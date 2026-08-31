"""Save and load an already-quantized Breeze checkpoint.

Quantizing at load time means shipping 7.2 GB and needing that much host RAM
before anything shrinks. A pre-quantized checkpoint is ~2.3 GB, loads without
ever materialising bf16 weights, and skips packing 280 layers on the target
machine -- which matters most on exactly the hardware that needs it.

The manifest records which modules were replaced and the shapes of their
buffers, so the loader can rebuild the skeleton without a source Linear to
measure. Buffer shapes are recorded rather than derived: the int4 packed layout
comes from _convert_weight_to_int4pack and is not something to re-derive by
hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from models.int4_linear import Int4Linear
from models.int8_linear import Int8Linear

FORMAT_VERSION = 1
WEIGHTS_NAME = "quantized.safetensors"
# Kept in host memory when offloading: gathered once per request. The depth
# decoder embedding is absent by design -- it is hit 16x per frame.
TEXT_EMBEDDING_PREFIXES = ("embed_text_tokens.", "text_encoder.embed_tokens.")
MANIFEST_NAME = "quant_manifest.json"


def describe(model: nn.Module) -> dict[str, dict]:
    """Record every quantized module so the loader can recreate its shape."""
    described: dict[str, dict] = {}
    for path, module in model.named_modules():
        if isinstance(module, Int4Linear):
            described[path] = {
                "kind": "int4",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "group_size": module.group_size,
                "shapes": {
                    "weight_int4": list(module.weight_int4.shape),
                    "scales_zeros": list(module.scales_zeros.shape),
                },
            }
        elif isinstance(module, Int8Linear):
            described[path] = {
                "kind": "int8",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
                "shapes": {
                    "weight_int8": list(module.weight_int8.shape),
                    "weight_scale": list(module.weight_scale.shape),
                },
            }
    return described


def save_quantized(
    model: nn.Module, out_dir: Path, quant_config: dict, source: str
) -> dict:
    """Write the quantized weights and a manifest describing them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = describe(model)

    # Offloaded tables live on CPU; safetensors needs contiguous tensors and
    # rejects shared storage, so clone anything that is a view.
    state = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(out_dir / WEIGHTS_NAME))

    manifest = {
        "version": FORMAT_VERSION,
        "source": source,
        "quant_config": quant_config,
        "modules": modules,
        "tensor_count": len(state),
        "bytes": sum(v.numel() * v.element_size() for v in state.values()),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _empty_int4(spec: dict, device) -> Int4Linear:
    module = Int4Linear.__new__(Int4Linear)
    nn.Module.__init__(module)
    module.in_features = spec["in_features"]
    module.out_features = spec["out_features"]
    module.group_size = spec["group_size"]
    module.register_buffer(
        "weight_int4",
        torch.empty(spec["shapes"]["weight_int4"], dtype=torch.int32, device=device),
    )
    module.register_buffer(
        "scales_zeros",
        torch.empty(
            spec["shapes"]["scales_zeros"], dtype=torch.bfloat16, device=device
        ),
    )
    return module


def _empty_int8(spec: dict, device) -> Int8Linear:
    module = Int8Linear.__new__(Int8Linear)
    nn.Module.__init__(module)
    module.in_features = spec["in_features"]
    module.out_features = spec["out_features"]
    module.register_buffer(
        "weight_int8",
        torch.empty(spec["shapes"]["weight_int8"], dtype=torch.int8, device=device),
    )
    module.register_buffer(
        "weight_scale",
        torch.empty(
            spec["shapes"]["weight_scale"], dtype=torch.float32, device=device
        ),
    )
    module.register_buffer("bias", None)
    return module


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def install_shells(model: nn.Module, manifest: dict, device) -> int:
    """Swap in empty quantized modules so the state dict has somewhere to land."""
    for path, spec in manifest["modules"].items():
        builder = _empty_int4 if spec["kind"] == "int4" else _empty_int8
        _set_submodule(model, path, builder(spec, device))
    return len(manifest["modules"])


def load_quantized_state(
    model: nn.Module, out_dir: Path, device, host_prefixes: tuple[str, ...] = ()
) -> dict:
    """Load quantized weights into a model whose shells are already installed.

    ``host_prefixes`` names tensors to land in host memory instead of on the
    device. Without this, tensors destined for CPU offload still visit the GPU
    first, so peak load is the whole checkpoint -- enough to OOM a card that the
    steady-state config would fit comfortably.
    """
    if host_prefixes:
        from safetensors import safe_open

        state = {}
        with safe_open(str(out_dir / WEIGHTS_NAME), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safetensors API
                target = (
                    "cpu" if key.startswith(host_prefixes) else str(device)
                )
                state[key] = handle.get_tensor(key).to(target)
    else:
        state = load_file(str(out_dir / WEIGHTS_NAME), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    return {
        "loaded": len(state),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_sample": list(missing)[:5],
        "unexpected_sample": list(unexpected)[:5],
    }


def read_manifest(out_dir: Path) -> dict:
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    if manifest.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint version {manifest.get('version')!r}")
    return manifest


def load_quantized_runtime(
    ckpt_dir: Path, device: torch.device | str, offload_embeddings: bool = False
):
    """Build the runtime from a pre-quantized checkpoint.

    The skeleton is constructed on the meta device so no bf16 weight is ever
    allocated -- the whole point on hardware that cannot hold them. Every
    parameter is then assigned from the checkpoint; anything left on meta would
    fail loudly at first use, so the loader checks for that explicitly.
    """
    from accelerate import init_empty_weights
    from transformers import AutoTokenizer

    from models.breeze import BreezeForConditionalGeneration
    from models.breeze_config import BreezeConfig

    manifest = read_manifest(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    config = BreezeConfig.from_pretrained(str(ckpt_dir))
    config._attn_implementation = "eager"
    with init_empty_weights():
        model = BreezeForConditionalGeneration(config)

    installed = install_shells(model, manifest, device="meta")
    # Tensors bound for host memory never touch the GPU, so peak load tracks
    # what actually stays resident.
    host_prefixes = TEXT_EMBEDDING_PREFIXES if offload_embeddings else ()
    stats = load_quantized_state(model, ckpt_dir, device, host_prefixes)

    stranded = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor is not None and tensor.is_meta
    ]
    if stranded:
        raise RuntimeError(
            f"{len(stranded)} tensors were not present in the checkpoint, "
            f"e.g. {stranded[:5]}"
        )

    if offload_embeddings:
        # Wrap now, before the .to(device) below would pull them across.
        from models.offload import offload_text_embeddings

        stats["offload"] = offload_text_embeddings(model, device)

    # Non-persistent buffers (audio_tokens_offsets, rotary inv_freq, ...) are
    # excluded from state_dict by design -- they are constants derived from the
    # config. init_empty_weights leaves buffers alone, so they exist on CPU and
    # need moving. Safe here only because the meta check above already proved
    # every real parameter was assigned.
    model.to(device)
    model.eval()

    from qwen_tts import Qwen3TTSTokenizer

    audio_dir = ckpt_dir / "audio_tokenizer"
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"{audio_dir} missing -- the checkpoint must be exported with its "
            "audio tokenizer to stand alone."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(audio_dir), device_map=str(device)
    )

    stats["shells_installed"] = installed
    stats["quant_config"] = manifest["quant_config"]
    return tokenizer, model, audio_tokenizer, stats
