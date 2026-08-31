"""Export a pre-quantized Breeze checkpoint.

Run once on a machine with room; ship the result to one that has none.

    python quantize_checkpoint.py /path/to/breeze-tts-2 --out breeze-int4 \
        --int4 all --int8-text --attention-precision int4

Embeddings are NOT offloaded here: offloading is a runtime placement decision,
not a property of the weights. The checkpoint stores them normally and the
server decides at load time.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
from models.quantize_config import QuantConfig, apply_quantization
from models.quantized_checkpoint import save_quantized

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("breeze-tts")

GB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp8", choices=("off", "depth", "backbone", "all"), default="off")
    parser.add_argument("--int4", choices=("off", "depth", "backbone", "all"), default="all")
    parser.add_argument("--int8-text", action="store_true")
    parser.add_argument("--text-precision", choices=("bf16", "int8", "int4"),
                        default="bf16")
    parser.add_argument(
        "--attention-precision", choices=("int4", "bf16"), default="int4"
    )
    parser.add_argument("--int4-group-depth", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Where to quantize. cpu keeps VRAM free but int4 packing needs CUDA.",
    )
    args = parser.parse_args()

    # int4 packing goes through a CUDA kernel, so quantization runs on GPU even
    # though the source model is staged on CPU.
    pack_device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("loading %s on cpu ...", args.model)
    _, model, _ = load_runtime(
        args.model, device="cpu", attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)

    config = QuantConfig(
        fp8=args.fp8,
        int4=args.int4,
        int8_text=args.int8_text,
        text_precision=args.text_precision,
        offload_embeddings=False,  # runtime placement, not a weight property
        attention_precision=args.attention_precision,
        int4_group_depth=args.int4_group_depth,
        low_memory=True,  # pack layer by layer
    )
    apply_quantization(model, config, pack_device)

    # Bring everything to one device so the state dict is uniform.
    model.to("cpu")

    log.info("writing %s ...", args.out)
    manifest = save_quantized(
        model, args.out, config.as_dict(), source=str(args.model)
    )

    # The checkpoint holds the model, but the runtime also needs the text
    # tokenizer and the separate audio tokenizer -- neither appears in
    # model.state_dict(). Copy them so the artifact stands alone.
    import shutil

    args.out.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        source_file = args.model / name
        if source_file.is_file():
            shutil.copy2(source_file, args.out / name)
    audio_dir = args.model / "audio_tokenizer"
    if audio_dir.is_dir() and not (args.out / "audio_tokenizer").exists():
        shutil.copytree(audio_dir, args.out / "audio_tokenizer")
    print(
        f"\nwrote {manifest['tensor_count']} tensors, "
        f"{manifest['bytes'] / GB:.2f} GB to {args.out}/"
    )
    print(f"quantized modules: {len(manifest['modules'])}")


if __name__ == "__main__":
    main()
