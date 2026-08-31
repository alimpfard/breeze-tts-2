"""One place that decides how a model gets quantized.

Shared by the server and the checkpoint exporter so the two cannot drift: a
pre-quantized checkpoint must be built exactly the way the server would have
built it in memory, or the saved weights will not match what the loader expects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from models.fp8_linear import quantize_module_fp8
from models.int4_linear import quantize_module_int4
from models.int8_linear import quantize_module_int8
from models.offload import offload_text_embeddings

log = logging.getLogger("breeze-tts")

# In low-memory mode the size threshold inverts: it normally protects throughput
# by leaving small L2-resident layers in bf16, but on a card that barely fits the
# model every byte counts -- attention included.
LOW_MEMORY_MIN_BYTES = 256 * 1024
ATTENTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_NAMES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class QuantConfig:
    fp8: str = "off"
    int4: str = "off"
    int8_text: bool = False
    # bf16 | int8 | int4. int4 is only sane on a card that cannot
    # otherwise fit: the text encoder feeds conditioning downstream.
    text_precision: str = "bf16"
    offload_embeddings: bool = False
    attention_precision: str = "int4"
    int4_group_depth: int = 128
    low_memory: bool = False

    def as_dict(self) -> dict:
        return {
            "fp8": self.fp8,
            "int4": self.int4,
            "int8_text": self.int8_text,
            "text_precision": self.text_precision,
            "offload_embeddings": self.offload_embeddings,
            "attention_precision": self.attention_precision,
            "int4_group_depth": self.int4_group_depth,
            "low_memory": self.low_memory,
        }


def _components(spec: str) -> list[str]:
    if spec == "off":
        return []
    if spec == "all":
        return ["depth_decoder", "backbone"]
    return ["depth_decoder"] if spec == "depth" else ["backbone"]


def apply_quantization(
    model: nn.Module,
    config: QuantConfig,
    device: torch.device | str,
) -> dict[str, object]:
    """Quantize in place. Returns per-stage stats for logging."""
    submodules = {
        "depth_decoder": model.depth_decoder,
        "backbone": model.backbone_model,
    }
    quantized: set[str] = set()
    stats: dict[str, object] = {}

    # In low-memory mode each layer is packed straight onto the target device so
    # only the packed form accumulates there.
    quant_device = device if config.low_memory else None
    base_kwargs: dict[str, object] = {"device": quant_device}
    if config.low_memory and config.attention_precision == "int4":
        base_kwargs["min_bytes"] = LOW_MEMORY_MIN_BYTES
        base_kwargs["target_names"] = MLP_NAMES + ATTENTION_NAMES

    for name in _components(config.int4):
        kwargs = dict(base_kwargs)
        # A smaller group cuts int4 error for a modest scale table; the depth
        # decoder is where it would matter most (16 rounds per frame).
        if name == "depth_decoder":
            kwargs["group_size"] = config.int4_group_depth
        stats[f"int4:{name}"] = quantize_module_int4(submodules[name], **kwargs)
        quantized.add(name)

    for name in _components(config.fp8):
        if name in quantized:
            continue
        stats[f"fp8:{name}"] = quantize_module_fp8(submodules[name])

    # Runs once per request rather than per frame, so precision here costs no
    # throughput -- and its errors feed conditioning for everything downstream.
    text_precision = config.text_precision
    if config.int8_text and text_precision == "bf16":
        text_precision = "int8"  # backward compatible with the older flag
    if text_precision != "bf16" and getattr(model, "text_encoder", None) is not None:
        if text_precision == "int4":
            stats["int4:text_encoder"] = quantize_module_int4(
                model.text_encoder,
                target_names=None,
                min_bytes=256 * 1024,
                device=quant_device,
            )
        else:
            stats["int8:text_encoder"] = quantize_module_int8(model.text_encoder)

    # Must precede any later .to(device): the tables are gathered once per
    # request and cost 1.67 GB resident. The depth decoder's embedding stays
    # put -- it is hit 16x per frame.
    if config.offload_embeddings:
        stats["offload"] = offload_text_embeddings(model, device)

    for key, value in stats.items():
        log.info("%s: %s", key, value)
    return stats
