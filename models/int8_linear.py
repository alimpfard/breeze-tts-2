"""int8 weight storage for components where memory matters more than speed.

Unlike fp8_linear/int4_linear, this is NOT a throughput optimisation. It
dequantises to bf16 inside forward and runs an ordinary matmul, so it moves
*more* bytes than bf16 for a single use (read int8, write bf16, read bf16).
What it buys is halved *resident* VRAM.

That trade is right for the text encoder specifically: it runs once per request
at prefill rather than per audio frame, so its speed barely registers, while its
1.4 GB is the largest single block on a memory-constrained card. It is the wrong
trade for the backbone or depth decoder -- use int4_linear there.

It is also architecture-agnostic, which matters: fp8_linear needs _scaled_mm
(sm_89+), so on Ampere-era cards this is the only 8-bit option available.

Per-output-channel symmetric scales. Errors here propagate into conditioning for
everything downstream, which is why the text encoder deserves 8 bits rather than
the 4 used in the decode path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Lower than the decode-path modules: here we want the memory back from every
# layer, and there is no throughput to protect.
MIN_QUANT_BYTES = 1024 * 1024


class Int8Linear(nn.Module):
    """nn.Linear with int8-stored weights, dequantised per forward."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.data
        self.out_features, self.in_features = weight.shape

        amax = weight.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
        scale = amax / 127.0
        quantized = (weight.float() / scale).round().clamp(-127, 127).to(torch.int8)

        self.register_buffer("weight_int8", quantized.contiguous())
        self.register_buffer("weight_scale", scale.to(torch.float32).contiguous())
        self.register_buffer(
            "bias", linear.bias.data.clone() if linear.bias is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transient: one layer's weight in x.dtype, freed on return.
        weight = self.weight_int8.to(x.dtype) * self.weight_scale.to(x.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            "int8=True"
        )


def quantize_module_int8(
    root: nn.Module,
    *,
    target_names: tuple[str, ...] | None = None,
    min_bytes: int = MIN_QUANT_BYTES,
) -> dict[str, int]:
    """Store qualifying Linear weights as int8. ``None`` targets every Linear."""
    converted = 0
    saved_bytes = 0
    skipped_small = 0

    for module in root.modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if target_names is not None and child_name not in target_names:
                continue
            nbytes = child.weight.numel() * child.weight.element_size()
            if nbytes < min_bytes:
                skipped_small += 1
                continue
            setattr(module, child_name, Int8Linear(child))
            converted += 1
            saved_bytes += nbytes // 2

    return {
        "converted": converted,
        "skipped_below_threshold": skipped_small,
        "weight_bytes_saved": saved_bytes,
    }
