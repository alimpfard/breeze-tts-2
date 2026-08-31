"""Weight-only int4 Linear for bandwidth-bound decode.

Same premise as models/fp8_linear.py but a step further down: at batch 1-2 every
weight is re-read from DRAM per token, and the depth decoder re-reads its
weights 16 times per audio frame, so halving bytes again roughly halves time.

Uses PyTorch's stock tinygemm path (``_convert_weight_to_int4pack`` /
``_weight_int4pack_mm``) -- no torchao dependency. It is *weight-only*: int4 is
what gets read from memory, then dequantised to bf16 for ordinary tensor-core
maths. Ampere/Blackwell int4 IMMA instructions are not involved and would not
help, because the bottleneck is bytes moved, not arithmetic.

Measured on GB10 (DGX Spark), DRAM-resident working set, vs bf16:

    depth gate/up 1024->8192   M=1 4.22x  M=2 3.78x
    depth down    8192->1024   M=1 3.06x  M=2 3.73x
    backbone up   2048->6144   M=1 4.50x  M=2 4.02x
    backbone down 6144->2048   M=1 4.87x  M=2 3.57x

Accuracy: group-128 int4 gives ~0.10 relative error per layer, against ~0.037
for FP8. That is inherent to 4 bits (step size ~0.39 sigma), not a bug -- and it
compounds across 12 depth-decoder layers x 16 codebooks per frame, which is the
main risk to audio quality.

The packing convention was determined empirically: the kernel expects the ODD
K index in the low nibble and the even index in the high nibble, dequantising as
``w = (q - 8) * scale + zero``. The natural ordering silently produces garbage
(relative error ~1.5) rather than failing, so do not "tidy" this.
"""

from __future__ import annotations

import torch
from torch import nn

GROUP_SIZE = 128
INNER_K_TILES = 8

# Below this, layers are L2-resident and launch-bound rather than
# bandwidth-bound, so quantisation costs more than it saves (FP8 measurably
# regressed on the 2.1 MB attention projections).
MIN_QUANT_BYTES = 8 * 1024 * 1024

DEFAULT_TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")


def quantize_int4(weight: torch.Tensor, group_size: int = GROUP_SIZE):
    """Group-wise asymmetric int4 quantisation in the tinygemm layout."""
    out_features, in_features = weight.shape
    grouped = weight.float().reshape(out_features, in_features // group_size, group_size)
    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    q = ((grouped - lo) / scale).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(out_features, in_features)

    # Odd index low nibble, even index high nibble. See module docstring.
    packed = (q[:, 1::2] | (q[:, 0::2] << 4)).contiguous()
    inner = torch._convert_weight_to_int4pack(packed, INNER_K_TILES)

    scales = scale.squeeze(-1).t().contiguous()
    zeros = (lo.squeeze(-1) + 8 * scale.squeeze(-1)).t().contiguous()
    scales_zeros = torch.stack([scales, zeros], dim=-1).to(torch.bfloat16).contiguous()
    return inner, scales_zeros


class Int4Linear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear using int4 weights."""

    def __init__(self, linear: nn.Linear, group_size: int = GROUP_SIZE) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("Int4Linear expects a bias-free Linear")

        weight = linear.weight.data
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        self.group_size = group_size
        if self.in_features % group_size:
            raise ValueError(
                f"in_features {self.in_features} not divisible by group {group_size}"
            )

        inner, scales_zeros = quantize_int4(weight, group_size)
        self.register_buffer("weight_int4", inner)
        self.register_buffer("scales_zeros", scales_zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        # tinygemm produces bf16; cast back if the caller is using something else.
        out = torch._weight_int4pack_mm(
            flat.to(torch.bfloat16),
            self.weight_int4,
            self.group_size,
            self.scales_zeros,
        )
        return out.to(x.dtype).reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"int4=True, group={self.group_size}"
        )


def quantize_module_int4(
    root: nn.Module,
    *,
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
    min_bytes: int = MIN_QUANT_BYTES,
) -> dict[str, int]:
    """Swap qualifying Linear layers under ``root`` for int4 equivalents.

    A layer that cannot be packed (awkward shape, unsupported geometry) is left
    untouched rather than failing the load -- reported as ``skipped_unsupported``
    so it is visible instead of silent.
    """
    converted = 0
    saved_bytes = 0
    skipped_small = 0
    skipped_unsupported = 0

    for module in root.modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear) or child.bias is not None:
                continue
            if target_names and child_name not in target_names:
                continue
            nbytes = child.weight.numel() * child.weight.element_size()
            if nbytes < min_bytes:
                skipped_small += 1
                continue
            try:
                replacement = Int4Linear(child)
            except Exception:  # noqa: BLE001 - shape/geometry rejections
                skipped_unsupported += 1
                continue
            setattr(module, child_name, replacement)
            converted += 1
            saved_bytes += nbytes - nbytes // 4

    return {
        "converted": converted,
        "skipped_below_threshold": skipped_small,
        "skipped_unsupported": skipped_unsupported,
        "weight_bytes_saved": saved_bytes,
    }
