"""FP8 (e4m3) dynamic-quantized Linear for bandwidth-bound decode.

Breeze's decode loop is memory-bound: at batch 1-2 every weight is re-read from
DRAM to produce one token, and the depth decoder alone re-reads its weights 16
times per audio frame (once per codebook). Halving weight bytes therefore buys
close to a proportional speedup on the layers that dominate traffic.

Measured on GB10 (DGX Spark, ~273 GB/s), DRAM-resident working set:

    gate/up 1024->8192  (16.8 MB)   1.56-1.75x
    down    8192->1024  (16.8 MB)   1.15-1.52x
    q/o     1024->1024  ( 2.1 MB)   0.81-0.90x   <- slower!

Small projections are L2-resident and launch-bound, so the extra scaling work
costs more than the bytes saved. Hence MIN_QUANT_BYTES: only quantize layers
big enough to actually be DRAM-bound. MLP is ~91% of per-layer traffic, so
skipping attention costs little.

Weights use per-output-channel scales (static, computed once). Activations use
per-row scales computed on the fly, which keeps this accurate at the cost of a
few small kernels -- torch.compile is expected to fuse those into the preceding
norm/activation. Everything here is shape-static and free of data-dependent
control flow so it survives torch.compile(fullgraph=True) and CUDA graph
capture, both of which the fast path applies to these modules.
"""

from __future__ import annotations

import torch
from torch import nn

# Max representable magnitude of float8_e4m3fn.
FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn

# Below this, FP8 loses to bf16 (see module docstring). 8 MB sits between the
# 4.2 MB projections that regress and the 16.8 MB MLP tensors that win.
MIN_QUANT_BYTES = 8 * 1024 * 1024

# Names worth quantizing: the MLP projections carrying the bulk of the traffic.
DEFAULT_TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")


class Fp8Linear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear using FP8 weights."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("Fp8Linear expects a bias-free Linear")

        weight = linear.weight.data
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

        # Per-output-channel scale keeps outlier channels from squashing the
        # rest, and is what _scaled_mm's RowWise mode expects.
        amax = weight.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
        weight_scale = amax / FP8_MAX
        quantized = (weight.float() / weight_scale).clamp(-FP8_MAX, FP8_MAX)

        self.register_buffer("weight_fp8", quantized.to(FP8_DTYPE).contiguous())
        # _scaled_mm wants scale_b shaped (1, N) and contiguous.
        self.register_buffer(
            "weight_scale", weight_scale.reshape(1, -1).contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        row_amax = flat.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
        row_scale = row_amax / FP8_MAX
        flat_fp8 = (flat.float() / row_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        out = torch._scaled_mm(
            flat_fp8,
            self.weight_fp8.t(),
            row_scale,
            self.weight_scale,
            out_dtype=x.dtype,
        )
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8=True"


def quantize_module_fp8(
    root: nn.Module,
    *,
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
    min_bytes: int = MIN_QUANT_BYTES,
) -> dict[str, int]:
    """Swap qualifying Linear layers under ``root`` for FP8 equivalents.

    Returns a summary of what was converted and what was skipped, so callers can
    log it rather than silently quantizing more or less than intended.
    """
    converted = 0
    saved_bytes = 0
    skipped_small = 0

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
            setattr(module, child_name, Fp8Linear(child))
            converted += 1
            saved_bytes += nbytes // 2

    return {
        "converted": converted,
        "skipped_below_threshold": skipped_small,
        "weight_bytes_saved": saved_bytes,
    }
