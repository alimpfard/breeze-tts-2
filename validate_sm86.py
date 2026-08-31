"""Validate the int4/int8 paths on Ampere (sm_86). Needs only torch.

The int4 packing convention in models/int4_linear.py was reverse-engineered on
sm_121 (Blackwell). _convert_weight_to_int4pack produces an architecture-
specific layout, so the convention may differ on Ampere -- and a mismatch does
not raise, it silently returns near-garbage (relative error ~1.5 rather than
~0.1). That is the failure this script exists to catch.

Deliberately standalone: no model download, no qwen_tts, no repo imports.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

GROUP = 128
COPIES = 12

# Depth decoder and backbone MLP shapes; batch 1-2 is what decode actually uses.
SHAPES = [
    (1024, 8192, "depth gate/up  1024->8192"),
    (8192, 1024, "depth down     8192->1024"),
    (2048, 6144, "backbone up    2048->6144"),
]


def quantize_int4(weight: torch.Tensor, group_size: int = GROUP, swap: bool = True):
    """Mirrors models/int4_linear.quantize_int4. `swap` toggles nibble order."""
    n, k = weight.shape
    grouped = weight.float().reshape(n, k // group_size, group_size)
    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    q = ((grouped - lo) / scale).round().clamp(0, 15).to(torch.uint8).reshape(n, k)

    a, b = (q[:, 1::2], q[:, 0::2]) if swap else (q[:, 0::2], q[:, 1::2])
    packed = (a | (b << 4)).contiguous()
    inner = torch._convert_weight_to_int4pack(packed, 8)

    scales = scale.squeeze(-1).t().contiguous()
    zeros = (lo.squeeze(-1) + 8 * scale.squeeze(-1)).t().contiguous()
    return inner, torch.stack([scales, zeros], dim=-1).to(torch.bfloat16).contiguous()


def quantize_int8(weight: torch.Tensor):
    amax = weight.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / 127.0
    q = (weight.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.float32).contiguous()


def bench(fn, reps=3):
    for i in range(COPIES):
        fn(i)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        for i in range(COPIES):
            fn(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / (reps * COPIES)


def main() -> None:
    cap = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"sm_{cap[0]}{cap[1]}")
    print(f"fp8 available (needs sm_89+): "
          f"{hasattr(torch, '_scaled_mm') and cap >= (8, 9)}\n")

    print("--- int4 nibble convention (ours is swap=True) ---")
    torch.manual_seed(0)
    w = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
    ref = (x @ w.t()).float()
    for swap in (True, False):
        try:
            inner, sz = quantize_int4(w, swap=swap)
            got = torch._weight_int4pack_mm(x, inner, GROUP, sz).float()
            err = ((got - ref).norm() / ref.norm()).item()
            verdict = "OK" if err < 0.25 else "GARBAGE"
            print(f"  swap={swap!s:5s} rel_err {err:.4f}  {verdict}")
        except Exception as exc:  # noqa: BLE001
            print(f"  swap={swap!s:5s} FAILED: {str(exc)[:70]}")

    print("\n--- int8 (text encoder path) ---")
    q, scale = quantize_int8(w)
    got = F.linear(x, q.to(x.dtype) * scale.to(x.dtype)).float()
    print(f"  rel_err {((got - ref).norm() / ref.norm()).item():.4f}")

    print("\n--- speed at decode batch sizes ---")
    print(f"{'shape':28s} {'M':>3s} {'bf16':>9s} {'int4':>9s} {'speedup':>8s}")
    for k, n, label in SHAPES:
        _bench_shape(k, n, label)
        torch.cuda.empty_cache()


def _bench_shape(k: int, n: int, label: str) -> None:
    """Own scope per shape: keeps the closures below bound to these tensors,
    and lets the weights fall out of scope before the next shape allocates."""
    weights = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(COPIES)
    ]
    packs = [quantize_int4(t) for t in weights]
    for m in (1, 2):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        t_bf = bench(lambda i: x @ weights[i].t())
        t_i4 = bench(
            lambda i: torch._weight_int4pack_mm(x, packs[i][0], GROUP, packs[i][1])
        )
        print(
            f"{label:28s} {m:>3d} {t_bf * 1e6:>8.1f}u {t_i4 * 1e6:>8.1f}u "
            f"{t_bf / t_i4:>7.2f}x"
        )


if __name__ == "__main__":
    main()
