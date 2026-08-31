"""Does weight-only int4 pay at the depth decoder's real shapes?

Decode is bandwidth-bound, so the win is reading int4 weights from DRAM (4x
less traffic), not int4 arithmetic -- tinygemm dequantises to bf16 and uses
ordinary tensor cores. Ampere int4 IMMA is irrelevant to us.

torch._convert_weight_to_int4pack wants uint8 of shape (N, K//2) with two
4-bit values per byte. The resulting layout satisfies
    K == B.size(1) * B.size(3) * 32
which is how the tile geometry was pinned down.

Working set exceeds L2 so this measures DRAM streaming, at the batch sizes the
depth decoder actually uses (1-2 CFG branches).
"""

from __future__ import annotations

import time

import torch

DEV = "cuda"
COPIES = 24
GROUP_SIZE = 128

SHAPES = [
    (1024, 8192, "depth gate/up  1024->8192"),
    (8192, 1024, "depth down     8192->1024"),
    (2048, 6144, "backbone up    2048->6144"),
    (6144, 2048, "backbone down  6144->2048"),
]


def run(fn, reps=4):
    for i in range(COPIES):
        fn(i)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        for i in range(COPIES):
            fn(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / (reps * COPIES)


def quantize_int4(weight: torch.Tensor, group_size: int = GROUP_SIZE):
    """Group-wise asymmetric int4 quantisation in the tinygemm layout."""
    n, k = weight.shape
    grouped = weight.float().reshape(n, k // group_size, group_size)
    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    zero = lo
    q = ((grouped - zero) / scale).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(n, k)

    # Two nibbles per byte along K. The kernel expects the ODD index in the low
    # nibble and the even index in the high nibble (determined empirically --
    # the natural ordering gives a relative error of ~1.5).
    packed = (q[:, 1::2] | (q[:, 0::2] << 4)).contiguous()
    inner = torch._convert_weight_to_int4pack(packed, 8)

    # tinygemm wants (K // group, N, 2) holding scale and the midpoint offset,
    # dequantising as w = (q - 8) * scale + zero.
    scales = scale.squeeze(-1).t().contiguous()
    zeros = (zero.squeeze(-1) + 8 * scale.squeeze(-1)).t().contiguous()
    scales_zeros = torch.stack([scales, zeros], dim=-1).to(torch.bfloat16)
    return inner, scales_zeros


def main() -> None:
    caps = "".join(map(str, torch.cuda.get_device_capability()))
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}  sm_{caps}\n")
    print(f"{'shape':28s} {'M':>3s} {'bf16':>9s} {'int4':>9s} {'speedup':>8s}")

    for k, n, label in SHAPES:
        weights = [
            torch.randn(n, k, device=DEV, dtype=torch.bfloat16) for _ in range(COPIES)
        ]
        try:
            packs = [quantize_int4(w) for w in weights]
        except Exception as exc:  # noqa: BLE001
            print(f"{label:28s}  pack FAILED: {str(exc)[:80]}")
            continue

        for m in (1, 2, 16):
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            t_bf16 = run(lambda i: x @ weights[i].t())
            try:
                t_int4 = run(
                    lambda i: torch._weight_int4pack_mm(
                        x, packs[i][0], GROUP_SIZE, packs[i][1]
                    )
                )
                print(
                    f"{label:28s} {m:>3d} {t_bf16 * 1e6:>8.1f}u {t_int4 * 1e6:>8.1f}u "
                    f"{t_bf16 / t_int4:>7.2f}x"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{label:28s} {m:>3d} {t_bf16 * 1e6:>8.1f}u  FAILED: "
                      f"{str(exc)[:70]}")
                break

        # Sanity: does the quantised path actually approximate the original?
        x = torch.randn(2, k, device=DEV, dtype=torch.bfloat16)
        ref = (x @ weights[0].t()).float()
        got = torch._weight_int4pack_mm(
            x, packs[0][0], GROUP_SIZE, packs[0][1]
        ).float()
        err = ((got - ref).norm() / ref.norm()).item()
        print(f"{'':28s}     relative error {err:.4f}")


if __name__ == "__main__":
    main()
