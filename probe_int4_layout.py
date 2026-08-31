"""Find the packing convention _weight_int4pack_mm actually expects.

Speed looked right but relative error was ~1.45, i.e. the result is
uncorrelated with the reference -- so the nibble order and/or the K
interleaving in my hand-rolled pack does not match the kernel's layout.
Try the plausible conventions and see which one reconstructs the matmul.
"""

from __future__ import annotations

import itertools

import torch

DEV = "cuda"
N, K, M = 256, 512, 2
GROUP = 128


def build(weight, *, nibble_swap: bool, halves: bool, zero_offset: bool):
    n, k = weight.shape
    grouped = weight.float().reshape(n, k // GROUP, GROUP)
    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    q = ((grouped - lo) / scale).round().clamp(0, 15).to(torch.uint8).reshape(n, k)

    if halves:  # first half in low nibble, second half in high nibble
        a, b = q[:, : k // 2], q[:, k // 2 :]
    else:  # adjacent pairs
        a, b = q[:, 0::2], q[:, 1::2]
    if nibble_swap:
        a, b = b, a
    packed = (a | (b << 4)).contiguous()

    inner = torch._convert_weight_to_int4pack(packed, 8)
    s = scale.squeeze(-1).t().contiguous()
    z = lo.squeeze(-1).t().contiguous()
    if zero_offset:
        z = z + 8 * s
    sz = torch.stack([s, z], dim=-1).to(torch.bfloat16)
    return inner, sz


def main() -> None:
    torch.manual_seed(0)
    weight = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    ref = (x @ weight.t()).float()

    print(f"{'nibble_swap':>12s} {'halves':>7s} {'zero+8s':>8s} {'rel_err':>9s}")
    best = None
    for swap, halves, zoff in itertools.product([False, True], repeat=3):
        try:
            inner, sz = build(weight, nibble_swap=swap, halves=halves, zero_offset=zoff)
            got = torch._weight_int4pack_mm(x, inner, GROUP, sz).float()
            err = ((got - ref).norm() / ref.norm()).item()
            print(f"{swap!s:>12s} {halves!s:>7s} {zoff!s:>8s} {err:>9.4f}")
            if best is None or err < best[0]:
                best = (err, swap, halves, zoff)
        except Exception as exc:  # noqa: BLE001
            print(f"{swap!s:>12s} {halves!s:>7s} {zoff!s:>8s}  FAILED {str(exc)[:40]}")

    if best:
        err, swap, halves, zoff = best
        print(f"\nbest: rel_err {err:.4f} "
              f"(nibble_swap={swap}, halves={halves}, zero_offset={zoff})")
        print("int4 group quantisation should land near 0.02-0.05 for random data;"
              " anything above ~0.2 means the layout is still wrong.")


if __name__ == "__main__":
    main()
