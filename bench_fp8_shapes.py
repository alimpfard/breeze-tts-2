"""Microbenchmark: does FP8 still win at the depth decoder's real batch sizes?

The depth decoder runs at batch 1 (cfg=1.0) or 2 (cfg>1), not the M=16 used in
earlier probes. Small-M GEMMs can lose the bandwidth advantage to kernel
overhead, so measure before committing to a quantization implementation.

Working set is deliberately larger than L2 so we measure DRAM streaming, which
is the regime the real decode loop lives in.
"""

import time

import torch

DEV = "cuda"
COPIES = 48  # x 16MB = ~800MB, far beyond L2

SHAPES = [
    (1024, 8192, "gate/up  1024->8192"),
    (8192, 1024, "down     8192->1024"),
    (1024, 1024, "q/o      1024->1024"),
    (2048, 1024, "in_proj  2048->1024"),
]


def run(fn, reps=6):
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
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    for k, n, label in SHAPES:
        weights = [
            torch.randn(n, k, device=DEV, dtype=torch.bfloat16) for _ in range(COPIES)
        ]
        weights_fp8 = [w.to(torch.float8_e4m3fn) for w in weights]
        print(f"\n{label}   ({n}x{k}, {weights[0].numel() * 2 / 1e6:.1f} MB bf16)")

        for m in (1, 2, 4, 16):
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            x_fp8 = x.to(torch.float8_e4m3fn)
            scale_a = torch.ones(m, 1, device=DEV, dtype=torch.float32)
            scale_b = torch.ones(1, n, device=DEV, dtype=torch.float32)

            t_bf16 = run(lambda i: x @ weights[i].t())
            try:
                t_fp8 = run(
                    lambda i: torch._scaled_mm(
                        x_fp8,
                        weights_fp8[i].t(),
                        scale_a,
                        scale_b,
                        out_dtype=torch.bfloat16,
                    )
                )
                print(
                    f"  M={m:2d}  bf16 {t_bf16 * 1e6:7.1f}us   "
                    f"fp8 {t_fp8 * 1e6:7.1f}us   {t_bf16 / t_fp8:.2f}x"
                )
            except Exception as exc:  # noqa: BLE001 - reporting tool
                print(f"  M={m:2d}  bf16 {t_bf16 * 1e6:7.1f}us   fp8 FAILED: {exc}"[:150])


if __name__ == "__main__":
    main()
