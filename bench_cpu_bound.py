"""Upper-bound CPU realtime factor from memory bandwidth at the decode shapes.

The repo's inference runtime hard-requires CUDA (models/fast_streaming.py:193),
so there is no CPU path to benchmark directly. But the decode loop is
bandwidth-bound, which makes the ceiling computable: measure how fast the CPU
can stream the weights it must touch per audio frame, and divide.

Per audio frame Breeze reads, at bf16:
    backbone       ~2.8 GB  x 1  (one forward per frame)
    depth decoder  ~0.67 GB x 16 (autoregressive over 16 codebooks)
    = ~13.5 GB, at 12.5 frames/sec -> ~169 GB/s to hit realtime.

This is a generous upper bound: it counts only MLP-shaped GEMMs and ignores
attention, norms, sampling, the codec decode, and all framework overhead. Real
throughput will be lower.
"""

from __future__ import annotations

import time

import torch

COPIES = 24  # keep the working set well beyond last-level cache
FRAME_RATE = 12.5
BYTES_PER_FRAME_BF16 = 13.5 * (1024**3)

SHAPES = [
    (1024, 8192, "depth gate/up  1024->8192"),
    (8192, 1024, "depth down     8192->1024"),
    (2048, 6144, "backbone up    2048->6144"),
]


def run(fn, reps=3):
    for i in range(COPIES):
        fn(i)
    started = time.perf_counter()
    for _ in range(reps):
        for i in range(COPIES):
            fn(i)
    return (time.perf_counter() - started) / (reps * COPIES)


def main() -> None:
    print(f"threads: {torch.get_num_threads()}   torch {torch.__version__}")
    for dtype in (torch.bfloat16, torch.float32):
        print(f"\n=== {dtype} ===")
        rates = []
        for k, n, label in SHAPES:
            weights = [torch.randn(n, k, dtype=dtype) for _ in range(COPIES)]
            x = torch.randn(2, k, dtype=dtype)
            elapsed = run(lambda i: x @ weights[i].t())
            nbytes = weights[0].numel() * weights[0].element_size()
            gbs = nbytes / elapsed / 1e9
            rates.append(gbs)
            print(f"  {label}  {elapsed * 1e6:9.1f}us  {gbs:6.1f} GB/s")

        # Scale the bf16 frame budget by dtype width.
        width = 2 if dtype == torch.bfloat16 else 4
        per_frame = BYTES_PER_FRAME_BF16 * (width / 2)
        achieved = sum(rates) / len(rates) * 1e9
        rtf = achieved / (per_frame * FRAME_RATE)
        print(
            f"  mean {achieved / 1e9:.1f} GB/s -> upper-bound RTF ~{rtf:.3f}x "
            f"({per_frame * FRAME_RATE / 1e9:.0f} GB/s needed for realtime)"
        )


if __name__ == "__main__":
    main()
