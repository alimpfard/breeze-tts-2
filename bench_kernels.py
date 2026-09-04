"""Kernel-level view of a frame: which kernels, how many per frame, how long.

    python bench_kernels.py /path/to/breeze [--fp8 depth --int4 backbone]
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import kokoro_compat_server as srv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--voices-dir", type=Path, default=Path("voices"))
    p.add_argument("--fp8", default="off")
    p.add_argument("--int4", default="off")
    p.add_argument("--device", default=None)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--fused", action="store_true")
    args = p.parse_args()
    engine = srv.BreezeEngine(
        args.model, cfg_scale=1.5, seed=42, fast=True, fast_stages="decode",
        fast_profile=Path("configs/fast_service.json"), fp8=args.fp8, int4=args.int4, device=args.device,
        fused=args.fused,
    )
    voice = srv.VoiceLibrary(args.voices_dir, "default").get("default")
    text = "It was not an unfriendly silence, Mira had learned over three seasons."
    engine.synthesize(text, voice)
    rt = engine.runtime
    frames = {"n": 0}
    orig = rt._depth_decoder_graph.run

    def counted(*a, **k):
        frames["n"] += 1
        return orig(*a, **k)

    rt._depth_decoder_graph.run = counted
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        engine.synthesize(text, voice)
        torch.cuda.synchronize()
    n = frames["n"]
    agg: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    total = 0.0
    for e in prof.events():
        if e.device_type.name != "CUDA":
            continue
        agg[e.name][0] += 1
        agg[e.name][1] += e.self_device_time_total if hasattr(e, "self_device_time_total") else e.cuda_time_total
        total += e.self_device_time_total if hasattr(e, "self_device_time_total") else e.cuda_time_total
    print(f"{n} frames; GPU kernel time {total / n / 1000:.1f} ms/frame over {sum(v[0] for v in agg.values()) / n:.0f} kernels/frame")
    print(f"{'kernel':<70}{'per frame':>10}{'us each':>9}{'ms/frame':>10}")
    for name, (cnt, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{name[:70]:<70}{cnt / n:10.1f}{us / cnt:9.1f}{us / n / 1000:10.2f}")


if __name__ == "__main__":
    main()
