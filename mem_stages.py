"""Compare VRAM for fast_all vs decode-only graphs on a given quant config.

Throughput turned out to be a wash between the two, so the question is whether
graphing prefill and the text encoder -- stages that run once per request -- is
costing memory for nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kokoro_compat_server import BreezeEngine

GB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--fp8", choices=("off", "depth", "backbone", "all"), default="depth")
    parser.add_argument("--int4", choices=("off", "depth", "backbone", "all"), default="backbone")
    parser.add_argument("--fast-profile", type=Path, default=Path("configs/fast_service.json"))
    parser.add_argument("--fast-stages", choices=("all", "decode"), default="all")
    args = parser.parse_args()

    engine = BreezeEngine(
        args.model,
        cfg_scale=1.5,
        seed=42,
        fast=True,
        fast_profile=args.fast_profile,
        fast_stages=args.fast_stages,
        fp8=args.fp8,
        int4=args.int4,
    )
    torch.cuda.synchronize()
    print(
        f"\nstages={args.fast_stages}  allocated {torch.cuda.memory_allocated() / GB:.2f} GB"
        f"  reserved {torch.cuda.memory_reserved() / GB:.2f} GB"
    )
    # Touch the model once so any lazily captured graph shows up too.
    _ = engine.sample_rate


if __name__ == "__main__":
    main()
