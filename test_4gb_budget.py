"""Verify a config fits a small card, without owning one.

torch.cuda.set_per_process_memory_fraction caps the caching allocator, so a
budget can be emulated on any GPU: if the config does not fit, allocation fails
loudly instead of silently succeeding on a machine with 128 GB.

What this proves: resident footprint, and that generation completes inside the
budget. What it cannot prove: throughput on the target card, which depends on
its bandwidth and clocks. Fit transfers; speed does not.

Caveat: the cap applies to torch allocations only. CUDA context and driver
overhead (~0.3-0.5 GB) sit outside it, so budget accordingly -- the default
below leaves room for that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from breeze_infer.conditioning import ConditioningTurn
from kokoro_compat_server import BreezeEngine, VoiceLibrary

GB = 1024**3

TURNS = [
    "Sure, I can take a look at that.",
    "He spread his hands.",
    "“Your kingdom is beautiful.",
    "Rich in potential, But potential does not reduce carriage strain.”",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--voice", default="default")
    parser.add_argument(
        "--budget-gb",
        type=float,
        default=3.5,
        help="Torch allocator cap. Default leaves ~0.5 GB of a 4 GB card for context.",
    )
    parser.add_argument("--fp8", choices=("off", "depth", "backbone", "all"), default="off")
    parser.add_argument("--int4", choices=("off", "depth", "backbone", "all"), default="all")
    parser.add_argument("--int8-text", action="store_true")
    parser.add_argument("--offload-embeddings", action="store_true")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--attention-precision", choices=("int4", "bf16"), default="int4")
    parser.add_argument("--int4-group-depth", type=int, default=128)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = (args.budget_gb * GB) / total
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(
        f"emulating a {args.budget_gb:.1f} GB budget on "
        f"{torch.cuda.get_device_name(0)} ({total / GB:.0f} GB) "
        f"-> fraction {fraction:.4f}\n"
    )

    # max_seq_len drives the KV cache, which is a real slice of a small budget.
    import kokoro_compat_server as server

    server.MAX_SEQ_LEN = args.max_seq_len

    library = VoiceLibrary(args.voices_dir, args.voice)
    voice = library.get(args.voice)
    if voice is None:
        raise SystemExit(f"voice {args.voice!r} not found")

    try:
        engine = BreezeEngine(
            args.model,
            cfg_scale=1.5,
            seed=42,
            fast=args.fast,
            fp8=args.fp8,
            int4=args.int4,
            int8_text=args.int8_text,
            offload_embeddings=args.offload_embeddings,
            low_memory=args.low_memory,
            attention_precision=args.attention_precision,
            int4_group_depth=args.int4_group_depth,
        )
    except torch.OutOfMemoryError as exc:
        print(f"FAILED to load inside {args.budget_gb} GB: {str(exc)[:200]}")
        raise SystemExit(1) from exc

    print(f"loaded.  allocated {torch.cuda.memory_allocated() / GB:.2f} GB  "
          f"reserved {torch.cuda.memory_reserved() / GB:.2f} GB")

    turns: list[ConditioningTurn] = []
    try:
        for index, text in enumerate(TURNS):
            audio, turns = engine.synthesize_stateless(text, voice, turns)
            rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
            print(
                f"  turn {index}: {len(audio) / engine.sample_rate:5.2f}s rms {rms:.4f}"
                f"  peak reserved {torch.cuda.max_memory_reserved() / GB:.2f} GB"
            )
    except torch.OutOfMemoryError as exc:
        print(f"FAILED during generation: {str(exc)[:200]}")
        raise SystemExit(1) from exc

    peak = torch.cuda.max_memory_reserved() / GB
    print(f"\nPASS: peak reserved {peak:.2f} GB of {args.budget_gb:.1f} GB budget")
    print(f"      implies ~{peak + 0.45:.2f} GB on a real card once CUDA context is counted")


if __name__ == "__main__":
    main()
