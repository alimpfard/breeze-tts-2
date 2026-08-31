"""Measure steady-state realtime factor for the kokoro-compatible Breeze server.

Loads the model once, then generates several utterances in-process, which is
what the long-running service actually does. Model load is reported separately
since it is a one-time startup cost.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kokoro_compat_server import BreezeEngine, VoiceLibrary

PHRASES = [
    "Sure, one sec.",
    "Hey, this is a test of the replacement voice. Seems to be working so far.",
    "I've charted the route and checked the weather, so we should be fine.",
    "That's a good question, and the honest answer is that it depends on "
    "what you're optimizing for. If you want it fast, we cut the reranking "
    "step. If you want it accurate, we keep it and eat the latency.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-profile", type=Path, default=None)
    parser.add_argument("--fast-stages", choices=("all", "decode"), default="all")
    parser.add_argument("--fp8", choices=("off", "depth", "backbone", "all"), default="off")
    parser.add_argument("--int4", choices=("off", "depth", "backbone", "all"), default="off")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--max-phrases",
        type=int,
        default=len(PHRASES),
        help="Trim the phrase list. CPU runs are slow enough to want this.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the discarded warmup generation (saves minutes on CPU).",
    )
    args = parser.parse_args()

    library = VoiceLibrary(args.voices_dir, args.voice)
    voice = library.get(args.voice)
    if voice is None:
        raise SystemExit(f"voice {args.voice!r} not found in {args.voices_dir}")

    load_started = time.time()
    engine = BreezeEngine(
        args.model,
        cfg_scale=args.cfg_scale,
        seed=42,
        fast=args.fast,
        fast_stages=args.fast_stages,
        fast_profile=args.fast_profile,
        fp8=args.fp8,
        int4=args.int4,
        device=args.device,
        dtype=args.dtype,
    )
    load_elapsed = time.time() - load_started

    # First generation pays lazy-init costs; run one and discard it.
    if not args.skip_warmup:
        engine.synthesize("Warming up the pipeline.", voice)

    print(f"\ndevice: {engine.device}  dtype: {args.dtype}  fp8: {args.fp8}  int4: {args.int4}")
    print(f"model load: {load_elapsed:.1f}s")
    print(f"{'chars':>6} {'audio_s':>8} {'gen_s':>7} {'RTF':>6}")

    total_audio = 0.0
    total_gen = 0.0
    for phrase in PHRASES[: args.max_phrases]:
        started = time.time()
        audio = engine.synthesize(phrase, voice)
        elapsed = time.time() - started
        duration = len(audio) / engine.sample_rate
        total_audio += duration
        total_gen += elapsed
        print(f"{len(phrase):>6} {duration:>8.2f} {elapsed:>7.2f} {duration/elapsed:>6.2f}x")

    print(f"\ntotal: {total_audio:.2f}s audio in {total_gen:.2f}s "
          f"= {total_audio/total_gen:.2f}x realtime")


if __name__ == "__main__":
    main()
