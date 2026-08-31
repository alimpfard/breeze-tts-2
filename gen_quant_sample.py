"""Render the merchant passage under a given quantisation config.

Mirrors real service usage: conditioning is chained turn to turn, so accumulated
quantisation error gets to feed back through the reference the way it would in
a conversation -- which is where 4-bit is most likely to fall apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from breeze_infer.conditioning import ConditioningTurn
from kokoro_compat_server import BreezeEngine, VoiceLibrary

TURNS = [
    "“With the deepest respect, Your Majesty,” the merchant said, bowing again, “all trade ultimately is.”",
    "He spread his hands.",
    "“Your kingdom is beautiful.",
    "Rich in potential, But potential does not reduce carriage strain.”",
    "The king saw us then.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--voice", default="default")
    parser.add_argument("--fp8", choices=("off", "depth", "backbone", "all"), default="off")
    parser.add_argument("--int4", choices=("off", "depth", "backbone", "all"), default="off")
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-profile", type=Path, default=Path("configs/fast_service.json"))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    library = VoiceLibrary(args.voices_dir, args.voice)
    voice = library.get(args.voice)
    if voice is None:
        raise SystemExit(f"voice {args.voice!r} not found")

    engine = BreezeEngine(
        args.model,
        cfg_scale=args.cfg_scale,
        seed=42,
        fast=args.fast,
        fast_profile=args.fast_profile,
        fp8=args.fp8,
        int4=args.int4,
    )

    turns: list[ConditioningTurn] = []
    pieces: list[np.ndarray] = []
    rate = engine.sample_rate
    for index, text in enumerate(TURNS):
        audio, turns = engine.synthesize_stateless(text, voice, turns)
        pieces.append(audio)
        rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
        print(
            f"  turn {index}: {len(audio) / rate:5.2f}s  rms {rms:.4f}  "
            f"chain={len(turns)}",
            flush=True,
        )

    gap = np.zeros(int(0.35 * rate), dtype=np.float32)
    stitched = np.concatenate([p for turn in pieces for p in (turn, gap)][:-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, stitched, rate, subtype="PCM_16")
    print(f"-> {args.out} ({len(stitched) / rate:.1f}s)  fp8={args.fp8} int4={args.int4}")


if __name__ == "__main__":
    main()
