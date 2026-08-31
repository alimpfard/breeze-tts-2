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
    parser.add_argument("--int8-text", action="store_true")
    parser.add_argument("--text-precision", choices=("bf16", "int8", "int4"),
                        default="bf16")
    parser.add_argument("--offload-embeddings", action="store_true")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--attention-precision", choices=("int4", "bf16"), default="int4")
    parser.add_argument("--int4-group-depth", type=int, default=128)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-profile", type=Path, default=Path("configs/fast_service.json"))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--turns", type=Path, default=None,
                        help="File with one turn per line. Defaults to the built-in set.")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Match the deployment: the 4 GB config runs 512.")
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
        max_seq_len=args.max_seq_len,
        int8_text=args.int8_text,
        text_precision=args.text_precision,
        offload_embeddings=args.offload_embeddings,
        low_memory=args.low_memory,
        attention_precision=args.attention_precision,
        int4_group_depth=args.int4_group_depth,
    )

    passage = TURNS
    if args.turns is not None:
        passage = [ln.strip() for ln in args.turns.read_text(encoding='utf-8').splitlines() if ln.strip()]

    turns: list[ConditioningTurn] = []
    pieces: list[np.ndarray] = []
    rate = engine.sample_rate
    for index, text in enumerate(passage):
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
