"""Render the same passage through Kokoro-82M for comparison.

Same turns, same stitching, so the two files can be A/B'd directly. Kokoro is
the model this service replaced: 82M parameters against Breeze's 3.5B.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang", default="a")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    passage = [
        line.strip()
        for line in args.turns.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    pipeline = KPipeline(lang_code=args.lang, repo_id="hexgrad/Kokoro-82M")
    # Warm up so the first turn is not paying model load.
    for _ in pipeline("warmup", voice=args.voice):
        pass

    pieces: list[np.ndarray] = []
    total_audio = total_gen = 0.0
    for index, text in enumerate(passage):
        started = time.time()
        chunks = [audio for _gs, _ps, audio in pipeline(text, voice=args.voice)]
        elapsed = time.time() - started
        if not chunks:
            print(f"  turn {index}: no audio")
            continue
        audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        pieces.append(audio)
        duration = len(audio) / SAMPLE_RATE
        total_audio += duration
        total_gen += elapsed
        print(
            f"  turn {index}: {duration:5.2f}s in {elapsed:5.2f}s "
            f"= {duration / elapsed:6.1f}x  rms {np.sqrt((audio**2).mean()):.4f}",
            flush=True,
        )

    gap = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    stitched = np.concatenate([p for t in pieces for p in (t, gap)][:-1])
    sf.write(args.out, stitched, SAMPLE_RATE, subtype="PCM_16")
    print(
        f"\n{total_audio:.1f}s audio in {total_gen:.1f}s = {total_audio / total_gen:.1f}x realtime"
    )
    print(f"-> {args.out} ({len(stitched) / SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
