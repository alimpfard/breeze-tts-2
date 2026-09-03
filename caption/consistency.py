"""Does the forward model realise an attribute strongly enough to read back?

For each clip, score a template description with the attribute set to every
one of its values, using the Breeze likelihood scorer over the full code
sequence (voice_invert.VoiceScorer). If the true value wins only slightly
above chance, the audio itself does not carry the attribute consistently,
and no captioner can do better on these labels.

    python -m caption.consistency /path/to/breeze --attrs mood texture --n 150
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.data import ATTRS
from voice_invert import VoiceScorer

REPO = Path(__file__).resolve().parents[1]

TEMPLATES = {
    "mood": "A {gender}, {age}. The mood is {value}.",
    "texture": "A {gender}, {age}. The voice has {value} texture.",
    "brightness": "A {gender}, {age}. The voice has {value} resonance.",
    "pitch": "A {gender}, {age}. The voice has a {value} pitch.",
    "energy": "A {gender}, {age}. The delivery is {value}.",
    "sex": "A {value}, {age}.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attrs", nargs="+", default=["mood", "texture", "brightness", "sex"]
    )
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--window", type=float, default=0.0)
    args = parser.parse_args()

    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    random.Random(0).shuffle(files)
    scorer = VoiceScorer(args.breeze, args.device, window_seconds=args.window)

    for attr in args.attrs:
        values = ["woman", "man"] if attr == "sex" else ATTRS[attr]
        hits = 0
        total = 0
        ranks: Counter = Counter()
        for p in files:
            if total >= args.n:
                break
            r = torch.load(p)
            a = r["attrs"]
            if attr == "sex":
                truth = (
                    "woman"
                    if a["gender"] in ("woman", "girl", "teenage girl")
                    else "man"
                )
                if a["gender"] not in ("woman", "man"):
                    continue  # keep the adult binary only
            else:
                truth = a.get(attr)
                if truth is None:
                    continue
            scores = {}
            for v in values:
                desc = TEMPLATES[attr].format(gender=a["gender"], age=a["age"], value=v)
                scores[v] = scorer.score(desc, r["text"], r["codes"]).total
            order = sorted(values, key=scores.get)
            rank = order.index(truth)
            ranks[rank] += 1
            hits += rank == 0
            total += 1
        print(
            f"{attr:<11} n={total} k={len(values)} chance={1 / len(values):.2f} "
            f"top-1 {hits / total:.2f}  top-2 {(ranks[0] + ranks[1]) / total:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
