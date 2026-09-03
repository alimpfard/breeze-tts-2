"""Minimal-prompt clips: one attribute per prompt, so the label is honoured.

caption.paired showed a mood word in a two-line template is rendered
readably (0.44-0.56 vs 0.10 chance) while the same word in a ten-attribute
prose prompt is honoured ~37% of the time. These clips are training data
with that property: gender + age + exactly one of the weak attributes,
rendered with Breeze, latents extracted like caption.gen. Records match
data/caption/latents so caption.train can take both dirs.

    python -m caption.minimal /path/to/breeze --n 1000 --shard 0/2 --device cuda:0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.consistency import TEMPLATES
from caption.data import ATTRS, sample_attrs
from caption.gen import LAYERS
from caption.rl import RoundTrip

REPO = Path(__file__).resolve().parents[1]
WEAK = ("mood", "brightness", "texture", "energy", "pitch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out", type=Path, default=REPO / "data/caption/minimal")
    args = parser.parse_args()

    si, sn = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(exist_ok=True)
    sentences = [
        l.strip() for l in (REPO / "data/caption/sentences.txt").read_text().splitlines() if l.strip()
    ]
    rng = random.Random(args.seed)
    plan = []
    for i in range(args.n):
        base = sample_attrs(rng)
        attr = WEAK[i % len(WEAK)]
        attrs = {"gender": base["gender"], "age": base["age"], attr: rng.choice(ATTRS[attr])}
        plan.append((attrs, attr, rng.choice(sentences)))

    rt = RoundTrip(args.breeze, args.device)
    for i, (attrs, attr, text) in enumerate(plan):
        if i % sn != si:
            continue
        path = args.out / f"{i:06d}.pt"
        if path.exists():
            continue
        prose = TEMPLATES[attr].format(gender=attrs["gender"], age=attrs["age"], value=attrs[attr])
        lat = rt.render(prose, text, seed=0, raw=True)
        if lat is None:
            continue
        torch.save(
            {
                "latents": lat.to(torch.bfloat16).cpu(),
                "layers": LAYERS,
                "attrs": attrs,
                "prose": prose,
                "text": text,
                "seconds": lat.shape[1] / 12.5,
            },
            path,
        )
        if i % 100 < sn:
            print(f"  {i}/{args.n}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
