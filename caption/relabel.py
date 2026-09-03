"""Relabel clips with what Breeze itself hears, not what was requested.

For each clip and each attribute, score a template description with the
attribute set to every value (full-sequence likelihood, voice_invert's
scorer) and keep the argmin. Writes jsonl records {id, attrs, requested,
margin} where margin is the nats between best and second-best per attribute.

    python -m caption.relabel /path/to/breeze --shard 0/2 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.consistency import TEMPLATES
from caption.data import ATTRS
from voice_invert import VoiceScorer

REPO = Path(__file__).resolve().parents[1]
RELABEL = ("mood", "texture", "brightness", "pitch", "energy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "data/caption/relabel")
    args = parser.parse_args()

    si, sn = (int(x) for x in args.shard.split("/"))
    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    if args.limit:
        files = files[: args.limit]
    files = [p for i, p in enumerate(files) if i % sn == si]
    args.out.mkdir(exist_ok=True)
    out_path = args.out / f"shard{si}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()}
    scorer = VoiceScorer(args.breeze, args.device)

    with out_path.open("a") as fh:
        for n, p in enumerate(files):
            if p.stem in done:
                continue
            r = torch.load(p)
            a = r["attrs"]
            attrs = dict(a)
            margin = {}
            for attr in RELABEL:
                scores = {}
                for v in ATTRS[attr]:
                    desc = TEMPLATES[attr].format(
                        gender=a["gender"], age=a["age"], value=v
                    )
                    scores[v] = scorer.score(desc, r["text"], r["codes"]).total
                order = sorted(scores, key=scores.get)
                attrs[attr] = order[0]
                margin[attr] = round(scores[order[1]] - scores[order[0]], 4)
            fh.write(
                json.dumps(
                    {"id": p.stem, "attrs": attrs, "requested": a, "margin": margin}
                )
                + "\n"
            )
            fh.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(files)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
