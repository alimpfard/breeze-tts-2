"""Can the roundtrip reward see an attribute at all?

For clips with a stated attribute, render the same text under a template
prompt with the true value, with the value swapped, and under the true
value with a different seed (the noise floor). If swapping the word moves
the cosine less than changing the seed, RL against this reward has no
gradient toward that attribute, whatever the theory says.

    python -m caption.reward_sensitivity /path/to/breeze --attr mood --n 24
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.consistency import TEMPLATES
from caption.data import ATTRS
from caption.model import pool_features
from caption.rl import RoundTrip

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--attr", default="mood")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument(
        "--labels", default="requested", choices=("requested", "breeze")
    )
    args = parser.parse_args()

    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    random.Random(7).shuffle(files)
    relabel = {}
    if args.labels == "breeze":
        import json

        for line in (REPO / "data/caption/relabel/all.jsonl").read_text().splitlines():
            d = json.loads(line)
            relabel[d["id"]] = d["attrs"]

    recs = []
    for p in files:
        r = torch.load(p)
        a = relabel.get(p.stem, r["attrs"]) if relabel else r["attrs"]
        if a.get(args.attr) and a["gender"] in ("woman", "man"):
            r["use"] = a
            recs.append(r)
        if len(recs) >= args.n:
            break

    # Standardisation stats from a sample of clips, for the whitened variants.
    sample = torch.stack(
        [
            pool_features(
                torch.load(p)["latents"].float().unsqueeze(0),
                torch.ones(1, torch.load(p)["latents"].shape[1], dtype=torch.bool),
            )[0].flatten()
            for p in files[-300:]
        ]
    )
    mu, sd = sample.mean(0), sample.std(0) + 1e-3

    rt = RoundTrip(args.breeze, args.device)
    rng = random.Random(0)

    def sims(feats, original):
        o = pool_features(
            original.float().unsqueeze(0),
            torch.ones(1, original.shape[1], dtype=torch.bool),
        )[0].flatten()
        f = feats.flatten()
        return {
            "raw": F.cosine_similarity(f, o, dim=0).item(),
            "centred": F.cosine_similarity(f - mu, o - mu, dim=0).item(),
            "standardised": F.cosine_similarity(
                (f - mu) / sd, (o - mu) / sd, dim=0
            ).item(),
        }

    conditions = ["truth", "truth-seed2", f"{args.attr}-swap", "sex-swap", "null"]
    results = {c: {"raw": [], "centred": [], "standardised": []} for c in conditions}
    for r in recs:
        a = r["use"]
        truth = TEMPLATES[args.attr].format(
            gender=a["gender"], age=a["age"], value=a[args.attr]
        )
        other = rng.choice([v for v in ATTRS[args.attr] if v != a[args.attr]])
        swapped = TEMPLATES[args.attr].format(
            gender=a["gender"], age=a["age"], value=other
        )
        sex = "man" if a["gender"] == "woman" else "woman"
        sex_swapped = TEMPLATES[args.attr].format(
            gender=sex, age=a["age"], value=a[args.attr]
        )
        prompts = {
            "truth": (truth, 0),
            "truth-seed2": (truth, 1),
            f"{args.attr}-swap": (swapped, 0),
            "sex-swap": (sex_swapped, 0),
            "null": ("Speak clearly and naturally.", 0),
        }
        for cond, (prompt, seed) in prompts.items():
            feats = rt.render(prompt, r["text"], seed=seed)
            if feats is None:
                continue
            for k, v in sims(feats, r["latents"]).items():
                results[cond][k].append(v)
        print(".", end="", flush=True)
    print(f"\n{len(recs)} clips, attribute {args.attr} ({args.labels} labels)\n")
    print(
        f"{'condition':<14}"
        + "".join(f"{k:>14}" for k in ("raw", "centred", "standardised"))
    )
    for cond in conditions:
        print(
            f"{cond:<14}"
            + "".join(
                f"{statistics.mean(results[cond][k]):14.4f}"
                for k in ("raw", "centred", "standardised")
            )
        )
    print("\npaired: truth minus condition, mean (and how often truth wins)")
    for cond in conditions[1:]:
        row = f"{cond:<14}"
        for k in ("raw", "centred", "standardised"):
            d = [t - c for t, c in zip(results["truth"][k], results[cond][k])]
            wins = sum(x > 0 for x in d) / len(d)
            row += f"  {statistics.mean(d):+.4f} ({wins:.2f})"
        print(row)


if __name__ == "__main__":
    main()
