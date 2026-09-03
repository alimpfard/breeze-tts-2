"""Does the tail of the sequence, or a cross-layer readout, carry more?

The backbone is causal: the state at the last frame has seen the whole
clip, the state at the first has heard nothing. Mean pooling weights them
equally. This probes, on the stored 4-layer latents:

    mean       all frames                    (what the captioner uses)
    last1/4/8  the final frames only
    first4     the opening frames only
    mean+last4 both
    x-layer    the same statistic over all four layers concatenated

    python -m caption.pooling --limit 2000 --attrs mood texture sex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.layers import fit_probe
from caption.probe import sex_of

REPO = Path(__file__).resolve().parents[1]

FEATURES = {
    "mean": lambda x: x.mean(1),
    "last1": lambda x: x[:, -1],
    "last4": lambda x: x[:, -4:].mean(1),
    "last8": lambda x: x[:, -8:].mean(1),
    "first4": lambda x: x[:, :4].mean(1),
    "mean+last4": lambda x: torch.cat([x.mean(1), x[:, -4:].mean(1)], -1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument(
        "--attrs", nargs="+", default=["mood", "texture", "brightness", "sex"]
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="jsonl of {id, attrs} to use instead of requested attrs",
    )
    args = parser.parse_args()

    files = sorted((REPO / "data/caption/latents").glob("*.pt"))[: args.limit]
    recs = [torch.load(p) for p in files]
    relabel = {}
    if args.labels:
        for line in args.labels.read_text().splitlines():
            d = json.loads(line)
            relabel[d["id"]] = d["attrs"]
        print(f"using relabelled attrs for {len(relabel)} clips")
    print(f"{len(recs)} clips, layers {recs[0]['layers']}")

    feats = {}
    for name, fn in FEATURES.items():
        per_layer = []
        for r in recs:
            x = r["latents"].float().to(args.device)  # (L, T, D)
            per_layer.append(fn(x))  # (L, D')
        feats[name] = torch.stack(per_layer)  # (N, L, D')

    layers = recs[0]["layers"]
    for attr in args.attrs:
        labels = []
        for r, p in zip(recs, files):
            a = relabel.get(p.stem, r["attrs"]) if relabel else r["attrs"]
            labels.append(sex_of(a["gender"]) if attr == "sex" else a.get(attr))
        counts = Counter(v for v in labels if v is not None)
        keep = sorted(v for v, c in counts.items() if c >= 30)
        idx = {v: i for i, v in enumerate(keep)}
        rows = [i for i, v in enumerate(labels) if v in idx]
        y = torch.tensor([idx[labels[i]] for i in rows], device=args.device)
        major = max(counts[v] for v in keep) / len(rows)
        print(f"\n{attr}: n={len(rows)} k={len(keep)} majority={major:.2f}")
        print(
            f"{'feature':<11}"
            + "".join(f"  L{layer:<4}" for layer in layers)
            + "  x-layer"
        )
        for name, f in feats.items():
            accs = [fit_probe(f[rows, li], y, len(keep)) for li in range(len(layers))]
            cross = fit_probe(f[rows].flatten(1), y, len(keep))
            print(
                f"{name:<11}"
                + "".join(f"  {a:5.2f}" for a in accs)
                + f"  {cross:5.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
