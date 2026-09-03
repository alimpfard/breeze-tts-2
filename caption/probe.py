"""Linear probe: do the backbone latents encode the voice attributes at all?

Cheap go/no-go before training a captioner. Mean-pools each layer's states
over time and fits a logistic regression per attribute, 5-fold. If gender is
not near-perfect here, nothing downstream will fix it.

    python -m caption.probe data/caption/probe
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sex_of(gender: str) -> str:
    return "female" if gender in ("woman", "girl", "teenage girl") else "male"


def pool(latents: torch.Tensor) -> np.ndarray:
    """(L, T, D) -> (L, 2D): mean and std over time."""
    x = latents.float()
    return torch.cat([x.mean(1), x.std(1)], dim=-1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--min-per-class", type=int, default=15)
    args = parser.parse_args()

    records = [torch.load(p) for p in sorted(args.dir.glob("*.pt"))]
    print(f"{len(records)} clips")
    layers = records[0]["layers"]
    feats = np.stack([pool(r["latents"]) for r in records])  # (N, L, 2D)
    dim = feats.shape[-1] // 2

    attributes = [
        "sex",
        "age",
        "pitch",
        "brightness",
        "phonation",
        "texture",
        "pace",
        "energy",
        "mood",
        "accent",
    ]
    header = (
        f"{'attribute':<11} {'n':>4} {'k':>2} {'major':>6} | "
        + " ".join(f"L{layer:<4}" for layer in layers)
        + f" | mean-only L{layers[1]}"
    )
    print(header)
    print("-" * len(header))
    for attr in attributes:
        labels, idx = [], []
        for i, r in enumerate(records):
            a = r["attrs"]
            value = sex_of(a["gender"]) if attr == "sex" else a.get(attr)
            if value is not None:
                labels.append(value)
                idx.append(i)
        counts = Counter(labels)
        keep = {k for k, c in counts.items() if c >= args.min_per_class}
        sel = [j for j, lab in enumerate(labels) if lab in keep]
        if len(keep) < 2 or len(sel) < 40:
            print(f"{attr:<11} {len(sel):>4}  (too few)")
            continue
        y = np.array([labels[j] for j in sel])
        rows = np.array([idx[j] for j in sel])
        majority = max(Counter(y).values()) / len(y)
        accs = []
        for li in range(len(layers)):
            x = feats[rows, li]
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)
            )
            accs.append(cross_val_score(clf, x, y, cv=5).mean())
        x = feats[rows, 1, :dim]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        mean_only = cross_val_score(clf, x, y, cv=5).mean()
        print(
            f"{attr:<11} {len(y):>4} {len(keep):>2} {majority:6.2f} | "
            + " ".join(f"{a:5.2f}" for a in accs)
            + f" | {mean_only:5.2f}"
        )


if __name__ == "__main__":
    main()
