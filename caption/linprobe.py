"""Learned linear pooling: let the probe choose its own (layer, time) mix.

Frames are binned into K relative-time slots (mean within slot) so clips of
different lengths line up, giving (L, K, D) per clip. Three linear readouts:

    mean      LR on the time-mean, per layer and across layers (baseline)
    factored  logits = (sum_lk alpha_lk x_lk) W + b, alpha and W learned
              jointly: a learned linear pooling over layers and time with
              one shared readout. "(l1 - l2 + l3) / 3" is an alpha.
    full      LR on the whole flattened (L*K*D) tensor, heavy L2: the
              unconstrained linear readout, upper bound for anything linear.

    python -m caption.linprobe --limit 2000 --attrs mood texture
    python -m caption.linprobe --labels data/caption/relabel/all.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools

from caption.layers import fit_probe
from caption.probe import sex_of

REPO = Path(__file__).resolve().parents[1]


def binned(x: torch.Tensor, k: int) -> torch.Tensor:
    """(L, T, D) -> (L, K, D): mean within K equal relative-time slots."""
    t = x.shape[1]
    edges = torch.linspace(0, t, k + 1).round().long().tolist()
    slots = []
    for a, b in itertools.pairwise(edges):
        b = max(b, a + 1)
        slots.append(x[:, a:b].mean(1))
    return torch.stack(slots, 1)


def fit_factored(x, y, k, folds=3, steps=400, wd=1e-2, lr=1e-2):
    """x (N, L, K, D) standardised; alpha (L, K) and W (D, k) learned."""
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(x.device)
    accs, alphas = [], []
    for f in range(folds):
        test = torch.zeros(n, dtype=torch.bool, device=x.device)
        test[perm[f::folds]] = True
        xt, yt, xv, yv = x[~test], y[~test], x[test], y[test]
        alpha = torch.full(
            x.shape[1:3],
            1.0 / (x.shape[1] * x.shape[2]),
            device=x.device,
            requires_grad=True,
        )
        w = torch.zeros(x.shape[-1], k, device=x.device, requires_grad=True)
        b = torch.zeros(k, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([alpha, w, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pooled = torch.einsum("nlkd,lk->nd", xt, alpha)
            loss = (
                F.cross_entropy(pooled @ w + b, yt) + wd * w.pow(2).sum() / x.shape[-1]
            )
            loss.backward()
            opt.step()
        with torch.no_grad():
            pooled = torch.einsum("nlkd,lk->nd", xv, alpha)
            accs.append(((pooled @ w + b).argmax(1) == yv).float().mean().item())
            alphas.append(alpha.detach().clone())
    return sum(accs) / folds, torch.stack(alphas).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument(
        "--attrs", nargs="+", default=["mood", "texture", "brightness", "pitch", "sex"]
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
    layers = recs[0]["layers"]
    x_all = torch.stack([binned(r["latents"].float(), args.slots) for r in recs]).to(
        args.device
    )
    mu, sd = x_all.mean(0, keepdim=True), x_all.std(0, keepdim=True) + 1e-3
    x_all = (x_all - mu) / sd  # (N, L, K, D)
    print(f"{len(recs)} clips, layers {layers}, {args.slots} time slots")

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
        x = x_all[rows]
        k = len(keep)
        major = max(counts[v] for v in keep) / len(rows)
        print(f"\n{attr}: n={len(rows)} k={k} majority={major:.2f}")
        mean_layer = [fit_probe(x[:, li].mean(1), y, k) for li in range(len(layers))]
        mean_x = fit_probe(x.mean(2).flatten(1), y, k)
        print(
            "  mean       "
            + " ".join(f"L{layer}={a:.2f}" for layer, a in zip(layers, mean_layer))
            + f"  x-layer={mean_x:.2f}"
        )
        for wd in (1e-2, 1e-1):
            acc, alpha = fit_factored(x, y, k, wd=wd)
            print(f"  factored   wd={wd:g} acc={acc:.2f}")
        a = alpha / alpha.abs().sum()
        print("    alpha (rows=layers, cols=time slots), normalised:")
        for layer, row in zip(layers, a.tolist()):
            print(f"      L{layer:<3}" + " ".join(f"{v:+.2f}" for v in row))
        full = fit_probe(x.flatten(1), y, k)
        print(f"  full       {x.flatten(1).shape[1]} dims acc={full:.2f}", flush=True)


if __name__ == "__main__":
    main()
