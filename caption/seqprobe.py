"""Sequence probe: a small bidirectional GRU over the raw frames, no pooling.

The direct test of whether an attribute lives in the temporal pattern of the
latents rather than in their average. Multi-task over the attributes with
missing labels masked, learned layer mix, early stopping on held-out
accuracy. Reports held-out accuracy per attribute next to the linear
mean-pooling probe on the same split.

    python -m caption.seqprobe --limit 5000
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.layers import fit_probe
from caption.probe import sex_of

REPO = Path(__file__).resolve().parents[1]
ATTRS = ("mood", "texture", "brightness", "pitch", "energy", "sex")


class SeqProbe(nn.Module):
    def __init__(
        self, n_layers: int, dim: int, heads: dict[str, int], hidden: int = 256
    ):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.att = nn.Linear(2 * hidden, 1)
        self.heads = nn.ModuleDict(
            {k: nn.Linear(4 * hidden, n) for k, n in heads.items()}
        )
        self.drop = nn.Dropout(0.3)

    def forward(self, x, mask):
        w = torch.softmax(self.layer_logits, 0).view(1, -1, 1, 1)
        h = self.proj((x * w).sum(1))
        h, _ = self.gru(h)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        attended = (h * a.unsqueeze(-1)).sum(1)
        mean = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        z = self.drop(torch.cat([attended, mean], -1))
        return {k: head(z) for k, head in self.heads.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--holdout", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    files = sorted((REPO / "data/caption/latents").glob("*.pt"))[: args.limit]
    recs = [torch.load(p) for p in files]
    random.Random(0).shuffle(recs)
    held, train = recs[: args.holdout], recs[args.holdout :]

    vocab = {}
    for attr in ATTRS:
        counts = Counter(
            sex_of(r["attrs"]["gender"]) if attr == "sex" else r["attrs"].get(attr)
            for r in train
        )
        vocab[attr] = sorted(v for v, c in counts.items() if v is not None and c >= 30)

    def targets(items):
        out = {}
        for attr in ATTRS:
            idx = {v: i for i, v in enumerate(vocab[attr])}
            out[attr] = torch.tensor(
                [
                    idx.get(
                        sex_of(r["attrs"]["gender"])
                        if attr == "sex"
                        else r["attrs"].get(attr),
                        -100,
                    )
                    for r in items
                ]
            )
        return out

    # Standardise per (layer, dim) with training-set statistics.
    sample = torch.cat([r["latents"].float() for r in train[:800]], dim=1)
    mu, sd = sample.mean(1, keepdim=True), sample.std(1, keepdim=True) + 1e-3

    def collate(items):
        t = max(r["latents"].shape[1] for r in items)
        x = torch.zeros(len(items), items[0]["latents"].shape[0], t, sample.shape[-1])
        m = torch.zeros(len(items), t, dtype=torch.bool)
        for i, r in enumerate(items):
            z = (r["latents"].float() - mu) / sd
            x[i, :, : z.shape[1]] = z
            m[i, : z.shape[1]] = True
        return (
            x.to(args.device),
            m.to(args.device),
            {k: v.to(args.device) for k, v in targets(items).items()},
        )

    model = SeqProbe(
        train[0]["latents"].shape[0],
        sample.shape[-1],
        {k: len(v) for k, v in vocab.items()},
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    print(
        f"{len(train)} train, {len(held)} held out; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    def evaluate():
        model.eval()
        hits, n = Counter(), Counter()
        with torch.no_grad():
            for i in range(0, len(held), args.batch):
                x, m, y = collate(held[i : i + args.batch])
                out = model(x, m)
                for k in ATTRS:
                    ok = y[k] != -100
                    hits[k] += (out[k].argmax(-1)[ok] == y[k][ok]).sum().item()
                    n[k] += ok.sum().item()
        model.train()
        return {k: hits[k] / max(1, n[k]) for k in ATTRS}

    best = {k: 0.0 for k in ATTRS}
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            x, m, y = collate(train[i : i + args.batch])
            out = model(x, m)
            loss = sum(
                F.cross_entropy(out[k], y[k], ignore_index=-100)
                for k in ATTRS
                if (y[k] != -100).any()
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = evaluate()
        best = {k: max(best[k], acc[k]) for k in ATTRS}
        print(
            f"epoch {epoch + 1:2d} " + " ".join(f"{k}={acc[k]:.2f}" for k in ATTRS),
            flush=True,
        )
    print("best  " + " ".join(f"{k}={best[k]:.2f}" for k in ATTRS))

    # Linear mean-pool baseline on the same held-out split, for reference.
    print("\nlinear mean-pool probe on the same clips (3-fold over train+held):")
    allrecs = train + held
    feats = torch.stack(
        [((r["latents"].float() - mu) / sd).mean(1).flatten() for r in allrecs]
    ).to(args.device)
    for attr in ATTRS:
        idx = {v: i for i, v in enumerate(vocab[attr])}
        labels = [
            idx.get(
                sex_of(r["attrs"]["gender"]) if attr == "sex" else r["attrs"].get(attr),
                -100,
            )
            for r in allrecs
        ]
        rows = [i for i, v in enumerate(labels) if v != -100]
        y = torch.tensor([labels[i] for i in rows], device=args.device)
        print(f"  {attr}={fit_probe(feats[rows], y, len(vocab[attr])):.2f}", end="")
    print()


if __name__ == "__main__":
    main()
