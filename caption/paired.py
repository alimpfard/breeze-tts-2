"""Paired renders: does an attribute survive rendering, and where?

Render the same text twice from a template prompt that differs only in one
attribute's word (mood by default), keep the full latent sequences, then
ask probes to read the word back:

  pooled     linear probe on mean+std pooled latents of one render
  delta      linear probe on pooled(A) - pooled(B), the corrector's input
  sequence   the seqprobe GRU over the frames, no pooling

Labels here are exact (the word is what was rendered), so the gap between
these and the seqprobe on the training set is label noise, and the gap
between sequence and pooled is what the temporal pattern adds.

    python -m caption.paired render /path/to/breeze --shard 0/2 --device cuda:0
    python -m caption.paired probe
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.consistency import TEMPLATES
from caption.data import ATTRS
from caption.layers import fit_probe
from caption.model import pool_features
from caption.seqprobe import SeqProbe

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/caption/paired"


def clips(n: int, seed: int) -> list[Path]:
    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    random.Random(seed).shuffle(files)
    return files[:n]


def cmd_render(args) -> None:
    from caption.rl import RoundTrip

    si, sn = (int(x) for x in args.shard.split("/"))
    OUT.mkdir(exist_ok=True)
    rt = RoundTrip(args.breeze, args.device)
    values = ATTRS[args.attr]
    for i, p in enumerate(clips(args.n, args.seed)):
        if i % sn != si:
            continue
        if (OUT / f"{p.stem}_b.pt").exists():
            continue
        r = torch.load(p)
        a = r["attrs"]
        rng = random.Random(f"{args.seed}:{p.stem}")
        va, vb = rng.sample(values, 2)
        for tag, v in (("a", va), ("b", vb)):
            desc = TEMPLATES[args.attr].format(gender=a["gender"], age=a["age"], value=v)
            lat = rt.render(desc, r["text"], seed=0, raw=True)
            if lat is None:
                break
            torch.save(
                {"latents": lat.to(torch.bfloat16).cpu(), "value": v, "text": r["text"], "id": p.stem},
                OUT / f"{p.stem}_{tag}.pt",
            )
        if i % 50 == 0:
            print(f"  {i}/{args.n}", flush=True)
    print("done")


def cmd_probe(args) -> None:
    dev = args.device
    recs = {}
    for p in sorted(OUT.glob("*_a.pt")):
        q = OUT.parent / "paired" / (p.stem[:-2] + "_b.pt")
        if q.exists():
            recs[p.stem[:-2]] = (torch.load(p), torch.load(q))
    ids = sorted(recs)
    values = sorted({r[0]["value"] for r in recs.values()} | {r[1]["value"] for r in recs.values()})
    idx = {v: i for i, v in enumerate(values)}
    print(f"{len(ids)} pairs, {len(values)} values, chance {1 / len(values):.2f}")

    def pooled(r):
        lat = r["latents"].float().unsqueeze(0)
        return pool_features(lat, torch.ones(1, lat.shape[2], dtype=torch.bool))[0].flatten()

    xa = torch.stack([pooled(recs[i][0]) for i in ids]).to(dev)
    xb = torch.stack([pooled(recs[i][1]) for i in ids]).to(dev)
    ya = torch.tensor([idx[recs[i][0]["value"]] for i in ids], device=dev)
    yb = torch.tensor([idx[recs[i][1]["value"]] for i in ids], device=dev)
    k = len(values)
    # Both renders as independent samples.
    print(f"pooled   {fit_probe(torch.cat([xa, xb]), torch.cat([ya, yb]), k):.3f}")
    # The corrector's view: target minus render, predict the target's word.
    print(f"delta    {fit_probe(torch.cat([xa - xb, xb - xa]), torch.cat([ya, yb]), k):.3f}")
    # Sanity: the same delta asked for the *other* render's word.
    print(f"delta->other {fit_probe(torch.cat([xa - xb, xb - xa]), torch.cat([yb, ya]), k):.3f}")

    # Sequence probe, split by pair so both renders of a text stay together.
    items = []
    for i in ids:
        for r in recs[i]:
            items.append({"latents": r["latents"], "y": idx[r["value"]], "id": i})
    random.Random(0).shuffle(ids)
    held_ids = set(ids[: len(ids) // 5])
    train = [it for it in items if it["id"] not in held_ids]
    held = [it for it in items if it["id"] in held_ids]
    sample = torch.cat([it["latents"].float().flatten(1) for it in train[:200]], 1)
    mu, sd = sample.mean(1).view(-1, 1, 1), sample.std(1).view(-1, 1, 1) + 1e-3
    mu, sd = mu.view(4, 1, -1), sd.view(4, 1, -1)

    def collate(batch):
        t = max(it["latents"].shape[1] for it in batch)
        x = torch.zeros(len(batch), 4, t, batch[0]["latents"].shape[-1])
        m = torch.zeros(len(batch), t, dtype=torch.bool)
        for j, it in enumerate(batch):
            z = (it["latents"].float() - mu) / sd
            x[j, :, : z.shape[1]] = z
            m[j, : z.shape[1]] = True
        y = torch.tensor([it["y"] for it in batch])
        return x.to(dev), m.to(dev), y.to(dev)

    model = SeqProbe(4, train[0]["latents"].shape[-1], {"v": k}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for j in range(0, len(train), 16):
            x, m, y = collate(train[j : j + 16])
            loss = nn.functional.cross_entropy(model(x, m)["v"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for j in range(0, len(held), 32):
                x, m, y = collate(held[j : j + 32])
                correct += (model(x, m)["v"].argmax(-1) == y).sum().item()
        acc = correct / len(held)
        best = max(best, acc)
        if epoch % 5 == 4:
            print(f"  epoch {epoch + 1} held-out {acc:.3f} (best {best:.3f})", flush=True)
    print(f"sequence {best:.3f} (best epoch, {len(held)} held-out renders)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("breeze", type=Path)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--attr", default="mood")
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--seed", type=int, default=11)
    r.set_defaults(fn=cmd_render)
    q = sub.add_parser("probe")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--epochs", type=int, default=40)
    q.set_defaults(fn=cmd_probe)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
