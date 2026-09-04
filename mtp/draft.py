"""Draft heads for the depth decoder: all 15 codebooks at once, from what
the depth decoder itself is given (backbone hidden state + codebook 0).

Reports, on held-out clips with the depth decoder's own distributions:
  top-1 agreement per codebook, and the expected accepted prefix length
  under speculative sampling (per position, P(accept) = sum_v min(q_v, p_v)
  with q the draft and p the target; the prefix ends at the first reject).

    python -m mtp.draft --epochs 3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/mtp"
V = 2051
K = 15  # codebooks 1..15
HIDDEN = 2048


class Draft(nn.Module):
    """hidden (2048) + embed(cb0) -> trunk -> 15 heads. Medusa-shaped:
    every head sees the same trunk state, no dependence between codebooks,
    so one forward gives every draft token."""

    def __init__(self, width: int = 1536, depth: int = 2, cb0_dim: int = 512):
        super().__init__()
        self.cb0 = nn.Embedding(V, cb0_dim)
        self.inp = nn.Linear(HIDDEN + cb0_dim, width)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.heads = nn.Linear(width, K * V)

    def forward(self, hidden: torch.Tensor, cb0: torch.Tensor) -> torch.Tensor:
        x = self.inp(torch.cat([hidden, self.cb0(cb0)], dim=-1))
        for b in self.blocks:
            x = x + b(x)
        return self.heads(self.norm(x)).view(-1, K, V)


def load(files, with_probs=False):
    hs, cbs, ys, ps = [], [], [], []
    for p in files:
        r = torch.load(p)
        h = r["hidden"].float()
        c = r["codes"].long()
        n = min(h.shape[0], c.shape[0])
        hs.append(h[:n])
        cbs.append(c[:n, 0])
        ys.append(c[:n, 1:])
        if with_probs:
            ps.append(r["target_probs"][:n].float())
    out = (torch.cat(hs), torch.cat(cbs), torch.cat(ys))
    return out + ((torch.cat(ps),) if with_probs else ())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=DATA / "draft.pt")
    args = parser.parse_args()
    dev = args.device

    files = sorted(DATA.glob("*.pt"))
    files = [f for f in files if f.name != "draft.pt"]
    evalf = [f for f in files if "target_probs" in torch.load(f, map_location="cpu").keys()]
    trainf = [f for f in files if f not in set(evalf)]
    print(f"{len(trainf)} train clips, {len(evalf)} eval clips (with target distributions)")
    h, cb0, y = load(trainf)
    mu, sd = h.mean(0), h.std(0) + 1e-3
    print(f"{h.shape[0]} train frames")
    he, cbe, ye, pe = load(evalf, with_probs=True)
    print(f"{he.shape[0]} eval frames")

    model = Draft(args.width, args.depth).to(dev)
    print(f"draft params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n = h.shape[0]
    steps = args.epochs * (n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    mu_d, sd_d = mu.to(dev), sd.to(dev)

    def evaluate():
        model.eval()
        top1 = torch.zeros(K)
        acc_len = 0.0
        acc_pos = torch.zeros(K)
        count = 0
        with torch.no_grad():
            for i in range(0, he.shape[0], 1024):
                hb = ((he[i : i + 1024].to(dev) - mu_d) / sd_d)
                logits = model(hb, cbe[i : i + 1024].to(dev))
                q = torch.softmax(logits.float() / args.temperature, dim=-1)  # draft dist
                p = pe[i : i + 1024].to(dev)  # target dist at temperature
                yb = ye[i : i + 1024].to(dev)
                top1 += (logits.argmax(-1) == yb).float().sum(0).cpu()
                # P(accept at position k) under speculative sampling, draft
                # sampled from q, verified against p.
                a = torch.minimum(q, p).sum(-1)  # (B, K)
                acc_pos += a.sum(0).cpu()
                # expected accepted prefix length: sum_k prod_{j<=k} a_j
                acc_len += torch.cumprod(a, dim=-1).sum(-1).sum().item()
                count += yb.shape[0]
        model.train()
        return top1 / count, acc_pos / count, acc_len / count

    started = time.perf_counter()
    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i : i + args.batch]
            hb = ((h[idx].to(dev) - mu_d) / sd_d)
            logits = model(hb, cb0[idx].to(dev))
            loss = F.cross_entropy(logits.reshape(-1, V), y[idx].to(dev).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {(time.perf_counter() - started) / 60:.1f}min", flush=True)
        top1, acc_pos, acc_len = evaluate()
        print(f"epoch {epoch + 1}: top-1 per codebook " + " ".join(f"{x:.2f}" for x in top1.tolist()), flush=True)
        print(f"         P(accept) per position " + " ".join(f"{x:.2f}" for x in acc_pos.tolist()), flush=True)
        print(f"         expected accepted prefix {acc_len:.2f} of {K}  -> depth steps/frame ~ {1 + (K - acc_len):.1f} instead of {K}", flush=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "width": args.width, "depth": args.depth}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
