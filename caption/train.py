"""Train the captioner on (latents, prose) pairs from caption.gen.

python -m caption.train data/caption/latents --out data/caption/captioner
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.model import (
    Captioner,
    ProjectorConfig,
    attr_targets,
    build_attr_vocab,
    collate,
)


def load_records(dirs: list[Path]) -> list[dict]:
    records = []
    for d in dirs:
        for p in sorted(d.glob("*.pt")):
            r = torch.load(p)
            records.append(
                {
                    "latents": r["latents"],
                    "prose": r["prose"],
                    "attrs": r["attrs"],
                    "id": p.stem,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("data/caption/captioner"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr-lm", type=float, default=2e-5)
    parser.add_argument("--lr-proj", type=float, default=3e-4)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument(
        "--freeze-frac",
        type=float,
        default=0.3,
        help="fraction of steps with the LM frozen, so the only way down is "
        "through the prefix",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projector", choices=("pool", "attn", "gru"), default="pool")
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument(
        "--init", type=Path, help="start from a saved captioner instead of the base LM"
    )
    parser.add_argument(
        "--max-per-dir", type=int, default=0, help="random subsample of each dir"
    )
    parser.add_argument(
        "--prose",
        type=Path,
        help="jsonl {id, attrs, prose} (caption.rewrite) replacing the records' "
        "requested labels with relabelled ones; records without an entry are dropped",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = []
    for d in args.dirs:
        recs = load_records([d])
        if args.max_per_dir and len(recs) > args.max_per_dir:
            recs = random.sample(recs, args.max_per_dir)
        print(f"{d}: {len(recs)} records", flush=True)
        records += recs
    if args.prose:
        over = {}
        for line in args.prose.read_text().splitlines():
            d = json.loads(line)
            over[d["id"]] = d
        kept = []
        for r in records:
            if r["id"] in over:
                r["prose"], r["attrs"] = over[r["id"]]["prose"], over[r["id"]]["attrs"]
                kept.append(r)
        print(f"{len(kept)}/{len(records)} records have relabelled prose", flush=True)
        records = kept
    random.shuffle(records)
    n_hold = max(32, int(len(records) * args.holdout))
    held, train = records[:n_hold], records[n_hold:]
    print(f"{len(train)} train, {len(held)} held out", flush=True)

    if args.init:
        model = Captioner.load(args.init, args.device).train()
        vocab = model.attr_vocab
    else:
        vocab = build_attr_vocab(train)
        model = Captioner(cfg=ProjectorConfig(kind=args.projector), attr_vocab=vocab).to(
            args.device
        )
        model.resampler.set_input_stats(
            [r["latents"].to(args.device) for r in train[:1000]]
        )
    model.lm.gradient_checkpointing_enable()
    tok = model.tokenizer
    params = [
        {"params": model.resampler.parameters(), "lr": args.lr_proj},
        {"params": model.aux_heads.parameters(), "lr": args.lr_proj},
        {"params": model.lm.parameters(), "lr": args.lr_lm},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = int(steps_per_epoch * args.epochs)
    warm = max(20, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warm)
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, s / total)))
        ),
    )

    def evaluate() -> tuple[float, float]:
        """Held-out loss with the right prefix and with prefixes shuffled
        across the batch. The gap is how much the model reads the latents."""
        model.eval()
        real, shuffled = [], []
        acc: dict[str, tuple[int, int]] = {}
        with torch.no_grad():
            lat, fm, _, _ = (b.to(args.device) for b in collate(held[:8], tok))
            pre = model.resampler(lat.float(), fm).flatten(1)
            cos = torch.nn.functional.cosine_similarity(pre[:1], pre[1:], dim=-1)
            print(f"  prefix cosine across clips {cos.mean():.4f}", flush=True)
            for i in range(0, len(held), args.batch):
                chunk = held[i : i + args.batch]
                lat, fm, ids, am = (b.to(args.device) for b in collate(chunk, tok))
                real.append(model(lat, fm, ids, am)[0].item())
                perm = torch.roll(torch.arange(lat.shape[0]), 1)
                shuffled.append(model(lat[perm], fm[perm], ids, am)[0].item())
                pooled = torch.nn.functional.layer_norm(
                    model._prefix(lat, fm)[:, : model.cfg.queries].float().mean(1),
                    (model.lm.config.hidden_size,),
                )
                tg = attr_targets(chunk, vocab)
                for k, head in model.aux_heads.items():
                    t = tg[k].to(args.device)
                    ok = t != -100
                    if ok.any():
                        hits = (head(pooled).argmax(-1)[ok] == t[ok]).sum().item()
                        acc[k] = (
                            acc.get(k, (0, 0))[0] + hits,
                            acc.get(k, (0, 0))[1] + ok.sum().item(),
                        )
            print(
                "  attr acc "
                + " ".join(f"{k}={h / n:.2f}" for k, (h, n) in acc.items()),
                flush=True,
            )
        model.train()
        return sum(real) / len(real), sum(shuffled) / len(shuffled)

    freeze_until = int(total * args.freeze_frac)
    for p in model.lm.parameters():
        p.requires_grad_(freeze_until == 0)
    print(f"LM frozen for the first {freeze_until} of {total} steps", flush=True)

    step = 0
    started = time.time()
    model.train()
    while step < total:
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            if step >= total:
                break
            chunk = train[i : i + args.batch]
            batch = [b.to(args.device) for b in collate(chunk, tok)]
            targets = {
                k: v.to(args.device) for k, v in attr_targets(chunk, vocab).items()
            }
            lm_loss, aux = model(*batch, attr_targets=targets)
            loss = lm_loss + args.aux_weight * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == freeze_until:
                for p in model.lm.parameters():
                    p.requires_grad_(True)
                print("  LM unfrozen", flush=True)
            if step % 25 == 0 or step == 1:
                w = torch.softmax(model.resampler.layer_logits, 0).tolist()
                print(
                    f"step {step}/{total} loss {lm_loss.item():.3f} aux {aux.item():.3f} "
                    f"layers {' '.join(f'{x:.2f}' for x in w)} {(time.time() - started) / 60:.1f}min",
                    flush=True,
                )
            if step % 200 == 0 or step == total:
                real, shuffled = evaluate()
                print(
                    f"  held-out loss {real:.3f}  shuffled-prefix {shuffled:.3f}  "
                    f"(gap {shuffled - real:+.3f})",
                    flush=True,
                )

    model.save(args.out)
    # A few held-out samples, truth vs caption, for eyeballing.
    model.eval()
    sample = held[:8]
    latents, frame_mask, _, _ = collate(sample, tok)
    captions = model.generate(
        latents.to(args.device), frame_mask.to(args.device), do_sample=False
    )
    for r, c in zip(sample, captions):
        print(f"\ntruth:   {r['prose']}\ncaption: {c}")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
