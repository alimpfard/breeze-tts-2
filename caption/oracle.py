"""Loss floor: the same LM fed the ground-truth attributes instead of audio.

The captioner's held-out loss mixes two things: what it failed to read from
the audio, and how the LLM happened to phrase the description, which no
audio can predict. Fine-tuning the same LM on attributes -> prose with the
same split gives the phrasing entropy; the captioner's loss minus this is
the recoverable part.

    python -m caption.oracle data/caption/latents
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.model import LM_NAME, PROMPT
from caption.train import load_records


def encode(tok, items, device):
    prompts = ["Attributes: " + json.dumps(it["attrs"]) + "\n" + PROMPT for it in items]
    targets = [it["prose"] + tok.eos_token for it in items]
    ids, labels = [], []
    for p, t in zip(prompts, targets):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ti = tok(t, add_special_tokens=False)["input_ids"][:128]
        ids.append(pi + ti)
        labels.append([-100] * len(pi) + ti)
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id or tok.eos_token_id
    ids_t = torch.tensor([x + [pad] * (n - len(x)) for x in ids], device=device)
    lab_t = torch.tensor([x + [-100] * (n - len(x)) for x in labels], device=device)
    att = (lab_t != -100) | (ids_t != pad)
    return ids_t, att.long(), lab_t


def loss_of(model, ids, att, lab):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Same shuffle as caption.train so the held-out set is identical.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.dirs)
    random.shuffle(records)
    n_hold = max(32, int(len(records) * args.holdout))
    held, train = records[:n_hold], records[n_hold:]
    for r in records:
        r.pop("latents", None)
    print(f"{len(train)} train, {len(held)} held out", flush=True)

    tok = AutoTokenizer.from_pretrained(LM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LM_NAME, dtype=torch.bfloat16).to(
        args.device
    )
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    total = int(math.ceil(len(train) / args.batch) * args.epochs)
    warm = max(20, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warm)
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, s / total)))
        ),
    )

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(held), args.batch):
                losses.append(
                    loss_of(
                        model, *encode(tok, held[i : i + args.batch], args.device)
                    ).item()
                )
        model.train()
        return sum(losses) / len(losses)

    step, started = 0, time.time()
    model.train()
    while step < total:
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            if step >= total:
                break
            loss = loss_of(model, *encode(tok, train[i : i + args.batch], args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0 or step == total:
                print(
                    f"step {step}/{total} loss {loss.item():.3f} held-out {evaluate():.3f} "
                    f"{(time.time() - started) / 60:.1f}min",
                    flush=True,
                )
    print(f"oracle held-out loss {evaluate():.3f}")


if __name__ == "__main__":
    main()
