"""Aux-head accuracy of one or more captioners on a set of .pt records.

    python -m caption.auxeval data/caption/captioner_v2 data/caption/captioner_v5 \
        --files data/caption/minimal/0009*.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.model import Captioner, attr_targets, collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captioners", type=Path, nargs="+")
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    recs = [torch.load(p) for p in args.files]
    print(f"{len(recs)} records")
    for path in args.captioners:
        model = Captioner.load(path, args.device)
        acc: dict[str, tuple[int, int]] = {}
        with torch.no_grad():
            for i in range(0, len(recs), 16):
                chunk = recs[i : i + 16]
                lat, fm, _, _ = collate(chunk, model.tokenizer)
                lat, fm = lat.to(args.device), fm.to(args.device)
                pooled = F.layer_norm(
                    model._prefix(lat, fm)[:, : model.cfg.queries].float().mean(1),
                    (model.lm.config.hidden_size,),
                )
                tg = attr_targets(chunk, model.attr_vocab)
                for k, head in model.aux_heads.items():
                    t = tg[k].to(args.device)
                    ok = t != -100
                    if ok.any():
                        hits = (head(pooled).argmax(-1)[ok] == t[ok]).sum().item()
                        h, n = acc.get(k, (0, 0))
                        acc[k] = (h + hits, n + ok.sum().item())
        print(
            f"{path.name:<16} "
            + " ".join(f"{k}={h / n:.2f}({n})" for k, (h, n) in sorted(acc.items()))
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
