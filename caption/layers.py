"""Which layers and which pooling carry which attribute?

Re-runs the clone-style prefill on the stored codes with hooks on every
backbone layer and keeps pooled statistics per layer:

    mean, std           climate: what the voice is like on average
    early, late         mean over the first and last third of the clip
    delta               mean |x[t+1] - x[t]|: how much the state moves

Then a GPU logistic-regression probe per (layer, stat) for a few attributes.

    python -m caption.layers /path/to/breeze extract --limit 3000
    python -m caption.layers /path/to/breeze probe --attrs mood texture
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breeze_infer.runtime import load_runtime
from breeze_infer.templates import _prepare_segment_batches
from caption.probe import sex_of

REPO = Path(__file__).resolve().parents[1]
STATS = ("mean", "std", "early", "late", "delta")


class AllLayers:
    def __init__(self, tokenizer, model, audio_tokenizer, device):
        self.tokenizer, self.model, self.audio_tokenizer, self.device = (
            tokenizer,
            model,
            audio_tokenizer,
            device,
        )
        self.captured: list[torch.Tensor] = []
        for layer in model.backbone_model.layers:
            layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self.captured.append(out[0] if isinstance(out, tuple) else out)

    @torch.no_grad()
    def __call__(self, text: str, codes: torch.Tensor) -> torch.Tensor:
        segments = [
            {"type": "text", "text": f"[S0]{text}"},
            {"type": "audio", "audio_codes": codes, "append_eos": True},
        ]
        inputs = _prepare_segment_batches(
            self.tokenizer,
            self.audio_tokenizer,
            self.model.config,
            self.device,
            [segments],
        )
        merged = self.model._merge_input_ids_with_input_values(
            inputs["input_ids"],
            inputs["input_values"].long(),
            None,
            text_ids_mask=inputs["text_ids_mask"],
            text_ids_len=inputs["text_ids_len"],
            attention_mask=inputs["attention_mask"],
        )
        self.captured.clear()
        self.model.backbone_model(
            inputs_embeds=merged["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            text_encoder_layer_hidden_states=merged["text_encoder_layer_hidden_states"],
            text_ids_mask=inputs["text_ids_mask"],
        )
        mask = inputs["input_ids"][0] == self.model.config.audio_token_id
        x = torch.stack([h[0, mask] for h in self.captured]).float()  # (28, T, D)
        t = x.shape[1]
        third = max(1, t // 3)
        stats = torch.stack(
            [
                x.mean(1),
                x.std(1),
                x[:, :third].mean(1),
                x[:, -third:].mean(1),
                (x[:, 1:] - x[:, :-1]).abs().mean(1)
                if t > 1
                else torch.zeros_like(x[:, 0]),
            ],
            dim=1,
        )  # (28, 5, D)
        return stats.to(torch.bfloat16).cpu()


def cmd_extract(args):
    src = sorted((REPO / "data/caption/latents").glob("*.pt"))
    if args.limit:
        src = src[: args.limit]
    out = REPO / "data/caption/layerstats"
    out.mkdir(exist_ok=True)
    tokenizer, model, audio_tokenizer = load_runtime(
        args.breeze, device=args.device, attn_implementation="eager"
    )
    extract = AllLayers(tokenizer, model, audio_tokenizer, args.device)
    for i, p in enumerate(src):
        dst = out / p.name
        if dst.exists():
            continue
        r = torch.load(p)
        torch.save({"stats": extract(r["text"], r["codes"]), "attrs": r["attrs"]}, dst)
        if i % 500 == 0:
            print(f"  {i}/{len(src)}", flush=True)
    print("done")


def _fit_fold(xt, yt, xv, yv, k: int) -> float:
    w = torch.zeros(xt.shape[1], k, device=xt.device, requires_grad=True)
    b = torch.zeros(k, device=xt.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ w + b, yt)
        loss = loss + 1e-2 * w.pow(2).sum() / xt.shape[1]
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return ((xv @ w + b).argmax(1) == yv).float().mean().item()


def fit_probe(x: torch.Tensor, y: torch.Tensor, k: int, folds: int = 3) -> float:
    """Multinomial logistic regression, standardised inputs, L2, k-fold."""
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(x.device)
    accs = []
    for f in range(folds):
        is_test = torch.zeros(n, dtype=torch.bool, device=x.device)
        is_test[perm[f::folds]] = True
        mu, sd = x[~is_test].mean(0), x[~is_test].std(0) + 1e-3
        xt, xv = (x[~is_test] - mu) / sd, (x[is_test] - mu) / sd
        accs.append(_fit_fold(xt, y[~is_test], xv, y[is_test], k))
    return sum(accs) / len(accs)


def cmd_probe(args):
    files = sorted((REPO / "data/caption/layerstats").glob("*.pt"))
    recs = [torch.load(p) for p in files]
    stats = (
        torch.stack([r["stats"] for r in recs]).float().to(args.device)
    )  # (N, 28, 5, D)
    print(f"{len(recs)} clips")
    for attr in args.attrs:
        labels = []
        for r in recs:
            a = r["attrs"]
            labels.append(sex_of(a["gender"]) if attr == "sex" else a.get(attr))
        counts = Counter(v for v in labels if v is not None)
        keep = sorted(v for v, c in counts.items() if c >= 30)
        idx = {v: i for i, v in enumerate(keep)}
        rows = [i for i, v in enumerate(labels) if v in idx]
        y = torch.tensor([idx[labels[i]] for i in rows], device=args.device)
        major = max(counts[v] for v in keep) / len(rows)
        print(f"\n{attr}: n={len(rows)} k={len(keep)} majority={major:.2f}")
        print("layer " + " ".join(f"{s:>6}" for s in STATS) + "   mean+std  all5")
        for layer in range(0, 28, args.layer_step):
            row = []
            for si in range(len(STATS)):
                row.append(fit_probe(stats[rows, layer, si], y, len(keep)))
            ms = fit_probe(stats[rows, layer, :2].flatten(1), y, len(keep))
            all5 = fit_probe(stats[rows, layer].flatten(1), y, len(keep))
            print(
                f"L{layer + 1:<4}"
                + " ".join(f"{a:6.2f}" for a in row)
                + f"   {ms:6.2f} {all5:6.2f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("probe")
    p.add_argument(
        "--attrs", nargs="+", default=["mood", "texture", "brightness", "sex"]
    )
    p.add_argument("--layer-step", type=int, default=3)
    args = parser.parse_args()
    {"extract": cmd_extract, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
