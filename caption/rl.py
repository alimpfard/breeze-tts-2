"""RL for the captioner: reward descriptions by how well Breeze explains
the original audio under them, no attribute labels involved.

GRPO-style. For each clip, sample K captions from the policy, score each,
normalise rewards within the group, and push the policy toward the ones
Breeze prefers, with a KL leash to the SFT captioner so it stays prose.

Rewards:
  likelihood  -NLL of the original codes under the caption (voice_invert's
              scorer, first --window seconds). One prefill, ~40ms, exact.
  roundtrip   (--roundtrip) render the caption with Breeze, pool the
              rendered clip's latents, cosine to the original's. A
              generation per sample, ~2s, stochastic; weighted in on top.

    python -m caption.rl data/caption/captioner_v2 /path/to/breeze \
        --out data/caption/captioner_rl --clips 3000 --epochs 2
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.model import Captioner, collate, pool_features
from voice_invert import VoiceScorer

REPO = Path(__file__).resolve().parents[1]


def sample_captions(model: Captioner, latents, frame_mask, k: int, max_new: int):
    """Returns token ids (B*K, T) with -100 past EOS and the prefix embeds."""
    prefix = model._prefix(latents, frame_mask)  # (B, P, D)
    prefix = prefix.repeat_interleave(k, dim=0)
    attn = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
    tok = model.tokenizer
    with torch.no_grad():
        out = model.lm.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    # Mask everything after (and excluding) the first EOS; keep the EOS
    # itself as a scored token so the policy learns to stop.
    ids = out.clone()
    mask = torch.ones_like(ids, dtype=torch.bool)
    for i in range(ids.shape[0]):
        eos = (ids[i] == tok.eos_token_id).nonzero()
        if len(eos):
            mask[i, eos[0, 0] + 1 :] = False
    return ids, mask, prefix


def token_logprobs(model: Captioner, prefix, ids, mask):
    """Sum of log-probs of the sampled tokens given the prefix, per row."""
    embed = model.lm.get_input_embeddings()
    inputs = torch.cat([prefix, embed(ids)], dim=1)
    attn = torch.cat(
        [
            torch.ones(prefix.shape[:2], dtype=torch.long, device=ids.device),
            mask.long(),
        ],
        dim=1,
    )
    logits = model.lm(inputs_embeds=inputs, attention_mask=attn).logits.float()
    # position P-1 predicts ids[:, 0]; position P+t-1 predicts ids[:, t]
    p = prefix.shape[1]
    pred = logits[:, p - 1 : p - 1 + ids.shape[1]]
    lp = torch.gather(F.log_softmax(pred, -1), -1, ids.unsqueeze(-1)).squeeze(-1)
    return lp * mask, mask.sum(1)


class RoundTrip:
    """Render a caption with Breeze and compare pooled latents."""

    def __init__(self, breeze: Path, device: str):
        from caption.gen import LatentExtractor, build_runtime

        self.tokenizer, self.model, self.audio_tokenizer, self.runtime = build_runtime(
            breeze, device, fast=True
        )
        self.extract = LatentExtractor(
            self.tokenizer, self.model, self.audio_tokenizer, device
        )
        self.device = device

    def __call__(self, caption: str, text: str, original: torch.Tensor) -> float:
        import numpy as np

        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs

        request = {"id": "rt", "text": text, "instruction": caption, "speaker": "S0"}
        set_all_seeds(0)
        inputs = prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            get_template("tts_instruction"),
            guidance_scale=4.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        parts = [
            np.asarray(c.codes)
            for c in self.runtime.iter_audio_chunks(inputs, request_id="rt")
            if c.codes is not None and len(c.codes)
        ]
        if not parts:
            return -1.0
        codes = torch.as_tensor(np.concatenate(parts), dtype=torch.int16)
        lat = self.extract(text, codes)
        a = pool_features(
            lat.float().unsqueeze(0), torch.ones(1, lat.shape[1], dtype=torch.bool)
        )[0]
        b = pool_features(
            original.float().unsqueeze(0),
            torch.ones(1, original.shape[1], dtype=torch.bool),
        )[0]
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captioner", type=Path)
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "data/caption/captioner_rl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scorer-device", default="cuda:1")
    parser.add_argument("--clips", type=int, default=3000)
    parser.add_argument("--holdout", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=8, help="clips per step")
    parser.add_argument("--k", type=int, default=6, help="samples per clip")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr-proj", type=float, default=1e-5)
    parser.add_argument("--kl", type=float, default=0.05)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument(
        "--roundtrip", type=float, default=0.0, help="weight of the roundtrip reward"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    random.shuffle(files)
    files = files[: args.clips + args.holdout]
    recs = [torch.load(p) for p in files]
    held, train = recs[: args.holdout], recs[args.holdout :]
    print(f"{len(train)} train clips, {len(held)} held out, K={args.k}", flush=True)

    policy = Captioner.load(args.captioner, args.device)
    policy.train()
    # No gradient checkpointing: it disables the KV cache and makes the
    # sampling step quadratic. The batch is small enough without it.
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    tok = policy.tokenizer
    scorer = VoiceScorer(args.breeze, args.scorer_device, window_seconds=args.window)
    roundtrip = RoundTrip(args.breeze, args.scorer_device) if args.roundtrip else None

    def reward(caption: str, r: dict) -> float:
        caption = caption.strip()
        if len(caption.split()) < 3:
            return -1e3  # rewards are around -70; this must lose the group
        s = -scorer.score(caption, r["text"], r["codes"]).total
        if roundtrip is not None:
            s = s + args.roundtrip * roundtrip(caption, r["text"], r["latents"])
        return s

    def held_out_eval() -> tuple[float, float]:
        """Mean reward of greedy captions on held-out, policy vs reference."""
        policy.eval()
        outs = []
        for model in (policy, ref):
            total = 0.0
            for i in range(0, len(held), 16):
                chunk = held[i : i + 16]
                lat, fm, _, _ = collate(chunk, tok)
                caps = model.generate(
                    lat.to(args.device), fm.to(args.device), do_sample=False
                )
                total += sum(
                    -scorer.score(c, r["text"], r["codes"]).total
                    for c, r in zip(caps, chunk)
                )
            outs.append(total / len(held))
        policy.train()
        return outs[0], outs[1]

    opt = torch.optim.AdamW(
        [
            {"params": policy.resampler.parameters(), "lr": args.lr_proj},
            {"params": policy.lm.parameters(), "lr": args.lr},
        ],
        weight_decay=0.0,
    )
    steps = int(len(train) / args.batch * args.epochs)
    print(f"held-out reward before: policy {held_out_eval()[0]:.3f}", flush=True)
    started = time.time()
    for step in range(steps):
        chunk = random.sample(train, args.batch)
        lat, fm, _, _ = collate(chunk, tok)
        lat, fm = lat.to(args.device), fm.to(args.device)
        policy.eval()
        ids, mask, prefix = sample_captions(policy, lat, fm, args.k, args.max_new)
        policy.train()
        caps = [
            tok.decode(row[m], skip_special_tokens=True) for row, m in zip(ids, mask)
        ]
        rewards = torch.tensor(
            [reward(c, chunk[i // args.k]) for i, c in enumerate(caps)],
            device=args.device,
        )
        groups = rewards.view(args.batch, args.k)
        adv = (groups - groups.mean(1, keepdim=True)) / (
            groups.std(1, keepdim=True) + 1e-4
        )
        adv = adv.view(-1)

        lp, n_tok = token_logprobs(policy, prefix.detach(), ids, mask)
        with torch.no_grad():
            lp_ref, _ = token_logprobs(ref, prefix.detach(), ids, mask)
        # Token-mean policy gradient with per-sequence advantage, plus a
        # k3-style KL estimate to the reference, both averaged per token.
        pg = -(adv.unsqueeze(1) * lp).sum(1) / n_tok.clamp_min(1)
        ratio = lp_ref - lp
        kl = (ratio.exp() - ratio - 1) * mask
        kl = kl.sum(1) / n_tok.clamp_min(1)
        loss = (pg + args.kl * kl).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 10 == 0:
            length = n_tok.float().mean().item()
            print(
                f"step {step}/{steps} reward {rewards.mean():.3f} (best-in-group {groups.max(1).values.mean():.3f}) "
                f"kl {kl.mean():.4f} len {length:.0f} {(time.time() - started) / 60:.1f}min",
                flush=True,
            )
            print(f"    {caps[0][:150]}", flush=True)
        if step and step % 100 == 0:
            pol, base = held_out_eval()
            print(
                f"  held-out reward: policy {pol:.3f}  reference {base:.3f}", flush=True
            )

    pol, base = held_out_eval()
    print(f"held-out reward after: policy {pol:.3f}  reference {base:.3f}")
    policy.save(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
