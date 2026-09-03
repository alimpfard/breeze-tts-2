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


def token_logprobs(model: Captioner, prefix, ids, mask, chunk: int = 16):
    """Sum of log-probs of the sampled tokens given the prefix, per row.
    Chunked over rows: the full-vocab logits for 64 rows do not fit."""
    embed = model.lm.get_input_embeddings()
    p = prefix.shape[1]
    lps = []
    for i in range(0, ids.shape[0], chunk):
        pre, tok_ids, m = prefix[i : i + chunk], ids[i : i + chunk], mask[i : i + chunk]
        inputs = torch.cat([pre, embed(tok_ids)], dim=1)
        attn = torch.cat(
            [torch.ones(pre.shape[:2], dtype=torch.long, device=ids.device), m.long()],
            dim=1,
        )
        logits = model.lm(inputs_embeds=inputs, attention_mask=attn).logits
        # position P-1 predicts ids[:, 0]; position P+t-1 predicts ids[:, t]
        pred = logits[:, p - 1 : p - 1 + tok_ids.shape[1]].float()
        lp = -F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), tok_ids.reshape(-1), reduction="none"
        ).view(tok_ids.shape)
        lps.append(lp * m)
    return torch.cat(lps), mask.sum(1)


class RoundTrip:
    """Render a caption with Breeze and pool the rendered clip's latents."""

    def __init__(self, breeze: Path, device: str):
        from caption.gen import LatentExtractor, build_runtime

        self.tokenizer, self.model, self.audio_tokenizer, self.runtime = build_runtime(
            breeze, device, fast=True
        )
        self.extract = LatentExtractor(
            self.tokenizer, self.model, self.audio_tokenizer, device
        )
        self.device = device

    def render(
        self, caption: str, text: str, seed: int = 0, raw: bool = False
    ) -> torch.Tensor | None:
        import numpy as np

        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs

        request = {"id": "rt", "text": text, "instruction": caption, "speaker": "S0"}
        set_all_seeds(seed)
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
            return None
        codes = torch.as_tensor(np.concatenate(parts), dtype=torch.int16)
        lat = self.extract(text, codes)
        return lat if raw else pooled(lat)


def pooled(lat: torch.Tensor) -> torch.Tensor:
    return pool_features(
        lat.float().unsqueeze(0), torch.ones(1, lat.shape[1], dtype=torch.bool)
    )[0]


def _render_worker(device: str, breeze: Path, in_q, out_q) -> None:
    """One Breeze runtime per process: CUDA graphs on two devices in one
    process trip over each other during capture."""
    rt = RoundTrip(breeze, device)
    out_q.put(("ready", device))
    while True:
        job = in_q.get()
        if job is None:
            return
        job_id, caption, text = job
        try:
            out_q.put((job_id, rt.render(caption, text)))
        except Exception as exc:  # noqa: BLE001
            out_q.put((job_id, f"error: {exc}"))


class Renderers:
    def __init__(
        self, breeze: Path, devices: list[str], centre: torch.Tensor | None = None
    ):
        # Cosine of raw pooled latents is compressed into 0.84-0.93 by the
        # shared mean vector: a null prompt scores 0.836 against 0.845 for
        # the true one. Centring on the dataset mean opens that to 0.13 vs
        # 0.23 with seed noise of 0.002, a far cleaner reward.
        self.centre = centre.flatten() if centre is not None else None
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.in_qs = [ctx.Queue() for _ in devices]
        self.out_q = ctx.Queue()
        self.procs = [
            ctx.Process(
                target=_render_worker, args=(d, breeze, q, self.out_q), daemon=True
            )
            for d, q in zip(devices, self.in_qs)
        ]
        for p in self.procs:
            p.start()
        for _ in devices:
            print("renderer", self.out_q.get(), flush=True)

    def similarities(self, jobs: list[tuple[str, str, torch.Tensor]]) -> list[float]:
        """jobs: (caption, text, original latents) -> cosine per job, -1 on failure."""
        for i, (caption, text, _) in enumerate(jobs):
            self.in_qs[i % len(self.in_qs)].put((i, caption, text))
        results = {}
        for _ in jobs:
            job_id, feats = self.out_q.get()
            results[job_id] = feats
        sims = []
        for i, (_, _, original) in enumerate(jobs):
            feats = results[i]
            if feats is None or isinstance(feats, str):
                sims.append(-1.0)
            else:
                a, b = feats.flatten(), pooled(original).flatten()
                if self.centre is not None:
                    a, b = a - self.centre, b - self.centre
                sims.append(F.cosine_similarity(a, b, dim=0).item())

        return sims

    def close(self):
        for q in self.in_qs:
            q.put(None)


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
    parser.add_argument(
        "--likelihood", type=float, default=1.0, help="weight of the likelihood reward"
    )
    parser.add_argument(
        "--roundtrip-devices",
        default="cuda:1",
        help="comma-separated devices, one Breeze runtime each, renders split across them",
    )
    parser.add_argument("--eval-clips", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--reward-centred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="centre pooled latents on the dataset mean before the cosine",
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
    scorer = (
        VoiceScorer(args.breeze, args.scorer_device, window_seconds=args.window)
        if args.likelihood
        else None
    )
    renderers = (
        Renderers(
            args.breeze,
            args.roundtrip_devices.split(","),
            centre=policy.resampler.in_mean.detach().cpu()
            if args.reward_centred
            else None,
        )
        if args.roundtrip
        else None
    )

    def rewards_for(caps: list[str], items: list[dict]) -> list[float]:
        """One reward per (caption, clip); renders split across the runtimes."""
        base: list[float | None] = []
        for c, r in zip(caps, items):
            if len(c.split()) < 3:
                base.append(None)  # degenerate: loses the group
            elif scorer is not None:
                base.append(
                    -args.likelihood * scorer.score(c, r["text"], r["codes"]).total
                )
            else:
                base.append(0.0)
        if renderers is not None:
            idxs = [i for i, b in enumerate(base) if b is not None]
            sims = renderers.similarities(
                [(caps[i], items[i]["text"], items[i]["latents"]) for i in idxs]
            )
            for i, sim in zip(idxs, sims):
                base[i] += args.roundtrip * sim
        floor = min((b for b in base if b is not None), default=0.0) - 100.0
        return [floor if b is None else b for b in base]

    def held_out_eval() -> tuple[float, float]:
        """Mean reward of greedy captions on held-out, policy vs reference."""
        policy.eval()
        outs = []
        subset = held[: args.eval_clips]
        for model in (policy, ref):
            total = 0.0
            for i in range(0, len(subset), 16):
                chunk = subset[i : i + 16]
                lat, fm, _, _ = collate(chunk, tok)
                caps = model.generate(
                    lat.to(args.device), fm.to(args.device), do_sample=False
                )
                total += sum(rewards_for(caps, chunk))
            outs.append(total / len(subset))
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
    best_held = float("-inf")
    for step in range(steps):
        chunk = random.sample(train, args.batch)
        lat, fm, _, _ = collate(chunk, tok)
        lat, fm = lat.to(args.device), fm.to(args.device)
        policy.eval()
        ids, mask, prefix = sample_captions(policy, lat, fm, args.k, args.max_new)
        policy.train()
        caps = [
            tok.decode(row[m], skip_special_tokens=True).strip()
            for row, m in zip(ids, mask)
        ]
        items = [chunk[i // args.k] for i in range(len(caps))]
        rewards = torch.tensor(rewards_for(caps, items), device=args.device)
        groups = rewards.view(args.batch, args.k)
        adv = (groups - groups.mean(1, keepdim=True)) / (
            groups.std(1, keepdim=True) + 1e-4
        )
        adv = adv.view(-1)

        policy.lm.gradient_checkpointing_enable()
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
        policy.lm.gradient_checkpointing_disable()

        if step % 10 == 0:
            length = n_tok.float().mean().item()
            print(
                f"step {step}/{steps} reward {rewards.mean():.3f} (best-in-group {groups.max(1).values.mean():.3f}) "
                f"kl {kl.mean():.4f} len {length:.0f} {(time.time() - started) / 60:.1f}min",
                flush=True,
            )
            print(f"    {caps[0][:150]}", flush=True)
        if step and step % args.eval_every == 0:
            pol, base = held_out_eval()
            print(
                f"  held-out reward: policy {pol:.3f}  reference {base:.3f}", flush=True
            )
            if pol > best_held:
                best_held = pol
                policy.save(args.out / "best")
                print(f"  saved best ({pol:.3f}) to {args.out / 'best'}", flush=True)

    pol, base = held_out_eval()
    print(f"held-out reward after: policy {pol:.3f}  reference {base:.3f}")
    policy.save(args.out)
    if renderers is not None:
        renderers.close()
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
