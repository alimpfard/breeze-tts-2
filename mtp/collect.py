"""Collect what the depth decoder sees and what it says, for a draft model.

Per frame the depth decoder gets the backbone's last hidden state and
codebook 0, then emits codebooks 1..15 one step at a time. That serial loop
is 60% of a frame on spark. Speculative decoding along the codebook axis
needs a draft that guesses all 15 at once from the same inputs, so this
stores, per clip: the hidden states (frames, 2048), the codes (frames, 16),
and for an evaluation subset the depth decoder's own per-codebook
distributions at the served temperature.

    python -m mtp.collect /path/to/breeze --shard 0/2 --device cuda:0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_invert import VoiceScorer

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/mtp"
DEPTH_TEMPERATURE = 0.9  # generation_config.depth_decoder_temperature
EVAL_CLIPS = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    si, sn = (int(x) for x in args.shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]
    scorer = VoiceScorer(args.breeze, args.device)
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, _args, kwargs):
        captured["hidden"] = kwargs["backbone_last_hidden_state"].detach()

    scorer.model.depth_decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)

    for i, p in enumerate(files):
        if i % sn != si:
            continue
        out_path = OUT / f"{p.stem}.pt"
        if out_path.exists():
            continue
        r = torch.load(p)
        codes = r["codes"]
        with torch.no_grad():
            # score() runs the teacher-forced pass; the hook grabs the hidden
            # states. Re-run the depth decoder outputs through the model output
            # by calling the scorer's internals would be cleaner, but the
            # public path is enough: depth logits come back on the model output.
            scorer._last_depth_logits = None
            score = scorer.score(r["prose"], r["text"], codes)
        rec = {
            "hidden": captured["hidden"].to(torch.float16).cpu(),
            "codes": codes.to(torch.int16),
            "text": r["text"],
        }
        if i < EVAL_CLIPS:
            probs = torch.softmax(scorer._last_depth_logits.float() / DEPTH_TEMPERATURE, dim=-1)
            rec["target_probs"] = probs.to(torch.float16).cpu()  # (frames, 15, V)
        torch.save(rec, out_path)
        if i % 100 < sn:
            print(f"  {i}/{len(files)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
