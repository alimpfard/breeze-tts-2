"""Roundtrip evaluation: caption -> Breeze -> latents, cosine to the original.

For held-out clips, compare captioners by how well a clip rendered from
their caption matches the original clip's pooled latents. The original
design prompt is rendered too, as the reference point: it is what the clip
was generated from, so it is as good as a description gets for this model.

    python -m caption.roundtrip_eval /path/to/breeze data/caption/captioner_v2 \
        data/caption/captioner_rl_hi --n 40
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.model import Captioner, collate
from caption.rl import RoundTrip, pooled
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("captioners", type=Path, nargs="+")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--breeze-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    files = sorted((REPO / "data/caption/latents").glob("*.pt"))
    random.Random(args.seed).shuffle(files)
    recs = [
        torch.load(p) for p in files[-args.n :]
    ]  # tail: never in any RL train slice
    rt = RoundTrip(args.breeze, args.breeze_device)
    # Centre on the projector's input mean so the shared component doesn't
    # compress every cosine into 0.83-0.91 (same as the RL reward).
    state = torch.load(args.captioners[0] / "resampler.pt", map_location="cpu")
    centre = state["resampler"]["in_mean"].flatten()

    def sim(caption, text, original):
        if not caption.strip():
            caption = "Speak clearly and naturally."  # empty output scores as null
        feats = rt.render(caption, text)
        if feats is None:
            return -1.0
        a, b = feats.flatten(), pooled(original).flatten()
        if centre is not None:
            a, b = a - centre, b - centre
        return F.cosine_similarity(a, b, dim=0).item()

    captions = {"truth": [r["prose"] for r in recs]}
    captions["null"] = ["Speak clearly and naturally."] * len(recs)
    for path in args.captioners:
        model = Captioner.load(path, args.device)
        out = []
        for i in range(0, len(recs), 16):
            chunk = recs[i : i + 16]
            lat, fm, _, _ = collate(chunk, model.tokenizer)
            out += model.generate(
                lat.to(args.device), fm.to(args.device), do_sample=False
            )
        captions[path.name] = out
        if empties := sum(not c.strip() for c in out):
            print(f"{path.name}: {empties} empty captions", flush=True)
        del model
        torch.cuda.empty_cache()

    print(
        f"{len(recs)} held-out clips, roundtrip cosine of pooled latents (higher is better)\n"
    )
    sims = {}
    for name, caps in captions.items():
        vals = [sim(c, r["text"], r["latents"]) for c, r in zip(caps, recs)]
        sims[name] = vals
        mean = sum(vals) / len(vals)
        print(f"{name:<18} mean {mean:.4f}", flush=True)
    names = [n for n in sims if n not in ("truth", "null")]
    if len(names) == 2:
        a, b = names
        a_wins = sum(x > y for x, y in zip(sims[a], sims[b]))
        print(f"\n{a} > {b} on {a_wins}/{len(recs)} clips")
    print("\nexamples:")
    for i in range(4):
        print(f"\n  text:  {recs[i]['text']}")
        for name, caps in captions.items():
            print(f"  {name:<16} {sims[name][i]:.3f}  {caps[i][:140]}")


if __name__ == "__main__":
    main()
