"""Describe the voice in a recording: wav + transcript -> prose.

    python -m caption.infer /path/to/breeze data/caption/captioner clip.wav \
        --text "what was said" [--samples 3]

Runs the clone-style prefill through Breeze to get the backbone latents,
then the captioner. Pass a directory of .pt records instead of a wav to
caption held-out training data (truth is printed alongside).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breeze_infer.audio import encode_prompt_audio
from breeze_infer.runtime import load_runtime
from caption.gen import LatentExtractor
from caption.model import Captioner, collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("breeze", type=Path)
    parser.add_argument("captioner", type=Path)
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="wav files, or a dir of .pt records"
    )
    parser.add_argument(
        "--text", help="transcript (one wav) or a file with one line per wav"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--samples", type=int, default=1, help="extra sampled captions per clip"
    )
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    captioner = Captioner.load(args.captioner, args.device)
    tok = captioner.tokenizer

    def describe(items: list[dict]) -> list[list[str]]:
        latents, frame_mask, _, _ = collate(items, tok)
        latents, frame_mask = latents.to(args.device), frame_mask.to(args.device)
        greedy = captioner.generate(latents, frame_mask, do_sample=False)
        outs = [[g] for g in greedy]
        for _ in range(args.samples):
            sampled = captioner.generate(
                latents, frame_mask, do_sample=True, temperature=0.8, top_p=0.95
            )
            for o, s in zip(outs, sampled):
                o.append(s)
        return outs

    if args.inputs[0].is_dir():
        paths = sorted(args.inputs[0].glob("*.pt"))[-args.limit :]
        items = [torch.load(p) for p in paths]
        for p, item, outs in zip(paths, items, describe(items)):
            print(f"\n[{p.stem}] {item['text']}\n  truth:  {item['prose']}")
            for o in outs:
                print(f"  ->      {o}")
        return

    if not args.text:
        sys.exit("--text is required for wav input")
    texts = (
        Path(args.text).read_text().splitlines()
        if Path(args.text).exists()
        else [args.text] * len(args.inputs)
    )
    tokenizer, model, audio_tokenizer = load_runtime(
        args.breeze, device=args.device, attn_implementation="eager"
    )
    extract = LatentExtractor(tokenizer, model, audio_tokenizer, args.device)
    items = []
    for wav, text in zip(args.inputs, texts):
        codes = encode_prompt_audio(audio_tokenizer, wav)
        items.append({"latents": extract(text, codes), "prose": "", "text": text})
    for wav, outs in zip(args.inputs, describe(items)):
        print(f"\n{wav}")
        for o in outs:
            print(f"  -> {o}")


if __name__ == "__main__":
    main()
