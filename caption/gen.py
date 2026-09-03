"""Synthesise the training set: description -> audio -> backbone latents.

For each description record, render a sentence with Breeze in voice-design
mode, then run a clone-style prefill (`[S0]{text}` + the generated codes) and
keep the backbone's hidden states at the audio positions from a few layers.
Those states are what the backbone itself conditions on when it clones a
reference, so they carry the voice by construction; the captioner learns to
read them back into words.

    python -m caption.gen /path/to/breeze --device cuda:0 --shard 0/2

Writes one .pt per clip into data/caption/latents/: latents (L, T, 2048)
bf16, codes (T, 16), attrs, prose, text. No wav is kept; the codes are
enough to decode one later.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breeze_infer.runtime import (
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import (
    _prepare_segment_batches,
    get_template,
    prepare_inputs,
)
from models.fast_streaming import (
    FastBreezeStreamingRuntime,
    FastStreamingConfig,
)
from models.warmup_profile import load_warmup_profile

REPO = Path(__file__).resolve().parents[1]
FAST_PROFILE = REPO / "configs" / "fast_service.json"

# Backbone has 28 layers. The last is specialised for next-code prediction;
# middle layers usually hold speaker identity better. Keep several and let
# the captioner learn a weighting.
LAYERS = (7, 14, 21, 28)
MAX_FRAMES = 128  # 10.24 s at 12.5 Hz
CFG_SCALE = 4.0  # model card's recommendation for instruction following
MIN_SECONDS = 2.0
MAX_SECONDS = 12.0


class LatentExtractor:
    """Clone-style prefill with forward hooks on the chosen backbone layers."""

    def __init__(self, tokenizer, model, audio_tokenizer, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self._captured: dict[int, torch.Tensor] = {}
        layers = model.backbone_model.layers
        for idx in LAYERS:
            layers[idx - 1].register_forward_hook(self._hook(idx))

    def _hook(self, idx: int):
        def fn(_module, _inputs, output):
            self._captured[idx] = output[0] if isinstance(output, tuple) else output

        return fn

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
        self._captured.clear()
        self.model.backbone_model(
            inputs_embeds=merged["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            text_encoder_layer_hidden_states=merged["text_encoder_layer_hidden_states"],
            text_ids_mask=inputs["text_ids_mask"],
        )
        audio_mask = inputs["input_ids"][0] == self.model.config.audio_token_id
        stacked = torch.stack([self._captured[idx][0, audio_mask] for idx in LAYERS])
        return stacked[:, :MAX_FRAMES].to(torch.bfloat16).cpu()


def build_runtime(ckpt: Path, device: str, fast: bool):
    tokenizer, model, audio_tokenizer = load_runtime(
        ckpt, device=device, attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    config = FastStreamingConfig(
        max_new_tokens=int(MAX_SECONDS * 12.5) + 10,
        max_seq_len=1024,
        fast_all=None,
        fast_backbone_decode=fast,
        fast_depth_decoder=fast,
        repetition_penalty=1.1,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )
    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_PROFILE)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        runtime.warmup_from_profile(profile)
    return tokenizer, model, audio_tokenizer, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--descriptions", type=Path, default=REPO / "data/caption/descriptions.jsonl"
    )
    parser.add_argument(
        "--sentences", type=Path, default=REPO / "data/caption/sentences.txt"
    )
    parser.add_argument("--out", type=Path, default=REPO / "data/caption/latents")
    parser.add_argument("--shard", default="0/1", help="i/n: this worker's slice")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shard_idx, shard_n = (int(x) for x in args.shard.split("/"))
    records = [json.loads(line) for line in args.descriptions.open()]
    sentences = [s.strip() for s in args.sentences.open() if s.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.device)
    tokenizer, model, audio_tokenizer, runtime = build_runtime(
        args.model, args.device, fast=not args.no_fast
    )
    extract = LatentExtractor(tokenizer, model, audio_tokenizer, args.device)

    todo = [
        i
        for i in range(len(records))
        if i % shard_n == shard_idx and not (args.out / f"{i:06d}.pt").exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"shard {args.shard}: {len(todo)} clips to render on {args.device}", flush=True
    )

    audio_total = gen_total = 0.0
    started = time.time()
    for n, i in enumerate(todo):
        record = records[i]
        rng = random.Random(args.seed * 1_000_003 + i)
        # One or two sentences: enough audio to judge a voice, short enough
        # to keep throughput up.
        text = " ".join(rng.sample(sentences, rng.choice((1, 1, 2))))
        request = {
            "id": f"cap{i:06d}",
            "text": text,
            "instruction": record["prose"],
            "speaker": "S0",
        }
        set_all_seeds(rng.randrange(1 << 30))
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template("tts_instruction"),
            guidance_scale=CFG_SCALE,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        t0 = time.time()
        codes_parts = []
        for chunk in runtime.iter_audio_chunks(inputs, request_id=request["id"]):
            if chunk.codes is not None and len(chunk.codes):
                codes_parts.append(np.asarray(chunk.codes))
        gen_s = time.time() - t0
        if not codes_parts:
            print(f"  {i:06d}: no audio, skipped", flush=True)
            continue
        codes = torch.as_tensor(np.concatenate(codes_parts), dtype=torch.int16)
        seconds = codes.shape[0] / 12.5
        if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
            print(f"  {i:06d}: {seconds:.1f}s out of range, skipped", flush=True)
            continue

        latents = extract(text, codes)
        torch.save(
            {
                "latents": latents,
                "layers": LAYERS,
                "codes": codes,
                "attrs": record["attrs"],
                "prose": record["prose"],
                "text": text,
                "seconds": seconds,
            },
            args.out / f"{i:06d}.pt",
        )
        audio_total += seconds
        gen_total += gen_s
        if n % 20 == 0:
            elapsed = time.time() - started
            rate = (n + 1) / elapsed * 3600
            print(
                f"  {n + 1}/{len(todo)}  {seconds:4.1f}s in {gen_s:4.1f}s  "
                f"rtf {audio_total / gen_total:.2f}x  {rate:.0f} clips/h",
                flush=True,
            )
    print(f"done: {len(todo)} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
