"""Discover which CUDA graph buckets real cloning traffic actually hits.

configs/fast_clone.json declares 96 graphs (~65 MB each) because I provisioned
defensively against "not declared in the warmup profile" crashes. Most are
never used: the reference transcript is fixed, so ref-side token lengths are
effectively constant and only the target text moves.

This runs representative requests with freeze_after_warmup disabled, so unknown
shapes are captured lazily instead of raising, then reports exactly which
buckets were touched. Feed that back into a tight profile.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.fp8_linear import quantize_module_fp8
from models.warmup_profile import load_warmup_profile

GB = 1024**3

# Span the range the server can produce: split_into_chunks caps chunks at ~300
# chars, so nothing longer than that reaches the model in one go.
PROBE_TEXTS = [
    "Sure.",
    "Alright, that works for me.",
    "Sure, I can do that. Give me a second to think it through.",
    "That is a good question, and the honest answer is that it depends on what "
    "you are optimizing for. If you want it fast, we cut the reranking step.",
    "A" * 8 + " " + ("word " * 55),  # ~300 chars, the chunker's ceiling
]


def minimal_profile(source: Path, target: Path) -> Path:
    """Stock profile, but lazy: declare little, capture the rest on demand."""
    spec = json.loads(source.read_text())
    spec["service"]["freeze_after_warmup"] = False
    # The service is pinned at cfg 4, which always yields 2 CFG branches, so
    # every branch_batch_size=1 graph is unreachable. Declaring only cfg 4 keeps
    # the validator happy (prefill must cover each declared decode batch) while
    # dropping the whole batch-1 half of the profile.
    spec["service"]["cfg_scales"] = [4.0]
    spec["stages"]["backbone_decode"]["graphs"] = [{"branch_batch_size": 2}]
    spec["stages"]["depth_decoder"]["graphs"] = [{"batch_size": 2}]
    spec["stages"]["text_encoder"]["graphs"] = [
        {"batch_size": 4, "token_length": 128}
    ]
    spec["stages"]["backbone_prefill"]["graphs"] = [
        {"branch_batch_size": 2, "sequence_length": 128}
    ]
    target.write_text(json.dumps(spec, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--voice", default="default")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    args = parser.parse_args()

    ref_wav = args.voices_dir / f"{args.voice}.wav"
    ref_text = (args.voices_dir / f"{args.voice}.txt").read_text().strip()

    tmp = Path(tempfile.mkdtemp()) / "probe.json"
    profile_path = minimal_profile(Path("configs/fast.json"), tmp)

    tokenizer, model, audio_tokenizer = load_runtime(
        args.model, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    quantize_module_fp8(model.depth_decoder)

    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=1500, max_seq_len=2048, fast_all=True,
            repetition_penalty=1.1,
        ),
        tokenizer=tokenizer,
    )
    profile = load_warmup_profile(profile_path)
    profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
    runtime.warmup_from_profile(profile)

    torch.cuda.synchronize()
    print(f"after minimal warmup: reserved {torch.cuda.memory_reserved() / GB:.2f} GB\n")

    for index, text in enumerate(PROBE_TEXTS):
        request = {
            "id": f"probe-{index}",
            "text": text,
            "instruction": "Speak clearly and naturally.",
            "speaker": "S0",
            "ref_audio_path": str(ref_wav),
            "ref_text": ref_text,
        }
        set_all_seeds(42)
        inputs = prepare_inputs(
            tokenizer, audio_tokenizer, model, [request],
            get_template("ref_edit_tata"),
            guidance_scale=args.cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        frames = sum(
            1 for _ in runtime.iter_audio_chunks(inputs, request_id=f"probe-{index}")
        )
        print(f"  probe {index} ({len(text):3d} chars): {frames} chunks", flush=True)

    cache = getattr(model, "_fast_text_encoder_graph_cache", None)
    print("\ntext-encoder graph keys actually used (batch, token_length):")
    print(f"  {sorted(cache.graph_keys) if cache else 'none'}")

    prefill = getattr(runtime, "_backbone_prefill_graphs", {})
    for batch, entry in sorted(prefill.items()):
        print(f"prefill branch_batch={batch}: {sorted(entry.graph_keys)}")

    torch.cuda.synchronize()
    print(f"\nfinal reserved: {torch.cuda.memory_reserved() / GB:.2f} GB")


if __name__ == "__main__":
    main()
