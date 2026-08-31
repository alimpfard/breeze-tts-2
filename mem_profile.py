"""Attribute the service's GPU memory to load / quantization / graph capture.

The service sits at ~19.6 GB while the weights are only ~7.2 GB. This reports
memory after each stage so the overhead lands on a specific cause rather than a
guess. Captured CUDA graphs are the prime suspect: each one retains its whole
activation working set in a private pool for the process lifetime, and
configs/fast_clone.json declares far more graphs than the stock profile.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    update_generation_config_for_breeze,
)
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.fp8_linear import quantize_module_fp8
from models.warmup_profile import load_warmup_profile

GB = 1024**3


def report(label: str) -> tuple[float, float]:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / GB
    print(
        f"{label:34s} alloc {alloc:6.2f}  reserved {reserved:6.2f}  device-used {used:6.2f} GB",
        flush=True,
    )
    return alloc, reserved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", type=Path, default=Path("configs/fast_clone.json"))
    parser.add_argument("--fp8", choices=("off", "depth", "all"), default="depth")
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    spec = json.loads(args.profile.read_text())
    print(
        f"profile {args.profile.name}: "
        f"{len(spec['stages']['text_encoder']['graphs'])} text-encoder graphs, "
        f"{len(spec['stages']['backbone_prefill']['graphs'])} prefill graphs\n"
    )

    report("start")
    tokenizer, model, audio_tokenizer = load_runtime(
        args.model, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    after_load, _ = report("after weight load")

    if args.fp8 != "off":
        targets = [model.depth_decoder]
        if args.fp8 == "all":
            targets.append(model.backbone_model)
        for submodule in targets:
            quantize_module_fp8(submodule)
        report("after fp8 quantize")
        # The displaced bf16 weights are freed but sit in the caching allocator;
        # hand them back so the saving is visible outside the process.
        torch.cuda.empty_cache()
        report("after empty_cache")

    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=1500,
            max_seq_len=2048,
            fast_all=True if args.fast else None,
            repetition_penalty=1.1,
        ),
        tokenizer=tokenizer,
    )
    report("after runtime construct")

    if runtime.fast_enabled:
        profile = load_warmup_profile(args.profile)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        runtime.warmup_from_profile(profile)
        after_warm, _ = report("after graph capture")
        # Capture leaves large reserved-but-unallocated segments behind. Graph
        # private pools are still in use so they survive; this should only
        # return the transient warmup slack.
        torch.cuda.empty_cache()
        report("after post-capture empty_cache")
        print(f"\ngraph capture cost: {after_warm - after_load:6.2f} GB allocated")


if __name__ == "__main__":
    main()
