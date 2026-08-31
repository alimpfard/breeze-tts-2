"""Single-request streaming inference for Breeze TTS 2."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.quantize_config import (
    QuantConfig,
    apply_quantization,
    place_runtime,
)
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parent
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one WAV with Breeze TTS 2")
    parser.add_argument("model", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default="Speak clearly and naturally.")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fp8", choices=("off", "depth", "backbone", "all"), default="off",
        help="Quantize MLP weights to FP8 (needs sm_89+).",
    )
    parser.add_argument(
        "--int4", choices=("off", "depth", "backbone", "all"), default="off",
        help="Quantize MLP weights to int4, group 128 (needs sm_80+).",
    )
    parser.add_argument(
        "--text-precision", choices=("bf16", "int8", "int4"), default="bf16",
        help="Text encoder precision. It runs once per request, so lowering it "
             "costs no throughput -- but it conditions everything downstream.",
    )
    parser.add_argument(
        "--attention-precision", choices=("int4", "bf16"), default="bf16",
        help="With --low-memory, also quantize attention projections.",
    )
    parser.add_argument(
        "--offload-embeddings", action="store_true",
        help="Keep text-side embedding tables in host memory (~1.7 GB freed).",
    )
    parser.add_argument(
        "--low-memory", action="store_true",
        help="Stage the load through host RAM and quantize layer by layer, so "
             "peak VRAM tracks the final footprint instead of the bf16 model.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        raise ValueError("--cfg-scale must be greater than 0")

    has_ref_audio = args.ref_audio is not None
    has_ref_text = bool(args.ref_text and args.ref_text.strip())
    if has_ref_audio != has_ref_text:
        raise ValueError("--ref-audio and --ref-text must be provided together")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    device = resolve_device()
    tokenizer, model, audio_tokenizer = load_runtime(
        args.model,
        device="cpu" if args.low_memory else device,
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)

    apply_quantization(
        model,
        QuantConfig(
            fp8=args.fp8,
            int4=args.int4,
            text_precision=args.text_precision,
            offload_embeddings=args.offload_embeddings,
            attention_precision=args.attention_precision,
            low_memory=args.low_memory,
        ),
        device,
    )
    if args.low_memory:
        place_runtime(model, audio_tokenizer, device)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )

    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms")

    request = {
        "id": "single-request",
        "text": args.text,
        "instruction": args.instruction,
        "speaker": "S0",
    }
    template_name = "tts_instruction"
    if args.ref_audio is not None:
        request["ref_audio_path"] = str(args.ref_audio)
        request["ref_text"] = args.ref_text.strip()
        template_name = "ref_edit_tata"

    set_all_seeds(args.seed)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template(template_name),
        guidance_scale=args.cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        args.output,
        mode="w",
        samplerate=runtime.sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output_file:
        for chunk in runtime.iter_audio_chunks(inputs, request_id="single-request"):
            output_file.write(chunk.audio)

    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
