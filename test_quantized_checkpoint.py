"""Round-trip a pre-quantized checkpoint: load it and generate.

Verifies on Linux what would otherwise be debugged on the target machine --
that the skeleton rebuilds, every tensor lands, nothing is stranded on meta,
and generation produces audio.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.quantized_checkpoint import load_quantized_runtime

GB = 1024**3

TURNS = [
    "Sure, I can take a look at that.",
    "He spread his hands.",
    "Rich in potential, But potential does not reduce carriage strain.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--voice", default="default")
    parser.add_argument("--offload-embeddings", action="store_true")
    parser.add_argument("--int8-audio", action="store_true",
                        help="int8 the audio tokenizer's Linear layers "
                             "(~47%% of it; the rest is conv and untouched).")
    parser.add_argument("--budget-gb", type=float, default=0.0)
    parser.add_argument(
        "--max-seq-len", type=int, default=1024,
        help="Drives KV cache and the backbone decode graph's static buffers.",
    )
    parser.add_argument("--fast", action="store_true",
                        help="Enable CUDA graphs for all stages. Costs VRAM.")
    parser.add_argument("--graph-backbone", action="store_true",
                        help="Graph backbone decode only (1 call per frame).")
    parser.add_argument("--graph-depth", action="store_true",
                        help="Graph depth decoder only (16 calls per frame).")
    parser.add_argument(
        "--fast-decode-only",
        action="store_true",
        help=(
            "Graph only the per-frame stages (backbone decode, depth decoder). "
            "Prefill and text-encoder run once per request, so graphing them "
            "buys little -- and they are what touches offloaded embeddings, "
            "which cannot be captured."
        ),
    )
    parser.add_argument("--fast-profile", type=Path,
                        default=Path("configs/fast_service.json"))
    parser.add_argument("--out", type=Path, default=Path("ckpt_roundtrip.wav"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.budget_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((args.budget_gb * GB) / total)
        print(f"emulating a {args.budget_gb:.1f} GB budget\n")

    started = time.time()
    tokenizer, model, audio_tokenizer, stats = load_quantized_runtime(
        args.checkpoint, device, offload_embeddings=args.offload_embeddings
    )
    print(f"load: {time.time() - started:.1f}s")
    print(f"  {stats}")

    if args.int8_audio:
        from models.int8_linear import quantize_module_int8
        inner = getattr(audio_tokenizer, 'model', None)
        if inner is not None:
            print(f'  int8 audio: {quantize_module_int8(inner)}')
            inner.to(device)

    update_generation_config_for_breeze(model)
    print(f"  allocated {torch.cuda.memory_allocated() / GB:.2f} GB")

    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=1500, max_seq_len=args.max_seq_len,
            repetition_penalty=1.1,
            fast_all=True if args.fast else None,
            fast_backbone_decode=args.fast_decode_only or args.graph_backbone,
            fast_depth_decoder=args.fast_decode_only or args.graph_depth,
        ),
        tokenizer=tokenizer,
    )
    if runtime.fast_enabled:
        from dataclasses import replace as _replace

        from models.warmup_profile import load_warmup_profile

        profile = load_warmup_profile(args.fast_profile)
        profile = _replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        warm_started = time.time()
        runtime.warmup_from_profile(profile)
        print(f"  graph capture: {time.time() - warm_started:.1f}s, "
              f"reserved {torch.cuda.memory_reserved() / GB:.2f} GB")

    ref_wav = args.voices_dir / f"{args.voice}.wav"
    ref_text = (args.voices_dir / f"{args.voice}.txt").read_text().strip()

    pieces = []
    total_audio = total_gen = 0.0
    for index, text in enumerate(TURNS):
        turn_started = time.time()
        request = {
            "id": f"ckpt-{index}",
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
            guidance_scale=1.5, guidance_scale_ref=None, guidance_scale_ins=None,
        )
        chunks = [
            c.audio for c in runtime.iter_audio_chunks(inputs, request_id=f"ckpt-{index}")
        ]
        audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
        elapsed = time.time() - turn_started
        pieces.append(audio)
        duration = len(audio) / runtime.sample_rate
        # Skip the first turn: it pays lazy-init costs, not steady-state.
        if index > 0:
            total_audio += duration
            total_gen += elapsed
        rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
        print(
            f"  turn {index}: {duration:5.2f}s in {elapsed:5.2f}s "
            f"= {duration / elapsed:4.2f}x  rms {rms:.4f}"
            f"  peak {torch.cuda.max_memory_reserved() / GB:.2f} GB"
        )

    gap = np.zeros(int(0.35 * runtime.sample_rate), dtype=np.float32)
    stitched = np.concatenate([p for t in pieces for p in (t, gap)][:-1])
    sf.write(args.out, stitched, runtime.sample_rate, subtype="PCM_16")
    if total_gen:
        print(
            f"\nsteady-state: {total_audio:.2f}s audio in {total_gen:.2f}s "
            f"= {total_audio / total_gen:.2f}x realtime"
        )
    print(f"PASS -> {args.out} ({len(stitched) / runtime.sample_rate:.1f}s)")


if __name__ == "__main__":
    main()
