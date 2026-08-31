"""Render the winning instruction at reference length across several seeds.

The chosen voice was found on a short (~7s) clip, but clone references work
better around 14s with dense phoneme coverage. Re-rendering changes the text,
which re-rolls voice character somewhat, so generate a few seeds and pick the
one that survived the transfer.

Whichever wins becomes voices/default.wav + default.txt, and the service picks
it up on the next request -- the reference is read per request, so no restart.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from gen_voice_euro import BASE, VARIANTS
from gen_voice_matrix import CFG_SCALE, analyze
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

# Winner of the prosody sweep: pitch behaviour only, no nationality named, so it
# imparts the intonation without segmental accent leakage.
INSTRUCTION = BASE + VARIANTS["eu_level_pitch"]

# Long, phonetically dense, and natural to say -- this doubles as the ref_text
# the cloning path needs, so it must match the audio exactly.
TEXT = (
    "Welcome aboard. Your journey begins now. I've charted the route, checked "
    "the weather, and packed just enough curiosity for the whole thing. There's "
    "a quiet pleasure in solving a puzzle before anyone thinks to ask you. Six "
    "or seven hours, maybe. We'll see how it goes."
)

SEEDS = (42, 7, 123, 2024, 31337)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/final"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer, model, audio_tokenizer = load_runtime(
        args.model, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=1500, max_seq_len=2048, repetition_penalty=1.1
        ),
        tokenizer=tokenizer,
    )

    (args.out / "instruction.txt").write_text(INSTRUCTION + "\n")
    (args.out / "transcript.txt").write_text(TEXT + "\n")

    print(f"generating {len(SEEDS)} seeds to {args.out}/\n")
    results = []
    for seed in SEEDS:
        name = f"seed{seed}"
        target = args.out / f"{name}.wav"
        request = {
            "id": name,
            "text": TEXT,
            "instruction": INSTRUCTION,
            "speaker": "S0",
        }
        set_all_seeds(seed)
        inputs = prepare_inputs(
            tokenizer, audio_tokenizer, model, [request],
            get_template("tts_instruction"),
            guidance_scale=CFG_SCALE,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        started = time.time()
        with sf.SoundFile(
            target, mode="w", samplerate=runtime.sample_rate, channels=1,
            subtype="PCM_16",
        ) as handle:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=name):
                handle.write(chunk.audio)
        stats = analyze(target)
        results.append((name, stats))
        print(f"  {name:12s} {stats}   ({time.time() - started:.1f}s)", flush=True)

    print("\n--- summary ---")
    for name, stats in results:
        print(f"{name:12s} {stats}")
    print(f"\nreference for comparison: outputs/euro/eu_level_pitch.wav (short winner)")


if __name__ == "__main__":
    main()
