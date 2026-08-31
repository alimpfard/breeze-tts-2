"""One-factor-at-a-time sweep over voice-quality axes.

Full factorial over 6 axes x 3 levels would be 729 clips. Instead we vary one
axis at a time against a neutral baseline, so each clip differs from the
baseline in exactly one respect and any audible difference is attributable.

The seed is pinned across every clip: voice design is stochastic, and without a
fixed seed you cannot tell an instruction effect from a sampling roll.

Listen within an axis (the three levels back to back), pick a level per axis,
then compose the winners into a single instruction.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

SEED = 42
CFG_SCALE = 4.0

# Short enough to iterate on, long enough to judge timbre. Covers plosives,
# fricatives, sibilants, nasals and a range of vowels.
TEXT = (
    "Welcome aboard. I've charted the route and checked the weather. There's a "
    "quiet pleasure in solving a puzzle before anyone thinks to ask."
)

# Persona held constant across every clip; only the quality axis moves.
PERSONA = (
    "A young adult woman, around 24 to 28. She sounds casually intelligent, "
    "observant, and quietly sharp. Cool, modern, understated, mentally alert."
)

NEUTRAL = "Natural phonation, balanced resonance, and normal conversational delivery."

AXES: dict[str, dict[str, str]] = {
    # What the vocal folds do. Prime suspect for "hazy".
    "phonation": {
        "breathy": "Noticeably breathy and airy, with a soft, whispery quality and audible air in the tone.",
        "neutral": "Natural phonation, neither breathy nor pressed.",
        "clear": "Clear, focused phonation with no breathiness at all, firmly supported and solid.",
    },
    # What the vocal tract does: spectral tilt / placement.
    "brightness": {
        "dark": "Dark, warm, back-placed resonance with a muted, veiled top end.",
        "neutral": "Balanced, natural resonance.",
        "bright": "Bright, forward, present resonance, placed well forward in the mask, with an open top end.",
    },
    # Tongue and lips: consonant definition.
    "articulation": {
        "soft": "Soft, relaxed articulation with gently blurred, understated consonants.",
        "neutral": "Natural, unremarkable articulation.",
        "crisp": "Crisp, precise articulation with clearly defined, cleanly separated consonants.",
    },
    # Time and pitch.
    "prosody": {
        "languid": "Slow, unhurried, legato delivery that lingers on phrases with little pitch variation.",
        "neutral": "Natural conversational pacing.",
        "animated": "Lively, slightly clipped, animated delivery with noticeably varied pitch and emphasis.",
    },
    # Dynamics / projection.
    "effort": {
        "intimate": "Quiet, intimate, low-effort delivery, spoken very close to the microphone.",
        "neutral": "Normal conversational effort.",
        "projected": "Well projected, energetic, strongly supported delivery with clear presence.",
    },
    # The texture you originally asked for, isolated so you can dial it exactly.
    "texture": {
        "smooth": "Perfectly smooth, clean tone with no rasp, creak, or vocal fry whatsoever.",
        "light": "A light, youthful vocal fry appearing only on phrase endings, used sparingly as texture.",
        "raspy": "A noticeably raspy, creaky texture present throughout the voice.",
    },
}


def analyze(path: Path) -> str:
    audio, rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / rate)
    total = spectrum.sum() or 1.0
    centroid = float((spectrum * freqs).sum() / total)
    high = float(spectrum[freqs > 4000].sum() / total) * 100
    rms = float(np.sqrt((audio**2).mean()))
    return f"{len(audio) / rate:5.2f}s  rms {rms:.4f}  centroid {centroid:6.0f}Hz  HF {high:5.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/matrix"))
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

    jobs: list[tuple[str, str]] = [("baseline", f"{PERSONA} {NEUTRAL}")]
    for axis, levels in AXES.items():
        for level, phrase in levels.items():
            if level == "neutral":
                continue  # the baseline already covers the neutral midpoint
            jobs.append((f"{axis}__{level}", f"{PERSONA} {phrase}"))

    print(f"generating {len(jobs)} clips to {args.out}/\n")
    results: list[tuple[str, str]] = []
    for name, instruction in jobs:
        target = args.out / f"{name}.wav"
        request = {
            "id": name,
            "text": TEXT,
            "instruction": instruction,
            "speaker": "S0",
        }
        set_all_seeds(SEED)
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
        started = time.time()
        with sf.SoundFile(
            target, mode="w", samplerate=runtime.sample_rate, channels=1,
            subtype="PCM_16",
        ) as handle:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=name):
                handle.write(chunk.audio)
        target.with_suffix(".txt").write_text(instruction + "\n")
        stats = analyze(target)
        results.append((name, stats))
        print(f"  {name:26s} {stats}   ({time.time() - started:.1f}s)", flush=True)

    print("\n--- summary (baseline first, then one axis at a time) ---")
    for name, stats in results:
        print(f"{name:26s} {stats}")


if __name__ == "__main__":
    main()
