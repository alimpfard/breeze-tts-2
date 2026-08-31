"""Compose the chosen voice axes, then sweep intonation/stress character.

Picks from the axis matrix: brightness=dark, phonation~clear (moderated, not
clinical), prosody=languid but well short of the 11s extreme, texture=raspy.

On top of that base we vary only prosody-level "accent feel" -- the thing that
survives even when segmental pronunciation is perfect. Deliberately phrased as
rhythm and pitch behaviour rather than as a named accent, to avoid pulling in
segmental accent leakage.

`control_american` exists to prove the axis moves at all. If it is
indistinguishable from `eu_combined`, this axis is dead (as brightness was) and
no amount of prompt wording will fix it.
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
from gen_voice_matrix import CFG_SCALE, SEED, TEXT, analyze
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

# The composed winner from the axis sweep. "Unhurried" is deliberately hedged --
# the languid extreme ran 11s against a 6.7s baseline, which was far too much.
BASE = (
    "A young adult woman, around 24 to 28. She sounds casually intelligent, "
    "observant, and quietly sharp. Cool, modern, understated, mentally alert. "
    "Dark, warm resonance with a mellow top end. Focused, clear phonation with "
    "no breathiness or airiness. A noticeably raspy, creaky texture running "
    "through the voice. Relaxed and unhurried, a little slower than average "
    "conversation, but still easy and natural, never drawn out or sluggish."
)

VARIANTS: dict[str, str] = {
    # The composed base with no prosody instruction, as the reference point.
    "composed": "",
    # Rhythm: reduced vowel reduction -> more even syllable weight.
    "eu_even_stress": (
        " Every syllable is given close to its full value, with little of the "
        "vowel reduction typical of American English, so the rhythm is evenly "
        "weighted rather than compressed around the stressed beats."
    ),
    # Pitch: narrow range, level or falling terminals.
    "eu_level_pitch": (
        " A narrow, level pitch range with restrained melodic movement. "
        "Statements end in a clean, definite fall, never rising at the end."
    ),
    # Timing: measured and deliberate.
    "eu_deliberate": (
        " Measured, deliberate phrasing with precise, even timing and clear "
        "separation between phrases, as though each thought is placed rather "
        "than tumbled out."
    ),
    # Explicit framing: perfect English, but non-American intonation.
    "eu_named": (
        " She speaks flawless, unaccented English, but her intonation and stress "
        "patterns are those of an educated central European speaker rather than "
        "an American one: even, measured, and restrained."
    ),
    # Everything at once.
    "eu_combined": (
        " She speaks flawless English, but with the intonation and stress of an "
        "educated central European speaker rather than an American one. Syllables "
        "are evenly weighted with little vowel reduction, the pitch range is "
        "narrow and level, statements end in a clean fall, and the phrasing is "
        "measured and deliberate."
    ),
    # Control: if this is indistinguishable from eu_combined, the axis is dead.
    "control_american": (
        " Distinctly American intonation and stress, with a wide, swooping pitch "
        "range, heavy reduction of unstressed syllables, and a casual rising "
        "lilt at the ends of phrases."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/euro"))
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

    print(f"generating {len(VARIANTS)} clips to {args.out}/\n")
    results = []
    for name, suffix in VARIANTS.items():
        instruction = BASE + suffix
        target = args.out / f"{name}.wav"
        request = {
            "id": name,
            "text": TEXT,
            "instruction": instruction,
            "speaker": "S0",
        }
        set_all_seeds(SEED)
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
        target.with_suffix(".txt").write_text(instruction + "\n")
        stats = analyze(target)
        results.append((name, stats))
        print(f"  {name:20s} {stats}   ({time.time() - started:.1f}s)", flush=True)

    print("\n--- summary ---")
    for name, stats in results:
        print(f"{name:20s} {stats}")


if __name__ == "__main__":
    main()
