"""Generate voice-design variants that dial clarity independently of character.

The original instruction stacked several haze-inducing descriptors ("raspy",
"soft vocal fry", "relaxed texture", "lazy cadence", "low-effort"). These
variants keep the persona -- cool, understated, casually intelligent -- while
moving one axis at a time so the difference is attributable.
"""

import subprocess
import sys
from pathlib import Path

MODEL = "/home/test/source/breeze-tts-2/"

# Phonetically dense, and long enough to be reusable as a clone reference.
TEXT = (
    "Welcome aboard. Your journey begins now. I've charted the route, checked "
    "the weather, and packed just enough curiosity for the whole thing. There's "
    "a quiet pleasure in solving a puzzle before anyone thinks to ask you. Six "
    "or seven hours, maybe. We'll see how it goes."
)

BASE = (
    "A young adult woman, around 24 to 28. She sounds casually intelligent, "
    "observant, and quietly sharp, with subtle emphasis that suggests she is "
    "always thinking a step ahead. Cool, modern, understated, mentally alert."
)

VARIANTS = {
    # Articulation axis: crisp consonants, everything else unchanged.
    "crisp": BASE + (
        " Clear, focused tone with crisp consonants and clean, precise diction. "
        "Unhurried and composed, never rushed, but every word is distinctly "
        "articulated. A light youthful texture on phrase endings, more character "
        "than roughness."
    ),
    # Resonance axis: forward/bright placement.
    "forward": BASE + (
        " Bright, forward, present tone with clear resonance and no breathiness. "
        "Well supported and easy, speaking close to the microphone with warmth "
        "but no haze. Relaxed pacing, clean articulation."
    ),
    # Minimal edit: the original, with only the haze-inducing words removed.
    "clean": BASE + (
        " Relaxed, unhurried cadence with a faintly amused quality, like someone "
        "very smart who does not feel the need to sound impressive. Clear and "
        "well defined throughout, with a light youthful vocal fry used sparingly "
        "for texture rather than softness."
    ),
}


def main() -> None:
    out_dir = Path("outputs/variants")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, instruction in VARIANTS.items():
        target = out_dir / f"{name}.wav"
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run(
            [
                sys.executable, "infer.py", MODEL,
                "--text", TEXT,
                "--instruction", instruction,
                "--cfg-scale", "4",
                "--output", str(target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAILED: {result.stderr[-400:]}")
        else:
            print(f"saved {target}")
        # Keep the instruction alongside the audio so the winner is reproducible.
        target.with_suffix(".txt").write_text(instruction + "\n")


if __name__ == "__main__":
    main()
