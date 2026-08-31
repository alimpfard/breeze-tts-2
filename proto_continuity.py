"""Prototype cross-request voice continuity strategies.

Each HTTP request currently conditions on the same fixed reference clip, so
consecutive replies are independent draws from "voices like this reference" and
audibly jump between them. The model has no memory of what it just said.

Three strategies, rendered over the same simulated conversation so they can be
compared by ear:

  independent  what the service does today: fixed reference every turn.
  rolling      condition on the tail of the previous reply. Maximum continuity,
               but each turn clones a clone, so identity can drift.
  anchored     canonical reference + previous tail concatenated. Pins identity
               to the original while still carrying prosody across the seam.

Anchored costs prompt length: canonical (~195 codec frames) plus the tail. At
12.5 Hz with max_seq_len 2048 there is ample room.
"""

from __future__ import annotations

import argparse
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

CFG_SCALE = 4.0
SEED = 42
TAIL_SECONDS = 6.0  # how much of the previous reply to carry forward

# "windowed" strategy bounds. A ~1s reference is too thin to clone from (a 3s
# one was already visibly worse than 14s early on), so accumulate whole recent
# turns until there is enough material, and cap it so the prompt stays bounded.
MIN_REF_SECONDS = 5.0
MAX_REF_SECONDS = 12.0

# A plausible multi-turn exchange: the seams between these are what jump today.
TURNS = [
    "Sure, I can take a look at that.",
    "The reranking step is the expensive part, so if you want it faster that is "
    "the first thing I would cut.",
    "That said, you will notice the difference in the results almost immediately.",
    "Let me know which way you want to go and I will set it up.",
]


def tail_of(audio: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    return audio[-int(seconds * rate) :] if len(audio) > seconds * rate else audio


def match_level(audio: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale to a target RMS so level drift cannot compound across turns."""
    rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
    if rms < 1e-6:
        return audio
    scaled = audio * (target_rms / rms)
    peak = float(np.abs(scaled).max())
    if peak > 0.99:  # never clip the conditioning
        scaled = scaled * (0.99 / peak)
    return scaled.astype(np.float32)


def build_window(
    history: list[tuple[np.ndarray, str]], rate: int, min_seconds: float
) -> tuple[np.ndarray, str, int] | None:
    """Newest-first accumulation of whole turns into a bounded reference.

    Whole turns keep audio and transcript aligned -- we have no word-level
    timings, so slicing mid-turn would desynchronise ref_text from ref_audio.
    """
    picked: list[tuple[np.ndarray, str]] = []
    total = 0.0
    for audio, text in reversed(history):
        picked.append((audio, text))
        total += len(audio) / rate
        if total >= min_seconds:
            break
    if total < min_seconds:
        return None  # not enough history yet; caller falls back to canonical
    picked.reverse()
    while total > MAX_REF_SECONDS and len(picked) > 1:
        dropped, _ = picked.pop(0)
        total -= len(dropped) / rate
    return (
        np.concatenate([audio for audio, _ in picked]).astype(np.float32),
        " ".join(text for _, text in picked),
        len(picked),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--voice", default="default")
    parser.add_argument("--out", type=Path, default=Path("outputs/continuity"))
    parser.add_argument(
        "--turns", type=Path, default=None, help="File with one turn per line."
    )
    parser.add_argument("--tag", default="", help="Suffix for output filenames.")
    args = parser.parse_args()

    turns = TURNS
    if args.turns is not None:
        turns = [
            line.strip()
            for line in args.turns.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    canon_wav = args.voices_dir / f"{args.voice}.wav"
    canon_text = (args.voices_dir / f"{args.voice}.txt").read_text().strip()
    canon_audio, rate = sf.read(canon_wav)

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "_scratch"
    scratch.mkdir(exist_ok=True)

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

    def generate(
        text: str, ref_path: Path, ref_text: str, tag: str, cfg: float = CFG_SCALE
    ) -> np.ndarray:
        request = {
            "id": tag,
            "text": text,
            "instruction": "Speak clearly and naturally.",
            "speaker": "S0",
            "ref_audio_path": str(ref_path),
            "ref_text": ref_text,
        }
        set_all_seeds(SEED)
        inputs = prepare_inputs(
            tokenizer, audio_tokenizer, model, [request],
            get_template("ref_edit_tata"),
            guidance_scale=cfg,
            guidance_scale_ref=None, guidance_scale_ins=None,
        )
        pieces = [c.audio for c in runtime.iter_audio_chunks(inputs, request_id=tag)]
        return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)

    canon_rms = float(np.sqrt((canon_audio.astype(np.float32) ** 2).mean()))

    by_strategy: dict[str, list[np.ndarray]] = {}
    # cfg_scale is the guidance strength: how hard generation is pushed toward
    # the conditioning. At 4.0 the reference may dominate enough that the model
    # stops adapting delivery to the text (quote vs narration). Sweep it.
    specs = {
        "rolling_c40": ("rolling", None, 4.0),
        "windowed3_c40": ("window", 3.0, 4.0),
        "windowed3_c25": ("window", 3.0, 2.5),
        "windowed3_c15": ("window", 3.0, 1.5),
        "rolling_c25": ("rolling", None, 2.5),
    }
    window_min = {k: v[1] for k, v in specs.items() if v[0] == "window"}
    for strategy in specs:
        kind, win_min, cfg = specs[strategy]
        print(f"\n=== {strategy} ===", flush=True)
        prev_audio: np.ndarray | None = None
        prev_text = ""
        history: list[tuple[np.ndarray, str]] = []
        turns_audio = []

        for index, text in enumerate(turns):
            note = ""
            if kind == "independent" or (prev_audio is None and kind != "window"):
                ref_path, ref_text = canon_wav, canon_text
            elif kind == "window":
                window = build_window(history, rate, win_min)
                if window is None:
                    ref_path, ref_text = canon_wav, canon_text
                    note = " [canonical: history too short]"
                else:
                    combined, ref_text, turn_count = window
                    combined = match_level(combined, canon_rms)
                    note = f" [ref {len(combined) / rate:.1f}s over {turn_count} turns]"
                    ref_path = scratch / f"{strategy}_{index}_ref.wav"
                    sf.write(ref_path, combined, rate, subtype="PCM_16")
            else:
                tail = tail_of(prev_audio, rate, TAIL_SECONDS)
                if kind == "rolling":
                    combined, ref_text = tail, prev_text
                else:  # anchored
                    combined = np.concatenate([canon_audio.astype(np.float32), tail])
                    ref_text = f"{canon_text} {prev_text}"
                ref_path = scratch / f"{strategy}_{index}_ref.wav"
                sf.write(ref_path, combined, rate, subtype="PCM_16")

            audio = generate(text, ref_path, ref_text, f"{strategy}-{index}", cfg)
            turns_audio.append(audio)
            history.append((audio, text))
            prev_audio, prev_text = audio, text
            rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
            print(
                f"  turn {index}: {len(audio) / rate:5.2f}s  rms {rms:.4f}{note}",
                flush=True,
            )

        by_strategy[strategy] = turns_audio

        # Stitch the turns with short gaps -- the seams are the whole point.
        gap = np.zeros(int(0.35 * rate), dtype=np.float32)
        stitched = np.concatenate(
            [piece for turn in turns_audio for piece in (turn, gap)][:-1]
        )
        target = args.out / f"{strategy}{args.tag}.wav"
        sf.write(target, stitched, rate, subtype="PCM_16")
        print(f"  -> {target} ({len(stitched) / rate:.1f}s)")

    # Same sentence rendered three ways, back to back: easier to A/B than
    # flipping between three long files.
    order = list(specs)
    wide_gap = np.zeros(int(0.5 * rate), dtype=np.float32)
    print()
    for index in range(len(turns)):
        seq: list[np.ndarray] = []
        for strategy in order:
            seq += [by_strategy[strategy][index], wide_gap]
        target = args.out / f"byturn{args.tag}_{index}.wav"
        sf.write(target, np.concatenate(seq[:-1]), rate, subtype="PCM_16")
        lens = "  |  ".join(
            f"{s[:5]} {len(by_strategy[s][index]) / rate:.2f}s" for s in order
        )
        print(f"  {target.name}  {lens}")
    print(f"\n  order in byturn files: {' -> '.join(order)}")


if __name__ == "__main__":
    main()
