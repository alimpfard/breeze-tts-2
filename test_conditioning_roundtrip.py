"""End-to-end check of stateless conditioning against a running server.

Simulates a client that echoes the X-Conditioning header back on each request,
which is the whole point of the design: the server holds nothing.
"""

from __future__ import annotations

import argparse
import io
import sys

import numpy as np
import requests
import soundfile as sf

TURNS = [
    "“With the deepest respect, Your Majesty,” the merchant said, bowing again, “all trade ultimately is.”",
    "He spread his hands.",
    "“Your kingdom is beautiful.",
    "Rich in potential, But potential does not reduce carriage strain.”",
    "The king saw us then.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    parser.add_argument("--out", default="outputs/stateless.wav")
    parser.add_argument(
        "--no-carry",
        action="store_true",
        help="Drop the conditioning between turns (control: old behaviour).",
    )
    args = parser.parse_args()

    conditioning = ""
    pieces: list[np.ndarray] = []
    rate = 24000

    for index, text in enumerate(TURNS):
        payload = {"text": text, "speaker_wav": "default"}
        if conditioning and not args.no_carry:
            payload["conditioning"] = conditioning
        sent = len(conditioning) if payload.get("conditioning") else 0

        response = requests.post(
            f"{args.url}/tts_to_audio", json=payload, timeout=600
        )
        if response.status_code != 200:
            print(f"turn {index}: HTTP {response.status_code} {response.text[:200]}")
            sys.exit(1)

        audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        pieces.append(audio)
        conditioning = response.headers.get("X-Conditioning", "")
        print(
            f"turn {index}: {len(audio) / rate:5.2f}s  rms {np.sqrt((audio**2).mean()):.4f}"
            f"  sent {sent / 1024:5.1f} KB  got {len(conditioning) / 1024:5.1f} KB"
        )

    gap = np.zeros(int(0.35 * rate), dtype=np.float32)
    stitched = np.concatenate([p for turn in pieces for p in (turn, gap)][:-1])
    sf.write(args.out, stitched, rate, subtype="PCM_16")
    print(f"\n-> {args.out} ({len(stitched) / rate:.1f}s)")


if __name__ == "__main__":
    main()
