"""Serialisable voice-conditioning state for stateless TTS servers.

Cross-request voice continuity needs the model to condition on recent speech
rather than always on a fixed clip. Rather than hold that state server-side --
which needs session keys, idle expiry, and leaks between concurrent
conversations -- the state is handed to the client as an opaque blob and echoed
back on the next request.

The blob is a list of *turns*, each pairing codec tokens with the text they
correspond to. Whole turns matter: there is no word-level alignment, so
trimming to a duration budget must drop entire turns or ref_text desynchronises
from ref_audio.

Size: codec tokens run 12.5 frames/sec x 16 codebooks of int16, so 12 seconds is
~4.8 KB raw and roughly 3-4 KB once deflated and base64'd. Compare ~560 KB for
the equivalent waveform, or ~224 MB for the model's KV cache.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass

import numpy as np

FRAME_RATE = 12.5
FORMAT_VERSION = 1


@dataclass(frozen=True)
class ConditioningTurn:
    codes: np.ndarray  # (frames, codebooks) int16
    text: str

    @property
    def seconds(self) -> float:
        return len(self.codes) / FRAME_RATE


def encode_state(turns: list[ConditioningTurn]) -> str:
    """Pack turns into a compact, self-describing, URL-safe string."""
    payload = {
        "v": FORMAT_VERSION,
        "turns": [
            {
                "shape": list(turn.codes.shape),
                "codes": base64.b64encode(
                    np.ascontiguousarray(turn.codes, dtype=np.int16).tobytes()
                ).decode("ascii"),
                "text": turn.text,
            }
            for turn in turns
        ],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(raw, 6)).decode("ascii")


def decode_state(blob: str) -> list[ConditioningTurn]:
    """Inverse of encode_state. Raises ValueError on anything malformed."""
    try:
        raw = zlib.decompress(base64.b64decode(blob))
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"malformed conditioning blob: {exc}") from exc

    if payload.get("v") != FORMAT_VERSION:
        raise ValueError(f"unsupported conditioning version {payload.get('v')!r}")

    turns: list[ConditioningTurn] = []
    for entry in payload.get("turns", []):
        shape = tuple(entry["shape"])
        if len(shape) != 2:
            raise ValueError(f"expected 2D codes, got shape {shape}")
        codes = np.frombuffer(
            base64.b64decode(entry["codes"]), dtype=np.int16
        ).reshape(shape)
        turns.append(ConditioningTurn(codes=codes, text=entry["text"]))
    return turns


def trim_to_budget(
    turns: list[ConditioningTurn], max_seconds: float
) -> list[ConditioningTurn]:
    """Drop oldest whole turns until within budget, always keeping the newest."""
    kept = list(turns)
    while len(kept) > 1 and sum(t.seconds for t in kept) > max_seconds:
        kept.pop(0)
    return kept


def total_seconds(turns: list[ConditioningTurn]) -> float:
    return sum(turn.seconds for turn in turns)


def merge(turns: list[ConditioningTurn]) -> tuple[np.ndarray, str]:
    """Concatenate turns into a single (codes, text) conditioning pair.

    Codec frames are independent along the time axis, so concatenating them is
    equivalent to concatenating the underlying audio.
    """
    if not turns:
        raise ValueError("cannot merge zero turns")
    codes = np.concatenate([turn.codes for turn in turns], axis=0)
    return codes, " ".join(turn.text for turn in turns)
