"""Reference-audio loading for Breeze inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def encode_prompt_array(
    audio_tokenizer: Any, wav: np.ndarray, sample_rate: int
) -> torch.Tensor:
    """Tokenise audio already in memory, skipping a file round-trip.

    Used to capture conditioning from freshly generated audio.
    """
    mono = np.asarray(wav, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    encoded = audio_tokenizer.encode(mono, sr=sample_rate)
    codes = torch.as_tensor(encoded["audio_codes"][0], dtype=torch.int16)
    if codes.ndim != 2:
        raise ValueError(f"Expected 2D audio codes, got shape {tuple(codes.shape)}")
    return codes.cpu().contiguous()


def encode_prompt_audio(audio_tokenizer: Any, audio_path: str | Path) -> torch.Tensor:
    wav, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    wav = np.mean(wav, axis=1)
    encoded = audio_tokenizer.encode(wav, sr=sample_rate)
    codes = torch.as_tensor(encoded["audio_codes"][0], dtype=torch.int16)
    if codes.ndim != 2:
        raise ValueError(
            f"Expected 2D audio codes for '{audio_path}', got shape {tuple(codes.shape)}"
        )
    return codes.cpu().contiguous()
