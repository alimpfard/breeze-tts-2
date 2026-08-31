"""Kokoro-compatible TTS server backed by Breeze TTS 2.

Speaks the same HTTP contract the old kokoro-tts service did
(the XTTS-api-server shape), so existing clients need no changes:

    POST /tts_to_audio  {text, speaker_wav, language} -> audio/wav
    GET  /health
    GET  /voices

A "voice" is a reference clip in VOICES_DIR: ``<name>.wav`` plus a
``<name>.txt`` holding its exact transcript, which Breeze needs for
reference-based cloning.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel

from breeze_infer.conditioning import (
    ConditioningTurn,
    decode_state,
    encode_state,
    merge,
    total_seconds,
    trim_to_budget,
)
from breeze_infer.normalize import normalize_text
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

MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1

# Guidance strength. Lower lets the text drive delivery (quoted speech vs
# narration) instead of the reference pinning it; 1.5 stays coherent because the
# windowed reference below supplies steady conditioning. Raising this back to 4
# makes delivery uniform across contexts.
DEFAULT_CFG_SCALE = 1.5
DEFAULT_SEED = 42
DEFAULT_INSTRUCTION = "Speak clearly and naturally."

# A client-supplied mood swaps "naturally" out of the instruction. Capped and
# sanitised: it lands in the prompt, and instruction length drives text-encoder
# token buckets, so an unbounded value would spawn a new CUDA graph per variant.
MAX_MOOD_CHARS = 40
_MOOD_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9 ,'-]+")

# Cross-request voice continuity. Each reply conditions on recent speech rather
# than always on the canonical clip, so consecutive replies do not jump. Whole
# turns only: without word-level timings, slicing mid-turn desynchronises
# ref_text from ref_audio. A ~1s reference is too thin to clone from, hence the
# minimum; the maximum bounds prompt growth.
MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 12.0

# Conditioning on generated audio compounds: each turn clones the previous
# clone, so artefacts feed back and eventually collapse into a degenerate loop.
# Bound the chain and re-anchor to the canonical voice when it is spent. The
# cost is an audible seam at each reset; the alternative is unbounded decay.
#
# The chain is capped by *generation count*, not duration, because compounding
# tracks rounds rather than seconds -- eight half-second turns is four seconds
# of audio but eight rounds of degradation.
MAX_CHAIN_TURNS = 6

# Runaway guard: at ~13 chars/sec of speech, audio far longer than the text can
# justify means generation has come off the rails (usually the degenerate loop
# above). Break the chain immediately rather than feeding the mess forward.
CHARS_PER_SECOND = 13.0
RUNAWAY_DURATION_FACTOR = 3.0
RUNAWAY_SLACK_SECONDS = 2.0

# The client owns conversation lifetime: to start fresh at any point, simply
# stop echoing the conditioning blob.

# Breeze degrades on very long single utterances and MAX_NEW_TOKENS caps the
# audio length outright, so batch sentences into chunks of roughly this size.
CHUNK_CHAR_TARGET = 300

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("breeze-tts")

# Carried over verbatim from the kokoro server: clients send chat text that
# contains emoticons, and a TTS model will happily try to pronounce them.
EMOTICON_RE = re.compile(
    r"(?<!\w)(?:"
    r"[><]?[:;=8xX][\-~]?[)(DPp3Oo*|/\\}{@\]\[]"
    r"|[)(DPp3\]\[][\-~]?[:;=8]"
    r"|<[/\\]?3"
    r"|[><]\.[><]"
    r"|[\\^]_[\\^]"
    r"|[oO]_[oO]"
    r"|-_-"
    r"|T[_.]T"
    r"|uwu|owo|UwU|OwO"
    r")(?!\w)"
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")


def strip_emoticons(text: str) -> str:
    cleaned = EMOTICON_RE.sub("", text)
    return re.sub(r"  +", " ", cleaned).strip()


def prepare_text(text: str) -> str:
    """Clean client text into something the model reads correctly.

    Emoticons first: a TTS model will try to pronounce them (":3" comes out as
    "three"). Then symbol/decimal normalisation, since the model drops currency
    symbols entirely and mis-reads decimal points.
    """
    return normalize_text(strip_emoticons(text))


def is_runaway(text: str, seconds: float) -> bool:
    """True if generated audio is far longer than the text could justify."""
    expected = len(text) / CHARS_PER_SECOND
    return seconds > expected * RUNAWAY_DURATION_FACTOR + RUNAWAY_SLACK_SECONDS


def instruction_for(mood: str) -> str:
    """Build the instruction, substituting a mood for the default manner."""
    cleaned = _MOOD_ALLOWED_RE.sub("", mood or "").strip()[:MAX_MOOD_CHARS].strip()
    if not cleaned:
        return DEFAULT_INSTRUCTION
    return DEFAULT_INSTRUCTION.replace("naturally", f"in a {cleaned} mood")


def split_into_chunks(text: str, target: int = CHUNK_CHAR_TARGET) -> list[str]:
    """Group sentences into chunks of about ``target`` characters."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > target:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


class Voice:
    def __init__(self, name: str, wav: Path, transcript: str) -> None:
        self.name = name
        self.wav = wav
        self.transcript = transcript
        self._reference_rms: float | None = None

    @property
    def reference_rms(self) -> float:
        """RMS of the canonical clip, used as the level-matching target."""
        if self._reference_rms is None:
            audio, _ = sf.read(self.wav, always_2d=True, dtype="float32")
            mono = audio.mean(axis=1)
            self._reference_rms = float(np.sqrt((mono**2).mean()))
        return self._reference_rms


class VoiceLibrary:
    def __init__(self, directory: Path, default_name: str) -> None:
        self.directory = directory
        self.default_name = default_name

    def names(self) -> list[str]:
        return sorted(
            p.stem for p in self.directory.glob("*.wav") if p.with_suffix(".txt").exists()
        )

    def get(self, name: str) -> Voice | None:
        wav = self.directory / f"{name}.wav"
        transcript = wav.with_suffix(".txt")
        if not wav.is_file() or not transcript.is_file():
            return None
        return Voice(name, wav, transcript.read_text(encoding="utf-8").strip())

    def resolve(self, requested: str) -> Voice | None:
        """Fall back to the default voice for unknown/blank names.

        Clients hardcode kokoro voice names like ``am_michael``; those no
        longer mean anything here, so treat any miss as "use the default"
        rather than failing the request.
        """
        candidates = [requested, self.default_name] if requested else [self.default_name]
        for name in candidates:
            voice = self.get(name)
            if voice is not None:
                return voice
        return None


class BreezeEngine:
    """Single-threaded wrapper around the Breeze streaming runtime."""

    def __init__(
        self,
        model_path: Path,
        cfg_scale: float,
        seed: int,
        fast: bool = False,
        fast_stages: str = "all",
        fast_profile: Path | None = None,
        fp8: str = "off",
        int4: str = "off",
        int8_text: bool = False,
        text_precision: str = "bf16",
        offload_embeddings: bool = False,
        low_memory: bool = False,
        attention_precision: str = "int4",
        int4_group_depth: int = 128,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_seq_len: int = MAX_SEQ_LEN,
        continuity: str = "windowed",
    ) -> None:
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.continuity = continuity
        # The runtime is not reentrant; requests are serialised. No conversation
        # state is kept here -- see synthesize_stateless.
        self._lock = threading.Lock()

        self.device = resolve_device(device)
        # On a card that cannot hold the bf16 model at all, loading straight to
        # GPU OOMs before quantisation ever runs -- peak is the full-precision
        # size regardless of the final footprint. Stage through host memory and
        # quantise layer by layer instead.
        load_device = "cpu" if low_memory else self.device
        log.info(
            "Loading Breeze TTS 2 from %s on %s%s ...",
            model_path,
            load_device,
            " (staging for low-memory load)" if low_memory else "",
        )
        tokenizer, model, audio_tokenizer = load_runtime(
            model_path,
            device=load_device,
            attn_implementation="eager",
        )
        # load_runtime always lands in bf16. That is right on CUDA, but on CPU
        # bf16 is only fast where there is hardware support (Intel AMX, ARM
        # BF16); elsewhere it is emulated and loses to fp32.
        if dtype != "bfloat16":
            model.to(getattr(torch, dtype))
            log.info("Converted model to %s", dtype)
        update_generation_config_for_breeze(model)

        # Shared with the checkpoint exporter so a pre-quantized checkpoint is
        # built exactly the way the server would build it in memory.
        apply_quantization(
            model,
            QuantConfig(
                fp8=fp8,
                int4=int4,
                int8_text=int8_text,
                text_precision=text_precision,
                offload_embeddings=offload_embeddings,
                attention_precision=attention_precision,
                int4_group_depth=int4_group_depth,
                low_memory=low_memory,
            ),
            self.device,
        )

        if low_memory:
            # Everything still in host memory (attention, codec, the audio
            # tokenizer, anything not quantised) goes across now. Offloaded
            # tables pin themselves to the host and are skipped -- the round
            # trip alone would spike past a small card's budget.
            place_runtime(model, audio_tokenizer, self.device)
            torch.cuda.empty_cache()
            log.info(
                "low-memory load complete: %.2f GB allocated",
                torch.cuda.memory_allocated() / 1024**3,
            )

        config = FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            max_seq_len=max_seq_len,
            # "decode" graphs only the per-frame stages. Prefill and the text
            # encoder run once per request, so graphing them costs memory and
            # capture time for little gain -- and they are what touches
            # offloaded embeddings, which CUDA graphs cannot capture.
            fast_all=True if (fast and fast_stages == "all") else None,
            fast_backbone_decode=fast and fast_stages == "decode",
            fast_depth_decoder=fast and fast_stages == "decode",
            repetition_penalty=REPETITION_PENALTY,
        )
        runtime = FastBreezeStreamingRuntime(
            model, audio_tokenizer, config, tokenizer=tokenizer
        )
        if runtime.fast_enabled:
            profile = load_warmup_profile(fast_profile or FAST_CONFIG)
            profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
            manifest = runtime.warmup_from_profile(profile)
            log.info("Fast warmup: %.2f ms", manifest["total_elapsed_ms"])

        self.tokenizer = tokenizer
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self.runtime = runtime
        log.info("Breeze ready on %s.", resolve_device())

    @property
    def sample_rate(self) -> int:
        return self.runtime.sample_rate

    def synthesize_stateless(
        self,
        text: str,
        voice: Voice,
        incoming: list[ConditioningTurn],
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> tuple[np.ndarray, list[ConditioningTurn]]:
        """Generate, conditioning on client-supplied state; return updated state.

        The server keeps nothing: whatever conditioning arrives is what is used,
        and the caller gets back the state to echo on the next request.
        """
        chunks = split_into_chunks(text)
        if not chunks:
            return np.zeros(0, dtype=np.float32), incoming

        turns = list(incoming)
        pieces: list[np.ndarray] = []
        runaway = False
        with self._lock:
            for chunk in chunks:
                ref_codes = ref_text = None
                if self.continuity != "off" and total_seconds(turns) >= MIN_REF_SECONDS:
                    window = trim_to_budget(turns, MAX_REF_SECONDS)
                    codes, ref_text = merge(window)
                    ref_codes = codes
                audio, codes = self._generate_chunk(
                    chunk,
                    voice,
                    ref_codes=ref_codes,
                    ref_text=ref_text,
                    instruction=instruction,
                )
                if audio.size:
                    pieces.append(audio)
                    if is_runaway(chunk, len(audio) / self.sample_rate):
                        runaway = True
                    if codes is not None and len(codes):
                        # Codes come straight off generation -- re-encoding the
                        # audio the model just produced would be pure waste.
                        turns.append(ConditioningTurn(codes=codes, text=chunk))

        if runaway:
            log.warning("runaway generation, breaking conditioning chain")
            turns = []
        elif len(turns) >= MAX_CHAIN_TURNS:
            # Chain spent. Emit nothing so the next request re-anchors to the
            # canonical voice rather than cloning a clone indefinitely.
            log.info("conditioning chain reached %d turns, re-anchoring", len(turns))
            turns = []

        if not pieces:
            return np.zeros(0, dtype=np.float32), turns
        merged = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return merged, turns

    def _generate_chunk(
        self,
        text: str,
        voice: Voice,
        *,
        ref_codes: np.ndarray | None = None,
        ref_text: str | None = None,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        request_id = f"tts-{uuid.uuid4().hex}"
        request = {
            "id": request_id,
            "text": text,
            "instruction": instruction,
            "speaker": "S0",
        }
        if ref_codes is not None and ref_text:
            request["ref_audio_codes"] = ref_codes
            request["ref_text"] = ref_text
        else:
            request["ref_audio_path"] = str(voice.wav)
            request["ref_text"] = voice.transcript

        set_all_seeds(self.seed)
        inputs = prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            get_template("ref_edit_tata"),
            guidance_scale=self.cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

        pieces: list[np.ndarray] = []
        code_blocks: list[np.ndarray] = []
        for streamed in self.runtime.iter_audio_chunks(inputs, request_id=request_id):
            pieces.append(streamed.audio)
            if streamed.codes is not None and len(streamed.codes):
                code_blocks.append(streamed.codes)

        if not pieces:
            return np.zeros(0, dtype=np.float32), None
        audio = np.concatenate(pieces).astype(np.float32, copy=False)
        codes = np.concatenate(code_blocks, axis=0) if code_blocks else None
        return audio, codes

    def synthesize(self, text: str, voice: Voice) -> np.ndarray:
        """Generate with no conditioning state (canonical reference only)."""
        audio, _ = self.synthesize_stateless(text, voice, [])
        return audio


engine: BreezeEngine | None = None
voices: VoiceLibrary | None = None

app = FastAPI(title="Breeze TTS (kokoro-compatible)")


class TTSRequest(BaseModel):
    text: str
    speaker_wav: str = ""
    language: str = "en"
    # Opaque state returned by a previous call in the X-Conditioning header.
    # Optional, so clients that never send it keep the old behaviour exactly.
    conditioning: str = ""
    # Free-text mood, e.g. "wry" -> "Speak clearly and in a wry mood."
    mood: str = ""


@app.post("/tts_to_audio")
async def tts_to_audio(req: TTSRequest) -> Response:
    assert engine is not None and voices is not None

    clean_text = prepare_text(req.text)
    if not clean_text:
        return Response(status_code=204)

    requested = req.speaker_wav if req.speaker_wav not in ("default", "") else ""
    voice = voices.resolve(requested)
    if voice is None:
        return Response(status_code=400, content="No usable voice configured")

    incoming: list[ConditioningTurn] = []
    if req.conditioning:
        try:
            incoming = decode_state(req.conditioning)
        except ValueError as exc:
            # Bad state should not fail the request -- fall back to the
            # canonical voice and let the client resynchronise.
            log.warning("ignoring conditioning: %s", exc)

    instruction = instruction_for(req.mood)

    started = time.time()
    audio, outgoing = engine.synthesize_stateless(
        clean_text, voice, incoming, instruction
    )
    if not audio.size:
        return Response(status_code=400, content="No audio generated")

    buf = io.BytesIO()
    sf.write(buf, audio, engine.sample_rate, format="WAV", subtype="PCM_16")

    elapsed = time.time() - started
    duration = len(audio) / engine.sample_rate
    log.info(
        "Generated %.1fs audio in %.2fs (%.2fx realtime) voice=%s mood=%r "
        "cond_in=%.1fs cond_out=%.1fs text=%.50s",
        duration,
        elapsed,
        duration / elapsed if elapsed > 0 else 0.0,
        voice.name,
        req.mood or "",
        total_seconds(incoming),
        total_seconds(outgoing),
        clean_text,
    )
    headers = {"X-Sample-Rate": str(engine.sample_rate)}
    if outgoing:
        headers["X-Conditioning"] = encode_state(outgoing)
    return Response(
        content=buf.getvalue(), media_type="audio/wav", headers=headers
    )


@app.get("/health")
async def health() -> dict:
    if engine is None or voices is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "backend": "breeze-tts-2",
        "device": engine.device,
        "sample_rate": engine.sample_rate,
        "voices": voices.names(),
        "default_voice": voices.default_name,
        "continuity": engine.continuity,
        "cfg_scale": engine.cfg_scale,
        # Stateless: continuity state lives with the client, echoed via the
        # `conditioning` request field and X-Conditioning response header.
        "stateful": False,
        "conditioning_window_seconds": [MIN_REF_SECONDS, MAX_REF_SECONDS],
    }


@app.get("/voices")
async def list_voices() -> dict:
    assert voices is not None
    names = voices.names()
    # Keep kokoro's key names so existing clients keep parsing the response.
    return {"custom_voicepacks": [], "cloned_voices": names}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 behind the kokoro-tts HTTP contract"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=REPO_ROOT / "voices")
    parser.add_argument("--default-voice", default="default")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fast-stages",
        choices=("all", "decode"),
        default="all",
        help=(
            "Which stages to CUDA-graph. 'decode' covers the per-frame path "
            "(backbone decode, depth decoder) and skips prefill/text-encoder."
        ),
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the CUDA-graph fast path. Needs a profile declaring text-encoder "
            "batch size 4, which reference-based cloning uses; the stock "
            "configs/fast.json only declares 1 and 2 and is frozen after warmup."
        ),
    )
    parser.add_argument(
        "--fast-profile",
        type=Path,
        default=REPO_ROOT / "configs" / "fast_clone.json",
        help="Warmup profile used when --fast is set.",
    )
    parser.add_argument(
        "--fp8",
        choices=("off", "depth", "backbone", "all"),
        default="off",
        help=(
            "Quantize MLP weights to FP8. 'depth' covers the depth decoder "
            "(~79%% of per-frame weight traffic); 'all' adds the backbone."
        ),
    )
    parser.add_argument(
        "--int4",
        choices=("off", "depth", "backbone", "all"),
        default="off",
        help=(
            "Quantize MLP weights to int4 (group 128, weight-only tinygemm). "
            "Roughly 3-4x the bandwidth win of FP8 at ~2.7x the per-layer error. "
            "Takes precedence over --fp8 for any component named by both, so "
            "e.g. --int4 backbone --fp8 all gives int4 backbone + fp8 depth."
        ),
    )
    parser.add_argument(
        "--int8-text",
        action="store_true",
        help=(
            "Store text encoder weights as int8. Halves its ~1.4 GB at no "
            "throughput cost (it runs once per request), and unlike --fp8 works "
            "on pre-sm_89 cards. Superseded by --text-precision int8."
        ),
    )
    parser.add_argument(
        "--text-precision",
        choices=("bf16", "int8", "int4"),
        default="bf16",
        help=(
            "Text encoder precision. It runs once per request so this costs no "
            "throughput, but it conditions everything downstream -- int4 is only "
            "sensible on a card that cannot otherwise fit."
        ),
    )
    parser.add_argument(
        "--offload-embeddings",
        action="store_true",
        help=(
            "Keep text-side embedding tables in host memory (~1.7 GB freed). "
            "They are gathered once per request; the depth decoder's embedding "
            "stays resident because it is hit 16x per frame."
        ),
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "Stage the load through host memory and quantize layer by layer, so "
            "peak VRAM tracks the final footprint instead of the bf16 model. "
            "Required on cards too small to hold the unquantized weights."
        ),
    )
    parser.add_argument(
        "--device", default=None, help="e.g. cpu, cuda:0. Default: auto-detect."
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--continuity",
        choices=("windowed", "off"),
        default="windowed",
        help=(
            "windowed: condition each reply on recent speech so consecutive "
            "replies do not jump. off: always use the canonical reference."
        ),
    )
    args = parser.parse_args()

    global engine, voices
    voices = VoiceLibrary(args.voices_dir, args.default_voice)
    available = voices.names()
    if not available:
        raise SystemExit(
            f"No voices found in {args.voices_dir}. "
            "Each voice needs <name>.wav plus <name>.txt with its transcript."
        )
    log.info("Voices: %s (default: %s)", ", ".join(available), args.default_voice)

    engine = BreezeEngine(
        args.model,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        fast=args.fast,
        fast_stages=args.fast_stages,
        fast_profile=args.fast_profile,
        fp8=args.fp8,
        int4=args.int4,
        int8_text=args.int8_text,
        text_precision=args.text_precision,
        offload_embeddings=args.offload_embeddings,
        low_memory=args.low_memory,
        device=args.device,
        dtype=args.dtype,
        continuity=args.continuity,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
