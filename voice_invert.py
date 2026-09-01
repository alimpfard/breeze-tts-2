"""Voice design run backwards: which description best explains this audio?

Breeze conditions on the instruction as plain tokens spliced ahead of the audio
codes, so the model is a scorer for p(codes | description, transcript). One
teacher-forced pass (backbone + depth decoder) gives the full per-codebook
negative log-likelihood; no decode loop, no CFG. Comparing candidate
descriptions on the same audio cancels the content and prosody terms to first
order, leaving how much each descriptor explains the codes.

Three commands:

    score   score a handful of descriptions against one clip
    rank    labelled directory (name.wav + name.txt): confusion matrix,
            top-1 accuracy, and which codebooks carry the signal
    search  coordinate ascent over description slots for one clip
    evolve  free-form prose: an LLM proposes descriptions, Breeze scores
            them, the leaderboard goes back to the LLM to mutate

The rank command is the honesty check. If the true instruction does not win on
audio the model itself generated from it, the search results are noise.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from breeze_infer.audio import encode_prompt_audio
from breeze_infer.runtime import load_runtime, resolve_device
from breeze_infer.templates import (
    INSTRUCTION_BOS,
    INSTRUCTION_EOS,
    _prepare_segment_batches,
)

FRAME_HZ = 12.5


@dataclass(frozen=True)
class Score:
    """NLL in nats per frame and codebook (0 = backbone), shape (frames, K).

    ``per_codebook`` is the mean over ``window`` frames only. The description
    should matter most before the audio itself takes over as conditioning, so
    a window over the first second or two is a sharper test than the full clip.
    """

    per_frame: np.ndarray
    window: slice = slice(None)

    @property
    def frames(self) -> int:
        return self.per_frame.shape[0]

    @property
    def per_codebook(self) -> np.ndarray:
        return self.per_frame[self.window].mean(axis=0)

    @property
    def total(self) -> float:
        return float(self.per_codebook.sum())

    @property
    def backbone(self) -> float:
        return float(self.per_codebook[0])

    @property
    def acoustic(self) -> float:
        return float(self.per_codebook[1:].sum())


class VoiceScorer:
    def __init__(self, ckpt_dir: Path, device: str, window_seconds: float = 0.0):
        self.device = device
        self.window = (
            slice(0, round(window_seconds * FRAME_HZ))
            if window_seconds > 0
            else slice(None)
        )
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(
            ckpt_dir, device=device, attn_implementation="eager"
        )
        self.model.eval()
        cfg = self.model.config
        self.audio_token_id = cfg.audio_token_id
        self.audio_eos_token_id = cfg.audio_eos_token_id
        self.num_codebooks = cfg.num_codebooks
        self.backbone_eos_token_id = self.model.backbone_eos_token_id

    def encode(self, wav: Path) -> torch.Tensor:
        return encode_prompt_audio(self.audio_tokenizer, wav)

    @torch.no_grad()
    def score(self, description: str, text: str, codes: torch.Tensor) -> Score:
        segments = [
            {
                "type": "text",
                "text": f"[S0]{INSTRUCTION_BOS}{description}{INSTRUCTION_EOS}{text}",
            },
            {"type": "audio", "audio_codes": codes, "append_eos": True},
        ]
        inputs = _prepare_segment_batches(
            self.tokenizer,
            self.audio_tokenizer,
            self.model.config,
            self.device,
            [segments],
        )
        input_ids = inputs["input_ids"]
        input_values = inputs["input_values"].long()
        frames = input_values.shape[1]

        # Any non -101 value at audio positions works: _merge_input_ids_with_
        # input_values overwrites them with the codes (and the eos frame).
        labels = torch.full_like(input_ids, -100)
        audio_mask = input_ids == self.audio_token_id
        eos_mask = input_ids == self.audio_eos_token_id
        labels[audio_mask | eos_mask] = self.audio_token_id

        out = self.model(
            input_ids=input_ids,
            input_values=input_values,
            attention_mask=inputs["attention_mask"],
            text_ids_mask=inputs["text_ids_mask"],
            text_ids_len=inputs["text_ids_len"],
            labels=labels,
            use_cache=False,
        )

        # Backbone predicts codebook 0 of the next position (plus the eos frame,
        # which is a constant across candidates and left out of the mean).
        targets = torch.full_like(input_ids, -100)
        targets[audio_mask] = input_values[0, :, 0]
        logits = out.logits[0, :-1].float()
        nll0 = F.cross_entropy(
            logits, targets[0, 1:], ignore_index=-100, reduction="none"
        )
        nll0 = nll0[targets[0, 1:] != -100]
        assert nll0.numel() == frames, (nll0.numel(), frames)

        # Depth decoder: logits[:, k-1] predicts codebook k, frames in order.
        depth_logits = out.depth_decoder_logits.float()  # (frames, K-1, V)
        assert depth_logits.shape[0] == frames, (depth_logits.shape, frames)
        depth_targets = input_values[0, :, 1:]
        nll_depth = F.cross_entropy(
            depth_logits.reshape(-1, depth_logits.shape[-1]),
            depth_targets.reshape(-1),
            reduction="none",
        ).view(frames, self.num_codebooks - 1)

        per_frame = torch.cat([nll0.view(frames, 1), nll_depth], dim=1)
        return Score(per_frame.cpu().numpy(), self.window)


# --- description slots for the search ---------------------------------------
#
# Phrased the way the model card phrases voice design ("A warm, thoughtful
# young woman with a clear voice and a calm, reflective delivery."): identity
# first, then quality, then delivery, as one or two sentences.

SLOTS: dict[str, dict[str, str]] = {
    "identity": {
        "young woman": "A young adult woman, in her twenties.",
        "middle-aged woman": "A middle-aged woman, around forty.",
        "older woman": "An older woman, in her sixties.",
        "teenage girl": "A teenage girl.",
        "young man": "A young adult man, in his twenties.",
        "middle-aged man": "A middle-aged man, around forty.",
        "older man": "An older man, in his sixties.",
        "teenage boy": "A teenage boy.",
    },
    "pitch": {
        "low": "Her voice sits low, deep in pitch.",
        "mid": "Her voice sits in a natural mid range.",
        "high": "Her voice sits high in pitch, light and youthful.",
    },
    "brightness": {
        "dark": "Dark, warm, back-placed resonance with a mellow top end.",
        "balanced": "Balanced, natural resonance.",
        "bright": "Bright, forward resonance with an open top end.",
    },
    "phonation": {
        "breathy": "Noticeably breathy and airy, with audible air in the tone.",
        "clear": "Clear, focused phonation with no breathiness.",
    },
    "texture": {
        "smooth": "A perfectly smooth, clean tone with no rasp or vocal fry.",
        "raspy": "A noticeably raspy, creaky texture running through the voice.",
    },
    "pace": {
        "slow": "Relaxed and unhurried, a little slower than average conversation.",
        "normal": "Normal conversational pacing.",
        "fast": "Quick, brisk pacing.",
    },
    "energy": {
        "intimate": "Quiet, intimate, low-effort delivery.",
        "calm": "Calm, even, understated delivery.",
        "animated": "Lively, animated delivery with varied pitch and emphasis.",
        "projected": "Well projected, energetic delivery with strong presence.",
    },
    "attitude": {
        "warm": "She sounds warm and friendly.",
        "cool": "She sounds cool, observant, and quietly sharp.",
        "serious": "She sounds serious and restrained.",
        "cheerful": "She sounds cheerful and upbeat.",
    },
}

# Values in the pitch/attitude slots are written with "her"; swap the pronoun
# when the identity slot picks a male voice so the text stays coherent.
_MALE = {"young man", "middle-aged man", "older man", "teenage boy"}


def compose(choice: dict[str, str]) -> str:
    parts = [SLOTS[slot][value] for slot, value in choice.items()]
    text = " ".join(parts)
    if choice.get("identity") in _MALE:
        text = text.replace("Her voice", "His voice").replace("She sounds", "He sounds")
    return text


def coordinate_ascent(
    scorer: VoiceScorer,
    text: str,
    codes: torch.Tensor,
    *,
    rounds: int = 3,
    objective: str = "total",
    log=print,
) -> tuple[dict[str, str], dict[str, float]]:
    choice = {slot: next(iter(values)) for slot, values in SLOTS.items()}
    margins: dict[str, float] = {}
    cache: dict[str, Score] = {}

    def evaluate(candidate: dict[str, str]) -> float:
        desc = compose(candidate)
        if desc not in cache:
            cache[desc] = scorer.score(desc, text, codes)
        return getattr(cache[desc], objective)

    for round_idx in range(rounds):
        changed = False
        for slot, values in SLOTS.items():
            results = {}
            for value in values:
                candidate = dict(choice, **{slot: value})
                results[value] = evaluate(candidate)
            best = min(results, key=results.get)
            margins[slot] = max(results.values()) - min(results.values())
            ranked = ", ".join(
                f"{v}={s - results[best]:+.3f}"
                for v, s in sorted(results.items(), key=lambda kv: kv[1])
            )
            log(f"  r{round_idx} {slot:<11} -> {best:<18} [{ranked}]")
            if best != choice[slot]:
                choice[slot] = best
                changed = True
        if not changed:
            break
    return choice, margins


# --- free-form search with an LLM proposer -----------------------------------
#
# The slot search can only say what its vocabulary can say. Here an
# OpenAI-compatible chat endpoint writes the descriptions and Breeze grades
# them; the graded list goes back so the proposer can infer which attributes
# the audio rewards. Evolutionary search with the LLM as mutation operator.

STYLE_EXAMPLE = (
    "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."
)

LLM_SYSTEM = (
    "You write voice descriptions for a text-to-speech voice designer. A "
    "description is one to three sentences about the speaker and how they "
    "sound: gender, age, pitch, resonance, breathiness, texture, pace, "
    "energy, mood, accent. Describe the voice, never the words being said. "
    f"House style: {STYLE_EXAMPLE!r}. Always reply with a JSON array of "
    "strings and nothing else."
)


def llm_chat(
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.9,
    max_tokens: int = 8000,
    reasoning_effort: str = "none",
) -> str:
    """Return the reply text. Reasoning models may spend the whole budget
    thinking and return empty content; in that case the reasoning itself often
    ends with the answer, so hand that back for parsing."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Off by default: a reasoning model spends the whole budget thinking
        # about a leaderboard and returns empty content. Proposals are cheap
        # to grade, so breadth beats deliberation here.
        "reasoning_effort": reasoning_effort,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from exc
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if "[" not in content:
        content = message.get("reasoning_content") or content
    return content


def parse_candidates(text: str) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in LLM reply: {text[:200]!r}")
    items = json.loads(match.group(0))
    out = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(" ".join(item.split()))
    return out


def _leaderboard(scored: dict[str, float], top: int = 10, bottom: int = 3) -> str:
    ranked = sorted(scored.items(), key=lambda kv: kv[1])
    best = ranked[0][1]
    rows = ranked[:top]
    if len(ranked) > top + bottom:
        rows += [("...", None)] + ranked[-bottom:]
    elif len(ranked) > top:
        rows += ranked[top:]
    lines = []
    for desc, score in rows:
        if score is None:
            lines.append("  ...")
        else:
            lines.append(f"  {score - best:+.3f}  {desc}")
    return "\n".join(lines)


ROLES = (
    (
        "Propose {k} variants of the best entry, each changing or adding exactly "
        "one attribute (age, pitch, resonance, breathiness, rasp or fry, pace, "
        "energy, mood, accent)."
    ),
    (
        "Propose {k} descriptions that recombine the strongest traits of the top "
        "three entries, written fresh rather than spliced."
    ),
    (
        "Propose {k} descriptions that keep what the top entries agree on but "
        "test an attribute nobody has tried yet."
    ),
    (
        "Propose {k} descriptions that keep what the top entries agree on and "
        "state the opposite of whatever the bottom entries claim."
    ),
)


def cmd_evolve(scorer: VoiceScorer, args: argparse.Namespace) -> None:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        sys.exit("set LLM_API_KEY for the proposer endpoint")
    base_url = args.llm_base_url or os.environ.get("LLM_BASE_URL")
    if not base_url:
        sys.exit("set --llm-base-url or LLM_BASE_URL")

    codes = scorer.encode(args.wav)
    text = _read_text(args)
    n = args.population
    print(f"{args.wav}: {codes.shape[0]} frames, proposer={args.llm_model}\n")

    def ask(prompt: str) -> list[str]:
        for attempt in range(3):
            try:
                reply = llm_chat(
                    base_url,
                    args.llm_model,
                    api_key,
                    [
                        {"role": "system", "content": LLM_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    reasoning_effort=args.llm_reasoning,
                )
                return parse_candidates(reply)
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"  (retry {attempt + 1}: {str(exc)[:160]})")
        return []

    def ask_many(prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=args.llm_parallel) as pool:
            replies = list(pool.map(ask, prompts))
        out: list[str] = []
        for reply in replies:
            out.extend(d for d in reply if d not in out)
        return out

    scored: dict[str, float] = {}

    def grade(candidates: list[str]) -> list[tuple[str, float]]:
        fresh = []
        for desc in candidates:
            if desc in scored:
                continue
            scored[desc] = getattr(scorer.score(desc, text, codes), args.objective)
            fresh.append((desc, scored[desc]))
        return fresh

    t0 = time.perf_counter()
    initial = []
    if args.seed_slots:
        choice, _ = coordinate_ascent(
            scorer, text, codes, rounds=2, objective=args.objective, log=lambda *_: None
        )
        initial.append(compose(choice))
    initial += ask(
        f"The recording is of someone saying: {text!r}. You cannot hear it. "
        f"Propose {n} maximally diverse voice descriptions spanning both "
        "genders, several ages, low and high pitch, dark and bright timbre, "
        "smooth and raspy texture, slow and quick delivery. JSON array only."
    )
    grade(initial)
    history = []
    for round_idx in range(args.rounds + 1):
        ranked = sorted(scored.items(), key=lambda kv: kv[1])
        best_desc, best = ranked[0]
        history.append(best)
        print(
            f"round {round_idx}: {len(scored)} scored, best {best:.3f}  {best_desc[:100]}"
        )
        if round_idx == args.rounds:
            break
        board = (
            "Each description below was scored by how well a TTS model's "
            "likelihood of the recording is explained by it. Lower is better; "
            "shown as the gap to the current best. Under 0.1 is noise, over 0.3 "
            "is real. Attributes shared by the top entries and absent from the "
            "bottom ones are probably true of the voice; attributes that flip "
            "between them without changing the score do not matter.\n\n"
            f"{_leaderboard(scored)}\n\n"
        )
        per = max(2, n // len(ROLES))
        prompts = [
            board
            + role.format(k=per)
            + " None may repeat a listed entry. JSON array only."
            for role in ROLES
        ]
        fresh = grade(ask_many(prompts))
        for desc, score in sorted(fresh, key=lambda kv: kv[1]):
            marker = " *" if score < best else ""
            print(f"    {score - best:+.3f}  {desc[:100]}{marker}")

    print(
        f"\n{time.perf_counter() - t0:.1f}s, best per round: "
        + " ".join(f"{h:.3f}" for h in history)
    )
    ranked = sorted(scored.items(), key=lambda kv: kv[1])
    print("\ntop 5:")
    for desc, score in ranked[:5]:
        print(f"  {score:.3f}  {desc}")
    if args.truth:
        truth = args.truth.read_text().strip()
        s_truth = getattr(scorer.score(truth, text, codes), args.objective)
        rank = sum(1 for _, v in ranked if v < s_truth) + 1
        print(f"\ntruth {s_truth:.3f} would rank {rank}/{len(ranked) + 1}")
        print(f"  {truth}")


# --- commands ---------------------------------------------------------------


def _fmt_codebooks(score: Score) -> str:
    return " ".join(f"{v:5.2f}" for v in score.per_codebook)


def cmd_score(scorer: VoiceScorer, args: argparse.Namespace) -> None:
    codes = scorer.encode(args.wav)
    text = _read_text(args)
    print(f"{args.wav}: {codes.shape[0]} frames\n")
    rows = []
    for desc in args.description:
        t0 = time.perf_counter()
        s = scorer.score(desc, text, codes)
        rows.append((s, desc, time.perf_counter() - t0))
    rows.sort(key=lambda r: r[0].total)
    base = rows[0][0]
    edges = [0, 1, 2, 4, 8, 1e9]
    print(
        "total (gap)   bb    acoustic   | gap to best by window, nats/frame: "
        + " ".join(f"{a}-{b if b < 1e9 else ''}s" for a, b in itertools.pairwise(edges))
    )
    for s, desc, dt in rows:
        gaps = []
        for a, b in itertools.pairwise(edges):
            w = slice(int(a * FRAME_HZ), int(min(b, 1e6) * FRAME_HZ))
            g = (
                s.per_frame[w].sum(1).mean() - base.per_frame[w].sum(1).mean()
                if s.per_frame[w].size
                else float("nan")
            )
            gaps.append(f"{g:+.2f}")
        print(
            f"{s.total:7.3f} ({s.total - base.total:+.3f})  bb {s.backbone:5.2f}  ac {s.acoustic:6.2f} | {' '.join(gaps)}  {dt * 1000:4.0f}ms  {desc[:70]}"
        )


def cmd_rank(scorer: VoiceScorer, args: argparse.Namespace) -> None:
    text = _read_text(args)
    clips = sorted(p for p in args.dir.glob("*.wav") if p.with_suffix(".txt").exists())
    if args.limit:
        clips = clips[: args.limit]
    names = [p.stem for p in clips]
    instructions = {p.stem: p.with_suffix(".txt").read_text().strip() for p in clips}
    # Distinct instructions only; seeds of one instruction share a label.
    labels = sorted(set(instructions.values()))
    label_idx = {ins: i for i, ins in enumerate(labels)}
    print(f"{len(clips)} clips, {len(labels)} distinct instructions\n")

    k = scorer.num_codebooks
    nll = np.zeros((len(clips), len(labels), k))
    t0 = time.perf_counter()
    for ci, clip in enumerate(clips):
        codes = scorer.encode(clip)
        for li, ins in enumerate(labels):
            nll[ci, li] = scorer.score(ins, text, codes).per_codebook
        true = label_idx[instructions[clip.stem]]
        totals = nll[ci].sum(-1)
        order = np.argsort(totals)
        rank = int(np.where(order == true)[0][0]) + 1
        print(
            f"  {clip.stem:<24} true rank {rank:2d}/{len(labels)}  best={names_for(labels, instructions, order[0])}"
        )
    print(f"\n{time.perf_counter() - t0:.1f}s")

    truth = np.array([label_idx[instructions[n]] for n in names])

    def top1(weights: np.ndarray) -> float:
        totals = (nll * weights).sum(-1)
        return float((totals.argmin(-1) == truth).mean())

    print("\ntop-1 accuracy by codebook subset")
    print(f"  all 16         {top1(np.ones(k)):.2f}")
    print(f"  backbone only  {top1(np.eye(k)[0]):.2f}")
    print(f"  acoustic only  {top1(1 - np.eye(k)[0]):.2f}")
    print(
        f"  cb 1-4         {top1(np.isin(np.arange(k), [1, 2, 3, 4]).astype(float)):.2f}"
    )
    print(f"  cb 5-15        {top1((np.arange(k) >= 5).astype(float)):.2f}")
    print("  single codebook:", " ".join(f"{top1(np.eye(k)[i]):.2f}" for i in range(k)))

    # Paired-opposite test on the matrix axes: does "dark" beat "bright" on
    # the clip generated from "dark"? Harder than it sounds: the persona
    # sentence is shared, so only the axis sentence separates the candidates.
    pairs = []
    for a, b in itertools.combinations(names, 2):
        if "__" in a and "__" in b and a.split("__")[0] == b.split("__")[0]:
            pairs.append((a, b))
    if pairs:
        wins = 0
        print("\npaired opposites (clip: own instruction vs sibling's)")
        for a, b in pairs:
            for clip, other in ((a, b), (b, a)):
                ci = names.index(clip)
                own = nll[ci, label_idx[instructions[clip]]].sum()
                sib = nll[ci, label_idx[instructions[other]]].sum()
                ok = own < sib
                wins += ok
                print(
                    f"  {clip:<24} vs {other:<24} {'ok ' if ok else 'BAD'} {sib - own:+.3f}"
                )
        print(f"  {wins}/{2 * len(pairs)}")

    if args.save:
        np.savez(args.save, nll=nll, names=names, labels=labels, truth=truth)


def names_for(labels, instructions, idx) -> str:
    ins = labels[idx]
    return ",".join(n for n, i in instructions.items() if i == ins)


def cmd_search(scorer: VoiceScorer, args: argparse.Namespace) -> None:
    codes = scorer.encode(args.wav)
    text = _read_text(args)
    print(f"{args.wav}: {codes.shape[0]} frames, objective={args.objective}\n")
    t0 = time.perf_counter()
    choice, margins = coordinate_ascent(
        scorer, text, codes, rounds=args.rounds, objective=args.objective
    )
    print(f"\n{time.perf_counter() - t0:.1f}s\n")
    print(
        "slot margins (nats between best and worst value; ~0 means the model does not care):"
    )
    for slot, m in sorted(margins.items(), key=lambda kv: -kv[1]):
        print(f"  {slot:<11} {m:.3f}  -> {choice[slot]}")
    print("\n" + compose(choice))
    if args.truth:
        truth = args.truth.read_text().strip()
        s_truth = scorer.score(truth, text, codes)
        s_found = scorer.score(compose(choice), text, codes)
        print(f"\nfound   {s_found.total:.3f}")
        print(f"truth   {s_truth.total:.3f}")


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text().strip()
    if args.text:
        return args.text
    sys.exit("need --text or --text-file (the transcript of the clip)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--window",
        type=float,
        default=0.0,
        help="score only the first N seconds of audio (0 = whole clip)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--text")
        p.add_argument("--text-file", type=Path)

    p = sub.add_parser("score")
    common(p)
    p.add_argument("wav", type=Path)
    p.add_argument("description", nargs="+")

    p = sub.add_parser("rank")
    common(p)
    p.add_argument("dir", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save", type=Path)

    p = sub.add_parser("search")
    common(p)
    p.add_argument("wav", type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--objective", choices=("total", "backbone", "acoustic"), default="total"
    )
    p.add_argument("--truth", type=Path, help="known instruction, for comparison")

    p = sub.add_parser("evolve")
    common(p)
    p.add_argument("wav", type=Path)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--population", type=int, default=12)
    p.add_argument(
        "--objective", choices=("total", "backbone", "acoustic"), default="total"
    )
    p.add_argument("--truth", type=Path, help="known instruction, for comparison")
    p.add_argument(
        "--seed-slots",
        action="store_true",
        help="add the slot-search winner to round 0",
    )
    p.add_argument(
        "--llm-base-url", help="OpenAI-compatible base URL (or LLM_BASE_URL)"
    )
    p.add_argument("--llm-model", default="qwen/qwen3.8-27b")
    p.add_argument("--llm-parallel", type=int, default=4)
    p.add_argument(
        "--llm-reasoning", default="none", help="reasoning_effort sent to the endpoint"
    )

    args = parser.parse_args()
    scorer = VoiceScorer(
        args.model, resolve_device(args.device), window_seconds=args.window
    )
    {"score": cmd_score, "rank": cmd_rank, "search": cmd_search, "evolve": cmd_evolve}[
        args.cmd
    ](scorer, args)


if __name__ == "__main__":
    main()
