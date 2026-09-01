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

The rank command is the honesty check. If the true instruction does not win on
audio the model itself generated from it, the search results are noise.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
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

    args = parser.parse_args()
    scorer = VoiceScorer(
        args.model, resolve_device(args.device), window_seconds=args.window
    )
    {"score": cmd_score, "rank": cmd_rank, "search": cmd_search}[args.cmd](scorer, args)


if __name__ == "__main__":
    main()
