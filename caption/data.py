"""Sample voice attributes and have an LLM write them up as descriptions.

Produces the label side of the captioner's training set. Each record has the
structured attributes it was written from (for probing and evaluation) and
the prose the TTS model will be asked to realise (the captioner's target).

    python -m caption.data --n 6000 --out data/caption/descriptions.jsonl

Attributes are sampled with "not stated" as a real option for the soft ones
so the prose varies in what it mentions, the way real design prompts do.
Gender and age are always stated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_invert import llm_chat, parse_candidates

ATTRS: dict[str, list[str]] = {
    "gender": ["woman", "man"],
    "age": [
        "child, around 9",
        "teenager",
        "young adult, twenties",
        "adult, thirties to forties",
        "middle-aged, fifties",
        "elderly, seventies",
    ],
    "pitch": ["low", "mid", "high"],
    "brightness": ["dark, warm, mellow", "balanced", "bright, forward"],
    "phonation": ["breathy, airy", "clear, focused", "tense, pressed"],
    "texture": ["smooth", "light vocal fry", "raspy, gravelly", "hoarse"],
    "pace": ["slow", "normal", "fast"],
    "energy": ["quiet, intimate", "calm", "animated", "loud, projected"],
    "mood": [
        "neutral",
        "warm, friendly",
        "cheerful",
        "serious",
        "tired",
        "wry, amused",
        "anxious",
        "angry",
        "sad",
        "excited",
    ],
    "accent": [
        "American",
        "British",
        "Irish",
        "Scottish",
        "Australian",
        "Indian English",
        "non-native European",
        "Southern US",
    ],
    "role": [
        "narrator",
        "news anchor",
        "teacher",
        "customer service agent",
        "radio host",
        "storyteller",
        "podcaster",
        "video game character",
        "announcer",
        "friend chatting",
    ],
}

ALWAYS = ("gender", "age")
# Probability that a soft attribute is left unstated.
OMIT = {
    "pitch": 0.3,
    "brightness": 0.4,
    "phonation": 0.5,
    "texture": 0.4,
    "pace": 0.4,
    "energy": 0.4,
    "mood": 0.3,
    "accent": 0.6,
    "role": 0.7,
}

STYLE = (
    "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."
)

SYSTEM = (
    "You write voice descriptions for a text-to-speech voice designer. Given "
    "a set of attributes, write one to three sentences of natural prose that "
    "a person would type to request that voice. Cover every given attribute "
    "and nothing not given; do not name the attribute categories; vary the "
    "sentence structure and vocabulary between items; use plain, concrete "
    f"words about how the voice sounds. House style: {STYLE!r}. Reply with a "
    "JSON array of strings, one per attribute set, in order, nothing else."
)


def sample_attrs(rng: random.Random) -> dict[str, str]:
    attrs = {}
    for key, values in ATTRS.items():
        if key in ALWAYS or rng.random() >= OMIT[key]:
            attrs[key] = rng.choice(values)
    # A nine-year-old is a girl or boy, not a woman or man.
    if attrs["age"].startswith("child"):
        attrs["gender"] = "girl" if attrs["gender"] == "woman" else "boy"
    elif attrs["age"] == "teenager":
        attrs["gender"] = (
            "teenage girl" if attrs["gender"] == "woman" else "teenage boy"
        )
    return attrs


def write_batch(base_url, model, key, batch: list[dict[str, str]]) -> list[str]:
    prompt = "Attribute sets:\n" + "\n".join(
        f"{i + 1}. " + json.dumps(a) for i, a in enumerate(batch)
    )
    for attempt in range(4):
        try:
            reply = llm_chat(
                base_url,
                model,
                key,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
            )
            out = parse_candidates(reply)
            if len(out) == len(batch):
                return out
        except (ValueError, json.JSONDecodeError):
            pass
    return []


SENTENCE_SYSTEM = (
    "You write short, natural English sentences for a text-to-speech test "
    "set: everyday statements, questions, small observations, bits of "
    "narration. Eight to twenty words each, varied topics and rhythms, no "
    "numerals, no names of real people, no lists. Reply with a JSON array of "
    "strings, nothing else."
)


def write_sentences(base_url, model, key, n: int, rng: random.Random) -> list[str]:
    topics = [
        "weather",
        "cooking",
        "travel",
        "work",
        "a garden",
        "the sea",
        "a city at night",
        "an old house",
        "friendship",
        "a museum",
        "sports",
        "music",
        "a train journey",
        "childhood",
        "a market",
        "the mountains",
        "technology",
        "a library",
        "a storm",
        "waiting for someone",
    ]

    def one(_):
        topic = rng.choice(topics)
        try:
            reply = llm_chat(
                base_url,
                model,
                key,
                [
                    {"role": "system", "content": SENTENCE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Twenty sentences loosely about {topic}.",
                    },
                ],
                temperature=1.0,
            )
            return parse_candidates(reply)
        except (ValueError, json.JSONDecodeError):
            return []

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(one, range(n // 20 + 1)))
    sentences = []
    seen = set()
    for batch in batches:
        for s in batch:
            if s not in seen and 5 <= len(s.split()) <= 24:
                seen.add(s)
                sentences.append(s)
    return sentences[:n]


# Template prose for when no LLM is reachable: fine for probing whether the
# latents encode the attributes, too monotonous to train the captioner on.
PHRASES = {
    "pitch": {
        "low": "a low, deep pitch",
        "mid": "a mid-range pitch",
        "high": "a high pitch",
    },
    "brightness": {
        "dark, warm, mellow": "dark, warm, mellow resonance",
        "balanced": "balanced resonance",
        "bright, forward": "bright, forward resonance",
    },
    "phonation": {
        "breathy, airy": "breathy, airy phonation",
        "clear, focused": "clear, focused phonation",
        "tense, pressed": "tense, pressed phonation",
    },
    "texture": {
        "smooth": "a smooth texture",
        "light vocal fry": "light vocal fry",
        "raspy, gravelly": "a raspy, gravelly texture",
        "hoarse": "a hoarse texture",
    },
    "pace": {"slow": "slow pacing", "normal": "normal pacing", "fast": "fast pacing"},
    "energy": {
        "quiet, intimate": "quiet, intimate delivery",
        "calm": "calm delivery",
        "animated": "animated delivery",
        "loud, projected": "loud, projected delivery",
    },
}


def template_prose(attrs: dict[str, str]) -> str:
    head = f"A {attrs['gender']}, {attrs['age']}."
    bits = [PHRASES[k][attrs[k]] for k in PHRASES if k in attrs]
    body = f" The voice has {', '.join(bits)}." if bits else ""
    tail = ""
    if "mood" in attrs:
        tail += f" The mood is {attrs['mood']}."
    if "accent" in attrs:
        tail += f" {attrs['accent']} accent."
    if "role" in attrs:
        tail += f" Sounds like a {attrs['role']}."
    return head + body + tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", action="store_true", help="no LLM: template prose"
    )
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--sentences", type=int, default=800)
    parser.add_argument(
        "--out", type=Path, default=Path("data/caption/descriptions.jsonl")
    )
    parser.add_argument(
        "--sentences-out", type=Path, default=Path("data/caption/sentences.txt")
    )
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm-model", default="huihui-qwen3.8-27b-abliterated")
    parser.add_argument("--llm-parallel", type=int, default=4)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.template:
        with args.out.open("w") as fh:
            for _ in range(args.n):
                attrs = sample_attrs(rng)
                fh.write(
                    json.dumps({"attrs": attrs, "prose": template_prose(attrs)}) + "\n"
                )
        print(f"{args.n} template descriptions -> {args.out}")
        return

    key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    if not key or not base_url:
        sys.exit("set LLM_API_KEY and LLM_BASE_URL")

    if not args.sentences_out.exists():
        sentences = write_sentences(base_url, args.llm_model, key, args.sentences, rng)
        args.sentences_out.write_text("\n".join(sentences) + "\n")
        print(f"{len(sentences)} sentences -> {args.sentences_out}")

    done = 0
    if args.out.exists():
        done = sum(1 for _ in args.out.open())
        print(f"resuming, {done} already written")

    batches = []
    for start in range(done, args.n, args.batch):
        batches.append(
            [sample_attrs(rng) for _ in range(min(args.batch, args.n - start))]
        )

    with (
        args.out.open("a") as fh,
        ThreadPoolExecutor(max_workers=args.llm_parallel) as pool,
    ):
        futures = [
            pool.submit(write_batch, base_url, args.llm_model, key, batch)
            for batch in batches
        ]
        written = done
        for batch, future in zip(batches, futures):
            prose = future.result()
            for attrs, text in zip(batch, prose):
                fh.write(json.dumps({"attrs": attrs, "prose": text}) + "\n")
                written += 1
            fh.flush()
            if written % 300 < args.batch:
                print(f"  {written}", flush=True)
    print(f"{written} descriptions -> {args.out}")


if __name__ == "__main__":
    main()
