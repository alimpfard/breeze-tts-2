"""Rewrite design prompts from relabelled attributes (what Breeze heard).

caption.relabel decides, per clip, which attribute values Breeze's own
likelihood prefers. This has the LLM write those up as prose, the same way
caption.data wrote the requested ones, so a captioner can be trained on
descriptions of the realised voice rather than the requested one.

Resumable: ids already in the output are skipped, so it can be run while
relabel is still filling its shards.

    LLM_BASE_URL=http://127.0.0.1:8089/v1 LLM_API_KEY=x \
        python -m caption.rewrite --model gemma-4-26B-A4B-it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.data import write_batch

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel", type=Path, default=REPO / "data/caption/relabel")
    parser.add_argument(
        "--out", type=Path, default=REPO / "data/caption/relabel/prose.jsonl"
    )
    parser.add_argument("--model", default="gemma-4-26B-A4B-it")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    if not key or not base_url:
        sys.exit("set LLM_API_KEY and LLM_BASE_URL")

    recs = {}
    for shard in sorted(args.relabel.glob("shard*.jsonl")):
        for line in shard.read_text().splitlines():
            d = json.loads(line)
            recs[d["id"]] = d["attrs"]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines()}
    todo = sorted(i for i in recs if i not in done)
    print(f"{len(recs)} relabelled, {len(done)} written, {len(todo)} to do")
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]

    def run(ids):
        prose = write_batch(base_url, args.model, key, [recs[i] for i in ids])
        return list(zip(ids, prose))

    n = 0
    with args.out.open("a") as fh, ThreadPoolExecutor(args.parallel) as pool:
        for got in pool.map(run, batches):
            for i, p in got:
                fh.write(json.dumps({"id": i, "attrs": recs[i], "prose": p}) + "\n")
            fh.flush()
            n += len(got)
            if n % 120 < args.batch:
                print(f"  {n}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
