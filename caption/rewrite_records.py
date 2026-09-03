"""Replace template prose in rendered records with LLM-written prompts.

caption.minimal and caption.paired write template prose ("A man, elderly,
seventies. The mood is neutral."). Mixed into training that teaches the
captioner template-ese (11/40 captions in v6). This rewrites each record's
prose from its attrs with caption.data's writer; the template is kept as
``template_prose``. Idempotent.

    LLM_BASE_URL=http://127.0.0.1:8089/v1 LLM_API_KEY=x \
        python -m caption.rewrite_records data/caption/paired_recs data/caption/minimal
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caption.data import write_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--model", default="gemma-4-26B-A4B-it")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    key, base_url = os.environ.get("LLM_API_KEY"), os.environ.get("LLM_BASE_URL")
    if not key or not base_url:
        sys.exit("set LLM_API_KEY and LLM_BASE_URL")

    todo = []
    for d in args.dirs:
        for p in sorted(d.glob("*.pt")):
            r = torch.load(p)
            if "template_prose" not in r:
                todo.append((p, r))
    print(f"{len(todo)} records to rewrite", flush=True)
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]

    def run(batch):
        prose = write_batch(base_url, args.model, key, [r["attrs"] for _, r in batch])
        return list(zip(batch, prose))

    n = 0
    with ThreadPoolExecutor(args.parallel) as pool:
        for got in pool.map(run, batches):
            for (p, r), prose in got:
                r["template_prose"], r["prose"] = r["prose"], prose
                torch.save(r, p)
            n += len(got)
            if n % 120 < args.batch:
                print(f"  {n}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
