#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wgram_lm.eval.memory_retrieval import load_cases
from wgram_lm.training.synthetic_memory_cases import write_synthetic_memory_cases_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic MemoryOS reasoning train cases.")
    parser.add_argument("--out", default="data/filtered/memory_reasoning_synth_train_cases.jsonl")
    parser.add_argument("--num-sets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--avoid-cases",
        action="append",
        default=[
            "data/eval/memory_reasoning_probe.jsonl",
            "data/eval/memory_reasoning_heldout_probe.jsonl",
        ],
    )
    args = parser.parse_args()

    avoid_ids = set()
    for path in args.avoid_cases:
        try:
            avoid_ids.update(case["id"] for case in load_cases(path))
        except FileNotFoundError:
            continue

    count = write_synthetic_memory_cases_jsonl(
        args.out,
        num_sets=args.num_sets,
        seed=args.seed,
        avoid_ids=avoid_ids,
        start_index=args.start_index,
    )
    print(
        f"wrote {args.out}, cases={count}, num_sets={args.num_sets}, "
        f"seed={args.seed}, start_index={args.start_index}"
    )


if __name__ == "__main__":
    main()
