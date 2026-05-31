#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wgram_lm.training.memory_trace_data import write_memory_trace_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build supervised MemoryOS trace JSONL from hard memory reasoning cases."
    )
    parser.add_argument("--cases", default="data/eval/memory_reasoning_probe.jsonl")
    parser.add_argument("--out", default="data/filtered/memory_abstention_traces.jsonl")
    parser.add_argument("--variant", action="append", default=None, help="target, all, or lexical. Can repeat.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evidence-chars", type=int, default=2000)
    args = parser.parse_args()

    variants = tuple(args.variant or ["target", "all", "lexical"])
    count = write_memory_trace_jsonl(
        args.cases,
        args.out,
        variants=variants,
        top_k=args.top_k,
        max_evidence_chars=args.max_evidence_chars,
    )
    print(f"wrote {args.out}, rows={count}, variants={','.join(variants)}")


if __name__ == "__main__":
    main()
