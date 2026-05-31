#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wgram_lm.training.critical_synthesis_data import write_critical_synthesis_trace_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build supervised critical-synthesis traces from religion/value probe cases."
    )
    parser.add_argument("--cases", default="data/eval/critical_synthesis_probe.jsonl")
    parser.add_argument("--out", default="data/filtered/critical_synthesis_traces.jsonl")
    parser.add_argument("--max-evidence-chars", type=int, default=4000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_critical_synthesis_trace_jsonl(
        args.cases,
        args.out,
        max_evidence_chars=args.max_evidence_chars,
    )
    print(f"wrote {args.out}, rows={count}")


if __name__ == "__main__":
    main()
