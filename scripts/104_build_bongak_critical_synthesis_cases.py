#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wgram_lm.training.bongak_critical_synthesis_cases import (
    DEFAULT_BONGAK_MANUAL,
    DEFAULT_BONGAK_SUMMARY,
    write_bongak_cases_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build critical-synthesis cases and traces from local 본각교 documents."
    )
    parser.add_argument("--summary", default=str(DEFAULT_BONGAK_SUMMARY))
    parser.add_argument("--manual", default=str(DEFAULT_BONGAK_MANUAL))
    parser.add_argument("--out", default="data/filtered/critical_synthesis_bongak_cases.jsonl")
    parser.add_argument("--traces-out", default="data/filtered/critical_synthesis_bongak_traces.jsonl")
    parser.add_argument("--max-cases", type=int, default=30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_bongak_cases_jsonl(
        summary_path=args.summary,
        manual_path=args.manual,
        out_path=args.out,
        traces_out_path=args.traces_out,
        max_cases=args.max_cases,
    )
    print(f"wrote {args.out}, rows={count}")
    if args.traces_out:
        print(f"wrote {args.traces_out}, rows={count}")


if __name__ == "__main__":
    main()
