#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from wgram_lm.distill.training_mix import build_training_mix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build QTRM training JSONL from converted teacher-record JSONL files."
    )
    parser.add_argument("--input", action="append", required=True, help="Teacher-record JSONL path.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-evidence-chars", type=int, default=4000)
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Do not wrap memory_docs into MemoryOS workspace prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_training_mix(
        inputs=[Path(item) for item in args.input],
        out_path=Path(args.out),
        max_rows_per_source=args.max_rows_per_source,
        include_evidence=not args.no_evidence,
        max_evidence_chars=args.max_evidence_chars,
        preference_weight=args.preference_weight,
    )
    print(
        "written={written} read={read} skipped={skipped} out={out} sources={sources}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
