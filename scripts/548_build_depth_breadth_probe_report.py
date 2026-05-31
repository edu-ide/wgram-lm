#!/usr/bin/env python3
"""Build an EqR-style depth/breadth convergence report from eval JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wgram_lm.eval.depth_breadth_probe import build_depth_breadth_report, load_jsonl  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="JSONL eval rows with depth/restart/residual fields")
    parser.add_argument("--out", default="", help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.rows)
    report = build_depth_breadth_report(rows)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
