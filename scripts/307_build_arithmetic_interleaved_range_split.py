#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_case_builder():
    path = Path(__file__).with_name("190_build_pure_recursive_reasoning_cases.py")
    spec = importlib.util.spec_from_file_location("pure_recursive_case_builder_307", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_interleaved_split(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    start_index: int = 18,
    train_cases: int = 64,
    eval_cases: int = 64,
) -> dict[str, Any]:
    builder = _load_case_builder()
    needed = int(train_cases) + int(eval_cases)
    candidates = builder.filter_cases_by_family(
        builder.build_cases(cases_per_family=max(1, needed * 2), start_index=int(start_index)),
        {"arithmetic_chain"},
    )
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for offset, row in enumerate(candidates):
        if offset % 2 == 0 and len(train_rows) < int(train_cases):
            train_rows.append(row)
        elif offset % 2 == 1 and len(eval_rows) < int(eval_cases):
            eval_rows.append(row)
        if len(train_rows) >= int(train_cases) and len(eval_rows) >= int(eval_cases):
            break
    if len(train_rows) != int(train_cases) or len(eval_rows) != int(eval_cases):
        raise RuntimeError("failed to build requested interleaved split")
    train_ids = {str(row["id"]) for row in train_rows}
    eval_ids = {str(row["id"]) for row in eval_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise RuntimeError(f"train/eval overlap: {overlap[:5]}")
    _write_jsonl(train_out, train_rows)
    _write_jsonl(eval_out, eval_rows)
    return {
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "start_index": int(start_index),
        "train_cases": len(train_rows),
        "eval_cases": len(eval_rows),
        "train_first_id": train_rows[0]["id"],
        "train_last_id": train_rows[-1]["id"],
        "eval_first_id": eval_rows[0]["id"],
        "eval_last_id": eval_rows[-1]["id"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an interleaved arithmetic train/eval split for renderer range-generalization gates."
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--start-index", type=int, default=18)
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--eval-cases", type=int, default=64)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_interleaved_split(
        train_out=args.train_out,
        eval_out=args.eval_out,
        start_index=args.start_index,
        train_cases=args.train_cases,
        eval_cases=args.eval_cases,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
