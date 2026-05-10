#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _step_matches(predicted: list[int], target: list[int]) -> bool:
    for index, target_value in enumerate(target):
        if int(target_value) == -100:
            continue
        if index >= len(predicted) or int(predicted[index]) != int(target_value):
            return False
    return True


def trace_exact_with_oracle_prefix(record: dict[str, Any], prefix_steps: int) -> bool:
    if int(prefix_steps) < 0:
        raise ValueError("prefix_steps must be non-negative")
    predicted_steps = record.get("predicted_values")
    target_steps = record.get("target_values")
    if not isinstance(predicted_steps, list) or not isinstance(target_steps, list):
        raise ValueError("record must contain predicted_values and target_values lists")
    for step_index, target in enumerate(target_steps):
        if step_index < int(prefix_steps):
            continue
        predicted = predicted_steps[step_index] if step_index < len(predicted_steps) else []
        if not _step_matches(list(predicted), list(target)):
            return False
    return True


def oracle_prefix_report(data: dict[str, Any], *, max_prefix_steps: int) -> dict[str, Any]:
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("input JSON must contain records")
    rows = len(records)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    raw_trace = int(summary.get("exact_rows", 0))
    prefix = []
    for prefix_steps in range(0, int(max_prefix_steps) + 1):
        exact_rows = sum(
            int(trace_exact_with_oracle_prefix(record, prefix_steps))
            for record in records
        )
        prefix.append(
            {
                "oracle_prefix_steps": prefix_steps,
                "exact_rows": exact_rows,
                "rows": rows,
                "trace_exact_accuracy": (float(exact_rows) / float(rows))
                if rows
                else 0.0,
            }
        )
    min_prefix_histogram: dict[str, int] = {}
    for record in records:
        required = None
        for prefix_steps in range(0, int(max_prefix_steps) + 1):
            if trace_exact_with_oracle_prefix(record, prefix_steps):
                required = prefix_steps
                break
        key = "unrecovered" if required is None else str(required)
        min_prefix_histogram[key] = min_prefix_histogram.get(key, 0) + 1
    return {
        "rows": rows,
        "raw_exact_rows": raw_trace,
        "oracle_prefix": prefix,
        "min_prefix_histogram": min_prefix_histogram,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how many leading value-state steps would need to be oracle "
            "gold before a predicted recurrent trace becomes exact."
        )
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--max-prefix-steps", type=int, default=8)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    data = json.loads(Path(args.input_json).read_text())
    report = oracle_prefix_report(data, max_prefix_steps=int(args.max_prefix_steps))
    report["input_json"] = str(args.input_json)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if str(args.out_json).strip():
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
