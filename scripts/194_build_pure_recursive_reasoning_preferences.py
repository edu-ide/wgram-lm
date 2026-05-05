#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if case.get("evidence"):
                raise ValueError(f"{path}:{line_no}: raw preference cases must not include evidence")
            if not case.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not case.get("answer_aliases"):
                raise ValueError(f"{path}:{line_no}: missing answer_aliases")
            if not case.get("choices"):
                raise ValueError(f"{path}:{line_no}: missing choices")
            cases.append(case)
    return cases


def preference_rows_for_case(
    case: dict[str, Any],
    *,
    max_rejected_per_case: int = 3,
) -> list[dict[str, Any]]:
    chosen = str(case["answer_aliases"][0])
    rejected_choices = [
        str(choice)
        for choice in case.get("choices", [])
        if str(choice).casefold() != chosen.casefold()
    ][: max(1, int(max_rejected_per_case))]
    rows: list[dict[str, Any]] = []
    for idx, rejected in enumerate(rejected_choices):
        rows.append(
            {
                "id": f"{case.get('id', 'case')}-pref-{idx}",
                "type": "pure_recursive_reasoning_preference",
                "source_id": case.get("id"),
                "raw_intelligence_axis": case.get("raw_intelligence_axis", "pure_recursive_reasoning"),
                "category": case.get("category", "uncategorized"),
                "task_family": case.get("task_family", case.get("category", "uncategorized")),
                "reasoning_family": case.get(
                    "reasoning_family",
                    case.get("task_family", case.get("category", "uncategorized")),
                ),
                "expected_unknown": bool(case.get("expected_unknown", False)),
                "uncertainty_type": case.get("uncertainty_type", "unknown"),
                "expected_paradigm": case.get("expected_paradigm", "unknown"),
                "requires_stochasticity": bool(case.get("requires_stochasticity", False)),
                "parallel_depth_estimate": case.get("parallel_depth_estimate"),
                "serial_trace_length_estimate": case.get("serial_trace_length_estimate"),
                "prompt": str(case["prompt"]),
                "chosen": chosen,
                "rejected": rejected,
                "answer": chosen,
                "answer_aliases": [chosen],
                "choices": case.get("choices", []),
                "depth_targets": case.get("depth_targets", {}),
                "transition_state_codes": case.get("transition_state_codes", {}),
                "solver_trace": case.get("solver_trace", []),
                "preference_weight": 1.0,
                "retrieval_allowed": False,
                "memoryos_allowed": False,
                "evidence": [],
            }
        )
    return rows


def write_preferences(
    cases_path: str | Path,
    out_path: str | Path,
    *,
    max_rejected_per_case: int = 3,
    only_expected_unknown: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in load_cases(cases_path):
        if bool(only_expected_unknown) and not bool(case.get("expected_unknown", False)):
            continue
        rows.extend(
            preference_rows_for_case(
                case,
                max_rejected_per_case=max_rejected_per_case,
            )
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pure recursive reasoning chosen/rejected preference rows."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", default="data/filtered/pure_recursive_reasoning_preferences_train.jsonl")
    parser.add_argument("--max-rejected-per-case", type=int, default=3)
    parser.add_argument(
        "--only-expected-unknown",
        action="store_true",
        help="Emit preferences only for cases marked expected_unknown=true.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = write_preferences(
        args.cases,
        args.out,
        max_rejected_per_case=args.max_rejected_per_case,
        only_expected_unknown=args.only_expected_unknown,
    )
    print(f"wrote {len(rows)} preference rows to {args.out}")


if __name__ == "__main__":
    main()
