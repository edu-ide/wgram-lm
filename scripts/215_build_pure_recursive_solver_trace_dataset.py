#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: solver-trace cases must not include evidence")
            yield row


def _trace_steps(case: dict[str, Any]) -> list[dict[str, Any]]:
    trace = case.get("solver_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"case {case.get('id', '<missing>')}: missing solver_trace")
    steps: list[dict[str, Any]] = []
    for raw_step in trace:
        if not isinstance(raw_step, dict):
            raise ValueError("solver_trace entries must be objects")
        if "depth" not in raw_step or "operation" not in raw_step or "state_text" not in raw_step:
            raise ValueError("solver_trace entries require depth, operation, and state_text")
        steps.append(
            {
                "depth": int(raw_step["depth"]),
                "operation": str(raw_step["operation"]),
                "state_text": str(raw_step["state_text"]),
            }
        )
    return sorted(steps, key=lambda item: int(item["depth"]))


def rows_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    final_answer = str((case.get("answer_aliases") or [case.get("answer", "")])[0])
    previous_state = ""
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(_trace_steps(case)):
        target_state = str(step["state_text"])
        rows.append(
            {
                "id": f"{case.get('id', 'case')}-trace-{index:02d}-d{int(step['depth'])}",
                "type": "pure_recursive_solver_trace",
                "source_id": case.get("id"),
                "raw_intelligence_axis": case.get(
                    "raw_intelligence_axis",
                    "pure_recursive_reasoning",
                ),
                "category": case.get("category", "uncategorized"),
                "task_family": case.get("task_family", case.get("category", "uncategorized")),
                "reasoning_family": case.get(
                    "reasoning_family",
                    case.get("task_family", case.get("category", "uncategorized")),
                ),
                "expected_paradigm": case.get("expected_paradigm", "unknown"),
                "prompt": str(case.get("prompt", "")),
                "question": str(case.get("question", "")),
                "depth": int(step["depth"]),
                "trace_index": int(index),
                "operation": str(step["operation"]),
                "previous_state_text": previous_state,
                "target_state_text": target_state,
                "final_answer": final_answer,
                "answer_aliases": [final_answer],
                "retrieval_allowed": False,
                "memoryos_allowed": False,
                "evidence": [],
            }
        )
        previous_state = target_state
    return rows


def write_trace_dataset(
    cases_path: str | Path,
    out_path: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in iter_jsonl(cases_path):
        rows.extend(rows_for_case(case))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten pure recursive cases into solver-trace state-machine rows."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = write_trace_dataset(args.cases, args.out)
    print(f"wrote {len(rows)} solver-trace rows to {args.out}")


if __name__ == "__main__":
    main()
