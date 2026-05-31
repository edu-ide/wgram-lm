#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.memory_retrieval import load_cases, normalize_answer


def _case_answer_key(case: dict[str, Any]) -> str:
    aliases = case.get("answer_aliases") or []
    return normalize_answer(str(aliases[0])) if aliases else ""


def _case_evidence_text(case: dict[str, Any]) -> str:
    records = list(case.get("evidence") or []) + list(case.get("distractors") or [])
    return "\n".join(str(record.get("text", "")) for record in records)


def _swap_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(case.get("evidence") or []) + list(case.get("distractors") or [])
    return [dict(record) for record in records]


def choose_swap_case(cases: list[dict[str, Any]], index: int) -> dict[str, Any]:
    original = cases[index]
    answer_key = _case_answer_key(original)
    for offset in range(1, len(cases)):
        candidate = cases[(index + offset) % len(cases)]
        if candidate.get("id") == original.get("id"):
            continue
        if answer_key and answer_key in normalize_answer(_case_evidence_text(candidate)):
            continue
        if _swap_records(candidate):
            return candidate
    raise ValueError(f"could not find counterfactual swap for {original.get('id')}")


def build_workspace_swap_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    case_list = list(cases)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(case_list):
        swap = choose_swap_case(case_list, index)
        row = dict(case)
        row["id"] = f"{case.get('id', f'case-{index}')}__workspace_swap"
        row["category"] = f"{case.get('category', 'uncategorized')}_workspace_swap"
        row["task_family"] = "workspace_counterfactual"
        row["expected_unknown"] = True
        row["answer_aliases"] = ["UNKNOWN", "unknown"]
        row["evidence"] = []
        row["distractors"] = _swap_records(swap)
        row["workspace_counterfactual_source_case_id"] = swap.get("id")
        base_instruction = str(case.get("instruction") or "").strip()
        counterfactual_instruction = (
            "The hidden workspace evidence has been counterfactually swapped. "
            "If it does not explicitly contain the requested answer, answer UNKNOWN."
        )
        row["instruction"] = (
            f"{base_instruction} {counterfactual_instruction}".strip()
            if base_instruction
            else counterfactual_instruction
        )
        rows.append(row)
    return rows


def write_workspace_swap_cases(
    cases_path: str | Path,
    out_path: str | Path,
    *,
    max_cases: int = 0,
) -> int:
    cases = load_cases(cases_path)
    rows = build_workspace_swap_cases(cases)
    if max_cases > 0:
        rows = rows[:max_cases]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build counterfactual MemoryOS eval cases by keeping the visible "
            "question fixed while swapping hidden workspace evidence."
        )
    )
    parser.add_argument("--cases", default="data/eval/memory_reasoning_heldout_expanded_72.jsonl")
    parser.add_argument("--out", default="data/eval/memory_reasoning_workspace_swap.jsonl")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_workspace_swap_cases(args.cases, args.out, max_cases=args.max_cases)
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
