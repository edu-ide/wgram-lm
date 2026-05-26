#!/usr/bin/env python3
"""Evaluate Stage59-style general answer candidates.

This script is intentionally only an answer-interface evaluator. It does not
run a task solver and it does not know how to execute any family-specific
program. Candidate answers may come from a model JSONL, from row choices for an
interface smoke test, or from gold answers for a canary check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.eval.general_answer_interface import (
    answer_aliases,
    answer_kind,
    select_candidate,
    summarize_records,
)


ID_KEYS = ("id", "case_id", "example_id", "uid")
CANDIDATE_LIST_KEYS = ("candidates", "candidate_texts", "answer_candidates", "topk_answers")
SINGLE_CANDIDATE_KEYS = ("pred_answer", "prediction", "completion", "raw_completion", "answer")


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row must be a JSON object at {path}:{line_no}")
        rows.append(row)
        if int(limit) > 0 and len(rows) >= int(limit):
            break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ID_KEYS:
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError(f"row has no id key among {ID_KEYS}: {row}")


def build_candidate_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_id(row)
        if key in index:
            raise ValueError(f"duplicate candidate id: {key}")
        index[key] = row
    return index


def candidate_list_from_row(row: dict[str, Any]) -> list[str]:
    for key in CANDIDATE_LIST_KEYS:
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    for key in SINGLE_CANDIDATE_KEYS:
        if row.get(key) is not None:
            return [str(row[key])]
    return []


def candidates_for_eval_row(
    eval_row: dict[str, Any],
    *,
    candidate_source: str,
    candidate_index: dict[str, dict[str, Any]] | None,
) -> list[str]:
    if candidate_source == "jsonl":
        if candidate_index is None:
            raise ValueError("candidate_index is required for candidate_source=jsonl")
        candidate_row = candidate_index.get(row_id(eval_row))
        return candidate_list_from_row(candidate_row or {})
    if candidate_source == "choices":
        choices = eval_row.get("choices")
        return [str(choice) for choice in choices] if isinstance(choices, list) else []
    if candidate_source == "gold":
        return list(answer_aliases(eval_row)[:1])
    raise ValueError(f"unsupported candidate_source: {candidate_source}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    eval_rows = load_jsonl(args.eval_jsonl, limit=int(args.limit))
    candidate_index = None
    if args.candidates_jsonl:
        candidate_rows = load_jsonl(args.candidates_jsonl)
        candidate_index = build_candidate_index(candidate_rows)
    elif args.candidate_source == "jsonl":
        raise ValueError("--candidates-jsonl is required when --candidate-source jsonl")

    scored: list[dict[str, Any]] = []
    missing_candidates = 0
    for row in eval_rows:
        candidates = candidates_for_eval_row(
            row,
            candidate_source=args.candidate_source,
            candidate_index=candidate_index,
        )
        if not candidates:
            missing_candidates += 1
        aliases = answer_aliases(row)
        selection = select_candidate(candidates, aliases, selection_mode=args.selection_mode)
        record = {
            "id": row_id(row),
            "task_family": row.get("task_family") or row.get("family") or "unknown",
            "answer_kind": answer_kind(aliases[0] if aliases else ""),
            "aliases": list(aliases),
            "candidates": candidates,
            "selected": selection.selected,
            "normalized_selected": selection.normalized_selected,
            "selected_index": selection.selected_index,
            "exact": selection.exact,
            "oracle_exact": selection.oracle_exact,
            "oracle_index": selection.oracle_index,
            "selection_mode": selection.selection_mode,
        }
        scored.append(record)

    summary = summarize_records(scored)
    oracle_hits = sum(1 for row in scored if bool(row.get("oracle_exact")))
    summary.update(
        {
            "stage": "Stage59 general answer candidate evaluation",
            "eval_jsonl": str(args.eval_jsonl),
            "candidates_jsonl": str(args.candidates_jsonl) if args.candidates_jsonl else "",
            "candidate_source": args.candidate_source,
            "selection_mode": args.selection_mode,
            "oracle_hits": oracle_hits,
            "oracle_accuracy": float(oracle_hits / max(1, len(scored))),
            "missing_candidates": missing_candidates,
            "plain_language_read": (
                "This evaluates whether the answer mouth can handle general answer objects. "
                "The 'exact' score is the selected answer score; 'oracle_accuracy' is only "
                "candidate coverage and must not be reported as deployed verifier accuracy."
            ),
        }
    )
    return {"summary": summary, "records": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--candidates-jsonl", default="")
    parser.add_argument("--candidate-source", choices=("jsonl", "choices", "gold"), default="jsonl")
    parser.add_argument("--selection-mode", choices=("first", "oracle"), default="first")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-json", default="local_eval/stage59_general_answer_candidates/report.json")
    parser.add_argument("--out-jsonl", default="local_eval/stage59_general_answer_candidates/records.jsonl")
    args = parser.parse_args()

    result = evaluate(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in result["records"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
