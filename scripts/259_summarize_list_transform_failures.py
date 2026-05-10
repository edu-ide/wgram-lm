#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _normalize(text: Any) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def _reverse_comma_text(text: Any) -> str:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) <= 1:
        return str(text).strip()
    return ",".join(reversed(parts))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _case_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in _load_jsonl(path)}


def _choice_score(row: dict[str, Any], target: str) -> float | None:
    target_norm = _normalize(target)
    for choice in row.get("choice_scores") or []:
        if _normalize(choice.get("choice")) == target_norm:
            return float(choice.get("logprob"))
    return None


def _choice_rank(row: dict[str, Any], target: str) -> int | None:
    target_norm = _normalize(target)
    for index, choice in enumerate(row.get("choice_scores") or [], start=1):
        if _normalize(choice.get("choice")) == target_norm:
            return index
    return None


def classify_list_failure(row: dict[str, Any], case: dict[str, Any]) -> str:
    completion = str(row.get("completion", "")).strip()
    answer = str((row.get("answer_aliases") or case.get("answer_aliases") or [""])[0])
    if _normalize(completion) == _normalize(answer):
        return "correct"
    depth_targets = case.get("depth_targets") or {}
    if _normalize(completion) == _normalize(depth_targets.get("1", "")):
        return "filtered_state_selected"
    if _normalize(completion) == _normalize(_reverse_comma_text(answer)):
        return "reversed_final_selected"
    if _normalize(completion) == _normalize("EMPTY"):
        return "empty_selected"
    return "other_wrong"


def summarize_failures(
    eval_rows: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    records = []
    for row in eval_rows:
        if row.get("mode") != mode or row.get("task_family") != "list_transform":
            continue
        case = cases_by_id.get(str(row.get("id")), {})
        answer = str((row.get("answer_aliases") or case.get("answer_aliases") or [""])[0])
        selected = str(row.get("completion", ""))
        selected_score = _choice_score(row, selected)
        correct_score = _choice_score(row, answer)
        correct_rank = _choice_rank(row, answer)
        gap = (
            None
            if selected_score is None or correct_score is None
            else correct_score - selected_score
        )
        records.append(
            {
                "id": row.get("id"),
                "hit": bool(row.get("hit")),
                "error_type": classify_list_failure(row, case),
                "completion": selected,
                "answer": answer,
                "correct_rank": correct_rank,
                "correct_minus_selected_score": gap,
                "depth_targets": case.get("depth_targets"),
                "choices": case.get("choices"),
                "prompt": case.get("prompt"),
            }
        )

    by_error = Counter(record["error_type"] for record in records)
    gaps = [
        float(record["correct_minus_selected_score"])
        for record in records
        if record["correct_minus_selected_score"] is not None
    ]
    return {
        "mode": mode,
        "total": len(records),
        "hits": sum(bool(record["hit"]) for record in records),
        "by_error": dict(by_error),
        "mean_correct_minus_selected_score": (
            sum(gaps) / len(gaps) if gaps else None
        ),
        "records": records,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# List Transform Failure Ledger",
        "",
        f"Mode: `{summary['mode']}`",
        "",
        "## Summary",
        "",
        "```text",
        f"hits: {summary['hits']}/{summary['total']}",
        f"by_error: {summary['by_error']}",
        f"mean_correct_minus_selected_score: {summary['mean_correct_minus_selected_score']}",
        "```",
        "",
        "## Records",
        "",
    ]
    for record in summary["records"]:
        lines.extend(
            [
                f"### {record['id']}",
                "",
                "```text",
                f"hit: {record['hit']}",
                f"error_type: {record['error_type']}",
                f"completion: {record['completion']}",
                f"answer: {record['answer']}",
                f"correct_rank: {record['correct_rank']}",
                f"correct_minus_selected_score: {record['correct_minus_selected_score']}",
                f"depth_targets: {record['depth_targets']}",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--cases-jsonl", required=True)
    parser.add_argument("--mode", default="qtrm_core_steps_8_no_evidence")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = summarize_failures(
        _load_jsonl(args.eval_jsonl),
        _case_by_id(args.cases_jsonl),
        mode=args.mode,
    )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_markdown(summary) + "\n", encoding="utf-8")
    print(
        f"{summary['mode']}: {summary['hits']}/{summary['total']} "
        f"by_error={summary['by_error']}"
    )


if __name__ == "__main__":
    main()
