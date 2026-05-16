#!/usr/bin/env python3
"""Build non-test MCQ knowledge-text records for QTRM-native language bootstrap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


OPTION_LETTERS = "ABCDEFGHIJ"


def clean_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


def normalize_answer(value: object) -> str:
    text = str(value).strip().upper()
    if text in OPTION_LETTERS:
        return text
    return ""


def correct_option_text(row: dict[str, Any]) -> str:
    options = row.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError("row missing non-empty options")
    answer = normalize_answer(row.get("answer", ""))
    if not answer:
        raise ValueError("row missing option-letter answer")
    answer_index = OPTION_LETTERS.index(answer)
    if answer_index >= len(options):
        raise ValueError("answer outside options")
    return clean_text(options[answer_index])


def load_rows(paths: list[str], *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row must be an object at {path}:{line_no}")
            if not clean_text(row.get("question", "")):
                raise ValueError(f"row missing question at {path}:{line_no}")
            correct_option_text(row)
            rows.append(row)
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                return rows
    if not rows:
        raise ValueError("no rows loaded")
    return rows


def record_text(row: dict[str, Any]) -> str:
    question = clean_text(row.get("question", ""))
    answer_text = correct_option_text(row)
    category = clean_text(row.get("category", "general")) or "general"
    return (
        "User: Answer the question in words, using the correct choice content.\n"
        f"Category: {category}\n"
        f"Question: {question}\n"
        f"Assistant: {answer_text}"
    )


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_rows(list(args.source_jsonl), max_rows=int(args.max_rows))
    records: list[dict[str, Any]] = []
    for repeat_index in range(max(1, int(args.repeats))):
        current = list(rows)
        if bool(args.shuffle):
            random.Random(int(args.seed) + repeat_index).shuffle(current)
        for row in current:
            records.append(
                {
                    "text": record_text(row),
                    "source_benchmark_id": row.get("benchmark_id", ""),
                    "source_case_id": row.get("case_id", ""),
                    "category": row.get("category", "unknown"),
                    "answer": normalize_answer(row.get("answer", "")),
                    "repeat_index": repeat_index,
                }
            )
    return records


def write_outputs(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_jsonl = Path(args.out_jsonl)
    out_report = Path(args.out_report)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    by_category: dict[str, int] = {}
    for record in records:
        category = str(record.get("category", "unknown"))
        by_category[category] = by_category.get(category, 0) + 1
    report = {
        "status": "complete",
        "decision": "built_mcq_knowledge_text_corpus",
        "accepted": True,
        "source_jsonl": list(args.source_jsonl),
        "out_jsonl": str(out_jsonl),
        "records": len(records),
        "repeats": int(args.repeats),
        "by_category": dict(sorted(by_category.items())),
        "policy": "non-test MCQ rows only; trains answer-content language, not public test labels",
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", action="append", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=406)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = write_outputs(args, build_records(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
