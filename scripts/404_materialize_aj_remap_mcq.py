#!/usr/bin/env python3
"""Build A-J remapped MCQ training data from non-test MCQ JSONL files.

Many auxiliary MCQ datasets only use A-D labels, while MMLU-Pro uses up to
A-J. This materializer preserves each source row's correct option text but
randomly places it into a 10-option A-J prompt, filling the remaining slots with
wrong options from the same/non-test pool. The purpose is to train the native LM
answer path in the same A-J label space without using MMLU-Pro test labels.
"""

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


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row must be an object at {path}:{line_no}")
            options = row.get("options")
            answer = normalize_answer(row.get("answer", ""))
            if not isinstance(options, list) or not options or not answer:
                raise ValueError(f"invalid MCQ row at {path}:{line_no}")
            if OPTION_LETTERS.index(answer) >= len(options):
                raise ValueError(f"answer outside options at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError("no rows loaded")
    return rows


def option_text(row: dict[str, Any], letter: str) -> str:
    return clean_text(row["options"][OPTION_LETTERS.index(letter)])


def wrong_option_pool(rows: list[dict[str, Any]]) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()
    for row in rows:
        answer = normalize_answer(row.get("answer", ""))
        for index, value in enumerate(row.get("options", [])):
            letter = OPTION_LETTERS[index]
            text = clean_text(value)
            if letter != answer and text and text not in seen:
                seen.add(text)
                pool.append(text)
    if len(pool) < 16:
        raise ValueError("not enough wrong options to build A-J remaps")
    return pool


def format_prompt(question: str, options: list[str], *, source_name: str) -> str:
    option_lines = "\n".join(
        f"{OPTION_LETTERS[index]}. {option}"
        for index, option in enumerate(options)
    )
    return (
        f"User: Answer the following {source_name} multiple-choice question.\n"
        "Return only one option letter, with no explanation.\n\n"
        f"Question: {clean_text(question)}\n"
        f"Options:\n{option_lines}\n\n"
        "Answer:\n"
        "Assistant:"
    )


def remap_row(
    row: dict[str, Any],
    *,
    pool: list[str],
    rng: random.Random,
    source_index: int,
    repeat_index: int,
) -> dict[str, Any]:
    question = clean_text(row.get("question", ""))
    if not question:
        raise ValueError("source row missing question")
    answer = normalize_answer(row.get("answer", ""))
    correct_text = option_text(row, answer)
    source_wrong = [
        clean_text(value)
        for index, value in enumerate(row.get("options", []))
        if OPTION_LETTERS[index] != answer and clean_text(value)
    ]
    chosen: list[str] = []
    seen = {correct_text}
    for value in source_wrong:
        if value not in seen:
            chosen.append(value)
            seen.add(value)
    pool_candidates = list(pool)
    rng.shuffle(pool_candidates)
    for value in pool_candidates:
        if len(chosen) >= 9:
            break
        if value and value not in seen:
            chosen.append(value)
            seen.add(value)
    if len(chosen) < 9:
        raise ValueError("failed to build 9 distractors")
    target_index = rng.randrange(10)
    options = chosen[:9]
    options.insert(target_index, correct_text)
    remapped_answer = OPTION_LETTERS[target_index]
    source_name = str(row.get("dataset") or row.get("benchmark_id") or "remapped")
    return {
        "benchmark_id": "aj_remap_public_mcq",
        "source_benchmark_id": row.get("benchmark_id", ""),
        "dataset": row.get("dataset", ""),
        "config": row.get("config", ""),
        "split": row.get("split", ""),
        "case_id": f"aj-remap-{source_index:06d}-{repeat_index:03d}",
        "source_case_id": row.get("case_id", ""),
        "row_idx": int(source_index),
        "category": row.get("category", "unknown"),
        "question": question,
        "options": options,
        "answer": remapped_answer,
        "answer_index": target_index,
        "original_answer": answer,
        "qtrm_prompt": format_prompt(question, options, source_name=source_name),
        "scorer": "exact option-letter match",
    }


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_rows = load_rows(list(args.source_jsonl))
    anchor_rows = load_rows(list(args.anchor_jsonl)) if args.anchor_jsonl else []
    pool = wrong_option_pool(source_rows + anchor_rows)
    rng = random.Random(int(args.seed))
    source_indexed = list(enumerate(source_rows))
    cases: list[dict[str, Any]] = []
    for repeat_index in range(max(1, int(args.augment_repeats))):
        current = list(source_indexed)
        if bool(args.shuffle):
            rng.shuffle(current)
        for source_index, row in current:
            cases.append(
                remap_row(
                    row,
                    pool=pool,
                    rng=rng,
                    source_index=int(source_index),
                    repeat_index=repeat_index,
                )
            )
            if int(args.max_augmented_cases) > 0 and len(cases) >= int(args.max_augmented_cases):
                break
        if int(args.max_augmented_cases) > 0 and len(cases) >= int(args.max_augmented_cases):
            break
    if anchor_rows:
        cases.extend(dict(row, benchmark_id=str(row.get("benchmark_id", "anchor_public_mcq"))) for row in anchor_rows)
    if bool(args.shuffle):
        rng.shuffle(cases)
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]
    return cases


def write_outputs(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    out_jsonl = Path(args.out_jsonl)
    out_report = Path(args.out_report)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )
    by_answer: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in cases:
        by_answer[str(row["answer"])] = by_answer.get(str(row["answer"]), 0) + 1
        by_category[str(row.get("category", "unknown"))] = by_category.get(str(row.get("category", "unknown")), 0) + 1
    report = {
        "status": "complete",
        "decision": "built_aj_remap_public_mcq",
        "accepted": True,
        "source_jsonl": list(args.source_jsonl),
        "anchor_jsonl": list(args.anchor_jsonl or []),
        "out_jsonl": str(out_jsonl),
        "cases": len(cases),
        "augment_repeats": int(args.augment_repeats),
        "max_augmented_cases": int(args.max_augmented_cases),
        "by_answer": dict(sorted(by_answer.items())),
        "by_category": dict(sorted(by_category.items())),
        "policy": "non-test remap data; do not use MMLU-Pro test labels for training or selection",
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", action="append", required=True)
    parser.add_argument("--anchor-jsonl", action="append", default=[])
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--augment-repeats", type=int, default=1)
    parser.add_argument("--max-augmented-cases", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = build_cases(args)
    print(json.dumps(write_outputs(args, cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
