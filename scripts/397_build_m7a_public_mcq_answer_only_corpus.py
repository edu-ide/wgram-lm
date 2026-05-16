#!/usr/bin/env python3
"""Build answer-only public MCQ healing data for QTRM-native M7A."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


OPTION_LETTERS = "ABCDEFGHIJ"


def normalize_answer(value: object) -> str:
    text = str(value).strip().upper()
    if text in OPTION_LETTERS:
        return text
    match = re.search(r"\b([A-J])\b", text)
    return match.group(1) if match else ""


def load_suite(path: str | Path, *, max_records: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row must be an object at {path}:{line_no}")
        for key in ("qtrm_prompt", "answer"):
            if key not in row:
                raise ValueError(f"row missing {key} at {path}:{line_no}")
        answer = normalize_answer(row["answer"])
        if not answer:
            raise ValueError(f"row has invalid answer at {path}:{line_no}: {row.get('answer')!r}")
        rows.append(row)
        if int(max_records) > 0 and len(rows) >= int(max_records):
            break
    if not rows:
        raise ValueError(f"suite is empty: {path}")
    return rows


def answer_record_text(row: dict[str, Any]) -> str:
    prompt = str(row["qtrm_prompt"]).rstrip()
    answer = normalize_answer(row["answer"])
    return f"{prompt} {answer}\n"


def build_records(
    rows: list[dict[str, Any]],
    *,
    repeats: int,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    records: list[dict[str, Any]] = []
    base_rows = list(rows)
    for repeat_index in range(max(1, int(repeats))):
        current = list(base_rows)
        if bool(shuffle):
            rng.shuffle(current)
        for row in current:
            records.append(
                {
                    "text": answer_record_text(row),
                    "benchmark_id": row.get("benchmark_id", ""),
                    "case_id": row.get("case_id", ""),
                    "category": row.get("category", ""),
                    "answer": normalize_answer(row["answer"]),
                    "repeat_index": repeat_index,
                }
            )
    return records


def repair_seed_texts(rows: list[dict[str, Any]], *, count: int) -> str:
    seeds = [str(row["qtrm_prompt"]).rstrip() + " " for row in rows[: max(1, int(count))]]
    return "||".join(seeds)


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_suite(args.suite_jsonl, max_records=int(args.max_records))
    if bool(args.shuffle):
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
    records = build_records(
        rows,
        repeats=int(args.repeats),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + 17,
    )
    out_jsonl = Path(args.out_jsonl)
    out_json = Path(args.out_json)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "decision": "built_m7a_public_mcq_answer_only_corpus",
        "suite_jsonl": str(args.suite_jsonl),
        "out_jsonl": str(out_jsonl),
        "source_rows": len(rows),
        "records": len(records),
        "repeats": int(args.repeats),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "repair_seed_texts": repair_seed_texts(rows, count=int(args.repair_seed_count)),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=397)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-seed-count", type=int, default=3)
    return parser


def main() -> None:
    report = build_corpus(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
