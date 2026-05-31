#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wgram_lm.eval.memory_retrieval import (
    audit_records,
    expected_unknown_case,
    score_answer,
    summarize_records,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rescore_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rescored: list[dict[str, Any]] = []
    for record in records:
        if "summary" in record:
            continue
        aliases = record.get("answer_aliases") or []
        score = score_answer(
            str(record.get("completion", "")),
            aliases,
            expected_unknown=expected_unknown_case(record),
        )
        updated = dict(record)
        updated.update(score)
        rescored.append(updated)
    summary = summarize_records(rescored)
    return rescored, summary, audit_records(rescored)


def write_jsonl(path: str | Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


def write_audit_jsonl(path: str | Path, audit_items: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in audit_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def rescore_file(input_jsonl: str, output_jsonl: str, *, audit_jsonl_out: str | None = None) -> dict[str, Any]:
    records, summary, audit_items = rescore_records(load_jsonl(input_jsonl))
    write_jsonl(output_jsonl, records, summary)
    if audit_jsonl_out:
        write_audit_jsonl(audit_jsonl_out, audit_items)
    return {
        "records": len(records),
        "audit_items": len(audit_items),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore MemoryOS eval JSONL with strict answer metrics.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--audit-jsonl-out", default=None)
    args = parser.parse_args()

    result = rescore_file(
        args.input_jsonl,
        args.output_jsonl,
        audit_jsonl_out=args.audit_jsonl_out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
