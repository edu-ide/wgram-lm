#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _canonical_answer(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    aliases = metadata.get("answer_aliases") or row.get("answer_aliases") or []
    if aliases:
        first = str(aliases[0]).strip()
        if first.lower() == "unknown":
            return "UNKNOWN"
        return first
    chosen = str(row.get("chosen") or "").strip()
    if chosen.lower().startswith("answer:"):
        chosen = chosen.split(":", 1)[1].strip()
    return chosen.splitlines()[0].strip()


def clean_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        answer = _canonical_answer(row)
        if not answer:
            continue
        out = dict(row)
        metadata = dict(out.get("metadata") or {})
        metadata["clean_intervention_preference"] = True
        out["metadata"] = metadata
        out["chosen"] = f"Answer: {answer}"
        out["rejected"] = str(out.get("rejected") or "").strip()
        if out["rejected"] and out["rejected"] != out["chosen"]:
            cleaned.append(out)
    return cleaned


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        default="data/filtered/memory_reasoning_intervention_preferences_train24.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/filtered/memory_reasoning_intervention_preferences_clean_train24.jsonl",
    )
    args = parser.parse_args(argv)

    rows = clean_rows(_read_jsonl(Path(args.input_jsonl)))
    _write_jsonl(Path(args.output_jsonl), rows)
    print(f"wrote {len(rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
