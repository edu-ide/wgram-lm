#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TIE_COMPLETION = "__FORCED_CHOICE_TIE__"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def normalize(text: Any) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def parse_mixed_question(question: str) -> dict[str, Any] | None:
    list_match = re.search(r"\[([^\]]+)\]", str(question))
    if list_match is None:
        return None
    values = [
        int(part.strip())
        for part in list_match.group(1).split(",")
        if part.strip()
    ]
    offset_match = re.search(
        r"\b(?:subtract|minus)[ -]?(\d+)\b",
        str(question).casefold(),
    )
    if offset_match is None:
        return None
    offset = int(offset_match.group(1))
    filtered = [value for value in values if value % 2 == 0]
    doubled = [value * 2 for value in filtered]
    doubled_text = ",".join(str(value) for value in doubled) if doubled else "EMPTY"
    summed = sum(doubled)
    final = summed - offset
    return {
        "values": values,
        "offset": offset,
        "doubled_text": doubled_text,
        "pre_subtract_sum": str(summed),
        "final_answer": str(final),
    }


def expected_final(record: dict[str, Any], parsed: dict[str, Any] | None) -> str:
    aliases = record.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    if parsed is not None:
        return str(parsed["final_answer"])
    return str(record.get("canonical_answer") or "")


def classify_record(record: dict[str, Any]) -> str:
    completion = str(record.get("completion") or "").strip()
    if bool(record.get("hit")):
        return "correct_final"
    if completion == TIE_COMPLETION or bool(record.get("choice_tied")):
        return "forced_choice_tie"

    parsed = parse_mixed_question(str(record.get("question") or ""))
    final = expected_final(record, parsed)
    if final and normalize(completion) == normalize(final):
        return "correct_final"
    if parsed is not None:
        if normalize(completion) == normalize(parsed["pre_subtract_sum"]):
            return "pre_subtract_sum"
        if normalize(completion) == normalize(parsed["doubled_text"]):
            return "doubled_list"
    if completion.casefold() == "empty":
        return "empty"
    if "," in completion:
        return "list_like_miss"
    if completion and re.fullmatch(r"-?\d+", completion):
        return "numeric_other_miss"
    return "other_miss"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode_family: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    labelled: list[dict[str, Any]] = []
    for record in records:
        mode = str(record.get("mode") or "unknown")
        family = str(record.get("task_family") or record.get("category") or "unknown")
        label = classify_record(record)
        by_mode[mode][label] += 1
        by_mode_family[mode][family][label] += 1
        labelled.append(
            {
                "id": record.get("id", ""),
                "mode": mode,
                "task_family": family,
                "completion": record.get("completion", ""),
                "answer_aliases": record.get("answer_aliases", []),
                "tail_error_class": label,
            }
        )

    def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
        return {key: int(counter[key]) for key in sorted(counter)}

    return {
        "records": len(records),
        "by_mode": {
            mode: counter_to_dict(counter)
            for mode, counter in sorted(by_mode.items())
        },
        "by_mode_family": {
            mode: {
                family: counter_to_dict(counter)
                for family, counter in sorted(families.items())
            }
            for mode, families in sorted(by_mode_family.items())
        },
        "labelled_records": labelled,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize mixed-composition final-tail failure classes."
    )
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = summarize(load_jsonl(args.eval_jsonl))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for mode, counts in result["by_mode"].items():
        compact = " ".join(f"{key}={value}" for key, value in counts.items())
        print(f"{mode}: {compact}")


if __name__ == "__main__":
    main()
