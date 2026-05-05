#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence


VISIBLE_REASONING_PATTERNS = (
    r"<\s*/?\s*think\s*>",
    r"\blet me\b",
    r"\bi need to\b",
    r"\bthe user (is|asked|wants|needs)\b",
    r"\bhidden reasoning\b",
)

ANSWER_DRIFT_PATTERNS = (
    r"(^|\n)\s*a\.\s+.+(\n|\s+)b\.\s+",
    r"\n\s*(what|why|how|when|where|who|which)\b.+\?",
    r"\bpls answer\b",
    r"\bplease answer\b",
    r"\bquestion\s*\d*\s*:",
    r"\banswer in a minimum\b",
    r"\bdo not (mention|reveal|add)\b",
    r"\bif the user'?s question\b",
    r"\byou should reply\b",
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Summarize visible reasoning and repetition failures in QTRM generation JSONL."
    )
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat-threshold", type=float, default=0.15)
    return ap


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_visible_reasoning(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered) for pattern in VISIBLE_REASONING_PATTERNS)


def has_answer_drift(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in ANSWER_DRIFT_PATTERNS)


def completion_text(record: dict) -> str:
    generated = str(record.get("greedy_text") or record.get("text") or "")
    prompt = str(record.get("text") or record.get("prompt") or "")
    if prompt and generated.startswith(prompt):
        return generated[len(prompt) :]
    return generated


def source_key(record: dict) -> str:
    sample = record.get("sample", record.get("source_sample", ""))
    return str(sample)


def candidate_id(record: dict) -> int:
    value = record.get("candidate_id")
    return int(value) if value is not None else 0


def summarize_records(records: Sequence[dict], *, repeat_threshold: float = 0.15) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    visible_count = repeat_count = drift_count = clean_count = 0
    per_record = []
    for record in records:
        text = completion_text(record)
        repetition = record.get("greedy_repetition") or {}
        rep2 = float(repetition.get("repeated_2gram_rate") or 0.0)
        visible = has_visible_reasoning(text)
        repeated = rep2 >= float(repeat_threshold)
        drift = has_answer_drift(text)
        clean = not visible and not repeated and not drift
        visible_count += int(visible)
        repeat_count += int(repeated)
        drift_count += int(drift)
        clean_count += int(clean)
        groups[source_key(record)].append(record)
        per_record.append(
            {
                "sample": source_key(record),
                "candidate_id": candidate_id(record),
                "visible_reasoning": visible,
                "repeat_failure": repeated,
                "answer_drift": drift,
                "clean": clean,
                "repeated_2gram_rate": rep2,
            }
        )

    group_visible = 0
    group_drift = 0
    group_has_clean = 0
    for rows in groups.values():
        row_texts = [completion_text(row) for row in rows]
        row_repeats = [
            float((row.get("greedy_repetition") or {}).get("repeated_2gram_rate") or 0.0)
            >= float(repeat_threshold)
            for row in rows
        ]
        row_visible = [has_visible_reasoning(text) for text in row_texts]
        row_drift = [has_answer_drift(text) for text in row_texts]
        group_visible += int(any(row_visible))
        group_drift += int(any(row_drift))
        group_has_clean += int(
            any(not v and not r and not d for v, r, d in zip(row_visible, row_repeats, row_drift))
        )

    n = len(records)
    g = len(groups)
    return {
        "records": n,
        "groups": g,
        "repeat_threshold": float(repeat_threshold),
        "visible_reasoning_count": visible_count,
        "visible_reasoning_rate": visible_count / max(1, n),
        "repeat_failure_count": repeat_count,
        "repeat_failure_rate": repeat_count / max(1, n),
        "answer_drift_count": drift_count,
        "answer_drift_rate": drift_count / max(1, n),
        "clean_count": clean_count,
        "clean_rate": clean_count / max(1, n),
        "group_visible_reasoning_rate": group_visible / max(1, g),
        "group_answer_drift_rate": group_drift / max(1, g),
        "group_has_clean_candidate_rate": group_has_clean / max(1, g),
        "records_detail": per_record,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    records = load_jsonl(args.eval_jsonl)
    summary = summarize_records(records, repeat_threshold=args.repeat_threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
