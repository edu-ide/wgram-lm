#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.memory_retrieval import (
    build_case_prompt_and_workspace_memory,
    load_cases,
    select_evidence_results,
)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_hard_negative_rows(
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = 3,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        evidence_results = select_evidence_results(
            case,
            evidence_mode="all",
            top_k=max(1, int(top_k)),
        )
        if not evidence_results:
            continue
        visible_prompt, workspace_text = build_case_prompt_and_workspace_memory(
            case,
            include_evidence=True,
            evidence_results=evidence_results,
            evidence_injection="workspace",
        )
        if not workspace_text:
            continue
        rows.append(
            {
                "type": "evidence_span_reader",
                "case_id": case.get("id"),
                "category": case.get("category"),
                "task_family": case.get("task_family"),
                "prompt": visible_prompt,
                "visible_prompt": visible_prompt,
                "workspace_text": workspace_text,
                "workspace_evidence": workspace_text,
                "answer": "Answer: UNKNOWN",
                "answer_text": "UNKNOWN",
                "no_answer": True,
                "answer_span": None,
                "span_status": "hard_no_answer",
                "source_training_scope": "counterfactual_hard_negative",
                "workspace_counterfactual_source_case_id": case.get(
                    "workspace_counterfactual_source_case_id"
                ),
            }
        )
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def write_training_mix(
    base_span_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    hard_negative_cases: str | Path | None = None,
    hard_negative_top_k: int = 3,
    max_hard_negatives: int = 0,
    hard_negative_repeat: int = 1,
) -> int:
    rows = list(iter_jsonl(base_span_jsonl))
    if hard_negative_cases is not None:
        hard_negative_rows = build_hard_negative_rows(
            load_cases(hard_negative_cases),
            top_k=hard_negative_top_k,
            max_rows=max_hard_negatives,
        )
        for _ in range(max(1, int(hard_negative_repeat))):
            rows.extend(dict(row) for row in hard_negative_rows)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an evidence span-reader training mix with optional "
            "counterfactual hard no-answer rows."
        )
    )
    parser.add_argument("--base-span-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--hard-negative-cases", default=None)
    parser.add_argument("--hard-negative-top-k", type=int, default=3)
    parser.add_argument("--max-hard-negatives", type=int, default=0)
    parser.add_argument("--hard-negative-repeat", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_training_mix(
        args.base_span_jsonl,
        args.output_jsonl,
        hard_negative_cases=args.hard_negative_cases,
        hard_negative_top_k=args.hard_negative_top_k,
        max_hard_negatives=args.max_hard_negatives,
        hard_negative_repeat=args.hard_negative_repeat,
    )
    print(f"wrote {count} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
