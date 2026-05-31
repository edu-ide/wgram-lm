#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wgram_lm.eval.memory_retrieval import (
    build_case_prompt_and_workspace_memory,
    case_task_family,
    expected_unknown_case,
    load_cases,
    select_evidence_results,
)


def _clean_answer(text: str) -> str:
    answer = str(text or "").strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer or "UNKNOWN"


def _case_answer(case: dict[str, Any]) -> str:
    if expected_unknown_case(case):
        return "UNKNOWN"
    aliases = case.get("answer_aliases") or []
    if aliases:
        return _clean_answer(str(aliases[0]))
    return _clean_answer(str(case.get("answer") or "UNKNOWN"))


def build_canonical_plain_answer_row(
    case: dict[str, Any],
    *,
    evidence_mode: str = "target",
    top_k: int = 3,
    max_evidence_chars: int = 4000,
) -> dict[str, Any]:
    results = select_evidence_results(case, evidence_mode=evidence_mode, top_k=top_k)
    prompt, workspace_memory = build_case_prompt_and_workspace_memory(
        case,
        include_evidence=True,
        evidence_results=results,
        max_evidence_chars=max_evidence_chars,
        evidence_injection="ssot",
    )
    if workspace_memory is not None:
        raise RuntimeError("canonical SSOT plain-answer rows must not create workspace memory")
    answer = _case_answer(case)
    return {
        "type": "canonical_plain_answer",
        "case_id": str(case.get("id", "")),
        "prompt": prompt,
        "answer": f"Answer: {answer}",
        "ssot_contract": "single_visible_prompt_stream",
        "metadata": {
            "answer_contract": "plain_short_answer",
            "answer_policy": "greedy_autoregressive",
            "expected_unknown": expected_unknown_case(case),
            "task_family": case_task_family(case),
            "evidence_mode": evidence_mode,
            "top_k": int(top_k),
        },
    }


def write_canonical_plain_answer_data(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    evidence_mode: str = "target",
    top_k: int = 3,
    max_source_rows: int = 0,
    max_evidence_chars: int = 4000,
) -> int:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(load_cases(input_jsonl)):
        if max_source_rows > 0 and index >= max_source_rows:
            break
        rows.append(
            build_canonical_plain_answer_row(
                case,
                evidence_mode=evidence_mode,
                top_k=top_k,
                max_evidence_chars=max_evidence_chars,
            )
        )

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SSOT canonical plain-answer SFT rows from MemoryOS QA cases."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--evidence-mode",
        choices=["target", "all", "lexical", "none"],
        default="target",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-evidence-chars", type=int, default=4000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    count = write_canonical_plain_answer_data(
        args.input_jsonl,
        args.output_jsonl,
        evidence_mode=args.evidence_mode,
        top_k=args.top_k,
        max_source_rows=args.max_source_rows,
        max_evidence_chars=args.max_evidence_chars,
    )
    print(f"wrote {count} canonical plain-answer rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
