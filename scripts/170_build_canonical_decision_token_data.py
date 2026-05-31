#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.memory_retrieval import (
    case_task_family,
    expected_unknown_case,
    load_cases,
    select_evidence_results,
)
from wgram_lm.infer import build_prompt_with_memory, format_memory_context


def _clean_answer(text: str) -> str:
    answer = str(text or "").strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer or "UNKNOWN"


def _case_answer(case: dict[str, Any]) -> str:
    aliases = case.get("answer_aliases") or []
    if aliases:
        return _clean_answer(str(aliases[0]))
    return _clean_answer(str(case.get("answer") or "UNKNOWN"))


def verification_label_for_case(case: dict[str, Any]) -> str:
    """Return the decision-token verifier label without hidden runtime state."""
    if expected_unknown_case(case):
        return "missing"
    family = case_task_family(case)
    if family == "conflict":
        return "supported_after_source_check"
    if family == "multi_hop":
        return "supported_by_multi_hop"
    return "supported"


def decision_target_for_verification(verification_label: str) -> str:
    if verification_label == "missing":
        return "ABSTAIN"
    return "ANSWER"


def build_decision_token_prompt(
    case: dict[str, Any],
    evidence_results: Iterable[tuple[float, dict[str, Any]]],
    *,
    max_evidence_chars: int = 4000,
) -> str:
    question = str(case.get("question", "")).strip()
    instruction = str(case.get("instruction") or "").strip()
    task_lines = [
        "Use one visible evidence stream. Do not use hidden evidence.",
        "Output exactly three lines: Verify, Decision, Answer.",
        "Verify must be one of: supported, supported_after_source_check, supported_by_multi_hop, missing.",
        "Decision must be ANSWER when evidence supports the answer, otherwise ABSTAIN.",
        "If the evidence does not explicitly contain the requested answer, Answer must be UNKNOWN.",
    ]
    if instruction:
        task_lines.append(instruction)
    task_lines.append(f"Question: {question}")
    memory_context = format_memory_context(
        evidence_results,
        max_chars=max_evidence_chars,
    )
    return build_prompt_with_memory("\n".join(task_lines), memory_context)


def build_canonical_decision_token_row(
    case: dict[str, Any],
    *,
    evidence_mode: str = "target",
    top_k: int = 3,
    max_evidence_chars: int = 4000,
) -> dict[str, Any]:
    results = select_evidence_results(case, evidence_mode=evidence_mode, top_k=top_k)
    verification_label = verification_label_for_case(case)
    decision_target = decision_target_for_verification(verification_label)
    answer = "UNKNOWN" if decision_target == "ABSTAIN" else _case_answer(case)
    prompt = build_decision_token_prompt(
        case,
        results,
        max_evidence_chars=max_evidence_chars,
    )
    return {
        "type": "canonical_decision_tokens",
        "case_id": str(case.get("id", "")),
        "prompt": prompt,
        "answer": (
            f"Verify: {verification_label}\n"
            f"Decision: {decision_target}\n"
            f"Answer: {answer}"
        ),
        "ssot_contract": "single_visible_prompt_stream",
        "metadata": {
            "answer_policy": "greedy_autoregressive",
            "decision_target": decision_target,
            "verification_label": verification_label,
            "expected_unknown": expected_unknown_case(case),
            "task_family": case_task_family(case),
            "evidence_mode": evidence_mode,
            "top_k": int(top_k),
        },
    }


def write_canonical_decision_token_data(
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
            build_canonical_decision_token_row(
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
        description=(
            "Build SSOT canonical decision-token SFT rows from MemoryOS QA cases."
        )
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
    count = write_canonical_decision_token_data(
        args.input_jsonl,
        args.output_jsonl,
        evidence_mode=args.evidence_mode,
        top_k=args.top_k,
        max_source_rows=args.max_source_rows,
        max_evidence_chars=args.max_evidence_chars,
    )
    print(f"wrote {count} canonical decision-token rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
