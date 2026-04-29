from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from qtrm_mm.eval.fact_verification import (
    action_for_verdict,
    build_fact_prompt,
    infer_fact_verdict,
    load_fact_cases,
    select_fact_evidence_results,
)


def _expected_answer(case: dict[str, Any], verdict: str, action: str) -> str:
    answer = str(case.get("expected_answer") or "").strip()
    if answer:
        return answer
    if action == "NEEDS_SEARCH":
        return "Insufficient or stale evidence; search is required before a final claim."
    if verdict == "CONFLICT":
        return "Available evidence conflicts; report the conflict instead of choosing a side blindly."
    return str(case.get("positive_conclusion") or case.get("claim") or "").strip()


def build_fact_trace_rows(
    cases: Iterable[dict[str, Any]],
    *,
    variants: Sequence[str] = ("target", "all", "lexical"),
    top_k: int = 5,
    max_evidence_chars: int = 4000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for variant in variants:
            evidence_results = select_fact_evidence_results(case, evidence_mode=variant, top_k=top_k)
            verdict = str(case.get("expected_verdict") or infer_fact_verdict(case, evidence_results)).strip().upper()
            action = str(case.get("expected_action") or action_for_verdict(verdict)).strip().upper()
            answer_text = _expected_answer(case, verdict, action)
            rows.append(
                {
                    "type": "fact_verification_trace",
                    "case_id": case.get("id"),
                    "category": case.get("category", "fact_verification"),
                    "variant": variant,
                    "expected_verdict": verdict,
                    "expected_action": action,
                    "prompt": build_fact_prompt(
                        case,
                        include_evidence=True,
                        evidence_results=evidence_results,
                        max_evidence_chars=max_evidence_chars,
                    ),
                    "answer": f"Verdict: {verdict}\nAction: {action}\nAnswer: {answer_text}",
                }
            )
    return rows


def write_fact_trace_jsonl(
    cases_path: str | Path,
    out_path: str | Path,
    *,
    variants: Sequence[str] = ("target", "all", "lexical"),
    top_k: int = 5,
    max_evidence_chars: int = 4000,
) -> int:
    rows = build_fact_trace_rows(
        load_fact_cases(cases_path),
        variants=variants,
        top_k=top_k,
        max_evidence_chars=max_evidence_chars,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
