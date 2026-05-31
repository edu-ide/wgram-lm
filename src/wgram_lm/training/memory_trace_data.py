from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from wgram_lm.eval.memory_retrieval import (
    build_case_prompt,
    case_task_family,
    expected_unknown_case,
    load_cases,
    select_evidence_results,
)


def canonical_case_answer(case: dict) -> str:
    if expected_unknown_case(case):
        return "UNKNOWN"
    aliases = case.get("answer_aliases") or []
    if not aliases:
        raise ValueError(f"{case.get('id', 'case')}: missing answer_aliases")
    return str(aliases[0])


def _variant_results(case: dict, variant: str, top_k: int):
    if variant == "target":
        return select_evidence_results(case, evidence_mode="target", top_k=top_k)
    if variant == "all":
        return select_evidence_results(case, evidence_mode="all", top_k=top_k)
    if variant == "lexical":
        return select_evidence_results(case, evidence_mode="lexical", top_k=top_k)
    raise ValueError(f"unknown memory trace variant: {variant}")


def build_memory_trace_rows(
    cases: Iterable[dict],
    *,
    variants: Sequence[str] = ("target", "all", "lexical"),
    top_k: int = 5,
    max_evidence_chars: int = 2000,
) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        answer = canonical_case_answer(case)
        for variant in variants:
            evidence_results = _variant_results(case, variant, top_k)
            rows.append(
                {
                    "type": "memory_trace",
                    "case_id": case.get("id"),
                    "category": case.get("category", "uncategorized"),
                    "task_family": case_task_family(case),
                    "expected_unknown": expected_unknown_case(case),
                    "variant": variant,
                    "prompt": build_case_prompt(
                        case,
                        include_evidence=True,
                        evidence_results=evidence_results,
                        max_evidence_chars=max_evidence_chars,
                    ),
                    "answer": f"Answer: {answer}",
                }
            )
    return rows


def write_memory_trace_jsonl(
    cases_path: str | Path,
    out_path: str | Path,
    *,
    variants: Sequence[str] = ("target", "all", "lexical"),
    top_k: int = 5,
    max_evidence_chars: int = 2000,
) -> int:
    rows = build_memory_trace_rows(
        load_cases(cases_path),
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
