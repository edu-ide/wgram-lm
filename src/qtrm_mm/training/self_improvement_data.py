from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.eval.memory_retrieval import (
    build_case_prompt,
    case_task_family,
    expected_unknown_case,
    evidence_records,
)
from qtrm_mm.training.memory_trace_data import canonical_case_answer


_UNKNOWN_RE = re.compile(r"\bunknown\b", re.IGNORECASE)


def _answer_text(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped if stripped.startswith("Answer:") or stripped.startswith("Action:") else f"Answer: {stripped}"


def _has_unknown_repetition(completion: str, *, min_count: int = 3) -> bool:
    return len(_UNKNOWN_RE.findall(completion or "")) >= min_count


def _case_by_id(cases: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case.get("id")): case for case in cases if case.get("id") is not None}


def _results_from_eval_record(case: dict[str, Any], record: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    records_by_source = {str(rec.get("source")): rec for rec in evidence_records(case, include_distractors=True)}
    sources = list(record.get("retrieved_sources") or [])
    roles = list(record.get("retrieved_roles") or [])
    rerank_scores = list(record.get("retrieved_rerank_scores") or [])
    retrieval_scores = list(record.get("retrieved_retrieval_scores") or [])
    rerank_backends = list(record.get("retrieved_rerank_backend") or [])
    results: list[tuple[float, dict[str, Any]]] = []

    for idx, source in enumerate(sources):
        base = dict(records_by_source.get(str(source), {"source": source, "text": ""}))
        if idx < len(roles):
            base["evidence_role"] = roles[idx]
            base["is_target"] = roles[idx] == "target"
        if idx < len(rerank_scores):
            base["rerank_score"] = rerank_scores[idx]
        if idx < len(retrieval_scores):
            base["retrieval_score"] = retrieval_scores[idx]
        if idx < len(rerank_backends):
            base["rerank_backend"] = rerank_backends[idx]
        score = base.get("rerank_score", base.get("retrieval_score", 0.0))
        results.append((float(score or 0.0), base))
    return results


def _failure_tags(
    *,
    case: dict[str, Any],
    record: dict[str, Any],
    missing_answer_policy: str,
) -> list[str]:
    tags: list[str] = []
    completion = str(record.get("completion", ""))
    if not bool(record.get("hit")):
        tags.append("wrong_answer")
    if expected_unknown_case(case):
        tags.append("abstention")
        if missing_answer_policy == "needs_search":
            tags.append("needs_search")
    if _has_unknown_repetition(completion):
        tags.append("unknown_repetition")
    return list(dict.fromkeys(tags))


def _chosen_answer(case: dict[str, Any], *, missing_answer_policy: str) -> tuple[str, str]:
    if expected_unknown_case(case) and missing_answer_policy == "needs_search":
        return "Action: NEEDS_SEARCH", "needs_search"
    return f"Answer: {canonical_case_answer(case)}", "answer"


def _agentic_needs_search_prompt(prompt: str) -> str:
    replacements = (
        (
            "If the evidence does not explicitly contain the requested answer, return UNKNOWN.",
            "If the evidence does not explicitly contain the requested answer, emit Action: NEEDS_SEARCH.",
        ),
        (
            "If the requested answer is not present in the evidence, answer UNKNOWN.",
            "If the requested answer is not present in the evidence, emit Action: NEEDS_SEARCH.",
        ),
        (
            "If the current requested answer is not present, answer UNKNOWN.",
            "If the current requested answer is not present, emit Action: NEEDS_SEARCH.",
        ),
        (
            "If the signed notice redacts the requested answer, answer UNKNOWN.",
            "If the signed notice redacts the requested answer, emit Action: NEEDS_SEARCH.",
        ),
        ("요청한 답이 증거에 없으면 UNKNOWN만 답하세요.", "요청한 답이 증거에 없으면 Action: NEEDS_SEARCH를 출력하세요."),
    )
    out = prompt
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def build_preference_rows(
    cases: Iterable[dict[str, Any]],
    eval_records: Iterable[dict[str, Any]],
    *,
    source_eval: str = "",
    training_scope: str = "analysis_only",
    include_hits_with_artifacts: bool = False,
    missing_answer_policy: str = "closed_evidence_unknown",
    max_evidence_chars: int = 2000,
) -> list[dict[str, Any]]:
    if missing_answer_policy not in {"closed_evidence_unknown", "needs_search"}:
        raise ValueError(f"unknown missing_answer_policy: {missing_answer_policy}")

    cases_by_id = _case_by_id(cases)
    rows: list[dict[str, Any]] = []
    for record in eval_records:
        case_id = str(record.get("id") or record.get("case_id") or "")
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        artifact = _has_unknown_repetition(str(record.get("completion", "")))
        if bool(record.get("hit")) and not (include_hits_with_artifacts and artifact):
            continue

        evidence_results = _results_from_eval_record(case, record)
        chosen, resolution_state = _chosen_answer(case, missing_answer_policy=missing_answer_policy)
        prompt = build_case_prompt(
            case,
            include_evidence=True,
            evidence_results=evidence_results,
            max_evidence_chars=max_evidence_chars,
        )
        if resolution_state == "needs_search":
            prompt = _agentic_needs_search_prompt(prompt)
        rows.append(
            {
                "type": "memory_preference",
                "case_id": case_id,
                "category": case.get("category", record.get("category", "uncategorized")),
                "task_family": record.get("task_family") or case_task_family(case),
                "source_eval": source_eval,
                "training_scope": training_scope,
                "missing_answer_policy": missing_answer_policy,
                "resolution_state": resolution_state,
                "failure_tags": _failure_tags(
                    case=case,
                    record=record,
                    missing_answer_policy=missing_answer_policy,
                ),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": _answer_text(str(record.get("completion", ""))),
                "mode": record.get("mode", ""),
                "retrieved_sources": list(record.get("retrieved_sources") or []),
                "retrieved_roles": list(record.get("retrieved_roles") or []),
                "retrieved_rerank_scores": list(record.get("retrieved_rerank_scores") or []),
                "retrieved_retrieval_scores": list(record.get("retrieved_retrieval_scores") or []),
            }
        )
    return rows


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "summary" not in obj:
                records.append(obj)
    return records


def write_preference_jsonl(
    cases: Iterable[dict[str, Any]],
    eval_records: Iterable[dict[str, Any]],
    out_path: str | Path,
    **kwargs: Any,
) -> int:
    rows = build_preference_rows(cases, eval_records, **kwargs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
