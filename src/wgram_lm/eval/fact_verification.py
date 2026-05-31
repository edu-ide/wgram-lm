from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.memory_retrieval import lexical_retrieve_case, target_retrieval_stats

VALID_VERDICTS = (
    "SUPPORTED",
    "REFUTED",
    "NOT_ENOUGH_INFO",
    "CONFLICT",
    "STALE_OR_TIME_DEPENDENT",
)
VALID_ACTIONS = ("ANSWER", "NEEDS_SEARCH")

_CREDIBILITY_RANK = {
    "peer_reviewed": 100,
    "official": 95,
    "primary_source": 90,
    "signed": 85,
    "signed_notice": 85,
    "expert": 75,
    "secondary": 55,
    "news": 45,
    "user_note": 30,
    "generated_trace": 20,
    "anonymous": 10,
    "unknown": 0,
}


def _normalize_verdict(value: Any) -> str:
    verdict = str(value or "NOT_ENOUGH_INFO").strip().upper()
    return verdict if verdict in VALID_VERDICTS else "NOT_ENOUGH_INFO"


def _credibility_rank(rec: dict[str, Any]) -> int:
    tier = str(rec.get("credibility_tier") or rec.get("source_type") or "unknown").strip().casefold()
    return _CREDIBILITY_RANK.get(tier, 0)


def _parse_date(value: Any) -> date:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return date.min


def fact_evidence_records(case: dict[str, Any], *, include_distractors: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, key in (("target", "evidence"), ("distractor", "distractors")):
        if role == "distractor" and not include_distractors:
            continue
        for idx, rec in enumerate(case.get(key) or []):
            enriched = dict(rec)
            enriched.setdefault("source", f"{case.get('id', 'case')}:{key}:{idx}")
            enriched.setdefault("chunk_id", idx)
            enriched.setdefault("source_type", "unknown")
            enriched.setdefault("credibility_tier", "unknown")
            enriched["case_id"] = case.get("id")
            enriched["evidence_role"] = role
            enriched["is_target"] = role == "target"
            enriched["verdict"] = _normalize_verdict(enriched.get("verdict"))
            records.append(enriched)
    return records


def select_fact_evidence_results(
    case: dict[str, Any],
    *,
    evidence_mode: str = "target",
    top_k: int = 5,
) -> list[tuple[float, dict[str, Any]]]:
    if evidence_mode == "target":
        return [(1.0, rec) for rec in fact_evidence_records(case, include_distractors=False)[:top_k]]
    if evidence_mode == "all":
        return [(1.0, rec) for rec in fact_evidence_records(case, include_distractors=True)[:top_k]]
    if evidence_mode == "lexical":
        query_case = {
            **case,
            "question": f"{case.get('question', '')} {case.get('claim', '')}",
            "evidence": fact_evidence_records(case, include_distractors=False),
            "distractors": [
                rec for rec in fact_evidence_records(case, include_distractors=True) if not rec.get("is_target")
            ],
        }
        return lexical_retrieve_case(query_case, top_k=top_k, include_distractors=True)
    if evidence_mode == "none":
        return []
    raise ValueError(f"unknown evidence_mode: {evidence_mode}")


def _format_fact_evidence(results: Iterable[tuple[float, dict[str, Any]]], *, max_chars: int) -> str:
    lines = ["Fact verification evidence"]
    for score, rec in results:
        header = (
            f"SOURCE={rec.get('source', '?')} CHUNK={rec.get('chunk_id', '?')} "
            f"SCORE={float(score):.4f} ROLE={rec.get('evidence_role', 'unknown')} "
            f"DATE={rec.get('published_at', '?')} TYPE={rec.get('source_type', '?')} "
            f"CREDIBILITY={rec.get('credibility_tier', '?')} VERDICT={_normalize_verdict(rec.get('verdict'))}"
        )
        text = str(rec.get("text", "")).replace("\n", " ").strip()
        block = f"{header}\n{text}"
        current = "\n".join(lines)
        tentative = f"{current}\n{block}"
        if len(tentative) <= max_chars:
            lines.append(block)
            continue
        remaining = max_chars - len(current) - len(header) - 2
        if remaining > 0:
            lines.append(f"{header}\n{text[:remaining]}")
        break
    return "\n".join(lines)[:max_chars]


def build_fact_prompt(
    case: dict[str, Any],
    *,
    include_evidence: bool = True,
    evidence_results: Iterable[tuple[float, dict[str, Any]]] | None = None,
    max_evidence_chars: int = 4000,
) -> str:
    claim = str(case.get("claim", "")).strip()
    question = str(case.get("question", "")).strip()
    instruction = str(case.get("instruction", "")).strip()
    task = (
        "You are a fact-verification and evidence-quality reasoner.\n"
        "Allowed verdicts: SUPPORTED, REFUTED, NOT_ENOUGH_INFO, CONFLICT, STALE_OR_TIME_DEPENDENT.\n"
        "Allowed actions: ANSWER, NEEDS_SEARCH.\n"
        "Use source credibility, publication time, directness, and contradictions before answering.\n"
        "Return this shape:\n"
        "Verdict: <allowed verdict>\n"
        "Action: <ANSWER or NEEDS_SEARCH>\n"
        "Answer: <short grounded conclusion>\n"
    )
    if instruction:
        task += f"Instruction: {instruction}\n"
    if claim:
        task += f"Claim: {claim}\n"
    if question:
        task += f"Question: {question}\n"
    if not include_evidence:
        return task.strip()

    results = list(evidence_results) if evidence_results is not None else select_fact_evidence_results(
        case,
        evidence_mode="target",
        top_k=20,
    )
    evidence = _format_fact_evidence(results, max_chars=max_evidence_chars)
    return f"{evidence}\n\n{task}".strip()


def _verdict_from_records(records: list[dict[str, Any]]) -> str:
    verdicts = {_normalize_verdict(rec.get("verdict")) for rec in records}
    if not verdicts:
        return "NOT_ENOUGH_INFO"
    if "CONFLICT" in verdicts:
        return "CONFLICT"
    if {"SUPPORTED", "REFUTED"}.issubset(verdicts):
        return "CONFLICT"
    if "SUPPORTED" in verdicts:
        return "SUPPORTED"
    if "REFUTED" in verdicts:
        return "REFUTED"
    if "STALE_OR_TIME_DEPENDENT" in verdicts:
        return "STALE_OR_TIME_DEPENDENT"
    return "NOT_ENOUGH_INFO"


def infer_fact_verdict(
    case: dict[str, Any],
    evidence_results: Iterable[tuple[float, dict[str, Any]]] | None = None,
) -> str:
    records = [rec for _, rec in evidence_results] if evidence_results is not None else fact_evidence_records(
        case,
        include_distractors=True,
    )
    if not records:
        return "NOT_ENOUGH_INFO"

    strategy = str(case.get("verification_strategy", "")).casefold()
    if strategy == "temporal":
        newest_date = max(_parse_date(rec.get("published_at")) for rec in records)
        newest = [rec for rec in records if _parse_date(rec.get("published_at")) == newest_date]
        return _verdict_from_records(newest)
    if strategy == "authority":
        best_rank = max(_credibility_rank(rec) for rec in records)
        best = [rec for rec in records if _credibility_rank(rec) == best_rank]
        return _verdict_from_records(best)
    return _verdict_from_records(records)


def action_for_verdict(verdict: str) -> str:
    verdict = _normalize_verdict(verdict)
    if verdict in {"NOT_ENOUGH_INFO", "STALE_OR_TIME_DEPENDENT"}:
        return "NEEDS_SEARCH"
    return "ANSWER"


def evaluate_fact_case(
    case: dict[str, Any],
    *,
    evidence_mode: str = "target",
    retrieval_top_k: int = 5,
    max_evidence_chars: int = 4000,
) -> dict[str, Any]:
    evidence_results = select_fact_evidence_results(case, evidence_mode=evidence_mode, top_k=retrieval_top_k)
    predicted_verdict = infer_fact_verdict(case, evidence_results)
    predicted_action = action_for_verdict(predicted_verdict)
    expected_verdict = _normalize_verdict(case.get("expected_verdict"))
    expected_action = str(case.get("expected_action") or action_for_verdict(expected_verdict)).strip().upper()
    if expected_action not in VALID_ACTIONS:
        expected_action = action_for_verdict(expected_verdict)
    retrieval_stats = target_retrieval_stats(case, evidence_results)
    prompt = build_fact_prompt(
        case,
        include_evidence=evidence_mode != "none",
        evidence_results=evidence_results,
        max_evidence_chars=max_evidence_chars,
    )
    return {
        "id": case.get("id"),
        "category": case.get("category", "fact_verification"),
        "verification_strategy": case.get("verification_strategy", ""),
        "expected_verdict": expected_verdict,
        "predicted_verdict": predicted_verdict,
        "verdict_hit": predicted_verdict == expected_verdict,
        "expected_action": expected_action,
        "predicted_action": predicted_action,
        "action_hit": predicted_action == expected_action,
        "evidence_mode": evidence_mode,
        "retrieved_target": retrieval_stats["retrieved_target"],
        "target_count": retrieval_stats["target_count"],
        "retrieved_target_count": retrieval_stats["retrieved_target_count"],
        "all_targets_retrieved": retrieval_stats["all_targets_retrieved"],
        "target_recall": retrieval_stats["target_recall"],
        "retrieved_roles": [rec.get("evidence_role", "unknown") for _, rec in evidence_results],
        "retrieved_sources": [rec.get("source", "?") for _, rec in evidence_results],
        "prompt": prompt,
    }


def _bucket_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(records)
    verdict_hits = sum(1 for rec in records if bool(rec.get("verdict_hit")))
    action_hits = sum(1 for rec in records if bool(rec.get("action_hit")))
    retrieved = sum(1 for rec in records if bool(rec.get("retrieved_target")))
    all_targets = sum(1 for rec in records if bool(rec.get("all_targets_retrieved")))
    target_recall = sum(float(rec.get("target_recall", 0.0)) for rec in records)
    return {
        "count": count,
        "verdict_hits": verdict_hits,
        "verdict_accuracy": verdict_hits / count if count else 0.0,
        "action_hits": action_hits,
        "action_accuracy": action_hits / count if count else 0.0,
        "retrieved_target_count": retrieved,
        "retrieved_target_rate": retrieved / count if count else 0.0,
        "all_targets_retrieved_count": all_targets,
        "all_targets_retrieved_rate": all_targets / count if count else 0.0,
        "target_recall_mean": target_recall / count if count else 0.0,
    }


def summarize_fact_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_expected_verdict: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_expected_verdict[str(row.get("expected_verdict", "UNKNOWN"))].append(row)
        by_category[str(row.get("category", "uncategorized"))].append(row)
    return {
        "overall": _bucket_summary(rows),
        "by_expected_verdict": {key: _bucket_summary(value) for key, value in by_expected_verdict.items()},
        "by_category": {key: _bucket_summary(value) for key, value in by_category.items()},
    }


def load_fact_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not case.get("id"):
                case["id"] = f"fact-case-{line_no}"
            if not case.get("claim"):
                raise ValueError(f"{path}:{line_no}: missing claim")
            if not case.get("question"):
                raise ValueError(f"{path}:{line_no}: missing question")
            if _normalize_verdict(case.get("expected_verdict")) not in VALID_VERDICTS:
                raise ValueError(f"{path}:{line_no}: invalid expected_verdict")
            case["expected_verdict"] = _normalize_verdict(case.get("expected_verdict"))
            case["expected_action"] = str(
                case.get("expected_action") or action_for_verdict(case["expected_verdict"])
            ).strip().upper()
            if case["expected_action"] not in VALID_ACTIONS:
                raise ValueError(f"{path}:{line_no}: invalid expected_action")
            cases.append(case)
    return cases
