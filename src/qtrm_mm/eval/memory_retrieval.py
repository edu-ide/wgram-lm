from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.infer import build_prompt_with_memory, format_memory_context

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "is",
    "of",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def normalize_answer(text: str) -> str:
    """Normalize short factual answers for exact-ish hit checks."""
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def answer_hit(text: str, aliases: Iterable[str]) -> bool:
    normalized_text = normalize_answer(text)
    return any(
        normalized_alias and normalized_alias in normalized_text
        for normalized_alias in (normalize_answer(alias) for alias in aliases)
    )


def expected_unknown_case(case_or_record: dict[str, Any]) -> bool:
    if bool(case_or_record.get("expected_unknown")):
        return True
    aliases = case_or_record.get("answer_aliases") or []
    if any(normalize_answer(str(alias)) == "unknown" for alias in aliases):
        return True
    category = str(case_or_record.get("category", "")).casefold()
    return category.startswith("negative") or "missing" in category


def case_task_family(case_or_record: dict[str, Any]) -> str:
    if case_or_record.get("task_family"):
        return str(case_or_record["task_family"])
    category = str(case_or_record.get("category", "")).strip() or "uncategorized"
    category_key = category.casefold()
    if expected_unknown_case(case_or_record):
        return "abstention"
    if "conflict" in category_key or category_key.startswith(("temporal", "authority")):
        return "conflict"
    if "multi" in category_key or "multihop" in category_key:
        return "multi_hop"
    return category


def build_case_prompt(
    case: dict[str, Any],
    *,
    include_evidence: bool = True,
    evidence_results: Iterable[tuple[float, dict[str, Any]]] | None = None,
    max_evidence_chars: int = 4000,
) -> str:
    question = str(case.get("question", "")).strip()
    instruction = str(case.get("instruction", "")).strip()
    instruction_line = f"{instruction}\n" if instruction else ""
    task = (
        "Answer using only the evidence. Return only the short answer.\n"
        "If the evidence does not explicitly contain the requested answer, return UNKNOWN. "
        "Do not answer with related but different entities.\n"
        f"{instruction_line}"
        f"Question: {question}"
    )
    if not include_evidence:
        return task

    results = list(evidence_results) if evidence_results is not None else [
        (1.0, rec) for rec in evidence_records(case, include_distractors=False)
    ]
    memory_context = format_memory_context(results, max_chars=max_evidence_chars)
    return build_prompt_with_memory(task, memory_context)


def evidence_records(case: dict[str, Any], *, include_distractors: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, key in (("target", "evidence"), ("distractor", "distractors")):
        if role == "distractor" and not include_distractors:
            continue
        for idx, rec in enumerate(case.get(key) or []):
            enriched = dict(rec)
            enriched.setdefault("source", f"{case.get('id', 'case')}:{key}")
            enriched.setdefault("chunk_id", idx)
            enriched["case_id"] = case.get("id")
            enriched["evidence_role"] = role
            enriched["is_target"] = role == "target"
            records.append(enriched)
    return records


def case_index_records(
    cases: Iterable[dict[str, Any]],
    *,
    include_distractors: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        for rec in evidence_records(case, include_distractors=include_distractors):
            records.append(rec)
    return records


def _lex_terms(text: str) -> set[str]:
    normalized = []
    for ch in text.casefold():
        normalized.append(ch if ch.isalnum() else " ")
    return {
        token
        for token in "".join(normalized).split()
        if len(token) > 1 and token not in _STOPWORDS
    }


def lexical_retrieve_case(
    case: dict[str, Any],
    *,
    top_k: int = 3,
    include_distractors: bool = True,
) -> list[tuple[float, dict[str, Any]]]:
    query_terms = _lex_terms(str(case.get("question", "")))
    scored = []
    for index, rec in enumerate(evidence_records(case, include_distractors=include_distractors)):
        haystack = f"{rec.get('source', '')} {rec.get('text', '')}"
        rec_terms = _lex_terms(haystack)
        overlap = query_terms & rec_terms
        phrase_bonus = sum(1 for term in query_terms if term in haystack.casefold())
        role_tiebreak = 1e-4 if rec.get("is_target") else 0.0
        order_tiebreak = -index * 1e-6
        score = float(len(overlap) + 0.1 * phrase_bonus + role_tiebreak + order_tiebreak)
        scored.append((score, rec))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def target_retrieved(results: Iterable[tuple[float, dict[str, Any]]]) -> bool:
    return any(bool(rec.get("is_target")) for _, rec in results)


def target_retrieval_stats(
    case: dict[str, Any],
    results: Iterable[tuple[float, dict[str, Any]]],
) -> dict[str, Any]:
    target_keys = {
        (rec.get("source"), rec.get("chunk_id"))
        for rec in evidence_records(case, include_distractors=False)
    }
    retrieved_target_keys = {
        (rec.get("source"), rec.get("chunk_id"))
        for _, rec in results
        if rec.get("is_target")
    }
    target_count = len(target_keys)
    retrieved_target_count = len(target_keys & retrieved_target_keys)
    return {
        "target_count": target_count,
        "retrieved_target_count": retrieved_target_count,
        "retrieved_target": retrieved_target_count > 0,
        "all_targets_retrieved": target_count > 0 and retrieved_target_count == target_count,
        "target_recall": retrieved_target_count / target_count if target_count else 0.0,
    }


def filter_results_for_case(
    results: Iterable[tuple[float, dict[str, Any]]],
    *,
    case_id: str,
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    filtered = [(score, rec) for score, rec in results if rec.get("case_id") == case_id]
    return filtered[:top_k]


def select_evidence_results(
    case: dict[str, Any],
    *,
    evidence_mode: str = "target",
    top_k: int = 3,
) -> list[tuple[float, dict[str, Any]]]:
    if evidence_mode == "target":
        return [(1.0, rec) for rec in evidence_records(case, include_distractors=False)[:top_k]]
    if evidence_mode == "all":
        return [(1.0, rec) for rec in evidence_records(case, include_distractors=True)[:top_k]]
    if evidence_mode == "lexical":
        return lexical_retrieve_case(case, top_k=top_k, include_distractors=True)
    if evidence_mode == "none":
        return []
    raise ValueError(f"unknown evidence_mode: {evidence_mode}")


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not case.get("id"):
                case["id"] = f"case-{line_no}"
            if not case.get("question"):
                raise ValueError(f"{path}:{line_no}: missing question")
            if not case.get("answer_aliases"):
                raise ValueError(f"{path}:{line_no}: missing answer_aliases")
            cases.append(case)
    return cases


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total_count = 0
    total_hits = 0
    total_retrieved_targets = 0
    total_all_targets = 0
    total_target_recall = 0.0
    bucket_factory = lambda: {  # noqa: E731
        "count": 0,
        "hits": 0,
        "retrieved_target_count": 0,
        "all_targets_retrieved_count": 0,
    }
    by_mode: dict[str, dict[str, int]] = defaultdict(bucket_factory)
    by_category: dict[str, dict[str, int]] = defaultdict(bucket_factory)
    by_task_family: dict[str, dict[str, int]] = defaultdict(bucket_factory)
    recall_by_mode: dict[str, float] = defaultdict(float)
    recall_by_category: dict[str, float] = defaultdict(float)
    recall_by_task_family: dict[str, float] = defaultdict(float)

    for record in records:
        hit = bool(record.get("hit"))
        retrieved_target = bool(record.get("retrieved_target"))
        all_targets_retrieved = bool(record.get("all_targets_retrieved"))
        target_recall = float(record.get("target_recall", 1.0 if retrieved_target else 0.0))
        mode = str(record.get("mode", "unknown"))
        category = str(record.get("category", "")).strip() or "uncategorized"
        task_family = case_task_family(record)
        total_count += 1
        total_hits += int(hit)
        total_retrieved_targets += int(retrieved_target)
        total_all_targets += int(all_targets_retrieved)
        total_target_recall += target_recall
        for key, buckets, recalls in (
            (mode, by_mode, recall_by_mode),
            (category, by_category, recall_by_category),
            (task_family, by_task_family, recall_by_task_family),
        ):
            buckets[key]["count"] += 1
            buckets[key]["hits"] += int(hit)
            buckets[key]["retrieved_target_count"] += int(retrieved_target)
            buckets[key]["all_targets_retrieved_count"] += int(all_targets_retrieved)
            recalls[key] += target_recall

    def with_accuracy(bucket: dict[str, int], *, target_recall_sum: float = 0.0) -> dict[str, float | int]:
        count = bucket["count"]
        hits = bucket["hits"]
        retrieved_targets = bucket.get("retrieved_target_count", 0)
        all_targets = bucket.get("all_targets_retrieved_count", 0)
        return {
            "count": count,
            "hits": hits,
            "accuracy": hits / count if count else 0.0,
            "retrieved_target_count": retrieved_targets,
            "retrieved_target_rate": retrieved_targets / count if count else 0.0,
            "all_targets_retrieved_count": all_targets,
            "all_targets_retrieved_rate": all_targets / count if count else 0.0,
            "target_recall_mean": target_recall_sum / count if count else 0.0,
        }

    return {
        "overall": with_accuracy(
            {
                "count": total_count,
                "hits": total_hits,
                "retrieved_target_count": total_retrieved_targets,
                "all_targets_retrieved_count": total_all_targets,
            },
            target_recall_sum=total_target_recall,
        ),
        "by_mode": {
            mode: with_accuracy(bucket, target_recall_sum=recall_by_mode[mode])
            for mode, bucket in by_mode.items()
        },
        "by_category": {
            category: with_accuracy(bucket, target_recall_sum=recall_by_category[category])
            for category, bucket in by_category.items()
        },
        "by_task_family": {
            family: with_accuracy(bucket, target_recall_sum=recall_by_task_family[family])
            for family, bucket in by_task_family.items()
        },
    }
