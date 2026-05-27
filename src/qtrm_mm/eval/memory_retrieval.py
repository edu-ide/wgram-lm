from __future__ import annotations

import json
import re
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


def canonical_answer_text(text: str) -> str:
    """Strip common answer wrappers while preserving the model's short answer text."""
    answer = str(text).strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().strip(" \t\r\n.。．:：;；")


def score_answer(
    text: str,
    aliases: Iterable[str],
    *,
    expected_unknown: bool = False,
    strict_exact: bool = False,
) -> dict[str, Any]:
    """Return layered deterministic answer scores and audit flags.

    `hit` preserves the old permissive answer check so historical runs remain
    comparable. The stricter fields make it clear whether a hit was exact,
    normalized exact, or only a loose substring match that should be audited.
    When strict_exact=True, only exact or normalized_exact (or unknown_exact)
    count as hit; loose contains is rejected (used by pure recursive RI gates).
    """
    alias_list = [str(alias) for alias in aliases]
    canonical = canonical_answer_text(text)
    compact = _strip_terminal_punctuation(canonical)
    normalized_text = normalize_answer(canonical)
    normalized_compact = normalize_answer(compact)
    normalized_aliases = [
        (alias, normalize_answer(alias), normalize_answer(_strip_terminal_punctuation(alias)))
        for alias in alias_list
    ]
    matched_aliases = [
        alias
        for alias, normalized_alias, _ in normalized_aliases
        if normalized_alias and normalized_alias in normalized_text
    ]

    exact_match = any(compact == _strip_terminal_punctuation(alias) for alias in alias_list)
    normalized_exact = any(
        normalized_compact and normalized_compact == normalized_alias_compact
        for _, _, normalized_alias_compact in normalized_aliases
    )
    normalized_contains = bool(matched_aliases)

    unknown_contains = "unknown" in normalized_text
    unknown_exact = normalized_compact == "unknown"
    unknown_correct = bool(expected_unknown and unknown_contains)

    if bool(strict_exact):
        hit = unknown_exact if expected_unknown else bool(exact_match or normalized_exact)
    else:
        hit = unknown_correct if expected_unknown else normalized_contains

    if expected_unknown and unknown_exact:
        match_type = "unknown_exact"
    elif exact_match:
        match_type = "exact"
    elif normalized_exact:
        match_type = "normalized_exact"
    elif unknown_correct:
        match_type = "unknown_contains"
    elif normalized_contains:
        match_type = "normalized_contains"
    else:
        match_type = "none"

    audit_reasons: list[str] = []
    if hit and normalized_contains and not (exact_match or normalized_exact):
        audit_reasons.append("loose_contains_match")
    if expected_unknown and unknown_correct and not unknown_exact:
        audit_reasons.append("unknown_with_extra_text")
    if bool(strict_exact) and normalized_contains and not (exact_match or normalized_exact):
        audit_reasons.append("strict_exact_miss")
    if not hit:
        audit_reasons.append("answer_miss")

    return {
        "hit": hit,
        "exact_match": exact_match,
        "normalized_exact": normalized_exact,
        "normalized_contains": normalized_contains,
        "unknown_correct": unknown_correct,
        "match_type": match_type,
        "matched_aliases": matched_aliases,
        "canonical_answer": canonical,
        "needs_human_audit": bool(audit_reasons),
        "audit_reasons": audit_reasons,
        "judge_status": "not_run",
    }


def answer_hit(text: str, aliases: Iterable[str]) -> bool:
    return bool(score_answer(text, aliases)["hit"])


def judge_prompt_for_record(record: dict[str, Any]) -> str:
    """Build a compact reference-based judge prompt for queued audits."""
    aliases = ", ".join(str(alias) for alias in record.get("answer_aliases", []))
    return (
        "Judge whether the completion answers the question using only the expected answer aliases.\n"
        "Return one JSON object with keys verdict, reason, corrected_short_answer.\n"
        f"Question: {record.get('question', '')}\n"
        f"Expected aliases: {aliases}\n"
        f"Expected UNKNOWN: {bool(record.get('expected_unknown'))}\n"
        f"Completion: {record.get('completion', '')}"
    )


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a small human/LLM-judge-ready audit item for an eval record."""
    return {
        "id": record.get("id"),
        "mode": record.get("mode"),
        "category": record.get("category"),
        "task_family": record.get("task_family"),
        "question": record.get("question"),
        "answer_aliases": record.get("answer_aliases", []),
        "expected_unknown": bool(record.get("expected_unknown")),
        "completion": record.get("completion", ""),
        "hit": bool(record.get("hit")),
        "match_type": record.get("match_type", "unknown"),
        "audit_reasons": record.get("audit_reasons", []),
        "judge_prompt": judge_prompt_for_record(record),
    }


def audit_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [audit_record(record) for record in records if bool(record.get("needs_human_audit"))]



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


def build_case_task_prompt(case: dict[str, Any]) -> str:
    question = str(case.get("question", "")).strip()
    instruction = str(case.get("instruction", "")).strip()
    instruction_line = f"{instruction}\n" if instruction else ""
    return (
        "Answer using only the evidence. Return only the short answer.\n"
        "If the evidence does not explicitly contain the requested answer, return UNKNOWN. "
        "Do not answer with related but different entities.\n"
        f"{instruction_line}"
        f"Question: {question}"
    )


def build_shared_evidence_context(
    case: dict[str, Any],
    *,
    evidence_results: Iterable[tuple[float, dict[str, Any]]] | None = None,
    max_evidence_chars: int = 4000,
) -> tuple[list[tuple[float, dict[str, Any]]], str]:
    """Materialize retrieved evidence once for all downstream views."""
    results = list(evidence_results) if evidence_results is not None else [
        (1.0, rec) for rec in evidence_records(case, include_distractors=False)
    ]
    return results, format_memory_context(results, max_chars=max_evidence_chars)


def build_case_prompt(
    case: dict[str, Any],
    *,
    include_evidence: bool = True,
    evidence_results: Iterable[tuple[float, dict[str, Any]]] | None = None,
    max_evidence_chars: int = 4000,
) -> str:
    task = build_case_task_prompt(case)
    if not include_evidence:
        return task

    _, memory_context = build_shared_evidence_context(
        case,
        evidence_results=evidence_results,
        max_evidence_chars=max_evidence_chars,
    )
    return build_prompt_with_memory(task, memory_context)


def build_workspace_memory_text(
    evidence_results: Iterable[tuple[float, dict[str, Any]]],
    *,
    max_evidence_chars: int = 4000,
) -> str:
    """Format retrieved evidence for workspace-side memory injection."""
    return format_memory_context(evidence_results, max_chars=max_evidence_chars)


def build_case_prompt_and_workspace_memory(
    case: dict[str, Any],
    *,
    include_evidence: bool = True,
    evidence_results: Iterable[tuple[float, dict[str, Any]]] | None = None,
    max_evidence_chars: int = 4000,
    evidence_injection: str = "prompt",
) -> tuple[str, str | None]:
    """Return the canonical prompt plus optional legacy workspace evidence text.

    `ssot` is the canonical path: retrieval is compiled into one donor-visible
    chat-template/token stream, and no second semantic evidence text is
    returned. `prompt` is kept as a legacy alias for the same visible-evidence
    shape. `workspace` and `dual` remain available as ablation/probe paths.
    """
    if evidence_injection in {"ssot", "prompt"}:
        return (
            build_case_prompt(
                case,
                include_evidence=include_evidence,
                evidence_results=evidence_results,
                max_evidence_chars=max_evidence_chars,
            ),
            None,
        )
    if evidence_injection not in {"workspace", "dual"}:
        raise ValueError(f"unknown evidence_injection: {evidence_injection}")

    _, memory_context = build_shared_evidence_context(
        case,
        evidence_results=evidence_results,
        max_evidence_chars=max_evidence_chars,
    )

    task = build_case_task_prompt(case)
    prompt = (
        build_prompt_with_memory(task, memory_context)
        if include_evidence and evidence_injection == "dual"
        else task
    )
    if not include_evidence:
        return prompt, None

    return prompt, memory_context


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


_DATE_RE = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_AUTHORITY_POSITIVE_CUES = (
    "signed",
    "supervisor",
    "official",
    "verified",
    "서명",
    "운영 공지",
    "공지",
)
_AUTHORITY_NEGATIVE_CUES = (
    "anonymous",
    "unverified",
    "rumor",
    "익명",
)
_TEMPORAL_CURRENT_CUES = (
    "current",
    "currently",
    "latest",
    "newest",
    "현재",
    "최신",
)
_TEMPORAL_STALE_CUES = (
    "previous",
    "older",
    "old ",
    "deprecated",
    "discarded",
    "이전",
    "폐기",
    "과거",
)
_DECOY_CUES = (
    "decoy",
    "other-",
    "other_",
    "project other",
    "bay decoy",
    "가짜",
)


def _latest_date_value(text: str) -> int:
    values: list[int] = []
    for year, month, day in _DATE_RE.findall(text):
        values.append(int(year) * 10000 + int(month) * 100 + int(day))
    return max(values) if values else 0


def evidence_source_reliability_score(case: dict[str, Any], rec: dict[str, Any]) -> float:
    """Score source trust without using target/evidence labels."""
    question = str(case.get("question", ""))
    category = str(case.get("category", ""))
    source_text = f"{rec.get('source', '')} {rec.get('text', '')}"
    haystack = source_text.casefold()
    category_key = category.casefold()
    task_family = case_task_family(case)

    score = 0.0
    score += 0.05 * len(_lex_terms(question) & _lex_terms(source_text))

    authority_context = (
        "authority" in category_key
        or "passphrase" in question.casefold()
        or "인증" in question
        or "통제실" in question
    )
    temporal_context = (
        "temporal" in category_key
        or "current" in question.casefold()
        or "현재" in question
    )

    if authority_context:
        score += 3.0 * sum(1 for cue in _AUTHORITY_POSITIVE_CUES if cue in haystack)
        score -= 3.0 * sum(1 for cue in _AUTHORITY_NEGATIVE_CUES if cue in haystack)
    if temporal_context:
        score += 2.5 * sum(1 for cue in _TEMPORAL_CURRENT_CUES if cue in haystack)
        score -= 2.5 * sum(1 for cue in _TEMPORAL_STALE_CUES if cue in haystack)
        date_value = _latest_date_value(source_text)
        if date_value:
            year = date_value // 10000
            month = (date_value // 100) % 100
            day = date_value % 100
            score += max(0, year - 2000) * 0.05 + month * 0.002 + day * 0.00005
    if task_family == "multi_hop":
        score -= 4.0 * sum(1 for cue in _DECOY_CUES if cue in haystack)
    return score


def govern_evidence_sources(
    case: dict[str, Any],
    results: Iterable[tuple[float, dict[str, Any]]],
    *,
    governor: str = "none",
) -> list[tuple[float, dict[str, Any]]]:
    """Filter/reorder evidence with non-label source reliability cues."""
    result_list = list(results)
    if governor in {"", "none"}:
        return result_list
    if governor != "reliability":
        raise ValueError(f"unknown evidence source governor: {governor}")

    scored: list[tuple[float, int, float, dict[str, Any]]] = []
    for index, (retrieval_score, rec) in enumerate(result_list):
        source_score = evidence_source_reliability_score(case, rec)
        enriched = dict(rec)
        enriched["source_governor"] = governor
        enriched["source_governor_score"] = source_score
        scored.append((source_score, index, float(retrieval_score), enriched))
    if not scored:
        return []

    task_family = case_task_family(case)
    best_score = max(item[0] for item in scored)
    kept = scored
    if task_family in {"conflict", "abstention"} and best_score >= 2.0:
        kept = [item for item in scored if item[0] >= max(1.0, best_score - 0.75)]
    elif task_family == "multi_hop" and any(item[0] < -1.0 for item in scored):
        kept = [item for item in scored if item[0] >= -1.0]

    if task_family in {"conflict", "abstention"}:
        kept.sort(key=lambda item: (item[0], item[2], -item[1]), reverse=True)
    else:
        kept.sort(key=lambda item: item[1])

    governed: list[tuple[float, dict[str, Any]]] = []
    for rank, (source_score, _index, retrieval_score, rec) in enumerate(kept):
        enriched = dict(rec)
        enriched["source_governor_rank"] = rank
        governed.append((retrieval_score, enriched))
    return governed


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


def _linked_record_key(rec: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (rec.get("case_id"), rec.get("source"), rec.get("chunk_id"))


def _link_normalize(text: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.casefold()).split())


def _link_terms(rec: dict[str, Any]) -> set[str]:
    text = _link_normalize(f"{rec.get('source', '')} {rec.get('text', '')}")
    tokens = [token for token in text.split() if len(token) > 1 and token not in _STOPWORDS]
    terms: set[str] = set()
    for width in (2, 3):
        for idx in range(0, max(0, len(tokens) - width + 1)):
            term = " ".join(tokens[idx : idx + width])
            if len(term) >= 5:
                terms.add(term)
    return terms


def expand_linked_evidence_results(
    selected: Iterable[tuple[float, dict[str, Any]]],
    candidates: Iterable[tuple[float, dict[str, Any]]],
    *,
    max_extra: int = 0,
) -> list[tuple[float, dict[str, Any]]]:
    selected_list = list(selected)
    if max_extra <= 0:
        return selected_list

    selected_keys = {_linked_record_key(rec) for _, rec in selected_list}
    terms: set[str] = set()
    for _, rec in selected_list:
        terms.update(_link_terms(rec))
    if not terms:
        return selected_list

    scored: list[tuple[int, float, tuple[float, dict[str, Any]]]] = []
    for score, rec in candidates:
        if _linked_record_key(rec) in selected_keys:
            continue
        haystack = _link_normalize(f"{rec.get('source', '')} {rec.get('text', '')}")
        link_score = sum(1 for term in terms if term in haystack)
        if link_score > 0:
            scored.append((link_score, float(score), (score, rec)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return selected_list + [item for _, _, item in scored[:max_extra]]


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
    total_exact_matches = 0
    total_normalized_exact = 0
    total_normalized_contains = 0
    total_unknown_correct = 0
    total_human_audits = 0
    total_retrieved_targets = 0
    total_all_targets = 0
    total_target_recall = 0.0
    bucket_factory = lambda: {  # noqa: E731
        "count": 0,
        "hits": 0,
        "exact_match_count": 0,
        "normalized_exact_count": 0,
        "normalized_contains_count": 0,
        "unknown_correct_count": 0,
        "human_audit_count": 0,
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
        exact_match = bool(record.get("exact_match", False))
        normalized_exact = bool(record.get("normalized_exact", False))
        normalized_contains = bool(record.get("normalized_contains", hit))
        unknown_correct = bool(record.get("unknown_correct", False))
        needs_human_audit = bool(record.get("needs_human_audit", False))
        retrieved_target = bool(record.get("retrieved_target"))
        all_targets_retrieved = bool(record.get("all_targets_retrieved"))
        target_recall = float(record.get("target_recall", 1.0 if retrieved_target else 0.0))
        mode = str(record.get("mode", "unknown"))
        category = str(record.get("category", "")).strip() or "uncategorized"
        task_family = case_task_family(record)
        total_count += 1
        total_hits += int(hit)
        total_exact_matches += int(exact_match)
        total_normalized_exact += int(normalized_exact)
        total_normalized_contains += int(normalized_contains)
        total_unknown_correct += int(unknown_correct)
        total_human_audits += int(needs_human_audit)
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
            buckets[key]["exact_match_count"] += int(exact_match)
            buckets[key]["normalized_exact_count"] += int(normalized_exact)
            buckets[key]["normalized_contains_count"] += int(normalized_contains)
            buckets[key]["unknown_correct_count"] += int(unknown_correct)
            buckets[key]["human_audit_count"] += int(needs_human_audit)
            buckets[key]["retrieved_target_count"] += int(retrieved_target)
            buckets[key]["all_targets_retrieved_count"] += int(all_targets_retrieved)
            recalls[key] += target_recall

    def with_accuracy(bucket: dict[str, int], *, target_recall_sum: float = 0.0) -> dict[str, float | int]:
        count = bucket["count"]
        hits = bucket["hits"]
        exact_matches = bucket.get("exact_match_count", 0)
        normalized_exact_matches = bucket.get("normalized_exact_count", 0)
        normalized_contains_matches = bucket.get("normalized_contains_count", hits)
        unknown_correct_matches = bucket.get("unknown_correct_count", 0)
        human_audits = bucket.get("human_audit_count", 0)
        retrieved_targets = bucket.get("retrieved_target_count", 0)
        all_targets = bucket.get("all_targets_retrieved_count", 0)
        return {
            "count": count,
            "hits": hits,
            "accuracy": hits / count if count else 0.0,
            "exact_match_count": exact_matches,
            "exact_match_rate": exact_matches / count if count else 0.0,
            "normalized_exact_count": normalized_exact_matches,
            "normalized_exact_rate": normalized_exact_matches / count if count else 0.0,
            "normalized_contains_count": normalized_contains_matches,
            "normalized_contains_rate": normalized_contains_matches / count if count else 0.0,
            "unknown_correct_count": unknown_correct_matches,
            "unknown_correct_rate": unknown_correct_matches / count if count else 0.0,
            "human_audit_count": human_audits,
            "human_audit_rate": human_audits / count if count else 0.0,
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
                "exact_match_count": total_exact_matches,
                "normalized_exact_count": total_normalized_exact,
                "normalized_contains_count": total_normalized_contains,
                "unknown_correct_count": total_unknown_correct,
                "human_audit_count": total_human_audits,
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
