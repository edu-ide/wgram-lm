"""Thin answer-object utilities for Stage59 generality gates.

This module intentionally does not execute a task-specific solver. It only
normalizes candidate answer strings, scores them against aliases, and selects a
candidate. That keeps the verifier thin enough for transfer experiments where
the thinking path, not a hand-coded executor, must carry the work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


_TRAILING_PUNCT_RE = re.compile(r"[\s\.;:!\?]+$")
_LEADING_ANSWER_RE = re.compile(
    r"^\s*(?:answer|final answer|final digit|result|output)\s*[:=]\s*",
    re.IGNORECASE,
)
_ANSWER_PHRASE_RE = re.compile(
    r"\b(?:final answer|answer|result|output)\s*(?:is|:|=)\s*([^\n\r]+)",
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(r"^[\"'`]+|[\"'`]+$")


@dataclass(frozen=True)
class CandidateSelection:
    selected: str
    normalized_selected: str
    exact: bool
    oracle_exact: bool
    selected_index: int
    oracle_index: int | None
    selection_mode: str
    normalized_aliases: tuple[str, ...]
    normalized_candidates: tuple[str, ...]


def _strip_answer_prefix(text: str) -> str:
    value = str(text).strip()
    for _ in range(2):
        next_value = _LEADING_ANSWER_RE.sub("", value).strip()
        if next_value == value:
            break
        value = next_value
    return value


def _clean_candidate_fragment(text: str) -> str:
    value = str(text).strip()
    value = _QUOTE_RE.sub("", value).strip()
    return _TRAILING_PUNCT_RE.sub("", value)


def _looks_like_concise_answer(text: str) -> bool:
    value = _clean_candidate_fragment(text)
    return bool(
        re.fullmatch(r"(?i)true|false", value)
        or re.fullmatch(r"-?\d+", value)
        or re.fullmatch(r"-?\d+(\s*,\s*-?\d+)+", value)
        or value.upper() == "EMPTY"
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value)
    )


def extract_answer_candidate_text(text: Any) -> str:
    """Extract a concise answer candidate from a model completion.

    This is response parsing, not task solving. It only handles common answer
    formatting such as ``Final answer: X`` or a final one-line answer.
    """
    raw = str(text).strip()
    if not raw:
        return ""
    answer_phrases = list(_ANSWER_PHRASE_RE.finditer(raw))
    if answer_phrases:
        return _clean_candidate_fragment(answer_phrases[-1].group(1))

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = _clean_candidate_fragment(_strip_answer_prefix(line))
        if _looks_like_concise_answer(candidate):
            return candidate
    return _clean_candidate_fragment(lines[-1] if lines else raw)


def normalize_answer_text(text: Any) -> str:
    """Normalize a model answer without solving the task.

    Rules are deliberately conservative:
    - trim common answer prefixes;
    - remove trailing sentence punctuation;
    - remove whitespace around CSV commas;
    - canonicalize booleans;
    - canonicalize plain integers, preserving CSV integer lists as numbers.
    """
    value = _strip_answer_prefix(extract_answer_candidate_text(text))
    value = _clean_candidate_fragment(value)
    value = re.sub(r"\s*,\s*", ",", value)
    if re.fullmatch(r"(?i)true|false", value):
        return value.upper()
    if re.fullmatch(r"-?\d+", value):
        return str(int(value))
    if re.fullmatch(r"-?\d+(,-?\d+)+", value):
        return ",".join(str(int(part)) for part in value.split(","))
    if value.upper() == "EMPTY":
        return "EMPTY"
    return value.lower()


def answer_aliases(row: dict[str, Any]) -> tuple[str, ...]:
    aliases = row.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return tuple(str(alias) for alias in aliases)
    for key in ("answer", "chosen", "answer_text", "gold_answer"):
        if row.get(key) is not None:
            return (str(row[key]),)
    return ("",)


def normalized_alias_set(aliases: Iterable[Any]) -> tuple[str, ...]:
    normalized = []
    seen = set()
    for alias in aliases:
        item = normalize_answer_text(alias)
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return tuple(normalized)


def select_candidate(
    candidates: Sequence[Any],
    aliases: Iterable[Any],
    *,
    selection_mode: str = "oracle",
) -> CandidateSelection:
    """Select and score a candidate answer.

    ``selection_mode="first"`` scores the first candidate as the model's actual
    answer. ``selection_mode="oracle"`` is an upper bound: it checks whether the
    correct answer appeared anywhere in the candidate list and selects that item
    if present. Keeping both modes explicit prevents a candidate-coverage probe
    from being mistaken for a deployable verifier.
    """
    if selection_mode not in {"first", "oracle"}:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")
    normalized_aliases = normalized_alias_set(aliases)
    normalized_candidates = tuple(normalize_answer_text(candidate) for candidate in candidates)
    alias_set = set(normalized_aliases)
    oracle_index = next(
        (index for index, candidate in enumerate(normalized_candidates) if candidate in alias_set),
        None,
    )
    selected_index = oracle_index if selection_mode == "oracle" and oracle_index is not None else 0
    selected = str(candidates[selected_index]) if candidates else ""
    normalized_selected = normalized_candidates[selected_index] if normalized_candidates else ""
    exact = normalized_selected in alias_set
    return CandidateSelection(
        selected=selected,
        normalized_selected=normalized_selected,
        exact=exact,
        oracle_exact=oracle_index is not None,
        selected_index=int(selected_index),
        oracle_index=oracle_index,
        selection_mode=selection_mode,
        normalized_aliases=normalized_aliases,
        normalized_candidates=normalized_candidates,
    )


def answer_kind(text: Any) -> str:
    value = normalize_answer_text(text)
    if re.fullmatch(r"\d", value):
        return "single_digit"
    if re.fullmatch(r"-?\d+", value):
        return "integer"
    if re.fullmatch(r"-?\d+(,-?\d+)+", value):
        return "csv_integer_list"
    if value in {"TRUE", "FALSE"}:
        return "boolean"
    if value == "EMPTY":
        return "empty"
    return "symbolic_or_text"


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    hits = sum(1 for row in records if bool(row.get("exact")))
    by_family: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for row in records:
        family = str(row.get("task_family") or row.get("family") or "unknown")
        kind = str(row.get("answer_kind") or "unknown")
        for bucket_map, key in ((by_family, family), (by_kind, kind)):
            bucket = bucket_map.setdefault(key, {"hits": 0, "total": 0})
            bucket["hits"] += int(bool(row.get("exact")))
            bucket["total"] += 1
    return {
        "hits": hits,
        "total": total,
        "accuracy": float(hits / max(1, total)),
        "by_family": {
            key: {**value, "accuracy": float(value["hits"] / max(1, value["total"]))}
            for key, value in sorted(by_family.items())
        },
        "by_kind": {
            key: {**value, "accuracy": float(value["hits"] / max(1, value["total"]))}
            for key, value in sorted(by_kind.items())
        },
    }
