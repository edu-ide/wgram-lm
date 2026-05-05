#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


_ANSWER_PREFIX_RE = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)
_SOURCE_HEADER_RE = re.compile(r"^SOURCE=.*? CHUNK=.*? SCORE=")
_EN_IS_RE = re.compile(
    r"\b(?:is|are|was|were)\s+[\"'“”]?([A-Za-z0-9가-힣_.:/-]+(?:[- ][A-Za-z0-9가-힣_.:/-]+){0,3})[\"'“”]?(?:[.。]|$)",
    re.IGNORECASE,
)
_KO_TOPIC_RE = re.compile(
    r"(?<!에)(?:은|는)\s*([A-Za-z0-9가-힣_.:/-]+(?:[- ][A-Za-z0-9가-힣_.:/-]+){0,3})\s*(?:이다|입니다|다[.]?|[.。]|$)"
)
_BAD_CANDIDATES = {
    "UNKNOWN",
    "unknown",
    "the",
    "a",
    "an",
    "it",
    "관련",
    "요청한",
}


def _clean_answer(text: str) -> str:
    answer = _ANSWER_PREFIX_RE.sub("", str(text or "").strip()).strip()
    return answer.rstrip(".。").strip()


def _clean_candidate(text: str) -> str:
    out = str(text or "").strip().strip("\"'“”` ")
    out = out.rstrip(".。,:;").strip()
    for suffix in ("입니다", "이다", "다"):
        if out.endswith(suffix) and len(out) > len(suffix):
            out = out[: -len(suffix)].strip()
            break
    if " (" in out:
        out = out.split(" (", 1)[0].strip()
    return out


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_candidate(text).casefold())


def candidate_answers_from_prompt(prompt: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(prompt or "").splitlines():
        line = raw_line.strip()
        if not line or _SOURCE_HEADER_RE.match(line):
            continue
        if line.startswith("Use the evidence") or line.startswith("User prompt:"):
            continue
        if line.startswith("Answer using") or line.startswith("If the evidence"):
            continue
        if line.startswith("Question:") or line.startswith("관련 증거"):
            continue
        if line.startswith("요청한 답"):
            continue
        for pattern in (_EN_IS_RE, _KO_TOPIC_RE):
            for match in pattern.finditer(line):
                candidate = _clean_candidate(match.group(1))
                if not candidate or candidate in _BAD_CANDIDATES:
                    continue
                if len(candidate) > 80:
                    continue
                candidates.append(candidate)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normal(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def build_preference_row(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = str(row.get("prompt") or "")
    chosen = str(row.get("answer") or row.get("chosen") or "")
    chosen_answer = _clean_answer(chosen)
    chosen_key = _normal(chosen_answer)
    rejected = ""
    for candidate in candidate_answers_from_prompt(prompt):
        if _normal(candidate) != chosen_key:
            rejected = candidate
            break
    if not prompt or not chosen_answer or not rejected:
        return None
    metadata = dict(row.get("metadata") or {})
    return {
        "type": "canonical_plain_answer_preference",
        "case_id": str(row.get("case_id") or row.get("id") or ""),
        "prompt": prompt,
        "chosen": f"Answer: {chosen_answer}",
        "rejected": f"Answer: {rejected}",
        "preference_weight": float(row.get("preference_weight", 1.0)),
        "ssot_contract": "single_visible_prompt_stream",
        "metadata": {
            **metadata,
            "answer_contract": "plain_short_answer_preference",
            "rejected_answer": rejected,
        },
    }


def build_preference_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pref = build_preference_row(row)
        if pref is not None:
            out.append(pref)
    return out


def write_preference_rows(input_jsonl: str | Path, output_jsonl: str | Path) -> int:
    rows: list[dict[str, Any]] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    prefs = build_preference_rows(rows)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for pref in prefs:
            handle.write(json.dumps(pref, ensure_ascii=False) + "\n")
    return len(prefs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SSOT chosen/rejected answer preferences from canonical plain-answer rows."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_preference_rows(args.input_jsonl, args.output_jsonl)
    print(f"wrote {count} canonical plain-answer preference rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
