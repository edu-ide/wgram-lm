#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.memory_retrieval import (
    build_case_prompt,
    case_task_family,
    load_cases,
    select_evidence_results,
)


QTRM_MODE = "qtrm_residual_with_evidence"
DONOR_MODE = "donor_only_with_evidence"
CORE_OFF_MODE = "qtrm_core_off_with_evidence"
WORKSPACE_OFF_MODE = "qtrm_workspace_off_with_evidence"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def records_by_case(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        case_id = str(row.get("id") or row.get("case_id") or "")
        mode = str(row.get("mode") or row.get("model_mode") or "")
        if case_id and mode:
            grouped.setdefault(case_id, {})[mode] = row
    return grouped


def clean_answer_text(text: str) -> str:
    answer = str(text or "").strip().replace("\r\n", "\n")
    if not answer:
        return ""
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    answer = answer.splitlines()[0].strip()
    answer = answer.rstrip(".。").strip()
    if not answer:
        return ""
    return f"Answer: {answer}"


def canonical_answer(case: dict[str, Any]) -> str:
    aliases = case.get("answer_aliases") or []
    if not aliases:
        return ""
    answer = str(aliases[0]).strip()
    if answer.casefold() == "unknown":
        answer = "UNKNOWN"
    return f"Answer: {answer}"


def _hit(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("hit"))


def is_reliability_focus_case(case: dict[str, Any]) -> bool:
    category = str(case.get("category") or "").casefold()
    family = str(case_task_family(case) or "").casefold()
    aliases = [str(a).casefold() for a in (case.get("answer_aliases") or [])]
    return (
        "negative" in category
        or "temporal" in category
        or "conflict" in category
        or family in {"abstention", "conflict"}
        or "unknown" in aliases
    )


def _add_row(
    rows: list[dict[str, Any]],
    *,
    case: dict[str, Any],
    prompt: str,
    chosen: str,
    rejected: str,
    reason: str,
    source_mode: str,
    rejected_mode: str,
    preference_weight: float,
) -> None:
    chosen = clean_answer_text(chosen)
    rejected = clean_answer_text(rejected)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return
    rows.append(
        {
            "type": "reliability_hard_negative_preference",
            "case_id": str(case.get("id") or ""),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "preference_weight": float(preference_weight),
            "ssot_contract": "single_visible_prompt_stream",
            "metadata": {
                "reliability_reasons": [reason],
                "source_mode": source_mode,
                "rejected_mode": rejected_mode,
                "category": case.get("category"),
                "task_family": case_task_family(case),
                "expected_unknown": "UNKNOWN" in chosen,
                "answer_aliases": case.get("answer_aliases"),
            },
        }
    )


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("chosen")), str(row.get("rejected")))
        if key not in merged:
            merged[key] = row
            continue
        existing = merged[key]
        metadata = dict(existing.get("metadata") or {})
        reasons = list(metadata.get("reliability_reasons") or [])
        for reason in (row.get("metadata") or {}).get("reliability_reasons") or []:
            if reason not in reasons:
                reasons.append(reason)
        metadata["reliability_reasons"] = reasons
        existing["metadata"] = metadata
        existing["preference_weight"] = max(
            float(existing.get("preference_weight", 1.0)),
            float(row.get("preference_weight", 1.0)),
        )
    return list(merged.values())


def build_reliability_preference_rows(
    cases: list[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    evidence_mode: str = "all",
    retrieval_top_k: int = 3,
    max_evidence_chars: int = 4000,
) -> list[dict[str, Any]]:
    grouped = records_by_case(records)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not is_reliability_focus_case(case):
            continue
        case_id = str(case.get("id") or "")
        modes = grouped.get(case_id) or {}
        qtrm = modes.get(QTRM_MODE)
        if qtrm is None:
            continue
        evidence_results = select_evidence_results(
            case,
            evidence_mode=evidence_mode,
            top_k=retrieval_top_k,
        )
        prompt = build_case_prompt(
            case,
            include_evidence=True,
            evidence_results=evidence_results,
            max_evidence_chars=max_evidence_chars,
        )
        chosen = canonical_answer(case)
        if not chosen:
            continue
        if not _hit(qtrm):
            _add_row(
                rows,
                case=case,
                prompt=prompt,
                chosen=chosen,
                rejected=str(qtrm.get("completion") or qtrm.get("raw_completion") or ""),
                reason="repair_qtrm_miss",
                source_mode="canonical_answer",
                rejected_mode=QTRM_MODE,
                preference_weight=1.30,
            )
        for baseline_mode, reason in (
            (DONOR_MODE, "strengthen_qtrm_win_over_donor"),
            (CORE_OFF_MODE, "strengthen_core_causal_win"),
            (WORKSPACE_OFF_MODE, "strengthen_workspace_causal_win"),
        ):
            baseline = modes.get(baseline_mode)
            if qtrm is not None and _hit(qtrm) and baseline is not None and not _hit(baseline):
                _add_row(
                    rows,
                    case=case,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=str(baseline.get("completion") or baseline.get("raw_completion") or ""),
                    reason=reason,
                    source_mode=QTRM_MODE,
                    rejected_mode=baseline_mode,
                    preference_weight=1.00,
                )
        for baseline_mode, reason in (
            (DONOR_MODE, "preserve_donor_correct"),
            (CORE_OFF_MODE, "suppress_bad_core_override"),
        ):
            baseline = modes.get(baseline_mode)
            if baseline is not None and _hit(baseline) and not _hit(qtrm):
                _add_row(
                    rows,
                    case=case,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=str(qtrm.get("completion") or qtrm.get("raw_completion") or ""),
                    reason=reason,
                    source_mode=baseline_mode,
                    rejected_mode=QTRM_MODE,
                    preference_weight=1.50,
                )
    return dedupe_rows(rows)


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SSOT reliability hard-negative preference rows from eval records."
    )
    parser.add_argument("--cases-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--evidence-mode", default="all", choices=["target", "all", "lexical", "none"])
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=4000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = build_reliability_preference_rows(
        load_cases(args.cases_jsonl),
        load_jsonl(args.eval_jsonl),
        evidence_mode=args.evidence_mode,
        retrieval_top_k=args.retrieval_top_k,
        max_evidence_chars=args.max_evidence_chars,
    )
    count = write_rows(rows, args.out_jsonl)
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in (row.get("metadata") or {}).get("reliability_reasons") or ["unknown"]:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    print(json.dumps({"rows": count, "out": args.out_jsonl, "reasons": reasons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
