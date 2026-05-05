#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.eval.memory_retrieval import (
    build_case_prompt,
    case_task_family,
    load_cases,
    select_evidence_results,
)


DONOR_MODE = "donor_only_with_evidence"
QTRM_MODE = "qtrm_residual_with_evidence"
CORE_OFF_MODE = "qtrm_core_off_with_evidence"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def records_by_case(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        case_id = str(row.get("id", ""))
        mode = str(row.get("mode", ""))
        if case_id and mode:
            grouped.setdefault(case_id, {})[mode] = row
    return grouped


def _hit(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("hit", False))


def clean_completion(row: dict[str, Any] | None) -> str:
    text = str((row or {}).get("completion") or (row or {}).get("raw_completion") or "").strip()
    return text.replace("\r\n", "\n")


def _append_preference(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
    prompt: str,
    chosen: str,
    rejected: str,
    reason: str,
    case: dict[str, Any],
    source_modes: tuple[str, str],
) -> None:
    chosen = clean_answer_text(chosen)
    rejected = clean_answer_text(rejected)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return
    rows.append(
        {
            "type": "intervention_policy_preference",
            "case_id": case_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "preference_weight": 1.0,
            "ssot_contract": "single_visible_prompt_stream",
            "metadata": {
                "intervention_reason": reason,
                "source_modes": list(source_modes),
                "category": case.get("category"),
                "task_family": case_task_family(case),
                "answer_aliases": case.get("answer_aliases"),
            },
        }
    )


def clean_answer_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.casefold().startswith("answer:"):
        return f"Answer: {text.split(':', 1)[1].strip()}"
    return f"Answer: {text}"


def build_intervention_preference_rows(
    cases: list[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    evidence_mode: str = "all",
    retrieval_top_k: int = 3,
    max_evidence_chars: int = 4000,
) -> list[dict[str, Any]]:
    grouped = records_by_case(records)
    out: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        modes = grouped.get(case_id)
        if not modes:
            continue
        donor = modes.get(DONOR_MODE)
        qtrm = modes.get(QTRM_MODE)
        core_off = modes.get(CORE_OFF_MODE)
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
        donor_completion = clean_completion(donor)
        qtrm_completion = clean_completion(qtrm)
        core_completion = clean_completion(core_off)
        if _hit(donor) and not _hit(qtrm):
            _append_preference(
                out,
                case_id=case_id,
                prompt=prompt,
                chosen=donor_completion,
                rejected=qtrm_completion,
                reason="preserve_donor",
                case=case,
                source_modes=(DONOR_MODE, QTRM_MODE),
            )
        if _hit(qtrm) and not _hit(donor):
            _append_preference(
                out,
                case_id=case_id,
                prompt=prompt,
                chosen=qtrm_completion,
                rejected=donor_completion,
                reason="allow_qtrm",
                case=case,
                source_modes=(QTRM_MODE, DONOR_MODE),
            )
        if _hit(core_off) and not _hit(qtrm):
            _append_preference(
                out,
                case_id=case_id,
                prompt=prompt,
                chosen=core_completion,
                rejected=qtrm_completion,
                reason="suppress_core_override",
                case=case,
                source_modes=(CORE_OFF_MODE, QTRM_MODE),
            )
    return dedupe_rows(out)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("case_id", "")),
            str(row.get("chosen", "")),
            str(row.get("rejected", "")),
            str((row.get("metadata") or {}).get("intervention_reason", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build on-policy intervention preference rows from donor/QTRM/core-off eval records."
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
    rows = build_intervention_preference_rows(
        load_cases(args.cases_jsonl),
        load_jsonl(args.eval_jsonl),
        evidence_mode=args.evidence_mode,
        retrieval_top_k=args.retrieval_top_k,
        max_evidence_chars=args.max_evidence_chars,
    )
    count = write_rows(rows, args.out_jsonl)
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str((row.get("metadata") or {}).get("intervention_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    print(json.dumps({"rows": count, "out": args.out_jsonl, "reasons": reasons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
