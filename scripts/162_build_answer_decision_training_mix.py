#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from wgram_lm.eval.memory_retrieval import (
    build_case_prompt,
    expected_unknown_case,
    load_cases,
    score_answer,
    select_evidence_results,
)


def _load_answer_decision_feature_module():
    path = Path("scripts/161_train_answer_decision_head.py")
    spec = importlib.util.spec_from_file_location("answer_decision_head_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_records(
    path: str | Path,
    *,
    record_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("id")
            if record_mode and str(row.get("mode", "")) != str(record_mode):
                continue
            if record_id:
                records[str(record_id)] = row
    return records


def label_answer_decision(record: dict[str, Any]) -> tuple[float, str]:
    aliases = record.get("answer_aliases") or []
    expected_unknown = expected_unknown_case(record)
    candidate = record.get("raw_completion") or record.get("completion") or ""
    candidate_hit = bool(
        score_answer(
            candidate,
            aliases,
            expected_unknown=expected_unknown,
        )["hit"]
    )
    unknown_hit = bool(
        score_answer(
            "Answer: UNKNOWN",
            aliases,
            expected_unknown=expected_unknown,
        )["hit"]
    )
    if unknown_hit and not candidate_hit:
        return 1.0, "unknown_block_improves"
    if candidate_hit:
        return 0.0, "candidate_already_correct"
    return 0.0, "positive_wrong_needs_revise_or_search"


def candidate_answer_text(record: dict[str, Any]) -> str:
    candidate = str(record.get("raw_completion") or record.get("completion") or "").strip()
    return (
        "Candidate answer:\n"
        f"{candidate}\n\n"
        "Decide whether the candidate must be blocked to UNKNOWN."
    )


def build_rows(
    cases: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    evidence_mode: str,
    retrieval_top_k: int,
    memory_max_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_module = _load_answer_decision_feature_module()
    answer_decision_feature_names = feature_module.feature_names(
        include_task_family=False,
    )
    for case in cases:
        case_id = str(case.get("id", ""))
        record = records.get(case_id)
        if record is None:
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
            max_evidence_chars=memory_max_chars,
        )
        target, reason = label_answer_decision(record)
        answer_decision_features = feature_module.extract_features(
            record,
            include_task_family=False,
        )
        rows.append(
            {
                "type": "answer_decision",
                "id": case_id,
                "prompt": prompt,
                "answer": candidate_answer_text(record),
                "answer_decision_target": target,
                "answer_decision_sample_weight": float(record.get("answer_decision_sample_weight", 1.0)),
                "answer_decision_features": answer_decision_features,
                "answer_decision_feature_names": answer_decision_feature_names,
                "metadata": {
                    "source_record": str(record.get("id", case_id)),
                    "candidate_completion": record.get("raw_completion")
                    or record.get("completion")
                    or "",
                    "expected_unknown": expected_unknown_case(record),
                    "baseline_hit": bool(record.get("hit", False)),
                    "label_reason": reason,
                    "task_family": record.get("task_family") or case.get("task_family"),
                    "category": record.get("category") or case.get("category"),
                    "retrieved_sources": [rec.get("source") for _, rec in evidence_results],
                },
            }
        )
    return rows


def apply_answer_decision_class_balance(rows: list[dict[str, Any]]) -> None:
    positives = sum(1 for row in rows if float(row.get("answer_decision_target", 0.0)) == 1.0)
    negatives = sum(1 for row in rows if float(row.get("answer_decision_target", 0.0)) == 0.0)
    if positives <= 0 or negatives <= 0:
        return
    positive_weight = max(1.0, float(negatives) / float(positives))
    for row in rows:
        if float(row.get("answer_decision_target", 0.0)) == 1.0:
            base_weight = float(row.get("answer_decision_sample_weight", 1.0))
            row["answer_decision_sample_weight"] = base_weight * positive_weight


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build QTRM in-model answer-decision training rows from eval records."
    )
    ap.add_argument("--cases-jsonl", required=True)
    ap.add_argument("--records-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--evidence-mode", default="all", choices=["target", "all", "lexical"])
    ap.add_argument("--retrieval-top-k", type=int, default=4)
    ap.add_argument("--memory-max-chars", type=int, default=4000)
    ap.add_argument(
        "--record-mode",
        default="qtrm_residual_with_evidence",
        help="Only use eval records from this mode. Set empty string to keep all modes.",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cases = load_cases(args.cases_jsonl)
    records = load_records(
        args.records_jsonl,
        record_mode=args.record_mode or None,
    )
    rows = build_rows(
        cases,
        records,
        evidence_mode=args.evidence_mode,
        retrieval_top_k=args.retrieval_top_k,
        memory_max_chars=args.memory_max_chars,
    )
    apply_answer_decision_class_balance(rows)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    positives = sum(1 for row in rows if float(row["answer_decision_target"]) == 1.0)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "positives": positives,
                "negatives": len(rows) - positives,
                "out": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
