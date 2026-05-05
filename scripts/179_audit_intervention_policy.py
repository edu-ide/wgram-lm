#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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


def _by_case(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        case_id = str(row.get("id", ""))
        mode = str(row.get("mode", ""))
        if not case_id or not mode:
            continue
        grouped.setdefault(case_id, {})[mode] = row
    return grouped


def _hit(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("hit", False))


def _completion(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return str(row.get("completion") or row.get("raw_completion") or "").strip()


def _case_summary(
    case_id: str,
    modes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    donor = modes.get(DONOR_MODE)
    qtrm = modes.get(QTRM_MODE)
    core_off = modes.get(CORE_OFF_MODE)
    reference = qtrm or donor or core_off or {}
    return {
        "id": case_id,
        "category": reference.get("category"),
        "task_family": reference.get("task_family"),
        "question": reference.get("question"),
        "answer_aliases": reference.get("answer_aliases"),
        "donor_hit": _hit(donor),
        "qtrm_hit": _hit(qtrm),
        "core_off_hit": _hit(core_off),
        "donor_completion": _completion(donor),
        "qtrm_completion": _completion(qtrm),
        "core_off_completion": _completion(core_off),
    }


def _changed(a: str, b: str) -> bool:
    return a.strip() != b.strip()


def build_intervention_audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped = _by_case(records)
    cases = [
        _case_summary(case_id, modes)
        for case_id, modes in sorted(grouped.items())
        if QTRM_MODE in modes
    ]
    donor_hit_qtrm_miss = [
        case for case in cases if case["donor_hit"] and not case["qtrm_hit"]
    ]
    qtrm_hit_donor_miss = [
        case for case in cases if case["qtrm_hit"] and not case["donor_hit"]
    ]
    core_off_beats_qtrm = [
        case for case in cases if case["core_off_hit"] and not case["qtrm_hit"]
    ]
    qtrm_beats_core_off = [
        case for case in cases if case["qtrm_hit"] and not case["core_off_hit"]
    ]
    changed_donor_completion = [
        case
        for case in cases
        if case["donor_completion"]
        and case["qtrm_completion"]
        and _changed(case["donor_completion"], case["qtrm_completion"])
    ]
    changed_core_off_completion = [
        case
        for case in cases
        if case["core_off_completion"]
        and case["qtrm_completion"]
        and _changed(case["core_off_completion"], case["qtrm_completion"])
    ]
    return {
        "case_count": len(cases),
        "donor_hit_qtrm_miss_count": len(donor_hit_qtrm_miss),
        "qtrm_hit_donor_miss_count": len(qtrm_hit_donor_miss),
        "core_off_beats_qtrm_count": len(core_off_beats_qtrm),
        "qtrm_beats_core_off_count": len(qtrm_beats_core_off),
        "qtrm_changed_donor_completion_count": len(changed_donor_completion),
        "qtrm_changed_core_off_completion_count": len(changed_core_off_completion),
        "donor_hit_qtrm_miss_cases": donor_hit_qtrm_miss,
        "qtrm_hit_donor_miss_cases": qtrm_hit_donor_miss,
        "core_off_beats_qtrm_cases": core_off_beats_qtrm,
        "qtrm_beats_core_off_cases": qtrm_beats_core_off,
        "qtrm_changed_donor_completion_cases": changed_donor_completion,
        "qtrm_changed_core_off_completion_cases": changed_core_off_completion,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit when QTRM should or should not override donor/core-off answers."
    )
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_intervention_audit(load_jsonl(args.eval_jsonl))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_cases")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
