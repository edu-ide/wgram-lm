#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_cases) > 0 and len(rows) >= int(max_cases):
                break
    return rows


def _csv(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in values) if values else "EMPTY"


def _gold_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "chosen", "canonical_answer"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    aliases = row.get("answer_aliases") or []
    for alias in aliases:
        value = str(alias).strip()
        if value:
            return value
    raise ValueError(f"row has no answer: {row.get('id')}")


def selected_copy_positions(row: dict[str, Any]) -> list[int]:
    values = [int(value) for value in row.get("input_list") or []]
    return [index for index, value in enumerate(values) if value % 2 == 0]


def oracle_pointer_copy_answer(row: dict[str, Any]) -> str:
    values = [int(value) for value in row.get("input_list") or []]
    positions = selected_copy_positions(row)
    return _csv([values[index] for index in positions])


def renderer_off_answer(row: dict[str, Any]) -> str:
    # Pointer/copy disabled: the lexicalizer has no selected source-token path.
    return "EMPTY"


def _exact(prediction: str, answer: str) -> bool:
    return str(prediction).strip() == str(answer).strip()


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_exact = 0
    renderer_off_exact = 0
    non_empty_rows = 0
    non_empty_full_exact = 0
    non_empty_renderer_off_exact = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        answer = _gold_answer(row)
        full = oracle_pointer_copy_answer(row)
        off = renderer_off_answer(row)
        is_non_empty = answer != "EMPTY"
        full_hit = _exact(full, answer)
        off_hit = _exact(off, answer)
        full_exact += int(full_hit)
        renderer_off_exact += int(off_hit)
        non_empty_rows += int(is_non_empty)
        non_empty_full_exact += int(is_non_empty and full_hit)
        non_empty_renderer_off_exact += int(is_non_empty and off_hit)
        if len(examples) < 5:
            examples.append(
                {
                    "id": row.get("id"),
                    "input_list": row.get("input_list"),
                    "selected_copy_positions": selected_copy_positions(row),
                    "answer": answer,
                    "oracle_pointer_copy_answer": full,
                    "renderer_off_answer": off,
                    "full_hit": full_hit,
                    "renderer_off_hit": off_hit,
                }
            )
    total = len(rows)
    full_accuracy = float(full_exact) / float(total) if total else 0.0
    renderer_off_accuracy = (
        float(renderer_off_exact) / float(total) if total else 0.0
    )
    non_empty_full_accuracy = (
        float(non_empty_full_exact) / float(non_empty_rows) if non_empty_rows else 0.0
    )
    non_empty_renderer_off_accuracy = (
        float(non_empty_renderer_off_exact) / float(non_empty_rows)
        if non_empty_rows
        else 0.0
    )
    accepted = (
        total > 0
        and full_accuracy == 1.0
        and non_empty_full_accuracy == 1.0
        and non_empty_full_accuracy > non_empty_renderer_off_accuracy
    )
    return {
        "decision": "accepted_l1_pointer_copy_oracle" if accepted else "rejected",
        "accepted": accepted,
        "target_level": "L1 official/minimal pointer-copy reproduction",
        "major_bottleneck": "latent/source state to token lexicalization",
        "method_class": "official/minimal reproduction",
        "prior_family": "pointer-generator / copy attention",
        "canonical_path": (
            "prompt source tokens -> selected source positions -> pointer/copy "
            "LM-token rendering"
        ),
        "rows": total,
        "non_empty_rows": non_empty_rows,
        "full_exact_rows": full_exact,
        "renderer_off_exact_rows": renderer_off_exact,
        "non_empty_full_exact_rows": non_empty_full_exact,
        "non_empty_renderer_off_exact_rows": non_empty_renderer_off_exact,
        "full_accuracy": full_accuracy,
        "renderer_off_accuracy": renderer_off_accuracy,
        "non_empty_full_accuracy": non_empty_full_accuracy,
        "non_empty_renderer_off_accuracy": non_empty_renderer_off_accuracy,
        "full_minus_renderer_off": full_accuracy - renderer_off_accuracy,
        "non_empty_full_minus_renderer_off": (
            non_empty_full_accuracy - non_empty_renderer_off_accuracy
        ),
        "examples": examples,
        "next_action": (
            "port this exact selected-source-position interface into QTRM L4 "
            "and require source-position logits to match the oracle copy path"
            if accepted
            else "fix the source-copy data/interface before QTRM integration"
        ),
    }


def run_gate(
    *,
    cases_path: str | Path,
    report_path: str | Path,
    max_cases: int = 0,
) -> dict[str, Any]:
    rows = load_jsonl(cases_path, max_cases=max_cases)
    report = evaluate_rows(rows)
    report["cases"] = str(cases_path)
    report["report_path"] = str(report_path)
    report["max_cases"] = int(max_cases)
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a minimal pointer/copy oracle renderer. This is a "
            "pre-QTRM L1 reproduction gate for source-position lexicalization."
        )
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_gate(
        cases_path=args.cases,
        report_path=args.report,
        max_cases=args.max_cases,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
