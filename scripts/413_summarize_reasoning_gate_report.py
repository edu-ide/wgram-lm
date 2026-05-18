#!/usr/bin/env python3
"""Summarize QTRM reasoning gate report.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_ORDER = (
    "full_generation_exact",
    "think0_generation_exact",
    "full_minus_think0",
    "full_minus_worst_ablation",
    "min_family_generation_exact",
    "state_reset_generation_exact",
    "op_zero_generation_exact",
    "teacher_forced_answer_loss",
)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path) -> str:
    report = load_json(path)
    lines: list[str] = []
    lines.append(f"report: {path}")
    lines.append(f"decision: {report.get('decision')} accepted={report.get('accepted')}")

    reject_reasons = report.get("reject_reasons") or []
    if reject_reasons:
        lines.append("reject_reasons: " + ", ".join(map(str, reject_reasons)))

    decisive = report.get("decisive_metrics") or {}
    if decisive:
        lines.append("decisive_metrics:")
        for key in METRIC_ORDER:
            if key in decisive:
                lines.append(f"  {key}: {fmt(decisive[key])}")
        for key in sorted(set(decisive) - set(METRIC_ORDER)):
            lines.append(f"  {key}: {fmt(decisive[key])}")

    best_periodic = report.get("best_periodic_eval")
    if not best_periodic:
        best_periodic = (report.get("train") or {}).get("best_periodic_eval")
    if best_periodic:
        lines.append("best_periodic:")
        for key in (
            "step",
            "generation_exact",
            "min_family_generation_exact",
            "teacher_forced_sequence_exact",
            "teacher_forced_answer_loss",
        ):
            if key in best_periodic:
                lines.append(f"  {key}: {fmt(best_periodic[key])}")

    periodic = report.get("periodic_eval") or []
    if periodic:
        last = periodic[-1]
        lines.append("last_periodic:")
        for key in (
            "step",
            "generation_exact",
            "min_family_generation_exact",
            "teacher_forced_sequence_exact",
            "teacher_forced_answer_loss",
        ):
            if key in last:
                lines.append(f"  {key}: {fmt(last[key])}")

    full_metrics = (report.get("eval_metrics") or {}).get("full") or {}
    by_family = full_metrics.get("by_family") or {}
    if by_family:
        lines.append("full_by_family:")
        for family, metrics in sorted(by_family.items()):
            exact = metrics.get("generation_exact")
            total = metrics.get("total")
            lines.append(f"  {family}: exact={fmt(exact)} total={total}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    for index, report_path in enumerate(args.reports):
        if index:
            print()
        print(summarize(report_path))


if __name__ == "__main__":
    main()
