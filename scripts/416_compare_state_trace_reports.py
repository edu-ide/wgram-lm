#!/usr/bin/env python3
"""Summarize recurrent state-trace diagnostics from QTRM report.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _last_float(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return None


def _summarize_report(path: Path, families: tuple[str, ...]) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    by_family = _get(report, "eval_metrics.think20.by_family")
    if by_family is None:
        by_family = _get(report, "eval_metrics.think6.by_family", {})
    trace_by_family = _get(report, "eval_metrics.state_trace.z_h_by_family", {})
    router_by_family = _get(report, "eval_metrics.order_router_probe.by_family", {})
    for family in families:
        trace = trace_by_family.get(family, {}) if isinstance(trace_by_family, dict) else {}
        family_metrics = by_family.get(family, {}) if isinstance(by_family, dict) else {}
        router = router_by_family.get(family, {}) if isinstance(router_by_family, dict) else {}
        rows.append(
            {
                "report": str(path),
                "decision": report.get("decision"),
                "accepted": report.get("accepted"),
                "family": family,
                "full": _get(report, "decisive_metrics.full_generation_exact"),
                "min_family": _get(
                    report,
                    "decisive_metrics.min_family_generation_exact",
                ),
                "family_exact": family_metrics.get("generation_exact"),
                "late_cosine": _last_float(trace.get("mean_consecutive_cosine")),
                "final_variance": _last_float(trace.get("mean_batch_variance_by_depth")),
                "final_delta": _last_float(trace.get("mean_step_delta_norm")),
                "last_hlh": router.get("last_hlh_prob"),
                "mean_hlh": router.get("mean_hlh_prob"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--families",
        default="checksum,modchain,revchain",
        help="Comma-separated families to show.",
    )
    args = parser.parse_args()
    families = tuple(item.strip() for item in str(args.families).split(",") if item.strip())
    rows: list[dict[str, Any]] = []
    for report_path in args.reports:
        rows.extend(_summarize_report(report_path, families))

    columns = (
        "report",
        "decision",
        "accepted",
        "family",
        "full",
        "min_family",
        "family_exact",
        "late_cosine",
        "final_variance",
        "final_delta",
        "last_hlh",
        "mean_hlh",
    )
    widths = {
        column: max(len(column), *(len(_fmt(row.get(column))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(_fmt(row.get(column)).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
