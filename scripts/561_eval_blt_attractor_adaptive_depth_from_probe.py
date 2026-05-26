#!/usr/bin/env python3
"""Offline attractor-style adaptive depth probe from BLT depth/residual rows.

This does not claim a full Attractor Model implementation.  It asks the first
cheap question implied by arXiv:2605.12466: if residual convergence is used as
the stopping rule, which depth would the current checkpoint choose, and what
held-out loss would that imply?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _group_rows_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id", f"row-{index:05d}"))
        grouped[case_id].append(row)
    for case_rows in grouped.values():
        case_rows.sort(key=lambda item: int(item["think_steps"]))
    return dict(grouped)


def _select_row_for_threshold(
    case_rows: list[dict[str, Any]],
    *,
    residual_threshold: float,
    min_depth: int,
    max_depth: int,
) -> tuple[dict[str, Any], bool]:
    candidates = [
        row
        for row in case_rows
        if int(row["think_steps"]) >= int(min_depth) and int(row["think_steps"]) <= int(max_depth)
    ]
    if not candidates:
        raise ValueError("case has no rows inside requested depth range")
    for row in candidates:
        residual = row.get("fixed_point_residual")
        if residual is not None and float(residual) <= float(residual_threshold):
            return row, False
    return candidates[-1], True


def build_adaptive_depth_report(
    rows: list[dict[str, Any]],
    *,
    residual_thresholds: list[float],
    min_depth: int,
    max_depth: int,
) -> dict[str, Any]:
    grouped = _group_rows_by_case(rows)
    oracle_weighted_loss = 0.0
    oracle_target_tokens = 0
    oracle_depth_sum = 0.0
    oracle_depth_counts: Counter[str] = Counter()
    for case_rows in grouped.values():
        candidates = [
            row
            for row in case_rows
            if int(row["think_steps"]) >= int(min_depth) and int(row["think_steps"]) <= int(max_depth)
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda item: float(item["loss"]))
        targets = max(0, int(selected.get("target_tokens", 0)))
        depth = int(selected["think_steps"])
        oracle_weighted_loss += float(selected["loss"]) * targets
        oracle_target_tokens += targets
        oracle_depth_sum += float(depth)
        oracle_depth_counts[str(depth)] += 1
    threshold_summaries: list[dict[str, Any]] = []
    for threshold in residual_thresholds:
        weighted_loss = 0.0
        target_tokens = 0
        selected_depth_sum = 0.0
        fallback_count = 0
        depth_counts: Counter[str] = Counter()
        selected_rows: list[dict[str, Any]] = []
        for case_id, case_rows in grouped.items():
            selected, used_fallback = _select_row_for_threshold(
                case_rows,
                residual_threshold=float(threshold),
                min_depth=int(min_depth),
                max_depth=int(max_depth),
            )
            targets = int(selected.get("target_tokens", 0))
            depth = int(selected["think_steps"])
            weighted_loss += float(selected["loss"]) * max(0, targets)
            target_tokens += max(0, targets)
            selected_depth_sum += float(depth)
            fallback_count += int(bool(used_fallback))
            depth_counts[str(depth)] += 1
            selected_rows.append(
                {
                    "case_id": str(case_id),
                    "think_steps": int(depth),
                    "loss": float(selected["loss"]),
                    "target_tokens": int(targets),
                    "fixed_point_residual": (
                        float(selected["fixed_point_residual"])
                        if selected.get("fixed_point_residual") is not None
                        else None
                    ),
                    "used_fallback": bool(used_fallback),
                }
            )
        selected_count = len(selected_rows)
        threshold_summaries.append(
            {
                "residual_threshold": float(threshold),
                "adaptive_loss": (
                    weighted_loss / float(target_tokens) if int(target_tokens) > 0 else float("nan")
                ),
                "target_tokens": int(target_tokens),
                "selected_count": int(selected_count),
                "mean_selected_depth": (
                    selected_depth_sum / float(selected_count) if selected_count > 0 else float("nan")
                ),
                "fallback_count": int(fallback_count),
                "fallback_rate": fallback_count / float(selected_count) if selected_count > 0 else float("nan"),
                "selected_depth_counts": dict(sorted(depth_counts.items(), key=lambda item: int(item[0]))),
            }
        )
    best = min(threshold_summaries, key=lambda item: float(item["adaptive_loss"]))
    return {
        "probe_type": "blt_attractor_adaptive_depth_from_probe",
        "min_depth": int(min_depth),
        "max_depth": int(max_depth),
        "case_count": int(len(grouped)),
        "oracle_best_depth": {
            "oracle_loss": (
                oracle_weighted_loss / float(oracle_target_tokens)
                if int(oracle_target_tokens) > 0
                else float("nan")
            ),
            "target_tokens": int(oracle_target_tokens),
            "mean_oracle_depth": (
                oracle_depth_sum / float(len(grouped)) if int(len(grouped)) > 0 else float("nan")
            ),
            "oracle_depth_counts": dict(sorted(oracle_depth_counts.items(), key=lambda item: int(item[0]))),
        },
        "threshold_summaries": threshold_summaries,
        "best_threshold": float(best["residual_threshold"]),
        "best_adaptive_loss": float(best["adaptive_loss"]),
        "best_mean_selected_depth": float(best["mean_selected_depth"]),
        "plain_language_read": (
            "This is the cheap Attractor-Model preflight: use convergence "
            "residual as the stop rule, then check whether stopping by stability "
            "chooses cheaper or better answer states than a fixed depth."
        ),
    }


def load_depth_probe_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"depth probe report has no rows list: {path}")
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-report", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.075, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_adaptive_depth_report(
        load_depth_probe_rows(Path(args.depth_report)),
        residual_thresholds=[float(value) for value in args.thresholds],
        min_depth=int(args.min_depth),
        max_depth=int(args.max_depth),
    )
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
