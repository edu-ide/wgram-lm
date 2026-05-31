from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RESIDUAL_FIELDS = (
    "fixed_point_residual",
    "core_fixed_point_residual",
    "convergence_residual",
    "mean_fixed_point_residual",
    "residual",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} in {path} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def extract_depth(record: dict[str, Any]) -> int | None:
    for field in ("depth", "core_steps", "think_steps"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return int(value)
    mode = str(record.get("mode") or "")
    match = re.search(r"core_steps_(\d+)", mode)
    if match:
        return int(match.group(1))
    match = re.search(r"think_steps_(\d+)", mode)
    if match:
        return int(match.group(1))
    return None


def _case_id(record: dict[str, Any]) -> str | None:
    value = record.get("case_id", record.get("id"))
    if value is None or not str(value).strip():
        return None
    return str(value)


def _residual(record: dict[str, Any]) -> float | None:
    for field in RESIDUAL_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return float(value)
    return None


def _canonical_completion(record: dict[str, Any]) -> str:
    completion = str(record.get("completion", record.get("cleaned_response", ""))).strip().casefold()
    return re.sub(r"\s+", " ", completion)


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(sum(items) / len(items))


def _trajectory_summary(rows: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    residuals = [_residual(row) for row in rows]
    residual_values = [value for value in residuals if value is not None]
    hits = sum(1 for row in rows if bool(row.get("hit", row.get("generation_hit", False))))
    case_ids = {_case_id(row) for row in rows if _case_id(row) is not None}
    return {
        "depth": int(depth),
        "case_count": len(case_ids),
        "trajectory_count": len(rows),
        "hits": int(hits),
        "trajectory_accuracy": float(hits / len(rows)) if rows else 0.0,
        "mean_residual": _mean(residual_values),
        "residual_count": len(residual_values),
    }


def _majority_hit(case_rows: list[dict[str, Any]]) -> bool:
    completions = [_canonical_completion(row) for row in case_rows]
    completions = [value for value in completions if value]
    if not completions:
        return any(bool(row.get("hit", row.get("generation_hit", False))) for row in case_rows)
    counts = Counter(completions)
    winner = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return any(
        _canonical_completion(row) == winner
        and bool(row.get("hit", row.get("generation_hit", False)))
        for row in case_rows
    )


def _breadth_summary(rows: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = _case_id(row)
        if case_id is None:
            continue
        by_case[case_id].append(row)

    top1_rows: list[dict[str, Any]] = []
    top1_examples: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(by_case.items()):
        rows_with_residual = [row for row in case_rows if _residual(row) is not None]
        if not rows_with_residual:
            continue
        selected = min(rows_with_residual, key=lambda row: _residual(row) or float("inf"))
        top1_rows.append(selected)
        top1_examples.append(
            {
                "case_id": case_id,
                "restart_id": selected.get("restart_id"),
                "residual": _residual(selected),
                "hit": bool(selected.get("hit", selected.get("generation_hit", False))),
                "completion": selected.get("completion", selected.get("cleaned_response", "")),
            }
        )

    trajectory = _trajectory_summary(rows, depth)
    majority_hits = sum(1 for case_rows in by_case.values() if _majority_hit(case_rows))
    oracle_hits = sum(
        1
        for case_rows in by_case.values()
        if any(bool(row.get("hit", row.get("generation_hit", False))) for row in case_rows)
    )
    top1_hits = sum(1 for row in top1_rows if bool(row.get("hit", row.get("generation_hit", False))))
    residual_hit_values = [
        _residual(row)
        for row in rows
        if _residual(row) is not None and bool(row.get("hit", row.get("generation_hit", False)))
    ]
    residual_miss_values = [
        _residual(row)
        for row in rows
        if _residual(row) is not None and not bool(row.get("hit", row.get("generation_hit", False)))
    ]
    top1_case_count = len(top1_rows)
    case_count = len(by_case)

    return {
        **trajectory,
        "case_count": int(case_count),
        "top1_case_count": int(top1_case_count),
        "top1_case_ids": [str(_case_id(row)) for row in top1_rows],
        "top1_convergence_accuracy": (
            float(top1_hits / top1_case_count) if top1_case_count else None
        ),
        "majority_vote_accuracy": float(majority_hits / case_count) if case_count else None,
        "oracle_accuracy": float(oracle_hits / case_count) if case_count else None,
        "hit_mean_residual": _mean(value for value in residual_hit_values if value is not None),
        "miss_mean_residual": _mean(value for value in residual_miss_values if value is not None),
        "top1_examples": top1_examples[:10],
    }


def build_depth_breadth_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped_rows = 0
    for row in rows:
        depth = extract_depth(row)
        if depth is None:
            skipped_rows += 1
            continue
        by_depth[int(depth)].append(row)

    depth_ladder = [
        _trajectory_summary(depth_rows, depth)
        for depth, depth_rows in sorted(by_depth.items())
    ]
    breadth_by_depth = {
        int(depth): _breadth_summary(depth_rows, int(depth))
        for depth, depth_rows in sorted(by_depth.items())
    }

    best_top1_depth = None
    measured_top1 = [
        (depth, summary)
        for depth, summary in breadth_by_depth.items()
        if summary.get("top1_convergence_accuracy") is not None
    ]
    if measured_top1:
        best_top1_depth = max(
            measured_top1,
            key=lambda item: (float(item[1]["top1_convergence_accuracy"]), int(item[0])),
        )[0]

    all_case_ids = {_case_id(row) for row in rows if _case_id(row) is not None}
    all_residuals = [_residual(row) for row in rows]
    residual_count = sum(1 for value in all_residuals if value is not None)
    trajectory_count = sum(len(depth_rows) for depth_rows in by_depth.values())
    passed_checks: list[str] = []
    failed_checks: list[str] = []

    if skipped_rows:
        failed_checks.append("rows_missing_depth")
    if residual_count:
        passed_checks.append("residual_records_present")
    else:
        failed_checks.append("no_residual_records")

    if len(depth_ladder) >= 2:
        shallow = depth_ladder[0]
        deep = depth_ladder[-1]
        if float(deep["trajectory_accuracy"]) > float(shallow["trajectory_accuracy"]):
            passed_checks.append("deepest_depth_beats_shallowest_trajectory_average")
        else:
            failed_checks.append("no_depth_trajectory_gain")
    else:
        failed_checks.append("fewer_than_two_depths")

    if best_top1_depth is not None:
        best = breadth_by_depth[int(best_top1_depth)]
        top1 = best.get("top1_convergence_accuracy")
        average = float(best["trajectory_accuracy"])
        if top1 is not None and float(top1) > average:
            passed_checks.append("top1_convergence_beats_trajectory_average")
        else:
            failed_checks.append("top1_convergence_does_not_beat_average")
        hit_residual = best.get("hit_mean_residual")
        miss_residual = best.get("miss_mean_residual")
        if hit_residual is not None and miss_residual is not None and float(hit_residual) < float(miss_residual):
            passed_checks.append("hits_have_lower_residual_than_misses")
        elif hit_residual is not None and miss_residual is not None:
            failed_checks.append("hits_do_not_have_lower_residual_than_misses")

    return {
        "probe_type": "depth_breadth_convergence",
        "claim": (
            "QTRM recurrent depth and stochastic breadth should improve answers when "
            "convergence residuals select better trajectories."
        ),
        "case_count": len(all_case_ids),
        "trajectory_count": int(trajectory_count),
        "skipped_rows": int(skipped_rows),
        "depths": sorted(int(depth) for depth in by_depth),
        "depth_ladder": depth_ladder,
        "breadth_by_depth": breadth_by_depth,
        "best_top1_depth": best_top1_depth,
        "residual_fields": list(RESIDUAL_FIELDS),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "status": "accepted" if not failed_checks else "inconclusive",
    }
