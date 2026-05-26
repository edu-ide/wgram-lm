#!/usr/bin/env python3
"""Evaluate whether deeper thinking converges toward answer-correct attractors.

This is a meta-gate over a depth sweep.  A recurrent model may become more
stable at deeper steps while moving toward the wrong answer basin.  This gate
therefore checks answer-facing signals, not residual stability alone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CRITICAL_TASKS = (
    "flipped_answer_icl",
    "successive_answer_icl",
    "truthy_answer_icl",
)


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def row_depth(row: dict[str, Any]) -> int:
    return int(row["depth"])


def select_baseline(rows: list[dict[str, Any]], baseline_depth: int) -> dict[str, Any]:
    exact = [row for row in rows if row_depth(row) == int(baseline_depth)]
    if exact:
        return exact[0]
    shallow = [row for row in rows if row_depth(row) <= int(baseline_depth)]
    if not shallow:
        raise ValueError(f"no baseline row at or below depth {baseline_depth}")
    with_loss = [row for row in shallow if finite_float(row.get("heldout_loss")) is not None]
    if with_loss:
        return min(with_loss, key=lambda row: float(row["heldout_loss"]))
    return max(shallow, key=lambda row: (float(row.get("gd_accuracy", 0.0)), float(row.get("gd_mean_margin", -1e9))))


def critical_pass_count(row: dict[str, Any], critical_tasks: tuple[str, ...]) -> int:
    passed = {str(task) for task in row.get("gd_passed_tasks", [])}
    return sum(1 for task in critical_tasks if task in passed)


def select_candidate(
    rows: list[dict[str, Any]],
    *,
    min_candidate_depth: int,
    critical_tasks: tuple[str, ...],
) -> dict[str, Any]:
    candidates = [row for row in rows if row_depth(row) >= int(min_candidate_depth)]
    if not candidates:
        raise ValueError(f"no candidate rows at or above depth {min_candidate_depth}")
    return max(
        candidates,
        key=lambda row: (
            critical_pass_count(row, critical_tasks),
            float(row.get("gd_accuracy", 0.0)),
            float(row.get("gd_mean_margin", -1e9)),
            -float(row.get("elapsed_sec", 1e9) or 1e9),
        ),
    )


def task_pass_status(row: dict[str, Any], critical_tasks: tuple[str, ...]) -> dict[str, bool]:
    failed = {str(task) for task in row.get("gd_failed_tasks", [])}
    passed = {str(task) for task in row.get("gd_passed_tasks", [])}
    return {task: task in passed and task not in failed for task in critical_tasks}


def build_checks(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    critical_tasks: tuple[str, ...],
    min_margin_gain: float,
    min_accuracy_gain: float,
    max_heldout_loss_regression: float,
    max_elapsed_ratio: float,
) -> dict[str, dict[str, Any]]:
    base_margin = float(baseline.get("gd_mean_margin", 0.0))
    cand_margin = float(candidate.get("gd_mean_margin", 0.0))
    base_acc = float(baseline.get("gd_accuracy", 0.0))
    cand_acc = float(candidate.get("gd_accuracy", 0.0))
    base_loss = finite_float(baseline.get("heldout_loss"))
    cand_loss = finite_float(candidate.get("heldout_loss"))
    base_residual = finite_float(baseline.get("mean_fixed_point_residual"))
    cand_residual = finite_float(candidate.get("mean_fixed_point_residual"))
    base_elapsed = finite_float(baseline.get("elapsed_sec"))
    cand_elapsed = finite_float(candidate.get("elapsed_sec"))

    critical_status = task_pass_status(candidate, critical_tasks)
    checks: dict[str, dict[str, Any]] = {
        "gd_mean_margin_improves": {
            "passed": bool(cand_margin >= base_margin + float(min_margin_gain)),
            "baseline": base_margin,
            "candidate": cand_margin,
            "required_gain": float(min_margin_gain),
        },
        "gd_accuracy_improves": {
            "passed": bool(cand_acc >= base_acc + float(min_accuracy_gain)),
            "baseline": base_acc,
            "candidate": cand_acc,
            "required_gain": float(min_accuracy_gain),
        },
        "critical_tasks_pass": {
            "passed": all(critical_status.values()),
            "tasks": critical_status,
        },
        "heldout_loss_not_regressed": {
            "passed": bool(
                base_loss is not None
                and cand_loss is not None
                and cand_loss <= base_loss + float(max_heldout_loss_regression)
            ),
            "baseline": base_loss,
            "candidate": cand_loss,
            "allowed_regression": float(max_heldout_loss_regression),
        },
        "residual_decreases": {
            "passed": bool(
                base_residual is not None
                and cand_residual is not None
                and cand_residual < base_residual
            ),
            "baseline": base_residual,
            "candidate": cand_residual,
        },
        "elapsed_not_exploded": {
            "passed": bool(
                base_elapsed is not None
                and cand_elapsed is not None
                and cand_elapsed <= base_elapsed * float(max_elapsed_ratio)
            ),
            "baseline": base_elapsed,
            "candidate": cand_elapsed,
            "max_ratio": float(max_elapsed_ratio),
        },
    }
    return checks


def failed_check_names(checks: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, payload in checks.items() if not bool(payload.get("passed"))]


def build_plain_korean_read(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> str:
    failed = failed_check_names(checks)
    if not failed:
        return (
            "더 깊게 생각할수록 같은 말문 기준 정답 쪽으로 이동했다. "
            "이 run은 조용한 생각이 아니라 정답으로 수렴하는 생각의 첫 증거다."
        )
    return (
        f"baseline depth {row_depth(baseline)}에서 candidate depth {row_depth(candidate)}로 갈 때 "
        "일부 안정화 또는 일부 문항 개선은 있지만, 정답 attractor로 승격할 증거는 부족하다. "
        f"실패 체크: {', '.join(failed)}. "
        "문과적으로는 마음이 더 차분해졌거나 몇 문제를 더 맞혔을 수는 있지만, "
        "같은 LM head가 정답 쪽으로 일관되게 끌려간 것은 아니다."
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.depth_sweep_summary)
    summary = load_json(summary_path)
    rows = list(summary.get("rows", []))
    if not rows:
        raise ValueError(f"depth sweep summary has no rows: {summary_path}")
    rows = sorted((dict(row) for row in rows), key=row_depth)
    critical_tasks = tuple(str(task) for task in args.critical_tasks)
    baseline = select_baseline(rows, int(args.baseline_depth))
    candidate = select_candidate(
        rows,
        min_candidate_depth=int(args.min_candidate_depth),
        critical_tasks=critical_tasks,
    )
    checks = build_checks(
        baseline=baseline,
        candidate=candidate,
        critical_tasks=critical_tasks,
        min_margin_gain=float(args.min_margin_gain),
        min_accuracy_gain=float(args.min_accuracy_gain),
        max_heldout_loss_regression=float(args.max_heldout_loss_regression),
        max_elapsed_ratio=float(args.max_elapsed_ratio),
    )
    failed = failed_check_names(checks)
    report = {
        "gate_type": "solution_aligned_answer_attractor",
        "source_summary": str(summary_path),
        "checkpoint": summary.get("checkpoint"),
        "critical_tasks": list(critical_tasks),
        "baseline_depth": row_depth(baseline),
        "candidate_depth": row_depth(candidate),
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "failed_checks": failed,
        "accepted": not failed,
        "plain_korean_read": build_plain_korean_read(
            baseline=baseline,
            candidate=candidate,
            checks=checks,
        ),
        "architecture_implication": (
            "Promote only if deeper recurrence improves answer-facing GD-lite "
            "margins and critical shortcut axes without damaging held-out loss. "
            "Residual convergence alone is not enough."
        ),
    }
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-sweep-summary", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--baseline-depth", type=int, default=2)
    parser.add_argument("--min-candidate-depth", type=int, default=4)
    parser.add_argument("--critical-tasks", nargs="+", default=list(DEFAULT_CRITICAL_TASKS))
    parser.add_argument("--min-margin-gain", type=float, default=0.02)
    parser.add_argument("--min-accuracy-gain", type=float, default=0.0)
    parser.add_argument("--max-heldout-loss-regression", type=float, default=0.01)
    parser.add_argument("--max-elapsed-ratio", type=float, default=1.5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_gate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
