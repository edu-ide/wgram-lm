#!/usr/bin/env python3
"""Score the M7B public-MCQ core-depth gate.

M7B is still not public benchmark parity. It checks whether the accepted M7A
answer-only surface is preserved and whether the deeper native recurrent core
improves strict greedy option-letter accuracy over no/shallow thinking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"report missing metrics: {path}")
    return report


def max_prediction_fraction(metrics: dict[str, Any]) -> float:
    cases = int(metrics.get("cases", 0))
    histogram = metrics.get("pred_answer_histogram", {})
    if cases <= 0 or not isinstance(histogram, dict) or not histogram:
        return 1.0
    return float(max(int(value) for value in histogram.values()) / max(1, cases))


def metric_summary(report: dict[str, Any], *, path: str | Path) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    return {
        "path": str(path),
        "cases": int(metrics.get("cases", 0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "invalid_pred_rate": float(metrics.get("invalid_pred_rate", 1.0)),
        "prompt_echo_rate": float(metrics.get("prompt_echo_rate", 1.0)),
        "max_pred_fraction": max_prediction_fraction(metrics),
        "pred_answer_histogram": metrics.get("pred_answer_histogram", {}),
    }


def score_gate(args: argparse.Namespace) -> dict[str, Any]:
    full_report = load_report(args.full_report)
    baseline_report = load_report(args.baseline_report)
    shallow_reports = [load_report(path) for path in args.shallow_report]
    full = metric_summary(full_report, path=args.full_report)
    baseline = metric_summary(baseline_report, path=args.baseline_report)
    shallow = [
        metric_summary(report, path=path)
        for report, path in zip(shallow_reports, args.shallow_report)
    ]
    comparison_pool = [baseline, *shallow]
    best_shallow = max(comparison_pool, key=lambda item: float(item["accuracy"]))
    gain_vs_baseline = float(full["accuracy"] - baseline["accuracy"])
    gain_vs_best_shallow = float(full["accuracy"] - best_shallow["accuracy"])
    checks = {
        "cases_ge_min": int(full["cases"]) >= int(args.min_cases),
        "full_accuracy_ge_min": float(full["accuracy"]) >= float(args.min_full_accuracy),
        "gain_vs_baseline_ge_min": gain_vs_baseline >= float(args.min_gain_vs_baseline),
        "gain_vs_best_shallow_ge_min": gain_vs_best_shallow >= float(args.min_gain_vs_best_shallow),
        "invalid_pred_rate_le_max": float(full["invalid_pred_rate"]) <= float(args.max_invalid_pred_rate),
        "prompt_echo_rate_le_max": float(full["prompt_echo_rate"]) <= float(args.max_prompt_echo_rate),
        "max_pred_fraction_le_max": float(full["max_pred_fraction"]) <= float(args.max_pred_fraction),
    }
    accepted = all(bool(value) for value in checks.values())
    report = {
        "status": "complete",
        "decision": "accepted_m7b_public_mcq_core_depth_gate" if accepted else "rejected_m7b_public_mcq_core_depth_gate",
        "accepted": accepted,
        "target_level": "M7B public MCQ core-depth correctness gate",
        "full_report": str(args.full_report),
        "baseline_report": str(args.baseline_report),
        "shallow_reports": [str(path) for path in args.shallow_report],
        "metrics": {
            "full": full,
            "baseline": baseline,
            "shallow": shallow,
            "best_shallow": best_shallow,
            "gain_vs_baseline": gain_vs_baseline,
            "gain_vs_best_shallow": gain_vs_best_shallow,
        },
        "thresholds": {
            "min_cases": int(args.min_cases),
            "min_full_accuracy": float(args.min_full_accuracy),
            "min_gain_vs_baseline": float(args.min_gain_vs_baseline),
            "min_gain_vs_best_shallow": float(args.min_gain_vs_best_shallow),
            "max_invalid_pred_rate": float(args.max_invalid_pred_rate),
            "max_prompt_echo_rate": float(args.max_prompt_echo_rate),
            "max_pred_fraction": float(args.max_pred_fraction),
        },
        "checks": checks,
        "reject_reasons": [key for key, ok in checks.items() if not ok],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--shallow-report", action="append", default=[])
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-cases", type=int, default=64)
    parser.add_argument("--min-full-accuracy", type=float, default=0.0)
    parser.add_argument("--min-gain-vs-baseline", type=float, default=0.03)
    parser.add_argument("--min-gain-vs-best-shallow", type=float, default=0.03)
    parser.add_argument("--max-invalid-pred-rate", type=float, default=0.05)
    parser.add_argument("--max-prompt-echo-rate", type=float, default=0.05)
    parser.add_argument("--max-pred-fraction", type=float, default=0.60)
    return parser


def main() -> None:
    report = score_gate(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
