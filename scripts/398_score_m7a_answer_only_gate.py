#!/usr/bin/env python3
"""Score the M7A public-MCQ answer-only gate from a strict eval report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def max_prediction_fraction(histogram: dict[str, Any], cases: int) -> float:
    if int(cases) <= 0 or not histogram:
        return 1.0
    return float(max(int(value) for value in histogram.values()) / max(1, int(cases)))


def score_gate(args: argparse.Namespace) -> dict[str, Any]:
    eval_report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    metrics = eval_report.get("metrics", {})
    cases = int(metrics.get("cases", 0))
    accuracy = float(metrics.get("accuracy", 0.0))
    invalid_rate = float(metrics.get("invalid_pred_rate", 1.0))
    prompt_echo_rate = float(metrics.get("prompt_echo_rate", 1.0))
    histogram = metrics.get("pred_answer_histogram", {})
    if not isinstance(histogram, dict):
        histogram = {}
    max_pred_fraction = max_prediction_fraction(histogram, cases)
    checks = {
        "cases_ge_min": cases >= int(args.min_cases),
        "accuracy_ge_min": accuracy >= float(args.min_accuracy),
        "invalid_pred_rate_le_max": invalid_rate <= float(args.max_invalid_pred_rate),
        "prompt_echo_rate_le_max": prompt_echo_rate <= float(args.max_prompt_echo_rate),
        "max_pred_fraction_le_max": max_pred_fraction <= float(args.max_pred_fraction),
    }
    accepted = all(bool(value) for value in checks.values())
    report = {
        "status": "complete",
        "decision": "accepted_m7a_public_mcq_answer_only_gate" if accepted else "rejected_m7a_public_mcq_answer_only_gate",
        "accepted": accepted,
        "target_level": "M7A public MCQ answer-only healing gate",
        "eval_report": str(args.eval_report),
        "metrics": {
            "cases": cases,
            "accuracy": accuracy,
            "invalid_pred_rate": invalid_rate,
            "prompt_echo_rate": prompt_echo_rate,
            "max_pred_fraction": max_pred_fraction,
            "pred_answer_histogram": histogram,
        },
        "thresholds": {
            "min_cases": int(args.min_cases),
            "min_accuracy": float(args.min_accuracy),
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
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-cases", type=int, default=64)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
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
