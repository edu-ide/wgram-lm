#!/usr/bin/env python3
"""Score whether QTRM has a TRM-like breakthrough signal.

This gate is intentionally stricter than M6/M7 diagnostics. A narrow synthetic
win is not enough. A breakthrough claim needs both:

1. large recurrent-depth gain on scoped raw reasoning, and
2. scaled public-style core-depth gain on held-out MCQ cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def score(args: argparse.Namespace) -> dict[str, Any]:
    m6 = load_json(args.m6_report)
    m7 = load_json(args.m7b_report)
    best = m6.get("best_qtrm_native", {})
    qwen = m6.get("qwen36_baseline", {})
    m7_metrics = m7.get("metrics", {})

    m6_qtrm = as_float(best.get("full_generation_exact"))
    m6_qwen = as_float(qwen.get("score"))
    m6_margin = m6_qtrm - m6_qwen
    m6_core_gain = as_float(best.get("core_gain"))
    m6_ablation_drop = as_float(best.get("ablation_drop"))
    m6_min_family = as_float(best.get("min_family_generation_exact"))
    m6_cases = as_int(best.get("cases"))

    m7_full = nested(m7_metrics, "full", default={}) or {}
    m7_best_shallow = nested(m7_metrics, "best_shallow", default={}) or {}
    m7_cases = as_int(m7_full.get("cases"))
    m7_full_accuracy = as_float(m7_full.get("accuracy"))
    m7_best_shallow_accuracy = as_float(m7_best_shallow.get("accuracy"))
    m7_gain_vs_baseline = as_float(m7_metrics.get("gain_vs_baseline"))
    m7_gain_vs_best_shallow = as_float(m7_metrics.get("gain_vs_best_shallow"))

    checks = {
        "m6_accepted": bool(m6.get("accepted")),
        "m6_cases_ge_min": m6_cases >= int(args.min_m6_cases),
        "m6_qtrm_ge_min": m6_qtrm >= float(args.min_m6_qtrm),
        "m6_margin_ge_min": m6_margin >= float(args.min_m6_margin),
        "m6_core_gain_ge_min": m6_core_gain >= float(args.min_m6_core_gain),
        "m6_ablation_drop_ge_min": m6_ablation_drop >= float(args.min_m6_ablation_drop),
        "m6_min_family_ge_min": m6_min_family >= float(args.min_m6_min_family),
        "m7b_accepted": bool(m7.get("accepted")),
        "m7b_cases_ge_min": m7_cases >= int(args.min_m7_cases),
        "m7b_full_accuracy_ge_min": m7_full_accuracy >= float(args.min_m7_full_accuracy),
        "m7b_gain_vs_baseline_ge_min": m7_gain_vs_baseline >= float(args.min_m7_gain),
        "m7b_gain_vs_best_shallow_ge_min": m7_gain_vs_best_shallow >= float(args.min_m7_gain),
    }
    accepted = all(checks.values())
    report = {
        "status": "complete",
        "decision": "accepted_trm_like_breakthrough" if accepted else "rejected_trm_like_breakthrough",
        "accepted": accepted,
        "target_level": "TRM-like QTRM breakthrough gate",
        "policy": (
            "Reject unless QTRM shows a large recurrent-depth gain on scoped raw "
            "reasoning and a scaled held-out public-style core-depth gain."
        ),
        "inputs": {
            "m6_report": str(args.m6_report),
            "m7b_report": str(args.m7b_report),
        },
        "metrics": {
            "m6_qtrm_full_generation_exact": m6_qtrm,
            "m6_qwen36_proxy_score": m6_qwen,
            "m6_margin": m6_margin,
            "m6_core_gain": m6_core_gain,
            "m6_ablation_drop": m6_ablation_drop,
            "m6_min_family_generation_exact": m6_min_family,
            "m6_cases": m6_cases,
            "m7b_full_accuracy": m7_full_accuracy,
            "m7b_best_shallow_accuracy": m7_best_shallow_accuracy,
            "m7b_gain_vs_baseline": m7_gain_vs_baseline,
            "m7b_gain_vs_best_shallow": m7_gain_vs_best_shallow,
            "m7b_cases": m7_cases,
        },
        "thresholds": {
            "min_m6_cases": int(args.min_m6_cases),
            "min_m6_qtrm": float(args.min_m6_qtrm),
            "min_m6_margin": float(args.min_m6_margin),
            "min_m6_core_gain": float(args.min_m6_core_gain),
            "min_m6_ablation_drop": float(args.min_m6_ablation_drop),
            "min_m6_min_family": float(args.min_m6_min_family),
            "min_m7_cases": int(args.min_m7_cases),
            "min_m7_full_accuracy": float(args.min_m7_full_accuracy),
            "min_m7_gain": float(args.min_m7_gain),
        },
        "checks": checks,
        "reject_reasons": [key for key, ok in checks.items() if not ok],
        "next_action": (
            "promote to 512/1k public-style and language non-regression"
            if accepted
            else "do not claim TRM-like breakthrough; repair M7B core-depth scale-out"
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m6-report", required=True)
    parser.add_argument("--m7b-report", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-m6-cases", type=int, default=512)
    parser.add_argument("--min-m6-qtrm", type=float, default=0.50)
    parser.add_argument("--min-m6-margin", type=float, default=0.20)
    parser.add_argument("--min-m6-core-gain", type=float, default=0.25)
    parser.add_argument("--min-m6-ablation-drop", type=float, default=0.25)
    parser.add_argument("--min-m6-min-family", type=float, default=0.30)
    parser.add_argument("--min-m7-cases", type=int, default=256)
    parser.add_argument("--min-m7-full-accuracy", type=float, default=0.18)
    parser.add_argument("--min-m7-gain", type=float, default=0.03)
    return parser


def main() -> None:
    report = score(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()

