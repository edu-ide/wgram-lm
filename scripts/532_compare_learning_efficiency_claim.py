#!/usr/bin/env python3
"""Evaluate whether a reported run supports a 10x learning-efficiency claim.

The script is intentionally conservative. It only compares reports that share
the same task/data contract, then measures whether the candidate reaches a
target score in at most baseline_steps / factor. If the reports do not expose
early-enough evaluations, the claim remains unproven rather than inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_KEYS = (
    "task_families",
    "eval_task_families",
    "eval_family_order_invariant",
    "include_family_tag",
    "tokenizer_mode",
    "number_tokenizer_max_value",
    "number_tokenizer_op_role_tokens",
    "value_codec",
    "program_len",
    "modulus",
    "eval_cases",
    "train_think_steps",
    "eval_think_steps",
)


def load_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["_report_path"] = str(report_path)
    return report


def train_cfg(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("train")
    return cfg if isinstance(cfg, dict) else {}


def contract(report: dict[str, Any]) -> dict[str, Any]:
    cfg = train_cfg(report)
    merged = {key: cfg.get(key, report.get(key)) for key in CONTRACT_KEYS}
    # These two are top-level in mixed-text reports.
    merged["eval_family_order_invariant"] = report.get(
        "eval_family_order_invariant",
        cfg.get("eval_family_order_invariant"),
    )
    merged["include_family_tag"] = report.get("include_family_tag", cfg.get("include_family_tag"))
    return merged


def mismatch(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    ca = contract(a)
    cb = contract(b)
    return {key: (ca.get(key), cb.get(key)) for key in CONTRACT_KEYS if ca.get(key) != cb.get(key)}


def final_exact(report: dict[str, Any]) -> float:
    decisive = report.get("decisive_metrics")
    if isinstance(decisive, dict) and decisive.get("full_generation_exact") is not None:
        return float(decisive["full_generation_exact"])
    periodic = report.get("periodic_eval")
    if isinstance(periodic, list) and periodic:
        return float(periodic[-1].get("generation_exact", 0.0))
    return float(report.get("generation_exact", report.get("accuracy", 0.0)) or 0.0)


def final_depth_gain(report: dict[str, Any]) -> float | None:
    decisive = report.get("decisive_metrics")
    if not isinstance(decisive, dict):
        return None
    if decisive.get("full_minus_think0") is None:
        return None
    return float(decisive["full_minus_think0"])


def steps(report: dict[str, Any]) -> int | None:
    value = train_cfg(report).get("steps")
    return int(value) if value is not None else None


def first_step_at_or_above(report: dict[str, Any], target: float) -> int | None:
    periodic = report.get("periodic_eval")
    if not isinstance(periodic, list):
        return None
    for row in periodic:
        if float(row.get("generation_exact", 0.0)) >= float(target):
            step = row.get("step")
            return int(step) if step is not None else None
    return None


def summarize_run(report: dict[str, Any]) -> dict[str, Any]:
    cfg = train_cfg(report)
    return {
        "path": report.get("_report_path"),
        "target_level": report.get("target_level") or cfg.get("target_level"),
        "steps": steps(report),
        "final_generation_exact": final_exact(report),
        "final_depth_gain": final_depth_gain(report),
        "backend_summary": report.get("backend_summary"),
        "reject_reasons": report.get("reject_reasons", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--gram-ptrm-report", default="")
    parser.add_argument("--factor", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_report(args.baseline_report)
    candidate = load_report(args.candidate_report)
    mismatches = mismatch(baseline, candidate)
    comparable = not mismatches

    baseline_steps = steps(baseline)
    baseline_exact = final_exact(baseline)
    candidate_exact = final_exact(candidate)
    candidate_step_to_baseline = first_step_at_or_above(candidate, baseline_exact)

    max_allowed_step = None
    proven_10x = False
    if comparable and baseline_steps is not None:
        max_allowed_step = baseline_steps / float(args.factor)
        proven_10x = (
            candidate_step_to_baseline is not None
            and candidate_step_to_baseline <= max_allowed_step
        )

    baseline_gain = final_depth_gain(baseline)
    candidate_gain = final_depth_gain(candidate)
    gain_ratio = None
    if baseline_gain is not None and candidate_gain is not None and abs(baseline_gain) > 1e-12:
        gain_ratio = candidate_gain / baseline_gain

    accuracy_ratio = None
    if abs(baseline_exact) > 1e-12:
        accuracy_ratio = candidate_exact / baseline_exact

    gram_ptrm_summary: dict[str, Any] | None = None
    if args.gram_ptrm_report:
        gram_ptrm = load_report(args.gram_ptrm_report)
        gram_mismatches = mismatch(baseline, gram_ptrm)
        gram_ptrm_summary = {
            "run": summarize_run(gram_ptrm),
            "comparable_with_baseline": not gram_mismatches,
            "contract_mismatches": gram_mismatches,
        }

    verdict = "unproven"
    reasons: list[str] = []
    if not comparable:
        verdict = "invalid_comparison"
        reasons.append("baseline and candidate task/data contracts differ")
    elif proven_10x:
        verdict = "supports_10x_on_this_gate"
    else:
        if candidate_step_to_baseline is None:
            reasons.append("candidate did not reach the baseline final score in logged evaluations")
        elif max_allowed_step is not None:
            reasons.append(
                f"candidate reached baseline score at step {candidate_step_to_baseline}, "
                f"which is later than the 10x cutoff {max_allowed_step:.1f}",
            )
        reasons.append("broad HRM-Text pretraining efficiency is not tested by this synthetic gate")
        if gram_ptrm_summary is None:
            reasons.append("no comparable GRAM/PTRM report was supplied")

    report = {
        "verdict": verdict,
        "reasons": reasons,
        "factor": float(args.factor),
        "baseline": summarize_run(baseline),
        "candidate": summarize_run(candidate),
        "comparable": comparable,
        "contract_mismatches": mismatches,
        "metrics": {
            "baseline_final_generation_exact": baseline_exact,
            "candidate_final_generation_exact": candidate_exact,
            "final_accuracy_ratio": accuracy_ratio,
            "baseline_final_depth_gain": baseline_gain,
            "candidate_final_depth_gain": candidate_gain,
            "depth_gain_ratio": gain_ratio,
            "candidate_step_to_baseline_final_exact": candidate_step_to_baseline,
            "max_step_for_factor_claim": max_allowed_step,
        },
        "gram_ptrm": gram_ptrm_summary,
        "plain_language_read": (
            "This gate asks whether the candidate learns the same exam much faster, "
            "not whether it is broadly more training-efficient than HRM-Text pretraining. "
            "GRAM/PTRM only counts here if its report shares the same task contract and "
            "the normal answer path uses the GRAM/PTRM-generated candidates."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if verdict == "invalid_comparison":
        raise SystemExit(2)
    if verdict != "supports_10x_on_this_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
