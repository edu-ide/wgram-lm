#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CRITICAL_MODES = (
    "qtrm_core_steps_8_no_evidence",
    "qtrm_core_steps_8_low_donor_no_evidence",
    "qtrm_core_steps_8_qtrm_only_no_evidence",
)

GATE_PROFILE_INCLUDED_MODES: dict[str, tuple[str, ...] | None] = {
    "strict": None,
    "qtrm_core": (
        "qtrm_core_steps_8_no_evidence",
        "qtrm_core_steps_8_qtrm_only_no_evidence",
    ),
    "qtrm_only": (
        "qtrm_core_steps_8_qtrm_only_no_evidence",
    ),
    "fused": (
        "qtrm_core_steps_8_no_evidence",
        "qtrm_core_steps_8_low_donor_no_evidence",
    ),
    "low_donor_fused": (
        "qtrm_core_steps_8_low_donor_no_evidence",
    ),
}


def _normalize_answer(text: str) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [1.0 / len(values) for _ in values]
    max_value = max(finite)
    exp_values = [math.exp(value - max_value) if math.isfinite(value) else 0.0 for value in values]
    total = sum(exp_values)
    if total <= 0.0:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in exp_values]


def _choice_matches_alias(choice: str, aliases: Iterable[str]) -> bool:
    normalized_choice = _normalize_answer(choice)
    if not normalized_choice:
        return False
    for alias in aliases:
        normalized_alias = _normalize_answer(str(alias).strip().strip(".:;"))
        if normalized_alias and normalized_alias == normalized_choice:
            return True
    return False


def record_calibration(record: dict[str, Any]) -> dict[str, Any]:
    scores = [
        row
        for row in record.get("choice_scores", [])
        if isinstance(row, dict) and "choice" in row and "logprob" in row
    ]
    if not scores:
        return {"available": False, "reason": "missing_choice_scores"}

    logprobs = [float(row["logprob"]) for row in scores]
    probs = _softmax(logprobs)
    best_index = max(range(len(scores)), key=lambda idx: probs[idx])
    predicted_choice = str(scores[best_index]["choice"])
    confidence = float(probs[best_index])
    conflict_gate_values = [
        float(row["donor_qtrm_conflict_gate_mean"])
        for row in scores
        if "donor_qtrm_conflict_gate_mean" in row
    ]
    aliases = record.get("answer_aliases", [])
    target_probability = sum(
        prob
        for row, prob in zip(scores, probs)
        if _choice_matches_alias(str(row.get("choice", "")), aliases)
    )
    correct = 1.0 if bool(record.get("hit")) else 0.0
    return {
        "available": True,
        "predicted_choice": predicted_choice,
        "confidence": confidence,
        "correct": correct,
        "target_probability": float(target_probability),
        "brier": float((confidence - correct) ** 2),
        "choice_count": len(scores),
        "predicted_conflict_gate_mean": (
            float(scores[best_index]["donor_qtrm_conflict_gate_mean"])
            if "donor_qtrm_conflict_gate_mean" in scores[best_index]
            else None
        ),
        "mean_choice_conflict_gate_mean": (
            sum(conflict_gate_values) / len(conflict_gate_values)
            if conflict_gate_values
            else None
        ),
    }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["calibration"] = record_calibration(row)
            rows.append(row)
    return rows


def calibration_summary(records: Iterable[dict[str, Any]], *, n_bins: int = 10) -> dict[str, Any]:
    rows = [row for row in records if bool(row.get("calibration", {}).get("available"))]
    count = len(rows)
    if count == 0:
        return {
            "count": 0,
            "accuracy": 0.0,
            "mean_confidence": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "avg_confidence_when_wrong": 0.0,
            "avg_confidence_when_correct": 0.0,
            "mean_predicted_conflict_gate": None,
            "mean_choice_conflict_gate": None,
            "bins": [],
        }

    confidences = [float(row["calibration"]["confidence"]) for row in rows]
    correct = [float(row["calibration"]["correct"]) for row in rows]
    briers = [float(row["calibration"]["brier"]) for row in rows]
    bins: list[dict[str, Any]] = []
    ece = 0.0
    bin_count = max(1, int(n_bins))
    for bin_index in range(bin_count):
        low = bin_index / bin_count
        high = (bin_index + 1) / bin_count
        if bin_index == bin_count - 1:
            indices = [idx for idx, conf in enumerate(confidences) if low <= conf <= high]
        else:
            indices = [idx for idx, conf in enumerate(confidences) if low <= conf < high]
        if not indices:
            bins.append({"low": low, "high": high, "count": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        bin_accuracy = sum(correct[idx] for idx in indices) / len(indices)
        bin_confidence = sum(confidences[idx] for idx in indices) / len(indices)
        ece += (len(indices) / count) * abs(bin_accuracy - bin_confidence)
        bins.append(
            {
                "low": low,
                "high": high,
                "count": len(indices),
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "gap": abs(bin_accuracy - bin_confidence),
            }
        )
    wrong_conf = [conf for conf, ok in zip(confidences, correct) if ok < 0.5]
    correct_conf = [conf for conf, ok in zip(confidences, correct) if ok >= 0.5]
    predicted_conflict_gates = [
        float(row["calibration"]["predicted_conflict_gate_mean"])
        for row in rows
        if row["calibration"].get("predicted_conflict_gate_mean") is not None
    ]
    choice_conflict_gates = [
        float(row["calibration"]["mean_choice_conflict_gate_mean"])
        for row in rows
        if row["calibration"].get("mean_choice_conflict_gate_mean") is not None
    ]
    return {
        "count": count,
        "accuracy": sum(correct) / count,
        "mean_confidence": sum(confidences) / count,
        "brier": sum(briers) / count,
        "ece": ece,
        "avg_confidence_when_wrong": sum(wrong_conf) / len(wrong_conf) if wrong_conf else 0.0,
        "avg_confidence_when_correct": sum(correct_conf) / len(correct_conf) if correct_conf else 0.0,
        "mean_predicted_conflict_gate": (
            sum(predicted_conflict_gates) / len(predicted_conflict_gates)
            if predicted_conflict_gates
            else None
        ),
        "mean_choice_conflict_gate": (
            sum(choice_conflict_gates) / len(choice_conflict_gates)
            if choice_conflict_gates
            else None
        ),
        "bins": bins,
    }


def _group_by_mode(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(str(record.get("mode", "unknown")), []).append(record)
    return out


def _group_by_field(records: Iterable[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(str(record.get(field, "unknown")), []).append(record)
    return out


def _compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "accuracy_delta": float(candidate["accuracy"]) - float(baseline["accuracy"]),
        "ece_delta": float(candidate["ece"]) - float(baseline["ece"]),
        "brier_delta": float(candidate["brier"]) - float(baseline["brier"]),
        "mean_confidence_delta": float(candidate["mean_confidence"]) - float(baseline["mean_confidence"]),
        "wrong_confidence_delta": (
            float(candidate["avg_confidence_when_wrong"])
            - float(baseline["avg_confidence_when_wrong"])
        ),
    }
    for metric in ("mean_predicted_conflict_gate", "mean_choice_conflict_gate"):
        baseline_value = baseline.get(metric)
        candidate_value = candidate.get(metric)
        comparison[f"{metric}_delta"] = (
            float(candidate_value) - float(baseline_value)
            if baseline_value is not None and candidate_value is not None
            else None
        )
    return comparison


def _filter_records_by_modes(
    records: Iterable[dict[str, Any]],
    included_modes: Iterable[str] | None,
) -> list[dict[str, Any]]:
    rows = list(records)
    if included_modes is None:
        return rows
    mode_set = set(included_modes)
    return [row for row in rows if str(row.get("mode", "unknown")) in mode_set]


def _resolve_gate_profile(gate_profile: str) -> tuple[str, tuple[str, ...] | None]:
    profile = str(gate_profile or "strict")
    if profile not in GATE_PROFILE_INCLUDED_MODES:
        choices = ", ".join(sorted(GATE_PROFILE_INCLUDED_MODES))
        raise ValueError(f"unknown gate profile {profile!r}; expected one of: {choices}")
    return profile, GATE_PROFILE_INCLUDED_MODES[profile]


def build_matched_metacognitive_gate(
    baseline_records: Iterable[dict[str, Any]],
    candidate_records: Iterable[dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
    n_bins: int = 10,
    tolerance: float = 1.0e-9,
    critical_modes: Iterable[str] | None = None,
    gate_profile: str = "strict",
) -> dict[str, Any]:
    source_baseline = [
        {**row, "calibration": row.get("calibration") or record_calibration(row)}
        for row in baseline_records
    ]
    source_candidate = [
        {**row, "calibration": row.get("calibration") or record_calibration(row)}
        for row in candidate_records
    ]
    resolved_profile, included_modes_tuple = _resolve_gate_profile(gate_profile)
    baseline = _filter_records_by_modes(source_baseline, included_modes_tuple)
    candidate = _filter_records_by_modes(source_candidate, included_modes_tuple)
    baseline_summary = calibration_summary(baseline, n_bins=n_bins)
    candidate_summary = calibration_summary(candidate, n_bins=n_bins)
    global_comparison = _compare_summaries(baseline_summary, candidate_summary)

    baseline_by_mode = _group_by_mode(baseline)
    candidate_by_mode = _group_by_mode(candidate)
    mode_comparisons: dict[str, dict[str, Any]] = {}
    for mode in sorted(set(baseline_by_mode) & set(candidate_by_mode)):
        mode_comparisons[mode] = _compare_summaries(
            calibration_summary(baseline_by_mode[mode], n_bins=n_bins),
            calibration_summary(candidate_by_mode[mode], n_bins=n_bins),
        )
    field_comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("category", "expected_unknown", "uncertainty_type"):
        baseline_by_field = _group_by_field(baseline, field)
        candidate_by_field = _group_by_field(candidate, field)
        field_comparisons[field] = {}
        for value in sorted(set(baseline_by_field) & set(candidate_by_field)):
            field_comparisons[field][value] = _compare_summaries(
                calibration_summary(baseline_by_field[value], n_bins=n_bins),
                calibration_summary(candidate_by_field[value], n_bins=n_bins),
            )

    failed_checks: list[str] = []
    passed_checks: list[str] = []
    if global_comparison["accuracy_delta"] < -tolerance:
        failed_checks.append("candidate_accuracy_dropped")
    else:
        passed_checks.append("candidate_accuracy_not_lower")
    if global_comparison["ece_delta"] > tolerance:
        failed_checks.append("candidate_ece_worse")
    else:
        passed_checks.append("candidate_ece_not_worse")
    if global_comparison["brier_delta"] > tolerance:
        failed_checks.append("candidate_brier_worse")
    else:
        passed_checks.append("candidate_brier_not_worse")
    if (
        global_comparison["ece_delta"] < -tolerance
        or global_comparison["brier_delta"] < -tolerance
    ):
        passed_checks.append("candidate_calibration_improved")
    else:
        failed_checks.append("candidate_no_calibration_gain")
    if critical_modes is not None:
        mode_gate_names = list(critical_modes)
    elif included_modes_tuple is not None:
        mode_gate_names = list(included_modes_tuple)
    else:
        mode_gate_names = list(DEFAULT_CRITICAL_MODES)

    if not baseline or not candidate:
        failed_checks.append("profile_has_no_matched_records")
    checked_critical_modes = 0
    critical_mode_gains = 0
    for mode in mode_gate_names:
        comparison = mode_comparisons.get(mode)
        if comparison is None:
            continue
        checked_critical_modes += 1
        if comparison["accuracy_delta"] < -tolerance:
            failed_checks.append(f"critical_mode_{mode}_accuracy_dropped")
        if comparison["ece_delta"] > tolerance:
            failed_checks.append(f"critical_mode_{mode}_ece_worse")
        if comparison["brier_delta"] > tolerance:
            failed_checks.append(f"critical_mode_{mode}_brier_worse")
        if (
            comparison["accuracy_delta"] >= -tolerance
            and comparison["ece_delta"] <= tolerance
            and comparison["brier_delta"] <= tolerance
        ):
            passed_checks.append(f"critical_mode_{mode}_not_worse")
        if comparison["ece_delta"] < -tolerance or comparison["brier_delta"] < -tolerance:
            critical_mode_gains += 1
    if checked_critical_modes > 0 and critical_mode_gains > 0:
        passed_checks.append("critical_qtrm_mode_calibration_improved")
    elif checked_critical_modes > 0:
        failed_checks.append("critical_qtrm_mode_no_calibration_gain")

    return {
        "gate_type": "metacognitive_calibration",
        "gate_profile": resolved_profile,
        "claim": (
            "Random-noise warm-up should reduce overconfidence/calibration error "
            "without lowering matched raw-answer accuracy."
        ),
        "status": "rejected" if failed_checks else "accepted",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "record_count": len(baseline) + len(candidate),
        "source_record_count": len(source_baseline) + len(source_candidate),
        "profile_record_count": len(baseline) + len(candidate),
        "included_modes": list(included_modes_tuple or []),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "global_comparison": global_comparison,
        "mode_comparisons": mode_comparisons,
        "critical_modes": mode_gate_names,
        "field_comparisons": field_comparisons,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "notes": [
            "Confidence is softmax over forced-choice logprob scores.",
            "This gate uses choice-score calibration; it is not a full generative calibration proof.",
            "Gate profiles filter records before global metrics; strict keeps all modes.",
            "Promotion still needs donor-only, fused, low-donor/QTRM-only, and core-off comparisons.",
        ],
    }


def write_gate(gate: dict[str, Any], *, markdown_out: str, json_out: str) -> None:
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Metacognitive Calibration Gate",
        "",
        f"Status: `{gate['status']}`",
        "",
        f"Baseline: `{gate['baseline_label']}`",
        f"Candidate: `{gate['candidate_label']}`",
        f"Gate profile: `{gate.get('gate_profile', 'strict')}`",
        f"Profile records: `{gate.get('profile_record_count', gate.get('record_count', 0))}` / source `{gate.get('source_record_count', gate.get('record_count', 0))}`",
        f"Included modes: `{', '.join(gate.get('included_modes', [])) or 'all'}`",
        "",
        "## Global Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    comparison = gate["global_comparison"]
    global_metrics = [
        "accuracy",
        "ece",
        "brier",
        "mean_confidence",
        "avg_confidence_when_wrong",
    ]
    if comparison["baseline"].get("mean_predicted_conflict_gate") is not None or comparison[
        "candidate"
    ].get("mean_predicted_conflict_gate") is not None:
        global_metrics.extend(["mean_predicted_conflict_gate", "mean_choice_conflict_gate"])
    for metric in global_metrics:
        baseline_value = float(comparison["baseline"].get(metric, 0.0))
        candidate_value = float(comparison["candidate"].get(metric, 0.0))
        delta_key = {
            "accuracy": "accuracy_delta",
            "ece": "ece_delta",
            "brier": "brier_delta",
            "mean_confidence": "mean_confidence_delta",
            "avg_confidence_when_wrong": "wrong_confidence_delta",
            "mean_predicted_conflict_gate": "mean_predicted_conflict_gate_delta",
            "mean_choice_conflict_gate": "mean_choice_conflict_gate_delta",
        }[metric]
        delta = comparison[delta_key]
        lines.append(
            f"| `{metric}` | {baseline_value:.6f} | {candidate_value:.6f} | "
            f"{float(delta):+.6f} |"
            if delta is not None
            else f"| `{metric}` | {baseline_value:.6f} | {candidate_value:.6f} | `n/a` |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"Passed: `{', '.join(gate['passed_checks'])}`",
            "",
            f"Failed: `{', '.join(gate['failed_checks'])}`",
            "",
            "## Mode Comparisons",
            "",
            "| Mode | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, mode_comparison in gate["mode_comparisons"].items():
        lines.append(
            "| `{}` | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} |".format(
                mode,
                float(mode_comparison["accuracy_delta"]),
                float(mode_comparison["ece_delta"]),
                float(mode_comparison["brier_delta"]),
                float(mode_comparison["wrong_confidence_delta"]),
            )
        )
    lines.extend(["", "Critical modes: `{}`".format(", ".join(gate.get("critical_modes", [])))])
    lines.extend(
        [
            "",
            "## Category Comparisons",
            "",
            "| Category | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, comparison in gate.get("field_comparisons", {}).get("category", {}).items():
        lines.append(
            "| `{}` | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} |".format(
                category,
                float(comparison["accuracy_delta"]),
                float(comparison["ece_delta"]),
                float(comparison["brier_delta"]),
                float(comparison["wrong_confidence_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Expected Unknown Comparisons",
            "",
            "| Expected Unknown | Acc Delta | ECE Delta | Brier Delta | Wrong-Conf Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for value, comparison in gate.get("field_comparisons", {}).get("expected_unknown", {}).items():
        lines.append(
            "| `{}` | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} |".format(
                value,
                float(comparison["accuracy_delta"]),
                float(comparison["ece_delta"]),
                float(comparison["brier_delta"]),
                float(comparison["wrong_confidence_delta"]),
            )
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in gate.get("notes", []))
    markdown_path = Path(markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metacognitive_gate_report(
    *,
    baseline_jsonl: str,
    candidate_jsonl: str,
    baseline_label: str,
    candidate_label: str,
    markdown_out: str,
    json_out: str,
    n_bins: int,
    gate_profile: str = "strict",
    critical_modes: Iterable[str] | None = None,
) -> dict[str, Any]:
    baseline = load_records(baseline_jsonl)
    candidate = load_records(candidate_jsonl)
    gate = build_matched_metacognitive_gate(
        baseline,
        candidate,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        n_bins=n_bins,
        gate_profile=gate_profile,
        critical_modes=critical_modes,
    )
    write_gate(gate, markdown_out=markdown_out, json_out=json_out)
    return gate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a matched metacognitive calibration gate from raw forced-choice eval JSONL."
    )
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--markdown-out", default="docs/wiki/decisions/metacognitive-calibration-gate.md")
    parser.add_argument("--json-out", default="docs/wiki/decisions/metacognitive-calibration-gate-summary.json")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--gate-profile",
        default="strict",
        choices=sorted(GATE_PROFILE_INCLUDED_MODES),
        help=(
            "Record-filtering gate profile. strict keeps all modes; qtrm_core "
            "separates the core metacognition claim; fused isolates donor/QTRM fusion."
        ),
    )
    parser.add_argument(
        "--critical-mode",
        action="append",
        default=None,
        help="Override critical modes checked inside the selected profile. Can be repeated.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gate = write_metacognitive_gate_report(
        baseline_jsonl=args.baseline_jsonl,
        candidate_jsonl=args.candidate_jsonl,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        markdown_out=args.markdown_out,
        json_out=args.json_out,
        n_bins=args.n_bins,
        gate_profile=args.gate_profile,
        critical_modes=args.critical_mode,
    )
    print(f"status={gate['status']}")
    print(f"wrote {args.markdown_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
