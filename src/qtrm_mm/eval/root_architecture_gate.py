from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.eval.memory_retrieval import summarize_records
from qtrm_mm.eval.residual_adapter_proof import load_eval_records

DEFAULT_BASELINE_MODE = "qtrm_residual_with_evidence"
DEFAULT_CRITICAL_MODES = [
    "qtrm_workspace_off_with_evidence",
    "qtrm_core_off_with_evidence",
    "qtrm_workspace_memory_off_with_evidence",
    "qtrm_core_context_off_with_evidence",
    "qtrm_core_to_text_off_with_evidence",
    "qtrm_evidence_bottleneck_off_with_evidence",
    "qtrm_evidence_span_reader_off_with_evidence",
    "qtrm_answer_residual_governor_off_with_evidence",
]
DEFAULT_COMPARISON_MODES = [
    "donor_only_with_evidence",
]


def _records_by_mode(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("mode", "unknown")), []).append(record)
    return grouped


def _overall(records: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_records(records)["overall"]


def _records_by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("id")): record for record in records}


def _completion_identity(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = _records_by_id(baseline_records)
    candidate_by_id = _records_by_id(candidate_records)
    paired_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    same_count = sum(
        str(baseline_by_id[case_id].get("completion", ""))
        == str(candidate_by_id[case_id].get("completion", ""))
        for case_id in paired_ids
    )
    paired_count = len(paired_ids)
    return {
        "paired_completion_count": paired_count,
        "same_completion_count": same_count,
        "same_completion_rate": same_count / paired_count if paired_count else 0.0,
    }


def _mode_check(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    identity_same_threshold: float,
) -> dict[str, Any]:
    baseline = _overall(baseline_records)
    candidate = _overall(candidate_records)
    identity = _completion_identity(baseline_records, candidate_records)
    hit_drop = int(baseline["hits"]) - int(candidate["hits"])
    accuracy_drop = float(baseline["accuracy"]) - float(candidate["accuracy"])
    same_completion_rate = float(identity["same_completion_rate"])
    return {
        "count": int(candidate["count"]),
        "hits": int(candidate["hits"]),
        "accuracy": float(candidate["accuracy"]),
        "hit_drop": hit_drop,
        "accuracy_drop": accuracy_drop,
        **identity,
        "has_causal_drop": hit_drop > 0 or accuracy_drop > 0.0,
        "matches_baseline_identity": (
            int(identity["paired_completion_count"]) > 0
            and same_completion_rate >= identity_same_threshold
        ),
    }


def _comparison_check(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = _overall(baseline_records)
    candidate = _overall(candidate_records)
    hit_advantage = int(baseline["hits"]) - int(candidate["hits"])
    accuracy_advantage = float(baseline["accuracy"]) - float(candidate["accuracy"])
    return {
        "count": int(candidate["count"]),
        "hits": int(candidate["hits"]),
        "accuracy": float(candidate["accuracy"]),
        "hit_advantage": hit_advantage,
        "accuracy_advantage": accuracy_advantage,
        "baseline_beats_candidate": hit_advantage > 0 or accuracy_advantage > 0.0,
    }


def build_root_architecture_gate(
    records: Iterable[dict[str, Any]],
    *,
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    critical_modes: Iterable[str] = DEFAULT_CRITICAL_MODES,
    comparison_modes: Iterable[str] = DEFAULT_COMPARISON_MODES,
    min_baseline_accuracy: float = 0.01,
    identity_same_threshold: float = 0.90,
    require_donor_advantage: bool = False,
    require_no_critical_ablation_improvement: bool = False,
) -> dict[str, Any]:
    """Classify whether the current QTRM root causal claim is supported.

    This gate is deliberately stricter than a normal eval summary. It rejects a
    checkpoint if a successful baseline keeps the same answers when critical
    workspace/core/evidence paths are disabled, because that means the claimed
    latent path is not causally necessary on the tested cases.
    """
    record_list = list(records)
    by_mode = _records_by_mode(record_list)
    baseline_records = by_mode.get(baseline_mode, [])
    baseline = _overall(baseline_records)
    critical_mode_list = list(critical_modes)
    comparison_mode_list = list(comparison_modes)
    missing_modes = [mode for mode in critical_mode_list if mode not in by_mode]
    missing_comparison_modes = [
        mode for mode in comparison_mode_list if mode not in by_mode
    ]
    mode_checks = {
        mode: _mode_check(
            baseline_records,
            by_mode.get(mode, []),
            identity_same_threshold=identity_same_threshold,
        )
        for mode in critical_mode_list
        if mode in by_mode
    }
    comparison_checks = {
        mode: _comparison_check(baseline_records, by_mode.get(mode, []))
        for mode in comparison_mode_list
        if mode in by_mode
    }

    failed_checks: list[str] = []
    passed_checks: list[str] = []

    if int(baseline["count"]) == 0:
        failed_checks.append("baseline_missing")
    if int(baseline["hits"]) == 0:
        failed_checks.append("baseline_has_no_successes")
    if float(baseline["accuracy"]) < min_baseline_accuracy:
        failed_checks.append("baseline_below_min_accuracy")

    causal_modes = [
        mode for mode, check in mode_checks.items() if bool(check["has_causal_drop"])
    ]
    identity_match_modes = [
        mode
        for mode, check in mode_checks.items()
        if not bool(check["has_causal_drop"]) and bool(check["matches_baseline_identity"])
    ]
    if causal_modes:
        passed_checks.append("critical_causal_drop_present")
    else:
        failed_checks.append("no_critical_causal_drop")
    if identity_match_modes:
        failed_checks.append("critical_ablations_match_baseline_identity")
    if missing_modes:
        failed_checks.append("critical_modes_missing")
    if require_donor_advantage and missing_comparison_modes:
        failed_checks.append("comparison_modes_missing")
    if require_donor_advantage:
        weak_comparisons = [
            mode
            for mode, check in comparison_checks.items()
            if not bool(check["baseline_beats_candidate"])
        ]
        if weak_comparisons:
            failed_checks.append("baseline_does_not_beat_comparison")
    else:
        weak_comparisons = []
    improving_critical_modes = [
        mode
        for mode, check in mode_checks.items()
        if int(check["hit_drop"]) < 0 or float(check["accuracy_drop"]) < 0.0
    ]
    if require_no_critical_ablation_improvement and improving_critical_modes:
        failed_checks.append("critical_ablation_beats_baseline")

    causal_gate_status = "rejected"
    if "baseline_missing" in failed_checks:
        causal_gate_status = "inconclusive"
    elif (
        "baseline_has_no_successes" not in failed_checks
        and "baseline_below_min_accuracy" not in failed_checks
        and causal_modes
    ):
        causal_gate_status = "accepted"
    elif not mode_checks:
        causal_gate_status = "inconclusive"

    if "baseline_missing" in failed_checks:
        status = "inconclusive"
    elif "baseline_has_no_successes" in failed_checks:
        status = "rejected"
    elif (
        causal_modes
        and "baseline_below_min_accuracy" not in failed_checks
        and "baseline_does_not_beat_comparison" not in failed_checks
        and "critical_ablation_beats_baseline" not in failed_checks
        and "comparison_modes_missing" not in failed_checks
    ):
        status = "accepted"
    elif not mode_checks:
        status = "inconclusive"
    else:
        status = "rejected"

    recommendation = (
        "Do not spend another local loss/threshold on this checkpoint. Move the "
        "answer signal onto a forced workspace/evidence bottleneck, then rerun "
        "workspace/core/memory-off ablations."
    )
    if status == "accepted":
        recommendation = (
            "Keep this architecture candidate under test. It has at least one "
            "causal component drop and passes the enabled promotion checks."
        )
    elif causal_gate_status == "accepted":
        recommendation = (
            "Keep the causal component signal as a diagnostic result, but do "
            "not promote the full architecture until it beats donor-only and "
            "critical ablations no longer outperform the full path."
        )
    elif status == "inconclusive":
        recommendation = (
            "Run the missing baseline and critical ablation modes before making an "
            "architecture decision."
        )

    return {
        "claim": (
            "QTRM canonical answer path should improve over donor-only and lose "
            "when critical causal components are disabled."
        ),
        "status": status,
        "causal_gate_status": causal_gate_status,
        "strict_promotion_required": bool(
            require_donor_advantage or require_no_critical_ablation_improvement
        ),
        "baseline_mode": baseline_mode,
        "baseline": baseline,
        "critical_modes": critical_mode_list,
        "comparison_modes": comparison_mode_list,
        "mode_checks": mode_checks,
        "comparison_checks": comparison_checks,
        "causal_modes": causal_modes,
        "identity_match_modes": identity_match_modes,
        "weak_comparison_modes": weak_comparisons,
        "improving_critical_modes": improving_critical_modes,
        "missing_modes": missing_modes,
        "missing_comparison_modes": missing_comparison_modes,
        "failed_checks": failed_checks,
        "passed_checks": passed_checks,
        "recommendation": recommendation,
    }


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_eval_records(path))
    return records


def render_markdown(gate: dict[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    baseline = gate.get("baseline", {})
    lines = [
        "# Root Architecture Causality Gate",
        "",
        "## Verdict",
        "",
        f"Status: `{status}`",
        "",
        f"Causal gate status: `{gate.get('causal_gate_status', status)}`",
        "",
        f"Strict promotion required: `{bool(gate.get('strict_promotion_required', False))}`",
        "",
        f"Claim: {gate.get('claim', '')}",
        "",
        f"Recommendation: {gate.get('recommendation', '')}",
        "",
        "## Baseline",
        "",
        "| Mode | Hits | Accuracy |",
        "| --- | ---: | ---: |",
        "| {mode} | {hits}/{count} | {accuracy:.3f} |".format(
            mode=gate.get("baseline_mode", DEFAULT_BASELINE_MODE),
            hits=int(baseline.get("hits", 0)),
            count=int(baseline.get("count", 0)),
            accuracy=float(baseline.get("accuracy", 0.0)),
        ),
        "",
        "## Comparison Checks",
        "",
        "| Mode | Hits | Hit advantage | Accuracy advantage | Gate |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for mode, check in gate.get("comparison_checks", {}).items():
        gate_text = "baseline-beats" if check.get("baseline_beats_candidate") else "not-beaten"
        lines.append(
            "| {mode} | {hits}/{count} | {hit_advantage:+d} | {accuracy_advantage:+.3f} | {gate} |".format(
                mode=mode,
                hits=int(check.get("hits", 0)),
                count=int(check.get("count", 0)),
                hit_advantage=int(check.get("hit_advantage", 0)),
                accuracy_advantage=float(check.get("accuracy_advantage", 0.0)),
                gate=gate_text,
            )
        )

    lines.extend(
        [
            "",
            "## Critical Mode Checks",
            "",
            "| Mode | Hits | Hit drop | Accuracy drop | Same completions | Gate |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for mode, check in gate.get("mode_checks", {}).items():
        gate_text = "causal-drop" if check.get("has_causal_drop") else "no-drop"
        if check.get("matches_baseline_identity"):
            gate_text += ", same-output"
        lines.append(
            "| {mode} | {hits}/{count} | {hit_drop:+d} | {accuracy_drop:+.3f} | {same}/{paired} ({rate:.3f}) | {gate} |".format(
                mode=mode,
                hits=int(check.get("hits", 0)),
                count=int(check.get("count", 0)),
                hit_drop=int(check.get("hit_drop", 0)),
                accuracy_drop=float(check.get("accuracy_drop", 0.0)),
                same=int(check.get("same_completion_count", 0)),
                paired=int(check.get("paired_completion_count", 0)),
                rate=float(check.get("same_completion_rate", 0.0)),
                gate=gate_text,
            )
        )

    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Passed: `{', '.join(gate.get('passed_checks', [])) or 'none'}`",
            f"- Failed: `{', '.join(gate.get('failed_checks', [])) or 'none'}`",
            f"- Missing modes: `{', '.join(gate.get('missing_modes', [])) or 'none'}`",
            f"- Missing comparison modes: `{', '.join(gate.get('missing_comparison_modes', [])) or 'none'}`",
            f"- Weak comparison modes: `{', '.join(gate.get('weak_comparison_modes', [])) or 'none'}`",
            f"- Improving critical modes: `{', '.join(gate.get('improving_critical_modes', [])) or 'none'}`",
            "",
            "## Interpretation Rule",
            "",
            "- `causal_gate_status=accepted` means at least one critical ablation worsened a successful baseline.",
            "- `status=accepted` means the causal gate passed and any enabled strict promotion checks also passed.",
            "- `rejected` means the checkpoint cannot support the current root architecture claim on this eval.",
            "- `inconclusive` means the required baseline or ablation rows are missing.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_gate(
    gate: dict[str, Any],
    *,
    markdown_out: str | Path,
    json_out: str | Path,
) -> None:
    markdown_path = Path(markdown_out)
    json_path = Path(json_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(gate), encoding="utf-8")
    json_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
