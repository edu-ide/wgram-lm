from __future__ import annotations

from typing import Any, Mapping


def evaluate_causal_loop_gate(
    metrics: Mapping[str, float],
    *,
    min_gain: float = 0.0,
    min_drop: float = 0.0,
) -> dict[str, Any]:
    """Accept QTRM loop claims only when gain and ablations are causal.

    Required comparison:
    - QTRM plus harness must beat donor-only plus harness and scripted harness.
    - Turning off latent core, world model, or verifier must drop score enough.
    """

    qtrm = _metric(metrics, "qtrm_harness")
    donor = _metric(metrics, "donor_harness")
    scripted = _metric(metrics, "scripted_harness")
    checks: list[str] = []

    gain_over_donor = qtrm - donor
    gain_over_scripted = qtrm - scripted
    if gain_over_donor < float(min_gain):
        checks.append("qtrm_does_not_beat_donor_harness")
    if gain_over_scripted < float(min_gain):
        checks.append("qtrm_does_not_beat_scripted_harness")

    ablation_specs = (
        ("qtrm_latent_core_off", "latent_core_not_causal"),
        ("qtrm_world_model_off", "world_model_not_causal"),
        ("qtrm_verifier_off", "verifier_not_causal"),
    )
    causal_drops: dict[str, float] = {}
    for key, failure in ablation_specs:
        if key not in metrics:
            checks.append(f"{key}_missing")
            continue
        drop = qtrm - float(metrics[key])
        causal_drops[key] = drop
        if drop < float(min_drop):
            checks.append(failure)

    return {
        "status": "accepted" if not checks else "rejected",
        "baseline": "qtrm_harness",
        "failed_checks": tuple(checks),
        "gain_over_donor_harness": gain_over_donor,
        "gain_over_scripted_harness": gain_over_scripted,
        "causal_drops": causal_drops,
        "min_gain": float(min_gain),
        "min_drop": float(min_drop),
    }


def _metric(metrics: Mapping[str, float], key: str) -> float:
    if key not in metrics:
        raise ValueError(f"missing required metric: {key}")
    return float(metrics[key])
