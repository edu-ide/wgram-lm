from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def _as_floats(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def summarize_preference_scores(
    *,
    chosen_logps: Sequence[float],
    rejected_logps: Sequence[float],
    sample_weights: Sequence[float] | None = None,
    target_margin: float = 0.0,
) -> dict[str, float | int]:
    if len(chosen_logps) != len(rejected_logps):
        raise ValueError("chosen and rejected log-prob lists must have the same length")
    chosen = _as_floats(chosen_logps)
    rejected = _as_floats(rejected_logps)
    weights = _as_floats(sample_weights) if sample_weights is not None else [1.0] * len(chosen)
    if len(weights) != len(chosen):
        raise ValueError("sample_weights must match chosen/rejected length")

    margins = [c - r for c, r in zip(chosen, rejected)]
    count = len(margins)
    if count == 0:
        return {
            "count": 0,
            "weighted_count": 0.0,
            "preference_accuracy": 0.0,
            "weighted_preference_accuracy": 0.0,
            "margin_pass_rate": 0.0,
            "weighted_margin_pass_rate": 0.0,
            "margin_mean": 0.0,
            "weighted_margin_mean": 0.0,
            "margin_min": 0.0,
            "margin_max": 0.0,
        }

    target = float(target_margin)
    wins = [1.0 if margin > 0.0 else 0.0 for margin in margins]
    passes = [1.0 if margin >= target else 0.0 for margin in margins]
    clipped_weights = [max(0.0, weight) for weight in weights]
    weight_total = sum(clipped_weights)

    def weighted_mean(values: list[float]) -> float:
        if weight_total <= 0.0:
            return 0.0
        return sum(value * weight for value, weight in zip(values, clipped_weights)) / weight_total

    return {
        "count": count,
        "weighted_count": float(weight_total),
        "preference_accuracy": sum(wins) / count,
        "weighted_preference_accuracy": weighted_mean(wins),
        "margin_pass_rate": sum(passes) / count,
        "weighted_margin_pass_rate": weighted_mean(passes),
        "margin_mean": sum(margins) / count,
        "weighted_margin_mean": weighted_mean(margins),
        "margin_min": min(margins),
        "margin_max": max(margins),
    }


def summarize_preference_records(
    records: Iterable[Mapping[str, object]],
    *,
    target_margin: float = 0.0,
) -> dict[str, float | int]:
    chosen: list[float] = []
    rejected: list[float] = []
    weights: list[float] = []
    for record in records:
        if "summary" in record:
            continue
        if "chosen_logp" not in record or "rejected_logp" not in record:
            continue
        chosen.append(float(record["chosen_logp"]))
        rejected.append(float(record["rejected_logp"]))
        weights.append(float(record.get("sample_weight", 1.0)))
    return summarize_preference_scores(
        chosen_logps=chosen,
        rejected_logps=rejected,
        sample_weights=weights,
        target_margin=target_margin,
    )
