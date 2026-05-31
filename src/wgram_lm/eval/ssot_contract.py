from __future__ import annotations

from typing import Any


CANONICAL_EVIDENCE_INJECTION = "ssot"
CANONICAL_ANSWER_CHANNEL = "greedy"
CANONICAL_CORE_WORLD_MODEL_ENABLED = False
CANONICAL_CORE_WORLD_MODEL_WEIGHT = 0.0


def validate_canonical_ssot_args(args: Any) -> None:
    """Fail unless an argparse-like object uses the canonical answer path."""
    if not bool(getattr(args, "require_canonical_ssot", False)):
        return
    evidence_injection = getattr(args, "evidence_injection", None)
    answer_channel = getattr(args, "answer_channel", None)
    if evidence_injection != CANONICAL_EVIDENCE_INJECTION:
        raise ValueError(
            "--require-canonical-ssot requires --evidence-injection "
            f"{CANONICAL_EVIDENCE_INJECTION}; got {evidence_injection!r}"
        )
    if answer_channel != CANONICAL_ANSWER_CHANNEL:
        raise ValueError(
            "--require-canonical-ssot requires --answer-channel "
            f"{CANONICAL_ANSWER_CHANNEL}; got {answer_channel!r}"
        )


def validate_canonical_model_config(cfg: Any) -> None:
    """Fail unless a config matches the canonical single-trace TRM path."""
    model = getattr(cfg, "model", cfg)
    train = getattr(cfg, "train", cfg)
    if bool(getattr(model, "core_world_model_enabled", False)):
        raise ValueError(
            "canonical QTRM keeps LeWorldModel out of the answer path: "
            "model.core_world_model_enabled must be false"
        )
    weight = float(getattr(train, "loss_core_world_model_weight", 0.0))
    if weight != CANONICAL_CORE_WORLD_MODEL_WEIGHT:
        raise ValueError(
            "canonical QTRM keeps LeWorldModel loss disabled: "
            "train.loss_core_world_model_weight must be 0.0"
        )
