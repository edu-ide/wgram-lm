#!/usr/bin/env python3
"""Average two QTRM-native checkpoints for model-soup triage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a dict: {path}")
    state_key = "model_state" if "model_state" in checkpoint else "model" if "model" in checkpoint else ""
    if not state_key:
        raise ValueError(f"checkpoint missing model_state/model: {path}")
    if not isinstance(checkpoint[state_key], dict):
        raise ValueError(f"checkpoint {state_key} is not a dict: {path}")
    checkpoint["_average_state_key"] = state_key
    return checkpoint


def average_model_states(
    base_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if set(base_state) != set(candidate_state):
        missing = sorted(set(base_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(base_state))
        raise ValueError(
            "checkpoint state keys differ: "
            f"missing_in_candidate={missing[:8]} extra_in_candidate={extra[:8]}"
        )

    averaged: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        candidate_value = candidate_state[key]
        if tuple(base_value.shape) != tuple(candidate_value.shape):
            raise ValueError(
                f"checkpoint tensor shape differs for {key}: "
                f"{tuple(base_value.shape)} != {tuple(candidate_value.shape)}"
            )
        if torch.is_floating_point(base_value):
            averaged[key] = base_value.detach().clone().mul(1.0 - alpha).add(
                candidate_value.detach(), alpha=alpha
            )
        else:
            if not torch.equal(base_value, candidate_value):
                raise ValueError(f"non-floating tensor differs for {key}")
            averaged[key] = base_value.detach().clone()
    return averaged


def build_averaged_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.base_checkpoint)
    candidate_path = Path(args.candidate_checkpoint)
    base = load_checkpoint(base_path)
    candidate = load_checkpoint(candidate_path)
    state_key = str(base["_average_state_key"])
    if state_key != str(candidate["_average_state_key"]):
        raise ValueError(
            "checkpoint state keys differ: "
            f"{state_key} != {candidate['_average_state_key']}"
        )

    if tuple(base.get("chars", ())) != tuple(candidate.get("chars", ())):
        raise ValueError("checkpoint tokenizer chars differ")

    averaged = dict(base)
    averaged.pop("_average_state_key", None)
    averaged[state_key] = average_model_states(
        base[state_key],
        candidate[state_key],
        alpha=float(args.alpha),
    )
    averaged["checkpoint_average"] = {
        "method": "linear_model_soup",
        "formula": "(1-alpha) * base + alpha * candidate",
        "alpha": float(args.alpha),
        "base_checkpoint": str(base_path),
        "candidate_checkpoint": str(candidate_path),
    }
    averaged["args"] = dict(base.get("args", {}))
    averaged["args"]["averaged_from"] = averaged["checkpoint_average"]
    return averaged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average two compatible QTRM-native checkpoints."
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint = build_averaged_checkpoint(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)

    summary = {
        "output_checkpoint": str(out),
        **checkpoint["checkpoint_average"],
    }
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
