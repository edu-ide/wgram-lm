#!/usr/bin/env python3
"""Interpolate two trainable QTRM checkpoint state dicts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _state(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state dict or contain a model state dict")
    return {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="checkpoint used at alpha=0")
    parser.add_argument("--b", required=True, help="checkpoint used at alpha=1")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--qwen-alpha",
        type=float,
        default=None,
        help="optional alpha for qwen.* tensors; defaults to --alpha",
    )
    parser.add_argument(
        "--core-alpha",
        type=float,
        default=None,
        help="optional alpha for non-qwen tensors; defaults to --alpha",
    )
    parser.add_argument(
        "--qwen-attn-alpha",
        type=float,
        default=None,
        help="optional alpha for qwen self-attention tensors",
    )
    parser.add_argument(
        "--qwen-mlp-alpha",
        type=float,
        default=None,
        help="optional alpha for qwen MLP tensors",
    )
    parser.add_argument(
        "--qwen-norm-alpha",
        type=float,
        default=None,
        help="optional alpha for qwen layer norm tensors",
    )
    parser.add_argument(
        "--core-state-alpha",
        type=float,
        default=None,
        help="optional alpha for recurrent state/norm/step-conditioning tensors",
    )
    parser.add_argument(
        "--core-adapter-alpha",
        type=float,
        default=None,
        help="optional alpha for core input/output/delta adapter tensors",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    alpha = float(args.alpha)
    qwen_alpha = alpha if args.qwen_alpha is None else float(args.qwen_alpha)
    core_alpha = alpha if args.core_alpha is None else float(args.core_alpha)
    group_alphas = {
        "alpha": alpha,
        "qwen_alpha": qwen_alpha,
        "core_alpha": core_alpha,
        "qwen_attn_alpha": qwen_alpha
        if args.qwen_attn_alpha is None
        else float(args.qwen_attn_alpha),
        "qwen_mlp_alpha": qwen_alpha
        if args.qwen_mlp_alpha is None
        else float(args.qwen_mlp_alpha),
        "qwen_norm_alpha": qwen_alpha
        if args.qwen_norm_alpha is None
        else float(args.qwen_norm_alpha),
        "core_state_alpha": core_alpha
        if args.core_state_alpha is None
        else float(args.core_state_alpha),
        "core_adapter_alpha": core_alpha
        if args.core_adapter_alpha is None
        else float(args.core_adapter_alpha),
    }
    for name, value in group_alphas.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    a_raw = torch.load(str(args.a), map_location="cpu")
    b_raw = torch.load(str(args.b), map_location="cpu")
    a = _state(a_raw)
    b = _state(b_raw)
    keys = sorted(set(a) | set(b))
    merged: dict[str, torch.Tensor] = {}
    incompatible: list[str] = []
    for key in keys:
        if key in a and key in b:
            if a[key].shape != b[key].shape:
                incompatible.append(key)
                continue
            key_alpha = _alpha_for_key(key, group_alphas)
            if a[key].is_floating_point() or b[key].is_floating_point():
                merged[key] = (1.0 - key_alpha) * a[key].float() + key_alpha * b[key].float()
                merged[key] = merged[key].to(dtype=a[key].dtype)
            else:
                merged[key] = b[key] if key_alpha >= 0.5 else a[key]
        elif key in a:
            merged[key] = a[key]
        else:
            merged[key] = b[key]

    if incompatible:
        raise RuntimeError(f"incompatible tensor shapes: {incompatible[:8]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": merged,
            "interpolation": {
                "a": str(args.a),
                "b": str(args.b),
                "alpha": alpha,
                "qwen_alpha": qwen_alpha,
                "core_alpha": core_alpha,
                **group_alphas,
                "num_tensors": len(merged),
            },
        },
        str(out),
    )
    print(
        f"saved {out} tensors={len(merged)} "
        f"alpha={alpha} qwen_alpha={qwen_alpha} core_alpha={core_alpha}"
    )


def _alpha_for_key(key: str, alphas: dict[str, float]) -> float:
    if key.startswith("qwen."):
        if ".self_attn." in key:
            return alphas["qwen_attn_alpha"]
        if ".mlp." in key:
            return alphas["qwen_mlp_alpha"]
        if key.endswith("layernorm.weight") or "norm.weight" in key:
            return alphas["qwen_norm_alpha"]
        return alphas["qwen_alpha"]
    if key.startswith("core_delta_adapter.") or key in {"core_in_norm.weight", "core_out_norm.weight"}:
        return alphas["core_adapter_alpha"]
    if key.startswith("core."):
        return alphas["core_state_alpha"]
    return alphas["core_alpha"]


if __name__ == "__main__":
    main()
