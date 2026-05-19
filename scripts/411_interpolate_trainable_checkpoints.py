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
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    alpha = float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

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
            if a[key].is_floating_point() or b[key].is_floating_point():
                merged[key] = (1.0 - alpha) * a[key].float() + alpha * b[key].float()
                merged[key] = merged[key].to(dtype=a[key].dtype)
            else:
                merged[key] = b[key] if alpha >= 0.5 else a[key]
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
                "num_tensors": len(merged),
            },
        },
        str(out),
    )
    print(f"saved {out} tensors={len(merged)} alpha={alpha}")


if __name__ == "__main__":
    main()
