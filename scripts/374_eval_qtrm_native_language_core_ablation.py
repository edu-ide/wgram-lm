#!/usr/bin/env python3
"""Evaluate QTRM-native language checkpoint core causality ablations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


def load_language_module():
    path = Path(__file__).with_name("354_train_qtrm_native_language_bootstrap.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load language bootstrap module: {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def restore_checkpoint_args(module, checkpoint: dict[str, Any], checkpoint_path: str) -> argparse.Namespace:
    args = module.build_arg_parser().parse_args([])
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, dict):
        for name, value in checkpoint_args.items():
            if hasattr(args, name):
                setattr(args, name, value)
    args.init_checkpoint = str(checkpoint_path)
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="local_eval/qtrm_native_language_core_ablation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-fraction", type=float, default=-1.0)
    parser.add_argument("--eval-depth-sweep", default="")
    parser.add_argument("--eval-think-steps", type=int, default=-1)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    module = load_language_module()
    device = torch.device(str(args.device))
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    train_args = restore_checkpoint_args(module, checkpoint, str(args.checkpoint))
    train_args.device = str(args.device)
    train_args.batch_size = int(args.batch_size)
    if float(args.eval_fraction) > 0.0:
        train_args.eval_fraction = float(args.eval_fraction)
    if str(args.eval_depth_sweep):
        train_args.eval_depth_sweep = str(args.eval_depth_sweep)
    if int(args.eval_think_steps) >= 0:
        train_args.eval_think_steps = int(args.eval_think_steps)

    tokenizer = module.tokenizer_from_payload(checkpoint.get("tokenizer", {}), train_args)
    stage_texts = module.build_stage_texts(train_args)
    _train_windows, eval_windows = module.split_eval_windows(
        tokenizer,
        stage_texts["teacher"],
        seq_len=int(train_args.seq_len),
        eval_fraction=float(train_args.eval_fraction),
    )
    model = module._text_probe.build_model(train_args, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    think_steps = int(train_args.eval_think_steps)
    full_loss = module._text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(train_args.batch_size),
        device=device,
        think_steps=think_steps,
    )
    think0_loss = module._text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(train_args.batch_size),
        device=device,
        think_steps=0,
    )
    off_loss = module._text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(train_args.batch_size),
        device=device,
        think_steps=think_steps,
        thinking_block_off=True,
    )
    state_reset_loss = module._text_probe.eval_loss(
        model,
        eval_windows,
        batch_size=int(train_args.batch_size),
        device=device,
        think_steps=think_steps,
        state_reset_each_step=True,
    )

    depth_sweep_losses: dict[str, float] = {}
    for depth in module._text_probe.parse_depth_sweep(str(train_args.eval_depth_sweep)):
        if int(depth) == think_steps:
            depth_sweep_losses[str(depth)] = float(full_loss)
        elif int(depth) == 0:
            depth_sweep_losses[str(depth)] = float(think0_loss)
        else:
            depth_sweep_losses[str(depth)] = module._text_probe.eval_loss(
                model,
                eval_windows,
                batch_size=int(train_args.batch_size),
                device=device,
                think_steps=int(depth),
            )
    shallow_losses = [
        loss for depth, loss in depth_sweep_losses.items() if int(depth) < think_steps
    ]
    best_shallow_loss = min(shallow_losses) if shallow_losses else None

    accepted = bool(
        full_loss < think0_loss
        and full_loss < off_loss
        and state_reset_loss > full_loss
        and (best_shallow_loss is None or full_loss < best_shallow_loss)
    )
    report = {
        "status": "complete",
        "decision": "accepted_qtrm_native_core_causality" if accepted else "rejected",
        "accepted": accepted,
        "checkpoint": str(args.checkpoint),
        "eval_windows": len(eval_windows),
        "vocab_size": int(tokenizer.vocab_size),
        "eval_metrics": {
            "think_eval_loss": float(full_loss),
            "think0_loss": float(think0_loss),
            "thinking_block_off_loss": float(off_loss),
            "state_reset_ablation": {
                "loss": float(state_reset_loss),
                "full_vs_state_reset": (
                    float(full_loss / state_reset_loss) if state_reset_loss > 0.0 else None
                ),
            },
            "loss_ratios": {
                "full_vs_think0": (
                    float(full_loss / think0_loss) if think0_loss > 0.0 else None
                ),
                "full_vs_thinking_block_off": (
                    float(full_loss / off_loss) if off_loss > 0.0 else None
                ),
                "full_vs_best_shallow_depth": (
                    float(full_loss / best_shallow_loss)
                    if best_shallow_loss and best_shallow_loss > 0.0
                    else None
                ),
            },
            "depth_sweep_loss": depth_sweep_losses,
            "best_shallow_depth_loss": best_shallow_loss,
        },
        "checks": {
            "full_beats_think0": bool(full_loss < think0_loss),
            "full_beats_off": bool(full_loss < off_loss),
            "state_reset_degrades": bool(state_reset_loss > full_loss),
            "full_beats_best_shallow": bool(best_shallow_loss is None or full_loss < best_shallow_loss),
        },
    }
    return report


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run(args)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
