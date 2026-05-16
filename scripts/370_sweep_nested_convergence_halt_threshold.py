#!/usr/bin/env python3
"""Sweep convergence early-exit thresholds for Qwen-wrapper nested QTRM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


def load_gate_module():
    path = Path(__file__).resolve().parent / "362_train_qwen_backbone_qtrm_core_gate.py"
    spec = importlib.util.spec_from_file_location("qwen_backbone_qtrm_core_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--out-dir", default="local_eval/qwen_nested_convergence_halt_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-cases", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--thresholds", default="0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--h-cycles", type=int, default=3)
    parser.add_argument("--l-cycles", type=int, default=6)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--core-adapter-dim", type=int, default=128)
    parser.add_argument("--core-gate-init", type=float, default=-2.0)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--case-mode", default="hard_v1")
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--min-outer", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gate = load_gate_module()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = gate._dtype(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = gate._label_token_ids(tokenizer)
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=True,
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl="qwen_layer_wrapped",
        qwen_core_layer_indices=gate.parse_int_list(str(args.qwen_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        core_convergence_halt_enabled=True,
        core_convergence_halt_threshold=0.0,
        core_convergence_halt_min_outer=int(args.min_outer),
        strict_backends=False,
        core_causal=True,
    ).to(device)
    model.eval()

    eval_cases = gate.build_synthetic_cases(
        count=int(args.eval_cases),
        seed=int(args.seed) + 10000,
        case_mode=str(args.case_mode),
    )
    eval_args = argparse.Namespace(
        batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_len),
        acceptance_metric="full_vocab",
        min_reasoning_gain=0.0,
        min_family_gain=-1.0,
        min_family_core_accuracy=0.0,
    )
    rows = []
    for threshold in parse_thresholds(str(args.thresholds)):
        model.core.cfg.core_convergence_halt_enabled = True
        model.core.cfg.core_convergence_halt_threshold = float(threshold)
        model.core.cfg.core_convergence_halt_min_outer = int(args.min_outer)
        evaluation = gate.evaluate_cases(model, tokenizer, eval_cases, eval_args, label_ids)
        family = gate.family_gain_summary(evaluation, metric="full_vocab")
        rows.append(
            {
                "threshold": float(threshold),
                "base_accuracy": evaluation["base_accuracy"],
                "core_accuracy": evaluation["core_accuracy"],
                "gain": evaluation["gain"],
                "mean_core_outer_iterations": evaluation["mean_core_outer_iterations"],
                "core_converged_fraction": evaluation["core_converged_fraction"],
                "min_family_gain": family["min_gain"],
                "min_family_core_accuracy": family["min_core_accuracy"],
                "by_family": evaluation["by_family"],
            }
        )
    result = {
        "status": "complete",
        "model_id": str(args.model_id),
        "case_mode": str(args.case_mode),
        "eval_cases": int(args.eval_cases),
        "h_cycles": int(args.h_cycles),
        "l_cycles": int(args.l_cycles),
        "outer_steps": int(args.outer_steps),
        "min_outer": int(args.min_outer),
        "residual_scale": float(args.residual_scale),
        "rows": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
