#!/usr/bin/env python3
"""Evaluate one Qwen-backbone QTRM checkpoint across multiple held-out seeds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from wgram_lm.qwen_backbone_wgram import QwenBackboneQTRM


def load_trainer_module():
    path = Path(__file__).with_name("362_train_qwen_backbone_wgram_core_gate.py")
    spec = importlib.util.spec_from_file_location("qwen_backbone_wgram_core_gate", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load trainer module: {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(value).replace(",", " ").split() if part.strip()]
    if not seeds:
        raise ValueError("at least one eval seed is required")
    return seeds


def aggregate_seed_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            current: Any = row
            for part in path.split("."):
                current = current.get(part, {}) if isinstance(current, dict) else {}
            if isinstance(current, (int, float)):
                result.append(float(current))
        return result

    gain_values = values("after_eval.gain")
    family_gain_values = values("accepted_family_summary.min_gain")
    family_accuracy_values = values("accepted_family_summary.min_core_accuracy")
    language_values = values("after_language.top1_agreement")
    accepted_values = [bool(row.get("accepted", False)) for row in rows]
    return {
        "num_seeds": len(rows),
        "accepted": bool(rows and all(accepted_values)),
        "num_accepted": sum(1 for value in accepted_values if value),
        "min_gain": min(gain_values) if gain_values else 0.0,
        "mean_gain": sum(gain_values) / max(1, len(gain_values)),
        "min_family_gain": min(family_gain_values) if family_gain_values else 0.0,
        "mean_family_gain": sum(family_gain_values) / max(1, len(family_gain_values)),
        "min_family_core_accuracy": min(family_accuracy_values) if family_accuracy_values else 0.0,
        "mean_family_core_accuracy": (
            sum(family_accuracy_values) / max(1, len(family_accuracy_values))
        ),
        "min_language_top1_agreement": min(language_values) if language_values else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trainer = load_trainer_module()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = trainer._dtype(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = trainer._label_token_ids(tokenizer)
    ouro_model = trainer._load_ouro_model(args, dtype=dtype, device=device)
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=True,
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        qwen_core_layer_indices=trainer.parse_int_list(str(args.qwen_core_layer_indices)),
        ouro_model=ouro_model,
        ouro_core_layer_indices=trainer.parse_int_list(str(args.ouro_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        n_core_layers=int(args.n_core_layers),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        delta_backend=str(args.delta_backend),
        strict_backends=bool(args.strict_backends),
        core_convergence_halt_enabled=bool(args.core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(args.core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(args.core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(args.core_step_conditioning_enabled),
        core_step_conditioning_max_steps=int(args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(args.core_step_conditioning_scale),
        core_causal=True,
    ).to(device)
    model.qwen.eval()
    if hasattr(model, "ouro_model"):
        model.ouro_model.eval()
    init_checkpoint = trainer._load_trainable_checkpoint(model, str(args.init_checkpoint))
    model.eval()

    rows: list[dict[str, Any]] = []
    for eval_seed in parse_seed_list(str(args.eval_seeds)):
        eval_cases = trainer.build_synthetic_cases(
            count=int(args.eval_cases),
            seed=int(eval_seed),
            case_mode=str(args.case_mode),
        )
        after = trainer.evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
        after_language = trainer.evaluate_language_non_regression(model, tokenizer, args)
        acceptance = trainer.evaluation_acceptance_summary(after, args)
        accepted_language = (
            float(after_language["top1_agreement"]) >= float(args.min_language_top1_agreement)
        )
        accepted = bool(
            acceptance["accepted_reasoning_gain"]
            and acceptance["accepted_family_gain"]
            and acceptance["accepted_family_core_accuracy"]
            and accepted_language
        )
        rows.append(
            {
                "eval_seed": int(eval_seed),
                "accepted": accepted,
                "accepted_language_non_regression": bool(accepted_language),
                "after_eval": after,
                "after_language": after_language,
                "accepted_family_summary": acceptance["family_summary"],
                "acceptance_summary": acceptance,
            }
        )

    summary = aggregate_seed_reports(rows)
    return {
        "status": "complete",
        "accepted": bool(summary["accepted"]),
        "model_id": str(args.model_id),
        "init_checkpoint": init_checkpoint,
        "core_impl": str(args.core_impl),
        "qwen_core_layer_indices": trainer.parse_int_list(str(args.qwen_core_layer_indices)),
        "core_adapter_dim": int(args.core_adapter_dim),
        "core_delta_adapter_mode": str(args.core_delta_adapter_mode),
        "h_cycles": int(args.h_cycles),
        "l_cycles": int(args.l_cycles),
        "outer_steps": int(args.outer_steps),
        "case_mode": str(args.case_mode),
        "eval_cases": int(args.eval_cases),
        "eval_seeds": parse_seed_list(str(args.eval_seeds)),
        "thresholds": {
            "min_reasoning_gain": float(args.min_reasoning_gain),
            "min_language_top1_agreement": float(args.min_language_top1_agreement),
            "min_family_gain": float(args.min_family_gain),
            "min_family_core_accuracy": float(args.min_family_core_accuracy),
        },
        "summary": summary,
        "per_seed": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--out-dir", default="local_eval/qwen_backbone_wgram_stability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=80)
    parser.add_argument(
        "--core-impl",
        choices=[
            "qwen_layer_wrapped",
            "qwen_shared_layer_wrapped",
            "ouro_shared_qwen_layer",
            "ouro_weight_wrapped",
        ],
        default="qwen_layer_wrapped",
    )
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--ouro-model-id", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--ouro-core-layer-indices", default="")
    parser.add_argument("--ouro-partial-safetensors", action="store_true")
    parser.add_argument("--core-adapter-dim", type=int, default=128)
    parser.add_argument("--core-delta-adapter-mode", choices=["add", "adapter_only"], default="adapter_only")
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--core-convergence-halt-enabled", action="store_true")
    parser.add_argument("--core-convergence-halt-threshold", type=float, default=1.0e-3)
    parser.add_argument("--core-convergence-halt-min-outer", type=int, default=1)
    parser.add_argument("--core-step-conditioning-enabled", action="store_true")
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--delta-backend", default="fla_gated_delta")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-2.0)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-cases", type=int, default=512)
    parser.add_argument("--case-mode", choices=["standard", "hard_v1", "hard_repair_v1", "mixed_v1"], default="hard_v1")
    parser.add_argument("--eval-seeds", default="20270515 20270518 20270519")
    parser.add_argument("--acceptance-metric", choices=["full_vocab", "label_choice"], default="full_vocab")
    parser.add_argument("--min-reasoning-gain", type=float, default=0.05)
    parser.add_argument("--min-language-top1-agreement", type=float, default=0.50)
    parser.add_argument("--min-family-gain", type=float, default=0.01)
    parser.add_argument("--min-family-core-accuracy", type=float, default=0.10)
    parser.add_argument("--init-checkpoint", required=True)
    return parser


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
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
