#!/usr/bin/env python3
"""Probe bundle2 chain5 regressions for Qwen-preinit QTRM checkpoints.

This is a diagnostic, not a capability gate. It finds cases where the normal
Qwen path is correct but the mandatory recurrent core pushes the digit-choice
margin in the wrong direction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "scripts" / "362_train_qwen_backbone_qtrm_core_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("qwen_backbone_qtrm_core_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load gate module from {GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _checkpoint_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise argparse.ArgumentTypeError("checkpoint must be path or name=path")
        return name, path
    path = value.strip()
    if not path:
        raise argparse.ArgumentTypeError("empty checkpoint path")
    return Path(path).parent.name or Path(path).stem, path


def _read_checkpoint_report(path: str) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu")
    report = checkpoint.get("report", {})
    return report if isinstance(report, dict) else {}


def _csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


ARCH_DEFAULT_KEYS = (
    "model_id",
    "core_impl",
    "qwen_core_layer_indices",
    "ouro_model_id",
    "ouro_core_layer_indices",
    "core_adapter_dim",
    "core_delta_adapter_mode",
    "core_insertion_mode",
    "core_insert_after_layer",
    "core_residual_gate_mode",
    "core_residual_gate_dim",
    "core_residual_gate_init",
    "core_trajectory_carry_mode",
    "core_trajectory_carry_gate_init",
    "residual_scale",
    "h_cycles",
    "l_cycles",
    "outer_steps",
    "core_convergence_halt_enabled",
    "core_convergence_halt_threshold",
    "core_convergence_halt_min_outer",
    "core_step_conditioning_enabled",
    "core_step_conditioning_max_steps",
    "core_step_conditioning_scale",
    "n_core_layers",
    "delta_backend",
    "mandatory_core",
    "train_qwen",
)


def _apply_checkpoint_report_defaults(args: argparse.Namespace, report: dict[str, Any]) -> None:
    for key in ARCH_DEFAULT_KEYS:
        if key not in report or not hasattr(args, key):
            continue
        value = report[key]
        current = getattr(args, key)
        if isinstance(current, bool):
            setattr(args, key, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(args, key, int(value))
        elif isinstance(current, float):
            setattr(args, key, float(value))
        else:
            setattr(args, key, _csv(value))


def _build_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = gate.build_arg_parser()
    eval_args = parser.parse_args([])
    first_report = _read_checkpoint_report(args.checkpoints[0][1])
    _apply_checkpoint_report_defaults(eval_args, first_report)

    eval_args.model_id = str(args.model_id or eval_args.model_id)
    eval_args.device = str(args.device)
    eval_args.dtype = str(args.dtype)
    eval_args.max_seq_len = int(args.max_seq_len)
    eval_args.batch_size = int(args.batch_size)
    eval_args.eval_force_trajectory_carry_off = bool(args.eval_force_trajectory_carry_off)
    return eval_args


def _build_model(eval_args: argparse.Namespace, checkpoint_path: str):
    from transformers import AutoTokenizer
    from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM

    device = torch.device(str(eval_args.device))
    dtype = gate._dtype(str(eval_args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(eval_args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ouro_model = gate._load_ouro_model(eval_args, dtype=dtype, device=device)
    model = QwenBackboneQTRM.from_pretrained(
        str(eval_args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(eval_args.max_seq_len),
        freeze_qwen=not bool(eval_args.train_qwen),
        core_gate_init=float(eval_args.core_gate_init),
        residual_scale=float(eval_args.residual_scale),
        core_impl=str(eval_args.core_impl),
        mandatory_core=bool(eval_args.mandatory_core),
        qwen_core_layer_indices=gate.parse_int_list(str(eval_args.qwen_core_layer_indices)),
        ouro_model=ouro_model,
        ouro_core_layer_indices=gate.parse_int_list(str(eval_args.ouro_core_layer_indices)),
        core_adapter_dim=int(eval_args.core_adapter_dim),
        core_delta_adapter_mode=str(eval_args.core_delta_adapter_mode),
        core_insertion_mode=str(eval_args.core_insertion_mode),
        core_insert_after_layer=int(eval_args.core_insert_after_layer),
        core_residual_gate_mode=str(eval_args.core_residual_gate_mode),
        core_residual_gate_dim=int(eval_args.core_residual_gate_dim),
        core_residual_gate_init=float(eval_args.core_residual_gate_init),
        core_trajectory_carry_mode=str(eval_args.core_trajectory_carry_mode),
        core_trajectory_carry_gate_init=float(eval_args.core_trajectory_carry_gate_init),
        n_core_layers=int(eval_args.n_core_layers),
        h_cycles=int(eval_args.h_cycles),
        l_cycles=int(eval_args.l_cycles),
        outer_steps=int(eval_args.outer_steps),
        delta_backend=str(eval_args.delta_backend),
        strict_backends=bool(eval_args.strict_backends),
        core_convergence_halt_enabled=bool(eval_args.core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(eval_args.core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(eval_args.core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(eval_args.core_step_conditioning_enabled),
        core_step_conditioning_max_steps=int(eval_args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(eval_args.core_step_conditioning_scale),
        core_causal=True,
    ).to(device)
    gate._load_trainable_checkpoint(model, checkpoint_path)
    model.eval()
    return model, tokenizer


def _cases(args: argparse.Namespace) -> list[Any]:
    offsets = gate.parse_int_list_or_default(str(args.eval_seed_offsets), (20000, 20001, 20002))
    cases = []
    for offset in offsets:
        built = gate.build_synthetic_cases(
            count=int(args.eval_cases),
            seed=int(args.seed) + int(offset),
            case_mode=str(args.case_mode),
        )
        cases.extend(case for case in built if case.family == str(args.family))
    return cases


@torch.no_grad()
def _probe_checkpoint(
    *,
    name: str,
    path: str,
    eval_args: argparse.Namespace,
    cases: list[Any],
    top_k: int,
) -> dict[str, Any]:
    model, tokenizer = _build_model(eval_args, path)
    label_ids = gate._label_token_ids(tokenizer)
    label_token_ids = torch.tensor(
        [label_ids[digit] for digit in "0123456789"],
        device=next(model.parameters()).device,
    )
    choice_targets = {digit: idx for idx, digit in enumerate("0123456789")}

    total = 0
    choice_counts = {
        "both_correct": 0,
        "base_correct_core_wrong": 0,
        "base_wrong_core_correct": 0,
        "both_wrong": 0,
    }
    full_counts = {
        "both_correct": 0,
        "base_correct_core_wrong": 0,
        "base_wrong_core_correct": 0,
        "both_wrong": 0,
    }
    rows = []
    margin_drops = []
    base_margins = []
    core_margins = []

    for chunk in gate._batch(cases, int(eval_args.batch_size)):
        input_ids, attention_mask = gate._encode_prompts(
            tokenizer,
            [case.prompt for case in chunk],
            max_seq_len=int(eval_args.max_seq_len),
            device=next(model.parameters()).device,
        )
        base_logits = gate._last_token_logits(
            model,
            input_ids,
            attention_mask,
            force_core_off=True,
        )
        core_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            force_trajectory_carry_off=bool(eval_args.eval_force_trajectory_carry_off),
        )
        core_logits = core_outputs.logits[:, -1, :]
        target_choice = torch.tensor(
            [choice_targets[case.label] for case in chunk],
            device=core_logits.device,
        )
        base_choice = gate._digit_choice_predictions(base_logits, label_ids)
        core_choice = gate._digit_choice_predictions(core_logits, label_ids)
        base_full = base_logits.argmax(dim=-1)
        core_full = core_logits.argmax(dim=-1)
        target_token = torch.tensor(
            [label_ids[case.label] for case in chunk],
            device=core_logits.device,
        )
        base_margin = gate._digit_choice_margins(base_logits, label_token_ids, target_choice)
        core_margin = gate._digit_choice_margins(core_logits, label_token_ids, target_choice)

        for idx, case in enumerate(chunk):
            base_choice_ok = int(base_choice[idx]) == int(target_choice[idx])
            core_choice_ok = int(core_choice[idx]) == int(target_choice[idx])
            base_full_ok = int(base_full[idx]) == int(target_token[idx])
            core_full_ok = int(core_full[idx]) == int(target_token[idx])
            if base_choice_ok and core_choice_ok:
                choice_category = "both_correct"
            elif base_choice_ok and not core_choice_ok:
                choice_category = "base_correct_core_wrong"
            elif (not base_choice_ok) and core_choice_ok:
                choice_category = "base_wrong_core_correct"
            else:
                choice_category = "both_wrong"
            if base_full_ok and core_full_ok:
                full_category = "both_correct"
            elif base_full_ok and not core_full_ok:
                full_category = "base_correct_core_wrong"
            elif (not base_full_ok) and core_full_ok:
                full_category = "base_wrong_core_correct"
            else:
                full_category = "both_wrong"
            choice_counts[choice_category] += 1
            full_counts[full_category] += 1
            total += 1
            base_m = float(base_margin[idx].detach().cpu())
            core_m = float(core_margin[idx].detach().cpu())
            drop = core_m - base_m
            base_margins.append(base_m)
            core_margins.append(core_m)
            margin_drops.append(drop)
            if (
                choice_category in {"base_correct_core_wrong", "base_wrong_core_correct", "both_wrong"}
                or full_category in {"base_correct_core_wrong", "base_wrong_core_correct"}
            ):
                rows.append(
                    {
                        "choice_category": choice_category,
                        "full_category": full_category,
                        "prompt": case.prompt,
                        "label": case.label,
                        "base_choice": str(int(base_choice[idx].detach().cpu())),
                        "core_choice": str(int(core_choice[idx].detach().cpu())),
                        "base_full_token_id": int(base_full[idx].detach().cpu()),
                        "core_full_token_id": int(core_full[idx].detach().cpu()),
                        "base_full_token": tokenizer.decode([int(base_full[idx].detach().cpu())]),
                        "core_full_token": tokenizer.decode([int(core_full[idx].detach().cpu())]),
                        "base_margin": base_m,
                        "core_margin": core_m,
                        "core_minus_base_margin": drop,
                    }
                )

    regressions = sorted(
        (row for row in rows if row["choice_category"] == "base_correct_core_wrong"),
        key=lambda row: float(row["core_minus_base_margin"]),
    )[:top_k]
    recoveries = sorted(
        (row for row in rows if row["choice_category"] == "base_wrong_core_correct"),
        key=lambda row: float(row["core_minus_base_margin"]),
        reverse=True,
    )[:top_k]
    both_wrong = sorted(
        (row for row in rows if row["choice_category"] == "both_wrong"),
        key=lambda row: float(row["core_margin"]),
    )[:top_k]
    full_regressions = sorted(
        (row for row in rows if row["full_category"] == "base_correct_core_wrong"),
        key=lambda row: float(row["core_minus_base_margin"]),
    )[:top_k]
    full_recoveries = sorted(
        (row for row in rows if row["full_category"] == "base_wrong_core_correct"),
        key=lambda row: float(row["core_minus_base_margin"]),
        reverse=True,
    )[:top_k]
    return {
        "name": name,
        "checkpoint": path,
        "total": total,
        "base_choice_accuracy": (
            choice_counts["both_correct"] + choice_counts["base_correct_core_wrong"]
        ) / max(1, total),
        "core_choice_accuracy": (
            choice_counts["both_correct"] + choice_counts["base_wrong_core_correct"]
        ) / max(1, total),
        "choice_gain": (
            choice_counts["base_wrong_core_correct"] - choice_counts["base_correct_core_wrong"]
        )
        / max(1, total),
        "choice_counts": choice_counts,
        "base_full_accuracy": (
            full_counts["both_correct"] + full_counts["base_correct_core_wrong"]
        ) / max(1, total),
        "core_full_accuracy": (
            full_counts["both_correct"] + full_counts["base_wrong_core_correct"]
        ) / max(1, total),
        "full_gain": (
            full_counts["base_wrong_core_correct"] - full_counts["base_correct_core_wrong"]
        )
        / max(1, total),
        "full_counts": full_counts,
        "mean_base_margin": sum(base_margins) / max(1, len(base_margins)),
        "mean_core_margin": sum(core_margins) / max(1, len(core_margins)),
        "mean_core_minus_base_margin": sum(margin_drops) / max(1, len(margin_drops)),
        "top_choice_regressions": regressions,
        "top_choice_recoveries": recoveries,
        "top_choice_both_wrong": both_wrong,
        "top_full_regressions": full_regressions,
        "top_full_recoveries": full_recoveries,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        type=_checkpoint_arg,
        action="append",
        required=True,
        help="Checkpoint path, or name=path. May be repeated.",
    )
    parser.add_argument("--out-dir", default="local_eval/qwen35_chain5_hard_negative_probe")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--eval-seed-offsets", default="20000,20001,20002")
    parser.add_argument("--eval-cases", type=int, default=192)
    parser.add_argument("--case-mode", default="hard_v1")
    parser.add_argument("--family", default="chain5")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--eval-force-trajectory-carry-off", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_args = _build_eval_args(args)
    cases = _cases(args)
    report = {
        "status": "complete",
        "purpose": "diagnose bundle2 family regressions before more HRM-Text-style scaling",
        "family": str(args.family),
        "case_mode": str(args.case_mode),
        "eval_seed_offsets": list(
            gate.parse_int_list_or_default(str(args.eval_seed_offsets), (20000, 20001, 20002))
        ),
        "eval_cases_per_seed_before_family_filter": int(args.eval_cases),
        "filtered_cases": len(cases),
        "eval_force_trajectory_carry_off": bool(args.eval_force_trajectory_carry_off),
        "architecture": {
            "model_id": str(eval_args.model_id),
            "core_impl": str(eval_args.core_impl),
            "qwen_core_layer_indices": str(eval_args.qwen_core_layer_indices),
            "mandatory_core": bool(eval_args.mandatory_core),
            "h_cycles": int(eval_args.h_cycles),
            "l_cycles": int(eval_args.l_cycles),
            "outer_steps": int(eval_args.outer_steps),
            "core_trajectory_carry_mode": str(eval_args.core_trajectory_carry_mode),
            "core_insertion_mode": str(eval_args.core_insertion_mode),
            "train_qwen": bool(eval_args.train_qwen),
        },
        "checkpoints": [],
    }
    for name, checkpoint_path in args.checkpoints:
        report["checkpoints"].append(
            _probe_checkpoint(
                name=name,
                path=checkpoint_path,
                eval_args=eval_args,
                cases=cases,
                top_k=int(args.top_k),
            )
        )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
