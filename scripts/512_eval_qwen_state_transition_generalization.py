"""
Evaluate Qwen state-transition checkpoints on unseen synthetic reasoning cases.

This is intentionally separate from the training script: it measures held-out
accuracy rather than tiny train-set overfit. It can also score the raw Qwen
next-token digit baseline on the same prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from qtrm_mm.qwen_backbone_state_transition import build_qwen_state_transition_model


@dataclass
class EvalCase:
    prompt_text: str
    operation_ids: List[int]
    answer_label: int
    state_labels: List[int]
    family: str
    depth: int


def build_eval_cases(*, count: int, seed: int, depth: int, condition_prefix: str = "synth") -> List[EvalCase]:
    rng = random.Random(seed)
    cases: List[EvalCase] = []
    families = ("chain", "chain", "checksum")

    for _ in range(count):
        family = rng.choice(families)
        if family == "checksum":
            digits = [rng.randint(0, 9) for _ in range(depth)]
            states: List[int] = []
            value = 0
            for digit in digits:
                value = (value + digit) % 10
                states.append(value)
            prompt = (
                f"Condition: {condition_prefix},direct\n"
                f"Reasoning task: checksum{depth} modulo 10.\n"
                f"digits={','.join(str(d) for d in digits)}.\n"
                f"ops={','.join(['add'] * depth)}.\n"
                "Return the final digit.\n"
                "Answer:"
            )
            cases.append(EvalCase(prompt, [0] * depth, value, states, family, depth))
            continue

        value = rng.randint(0, 9)
        start = value
        ops: List[int] = []
        op_text: List[str] = []
        states = []
        for _step in range(depth):
            op = rng.choice(("add", "mul", "sub"))
            arg = rng.randint(0, 9)
            if op == "add":
                value = (value + arg) % 10
                ops.append(0)
            elif op == "mul":
                value = (value * arg) % 10
                ops.append(1)
            else:
                value = (value - arg) % 10
                ops.append(2)
            op_text.append(f"{op}:{arg}")
            states.append(value)
        prompt = (
            f"Condition: {condition_prefix},cot\n"
            f"Reasoning task: chain{depth} modulo 10.\n"
            f"start={start}.\n"
            f"steps={','.join(op_text)}.\n"
            "Return the final digit.\n"
            "Answer:"
        )
        cases.append(EvalCase(prompt, ops, value, states, family, depth))

    return cases


def encode_batch(tokenizer: Any, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def load_flexible_checkpoint(model: Any, checkpoint_path: str, device: torch.device) -> Dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    current = model.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    exact = 0
    partial = 0
    skipped = 0

    for name, value in checkpoint.items():
        if name not in current:
            skipped += 1
            continue
        target = current[name]
        if tuple(value.shape) == tuple(target.shape):
            loaded[name] = value
            exact += 1
            continue
        if value.ndim == target.ndim and value.ndim >= 1 and tuple(value.shape[1:]) == tuple(target.shape[1:]):
            merged = target.clone()
            rows = min(value.shape[0], target.shape[0])
            merged[:rows] = value[:rows].to(dtype=target.dtype, device=target.device)
            loaded[name] = merged
            partial += 1
            continue
        skipped += 1

    current.update(loaded)
    model.load_state_dict(current, strict=True)
    return {"exact": exact, "partial": partial, "skipped": skipped, "checkpoint_tensors": len(checkpoint)}


def apply_recurrent_eval_overrides(model: Any, args: argparse.Namespace) -> Dict[str, int]:
    stats = {
        "transition_scale": 0,
        "injection_gate": 0,
        "step_embeddings_zeroed": 0,
        "step_embeddings_frozen": 0,
    }
    if args.override_transition_scale is not None:
        value = float(args.override_transition_scale)
        for module in model.modules():
            scale = getattr(module, "transition_scale", None)
            if isinstance(scale, torch.nn.Parameter):
                with torch.no_grad():
                    scale.fill_(value)
                stats["transition_scale"] += 1
    if args.override_injection_gate_logit is not None:
        value = float(args.override_injection_gate_logit)
        for module in model.modules():
            gate = getattr(module, "injection_gate", None)
            if isinstance(gate, torch.nn.Parameter):
                with torch.no_grad():
                    gate.fill_(value)
                stats["injection_gate"] += 1
    if args.zero_step_embeddings or args.freeze_step_embeddings:
        for module in model.modules():
            step_embed = getattr(module, "step_embed", None)
            weight = getattr(step_embed, "weight", None)
            if isinstance(weight, torch.nn.Parameter):
                if args.zero_step_embeddings:
                    with torch.no_grad():
                        weight.zero_()
                    stats["step_embeddings_zeroed"] += 1
                if args.freeze_step_embeddings:
                    weight.requires_grad_(False)
                    stats["step_embeddings_frozen"] += 1
    return stats


def digit_token_ids(tokenizer: Any) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for digit in range(10):
        ids = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) == 1:
            result[digit] = int(ids[0])
    return result


def logits_from_recurrent_state(model: Any, state: torch.Tensor) -> torch.Tensor:
    if getattr(model, "answer_path", "state_head") == "lm_head":
        logits, _ = model._lm_head_logits_from_state(state)
        return logits
    return model.answer_head(model.core_out_norm(state))


def attention_readout_state(model: Any, trajectory: torch.Tensor) -> torch.Tensor:
    states = trajectory[:, 1:, :]
    attn_input = states.to(model.recurrent_readout_attention.weight.dtype)
    scores = model.recurrent_readout_attention(attn_input).squeeze(-1)
    if getattr(model, "recurrent_readout_pooling", "attention") == "sharp_attention":
        scores = scores / float(getattr(model, "recurrent_readout_temperature", 1.0))
    weights = torch.softmax(scores, dim=1).unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(dim=1)


def select_eval_answer_logits(model: Any, out: Dict[str, Any], mode: str) -> torch.Tensor:
    if mode == "model":
        return out["answer_logits"]
    trajectory = out["qtrm_core_step_states"]
    if mode == "final":
        return logits_from_recurrent_state(model, trajectory[:, -1, :])
    if mode == "attention":
        return logits_from_recurrent_state(model, attention_readout_state(model, trajectory))
    if mode == "final_attention_logit_mean":
        final_logits = logits_from_recurrent_state(model, trajectory[:, -1, :])
        attention_logits = logits_from_recurrent_state(model, attention_readout_state(model, trajectory))
        return 0.5 * (final_logits + attention_logits)
    raise ValueError(f"unknown eval readout mode: {mode}")


@torch.inference_mode()
def eval_trm(
    model: Any,
    tokenizer: Any,
    cases: List[EvalCase],
    *,
    batch_size: int,
    max_length: int,
    n_steps: int,
    device: torch.device,
    eval_readout_mode: str,
    stochastic_eval_samples: int = 1,
    stochastic_selection_mode: str = "mean",
    condition_on_operation_ids: bool = True,
) -> Dict[str, Any]:
    correct = 0
    oracle_correct = 0
    total = 0
    by_family: Dict[str, Dict[str, int]] = {}

    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start : start + batch_size]
        encoded = encode_batch(tokenizer, [case.prompt_text for case in batch_cases], max_length, device)
        ops = torch.tensor([case.operation_ids for case in batch_cases], dtype=torch.long, device=device)
        labels = torch.tensor([case.answer_label for case in batch_cases], dtype=torch.long, device=device)
        sample_probs = []
        sample_preds = []
        sample_rewards = []
        for _ in range(max(1, int(stochastic_eval_samples))):
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                operation_ids=ops if condition_on_operation_ids else None,
                n_steps=n_steps,
            )
            answer_logits = select_eval_answer_logits(model, out, eval_readout_mode)
            probs = torch.softmax(answer_logits.float(), dim=-1)
            sample_probs.append(probs)
            sample_preds.append(probs.argmax(dim=-1))
            sample_rewards.append(out["qtrm_trajectory_reward_logits"].float())
        probs_by_sample = torch.stack(sample_probs, dim=0)
        pred_by_sample = torch.stack(sample_preds, dim=0)
        oracle_matches = pred_by_sample.eq(labels.unsqueeze(0)).any(dim=0)
        oracle_correct += int(oracle_matches.sum().item())
        if stochastic_selection_mode == "lprm":
            reward_by_sample = torch.stack(sample_rewards, dim=0)
            best_sample = reward_by_sample.argmax(dim=0)
            pred = probs_by_sample[best_sample, torch.arange(probs_by_sample.size(1), device=device)].argmax(dim=-1)
        elif stochastic_selection_mode == "confidence":
            confidence_by_sample = probs_by_sample.max(dim=-1).values
            best_sample = confidence_by_sample.argmax(dim=0)
            pred = probs_by_sample[best_sample, torch.arange(probs_by_sample.size(1), device=device)].argmax(dim=-1)
        elif stochastic_selection_mode == "vote":
            votes = torch.nn.functional.one_hot(pred_by_sample, num_classes=10).float().sum(dim=0)
            pred = votes.argmax(dim=-1)
        else:
            pred = probs_by_sample.mean(dim=0).argmax(dim=-1)
        matches = pred.eq(labels)
        correct += int(matches.sum().item())
        total += len(batch_cases)
        for case, ok in zip(batch_cases, matches.tolist()):
            bucket = by_family.setdefault(case.family, {"correct": 0, "total": 0})
            bucket["correct"] += int(bool(ok))
            bucket["total"] += 1

    return {
        "correct": correct,
        "oracle_correct": oracle_correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "oracle_accuracy": oracle_correct / total if total else 0.0,
        "by_family": {
            family: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for family, stats in by_family.items()
        },
    }


@torch.inference_mode()
def eval_raw_qwen(
    model: Any,
    tokenizer: Any,
    cases: List[EvalCase],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, Any]:
    ids_by_digit = digit_token_ids(tokenizer)
    if len(ids_by_digit) != 10:
        return {"correct": 0, "total": len(cases), "accuracy": 0.0, "error": "not all digits are single tokens"}

    digit_ids = torch.tensor([ids_by_digit[digit] for digit in range(10)], dtype=torch.long, device=device)
    correct = 0
    total = 0
    by_family: Dict[str, Dict[str, int]] = {}
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start : start + batch_size]
        encoded = encode_batch(tokenizer, [case.prompt_text for case in batch_cases], max_length, device)
        labels = torch.tensor([case.answer_label for case in batch_cases], dtype=torch.long, device=device)
        out = model.qwen(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        lengths = encoded["attention_mask"].sum(dim=1).clamp(min=1)
        row = torch.arange(encoded["input_ids"].size(0), device=device)
        logits = out.logits[row, lengths - 1][:, digit_ids]
        pred = logits.argmax(dim=-1)
        matches = pred.eq(labels)
        correct += int(matches.sum().item())
        total += len(batch_cases)
        for case, ok in zip(batch_cases, matches.tolist()):
            bucket = by_family.setdefault(case.family, {"correct": 0, "total": 0})
            bucket["correct"] += int(bool(ok))
            bucket["total"] += 1

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "by_family": {
            family: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for family, stats in by_family.items()
        },
    }


def init_aim(args: Any) -> Optional[Any]:
    if not args.aim_repo:
        return None
    try:
        from aim import Run
    except ImportError:
        return None
    run = Run(repo=args.aim_repo, experiment=args.aim_experiment)
    run.name = args.aim_run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run["hparams"] = vars(args)
    return run


def track_eval(aim_run: Optional[Any], result: Dict[str, Any], *, prefix: str, depth: int) -> None:
    if aim_run is None:
        return
    context = {"phase": "generalization", "split": "held_out", "depth": str(depth), "metric_set": prefix}
    aim_run.track(float(result["accuracy"]), name=f"{prefix}_accuracy", context=context)
    aim_run.track(float(result["correct"]), name=f"{prefix}_correct", context=context)
    aim_run.track(float(result["total"]), name=f"{prefix}_total", context=context)
    aim_run.track(float(result["accuracy"]), name=f"generalization_{prefix}_accuracy", context=context)
    aim_run.track(float(result["correct"]), name=f"generalization_{prefix}_correct", context=context)
    aim_run.track(float(result["total"]), name=f"generalization_{prefix}_total", context=context)
    for family, stats in result.get("by_family", {}).items():
        family_context = {
            "phase": "generalization",
            "split": "held_out",
            "depth": str(depth),
            "family": str(family),
            "metric_set": prefix,
        }
        aim_run.track(float(stats["accuracy"]), name=f"generalization_{prefix}_family_accuracy", context=family_context)
        aim_run.track(float(stats["correct"]), name=f"generalization_{prefix}_family_correct", context=family_context)
        aim_run.track(float(stats["total"]), name=f"generalization_{prefix}_family_total", context=family_context)


def default_tensorboard_logdir(out_path: str) -> str:
    path = Path(out_path)
    return str(path.with_suffix("").parent / f"{path.with_suffix('').name}_tb_logs")


def track_tensorboard_eval(
    writer: Optional[SummaryWriter],
    *,
    depth: int,
    trm_result: Dict[str, Any],
    raw_result: Optional[Dict[str, Any]],
) -> None:
    if writer is None:
        return
    trm_acc = float(trm_result["accuracy"])
    raw_acc = float((raw_result or {}).get("accuracy", 0.0))
    writer.add_scalar("Generalization/HeldOut/Accuracy_TRM", trm_acc, depth)
    writer.add_scalar("Generalization/HeldOut/Accuracy_RawQwen", raw_acc, depth)
    writer.add_scalar("Generalization/HeldOut/Accuracy_Delta_TRM_minus_RawQwen", trm_acc - raw_acc, depth)
    writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Accuracy_TRM", trm_acc, 0)
    writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Accuracy_RawQwen", raw_acc, 0)
    writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Accuracy_Delta_TRM_minus_RawQwen", trm_acc - raw_acc, 0)
    raw_by_family = (raw_result or {}).get("by_family", {})
    for family, trm_stats in trm_result.get("by_family", {}).items():
        trm_family_acc = float(trm_stats["accuracy"])
        raw_family_acc = float(raw_by_family.get(family, {}).get("accuracy", 0.0))
        writer.add_scalar(f"Generalization/HeldOut/Family_{family}/Accuracy_TRM", trm_family_acc, depth)
        writer.add_scalar(f"Generalization/HeldOut/Family_{family}/Accuracy_RawQwen", raw_family_acc, depth)
        writer.add_scalar(
            f"Generalization/HeldOut/Family_{family}/Accuracy_Delta_TRM_minus_RawQwen",
            trm_family_acc - raw_family_acc,
            depth,
        )
        writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Family_{family}/Accuracy_TRM", trm_family_acc, 0)
        writer.add_scalar(f"Generalization/HeldOut/Depth_{depth}/Family_{family}/Accuracy_RawQwen", raw_family_acc, 0)
        writer.add_scalar(
            f"Generalization/HeldOut/Depth_{depth}/Family_{family}/Accuracy_Delta_TRM_minus_RawQwen",
            trm_family_acc - raw_family_acc,
            0,
        )
    writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model-id", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--condition-prefix", type=str, default="synth")
    parser.add_argument("--model-n-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--workspace-pooling", choices=("mean", "last", "attention", "sequence", "none"), default="attention")
    parser.add_argument("--core-impl", choices=("state_transition", "hybrid_state_transition"), default="state_transition")
    parser.add_argument("--core-update", choices=("mlp", "mini_gated_delta"), default="mlp")
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="state_head")
    parser.add_argument("--condition-on-operation-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--recurrent-readout-pooling",
        choices=("final", "mean", "attention", "sharp_attention", "hybrid_gate"),
        default="final",
    )
    parser.add_argument("--recurrent-readout-temperature", type=float, default=1.0)
    parser.add_argument("--state-update-schedule", choices=("nested", "two_stream"), default="nested")
    parser.add_argument("--latent-feedback-passes", type=int, default=1)
    parser.add_argument("--correction-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--correction-feedback-scale", type=float, default=1.0)
    parser.add_argument("--correction-feedback-gate-init-bias", type=float, default=-1.0)
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="delta")
    parser.add_argument("--stochastic-high-level-scale", type=float, default=0.05)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=0.2)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-slots", type=int, default=4)
    parser.add_argument("--working-register-update-scale", type=float, default=0.25)
    parser.add_argument("--working-register-feedback-scale", type=float, default=1.0)
    parser.add_argument("--working-register-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--working-register-summary-mode", choices=("mean", "query_attention", "query_dot"), default="mean")
    parser.add_argument("--working-register-role-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-role-anchor-scale", type=float, default=0.0)
    parser.add_argument("--working-register-update-mode", choices=("all", "cyclic"), default="all")
    parser.add_argument("--semantic-token-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-token-feedback-scale", type=float, default=0.0)
    parser.add_argument("--semantic-token-feedback-temperature", type=float, default=1.0)
    parser.add_argument("--semantic-token-feedback-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--semantic-token-feedback-score-mode", choices=("cosine", "dot"), default="cosine")
    parser.add_argument("--semantic-token-feedback-teacher-forcing", type=float, default=0.0)
    parser.add_argument("--override-transition-scale", type=float, default=None)
    parser.add_argument("--override-injection-gate-logit", type=float, default=None)
    parser.add_argument("--zero-step-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-step-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-eval-samples", type=int, default=1)
    parser.add_argument("--stochastic-selection-mode", choices=("mean", "vote", "confidence", "lprm"), default="mean")
    parser.add_argument(
        "--eval-readout-mode",
        choices=("model", "final", "attention", "final_attention_logit_mean"),
        default="model",
        help=(
            "Which recurrent readout to use at evaluation. "
            "'model' uses the model's configured readout; "
            "'final_attention_logit_mean' averages final-step and attention-readout logits."
        ),
    )
    parser.add_argument("--freeze-qwen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-raw-qwen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aim-repo", type=str, default=os.environ.get("QTRM_AIM_REPO"))
    parser.add_argument("--aim-experiment", type=str, default="qwen35_hrmtext_generalization_eval")
    parser.add_argument("--aim-run-name", type=str, default=None)
    parser.add_argument("--tensorboard-logdir", type=str, default=None)
    args = parser.parse_args()
    if args.stochastic_high_level_scale < 0:
        raise ValueError("--stochastic-high-level-scale must be >= 0")
    if args.stochastic_high_level_min_std < 0:
        raise ValueError("--stochastic-high-level-min-std must be >= 0")
    if args.stochastic_high_level_max_std < args.stochastic_high_level_min_std:
        raise ValueError("--stochastic-high-level-max-std must be >= --stochastic-high-level-min-std")
    if args.stochastic_eval_samples <= 0:
        raise ValueError("--stochastic-eval-samples must be positive")
    if args.stochastic_posterior_guidance and not args.stochastic_high_level_guidance:
        raise ValueError("--stochastic-posterior-guidance requires --stochastic-high-level-guidance")
    if args.working_register_slots <= 0:
        raise ValueError("--working-register-slots must be positive")
    if args.working_register_update_scale < 0:
        raise ValueError("--working-register-update-scale must be >= 0")
    if args.working_register_feedback_scale < 0:
        raise ValueError("--working-register-feedback-scale must be >= 0")
    if args.working_register_role_anchor_scale < 0:
        raise ValueError("--working-register-role-anchor-scale must be >= 0")
    if args.semantic_token_feedback and args.answer_path != "lm_head":
        raise ValueError("--semantic-token-feedback requires --answer-path lm_head")
    if args.semantic_token_feedback_scale < 0:
        raise ValueError("--semantic-token-feedback-scale must be >= 0")
    if args.semantic_token_feedback_temperature <= 0:
        raise ValueError("--semantic-token-feedback-temperature must be > 0")
    if not 0.0 <= args.semantic_token_feedback_teacher_forcing <= 1.0:
        raise ValueError("--semantic-token-feedback-teacher-forcing must be in [0, 1]")
    if args.override_transition_scale is not None and args.override_transition_scale < 0:
        raise ValueError("--override-transition-scale must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_qwen_state_transition_model(
        args.qwen_model_id,
        freeze_qwen=args.freeze_qwen,
        device=device,
        core_impl=args.core_impl,
        core_update=args.core_update,
        answer_path=args.answer_path,
        workspace_pooling=args.workspace_pooling,
        recurrent_readout_pooling=args.recurrent_readout_pooling,
        recurrent_readout_temperature=args.recurrent_readout_temperature,
        n_steps=args.model_n_steps,
        state_update_schedule=args.state_update_schedule,
        latent_feedback_passes=args.latent_feedback_passes,
        correction_feedback=args.correction_feedback,
        correction_feedback_scale=args.correction_feedback_scale,
        correction_feedback_gate_init_bias=args.correction_feedback_gate_init_bias,
        stochastic_high_level_guidance=args.stochastic_high_level_guidance,
        stochastic_transition_mode=args.stochastic_transition_mode,
        stochastic_high_level_scale=args.stochastic_high_level_scale,
        stochastic_high_level_min_std=args.stochastic_high_level_min_std,
        stochastic_high_level_max_std=args.stochastic_high_level_max_std,
        stochastic_high_level_eval=args.stochastic_high_level_eval,
        stochastic_posterior_guidance=args.stochastic_posterior_guidance,
        working_register_enabled=args.working_register_enabled,
        working_register_slots=args.working_register_slots,
        working_register_update_scale=args.working_register_update_scale,
        working_register_feedback_scale=args.working_register_feedback_scale,
        working_register_gate_init_bias=args.working_register_gate_init_bias,
        working_register_summary_mode=args.working_register_summary_mode,
        working_register_role_conditioning=args.working_register_role_conditioning,
        working_register_role_anchor_scale=args.working_register_role_anchor_scale,
        working_register_update_mode=args.working_register_update_mode,
        semantic_token_feedback=args.semantic_token_feedback,
        semantic_token_feedback_scale=args.semantic_token_feedback_scale,
        semantic_token_feedback_temperature=args.semantic_token_feedback_temperature,
        semantic_token_feedback_gate_init_bias=args.semantic_token_feedback_gate_init_bias,
        semantic_token_feedback_score_mode=args.semantic_token_feedback_score_mode,
        semantic_token_feedback_teacher_forcing=args.semantic_token_feedback_teacher_forcing,
    )
    load_stats = None
    if args.checkpoint:
        load_stats = load_flexible_checkpoint(model, args.checkpoint, device)
    override_stats = apply_recurrent_eval_overrides(model, args)
    model.eval()

    aim_run = init_aim(args)
    if aim_run is not None:
        aim_run["checkpoint"] = args.checkpoint
        aim_run["load_stats"] = load_stats
        aim_run["override_stats"] = override_stats
    tb_logdir = args.tensorboard_logdir or default_tensorboard_logdir(args.out)
    writer = SummaryWriter(log_dir=tb_logdir)

    summary: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "load_stats": load_stats,
        "qwen_model_id": args.qwen_model_id,
        "core_impl": args.core_impl,
        "core_update": args.core_update,
        "eval_count": args.eval_count,
        "eval_seed": args.eval_seed,
        "model_n_steps": args.model_n_steps,
        "answer_path": args.answer_path,
        "condition_on_operation_ids": args.condition_on_operation_ids,
        "recurrent_readout_pooling": args.recurrent_readout_pooling,
        "recurrent_readout_temperature": args.recurrent_readout_temperature,
        "state_update_schedule": args.state_update_schedule,
        "latent_feedback_passes": args.latent_feedback_passes,
        "correction_feedback": args.correction_feedback,
        "correction_feedback_scale": args.correction_feedback_scale,
        "stochastic_high_level_guidance": args.stochastic_high_level_guidance,
        "stochastic_transition_mode": args.stochastic_transition_mode,
        "stochastic_high_level_scale": args.stochastic_high_level_scale,
        "stochastic_high_level_eval": args.stochastic_high_level_eval,
        "stochastic_posterior_guidance": args.stochastic_posterior_guidance,
        "working_register_enabled": args.working_register_enabled,
        "working_register_slots": args.working_register_slots,
        "working_register_update_scale": args.working_register_update_scale,
        "working_register_feedback_scale": args.working_register_feedback_scale,
        "working_register_summary_mode": args.working_register_summary_mode,
        "working_register_role_conditioning": args.working_register_role_conditioning,
        "working_register_role_anchor_scale": args.working_register_role_anchor_scale,
        "working_register_update_mode": args.working_register_update_mode,
        "semantic_token_feedback": args.semantic_token_feedback,
        "semantic_token_feedback_scale": args.semantic_token_feedback_scale,
        "semantic_token_feedback_temperature": args.semantic_token_feedback_temperature,
        "semantic_token_feedback_score_mode": args.semantic_token_feedback_score_mode,
        "semantic_token_feedback_teacher_forcing": args.semantic_token_feedback_teacher_forcing,
        "override_transition_scale": args.override_transition_scale,
        "override_injection_gate_logit": args.override_injection_gate_logit,
        "zero_step_embeddings": args.zero_step_embeddings,
        "freeze_step_embeddings": args.freeze_step_embeddings,
        "override_stats": override_stats,
        "stochastic_eval_samples": args.stochastic_eval_samples,
        "stochastic_selection_mode": args.stochastic_selection_mode,
        "eval_readout_mode": args.eval_readout_mode,
        "depths": {},
    }

    for depth in args.eval_depths:
        cases = build_eval_cases(
            count=args.eval_count,
            seed=args.eval_seed + depth,
            depth=depth,
            condition_prefix=args.condition_prefix,
        )
        trm_result = eval_trm(
            model,
            tokenizer,
            cases,
            batch_size=args.batch_size,
            max_length=args.max_length,
            n_steps=depth,
            device=device,
            eval_readout_mode=args.eval_readout_mode,
            stochastic_eval_samples=args.stochastic_eval_samples,
            stochastic_selection_mode=args.stochastic_selection_mode,
            condition_on_operation_ids=args.condition_on_operation_ids,
        )
        raw_result = None
        if args.eval_raw_qwen:
            raw_result = eval_raw_qwen(
                model,
                tokenizer,
                cases,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )
        summary["depths"][str(depth)] = {
            "trm": trm_result,
            "raw_qwen": raw_result,
            "sample_cases": [asdict(case) for case in cases[:3]],
        }
        track_eval(aim_run, trm_result, prefix="trm", depth=depth)
        if raw_result is not None:
            track_eval(aim_run, raw_result, prefix="raw_qwen", depth=depth)
            delta = float(trm_result["accuracy"]) - float(raw_result["accuracy"])
            if aim_run is not None:
                aim_run.track(
                    delta,
                    name="generalization_delta_accuracy_trm_minus_raw_qwen",
                    context={"phase": "generalization", "split": "held_out", "depth": str(depth)},
                )
                for family, trm_stats in trm_result.get("by_family", {}).items():
                    raw_stats = raw_result.get("by_family", {}).get(family, {})
                    aim_run.track(
                        float(trm_stats["accuracy"]) - float(raw_stats.get("accuracy", 0.0)),
                        name="generalization_delta_family_accuracy_trm_minus_raw_qwen",
                        context={
                            "phase": "generalization",
                            "split": "held_out",
                            "depth": str(depth),
                            "family": str(family),
                        },
                    )
        track_tensorboard_eval(writer, depth=depth, trm_result=trm_result, raw_result=raw_result)
        print(
            f"depth={depth} trm_acc={trm_result['accuracy']:.4f} "
            f"raw_qwen_acc={(raw_result or {}).get('accuracy', 0.0):.4f}",
            flush=True,
        )

    if aim_run is not None:
        aim_run["summary"] = summary
        aim_run.close()
    writer.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
