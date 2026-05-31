"""Evaluate a structured typed-register selector on GRAM candidates.

Stage52A is a local-only upper-bound probe. It does not train a neural judge.
It asks: if the verifier had the typed operation/operand registers, could it
select the correct sampled trajectory? If yes, the next architecture target is
not another scalar reward head but a Qwen reader that emits reliable registers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from wgram_lm.qwen_backbone_state_transition import build_qwen_state_transition_model


def _load_train511() -> Any:
    path = Path(__file__).resolve().parent / "511_train_qwen_state_transition_hrmtext.py"
    spec = importlib.util.spec_from_file_location("qtrm_stage511", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


train511 = _load_train511()


def configure_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def encode_batch(tokenizer: Any, texts: Sequence[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=int(max_length),
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def apply_recurrent_overrides(model: Any, args: argparse.Namespace) -> Dict[str, int]:
    stats = {"transition_scale": 0, "injection_gate": 0, "step_embeddings_zeroed": 0, "step_embeddings_frozen": 0}
    if args.override_transition_scale is not None:
        for module in model.modules():
            scale = getattr(module, "transition_scale", None)
            if isinstance(scale, torch.nn.Parameter):
                with torch.no_grad():
                    scale.fill_(float(args.override_transition_scale))
                stats["transition_scale"] += 1
    if args.override_injection_gate_logit is not None:
        for module in model.modules():
            gate = getattr(module, "injection_gate", None)
            if isinstance(gate, torch.nn.Parameter):
                with torch.no_grad():
                    gate.fill_(float(args.override_injection_gate_logit))
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


def build_cases(count: int, seed: int, depths: Sequence[int], args: argparse.Namespace) -> List[Any]:
    return train511.build_generalized_synthetic_cases(
        count=int(count),
        seed=int(seed),
        depths=[int(depth) for depth in depths],
        max_steps=max(int(depth) for depth in depths),
        condition_prefix=args.reasoning_condition_prefix,
        family_mix=args.synthetic_family_mix,
        sampling_strategy=args.synthetic_sampling_strategy,
    )


def execute_register(case: Any) -> Tuple[int, List[int]]:
    """Compute the modulo trace from typed operation/operand registers."""
    value = int(case.initial_label) % 10
    states: List[int] = []
    operation_ids = list(case.operation_ids)
    operation_args = list(case.operation_args or [0] * len(operation_ids))
    for step in range(int(case.depth)):
        op_id = int(operation_ids[step])
        arg = int(operation_args[step]) % 10
        if op_id == 0:  # add
            value = (value + arg) % 10
        elif op_id == 1:  # mul
            value = (value * arg) % 10
        elif op_id == 2:  # sub
            value = (value - arg) % 10
        elif op_id == 3:  # copy / no-op
            value = value % 10
        else:
            raise ValueError(f"unknown op id: {op_id}")
        states.append(value)
    return value, states


def cases_to_tensors(cases: Sequence[Any], n_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    operation_ids = torch.tensor([case.operation_ids for case in cases], dtype=torch.long, device=device)
    operation_args = torch.tensor(
        [case.operation_args or [0] * len(case.operation_ids) for case in cases],
        dtype=torch.long,
        device=device,
    )
    initial_labels = torch.tensor([case.initial_label for case in cases], dtype=torch.long, device=device)
    answer_labels = torch.tensor([case.answer_label for case in cases], dtype=torch.long, device=device)
    return {
        "operation_ids": train511.fit_step_sequence(operation_ids, int(n_steps)),
        "operation_arg_ids": train511.fit_step_sequence(operation_args, int(n_steps)),
        "initial_labels": initial_labels,
        "answer_labels": answer_labels,
    }


@torch.inference_mode()
def sample_candidate_digits(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    samples: int,
    n_steps: int,
    max_length: int,
    condition_on_operation_ids: bool,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_batch(tokenizer, [case.prompt_text for case in cases], max_length, device)
    tensors = cases_to_tensors(cases, n_steps, device)
    batch = len(cases)
    out = model(
        input_ids=encoded["input_ids"].repeat_interleave(samples, dim=0),
        attention_mask=encoded["attention_mask"].repeat_interleave(samples, dim=0),
        operation_ids=(
            tensors["operation_ids"].repeat_interleave(samples, dim=0)
            if condition_on_operation_ids
            else None
        ),
        operation_arg_ids=(
            tensors["operation_arg_ids"].repeat_interleave(samples, dim=0)
            if condition_on_operation_ids
            else None
        ),
        initial_labels=tensors["initial_labels"].repeat_interleave(samples, dim=0),
        n_steps=int(n_steps),
        posterior_labels=None,
    )
    logits = out["answer_logits"].detach().float().reshape(batch, samples, -1)
    return logits.argmax(dim=-1)


def score_cases(candidate_digits: torch.Tensor, cases: Sequence[Any]) -> Dict[str, Any]:
    labels = torch.tensor([int(case.answer_label) for case in cases], dtype=torch.long, device=candidate_digits.device)
    register_answers = torch.tensor([execute_register(case)[0] for case in cases], dtype=torch.long, device=candidate_digits.device)
    register_ok = register_answers.eq(labels)
    matches = candidate_digits.eq(labels.unsqueeze(1))
    oracle = matches.any(dim=1)
    register_matches = candidate_digits.eq(register_answers.unsqueeze(1))
    selected_by_register = register_matches.float().argmax(dim=1)
    has_register_match = register_matches.any(dim=1)
    selected_digits = candidate_digits.gather(1, selected_by_register.unsqueeze(1)).squeeze(1)
    selected_correct = selected_digits.eq(labels) & has_register_match
    first_correct = candidate_digits[:, 0].eq(labels)
    return {
        "selected_correct": int(selected_correct.sum().item()),
        "oracle_correct": int(oracle.sum().item()),
        "first_correct": int(first_correct.sum().item()),
        "register_match": int(has_register_match.sum().item()),
        "register_ok": int(register_ok.sum().item()),
        "total": int(len(cases)),
    }


def merge_counts(target: Dict[str, int], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, int):
            target[key] = int(target.get(key, 0)) + int(value)


def finalize_counts(counts: Dict[str, int]) -> Dict[str, float | int]:
    total = max(1, int(counts.get("total", 0)))
    return {
        **counts,
        "selected_accuracy": float(counts.get("selected_correct", 0)) / total,
        "oracle_accuracy": float(counts.get("oracle_correct", 0)) / total,
        "first_accuracy": float(counts.get("first_correct", 0)) / total,
        "register_match_rate": float(counts.get("register_match", 0)) / total,
        "register_correctness": float(counts.get("register_ok", 0)) / total,
    }


def evaluate_depth(
    model: Any,
    tokenizer: Any,
    *,
    depth: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    cases = build_cases(args.eval_count, args.eval_seed + int(depth), [int(depth)], args)
    counts: Dict[str, int] = {}
    by_family: Dict[str, Dict[str, int]] = {}
    examples: List[Dict[str, Any]] = []
    for start in range(0, len(cases), int(args.batch_size)):
        batch_cases = cases[start : start + int(args.batch_size)]
        digits = sample_candidate_digits(
            model,
            tokenizer,
            batch_cases,
            samples=args.samples,
            n_steps=args.model_n_steps,
            max_length=args.max_length,
            condition_on_operation_ids=args.condition_on_operation_ids,
            device=device,
        )
        batch_counts = score_cases(digits, batch_cases)
        merge_counts(counts, batch_counts)
        for family in sorted({case.family for case in batch_cases}):
            indices = [i for i, case in enumerate(batch_cases) if case.family == family]
            family_digits = digits[indices]
            family_cases = [batch_cases[i] for i in indices]
            family_counts = score_cases(family_digits, family_cases)
            merge_counts(by_family.setdefault(family, {}), family_counts)
        if len(examples) < 3:
            for case, row in zip(batch_cases, digits.tolist()):
                register_answer, register_trace = execute_register(case)
                examples.append(
                    {
                        "prompt_text": case.prompt_text,
                        "family": case.family,
                        "depth": case.depth,
                        "answer_label": case.answer_label,
                        "candidate_digits": row,
                        "register_answer": register_answer,
                        "register_trace": register_trace,
                    }
                )
                if len(examples) >= 3:
                    break
    result = finalize_counts(counts)
    result["by_family"] = {family: finalize_counts(stats) for family, stats in sorted(by_family.items())}
    result["sample_cases"] = examples
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tensorboard-logdir", default=None)
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", choices=("mean", "last", "attention", "sequence", "none"), default="sequence")
    parser.add_argument("--core-impl", choices=("state_transition", "hybrid_state_transition"), default="state_transition")
    parser.add_argument("--core-update", choices=("mlp", "mini_gated_delta"), default="mlp")
    parser.add_argument("--state-update-schedule", choices=("nested", "two_stream"), default="nested")
    parser.add_argument("--recurrent-readout-pooling", choices=("final", "mean", "attention", "sharp_attention", "hybrid_gate"), default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--model-n-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--reasoning-condition-prefix", default="synth")
    parser.add_argument("--synthetic-family-mix", default="balanced")
    parser.add_argument("--synthetic-sampling-strategy", default="random")
    parser.add_argument("--condition-on-operation-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=95)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--override-transition-scale", type=float, default=0.05)
    parser.add_argument("--override-injection-gate-logit", type=float, default=3.0)
    parser.add_argument("--zero-step-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-step-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="true_gram")
    parser.add_argument("--stochastic-high-level-scale", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.samples <= 1:
        raise ValueError("--samples must be > 1")
    configure_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_qwen_state_transition_model(
        args.qwen_model_id,
        freeze_qwen=True,
        device=device,
        core_impl=args.core_impl,
        core_update=args.core_update,
        answer_path=args.answer_path,
        workspace_pooling=args.workspace_pooling,
        recurrent_readout_pooling=args.recurrent_readout_pooling,
        recurrent_readout_temperature=args.recurrent_readout_temperature,
        n_steps=args.model_n_steps,
        state_update_schedule=args.state_update_schedule,
        stochastic_high_level_guidance=args.stochastic_high_level_guidance,
        stochastic_high_level_scale=args.stochastic_high_level_scale,
        stochastic_high_level_min_std=args.stochastic_high_level_min_std,
        stochastic_high_level_max_std=args.stochastic_high_level_max_std,
        stochastic_high_level_eval=args.stochastic_high_level_eval,
        stochastic_posterior_guidance=args.stochastic_posterior_guidance,
        stochastic_transition_mode=args.stochastic_transition_mode,
    )
    load_stats = train511.load_flexible_checkpoint(model, args.checkpoint, device)
    override_stats = apply_recurrent_overrides(model, args)
    model.eval()

    writer = SummaryWriter(log_dir=args.tensorboard_logdir) if args.tensorboard_logdir else None
    depths: Dict[str, Any] = {}
    selected_values: List[float] = []
    oracle_values: List[float] = []
    first_values: List[float] = []
    for depth in args.eval_depths:
        result = evaluate_depth(model, tokenizer, depth=int(depth), args=args, device=device)
        depths[str(depth)] = result
        selected_values.append(float(result["selected_accuracy"]))
        oracle_values.append(float(result["oracle_accuracy"]))
        first_values.append(float(result["first_accuracy"]))
        if writer is not None:
            writer.add_scalar("Stage52/StructuredRegister/SelectedAccuracy", float(result["selected_accuracy"]), int(depth))
            writer.add_scalar("Stage52/StructuredRegister/OracleAccuracy", float(result["oracle_accuracy"]), int(depth))
            writer.add_scalar("Stage52/StructuredRegister/FirstAccuracy", float(result["first_accuracy"]), int(depth))
            writer.add_scalar("Stage52/StructuredRegister/RegisterCorrectness", float(result["register_correctness"]), int(depth))
        print(
            f"depth={int(depth):2d} selected={float(result['selected_accuracy']):.4f} "
            f"oracle={float(result['oracle_accuracy']):.4f} first={float(result['first_accuracy']):.4f} "
            f"register_ok={float(result['register_correctness']):.4f}",
            flush=True,
        )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "load_stats": load_stats,
        "override_stats": override_stats,
        "depths": depths,
        "mean_selected_accuracy": sum(selected_values) / len(selected_values) if selected_values else 0.0,
        "mean_oracle_accuracy": sum(oracle_values) / len(oracle_values) if oracle_values else 0.0,
        "mean_first_accuracy": sum(first_values) / len(first_values) if first_values else 0.0,
    }
    if writer is not None:
        writer.add_scalar("Stage52/StructuredRegister/MeanSelectedAccuracy", summary["mean_selected_accuracy"], 0)
        writer.add_scalar("Stage52/StructuredRegister/MeanOracleAccuracy", summary["mean_oracle_accuracy"], 0)
        writer.add_scalar("Stage52/StructuredRegister/MeanFirstAccuracy", summary["mean_first_accuracy"], 0)
        writer.flush()
        writer.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
