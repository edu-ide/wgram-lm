"""Train a local Qwen-hidden typed-register extractor.

Stage53 is the follow-up to Stage52A. Stage52A showed that a clean typed
register ledger closes the K-sample selector gap. This script asks the next
causal question: can frozen Qwen hidden states read that ledger from the prompt?

The probe is local-only by design:
  prompt text -> frozen Qwen token hidden states -> typed registers
  typed registers -> deterministic executor -> GRAM candidate selection

It is not a final product architecture. It is the smallest falsifiable bridge
between "Qwen reads" and "typed verifier selects".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from qtrm_mm.qwen_backbone_state_transition import build_qwen_state_transition_model


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


class QwenRegisterExtractor(nn.Module):
    """Cross-attention reader from Qwen token states to typed registers."""

    def __init__(
        self,
        *,
        hidden_size: int,
        max_steps: int,
        num_heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.step_query = nn.Embedding(self.max_steps, hidden_size)
        self.initial_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.depth_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.initial_head = nn.Linear(hidden_size, 10)
        self.depth_head = nn.Linear(hidden_size, self.max_steps + 1)
        self.operation_head = nn.Linear(hidden_size, 4)
        self.argument_head = nn.Linear(hidden_size, 10)
        nn.init.normal_(self.initial_query, std=0.02)
        nn.init.normal_(self.depth_query, std=0.02)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.hidden_norm(hidden_states.float())
        batch = hidden.size(0)
        step_ids = torch.arange(self.max_steps, device=hidden.device)
        step_queries = self.step_query(step_ids).unsqueeze(0).expand(batch, -1, -1)
        queries = torch.cat(
            [
                self.initial_query.expand(batch, -1, -1),
                self.depth_query.expand(batch, -1, -1),
                step_queries,
            ],
            dim=1,
        )
        key_padding_mask = attention_mask.to(torch.bool).logical_not()
        read_states, _ = self.attn(
            queries,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        read_states = self.dropout(self.out_norm(read_states))
        initial_state = read_states[:, 0]
        depth_state = read_states[:, 1]
        step_states = read_states[:, 2:]
        return {
            "initial_logits": self.initial_head(initial_state),
            "depth_logits": self.depth_head(depth_state),
            "operation_logits": self.operation_head(step_states),
            "argument_logits": self.argument_head(step_states),
        }


def configure_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def init_aim_run(args: argparse.Namespace) -> Optional[Any]:
    if not args.aim_repo:
        return None
    try:
        from aim import Run
    except ImportError as exc:
        print(f"[warn] Aim logging disabled; package is not installed: {exc}", flush=True)
        return None
    run = Run(repo=args.aim_repo, experiment=args.aim_experiment)
    run.name = args.aim_run_name or args.run_name or os.path.basename(os.path.normpath(args.out_dir))
    run["hparams"] = dict(vars(args))
    run["paths"] = {
        "out_dir": args.out_dir,
        "tensorboard_logdir": os.path.join(args.out_dir, "logs"),
    }
    return run


def track_aim_scalar(
    aim_run: Optional[Any],
    value: float,
    *,
    name: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    context: Optional[Dict[str, str]] = None,
) -> None:
    if aim_run is None:
        return
    aim_run.track(float(value), name=name, step=step, epoch=epoch, context=context or {})


def build_cases(
    *,
    count: int,
    seed: int,
    depths: Sequence[int],
    max_steps: int,
    args: argparse.Namespace,
) -> List[Any]:
    return train511.build_generalized_synthetic_cases(
        count=int(count),
        seed=int(seed),
        depths=[int(depth) for depth in depths],
        max_steps=int(max_steps),
        condition_prefix=args.reasoning_condition_prefix,
        family_mix=args.synthetic_family_mix,
        sampling_strategy=args.synthetic_sampling_strategy,
    )


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


@torch.inference_mode()
def qwen_hidden(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model.qwen(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        hidden = outputs.hidden_states[-1]
    elif hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    else:
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
    return hidden.detach()


def case_targets(cases: Sequence[Any], max_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    operation_ids = torch.tensor([case.operation_ids for case in cases], dtype=torch.long, device=device)
    operation_args = torch.tensor(
        [case.operation_args or [0] * len(case.operation_ids) for case in cases],
        dtype=torch.long,
        device=device,
    )
    initial_labels = torch.tensor([case.initial_label for case in cases], dtype=torch.long, device=device)
    answer_labels = torch.tensor([case.answer_label for case in cases], dtype=torch.long, device=device)
    depths = torch.tensor([case.depth for case in cases], dtype=torch.long, device=device)
    operation_ids = train511.fit_step_sequence(operation_ids, int(max_steps))
    operation_args = train511.fit_step_sequence(operation_args, int(max_steps))
    active_mask = torch.arange(int(max_steps), device=device).unsqueeze(0) < depths.unsqueeze(1)
    return {
        "operation_ids": operation_ids,
        "operation_args": operation_args,
        "initial_labels": initial_labels,
        "answer_labels": answer_labels,
        "depths": depths,
        "active_mask": active_mask,
    }


def compute_register_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = targets["active_mask"]
    initial_loss = F.cross_entropy(outputs["initial_logits"], targets["initial_labels"])
    depth_loss = F.cross_entropy(outputs["depth_logits"], targets["depths"].clamp(min=0, max=outputs["depth_logits"].size(-1) - 1))
    op_logits = outputs["operation_logits"][active]
    arg_logits = outputs["argument_logits"][active]
    op_loss = F.cross_entropy(op_logits, targets["operation_ids"][active])
    arg_loss = F.cross_entropy(arg_logits, targets["operation_args"][active])
    loss = initial_loss + depth_loss + op_loss + arg_loss
    metrics = {
        "loss_initial": float(initial_loss.detach().item()),
        "loss_depth": float(depth_loss.detach().item()),
        "loss_operation": float(op_loss.detach().item()),
        "loss_argument": float(arg_loss.detach().item()),
    }
    return loss, metrics


def execute_register(
    *,
    initial: int,
    operation_ids: Sequence[int],
    operation_args: Sequence[int],
    depth: int,
) -> int:
    value = int(initial) % 10
    for index in range(int(depth)):
        op_id = int(operation_ids[index])
        arg = int(operation_args[index]) % 10
        if op_id == 0:
            value = (value + arg) % 10
        elif op_id == 1:
            value = (value * arg) % 10
        elif op_id == 2:
            value = (value - arg) % 10
        elif op_id == 3:
            value = value % 10
        else:
            raise ValueError(f"unknown op id: {op_id}")
    return value


def execute_predicted_registers(
    *,
    initial_digits: torch.Tensor,
    operation_ids: torch.Tensor,
    operation_args: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    answers: List[int] = []
    initial_list = initial_digits.detach().cpu().tolist()
    op_list = operation_ids.detach().cpu().tolist()
    arg_list = operation_args.detach().cpu().tolist()
    depth_list = depths.detach().cpu().tolist()
    for initial, ops, args, depth in zip(initial_list, op_list, arg_list, depth_list):
        answers.append(
            execute_register(
                initial=int(initial),
                operation_ids=ops,
                operation_args=args,
                depth=max(1, min(int(depth), len(ops))),
            )
        )
    return torch.tensor(answers, dtype=torch.long, device=initial_digits.device)


@torch.inference_mode()
def sample_candidate_digits(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    samples: int,
    topk_per_sample: int = 1,
    n_steps: int,
    max_length: int,
    condition_on_operation_ids: bool,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_batch(tokenizer, [case.prompt_text for case in cases], max_length, device)
    targets = case_targets(cases, n_steps, device)
    batch = len(cases)
    out = model(
        input_ids=encoded["input_ids"].repeat_interleave(samples, dim=0),
        attention_mask=encoded["attention_mask"].repeat_interleave(samples, dim=0),
        operation_ids=(
            targets["operation_ids"].repeat_interleave(samples, dim=0)
            if condition_on_operation_ids
            else None
        ),
        operation_arg_ids=(
            targets["operation_args"].repeat_interleave(samples, dim=0)
            if condition_on_operation_ids
            else None
        ),
        initial_labels=targets["initial_labels"].repeat_interleave(samples, dim=0),
        n_steps=int(n_steps),
        posterior_labels=None,
    )
    logits = out["answer_logits"].detach().float().reshape(batch, samples, -1)
    topk = max(1, min(int(topk_per_sample), int(logits.size(-1))))
    if topk == 1:
        return logits.argmax(dim=-1)
    return logits.topk(k=topk, dim=-1).indices.reshape(batch, samples * topk)


def select_by_register(candidate_digits: torch.Tensor, register_answers: torch.Tensor, answer_labels: torch.Tensor) -> Dict[str, int]:
    matches_label = candidate_digits.eq(answer_labels.unsqueeze(1))
    oracle = matches_label.any(dim=1)
    register_matches = candidate_digits.eq(register_answers.unsqueeze(1))
    selected_by_register = register_matches.float().argmax(dim=1)
    has_register_match = register_matches.any(dim=1)
    selected_digits = candidate_digits.gather(1, selected_by_register.unsqueeze(1)).squeeze(1)
    selected_correct = selected_digits.eq(answer_labels) & has_register_match
    first_correct = candidate_digits[:, 0].eq(answer_labels)
    return {
        "selected_correct": int(selected_correct.sum().item()),
        "oracle_correct": int(oracle.sum().item()),
        "first_correct": int(first_correct.sum().item()),
        "register_match": int(has_register_match.sum().item()),
        "total": int(candidate_digits.size(0)),
    }


def merge_counts(target: Dict[str, int], update: Dict[str, int]) -> None:
    for key, value in update.items():
        target[key] = int(target.get(key, 0)) + int(value)


def finalize_counts(counts: Dict[str, int]) -> Dict[str, float | int]:
    total = max(1, int(counts.get("total", 0)))
    return {
        **counts,
        "selected_accuracy": float(counts.get("selected_correct", 0)) / total,
        "oracle_accuracy": float(counts.get("oracle_correct", 0)) / total,
        "first_accuracy": float(counts.get("first_correct", 0)) / total,
        "register_match_rate": float(counts.get("register_match", 0)) / total,
    }


def prediction_metrics(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    active = targets["active_mask"]
    initial_pred = outputs["initial_logits"].argmax(dim=-1)
    depth_pred = outputs["depth_logits"].argmax(dim=-1).clamp(min=1, max=targets["operation_ids"].size(1))
    operation_pred = outputs["operation_logits"].argmax(dim=-1)
    argument_pred = outputs["argument_logits"].argmax(dim=-1)
    oracle_depth_answer = execute_predicted_registers(
        initial_digits=initial_pred,
        operation_ids=operation_pred,
        operation_args=argument_pred,
        depths=targets["depths"],
    )
    predicted_depth_answer = execute_predicted_registers(
        initial_digits=initial_pred,
        operation_ids=operation_pred,
        operation_args=argument_pred,
        depths=depth_pred,
    )
    return {
        "initial_accuracy": float(initial_pred.eq(targets["initial_labels"]).float().mean().item()),
        "depth_accuracy": float(depth_pred.eq(targets["depths"]).float().mean().item()),
        "operation_accuracy": float(operation_pred[active].eq(targets["operation_ids"][active]).float().mean().item()),
        "argument_accuracy": float(argument_pred[active].eq(targets["operation_args"][active]).float().mean().item()),
        "register_answer_accuracy_oracle_depth": float(oracle_depth_answer.eq(targets["answer_labels"]).float().mean().item()),
        "register_answer_accuracy_predicted_depth": float(predicted_depth_answer.eq(targets["answer_labels"]).float().mean().item()),
    }


def shuffled_batches(cases: Sequence[Any], batch_size: int, rng: random.Random) -> Sequence[List[Any]]:
    indices = list(range(len(cases)))
    rng.shuffle(indices)
    batches: List[List[Any]] = []
    for start in range(0, len(indices), int(batch_size)):
        batches.append([cases[index] for index in indices[start : start + int(batch_size)]])
    return batches


def train_epoch(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    extractor: QwenRegisterExtractor,
    optimizer: torch.optim.Optimizer,
    cases: Sequence[Any],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    aim_run: Optional[Any],
) -> Tuple[int, Dict[str, float]]:
    extractor.train()
    rng = random.Random(int(args.seed) + int(epoch) * 1009)
    totals: Dict[str, float] = {}
    count = 0
    started = time.time()
    for batch_cases in shuffled_batches(cases, args.batch_size, rng):
        encoded = encode_batch(tokenizer, [case.prompt_text for case in batch_cases], args.max_length, device)
        targets = case_targets(batch_cases, args.max_steps, device)
        with torch.no_grad():
            hidden = qwen_hidden(qtrm_model, encoded["input_ids"], encoded["attention_mask"])
        outputs = extractor(hidden, encoded["attention_mask"])
        loss, loss_metrics = compute_register_loss(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), float(args.grad_clip))
        optimizer.step()

        metrics = prediction_metrics(outputs, targets)
        metrics.update(loss_metrics)
        metrics["loss"] = float(loss.detach().item())
        batch_size = len(batch_cases)
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        if global_step % int(args.log_every) == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"Stage53/Train/{key}", float(value), global_step)
                track_aim_scalar(aim_run, float(value), name=key, step=global_step, epoch=epoch, context={"split": "train"})
        global_step += 1

    averaged = {key: value / max(1, count) for key, value in totals.items()}
    averaged["seconds"] = time.time() - started
    return global_step, averaged


@torch.inference_mode()
def evaluate_depth(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    extractor: QwenRegisterExtractor,
    depth: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    extractor.eval()
    cases = build_cases(
        count=args.eval_count,
        seed=args.eval_seed + int(depth),
        depths=[int(depth)],
        max_steps=args.max_steps,
        args=args,
    )
    metric_totals: Dict[str, float] = {}
    metric_count = 0
    strict_counts: Dict[str, int] = {}
    predicted_depth_counts: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []

    for start in range(0, len(cases), int(args.batch_size)):
        batch_cases = cases[start : start + int(args.batch_size)]
        encoded = encode_batch(tokenizer, [case.prompt_text for case in batch_cases], args.max_length, device)
        targets = case_targets(batch_cases, args.max_steps, device)
        hidden = qwen_hidden(qtrm_model, encoded["input_ids"], encoded["attention_mask"])
        outputs = extractor(hidden, encoded["attention_mask"])
        metrics = prediction_metrics(outputs, targets)
        batch_size = len(batch_cases)
        metric_count += batch_size
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * batch_size

        initial_pred = outputs["initial_logits"].argmax(dim=-1)
        depth_pred = outputs["depth_logits"].argmax(dim=-1).clamp(min=1, max=args.max_steps)
        operation_pred = outputs["operation_logits"].argmax(dim=-1)
        argument_pred = outputs["argument_logits"].argmax(dim=-1)
        oracle_depth_answer = execute_predicted_registers(
            initial_digits=initial_pred,
            operation_ids=operation_pred,
            operation_args=argument_pred,
            depths=targets["depths"],
        )
        predicted_depth_answer = execute_predicted_registers(
            initial_digits=initial_pred,
            operation_ids=operation_pred,
            operation_args=argument_pred,
            depths=depth_pred,
        )
        candidate_digits = sample_candidate_digits(
            qtrm_model,
            tokenizer,
            batch_cases,
            samples=args.samples,
            n_steps=args.model_n_steps,
            max_length=args.max_length,
            condition_on_operation_ids=args.condition_on_operation_ids,
            device=device,
        )
        merge_counts(strict_counts, select_by_register(candidate_digits, oracle_depth_answer, targets["answer_labels"]))
        merge_counts(predicted_depth_counts, select_by_register(candidate_digits, predicted_depth_answer, targets["answer_labels"]))

        if len(examples) < 3:
            for row_index, case in enumerate(batch_cases):
                examples.append(
                    {
                        "prompt_text": case.prompt_text,
                        "family": case.family,
                        "depth": int(case.depth),
                        "label": int(case.answer_label),
                        "candidate_digits": [int(value) for value in candidate_digits[row_index].detach().cpu().tolist()],
                        "pred_initial": int(initial_pred[row_index].item()),
                        "pred_depth": int(depth_pred[row_index].item()),
                        "pred_ops": [int(value) for value in operation_pred[row_index, : int(case.depth)].detach().cpu().tolist()],
                        "pred_args": [int(value) for value in argument_pred[row_index, : int(case.depth)].detach().cpu().tolist()],
                        "oracle_depth_answer": int(oracle_depth_answer[row_index].item()),
                        "predicted_depth_answer": int(predicted_depth_answer[row_index].item()),
                    }
                )
                if len(examples) >= 3:
                    break

    field_metrics = {key: value / max(1, metric_count) for key, value in metric_totals.items()}
    return {
        **field_metrics,
        "selector_oracle_depth": finalize_counts(strict_counts),
        "selector_predicted_depth": finalize_counts(predicted_depth_counts),
        "sample_cases": examples,
    }


def evaluate_all(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    extractor: QwenRegisterExtractor,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    aim_run: Optional[Any],
) -> Dict[str, Any]:
    depths: Dict[str, Any] = {}
    selected_values: List[float] = []
    predicted_depth_selected_values: List[float] = []
    oracle_values: List[float] = []
    register_values: List[float] = []
    for depth in args.eval_depths:
        result = evaluate_depth(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
            extractor=extractor,
            depth=int(depth),
            args=args,
            device=device,
        )
        depths[str(depth)] = result
        selected = float(result["selector_oracle_depth"]["selected_accuracy"])
        predicted_depth_selected = float(result["selector_predicted_depth"]["selected_accuracy"])
        oracle = float(result["selector_oracle_depth"]["oracle_accuracy"])
        register_acc = float(result["register_answer_accuracy_oracle_depth"])
        selected_values.append(selected)
        predicted_depth_selected_values.append(predicted_depth_selected)
        oracle_values.append(oracle)
        register_values.append(register_acc)
        context = {"split": "eval", "depth": str(int(depth))}
        writer.add_scalar(f"Stage53/EvalDepth{int(depth)}/SelectedOracleDepth", selected, epoch)
        writer.add_scalar(f"Stage53/EvalDepth{int(depth)}/SelectedPredictedDepth", predicted_depth_selected, epoch)
        writer.add_scalar(f"Stage53/EvalDepth{int(depth)}/Oracle", oracle, epoch)
        writer.add_scalar(f"Stage53/EvalDepth{int(depth)}/RegisterAnswerOracleDepth", register_acc, epoch)
        for key in ("initial_accuracy", "depth_accuracy", "operation_accuracy", "argument_accuracy"):
            writer.add_scalar(f"Stage53/EvalDepth{int(depth)}/{key}", float(result[key]), epoch)
            track_aim_scalar(aim_run, float(result[key]), name=key, epoch=epoch, context=context)
        track_aim_scalar(aim_run, selected, name="selected_accuracy_oracle_depth", epoch=epoch, context=context)
        track_aim_scalar(aim_run, predicted_depth_selected, name="selected_accuracy_predicted_depth", epoch=epoch, context=context)
        track_aim_scalar(aim_run, oracle, name="oracle_accuracy", epoch=epoch, context=context)
        track_aim_scalar(aim_run, register_acc, name="register_answer_accuracy_oracle_depth", epoch=epoch, context=context)
        print(
            f"eval epoch={epoch:02d} depth={int(depth):2d} "
            f"sel_true_depth={selected:.4f} sel_pred_depth={predicted_depth_selected:.4f} "
            f"oracle={oracle:.4f} reg={register_acc:.4f} "
            f"init={float(result['initial_accuracy']):.4f} op={float(result['operation_accuracy']):.4f} "
            f"arg={float(result['argument_accuracy']):.4f} depth_acc={float(result['depth_accuracy']):.4f}",
            flush=True,
        )

    summary = {
        "depths": depths,
        "mean_selected_accuracy_oracle_depth": sum(selected_values) / len(selected_values) if selected_values else 0.0,
        "mean_selected_accuracy_predicted_depth": (
            sum(predicted_depth_selected_values) / len(predicted_depth_selected_values)
            if predicted_depth_selected_values
            else 0.0
        ),
        "mean_oracle_accuracy": sum(oracle_values) / len(oracle_values) if oracle_values else 0.0,
        "mean_register_answer_accuracy_oracle_depth": sum(register_values) / len(register_values) if register_values else 0.0,
    }
    writer.add_scalar("Stage53/EvalMean/SelectedOracleDepth", summary["mean_selected_accuracy_oracle_depth"], epoch)
    writer.add_scalar("Stage53/EvalMean/SelectedPredictedDepth", summary["mean_selected_accuracy_predicted_depth"], epoch)
    writer.add_scalar("Stage53/EvalMean/Oracle", summary["mean_oracle_accuracy"], epoch)
    writer.add_scalar("Stage53/EvalMean/RegisterAnswerOracleDepth", summary["mean_register_answer_accuracy_oracle_depth"], epoch)
    track_aim_scalar(
        aim_run,
        summary["mean_selected_accuracy_oracle_depth"],
        name="mean_selected_accuracy_oracle_depth",
        epoch=epoch,
        context={"split": "eval"},
    )
    track_aim_scalar(
        aim_run,
        summary["mean_selected_accuracy_predicted_depth"],
        name="mean_selected_accuracy_predicted_depth",
        epoch=epoch,
        context={"split": "eval"},
    )
    track_aim_scalar(
        aim_run,
        summary["mean_register_answer_accuracy_oracle_depth"],
        name="mean_register_answer_accuracy_oracle_depth",
        epoch=epoch,
        context={"split": "eval"},
    )
    return summary


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


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="")
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
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--train-depths", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--reasoning-condition-prefix", default="synth")
    parser.add_argument("--synthetic-family-mix", default="balanced")
    parser.add_argument("--synthetic-sampling-strategy", default="random")
    parser.add_argument("--condition-on-operation-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=96)
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
    parser.add_argument("--aim-repo", default=os.environ.get("QTRM_AIM_REPO", ""))
    parser.add_argument("--aim-experiment", default="qwen35_hrmtext_stage53_register_extractor")
    parser.add_argument("--aim-run-name", default="")
    args = parser.parse_args()

    configure_reproducibility(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qtrm_model, tokenizer = build_qwen_state_transition_model(
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
    load_stats = train511.load_flexible_checkpoint(qtrm_model, args.checkpoint, device)
    override_stats = apply_recurrent_overrides(qtrm_model, args)
    qtrm_model.eval()
    for parameter in qtrm_model.parameters():
        parameter.requires_grad_(False)

    extractor = QwenRegisterExtractor(
        hidden_size=int(qtrm_model.hidden_size),
        max_steps=int(args.max_steps),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(extractor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    aim_run = init_aim_run(args)
    train_cases = build_cases(
        count=args.train_count,
        seed=args.seed,
        depths=args.train_depths,
        max_steps=args.max_steps,
        args=args,
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "load_stats": load_stats,
        "override_stats": override_stats,
        "train_case_preview": [asdict(case) for case in train_cases[:3]],
        "device": str(device),
    }
    save_json(os.path.join(args.out_dir, "metadata.json"), metadata)
    print(
        f"Stage53 local-only register extractor | train_count={len(train_cases)} "
        f"epochs={args.epochs} device={device} out_dir={args.out_dir}",
        flush=True,
    )

    global_step = 0
    best_score = -1.0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        global_step, train_metrics = train_epoch(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
            extractor=extractor,
            optimizer=optimizer,
            cases=train_cases,
            args=args,
            device=device,
            epoch=epoch,
            global_step=global_step,
            writer=writer,
            aim_run=aim_run,
        )
        for key, value in train_metrics.items():
            writer.add_scalar(f"Stage53/TrainEpoch/{key}", float(value), epoch)
            track_aim_scalar(aim_run, float(value), name=key, epoch=epoch, context={"split": "train_epoch"})
        print(
            f"epoch={epoch:02d} train loss={train_metrics['loss']:.4f} "
            f"init={train_metrics['initial_accuracy']:.4f} depth={train_metrics['depth_accuracy']:.4f} "
            f"op={train_metrics['operation_accuracy']:.4f} arg={train_metrics['argument_accuracy']:.4f} "
            f"reg={train_metrics['register_answer_accuracy_oracle_depth']:.4f} "
            f"time={train_metrics['seconds']:.1f}s",
            flush=True,
        )
        eval_summary = evaluate_all(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
            extractor=extractor,
            args=args,
            device=device,
            epoch=epoch,
            writer=writer,
            aim_run=aim_run,
        )
        score = float(eval_summary["mean_selected_accuracy_oracle_depth"])
        record = {"epoch": epoch, "train": train_metrics, "eval": eval_summary}
        history.append(record)
        save_json(os.path.join(args.out_dir, "summary.json"), {"metadata": metadata, "history": history})
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "extractor": extractor.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_score": best_score,
                    "eval": eval_summary,
                },
                os.path.join(args.out_dir, "best_register_extractor.pt"),
            )
        print(
            f"epoch={epoch:02d} eval mean_selected_true_depth={eval_summary['mean_selected_accuracy_oracle_depth']:.4f} "
            f"mean_selected_pred_depth={eval_summary['mean_selected_accuracy_predicted_depth']:.4f} "
            f"mean_oracle={eval_summary['mean_oracle_accuracy']:.4f} "
            f"mean_register={eval_summary['mean_register_answer_accuracy_oracle_depth']:.4f}",
            flush=True,
        )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
