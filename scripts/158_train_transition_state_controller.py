#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qtrm_mm.agentic.cognitive_loop import Action
from qtrm_mm.agentic.transition_controller import (
    TransitionStatePredictor,
    TransitionStateController,
    transition_action_loss,
    transition_state_prediction_loss,
)
from qtrm_mm.config import load_config
from qtrm_mm.data.jsonl_dataset import (
    _render_trace_replay_action_input,
    _trace_action_id,
    build_text_tokenizer,
)
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter
from qtrm_mm.training.train import prepare_donor_batch


TRANSITION_STATE_DIM = 9
RUNTIME_STATE_SUMMARY = (
    "Runtime controller state. Choose the next action from the task context "
    "and previous_observation."
)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sequence_key(row: dict[str, Any]) -> str:
    task_id = str(row.get("task_id") or row.get("id") or "")
    if not task_id:
        return ""
    prompt = str(row.get("chat_prompt") or row.get("prompt") or "")
    workspace = str(
        row.get("workspace_context")
        or row.get("workspace_text")
        or row.get("workspace_evidence")
        or ""
    )
    variant = hashlib.blake2b(
        f"{prompt}\n{workspace}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"{task_id}:{variant}"


def read_trace_sequences(
    paths: list[str | Path],
    *,
    max_sequences: int = 0,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for path in paths:
        for row in iter_jsonl(path):
            if row.get("type") != "trace_replay" or "action_target" not in row:
                continue
            key = _sequence_key(row)
            if not key:
                continue
            if key not in grouped:
                order.append(key)
            grouped[key].append(row)
    sequences: list[list[dict[str, Any]]] = []
    for key in order:
        by_step: dict[int, dict[str, Any]] = {}
        for row in sorted(grouped[key], key=lambda item: int(item.get("step", 0))):
            step = int(row.get("step", 0))
            by_step.setdefault(step, row)
        rows = [by_step[step] for step in sorted(by_step)]
        if rows:
            sequences.append(rows)
        if max_sequences > 0 and len(sequences) >= max_sequences:
            break
    return sequences


def _signal_values(row: dict[str, Any] | None) -> tuple[float, float]:
    if row is None:
        return 0.0, 0.0
    values = list(row.get("controller_signal") or [])
    world = (
        row.get("controller_world_model_signal")
        if "controller_world_model_signal" in row
        else values[0] if len(values) >= 1 else 0.0
    )
    verifier = (
        row.get("controller_verifier_signal")
        if "controller_verifier_signal" in row
        else values[1] if len(values) >= 2 else 0.0
    )
    return float(world or 0.0), float(verifier or 0.0)


def transition_state_features_for_row(
    row: dict[str, Any],
    previous_row: dict[str, Any] | None,
) -> torch.Tensor:
    previous_observation = str(row.get("previous_observation") or "")
    obs_lower = previous_observation.casefold()
    previous_reward = float(previous_row.get("reward", 0.0)) if previous_row else 0.0
    previous_world, previous_verifier = _signal_values(previous_row)
    source_count = min(8, previous_observation.count("SOURCE=")) / 8.0
    values = [
        1.0 if previous_observation else 0.0,
        1.0 if "memoryos evidence" in obs_lower or "source=" in obs_lower else 0.0,
        1.0 if "candidate_answer=" in obs_lower else 0.0,
        1.0 if "verified_candidate_answer=" in obs_lower else 0.0,
        1.0 if "unknown" in obs_lower else 0.0,
        float(source_count),
        float(previous_reward),
        float(previous_world),
        float(previous_verifier),
    ]
    return torch.tensor(values, dtype=torch.float32)


def runtime_state_training_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["state_summary"] = RUNTIME_STATE_SUMMARY
    out["hide_trace_step_from_input"] = True
    return out


def collate_trace_sequences(
    sequences: list[list[dict[str, Any]]],
    *,
    tokenizer: Any,
    seq_len: int,
    strict_runtime_state_inputs: bool = False,
) -> dict[str, torch.Tensor]:
    if not sequences:
        raise ValueError("empty sequence batch")
    batch_size = len(sequences)
    max_steps = max(len(seq) for seq in sequences)
    pad_id = int(getattr(tokenizer, "pad_id", 0))
    input_ids = torch.full((batch_size, max_steps, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    workspace_input_ids = torch.full_like(input_ids, pad_id)
    workspace_attention_mask = torch.zeros_like(input_ids)
    action_targets = torch.full((batch_size, max_steps), -100, dtype=torch.long)
    sequence_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    controller_signal_targets = torch.zeros((batch_size, max_steps, 2), dtype=torch.float32)
    transition_state_features = torch.zeros(
        (batch_size, max_steps, TRANSITION_STATE_DIM),
        dtype=torch.float32,
    )

    for batch_idx, sequence in enumerate(sequences):
        for step_idx, row in enumerate(sequence):
            previous_row = sequence[step_idx - 1] if step_idx > 0 else None
            input_row = (
                runtime_state_training_row(row)
                if bool(strict_runtime_state_inputs)
                else row
            )
            text = _render_trace_replay_action_input(input_row)
            ids = tokenizer.encode(text, seq_len)
            input_ids[batch_idx, step_idx] = ids
            attention_mask[batch_idx, step_idx] = (ids != pad_id).long()
            workspace_text = (
                row.get("workspace_context")
                or row.get("workspace_text")
                or row.get("workspace_evidence")
                or ""
            )
            if workspace_text:
                workspace_ids = tokenizer.encode(str(workspace_text), seq_len)
                workspace_input_ids[batch_idx, step_idx] = workspace_ids
                workspace_attention_mask[batch_idx, step_idx] = (workspace_ids != pad_id).long()
            action_targets[batch_idx, step_idx] = int(_trace_action_id(row["action_target"]))
            sequence_mask[batch_idx, step_idx] = True
            values = list(row.get("controller_signal") or [])
            if len(values) >= 2:
                controller_signal_targets[batch_idx, step_idx, 0] = float(values[0])
                controller_signal_targets[batch_idx, step_idx, 1] = float(values[1])
            transition_state_features[batch_idx, step_idx] = (
                transition_state_features_for_row(row, previous_row)
            )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "workspace_input_ids": workspace_input_ids,
        "workspace_attention_mask": workspace_attention_mask,
        "action_targets": action_targets,
        "sequence_mask": sequence_mask,
        "controller_signal_targets": controller_signal_targets,
        "transition_state_features": transition_state_features,
    }


def _flatten_step_batch(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
    b, t, s = batch["input_ids"].shape
    flat: dict[str, torch.Tensor] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "workspace_input_ids",
        "workspace_attention_mask",
    ):
        flat[key] = batch[key].reshape(b * t, s)
    return flat, (b, t)


@torch.no_grad()
def extract_sequence_features(
    qtrm: QTRMMultimodalModel,
    donor: QwenDonorAdapter | None,
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    use_amp: bool,
    feature_key: str,
) -> torch.Tensor:
    flat, (b, t) = _flatten_step_batch(batch)
    flat = {key: value.to(device) for key, value in flat.items()}
    model_kwargs: dict[str, torch.Tensor] = {"attention_mask": flat["attention_mask"]}
    if donor is not None:
        model_kwargs.update(prepare_donor_batch(donor, flat, return_logits=False))
    with torch.amp.autocast(
        "cuda",
        enabled=(device == "cuda" and bool(use_amp)),
        dtype=torch.bfloat16,
    ):
        outputs = qtrm(
            input_ids=flat["input_ids"],
            return_features_only=True,
            **model_kwargs,
        )
    if feature_key not in outputs:
        raise KeyError(f"feature key not returned by model: {feature_key}")
    return outputs[feature_key].detach().float().reshape(b, t, -1).cpu()


def precompute_features(
    sequences: list[list[dict[str, Any]]],
    *,
    tokenizer: Any,
    seq_len: int,
    qtrm: QTRMMultimodalModel,
    donor: QwenDonorAdapter | None,
    device: str,
    batch_size: int,
    use_amp: bool,
    feature_key: str,
    strict_runtime_state_inputs: bool = False,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        sequences,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=lambda rows: collate_trace_sequences(
            rows,
            tokenizer=tokenizer,
            seq_len=seq_len,
            strict_runtime_state_inputs=strict_runtime_state_inputs,
        ),
    )
    features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    signals: list[torch.Tensor] = []
    transition_states: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="features"):
        features.append(
            extract_sequence_features(
                qtrm,
                donor,
                batch,
                device=device,
                use_amp=use_amp,
                feature_key=feature_key,
            )
        )
        actions.append(batch["action_targets"].cpu())
        masks.append(batch["sequence_mask"].cpu())
        signals.append(batch["controller_signal_targets"].cpu())
        transition_states.append(batch["transition_state_features"].cpu())
    return {
        "features": torch.cat(features, dim=0),
        "action_targets": torch.cat(actions, dim=0),
        "sequence_mask": torch.cat(masks, dim=0),
        "controller_signal_targets": torch.cat(signals, dim=0),
        "transition_state_features": torch.cat(transition_states, dim=0),
    }


def _batches(num_items: int, batch_size: int, *, shuffle: bool) -> Iterable[torch.Tensor]:
    indices = torch.arange(num_items)
    if shuffle:
        indices = indices[torch.randperm(num_items)]
    for start in range(0, num_items, max(1, int(batch_size))):
        yield indices[start : start + max(1, int(batch_size))]


def train_transition_controller(
    controller: TransitionStateController,
    data: dict[str, torch.Tensor],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    reset_hidden: bool = False,
    state_predictor: TransitionStatePredictor | None = None,
    state_loss_weight: float = 0.0,
    controller_feature_scale: float = 1.0,
) -> list[dict[str, float]]:
    controller.to(device)
    params = list(controller.parameters())
    if state_predictor is not None:
        state_predictor.to(device)
        params.extend(state_predictor.parameters())
    opt = torch.optim.AdamW(params, lr=float(lr))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        controller.train()
        if state_predictor is not None:
            state_predictor.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_state_loss = 0.0
        total_correct = 0.0
        total_state_binary_acc = 0.0
        total_samples = 0
        for idx in _batches(data["features"].shape[0], batch_size, shuffle=True):
            features = data["features"][idx].to(device)
            controller_features = features * float(controller_feature_scale)
            targets = data["action_targets"][idx].to(device)
            mask = data["sequence_mask"][idx].to(device)
            transition_state_targets = data["transition_state_features"][idx].to(device)
            state_loss = features.sum() * 0.0
            state_metrics_samples = 0
            state_binary_accuracy = 0.0
            if controller.transition_state_dim > 0:
                if state_predictor is None:
                    transition_state_features = transition_state_targets
                else:
                    state_outputs = state_predictor(features)
                    transition_state_features = state_outputs["transition_state_features"]
                    state_loss, state_metrics = transition_state_prediction_loss(
                        state_outputs["transition_state_logits"],
                        transition_state_targets,
                        mask,
                    )
                    state_metrics_samples = int(state_metrics.samples)
                    state_binary_accuracy = float(state_metrics.binary_accuracy)
            else:
                transition_state_features = None
            prev_actions = (
                controller.teacher_forced_prev_actions(targets)
                if controller.use_prev_action
                else None
            )
            outputs = controller(
                controller_features,
                prev_actions=prev_actions,
                transition_state_features=transition_state_features,
                reset_each_step=reset_hidden,
            )
            action_loss, metrics = transition_action_loss(
                outputs["action_logits"],
                targets,
                mask,
            )
            loss = action_loss + float(state_loss_weight) * state_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu().item()) * int(metrics.samples)
            total_action_loss += float(action_loss.detach().cpu().item()) * int(metrics.samples)
            total_state_loss += float(state_loss.detach().cpu().item()) * int(metrics.samples)
            total_correct += float(metrics.accuracy) * int(metrics.samples)
            total_state_binary_acc += state_binary_accuracy * int(state_metrics_samples)
            total_samples += int(metrics.samples)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / max(1, total_samples),
                "action_loss": total_action_loss / max(1, total_samples),
                "state_loss": total_state_loss / max(1, total_samples),
                "accuracy": total_correct / max(1, total_samples),
                "state_binary_accuracy": total_state_binary_acc / max(1, total_samples),
            }
        )
    return history


@torch.no_grad()
def evaluate_transition_controller(
    controller: TransitionStateController,
    data: dict[str, torch.Tensor],
    *,
    device: str,
    batch_size: int,
    reset_each_step: bool = False,
    force_start_prev_action: bool = False,
    zero_transition_state: bool = False,
    state_predictor: TransitionStatePredictor | None = None,
    controller_feature_scale: float = 1.0,
) -> dict[str, Any]:
    controller.eval().to(device)
    if state_predictor is not None:
        state_predictor.eval().to(device)
    preds: list[int] = []
    targets_all: list[int] = []
    state_loss_sum = 0.0
    state_mae_sum = 0.0
    state_binary_acc_sum = 0.0
    state_samples = 0
    for idx in _batches(data["features"].shape[0], batch_size, shuffle=False):
        features = data["features"][idx].to(device)
        controller_features = features * float(controller_feature_scale)
        targets = data["action_targets"][idx].to(device)
        mask = data["sequence_mask"][idx].to(device)
        transition_state_targets = data["transition_state_features"][idx].to(device)
        if controller.transition_state_dim > 0:
            if state_predictor is None:
                transition_state_features = transition_state_targets
            else:
                state_outputs = state_predictor(features)
                transition_state_features = state_outputs["transition_state_features"]
                _, state_metrics = transition_state_prediction_loss(
                    state_outputs["transition_state_logits"],
                    transition_state_targets,
                    mask,
                )
                state_loss_sum += float(state_metrics.loss) * int(state_metrics.samples)
                state_mae_sum += float(state_metrics.mae) * int(state_metrics.samples)
                state_binary_acc_sum += (
                    float(state_metrics.binary_accuracy) * int(state_metrics.samples)
                )
                state_samples += int(state_metrics.samples)
        else:
            transition_state_features = None
        outputs = controller.predict_autoregressive(
            controller_features,
            transition_state_features=transition_state_features,
            reset_each_step=reset_each_step,
            force_start_prev_action=force_start_prev_action,
            zero_transition_state=zero_transition_state,
        )
        pred = outputs["action_logits"].argmax(dim=-1)
        valid = mask.to(torch.bool) & (targets >= 0)
        preds.extend(pred[valid].detach().cpu().tolist())
        targets_all.extend(targets[valid].detach().cpu().tolist())
    summary = summarize_action_predictions(preds, targets_all)
    if state_predictor is not None:
        summary["state_prediction"] = {
            "samples": int(state_samples),
            "loss": state_loss_sum / max(1, state_samples),
            "mae": state_mae_sum / max(1, state_samples),
            "binary_accuracy": state_binary_acc_sum / max(1, state_samples),
        }
    return summary


def summarize_action_predictions(preds: list[int], targets: list[int]) -> dict[str, Any]:
    total = len(targets)
    correct = sum(int(pred == target) for pred, target in zip(preds, targets))
    per_target: dict[str, dict[str, int | float]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for pred, target in zip(preds, targets):
        target_name = _action_name(target)
        pred_name = _action_name(pred)
        row = per_target.setdefault(target_name, {"total": 0, "correct": 0, "accuracy": 0.0})
        row["total"] = int(row["total"]) + 1
        row["correct"] = int(row["correct"]) + int(pred == target)
        confusion.setdefault(target_name, {})
        confusion[target_name][pred_name] = confusion[target_name].get(pred_name, 0) + 1
    for row in per_target.values():
        row["accuracy"] = float(row["correct"]) / max(1, int(row["total"]))
    return {
        "samples": total,
        "accuracy": float(correct) / max(1, total),
        "per_target": per_target,
        "confusion": confusion,
    }


def _action_name(action_id: int) -> str:
    try:
        return Action.from_id(int(action_id)).value
    except Exception:
        return f"ACTION_{int(action_id)}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a recurrent transition-state controller on trace sequences."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--use-donor", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-key", default="generation_verifier_pooled")
    parser.add_argument("--feature-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--controller-batch-size", type=int, default=64)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-eval-sequences", type=int, default=0)
    parser.add_argument("--no-prev-action", action="store_true")
    parser.add_argument("--use-transition-state", action="store_true")
    parser.add_argument("--learn-transition-state", action="store_true")
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--controller-feature-scale", type=float, default=1.0)
    parser.add_argument("--transition-state-scale", type=float, default=1.0)
    parser.add_argument("--strict-runtime-state-inputs", action="store_true")
    parser.add_argument(
        "--reset-hidden",
        action="store_true",
        help="Use explicit transition inputs only; reset recurrent hidden state at every step.",
    )
    parser.add_argument("--tokenizer-model-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    qtrm = QTRMMultimodalModel(cfg.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = qtrm.load_state_dict(state.get("model", state), strict=False)
    qtrm.eval()
    donor = QwenDonorAdapter(cfg.donor) if args.use_donor else None
    tokenizer_model_id = args.tokenizer_model_id
    if tokenizer_model_id is None:
        tokenizer_model_id = cfg.donor.model_id
    tokenizer = build_text_tokenizer(cfg.model.vocab_size, tokenizer_model_id=tokenizer_model_id)

    train_sequences = read_trace_sequences(
        [args.train_jsonl],
        max_sequences=int(args.max_train_sequences),
    )
    if not train_sequences:
        raise SystemExit("no train trace sequences found")
    eval_sequences = (
        read_trace_sequences([args.eval_jsonl], max_sequences=int(args.max_eval_sequences))
        if args.eval_jsonl
        else train_sequences
    )
    train_data = precompute_features(
        train_sequences,
        tokenizer=tokenizer,
        seq_len=cfg.train.seq_len,
        qtrm=qtrm,
        donor=donor,
        device=device,
        batch_size=args.feature_batch_size,
        use_amp=bool(cfg.train.use_amp),
        feature_key=args.feature_key,
        strict_runtime_state_inputs=bool(args.strict_runtime_state_inputs),
    )
    eval_data = (
        train_data
        if eval_sequences is train_sequences
        else precompute_features(
            eval_sequences,
            tokenizer=tokenizer,
            seq_len=cfg.train.seq_len,
            qtrm=qtrm,
            donor=donor,
            device=device,
            batch_size=args.feature_batch_size,
            use_amp=bool(cfg.train.use_amp),
            feature_key=args.feature_key,
            strict_runtime_state_inputs=bool(args.strict_runtime_state_inputs),
        )
    )
    feature_scale = float(args.feature_scale)
    if feature_scale != 1.0:
        train_data["features"] = train_data["features"] * feature_scale
        if eval_data is not train_data:
            eval_data["features"] = eval_data["features"] * feature_scale
    transition_state_scale = float(args.transition_state_scale)
    if transition_state_scale != 1.0:
        train_data["transition_state_features"] = (
            train_data["transition_state_features"] * transition_state_scale
        )
        if eval_data is not train_data:
            eval_data["transition_state_features"] = (
                eval_data["transition_state_features"] * transition_state_scale
            )

    use_transition_state = bool(args.use_transition_state or args.learn_transition_state)
    state_predictor = (
        TransitionStatePredictor(
            d_model=train_data["features"].shape[-1],
            state_dim=TRANSITION_STATE_DIM,
            hidden_dim=int(args.state_predictor_hidden_dim),
        )
        if bool(args.learn_transition_state)
        else None
    )
    controller = TransitionStateController(
        d_model=train_data["features"].shape[-1],
        num_actions=cfg.model.num_actions,
        hidden_dim=int(args.hidden_dim),
        signal_dim=cfg.model.controller_signal_dim,
        transition_state_dim=(TRANSITION_STATE_DIM if use_transition_state else 0),
        use_prev_action=not bool(args.no_prev_action),
    )
    history = train_transition_controller(
        controller,
        train_data,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.controller_batch_size),
        lr=float(args.lr),
        reset_hidden=bool(args.reset_hidden),
        state_predictor=state_predictor,
        state_loss_weight=float(args.state_loss_weight),
        controller_feature_scale=float(args.controller_feature_scale),
    )
    train_eval = evaluate_transition_controller(
        controller,
        train_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=bool(args.reset_hidden),
        state_predictor=state_predictor,
        controller_feature_scale=float(args.controller_feature_scale),
    )
    eval_full = evaluate_transition_controller(
        controller,
        eval_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=bool(args.reset_hidden),
        state_predictor=state_predictor,
        controller_feature_scale=float(args.controller_feature_scale),
    )
    eval_reset = evaluate_transition_controller(
        controller,
        eval_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=True,
        state_predictor=state_predictor,
        controller_feature_scale=float(args.controller_feature_scale),
    )
    eval_reset_transition_state = evaluate_transition_controller(
        controller,
        eval_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=True,
        force_start_prev_action=True,
        zero_transition_state=True,
        state_predictor=state_predictor,
        controller_feature_scale=float(args.controller_feature_scale),
    )
    eval_zero_transition_state = evaluate_transition_controller(
        controller,
        eval_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=bool(args.reset_hidden),
        zero_transition_state=True,
        state_predictor=state_predictor,
        controller_feature_scale=float(args.controller_feature_scale),
    )
    eval_force_start_prev_action = evaluate_transition_controller(
        controller,
        eval_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=bool(args.reset_hidden),
        force_start_prev_action=True,
        state_predictor=state_predictor,
        controller_feature_scale=float(args.controller_feature_scale),
    )
    eval_gold_transition_state = evaluate_transition_controller(
        controller,
        eval_data,
        device=device,
        batch_size=int(args.controller_batch_size),
        reset_each_step=bool(args.reset_hidden),
        controller_feature_scale=float(args.controller_feature_scale),
    )
    recurrent_drop = float(eval_full["accuracy"]) - float(eval_reset["accuracy"])
    transition_state_drop = (
        float(eval_full["accuracy"]) - float(eval_reset_transition_state["accuracy"])
    )
    zero_transition_state_drop = (
        float(eval_full["accuracy"]) - float(eval_zero_transition_state["accuracy"])
    )
    prev_action_drop = (
        float(eval_full["accuracy"]) - float(eval_force_start_prev_action["accuracy"])
    )
    state_prediction_accuracy = float(
        eval_full.get("state_prediction", {}).get(
            "binary_accuracy",
            1.0 if state_predictor is None else 0.0,
        )
    )
    state_prediction_gate = (
        state_predictor is None or state_prediction_accuracy >= 0.90
    )
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl or args.train_jsonl,
        "train_sequences": len(train_sequences),
        "eval_sequences": len(eval_sequences),
        "feature_key": args.feature_key,
        "feature_scale": feature_scale,
        "controller_feature_scale": float(args.controller_feature_scale),
        "strict_runtime_state_inputs": bool(args.strict_runtime_state_inputs),
        "use_transition_state": use_transition_state,
        "learn_transition_state": bool(args.learn_transition_state),
        "state_loss_weight": float(args.state_loss_weight),
        "state_predictor_hidden_dim": int(args.state_predictor_hidden_dim),
        "transition_state_dim": (TRANSITION_STATE_DIM if use_transition_state else 0),
        "transition_state_scale": transition_state_scale,
        "use_donor": bool(args.use_donor),
        "use_prev_action": not bool(args.no_prev_action),
        "reset_hidden": bool(args.reset_hidden),
        "controller_mode": (
            "explicit_markov_transition_state"
            if bool(args.reset_hidden)
            else "recurrent_hidden_transition_state"
        ),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "history": history,
        "train_eval": train_eval,
        "eval_full": eval_full,
        "eval_reset_each_step": eval_reset,
        "eval_reset_transition_state": eval_reset_transition_state,
        "eval_zero_transition_state": eval_zero_transition_state,
        "eval_force_start_prev_action": eval_force_start_prev_action,
        "eval_gold_transition_state": eval_gold_transition_state,
        "recurrent_drop": recurrent_drop,
        "transition_state_drop": transition_state_drop,
        "zero_transition_state_drop": zero_transition_state_drop,
        "prev_action_drop": prev_action_drop,
        "state_prediction_binary_accuracy": state_prediction_accuracy,
        "gate": {
            "status": (
                "accepted"
                if (
                    transition_state_drop >= 0.20
                    and float(eval_full["accuracy"]) >= 0.80
                    and state_prediction_gate
                )
                else "rejected"
            ),
            "min_accuracy": 0.80,
            "min_transition_state_drop": 0.20,
            "min_state_prediction_binary_accuracy": 0.90,
            "state_prediction_gate": bool(state_prediction_gate),
        },
    }
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "controller": controller.state_dict(),
            "state_predictor": (
                state_predictor.state_dict() if state_predictor is not None else None
            ),
            "summary": summary,
            "args": vars(args),
        },
        out_pt,
    )
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
