#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qtrm_mm.algorithmic_value_state import (
    relative_source_slot_parity_ids,
    role_value_initial_targets_from_row,
    role_value_targets_from_row,
    token_numeric_source_slot_ids,
)


def load_rows(path: str | Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def source_slot_tensors_from_offsets(
    rows: list[dict[str, Any]],
    *,
    offsets: list[list[tuple[int, int]]],
    max_slots: int,
    value_vocab_size: int,
    id_mode: str = "absolute_value",
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(rows) != len(offsets):
        raise ValueError("rows and offsets must have the same length")
    if str(id_mode) == "relative_parity":
        ids_and_masks = [
            relative_source_slot_parity_ids(
                row,
                max_list_len=int(max_slots),
            )
            for row in rows
        ]
    else:
        ids_and_masks = [
            token_numeric_source_slot_ids(
                row,
                offsets=list(row_offsets),
                max_list_len=int(max_slots),
                value_vocab_size=int(value_vocab_size),
            )
            for row, row_offsets in zip(rows, offsets)
        ]
    ids = torch.tensor(
        [item[0] for item in ids_and_masks],
        dtype=torch.long,
        device=device,
    )
    mask = torch.tensor(
        [item[1] for item in ids_and_masks],
        dtype=torch.long,
        device=device,
    )
    return ids, mask


def batch_role_value_targets(
    rows: list[dict[str, Any]],
    *,
    num_depths: int,
    num_roles: int,
    value_vocab_size: int,
    device: str,
    target_mode: str = "staged",
    source_copy_answer_role_targets: bool = False,
) -> torch.Tensor:
    if str(target_mode).lower() != "staged":
        targets = [
            [[-100] * int(num_roles) for _ in range(int(num_depths))]
            for _row in rows
        ]
    else:
        targets = [
            role_value_targets_from_row(
                row,
                num_steps=int(num_depths),
                num_roles=int(num_roles),
                value_vocab_size=int(value_vocab_size),
            )
            for row in rows
        ]
    if bool(source_copy_answer_role_targets):
        max_list_fields = max(1, (int(num_roles) - 2) // 2)
        answer_start = max_list_fields
        for row_index, row in enumerate(rows):
            if not bool(row.get("role_value_source_copy_no_doubled")):
                continue
            for step_index in range(min(int(num_depths), len(targets[row_index]))):
                source_values = list(targets[row_index][step_index][:max_list_fields])
                shifted = [-100] * int(num_roles)
                for offset, class_id in enumerate(source_values):
                    role_index = answer_start + offset
                    if role_index < int(num_roles):
                        shifted[role_index] = int(class_id)
                targets[row_index][step_index] = shifted
    return torch.tensor(targets, dtype=torch.long, device=device)


def batch_initial_role_value_targets(
    rows: list[dict[str, Any]],
    *,
    num_steps: int,
    num_roles: int,
    value_vocab_size: int,
    device: str,
) -> torch.Tensor:
    targets = [
        role_value_initial_targets_from_row(
            row,
            num_steps=int(num_steps),
            num_roles=int(num_roles),
            value_vocab_size=int(value_vocab_size),
        )
        for row in rows
    ]
    return torch.tensor(targets, dtype=torch.long, device=device)


def role_value_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, steps, roles, vocab]")
    targets = targets.to(device=logits.device, dtype=torch.long)
    steps = min(int(logits.shape[1]), int(targets.shape[1]))
    roles = min(int(logits.shape[2]), int(targets.shape[2]))
    logits = logits[:, :steps, :roles, :].float()
    targets = targets[:, :steps, :roles]
    mask = targets >= 0
    if not bool(mask.any()):
        zero = logits.sum() * 0.0
        return zero, {
            "ce": zero.detach(),
            "acc": zero.detach(),
            "step_exact": zero.detach(),
            "samples": zero.detach(),
        }
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    exact_values = []
    for batch_index in range(int(targets.shape[0])):
        for step_index in range(int(targets.shape[1])):
            row_mask = mask[batch_index, step_index]
            if bool(row_mask.any()):
                exact_values.append(
                    (
                        pred[batch_index, step_index, row_mask]
                        == targets[batch_index, step_index, row_mask]
                    )
                    .all()
                    .float()
                )
    step_exact = torch.stack(exact_values).mean() if exact_values else loss.detach() * 0.0
    samples = logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "ce": loss.detach(),
        "acc": acc.detach(),
        "step_exact": step_exact.detach(),
        "samples": samples,
    }


def role_value_previous_target_contrast_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, steps, roles, vocab]")
    targets = targets.to(device=logits.device, dtype=torch.long)
    steps = min(int(logits.shape[1]), int(targets.shape[1]))
    roles = min(int(logits.shape[2]), int(targets.shape[2]))
    vocab = int(logits.shape[-1])
    logits = logits[:, :steps, :roles, :].float()
    targets = targets[:, :steps, :roles]
    losses: list[torch.Tensor] = []
    wins: list[torch.Tensor] = []
    for batch_index in range(int(targets.shape[0])):
        for step_index in range(steps):
            previous_targets: list[int] = []
            for role_index in range(roles):
                target = int(targets[batch_index, step_index, role_index].item())
                if target < 0 or target >= vocab:
                    continue
                for negative in previous_targets:
                    if negative == target or negative < 0 or negative >= vocab:
                        continue
                    delta = (
                        logits[batch_index, step_index, role_index, target]
                        - logits[batch_index, step_index, role_index, negative]
                    )
                    losses.append(F.relu(float(margin) - delta))
                    wins.append((delta.detach() >= float(margin)).float())
                previous_targets.append(target)
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "loss": zero.detach(),
            "win_rate": zero.detach(),
            "samples": zero.detach(),
        }
    loss = torch.stack(losses).mean()
    win_rate = torch.stack(wins).mean()
    samples = logits.detach().new_tensor(float(len(losses)))
    return loss, {
        "loss": loss.detach(),
        "win_rate": win_rate.detach(),
        "samples": samples,
    }


def role_value_source_slot_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_slot_mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, steps, roles, vocab]")
    targets = targets.to(device=logits.device, dtype=torch.long)
    source_slot_mask = source_slot_mask.to(device=logits.device, dtype=torch.long)
    if source_slot_mask.ndim != 2:
        raise ValueError("source_slot_mask must have shape [batch, slots]")
    steps = min(int(logits.shape[1]), int(targets.shape[1]))
    roles = min(int(logits.shape[2]), int(targets.shape[2]))
    vocab = int(logits.shape[-1])
    batch_size = min(int(logits.shape[0]), int(targets.shape[0]), int(source_slot_mask.shape[0]))
    logits = logits[:batch_size, :steps, :roles, :].float()
    targets = targets[:batch_size, :steps, :roles]
    source_slot_mask = source_slot_mask[:batch_size]
    losses: list[torch.Tensor] = []
    wins: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        valid_source_classes = [
            slot_index + 1
            for slot_index in range(int(source_slot_mask.shape[1]))
            if int(source_slot_mask[batch_index, slot_index].item()) > 0
            and slot_index + 1 < vocab
        ]
        valid_source_set = set(valid_source_classes)
        if not valid_source_classes:
            continue
        for step_index in range(steps):
            for role_index in range(roles):
                target = int(targets[batch_index, step_index, role_index].item())
                if target <= 0 or target >= vocab or target not in valid_source_set:
                    continue
                for negative in valid_source_classes:
                    if negative == target:
                        continue
                    delta = (
                        logits[batch_index, step_index, role_index, target]
                        - logits[batch_index, step_index, role_index, negative]
                    )
                    losses.append(F.relu(float(margin) - delta))
                    wins.append((delta.detach() >= float(margin)).float())
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "loss": zero.detach(),
            "win_rate": zero.detach(),
            "samples": zero.detach(),
        }
    loss = torch.stack(losses).mean()
    win_rate = torch.stack(wins).mean()
    samples = logits.detach().new_tensor(float(len(losses)))
    return loss, {
        "loss": loss.detach(),
        "win_rate": win_rate.detach(),
        "samples": samples,
    }


def slot_predicate_ce_loss(logits: torch.Tensor, source_slot_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 3 or int(logits.shape[-1]) != 2:
        raise ValueError("slot predicate logits must have shape [batch, slots, 2]")
    ids = source_slot_ids.to(device=logits.device, dtype=torch.long)
    slots = min(int(logits.shape[1]), int(ids.shape[1]))
    logits = logits[:, :slots, :].float()
    ids = ids[:, :slots]
    mask = ids > 0
    if not bool(mask.any()):
        zero = logits.sum() * 0.0
        return zero, {"ce": zero.detach(), "acc": zero.detach(), "samples": zero.detach()}
    targets = (((ids - 1) % 2) == 0).long()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {"ce": loss.detach(), "acc": acc.detach(), "samples": samples}


def configure_model_from_args(cfg: Any, args: argparse.Namespace) -> None:
    if str(args.trainable_param_policy).strip():
        cfg.train.trainable_param_policy = str(args.trainable_param_policy).strip()
    elif bool(args.token_numeric_source_slots) and bool(args.core_source_position_binder):
        cfg.train.trainable_param_policy = (
            "token_numeric_source_slot_context_binder_primitive_role_value_state_machine"
        )
    if bool(args.token_numeric_source_slots):
        cfg.model.token_numeric_source_slot_embedding_enabled = True
        cfg.model.token_numeric_source_slot_vocab_size = int(
            args.token_numeric_source_slot_vocab_size
        )
        cfg.model.token_numeric_source_slot_max_slots = int(
            args.token_numeric_source_slot_max_slots
        )
        cfg.model.token_numeric_source_slot_gate_min = float(
            args.token_numeric_source_slot_gate_min
        )
        cfg.model.token_numeric_source_slot_predicate_feedback_enabled = bool(
            args.token_numeric_source_slot_predicate_feedback
        )
        cfg.model.token_numeric_source_slot_predicate_gate_min = float(
            args.token_numeric_source_slot_predicate_gate_min
        )
    if bool(args.core_source_position_binder):
        cfg.model.core_source_position_binder_enabled = True
        cfg.model.core_source_position_binder_gate_min = float(
            args.core_source_position_binder_gate_min
        )
        cfg.model.core_source_position_binder_state_gate_min = float(
            args.core_source_position_binder_state_gate_min
        )
        cfg.model.core_source_position_binder_state_straight_through = bool(
            args.core_source_position_binder_state_st
        )
        cfg.model.core_source_position_binder_source_slots_only = bool(
            args.core_source_position_binder_source_slots_only
        )
        cfg.model.core_source_position_binder_raw_source_slots_enabled = bool(
            args.core_source_position_binder_raw_source_slots
        )


def prompt_batch(
    *,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    device: str,
    source_slots: bool,
    source_slot_max_slots: int,
    source_slot_value_vocab_size: int,
    source_slot_id_mode: str = "absolute_value",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    enc = tokenizer(
        [str(row["prompt"]) for row in rows],
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=True,
        add_special_tokens=True,
        return_offsets_mapping=bool(source_slots),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    source_slot_ids = None
    source_slot_mask = None
    if bool(source_slots):
        offset_mapping = enc.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError("tokenizer did not return offset_mapping")
        source_slot_ids, source_slot_mask = source_slot_tensors_from_offsets(
            rows,
            offsets=[
                [(int(start), int(end)) for start, end in row_offsets.tolist()]
                for row_offsets in offset_mapping
            ],
            max_slots=int(source_slot_max_slots),
            value_vocab_size=int(source_slot_value_vocab_size),
            id_mode=str(source_slot_id_mode),
            device=device,
        )
    return input_ids, attention_mask, source_slot_ids, source_slot_mask


def metric_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().cpu().item())
    return float(value)


def save_trainable_checkpoint(
    *,
    model: Any,
    path: Path,
    trainable_names: list[str],
    base_checkpoint: str,
    metadata: dict[str, Any],
) -> None:
    state = model.state_dict()
    payload = {
        "model": {
            name: state[name].detach().cpu()
            for name in trainable_names
            if name in state
        },
        "base_checkpoint": str(base_checkpoint or ""),
        "trainable_param_policy": str(metadata.get("trainable_param_policy") or ""),
        "format": "qtrm_trainable_delta_v1",
        "training_metadata": metadata,
    }
    torch.save(payload, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-train the integrated QTRM source-position pointer path. "
            "This is an L2 prerequisite repair for paired hard-negative "
            "template collapse, not a general-LM promotion script."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=324)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--trainable-param-policy", default="")
    parser.add_argument("--token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument(
        "--token-numeric-source-slot-id-mode",
        choices=["absolute_value", "relative_parity"],
        default="absolute_value",
    )
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument("--token-numeric-source-slot-parity-ce-weight", type=float, default=0.0)
    parser.add_argument("--token-numeric-source-slot-predicate-feedback", action="store_true")
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder", action="store_true")
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder-state-st", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-source-slots-only",
        action="store_true",
    )
    parser.add_argument(
        "--core-source-position-binder-raw-source-slots",
        action="store_true",
    )
    parser.add_argument("--core-role-value-prompt-ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--core-role-value-prompt-target-mode",
        choices=["initial", "staged"],
        default="initial",
    )
    parser.add_argument(
        "--core-role-value-source-copy-answer-role-targets",
        action="store_true",
        help=(
            "For source-copy rows, train the same final answer role block that "
            "the pointer/copy probe and renderer read."
        ),
    )
    parser.add_argument("--core-primitive-role-value-state-ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--core-primitive-role-value-order-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Margin loss that makes each later source-position role beat the "
            "gold source classes used by earlier roles. This targets the "
            "paired source-copy failure where role 5 copies role 4's source."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-order-contrast-margin",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--core-primitive-role-value-source-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Margin loss that makes each gold source-position class beat every "
            "other valid source slot for the same role. This is a pointer/copy "
            "hard-negative pressure, not an answer renderer."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-source-margin",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--core-primitive-role-value-update-gate-bce-weight",
        type=float,
        default=0.0,
    )
    return parser


def main() -> int:
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter
    from qtrm_mm.training.train import (
        configure_trainable_parameters,
        load_initial_checkpoint,
    )

    args = build_arg_parser().parse_args()
    if int(args.row_batch_size) <= 0:
        raise ValueError("--row-batch-size must be positive")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = load_config(args.config)
    configure_model_from_args(cfg, args)
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_rows(args.data_jsonl, max_rows=int(args.max_train_rows))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model).to(device)
    if args.init_checkpoint:
        load_initial_checkpoint(model, args.init_checkpoint, map_location=device)
    elif not bool(args.allow_random_init):
        raise ValueError("--init-checkpoint is required unless --allow-random-init is set")
    donor = QwenDonorAdapter(cfg.donor)

    trainable_names = configure_trainable_parameters(
        model,
        cfg.train.trainable_param_policy,
    )
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected")
    opt = torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.use_amp and device == "cuda"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "qtrm_source_pointer_batch_v1",
        "command": list(sys.argv),
        "config": str(args.config),
        "init_checkpoint": str(args.init_checkpoint or ""),
        "seed": int(args.seed),
        "train_data": str(args.data_jsonl),
        "row_batch_size": int(args.row_batch_size),
        "core_steps": int(args.core_steps),
        "trainable_param_policy": str(cfg.train.trainable_param_policy),
    }
    reports: list[dict[str, Any]] = []
    old_outer_steps = int(model.cfg.outer_steps)
    model.cfg.outer_steps = int(args.core_steps)
    try:
        for step in range(1, int(args.steps) + 1):
            model.train()
            batch = random.choices(rows, k=int(args.row_batch_size))
            input_ids, attention_mask, source_slot_ids, source_slot_mask = prompt_batch(
                tokenizer=tokenizer,
                rows=batch,
                max_length=int(args.max_length),
                device=device,
                source_slots=bool(args.token_numeric_source_slots),
                source_slot_max_slots=int(args.token_numeric_source_slot_max_slots),
                source_slot_value_vocab_size=int(
                    args.token_numeric_source_slot_vocab_size
                ),
                source_slot_id_mode=str(args.token_numeric_source_slot_id_mode),
            )
            with torch.no_grad():
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
            text_states = donor_out["text_states"].detach().to(device=device)
            donor_attention_mask = donor_out.get("attention_mask")
            if donor_attention_mask is not None:
                donor_attention_mask = donor_attention_mask.to(device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=(cfg.train.use_amp and device == "cuda"),
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    text_states=text_states,
                    token_numeric_source_slot_ids=source_slot_ids,
                    token_numeric_source_slot_mask=source_slot_mask,
                    return_core_depth_logits=False,
                    return_core_depth_text_logits=False,
                )
                losses: list[torch.Tensor] = []
                metrics: dict[str, torch.Tensor] = {}
                if (
                    float(args.token_numeric_source_slot_parity_ce_weight) != 0.0
                    and source_slot_ids is not None
                ):
                    parity_loss, parity_metrics = slot_predicate_ce_loss(
                        outputs["token_numeric_source_slot_parity_logits"],
                        source_slot_ids,
                    )
                    losses.append(
                        float(args.token_numeric_source_slot_parity_ce_weight)
                        * parity_loss
                    )
                    metrics.update(
                        {f"source_slot_parity_{key}": value for key, value in parity_metrics.items()}
                    )
                if (
                    float(args.token_numeric_source_slot_predicate_ce_weight) != 0.0
                    and source_slot_ids is not None
                ):
                    predicate_loss, predicate_metrics = slot_predicate_ce_loss(
                        outputs["token_numeric_source_slot_predicate_logits"],
                        source_slot_ids,
                    )
                    losses.append(
                        float(args.token_numeric_source_slot_predicate_ce_weight)
                        * predicate_loss
                    )
                    metrics.update(
                        {f"source_slot_predicate_{key}": value for key, value in predicate_metrics.items()}
                    )
                if float(args.core_role_value_prompt_ce_weight) != 0.0:
                    prompt_logits = outputs["core_role_value_state_prompt_logits"]
                    if str(args.core_role_value_prompt_target_mode) == "initial":
                        prompt_targets = batch_initial_role_value_targets(
                            batch,
                            num_steps=int(prompt_logits.shape[1]),
                            num_roles=int(prompt_logits.shape[2]),
                            value_vocab_size=int(prompt_logits.shape[3]),
                            device=device,
                        )
                    else:
                        prompt_targets = batch_role_value_targets(
                        batch,
                        num_depths=int(prompt_logits.shape[1]),
                        num_roles=int(prompt_logits.shape[2]),
                        value_vocab_size=int(prompt_logits.shape[3]),
                        device=device,
                        source_copy_answer_role_targets=bool(
                            args.core_role_value_source_copy_answer_role_targets
                        ),
                    )
                    prompt_loss, prompt_metrics = role_value_ce_loss(
                        prompt_logits,
                        prompt_targets,
                    )
                    losses.append(float(args.core_role_value_prompt_ce_weight) * prompt_loss)
                    metrics.update(
                        {f"prompt_role_value_{key}": value for key, value in prompt_metrics.items()}
                    )
                if float(args.core_primitive_role_value_state_ce_weight) != 0.0:
                    primitive_logits = outputs["core_primitive_role_value_state_logits"]
                    primitive_targets = batch_role_value_targets(
                        batch,
                        num_depths=int(primitive_logits.shape[1]),
                        num_roles=int(primitive_logits.shape[2]),
                        value_vocab_size=int(primitive_logits.shape[3]),
                        device=device,
                        source_copy_answer_role_targets=bool(
                            args.core_role_value_source_copy_answer_role_targets
                        ),
                    )
                    primitive_loss, primitive_metrics = role_value_ce_loss(
                        primitive_logits,
                        primitive_targets,
                    )
                    losses.append(
                        float(args.core_primitive_role_value_state_ce_weight)
                        * primitive_loss
                    )
                    metrics.update(
                        {
                            f"primitive_role_value_{key}": value
                            for key, value in primitive_metrics.items()
                        }
                    )
                if (
                    float(args.core_primitive_role_value_order_contrast_weight)
                    != 0.0
                ):
                    primitive_logits = outputs["core_primitive_role_value_state_logits"]
                    primitive_targets = batch_role_value_targets(
                        batch,
                        num_depths=int(primitive_logits.shape[1]),
                        num_roles=int(primitive_logits.shape[2]),
                        value_vocab_size=int(primitive_logits.shape[3]),
                        device=device,
                        source_copy_answer_role_targets=bool(
                            args.core_role_value_source_copy_answer_role_targets
                        ),
                    )
                    order_contrast, order_metrics = (
                        role_value_previous_target_contrast_loss(
                            primitive_logits,
                            primitive_targets,
                            margin=float(
                                args.core_primitive_role_value_order_contrast_margin
                            ),
                        )
                    )
                    losses.append(
                        float(args.core_primitive_role_value_order_contrast_weight)
                        * order_contrast
                    )
                    metrics.update(
                        {
                            f"primitive_role_value_order_contrast_{key}": value
                            for key, value in order_metrics.items()
                        }
                    )
                if (
                    float(args.core_primitive_role_value_source_margin_weight)
                    != 0.0
                    and source_slot_mask is not None
                ):
                    primitive_logits = outputs["core_primitive_role_value_state_logits"]
                    primitive_targets = batch_role_value_targets(
                        batch,
                        num_depths=int(primitive_logits.shape[1]),
                        num_roles=int(primitive_logits.shape[2]),
                        value_vocab_size=int(primitive_logits.shape[3]),
                        device=device,
                        source_copy_answer_role_targets=bool(
                            args.core_role_value_source_copy_answer_role_targets
                        ),
                    )
                    source_margin, source_margin_metrics = (
                        role_value_source_slot_margin_loss(
                            primitive_logits,
                            primitive_targets,
                            source_slot_mask,
                            margin=float(
                                args.core_primitive_role_value_source_margin
                            ),
                        )
                    )
                    losses.append(
                        float(args.core_primitive_role_value_source_margin_weight)
                        * source_margin
                    )
                    metrics.update(
                        {
                            f"primitive_role_value_source_margin_{key}": value
                            for key, value in source_margin_metrics.items()
                        }
                    )
                if not losses:
                    raise ValueError("no active losses")
                loss = torch.stack(losses).sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            if step == 1 or step % int(args.log_every) == 0:
                report = {
                    "step": step,
                    "loss": float(loss.detach().cpu().item()),
                    **{key: metric_float(value) for key, value in metrics.items()},
                }
                reports.append(report)
                print(json.dumps(report, ensure_ascii=False))
            if int(args.save_every) > 0 and step % int(args.save_every) == 0:
                save_trainable_checkpoint(
                    model=model,
                    path=out_dir / f"step_{step:06d}.pt",
                    trainable_names=trainable_names,
                    base_checkpoint=str(args.init_checkpoint or ""),
                    metadata={**metadata, "step": step},
                )
        save_trainable_checkpoint(
            model=model,
            path=out_dir / "last.pt",
            trainable_names=trainable_names,
            base_checkpoint=str(args.init_checkpoint or ""),
            metadata={**metadata, "step": int(args.steps)},
        )
    finally:
        model.cfg.outer_steps = old_outer_steps

    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "decision": "trained",
                "accepted": False,
                "target_level": "L2 prerequisite repair",
                "major_bottleneck": "integrated source-position batch coupling",
                "reports": reports,
                "checkpoint": str(out_dir / "last.pt"),
                **metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
