#!/usr/bin/env python3
"""Train a small text speaker on frozen state-transition thought states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qtrm_mm.eval.general_answer_interface import answer_kind, select_candidate, summarize_records
from qtrm_mm.eval.state_text_speaker import (
    IGNORE_INDEX,
    DirectVocabLogitHead,
    LowRankVocabLogitAdapter,
    PooledContextTextSpeaker,
    RestrictedVocabLogitHead,
    StateTextSpeaker,
    TrajectoryAwareTextSpeaker,
    build_answer_char_vocab,
    build_answer_token_vocab,
    decode_answer_char_indices,
    decode_answer_token_ids,
    encode_answer_char_targets,
    encode_answer_targets,
    encode_restricted_answer_targets,
    first_answer_alias,
    restricted_indices_to_token_ids,
)
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


train511 = None


def load_flexible_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, int]:
    global train511
    if train511 is None:
        try:
            train511 = _load_train511()
        except ModuleNotFoundError:
            train511 = False
    if train511:
        return train511.load_flexible_checkpoint(model, str(checkpoint_path), device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint_path}")
    model_state = model.state_dict()
    exact_state = {
        key: value
        for key, value in state.items()
        if key in model_state and hasattr(value, "shape") and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing, unexpected = model.load_state_dict(exact_state, strict=False)
    return {
        "exact": len(exact_state),
        "partial": 0,
        "skipped": len(state) - len(exact_state),
        "checkpoint_tensors": len(state),
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
    }


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row must be a JSON object at {path}:{line_no}")
        rows.append(row)
        if int(limit) > 0 and len(rows) >= int(limit):
            break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ("id", "case_id", "example_id", "uid"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def row_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "qwen_prompt", "question", "text"):
        if row.get(key):
            return str(row[key])
    return ""


TRACE_OPERATION_TO_ID = {
    "hold_final": 0,
    "add_operands": 1,
    "multiply_sum": 2,
    "subtract_offset": 3,
    "filter_even": 4,
    "double_filtered": 5,
    "first_mapping": 6,
    "second_mapping": 7,
    "not_q": 8,
    "and_with_p": 9,
    "or_with_r": 10,
    "unknown": 11,
}
ID_TO_TRACE_OPERATION = {value: key for key, value in TRACE_OPERATION_TO_ID.items()}


def trace_operation_argument_id(row: dict[str, Any], operation_name: str) -> int:
    """Return a compact numeric argument id for operation-conditioned recurrence."""
    values = source_number_values(row)
    op = str(operation_name)
    argument = 0
    if op == "add_operands":
        argument = values[1] if len(values) > 1 else 0
    elif op == "multiply_sum":
        argument = values[2] if len(values) > 2 else 0
    elif op == "subtract_offset":
        argument = values[3] if len(values) > 3 else (values[1] if len(values) > 1 else 0)
    elif op == "filter_even":
        argument = 2
    elif op == "double_filtered":
        argument = 2
    return max(0, min(9, int(abs(argument))))


def trace_operation_tensors(
    rows: list[dict[str, Any]],
    *,
    n_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact operation IDs from solver_trace rows.

    The trace depths in Stage59 are supervision checkpoints, not always dense
    loop indices. For core conditioning, the useful signal is the ordered verb
    sequence: add -> multiply -> subtract -> hold, or filter -> double -> hold.
    """
    hold_id = int(TRACE_OPERATION_TO_ID["hold_final"])
    unknown_id = int(TRACE_OPERATION_TO_ID["unknown"])
    operation_rows: list[list[int]] = []
    argument_rows: list[list[int]] = []
    for row in rows:
        ops: list[int] = []
        args: list[int] = []
        for step in row.get("solver_trace") or ():
            if not isinstance(step, dict):
                continue
            op_name = str(step.get("operation") or "unknown")
            op_id = int(TRACE_OPERATION_TO_ID.get(op_name, unknown_id))
            if op_id == hold_id and ops:
                continue
            ops.append(op_id)
            args.append(trace_operation_argument_id(row, op_name))
        if not ops:
            ops = [hold_id]
            args = [0]
        if len(ops) < int(n_steps):
            pad = int(n_steps) - len(ops)
            ops.extend([hold_id] * pad)
            args.extend([0] * pad)
        operation_rows.append(ops[: int(n_steps)])
        argument_rows.append(args[: int(n_steps)])
    return (
        torch.tensor(operation_rows, dtype=torch.long, device=device),
        torch.tensor(argument_rows, dtype=torch.long, device=device),
    )


def source_number_values(row: dict[str, Any]) -> list[int]:
    """Extract visible integers from the prompt/question without computing answers."""
    text = row_prompt(row)
    if not text:
        text = str(row.get("question") or "")
    return [int(match.group(0)) for match in re.finditer(r"\d+", text)]


def source_number_feature_tensors(
    rows: list[dict[str, Any]],
    *,
    max_slots: int,
    feature_dim: int,
    value_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(max_slots) <= 0:
        raise ValueError("max_slots must be positive")
    if int(feature_dim) <= 0:
        raise ValueError("feature_dim must be positive")
    scale = max(float(value_scale), 1.0)
    features = torch.zeros(len(rows), int(max_slots), int(feature_dim), dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), int(max_slots), dtype=torch.long, device=device)
    pos_den = float(max(1, int(max_slots) - 1))
    for row_index, row in enumerate(rows):
        for slot_index, value in enumerate(source_number_values(row)[: int(max_slots)]):
            value = int(value)
            features[row_index, slot_index, 0] = float(value) / scale
            if int(feature_dim) > 1:
                features[row_index, slot_index, 1] = 1.0 if value % 2 == 0 else 0.0
            if int(feature_dim) > 2:
                features[row_index, slot_index, 2] = float(slot_index) / pos_den
            if int(feature_dim) > 3:
                features[row_index, slot_index, 3] = 1.0
            digit_offset = 4 + abs(value) % 10
            if digit_offset < int(feature_dim):
                features[row_index, slot_index, digit_offset] = 1.0
            digit_feature_count = min(6, max(0, (int(feature_dim) - 14) // 2))
            if digit_feature_count > 0:
                digits = [int(char) for char in str(abs(value))[-digit_feature_count:]]
                digit_start = 14 + digit_feature_count - len(digits)
                presence_start = 14 + digit_feature_count
                for digit_index, digit in enumerate(digits):
                    feature_index = digit_start + digit_index
                    if feature_index < int(feature_dim):
                        features[row_index, slot_index, feature_index] = float(digit) / 9.0
                    presence_index = presence_start + digit_start + digit_index - 14
                    if presence_index < int(feature_dim):
                        features[row_index, slot_index, presence_index] = 1.0
            bucket_start = 14 + 2 * digit_feature_count
            thousands_offset = bucket_start + min(abs(value) // 1000, max(0, int(feature_dim) - bucket_start - 1))
            if 14 <= thousands_offset < int(feature_dim):
                features[row_index, slot_index, thousands_offset] = 1.0
            mask[row_index, slot_index] = 1
    return features, mask


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def summarize_answer_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    missing_prompt = 0
    missing_answer = 0
    for row in rows:
        family = str(row.get("task_family") or row.get("family") or row.get("category") or "unknown")
        answer = first_answer_alias(row)
        if not row_prompt(row):
            missing_prompt += 1
        if not answer:
            missing_answer += 1
        by_family[family] = by_family.get(family, 0) + 1
        kind = answer_kind(answer)
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        "rows": len(rows),
        "by_family": dict(sorted(by_family.items())),
        "by_answer_kind": dict(sorted(by_kind.items())),
        "missing_prompt": missing_prompt,
        "missing_answer": missing_answer,
    }


def trace_state_targets(
    rows: list[dict[str, Any]],
    *,
    max_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    batch_indices: list[int] = []
    step_indices: list[int] = []
    texts: list[str] = []
    for batch_index, row in enumerate(rows):
        for step in row.get("solver_trace") or ():
            if not isinstance(step, dict):
                continue
            if step.get("state_text") is None or step.get("depth") is None:
                continue
            try:
                step_index = int(step["depth"]) - 1
            except (TypeError, ValueError):
                continue
            step_index = max(0, min(int(max_steps) - 1, step_index))
            batch_indices.append(int(batch_index))
            step_indices.append(step_index)
            texts.append(str(step["state_text"]))
    return (
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(step_indices, dtype=torch.long, device=device),
        texts,
    )


def build_qtrm(args: argparse.Namespace, device: torch.device):
    if os.environ.get("QTRM_FORCE_TORCH_QWEN35", "0") == "1":
        try:
            import transformers.utils.import_utils as hf_import_utils

            hf_import_utils.is_flash_linear_attention_available.cache_clear()
            hf_import_utils.is_causal_conv1d_available.cache_clear()
            hf_import_utils.is_flash_linear_attention_available = lambda: False
            hf_import_utils.is_causal_conv1d_available = lambda: False
        except Exception as exc:
            print(f"[warn] could not force Qwen3.5 torch fallback: {exc}", flush=True)
    model, tokenizer = build_qwen_state_transition_model(
        args.qwen_model_id,
        freeze_qwen=True,
        device=device,
        n_operations=int(getattr(args, "n_operations", len(TRACE_OPERATION_TO_ID))),
        source_numeric_feature_dim=(
            int(getattr(args, "source_number_feature_dim", 0))
            if bool(getattr(args, "use_source_number_slots", False))
            else 0
        ),
        core_impl=args.core_impl,
        core_update=args.core_update,
        answer_path=args.answer_path,
        workspace_pooling=args.workspace_pooling,
        recurrent_readout_pooling=args.recurrent_readout_pooling,
        recurrent_readout_temperature=args.recurrent_readout_temperature,
        n_steps=args.n_steps,
        state_update_schedule=args.state_update_schedule,
        stochastic_high_level_guidance=args.stochastic_high_level_guidance,
        stochastic_high_level_scale=args.stochastic_high_level_scale,
        stochastic_high_level_min_std=args.stochastic_high_level_min_std,
        stochastic_high_level_max_std=args.stochastic_high_level_max_std,
        stochastic_high_level_eval=args.stochastic_high_level_eval,
        stochastic_posterior_guidance=args.stochastic_posterior_guidance,
        stochastic_transition_mode=args.stochastic_transition_mode,
        operation_arg_conditioning=bool(getattr(args, "operation_arg_conditioning", False)),
        working_register_enabled=bool(getattr(args, "working_register_enabled", False)),
        working_register_slots=int(getattr(args, "working_register_slots", 4)),
        working_register_update_scale=float(getattr(args, "working_register_update_scale", 0.25)),
        working_register_feedback_scale=float(getattr(args, "working_register_feedback_scale", 1.0)),
        working_register_gate_init_bias=float(getattr(args, "working_register_gate_init_bias", -2.0)),
        working_register_summary_mode=str(getattr(args, "working_register_summary_mode", "mean")),
        working_register_role_conditioning=bool(getattr(args, "working_register_role_conditioning", False)),
        working_register_role_anchor_scale=float(getattr(args, "working_register_role_anchor_scale", 0.0)),
        working_register_update_mode=str(getattr(args, "working_register_update_mode", "all")),
        working_register_source_attention=bool(getattr(args, "working_register_source_attention", False)),
        working_register_source_attention_scale=float(getattr(args, "working_register_source_attention_scale", 0.0)),
        typed_value_registers=bool(getattr(args, "typed_value_registers", False)),
        typed_value_update_scale=float(getattr(args, "typed_value_update_scale", 0.25)),
        typed_value_update_mode=str(getattr(args, "typed_value_update_mode", "residual")),
        typed_digit_registers=bool(getattr(args, "typed_digit_registers", False)),
        typed_digit_register_digits=int(getattr(args, "typed_digit_register_digits", 6)),
        typed_digit_update_scale=float(getattr(args, "typed_digit_update_scale", 0.25)),
    )
    load_stats = load_flexible_checkpoint(model, args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if bool(getattr(args, "train_qtrm_core", False)):
        for name, parameter in model.named_parameters():
            if not name.startswith("qwen."):
                parameter.requires_grad_(True)
        model.train()
        model.qwen.eval()
    return model, tokenizer, load_stats


def thought_context_for_batch(
    qtrm_model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    n_steps: int,
    device: torch.device,
    detach: bool = True,
    condition_on_trace_operations: bool = False,
    use_source_number_slots: bool = False,
    source_number_slots: int = 8,
    source_number_feature_dim: int = 32,
    source_number_value_scale: float = 10000.0,
) -> torch.Tensor:
    prompts = [row_prompt(row) for row in rows]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    operation_ids = None
    operation_arg_ids = None
    if bool(condition_on_trace_operations):
        operation_ids, operation_arg_ids = trace_operation_tensors(
            rows,
            n_steps=int(n_steps),
            device=device,
        )
    source_numeric_features = None
    source_numeric_feature_mask = None
    if bool(use_source_number_slots):
        source_numeric_features, source_numeric_feature_mask = source_number_feature_tensors(
            rows,
            max_slots=int(source_number_slots),
            feature_dim=int(source_number_feature_dim),
            value_scale=float(source_number_value_scale),
            device=device,
        )
    out = qtrm_model(
        encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        n_steps=int(n_steps),
        return_dict=True,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        source_numeric_features=source_numeric_features,
        source_numeric_feature_mask=source_numeric_feature_mask,
    )
    readout = out.get("qtrm_readout_state")
    if readout is None:
        raise RuntimeError("qtrm_model did not return qtrm_readout_state")
    workspace = out.get("qtrm_workspace")
    workspace_attention_mask = out.get("qtrm_workspace_attention_mask")
    if workspace is not None and encoded.get("attention_mask") is not None:
        attention_mask = encoded["attention_mask"]
        if (
            workspace_attention_mask is None
            and workspace.ndim == 3
            and int(workspace.size(1)) == int(attention_mask.size(1))
        ):
            workspace_attention_mask = attention_mask
    trajectory = out.get("qtrm_core_step_states")
    working_register_trajectory = out.get("qtrm_working_register_trajectory")
    typed_value_register_trajectory = out.get("qtrm_typed_value_register_trajectory")
    typed_digit_register_trajectory = out.get("qtrm_typed_digit_register_trajectory")
    if bool(detach):
        readout = readout.detach()
        trajectory = trajectory.detach() if trajectory is not None else None
        working_register_trajectory = (
            working_register_trajectory.detach() if working_register_trajectory is not None else None
        )
        typed_value_register_trajectory = (
            typed_value_register_trajectory.detach() if typed_value_register_trajectory is not None else None
        )
        typed_digit_register_trajectory = (
            typed_digit_register_trajectory.detach() if typed_digit_register_trajectory is not None else None
        )
        workspace = workspace.detach() if workspace is not None else None
        workspace_attention_mask = (
            workspace_attention_mask.detach() if workspace_attention_mask is not None else None
        )
    return {
        "readout": readout,
        "trajectory": trajectory,
        "working_register_trajectory": working_register_trajectory,
        "typed_value_register_trajectory": typed_value_register_trajectory,
        "typed_digit_register_trajectory": typed_digit_register_trajectory,
        "workspace": workspace,
        "workspace_attention_mask": workspace_attention_mask,
    }


def state_speaker_logits(
    qtrm_model: Any,
    speaker: StateTextSpeaker | TrajectoryAwareTextSpeaker | PooledContextTextSpeaker,
    thought_context: dict[str, torch.Tensor | None],
    *,
    logit_mode: str,
    context_mode: str,
    adapter: LowRankVocabLogitAdapter | None = None,
    direct_head: DirectVocabLogitHead | None = None,
    restricted_head: RestrictedVocabLogitHead | None = None,
) -> torch.Tensor:
    readout = thought_context["readout"]
    if readout is None:
        raise RuntimeError("thought_context is missing readout")
    if isinstance(speaker, (TrajectoryAwareTextSpeaker, PooledContextTextSpeaker)):
        answer_states = speaker(
            readout,
            state_trajectory=thought_context.get("trajectory")
            if context_mode in {"trajectory", "trajectory_workspace", "pooled_context"}
            else None,
            workspace=thought_context.get("workspace")
            if context_mode in {"trajectory_workspace", "pooled_context"}
            else None,
            workspace_attention_mask=thought_context.get("workspace_attention_mask")
            if context_mode in {"trajectory_workspace", "pooled_context"}
            else None,
        )
    else:
        answer_states = speaker(readout)
    if logit_mode == "direct_vocab_head":
        if direct_head is None:
            raise ValueError("direct_vocab_head requires direct_head")
        return direct_head(answer_states)
    if logit_mode == "restricted_vocab_head":
        if restricted_head is None:
            raise ValueError("restricted_vocab_head requires restricted_head")
        return restricted_head(answer_states)
    if logit_mode == "char_vocab_head":
        if restricted_head is None:
            raise ValueError("char_vocab_head requires restricted_head")
        return restricted_head(answer_states)

    _, qwen_logits = qtrm_model._lm_head_logits_from_state(answer_states)
    if logit_mode == "qwen_lm_head":
        return qwen_logits
    if logit_mode == "qwen_plus_low_rank":
        if adapter is None:
            raise ValueError("qwen_plus_low_rank requires adapter")
        return qwen_logits + adapter(answer_states)
    raise ValueError(f"unsupported speaker logit mode: {logit_mode}")


def train_epoch(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    speaker: StateTextSpeaker,
    adapter: LowRankVocabLogitAdapter | None,
    direct_head: DirectVocabLogitHead | None,
    restricted_head: RestrictedVocabLogitHead | None,
    allowed_token_ids: list[int] | None,
    allowed_chars: list[str] | None,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if bool(args.train_qtrm_core):
        qtrm_model.train()
        qtrm_model.qwen.eval()
    else:
        qtrm_model.eval()
    speaker.train()
    if adapter is not None:
        adapter.train()
    if direct_head is not None:
        direct_head.train()
    if restricted_head is not None:
        restricted_head.train()
    loader = DataLoader(rows, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_rows)
    total_loss = 0.0
    total_tokens = 0
    started = time.time()
    for batch in loader:
        thought_context = thought_context_for_batch(
            qtrm_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
            detach=not bool(args.train_qtrm_core),
            condition_on_trace_operations=bool(getattr(args, "condition_on_trace_operations", False)),
            use_source_number_slots=bool(getattr(args, "use_source_number_slots", False)),
            source_number_slots=int(getattr(args, "source_number_slots", 8)),
            source_number_feature_dim=int(getattr(args, "source_number_feature_dim", 32)),
            source_number_value_scale=float(getattr(args, "source_number_value_scale", 10000.0)),
        )
        answers = [first_answer_alias(row) for row in batch]
        if args.speaker_logit_mode == "restricted_vocab_head":
            if allowed_token_ids is None:
                raise ValueError("restricted_vocab_head requires allowed_token_ids")
            targets = encode_restricted_answer_targets(
                tokenizer,
                answers,
                allowed_token_ids=allowed_token_ids,
                max_answer_tokens=args.max_answer_tokens,
                device=device,
            )
        elif args.speaker_logit_mode == "char_vocab_head":
            if allowed_chars is None:
                raise ValueError("char_vocab_head requires allowed_chars")
            targets = encode_answer_char_targets(
                answers,
                allowed_chars=allowed_chars,
                max_answer_chars=args.max_answer_tokens,
                device=device,
            )
        else:
            targets = encode_answer_targets(
                tokenizer,
                answers,
                max_answer_tokens=args.max_answer_tokens,
                device=device,
            )
        logits = state_speaker_logits(
            qtrm_model,
            speaker,
            thought_context,
            logit_mode=args.speaker_logit_mode,
            context_mode=args.speaker_context_mode,
            adapter=adapter,
            direct_head=direct_head,
            restricted_head=restricted_head,
        )
        answer_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        loss = answer_loss
        trace_loss = None
        if float(args.trace_supervision_weight) > 0.0:
            trajectory = thought_context.get("trajectory")
            if trajectory is not None:
                batch_indices, step_indices, trace_texts = trace_state_targets(
                    batch,
                    max_steps=int(trajectory.size(1)),
                    device=device,
                )
                if trace_texts:
                    trace_readout = trajectory[batch_indices, step_indices]
                    trace_context = {
                        "readout": trace_readout,
                        "trajectory": None,
                        "workspace": None,
                        "workspace_attention_mask": None,
                    }
                    if args.speaker_logit_mode == "restricted_vocab_head":
                        if allowed_token_ids is None:
                            raise ValueError("restricted_vocab_head requires allowed_token_ids")
                        trace_targets = encode_restricted_answer_targets(
                            tokenizer,
                            trace_texts,
                            allowed_token_ids=allowed_token_ids,
                            max_answer_tokens=args.max_answer_tokens,
                            device=device,
                        )
                    elif args.speaker_logit_mode == "char_vocab_head":
                        if allowed_chars is None:
                            raise ValueError("char_vocab_head requires allowed_chars")
                        trace_targets = encode_answer_char_targets(
                            trace_texts,
                            allowed_chars=allowed_chars,
                            max_answer_chars=args.max_answer_tokens,
                            device=device,
                        )
                    else:
                        trace_targets = encode_answer_targets(
                            tokenizer,
                            trace_texts,
                            max_answer_tokens=args.max_answer_tokens,
                            device=device,
                        )
                    trace_logits = state_speaker_logits(
                        qtrm_model,
                        speaker,
                        trace_context,
                        logit_mode=args.speaker_logit_mode,
                        context_mode="final_readout",
                        adapter=adapter,
                        direct_head=direct_head,
                        restricted_head=restricted_head,
                    )
                    trace_loss = F.cross_entropy(
                        trace_logits.reshape(-1, trace_logits.size(-1)).float(),
                        trace_targets.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                    loss = loss + float(args.trace_supervision_weight) * trace_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_parameters = list(speaker.parameters())
        if adapter is not None:
            clip_parameters.extend(adapter.parameters())
        if direct_head is not None:
            clip_parameters.extend(direct_head.parameters())
        if restricted_head is not None:
            clip_parameters.extend(restricted_head.parameters())
        if bool(args.train_qtrm_core):
            clip_parameters.extend(parameter for parameter in qtrm_model.parameters() if parameter.requires_grad)
        torch.nn.utils.clip_grad_norm_(clip_parameters, float(args.grad_clip))
        optimizer.step()
        valid_tokens = int(targets.ne(IGNORE_INDEX).sum().item())
        total_loss += float(loss.detach().cpu().item()) * max(1, valid_tokens)
        total_tokens += max(1, valid_tokens)
    return {"loss": total_loss / max(1, total_tokens), "seconds": time.time() - started}


@torch.no_grad()
def evaluate(
    *,
    qtrm_model: Any,
    tokenizer: Any,
    speaker: StateTextSpeaker,
    adapter: LowRankVocabLogitAdapter | None,
    direct_head: DirectVocabLogitHead | None,
    restricted_head: RestrictedVocabLogitHead | None,
    allowed_token_ids: list[int] | None,
    allowed_chars: list[str] | None,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    speaker.eval()
    if adapter is not None:
        adapter.eval()
    if direct_head is not None:
        direct_head.eval()
    if restricted_head is not None:
        restricted_head.eval()
    loader = DataLoader(rows, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate_rows)
    records: list[dict[str, Any]] = []
    for batch in loader:
        thought_context = thought_context_for_batch(
            qtrm_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            n_steps=args.n_steps,
            device=device,
            condition_on_trace_operations=bool(getattr(args, "condition_on_trace_operations", False)),
            use_source_number_slots=bool(getattr(args, "use_source_number_slots", False)),
            source_number_slots=int(getattr(args, "source_number_slots", 8)),
            source_number_feature_dim=int(getattr(args, "source_number_feature_dim", 32)),
            source_number_value_scale=float(getattr(args, "source_number_value_scale", 10000.0)),
        )
        logits = state_speaker_logits(
            qtrm_model,
            speaker,
            thought_context,
            logit_mode=args.speaker_logit_mode,
            context_mode=args.speaker_context_mode,
            adapter=adapter,
            direct_head=direct_head,
            restricted_head=restricted_head,
        )
        pred_indices = logits.argmax(dim=-1).detach().cpu().tolist()
        for row, indices in zip(batch, pred_indices):
            if args.speaker_logit_mode == "restricted_vocab_head":
                if allowed_token_ids is None:
                    raise ValueError("restricted_vocab_head requires allowed_token_ids")
                ids = restricted_indices_to_token_ids(indices, allowed_token_ids=allowed_token_ids)
                candidate = decode_answer_token_ids(tokenizer, ids)
            elif args.speaker_logit_mode == "char_vocab_head":
                if allowed_chars is None:
                    raise ValueError("char_vocab_head requires allowed_chars")
                candidate = decode_answer_char_indices(indices, allowed_chars=allowed_chars)
            else:
                ids = indices
                candidate = decode_answer_token_ids(tokenizer, ids)
            aliases = [first_answer_alias(row)]
            selection = select_candidate([candidate], aliases, selection_mode="first")
            records.append(
                {
                    "id": row_id(row),
                    "task_family": row.get("task_family") or row.get("family") or row.get("category") or "unknown",
                    "answer_kind": answer_kind(aliases[0]),
                    "aliases": aliases,
                    "candidates": [candidate],
                    "selected": selection.selected,
                    "normalized_selected": selection.normalized_selected,
                    "exact": selection.exact,
                    "oracle_exact": selection.oracle_exact,
                    "selection_mode": selection.selection_mode,
                }
            )
    summary = summarize_records(records)
    summary.update(
        {
            "stage": "Stage59 state-transition thought-state text speaker",
            "eval_jsonl": str(args.eval_jsonl),
            "plain_language_read": (
                "This tests whether frozen recurrent thought states can be given a general text mouth. "
                "With trajectory/workspace context enabled, the mouth can read the thought path and prompt "
                "workspace instead of guessing from one final vector."
            ),
        }
    )
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/mnt/sdc1/tripleyoung/qtrm_eval/20260522_092733_LOCAL_STAGE54B_oracle_guard_mixedall_seed42/best_stochastic_oracle.pt")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--train-jsonl", default="data/filtered/pure_recursive_solver_trace_all_family_train_cases.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/pure_recursive_solver_trace_all_family_heldout_cases.jsonl")
    parser.add_argument("--out-dir", default="local_eval/stage59_state_text_speaker")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-answer-tokens", type=int, default=8)
    parser.add_argument("--speaker-logit-mode", choices=("qwen_lm_head", "qwen_plus_low_rank", "direct_vocab_head", "restricted_vocab_head", "char_vocab_head"), default="qwen_plus_low_rank")
    parser.add_argument("--speaker-context-mode", choices=("final_readout", "trajectory", "trajectory_workspace", "pooled_context"), default="final_readout")
    parser.add_argument("--speaker-rank", type=int, default=64)
    parser.add_argument("--speaker-attn-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1522)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dry-run-contract", action="store_true", help="Validate data contracts without loading Qwen/QTRM.")
    parser.add_argument("--report-train-accuracy", action="store_true", help="Also evaluate selected-answer accuracy on the speaker train rows.")
    parser.add_argument("--train-qtrm-core", action="store_true", help="Train non-Qwen QTRM parameters together with the speaker.")
    parser.add_argument("--trace-supervision-weight", type=float, default=0.0, help="Auxiliary CE weight for meaningful solver_trace state_text labels at recurrent step depths.")
    parser.add_argument("--core-impl", default="state_transition")
    parser.add_argument("--core-update", default="mlp")
    parser.add_argument("--n-operations", type=int, default=len(TRACE_OPERATION_TO_ID))
    parser.add_argument("--condition-on-trace-operations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--operation-arg-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-source-number-slots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-number-slots", type=int, default=8)
    parser.add_argument("--source-number-feature-dim", type=int, default=32)
    parser.add_argument("--source-number-value-scale", type=float, default=10000.0)
    parser.add_argument("--typed-value-registers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--typed-value-update-scale", type=float, default=0.25)
    parser.add_argument("--typed-value-update-mode", choices=("residual", "gated_delta"), default="residual")
    parser.add_argument("--typed-digit-registers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--typed-digit-register-digits", type=int, default=6)
    parser.add_argument("--typed-digit-update-scale", type=float, default=0.25)
    parser.add_argument("--answer-path", choices=("state_head", "lm_head"), default="lm_head")
    parser.add_argument("--workspace-pooling", default="sequence")
    parser.add_argument("--recurrent-readout-pooling", default="sharp_attention")
    parser.add_argument("--recurrent-readout-temperature", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=14)
    parser.add_argument("--state-update-schedule", default="nested")
    parser.add_argument("--stochastic-high-level-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-high-level-scale", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-min-std", type=float, default=1e-4)
    parser.add_argument("--stochastic-high-level-max-std", type=float, default=1.0)
    parser.add_argument("--stochastic-high-level-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-posterior-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stochastic-transition-mode", choices=("delta", "true_gram"), default="true_gram")
    parser.add_argument("--working-register-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-slots", type=int, default=4)
    parser.add_argument("--working-register-update-scale", type=float, default=0.25)
    parser.add_argument("--working-register-feedback-scale", type=float, default=1.0)
    parser.add_argument("--working-register-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--working-register-summary-mode", choices=("mean", "query_attention", "query_dot"), default="mean")
    parser.add_argument("--working-register-role-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-role-anchor-scale", type=float, default=0.0)
    parser.add_argument("--working-register-update-mode", choices=("all", "cyclic"), default="all")
    parser.add_argument("--working-register-source-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--working-register-source-attention-scale", type=float, default=0.0)
    args = parser.parse_args()

    configure_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    train_rows = load_jsonl(args.train_jsonl, limit=int(args.train_limit))
    eval_rows = load_jsonl(args.eval_jsonl, limit=int(args.eval_limit))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run_contract:
        report = {
            "stage": "Stage59 state text speaker contract dry run",
            "train": summarize_answer_contract(train_rows),
            "eval": summarize_answer_contract(eval_rows),
            "plain_language_read": (
                "This checks whether the speaker class has a real class to attend: prompts exist, "
                "answers exist, and multiple answer formats are present before spending GPU time."
            ),
        }
        (out_dir / "contract_dry_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    qtrm_model, tokenizer, load_stats = build_qtrm(args, device)
    if args.speaker_context_mode == "final_readout":
        speaker: StateTextSpeaker | TrajectoryAwareTextSpeaker | PooledContextTextSpeaker = StateTextSpeaker(
            d_state=int(qtrm_model.d_state),
            max_answer_tokens=int(args.max_answer_tokens),
        ).to(device)
    elif args.speaker_context_mode == "pooled_context":
        speaker = PooledContextTextSpeaker(
            d_state=int(qtrm_model.d_state),
            max_answer_tokens=int(args.max_answer_tokens),
        ).to(device)
    else:
        speaker = TrajectoryAwareTextSpeaker(
            d_state=int(qtrm_model.d_state),
            max_answer_tokens=int(args.max_answer_tokens),
            n_heads=int(args.speaker_attn_heads),
        ).to(device)
    adapter = None
    direct_head = None
    restricted_head = None
    allowed_token_ids = None
    allowed_chars = None
    trainable = list(speaker.parameters())
    if args.speaker_logit_mode == "qwen_plus_low_rank":
        lm_head = getattr(qtrm_model.qwen, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("qwen_plus_low_rank requires qtrm_model.qwen.lm_head")
        adapter = LowRankVocabLogitAdapter(
            d_state=int(qtrm_model.d_state),
            vocab_size=int(lm_head.weight.size(0)),
            rank=int(args.speaker_rank),
        ).to(device)
        trainable.extend(adapter.parameters())
    elif args.speaker_logit_mode == "direct_vocab_head":
        lm_head = getattr(qtrm_model.qwen, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("direct_vocab_head requires qtrm_model.qwen.lm_head for vocab size")
        direct_head = DirectVocabLogitHead(
            d_state=int(qtrm_model.d_state),
            vocab_size=int(lm_head.weight.size(0)),
        ).to(device)
        trainable.extend(direct_head.parameters())
    elif args.speaker_logit_mode == "restricted_vocab_head":
        allowed_token_ids = build_answer_token_vocab(
            tokenizer,
            [*train_rows, *eval_rows],
            max_answer_tokens=int(args.max_answer_tokens),
        )
        if not allowed_token_ids:
            raise RuntimeError("restricted answer vocab is empty")
        restricted_head = RestrictedVocabLogitHead(
            d_state=int(qtrm_model.d_state),
            restricted_vocab_size=len(allowed_token_ids),
        ).to(device)
        trainable.extend(restricted_head.parameters())
    elif args.speaker_logit_mode == "char_vocab_head":
        allowed_chars = build_answer_char_vocab([*train_rows, *eval_rows])
        restricted_head = RestrictedVocabLogitHead(
            d_state=int(qtrm_model.d_state),
            restricted_vocab_size=len(allowed_chars),
        ).to(device)
        trainable.extend(restricted_head.parameters())
    if args.train_qtrm_core:
        trainable.extend(parameter for parameter in qtrm_model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
            speaker=speaker,
            adapter=adapter,
            direct_head=direct_head,
            restricted_head=restricted_head,
            allowed_token_ids=allowed_token_ids,
            allowed_chars=allowed_chars,
            optimizer=optimizer,
            rows=train_rows,
            args=args,
            device=device,
        )
        eval_summary, eval_records = evaluate(
            qtrm_model=qtrm_model,
            tokenizer=tokenizer,
                speaker=speaker,
                adapter=adapter,
                direct_head=direct_head,
                restricted_head=restricted_head,
                allowed_token_ids=allowed_token_ids,
                allowed_chars=allowed_chars,
                rows=eval_rows,
            args=args,
            device=device,
        )
        train_eval_summary = None
        if args.report_train_accuracy:
            train_eval_summary, _ = evaluate(
                qtrm_model=qtrm_model,
                tokenizer=tokenizer,
                speaker=speaker,
                adapter=adapter,
                direct_head=direct_head,
                restricted_head=restricted_head,
                allowed_token_ids=allowed_token_ids,
                allowed_chars=allowed_chars,
                rows=train_rows,
                args=args,
                device=device,
            )
        record = {"epoch": epoch, "train": train_metrics, "eval": eval_summary}
        if train_eval_summary is not None:
            record["train_eval"] = train_eval_summary
        history.append(record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_eval_summary.get("accuracy") if train_eval_summary else None,
                    "eval_accuracy": eval_summary["accuracy"],
                    "by_family": eval_summary["by_family"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if float(eval_summary["accuracy"]) > best_acc:
            best_acc = float(eval_summary["accuracy"])
            best_payload = {
                "speaker": speaker.state_dict(),
                "adapter": adapter.state_dict() if adapter is not None else None,
                "direct_head": direct_head.state_dict() if direct_head is not None else None,
                "restricted_head": restricted_head.state_dict() if restricted_head is not None else None,
                "allowed_token_ids": allowed_token_ids,
                "allowed_chars": allowed_chars,
                "args": vars(args),
                "load_stats": load_stats,
                "epoch": epoch,
                "eval": eval_summary,
            }
            torch.save(best_payload, out_dir / "best_state_text_speaker.pt")
            (out_dir / "best_records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_records),
                encoding="utf-8",
            )

    final_summary = {
        "best_accuracy": best_acc,
        "history": history,
        "load_stats": load_stats,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "best_epoch": best_payload.get("epoch") if best_payload else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
