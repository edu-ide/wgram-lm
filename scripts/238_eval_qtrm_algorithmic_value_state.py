#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wgram_lm.algorithmic_value_state import (
    algorithmic_targets_from_row as _algorithmic_targets_from_row,
    apply_role_value_list_class_mode,
    numeric_source_feature_matrix,
    relative_source_slot_parity_ids,
    role_value_initial_targets_from_row,
    role_value_targets_from_row,
    score_algorithmic_sequences,
    score_role_value_predictions,
    score_typed_algorithmic_field_predictions,
    token_numeric_source_slot_ids,
    token_numeric_value_ids,
    typed_algorithmic_field_targets_from_row,
)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not isinstance(row.get("depth_targets"), dict):
                raise ValueError(f"{path}:{line_no}: missing depth_targets")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def filter_rows_by_family(
    rows: list[dict[str, Any]],
    *,
    include_family: str = "",
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    family = str(include_family or "").strip()
    filtered = [
        row
        for row in rows
        if not family or str(row.get("task_family") or row.get("category") or "") == family
    ]
    if int(max_cases) > 0:
        filtered = filtered[: int(max_cases)]
    return filtered


def algorithmic_targets_from_row_for_eval(
    row: dict[str, Any],
    *,
    num_steps: int,
    max_slots: int,
    slot_vocab_size: int,
    list_class_mode: str | None = None,
) -> tuple[list[int], list[list[int]]]:
    return _algorithmic_targets_from_row(
        row,
        num_steps=num_steps,
        max_slots=max_slots,
        slot_vocab_size=slot_vocab_size,
        list_class_mode=list_class_mode,
    )


# Test-facing alias with a stable name.
algorithmic_targets_from_row = algorithmic_targets_from_row_for_eval


def predicted_algorithmic_sequences_from_logits(
    *,
    kind_logits: Any,
    slot_logits: Any,
) -> tuple[list[int], list[list[int]]]:
    if (
        kind_logits.ndim != 3
        or slot_logits.ndim != 4
        or int(kind_logits.shape[1]) == 0
        or int(slot_logits.shape[1]) == 0
        or int(kind_logits.shape[-1]) == 0
        or int(slot_logits.shape[-1]) == 0
    ):
        return [], []
    kinds = kind_logits.detach().float().argmax(dim=-1)[0].cpu().tolist()
    slots = slot_logits.detach().float().argmax(dim=-1)[0].cpu().tolist()
    return [int(kind) for kind in kinds], [
        [int(slot) for slot in row] for row in slots
    ]


def predicted_role_values_from_logits(logits: Any) -> list[list[int]]:
    if (
        logits.ndim != 4
        or int(logits.shape[1]) == 0
        or int(logits.shape[2]) == 0
        or int(logits.shape[-1]) == 0
    ):
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def predicted_typed_algorithmic_fields_from_outputs(
    outputs: dict[str, Any],
    *,
    prefer_scalar_regression: bool = False,
) -> dict[str, Any]:
    import torch

    def _scalar_prediction(field: str, logit_key: str, value_key: str) -> list[int]:
        if bool(prefer_scalar_regression) and value_key in outputs:
            values = outputs[value_key].detach().float()[0].cpu()
            vocab = int(outputs[logit_key].shape[-1])
            denom = max(1.0, float(vocab - 1))
            return (
                (values.clamp(0.0, 1.0) * denom)
                .round()
                .to(dtype=torch.long)
                .tolist()
            )
        return (
            outputs[logit_key]
            .detach()
            .float()
            .argmax(dim=-1)[0]
            .cpu()
            .tolist()
        )

    predicted = {
        "kind": outputs["typed_algorithmic_kind_logits"]
        .detach()
        .float()
        .argmax(dim=-1)[0]
        .cpu()
        .tolist(),
        "raw_list_offsets": outputs["typed_algorithmic_raw_list_offset_logits"]
        .detach()
        .float()
        .argmax(dim=-1)[0]
        .cpu()
        .tolist(),
        "doubled_list_offsets": outputs[
            "typed_algorithmic_doubled_list_offset_logits"
        ]
        .detach()
        .float()
        .argmax(dim=-1)[0]
        .cpu()
        .tolist(),
        "scalar_coeff": _scalar_prediction(
            "scalar_coeff",
            "typed_algorithmic_scalar_coeff_logits",
            "typed_algorithmic_scalar_coeff_value",
        ),
        "scalar_offset": _scalar_prediction(
            "scalar_offset",
            "typed_algorithmic_scalar_offset_logits",
            "typed_algorithmic_scalar_offset_value",
        ),
        "scalar_residual": _scalar_prediction(
            "scalar_residual",
            "typed_algorithmic_scalar_residual_logits",
            "typed_algorithmic_scalar_residual_value",
        ),
        "final_residual": _scalar_prediction(
            "final_residual",
            "typed_algorithmic_final_residual_logits",
            "typed_algorithmic_final_residual_value",
        ),
    }
    if "typed_algorithmic_scalar_residual_delta_logits" in outputs:
        predicted["scalar_residual_delta"] = (
            outputs["typed_algorithmic_scalar_residual_delta_logits"]
            .detach()
            .float()
            .argmax(dim=-1)[0]
            .cpu()
            .tolist()
        )
    return predicted


def select_role_value_logits(
    outputs: dict[str, Any],
    *,
    use_core_role_value_state: bool = False,
    use_core_role_value_prompt_state: bool = False,
    use_core_value_delta_code: bool = False,
    use_core_primitive_role_value_state: bool = False,
    use_core_typed_register_state: bool = False,
    core_primitive_typed_register_blend: str = "",
    core_primitive_typed_register_blend_margin_bias: float = 0.0,
) -> Any:
    import torch

    blend = str(core_primitive_typed_register_blend or "").strip().lower()
    margin_bias = float(core_primitive_typed_register_blend_margin_bias)
    if blend:
        primitive_logits = outputs["core_primitive_role_value_state_logits"]
        typed_logits = outputs["core_typed_register_value_logits"]
        steps = min(int(primitive_logits.shape[1]), int(typed_logits.shape[1]))
        roles = min(int(primitive_logits.shape[2]), int(typed_logits.shape[2]))
        vocab = min(int(primitive_logits.shape[3]), int(typed_logits.shape[3]))
        primitive_logits = primitive_logits[:, :steps, :roles, :vocab]
        typed_logits = typed_logits[:, :steps, :roles, :vocab]
        if blend == "sum":
            return primitive_logits + typed_logits
        if blend == "confidence_switch":
            primitive_top2 = primitive_logits.float().topk(k=2, dim=-1).values
            typed_top2 = typed_logits.float().topk(k=2, dim=-1).values
            primitive_margin = primitive_top2[..., 0] - primitive_top2[..., 1]
            typed_margin = typed_top2[..., 0] - typed_top2[..., 1]
            use_primitive = primitive_margin >= typed_margin + margin_bias
            return torch.where(use_primitive.unsqueeze(-1), primitive_logits, typed_logits)
        if blend == "step_confidence_switch":
            primitive_top2 = primitive_logits.float().topk(k=2, dim=-1).values
            typed_top2 = typed_logits.float().topk(k=2, dim=-1).values
            primitive_margin = primitive_top2[..., 0] - primitive_top2[..., 1]
            typed_margin = typed_top2[..., 0] - typed_top2[..., 1]
            use_primitive = (
                primitive_margin.mean(dim=-1)
                >= typed_margin.mean(dim=-1) + margin_bias
            )
            return torch.where(
                use_primitive[:, :, None, None],
                primitive_logits,
                typed_logits,
            )
        raise ValueError(
            "core_primitive_typed_register_blend must be '', 'sum', "
            "'confidence_switch', or 'step_confidence_switch'"
        )
    if bool(use_core_typed_register_state):
        return outputs["core_typed_register_value_logits"]
    if bool(use_core_primitive_role_value_state):
        return outputs["core_primitive_role_value_state_logits"]
    if bool(use_core_value_delta_code):
        return outputs["core_value_delta_code_logits"]
    if bool(use_core_role_value_prompt_state):
        return outputs["core_role_value_state_prompt_logits"]
    key = (
        "core_role_value_state_logits"
        if bool(use_core_role_value_state)
        else "role_value_state_logits"
    )
    return outputs[key]


def _prepare_prompt(tokenizer: Any, prompt: str, *, max_length: int, device: str):
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def _prepare_prompt_with_token_numeric(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
    device: str,
    token_numeric_value_features: bool = False,
    disable_token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    disable_token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
):
    import torch

    numeric_enabled = bool(token_numeric_value_features) and not bool(
        disable_token_numeric_value_features
    )
    source_slot_enabled = bool(token_numeric_source_slots) and not bool(
        disable_token_numeric_source_slots
    )
    enc = tokenizer(
        str(row["prompt"]),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=bool(numeric_enabled or source_slot_enabled),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    token_numeric_ids = None
    source_slot_ids = None
    source_slot_mask = None
    if numeric_enabled or source_slot_enabled:
        offset_mapping = enc.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError("tokenizer did not return offset_mapping")
    if numeric_enabled:
        token_numeric_ids = torch.tensor(
            [
                token_numeric_value_ids(
                    row,
                    offsets=offset_mapping[0].tolist(),
                    value_vocab_size=int(token_numeric_value_vocab_size),
                )
            ],
            dtype=torch.long,
            device=device,
        )
    if source_slot_enabled:
        mode = str(token_numeric_source_slot_id_mode or "absolute_value")
        if mode == "relative_parity":
            ids, mask = relative_source_slot_parity_ids(
                row,
                max_list_len=int(token_numeric_source_slot_max_slots),
            )
        elif mode == "absolute_value":
            ids, mask = token_numeric_source_slot_ids(
                row,
                offsets=offset_mapping[0].tolist(),
                max_list_len=int(token_numeric_source_slot_max_slots),
                value_vocab_size=int(token_numeric_source_slot_vocab_size),
            )
        else:
            raise ValueError(f"unknown token numeric source slot id mode: {mode}")
        source_slot_ids = torch.tensor([ids], dtype=torch.long, device=device)
        source_slot_mask = torch.tensor([mask], dtype=torch.long, device=device)
    return input_ids, attention_mask, token_numeric_ids, source_slot_ids, source_slot_mask


def _family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "unknown")


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_kinds = sum(int(record["total_kinds"]) for record in records)
    correct_kinds = sum(int(record["correct_kinds"]) for record in records)
    total_slots = sum(int(record["total_slots"]) for record in records)
    correct_slots = sum(int(record["correct_slots"]) for record in records)
    total_content_slots = sum(int(record["total_content_slots"]) for record in records)
    correct_content_slots = sum(
        int(record["correct_content_slots"]) for record in records
    )
    total_steps = sum(int(record["total_steps"]) for record in records)
    exact_steps = sum(int(record["exact_steps"]) for record in records)
    exact_rows = sum(int(bool(record["trace_exact"])) for record in records)
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records))
        if records
        else 0.0,
        "correct_kinds": correct_kinds,
        "total_kinds": total_kinds,
        "kind_accuracy": float(correct_kinds) / float(total_kinds)
        if total_kinds
        else 0.0,
        "correct_slots": correct_slots,
        "total_slots": total_slots,
        "slot_accuracy": float(correct_slots) / float(total_slots)
        if total_slots
        else 0.0,
        "correct_content_slots": correct_content_slots,
        "total_content_slots": total_content_slots,
        "content_slot_accuracy": float(correct_content_slots)
        / float(total_content_slots)
        if total_content_slots
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
    }


def _summarize_role_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_values = sum(int(record["total_values"]) for record in records)
    correct_values = sum(int(record["correct_values"]) for record in records)
    total_steps = sum(int(record["total_steps"]) for record in records)
    exact_steps = sum(int(record["exact_steps"]) for record in records)
    exact_rows = sum(int(bool(record["trace_exact"])) for record in records)
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records))
        if records
        else 0.0,
        "correct_values": correct_values,
        "total_values": total_values,
        "value_accuracy": float(correct_values) / float(total_values)
        if total_values
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
    }


def _summarize_typed_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_fields = sum(int(record["total_fields"]) for record in records)
    correct_fields = sum(int(record["correct_fields"]) for record in records)
    total_content_fields = sum(int(record["total_content_fields"]) for record in records)
    correct_content_fields = sum(
        int(record["correct_content_fields"]) for record in records
    )
    total_steps = sum(int(record["total_steps"]) for record in records)
    exact_steps = sum(int(record["exact_steps"]) for record in records)
    exact_rows = sum(int(bool(record["trace_exact"])) for record in records)
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records))
        if records
        else 0.0,
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "field_accuracy": float(correct_fields) / float(total_fields)
        if total_fields
        else 0.0,
        "correct_content_fields": correct_content_fields,
        "total_content_fields": total_content_fields,
        "content_field_accuracy": float(correct_content_fields)
        / float(total_content_fields)
        if total_content_fields
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
    }


def evaluate_rows(
    *,
    config: str,
    checkpoint: str,
    data_jsonl: str,
    out_json: str | None = None,
    tokenizer_model_id: str = "Qwen/Qwen3.5-2B-Base",
    max_length: int | None = None,
    core_steps: int = 8,
    max_cases: int = 0,
    include_family: str = "",
    max_slots: int = 8,
    slot_vocab_size: int = 128,
    disable_transition_state: bool = False,
    disable_core_state_carry: bool = False,
    disable_core_role_value_delta: bool = False,
    disable_core_value_delta_code: bool = False,
    disable_donor_context: bool = False,
    disable_typed_algorithmic_value_state: bool = False,
    disable_typed_algorithmic_value_state_recurrent: bool = False,
    disable_core_typed_register_executor: bool = False,
    disable_core_primitive_role_value_executor: bool = False,
    disable_core_primitive_prompt_context: bool = False,
    disable_core_role_value_prompt_extract: bool = False,
    core_source_position_binder: bool = False,
    disable_core_source_position_binder: bool = False,
    core_source_position_binder_gate_min: float = 0.0,
    core_source_position_binder_state_gate_min: float = 0.0,
    core_source_position_binder_state_st: bool = False,
    core_source_position_binder_source_slots_only: bool = False,
    core_source_position_binder_raw_source_slots: bool = False,
    core_source_position_binder_query_state: bool = False,
    disable_core_source_position_binder_query_state: bool = False,
    core_source_position_binder_query_state_gate_min: float = 0.0,
    core_source_value_binder: bool = False,
    disable_core_source_value_binder: bool = False,
    core_source_value_binder_state_gate_min: float = 0.0,
    core_source_value_binder_state_st: bool = False,
    core_primitive_role_value_source_value_conditioning: bool = False,
    core_primitive_role_value_source_value_gate_min: float = 0.0,
    use_typed_algorithmic_value_state: bool = False,
    use_typed_scalar_regression_values: bool = False,
    use_role_value_state: bool = False,
    use_core_role_value_state: bool = False,
    use_core_role_value_prompt_state: bool = False,
    use_core_value_delta_code: bool = False,
    use_core_primitive_role_value_state: bool = False,
    use_core_typed_register_state: bool = False,
    core_primitive_typed_register_blend: str = "",
    core_primitive_typed_register_blend_margin_bias: float = 0.0,
    role_value_list_class_mode: str = "source_position",
    role_value_target_mode: str = "transition",
    numeric_source_features: bool = False,
    disable_numeric_source_features: bool = False,
    numeric_source_max_list_len: int = 5,
    numeric_source_value_vocab_size: int = 128,
    token_numeric_value_features: bool = False,
    disable_token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    disable_token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
    token_numeric_source_slot_gate_min: float = 0.0,
    token_numeric_source_slot_predicate_feedback: bool = False,
    token_numeric_source_slot_predicate_gate_min: float = 0.0,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import load_initial_checkpoint

    cfg = load_config(config)
    if bool(token_numeric_value_features):
        cfg.model.token_numeric_value_embedding_enabled = True
        cfg.model.token_numeric_value_vocab_size = int(token_numeric_value_vocab_size)
    if bool(token_numeric_source_slots):
        if (
            str(token_numeric_source_slot_id_mode) == "relative_parity"
            and int(token_numeric_source_slot_vocab_size) < 3
        ):
            raise ValueError(
                "relative_parity source slots require "
                "token_numeric_source_slot_vocab_size >= 3"
            )
        cfg.model.token_numeric_source_slot_embedding_enabled = True
        cfg.model.token_numeric_source_slot_vocab_size = int(
            token_numeric_source_slot_vocab_size
        )
        cfg.model.token_numeric_source_slot_max_slots = int(
            token_numeric_source_slot_max_slots
        )
        cfg.model.token_numeric_source_slot_gate_min = float(
            token_numeric_source_slot_gate_min
        )
        cfg.model.token_numeric_source_slot_predicate_feedback_enabled = bool(
            token_numeric_source_slot_predicate_feedback
        )
        cfg.model.token_numeric_source_slot_predicate_gate_min = float(
            token_numeric_source_slot_predicate_gate_min
        )
    if bool(core_source_position_binder):
        cfg.model.core_source_position_binder_enabled = True
        cfg.model.core_source_position_binder_gate_min = float(
            core_source_position_binder_gate_min
        )
        cfg.model.core_source_position_binder_state_gate_min = float(
            core_source_position_binder_state_gate_min
        )
        cfg.model.core_source_position_binder_state_straight_through = bool(
            core_source_position_binder_state_st
        )
        cfg.model.core_source_position_binder_source_slots_only = bool(
            core_source_position_binder_source_slots_only
        )
        cfg.model.core_source_position_binder_raw_source_slots_enabled = bool(
            core_source_position_binder_raw_source_slots
        )
        cfg.model.core_source_position_binder_query_state_enabled = bool(
            core_source_position_binder_query_state
        )
        cfg.model.core_source_position_binder_query_state_gate_min = float(
            core_source_position_binder_query_state_gate_min
        )
        cfg.model.core_source_value_binder_enabled = bool(core_source_value_binder)
        cfg.model.core_source_value_binder_state_gate_min = float(
            core_source_value_binder_state_gate_min
        )
        cfg.model.core_source_value_binder_state_straight_through = bool(
            core_source_value_binder_state_st
        )
        cfg.model.core_primitive_role_value_source_value_conditioning_enabled = bool(
            core_primitive_role_value_source_value_conditioning
        )
        cfg.model.core_primitive_role_value_source_value_gate_min = float(
            core_primitive_role_value_source_value_gate_min
        )
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_rows(data_jsonl)
    role_value_list_class_mode = apply_role_value_list_class_mode(
        rows,
        role_value_list_class_mode,
    )
    rows = filter_rows_by_family(
        rows,
        include_family=str(include_family or ""),
        max_cases=int(max_cases),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QTRMMultimodalModel(cfg.model).to(device)
    missing, unexpected = load_initial_checkpoint(model, checkpoint, map_location=device)
    donor = QwenDonorAdapter(cfg.donor)
    model.eval()
    max_len = int(max_length or cfg.train.seq_len)
    records: list[dict[str, Any]] = []
    old_outer_steps = int(model.cfg.outer_steps)
    model.cfg.outer_steps = int(core_steps)
    try:
        with torch.no_grad():
            for row in rows:
                (
                    input_ids,
                    attention_mask,
                    token_numeric_ids,
                    source_slot_ids,
                    source_slot_mask,
                ) = (
                    _prepare_prompt_with_token_numeric(
                        tokenizer,
                        row,
                        max_length=max_len,
                        device=device,
                        token_numeric_value_features=bool(token_numeric_value_features),
                        disable_token_numeric_value_features=bool(
                            disable_token_numeric_value_features
                        ),
                        token_numeric_value_vocab_size=int(
                            token_numeric_value_vocab_size
                        ),
                        token_numeric_source_slots=bool(token_numeric_source_slots),
                        disable_token_numeric_source_slots=bool(
                            disable_token_numeric_source_slots
                        ),
                        token_numeric_source_slot_vocab_size=int(
                            token_numeric_source_slot_vocab_size
                        ),
                        token_numeric_source_slot_max_slots=int(
                            token_numeric_source_slot_max_slots
                        ),
                        token_numeric_source_slot_id_mode=str(
                            token_numeric_source_slot_id_mode
                        ),
                    )
                )
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                with torch.amp.autocast(
                    "cuda",
                    enabled=(cfg.train.use_amp and device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    visual_features = None
                    visual_feature_mask = None
                    if bool(numeric_source_features) and not bool(
                        disable_numeric_source_features
                    ):
                        matrix, feature_mask = numeric_source_feature_matrix(
                            row,
                            visual_dim=int(cfg.model.visual_dim),
                            max_list_len=int(numeric_source_max_list_len),
                            value_vocab_size=int(numeric_source_value_vocab_size),
                        )
                        visual_features = torch.tensor(
                            matrix,
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                        visual_feature_mask = torch.tensor(
                            feature_mask,
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(0)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_numeric_value_ids=token_numeric_ids,
                        token_numeric_source_slot_ids=source_slot_ids,
                        token_numeric_source_slot_mask=source_slot_mask,
                        visual_features=visual_features,
                        workspace_attention_mask=None,
                        text_states=donor_out["text_states"].detach().to(device),
                        workspace_text_states=None,
                        disable_donor_context=bool(disable_donor_context),
                        disable_transition_state=bool(disable_transition_state),
                        disable_core_state_carry=bool(disable_core_state_carry),
                        disable_core_role_value_delta=bool(
                            disable_core_role_value_delta
                        ),
                        disable_core_value_delta_code=bool(
                            disable_core_value_delta_code
                        ),
                        disable_typed_algorithmic_value_state=bool(
                            disable_typed_algorithmic_value_state
                        ),
                        disable_typed_algorithmic_value_state_recurrent=bool(
                            disable_typed_algorithmic_value_state_recurrent
                        ),
                        disable_core_typed_register_executor=bool(
                            disable_core_typed_register_executor
                        ),
                        disable_core_primitive_role_value_executor=bool(
                            disable_core_primitive_role_value_executor
                        ),
                        disable_core_primitive_prompt_context=bool(
                            disable_core_primitive_prompt_context
                        ),
                        disable_core_role_value_prompt_extract=bool(
                            disable_core_role_value_prompt_extract
                        ),
                        disable_core_source_position_binder=bool(
                            disable_core_source_position_binder
                        ),
                        disable_core_source_position_binder_query_state=bool(
                            disable_core_source_position_binder_query_state
                        ),
                        disable_core_source_value_binder=bool(
                            disable_core_source_value_binder
                        ),
                    )
                if use_typed_algorithmic_value_state:
                    predicted_fields = predicted_typed_algorithmic_fields_from_outputs(
                        outputs,
                        prefer_scalar_regression=bool(
                            use_typed_scalar_regression_values
                        ),
                    )
                    target_fields = typed_algorithmic_field_targets_from_row(
                        row,
                        num_steps=len(predicted_fields["kind"]),
                        max_list_slots=int(
                            outputs[
                                "typed_algorithmic_raw_list_offset_logits"
                            ].shape[2]
                        ),
                        offset_vocab_size=int(
                            outputs[
                                "typed_algorithmic_raw_list_offset_logits"
                            ].shape[-1]
                        ),
                        scalar_vocab_size=int(
                            outputs["typed_algorithmic_scalar_coeff_logits"].shape[-1]
                        ),
                    )
                    score = score_typed_algorithmic_field_predictions(
                        predicted=predicted_fields,
                        target=target_fields,
                    )
                    score.update(
                        {
                            "id": row.get("id"),
                            "family": _family(row),
                            "predicted_fields": predicted_fields,
                            "target_fields": target_fields,
                        }
                    )
                elif use_role_value_state:
                    role_logits = select_role_value_logits(
                        outputs,
                        use_core_role_value_state=bool(use_core_role_value_state),
                        use_core_role_value_prompt_state=bool(
                            use_core_role_value_prompt_state
                        ),
                        use_core_value_delta_code=bool(use_core_value_delta_code),
                        use_core_primitive_role_value_state=bool(
                            use_core_primitive_role_value_state
                        ),
                        use_core_typed_register_state=bool(
                            use_core_typed_register_state
                        ),
                        core_primitive_typed_register_blend=str(
                            core_primitive_typed_register_blend or ""
                        ),
                        core_primitive_typed_register_blend_margin_bias=float(
                            core_primitive_typed_register_blend_margin_bias
                        ),
                    )
                    role_logits_empty = (
                        role_logits.ndim != 4
                        or int(role_logits.shape[1]) == 0
                        or int(role_logits.shape[2]) == 0
                        or int(role_logits.shape[-1]) == 0
                    )
                    if role_logits_empty:
                        target_steps = int(core_steps)
                        target_roles = max(
                            1,
                            int(
                                cfg.model.core_role_value_state_num_roles
                                or cfg.model.role_value_state_num_roles
                                or max_slots
                            ),
                        )
                        target_vocab = max(
                            2,
                            int(
                                cfg.model.core_role_value_state_vocab_size
                                or cfg.model.role_value_state_vocab_size
                                or slot_vocab_size
                            ),
                        )
                        predicted_values = [
                            [-1] * int(target_roles) for _ in range(int(target_steps))
                        ]
                    else:
                        target_steps = int(role_logits.shape[1])
                        target_roles = int(role_logits.shape[2])
                        target_vocab = int(role_logits.shape[-1])
                        predicted_values = predicted_role_values_from_logits(role_logits)
                    if str(role_value_target_mode) == "initial":
                        target_values = role_value_initial_targets_from_row(
                            row,
                            num_steps=int(target_steps),
                            num_roles=int(target_roles),
                            value_vocab_size=int(target_vocab),
                            list_class_mode=role_value_list_class_mode,
                        )
                    else:
                        target_values = role_value_targets_from_row(
                            row,
                            num_steps=int(target_steps),
                            num_roles=int(target_roles),
                            value_vocab_size=int(target_vocab),
                            list_class_mode=role_value_list_class_mode,
                        )
                    score = score_role_value_predictions(
                        predicted_values=predicted_values,
                        target_values=target_values,
                    )
                    score.update(
                        {
                            "id": row.get("id"),
                            "family": _family(row),
                            "predicted_values": predicted_values,
                            "target_values": target_values,
                        }
                    )
                else:
                    kind_logits = outputs["factorized_value_state_kind_logits"]
                    slot_logits = outputs["factorized_value_state_logits"]
                    slot_count = min(int(max_slots), int(slot_logits.shape[2]))
                    predicted_kinds, predicted_slots = (
                        predicted_algorithmic_sequences_from_logits(
                            kind_logits=kind_logits,
                            slot_logits=slot_logits[:, :, :slot_count, :],
                        )
                    )
                    target_kinds, target_slots = algorithmic_targets_from_row(
                        row,
                        num_steps=len(predicted_kinds),
                        max_slots=slot_count,
                        slot_vocab_size=min(int(slot_vocab_size), int(slot_logits.shape[-1])),
                    )
                    score = score_algorithmic_sequences(
                        predicted_kinds=predicted_kinds,
                        predicted_slots=predicted_slots,
                        target_kinds=target_kinds,
                        target_slots=target_slots,
                    )
                    score.update(
                        {
                            "id": row.get("id"),
                            "family": _family(row),
                            "predicted_kinds": predicted_kinds,
                            "predicted_slots": predicted_slots,
                            "target_kinds": target_kinds,
                            "target_slots": target_slots,
                        }
                    )
                records.append(score)
    finally:
        model.cfg.outer_steps = old_outer_steps
    summarize = (
        _summarize_typed_records
        if use_typed_algorithmic_value_state
        else _summarize_role_records
        if use_role_value_state
        else _summarize_records
    )
    summary = summarize(records)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(str(record["family"]), []).append(record)
    report = {
        "config": config,
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
        "core_steps": int(core_steps),
        "max_slots": int(max_slots),
        "slot_vocab_size": int(slot_vocab_size),
        "disable_transition_state": bool(disable_transition_state),
        "disable_core_state_carry": bool(disable_core_state_carry),
        "disable_core_role_value_delta": bool(disable_core_role_value_delta),
        "disable_core_value_delta_code": bool(disable_core_value_delta_code),
        "disable_typed_algorithmic_value_state": bool(
            disable_typed_algorithmic_value_state
        ),
        "disable_typed_algorithmic_value_state_recurrent": bool(
            disable_typed_algorithmic_value_state_recurrent
        ),
        "disable_core_typed_register_executor": bool(
            disable_core_typed_register_executor
        ),
        "disable_core_primitive_role_value_executor": bool(
            disable_core_primitive_role_value_executor
        ),
        "disable_core_primitive_prompt_context": bool(
            disable_core_primitive_prompt_context
        ),
        "disable_core_role_value_prompt_extract": bool(
            disable_core_role_value_prompt_extract
        ),
        "core_source_position_binder": bool(core_source_position_binder),
        "disable_core_source_position_binder": bool(
            disable_core_source_position_binder
        ),
        "core_source_position_binder_gate_min": float(
            core_source_position_binder_gate_min
        ),
        "core_source_position_binder_state_gate_min": float(
            core_source_position_binder_state_gate_min
        ),
        "core_source_position_binder_state_st": bool(
            core_source_position_binder_state_st
        ),
        "core_source_position_binder_source_slots_only": bool(
            core_source_position_binder_source_slots_only
        ),
        "core_source_position_binder_raw_source_slots": bool(
            core_source_position_binder_raw_source_slots
        ),
        "core_source_position_binder_query_state": bool(
            core_source_position_binder_query_state
        ),
        "disable_core_source_position_binder_query_state": bool(
            disable_core_source_position_binder_query_state
        ),
        "core_source_position_binder_query_state_gate_min": float(
            core_source_position_binder_query_state_gate_min
        ),
        "core_source_value_binder": bool(core_source_value_binder),
        "disable_core_source_value_binder": bool(disable_core_source_value_binder),
        "core_source_value_binder_state_gate_min": float(
            core_source_value_binder_state_gate_min
        ),
        "core_source_value_binder_state_st": bool(core_source_value_binder_state_st),
        "core_primitive_role_value_source_value_conditioning": bool(
            core_primitive_role_value_source_value_conditioning
        ),
        "core_primitive_role_value_source_value_gate_min": float(
            core_primitive_role_value_source_value_gate_min
        ),
        "numeric_source_features": bool(numeric_source_features),
        "disable_numeric_source_features": bool(disable_numeric_source_features),
        "token_numeric_value_features": bool(token_numeric_value_features),
        "disable_token_numeric_value_features": bool(
            disable_token_numeric_value_features
        ),
        "token_numeric_source_slots": bool(token_numeric_source_slots),
        "disable_token_numeric_source_slots": bool(
            disable_token_numeric_source_slots
        ),
        "token_numeric_source_slot_id_mode": str(token_numeric_source_slot_id_mode),
        "token_numeric_source_slot_predicate_feedback": bool(
            token_numeric_source_slot_predicate_feedback
        ),
        "use_typed_scalar_regression_values": bool(
            use_typed_scalar_regression_values
        ),
        "use_typed_algorithmic_value_state": bool(use_typed_algorithmic_value_state),
        "use_role_value_state": bool(use_role_value_state),
        "use_core_role_value_state": bool(use_core_role_value_state),
        "use_core_role_value_prompt_state": bool(use_core_role_value_prompt_state),
        "use_core_value_delta_code": bool(use_core_value_delta_code),
        "use_core_primitive_role_value_state": bool(
            use_core_primitive_role_value_state
        ),
        "use_core_typed_register_state": bool(use_core_typed_register_state),
        "core_primitive_typed_register_blend": str(
            core_primitive_typed_register_blend or ""
        ),
        "core_primitive_typed_register_blend_margin_bias": float(
            core_primitive_typed_register_blend_margin_bias
        ),
        "role_value_list_class_mode": str(role_value_list_class_mode),
        "role_value_target_mode": str(role_value_target_mode),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "summary": summary,
        "by_family": {
            family: summarize(family_records)
            for family, family_records in sorted(by_family.items())
        },
        "records": records,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate QTRM structured algorithmic value-state probes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--include-family",
        default="",
        help="Optional task_family/category filter applied before --max-cases.",
    )
    parser.add_argument("--max-slots", type=int, default=8)
    parser.add_argument("--slot-vocab-size", type=int, default=128)
    parser.add_argument("--disable-transition-state", action="store_true")
    parser.add_argument("--disable-core-state-carry", action="store_true")
    parser.add_argument("--disable-core-role-value-delta", action="store_true")
    parser.add_argument("--disable-core-value-delta-code", action="store_true")
    parser.add_argument(
        "--disable-donor-context",
        action="store_true",
        help=(
            "Evaluate the QTRM token/core path without injecting frozen donor "
            "hidden states as context."
        ),
    )
    parser.add_argument("--disable-typed-algorithmic-value-state", action="store_true")
    parser.add_argument(
        "--disable-typed-algorithmic-value-state-recurrent",
        action="store_true",
    )
    parser.add_argument("--disable-core-typed-register-executor", action="store_true")
    parser.add_argument(
        "--disable-core-primitive-role-value-executor",
        action="store_true",
        help=(
            "Ablate the primitive role/value recurrent executor when evaluating "
            "core_primitive_role_value_state_logits."
        ),
    )
    parser.add_argument(
        "--disable-core-primitive-prompt-context",
        action="store_true",
        help=(
            "Ablate direct prompt-context access inside the primitive role/value "
            "executor while leaving the recurrent executor itself enabled."
        ),
    )
    parser.add_argument("--disable-core-role-value-prompt-extract", action="store_true")
    parser.add_argument("--core-source-position-binder", action="store_true")
    parser.add_argument("--disable-core-source-position-binder", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-gate-min",
        type=float,
        default=0.0,
    )
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
    parser.add_argument("--core-source-position-binder-query-state", action="store_true")
    parser.add_argument(
        "--disable-core-source-position-binder-query-state",
        action="store_true",
    )
    parser.add_argument(
        "--core-source-position-binder-query-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-value-binder", action="store_true")
    parser.add_argument("--disable-core-source-value-binder", action="store_true")
    parser.add_argument(
        "--core-source-value-binder-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-value-binder-state-st", action="store_true")
    parser.add_argument(
        "--core-primitive-role-value-source-value-conditioning",
        action="store_true",
    )
    parser.add_argument(
        "--core-primitive-role-value-source-value-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--use-typed-algorithmic-value-state", action="store_true")
    parser.add_argument(
        "--use-typed-scalar-regression-values",
        action="store_true",
        help=(
            "When typed algorithmic state is used, round continuous scalar "
            "value heads instead of argmaxing scalar class logits."
        ),
    )
    parser.add_argument("--use-role-value-state", action="store_true")
    parser.add_argument("--use-core-role-value-state", action="store_true")
    parser.add_argument("--use-core-role-value-prompt-state", action="store_true")
    parser.add_argument("--use-core-value-delta-code", action="store_true")
    parser.add_argument("--use-core-primitive-role-value-state", action="store_true")
    parser.add_argument("--use-core-typed-register-state", action="store_true")
    parser.add_argument(
        "--core-primitive-typed-register-blend",
        choices=["", "sum", "confidence_switch", "step_confidence_switch"],
        default="",
        help=(
            "Diagnostic internal readout fusion between primitive role-value "
            "state logits and typed-register value logits. Both sources remain "
            "inside the same causal token/core path."
        ),
    )
    parser.add_argument(
        "--core-primitive-typed-register-blend-margin-bias",
        type=float,
        default=0.0,
        help=(
            "Extra primitive margin required before diagnostic confidence "
            "fusion selects the primitive readout over the typed-register "
            "readout. Positive values make the selector more conservative."
        ),
    )
    parser.add_argument(
        "--role-value-list-class-mode",
        choices=["source_position", "absolute"],
        default="source_position",
        help=(
            "Target encoding for plain list-transform role/value eval. Use "
            "absolute with matching training runs that store value+1 classes."
        ),
    )
    parser.add_argument(
        "--role-value-target-mode",
        choices=["transition", "initial"],
        default="transition",
        help=(
            "Target family for role/value eval. Use initial when evaluating "
            "core_role_value_state_prompt_logits trained with "
            "core_role_value_prompt_target_mode=initial."
        ),
    )
    parser.add_argument("--numeric-source-features", action="store_true")
    parser.add_argument("--disable-numeric-source-features", action="store_true")
    parser.add_argument("--numeric-source-max-list-len", type=int, default=5)
    parser.add_argument("--numeric-source-value-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-value-features", action="store_true")
    parser.add_argument("--disable-token-numeric-value-features", action="store_true")
    parser.add_argument("--token-numeric-value-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slots", action="store_true")
    parser.add_argument("--disable-token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument(
        "--token-numeric-source-slot-id-mode",
        choices=["absolute_value", "relative_parity"],
        default="absolute_value",
    )
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-feedback",
        action="store_true",
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_rows(
        config=args.config,
        checkpoint=args.checkpoint,
        data_jsonl=args.data_jsonl,
        out_json=args.out_json or None,
        tokenizer_model_id=args.tokenizer_model_id,
        max_length=args.max_length,
        core_steps=args.core_steps,
        max_cases=args.max_cases,
        include_family=args.include_family,
        max_slots=args.max_slots,
        slot_vocab_size=args.slot_vocab_size,
        disable_transition_state=args.disable_transition_state,
        disable_core_state_carry=args.disable_core_state_carry,
        disable_core_role_value_delta=args.disable_core_role_value_delta,
        disable_core_value_delta_code=args.disable_core_value_delta_code,
        disable_donor_context=args.disable_donor_context,
        disable_typed_algorithmic_value_state=(
            args.disable_typed_algorithmic_value_state
        ),
        disable_typed_algorithmic_value_state_recurrent=(
            args.disable_typed_algorithmic_value_state_recurrent
        ),
        disable_core_typed_register_executor=args.disable_core_typed_register_executor,
        disable_core_primitive_role_value_executor=(
            args.disable_core_primitive_role_value_executor
        ),
        disable_core_primitive_prompt_context=(
            args.disable_core_primitive_prompt_context
        ),
        disable_core_role_value_prompt_extract=(
            args.disable_core_role_value_prompt_extract
        ),
        core_source_position_binder=args.core_source_position_binder,
        disable_core_source_position_binder=args.disable_core_source_position_binder,
        core_source_position_binder_gate_min=(
            args.core_source_position_binder_gate_min
        ),
        core_source_position_binder_state_gate_min=(
            args.core_source_position_binder_state_gate_min
        ),
        core_source_position_binder_state_st=args.core_source_position_binder_state_st,
        core_source_position_binder_source_slots_only=(
            args.core_source_position_binder_source_slots_only
        ),
        core_source_position_binder_raw_source_slots=(
            args.core_source_position_binder_raw_source_slots
        ),
        core_source_position_binder_query_state=(
            args.core_source_position_binder_query_state
        ),
        disable_core_source_position_binder_query_state=(
            args.disable_core_source_position_binder_query_state
        ),
        core_source_position_binder_query_state_gate_min=(
            args.core_source_position_binder_query_state_gate_min
        ),
        core_source_value_binder=args.core_source_value_binder,
        disable_core_source_value_binder=args.disable_core_source_value_binder,
        core_source_value_binder_state_gate_min=(
            args.core_source_value_binder_state_gate_min
        ),
        core_source_value_binder_state_st=args.core_source_value_binder_state_st,
        core_primitive_role_value_source_value_conditioning=(
            args.core_primitive_role_value_source_value_conditioning
        ),
        core_primitive_role_value_source_value_gate_min=(
            args.core_primitive_role_value_source_value_gate_min
        ),
        use_typed_algorithmic_value_state=args.use_typed_algorithmic_value_state,
        use_typed_scalar_regression_values=args.use_typed_scalar_regression_values,
        use_role_value_state=args.use_role_value_state,
        use_core_role_value_state=args.use_core_role_value_state,
        use_core_role_value_prompt_state=args.use_core_role_value_prompt_state,
        use_core_value_delta_code=args.use_core_value_delta_code,
        use_core_primitive_role_value_state=args.use_core_primitive_role_value_state,
        use_core_typed_register_state=args.use_core_typed_register_state,
        core_primitive_typed_register_blend=args.core_primitive_typed_register_blend,
        core_primitive_typed_register_blend_margin_bias=(
            args.core_primitive_typed_register_blend_margin_bias
        ),
        role_value_list_class_mode=args.role_value_list_class_mode,
        role_value_target_mode=args.role_value_target_mode,
        numeric_source_features=args.numeric_source_features,
        disable_numeric_source_features=args.disable_numeric_source_features,
        numeric_source_max_list_len=args.numeric_source_max_list_len,
        numeric_source_value_vocab_size=args.numeric_source_value_vocab_size,
        token_numeric_value_features=args.token_numeric_value_features,
        disable_token_numeric_value_features=args.disable_token_numeric_value_features,
        token_numeric_value_vocab_size=args.token_numeric_value_vocab_size,
        token_numeric_source_slots=args.token_numeric_source_slots,
        disable_token_numeric_source_slots=args.disable_token_numeric_source_slots,
        token_numeric_source_slot_vocab_size=(
            args.token_numeric_source_slot_vocab_size
        ),
        token_numeric_source_slot_max_slots=args.token_numeric_source_slot_max_slots,
        token_numeric_source_slot_id_mode=args.token_numeric_source_slot_id_mode,
        token_numeric_source_slot_gate_min=args.token_numeric_source_slot_gate_min,
        token_numeric_source_slot_predicate_feedback=(
            args.token_numeric_source_slot_predicate_feedback
        ),
        token_numeric_source_slot_predicate_gate_min=(
            args.token_numeric_source_slot_predicate_gate_min
        ),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
