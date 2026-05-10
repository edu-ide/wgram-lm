#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from qtrm_mm.algorithmic_value_state import (
    algorithmic_targets_from_row,
    apply_role_value_list_class_mode,
    mixed_even_offsets,
    numeric_source_feature_matrix,
    role_value_targets_from_row,
    row_input_list,
    row_mixed_list_base,
    token_numeric_source_slot_ids,
    token_numeric_source_slot_token_ids,
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
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: depth-supervised rows must not include evidence")
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            answer = row.get("chosen") or row.get("answer")
            if not answer:
                aliases = row.get("answer_aliases")
                if isinstance(aliases, list) and aliases and str(aliases[0]).strip():
                    row["answer"] = str(aliases[0]).strip()
                else:
                    raise ValueError(f"{path}:{line_no}: missing chosen/answer")
            rows.append(row)
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return rows


def _source_even_position_signature(row: dict[str, Any]) -> tuple[int, ...]:
    raw = row.get("source_even_position_signature")
    if not isinstance(raw, list):
        return ()
    signature: list[int] = []
    for value in raw:
        try:
            signature.append(int(value))
        except (TypeError, ValueError):
            return ()
    return tuple(signature)


def build_paired_hard_negative_lookup(rows: list[dict[str, Any]]) -> dict[int, int]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group_id = str(row.get("pair_group_id") or "").strip()
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(index)

    lookup: dict[int, int] = {}
    for indices in groups.values():
        if len(indices) < 2:
            continue
        signatures = {
            index: _source_even_position_signature(rows[index])
            for index in indices
        }
        for index in indices:
            signature = signatures[index]
            if not signature:
                continue
            for candidate in indices:
                if candidate == index:
                    continue
                candidate_signature = signatures[candidate]
                if candidate_signature and candidate_signature != signature:
                    lookup[index] = candidate
                    break
    return lookup


def answer_first_token_id(tokenizer: Any, answer: str) -> int:
    token_ids = answer_token_ids(tokenizer, answer)
    return int(token_ids[0])


def answer_content_first_token_id(tokenizer: Any, answer: str) -> int:
    stripped = str(answer).strip()
    token_ids = tokenizer.encode(stripped, add_special_tokens=False)
    if not token_ids:
        token_ids = answer_token_ids(tokenizer, answer)
    if not token_ids:
        raise ValueError(f"answer produced no content tokens: {answer!r}")
    return int(token_ids[0])


def answer_token_ids(tokenizer: Any, answer: str) -> list[int]:
    token_ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"answer produced no tokens: {answer!r}")
    return [int(token_id) for token_id in token_ids]


def causal_prefix_answer_token_ids(
    tokenizer: Any,
    answer: str,
    *,
    skip_leading_whitespace_targets: bool = False,
) -> list[int]:
    if bool(skip_leading_whitespace_targets):
        token_ids = tokenizer.encode(str(answer).strip(), add_special_tokens=False)
        if token_ids:
            return [int(token_id) for token_id in token_ids]
    return answer_token_ids(tokenizer, answer)


def _choice_margin_normalize_text(text: Any) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def choice_margin_rejected_texts(
    row: dict[str, Any],
    *,
    current_answer: str | None = None,
) -> list[str]:
    explicit = row.get("rejected")
    if explicit is not None and str(explicit).strip():
        explicit_text = str(explicit).strip()
        if current_answer is not None and (
            _choice_margin_normalize_text(explicit_text)
            == _choice_margin_normalize_text(current_answer)
        ):
            return []
        return [explicit_text]
    choices = row.get("choices")
    if not isinstance(choices, list):
        return []
    accepted_texts = [
        row.get("chosen"),
        row.get("answer"),
        current_answer,
        *(row.get("answer_aliases") or []),
    ]
    accepted = {
        _choice_margin_normalize_text(text)
        for text in accepted_texts
        if text is not None and str(text).strip()
    }
    rejected: list[str] = []
    seen: set[str] = set()
    for choice in choices:
        text = str(choice).strip()
        normalized = _choice_margin_normalize_text(text)
        if not text or not normalized or normalized in accepted or normalized in seen:
            continue
        rejected.append(text)
        seen.add(normalized)
    return rejected


def _row_family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "")


def _final_answer_text(row: dict[str, Any]) -> str:
    aliases = row.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    return str(row.get("chosen") or row.get("answer") or "")


def tail_negative_rejected_texts(
    row: dict[str, Any],
    *,
    current_answer: str | None = None,
    family_filter: str = "mixed_list_arithmetic",
) -> list[str]:
    family = _row_family(row)
    allowed_family = str(family_filter or "").strip()
    if allowed_family and family != allowed_family:
        return []

    final_answer = _final_answer_text(row)
    if current_answer is not None and (
        _choice_margin_normalize_text(current_answer)
        != _choice_margin_normalize_text(final_answer)
    ):
        return []

    depth_targets = row.get("depth_targets")
    if not isinstance(depth_targets, dict):
        return []
    finality = row.get("transition_finality_targets")
    rejected: str | None = None
    if isinstance(finality, dict):
        final_depths = sorted(
            int(depth)
            for depth, value in finality.items()
            if float(value) > 0.0
        )
        if final_depths:
            preterminal_depth = final_depths[0] - 1
            rejected = depth_targets.get(str(preterminal_depth))
    if rejected is None and family == "mixed_list_arithmetic":
        rejected = depth_targets.get("3")
    if rejected is None:
        return []
    rejected_text = str(rejected).strip()
    if not rejected_text:
        return []
    if (
        _choice_margin_normalize_text(rejected_text)
        == _choice_margin_normalize_text(final_answer)
    ):
        return []
    return [rejected_text]


def _parse_int_answer_text(text: Any) -> int | None:
    stripped = str(text).strip()
    if not stripped or "," in stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def subtract_tail_counterfactual_rejected_texts(
    row: dict[str, Any],
    *,
    current_answer: str | None = None,
    family_filter: str = "mixed_list_arithmetic",
) -> list[str]:
    family = _row_family(row)
    allowed_family = str(family_filter or "").strip()
    if allowed_family and family != allowed_family:
        return []

    final_answer = _final_answer_text(row)
    if current_answer is not None and (
        _choice_margin_normalize_text(current_answer)
        != _choice_margin_normalize_text(final_answer)
    ):
        return []

    final_value = _parse_int_answer_text(final_answer)
    if final_value is None:
        return []

    rejected: list[str] = []
    seen: set[str] = {_choice_margin_normalize_text(final_answer)}
    for text in tail_negative_rejected_texts(
        row,
        current_answer=current_answer,
        family_filter=family_filter,
    ):
        normalized = _choice_margin_normalize_text(text)
        if normalized and normalized not in seen:
            rejected.append(str(text))
            seen.add(normalized)

    candidates = [final_value - 1, final_value + 1]
    preterminal_value = _parse_int_answer_text(rejected[0]) if rejected else None
    try:
        offset = int(row.get("mixed_offset"))
    except (TypeError, ValueError):
        offset = None
    if preterminal_value is not None and offset is not None:
        candidates.extend(
            [
                preterminal_value - offset - 1,
                preterminal_value - offset + 1,
            ]
        )

    for value in candidates:
        text = str(int(value))
        normalized = _choice_margin_normalize_text(text)
        if normalized and normalized not in seen:
            rejected.append(text)
            seen.add(normalized)
    return rejected


VALUE_STATE_CHAR_TO_ID = {str(index): index for index in range(10)}
VALUE_STATE_CHAR_TO_ID.update({",": 10, "-": 11})


def value_state_token_ids(text: str) -> list[int] | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    ids: list[int] = []
    for char in stripped:
        if char not in VALUE_STATE_CHAR_TO_ID:
            return None
        ids.append(int(VALUE_STATE_CHAR_TO_ID[char]))
    return ids


def target_for_core_steps(
    row: dict[str, Any],
    core_steps: int,
    *,
    target_mode: str = "staged",
) -> str:
    final_answer = str(row.get("chosen") or row.get("answer"))
    if str(target_mode).lower() == "final":
        return final_answer
    if str(target_mode).lower() != "staged":
        raise ValueError("target_mode must be 'staged' or 'final'")
    depth_targets = row.get("depth_targets")
    if isinstance(depth_targets, dict):
        target = depth_targets.get(str(int(core_steps)))
        if target:
            return str(target)
    return final_answer


def _row_temporal_spatial_context(row: dict[str, Any], *, device: str):
    value = row.get("temporal_spatial_context")
    if value is None:
        return None
    import torch

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return tensor.view(1, -1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        return tensor
    raise ValueError(
        "temporal_spatial_context must be a vector, token list, or single-batch token list"
    )


def row_numeric_source_visual_tensors(
    row: dict[str, Any],
    *,
    visual_dim: int,
    max_list_len: int,
    value_vocab_size: int,
    device: str,
):
    import torch

    features, mask = numeric_source_feature_matrix(
        row,
        visual_dim=int(visual_dim),
        max_list_len=int(max_list_len),
        value_vocab_size=int(value_vocab_size),
    )
    return (
        torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0),
    )


def parse_depth_steps(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        steps = [int(part) for part in parts]
    else:
        steps = [int(part) for part in value]
    if not steps:
        raise ValueError("at least one depth step is required")
    if any(step <= 0 for step in steps):
        raise ValueError("depth steps must be positive")
    return steps


def scheduled_row_and_core_steps(
    step: int,
    *,
    row_count: int,
    depth_steps: list[int],
    row_indices: list[int] | None = None,
) -> tuple[int, int]:
    effective_count = len(row_indices) if row_indices is not None else int(row_count)
    if effective_count <= 0:
        raise ValueError("row_count must be positive")
    if not depth_steps:
        raise ValueError("depth_steps must not be empty")
    depth_index = int(step) % len(depth_steps)
    curriculum_index = (int(step) // len(depth_steps)) % effective_count
    row_index = int(row_indices[curriculum_index]) if row_indices is not None else curriculum_index
    return row_index, int(depth_steps[depth_index])


def parse_family_repeat_spec(value: str | None) -> dict[str, int]:
    repeats: dict[str, int] = {}
    if not value:
        return repeats
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("family repeat entries must look like family=repeat")
        family, repeat_text = item.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError("family repeat entry has empty family name")
        repeat = int(repeat_text.strip())
        if repeat <= 0:
            raise ValueError("family repeat must be positive")
        repeats[family] = repeat
    return repeats


def build_curriculum_indices(
    rows: list[dict[str, Any]],
    family_repeats: dict[str, int] | None,
) -> list[int]:
    repeats = family_repeats or {}
    indices: list[int] = []
    for idx, row in enumerate(rows):
        family = str(row.get("task_family") or row.get("category") or "")
        repeat = max(1, int(repeats.get(family, 1)))
        indices.extend([idx] * repeat)
    if not indices:
        raise ValueError("curriculum has no rows")
    return indices


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train QTRM recursive core depth logits on prompt-only pure reasoning rows. "
            "This avoids MemoryOS/retrieval shortcuts and gives the core a direct "
            "per-depth answer pressure."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument(
        "--shuffle-rows",
        action="store_true",
        help=(
            "Deterministically shuffle training rows with --seed before the "
            "cyclic depth-step scheduler. This prevents short gate runs from "
            "seeing only the first few sorted rows."
        ),
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help=(
            "Explicitly start from a freshly initialized QTRM model when no "
            "--init-checkpoint is provided. This is for checkpoint-loss recovery "
            "and matched random-init baselines only."
        ),
    )
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--target-logit-positions-only",
        action="store_true",
        help=(
            "For final-path causal-prefix training, ask the model to compute "
            "full-vocab logits only at supervised target positions. This keeps "
            "the canonical LM head loss while avoiding full-sequence vocab "
            "logits on memory-constrained GPUs."
        ),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--depth-steps", default="1,2,4,8")
    parser.add_argument("--target-mode", choices=["staged", "final"], default="staged")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--trainable-param-policy",
        default="",
        help="Override train.trainable_param_policy from the config for one run.",
    )
    parser.add_argument("--final-logit-ce-weight", type=float, default=1.0)
    parser.add_argument("--depth-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--all-depth-ce-weight", type=float, default=0.0)
    parser.add_argument("--progress-margin-weight", type=float, default=0.25)
    parser.add_argument("--progress-margin", type=float, default=0.10)
    parser.add_argument(
        "--depth-trajectory-monotonic-weight",
        type=float,
        default=0.0,
        help=(
            "Adjacent-depth process credit: penalize recursive steps whose "
            "target sequence log-prob regresses relative to the previous depth."
        ),
    )
    parser.add_argument(
        "--depth-trajectory-monotonic-margin",
        type=float,
        default=0.02,
        help="Required adjacent-depth target log-prob improvement margin.",
    )
    parser.add_argument("--final-greedy-token-margin-weight", type=float, default=0.0)
    parser.add_argument("--depth-greedy-token-margin-weight", type=float, default=0.0)
    parser.add_argument("--greedy-token-margin", type=float, default=0.0)
    parser.add_argument(
        "--core-role-value-vocab-renderer-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Directly supervise core_role_value_vocab_renderer_logits on "
            "causal-prefix answer tokens. This is a renderer-only pressure, "
            "separate from donor-fused final_path_ce."
        ),
    )
    parser.add_argument(
        "--core-role-value-vocab-renderer-greedy-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Greedy-token margin for core_role_value_vocab_renderer_logits. "
            "Use with the renderer CE when donor logits otherwise dominate."
        ),
    )
    parser.add_argument(
        "--core-role-value-vocab-renderer-primitive-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast full renderer target log-prob against the same renderer "
            "with core_primitive_role_value_executor disabled. This forces the "
            "LM renderer to depend on the primitive recursive state, not just "
            "prompt-copy features."
        ),
    )
    parser.add_argument(
        "--core-role-value-vocab-renderer-primitive-contrast-margin",
        type=float,
        default=0.05,
        help="Target log-prob margin for primitive-on vs primitive-off renderer logits.",
    )
    parser.add_argument(
        "--core-role-value-vocab-renderer-source-binder-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast full renderer target log-prob against the same renderer "
            "with core_source_position_binder disabled. This forces direct "
            "source-position state tokens to matter before L4 promotion."
        ),
    )
    parser.add_argument(
        "--core-role-value-vocab-renderer-source-binder-contrast-margin",
        type=float,
        default=0.05,
        help="Target log-prob margin for source-binder-on vs source-binder-off renderer logits.",
    )
    parser.add_argument(
        "--terminal-depth-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Apply CE only to recursive depths whose transition_finality_targets "
            "mark the latent state as terminal. This pressures final depths to "
            "hold the answer without forcing nonterminal depths to emit it."
        ),
    )
    parser.add_argument(
        "--answer-state-loop-halt-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Train the answer-state-loop halt head to choose the first terminal "
            "recursive depth. This is the causal halting signal used by the "
            "answer_state_loop_halt_gate path."
        ),
    )
    parser.add_argument(
        "--answer-state-loop-logit-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Directly supervise answer_state_loop_logits on causal-prefix answer "
            "tokens. This tests whether the recurrent answer loop itself can "
            "carry the LM rendering path."
        ),
    )
    parser.add_argument(
        "--answer-state-loop-future-token-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Same-prefix latent-lookahead auxiliary CE. From the prompt-only "
            "answer-state-loop hidden state, predict the next answer tokens in "
            "parallel without changing the runtime autoregressive LM path."
        ),
    )
    parser.add_argument(
        "--answer-state-loop-future-token-max-target-tokens",
        type=int,
        default=0,
        help=(
            "Maximum target tokens for future-token lookahead CE. "
            "0 uses model.answer_state_loop_future_token_max_tokens."
        ),
    )
    parser.add_argument(
        "--family-repeat",
        default="",
        help="Comma-separated hard-family repeat spec, for example list_transform=4,boolean_logic=2.",
    )
    parser.add_argument("--choice-margin-weight", type=float, default=0.0)
    parser.add_argument("--choice-margin", type=float, default=0.10)
    parser.add_argument(
        "--choice-margin-mode",
        choices=["first_token", "sequence"],
        default="first_token",
        help=(
            "Preference margin contract. first_token matches legacy training; "
            "sequence applies the margin to each available causal-prefix answer "
            "token, aligning better with mean forced-choice eval scoring."
        ),
    )
    parser.add_argument(
        "--final-choice-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Final LM-path-only preference margin over row choices. Unlike "
            "--choice-margin-weight, this is compatible with "
            "--final-path-only-supervision and does not touch depth logits."
        ),
    )
    parser.add_argument("--final-choice-margin", type=float, default=0.10)
    parser.add_argument("--tail-negative-margin-weight", type=float, default=0.0)
    parser.add_argument("--tail-negative-margin", type=float, default=0.10)
    parser.add_argument(
        "--tail-negative-family-filter",
        default="mixed_list_arithmetic",
        help=(
            "Task family for preterminal-state tail-negative margin. Empty means "
            "all families with depth_targets + transition_finality_targets."
        ),
    )
    parser.add_argument("--subtract-tail-counterfactual-margin-weight", type=float, default=0.0)
    parser.add_argument("--subtract-tail-counterfactual-margin", type=float, default=0.05)
    parser.add_argument(
        "--subtract-tail-counterfactual-family-filter",
        default="mixed_list_arithmetic",
        help=(
            "Task family for numeric subtract-tail counterfactual negatives. "
            "The generated reject set includes the preterminal sum plus "
            "final-answer +/- 1 variants."
        ),
    )
    parser.add_argument("--temporal-spatial-context-contrast-weight", type=float, default=0.0)
    parser.add_argument("--temporal-spatial-context-contrast-margin", type=float, default=0.10)
    parser.add_argument("--transition-state-contrast-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-contrast-margin", type=float, default=0.10)
    parser.add_argument("--transition-state-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-depth-contrast-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-depth-contrast-margin", type=float, default=0.10)
    parser.add_argument("--transition-state-code-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-finality-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-joint-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-joint-order-contrast-weight", type=float, default=0.0)
    parser.add_argument("--transition-state-joint-order-contrast-margin", type=float, default=0.10)
    parser.add_argument(
        "--transition-joint-answer-bridge-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast the full answer-state loop against the same forward pass "
            "with transition_state_joint_answer_bridge disabled."
        ),
    )
    parser.add_argument(
        "--transition-joint-answer-bridge-contrast-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--transition-joint-answer-bridge-contrast-all-prefix-tokens",
        action="store_true",
        help=(
            "When causal-prefix supervision creates one example per answer token, "
            "apply transition-joint answer-bridge contrast to every prefix token "
            "instead of only the first token."
        ),
    )
    parser.add_argument(
        "--core-role-value-answer-bridge-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast the full answer-state loop against the same forward pass "
            "with core_role_value_state_answer_bridge disabled."
        ),
    )
    parser.add_argument(
        "--core-role-value-answer-bridge-contrast-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--core-role-value-answer-bridge-contrast-all-prefix-tokens",
        action="store_true",
        help=(
            "When causal-prefix supervision creates one example per answer token, "
            "apply core role-value answer-bridge contrast to every prefix token "
            "instead of only the first token."
        ),
    )
    parser.add_argument(
        "--core-role-value-answer-bridge-final-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast final LM-path target log-prob against the same forward pass "
            "with core_role_value_state_answer_bridge disabled. Unlike the legacy "
            "depth-text contrast, this directly pressures the canonical answer logits."
        ),
    )
    parser.add_argument(
        "--core-role-value-answer-bridge-final-contrast-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--typed-value-answer-bridge-final-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast final LM-path target log-prob against the same forward pass "
            "with typed_algorithmic_value_state_answer_bridge disabled. This tests "
            "whether typed latent value state causally improves the answer logits."
        ),
    )
    parser.add_argument(
        "--typed-value-answer-bridge-final-contrast-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--typed-value-answer-bridge-final-contrast-all-prefix-tokens",
        action="store_true",
        help=(
            "When causal-prefix supervision creates one example per answer token, "
            "apply typed value answer-bridge final contrast to every prefix token."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-answer-final-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Contrast final LM-path target log-prob against the same forward pass "
            "with core_primitive_role_value_executor disabled. This is the causal "
            "pressure needed for source-pointer/primitive state to matter at answer time."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-answer-final-contrast-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--core-primitive-role-value-answer-final-contrast-all-prefix-tokens",
        action="store_true",
        help=(
            "When causal-prefix supervision creates one example per answer token, "
            "apply primitive-role-value final contrast to every prefix token."
        ),
    )
    parser.add_argument("--primitive-transition-operation-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-transition-feedback-operation-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-transition-feedback-finality-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-transition-feedback-teacher-forcing",
        action="store_true",
        help=(
            "Feed gold primitive operation/finality hints into the core "
            "transition-feedback recurrent update during training."
        ),
    )
    parser.add_argument(
        "--core-transition-order-bottleneck-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--transition-phase-ce-weight", type=float, default=0.0)
    parser.add_argument("--transition-source-router-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--algorithmic-role-value-transition-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-role-value-prompt-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Direct CE on prompt-extracted role-value logits before the "
            "recursive typed-register executor. This tests whether trace "
            "failures start from a bad prompt-to-register initialization."
        ),
    )
    parser.add_argument(
        "--core-role-value-prompt-target-mode",
        choices=["staged", "initial"],
        default="staged",
        help=(
            "Target for prompt role-value initialization. initial supervises "
            "the state before the first primitive operation; staged keeps the "
            "legacy depth-1 target."
        ),
    )
    parser.add_argument(
        "--core-role-value-prompt-initial-metadata-targets",
        action="store_true",
        help=(
            "When prompt-target-mode=initial, also supervise the scalar roles "
            "with input metadata needed by the recurrent core: even-count "
            "coefficient and subtract-offset. This keeps the answer out of the "
            "prompt state while preserving the causal inputs needed for later "
            "latent transitions."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder",
        action="store_true",
        help=(
            "Enable the internal token-context source-position binder that "
            "initializes core_role_value_state_prompt_logits before recurrent "
            "primitive state updates."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder-gate-min",
        type=float,
        default=0.0,
        help="Minimum blend gate for source-position binder prompt logits.",
    )
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.0,
        help=(
            "Minimum gate for routing source-position binder logits into core "
            "role-state tokens. This is separate from the prompt-logit binder "
            "gate so causal state injection can be bounded."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder-state-st",
        action="store_true",
        help=(
            "Use straight-through hard source-position classes when routing "
            "source binder logits into core role-state tokens."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder-source-slots-only",
        action="store_true",
        help=(
            "Restrict the source-position binder context to prepended "
            "token-numeric source slots. This is a diagnostic path for "
            "testing whether compact prompt-derived slots are causally used."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder-raw-source-slots",
        action="store_true",
        help=(
            "When source-slots-only is active, read raw prepended source-slot "
            "states before the prelude. This matches the standalone pointer "
            "binder scaffold."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder-query-state",
        action="store_true",
        help=(
            "Route source-binder slot-query hidden states into core role-state "
            "tokens. This keeps prompt value information in the recurrent "
            "state instead of reducing it to a position class embedding."
        ),
    )
    parser.add_argument(
        "--core-source-position-binder-query-state-gate-min",
        type=float,
        default=0.0,
        help="Minimum gate for source-binder query-state injection.",
    )
    parser.add_argument(
        "--core-source-value-binder",
        action="store_true",
        help=(
            "Enable a factorized source-value head from the same prompt slot "
            "queries. This predicts numeric source values separately from "
            "source positions and routes them into core role-state tokens."
        ),
    )
    parser.add_argument(
        "--core-source-value-binder-state-gate-min",
        type=float,
        default=0.0,
        help="Minimum gate for factorized source-value state injection.",
    )
    parser.add_argument(
        "--core-source-value-binder-state-st",
        action="store_true",
        help="Use straight-through hard source-value classes for state injection.",
    )
    parser.add_argument(
        "--core-source-value-prompt-ce-weight",
        type=float,
        default=0.0,
        help=(
            "CE weight for the factorized source-value prompt reader. Targets "
            "are absolute input source values, not final answers."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-source-value-conditioning",
        action="store_true",
        help=(
            "Condition the primitive role/value update MLP on the factorized "
            "source-value reader state."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-source-value-gate-min",
        type=float,
        default=0.0,
        help="Minimum gate for source-value conditioning inside primitive update.",
    )
    parser.add_argument(
        "--core-role-value-prompt-parity-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary CE on the prompt-derived base-parity bottleneck used "
            "to condition prompt role-value initialization."
        ),
    )
    parser.add_argument(
        "--core-role-value-template-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary CE for the prompt/core-conditioned latent role-value "
            "template code used by the explicit state-codec scaffold."
        ),
    )
    parser.add_argument(
        "--core-role-value-template-table-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Teacher-forced CE for the latent role-value template table. "
            "This trains the codec table with the gold template id while the "
            "template classifier is still learning."
        ),
    )
    parser.add_argument("--core-value-delta-code-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-primitive-role-value-state-ce-weight",
        type=float,
        default=0.0,
        help=(
            "CE on operation-conditioned primitive recurrent role-value logits. "
            "This is the state-machine scaffold path that remains inside the "
            "token-derived QTRM causal graph."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-step-margin-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-primitive-role-value-step-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-primitive-role-value-trace-margin-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-primitive-role-value-trace-margin",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--core-primitive-role-value-pair-trace-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Paired hard-negative trace contrast for source-position gates. "
            "For rows sharing pair_group_id, the current trace must beat a "
            "different source-position signature from the same value multiset."
        ),
    )
    parser.add_argument(
        "--core-primitive-role-value-pair-trace-contrast-margin",
        type=float,
        default=0.10,
        help="Trace log-prob margin against the paired hard-negative trace.",
    )
    parser.add_argument(
        "--core-primitive-role-value-update-gate-bce-weight",
        type=float,
        default=0.0,
        help=(
            "Train the primitive role-value update gate to open only when a "
            "role changes relative to the previous recurrent value state."
        ),
    )
    parser.add_argument(
        "--core-primitive-typed-selector-bce-weight",
        type=float,
        default=0.0,
        help=(
            "Supervise the internal primitive-vs-typed selector gate only on "
            "positions where exactly one readout matches the gold role-value "
            "target. This trains source selection without adding a runtime "
            "rule solver."
        ),
    )
    parser.add_argument("--core-typed-register-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-typed-register-operation-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-typed-register-operation-target-shift",
        type=int,
        default=0,
        help=(
            "Shift operation-code targets forward for transition-readout "
            "experiments where register state[t] predicts value state[t+1]."
        ),
    )
    parser.add_argument(
        "--core-typed-register-transition-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-typed-register-step-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Optional row-level margin for core_typed_register_value_logits. "
            "Targets step-exact register readout rather than average value CE."
        ),
    )
    parser.add_argument(
        "--core-typed-register-step-margin",
        type=float,
        default=0.0,
        help="Required per-role logit margin for typed-register step exactness.",
    )
    parser.add_argument(
        "--core-typed-register-trace-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Trace-level hard margin for core_typed_register_value_logits. "
            "This optimizes the worst labelled role over the whole trace, so "
            "one broken step cannot hide in average CE."
        ),
    )
    parser.add_argument(
        "--core-typed-register-trace-margin",
        type=float,
        default=0.15,
        help="Required per-role logit margin for typed-register trace exactness.",
    )
    parser.add_argument(
        "--core-typed-register-scalar-role-ce-multiplier",
        type=float,
        default=1.0,
        help=(
            "CE multiplier for the last two typed-register value roles, which "
            "are the scalar coeff/residual roles in the role-value contract."
        ),
    )
    parser.add_argument(
        "--causal-prefix-supervision",
        action="store_true",
        help=(
            "Train answer logits from prompt/prefix-only inputs. This prevents "
            "workspace/core paths from seeing future answer tokens during depth supervision."
        ),
    )
    parser.add_argument(
        "--final-path-only-supervision",
        action="store_true",
        help=(
            "Skip core-depth text logits and train only the final autoregressive "
            "LM path. Use this for lightweight renderer sharpener probes after "
            "a causal core/readout checkpoint already exists."
        ),
    )
    parser.add_argument(
        "--causal-prefix-max-target-tokens",
        type=int,
        default=1,
        help=(
            "When causal-prefix supervision is enabled, train up to this many "
            "answer tokens by appending the previous answer-token prefix only."
        ),
    )
    parser.add_argument(
        "--causal-prefix-later-token-weight",
        type=float,
        default=1.0,
        help=(
            "Loss weight for causal-prefix answer tokens after the first. "
            "The first answer token always keeps weight 1.0."
        ),
    )
    parser.add_argument(
        "--causal-prefix-skip-leading-whitespace-targets",
        action="store_true",
        help=(
            "Use stripped answer tokens for causal-prefix supervision. This "
            "aligns hard-token lexicalization gates with visible answer tokens "
            "instead of spending the first target on a leading whitespace token."
        ),
    )
    parser.add_argument(
        "--causal-prefix-self-rollout-weight",
        type=float,
        default=0.0,
        help=(
            "Add DAgger/scheduled-sampling style examples where previous answer "
            "prefix tokens come from the current model's greedy rollout instead "
            "of the gold prefix."
        ),
    )
    parser.add_argument(
        "--causal-prefix-self-rollout-max-target-tokens",
        type=int,
        default=0,
        help=(
            "Maximum answer tokens supervised on self-rollout prefixes. "
            "0 reuses --causal-prefix-max-target-tokens."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="",
        help="Optional frozen QTRM checkpoint used to preserve first-token recursive depth logits.",
    )
    parser.add_argument("--teacher-first-token-depth-kl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-final-logit-kl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-depth-kl-temperature", type=float, default=1.0)
    parser.add_argument(
        "--answer-selective-context-alignment-weight",
        type=float,
        default=0.0,
        help=(
            "Self-distill a sparse answer selective-context router from the "
            "same checkpoint run with a dense state+prompt context teacher."
        ),
    )
    parser.add_argument(
        "--answer-selective-context-alignment-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--core-world-model-weight",
        type=float,
        default=None,
        help=(
            "Optional LeWM-style auxiliary loss weight over recursive core states. "
            "Defaults to train.loss_core_world_model_weight from the config."
        ),
    )
    parser.add_argument(
        "--staged-internal-first-token-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional semantic transition pressure: at labelled internal core "
            "depths, train the first answer-token logit toward depth_targets[depth]."
        ),
    )
    parser.add_argument(
        "--staged-internal-sequence-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional value-bearing state pressure: at labelled internal core "
            "depths, train depth readout logits toward the full token sequence "
            "from depth_targets[depth]."
        ),
    )
    parser.add_argument(
        "--staged-internal-sequence-max-target-tokens",
        type=int,
        default=6,
        help="Maximum target tokens used by staged internal sequence CE.",
    )
    parser.add_argument(
        "--transition-state-sequence-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional direct transition-state sequence pressure: train the "
            "transition_state_sequence_logits probe toward depth_targets[depth]."
        ),
    )
    parser.add_argument(
        "--transition-value-state-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional compact value-state pressure: train transition_value_state_logits "
            "toward digit/comma/minus encodings of numeric depth targets."
        ),
    )
    parser.add_argument(
        "--transition-value-state-max-target-tokens",
        type=int,
        default=32,
        help="Maximum digit/comma/minus tokens used by transition value-state CE.",
    )
    parser.add_argument(
        "--algorithmic-value-state-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional structured value-state pressure for factorized slots: "
            "kind=list/scalar plus relative numeric slots."
        ),
    )
    parser.add_argument(
        "--algorithmic-value-state-pad-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Relative CE weight for algorithmic value-state pad slots. Keep at "
            "0.0 for content-slot learning; raise only if exact padding is needed."
        ),
    )
    parser.add_argument(
        "--algorithmic-role-value-state-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional role-filler value pressure: train role_value_state_logits "
            "toward stable algorithmic field roles."
        ),
    )
    parser.add_argument(
        "--role-value-list-class-mode",
        choices=["source_position", "absolute"],
        default="source_position",
        help=(
            "How plain list-transform states are encoded when no numeric base "
            "metadata is available. source_position preserves copy pointers; "
            "absolute stores value+1 classes so final doubled states are "
            "answer-renderable for small vocab gates."
        ),
    )
    parser.add_argument(
        "--algorithmic-role-value-step-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Optional row-level pressure for role-value states: every active "
            "role in a step must beat its strongest non-target class by the margin."
        ),
    )
    parser.add_argument(
        "--algorithmic-role-value-step-margin",
        type=float,
        default=0.0,
        help="Required per-role logit margin for role-value step exactness.",
    )
    parser.add_argument(
        "--typed-algorithmic-value-state-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional typed-field value pressure: train raw-list, doubled-list, "
            "scalar coeff, scalar residual, and final residual heads separately."
        ),
    )
    parser.add_argument(
        "--typed-algorithmic-value-state-pad-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Relative CE weight for typed list pad offsets. Keep at 0.0 for "
            "content learning; raise only if exact padding becomes the bottleneck."
        ),
    )
    parser.add_argument(
        "--typed-algorithmic-kind-ce-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for typed algorithmic kind CE.",
    )
    parser.add_argument(
        "--typed-algorithmic-list-ce-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for typed raw/doubled list offset CE.",
    )
    parser.add_argument(
        "--typed-algorithmic-scalar-ce-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for typed scalar coeff/residual/final residual CE. "
            "Use >1 when final affine answer state is the bottleneck."
        ),
    )
    parser.add_argument(
        "--typed-algorithmic-residual-delta-ce-multiplier",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary CE multiplier for signed scalar residual delta. "
            "This trains a latent transition target without changing the final "
            "trace gate."
        ),
    )
    parser.add_argument(
        "--typed-algorithmic-scalar-ordinal-weight",
        type=float,
        default=0.0,
        help=(
            "Optional smooth expected-class loss for typed scalar residual fields. "
            "This preserves CE but adds ordinal numeric pressure so nearby scalar "
            "classes are treated as closer than distant classes."
        ),
    )
    parser.add_argument(
        "--typed-algorithmic-scalar-regression-weight",
        type=float,
        default=0.0,
        help=(
            "Optional continuous scalar codec loss. The model predicts normalized "
            "class values from the same latent state, and eval can round them "
            "back to scalar classes as a diagnostic for numeric value transition."
        ),
    )
    parser.add_argument(
        "--noise-warmup-steps",
        type=int,
        default=0,
        help=(
            "Optional random-token/random-label warm-up before real training. "
            "This calibrates only QTRM trainable modules; the Qwen donor remains frozen."
        ),
    )
    parser.add_argument("--noise-warmup-seq-len", type=int, default=32)
    parser.add_argument("--noise-warmup-batch-size", type=int, default=1)
    parser.add_argument("--noise-warmup-core-steps", type=int, default=2)
    parser.add_argument(
        "--noise-warmup-target-vocab-size",
        type=int,
        default=0,
        help="Optional target-token range for random labels. 0 means full model vocab.",
    )
    parser.add_argument("--noise-warmup-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--noise-warmup-depth-ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--disable-donor-context",
        action="store_true",
        help=(
            "Keep the donor tokenizer/optional logits path available but do not "
            "inject donor hidden states into QTRM. Use this for token-only "
            "recursive raw-intelligence gates."
        ),
    )
    parser.add_argument(
        "--numeric-source-features",
        action="store_true",
        help=(
            "Inject input_list-derived numeric source-slot features through the "
            "visual feature context path. This is an input-representation gate, "
            "not an answer sidecar."
        ),
    )
    parser.add_argument("--numeric-source-max-list-len", type=int, default=5)
    parser.add_argument("--numeric-source-value-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-value-features", action="store_true")
    parser.add_argument("--token-numeric-value-vocab-size", type=int, default=128)
    parser.add_argument(
        "--token-numeric-source-slots",
        action="store_true",
        help=(
            "Prepend compact source-slot tokens derived from tokenizer offsets "
            "over the visible prompt. This is a canonical prompt-derived input "
            "candidate for source-position gates."
        ),
    )
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--token-numeric-source-slot-parity-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary CE for source-slot even/odd perception. This is a "
            "diagnostic predicate bottleneck, not a final answer solver."
        ),
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-feedback",
        action="store_true",
        help=(
            "Feed a learned source-slot predicate embedding back into source-slot "
            "token states before the binder/core path."
        ),
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary CE for the source-slot predicate feedback logits. This "
            "must remain a token-state scaffold, not a final answer channel."
        ),
    )
    parser.add_argument(
        "--noise-warmup-uniform-weight",
        type=float,
        default=0.0,
        help=(
            "Optional uncertainty-calibration loss for random-token warm-up. "
            "When non-zero, pushes final/depth logits toward high-entropy "
            "uniform predictions on random inputs instead of overconfident guesses."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Deterministic run seed for python/random, numpy, torch, and CUDA. "
            "Accepted-gate candidates must keep this recorded for reproduction."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help=(
            "Optional training checkpoint interval in steps. Saves "
            "step_000040.pt-style snapshots for validation-gated selection."
        ),
    )
    parser.add_argument(
        "--save-trainable-only",
        action="store_true",
        help=(
            "Save only trainable parameter deltas plus base_checkpoint metadata. "
            "Use for frozen-donor adapter runs to avoid writing full donor weights."
        ),
    )
    return parser


def validate_init_checkpoint_args(init_checkpoint: str, *, allow_random_init: bool) -> str:
    if str(init_checkpoint or "").strip():
        return "checkpoint"
    if bool(allow_random_init):
        return "random_init"
    raise ValueError(
        "--init-checkpoint is required unless --allow-random-init is explicitly set"
    )


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


def _prepare_prompt_answer(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    device: str,
):
    import torch

    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    full_text = f"{prompt} {answer}"
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    prompt_len = int(prompt_enc["input_ids"].shape[1])
    full_len = int(input_ids.shape[1])
    if full_len <= prompt_len:
        raise ValueError(
            f"answer tokens were truncated or missing for prompt={prompt!r} answer={answer!r}"
        )
    target_start = prompt_len
    target_end = full_len
    target_ids = input_ids[:, target_start:target_end]
    return input_ids, attention_mask, target_ids, target_start, target_end


def _token_numeric_value_ids_for_prompt_prefix(
    tokenizer: Any,
    row: dict[str, Any],
    prompt: str,
    *,
    input_ids,
    max_length: int,
    value_vocab_size: int,
    device: str,
):
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    ids = list(
        token_numeric_value_ids(
            row,
            offsets=enc["offset_mapping"][0].tolist(),
            value_vocab_size=int(value_vocab_size),
        )
    )
    target_len = int(input_ids.shape[1])
    if len(ids) < target_len:
        ids.extend([0] * (target_len - len(ids)))
    ids = ids[:target_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _token_numeric_source_slots_for_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    prompt: str,
    *,
    max_length: int,
    max_slots: int,
    value_vocab_size: int,
    device: str,
):
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    ids, mask = token_numeric_source_slot_ids(
        row,
        offsets=enc["offset_mapping"][0].tolist(),
        max_list_len=int(max_slots),
        value_vocab_size=int(value_vocab_size),
    )
    slot_token_ids = token_numeric_source_slot_token_ids(
        row,
        offsets=enc["offset_mapping"][0].tolist(),
        input_ids=enc["input_ids"][0].tolist(),
        max_list_len=int(max_slots),
        value_vocab_size=int(value_vocab_size),
    )
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([slot_token_ids], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.long, device=device),
    )


def _prepare_causal_prefix_answer(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    device: str,
):
    return _prepare_causal_prefix_answer_examples(
        tokenizer,
        prompt,
        answer,
        max_length=max_length,
        device=device,
        max_target_tokens=1,
    )[0]


def _prepare_causal_prefix_answer_examples(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    device: str,
    max_target_tokens: int,
    skip_leading_whitespace_targets: bool = False,
):
    import torch

    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")

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
    answer_ids = causal_prefix_answer_token_ids(
        tokenizer,
        answer,
        skip_leading_whitespace_targets=bool(skip_leading_whitespace_targets),
    )

    examples = []
    target_count = min(len(answer_ids), int(max_target_tokens))
    for target_index in range(target_count):
        prefix_ids = [int(token_id) for token_id in answer_ids[:target_index]]
        example_input_ids = input_ids
        example_attention_mask = attention_mask
        if prefix_ids:
            prefix_tensor = input_ids.new_tensor([prefix_ids])
            prefix_mask = attention_mask.new_ones((attention_mask.shape[0], len(prefix_ids)))
            example_input_ids = torch.cat([input_ids, prefix_tensor], dim=1)
            example_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        if int(example_input_ids.shape[1]) > int(max_length):
            break
        target_start = int(example_input_ids.shape[1])
        target_ids = input_ids.new_tensor([[int(answer_ids[target_index])]])
        examples.append(
            (
                example_input_ids,
                example_attention_mask,
                target_ids,
                target_start,
                target_start + 1,
            )
        )
    if not examples:
        raise ValueError(
            f"causal-prefix answer produced no target examples: prompt={prompt!r} answer={answer!r}"
        )
    return examples


def _prepare_causal_prefix_rollout_answer_examples(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    rollout_prefix_ids: Iterable[int],
    max_length: int,
    device: str,
    max_target_tokens: int,
    skip_leading_whitespace_targets: bool = False,
):
    import torch

    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")

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
    answer_ids = causal_prefix_answer_token_ids(
        tokenizer,
        answer,
        skip_leading_whitespace_targets=bool(skip_leading_whitespace_targets),
    )
    rollout_ids = [int(token_id) for token_id in rollout_prefix_ids]

    target_count = min(
        len(answer_ids),
        int(max_target_tokens),
        len(rollout_ids) + 1,
    )
    examples = []
    for target_index in range(target_count):
        prefix_ids = rollout_ids[:target_index]
        example_input_ids = input_ids
        example_attention_mask = attention_mask
        if prefix_ids:
            prefix_tensor = input_ids.new_tensor([prefix_ids])
            prefix_mask = attention_mask.new_ones((attention_mask.shape[0], len(prefix_ids)))
            example_input_ids = torch.cat([input_ids, prefix_tensor], dim=1)
            example_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        if int(example_input_ids.shape[1]) > int(max_length):
            break
        target_start = int(example_input_ids.shape[1])
        target_ids = input_ids.new_tensor([[int(answer_ids[target_index])]])
        examples.append(
            (
                example_input_ids,
                example_attention_mask,
                target_ids,
                target_start,
                target_start + 1,
            )
        )
    if not examples:
        raise ValueError(
            f"self-rollout answer produced no target examples: prompt={prompt!r} answer={answer!r}"
        )
    return examples


def _causal_prefix_example_loss_weight(example_index: int, later_token_weight: float) -> float:
    if float(later_token_weight) < 0.0:
        raise ValueError("later_token_weight must be non-negative")
    if int(example_index) == 0:
        return 1.0
    return float(later_token_weight)


def _should_apply_teacher_first_token_depth_kl(example_index: int, weight: float) -> bool:
    return int(example_index) == 0 and float(weight) > 0.0


def _should_apply_transition_joint_answer_bridge_contrast(
    example_index: int,
    weight: float,
    *,
    all_prefix_tokens: bool = False,
) -> bool:
    if float(weight) == 0.0:
        return False
    return bool(all_prefix_tokens) or int(example_index) == 0


def _target_log_probs(depth_logits, target_ids):
    import torch

    log_probs = depth_logits.float().log_softmax(dim=-1)
    gather_index = target_ids[:, None, None].expand(-1, depth_logits.shape[1], 1)
    return log_probs.gather(dim=-1, index=gather_index).squeeze(-1)


def depth_supervision_loss(
    depth_logits,
    final_logits,
    target_ids,
    *,
    final_logit_ce_weight: float,
    depth_final_ce_weight: float = 1.0,
    all_depth_ce_weight: float,
    progress_margin_weight: float,
    progress_margin: float,
):
    import torch
    import torch.nn.functional as F

    if depth_logits.ndim != 3:
        raise ValueError("depth_logits must have shape [batch, steps, vocab]")
    if depth_logits.shape[1] == 0:
        raise ValueError("recursive core produced no depth logits")
    final_ce = F.cross_entropy(depth_logits[:, -1, :].float(), target_ids)
    final_path_ce = F.cross_entropy(final_logits.float(), target_ids)
    all_depth_ce = final_ce.new_zeros(())
    if float(all_depth_ce_weight) != 0.0:
        repeated_targets = target_ids[:, None].expand(-1, depth_logits.shape[1]).reshape(-1)
        all_depth_ce = F.cross_entropy(
            depth_logits.float().reshape(-1, depth_logits.shape[-1]),
            repeated_targets,
        )
    progress = final_ce.new_zeros(())
    if depth_logits.shape[1] > 1 and float(progress_margin_weight) != 0.0:
        target_logps = _target_log_probs(depth_logits, target_ids)
        final_logp = target_logps[:, -1:]
        progress = F.relu(float(progress_margin) + target_logps[:, :-1] - final_logp).mean()
    loss = (
        float(final_logit_ce_weight) * final_path_ce
        + float(depth_final_ce_weight) * final_ce
        + float(all_depth_ce_weight) * all_depth_ce
        + float(progress_margin_weight) * progress
    )
    with torch.no_grad():
        pred = depth_logits[:, -1, :].argmax(dim=-1)
        acc = (pred == target_ids).float().mean()
        final_path_pred = final_logits.argmax(dim=-1)
        final_path_acc = (final_path_pred == target_ids).float().mean()
        target_logps = _target_log_probs(depth_logits, target_ids)
        depth_delta = (
            target_logps[:, -1] - target_logps[:, 0]
            if depth_logits.shape[1] > 1
            else target_logps[:, -1] * 0.0
        )
    return loss, {
        "depth_final_ce": final_ce.detach(),
        "final_path_ce": final_path_ce.detach(),
        "depth_all_ce": all_depth_ce.detach(),
        "depth_progress": progress.detach(),
        "depth_final_acc": acc.detach(),
        "final_path_acc": final_path_acc.detach(),
        "depth_target_logp_delta": depth_delta.mean().detach(),
    }


def _token_sequence_cross_entropy(logits, target_ids):
    import torch.nn.functional as F

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits token dimension must match target_ids")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
    )


def _token_top_competitor_margin(logits, target_ids, *, margin: float):
    import torch

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits token dimension must match target_ids")
    if logits.shape[-1] <= 1:
        zero = logits.sum() * 0.0
        return zero, zero.detach()
    valid = target_ids >= 0
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        return zero, zero.detach()
    safe_targets = target_ids.masked_fill(~valid, 0)
    target_logits = logits.float().gather(
        dim=-1,
        index=safe_targets[:, :, None],
    ).squeeze(-1)
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    target_mask.scatter_(-1, safe_targets[:, :, None], True)
    top_competitor = logits.float().masked_fill(target_mask, -torch.inf).max(dim=-1).values
    violations = torch.relu(float(margin) + top_competitor - target_logits)
    loss = violations.masked_select(valid).mean()
    with torch.no_grad():
        wins = target_logits >= top_competitor + float(margin)
        win_rate = wins.masked_select(valid).float().mean()
    return loss, win_rate.detach()


def _sequence_target_log_probs(depth_text_logits, target_ids):
    log_probs = depth_text_logits.float().log_softmax(dim=-1)
    gather_index = target_ids[:, None, :, None].expand(
        -1,
        depth_text_logits.shape[1],
        -1,
        1,
    )
    return log_probs.gather(dim=-1, index=gather_index).squeeze(-1).mean(dim=-1)


def depth_text_logit_distillation_loss(
    student_depth_text_logits,
    teacher_depth_text_logits,
    *,
    temperature: float,
):
    import torch.nn.functional as F

    if student_depth_text_logits.shape != teacher_depth_text_logits.shape:
        raise ValueError("student and teacher depth text logits must have matching shapes")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    vocab = student_depth_text_logits.shape[-1]
    student = student_depth_text_logits.float().reshape(-1, vocab) / float(temperature)
    teacher = teacher_depth_text_logits.detach().float().reshape(-1, vocab) / float(temperature)
    return F.kl_div(
        student.log_softmax(dim=-1),
        teacher.softmax(dim=-1),
        reduction="batchmean",
    ) * (float(temperature) ** 2)


def answer_selective_context_alignment_loss(
    student_text_logits,
    dense_teacher_text_logits,
    *,
    temperature: float,
):
    import torch.nn.functional as F

    if student_text_logits.shape != dense_teacher_text_logits.shape:
        raise ValueError("student and dense teacher text logits must have matching shapes")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    vocab = student_text_logits.shape[-1]
    student = student_text_logits.float().reshape(-1, vocab) / float(temperature)
    teacher = (
        dense_teacher_text_logits.detach().float().reshape(-1, vocab)
        / float(temperature)
    )
    loss = F.kl_div(
        student.log_softmax(dim=-1),
        teacher.softmax(dim=-1),
        reduction="batchmean",
    ) * (float(temperature) ** 2)
    return loss, {"answer_selective_context_alignment_kl": loss.detach()}


def depth_sequence_supervision_loss(
    depth_text_logits,
    final_text_logits,
    target_ids,
    *,
    final_logit_ce_weight: float,
    depth_final_ce_weight: float = 1.0,
    all_depth_ce_weight: float,
    progress_margin_weight: float,
    progress_margin: float,
    depth_trajectory_monotonic_weight: float = 0.0,
    depth_trajectory_monotonic_margin: float = 0.02,
    final_greedy_token_margin_weight: float = 0.0,
    depth_greedy_token_margin_weight: float = 0.0,
    greedy_token_margin: float = 0.0,
):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if depth_text_logits.shape[1] == 0:
        raise ValueError("recursive core produced no depth text logits")
    if depth_text_logits.shape[2] != target_ids.shape[1]:
        raise ValueError("depth text logits must align to target token count")

    final_ce = _token_sequence_cross_entropy(depth_text_logits[:, -1, :, :], target_ids)
    final_path_ce = _token_sequence_cross_entropy(final_text_logits, target_ids)
    all_depth_ce = final_ce.new_zeros(())
    if float(all_depth_ce_weight) != 0.0:
        repeated_targets = target_ids[:, None, :].expand(
            -1,
            depth_text_logits.shape[1],
            -1,
        )
        all_depth_ce = _token_sequence_cross_entropy(
            depth_text_logits.reshape(
                -1,
                depth_text_logits.shape[2],
                depth_text_logits.shape[-1],
            ),
            repeated_targets.reshape(-1, target_ids.shape[1]),
        )
    progress = final_ce.new_zeros(())
    if depth_text_logits.shape[1] > 1 and float(progress_margin_weight) != 0.0:
        target_logps = _sequence_target_log_probs(depth_text_logits, target_ids)
        final_logp = target_logps[:, -1:]
        progress = torch.relu(
            float(progress_margin) + target_logps[:, :-1] - final_logp
        ).mean()
    trajectory_monotonic = final_ce.new_zeros(())
    trajectory_step_delta = final_ce.new_zeros(())
    if (
        depth_text_logits.shape[1] > 1
        and float(depth_trajectory_monotonic_weight) != 0.0
    ):
        target_logps = _sequence_target_log_probs(depth_text_logits, target_ids)
        step_deltas = target_logps[:, 1:] - target_logps[:, :-1]
        trajectory_monotonic = torch.relu(
            float(depth_trajectory_monotonic_margin) - step_deltas
        ).mean()
        trajectory_step_delta = step_deltas.mean()
    final_greedy_margin = final_ce.new_zeros(())
    final_greedy_win_rate = final_ce.new_zeros(())
    if float(final_greedy_token_margin_weight) != 0.0:
        final_greedy_margin, final_greedy_win_rate = _token_top_competitor_margin(
            final_text_logits,
            target_ids,
            margin=greedy_token_margin,
        )
    depth_greedy_margin = final_ce.new_zeros(())
    depth_greedy_win_rate = final_ce.new_zeros(())
    if float(depth_greedy_token_margin_weight) != 0.0:
        depth_greedy_margin, depth_greedy_win_rate = _token_top_competitor_margin(
            depth_text_logits[:, -1, :, :],
            target_ids,
            margin=greedy_token_margin,
        )
    loss = (
        float(final_logit_ce_weight) * final_path_ce
        + float(depth_final_ce_weight) * final_ce
        + float(all_depth_ce_weight) * all_depth_ce
        + float(progress_margin_weight) * progress
        + float(depth_trajectory_monotonic_weight) * trajectory_monotonic
        + float(final_greedy_token_margin_weight) * final_greedy_margin
        + float(depth_greedy_token_margin_weight) * depth_greedy_margin
    )
    with torch.no_grad():
        pred = depth_text_logits[:, -1, :, :].argmax(dim=-1)
        acc = (pred == target_ids).float().mean()
        final_path_pred = final_text_logits.argmax(dim=-1)
        final_path_acc = (final_path_pred == target_ids).float().mean()
        target_logps = _sequence_target_log_probs(depth_text_logits, target_ids)
        depth_delta = (
            target_logps[:, -1] - target_logps[:, 0]
            if depth_text_logits.shape[1] > 1
            else target_logps[:, -1] * 0.0
        )
    return loss, {
        "depth_final_ce": final_ce.detach(),
        "final_path_ce": final_path_ce.detach(),
        "depth_all_ce": all_depth_ce.detach(),
        "depth_progress": progress.detach(),
        "depth_trajectory_monotonic": trajectory_monotonic.detach(),
        "depth_trajectory_step_delta": trajectory_step_delta.detach(),
        "final_greedy_token_margin": final_greedy_margin.detach(),
        "final_greedy_token_win_rate": final_greedy_win_rate.detach(),
        "depth_greedy_token_margin": depth_greedy_margin.detach(),
        "depth_greedy_token_win_rate": depth_greedy_win_rate.detach(),
        "depth_final_acc": acc.detach(),
        "final_path_acc": final_path_acc.detach(),
        "depth_target_logp_delta": depth_delta.mean().detach(),
    }


def final_path_sequence_supervision_loss(
    final_text_logits,
    target_ids,
    *,
    final_logit_ce_weight: float,
    final_greedy_token_margin_weight: float = 0.0,
    greedy_token_margin: float = 0.0,
):
    import torch

    final_path_ce = _token_sequence_cross_entropy(final_text_logits, target_ids)
    final_greedy_margin = final_path_ce.new_zeros(())
    final_greedy_win_rate = final_path_ce.new_zeros(())
    if float(final_greedy_token_margin_weight) != 0.0:
        final_greedy_margin, final_greedy_win_rate = _token_top_competitor_margin(
            final_text_logits,
            target_ids,
            margin=greedy_token_margin,
        )
    loss = (
        float(final_logit_ce_weight) * final_path_ce
        + float(final_greedy_token_margin_weight) * final_greedy_margin
    )
    with torch.no_grad():
        final_path_pred = final_text_logits.argmax(dim=-1)
        final_path_acc = (final_path_pred == target_ids).float().mean()
    zero = final_path_ce.new_zeros(())
    return loss, {
        "depth_final_ce": zero.detach(),
        "final_path_ce": final_path_ce.detach(),
        "depth_all_ce": zero.detach(),
        "depth_progress": zero.detach(),
        "depth_trajectory_monotonic": zero.detach(),
        "depth_trajectory_step_delta": zero.detach(),
        "final_greedy_token_margin": final_greedy_margin.detach(),
        "final_greedy_token_win_rate": final_greedy_win_rate.detach(),
        "depth_greedy_token_margin": zero.detach(),
        "depth_greedy_token_win_rate": zero.detach(),
        "depth_final_acc": zero.detach(),
        "final_path_acc": final_path_acc.detach(),
        "depth_target_logp_delta": zero.detach(),
    }


def core_role_value_vocab_renderer_sequence_supervision_loss(
    renderer_text_logits,
    target_ids,
    *,
    renderer_ce_weight: float,
    renderer_greedy_token_margin_weight: float = 0.0,
    greedy_token_margin: float = 0.0,
):
    import torch

    renderer_ce = _token_sequence_cross_entropy(renderer_text_logits, target_ids)
    renderer_greedy_margin = renderer_ce.new_zeros(())
    renderer_greedy_win_rate = renderer_ce.new_zeros(())
    if float(renderer_greedy_token_margin_weight) != 0.0:
        renderer_greedy_margin, renderer_greedy_win_rate = (
            _token_top_competitor_margin(
                renderer_text_logits,
                target_ids,
                margin=greedy_token_margin,
            )
        )
    loss = (
        float(renderer_ce_weight) * renderer_ce
        + float(renderer_greedy_token_margin_weight) * renderer_greedy_margin
    )
    with torch.no_grad():
        renderer_pred = renderer_text_logits.argmax(dim=-1)
        renderer_acc = (renderer_pred == target_ids).float().mean()
    return loss, {
        "core_role_value_vocab_renderer_ce": renderer_ce.detach(),
        "core_role_value_vocab_renderer_acc": renderer_acc.detach(),
        "core_role_value_vocab_renderer_greedy_token_margin": (
            renderer_greedy_margin.detach()
        ),
        "core_role_value_vocab_renderer_greedy_token_win_rate": (
            renderer_greedy_win_rate.detach()
        ),
    }


def terminal_depth_mask_from_row(
    row: dict[str, Any],
    *,
    num_depths: int,
    device: Any,
):
    import torch

    if int(num_depths) <= 0:
        raise ValueError("num_depths must be positive")
    finality = row.get("transition_finality_targets")
    mask = torch.zeros(int(num_depths), dtype=torch.bool, device=device)
    if isinstance(finality, dict):
        for depth_text, value in finality.items():
            try:
                depth = int(depth_text)
            except (TypeError, ValueError):
                continue
            index = depth - 1
            if 0 <= index < int(num_depths) and float(value) > 0.0:
                mask[index] = True
        return mask

    depth_targets = row.get("depth_targets")
    final_answer = _final_answer_text(row).strip()
    final_norm = _choice_margin_normalize_text(final_answer)
    if not isinstance(depth_targets, dict) or not final_norm:
        return mask
    for depth_text, target in depth_targets.items():
        try:
            depth = int(depth_text)
        except (TypeError, ValueError):
            continue
        index = depth - 1
        if not 0 <= index < int(num_depths):
            continue
        if _choice_margin_normalize_text(target) == final_norm:
            mask[index] = True
    return mask


def terminal_depth_ce_loss(depth_text_logits, target_ids, terminal_mask):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if depth_text_logits.shape[2] != target_ids.shape[1]:
        raise ValueError("depth text logits token dimension must match target ids")
    mask = terminal_mask.to(device=depth_text_logits.device, dtype=torch.bool)
    if mask.ndim != 1 or int(mask.shape[0]) != int(depth_text_logits.shape[1]):
        raise ValueError("terminal_mask must have shape [steps]")
    if not bool(mask.any()):
        zero = depth_text_logits.new_zeros(())
        return zero, {
            "terminal_depth_ce": zero.detach(),
            "terminal_depth_acc": zero.detach(),
            "terminal_depth_count": zero.detach(),
        }
    selected_logits = depth_text_logits[:, mask, :, :]
    repeated_targets = target_ids[:, None, :].expand(
        -1,
        int(selected_logits.shape[1]),
        -1,
    )
    ce = _token_sequence_cross_entropy(
        selected_logits.reshape(
            -1,
            selected_logits.shape[2],
            selected_logits.shape[-1],
        ),
        repeated_targets.reshape(-1, target_ids.shape[1]),
    )
    with torch.no_grad():
        pred = selected_logits.argmax(dim=-1)
        acc = (pred == repeated_targets).float().mean()
    return ce, {
        "terminal_depth_ce": ce.detach(),
        "terminal_depth_acc": acc.detach(),
        "terminal_depth_count": depth_text_logits.new_tensor(float(mask.sum().item())),
    }


def answer_state_loop_halt_ce_loss(halt_logits, terminal_mask):
    import torch

    if halt_logits.ndim != 2:
        raise ValueError("halt_logits must have shape [batch, steps]")
    mask = terminal_mask.to(device=halt_logits.device, dtype=torch.bool)
    if mask.ndim != 1 or int(mask.shape[0]) != int(halt_logits.shape[1]):
        raise ValueError("terminal_mask must have shape [steps]")
    if not bool(mask.any()) or halt_logits.shape[1] == 0:
        zero = halt_logits.new_zeros(())
        return zero, {
            "answer_state_halt_ce": zero.detach(),
            "answer_state_halt_acc": zero.detach(),
            "answer_state_halt_count": zero.detach(),
        }
    target_index = int(torch.nonzero(mask, as_tuple=False)[0].item())
    targets = torch.full(
        (halt_logits.shape[0],),
        target_index,
        dtype=torch.long,
        device=halt_logits.device,
    )
    ce = torch.nn.functional.cross_entropy(halt_logits.float(), targets)
    with torch.no_grad():
        acc = (halt_logits.float().argmax(dim=1) == targets).float().mean()
    return ce, {
        "answer_state_halt_ce": ce.detach(),
        "answer_state_halt_acc": acc.detach(),
        "answer_state_halt_count": halt_logits.new_tensor(float(halt_logits.shape[0])),
    }


def answer_state_loop_future_token_targets(
    tokenizer: Any,
    answer: str,
    *,
    max_target_tokens: int,
    device: str,
):
    import torch

    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")
    token_ids = answer_token_ids(tokenizer, answer)[: int(max_target_tokens)]
    padded = [int(token_id) for token_id in token_ids]
    padded.extend([-100] * (int(max_target_tokens) - len(padded)))
    return torch.tensor([padded], dtype=torch.long, device=device)


def answer_state_loop_future_token_ce_loss(future_token_logits, target_ids):
    import torch
    import torch.nn.functional as F

    if future_token_logits.ndim != 3:
        raise ValueError("future_token_logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    token_count = min(int(future_token_logits.shape[1]), int(target_ids.shape[1]))
    if token_count <= 0:
        raise ValueError("future-token CE requires at least one predicted token")
    logits = future_token_logits[:, :token_count, :].float()
    targets = target_ids[:, :token_count].to(
        device=future_token_logits.device,
        dtype=torch.long,
    )
    mask = targets >= 0
    if not bool(mask.any()):
        zero = future_token_logits.sum() * 0.0
        return zero, {
            "answer_state_future_token_ce": zero.detach(),
            "answer_state_future_token_acc": zero.detach(),
            "answer_state_future_token_samples": zero.detach(),
        }
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = future_token_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "answer_state_future_token_ce": loss.detach(),
        "answer_state_future_token_acc": acc.detach(),
        "answer_state_future_token_samples": samples,
    }


def answer_state_loop_logit_ce_loss(answer_loop_logits, target_ids):
    import torch
    import torch.nn.functional as F

    if answer_loop_logits.ndim != 3:
        raise ValueError("answer_loop_logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    token_count = min(int(answer_loop_logits.shape[1]), int(target_ids.shape[1]))
    if token_count <= 0:
        raise ValueError("answer-state-loop logit CE requires at least one token")
    logits = answer_loop_logits[:, :token_count, :].float()
    targets = target_ids[:, :token_count].to(
        device=answer_loop_logits.device,
        dtype=torch.long,
    )
    mask = targets >= 0
    if not bool(mask.any()):
        zero = answer_loop_logits.sum() * 0.0
        return zero, {
            "answer_state_loop_logit_ce": zero.detach(),
            "answer_state_loop_logit_acc": zero.detach(),
            "answer_state_loop_logit_samples": zero.detach(),
        }
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = answer_loop_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "answer_state_loop_logit_ce": loss.detach(),
        "answer_state_loop_logit_acc": acc.detach(),
        "answer_state_loop_logit_samples": samples,
    }


def context_ablation_contrastive_loss(
    context_depth_text_logits,
    context_off_depth_text_logits,
    target_ids,
    *,
    margin: float,
):
    import torch

    if context_depth_text_logits.shape != context_off_depth_text_logits.shape:
        raise ValueError("context-on and context-off depth logits must have matching shapes")
    on_logp = _sequence_target_log_probs(context_depth_text_logits, target_ids)[:, -1]
    off_logp = _sequence_target_log_probs(context_off_depth_text_logits, target_ids)[:, -1]
    delta = on_logp - off_logp
    loss = torch.relu(float(margin) - delta).mean()
    return loss, {
        "context_contrast": loss.detach(),
        "context_contrast_target_logp_delta": delta.mean().detach(),
    }


def transition_state_ablation_contrastive_loss(
    transition_depth_text_logits,
    transition_off_depth_text_logits,
    target_ids,
    *,
    margin: float,
):
    import torch

    if transition_depth_text_logits.shape != transition_off_depth_text_logits.shape:
        raise ValueError(
            "transition-state-on and transition-state-off depth logits must have matching shapes"
        )
    on_logp = _sequence_target_log_probs(transition_depth_text_logits, target_ids)[:, -1]
    off_logp = _sequence_target_log_probs(
        transition_off_depth_text_logits,
        target_ids,
    )[:, -1].detach()
    delta = on_logp - off_logp
    loss = torch.relu(float(margin) - delta).mean()
    return loss, {
        "transition_state_contrast": loss.detach(),
        "transition_state_contrast_target_logp_delta": delta.mean().detach(),
    }


def transition_joint_answer_bridge_contrastive_loss(
    bridge_depth_text_logits,
    bridge_off_depth_text_logits,
    target_ids,
    *,
    margin: float,
):
    loss, metrics = transition_state_ablation_contrastive_loss(
        bridge_depth_text_logits,
        bridge_off_depth_text_logits,
        target_ids,
        margin=margin,
    )
    return loss, {
        key.replace("transition_state", "transition_joint_answer_bridge"): value
        for key, value in metrics.items()
    }


def core_role_value_answer_bridge_contrastive_loss(
    bridge_depth_text_logits,
    bridge_off_depth_text_logits,
    target_ids,
    *,
    margin: float,
):
    loss, metrics = transition_state_ablation_contrastive_loss(
        bridge_depth_text_logits,
        bridge_off_depth_text_logits,
        target_ids,
        margin=margin,
    )
    return loss, {
        key.replace("transition_state", "core_role_value_answer_bridge"): value
        for key, value in metrics.items()
    }


def final_path_ablation_contrastive_loss(
    final_text_logits,
    ablated_final_text_logits,
    target_ids,
    *,
    margin: float,
    metric_prefix: str,
):
    import torch

    if final_text_logits.shape != ablated_final_text_logits.shape:
        raise ValueError("final and ablated final logits must have matching shapes")
    on_logp = _sequence_target_log_probs(
        final_text_logits.unsqueeze(1),
        target_ids,
    )[:, -1]
    off_logp = _sequence_target_log_probs(
        ablated_final_text_logits.unsqueeze(1),
        target_ids,
    )[:, -1].detach()
    delta = on_logp - off_logp
    loss = torch.relu(float(margin) - delta).mean()
    return loss, {
        f"{metric_prefix}_final_contrast": loss.detach(),
        f"{metric_prefix}_final_target_logp_delta": delta.mean().detach(),
    }


def staged_internal_first_token_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_depths: int,
    device: str,
    target_mode: str = "staged",
    content_token: bool = False,
):
    import torch

    if int(num_depths) <= 0:
        raise ValueError("num_depths must be positive")
    target_ids: list[int] = []
    depth_targets = row.get("depth_targets") if str(target_mode).lower() == "staged" else None
    for depth in range(1, int(num_depths) + 1):
        target_text: str | None = None
        if isinstance(depth_targets, dict):
            exact = depth_targets.get(str(depth))
            if exact:
                target_text = str(exact)
        elif str(target_mode).lower() == "final":
            target_text = target_for_core_steps(row, depth, target_mode="final")
        if target_text is None:
            target_ids.append(-100)
            continue
        if bool(content_token):
            target_ids.append(answer_content_first_token_id(tokenizer, target_text))
        else:
            target_ids.append(answer_first_token_id(tokenizer, target_text))
    return torch.tensor([target_ids], dtype=torch.long, device=device)


def staged_internal_first_token_ce_loss(depth_text_logits, staged_target_ids):
    import torch
    import torch.nn.functional as F

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if depth_text_logits.shape[2] < 1:
        raise ValueError("staged internal CE needs at least one answer-token logit")
    targets = staged_target_ids.to(device=depth_text_logits.device, dtype=torch.long)
    if targets.shape != depth_text_logits.shape[:2]:
        raise ValueError("staged_target_ids must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = depth_text_logits.sum() * 0.0
        return zero, {
            "staged_internal_first_token_ce": zero.detach(),
            "staged_internal_first_token_acc": zero.detach(),
            "staged_internal_first_token_samples": zero.detach(),
        }
    logits = depth_text_logits[:, :, 0, :].float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = depth_text_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "staged_internal_first_token_ce": loss.detach(),
        "staged_internal_first_token_acc": acc.detach(),
        "staged_internal_first_token_samples": samples,
    }


def staged_internal_sequence_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_depths: int,
    max_target_tokens: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if int(num_depths) <= 0:
        raise ValueError("num_depths must be positive")
    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")
    targets: list[list[int]] = []
    depth_targets = row.get("depth_targets") if str(target_mode).lower() == "staged" else None
    for depth in range(1, int(num_depths) + 1):
        target_text: str | None = None
        if isinstance(depth_targets, dict):
            exact = depth_targets.get(str(depth))
            if exact:
                target_text = str(exact)
        elif str(target_mode).lower() == "final":
            target_text = target_for_core_steps(row, depth, target_mode="final")
        if target_text is None:
            targets.append([-100] * int(max_target_tokens))
            continue
        token_ids = answer_token_ids(tokenizer, target_text)[: int(max_target_tokens)]
        padded = [int(token_id) for token_id in token_ids]
        padded.extend([-100] * (int(max_target_tokens) - len(padded)))
        targets.append(padded)
    return torch.tensor([targets], dtype=torch.long, device=device)


def staged_internal_sequence_ce_loss(depth_text_logits, staged_sequence_target_ids):
    import torch
    import torch.nn.functional as F

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    targets = staged_sequence_target_ids.to(
        device=depth_text_logits.device,
        dtype=torch.long,
    )
    if targets.ndim != 3:
        raise ValueError("staged_sequence_target_ids must have shape [batch, steps, tokens]")
    if targets.shape[:2] != depth_text_logits.shape[:2]:
        raise ValueError("staged sequence targets must match batch and depth dimensions")
    token_count = min(int(targets.shape[2]), int(depth_text_logits.shape[2]))
    if token_count <= 0:
        zero = depth_text_logits.sum() * 0.0
        return zero, {
            "staged_internal_sequence_ce": zero.detach(),
            "staged_internal_sequence_acc": zero.detach(),
            "staged_internal_sequence_samples": zero.detach(),
        }
    targets = targets[:, :, :token_count]
    logits = depth_text_logits[:, :, :token_count, :].float()
    mask = targets >= 0
    if not bool(mask.any()):
        zero = depth_text_logits.sum() * 0.0
        return zero, {
            "staged_internal_sequence_ce": zero.detach(),
            "staged_internal_sequence_acc": zero.detach(),
            "staged_internal_sequence_samples": zero.detach(),
        }
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = depth_text_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "staged_internal_sequence_ce": loss.detach(),
        "staged_internal_sequence_acc": acc.detach(),
        "staged_internal_sequence_samples": samples,
    }


def transition_value_state_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    max_target_tokens: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if int(num_depths) <= 0:
        raise ValueError("num_depths must be positive")
    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")
    targets: list[list[int]] = []
    depth_targets = row.get("depth_targets") if str(target_mode).lower() == "staged" else None
    for depth in range(1, int(num_depths) + 1):
        target_text: str | None = None
        if isinstance(depth_targets, dict):
            exact = depth_targets.get(str(depth))
            if exact:
                target_text = str(exact)
        elif str(target_mode).lower() == "final":
            target_text = target_for_core_steps(row, depth, target_mode="final")
        token_ids = value_state_token_ids(target_text or "")
        if token_ids is None:
            targets.append([-100] * int(max_target_tokens))
            continue
        padded = [int(token_id) for token_id in token_ids[: int(max_target_tokens)]]
        padded.extend([-100] * (int(max_target_tokens) - len(padded)))
        targets.append(padded)
    return torch.tensor([targets], dtype=torch.long, device=device)


def algorithmic_value_state_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    max_slots: int,
    slot_vocab_size: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if str(target_mode).lower() != "staged":
        kind_targets = [-100] * int(num_depths)
        slot_targets = [[-100] * int(max_slots) for _ in range(int(num_depths))]
    else:
        kind_targets, slot_targets = algorithmic_targets_from_row(
            row,
            num_steps=int(num_depths),
            max_slots=int(max_slots),
            slot_vocab_size=int(slot_vocab_size),
        )
    return (
        torch.tensor([kind_targets], dtype=torch.long, device=device),
        torch.tensor([slot_targets], dtype=torch.long, device=device),
    )


def algorithmic_value_state_ce_loss(
    kind_logits,
    slot_logits,
    kind_targets,
    slot_targets,
    *,
    pad_ce_weight: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    if kind_logits.ndim != 3:
        raise ValueError("kind_logits must have shape [batch, steps, kinds]")
    if slot_logits.ndim != 4:
        raise ValueError("slot_logits must have shape [batch, steps, slots, vocab]")
    kind_targets = kind_targets.to(device=kind_logits.device, dtype=torch.long)
    slot_targets = slot_targets.to(device=slot_logits.device, dtype=torch.long)
    if kind_targets.shape != kind_logits.shape[:2]:
        raise ValueError("kind targets must match kind logits batch/depth")
    if slot_targets.shape[:2] != slot_logits.shape[:2]:
        raise ValueError("slot targets must match slot logits batch/depth")
    token_count = min(int(slot_targets.shape[2]), int(slot_logits.shape[2]))
    slot_targets = slot_targets[:, :, :token_count]
    slot_logits = slot_logits[:, :, :token_count, :].float()

    losses = []
    kind_mask = kind_targets >= 0
    if kind_logits.shape[-1] > 0 and bool(kind_mask.any()):
        kind_loss = F.cross_entropy(kind_logits.float()[kind_mask], kind_targets[kind_mask])
        losses.append(kind_loss)
        kind_pred = kind_logits.detach().argmax(dim=-1)
        kind_acc = (kind_pred[kind_mask] == kind_targets[kind_mask]).float().mean()
    else:
        zero = slot_logits.sum() * 0.0
        kind_loss = zero
        kind_pred = kind_targets.new_full(kind_targets.shape, -1)
        kind_acc = zero.detach()

    pad_ce_weight = max(0.0, float(pad_ce_weight))
    content_slot_mask = slot_targets > 0
    pad_slot_mask = (slot_targets == 0) & (pad_ce_weight > 0.0)
    slot_loss_mask = content_slot_mask | pad_slot_mask
    slot_metric_mask = slot_targets >= 0
    if bool(slot_loss_mask.any()):
        flat_logits = slot_logits[slot_loss_mask]
        flat_targets = slot_targets[slot_loss_mask]
        if pad_ce_weight == 1.0:
            slot_loss = F.cross_entropy(flat_logits, flat_targets)
        else:
            per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            weights = torch.ones_like(per_token)
            weights = torch.where(
                flat_targets == 0,
                weights * float(pad_ce_weight),
                weights,
            )
            slot_loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
        losses.append(slot_loss)
        slot_pred = slot_logits.detach().argmax(dim=-1)
        if bool(slot_metric_mask.any()):
            slot_acc = (
                slot_pred[slot_metric_mask] == slot_targets[slot_metric_mask]
            ).float().mean()
        else:
            slot_acc = slot_loss.detach() * 0.0
        if bool(content_slot_mask.any()):
            content_slot_acc = (
                slot_pred[content_slot_mask] == slot_targets[content_slot_mask]
            ).float().mean()
        else:
            content_slot_acc = slot_acc.detach() * 0.0
    else:
        zero = slot_logits.sum() * 0.0
        slot_loss = zero
        slot_pred = slot_targets.new_full(slot_targets.shape, -1)
        slot_acc = zero.detach()
        content_slot_acc = zero.detach()

    if losses:
        loss = torch.stack(losses).sum()
    else:
        loss = slot_logits.sum() * 0.0
    labelled_steps = kind_mask
    if bool(labelled_steps.any()) and token_count > 0:
        slot_step_match = torch.ones_like(labelled_steps, dtype=torch.bool)
        for batch_index in range(int(slot_targets.shape[0])):
            for step_index in range(int(slot_targets.shape[1])):
                mask = slot_targets[batch_index, step_index] >= 0
                if bool(mask.any()):
                    slot_step_match[batch_index, step_index] = bool(
                        (
                            slot_pred[batch_index, step_index, mask]
                            == slot_targets[batch_index, step_index, mask]
                        )
                        .all()
                        .item()
                    )
        step_exact = (
            ((kind_pred == kind_targets) & slot_step_match)[labelled_steps]
            .float()
            .mean()
        )
    else:
        step_exact = loss.detach() * 0.0
    samples = slot_logits.detach().new_tensor(float(int(slot_metric_mask.sum().item())))
    content_samples = slot_logits.detach().new_tensor(
        float(int(content_slot_mask.sum().item()))
    )
    return loss, {
        "algorithmic_value_state_ce": loss.detach(),
        "algorithmic_value_state_kind_ce": kind_loss.detach(),
        "algorithmic_value_state_slot_ce": slot_loss.detach(),
        "algorithmic_value_state_kind_acc": kind_acc.detach(),
        "algorithmic_value_state_slot_acc": slot_acc.detach(),
        "algorithmic_value_state_content_slot_acc": content_slot_acc.detach(),
        "algorithmic_value_state_step_exact": step_exact.detach(),
        "algorithmic_value_state_samples": samples,
        "algorithmic_value_state_content_samples": content_samples,
    }


def algorithmic_role_value_state_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    num_roles: int,
    value_vocab_size: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    staged_targets = role_value_targets_from_row(
        row,
        num_steps=int(num_depths),
        num_roles=int(num_roles),
        value_vocab_size=int(value_vocab_size),
    )
    if str(target_mode).lower() == "staged":
        targets = staged_targets
    else:
        targets = [[-100] * int(num_roles) for _ in range(int(num_depths))]
        if int(num_depths) > 0:
            final_row = dict(row)
            final_row["depth_targets"] = {
                str(int(num_depths)): target_for_core_steps(
                    row,
                    int(num_depths),
                    target_mode="final",
                )
            }
            final_targets = role_value_targets_from_row(
                final_row,
                num_steps=int(num_depths),
                num_roles=int(num_roles),
                value_vocab_size=int(value_vocab_size),
            )
            targets[-1] = list(final_targets[-1])
    return torch.tensor([targets], dtype=torch.long, device=device)


def core_role_value_prompt_parity_target(
    row: dict[str, Any],
    *,
    device: str,
):
    import torch

    base = row_mixed_list_base(row)
    target = -100 if base is None else int(base) % 2
    return torch.tensor([target], dtype=torch.long, device=device)


def core_role_value_prompt_parity_ce_loss(parity_logits, parity_targets):
    import torch
    import torch.nn.functional as F

    if parity_logits.ndim != 2:
        raise ValueError("parity_logits must have shape [batch, classes]")
    targets = parity_targets.to(device=parity_logits.device, dtype=torch.long)
    if targets.shape != parity_logits.shape[:1]:
        raise ValueError("parity targets must match parity logits batch")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = parity_logits.sum() * 0.0
        return zero, {
            "core_role_value_prompt_parity_ce": zero.detach(),
            "core_role_value_prompt_parity_acc": zero.detach(),
            "core_role_value_prompt_parity_samples": zero.detach(),
        }
    loss = F.cross_entropy(parity_logits.float()[mask], targets[mask])
    pred = parity_logits.detach().float().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = parity_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "core_role_value_prompt_parity_ce": loss.detach(),
        "core_role_value_prompt_parity_acc": acc.detach(),
        "core_role_value_prompt_parity_samples": samples,
    }


def algorithmic_role_value_state_ce_loss(role_logits, role_targets, *, role_weights=None):
    import torch
    import torch.nn.functional as F

    if role_logits.ndim != 4:
        raise ValueError("role_logits must have shape [batch, steps, roles, vocab]")
    targets = role_targets.to(device=role_logits.device, dtype=torch.long)
    if targets.shape != role_logits.shape[:3]:
        raise ValueError("role targets must match role logits batch/depth/role")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = role_logits.sum() * 0.0
        return zero, {
            "algorithmic_role_value_state_ce": zero.detach(),
            "algorithmic_role_value_state_acc": zero.detach(),
            "algorithmic_role_value_state_step_exact": zero.detach(),
            "algorithmic_role_value_state_samples": zero.detach(),
        }
    logits = role_logits.float()
    if role_weights is not None:
        weights = role_weights.to(device=role_logits.device, dtype=logits.dtype)
        if weights.shape != targets.shape:
            raise ValueError("role weights must match role targets")
        token_losses = F.cross_entropy(logits[mask], targets[mask], reduction="none")
        token_weights = weights[mask].clamp_min(0.0)
        weight_sum = token_weights.sum()
        if float(weight_sum.detach().item()) > 0.0:
            loss = (token_losses * token_weights).sum() / weight_sum
        else:
            loss = token_losses.mean()
    else:
        loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    step_mask = mask.any(dim=-1)
    exact_values = []
    for batch_index in range(int(targets.shape[0])):
        for step_index in range(int(targets.shape[1])):
            row_mask = mask[batch_index, step_index]
            if not bool(row_mask.any()):
                continue
            exact_values.append(
                (
                    pred[batch_index, step_index, row_mask]
                    == targets[batch_index, step_index, row_mask]
                )
                .all()
                .float()
            )
    if exact_values:
        step_exact = torch.stack(exact_values).mean()
    else:
        step_exact = loss.detach() * 0.0
    samples = role_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "algorithmic_role_value_state_ce": loss.detach(),
        "algorithmic_role_value_state_acc": acc.detach(),
        "algorithmic_role_value_state_step_exact": step_exact.detach(),
        "algorithmic_role_value_state_samples": samples,
    }


def core_primitive_typed_selector_bce_loss(
    selector_gate,
    primitive_logits,
    typed_logits,
    role_targets,
):
    import torch
    import torch.nn.functional as F

    if selector_gate.ndim != 3:
        raise ValueError("selector_gate must have shape [batch, steps, roles]")
    if primitive_logits.ndim != 4 or typed_logits.ndim != 4:
        raise ValueError("primitive/typed logits must have shape [batch, steps, roles, vocab]")
    steps = min(
        int(selector_gate.shape[1]),
        int(primitive_logits.shape[1]),
        int(typed_logits.shape[1]),
        int(role_targets.shape[1]),
    )
    roles = min(
        int(selector_gate.shape[2]),
        int(primitive_logits.shape[2]),
        int(typed_logits.shape[2]),
        int(role_targets.shape[2]),
    )
    vocab = min(int(primitive_logits.shape[3]), int(typed_logits.shape[3]))
    if steps <= 0 or roles <= 0 or vocab <= 1:
        zero = selector_gate.sum() * 0.0
        return zero, {
            "core_primitive_typed_selector_bce": zero.detach(),
            "core_primitive_typed_selector_acc": zero.detach(),
            "core_primitive_typed_selector_samples": zero.detach(),
            "core_primitive_typed_selector_primitive_target_rate": zero.detach(),
            "core_primitive_typed_selector_gate_mean": zero.detach(),
        }
    gate = selector_gate[:, :steps, :roles].float()
    primitive_pred = primitive_logits[:, :steps, :roles, :vocab].float().argmax(dim=-1)
    typed_pred = typed_logits[:, :steps, :roles, :vocab].float().argmax(dim=-1)
    targets = role_targets[:, :steps, :roles].to(
        device=selector_gate.device,
        dtype=torch.long,
    )
    mask = targets >= 0
    primitive_correct = primitive_pred == targets
    typed_correct = typed_pred == targets
    supervise = mask & (primitive_correct != typed_correct)
    if not bool(supervise.any()):
        zero = selector_gate.sum() * 0.0
        return zero, {
            "core_primitive_typed_selector_bce": zero.detach(),
            "core_primitive_typed_selector_acc": zero.detach(),
            "core_primitive_typed_selector_samples": zero.detach(),
            "core_primitive_typed_selector_primitive_target_rate": zero.detach(),
            "core_primitive_typed_selector_gate_mean": gate.detach().mean(),
        }
    labels = primitive_correct.float()
    gate_values = gate[supervise].clamp(1e-5, 1.0 - 1e-5)
    label_values = labels[supervise].float()
    loss = -(
        label_values * torch.log(gate_values)
        + (1.0 - label_values) * torch.log(1.0 - gate_values)
    ).mean()
    hard = gate.detach() >= 0.5
    acc = (hard[supervise] == labels[supervise].bool()).float().mean()
    samples = selector_gate.detach().new_tensor(float(int(supervise.sum().item())))
    primitive_rate = labels[supervise].detach().mean()
    return loss, {
        "core_primitive_typed_selector_bce": loss.detach(),
        "core_primitive_typed_selector_acc": acc.detach(),
        "core_primitive_typed_selector_samples": samples,
        "core_primitive_typed_selector_primitive_target_rate": primitive_rate.detach(),
        "core_primitive_typed_selector_gate_mean": gate.detach().mean(),
    }


def core_primitive_role_value_update_gate_bce_loss(
    update_gate,
    role_targets,
    initial_targets,
):
    import torch

    if update_gate.ndim != 3:
        raise ValueError("update_gate must have shape [batch, steps, roles]")
    targets = role_targets.to(device=update_gate.device, dtype=torch.long)
    initial = initial_targets.to(device=update_gate.device, dtype=torch.long)
    steps = min(int(update_gate.shape[1]), int(targets.shape[1]))
    roles = min(int(update_gate.shape[2]), int(targets.shape[2]), int(initial.shape[2]))
    if steps <= 0 or roles <= 0:
        zero = update_gate.sum() * 0.0
        return zero, {
            "core_primitive_role_value_update_gate_bce": zero.detach(),
            "core_primitive_role_value_update_gate_acc": zero.detach(),
            "core_primitive_role_value_update_gate_samples": zero.detach(),
            "core_primitive_role_value_update_gate_changed_rate": zero.detach(),
            "core_primitive_role_value_update_gate_mean": zero.detach(),
        }
    gate = update_gate[:, :steps, :roles].float()
    target = targets[:, :steps, :roles]
    prev_values = []
    for step in range(steps):
        if step == 0:
            prev_values.append(initial[:, 0, :roles])
        else:
            prev_values.append(targets[:, step - 1, :roles])
    prev = torch.stack(prev_values, dim=1).to(device=update_gate.device)
    mask = (target >= 0) & (prev >= 0)
    if not bool(mask.any()):
        zero = update_gate.sum() * 0.0
        return zero, {
            "core_primitive_role_value_update_gate_bce": zero.detach(),
            "core_primitive_role_value_update_gate_acc": zero.detach(),
            "core_primitive_role_value_update_gate_samples": zero.detach(),
            "core_primitive_role_value_update_gate_changed_rate": zero.detach(),
            "core_primitive_role_value_update_gate_mean": gate.detach().mean(),
        }
    labels = (target != prev).float()
    gate_values = gate[mask].clamp(1e-5, 1.0 - 1e-5)
    label_values = labels[mask]
    loss = -(
        label_values * torch.log(gate_values)
        + (1.0 - label_values) * torch.log(1.0 - gate_values)
    ).mean()
    hard = gate.detach() >= 0.5
    acc = (hard[mask] == labels[mask].bool()).float().mean()
    samples = update_gate.detach().new_tensor(float(int(mask.sum().item())))
    changed_rate = labels[mask].detach().mean()
    return loss, {
        "core_primitive_role_value_update_gate_bce": loss.detach(),
        "core_primitive_role_value_update_gate_acc": acc.detach(),
        "core_primitive_role_value_update_gate_samples": samples,
        "core_primitive_role_value_update_gate_changed_rate": changed_rate.detach(),
        "core_primitive_role_value_update_gate_mean": gate.detach().mean(),
    }


def token_numeric_source_slot_parity_ce_loss(parity_logits, source_slot_ids):
    import torch
    import torch.nn.functional as F

    if parity_logits.ndim != 3 or int(parity_logits.shape[-1]) != 2:
        raise ValueError("parity_logits must have shape [batch, slots, 2]")
    if source_slot_ids is None:
        zero = parity_logits.sum() * 0.0
        return zero, {
            "token_numeric_source_slot_parity_ce": zero.detach(),
            "token_numeric_source_slot_parity_acc": zero.detach(),
            "token_numeric_source_slot_parity_samples": zero.detach(),
        }
    ids = source_slot_ids.to(device=parity_logits.device, dtype=torch.long)
    slots = min(int(parity_logits.shape[1]), int(ids.shape[1]))
    if slots <= 0:
        zero = parity_logits.sum() * 0.0
        return zero, {
            "token_numeric_source_slot_parity_ce": zero.detach(),
            "token_numeric_source_slot_parity_acc": zero.detach(),
            "token_numeric_source_slot_parity_samples": zero.detach(),
        }
    logits = parity_logits[:, :slots, :].float()
    ids = ids[:, :slots]
    mask = ids > 0
    if not bool(mask.any()):
        zero = parity_logits.sum() * 0.0
        return zero, {
            "token_numeric_source_slot_parity_ce": zero.detach(),
            "token_numeric_source_slot_parity_acc": zero.detach(),
            "token_numeric_source_slot_parity_samples": zero.detach(),
        }
    values = ids - 1
    targets = ((values % 2) == 0).long()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = parity_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "token_numeric_source_slot_parity_ce": loss.detach(),
        "token_numeric_source_slot_parity_acc": acc.detach(),
        "token_numeric_source_slot_parity_samples": samples,
    }


def token_numeric_source_slot_predicate_ce_loss(predicate_logits, source_slot_ids):
    loss, metrics = token_numeric_source_slot_parity_ce_loss(
        predicate_logits,
        source_slot_ids,
    )
    return loss, {
        key.replace("parity", "predicate"): value
        for key, value in metrics.items()
    }


def algorithmic_role_value_scalar_role_weights(role_targets, *, multiplier: float):
    import torch

    targets = role_targets
    if targets.ndim != 3:
        raise ValueError("role targets must have shape [batch, steps, roles]")
    role_count = int(targets.shape[-1])
    if role_count < 2 or float(multiplier) == 1.0:
        return None
    weights = torch.ones_like(targets, dtype=torch.float32)
    weights[..., role_count - 2 :] = float(multiplier)
    return weights


def algorithmic_role_value_initial_state_targets(
    row: dict[str, Any],
    *,
    num_steps: int,
    num_roles: int,
    value_vocab_size: int,
    device: str,
    include_metadata: bool = False,
):
    import torch

    targets = [[-100] * int(num_roles) for _ in range(int(num_steps))]
    if int(num_steps) <= 0:
        return torch.tensor([targets], dtype=torch.long, device=device)
    base = row_mixed_list_base(row)
    if base is None:
        list_mode = str(
            row.get("role_value_list_class_mode")
            or row.get("list_class_mode")
            or ""
        ).strip().lower()
        values = row_input_list(row) if list_mode == "absolute" else None
        if values:
            max_list_fields = max(1, (int(num_roles) - 2) // 2)
            for role_index, value in enumerate(values[:max_list_fields]):
                class_id = int(value) + 1
                if 0 <= class_id < int(value_vocab_size):
                    targets[0][role_index] = int(class_id)
        elif list_mode == "source_position":
            values = row_input_list(row)
            if values:
                max_list_fields = max(1, (int(num_roles) - 2) // 2)
                for role_index, _value in enumerate(values[:max_list_fields]):
                    class_id = int(role_index) + 1
                    if 0 <= class_id < int(value_vocab_size):
                        targets[0][role_index] = int(class_id)
        return torch.tensor([targets], dtype=torch.long, device=device)
    raw_length = row.get("list_length")
    try:
        length = int(raw_length)
    except (TypeError, ValueError):
        length = 0
    max_list_fields = max(1, (int(num_roles) - 2) // 2)
    doubled_role_start = max_list_fields
    for offset in range(min(length, max_list_fields)):
        class_id = int(offset) + 1
        if 0 <= class_id < int(value_vocab_size):
            targets[0][offset] = class_id
    scalar_coeff_role = 2 * max_list_fields
    scalar_residual_role = scalar_coeff_role + 1
    if scalar_coeff_role < int(num_roles):
        targets[0][scalar_coeff_role] = int(base) % 2 + 1
    if bool(include_metadata) and scalar_residual_role < int(num_roles):
        depth_targets = row.get("depth_targets")
        even_offsets = (
            mixed_even_offsets(row, depth_targets)
            if isinstance(depth_targets, dict)
            else None
        )
        if even_offsets:
            coeff_class = 2 * len(even_offsets) + 1
            if 0 <= coeff_class < int(value_vocab_size):
                targets[0][doubled_role_start] = int(coeff_class)
        try:
            offset_class = int(row.get("mixed_offset")) + 1
        except (TypeError, ValueError):
            offset_class = -1
        if 0 <= offset_class < int(value_vocab_size):
            targets[0][scalar_residual_role] = int(offset_class)
    return torch.tensor([targets], dtype=torch.long, device=device)


def algorithmic_role_value_initial_source_value_targets(
    row: dict[str, Any],
    *,
    num_steps: int,
    num_roles: int,
    value_vocab_size: int,
    device: str,
):
    import torch

    targets = [[-100] * int(num_roles) for _ in range(int(num_steps))]
    if int(num_steps) <= 0:
        return torch.tensor([targets], dtype=torch.long, device=device)
    values = row_input_list(row)
    if not values:
        return torch.tensor([targets], dtype=torch.long, device=device)
    max_list_fields = max(1, (int(num_roles) - 2) // 2)
    for role_index in range(max_list_fields):
        if role_index < len(values):
            class_id = int(values[role_index]) + 1
            if 0 <= class_id < int(value_vocab_size):
                targets[0][role_index] = int(class_id)
        else:
            targets[0][role_index] = 0
    return torch.tensor([targets], dtype=torch.long, device=device)


def algorithmic_role_value_step_margin_loss(role_logits, role_targets, *, margin: float):
    import torch
    import torch.nn.functional as F

    if role_logits.ndim != 4:
        raise ValueError("role_logits must have shape [batch, steps, roles, vocab]")
    targets = role_targets.to(device=role_logits.device, dtype=torch.long)
    if targets.shape != role_logits.shape[:3]:
        raise ValueError("role targets must match role logits batch/depth/role")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = role_logits.sum() * 0.0
        return zero, {
            "algorithmic_role_value_step_margin": zero.detach(),
            "algorithmic_role_value_step_margin_win_rate": zero.detach(),
            "algorithmic_role_value_step_margin_pass_rate": zero.detach(),
            "algorithmic_role_value_step_margin_samples": zero.detach(),
        }

    logits = role_logits.float()
    safe_targets = targets.masked_fill(mask.logical_not(), 0)
    target_logits = logits.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    competitor_logits = logits.clone()
    competitor_logits.scatter_(dim=-1, index=safe_targets.unsqueeze(-1), value=-1.0e9)
    top_competitor = competitor_logits.max(dim=-1).values
    violations = F.relu(float(margin) + top_competitor - target_logits)
    step_mask = mask.any(dim=-1)
    step_violations = violations.masked_fill(mask.logical_not(), -1.0e9).amax(dim=-1)
    loss = step_violations[step_mask].mean()
    role_wins = (target_logits >= top_competitor + float(margin)).to(logits.dtype)
    win_rate = role_wins[mask].mean()
    step_pass = role_wins.masked_fill(mask.logical_not(), 1.0).bool().all(dim=-1)
    pass_rate = step_pass[step_mask].to(logits.dtype).mean()
    samples = role_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "algorithmic_role_value_step_margin": loss.detach(),
        "algorithmic_role_value_step_margin_win_rate": win_rate.detach(),
        "algorithmic_role_value_step_margin_pass_rate": pass_rate.detach(),
        "algorithmic_role_value_step_margin_samples": samples,
    }


def algorithmic_role_value_trace_margin_loss(role_logits, role_targets, *, margin: float):
    import torch
    import torch.nn.functional as F

    if role_logits.ndim != 4:
        raise ValueError("role_logits must have shape [batch, steps, roles, vocab]")
    targets = role_targets.to(device=role_logits.device, dtype=torch.long)
    if targets.shape != role_logits.shape[:3]:
        raise ValueError("role targets must match role logits batch/depth/role")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = role_logits.sum() * 0.0
        return zero, {
            "algorithmic_role_value_trace_margin": zero.detach(),
            "algorithmic_role_value_trace_margin_win_rate": zero.detach(),
            "algorithmic_role_value_trace_margin_step_pass_rate": zero.detach(),
            "algorithmic_role_value_trace_margin_trace_pass_rate": zero.detach(),
            "algorithmic_role_value_trace_margin_samples": zero.detach(),
        }

    logits = role_logits.float()
    safe_targets = targets.masked_fill(mask.logical_not(), 0)
    target_logits = logits.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    competitor_logits = logits.clone()
    competitor_logits.scatter_(dim=-1, index=safe_targets.unsqueeze(-1), value=-1.0e9)
    top_competitor = competitor_logits.max(dim=-1).values
    violations = F.relu(float(margin) + top_competitor - target_logits)
    trace_mask = mask.any(dim=(1, 2))
    trace_violations = violations.masked_fill(mask.logical_not(), -1.0e9).amax(
        dim=(1, 2)
    )
    loss = trace_violations[trace_mask].mean()
    role_wins = (target_logits >= top_competitor + float(margin)).to(logits.dtype)
    win_rate = role_wins[mask].mean()
    step_mask = mask.any(dim=-1)
    step_pass = role_wins.masked_fill(mask.logical_not(), 1.0).bool().all(dim=-1)
    step_pass_rate = step_pass[step_mask].to(logits.dtype).mean()
    trace_pass = step_pass.masked_fill(step_mask.logical_not(), True).all(dim=-1)
    trace_pass_rate = trace_pass[trace_mask].to(logits.dtype).mean()
    samples = role_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "algorithmic_role_value_trace_margin": loss.detach(),
        "algorithmic_role_value_trace_margin_win_rate": win_rate.detach(),
        "algorithmic_role_value_trace_margin_step_pass_rate": step_pass_rate.detach(),
        "algorithmic_role_value_trace_margin_trace_pass_rate": trace_pass_rate.detach(),
        "algorithmic_role_value_trace_margin_samples": samples,
    }


def core_primitive_role_value_pair_trace_contrastive_loss(
    role_logits,
    positive_targets,
    negative_targets,
    *,
    margin: float,
):
    import torch
    import torch.nn.functional as F

    if role_logits.ndim != 4:
        raise ValueError("role_logits must have shape [batch, steps, roles, vocab]")
    positive = positive_targets.to(device=role_logits.device, dtype=torch.long)
    negative = negative_targets.to(device=role_logits.device, dtype=torch.long)
    steps = min(int(role_logits.shape[1]), int(positive.shape[1]), int(negative.shape[1]))
    roles = min(int(role_logits.shape[2]), int(positive.shape[2]), int(negative.shape[2]))
    vocab = int(role_logits.shape[3])
    if steps <= 0 or roles <= 0 or vocab <= 0:
        zero = role_logits.sum() * 0.0
        return zero, {
            "core_primitive_role_value_pair_trace_contrast": zero.detach(),
            "core_primitive_role_value_pair_trace_contrast_win_rate": zero.detach(),
            "core_primitive_role_value_pair_trace_contrast_samples": zero.detach(),
        }
    logits = role_logits[:, :steps, :roles, :].float()
    positive = positive[:, :steps, :roles]
    negative = negative[:, :steps, :roles]
    mask = (positive >= 0) & (negative >= 0) & (positive != negative)
    if not bool(mask.any()):
        zero = role_logits.sum() * 0.0
        return zero, {
            "core_primitive_role_value_pair_trace_contrast": zero.detach(),
            "core_primitive_role_value_pair_trace_contrast_win_rate": zero.detach(),
            "core_primitive_role_value_pair_trace_contrast_samples": zero.detach(),
        }

    safe_positive = positive.clamp(min=0, max=vocab - 1)
    safe_negative = negative.clamp(min=0, max=vocab - 1)
    log_probs = F.log_softmax(logits, dim=-1)
    positive_logp = log_probs.gather(
        dim=-1,
        index=safe_positive.unsqueeze(-1),
    ).squeeze(-1)
    negative_logp = log_probs.gather(
        dim=-1,
        index=safe_negative.unsqueeze(-1),
    ).squeeze(-1)
    valid_steps = mask.any(dim=-1)
    positive_trace_score = positive_logp.masked_fill(mask.logical_not(), 0.0).sum(dim=-1)
    negative_trace_score = negative_logp.masked_fill(mask.logical_not(), 0.0).sum(dim=-1)
    violations = F.relu(
        float(margin) + negative_trace_score - positive_trace_score
    )
    loss = violations[valid_steps].mean()
    wins = (positive_trace_score >= negative_trace_score + float(margin)).to(
        logits.dtype
    )
    win_rate = wins[valid_steps].mean()
    samples = role_logits.detach().new_tensor(float(int(valid_steps.sum().item())))
    return loss, {
        "core_primitive_role_value_pair_trace_contrast": loss.detach(),
        "core_primitive_role_value_pair_trace_contrast_win_rate": win_rate.detach(),
        "core_primitive_role_value_pair_trace_contrast_samples": samples,
    }


def typed_algorithmic_value_state_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    max_list_slots: int,
    offset_vocab_size: int,
    scalar_vocab_size: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if str(target_mode).lower() != "staged":
        raw = {
            "kind": [-100] * int(num_depths),
            "raw_list_offsets": [
                [-100] * int(max_list_slots) for _ in range(int(num_depths))
            ],
            "doubled_list_offsets": [
                [-100] * int(max_list_slots) for _ in range(int(num_depths))
            ],
            "scalar_coeff": [-100] * int(num_depths),
            "scalar_offset": [-100] * int(num_depths),
            "scalar_residual": [-100] * int(num_depths),
            "scalar_residual_delta": [-100] * int(num_depths),
            "final_residual": [-100] * int(num_depths),
        }
    else:
        raw = typed_algorithmic_field_targets_from_row(
            row,
            num_steps=int(num_depths),
            max_list_slots=int(max_list_slots),
            offset_vocab_size=int(offset_vocab_size),
            scalar_vocab_size=int(scalar_vocab_size),
        )
    return {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in raw.items()
    }


def typed_algorithmic_value_state_ce_loss(
    logits: dict[str, Any],
    targets: dict[str, Any],
    *,
    pad_ce_weight: float = 0.0,
    kind_ce_multiplier: float = 1.0,
    list_ce_multiplier: float = 1.0,
    scalar_ce_multiplier: float = 1.0,
    residual_delta_ce_multiplier: float = 0.0,
    scalar_ordinal_weight: float = 0.0,
    scalar_regression_weight: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    required = {
        "kind": "kind_logits",
        "raw_list_offsets": "raw_list_offset_logits",
        "doubled_list_offsets": "doubled_list_offset_logits",
        "scalar_coeff": "scalar_coeff_logits",
        "scalar_residual": "scalar_residual_logits",
        "final_residual": "final_residual_logits",
    }
    for target_key, logit_key in required.items():
        if logit_key not in logits:
            raise ValueError(f"missing typed algorithmic logits: {logit_key}")
        if target_key not in targets:
            raise ValueError(f"missing typed algorithmic targets: {target_key}")
    kind_logits = logits["kind_logits"]
    if kind_logits.ndim != 3:
        raise ValueError("kind_logits must have shape [batch, steps, kinds]")
    losses = []
    correct = kind_logits.new_tensor(0.0)
    total = kind_logits.new_tensor(0.0)
    content_correct = kind_logits.new_tensor(0.0)
    content_total = kind_logits.new_tensor(0.0)
    step_exact_values = []

    kind_targets = targets["kind"].to(device=kind_logits.device, dtype=torch.long)
    kind_mask = kind_targets >= 0
    kind_ce_multiplier = max(0.0, float(kind_ce_multiplier))
    list_ce_multiplier = max(0.0, float(list_ce_multiplier))
    scalar_ce_multiplier = max(0.0, float(scalar_ce_multiplier))
    residual_delta_ce_multiplier = max(0.0, float(residual_delta_ce_multiplier))
    scalar_ordinal_weight = max(0.0, float(scalar_ordinal_weight))
    scalar_regression_weight = max(0.0, float(scalar_regression_weight))
    if bool(kind_mask.any()) and kind_ce_multiplier > 0.0:
        kind_loss = F.cross_entropy(kind_logits.float()[kind_mask], kind_targets[kind_mask])
        losses.append(kind_loss * kind_ce_multiplier)
    if bool(kind_mask.any()):
        kind_pred = kind_logits.detach().argmax(dim=-1)
        hits = (kind_pred[kind_mask] == kind_targets[kind_mask]).float()
        correct = correct + hits.sum()
        total = total + hits.new_tensor(float(hits.numel()))
    else:
        kind_pred = kind_targets.new_full(kind_targets.shape, -1)

    pad_ce_weight = max(0.0, float(pad_ce_weight))
    field_preds: dict[str, Any] = {"kind": kind_pred}

    def _field_loss(field: str, logit_key: str, *, multiplier: float):
        nonlocal correct, total, content_correct, content_total
        field_logits = logits[logit_key]
        field_targets = targets[field].to(device=field_logits.device, dtype=torch.long)
        if field_logits.shape[: field_targets.ndim] != field_targets.shape:
            raise ValueError(f"{logit_key} target shape mismatch")
        pred = field_logits.detach().argmax(dim=-1)
        field_preds[field] = pred
        if field_targets.ndim == 3:
            loss_mask = field_targets > 0
            pad_mask = (field_targets == 0) & (pad_ce_weight > 0.0)
            metric_mask = field_targets >= 0
            loss_mask = loss_mask | pad_mask
        else:
            loss_mask = field_targets >= 0
            metric_mask = loss_mask
        if bool(loss_mask.any()) and float(multiplier) > 0.0:
            flat_logits = field_logits.float()[loss_mask]
            flat_targets = field_targets[loss_mask]
            if field_targets.ndim == 3 and pad_ce_weight not in {0.0, 1.0}:
                per_item = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                weights = torch.ones_like(per_item)
                weights = torch.where(
                    flat_targets == 0,
                    weights * float(pad_ce_weight),
                    weights,
                )
                losses.append(
                    ((per_item * weights).sum() / weights.sum().clamp_min(1.0))
                    * float(multiplier)
                )
            else:
                losses.append(F.cross_entropy(flat_logits, flat_targets) * float(multiplier))
        if bool(metric_mask.any()):
            hits = (pred[metric_mask] == field_targets[metric_mask]).float()
            correct = correct + hits.sum()
            total = total + hits.new_tensor(float(hits.numel()))
        content_mask = field_targets > 0
        if bool(content_mask.any()):
            hits = (pred[content_mask] == field_targets[content_mask]).float()
            content_correct = content_correct + hits.sum()
            content_total = content_total + hits.new_tensor(float(hits.numel()))

    ordinal_abs_error = kind_logits.new_tensor(0.0)
    ordinal_total = kind_logits.new_tensor(0.0)
    regression_abs_error = kind_logits.new_tensor(0.0)
    regression_total = kind_logits.new_tensor(0.0)

    def _scalar_ordinal_loss(field: str, logit_key: str):
        nonlocal ordinal_abs_error, ordinal_total
        if scalar_ordinal_weight <= 0.0:
            return
        if field not in targets or logit_key not in logits:
            return
        field_logits = logits[logit_key]
        field_targets = targets[field].to(device=field_logits.device, dtype=torch.long)
        if field_logits.shape[: field_targets.ndim] != field_targets.shape:
            raise ValueError(f"{logit_key} target shape mismatch")
        mask = field_targets >= 0
        if not bool(mask.any()):
            return
        vocab = int(field_logits.shape[-1])
        class_values = torch.arange(
            vocab,
            device=field_logits.device,
            dtype=field_logits.float().dtype,
        )
        view_shape = [1] * (field_logits.ndim - 1) + [vocab]
        expected = (
            F.softmax(field_logits.float(), dim=-1)
            * class_values.view(*view_shape)
        ).sum(dim=-1)
        denom = max(1.0, float(vocab - 1))
        losses.append(
            F.smooth_l1_loss(
                expected[mask] / denom,
                field_targets.float()[mask] / denom,
            )
            * scalar_ordinal_weight
        )
        ordinal_abs_error = ordinal_abs_error + (
            expected.detach()[mask] - field_targets.float()[mask]
        ).abs().sum()
        ordinal_total = ordinal_total + field_targets.new_tensor(
            float(mask.sum().item()),
            dtype=kind_logits.dtype,
        )

    def _scalar_regression_loss(field: str, value_key: str, logit_key: str):
        nonlocal regression_abs_error, regression_total
        if scalar_regression_weight <= 0.0:
            return
        if field not in targets or value_key not in logits or logit_key not in logits:
            return
        field_values = logits[value_key]
        field_logits = logits[logit_key]
        field_targets = targets[field].to(device=field_values.device, dtype=torch.long)
        if field_values.shape != field_targets.shape:
            raise ValueError(f"{value_key} target shape mismatch")
        mask = field_targets >= 0
        if not bool(mask.any()):
            return
        vocab = int(field_logits.shape[-1])
        denom = max(1.0, float(vocab - 1))
        pred = field_values.float()
        target = field_targets.float() / denom
        losses.append(
            F.smooth_l1_loss(pred[mask], target[mask]) * scalar_regression_weight
        )
        rounded = (pred.detach()[mask].clamp(0.0, 1.0) * denom).round()
        regression_abs_error = regression_abs_error + (
            rounded - field_targets.float()[mask]
        ).abs().sum()
        regression_total = regression_total + field_targets.new_tensor(
            float(mask.sum().item()),
            dtype=kind_logits.dtype,
        )

    _field_loss("raw_list_offsets", "raw_list_offset_logits", multiplier=list_ce_multiplier)
    _field_loss("doubled_list_offsets", "doubled_list_offset_logits", multiplier=list_ce_multiplier)
    _field_loss("scalar_coeff", "scalar_coeff_logits", multiplier=scalar_ce_multiplier)
    if "scalar_offset" in targets and "scalar_offset_logits" in logits:
        _field_loss("scalar_offset", "scalar_offset_logits", multiplier=scalar_ce_multiplier)
    _field_loss("scalar_residual", "scalar_residual_logits", multiplier=scalar_ce_multiplier)
    if (
        "scalar_residual_delta" in targets
        and "scalar_residual_delta_logits" in logits
    ):
        _field_loss(
            "scalar_residual_delta",
            "scalar_residual_delta_logits",
            multiplier=residual_delta_ce_multiplier,
        )
    _field_loss("final_residual", "final_residual_logits", multiplier=scalar_ce_multiplier)
    _scalar_ordinal_loss("scalar_offset", "scalar_offset_logits")
    _scalar_ordinal_loss("scalar_residual", "scalar_residual_logits")
    _scalar_ordinal_loss("scalar_residual_delta", "scalar_residual_delta_logits")
    _scalar_ordinal_loss("final_residual", "final_residual_logits")
    _scalar_regression_loss(
        "scalar_coeff",
        "scalar_coeff_value",
        "scalar_coeff_logits",
    )
    _scalar_regression_loss(
        "scalar_offset",
        "scalar_offset_value",
        "scalar_offset_logits",
    )
    _scalar_regression_loss(
        "scalar_residual",
        "scalar_residual_value",
        "scalar_residual_logits",
    )
    _scalar_regression_loss(
        "final_residual",
        "final_residual_value",
        "final_residual_logits",
    )

    for batch_index in range(int(kind_targets.shape[0])):
        for step_index in range(int(kind_targets.shape[1])):
            if int(kind_targets[batch_index, step_index].item()) < 0:
                continue
            matches = [
                field_preds["kind"][batch_index, step_index]
                == kind_targets[batch_index, step_index]
            ]
            for field in ("raw_list_offsets", "doubled_list_offsets"):
                mask = targets[field][batch_index, step_index].to(
                    device=field_preds[field].device
                ) >= 0
                if bool(mask.any()):
                    matches.append(
                        (
                            field_preds[field][batch_index, step_index, mask]
                            == targets[field][batch_index, step_index].to(
                                device=field_preds[field].device
                            )[mask]
                        ).all()
                    )
            for field in (
                "scalar_coeff",
                "scalar_offset",
                "scalar_residual",
                "scalar_residual_delta",
                "final_residual",
            ):
                if field not in targets or field not in field_preds:
                    continue
                target_value = targets[field][batch_index, step_index].to(
                    device=field_preds[field].device
                )
                if int(target_value.item()) >= 0:
                    matches.append(
                        field_preds[field][batch_index, step_index] == target_value
                    )
            step_exact_values.append(torch.stack([m.float() for m in matches]).all())

    loss = torch.stack(losses).sum() if losses else kind_logits.sum() * 0.0
    acc = correct / total.clamp_min(1.0)
    content_acc = content_correct / content_total.clamp_min(1.0)
    if step_exact_values:
        step_exact = torch.stack([value.float() for value in step_exact_values]).mean()
    else:
        step_exact = loss.detach() * 0.0
    return loss, {
        "typed_algorithmic_value_state_ce": loss.detach(),
        "typed_algorithmic_value_state_acc": acc.detach(),
        "typed_algorithmic_value_state_content_acc": content_acc.detach(),
        "typed_algorithmic_value_state_step_exact": step_exact.detach(),
        "typed_algorithmic_value_state_samples": total.detach(),
        "typed_algorithmic_value_state_content_samples": content_total.detach(),
        "typed_algorithmic_scalar_ordinal_mae": (
            ordinal_abs_error / ordinal_total.clamp_min(1.0)
        ).detach(),
        "typed_algorithmic_scalar_regression_mae": (
            regression_abs_error / regression_total.clamp_min(1.0)
        ).detach(),
    }


def algorithmic_role_value_transition_ce_loss(transition_logits, role_targets):
    import torch
    import torch.nn.functional as F

    if transition_logits.ndim != 4:
        raise ValueError(
            "transition_logits must have shape [batch, transitions, roles, vocab]"
        )
    targets = role_targets.to(device=transition_logits.device, dtype=torch.long)
    if targets.ndim != 3:
        raise ValueError("role targets must have shape [batch, steps, roles]")
    transitions = int(transition_logits.shape[1])
    if targets.shape[0] != transition_logits.shape[0]:
        raise ValueError("role targets batch must match transition logits")
    if targets.shape[2] != transition_logits.shape[2]:
        raise ValueError("role targets role dimension must match transition logits")
    if targets.shape[1] < transitions + 1:
        raise ValueError("role targets must include one more step than transitions")
    shifted_targets = targets[:, 1 : transitions + 1, :]
    mask = shifted_targets >= 0
    if not bool(mask.any()):
        zero = transition_logits.sum() * 0.0
        return zero, {
            "algorithmic_role_value_transition_ce": zero.detach(),
            "algorithmic_role_value_transition_acc": zero.detach(),
            "algorithmic_role_value_transition_step_exact": zero.detach(),
            "algorithmic_role_value_transition_samples": zero.detach(),
        }
    logits = transition_logits.float()
    loss = F.cross_entropy(logits[mask], shifted_targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == shifted_targets[mask]).float().mean()
    exact_values = []
    for batch_index in range(int(shifted_targets.shape[0])):
        for transition_index in range(int(shifted_targets.shape[1])):
            row_mask = mask[batch_index, transition_index]
            if not bool(row_mask.any()):
                continue
            exact_values.append(
                (
                    pred[batch_index, transition_index, row_mask]
                    == shifted_targets[batch_index, transition_index, row_mask]
                )
                .all()
                .float()
            )
    if exact_values:
        step_exact = torch.stack(exact_values).mean()
    else:
        step_exact = loss.detach() * 0.0
    samples = transition_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "algorithmic_role_value_transition_ce": loss.detach(),
        "algorithmic_role_value_transition_acc": acc.detach(),
        "algorithmic_role_value_transition_step_exact": step_exact.detach(),
        "algorithmic_role_value_transition_samples": samples,
    }


def transition_state_first_token_ce_loss(transition_state_text_logits, staged_target_ids):
    import torch
    import torch.nn.functional as F

    if transition_state_text_logits.ndim != 3:
        raise ValueError(
            "transition_state_text_logits must have shape [batch, steps, vocab]"
        )
    targets = staged_target_ids.to(
        device=transition_state_text_logits.device,
        dtype=torch.long,
    )
    if targets.shape != transition_state_text_logits.shape[:2]:
        raise ValueError("staged_target_ids must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = transition_state_text_logits.sum() * 0.0
        return zero, {
            "transition_state_first_token_ce": zero.detach(),
            "transition_state_first_token_acc": zero.detach(),
            "transition_state_first_token_samples": zero.detach(),
        }
    logits = transition_state_text_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = transition_state_text_logits.detach().new_tensor(
        float(int(mask.sum().item()))
    )
    return loss, {
        "transition_state_first_token_ce": loss.detach(),
        "transition_state_first_token_acc": acc.detach(),
        "transition_state_first_token_samples": samples,
    }


def transition_state_depth_contrast_loss(
    transition_state_text_logits,
    staged_target_ids,
    *,
    margin: float = 0.10,
):
    import torch

    if transition_state_text_logits.ndim != 3:
        raise ValueError(
            "transition_state_text_logits must have shape [batch, steps, vocab]"
        )
    targets = staged_target_ids.to(
        device=transition_state_text_logits.device,
        dtype=torch.long,
    )
    if targets.shape != transition_state_text_logits.shape[:2]:
        raise ValueError("staged_target_ids must have shape [batch, steps]")
    logits = transition_state_text_logits.float()
    losses = []
    for batch_index in range(int(targets.shape[0])):
        labelled = targets[batch_index] >= 0
        if not bool(labelled.any()):
            continue
        row_targets = targets[batch_index, labelled]
        unique_targets = torch.unique(row_targets)
        for step_index in torch.where(labelled)[0]:
            target = targets[batch_index, step_index]
            negatives = unique_targets[unique_targets != target]
            if negatives.numel() == 0:
                continue
            pos = logits[batch_index, step_index, target]
            neg = logits[batch_index, step_index, negatives]
            losses.append(torch.relu(float(margin) + neg - pos).mean())
    if not losses:
        zero = transition_state_text_logits.sum() * 0.0
        return zero, {
            "transition_state_depth_contrast": zero.detach(),
            "transition_state_depth_contrast_pairs": zero.detach(),
        }
    loss = torch.stack(losses).mean()
    pairs = transition_state_text_logits.detach().new_tensor(float(len(losses)))
    return loss, {
        "transition_state_depth_contrast": loss.detach(),
        "transition_state_depth_contrast_pairs": pairs,
    }


def transition_state_code_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_depths: int,
    codebook_size: int,
    device: str,
    target_mode: str = "staged",
):
    import torch

    if int(codebook_size) <= 0:
        raise ValueError("codebook_size must be positive")
    explicit_codes = row.get("transition_state_codes")
    if isinstance(explicit_codes, dict) and explicit_codes:
        targets: list[int] = []
        for depth in range(1, int(num_depths) + 1):
            raw_code = explicit_codes.get(str(depth))
            if raw_code is None:
                targets.append(-100)
                continue
            code = int(raw_code)
            if code < 0 or code >= int(codebook_size):
                raise ValueError(
                    "transition_state_codes must be within the transition-state codebook"
                )
            targets.append(code)
        return torch.tensor([targets], dtype=torch.long, device=device)
    first_token_targets = staged_internal_first_token_targets(
        tokenizer,
        row,
        num_depths=num_depths,
        device=device,
        target_mode=target_mode,
    )
    targets = first_token_targets.clone()
    mask = targets >= 0
    targets[mask] = targets[mask] % int(codebook_size)
    return targets


def transition_state_code_ce_loss(transition_state_code_logits, code_targets):
    import torch
    import torch.nn.functional as F

    if transition_state_code_logits.ndim != 3:
        raise ValueError(
            "transition_state_code_logits must have shape [batch, steps, codebook]"
        )
    targets = code_targets.to(
        device=transition_state_code_logits.device,
        dtype=torch.long,
    )
    if targets.shape != transition_state_code_logits.shape[:2]:
        raise ValueError("code_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = transition_state_code_logits.sum() * 0.0
        return zero, {
            "transition_state_code_ce": zero.detach(),
            "transition_state_code_acc": zero.detach(),
            "transition_state_code_samples": zero.detach(),
        }
    logits = transition_state_code_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = transition_state_code_logits.detach().new_tensor(
        float(int(mask.sum().item()))
    )
    return loss, {
        "transition_state_code_ce": loss.detach(),
        "transition_state_code_acc": acc.detach(),
        "transition_state_code_samples": samples,
    }


def transition_state_finality_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    device: str,
):
    import torch

    targets = [-100.0] * int(num_depths)
    explicit_targets = row.get("transition_finality_targets")
    if not isinstance(explicit_targets, dict):
        return torch.tensor([targets], dtype=torch.float32, device=device)
    for depth in range(1, int(num_depths) + 1):
        raw_target = explicit_targets.get(str(depth))
        if raw_target is None:
            continue
        value = float(raw_target)
        if value not in {0.0, 1.0}:
            raise ValueError("transition_finality_targets must be binary")
        targets[depth - 1] = value
    return torch.tensor([targets], dtype=torch.float32, device=device)


def transition_state_finality_bce_loss(finality_logits, finality_targets):
    import torch
    import torch.nn.functional as F

    if finality_logits.ndim != 2:
        raise ValueError("finality_logits must have shape [batch, steps]")
    targets = finality_targets.to(device=finality_logits.device, dtype=torch.float32)
    if targets.shape != finality_logits.shape:
        raise ValueError("finality_targets must have shape [batch, steps]")
    mask = targets >= 0.0
    if not bool(mask.any()):
        zero = finality_logits.sum() * 0.0
        return zero, {
            "transition_state_finality_bce": zero.detach(),
            "transition_state_finality_acc": zero.detach(),
            "transition_state_finality_samples": zero.detach(),
        }
    logits = finality_logits.float()
    loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
    pred = (torch.sigmoid(logits.detach()) >= 0.5).float()
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = finality_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "transition_state_finality_bce": loss.detach(),
        "transition_state_finality_acc": acc.detach(),
        "transition_state_finality_samples": samples,
    }


def transition_state_joint_targets(
    row: dict[str, Any],
    *,
    num_depths: int,
    joint_size: int,
    device: str,
):
    import torch

    if int(joint_size) <= 0:
        raise ValueError("joint_size must be positive")
    codes = row.get("transition_state_codes")
    finality = row.get("transition_finality_targets")
    if not isinstance(codes, dict) or not isinstance(finality, dict):
        return torch.full((1, int(num_depths)), -100, dtype=torch.long, device=device)
    targets: list[int] = []
    for depth in range(1, int(num_depths) + 1):
        raw_code = codes.get(str(depth))
        raw_finality = finality.get(str(depth))
        if raw_code is None or raw_finality is None:
            targets.append(-100)
            continue
        code = int(raw_code)
        final_bit = int(float(raw_finality) > 0.0)
        joint = code * 2 + final_bit
        if joint < 0 or joint >= int(joint_size):
            raise ValueError(
                "transition_state_codes and transition_finality_targets exceed "
                "the transition-state joint size"
            )
        targets.append(joint)
    return torch.tensor([targets], dtype=torch.long, device=device)


def transition_state_joint_ce_loss(transition_state_joint_logits, joint_targets):
    import torch
    import torch.nn.functional as F

    if transition_state_joint_logits.ndim != 3:
        raise ValueError(
            "transition_state_joint_logits must have shape [batch, steps, joint_size]"
        )
    targets = joint_targets.to(
        device=transition_state_joint_logits.device,
        dtype=torch.long,
    )
    if targets.shape != transition_state_joint_logits.shape[:2]:
        raise ValueError("joint_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = transition_state_joint_logits.sum() * 0.0
        return zero, {
            "transition_state_joint_ce": zero.detach(),
            "transition_state_joint_acc": zero.detach(),
            "transition_state_joint_samples": zero.detach(),
        }
    logits = transition_state_joint_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = transition_state_joint_logits.detach().new_tensor(
        float(int(mask.sum().item()))
    )
    return loss, {
        "transition_state_joint_ce": loss.detach(),
        "transition_state_joint_acc": acc.detach(),
        "transition_state_joint_samples": samples,
    }


OPPOSITE_COMPOSITION_CODE_SEQUENCES = {
    "list_to_arithmetic": (0, 2, 3, 1, 1, 4, 4, 4),
    "arithmetic_to_list": (0, 1, 2, 3, 4, 4, 4, 4),
}


def _composition_order_for_row(row: dict[str, Any]) -> str:
    order = str(row.get("composition_order") or "")
    if order:
        return order
    family = str(row.get("task_family") or row.get("category") or "")
    if family == "mixed_list_arithmetic":
        return "list_to_arithmetic"
    if family == "mixed_arithmetic_list":
        return "arithmetic_to_list"
    return ""


def transition_state_joint_order_contrast_loss(
    transition_state_joint_logits,
    row: dict[str, Any],
    *,
    margin: float = 0.10,
):
    import torch

    if transition_state_joint_logits.ndim != 3:
        raise ValueError(
            "transition_state_joint_logits must have shape [batch, steps, joint_size]"
        )
    b, steps, joint_size = transition_state_joint_logits.shape
    if int(b) != 1:
        raise ValueError("transition_state_joint_order_contrast_loss expects batch size 1")
    order = _composition_order_for_row(row)
    opposite_codes = OPPOSITE_COMPOSITION_CODE_SEQUENCES.get(order)
    finality = row.get("transition_finality_targets")
    if opposite_codes is None or not isinstance(finality, dict):
        zero = transition_state_joint_logits.sum() * 0.0
        return zero, {
            "transition_state_joint_order_contrast": zero.detach(),
            "transition_state_joint_order_contrast_win_rate": zero.detach(),
            "transition_state_joint_order_contrast_pairs": zero.detach(),
        }
    target_joints = transition_state_joint_targets(
        row,
        num_depths=int(steps),
        joint_size=int(joint_size),
        device=str(transition_state_joint_logits.device),
    ).to(device=transition_state_joint_logits.device)
    losses = []
    wins = []
    logits = transition_state_joint_logits.float()
    for step_index in range(int(steps)):
        target = int(target_joints[0, step_index].item())
        if target < 0:
            continue
        raw_finality = finality.get(str(step_index + 1))
        if raw_finality is None:
            continue
        opposite_code = int(
            opposite_codes[step_index]
            if step_index < len(opposite_codes)
            else opposite_codes[-1]
        )
        final_bit = int(float(raw_finality) > 0.0)
        opposite = opposite_code * 2 + final_bit
        if opposite < 0 or opposite >= int(joint_size) or opposite == target:
            continue
        target_logit = logits[0, step_index, target]
        opposite_logit = logits[0, step_index, opposite]
        losses.append(torch.relu(float(margin) + opposite_logit - target_logit))
        wins.append((target_logit.detach() > opposite_logit.detach()).float())
    if not losses:
        zero = transition_state_joint_logits.sum() * 0.0
        return zero, {
            "transition_state_joint_order_contrast": zero.detach(),
            "transition_state_joint_order_contrast_win_rate": zero.detach(),
            "transition_state_joint_order_contrast_pairs": zero.detach(),
        }
    loss = torch.stack(losses).mean()
    win_rate = torch.stack(wins).mean()
    pairs = transition_state_joint_logits.detach().new_tensor(float(len(losses)))
    return loss, {
        "transition_state_joint_order_contrast": loss.detach(),
        "transition_state_joint_order_contrast_win_rate": win_rate.detach(),
        "transition_state_joint_order_contrast_pairs": pairs,
    }


PRIMITIVE_TRANSITION_OPERATION_ORDER = (
    "add_operands",
    "multiply_sum",
    "subtract_offset",
    "hold_final",
    "filter_even",
    "double_filtered",
    "first_mapping",
    "second_mapping",
    "not_q",
    "and_with_p",
    "or_with_r",
    "filter_above_threshold",
)


def primitive_transition_operation_id_map(num_operations: int) -> dict[str, int]:
    if int(num_operations) < len(PRIMITIVE_TRANSITION_OPERATION_ORDER):
        raise ValueError(
            "primitive transition operation head is smaller than the canonical operation set"
        )
    operation_to_id = {
        operation: index
        for index, operation in enumerate(PRIMITIVE_TRANSITION_OPERATION_ORDER)
    }
    for operation in ("filter_even_base_even", "filter_even_base_odd"):
        if len(operation_to_id) >= int(num_operations):
            break
        operation_to_id[operation] = len(operation_to_id)
    return operation_to_id


def primitive_transition_operation_targets(
    row: dict[str, Any],
    *,
    num_steps: int,
    operation_to_id: dict[str, int],
    device: str,
):
    import torch

    targets = [-100] * int(num_steps)
    solver_trace = row.get("solver_trace")
    if not isinstance(solver_trace, list):
        return torch.tensor([targets], dtype=torch.long, device=device)
    transition_codes = row.get("transition_state_codes")
    if not isinstance(transition_codes, dict):
        transition_codes = {}
    for index, step in enumerate(solver_trace[: int(num_steps)]):
        if not isinstance(step, dict):
            continue
        operation = str(step.get("operation") or "")
        if not operation:
            raw_code = transition_codes.get(str(index + 1))
            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                code = -1
            if str(row.get("task_family") or "") == "mixed_list_arithmetic":
                operation = {
                    0: "filter_even",
                    1: "double_filtered",
                    2: "multiply_sum",
                    3: "subtract_offset",
                    4: "hold_final",
                }.get(code, "")
            elif str(row.get("task_family") or "") == "arithmetic_chain":
                operation = {
                    0: "add_operands",
                    2: "multiply_sum",
                    3: "subtract_offset",
                    4: "hold_final",
                }.get(code, "")
        if operation == "filter_even":
            base = row_mixed_list_base(row)
            if base is not None and int(base) % 2 == 0:
                operation = (
                    "filter_even_base_even"
                    if "filter_even_base_even" in operation_to_id
                    else operation
                )
            elif base is not None:
                operation = (
                    "filter_even_base_odd"
                    if "filter_even_base_odd" in operation_to_id
                    else operation
                )
        if operation not in operation_to_id:
            raise ValueError(f"unknown primitive transition operation: {operation}")
        targets[index] = int(operation_to_id[operation])
    if "hold_final" in operation_to_id:
        codes = row.get("transition_state_codes")
        if isinstance(codes, dict):
            hold_id = int(operation_to_id["hold_final"])
            for depth in range(1, int(num_steps) + 1):
                if targets[depth - 1] >= 0:
                    continue
                raw_code = codes.get(str(depth))
                if raw_code is not None and int(raw_code) == 4:
                    targets[depth - 1] = hold_id
    return torch.tensor([targets], dtype=torch.long, device=device)


def primitive_transition_operation_ce_loss(operation_logits, operation_targets):
    import torch
    import torch.nn.functional as F

    if operation_logits.ndim != 3:
        raise ValueError("operation_logits must have shape [batch, steps, operations]")
    targets = operation_targets.to(device=operation_logits.device, dtype=torch.long)
    if targets.shape != operation_logits.shape[:2]:
        raise ValueError("operation_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = operation_logits.sum() * 0.0
        return zero, {
            "primitive_transition_operation_ce": zero.detach(),
            "primitive_transition_operation_acc": zero.detach(),
            "primitive_transition_operation_samples": zero.detach(),
        }
    logits = operation_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = operation_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "primitive_transition_operation_ce": loss.detach(),
        "primitive_transition_operation_acc": acc.detach(),
        "primitive_transition_operation_samples": samples,
    }


def transition_source_router_targets(
    row: dict[str, Any],
    *,
    num_steps: int,
    device: str,
):
    import torch

    target = -100
    order = str(row.get("composition_order") or "")
    family = str(row.get("task_family") or row.get("category") or "")
    if order == "list_to_arithmetic" or family == "mixed_list_arithmetic":
        target = 0
    elif order == "arithmetic_to_list" or family == "mixed_arithmetic_list":
        target = 1
    elif family:
        target = 0
    targets = [int(target)] * int(num_steps)
    return torch.tensor([targets], dtype=torch.long, device=device)


def transition_phase_targets(
    row: dict[str, Any],
    *,
    num_steps: int,
    device: str,
):
    return transition_source_router_targets(
        row,
        num_steps=num_steps,
        device=device,
    )


def transition_phase_ce_loss(phase_logits, phase_targets):
    import torch
    import torch.nn.functional as F

    if phase_logits.ndim != 3:
        raise ValueError("phase_logits must have shape [batch, steps, phases]")
    targets = phase_targets.to(device=phase_logits.device, dtype=torch.long)
    if targets.shape != phase_logits.shape[:2]:
        raise ValueError("phase_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = phase_logits.sum() * 0.0
        return zero, {
            "transition_phase_ce": zero.detach(),
            "transition_phase_acc": zero.detach(),
            "transition_phase_samples": zero.detach(),
        }
    logits = phase_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = phase_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "transition_phase_ce": loss.detach(),
        "transition_phase_acc": acc.detach(),
        "transition_phase_samples": samples,
    }


def transition_source_router_ce_loss(router_logits, router_targets):
    import torch
    import torch.nn.functional as F

    if router_logits.ndim != 3:
        raise ValueError("router_logits must have shape [batch, steps, sources]")
    targets = router_targets.to(device=router_logits.device, dtype=torch.long)
    if targets.shape != router_logits.shape[:2]:
        raise ValueError("router_targets must have shape [batch, steps]")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = router_logits.sum() * 0.0
        return zero, {
            "transition_source_router_ce": zero.detach(),
            "transition_source_router_acc": zero.detach(),
            "transition_source_router_samples": zero.detach(),
        }
    logits = router_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = router_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "transition_source_router_ce": loss.detach(),
        "transition_source_router_acc": acc.detach(),
        "transition_source_router_samples": samples,
    }


def core_typed_register_operation_targets(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_steps: int,
    num_operations: int,
    device: str,
    target_mode: str = "staged",
    target_shift: int = 0,
):
    target_shift = max(0, int(target_shift))
    targets = transition_state_code_targets(
        tokenizer,
        row,
        num_depths=int(num_steps) + target_shift,
        codebook_size=int(num_operations),
        device=device,
        target_mode=target_mode,
    )
    if target_shift > 0:
        return targets[:, target_shift : target_shift + int(num_steps)]
    return targets


def core_role_value_template_targets(
    row: dict[str, Any],
    *,
    num_templates: int,
    device: str,
):
    import re
    import torch

    target = -100
    base = row_mixed_list_base(row)
    raw_length = row.get("list_length")
    raw_offset = row.get("mixed_offset")
    if raw_offset is None:
        match = re.search(r"subtract[- ](\d+)", str(row.get("prompt", "")))
        raw_offset = match.group(1) if match else None
    try:
        length = int(raw_length)
        offset = int(raw_offset)
    except (TypeError, ValueError):
        length = -1
        offset = -1
    length_slots = {5: 0, 7: 1, 9: 2, 11: 3, 13: 4}
    if base is not None and length in length_slots and 3 <= offset <= 9:
        candidate = int(length_slots[length]) * 14 + (int(base) % 2) * 7 + (
            int(offset) - 3
        )
        if 0 <= candidate < int(num_templates):
            target = candidate
    return torch.tensor([target], dtype=torch.long, device=device)


def core_role_value_template_ce_loss(template_logits, template_targets):
    import torch
    import torch.nn.functional as F

    if template_logits.ndim != 2:
        raise ValueError("template_logits must have shape [batch, templates]")
    targets = template_targets.to(device=template_logits.device, dtype=torch.long)
    if targets.shape != template_logits.shape[:1]:
        raise ValueError("template targets must match template logits batch")
    mask = targets >= 0
    if not bool(mask.any()):
        zero = template_logits.sum() * 0.0
        return zero, {
            "core_role_value_template_ce": zero.detach(),
            "core_role_value_template_acc": zero.detach(),
            "core_role_value_template_samples": zero.detach(),
        }
    logits = template_logits.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    samples = template_logits.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "core_role_value_template_ce": loss.detach(),
        "core_role_value_template_acc": acc.detach(),
        "core_role_value_template_samples": samples,
    }


def core_role_value_template_table_ce_loss(
    template_table,
    template_targets,
    role_value_targets,
    *,
    num_steps: int,
):
    import torch
    import torch.nn.functional as F

    if template_table is None:
        zero = role_value_targets.sum().float() * 0.0
        return zero, {
            "core_role_value_template_table_ce": zero.detach(),
            "core_role_value_template_table_acc": zero.detach(),
            "core_role_value_template_table_step_exact": zero.detach(),
            "core_role_value_template_table_samples": zero.detach(),
        }
    if template_table.ndim != 4:
        raise ValueError("template_table must have shape [templates, steps, roles, vocab]")
    template_targets = template_targets.to(
        device=template_table.device,
        dtype=torch.long,
    )
    role_value_targets = role_value_targets.to(
        device=template_table.device,
        dtype=torch.long,
    )
    if role_value_targets.ndim != 3:
        raise ValueError("role_value_targets must have shape [batch, steps, roles]")
    if template_targets.shape != role_value_targets.shape[:1]:
        raise ValueError("template targets batch must match role-value targets")
    steps = min(
        int(num_steps),
        int(template_table.shape[1]),
        int(role_value_targets.shape[1]),
    )
    roles = min(int(template_table.shape[2]), int(role_value_targets.shape[2]))
    if steps <= 0 or roles <= 0:
        zero = template_table.sum() * 0.0
        return zero, {
            "core_role_value_template_table_ce": zero.detach(),
            "core_role_value_template_table_acc": zero.detach(),
            "core_role_value_template_table_step_exact": zero.detach(),
            "core_role_value_template_table_samples": zero.detach(),
        }
    valid_template = (template_targets >= 0) & (
        template_targets < int(template_table.shape[0])
    )
    if not bool(valid_template.any()):
        zero = template_table.sum() * 0.0
        return zero, {
            "core_role_value_template_table_ce": zero.detach(),
            "core_role_value_template_table_acc": zero.detach(),
            "core_role_value_template_table_step_exact": zero.detach(),
            "core_role_value_template_table_samples": zero.detach(),
        }
    selected = template_table.index_select(0, template_targets[valid_template])
    selected = selected[:, :steps, :roles, :]
    targets = role_value_targets[valid_template, :steps, :roles]
    mask = targets >= 0
    if not bool(mask.any()):
        zero = selected.sum() * 0.0
        return zero, {
            "core_role_value_template_table_ce": zero.detach(),
            "core_role_value_template_table_acc": zero.detach(),
            "core_role_value_template_table_step_exact": zero.detach(),
            "core_role_value_template_table_samples": zero.detach(),
        }
    logits = selected.float()
    loss = F.cross_entropy(logits[mask], targets[mask])
    pred = logits.detach().argmax(dim=-1)
    acc = (pred[mask] == targets[mask]).float().mean()
    step_values = []
    for batch_index in range(int(targets.shape[0])):
        for step_index in range(int(targets.shape[1])):
            step_mask = mask[batch_index, step_index]
            if not bool(step_mask.any()):
                continue
            step_values.append(
                (
                    pred[batch_index, step_index, step_mask]
                    == targets[batch_index, step_index, step_mask]
                )
                .all()
                .float()
            )
    if step_values:
        step_exact = torch.stack(step_values).mean()
    else:
        step_exact = loss.detach() * 0.0
    samples = selected.detach().new_tensor(float(int(mask.sum().item())))
    return loss, {
        "core_role_value_template_table_ce": loss.detach(),
        "core_role_value_template_table_acc": acc.detach(),
        "core_role_value_template_table_step_exact": step_exact.detach(),
        "core_role_value_template_table_samples": samples,
    }


def core_typed_register_operation_ce_loss(operation_logits, operation_targets):
    loss, metrics = transition_state_code_ce_loss(
        operation_logits,
        operation_targets,
    )
    return loss, {
        key.replace("transition_state_code", "core_typed_register_operation"): value
        for key, value in metrics.items()
    }


def core_typed_register_transition_ce_loss(transition_logits, role_targets):
    loss, metrics = algorithmic_role_value_transition_ce_loss(
        transition_logits,
        role_targets,
    )
    return loss, {
        key.replace(
            "algorithmic_role_value_transition",
            "core_typed_register_transition",
        ): value
        for key, value in metrics.items()
    }


def build_random_noise_warmup_batch(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    target_vocab_size: int | None = None,
    generator: Any | None = None,
):
    import torch

    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    target_limit = int(target_vocab_size or vocab_size)
    if target_limit <= 0 or target_limit > int(vocab_size):
        raise ValueError("target_vocab_size must be in [1, vocab_size]")
    input_ids = torch.randint(
        low=0,
        high=int(vocab_size),
        size=(int(batch_size), int(seq_len)),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    target_ids = torch.randint(
        low=0,
        high=target_limit,
        size=(int(batch_size),),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    return input_ids, attention_mask, target_ids


def random_noise_warmup_loss(
    final_logits,
    depth_text_logits,
    target_ids,
    *,
    final_ce_weight: float,
    depth_ce_weight: float,
    uniform_weight: float = 0.0,
):
    import torch
    import torch.nn.functional as F

    if final_logits.ndim != 2:
        raise ValueError("final_logits must have shape [batch, vocab]")
    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if depth_text_logits.shape[2] < 1:
        raise ValueError("depth_text_logits must include at least one token position")
    targets = target_ids.to(device=final_logits.device, dtype=torch.long)
    if targets.ndim != 1 or targets.shape[0] != final_logits.shape[0]:
        raise ValueError("target_ids must have shape [batch]")

    final_ce = F.cross_entropy(final_logits.float(), targets)
    depth_logits = depth_text_logits[:, :, 0, :].float()
    repeated_targets = targets[:, None].expand(-1, depth_logits.shape[1]).reshape(-1)
    depth_ce = F.cross_entropy(
        depth_logits.reshape(-1, depth_logits.shape[-1]),
        repeated_targets,
    )
    final_uniform_ce = -final_logits.float().log_softmax(dim=-1).mean(dim=-1).mean()
    depth_uniform_ce = -depth_logits.log_softmax(dim=-1).mean(dim=-1).mean()
    uniform_loss = 0.5 * (final_uniform_ce + depth_uniform_ce)
    loss = (
        float(final_ce_weight) * final_ce
        + float(depth_ce_weight) * depth_ce
        + float(uniform_weight) * uniform_loss
    )
    with torch.no_grad():
        final_acc = (final_logits.argmax(dim=-1) == targets).float().mean()
        depth_acc = (depth_logits.argmax(dim=-1) == targets[:, None]).float().mean()
    return loss, {
        "noise_warmup_final_ce": final_ce.detach(),
        "noise_warmup_depth_ce": depth_ce.detach(),
        "noise_warmup_final_uniform_ce": final_uniform_ce.detach(),
        "noise_warmup_depth_uniform_ce": depth_uniform_ce.detach(),
        "noise_warmup_final_acc": final_acc.detach(),
        "noise_warmup_depth_acc": depth_acc.detach(),
    }


def depth_choice_margin_loss(
    depth_text_logits,
    final_text_logits,
    chosen_first_token_ids,
    rejected_first_token_ids,
    *,
    margin: float,
    all_depth_weight: float,
    final_weight: float,
):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if final_text_logits.ndim != 3:
        raise ValueError("final_text_logits must have shape [batch, tokens, vocab]")
    if depth_text_logits.shape[2] < 1 or final_text_logits.shape[1] < 1:
        raise ValueError("choice margin needs at least one answer-token logit")

    chosen = chosen_first_token_ids.to(device=depth_text_logits.device, dtype=torch.long)
    rejected = rejected_first_token_ids.to(device=depth_text_logits.device, dtype=torch.long)
    if chosen.ndim == 0:
        chosen = chosen[None]
    if rejected.ndim == 0:
        rejected = rejected[None]
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected token ids must have matching shapes")

    depth_log_probs = depth_text_logits[:, :, 0, :].float().log_softmax(dim=-1)
    chosen_depth = depth_log_probs.gather(
        dim=-1,
        index=chosen[:, None, None].expand(-1, depth_log_probs.shape[1], 1),
    ).squeeze(-1)
    rejected_depth = depth_log_probs.gather(
        dim=-1,
        index=rejected[:, None, None].expand(-1, depth_log_probs.shape[1], 1),
    ).squeeze(-1)
    all_depth_margin = torch.relu(float(margin) + rejected_depth - chosen_depth).mean()

    final_log_probs = final_text_logits[:, 0, :].float().log_softmax(dim=-1)
    chosen_final = final_log_probs.gather(dim=-1, index=chosen[:, None]).squeeze(-1)
    rejected_final = final_log_probs.gather(dim=-1, index=rejected[:, None]).squeeze(-1)
    final_margin = torch.relu(float(margin) + rejected_final - chosen_final).mean()

    loss = float(all_depth_weight) * all_depth_margin + float(final_weight) * final_margin
    return loss, {
        "choice_margin_all_depth": all_depth_margin.detach(),
        "choice_margin_final_path": final_margin.detach(),
    }


def _masked_mean_log_probs(logits, target_ids):
    import torch

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits token dimension must match target ids")
    mask = target_ids >= 0
    if not bool(mask.any()):
        raise ValueError("at least one target token is required")
    safe_targets = target_ids.clamp_min(0)
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=safe_targets[:, :, None],
    ).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    lengths = mask.sum(dim=-1).clamp_min(1)
    return token_log_probs.sum(dim=-1) / lengths


def depth_choice_sequence_margin_loss(
    depth_text_logits,
    final_text_logits,
    chosen_target_ids,
    rejected_target_ids,
    *,
    margin: float,
    all_depth_weight: float,
    final_weight: float,
):
    import torch

    if depth_text_logits.ndim != 4:
        raise ValueError("depth_text_logits must have shape [batch, steps, tokens, vocab]")
    if final_text_logits.ndim != 3:
        raise ValueError("final_text_logits must have shape [batch, tokens, vocab]")
    chosen = chosen_target_ids.to(device=depth_text_logits.device, dtype=torch.long)
    rejected = rejected_target_ids.to(device=depth_text_logits.device, dtype=torch.long)
    if chosen.ndim == 1:
        chosen = chosen[None, :]
    if rejected.ndim == 1:
        rejected = rejected[None, :]
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected target ids must have matching shapes")
    if depth_text_logits.shape[2] != chosen.shape[1]:
        raise ValueError("depth text logits token dimension must match target ids")
    if final_text_logits.shape[1] != chosen.shape[1]:
        raise ValueError("final text logits token dimension must match target ids")

    depth_chosen = []
    depth_rejected = []
    for step_index in range(depth_text_logits.shape[1]):
        depth_chosen.append(
            _masked_mean_log_probs(depth_text_logits[:, step_index, :, :], chosen)
        )
        depth_rejected.append(
            _masked_mean_log_probs(depth_text_logits[:, step_index, :, :], rejected)
        )
    chosen_depth = torch.stack(depth_chosen, dim=1)
    rejected_depth = torch.stack(depth_rejected, dim=1)
    all_depth_margin = torch.relu(float(margin) + rejected_depth - chosen_depth).mean()

    chosen_final = _masked_mean_log_probs(final_text_logits, chosen)
    rejected_final = _masked_mean_log_probs(final_text_logits, rejected)
    final_margin = torch.relu(float(margin) + rejected_final - chosen_final).mean()

    loss = float(all_depth_weight) * all_depth_margin + float(final_weight) * final_margin
    return loss, {
        "choice_sequence_margin_all_depth": all_depth_margin.detach(),
        "choice_sequence_margin_final_path": final_margin.detach(),
    }


def final_choice_sequence_margin_loss(
    final_text_logits,
    chosen_target_ids,
    rejected_target_ids,
    *,
    margin: float,
):
    import torch

    if final_text_logits.ndim != 3:
        raise ValueError("final_text_logits must have shape [batch, tokens, vocab]")
    chosen = chosen_target_ids.to(device=final_text_logits.device, dtype=torch.long)
    rejected = rejected_target_ids.to(device=final_text_logits.device, dtype=torch.long)
    if chosen.ndim == 1:
        chosen = chosen[None, :]
    if rejected.ndim == 1:
        rejected = rejected[None, :]
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected target ids must have matching shapes")
    if final_text_logits.shape[1] != chosen.shape[1]:
        raise ValueError("final text logits token dimension must match target ids")

    chosen_final = _masked_mean_log_probs(final_text_logits, chosen)
    rejected_final = _masked_mean_log_probs(final_text_logits, rejected)
    final_margin = torch.relu(float(margin) + rejected_final - chosen_final).mean()
    return final_margin, {
        "final_choice_sequence_margin_final_path": final_margin.detach(),
    }


def tail_negative_sequence_margin_loss(
    depth_text_logits,
    final_text_logits,
    chosen_target_ids,
    rejected_target_ids,
    *,
    margin: float,
):
    loss, metrics = depth_choice_sequence_margin_loss(
        depth_text_logits,
        final_text_logits,
        chosen_target_ids,
        rejected_target_ids,
        margin=margin,
        all_depth_weight=1.0,
        final_weight=1.0,
    )
    return loss, {
        key.replace("choice_sequence", "tail_negative"): value
        for key, value in metrics.items()
    }


def subtract_tail_counterfactual_sequence_margin_loss(
    depth_text_logits,
    final_text_logits,
    chosen_target_ids,
    rejected_target_ids,
    *,
    margin: float,
):
    loss, metrics = depth_choice_sequence_margin_loss(
        depth_text_logits,
        final_text_logits,
        chosen_target_ids,
        rejected_target_ids,
        margin=margin,
        all_depth_weight=1.0,
        final_weight=1.0,
    )
    return loss, {
        key.replace("choice_sequence", "subtract_tail_counterfactual"): value
        for key, value in metrics.items()
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    if (
        float(args.tail_negative_margin_weight) != 0.0
        and not bool(args.causal_prefix_supervision)
    ):
        raise ValueError(
            "--tail-negative-margin-weight requires --causal-prefix-supervision"
        )
    if (
        float(args.subtract_tail_counterfactual_margin_weight) != 0.0
        and not bool(args.causal_prefix_supervision)
    ):
        raise ValueError(
            "--subtract-tail-counterfactual-margin-weight requires "
            "--causal-prefix-supervision"
        )
    if (
        float(args.causal_prefix_self_rollout_weight) != 0.0
        and not bool(args.causal_prefix_supervision)
    ):
        raise ValueError(
            "--causal-prefix-self-rollout-weight requires --causal-prefix-supervision"
        )
    if bool(args.final_path_only_supervision):
        depth_only_weights = {
            "depth_final_ce_weight": args.depth_final_ce_weight,
            "all_depth_ce_weight": args.all_depth_ce_weight,
            "progress_margin_weight": args.progress_margin_weight,
            "depth_trajectory_monotonic_weight": args.depth_trajectory_monotonic_weight,
            "depth_greedy_token_margin_weight": args.depth_greedy_token_margin_weight,
            "terminal_depth_ce_weight": args.terminal_depth_ce_weight,
            "choice_margin_weight": args.choice_margin_weight,
            "tail_negative_margin_weight": args.tail_negative_margin_weight,
            "subtract_tail_counterfactual_margin_weight": (
                args.subtract_tail_counterfactual_margin_weight
            ),
            "temporal_spatial_context_contrast_weight": (
                args.temporal_spatial_context_contrast_weight
            ),
            "transition_state_contrast_weight": args.transition_state_contrast_weight,
            "transition_state_depth_contrast_weight": (
                args.transition_state_depth_contrast_weight
            ),
            "transition_joint_answer_bridge_contrast_weight": (
                args.transition_joint_answer_bridge_contrast_weight
            ),
            "core_role_value_answer_bridge_contrast_weight": (
                args.core_role_value_answer_bridge_contrast_weight
            ),
            "teacher_first_token_depth_kl_weight": (
                args.teacher_first_token_depth_kl_weight
            ),
        }
        active_depth_weights = [
            name for name, value in depth_only_weights.items() if float(value) != 0.0
        ]
        if active_depth_weights:
            raise ValueError(
                "--final-path-only-supervision is incompatible with depth/contrast "
                f"weights: {', '.join(active_depth_weights)}"
            )
    if bool(args.target_logit_positions_only) and not bool(
        args.final_path_only_supervision
    ):
        raise ValueError(
            "--target-logit-positions-only requires --final-path-only-supervision"
        )

    import random
    import sys

    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.losses import jepa_world_model_loss
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter
    from qtrm_mm.training.train import (
        build_core_world_model_actions,
        configure_trainable_parameters,
        load_initial_checkpoint,
    )

    seed = int(args.seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] {seed}")

    cfg = load_config(args.config)
    if str(args.trainable_param_policy).strip():
        cfg.train.trainable_param_policy = str(args.trainable_param_policy).strip()
    if bool(args.token_numeric_value_features):
        cfg.model.token_numeric_value_embedding_enabled = True
        cfg.model.token_numeric_value_vocab_size = int(args.token_numeric_value_vocab_size)
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
        cfg.model.core_source_position_binder_query_state_enabled = bool(
            args.core_source_position_binder_query_state
        )
        cfg.model.core_source_position_binder_query_state_gate_min = float(
            args.core_source_position_binder_query_state_gate_min
        )
        cfg.model.core_source_value_binder_enabled = bool(
            args.core_source_value_binder
        )
        cfg.model.core_source_value_binder_state_gate_min = float(
            args.core_source_value_binder_state_gate_min
        )
        cfg.model.core_source_value_binder_state_straight_through = bool(
            args.core_source_value_binder_state_st
        )
        cfg.model.core_primitive_role_value_source_value_conditioning_enabled = bool(
            args.core_primitive_role_value_source_value_conditioning
        )
        cfg.model.core_primitive_role_value_source_value_gate_min = float(
            args.core_primitive_role_value_source_value_gate_min
        )
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_rows(args.data_jsonl)
    role_value_list_class_mode = apply_role_value_list_class_mode(
        rows,
        args.role_value_list_class_mode,
    )
    print(f"[data] role_value_list_class_mode={role_value_list_class_mode}")
    if bool(args.shuffle_rows):
        shuffle_rng = random.Random(seed)
        shuffle_rng.shuffle(rows)
        print(f"[data] shuffle_rows=true rows={len(rows)} seed={seed}")
    paired_hard_negative_lookup = (
        build_paired_hard_negative_lookup(rows)
        if float(args.core_primitive_role_value_pair_trace_contrast_weight) != 0.0
        else {}
    )
    if paired_hard_negative_lookup:
        print(f"[data] paired_hard_negative_lookup={len(paired_hard_negative_lookup)}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model).to(device)
    init_missing_keys: list[str] = []
    init_unexpected_keys: list[str] = []
    init_mode = validate_init_checkpoint_args(
        args.init_checkpoint,
        allow_random_init=bool(args.allow_random_init),
    )
    if init_mode == "checkpoint":
        missing, unexpected = load_initial_checkpoint(
            model,
            args.init_checkpoint,
            map_location=device,
        )
        init_missing_keys = list(missing)
        init_unexpected_keys = list(unexpected)
        if missing:
            print(f"[init] missing keys: {len(missing)}")
        if unexpected:
            print(f"[init] unexpected keys: {len(unexpected)}")
    else:
        print("[init] random QTRM initialization (--allow-random-init)")
    teacher_model = None
    if (
        float(args.teacher_first_token_depth_kl_weight) != 0.0
        or float(args.teacher_final_logit_kl_weight) != 0.0
    ):
        if not args.teacher_checkpoint:
            raise ValueError(
                "--teacher-checkpoint is required when teacher KL weight is non-zero"
            )
        teacher_model = QTRMMultimodalModel(cfg.model).to(device)
        teacher_missing, teacher_unexpected = load_initial_checkpoint(
            teacher_model,
            args.teacher_checkpoint,
            map_location=device,
        )
        if teacher_missing:
            print(f"[teacher] missing keys: {len(teacher_missing)}")
        if teacher_unexpected:
            print(f"[teacher] unexpected keys: {len(teacher_unexpected)}")
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
    donor = QwenDonorAdapter(cfg.donor)

    def needs_donor_logits_for(*model_candidates) -> bool:
        return any(
            candidate is not None
            and float(getattr(candidate.cfg, "donor_logits_scale", 0.0)) != 0.0
            for candidate in model_candidates
        )

    def donor_forward_kwargs(donor_out: dict, *, dtype=None) -> dict:
        text_states = donor_out["text_states"].detach()
        text_states = (
            text_states.to(device=device, dtype=dtype)
            if dtype is not None
            else text_states.to(device=device)
        )
        kwargs = {
            "text_states": text_states,
            "disable_donor_context": bool(args.disable_donor_context),
        }
        if donor_out.get("logits") is not None:
            kwargs["donor_logits"] = donor_out["logits"].detach().to(device=device)
        return kwargs

    trainable_names = configure_trainable_parameters(model, cfg.train.trainable_param_policy)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected")
    print(
        f"[trainable] policy={cfg.train.trainable_param_policy} "
        f"params={sum(p.numel() for p in params):,} tensors={len(trainable_names)}"
    )
    lr = float(args.lr if args.lr is not None else cfg.train.lr)
    if str(args.optimizer) == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    def save_training_checkpoint(path):
        metadata = {
            "format": "qtrm_depth_supervised_checkpoint_v2",
            "command": list(sys.argv),
            "config": str(args.config),
            "init_checkpoint": str(args.init_checkpoint or ""),
            "seed": seed,
            "train_data": str(args.data_jsonl),
            "shuffle_rows": bool(args.shuffle_rows),
            "init_missing_keys": list(init_missing_keys),
            "init_unexpected_keys": list(init_unexpected_keys),
            "trainable_param_policy": str(cfg.train.trainable_param_policy),
        }
        if bool(args.save_trainable_only):
            state = model.state_dict()
            payload = {
                "model": {
                    name: state[name].detach().cpu()
                    for name in trainable_names
                    if name in state
                },
                "base_checkpoint": str(args.init_checkpoint or ""),
                "trainable_param_policy": str(cfg.train.trainable_param_policy),
                "format": "qtrm_trainable_delta_v1",
                "training_metadata": metadata,
            }
        else:
            payload = {"model": model.state_dict(), "training_metadata": metadata}
        torch.save(payload, path)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.use_amp and device == "cuda"))
    steps = int(args.steps if args.steps is not None else cfg.train.steps)
    depth_steps = parse_depth_steps(args.depth_steps)
    family_repeats = parse_family_repeat_spec(args.family_repeat)
    curriculum_indices = build_curriculum_indices(rows, family_repeats)
    if family_repeats:
        print(f"[curriculum] family_repeat={family_repeats} effective_rows={len(curriculum_indices)}")
    max_length = int(args.max_length or cfg.train.seq_len)
    core_world_model_weight = (
        float(args.core_world_model_weight)
        if args.core_world_model_weight is not None
        else float(getattr(cfg.train, "loss_core_world_model_weight", 0.0))
    )
    if core_world_model_weight != 0.0 and not bool(cfg.model.core_world_model_enabled):
        raise ValueError(
            "core world-model loss requires model.core_world_model_enabled=true"
        )
    if (
        float(args.transition_state_contrast_weight) != 0.0
        and not (
            bool(cfg.model.transition_state_enabled)
            or bool(cfg.model.transition_state_code_enabled)
        )
    ):
        raise ValueError(
            "transition-state contrast requires model.transition_state_enabled=true "
            "or model.transition_state_code_enabled=true"
        )
    if (
        float(args.answer_selective_context_alignment_weight) != 0.0
        and not bool(cfg.model.answer_state_loop_selective_context_enabled)
    ):
        raise ValueError(
            "answer selective-context alignment requires "
            "model.answer_state_loop_selective_context_enabled=true"
        )
    if (
        float(args.answer_state_loop_halt_ce_weight) != 0.0
        and not bool(cfg.model.answer_state_loop_halt_enabled)
    ):
        raise ValueError(
            "answer-state-loop halt CE requires "
            "model.answer_state_loop_halt_enabled=true"
        )
    if float(args.answer_state_loop_logit_ce_weight) != 0.0:
        if not bool(args.causal_prefix_supervision):
            raise ValueError(
                "answer-state-loop logit CE requires "
                "--causal-prefix-supervision to avoid answer leakage"
            )
        if not bool(cfg.model.answer_state_loop_enabled):
            raise ValueError(
                "answer-state-loop logit CE requires "
                "model.answer_state_loop_enabled=true"
            )
    if float(args.answer_state_loop_future_token_ce_weight) != 0.0:
        if not bool(args.causal_prefix_supervision):
            raise ValueError(
                "answer-state-loop future-token CE requires "
                "--causal-prefix-supervision to avoid answer leakage"
            )
        if not bool(cfg.model.answer_state_loop_future_token_decoder_enabled):
            raise ValueError(
                "answer-state-loop future-token CE requires "
                "model.answer_state_loop_future_token_decoder_enabled=true"
            )
    if float(args.answer_selective_context_alignment_temperature) <= 0.0:
        raise ValueError("--answer-selective-context-alignment-temperature must be positive")
    if (
        float(args.transition_joint_answer_bridge_contrast_weight) != 0.0
        and not bool(cfg.model.transition_state_joint_answer_bridge_enabled)
    ):
        raise ValueError(
            "transition joint answer bridge contrast requires "
            "model.transition_state_joint_answer_bridge_enabled=true"
        )
    if (
        float(args.core_role_value_answer_bridge_contrast_weight) != 0.0
        and not bool(cfg.model.core_role_value_state_answer_bridge_enabled)
    ):
        raise ValueError(
            "core role-value answer bridge contrast requires "
            "model.core_role_value_state_answer_bridge_enabled=true"
        )
    if (
        float(args.core_role_value_answer_bridge_final_contrast_weight) != 0.0
        and not bool(cfg.model.core_role_value_state_answer_bridge_enabled)
    ):
        raise ValueError(
            "core role-value answer bridge final contrast requires "
            "model.core_role_value_state_answer_bridge_enabled=true"
        )
    if (
        float(args.core_primitive_role_value_answer_final_contrast_weight) != 0.0
        and not bool(cfg.model.core_primitive_role_value_executor_enabled)
    ):
        raise ValueError(
            "core primitive role-value answer final contrast requires "
            "model.core_primitive_role_value_executor_enabled=true"
        )
    if (
        float(args.transition_state_ce_weight) != 0.0
        or float(args.transition_state_depth_contrast_weight) != 0.0
    ) and not bool(cfg.model.transition_state_enabled):
        raise ValueError(
            "transition-state CE/contrast requires model.transition_state_enabled=true"
        )
    if (
        float(args.transition_state_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_enabled)
    ):
        raise ValueError(
            "transition-state CE requires model.transition_state_enabled=true"
        )
    if (
        float(args.transition_state_code_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_code_enabled)
    ):
        raise ValueError(
            "transition-state code CE requires model.transition_state_code_enabled=true"
        )
    if (
        float(args.transition_state_finality_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_finality_enabled)
    ):
        raise ValueError(
            "transition-state finality CE requires model.transition_state_finality_enabled=true"
        )
    if (
        float(args.transition_state_joint_ce_weight) != 0.0
        and not bool(cfg.model.transition_state_joint_enabled)
    ):
        raise ValueError(
            "transition-state joint CE requires model.transition_state_joint_enabled=true"
        )
    if (
        float(args.primitive_transition_operation_ce_weight) != 0.0
        and not bool(cfg.model.primitive_transition_enabled)
    ):
        raise ValueError(
            "primitive transition operation CE requires model.primitive_transition_enabled=true"
        )
    if (
        float(args.core_transition_feedback_operation_ce_weight) != 0.0
        or float(args.core_transition_feedback_finality_ce_weight) != 0.0
        or bool(args.core_transition_feedback_teacher_forcing)
    ) and not bool(cfg.model.core_transition_feedback_enabled):
        raise ValueError(
            "core transition feedback CE requires "
            "model.core_transition_feedback_enabled=true"
        )
    if (
        float(args.core_transition_order_bottleneck_ce_weight) != 0.0
        and not bool(cfg.model.core_transition_order_bottleneck_enabled)
    ):
        raise ValueError(
            "core transition order bottleneck CE requires "
            "model.core_transition_order_bottleneck_enabled=true"
        )
    if (
        float(args.transition_phase_ce_weight) != 0.0
        and not bool(cfg.model.transition_phase_enabled)
    ):
        raise ValueError("transition phase CE requires model.transition_phase_enabled=true")
    if (
        float(args.transition_source_router_ce_weight) != 0.0
        and not bool(cfg.model.transition_source_router_enabled)
    ):
        raise ValueError(
            "transition source router CE requires model.transition_source_router_enabled=true"
        )
    if (
        float(args.algorithmic_role_value_transition_ce_weight) != 0.0
        and not bool(cfg.model.core_role_value_transition_enabled)
    ):
        raise ValueError(
            "algorithmic role-value transition CE requires "
            "model.core_role_value_transition_enabled=true"
        )
    if (
        float(args.core_value_delta_code_ce_weight) != 0.0
        and not bool(cfg.model.core_value_delta_code_enabled)
    ):
        raise ValueError(
            "core value-delta code CE requires "
            "model.core_value_delta_code_enabled=true"
        )
    if (
        float(args.typed_algorithmic_value_state_ce_weight) != 0.0
        and not bool(cfg.model.typed_algorithmic_value_state_enabled)
    ):
        raise ValueError(
            "typed algorithmic value-state CE requires "
            "model.typed_algorithmic_value_state_enabled=true"
        )
    if (
        float(args.typed_algorithmic_scalar_regression_weight) != 0.0
        and not bool(cfg.model.typed_algorithmic_value_state_scalar_regression_enabled)
    ):
        raise ValueError(
            "typed algorithmic scalar regression requires "
            "model.typed_algorithmic_value_state_scalar_regression_enabled=true"
        )
    if (
        float(args.core_typed_register_ce_weight) != 0.0
        and not bool(cfg.model.core_typed_register_executor_enabled)
    ):
        raise ValueError(
            "core typed-register CE requires "
            "model.core_typed_register_executor_enabled=true"
        )
    if (
        float(args.core_typed_register_operation_ce_weight) != 0.0
        and not bool(cfg.model.core_typed_register_executor_enabled)
    ):
        raise ValueError(
            "core typed-register operation CE requires "
            "model.core_typed_register_executor_enabled=true"
        )
    if (
        float(args.core_typed_register_transition_ce_weight) != 0.0
        and not bool(cfg.model.core_typed_register_executor_enabled)
    ):
        raise ValueError(
            "core typed-register transition CE requires "
            "model.core_typed_register_executor_enabled=true"
        )
    primitive_operation_to_id = primitive_transition_operation_id_map(
        int(cfg.model.primitive_transition_num_operations)
    ) if (
        float(args.primitive_transition_operation_ce_weight) != 0.0
        or float(args.core_transition_feedback_operation_ce_weight) != 0.0
        or bool(args.core_transition_feedback_teacher_forcing)
    ) else {}
    out_dir = Path(args.out_dir or cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(args.save_every)
    if save_every < 0:
        raise ValueError("--save-every must be non-negative")
    model.train()

    noise_warmup_steps = int(args.noise_warmup_steps)
    if noise_warmup_steps < 0:
        raise ValueError("--noise-warmup-steps must be non-negative")
    if noise_warmup_steps > 0:
        vocab_size = int(getattr(cfg.model, "vocab_size", 0) or len(tokenizer))
        target_vocab_size = int(args.noise_warmup_target_vocab_size or vocab_size)
        print(
            "[noise_warmup] "
            f"steps={noise_warmup_steps} seq_len={int(args.noise_warmup_seq_len)} "
            f"batch={int(args.noise_warmup_batch_size)} core_steps={int(args.noise_warmup_core_steps)} "
            f"target_vocab_size={target_vocab_size}"
        )
        warmup_pbar = tqdm(range(noise_warmup_steps), desc="noise_warmup")
        for warmup_step in warmup_pbar:
            input_ids, attention_mask, target_ids = build_random_noise_warmup_batch(
                vocab_size=vocab_size,
                seq_len=int(args.noise_warmup_seq_len),
                batch_size=int(args.noise_warmup_batch_size),
                device=device,
                target_vocab_size=target_vocab_size,
            )
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=(cfg.train.use_amp and device == "cuda"),
                dtype=torch.bfloat16,
            ):
                with torch.no_grad():
                    donor_out = donor.encode_inputs(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_logits=needs_donor_logits_for(model),
                    )
                old_outer_steps = int(model.cfg.outer_steps)
                model.cfg.outer_steps = int(args.noise_warmup_core_steps)
                core_world_model_actions = None
                if bool(model.cfg.core_world_model_enabled):
                    core_world_model_actions = build_core_world_model_actions(
                        {"input_ids": input_ids},
                        num_steps=int(args.noise_warmup_core_steps),
                        num_actions=int(model.cfg.num_actions),
                        device=device,
                    )
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **donor_forward_kwargs(donor_out),
                        core_world_model_actions=core_world_model_actions,
                        return_core_depth_logits=True,
                        return_core_depth_text_logits=True,
                    )
                finally:
                    model.cfg.outer_steps = old_outer_steps
                offset = outputs["logits"].shape[1] - input_ids.shape[1]
                final_logits = outputs["logits"][
                    :,
                    offset + input_ids.shape[1] - 1,
                    :,
                ]
                depth_text_logits = outputs["core_depth_text_logits"][
                    :,
                    :,
                    input_ids.shape[1] - 1 : input_ids.shape[1],
                    :,
                ]
                loss, metrics = random_noise_warmup_loss(
                    final_logits,
                    depth_text_logits,
                    target_ids,
                    final_ce_weight=args.noise_warmup_final_ce_weight,
                    depth_ce_weight=args.noise_warmup_depth_ce_weight,
                    uniform_weight=args.noise_warmup_uniform_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            if warmup_step % max(1, int(args.log_every)) == 0:
                warmup_pbar.set_description(
                    " ".join(
                        f"{key}={float(value):.4f}"
                        for key, value in {"noise_loss": loss.detach(), **metrics}.items()
                    )
                )

    def greedy_self_rollout_prefix_ids(
        input_ids,
        attention_mask,
        *,
        max_new_tokens: int,
        core_steps: int,
        temporal_spatial_context=None,
    ) -> list[int]:
        if int(max_new_tokens) <= 0:
            return []
        generated: list[int] = []
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        was_training = model.training
        old_outer_steps = int(model.cfg.outer_steps)
        model_dtype = next(model.parameters()).dtype
        model.eval()
        model.cfg.outer_steps = int(core_steps)
        try:
            with torch.no_grad():
                for _ in range(int(max_new_tokens)):
                    if int(current_input_ids.shape[1]) >= int(max_length):
                        break
                    donor_out = donor.encode_inputs(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        return_logits=needs_donor_logits_for(model),
                    )
                    core_world_model_actions = None
                    if bool(model.cfg.core_world_model_enabled):
                        core_world_model_actions = build_core_world_model_actions(
                            {"input_ids": current_input_ids},
                            num_steps=int(core_steps),
                            num_actions=int(model.cfg.num_actions),
                            device=device,
                        )
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        **donor_forward_kwargs(donor_out, dtype=model_dtype),
                        core_world_model_actions=core_world_model_actions,
                        temporal_spatial_context=temporal_spatial_context,
                    )
                    offset = outputs["logits"].shape[1] - current_input_ids.shape[1]
                    next_logits = outputs["logits"][
                        :,
                        offset + current_input_ids.shape[1] - 1,
                        :,
                    ]
                    next_token = int(next_logits.float().argmax(dim=-1)[0].item())
                    generated.append(next_token)
                    next_tensor = current_input_ids.new_tensor([[next_token]])
                    next_mask = current_attention_mask.new_ones((current_attention_mask.shape[0], 1))
                    current_input_ids = torch.cat([current_input_ids, next_tensor], dim=1)
                    current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
        finally:
            model.cfg.outer_steps = old_outer_steps
            if was_training:
                model.train()
        return generated

    pbar = tqdm(range(steps))
    for step in pbar:
        row_index, core_steps = scheduled_row_and_core_steps(
            step,
            row_count=len(rows),
            depth_steps=depth_steps,
            row_indices=curriculum_indices,
        )
        row = rows[row_index]
        prompt = str(row["prompt"])
        answer = target_for_core_steps(row, core_steps, target_mode=args.target_mode)
        temporal_spatial_context = _row_temporal_spatial_context(row, device=device)
        numeric_source_visual_features = None
        if bool(args.numeric_source_features):
            numeric_source_visual_features, _numeric_source_visual_mask = (
                row_numeric_source_visual_tensors(
                    row,
                    visual_dim=int(model.cfg.visual_dim),
                    max_list_len=int(args.numeric_source_max_list_len),
                    value_vocab_size=int(args.numeric_source_value_vocab_size),
                    device=device,
                )
            )
        if bool(args.causal_prefix_supervision):
            train_examples = _prepare_causal_prefix_answer_examples(
                tokenizer,
                prompt,
                answer,
                max_length=max_length,
                device=device,
                max_target_tokens=args.causal_prefix_max_target_tokens,
                skip_leading_whitespace_targets=bool(
                    args.causal_prefix_skip_leading_whitespace_targets
                ),
            )
            train_example_weights = [
                _causal_prefix_example_loss_weight(
                    example_index,
                    args.causal_prefix_later_token_weight,
                )
                for example_index in range(len(train_examples))
            ]
        else:
            train_examples = [
                _prepare_prompt_answer(
                    tokenizer,
                    prompt,
                    answer,
                    max_length=max_length,
                    device=device,
                )
            ]
            train_example_weights = [1.0]
        train_token_numeric_value_ids = []
        if bool(args.token_numeric_value_features):
            for input_ids, _attention_mask, _target_ids, _target_start, _target_end in (
                train_examples
            ):
                train_token_numeric_value_ids.append(
                    _token_numeric_value_ids_for_prompt_prefix(
                        tokenizer,
                        row,
                        prompt,
                        input_ids=input_ids,
                        max_length=max_length,
                        value_vocab_size=int(args.token_numeric_value_vocab_size),
                        device=device,
                    )
                )
        else:
            train_token_numeric_value_ids = [None for _ in train_examples]
        token_numeric_source_slot_ids_tensor = None
        token_numeric_source_slot_token_ids_tensor = None
        token_numeric_source_slot_mask_tensor = None
        if bool(args.token_numeric_source_slots):
            (
                token_numeric_source_slot_ids_tensor,
                token_numeric_source_slot_token_ids_tensor,
                token_numeric_source_slot_mask_tensor,
            ) = _token_numeric_source_slots_for_prompt(
                tokenizer,
                row,
                prompt,
                max_length=max_length,
                max_slots=int(args.token_numeric_source_slot_max_slots),
                value_vocab_size=int(args.token_numeric_source_slot_vocab_size),
                device=device,
            )
        self_rollout_examples_count = 0
        self_rollout_prefix_tokens = 0
        self_rollout_prefix_mismatch_rate = 0.0
        if float(args.causal_prefix_self_rollout_weight) != 0.0:
            rollout_max_targets = int(
                args.causal_prefix_self_rollout_max_target_tokens
                or args.causal_prefix_max_target_tokens
            )
            if rollout_max_targets <= 0:
                raise ValueError("--causal-prefix-self-rollout-max-target-tokens must be positive")
            prompt_input_ids = train_examples[0][0]
            prompt_attention_mask = train_examples[0][1]
            rollout_prefix_ids = greedy_self_rollout_prefix_ids(
                prompt_input_ids,
                prompt_attention_mask,
                max_new_tokens=max(0, rollout_max_targets - 1),
                core_steps=core_steps,
                temporal_spatial_context=temporal_spatial_context,
            )
            rollout_examples = _prepare_causal_prefix_rollout_answer_examples(
                tokenizer,
                prompt,
                answer,
                rollout_prefix_ids=rollout_prefix_ids,
                max_length=max_length,
                device=device,
                max_target_tokens=rollout_max_targets,
                skip_leading_whitespace_targets=bool(
                    args.causal_prefix_skip_leading_whitespace_targets
                ),
            )
            train_examples.extend(rollout_examples)
            train_example_weights.extend(
                [
                    float(args.causal_prefix_self_rollout_weight)
                    * _causal_prefix_example_loss_weight(
                        example_index,
                        args.causal_prefix_later_token_weight,
                    )
                    for example_index in range(len(rollout_examples))
                ]
            )
            if bool(args.token_numeric_value_features):
                for input_ids, _attention_mask, _target_ids, _target_start, _target_end in (
                    rollout_examples
                ):
                    train_token_numeric_value_ids.append(
                        _token_numeric_value_ids_for_prompt_prefix(
                            tokenizer,
                            row,
                            prompt,
                            input_ids=input_ids,
                            max_length=max_length,
                            value_vocab_size=int(args.token_numeric_value_vocab_size),
                            device=device,
                        )
                    )
            else:
                train_token_numeric_value_ids.extend([None for _ in rollout_examples])
            self_rollout_examples_count = len(rollout_examples)
            self_rollout_prefix_tokens = len(rollout_prefix_ids)
            gold_prefix_ids = causal_prefix_answer_token_ids(
                tokenizer,
                answer,
                skip_leading_whitespace_targets=bool(
                    args.causal_prefix_skip_leading_whitespace_targets
                ),
            )
            compare_count = min(len(rollout_prefix_ids), len(gold_prefix_ids))
            if compare_count > 0:
                mismatches = sum(
                    int(rollout_prefix_ids[i] != gold_prefix_ids[i])
                    for i in range(compare_count)
                )
                self_rollout_prefix_mismatch_rate = float(mismatches) / float(compare_count)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(cfg.train.use_amp and device == "cuda"), dtype=torch.bfloat16):
            losses = []
            loss_weights = []
            metric_sums = {}
            metric_counts = {}

            def add_metrics(new_metrics):
                for key, value in new_metrics.items():
                    detached = value.detach()
                    if key not in metric_sums:
                        metric_sums[key] = detached.new_zeros(())
                        metric_counts[key] = 0
                    metric_sums[key] = metric_sums[key] + detached
                    metric_counts[key] += 1

            for example_index, (
                input_ids,
                attention_mask,
                target_ids,
                target_start,
                target_end,
            ) in enumerate(train_examples):
                token_numeric_value_ids_tensor = train_token_numeric_value_ids[
                    example_index
                ]
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=needs_donor_logits_for(model, teacher_model),
                )
                old_outer_steps = int(model.cfg.outer_steps)
                model.cfg.outer_steps = int(core_steps)
                core_world_model_actions = None
                if bool(model.cfg.core_world_model_enabled):
                    core_world_model_actions = build_core_world_model_actions(
                        {"input_ids": input_ids},
                        num_steps=int(core_steps),
                        num_actions=int(model.cfg.num_actions),
                        device=device,
                    )
                feedback_operation_targets = None
                feedback_finality_targets = None
                if bool(args.core_transition_feedback_teacher_forcing):
                    feedback_operation_targets = primitive_transition_operation_targets(
                        row,
                        num_steps=int(core_steps),
                        operation_to_id=primitive_operation_to_id,
                        device=device,
                    )
                    feedback_finality_targets = transition_state_finality_targets(
                        row,
                        num_depths=int(core_steps),
                        device=device,
                    )
                context_off_depth_text_logits = None
                context_off_outputs = None
                transition_state_off_depth_text_logits = None
                transition_state_off_outputs = None
                bridge_off_depth_text_logits = None
                bridge_off_outputs = None
                role_bridge_off_depth_text_logits = None
                role_bridge_off_final_text_logits = None
                role_bridge_off_outputs = None
                typed_value_bridge_off_final_text_logits = None
                typed_value_bridge_off_outputs = None
                primitive_role_value_off_final_text_logits = None
                primitive_role_value_off_renderer_text_logits = None
                primitive_role_value_off_outputs = None
                source_binder_off_renderer_text_logits = None
                source_binder_off_outputs = None
                dense_context_final_text_logits = None
                logit_token_indices = None
                if bool(args.target_logit_positions_only):
                    logit_token_indices = torch.arange(
                        int(target_start) - 1,
                        int(target_end) - 1,
                        device=device,
                        dtype=torch.long,
                    )
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_numeric_value_ids=token_numeric_value_ids_tensor,
                        token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                        token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                        token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                        visual_features=numeric_source_visual_features,
                        **donor_forward_kwargs(donor_out),
                        core_world_model_actions=core_world_model_actions,
                        temporal_spatial_context=temporal_spatial_context,
                        core_transition_feedback_operation_targets=feedback_operation_targets,
                        core_transition_feedback_finality_targets=feedback_finality_targets,
                        core_transition_feedback_teacher_forcing=bool(
                            args.core_transition_feedback_teacher_forcing
                        ),
                        return_core_depth_logits=not bool(
                            args.final_path_only_supervision
                        ),
                        return_core_depth_text_logits=not bool(
                            args.final_path_only_supervision
                        ),
                        logit_token_indices=logit_token_indices,
                    )
                    if (
                        float(args.token_numeric_source_slot_parity_ce_weight) != 0.0
                        and token_numeric_source_slot_ids_tensor is not None
                    ):
                        parity_logits = outputs.get(
                            "token_numeric_source_slot_parity_logits"
                        )
                        if parity_logits is None:
                            raise ValueError(
                                "model did not return "
                                "token_numeric_source_slot_parity_logits"
                            )
                        parity_loss, parity_metrics = (
                            token_numeric_source_slot_parity_ce_loss(
                                parity_logits,
                                token_numeric_source_slot_ids_tensor,
                            )
                        )
                        losses.append(parity_loss)
                        loss_weights.append(
                            float(args.token_numeric_source_slot_parity_ce_weight)
                            * _causal_prefix_example_loss_weight(
                                example_index,
                                args.causal_prefix_later_token_weight,
                            )
                        )
                        add_metrics(parity_metrics)
                    if (
                        float(args.token_numeric_source_slot_predicate_ce_weight) != 0.0
                        and token_numeric_source_slot_ids_tensor is not None
                    ):
                        predicate_logits = outputs.get(
                            "token_numeric_source_slot_predicate_logits"
                        )
                        if predicate_logits is None:
                            raise ValueError(
                                "model did not return "
                                "token_numeric_source_slot_predicate_logits"
                            )
                        predicate_loss, predicate_metrics = (
                            token_numeric_source_slot_predicate_ce_loss(
                                predicate_logits,
                                token_numeric_source_slot_ids_tensor,
                            )
                        )
                        losses.append(predicate_loss)
                        loss_weights.append(
                            float(args.token_numeric_source_slot_predicate_ce_weight)
                            * _causal_prefix_example_loss_weight(
                                example_index,
                                args.causal_prefix_later_token_weight,
                            )
                        )
                        add_metrics(predicate_metrics)
                    if float(args.answer_selective_context_alignment_weight) != 0.0:
                        with torch.no_grad():
                            dense_context_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_numeric_value_ids=token_numeric_value_ids_tensor,
                                token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                                token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                                token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                                **donor_forward_kwargs(donor_out),
                                core_world_model_actions=core_world_model_actions,
                                temporal_spatial_context=temporal_spatial_context,
                                force_answer_state_loop_dense_context=True,
                                logit_token_indices=logit_token_indices,
                            )
                    if (
                        float(args.temporal_spatial_context_contrast_weight) != 0.0
                        and temporal_spatial_context is not None
                        and example_index == 0
                    ):
                        context_off_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_numeric_value_ids=token_numeric_value_ids_tensor,
                            token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                            token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                            token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                            **donor_forward_kwargs(donor_out),
                            core_world_model_actions=core_world_model_actions,
                            temporal_spatial_context=temporal_spatial_context,
                            disable_temporal_spatial_context=True,
                            return_core_depth_logits=not bool(
                                args.final_path_only_supervision
                            ),
                            return_core_depth_text_logits=not bool(
                                args.final_path_only_supervision
                            ),
                            logit_token_indices=logit_token_indices,
                        )
                    if (
                        float(args.transition_state_contrast_weight) != 0.0
                        and example_index == 0
                    ):
                        transition_state_off_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_numeric_value_ids=token_numeric_value_ids_tensor,
                            token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                            token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                            token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                            **donor_forward_kwargs(donor_out),
                            core_world_model_actions=core_world_model_actions,
                            temporal_spatial_context=temporal_spatial_context,
                            disable_transition_state=True,
                            return_core_depth_logits=not bool(
                                args.final_path_only_supervision
                            ),
                            return_core_depth_text_logits=not bool(
                                args.final_path_only_supervision
                            ),
                            logit_token_indices=logit_token_indices,
                        )
                    if (
                        _should_apply_transition_joint_answer_bridge_contrast(
                            example_index,
                            args.transition_joint_answer_bridge_contrast_weight,
                            all_prefix_tokens=bool(
                                args.transition_joint_answer_bridge_contrast_all_prefix_tokens
                            ),
                        )
                    ):
                        bridge_off_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_numeric_value_ids=token_numeric_value_ids_tensor,
                            token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                            token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                            token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                            **donor_forward_kwargs(donor_out),
                            core_world_model_actions=core_world_model_actions,
                            temporal_spatial_context=temporal_spatial_context,
                            disable_transition_state_joint_answer_bridge=True,
                            return_core_depth_logits=not bool(
                                args.final_path_only_supervision
                            ),
                            return_core_depth_text_logits=not bool(
                                args.final_path_only_supervision
                            ),
                            logit_token_indices=logit_token_indices,
                        )
                    if (
                        _should_apply_transition_joint_answer_bridge_contrast(
                            example_index,
                            args.core_role_value_answer_bridge_contrast_weight,
                            all_prefix_tokens=bool(
                                args.core_role_value_answer_bridge_contrast_all_prefix_tokens
                            ),
                        )
                    ):
                        with torch.no_grad():
                            role_bridge_off_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_numeric_value_ids=token_numeric_value_ids_tensor,
                                token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                                token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                                token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                                **donor_forward_kwargs(donor_out),
                                core_world_model_actions=core_world_model_actions,
                                temporal_spatial_context=temporal_spatial_context,
                                disable_core_role_value_answer_bridge=True,
                                return_core_depth_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                return_core_depth_text_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                logit_token_indices=logit_token_indices,
                            )
                    if (
                        _should_apply_transition_joint_answer_bridge_contrast(
                            example_index,
                            args.core_role_value_answer_bridge_final_contrast_weight,
                            all_prefix_tokens=bool(
                                args.core_role_value_answer_bridge_contrast_all_prefix_tokens
                            ),
                        )
                        and role_bridge_off_outputs is None
                    ):
                        with torch.no_grad():
                            role_bridge_off_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_numeric_value_ids=token_numeric_value_ids_tensor,
                                token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                                token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                                token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                                **donor_forward_kwargs(donor_out),
                                core_world_model_actions=core_world_model_actions,
                                temporal_spatial_context=temporal_spatial_context,
                                disable_core_role_value_answer_bridge=True,
                                return_core_depth_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                return_core_depth_text_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                logit_token_indices=logit_token_indices,
                            )
                    if (
                        _should_apply_transition_joint_answer_bridge_contrast(
                            example_index,
                            args.typed_value_answer_bridge_final_contrast_weight,
                            all_prefix_tokens=bool(
                                args.typed_value_answer_bridge_final_contrast_all_prefix_tokens
                            ),
                        )
                    ):
                        with torch.no_grad():
                            typed_value_bridge_off_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_numeric_value_ids=token_numeric_value_ids_tensor,
                                token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                                token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                                token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                                **donor_forward_kwargs(donor_out),
                                core_world_model_actions=core_world_model_actions,
                                temporal_spatial_context=temporal_spatial_context,
                                disable_typed_algorithmic_value_state_answer_bridge=True,
                                return_core_depth_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                return_core_depth_text_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                logit_token_indices=logit_token_indices,
                            )
                    if (
                        _should_apply_transition_joint_answer_bridge_contrast(
                            example_index,
                            args.core_primitive_role_value_answer_final_contrast_weight,
                            all_prefix_tokens=bool(
                                args.core_primitive_role_value_answer_final_contrast_all_prefix_tokens
                            ),
                        )
                        or (
                            float(
                                args.core_role_value_vocab_renderer_primitive_contrast_weight
                            )
                            != 0.0
                        )
                    ):
                        with torch.no_grad():
                            primitive_role_value_off_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_numeric_value_ids=token_numeric_value_ids_tensor,
                                token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                                token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                                token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                                **donor_forward_kwargs(donor_out),
                                core_world_model_actions=core_world_model_actions,
                                temporal_spatial_context=temporal_spatial_context,
                                disable_core_primitive_role_value_executor=True,
                                return_core_depth_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                return_core_depth_text_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                logit_token_indices=logit_token_indices,
                            )
                    if (
                        float(
                            args.core_role_value_vocab_renderer_source_binder_contrast_weight
                        )
                        != 0.0
                    ):
                        with torch.no_grad():
                            source_binder_off_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_numeric_value_ids=token_numeric_value_ids_tensor,
                                token_numeric_source_slot_ids=token_numeric_source_slot_ids_tensor,
                                token_numeric_source_slot_token_ids=token_numeric_source_slot_token_ids_tensor,
                                token_numeric_source_slot_mask=token_numeric_source_slot_mask_tensor,
                                **donor_forward_kwargs(donor_out),
                                core_world_model_actions=core_world_model_actions,
                                temporal_spatial_context=temporal_spatial_context,
                                disable_core_source_position_binder=True,
                                return_core_depth_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                return_core_depth_text_logits=not bool(
                                    args.final_path_only_supervision
                                ),
                                logit_token_indices=logit_token_indices,
                            )
                finally:
                    model.cfg.outer_steps = old_outer_steps
                if bool(args.target_logit_positions_only):
                    offset = 0
                    final_text_logits = outputs["logits"]
                else:
                    offset = outputs["logits"].shape[1] - input_ids.shape[1]
                    final_text_logits = outputs["logits"][
                        :,
                        offset + target_start - 1 : offset + target_end - 1,
                        :,
                    ]
                if float(args.answer_selective_context_alignment_weight) != 0.0:
                    if bool(args.target_logit_positions_only):
                        dense_context_final_text_logits = dense_context_outputs["logits"]
                    else:
                        dense_offset = dense_context_outputs["logits"].shape[1] - input_ids.shape[1]
                        if dense_offset != offset:
                            raise ValueError("dense teacher and sparse student logit offsets must match")
                        dense_context_final_text_logits = dense_context_outputs["logits"][
                            :,
                            dense_offset + target_start - 1 : dense_offset + target_end - 1,
                            :,
                        ]
                if role_bridge_off_outputs is not None:
                    if bool(args.target_logit_positions_only):
                        role_bridge_off_final_text_logits = role_bridge_off_outputs[
                            "logits"
                        ]
                    else:
                        role_bridge_offset = (
                            role_bridge_off_outputs["logits"].shape[1] - input_ids.shape[1]
                        )
                        if role_bridge_offset != offset:
                            raise ValueError("role bridge-off and full logit offsets must match")
                        role_bridge_off_final_text_logits = role_bridge_off_outputs["logits"][
                            :,
                            role_bridge_offset + target_start - 1 : role_bridge_offset + target_end - 1,
                            :,
                        ]
                if typed_value_bridge_off_outputs is not None:
                    if bool(args.target_logit_positions_only):
                        typed_value_bridge_off_final_text_logits = (
                            typed_value_bridge_off_outputs["logits"]
                        )
                    else:
                        typed_bridge_offset = (
                            typed_value_bridge_off_outputs["logits"].shape[1]
                            - input_ids.shape[1]
                        )
                        if typed_bridge_offset != offset:
                            raise ValueError(
                                "typed bridge-off and full logit offsets must match"
                            )
                        typed_value_bridge_off_final_text_logits = (
                            typed_value_bridge_off_outputs["logits"][
                                :,
                                typed_bridge_offset
                                + target_start
                                - 1 : typed_bridge_offset
                                + target_end
                                - 1,
                                :,
                            ]
                        )
                if primitive_role_value_off_outputs is not None:
                    if bool(args.target_logit_positions_only):
                        primitive_role_value_off_final_text_logits = (
                            primitive_role_value_off_outputs["logits"]
                        )
                    else:
                        primitive_role_value_offset = (
                            primitive_role_value_off_outputs["logits"].shape[1]
                            - input_ids.shape[1]
                        )
                        if primitive_role_value_offset != offset:
                            raise ValueError("primitive-off and full logit offsets must match")
                        primitive_role_value_off_final_text_logits = primitive_role_value_off_outputs[
                            "logits"
                        ][
                            :,
                            primitive_role_value_offset
                            + target_start
                            - 1 : primitive_role_value_offset
                            + target_end
                            - 1,
                            :,
                        ]
                    if (
                        float(
                            args.core_role_value_vocab_renderer_primitive_contrast_weight
                        )
                        != 0.0
                    ):
                        if bool(args.target_logit_positions_only):
                            primitive_role_value_off_renderer_text_logits = (
                                primitive_role_value_off_outputs[
                                    "core_role_value_vocab_renderer_logits"
                                ]
                            )
                        else:
                            primitive_role_value_off_renderer_text_logits = (
                                primitive_role_value_off_outputs[
                                    "core_role_value_vocab_renderer_logits"
                                ][
                                    :,
                                    target_start - 1 : target_end - 1,
                                    :,
                                ]
                            )
                if source_binder_off_outputs is not None:
                    if bool(args.target_logit_positions_only):
                        source_binder_off_renderer_text_logits = (
                            source_binder_off_outputs[
                                "core_role_value_vocab_renderer_logits"
                            ]
                        )
                    else:
                        source_binder_off_renderer_text_logits = (
                            source_binder_off_outputs[
                                "core_role_value_vocab_renderer_logits"
                            ][
                                :,
                                target_start - 1 : target_end - 1,
                                :,
                            ]
                        )
                depth_text_logits = None
                if not bool(args.final_path_only_supervision):
                    depth_text_logits = outputs["core_depth_text_logits"][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                if context_off_depth_text_logits is None and (
                    float(args.temporal_spatial_context_contrast_weight) != 0.0
                    and temporal_spatial_context is not None
                    and example_index == 0
                ):
                    if context_off_outputs is None:
                        raise ValueError("context contrast requested but context-off output is missing")
                    context_off_depth_text_logits = context_off_outputs["core_depth_text_logits"][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                if transition_state_off_depth_text_logits is None and (
                    float(args.transition_state_contrast_weight) != 0.0
                    and example_index == 0
                ):
                    if transition_state_off_outputs is None:
                        raise ValueError(
                            "transition-state contrast requested but state-off output is missing"
                        )
                    transition_state_off_depth_text_logits = transition_state_off_outputs[
                        "core_depth_text_logits"
                    ][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                if bridge_off_depth_text_logits is None and (
                    _should_apply_transition_joint_answer_bridge_contrast(
                        example_index,
                        args.transition_joint_answer_bridge_contrast_weight,
                        all_prefix_tokens=bool(
                            args.transition_joint_answer_bridge_contrast_all_prefix_tokens
                        ),
                    )
                ):
                    if bridge_off_outputs is None:
                        raise ValueError(
                            "bridge contrast requested but bridge-off output is missing"
                        )
                    bridge_off_depth_text_logits = bridge_off_outputs[
                        "core_depth_text_logits"
                    ][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                if role_bridge_off_depth_text_logits is None and (
                    _should_apply_transition_joint_answer_bridge_contrast(
                        example_index,
                        args.core_role_value_answer_bridge_contrast_weight,
                        all_prefix_tokens=bool(
                            args.core_role_value_answer_bridge_contrast_all_prefix_tokens
                        ),
                    )
                ):
                    if role_bridge_off_outputs is None:
                        raise ValueError(
                            "core role-value bridge contrast requested but "
                            "bridge-off output is missing"
                        )
                    role_bridge_off_depth_text_logits = role_bridge_off_outputs[
                        "core_depth_text_logits"
                    ][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                teacher_depth_text_logits = None
                teacher_final_text_logits = None
                if (
                    _should_apply_teacher_first_token_depth_kl(
                        example_index,
                        args.teacher_first_token_depth_kl_weight,
                    )
                    or float(args.teacher_final_logit_kl_weight) != 0.0
                ):
                    if teacher_model is None:
                        raise ValueError("teacher model is not loaded")
                    teacher_old_outer_steps = int(teacher_model.cfg.outer_steps)
                    teacher_model.cfg.outer_steps = int(core_steps)
                    teacher_temporal_spatial_context = temporal_spatial_context
                    try:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **donor_forward_kwargs(donor_out),
                                temporal_spatial_context=teacher_temporal_spatial_context,
                                return_core_depth_logits=True,
                                return_core_depth_text_logits=True,
                                disable_answer_state_loop_talker=True,
                                logit_token_indices=logit_token_indices,
                            )
                    finally:
                        teacher_model.cfg.outer_steps = teacher_old_outer_steps
                    if bool(args.target_logit_positions_only):
                        teacher_offset = 0
                    else:
                        teacher_offset = (
                            teacher_outputs["logits"].shape[1] - input_ids.shape[1]
                        )
                        if teacher_offset != offset:
                            raise ValueError("teacher and student logit offsets must match")
                    teacher_depth_text_logits = teacher_outputs["core_depth_text_logits"][
                        :,
                        :,
                        target_start - 1 : target_end - 1,
                        :,
                    ]
                    if bool(args.target_logit_positions_only):
                        teacher_final_text_logits = teacher_outputs["logits"]
                    else:
                        teacher_final_text_logits = teacher_outputs["logits"][
                            :,
                            teacher_offset + target_start - 1 : teacher_offset + target_end - 1,
                            :,
                        ]
                if bool(args.final_path_only_supervision):
                    example_loss, example_metrics = final_path_sequence_supervision_loss(
                        final_text_logits,
                        target_ids,
                        final_logit_ce_weight=args.final_logit_ce_weight,
                        final_greedy_token_margin_weight=(
                            args.final_greedy_token_margin_weight
                        ),
                        greedy_token_margin=args.greedy_token_margin,
                    )
                else:
                    example_loss, example_metrics = depth_sequence_supervision_loss(
                        depth_text_logits,
                        final_text_logits,
                        target_ids,
                        final_logit_ce_weight=args.final_logit_ce_weight,
                        depth_final_ce_weight=args.depth_final_ce_weight,
                        all_depth_ce_weight=args.all_depth_ce_weight,
                        progress_margin_weight=args.progress_margin_weight,
                        progress_margin=args.progress_margin,
                        depth_trajectory_monotonic_weight=(
                            args.depth_trajectory_monotonic_weight
                        ),
                        depth_trajectory_monotonic_margin=(
                            args.depth_trajectory_monotonic_margin
                        ),
                        final_greedy_token_margin_weight=(
                            args.final_greedy_token_margin_weight
                        ),
                        depth_greedy_token_margin_weight=(
                            args.depth_greedy_token_margin_weight
                        ),
                        greedy_token_margin=args.greedy_token_margin,
                    )
                final_choice_weight = float(args.final_choice_margin_weight)
                if final_choice_weight != 0.0 and (
                    _choice_margin_normalize_text(answer)
                    == _choice_margin_normalize_text(_final_answer_text(row))
                ):
                    final_rejected_texts = choice_margin_rejected_texts(
                        row,
                        current_answer=answer,
                    )
                else:
                    final_rejected_texts = []
                if final_rejected_texts:
                    final_choice_losses = []
                    final_choice_metric_sums: dict[str, Any] = {}
                    target_len = int(target_ids.shape[1])
                    for rejected_text in final_rejected_texts:
                        rejected_ids = causal_prefix_answer_token_ids(
                            tokenizer,
                            rejected_text,
                            skip_leading_whitespace_targets=bool(
                                args.causal_prefix_skip_leading_whitespace_targets
                            ),
                        )
                        rejected_start = int(example_index)
                        rejected_end = rejected_start + target_len
                        if rejected_end > len(rejected_ids):
                            continue
                        rejected_target_ids = input_ids.new_tensor(
                            [rejected_ids[rejected_start:rejected_end]]
                        )
                        final_choice_loss, final_choice_metrics = (
                            final_choice_sequence_margin_loss(
                                final_text_logits,
                                target_ids,
                                rejected_target_ids,
                                margin=args.final_choice_margin,
                            )
                        )
                        final_choice_losses.append(final_choice_loss)
                        for key, value in final_choice_metrics.items():
                            final_choice_metric_sums[key] = (
                                final_choice_metric_sums.get(key, value.detach() * 0.0)
                                + value
                            )
                    if final_choice_losses:
                        final_choice_count = float(len(final_choice_losses))
                        averaged_final_choice = final_choice_losses[0]
                        for extra_final_choice in final_choice_losses[1:]:
                            averaged_final_choice = (
                                averaged_final_choice + extra_final_choice
                            )
                        averaged_final_choice = (
                            averaged_final_choice / final_choice_count
                        )
                        example_loss = (
                            example_loss + final_choice_weight * averaged_final_choice
                        )
                        for key, value in final_choice_metric_sums.items():
                            example_metrics[key] = value / final_choice_count
                if (
                    float(args.core_role_value_vocab_renderer_ce_weight) != 0.0
                    or float(
                        args.core_role_value_vocab_renderer_greedy_margin_weight
                    )
                    != 0.0
                ):
                    renderer_logits = outputs["core_role_value_vocab_renderer_logits"]
                    if bool(args.target_logit_positions_only):
                        renderer_text_logits = renderer_logits
                    else:
                        renderer_text_logits = renderer_logits[
                            :,
                            target_start - 1 : target_end - 1,
                            :,
                        ]
                    renderer_loss, renderer_metrics = (
                        core_role_value_vocab_renderer_sequence_supervision_loss(
                            renderer_text_logits,
                            target_ids,
                            renderer_ce_weight=(
                                args.core_role_value_vocab_renderer_ce_weight
                            ),
                            renderer_greedy_token_margin_weight=(
                                args.core_role_value_vocab_renderer_greedy_margin_weight
                            ),
                            greedy_token_margin=args.greedy_token_margin,
                        )
                    )
                    example_loss = example_loss + renderer_loss
                    example_metrics.update(renderer_metrics)
                if primitive_role_value_off_renderer_text_logits is not None:
                    renderer_logits = outputs["core_role_value_vocab_renderer_logits"]
                    if bool(args.target_logit_positions_only):
                        renderer_text_logits = renderer_logits
                    else:
                        renderer_text_logits = renderer_logits[
                            :,
                            target_start - 1 : target_end - 1,
                            :,
                        ]
                    renderer_primitive_contrast, renderer_primitive_metrics = (
                        final_path_ablation_contrastive_loss(
                            renderer_text_logits,
                            primitive_role_value_off_renderer_text_logits,
                            target_ids,
                            margin=(
                                args.core_role_value_vocab_renderer_primitive_contrast_margin
                            ),
                            metric_prefix=(
                                "core_role_value_vocab_renderer_primitive"
                            ),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(
                            args.core_role_value_vocab_renderer_primitive_contrast_weight
                        )
                        * renderer_primitive_contrast
                    )
                    example_metrics.update(renderer_primitive_metrics)
                if source_binder_off_renderer_text_logits is not None:
                    renderer_logits = outputs["core_role_value_vocab_renderer_logits"]
                    if bool(args.target_logit_positions_only):
                        renderer_text_logits = renderer_logits
                    else:
                        renderer_text_logits = renderer_logits[
                            :,
                            target_start - 1 : target_end - 1,
                            :,
                        ]
                    renderer_source_contrast, renderer_source_metrics = (
                        final_path_ablation_contrastive_loss(
                            renderer_text_logits,
                            source_binder_off_renderer_text_logits,
                            target_ids,
                            margin=(
                                args.core_role_value_vocab_renderer_source_binder_contrast_margin
                            ),
                            metric_prefix=(
                                "core_role_value_vocab_renderer_source_binder"
                            ),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(
                            args.core_role_value_vocab_renderer_source_binder_contrast_weight
                        )
                        * renderer_source_contrast
                    )
                    example_metrics.update(renderer_source_metrics)
                if float(args.terminal_depth_ce_weight) != 0.0:
                    terminal_mask = terminal_depth_mask_from_row(
                        row,
                        num_depths=int(depth_text_logits.shape[1]),
                        device=depth_text_logits.device,
                    )
                    terminal_ce, terminal_metrics = terminal_depth_ce_loss(
                        depth_text_logits,
                        target_ids,
                        terminal_mask,
                    )
                    example_loss = (
                        example_loss + float(args.terminal_depth_ce_weight) * terminal_ce
                    )
                    example_metrics.update(terminal_metrics)
                if (
                    float(args.answer_state_loop_halt_ce_weight) != 0.0
                    and example_index == 0
                ):
                    halt_logits = outputs["answer_state_loop_halt_logits"]
                    terminal_mask = terminal_depth_mask_from_row(
                        row,
                        num_depths=int(halt_logits.shape[1]),
                        device=halt_logits.device,
                    )
                    halt_ce, halt_metrics = answer_state_loop_halt_ce_loss(
                        halt_logits,
                        terminal_mask,
                    )
                    example_loss = (
                        example_loss
                        + float(args.answer_state_loop_halt_ce_weight) * halt_ce
                    )
                    example_metrics.update(halt_metrics)
                if float(args.answer_state_loop_logit_ce_weight) != 0.0:
                    answer_loop_logits = outputs["answer_state_loop_logits"]
                    if bool(args.target_logit_positions_only):
                        answer_loop_text_logits = answer_loop_logits
                    else:
                        answer_loop_text_logits = answer_loop_logits[
                            :,
                            target_start - 1 : target_end - 1,
                            :,
                        ]
                    answer_loop_ce, answer_loop_metrics = (
                        answer_state_loop_logit_ce_loss(
                            answer_loop_text_logits,
                            target_ids,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.answer_state_loop_logit_ce_weight)
                        * answer_loop_ce
                    )
                    example_metrics.update(answer_loop_metrics)
                if (
                    float(args.answer_state_loop_future_token_ce_weight) != 0.0
                    and example_index == 0
                ):
                    future_target_count = int(
                        args.answer_state_loop_future_token_max_target_tokens
                        or cfg.model.answer_state_loop_future_token_max_tokens
                    )
                    future_targets = answer_state_loop_future_token_targets(
                        tokenizer,
                        answer,
                        max_target_tokens=future_target_count,
                        device=device,
                    )
                    future_ce, future_metrics = (
                        answer_state_loop_future_token_ce_loss(
                            outputs["answer_state_loop_future_token_logits"],
                            future_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.answer_state_loop_future_token_ce_weight)
                        * future_ce
                    )
                    example_metrics.update(future_metrics)
                if context_off_depth_text_logits is not None:
                    context_contrast, context_metrics = context_ablation_contrastive_loss(
                        depth_text_logits,
                        context_off_depth_text_logits,
                        target_ids,
                        margin=args.temporal_spatial_context_contrast_margin,
                    )
                    example_loss = (
                        example_loss
                        + float(args.temporal_spatial_context_contrast_weight)
                        * context_contrast
                    )
                    example_metrics.update(context_metrics)
                if transition_state_off_depth_text_logits is not None:
                    transition_contrast, transition_metrics = (
                        transition_state_ablation_contrastive_loss(
                            depth_text_logits,
                            transition_state_off_depth_text_logits,
                            target_ids,
                            margin=args.transition_state_contrast_margin,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_contrast_weight)
                        * transition_contrast
                    )
                    example_metrics.update(transition_metrics)
                if bridge_off_depth_text_logits is not None:
                    bridge_contrast, bridge_metrics = (
                        transition_joint_answer_bridge_contrastive_loss(
                            depth_text_logits,
                            bridge_off_depth_text_logits,
                            target_ids,
                            margin=args.transition_joint_answer_bridge_contrast_margin,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_joint_answer_bridge_contrast_weight)
                        * bridge_contrast
                    )
                    example_metrics.update(bridge_metrics)
                if role_bridge_off_depth_text_logits is not None:
                    role_bridge_contrast, role_bridge_metrics = (
                        core_role_value_answer_bridge_contrastive_loss(
                            depth_text_logits,
                            role_bridge_off_depth_text_logits,
                            target_ids,
                            margin=args.core_role_value_answer_bridge_contrast_margin,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_role_value_answer_bridge_contrast_weight)
                        * role_bridge_contrast
                    )
                    example_metrics.update(role_bridge_metrics)
                if role_bridge_off_final_text_logits is not None and (
                    _should_apply_transition_joint_answer_bridge_contrast(
                        example_index,
                        args.core_role_value_answer_bridge_final_contrast_weight,
                        all_prefix_tokens=bool(
                            args.core_role_value_answer_bridge_contrast_all_prefix_tokens
                        ),
                    )
                ):
                    role_final_contrast, role_final_metrics = (
                        final_path_ablation_contrastive_loss(
                            final_text_logits,
                            role_bridge_off_final_text_logits,
                            target_ids,
                            margin=args.core_role_value_answer_bridge_final_contrast_margin,
                            metric_prefix="core_role_value_answer_bridge",
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_role_value_answer_bridge_final_contrast_weight)
                        * role_final_contrast
                    )
                    example_metrics.update(role_final_metrics)
                if typed_value_bridge_off_final_text_logits is not None and (
                    _should_apply_transition_joint_answer_bridge_contrast(
                        example_index,
                        args.typed_value_answer_bridge_final_contrast_weight,
                        all_prefix_tokens=bool(
                            args.typed_value_answer_bridge_final_contrast_all_prefix_tokens
                        ),
                    )
                ):
                    typed_bridge_final_contrast, typed_bridge_final_metrics = (
                        final_path_ablation_contrastive_loss(
                            final_text_logits,
                            typed_value_bridge_off_final_text_logits,
                            target_ids,
                            margin=args.typed_value_answer_bridge_final_contrast_margin,
                            metric_prefix="typed_value_answer_bridge",
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.typed_value_answer_bridge_final_contrast_weight)
                        * typed_bridge_final_contrast
                    )
                    example_metrics.update(typed_bridge_final_metrics)
                if primitive_role_value_off_final_text_logits is not None and (
                    _should_apply_transition_joint_answer_bridge_contrast(
                        example_index,
                        args.core_primitive_role_value_answer_final_contrast_weight,
                        all_prefix_tokens=bool(
                            args.core_primitive_role_value_answer_final_contrast_all_prefix_tokens
                        ),
                    )
                ):
                    primitive_final_contrast, primitive_final_metrics = (
                        final_path_ablation_contrastive_loss(
                            final_text_logits,
                            primitive_role_value_off_final_text_logits,
                            target_ids,
                            margin=(
                                args.core_primitive_role_value_answer_final_contrast_margin
                            ),
                            metric_prefix="core_primitive_role_value_answer",
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_primitive_role_value_answer_final_contrast_weight)
                        * primitive_final_contrast
                    )
                    example_metrics.update(primitive_final_metrics)
                if dense_context_final_text_logits is not None:
                    alignment_loss, alignment_metrics = (
                        answer_selective_context_alignment_loss(
                            final_text_logits,
                            dense_context_final_text_logits,
                            temperature=args.answer_selective_context_alignment_temperature,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.answer_selective_context_alignment_weight)
                        * alignment_loss
                    )
                    example_metrics.update(alignment_metrics)
                if (
                    float(args.staged_internal_first_token_ce_weight) != 0.0
                    and example_index == 0
                ):
                    staged_target_ids = staged_internal_first_token_targets(
                        tokenizer,
                        row,
                        num_depths=int(depth_text_logits.shape[1]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    staged_ce, staged_metrics = staged_internal_first_token_ce_loss(
                        depth_text_logits,
                        staged_target_ids,
                    )
                    example_loss = (
                        example_loss
                        + float(args.staged_internal_first_token_ce_weight) * staged_ce
                    )
                    example_metrics.update(staged_metrics)
                if (
                    float(args.staged_internal_sequence_ce_weight) != 0.0
                    and example_index == 0
                ):
                    staged_sequence_target_ids = staged_internal_sequence_targets(
                        tokenizer,
                        row,
                        num_depths=int(depth_text_logits.shape[1]),
                        max_target_tokens=min(
                            int(args.staged_internal_sequence_max_target_tokens),
                            int(depth_text_logits.shape[2]),
                        ),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    staged_sequence_ce, staged_sequence_metrics = (
                        staged_internal_sequence_ce_loss(
                            depth_text_logits,
                            staged_sequence_target_ids,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.staged_internal_sequence_ce_weight)
                        * staged_sequence_ce
                    )
                    example_metrics.update(staged_sequence_metrics)
                if (
                    float(args.transition_state_sequence_ce_weight) != 0.0
                    and example_index == 0
                ):
                    transition_sequence_logits = outputs[
                        "transition_state_sequence_logits"
                    ]
                    transition_sequence_target_ids = staged_internal_sequence_targets(
                        tokenizer,
                        row,
                        num_depths=int(transition_sequence_logits.shape[1]),
                        max_target_tokens=min(
                            int(args.staged_internal_sequence_max_target_tokens),
                            int(transition_sequence_logits.shape[2]),
                        ),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    transition_sequence_ce, transition_sequence_metrics = (
                        staged_internal_sequence_ce_loss(
                            transition_sequence_logits,
                            transition_sequence_target_ids,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_sequence_ce_weight)
                        * transition_sequence_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "staged_internal_sequence",
                                "transition_state_sequence",
                            ): value
                            for key, value in transition_sequence_metrics.items()
                        }
                    )
                if (
                    float(args.transition_value_state_ce_weight) != 0.0
                    and example_index == 0
                ):
                    value_state_logits = outputs["transition_value_state_logits"]
                    value_state_target_ids = transition_value_state_targets(
                        row,
                        num_depths=int(value_state_logits.shape[1]),
                        max_target_tokens=min(
                            int(args.transition_value_state_max_target_tokens),
                            int(value_state_logits.shape[2]),
                        ),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    value_state_ce, value_state_metrics = staged_internal_sequence_ce_loss(
                        value_state_logits,
                        value_state_target_ids,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_value_state_ce_weight) * value_state_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "staged_internal_sequence",
                                "transition_value_state",
                            ): value
                            for key, value in value_state_metrics.items()
                        }
                    )
                if (
                    float(args.algorithmic_value_state_ce_weight) != 0.0
                    and example_index == 0
                ):
                    algorithmic_kind_logits = outputs[
                        "factorized_value_state_kind_logits"
                    ]
                    algorithmic_slot_logits = outputs["factorized_value_state_logits"]
                    algorithmic_kind_targets, algorithmic_slot_targets = (
                        algorithmic_value_state_targets(
                            row,
                            num_depths=int(algorithmic_slot_logits.shape[1]),
                            max_slots=int(algorithmic_slot_logits.shape[2]),
                            slot_vocab_size=int(algorithmic_slot_logits.shape[3]),
                            device=device,
                            target_mode=args.target_mode,
                        )
                    )
                    algorithmic_ce, algorithmic_metrics = (
                        algorithmic_value_state_ce_loss(
                            algorithmic_kind_logits,
                            algorithmic_slot_logits,
                            algorithmic_kind_targets,
                            algorithmic_slot_targets,
                            pad_ce_weight=float(
                                args.algorithmic_value_state_pad_ce_weight
                            ),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.algorithmic_value_state_ce_weight)
                        * algorithmic_ce
                    )
                    example_metrics.update(algorithmic_metrics)
                if (
                    float(args.algorithmic_role_value_state_ce_weight) != 0.0
                    and example_index == 0
                ):
                    role_value_logits = outputs["role_value_state_logits"]
                    core_role_value_logits = outputs.get(
                        "core_role_value_state_logits"
                    )
                    if (
                        core_role_value_logits is not None
                        and core_role_value_logits.ndim == 4
                        and int(core_role_value_logits.shape[1]) > 0
                        and int(core_role_value_logits.shape[2]) > 0
                    ):
                        role_value_logits = core_role_value_logits
                    role_value_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(role_value_logits.shape[1]),
                        num_roles=int(role_value_logits.shape[2]),
                        value_vocab_size=int(role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    role_value_ce, role_value_metrics = (
                        algorithmic_role_value_state_ce_loss(
                            role_value_logits,
                            role_value_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.algorithmic_role_value_state_ce_weight)
                        * role_value_ce
                    )
                    example_metrics.update(role_value_metrics)
                if (
                    float(args.algorithmic_role_value_step_margin_weight) != 0.0
                    and example_index == 0
                ):
                    role_value_logits = outputs["role_value_state_logits"]
                    core_role_value_logits = outputs.get(
                        "core_role_value_state_logits"
                    )
                    if (
                        core_role_value_logits is not None
                        and core_role_value_logits.ndim == 4
                        and int(core_role_value_logits.shape[1]) > 0
                        and int(core_role_value_logits.shape[2]) > 0
                    ):
                        role_value_logits = core_role_value_logits
                    role_value_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(role_value_logits.shape[1]),
                        num_roles=int(role_value_logits.shape[2]),
                        value_vocab_size=int(role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    role_value_margin, role_value_margin_metrics = (
                        algorithmic_role_value_step_margin_loss(
                            role_value_logits,
                            role_value_targets,
                            margin=float(args.algorithmic_role_value_step_margin),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.algorithmic_role_value_step_margin_weight)
                        * role_value_margin
                    )
                    example_metrics.update(role_value_margin_metrics)
                if (
                    float(args.core_role_value_prompt_ce_weight) != 0.0
                    and example_index == 0
                ):
                    prompt_role_value_logits = outputs.get(
                        "core_role_value_state_prompt_logits"
                    )
                    if prompt_role_value_logits is None:
                        raise RuntimeError(
                            "model did not return core_role_value_state_prompt_logits"
                        )
                    if str(args.core_role_value_prompt_target_mode) == "initial":
                        prompt_role_value_targets = (
                            algorithmic_role_value_initial_state_targets(
                                row,
                                num_steps=int(prompt_role_value_logits.shape[1]),
                                num_roles=int(prompt_role_value_logits.shape[2]),
                                value_vocab_size=int(prompt_role_value_logits.shape[3]),
                                device=device,
                                include_metadata=bool(
                                    args.core_role_value_prompt_initial_metadata_targets
                                ),
                            )
                        )
                    else:
                        prompt_role_value_targets = (
                            algorithmic_role_value_state_targets(
                                row,
                                num_depths=int(prompt_role_value_logits.shape[1]),
                                num_roles=int(prompt_role_value_logits.shape[2]),
                                value_vocab_size=int(prompt_role_value_logits.shape[3]),
                                device=device,
                                target_mode=args.target_mode,
                            )
                        )
                    prompt_role_value_ce, prompt_role_value_metrics = (
                        algorithmic_role_value_state_ce_loss(
                            prompt_role_value_logits,
                            prompt_role_value_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_role_value_prompt_ce_weight)
                        * prompt_role_value_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value_state",
                                "core_role_value_prompt",
                            ): value
                            for key, value in prompt_role_value_metrics.items()
                        }
                    )
                if (
                    float(args.core_source_value_prompt_ce_weight) != 0.0
                    and example_index == 0
                ):
                    source_value_logits = outputs.get(
                        "core_source_value_prompt_logits"
                    )
                    if source_value_logits is None:
                        raise RuntimeError(
                            "model did not return core_source_value_prompt_logits"
                        )
                    source_value_targets = (
                        algorithmic_role_value_initial_source_value_targets(
                            row,
                            num_steps=int(source_value_logits.shape[1]),
                            num_roles=int(source_value_logits.shape[2]),
                            value_vocab_size=int(source_value_logits.shape[3]),
                            device=device,
                        )
                    )
                    source_value_ce, source_value_metrics = (
                        algorithmic_role_value_state_ce_loss(
                            source_value_logits,
                            source_value_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_source_value_prompt_ce_weight)
                        * source_value_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value_state",
                                "core_source_value_prompt",
                            ): value
                            for key, value in source_value_metrics.items()
                        }
                    )
                if (
                    float(args.core_role_value_prompt_parity_ce_weight) != 0.0
                    and example_index == 0
                ):
                    parity_logits = outputs.get(
                        "core_role_value_state_prompt_parity_logits"
                    )
                    if parity_logits is None:
                        raise RuntimeError(
                            "model did not return core_role_value_state_prompt_parity_logits"
                        )
                    if parity_logits.ndim != 2 or int(parity_logits.shape[1]) < 2:
                        raise RuntimeError(
                            "core role-value prompt parity CE requires "
                            "model.core_role_value_state_prompt_parity_enabled=true"
                        )
                    parity_targets = core_role_value_prompt_parity_target(
                        row,
                        device=device,
                    )
                    parity_ce, parity_metrics = core_role_value_prompt_parity_ce_loss(
                        parity_logits,
                        parity_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_role_value_prompt_parity_ce_weight)
                        * parity_ce
                    )
                    example_metrics.update(parity_metrics)
                if (
                    float(args.typed_algorithmic_value_state_ce_weight) != 0.0
                    and example_index == 0
                ):
                    typed_logits = {
                        "kind_logits": outputs["typed_algorithmic_kind_logits"],
                        "raw_list_offset_logits": outputs[
                            "typed_algorithmic_raw_list_offset_logits"
                        ],
                        "doubled_list_offset_logits": outputs[
                            "typed_algorithmic_doubled_list_offset_logits"
                        ],
                        "scalar_coeff_logits": outputs[
                            "typed_algorithmic_scalar_coeff_logits"
                        ],
                        "scalar_coeff_value": outputs[
                            "typed_algorithmic_scalar_coeff_value"
                        ],
                        "scalar_offset_logits": outputs[
                            "typed_algorithmic_scalar_offset_logits"
                        ],
                        "scalar_offset_value": outputs[
                            "typed_algorithmic_scalar_offset_value"
                        ],
                        "scalar_residual_logits": outputs[
                            "typed_algorithmic_scalar_residual_logits"
                        ],
                        "scalar_residual_value": outputs[
                            "typed_algorithmic_scalar_residual_value"
                        ],
                        "final_residual_logits": outputs[
                            "typed_algorithmic_final_residual_logits"
                        ],
                        "final_residual_value": outputs[
                            "typed_algorithmic_final_residual_value"
                        ],
                    }
                    typed_targets = typed_algorithmic_value_state_targets(
                        row,
                        num_depths=int(typed_logits["kind_logits"].shape[1]),
                        max_list_slots=int(
                            typed_logits["raw_list_offset_logits"].shape[2]
                        ),
                        offset_vocab_size=int(
                            typed_logits["raw_list_offset_logits"].shape[3]
                        ),
                        scalar_vocab_size=int(
                            typed_logits["scalar_coeff_logits"].shape[2]
                        ),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    typed_ce, typed_metrics = typed_algorithmic_value_state_ce_loss(
                        typed_logits,
                        typed_targets,
                        pad_ce_weight=float(
                            args.typed_algorithmic_value_state_pad_ce_weight
                        ),
                        kind_ce_multiplier=float(
                            args.typed_algorithmic_kind_ce_multiplier
                        ),
                        list_ce_multiplier=float(
                            args.typed_algorithmic_list_ce_multiplier
                        ),
                        scalar_ce_multiplier=float(
                            args.typed_algorithmic_scalar_ce_multiplier
                        ),
                        residual_delta_ce_multiplier=float(
                            args.typed_algorithmic_residual_delta_ce_multiplier
                        ),
                        scalar_ordinal_weight=float(
                            args.typed_algorithmic_scalar_ordinal_weight
                        ),
                        scalar_regression_weight=float(
                            args.typed_algorithmic_scalar_regression_weight
                        ),
                    )
                    example_loss = (
                        example_loss
                        + float(args.typed_algorithmic_value_state_ce_weight)
                        * typed_ce
                    )
                    example_metrics.update(typed_metrics)
                if (
                    float(args.algorithmic_role_value_transition_ce_weight) != 0.0
                    and example_index == 0
                ):
                    transition_logits = outputs.get(
                        "core_role_value_transition_logits"
                    )
                    if transition_logits is None:
                        raise ValueError(
                            "model did not return core_role_value_transition_logits"
                        )
                    role_value_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(transition_logits.shape[1]) + 1,
                        num_roles=int(transition_logits.shape[2]),
                        value_vocab_size=int(transition_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    role_transition_ce, role_transition_metrics = (
                        algorithmic_role_value_transition_ce_loss(
                            transition_logits,
                            role_value_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.algorithmic_role_value_transition_ce_weight)
                        * role_transition_ce
                    )
                    example_metrics.update(role_transition_metrics)
                if (
                    float(args.core_value_delta_code_ce_weight) != 0.0
                    and example_index == 0
                ):
                    code_logits = outputs.get("core_value_delta_code_logits")
                    if code_logits is None:
                        raise ValueError(
                            "model did not return core_value_delta_code_logits"
                        )
                    code_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(code_logits.shape[1]),
                        num_roles=int(code_logits.shape[2]),
                        value_vocab_size=int(code_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    delta_code_ce, delta_code_metrics = (
                        algorithmic_role_value_state_ce_loss(
                            code_logits,
                            code_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_value_delta_code_ce_weight)
                        * delta_code_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value_state",
                                "core_value_delta_code",
                            ): value
                            for key, value in delta_code_metrics.items()
                        }
                    )
                primitive_role_value_logits = outputs.get(
                    "core_primitive_role_value_state_logits"
                )
                if (
                    float(args.core_primitive_role_value_state_ce_weight) != 0.0
                    and example_index == 0
                ):
                    if primitive_role_value_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_state_logits"
                        )
                    primitive_role_value_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(primitive_role_value_logits.shape[1]),
                        num_roles=int(primitive_role_value_logits.shape[2]),
                        value_vocab_size=int(primitive_role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    primitive_role_value_ce, primitive_role_value_metrics = (
                        algorithmic_role_value_state_ce_loss(
                            primitive_role_value_logits,
                            primitive_role_value_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_primitive_role_value_state_ce_weight)
                        * primitive_role_value_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value_state",
                                "core_primitive_role_value",
                            ): value
                            for key, value in primitive_role_value_metrics.items()
                        }
                    )
                if (
                    float(args.core_primitive_role_value_step_margin_weight) != 0.0
                    and example_index == 0
                ):
                    if primitive_role_value_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_state_logits"
                        )
                    primitive_role_value_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(primitive_role_value_logits.shape[1]),
                        num_roles=int(primitive_role_value_logits.shape[2]),
                        value_vocab_size=int(primitive_role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    primitive_role_value_margin, primitive_role_value_margin_metrics = (
                        algorithmic_role_value_step_margin_loss(
                            primitive_role_value_logits,
                            primitive_role_value_targets,
                            margin=float(
                                args.core_primitive_role_value_step_margin
                            ),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_primitive_role_value_step_margin_weight)
                        * primitive_role_value_margin
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value",
                                "core_primitive_role_value",
                            ): value
                            for key, value in primitive_role_value_margin_metrics.items()
                        }
                    )
                if (
                    float(args.core_primitive_role_value_trace_margin_weight) != 0.0
                    and example_index == 0
                ):
                    if primitive_role_value_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_state_logits"
                        )
                    primitive_role_value_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(primitive_role_value_logits.shape[1]),
                        num_roles=int(primitive_role_value_logits.shape[2]),
                        value_vocab_size=int(primitive_role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    primitive_role_value_trace, primitive_role_value_trace_metrics = (
                        algorithmic_role_value_trace_margin_loss(
                            primitive_role_value_logits,
                            primitive_role_value_targets,
                            margin=float(
                                args.core_primitive_role_value_trace_margin
                            ),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_primitive_role_value_trace_margin_weight)
                        * primitive_role_value_trace
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value",
                                "core_primitive_role_value",
                            ): value
                            for key, value in primitive_role_value_trace_metrics.items()
                        }
                    )
                if (
                    float(args.core_primitive_role_value_pair_trace_contrast_weight)
                    != 0.0
                    and example_index == 0
                    and row_index in paired_hard_negative_lookup
                ):
                    if primitive_role_value_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_state_logits"
                        )
                    paired_row = rows[paired_hard_negative_lookup[row_index]]
                    positive_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(primitive_role_value_logits.shape[1]),
                        num_roles=int(primitive_role_value_logits.shape[2]),
                        value_vocab_size=int(primitive_role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    negative_targets = algorithmic_role_value_state_targets(
                        paired_row,
                        num_depths=int(primitive_role_value_logits.shape[1]),
                        num_roles=int(primitive_role_value_logits.shape[2]),
                        value_vocab_size=int(primitive_role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    pair_trace_contrast, pair_trace_metrics = (
                        core_primitive_role_value_pair_trace_contrastive_loss(
                            primitive_role_value_logits,
                            positive_targets,
                            negative_targets,
                            margin=float(
                                args.core_primitive_role_value_pair_trace_contrast_margin
                            ),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_primitive_role_value_pair_trace_contrast_weight)
                        * pair_trace_contrast
                    )
                    example_metrics.update(pair_trace_metrics)
                if (
                    float(args.core_primitive_role_value_update_gate_bce_weight) != 0.0
                    and example_index == 0
                ):
                    primitive_update_gate = outputs.get(
                        "core_primitive_role_value_update_gate"
                    )
                    if primitive_update_gate is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_update_gate"
                        )
                    if primitive_role_value_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_state_logits"
                        )
                    primitive_gate_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(primitive_update_gate.shape[1]),
                        num_roles=int(primitive_update_gate.shape[2]),
                        value_vocab_size=int(primitive_role_value_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    primitive_gate_initial_targets = (
                        algorithmic_role_value_initial_state_targets(
                            row,
                            num_steps=1,
                            num_roles=int(primitive_update_gate.shape[2]),
                            value_vocab_size=int(primitive_role_value_logits.shape[3]),
                            device=device,
                            include_metadata=bool(
                                args.core_role_value_prompt_initial_metadata_targets
                            ),
                        )
                    )
                    primitive_gate_bce, primitive_gate_metrics = (
                        core_primitive_role_value_update_gate_bce_loss(
                            primitive_update_gate,
                            primitive_gate_targets,
                            primitive_gate_initial_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(
                            args.core_primitive_role_value_update_gate_bce_weight
                        )
                        * primitive_gate_bce
                    )
                    example_metrics.update(primitive_gate_metrics)
                if (
                    float(args.core_role_value_template_ce_weight) != 0.0
                    and example_index == 0
                ):
                    template_logits = outputs.get("core_role_value_template_logits")
                    if template_logits is None:
                        raise ValueError(
                            "model did not return core_role_value_template_logits"
                        )
                    template_targets = core_role_value_template_targets(
                        row,
                        num_templates=int(template_logits.shape[-1]),
                        device=device,
                    )
                    template_ce, template_metrics = core_role_value_template_ce_loss(
                        template_logits,
                        template_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_role_value_template_ce_weight)
                        * template_ce
                    )
                    example_metrics.update(template_metrics)
                if (
                    float(args.core_role_value_template_table_ce_weight) != 0.0
                    and example_index == 0
                ):
                    template_logits = outputs.get("core_role_value_template_logits")
                    if template_logits is None:
                        raise ValueError(
                            "model did not return core_role_value_template_logits"
                        )
                    template_targets = core_role_value_template_targets(
                        row,
                        num_templates=int(template_logits.shape[-1]),
                        device=device,
                    )
                    table_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(model.cfg.core_role_value_template_max_steps),
                        num_roles=int(model.cfg.core_role_value_state_num_roles),
                        value_vocab_size=int(model.cfg.core_role_value_state_vocab_size),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    table_ce, table_metrics = core_role_value_template_table_ce_loss(
                        getattr(model, "core_role_value_template_table", None),
                        template_targets,
                        table_targets,
                        num_steps=int(model.cfg.core_role_value_template_max_steps),
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_role_value_template_table_ce_weight)
                        * table_ce
                    )
                    example_metrics.update(table_metrics)
                if (
                    float(args.core_typed_register_ce_weight) != 0.0
                    and example_index == 0
                ):
                    typed_register_logits = outputs.get(
                        "core_typed_register_value_logits"
                    )
                    if typed_register_logits is None:
                        raise ValueError(
                            "model did not return core_typed_register_value_logits"
                        )
                    typed_register_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(typed_register_logits.shape[1]),
                        num_roles=int(typed_register_logits.shape[2]),
                        value_vocab_size=int(typed_register_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    typed_register_role_weights = (
                        algorithmic_role_value_scalar_role_weights(
                            typed_register_targets,
                            multiplier=float(
                                args.core_typed_register_scalar_role_ce_multiplier
                            ),
                        )
                    )
                    typed_register_ce, typed_register_metrics = (
                        algorithmic_role_value_state_ce_loss(
                            typed_register_logits,
                            typed_register_targets,
                            role_weights=typed_register_role_weights,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_typed_register_ce_weight)
                        * typed_register_ce
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value_state",
                                "core_typed_register",
                            ): value
                            for key, value in typed_register_metrics.items()
                        }
                    )
                if (
                    float(args.core_typed_register_step_margin_weight) != 0.0
                    and example_index == 0
                ):
                    typed_register_logits = outputs.get(
                        "core_typed_register_value_logits"
                    )
                    if typed_register_logits is None:
                        raise ValueError(
                            "model did not return core_typed_register_value_logits"
                        )
                    typed_register_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(typed_register_logits.shape[1]),
                        num_roles=int(typed_register_logits.shape[2]),
                        value_vocab_size=int(typed_register_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    typed_register_margin, typed_register_margin_metrics = (
                        algorithmic_role_value_step_margin_loss(
                            typed_register_logits,
                            typed_register_targets,
                            margin=float(args.core_typed_register_step_margin),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_typed_register_step_margin_weight)
                        * typed_register_margin
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value",
                                "core_typed_register",
                            ): value
                            for key, value in typed_register_margin_metrics.items()
                        }
                    )
                if (
                    float(args.core_typed_register_trace_margin_weight) != 0.0
                    and example_index == 0
                ):
                    typed_register_logits = outputs.get(
                        "core_typed_register_value_logits"
                    )
                    if typed_register_logits is None:
                        raise ValueError(
                            "model did not return core_typed_register_value_logits"
                        )
                    typed_register_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(typed_register_logits.shape[1]),
                        num_roles=int(typed_register_logits.shape[2]),
                        value_vocab_size=int(typed_register_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    typed_register_trace, typed_register_trace_metrics = (
                        algorithmic_role_value_trace_margin_loss(
                            typed_register_logits,
                            typed_register_targets,
                            margin=float(args.core_typed_register_trace_margin),
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_typed_register_trace_margin_weight)
                        * typed_register_trace
                    )
                    example_metrics.update(
                        {
                            key.replace(
                                "algorithmic_role_value",
                                "core_typed_register",
                            ): value
                            for key, value in typed_register_trace_metrics.items()
                        }
                    )
                if (
                    float(args.core_typed_register_operation_ce_weight) != 0.0
                    and example_index == 0
                ):
                    typed_operation_logits = outputs.get(
                        "core_typed_register_operation_logits"
                    )
                    if typed_operation_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_typed_register_operation_logits"
                        )
                    typed_operation_targets = core_typed_register_operation_targets(
                        tokenizer,
                        row,
                        num_steps=int(typed_operation_logits.shape[1]),
                        num_operations=int(typed_operation_logits.shape[2]),
                        device=device,
                        target_mode=args.target_mode,
                        target_shift=int(
                            args.core_typed_register_operation_target_shift
                        ),
                    )
                    typed_operation_ce, typed_operation_metrics = (
                        core_typed_register_operation_ce_loss(
                            typed_operation_logits,
                            typed_operation_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_typed_register_operation_ce_weight)
                        * typed_operation_ce
                    )
                    example_metrics.update(typed_operation_metrics)
                if (
                    float(args.core_typed_register_transition_ce_weight) != 0.0
                    and example_index == 0
                ):
                    typed_transition_logits = outputs.get(
                        "core_typed_register_transition_logits"
                    )
                    if typed_transition_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_typed_register_transition_logits"
                        )
                    typed_transition_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(typed_transition_logits.shape[1]) + 1,
                        num_roles=int(typed_transition_logits.shape[2]),
                        value_vocab_size=int(typed_transition_logits.shape[3]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    typed_transition_ce, typed_transition_metrics = (
                        core_typed_register_transition_ce_loss(
                            typed_transition_logits,
                            typed_transition_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_typed_register_transition_ce_weight)
                        * typed_transition_ce
                    )
                    example_metrics.update(typed_transition_metrics)
                if (
                    float(args.core_primitive_typed_selector_bce_weight) != 0.0
                    and example_index == 0
                ):
                    selector_gate = outputs.get("core_primitive_typed_selector_gate")
                    primitive_role_value_logits = outputs.get(
                        "core_primitive_role_value_state_logits"
                    )
                    typed_register_logits = outputs.get(
                        "core_typed_register_value_logits"
                    )
                    if selector_gate is None:
                        raise ValueError(
                            "model did not return core_primitive_typed_selector_gate"
                        )
                    if primitive_role_value_logits is None:
                        raise ValueError(
                            "model did not return "
                            "core_primitive_role_value_state_logits"
                        )
                    if typed_register_logits is None:
                        raise ValueError(
                            "model did not return core_typed_register_value_logits"
                        )
                    selector_targets = algorithmic_role_value_state_targets(
                        row,
                        num_depths=int(selector_gate.shape[1]),
                        num_roles=int(selector_gate.shape[2]),
                        value_vocab_size=min(
                            int(primitive_role_value_logits.shape[3]),
                            int(typed_register_logits.shape[3]),
                        ),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    selector_bce, selector_metrics = (
                        core_primitive_typed_selector_bce_loss(
                            selector_gate,
                            primitive_role_value_logits,
                            typed_register_logits,
                            selector_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_primitive_typed_selector_bce_weight)
                        * selector_bce
                    )
                    example_metrics.update(selector_metrics)
                if (
                    (
                        float(args.transition_state_ce_weight) != 0.0
                        or float(args.transition_state_depth_contrast_weight) != 0.0
                    )
                    and example_index == 0
                ):
                    transition_targets = staged_internal_first_token_targets(
                        tokenizer,
                        row,
                        num_depths=int(outputs["transition_state_text_logits"].shape[1]),
                        device=device,
                        target_mode=args.target_mode,
                        content_token=True,
                    )
                    if float(args.transition_state_ce_weight) != 0.0:
                        transition_ce, transition_metrics = (
                            transition_state_first_token_ce_loss(
                                outputs["transition_state_text_logits"],
                                transition_targets,
                            )
                        )
                        example_loss = (
                            example_loss
                            + float(args.transition_state_ce_weight) * transition_ce
                        )
                        example_metrics.update(transition_metrics)
                    if float(args.transition_state_depth_contrast_weight) != 0.0:
                        depth_contrast, depth_contrast_metrics = (
                            transition_state_depth_contrast_loss(
                                outputs["transition_state_text_logits"],
                                transition_targets,
                                margin=args.transition_state_depth_contrast_margin,
                            )
                        )
                        example_loss = (
                            example_loss
                            + float(args.transition_state_depth_contrast_weight)
                            * depth_contrast
                        )
                        example_metrics.update(depth_contrast_metrics)
                if (
                    float(args.transition_state_code_ce_weight) != 0.0
                    and example_index == 0
                ):
                    code_logits = outputs["transition_state_code_logits"]
                    code_targets = transition_state_code_targets(
                        tokenizer,
                        row,
                        num_depths=int(code_logits.shape[1]),
                        codebook_size=int(code_logits.shape[-1]),
                        device=device,
                        target_mode=args.target_mode,
                    )
                    code_ce, code_metrics = transition_state_code_ce_loss(
                        code_logits,
                        code_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_code_ce_weight) * code_ce
                    )
                    example_metrics.update(code_metrics)
                if (
                    float(args.transition_state_finality_ce_weight) != 0.0
                    and example_index == 0
                ):
                    finality_logits = outputs["transition_state_finality_logits"]
                    finality_targets = transition_state_finality_targets(
                        row,
                        num_depths=int(finality_logits.shape[1]),
                        device=device,
                    )
                    finality_ce, finality_metrics = transition_state_finality_bce_loss(
                        finality_logits,
                        finality_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_finality_ce_weight) * finality_ce
                    )
                    example_metrics.update(finality_metrics)
                if (
                    float(args.transition_state_joint_ce_weight) != 0.0
                    and example_index == 0
                ):
                    joint_logits = outputs["transition_state_joint_logits"]
                    joint_targets = transition_state_joint_targets(
                        row,
                        num_depths=int(joint_logits.shape[1]),
                        joint_size=int(joint_logits.shape[-1]),
                        device=device,
                    )
                    joint_ce, joint_metrics = transition_state_joint_ce_loss(
                        joint_logits,
                        joint_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_joint_ce_weight) * joint_ce
                    )
                    example_metrics.update(joint_metrics)
                if (
                    float(args.transition_state_joint_order_contrast_weight) != 0.0
                    and example_index == 0
                ):
                    joint_logits = outputs["transition_state_joint_logits"]
                    order_contrast, order_contrast_metrics = (
                        transition_state_joint_order_contrast_loss(
                            joint_logits,
                            row,
                            margin=args.transition_state_joint_order_contrast_margin,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_state_joint_order_contrast_weight)
                        * order_contrast
                    )
                    example_metrics.update(order_contrast_metrics)
                if (
                    float(args.primitive_transition_operation_ce_weight) != 0.0
                    and example_index == 0
                ):
                    operation_logits = outputs["primitive_transition_operation_logits"]
                    operation_targets = primitive_transition_operation_targets(
                        row,
                        num_steps=int(operation_logits.shape[1]),
                        operation_to_id=primitive_operation_to_id,
                        device=device,
                    )
                    operation_ce, operation_metrics = primitive_transition_operation_ce_loss(
                        operation_logits,
                        operation_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.primitive_transition_operation_ce_weight)
                        * operation_ce
                    )
                    example_metrics.update(operation_metrics)
                if (
                    float(args.core_transition_feedback_operation_ce_weight) != 0.0
                    and example_index == 0
                ):
                    feedback_operation_logits = outputs[
                        "core_transition_feedback_operation_logits"
                    ]
                    feedback_operation_targets = primitive_transition_operation_targets(
                        row,
                        num_steps=int(feedback_operation_logits.shape[1]),
                        operation_to_id=primitive_operation_to_id,
                        device=device,
                    )
                    feedback_operation_ce, feedback_operation_metrics = (
                        primitive_transition_operation_ce_loss(
                            feedback_operation_logits,
                            feedback_operation_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_transition_feedback_operation_ce_weight)
                        * feedback_operation_ce
                    )
                    example_metrics.update(
                        {
                            f"core_transition_feedback_{key}": value
                            for key, value in feedback_operation_metrics.items()
                        }
                    )
                if (
                    float(args.core_transition_feedback_finality_ce_weight) != 0.0
                    and example_index == 0
                ):
                    feedback_finality_logits = outputs[
                        "core_transition_feedback_finality_logits"
                    ]
                    feedback_finality_targets = transition_state_finality_targets(
                        row,
                        num_depths=int(feedback_finality_logits.shape[1]),
                        device=device,
                    )
                    feedback_finality_ce, feedback_finality_metrics = (
                        transition_state_finality_bce_loss(
                            feedback_finality_logits,
                            feedback_finality_targets,
                        )
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_transition_feedback_finality_ce_weight)
                        * feedback_finality_ce
                    )
                    example_metrics.update(
                        {
                            f"core_transition_feedback_{key}": value
                            for key, value in feedback_finality_metrics.items()
                        }
                    )
                if (
                    float(args.core_transition_order_bottleneck_ce_weight) != 0.0
                    and example_index == 0
                ):
                    order_logits = outputs["core_transition_order_bottleneck_logits"]
                    order_targets = transition_phase_targets(
                        row,
                        num_steps=int(order_logits.shape[1]),
                        device=device,
                    )
                    order_ce, order_metrics = transition_phase_ce_loss(
                        order_logits,
                        order_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.core_transition_order_bottleneck_ce_weight)
                        * order_ce
                    )
                    example_metrics.update(
                        {
                            f"core_transition_order_bottleneck_{key}": value
                            for key, value in order_metrics.items()
                        }
                    )
                if (
                    float(args.transition_phase_ce_weight) != 0.0
                    and example_index == 0
                ):
                    phase_logits = outputs["transition_phase_logits"]
                    phase_targets = transition_phase_targets(
                        row,
                        num_steps=int(phase_logits.shape[1]),
                        device=device,
                    )
                    phase_ce, phase_metrics = transition_phase_ce_loss(
                        phase_logits,
                        phase_targets,
                    )
                    example_loss = (
                        example_loss + float(args.transition_phase_ce_weight) * phase_ce
                    )
                    example_metrics.update(phase_metrics)
                if (
                    float(args.transition_source_router_ce_weight) != 0.0
                    and example_index == 0
                ):
                    router_logits = outputs["transition_source_router_logits"]
                    router_targets = transition_source_router_targets(
                        row,
                        num_steps=int(router_logits.shape[1]),
                        device=device,
                    )
                    router_ce, router_metrics = transition_source_router_ce_loss(
                        router_logits,
                        router_targets,
                    )
                    example_loss = (
                        example_loss
                        + float(args.transition_source_router_ce_weight) * router_ce
                    )
                    example_metrics.update(router_metrics)
                if core_world_model_weight != 0.0:
                    core_world_model_loss = jepa_world_model_loss(
                        outputs["core_world_model_pred"],
                        outputs["core_world_model_target"],
                        outputs["core_world_model_mask"],
                        latents=outputs.get("core_world_model_latents"),
                        latent_mask=outputs.get("core_world_model_latent_mask"),
                        sigreg=getattr(model, "core_world_model_sigreg", None),
                        sigreg_weight=float(model.cfg.core_world_model_sigreg_weight),
                    )
                    example_loss = (
                        example_loss + core_world_model_weight * core_world_model_loss
                    )
                    example_metrics["core_world_model"] = core_world_model_loss.detach()
                if teacher_depth_text_logits is not None:
                    teacher_depth_kl = depth_text_logit_distillation_loss(
                        depth_text_logits,
                        teacher_depth_text_logits,
                        temperature=args.teacher_depth_kl_temperature,
                    )
                    example_loss = (
                        example_loss
                        + float(args.teacher_first_token_depth_kl_weight) * teacher_depth_kl
                    )
                    example_metrics["teacher_first_token_depth_kl"] = teacher_depth_kl.detach()
                if teacher_final_text_logits is not None and float(
                    args.teacher_final_logit_kl_weight
                ) != 0.0:
                    teacher_final_kl = depth_text_logit_distillation_loss(
                        final_text_logits,
                        teacher_final_text_logits,
                        temperature=args.teacher_depth_kl_temperature,
                    )
                    example_loss = (
                        example_loss
                        + float(args.teacher_final_logit_kl_weight) * teacher_final_kl
                    )
                    example_metrics["teacher_final_logit_kl"] = teacher_final_kl.detach()
                if (
                    float(args.choice_margin_weight) != 0.0
                ):
                    rejected_texts = choice_margin_rejected_texts(
                        row,
                        current_answer=answer,
                    )
                else:
                    rejected_texts = []
                if rejected_texts:
                    margin_losses = []
                    margin_metric_sums: dict[str, Any] = {}
                    if str(args.choice_margin_mode) == "sequence":
                        for rejected_text in rejected_texts:
                            rejected_ids = answer_token_ids(tokenizer, rejected_text)
                            if example_index >= len(rejected_ids):
                                continue
                            rejected_target_ids = input_ids.new_tensor(
                                [[int(rejected_ids[example_index])]]
                            )
                            margin_loss, margin_metrics = depth_choice_sequence_margin_loss(
                                depth_text_logits,
                                final_text_logits,
                                target_ids,
                                rejected_target_ids,
                                margin=args.choice_margin,
                                all_depth_weight=1.0,
                                final_weight=1.0,
                            )
                            margin_losses.append(margin_loss)
                            for key, value in margin_metrics.items():
                                margin_metric_sums[key] = (
                                    margin_metric_sums.get(key, value.detach() * 0.0)
                                    + value
                                )
                    elif example_index == 0:
                        chosen_first = input_ids.new_tensor(
                            [
                                answer_first_token_id(
                                    tokenizer,
                                    str(row.get("chosen") or row.get("answer")),
                                )
                            ]
                        )
                        for rejected_text in rejected_texts:
                            rejected_first = input_ids.new_tensor(
                                [answer_first_token_id(tokenizer, rejected_text)]
                            )
                            margin_loss, margin_metrics = depth_choice_margin_loss(
                                depth_text_logits,
                                final_text_logits,
                                chosen_first,
                                rejected_first,
                                margin=args.choice_margin,
                                all_depth_weight=1.0,
                                final_weight=1.0,
                            )
                            margin_losses.append(margin_loss)
                            for key, value in margin_metrics.items():
                                margin_metric_sums[key] = (
                                    margin_metric_sums.get(key, value.detach() * 0.0)
                                    + value
                                )
                    if margin_losses:
                        margin_count = float(len(margin_losses))
                        averaged_margin = margin_losses[0]
                        for extra_margin in margin_losses[1:]:
                            averaged_margin = averaged_margin + extra_margin
                        averaged_margin = averaged_margin / margin_count
                        example_loss = (
                            example_loss
                            + float(args.choice_margin_weight) * averaged_margin
                        )
                        for key, value in margin_metric_sums.items():
                            example_metrics[key] = value / margin_count
                if float(args.tail_negative_margin_weight) != 0.0:
                    tail_rejected_texts = tail_negative_rejected_texts(
                        row,
                        current_answer=answer,
                        family_filter=args.tail_negative_family_filter,
                    )
                else:
                    tail_rejected_texts = []
                if tail_rejected_texts:
                    tail_losses = []
                    tail_metric_sums: dict[str, Any] = {}
                    for rejected_text in tail_rejected_texts:
                        rejected_ids = answer_token_ids(tokenizer, rejected_text)
                        if example_index >= len(rejected_ids):
                            continue
                        rejected_target_ids = input_ids.new_tensor(
                            [[int(rejected_ids[example_index])]]
                        )
                        tail_loss, tail_metrics = tail_negative_sequence_margin_loss(
                            depth_text_logits,
                            final_text_logits,
                            target_ids,
                            rejected_target_ids,
                            margin=args.tail_negative_margin,
                        )
                        tail_losses.append(tail_loss)
                        for key, value in tail_metrics.items():
                            tail_metric_sums[key] = (
                                tail_metric_sums.get(key, value.detach() * 0.0)
                                + value
                            )
                    if tail_losses:
                        tail_count = float(len(tail_losses))
                        averaged_tail = tail_losses[0]
                        for extra_tail in tail_losses[1:]:
                            averaged_tail = averaged_tail + extra_tail
                        averaged_tail = averaged_tail / tail_count
                        example_loss = (
                            example_loss
                            + float(args.tail_negative_margin_weight) * averaged_tail
                        )
                        for key, value in tail_metric_sums.items():
                            example_metrics[key] = value / tail_count
                if float(args.subtract_tail_counterfactual_margin_weight) != 0.0:
                    subtract_tail_rejected_texts = (
                        subtract_tail_counterfactual_rejected_texts(
                            row,
                            current_answer=answer,
                            family_filter=(
                                args.subtract_tail_counterfactual_family_filter
                            ),
                        )
                    )
                else:
                    subtract_tail_rejected_texts = []
                if subtract_tail_rejected_texts:
                    subtract_tail_losses = []
                    subtract_tail_metric_sums: dict[str, Any] = {}
                    for rejected_text in subtract_tail_rejected_texts:
                        rejected_ids = answer_token_ids(tokenizer, rejected_text)
                        if example_index >= len(rejected_ids):
                            continue
                        rejected_target_ids = input_ids.new_tensor(
                            [[int(rejected_ids[example_index])]]
                        )
                        subtract_tail_loss, subtract_tail_metrics = (
                            subtract_tail_counterfactual_sequence_margin_loss(
                                depth_text_logits,
                                final_text_logits,
                                target_ids,
                                rejected_target_ids,
                                margin=args.subtract_tail_counterfactual_margin,
                            )
                        )
                        subtract_tail_losses.append(subtract_tail_loss)
                        for key, value in subtract_tail_metrics.items():
                            subtract_tail_metric_sums[key] = (
                                subtract_tail_metric_sums.get(
                                    key,
                                    value.detach() * 0.0,
                                )
                                + value
                            )
                    if subtract_tail_losses:
                        subtract_tail_count = float(len(subtract_tail_losses))
                        averaged_subtract_tail = subtract_tail_losses[0]
                        for extra_subtract_tail in subtract_tail_losses[1:]:
                            averaged_subtract_tail = (
                                averaged_subtract_tail + extra_subtract_tail
                            )
                        averaged_subtract_tail = (
                            averaged_subtract_tail / subtract_tail_count
                        )
                        example_loss = (
                            example_loss
                            + float(args.subtract_tail_counterfactual_margin_weight)
                            * averaged_subtract_tail
                        )
                        for key, value in subtract_tail_metric_sums.items():
                            example_metrics[key] = value / subtract_tail_count
                losses.append(example_loss)
                loss_weights.append(float(train_example_weights[example_index]))
                add_metrics(example_metrics)
            weighted_loss = losses[0] * float(loss_weights[0])
            for extra_loss, extra_weight in zip(losses[1:], loss_weights[1:]):
                weighted_loss = weighted_loss + extra_loss * float(extra_weight)
            loss = weighted_loss / max(1.0, float(sum(loss_weights)))
            metrics = {
                key: metric_sums[key] / float(metric_counts[key])
                for key in metric_sums
            }
            metrics["causal_prefix_examples"] = loss.detach().new_tensor(float(len(train_examples)))
            metrics["causal_prefix_later_token_weight"] = loss.detach().new_tensor(
                float(args.causal_prefix_later_token_weight)
                if bool(args.causal_prefix_supervision)
                else 1.0
            )
            metrics["causal_prefix_self_rollout_examples"] = loss.detach().new_tensor(
                float(self_rollout_examples_count)
            )
            metrics["causal_prefix_self_rollout_prefix_tokens"] = loss.detach().new_tensor(
                float(self_rollout_prefix_tokens)
            )
            metrics["causal_prefix_self_rollout_prefix_mismatch_rate"] = loss.detach().new_tensor(
                float(self_rollout_prefix_mismatch_rate)
            )
            metrics["core_world_model_weight"] = loss.detach().new_tensor(core_world_model_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, int(args.log_every)) == 0:
            metrics["core_steps"] = loss.detach().new_tensor(float(core_steps))
            pbar.set_description(
                " ".join(
                    f"{key}={float(value):.4f}"
                    for key, value in {
                        "loss": loss.detach(),
                        **metrics,
                    }.items()
                )
            )
        if save_every > 0 and (step + 1) % save_every == 0:
            checkpoint_path = out_dir / f"step_{step + 1:06d}.pt"
            save_training_checkpoint(checkpoint_path)
            print(f"saved {checkpoint_path}")
        del loss, weighted_loss, losses, loss_weights
        del metrics, outputs, donor_out, final_text_logits, example_loss
        del train_examples, train_example_weights, train_token_numeric_value_ids
        del (
            context_off_outputs,
            transition_state_off_outputs,
            bridge_off_outputs,
            role_bridge_off_outputs,
            primitive_role_value_off_outputs,
            source_binder_off_outputs,
            context_off_depth_text_logits,
            transition_state_off_depth_text_logits,
            bridge_off_depth_text_logits,
            role_bridge_off_depth_text_logits,
            role_bridge_off_final_text_logits,
            primitive_role_value_off_final_text_logits,
            primitive_role_value_off_renderer_text_logits,
            source_binder_off_renderer_text_logits,
            dense_context_final_text_logits,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    save_training_checkpoint(out_dir / "last.pt")
    print(f"saved {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
