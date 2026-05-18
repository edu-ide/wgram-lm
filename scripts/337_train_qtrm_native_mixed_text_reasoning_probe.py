#!/usr/bin/env python3
"""Donorless QTRM-native mixed text reasoning probe.

The prompt is ordinary fixed-width text, and the answer is generated as text.
This is the first L4-style scaffold: language-form input plus recursive
algorithmic reasoning through the normal LM logits path.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from qtrm_mm.mixers import FLADeltaMixer, OfficialMamba3Mixer, TorchGatedDeltaMixer


def load_native_module():
    path = Path(__file__).with_name("335_train_qtrm_native_etd_probe.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_etd_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_native = load_native_module()
NativeQTRMETDLM = _native.NativeQTRMETDLM
OP_SPECS = _native.OP_SPECS
NOOP_OP_ID = _native.NOOP_OP_ID
apply_op = _native.apply_op
active_program_len_for_step = _native.active_program_len_for_step
applicable_ablation_names = _native.applicable_ablation_names


def backend_summary(model) -> dict[str, object]:
    fla_total = 0
    fla_official = 0
    mamba3_total = 0
    mamba3_official = 0
    torch_delta = 0
    for module in model.modules():
        if isinstance(module, FLADeltaMixer):
            fla_total += 1
            fla_official += int(bool(module.is_official_backend))
        elif isinstance(module, OfficialMamba3Mixer):
            mamba3_total += 1
            mamba3_official += int(bool(module.is_official_backend))
        elif isinstance(module, TorchGatedDeltaMixer):
            torch_delta += 1
    return {
        "fla_delta_mixers": fla_total,
        "official_fla_delta_mixers": fla_official,
        "mamba3_mixers": mamba3_total,
        "official_mamba3_mixers": mamba3_official,
        "torch_delta_mixers": torch_delta,
        "all_fla_mixers_official": bool(fla_total > 0 and fla_total == fla_official),
        "all_mamba3_mixers_official": bool(mamba3_total > 0 and mamba3_total == mamba3_official),
    }


def lr_scale_for_step(
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    """Return an optimizer LR multiplier for the current training step."""
    if str(schedule) == "constant":
        return 1.0
    if str(schedule) != "linear_warmup_cosine":
        raise ValueError(f"Unsupported lr schedule: {schedule}")
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_ratio = min(1.0, max(0.0, float(min_ratio)))
    if warmup_steps > 0 and int(step) <= warmup_steps:
        return max(1, int(step)) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (int(step) - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def load_model_state_flexible(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    pos_embed_resize_strategy: str = "random_tail",
    source_chars: tuple[str, ...] = (),
    target_chars: tuple[str, ...] = (),
) -> dict[str, object]:
    """Load matching tensors and resize length/vocab embeddings when safe."""
    pos_strategy = str(pos_embed_resize_strategy)
    if pos_strategy not in {"random_tail", "repeat_last", "tail_shift"}:
        raise ValueError(f"unsupported pos_embed_resize_strategy: {pos_embed_resize_strategy}")
    target = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    resized: dict[str, dict[str, object]] = {}
    skipped: dict[str, dict[str, object]] = {}
    prefix_copy_names = {"pos_embed.weight", "token_embed.weight", "lm_head.weight"}
    vocab_remap_names = {"token_embed.weight", "lm_head.weight"}
    for name, value in state.items():
        if name not in target:
            skipped[name] = {"reason": "unexpected"}
            continue
        target_value = target[name]
        if tuple(value.shape) == tuple(target_value.shape):
            filtered[name] = value
            continue
        can_vocab_remap = (
            name in vocab_remap_names
            and source_chars
            and target_chars
            and value.ndim == target_value.ndim
            and tuple(value.shape[1:]) == tuple(target_value.shape[1:])
            and int(value.shape[0]) == len(source_chars)
            and int(target_value.shape[0]) == len(target_chars)
        )
        if can_vocab_remap:
            merged = target_value.detach().clone()
            source_index = {str(ch): idx for idx, ch in enumerate(source_chars)}
            copied_chars: list[str] = []
            composed_chars: list[str] = []
            for target_idx, ch in enumerate(target_chars):
                target_token = str(ch)
                source_idx = source_index.get(target_token)
                if source_idx is None:
                    pieces = tuple(target_token)
                    if target_token.startswith("op") and len(target_token) == 4:
                        pieces = tuple(target_token)
                    if not pieces or any(piece not in source_index for piece in pieces):
                        continue
                    piece_rows = torch.stack(
                        [
                            value[source_index[piece]].to(
                                dtype=merged.dtype,
                                device=merged.device,
                            )
                            for piece in pieces
                        ],
                        dim=0,
                    )
                    merged[target_idx] = piece_rows.mean(dim=0)
                    composed_chars.append(target_token)
                    continue
                merged[target_idx] = value[source_idx].to(
                    dtype=merged.dtype,
                    device=merged.device,
                )
                copied_chars.append(target_token)
            filtered[name] = merged
            initialized_chars = set(copied_chars) | set(composed_chars)
            resized[name] = {
                "source_shape": list(value.shape),
                "target_shape": list(target_value.shape),
                "copied_rows": len(copied_chars),
                "composed_rows": len(composed_chars),
                "resize_strategy": "token_remap",
                "composed_target_tokens": composed_chars,
                "new_target_tokens": [
                    str(ch) for ch in target_chars if str(ch) not in initialized_chars
                ],
                "dropped_source_tokens": [
                    str(ch) for ch in source_chars if str(ch) not in set(target_chars)
                ],
            }
            continue
        can_prefix_copy = (
            name in prefix_copy_names
            and value.ndim == target_value.ndim
            and tuple(value.shape[1:]) == tuple(target_value.shape[1:])
            and int(value.shape[0]) > 0
            and int(target_value.shape[0]) > 0
        )
        if can_prefix_copy:
            merged = target_value.detach().clone()
            rows = min(int(value.shape[0]), int(target_value.shape[0]))
            merged[:rows] = value[:rows].to(dtype=merged.dtype, device=merged.device)
            filled_rows = 0
            if (
                name == "pos_embed.weight"
                and int(target_value.shape[0]) > int(value.shape[0])
                and rows == int(value.shape[0])
            ):
                filled_rows = int(target_value.shape[0]) - int(value.shape[0])
                if pos_strategy == "repeat_last":
                    merged[rows:] = value[-1:].to(dtype=merged.dtype, device=merged.device)
                elif pos_strategy == "tail_shift":
                    start = max(0, int(value.shape[0]) - int(filled_rows))
                    tail = value[start:].to(dtype=merged.dtype, device=merged.device)
                    if int(tail.shape[0]) < int(filled_rows):
                        repeats = math.ceil(float(filled_rows) / max(1, int(tail.shape[0])))
                        tail = tail.repeat((repeats, *([1] * (tail.ndim - 1))))
                    merged[rows:] = tail[:filled_rows]
            filtered[name] = merged
            resized[name] = {
                "source_shape": list(value.shape),
                "target_shape": list(target_value.shape),
                "copied_rows": rows,
            }
            if name == "pos_embed.weight":
                resized[name]["resize_strategy"] = pos_strategy
                resized[name]["filled_rows"] = filled_rows
            continue
        skipped[name] = {
            "reason": "shape_mismatch",
            "source_shape": list(value.shape),
            "target_shape": list(target_value.shape),
        }
    incompatible = model.load_state_dict(filtered, strict=False)
    return {
        "loaded_tensors": len(filtered),
        "resized_tensors": resized,
        "skipped_tensors": skipped,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


@dataclass(frozen=True)
class TextReasoningCase:
    case_id: str
    start: int
    op_ids: tuple[int, ...]
    answer: int
    family: str = "modchain"


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]
    char_to_id: dict[str, int]
    op_role_tokens: bool = False

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        *,
        mode: str = "char",
        number_max_value: int = 99,
        op_role_tokens: bool = False,
    ) -> "CharTokenizer":
        tokens: set[str] = set()
        for text in texts:
            tokens.update(
                tokenize_text(
                    text,
                    mode=mode,
                    op_role_tokens=bool(op_role_tokens),
                )
            )
        if str(mode) == "number" and int(number_max_value) >= 0:
            tokens.update(f"{value:02d}" for value in range(int(number_max_value) + 1))
        chars = tuple(sorted(tokens))
        return cls(
            chars=chars,
            char_to_id={ch: index for index, ch in enumerate(chars)},
            op_role_tokens=bool(op_role_tokens),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [
            self.char_to_id[token]
            for token in tokenize_text(
                text,
                mode=self.mode,
                op_role_tokens=bool(self.op_role_tokens),
            )
        ]

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.chars[int(token_id)]
            if token.startswith("op") and len(token) == 4 and token[2:].isdigit():
                pieces.append(token[2:])
            else:
                pieces.append(token)
        return "".join(pieces)

    @property
    def mode(self) -> str:
        return "number" if any(len(token) == 2 and token.isdigit() for token in self.chars) else "char"


def tokenize_text(
    text: str,
    *,
    mode: str = "char",
    op_role_tokens: bool = False,
) -> list[str]:
    if str(mode) == "char":
        return list(text)
    if str(mode) == "number":
        tokens: list[str] = []
        index = 0
        in_ops = False
        while index < len(text):
            if bool(op_role_tokens) and text.startswith("ops", index):
                tokens.extend(("o", "p", "s"))
                index += len("ops")
                in_ops = True
                continue
            if bool(op_role_tokens) and in_ops and (
                text.startswith("answer", index) or text.startswith("state", index)
            ):
                in_ops = False
            if index + 1 < len(text) and text[index : index + 2].isdigit():
                raw_token = text[index : index + 2]
                tokens.append(f"op{raw_token}" if bool(op_role_tokens) and in_ops else raw_token)
                index += 2
            else:
                tokens.append(text[index])
                index += 1
        return tokens
    raise ValueError(f"unsupported tokenizer mode: {mode}")


def fmt2(value: int) -> str:
    return f"{int(value):02d}"


def value_token_ids_for_tokenizer(
    tokenizer: CharTokenizer,
    *,
    modulus: int,
) -> tuple[int, ...]:
    missing: list[str] = []
    token_ids: list[int] = []
    for value in range(int(modulus)):
        token = fmt2(value)
        token_id = tokenizer.char_to_id.get(token)
        if token_id is None:
            missing.append(token)
        else:
            token_ids.append(int(token_id))
    if missing:
        raise ValueError(f"number tokenizer is missing value tokens: {missing[:8]}")
    return tuple(token_ids)


SUPPORTED_FAMILIES = ("modchain", "revchain", "checksum")
FAMILY_SEED_OFFSETS = {
    "modchain": 101,
    "revchain": 211,
    "checksum": 307,
}
FAMILY_WIDTH = max(len(item) for item in SUPPORTED_FAMILIES)


def parse_families(value: str) -> tuple[str, ...]:
    families = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not families:
        raise ValueError("at least one task family is required")
    unknown = sorted(set(families) - set(SUPPORTED_FAMILIES))
    if unknown:
        raise ValueError(f"unsupported task families: {unknown}")
    return families


def compute_answer(
    *,
    start: int,
    op_ids: tuple[int, ...],
    family: str,
    modulus: int,
) -> int:
    family = str(family)
    value = int(start)
    if family == "modchain":
        for op_id in op_ids:
            value = apply_op(value, int(op_id), int(modulus))
        return int(value)
    if family == "revchain":
        for op_id in reversed(op_ids):
            value = apply_op(value, int(op_id), int(modulus))
        return int(value)
    if family == "checksum":
        for op_id in op_ids:
            value = (value + int(op_id)) % int(modulus)
        return int(value)
    raise ValueError(f"unsupported task family: {family}")


def parse_op_ids(value: str) -> tuple[int, ...]:
    raw = str(value).replace(",", " ").split()
    if not raw:
        return ()
    parsed = tuple(int(item) for item in raw)
    invalid = [item for item in parsed if item <= 0 or item >= len(OP_SPECS)]
    if invalid:
        raise ValueError(f"unsupported op ids: {invalid}")
    return parsed


def parse_residue_moduli(value: str) -> tuple[int, ...]:
    raw = str(value).replace(",", " ").split()
    if not raw:
        return ()
    parsed = tuple(int(item) for item in raw)
    invalid = [item for item in parsed if item <= 1 or item > 9]
    if invalid:
        raise ValueError(f"unsupported residue moduli: {invalid}")
    return parsed


def parse_preference_deltas(value: str) -> tuple[int, ...]:
    raw = str(value).replace(",", " ").split()
    if not raw:
        return ()
    parsed = tuple(int(item) for item in raw)
    invalid = [item for item in parsed if item <= 0]
    if invalid:
        raise ValueError(f"unsupported preference deltas: {invalid}")
    return parsed


def parse_positions(value: str) -> tuple[int, ...]:
    raw = str(value).replace(",", " ").split()
    if not raw:
        return ()
    parsed = tuple(int(item) for item in raw)
    invalid = [item for item in parsed if item <= 0]
    if invalid:
        raise ValueError(f"unsupported 1-indexed positions: {invalid}")
    return parsed


def sample_op_id(
    rng: random.Random,
    *,
    hard_op_ids: tuple[int, ...] = (),
    hard_op_probability: float = 0.0,
) -> int:
    if hard_op_ids and rng.random() < float(hard_op_probability):
        return int(rng.choice(hard_op_ids))
    return int(rng.randrange(1, len(OP_SPECS)))


def build_cases(
    *,
    count: int,
    seed: int,
    program_len: int,
    modulus: int,
    families: tuple[str, ...] = ("modchain",),
    hard_op_ids: tuple[int, ...] = (),
    hard_op_probability: float = 0.0,
    hard_op_positions: tuple[int, ...] = (),
) -> list[TextReasoningCase]:
    rng = random.Random(int(seed))
    families = parse_families(",".join(families))
    hard_op_ids = parse_op_ids(" ".join(str(item) for item in hard_op_ids))
    hard_op_positions = parse_positions(" ".join(str(item) for item in hard_op_positions))
    rows: list[TextReasoningCase] = []
    for index in range(int(count)):
        family = families[index % len(families)]
        start = rng.randrange(int(modulus))
        op_ids = tuple(
            sample_op_id(
                rng,
                hard_op_ids=(
                    hard_op_ids
                    if not hard_op_positions or position + 1 in hard_op_positions
                    else ()
                ),
                hard_op_probability=float(hard_op_probability),
            )
            for position in range(int(program_len))
        )
        value = compute_answer(
            start=start,
            op_ids=op_ids,
            family=family,
            modulus=int(modulus),
        )
        rows.append(
            TextReasoningCase(
                case_id=f"text-reason-{seed}-{index:06d}",
                start=start,
                op_ids=op_ids,
                answer=value,
                family=family,
            )
        )
    return rows


def build_family_order_invariant_eval_cases(
    *,
    count: int,
    seed: int,
    program_len: int,
    modulus: int,
    families: tuple[str, ...],
) -> list[TextReasoningCase]:
    """Build eval cases whose per-family samples do not depend on family order.

    `build_cases` intentionally lets duplicate families weight the training
    distribution. For evaluation, changing `"modchain,revchain,checksum"` to
    `"checksum,modchain,revchain"` should not silently change each family's
    held-out examples. This helper assigns each unique family a stable seed and
    then interleaves the generated rows.
    """
    unique_families = tuple(dict.fromkeys(parse_families(",".join(families))))
    if len(unique_families) <= 1:
        return build_cases(
            count=int(count),
            seed=int(seed),
            program_len=int(program_len),
            modulus=int(modulus),
            families=unique_families,
        )
    base = int(count) // len(unique_families)
    remainder = int(count) % len(unique_families)
    rows_by_family: list[list[TextReasoningCase]] = []
    for family_index, family in enumerate(unique_families):
        family_count = base + (1 if family_index < remainder else 0)
        family_seed = int(seed) * 1009 + int(FAMILY_SEED_OFFSETS[str(family)])
        rows_by_family.append(
            build_cases(
                count=family_count,
                seed=family_seed,
                program_len=int(program_len),
                modulus=int(modulus),
                families=(str(family),),
            )
        )
    interleaved: list[TextReasoningCase] = []
    max_count = max(len(rows) for rows in rows_by_family)
    for row_index in range(max_count):
        for rows in rows_by_family:
            if row_index < len(rows):
                interleaved.append(rows[row_index])
    return interleaved[: int(count)]


def causal_prefix_op_ids(
    case: TextReasoningCase,
    *,
    prefix_len: int,
) -> tuple[tuple[int, ...], int]:
    """Return op ids with only the first prefix operations in causal order.

    `modchain` and `checksum` execute left-to-right. `revchain` executes
    right-to-left, so its causal prefix is a suffix in prompt order.
    """
    non_noop_positions = [
        (index, int(op_id))
        for index, op_id in enumerate(case.op_ids)
        if int(op_id) != int(NOOP_OP_ID)
    ]
    prefix = max(0, min(int(prefix_len), len(non_noop_positions)))
    if str(case.family) == "revchain":
        causal_positions = list(reversed(non_noop_positions))
    else:
        causal_positions = non_noop_positions
    selected = causal_positions[:prefix]
    op_ids = [int(NOOP_OP_ID) for _ in case.op_ids]
    for index, op_id in selected:
        op_ids[int(index)] = int(op_id)
    return tuple(op_ids), int(prefix)


def causal_op_id_at_depth(case: TextReasoningCase, *, depth_index: int) -> int:
    """Return the operation id consumed at a recurrent depth in family order."""
    non_noop_positions = [
        (index, int(op_id))
        for index, op_id in enumerate(case.op_ids)
        if int(op_id) != int(NOOP_OP_ID)
    ]
    if str(case.family) == "revchain":
        causal_positions = list(reversed(non_noop_positions))
    else:
        causal_positions = non_noop_positions
    index = int(depth_index)
    if index < 0 or index >= len(causal_positions):
        return int(NOOP_OP_ID)
    return int(causal_positions[index][1])


def causal_op_position_at_depth(case: TextReasoningCase, *, depth_index: int) -> int:
    """Return 1-indexed prompt op position consumed at a recurrent depth.

    Class 0 means no causal operation is available at that depth. For revchain,
    the consumed positions run right-to-left even though the prompt text remains
    left-to-right.
    """
    non_noop_positions = [
        (index, int(op_id))
        for index, op_id in enumerate(case.op_ids)
        if int(op_id) != int(NOOP_OP_ID)
    ]
    if str(case.family) == "revchain":
        causal_positions = list(reversed(non_noop_positions))
    else:
        causal_positions = non_noop_positions
    index = int(depth_index)
    if index < 0 or index >= len(causal_positions):
        return 0
    return int(causal_positions[index][0]) + 1


def case_with_active_program_len(
    case: TextReasoningCase,
    *,
    active_len: int,
    modulus: int,
) -> TextReasoningCase:
    op_ids, active = causal_prefix_op_ids(case, prefix_len=int(active_len))
    value = compute_answer(
        start=int(case.start),
        op_ids=op_ids,
        family=str(case.family),
        modulus=int(modulus),
    )
    return TextReasoningCase(
        case_id=f"{case.case_id}-active{active}",
        start=int(case.start),
        op_ids=op_ids,
        answer=int(value),
        family=str(case.family),
    )


def case_with_causal_prefix_len(
    case: TextReasoningCase,
    *,
    prefix_len: int,
    modulus: int,
) -> TextReasoningCase:
    """Keep the first `prefix_len` operations in the family's causal order."""
    op_ids, prefix = causal_prefix_op_ids(case, prefix_len=int(prefix_len))
    value = compute_answer(
        start=int(case.start),
        op_ids=op_ids,
        family=str(case.family),
        modulus=int(modulus),
    )
    return TextReasoningCase(
        case_id=f"{case.case_id}-causalprefix{prefix}",
        start=int(case.start),
        op_ids=op_ids,
        answer=int(value),
        family=str(case.family),
    )


def case_prompt(
    case: TextReasoningCase,
    *,
    include_family_tag: bool = False,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> str:
    ops = " ".join(fmt2(op_id) for op_id in case.op_ids)
    position = str(state_anchor_position)
    if position not in {"before_answer", "after_answer"}:
        raise ValueError("state_anchor_position must be before_answer or after_answer")
    if bool(state_anchor) and position == "before_answer":
        answer_prefix = "state answer "
    elif bool(state_anchor):
        answer_prefix = "answer state "
    else:
        answer_prefix = "answer "
    if bool(include_family_tag):
        family = str(case.family).ljust(FAMILY_WIDTH)
        return f"task {family} start {fmt2(case.start)} ops {ops} {answer_prefix}"
    return f"start {fmt2(case.start)} ops {ops} {answer_prefix}"


def case_prompt_with_answer_label(
    case: TextReasoningCase,
    *,
    answer_label: str,
    include_family_tag: bool = False,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> str:
    label = str(answer_label)
    if len(label) != len("answer "):
        raise ValueError("answer_label must preserve the fixed-width prompt length")
    prompt = case_prompt(
        case,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    if bool(state_anchor) and str(state_anchor_position) == "after_answer":
        suffix = "answer state "
        if not prompt.endswith(suffix):
            raise ValueError("expected prompt to end with the canonical answer-state label")
        return prompt[: -len(suffix)] + label + "state "
    if not prompt.endswith("answer "):
        raise ValueError("expected prompt to end with the canonical answer label")
    return prompt[: -len("answer ")] + label


def case_answer(case: TextReasoningCase) -> str:
    return f"{fmt2(case.answer)}\n"


def answer_format_valid(text: str) -> bool:
    return len(text) == 3 and text[:2].isdigit() and text[2] == "\n"


def answer_value(text: str) -> int | None:
    if not answer_format_valid(text):
        return None
    return int(text[:2])


def case_full_text(
    case: TextReasoningCase,
    *,
    include_family_tag: bool = False,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> str:
    return (
        case_prompt(
            case,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        + case_answer(case)
    )


def zero_ops_case(case: TextReasoningCase, *, modulus: int) -> TextReasoningCase:
    return case_with_active_program_len(case, active_len=0, modulus=int(modulus))


def effective_program_len(case: TextReasoningCase) -> int:
    return int(sum(1 for op_id in case.op_ids if int(op_id) != int(NOOP_OP_ID)))


def _bump_generation_count(
    counts: dict[str, dict[str, int]],
    key: str,
    *,
    exact: bool,
    format_valid: bool,
) -> None:
    row = counts.setdefault(str(key), {"correct": 0, "format_valid": 0, "total": 0})
    row["correct"] += int(exact)
    row["format_valid"] += int(format_valid)
    row["total"] += 1


def _summarize_generation_counts(
    counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "correct": row["correct"],
            "format_valid": row["format_valid"],
            "total": row["total"],
            "generation_exact": float(row["correct"] / max(1, row["total"])),
            "generation_format_valid": float(row["format_valid"] / max(1, row["total"])),
        }
        for key, row in sorted(counts.items())
    }


def generation_operation_breakdown(
    cases: list[TextReasoningCase],
    predictions: list[str],
    *,
    modulus: int,
) -> dict[str, object]:
    by_last_op: dict[str, dict[str, int]] = {}
    by_position_op: dict[str, dict[str, int]] = {}
    by_error_delta: dict[str, dict[str, int]] = {}
    by_family_last_op: dict[str, dict[str, dict[str, int]]] = {}
    by_family_position_op: dict[str, dict[str, dict[str, int]]] = {}
    by_family_error_delta: dict[str, dict[str, dict[str, int]]] = {}
    for case, pred in zip(cases, predictions):
        gold = case_answer(case)
        exact = pred == gold
        pred_format_valid = answer_format_valid(pred)
        active_op_ids = [
            int(op_id) for op_id in case.op_ids if int(op_id) != int(NOOP_OP_ID)
        ]
        active_len = len(active_op_ids)
        family = str(case.family)
        family_last_op = by_family_last_op.setdefault(family, {})
        family_position_op = by_family_position_op.setdefault(family, {})
        family_error_delta = by_family_error_delta.setdefault(family, {})
        if active_len > 0:
            last_op_key = fmt2(active_op_ids[-1])
            _bump_generation_count(
                by_last_op,
                last_op_key,
                exact=exact,
                format_valid=pred_format_valid,
            )
            _bump_generation_count(
                family_last_op,
                last_op_key,
                exact=exact,
                format_valid=pred_format_valid,
            )
        for position, op_id in enumerate(active_op_ids, start=1):
            position_op_key = f"{position}:{fmt2(op_id)}"
            _bump_generation_count(
                by_position_op,
                position_op_key,
                exact=exact,
                format_valid=pred_format_valid,
            )
            _bump_generation_count(
                family_position_op,
                position_op_key,
                exact=exact,
                format_valid=pred_format_valid,
            )
        pred_value = answer_value(pred)
        if pred_value is None:
            delta_key = "invalid"
        else:
            gold_value = int(case.answer)
            delta_key = fmt2((int(pred_value) - gold_value) % int(modulus))
        _bump_generation_count(
            by_error_delta,
            delta_key,
            exact=exact,
            format_valid=pred_format_valid,
        )
        _bump_generation_count(
            family_error_delta,
            delta_key,
            exact=exact,
            format_valid=pred_format_valid,
        )
    return {
        "by_last_op": _summarize_generation_counts(by_last_op),
        "by_position_op": _summarize_generation_counts(by_position_op),
        "by_error_delta": _summarize_generation_counts(by_error_delta),
        "by_family": {
            family: {
                "by_last_op": _summarize_generation_counts(
                    by_family_last_op.get(family, {})
                ),
                "by_position_op": _summarize_generation_counts(
                    by_family_position_op.get(family, {})
                ),
                "by_error_delta": _summarize_generation_counts(
                    by_family_error_delta.get(family, {})
                ),
            }
            for family in sorted(
                set(by_family_last_op)
                | set(by_family_position_op)
                | set(by_family_error_delta)
            )
        },
    }


def apply_eval_active_len_cycle(
    cases: list[TextReasoningCase],
    *,
    modulus: int,
    min_active_len: int = 0,
    max_active_len: int | None = None,
    balance_by_family: bool = True,
) -> list[TextReasoningCase]:
    cycled: list[TextReasoningCase] = []
    family_offsets: dict[str, int] = {}
    for index, case in enumerate(cases):
        if bool(balance_by_family):
            cycle_index = family_offsets.get(case.family, 0)
            family_offsets[case.family] = cycle_index + 1
        else:
            cycle_index = index
        active_len = active_len_cycle_value(
            index=cycle_index,
            program_len=len(case.op_ids),
            min_active_len=int(min_active_len),
            max_active_len=max_active_len,
        )
        cycled.append(
            case_with_active_program_len(case, active_len=active_len, modulus=int(modulus))
        )
    return cycled


def active_len_cycle_value(
    *,
    index: int,
    program_len: int,
    min_active_len: int = 0,
    max_active_len: int | None = None,
) -> int:
    program = max(0, int(program_len))
    lo = max(0, min(int(min_active_len), program))
    hi = program if max_active_len is None else max(0, min(int(max_active_len), program))
    if hi < lo:
        hi = lo
    width = hi - lo + 1
    return int(lo + (int(index) % max(1, width)))


def apply_active_len_batch_cycle(
    cases: list[TextReasoningCase],
    *,
    step: int,
    modulus: int,
    min_active_len: int = 0,
    max_active_len: int | None = None,
) -> list[TextReasoningCase]:
    mixed: list[TextReasoningCase] = []
    for index, case in enumerate(cases):
        active_len = active_len_cycle_value(
            index=int(step) - 1 + index,
            program_len=len(case.op_ids),
            min_active_len=int(min_active_len),
            max_active_len=max_active_len,
        )
        mixed.append(
            case_with_active_program_len(case, active_len=active_len, modulus=int(modulus))
        )
    return mixed


def cases_to_batch(
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    prompt_cases: list[TextReasoningCase] | None = None,
    include_family_tag: bool = False,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    prompt_cases = cases if prompt_cases is None else prompt_cases
    texts = [
        case_prompt(
            prompt_case,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        + case_answer(answer_case)
        for prompt_case, answer_case in zip(prompt_cases, cases)
    ]
    prompt_len = len(
        tokenizer.encode(
            case_prompt(
                prompt_cases[0],
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
        )
    )
    answer_len = len(tokenizer.encode(case_answer(cases[0])))
    encoded = [tokenizer.encode(text) for text in texts]
    full = torch.tensor(encoded, dtype=torch.long, device=device)
    return full[:, :-1], full[:, 1:], prompt_len, answer_len


def include_family_tag_for_args(args: argparse.Namespace) -> bool:
    families = tuple(dict.fromkeys(train_families_for_args(args) + eval_families_for_args(args)))
    return bool(args.include_family_tag) or len(families) > 1


def state_anchor_for_args(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "prompt_state_anchor", False))


def state_anchor_position_for_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "prompt_state_anchor_position", "before_answer"))


def train_families_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    return parse_families(str(args.task_families))


def eval_families_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    value = str(args.eval_task_families).strip()
    return parse_families(value) if value else train_families_for_args(args)


def stablemax_log_probs(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    values = logits.to(torch.float64)
    transformed = torch.where(
        values < 0,
        1.0 / (1.0 - values + 1e-30),
        values + 1.0,
    )
    return torch.log(transformed / transformed.sum(dim=dim, keepdim=True)).to(logits.dtype)


def answer_text_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    loss_type: str = "cross_entropy",
) -> torch.Tensor:
    return answer_case_losses(
        logits,
        targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
        loss_type=loss_type,
    ).mean()


def answer_case_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    loss_type: str = "cross_entropy",
) -> torch.Tensor:
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    selected_logits = logits[:, start:end, :]
    selected_targets = targets[:, start:end]
    if str(loss_type) == "stablemax_cross_entropy":
        log_probs = stablemax_log_probs(selected_logits, dim=-1)
        selected = log_probs.gather(dim=-1, index=selected_targets.unsqueeze(-1)).squeeze(-1)
        return -selected.mean(dim=1)
    token_losses = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        selected_targets.reshape(-1),
        reduction="none",
    )
    return token_losses.reshape(selected_logits.shape[0], -1).mean(dim=1)


def family_dro_answer_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    prompt_len: int,
    answer_len: int,
    loss_type: str = "cross_entropy",
    temperature: float = 0.0,
) -> torch.Tensor:
    per_case = answer_case_losses(
        logits,
        targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
        loss_type=loss_type,
    )
    return family_dro_from_case_losses(per_case, cases, temperature=temperature)


def family_dro_from_case_losses(
    per_case: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    temperature: float = 0.0,
) -> torch.Tensor:
    family_losses: list[torch.Tensor] = []
    for family in sorted({str(case.family) for case in cases}):
        indexes = [
            index for index, case in enumerate(cases) if str(case.family) == family
        ]
        if indexes:
            index_tensor = torch.tensor(indexes, dtype=torch.long, device=per_case.device)
            family_losses.append(per_case.index_select(0, index_tensor).mean())
    if not family_losses:
        return torch.zeros((), device=per_case.device)
    stacked = torch.stack(family_losses)
    if float(temperature) <= 0.0:
        return stacked.max()
    return float(temperature) * torch.logsumexp(stacked / float(temperature), dim=0)


def residue_auxiliary_loss(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    residue_moduli: tuple[int, ...],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for residue_modulus in residue_moduli:
        label = f"answer{int(residue_modulus) % 10}"
        prompt_cases = cases
        answer_cases = [
            TextReasoningCase(
                case_id=case.case_id,
                start=int(case.start),
                op_ids=case.op_ids,
                answer=int(case.answer) % int(residue_modulus),
                family=str(case.family),
            )
            for case in cases
        ]
        texts = [
            case_prompt_with_answer_label(
                prompt_case,
                answer_label=label,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            + case_answer(answer_case)
            for prompt_case, answer_case in zip(prompt_cases, answer_cases)
        ]
        prompt_len = len(
            tokenizer.encode(
                case_prompt_with_answer_label(
                    prompt_cases[0],
                    answer_label=label,
                    include_family_tag=include_family_tag,
                    state_anchor=state_anchor,
                    state_anchor_position=state_anchor_position,
                )
            )
        )
        answer_len = len(tokenizer.encode(case_answer(answer_cases[0])))
        full = torch.tensor(
            [tokenizer.encode(text) for text in texts],
            dtype=torch.long,
            device=device,
        )
        input_ids = full[:, :-1]
        targets = full[:, 1:]
        logits = model(input_ids, think_steps=int(think_steps))
        losses.append(
            answer_text_loss(
                logits,
                targets,
                prompt_len=prompt_len,
                answer_len=answer_len,
            )
        )
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def answer_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    margin: float,
) -> torch.Tensor:
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    selected_logits = logits[:, start:end, :]
    selected_targets = targets[:, start:end]
    target_logits = selected_logits.gather(
        dim=-1,
        index=selected_targets.unsqueeze(-1),
    ).squeeze(-1)
    competitor_logits = selected_logits.masked_fill(
        F.one_hot(selected_targets, num_classes=selected_logits.shape[-1]).bool(),
        float("-inf"),
    ).amax(dim=-1)
    return F.relu(float(margin) - (target_logits - competitor_logits)).mean()


def answer_sequence_logprob(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
) -> torch.Tensor:
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    selected_logits = logits[:, start:end, :]
    selected_targets = targets[:, start:end]
    token_logprobs = F.log_softmax(selected_logits, dim=-1).gather(
        dim=-1,
        index=selected_targets.unsqueeze(-1),
    ).squeeze(-1)
    return token_logprobs.sum(dim=1)


def sequence_preference_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    prompt_len: int,
    answer_len: int,
    think_steps: int,
    modulus: int,
    deltas: tuple[int, ...],
    margin: float,
) -> torch.Tensor:
    chosen_logprob = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    losses: list[torch.Tensor] = []
    for delta in deltas:
        rejected_cases = [
            TextReasoningCase(
                case_id=case.case_id,
                start=int(case.start),
                op_ids=case.op_ids,
                answer=(int(case.answer) + int(delta)) % int(modulus),
                family=str(case.family),
            )
            for case in cases
        ]
        rejected_x, rejected_y, rejected_prompt_len, rejected_answer_len = cases_to_batch(
            rejected_cases,
            tokenizer=tokenizer,
            device=device,
            prompt_cases=cases,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        rejected_logits = model(rejected_x, think_steps=int(think_steps))
        rejected_logprob = answer_sequence_logprob(
            rejected_logits,
            rejected_y,
            prompt_len=rejected_prompt_len,
            answer_len=rejected_answer_len,
        )
        losses.append(
            F.relu(float(margin) - (chosen_logprob - rejected_logprob)).mean()
        )
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def operation_counterfactual_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    prompt_len: int,
    answer_len: int,
    think_steps: int,
    modulus: int,
    margin: float,
    max_cases: int,
    active_len_min: int = 1,
    active_len_max: int = -1,
) -> torch.Tensor:
    candidate_indexes = [
        index
        for index, case in enumerate(cases)
        if effective_program_len(case) >= int(active_len_min)
        and (
            int(active_len_max) < 0
            or effective_program_len(case) <= int(active_len_max)
        )
    ]
    if int(max_cases) > 0:
        candidate_indexes = candidate_indexes[: int(max_cases)]
    if not candidate_indexes:
        return torch.zeros((), device=device)
    chosen_logprob_all = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    selected_cases = [cases[index] for index in candidate_indexes]
    counterfactual_prompt_cases = [
        zero_ops_case(case, modulus=int(modulus)) for case in selected_cases
    ]
    counterfactual_x, counterfactual_y, counterfactual_prompt_len, counterfactual_answer_len = (
        cases_to_batch(
            selected_cases,
            tokenizer=tokenizer,
            device=device,
            prompt_cases=counterfactual_prompt_cases,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
    )
    counterfactual_logits = model(counterfactual_x, think_steps=int(think_steps))
    counterfactual_logprob = answer_sequence_logprob(
        counterfactual_logits,
        counterfactual_y,
        prompt_len=counterfactual_prompt_len,
        answer_len=counterfactual_answer_len,
    )
    index_tensor = torch.tensor(candidate_indexes, dtype=torch.long, device=device)
    chosen_logprob = chosen_logprob_all.index_select(0, index_tensor)
    return F.relu(float(margin) - (chosen_logprob - counterfactual_logprob)).mean()


def operation_counterfactual_schedule_enabled(args: argparse.Namespace, step: int) -> bool:
    if float(args.operation_counterfactual_loss_weight) <= 0.0:
        return False
    if int(step) <= int(args.operation_counterfactual_warmup_steps):
        return False
    end_step = int(getattr(args, "operation_counterfactual_end_step", -1))
    if end_step >= 0 and int(step) > end_step:
        return False
    every = int(args.operation_counterfactual_every)
    return every <= 1 or int(step) % every == 0


def depth_counterfactual_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    counterfactual_think_steps: int,
    margin: float,
) -> torch.Tensor:
    chosen_logprob = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    counterfactual_logits = model(
        input_ids,
        think_steps=int(counterfactual_think_steps),
    )
    counterfactual_logprob = answer_sequence_logprob(
        counterfactual_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    return F.relu(float(margin) - (chosen_logprob - counterfactual_logprob)).mean()


def state_reset_counterfactual_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    think_steps: int,
    margin: float,
) -> torch.Tensor:
    chosen_logprob = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    counterfactual_logits = model(
        input_ids,
        think_steps=int(think_steps),
        state_reset_each_step=True,
    )
    counterfactual_logprob = answer_sequence_logprob(
        counterfactual_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    return F.relu(float(margin) - (chosen_logprob - counterfactual_logprob)).mean()


def z_l_counterfactual_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    think_steps: int,
    margin: float,
) -> torch.Tensor:
    chosen_logprob = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    counterfactual_logits = model(
        input_ids,
        think_steps=int(think_steps),
        z_l_zero=True,
    )
    counterfactual_logprob = answer_sequence_logprob(
        counterfactual_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    return F.relu(float(margin) - (chosen_logprob - counterfactual_logprob)).mean()


def fast_slow_latent_counterfactual_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    think_steps: int,
    z_l_margin: float,
    z_h_margin: float,
    z_l_weight: float,
    z_h_weight: float,
) -> torch.Tensor:
    """Force both fast z_L and slow z_H to be causally useful for LM logits."""
    chosen_logprob = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    losses: list[torch.Tensor] = []
    if float(z_l_weight) > 0.0:
        z_l_logits = model(
            input_ids,
            think_steps=int(think_steps),
            z_l_zero=True,
        )
        z_l_logprob = answer_sequence_logprob(
            z_l_logits,
            chosen_targets,
            prompt_len=prompt_len,
            answer_len=answer_len,
        )
        losses.append(
            float(z_l_weight)
            * F.relu(float(z_l_margin) - (chosen_logprob - z_l_logprob)).mean()
        )
    if float(z_h_weight) > 0.0:
        z_h_logits = model(
            input_ids,
            think_steps=int(think_steps),
            z_h_zero=True,
        )
        z_h_logprob = answer_sequence_logprob(
            z_h_logits,
            chosen_targets,
            prompt_len=prompt_len,
            answer_len=answer_len,
        )
        losses.append(
            float(z_h_weight)
            * F.relu(float(z_h_margin) - (chosen_logprob - z_h_logprob)).mean()
        )
    if not losses:
        return chosen_logits.sum() * 0.0
    normalizer = max(float(z_l_weight) + float(z_h_weight), 1e-6)
    return torch.stack(losses).sum() / normalizer


def answer_space_ranking_loss(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    modulus: int,
    max_cases: int,
    temperature: float,
) -> torch.Tensor:
    selected_cases = cases[: int(max_cases)] if int(max_cases) > 0 else cases
    if not selected_cases:
        return torch.zeros((), device=device)
    candidate_cases: list[TextReasoningCase] = []
    prompt_cases: list[TextReasoningCase] = []
    labels: list[int] = []
    modulus = int(modulus)
    for case in selected_cases:
        labels.append(int(case.answer) % modulus)
        for answer in range(modulus):
            candidate_cases.append(
                TextReasoningCase(
                    case_id=case.case_id,
                    start=int(case.start),
                    op_ids=case.op_ids,
                    answer=int(answer),
                    family=str(case.family),
                )
            )
            prompt_cases.append(case)
    candidate_x, candidate_y, candidate_prompt_len, candidate_answer_len = cases_to_batch(
        candidate_cases,
        tokenizer=tokenizer,
        device=device,
        prompt_cases=prompt_cases,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    candidate_logits = model(candidate_x, think_steps=int(think_steps))
    sequence_scores = answer_sequence_logprob(
        candidate_logits,
        candidate_y,
        prompt_len=candidate_prompt_len,
        answer_len=candidate_answer_len,
    ).reshape(len(selected_cases), modulus)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    scale = max(float(temperature), 1e-6)
    return F.cross_entropy(sequence_scores / scale, label_tensor)


@torch.no_grad()
def answer_space_argmax_metrics(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    prompt_cases: list[TextReasoningCase],
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    modulus: int,
    max_candidate_batch: int,
) -> dict[str, object]:
    if not cases:
        return {
            "answer_space_argmax_exact": 0.0,
            "answer_space_gold_mean_rank": 0.0,
            "answer_space_gold_top3": 0.0,
            "answer_space_gold_top5": 0.0,
        }
    modulus = int(modulus)
    candidate_cases: list[TextReasoningCase] = []
    candidate_prompt_cases: list[TextReasoningCase] = []
    labels: list[int] = []
    for case, prompt_case in zip(cases, prompt_cases):
        labels.append(int(case.answer) % modulus)
        for answer in range(modulus):
            candidate_cases.append(
                TextReasoningCase(
                    case_id=case.case_id,
                    start=int(case.start),
                    op_ids=case.op_ids,
                    answer=int(answer),
                    family=str(case.family),
                )
            )
            candidate_prompt_cases.append(prompt_case)
    scores: list[torch.Tensor] = []
    chunk_size = max(1, int(max_candidate_batch))
    for start in range(0, len(candidate_cases), chunk_size):
        end = min(start + chunk_size, len(candidate_cases))
        candidate_x, candidate_y, candidate_prompt_len, candidate_answer_len = cases_to_batch(
            candidate_cases[start:end],
            tokenizer=tokenizer,
            device=device,
            prompt_cases=candidate_prompt_cases[start:end],
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        candidate_logits = model(candidate_x, think_steps=int(think_steps))
        scores.append(
            answer_sequence_logprob(
                candidate_logits,
                candidate_y,
                prompt_len=candidate_prompt_len,
                answer_len=candidate_answer_len,
            )
            .detach()
            .cpu()
        )
    score_tensor = torch.cat(scores, dim=0).reshape(len(cases), modulus)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    predicted = score_tensor.argmax(dim=1)
    gold_scores = score_tensor.gather(dim=1, index=label_tensor.unsqueeze(1)).squeeze(1)
    ranks = score_tensor.gt(gold_scores.unsqueeze(1)).sum(dim=1).add(1).float()
    exact = predicted.eq(label_tensor)
    by_active_len_counts: dict[int, dict[str, int]] = {}
    for case, is_exact in zip(cases, exact.tolist()):
        active_len = effective_program_len(case)
        counts = by_active_len_counts.setdefault(int(active_len), {"correct": 0, "total": 0})
        counts["correct"] += int(bool(is_exact))
        counts["total"] += 1
    return {
        "answer_space_argmax_exact": float(exact.float().mean()),
        "answer_space_gold_mean_rank": float(ranks.mean()),
        "answer_space_gold_top3": float(ranks.le(3).float().mean()),
        "answer_space_gold_top5": float(ranks.le(5).float().mean()),
        "answer_space_argmax_by_active_len": {
            str(active_len): {
                "correct": counts["correct"],
                "total": counts["total"],
                "exact": float(counts["correct"] / max(1, counts["total"])),
            }
            for active_len, counts in sorted(by_active_len_counts.items())
        },
    }


def online_greedy_preference_loss(
    model,
    chosen_logits: torch.Tensor,
    chosen_targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    prompt_len: int,
    answer_len: int,
    think_steps: int,
    margin: float,
    max_cases: int,
) -> torch.Tensor:
    chosen_logprob_all = answer_sequence_logprob(
        chosen_logits,
        chosen_targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    selected_indexes: list[int] = []
    rejected_cases: list[TextReasoningCase] = []
    prompt_cases: list[TextReasoningCase] = []
    candidate_indexes = list(range(len(cases)))
    if int(max_cases) > 0:
        candidate_indexes = candidate_indexes[: int(max_cases)]
    with torch.no_grad():
        for index in candidate_indexes:
            case = cases[index]
            prompt = case_prompt(
                case,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            prompt_ids = torch.tensor(
                [tokenizer.encode(prompt)],
                dtype=torch.long,
                device=device,
            )
            generated = generate_answer(
                model,
                prompt_ids,
                answer_len=answer_len,
                think_steps=int(think_steps),
            )
            pred = tokenizer.decode(generated)
            pred_value = answer_value(pred)
            if pred_value is None or pred == case_answer(case):
                continue
            selected_indexes.append(index)
            prompt_cases.append(case)
            rejected_cases.append(
                TextReasoningCase(
                    case_id=case.case_id,
                    start=int(case.start),
                    op_ids=case.op_ids,
                    answer=int(pred_value),
                    family=str(case.family),
                )
            )
    if not selected_indexes:
        return torch.zeros((), device=device)
    rejected_x, rejected_y, rejected_prompt_len, rejected_answer_len = cases_to_batch(
        rejected_cases,
        tokenizer=tokenizer,
        device=device,
        prompt_cases=prompt_cases,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    rejected_logits = model(rejected_x, think_steps=int(think_steps))
    rejected_logprob = answer_sequence_logprob(
        rejected_logits,
        rejected_y,
        prompt_len=rejected_prompt_len,
        answer_len=rejected_answer_len,
    )
    index_tensor = torch.tensor(selected_indexes, dtype=torch.long, device=device)
    chosen_logprob = chosen_logprob_all.index_select(0, index_tensor)
    return F.relu(float(margin) - (chosen_logprob - rejected_logprob)).mean()


def state_trace_anti_collapse_loss(
    runtime: dict[str, torch.Tensor],
    *,
    min_variance: float,
    min_delta_norm: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in ("core_state_trace_h", "core_state_trace_l"):
        trace = runtime.get(key)
        if trace is None or int(trace.shape[1]) == 0:
            continue
        last_token_trace = trace[:, :, -1, :].float()
        if float(min_variance) > 0.0:
            variance_by_depth = last_token_trace.var(dim=0, unbiased=False).mean(dim=-1)
            losses.append(F.relu(float(min_variance) - variance_by_depth).mean())
        if float(min_delta_norm) > 0.0 and int(last_token_trace.shape[1]) > 1:
            deltas = last_token_trace[:, 1:, :] - last_token_trace[:, :-1, :]
            delta_norm = deltas.norm(dim=-1).mean(dim=0)
            losses.append(F.relu(float(min_delta_norm) - delta_norm).mean())
    if not losses:
        return torch.zeros((), device=next(iter(runtime.values())).device)
    return torch.stack(losses).mean()


def state_trace_family_anti_collapse_loss(
    runtime: dict[str, torch.Tensor],
    cases: list[TextReasoningCase],
    *,
    families: tuple[str, ...],
    state_source: str,
    max_consecutive_cosine: float,
    min_final_variance: float,
    late_fraction: float,
    cosine_loss_scale: float,
    reduction: str,
) -> torch.Tensor:
    """Prevent hard-family recurrent traces from freezing at late depths.

    This is not a side answer head. It only regularizes the recurrent state
    trace that already feeds the canonical LM logits.
    """
    if not runtime:
        return torch.zeros(())
    device = next(iter(runtime.values())).device
    family_set = {str(family) for family in families}
    if not family_set:
        return torch.zeros((), device=device)
    source = str(state_source)
    trace_keys = [
        key
        for key, enabled in (
            ("core_state_trace_h", source in {"h", "both"}),
            ("core_state_trace_l", source in {"l", "both"}),
        )
        if enabled and key in runtime
    ]
    if not trace_keys:
        return torch.zeros((), device=device)

    family_losses: list[torch.Tensor] = []
    for family in sorted(family_set):
        indexes = [
            index for index, case in enumerate(cases) if str(case.family) == family
        ]
        if not indexes:
            continue
        index_tensor = torch.tensor(indexes, dtype=torch.long, device=device)
        trace_losses: list[torch.Tensor] = []
        for key in trace_keys:
            trace = runtime.get(key)
            if trace is None or int(trace.shape[1]) == 0:
                continue
            last_token_trace = trace.index_select(0, index_tensor)[:, :, -1, :].float()
            if (
                float(max_consecutive_cosine) < 1.0
                and int(last_token_trace.shape[1]) > 1
            ):
                cosine = F.cosine_similarity(
                    last_token_trace[:, 1:, :],
                    last_token_trace[:, :-1, :],
                    dim=-1,
                )
                late_count = max(
                    1,
                    int(math.ceil(float(late_fraction) * int(cosine.shape[1]))),
                )
                late_cosine = cosine[:, -late_count:]
                trace_losses.append(
                    float(cosine_loss_scale)
                    * F.relu(late_cosine - float(max_consecutive_cosine)).mean()
                )
            if float(min_final_variance) > 0.0 and int(last_token_trace.shape[0]) > 1:
                final_variance = last_token_trace[:, -1, :].var(
                    dim=0,
                    unbiased=False,
                ).mean()
                trace_losses.append(
                    F.relu(float(min_final_variance) - final_variance)
                    / max(float(min_final_variance), 1.0e-6)
                )
        if trace_losses:
            family_losses.append(torch.stack(trace_losses).mean())
    if not family_losses:
        return torch.zeros((), device=device)
    stacked = torch.stack(family_losses)
    if str(reduction) == "max":
        return stacked.max()
    return stacked.mean()


def shared_lm_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Use the model's canonical LM/value-codec readout when available."""
    lm_logits = getattr(model, "_lm_logits", None)
    if callable(lm_logits):
        return lm_logits(hidden)
    return model.lm_head(hidden)


def latent_refinement_loss_from_runtime(
    model,
    runtime: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    min_depth: int = 1,
    noise_std: float = 0.0,
    depth_weight_power: float = 0.0,
    final_kl_weight: float = 0.0,
) -> torch.Tensor:
    """Train intermediate core states to refine into answer-ready LM logits.

    This is an auxiliary-only latent diffusion/readout probe: no side decoder is
    introduced. Each intermediate z_H state is optionally noised, passed through
    the model's existing decode stack, norm, and shared LM head, then supervised
    on the same answer tokens as the canonical autoregressive path.
    """
    trace = runtime.get("core_state_trace_h")
    if trace is None:
        raise ValueError("core_state_trace_h is unavailable for latent refinement")
    if trace.ndim != 4:
        raise ValueError("core_state_trace_h must have shape [batch, depth, seq, dim]")
    depth = int(trace.shape[1])
    if depth <= 0:
        return trace.sum() * 0.0
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    min_index = max(0, int(min_depth) - 1)
    if min_index >= depth:
        return trace.sum() * 0.0

    seq_len = int(trace.shape[2])
    causal_mask = model._causal_mask(seq_len, trace.device)
    losses: list[torch.Tensor] = []
    final_logits = runtime.get("logits")
    final_answer_probs = None
    if final_logits is not None and float(final_kl_weight) > 0.0:
        final_answer_probs = F.softmax(final_logits[:, start:end, :].detach(), dim=-1)

    for depth_index in range(min_index, depth):
        state = trace[:, depth_index]
        if float(noise_std) > 0.0:
            remaining = max(0.0, 1.0 - float(depth_index + 1) / max(1.0, float(depth)))
            state = state + torch.randn_like(state) * float(noise_std) * remaining
        decoded = model._run_stage(model.decode, state, causal_mask=causal_mask)
        logits = shared_lm_logits(model, model.norm(decoded))
        ce = answer_text_loss(
            logits,
            targets,
            prompt_len=prompt_len,
            answer_len=answer_len,
        )
        if final_answer_probs is not None:
            answer_log_probs = F.log_softmax(logits[:, start:end, :], dim=-1)
            kl = F.kl_div(
                answer_log_probs.reshape(-1, answer_log_probs.shape[-1]),
                final_answer_probs.reshape(-1, final_answer_probs.shape[-1]),
                reduction="batchmean",
            )
            ce = ce + float(final_kl_weight) * kl
        if float(depth_weight_power) > 0.0:
            weight = (
                float(depth_index + 1) / max(1.0, float(depth))
            ) ** float(depth_weight_power)
            ce = ce * float(weight)
        losses.append(ce)
    return torch.stack(losses).mean()


def state_trace_depth_answer_loss_from_runtime(
    model,
    runtime: dict[str, torch.Tensor],
    targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    prompt_len: int,
    answer_len: int,
    max_depth: int,
    min_depth: int = 1,
    max_depth_samples: int = 0,
    depth_sample_mode: str = "uniform",
    depth_weight_power: float = 0.0,
    state_source: str = "h",
    family_dro: bool = False,
    family_dro_temperature: float = 0.0,
) -> torch.Tensor:
    """Supervise traced recurrent states on stepwise answers via the LM path."""
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported state-trace depth source: {state_source}")
    trace_parts: list[torch.Tensor] = []
    if source in {"h", "both"}:
        trace_h = runtime.get("core_state_trace_h")
        if trace_h is None or int(trace_h.shape[1]) == 0:
            raise ValueError("core_state_trace_h is unavailable for state-trace depth loss")
        trace_parts.append(trace_h)
    if source in {"l", "both"}:
        trace_l = runtime.get("core_state_trace_l")
        if trace_l is None or int(trace_l.shape[1]) == 0:
            raise ValueError("core_state_trace_l is unavailable for state-trace depth loss")
        trace_parts.append(trace_l)
    depth_count = min(
        int(max_depth),
        int(targets.shape[1]),
        *(int(trace.shape[1]) for trace in trace_parts),
    )
    if depth_count <= 0:
        first_value = next(iter(runtime.values()))
        return first_value.float().sum() * 0.0
    first_depth = max(1, int(min_depth))
    if first_depth > depth_count:
        first_value = next(iter(runtime.values()))
        return first_value.float().sum() * 0.0
    depths = list(range(first_depth, depth_count + 1))
    sample_count = int(max_depth_samples)
    if sample_count > 0 and len(depths) > sample_count:
        mode = str(depth_sample_mode)
        if mode == "late":
            depths = depths[-sample_count:]
        elif mode == "uniform":
            if sample_count == 1:
                selected_positions = [len(depths) - 1]
            else:
                span = len(depths) - 1
                selected_positions = sorted(
                    {
                        int(round(float(index) * float(span) / float(sample_count - 1)))
                        for index in range(sample_count)
                    }
                )
                fill = len(depths) - 1
                while len(selected_positions) < sample_count and fill >= 0:
                    if fill not in selected_positions:
                        selected_positions.append(fill)
                    fill -= 1
                selected_positions = sorted(selected_positions[:sample_count])
            depths = [depths[position] for position in selected_positions]
        else:
            raise ValueError(
                f"unsupported state-trace depth sample mode: {depth_sample_mode}"
            )

    start = int(prompt_len) - 1
    end = start + int(answer_len)
    losses: list[tuple[float, torch.Tensor]] = []
    for trace in trace_parts:
        seq_len = int(trace.shape[2])
        causal_mask = model._causal_mask(seq_len, trace.device)
        for depth in depths:
            depth_index = depth - 1
            state = trace[:, depth_index]
            decoded = model._run_stage(model.decode, state, causal_mask=causal_mask)
            logits = shared_lm_logits(model, model.norm(decoded))
            selected_logits = logits[:, start:end, :]
            selected_targets = targets[:, depth_index, :]
            token_losses = F.cross_entropy(
                selected_logits.reshape(-1, selected_logits.shape[-1]),
                selected_targets.reshape(-1),
                reduction="none",
            )
            per_case = token_losses.reshape(selected_logits.shape[0], -1).mean(dim=1)
            if bool(family_dro):
                depth_loss = family_dro_from_case_losses(
                    per_case,
                    cases,
                    temperature=float(family_dro_temperature),
                )
            else:
                depth_loss = per_case.mean()
            weight = float(depth) ** max(0.0, float(depth_weight_power))
            losses.append((weight, depth_loss))
    if not losses:
        first_value = next(iter(runtime.values()))
        return first_value.float().sum() * 0.0
    total_weight = sum(weight for weight, _ in losses)
    return torch.stack(
        [loss * (weight / max(total_weight, 1e-12)) for weight, loss in losses]
    ).sum()


def prefix_state_alignment_loss(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    max_depth: int,
    modulus: int,
    max_cases: int,
) -> torch.Tensor:
    selected_cases = cases[: int(max_cases)] if int(max_cases) > 0 else cases
    if not selected_cases:
        return torch.zeros((), device=device)
    full_x, _, prompt_len, _ = cases_to_batch(
        selected_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    prompt_state_index = int(prompt_len) - 1
    losses: list[torch.Tensor] = []
    for depth in range(1, max(1, int(max_depth)) + 1):
        prefix_cases = [
            case_with_causal_prefix_len(
                case,
                prefix_len=int(depth),
                modulus=int(modulus),
            )
            for case in selected_cases
        ]
        prefix_x, _, _, _ = cases_to_batch(
            prefix_cases,
            tokenizer=tokenizer,
            device=device,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        full_runtime = model.forward_with_runtime(
            full_x,
            think_steps=int(depth),
            return_state_trace=True,
        )
        with torch.no_grad():
            prefix_runtime = model.forward_with_runtime(
                prefix_x,
                think_steps=int(depth),
                return_state_trace=True,
            )
        for key in ("core_state_trace_h", "core_state_trace_l"):
            if key not in full_runtime or key not in prefix_runtime:
                continue
            full_trace = full_runtime[key]
            prefix_trace = prefix_runtime[key]
            if int(full_trace.shape[1]) == 0 or int(prefix_trace.shape[1]) == 0:
                continue
            full_state = full_trace[:, -1, prompt_state_index, :].float()
            prefix_state = prefix_trace[:, -1, prompt_state_index, :].float().detach()
            losses.append(F.mse_loss(full_state, prefix_state))
    if not losses:
        return full_x.float().sum() * 0.0
    return torch.stack(losses).mean()


def recurrent_state_trace_feature(
    runtime: dict[str, torch.Tensor],
    *,
    depth_index: int,
    prompt_len: int,
    state_source: str,
    pooling: str,
) -> torch.Tensor:
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported recurrent state source: {state_source}")
    pooling = str(pooling)
    if pooling not in {"last", "mean", "flatten"}:
        raise ValueError(f"unsupported recurrent state pooling: {pooling}")
    prompt_end = int(prompt_len)
    prompt_state_index = prompt_end - 1
    parts: list[torch.Tensor] = []
    for key, enabled in (
        ("core_state_trace_h", source in {"h", "both"}),
        ("core_state_trace_l", source in {"l", "both"}),
    ):
        if not enabled:
            continue
        trace = runtime.get(key)
        if trace is None or int(trace.shape[1]) == 0:
            raise ValueError(f"{key} is unavailable for recurrent state feature")
        index = min(max(0, int(depth_index)), int(trace.shape[1]) - 1)
        state = trace[:, index, :prompt_end, :].float()
        if pooling == "last":
            parts.append(state[:, prompt_state_index, :])
        elif pooling == "mean":
            parts.append(state.mean(dim=1))
        else:
            parts.append(state.reshape(state.shape[0], -1))
    if not parts:
        first_value = next(iter(runtime.values()))
        return first_value.float().new_empty((0, 0))
    return torch.cat(parts, dim=-1)


def prefix_state_contrastive_loss(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    max_depth: int,
    modulus: int,
    max_cases: int,
    temperature: float,
    state_source: str,
    pooling: str,
) -> torch.Tensor:
    selected_cases = cases[: int(max_cases)] if int(max_cases) > 0 else cases
    if len(selected_cases) < 2:
        return torch.zeros((), device=device)
    full_x, _, prompt_len, _ = cases_to_batch(
        selected_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    full_runtime = model.forward_with_runtime(
        full_x,
        think_steps=max(1, int(max_depth)),
        return_state_trace=True,
    )
    labels = torch.arange(len(selected_cases), dtype=torch.long, device=device)
    losses: list[torch.Tensor] = []
    scale = max(float(temperature), 1e-6)
    for depth in range(1, max(1, int(max_depth)) + 1):
        prefix_cases = [
            case_with_causal_prefix_len(
                case,
                prefix_len=int(depth),
                modulus=int(modulus),
            )
            for case in selected_cases
        ]
        prefix_x, _, prefix_prompt_len, _ = cases_to_batch(
            prefix_cases,
            tokenizer=tokenizer,
            device=device,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        with torch.no_grad():
            prefix_runtime = model.forward_with_runtime(
                prefix_x,
                think_steps=int(depth),
                return_state_trace=True,
            )
        full_feature = recurrent_state_trace_feature(
            full_runtime,
            depth_index=depth - 1,
            prompt_len=prompt_len,
            state_source=state_source,
            pooling=pooling,
        )
        prefix_feature = recurrent_state_trace_feature(
            prefix_runtime,
            depth_index=depth - 1,
            prompt_len=prefix_prompt_len,
            state_source=state_source,
            pooling=pooling,
        ).detach()
        full_feature = F.normalize(full_feature, dim=-1)
        prefix_feature = F.normalize(prefix_feature, dim=-1)
        logits = full_feature.matmul(prefix_feature.t()) / scale
        losses.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.t(), labels)
            )
        )
    return torch.stack(losses).mean()


def reference_retention_kl_loss(
    model,
    reference_model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    modulus: int,
    active_len_min: int,
    active_len_max: int,
    max_cases: int,
    temperature: float,
) -> torch.Tensor:
    selected_cases = cases[: int(max_cases)] if int(max_cases) > 0 else cases
    if not selected_cases:
        return torch.zeros((), device=device)
    retention_cases: list[TextReasoningCase] = []
    max_program_len = max(len(case.op_ids) for case in selected_cases)
    lo = max(0, int(active_len_min))
    hi = max_program_len if int(active_len_max) < 0 else int(active_len_max)
    hi = max(lo, min(hi, max_program_len))
    for case in selected_cases:
        for active_len in range(lo, hi + 1):
            retention_cases.append(
                case_with_active_program_len(
                    case,
                    active_len=active_len,
                    modulus=int(modulus),
                )
            )
    if not retention_cases:
        return torch.zeros((), device=device)
    x, _, prompt_len, answer_len = cases_to_batch(
        retention_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    student_logits = model(x, think_steps=int(think_steps))
    with torch.no_grad():
        teacher_logits = reference_model(x, think_steps=int(think_steps))
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    scale = max(float(temperature), 1e-6)
    student_answer_logits = student_logits[:, start:end, :] / scale
    teacher_answer_logits = teacher_logits[:, start:end, :] / scale
    return F.kl_div(
        F.log_softmax(student_answer_logits, dim=-1),
        F.softmax(teacher_answer_logits, dim=-1),
        reduction="batchmean",
    ) * (scale * scale)


def active_len_replay_ce_loss(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    modulus: int,
    active_len_min: int,
    active_len_max: int,
    max_cases: int,
) -> torch.Tensor:
    selected_cases = cases[: int(max_cases)] if int(max_cases) > 0 else cases
    if not selected_cases:
        return torch.zeros((), device=device)
    replay_cases: list[TextReasoningCase] = []
    max_program_len = max(len(case.op_ids) for case in selected_cases)
    lo = max(0, int(active_len_min))
    hi = max_program_len if int(active_len_max) < 0 else int(active_len_max)
    hi = max(lo, min(hi, max_program_len))
    for case in selected_cases:
        for active_len in range(lo, hi + 1):
            replay_cases.append(
                case_with_active_program_len(
                    case,
                    active_len=active_len,
                    modulus=int(modulus),
                )
            )
    if not replay_cases:
        return torch.zeros((), device=device)
    replay_x, replay_y, replay_prompt_len, replay_answer_len = cases_to_batch(
        replay_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    replay_logits = model(replay_x, think_steps=int(think_steps))
    return answer_text_loss(
        replay_logits,
        replay_y,
        prompt_len=replay_prompt_len,
        answer_len=replay_answer_len,
    )


@torch.no_grad()
def core_answer_probe_features(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    state_source: str,
    pooling: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    active_lengths: list[int] = []
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported core probe state source: {state_source}")
    pooling = str(pooling)
    if pooling not in {"last", "mean", "flatten"}:
        raise ValueError(f"unsupported core probe pooling: {pooling}")
    size = max(1, int(batch_size))
    for start in range(0, len(cases), size):
        batch = cases[start : start + size]
        prompts = [
            case_prompt(
                case,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            for case in batch
        ]
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt) for prompt in prompts],
            dtype=torch.long,
            device=device,
        )
        runtime = model.forward_with_runtime(
            prompt_ids,
            think_steps=int(think_steps),
            return_state_trace=True,
        )
        prompt_state_index = int(prompt_ids.shape[1]) - 1
        parts: list[torch.Tensor] = []
        if source in {"h", "both"}:
            trace_h = runtime.get("core_state_trace_h")
            if trace_h is None or int(trace_h.shape[1]) == 0:
                raise ValueError("core_state_trace_h is unavailable for core probe")
            final_h = trace_h[:, -1, :, :].float()
            if pooling == "last":
                parts.append(final_h[:, prompt_state_index, :])
            elif pooling == "mean":
                parts.append(final_h.mean(dim=1))
            else:
                parts.append(final_h.reshape(final_h.shape[0], -1))
        if source in {"l", "both"}:
            trace_l = runtime.get("core_state_trace_l")
            if trace_l is None or int(trace_l.shape[1]) == 0:
                raise ValueError("core_state_trace_l is unavailable for core probe")
            final_l = trace_l[:, -1, :, :].float()
            if pooling == "last":
                parts.append(final_l[:, prompt_state_index, :])
            elif pooling == "mean":
                parts.append(final_l.mean(dim=1))
            else:
                parts.append(final_l.reshape(final_l.shape[0], -1))
        features.append(torch.cat(parts, dim=-1).detach())
        labels.append(
            torch.tensor(
                [int(case.answer) for case in batch],
                dtype=torch.long,
                device=device,
            )
        )
        active_lengths.extend(effective_program_len(case) for case in batch)
    if not features:
        return (
            torch.empty((0, 0), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [],
        )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0), active_lengths


def core_answer_probe_metrics(
    model,
    train_cases: list[TextReasoningCase],
    eval_cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
) -> dict[str, object]:
    device = torch.device(args.device)
    include_family_tag = include_family_tag_for_args(args)
    state_anchor = state_anchor_for_args(args)
    state_anchor_position = state_anchor_position_for_args(args)
    model.eval()
    train_subset = (
        train_cases
        if int(args.core_answer_probe_train_cases) <= 0
        else train_cases[: int(args.core_answer_probe_train_cases)]
    )
    eval_subset = (
        eval_cases
        if int(args.core_answer_probe_eval_cases) <= 0
        else eval_cases[: int(args.core_answer_probe_eval_cases)]
    )
    train_x, train_y, _ = core_answer_probe_features(
        model,
        train_subset,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
        think_steps=int(args.eval_think_steps),
        state_source=str(args.core_answer_probe_state_source),
        pooling=str(args.core_answer_probe_pooling),
        batch_size=int(args.core_answer_probe_batch_size),
    )
    eval_x, eval_y, active_lengths = core_answer_probe_features(
        model,
        eval_subset,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
        think_steps=int(args.eval_think_steps),
        state_source=str(args.core_answer_probe_state_source),
        pooling=str(args.core_answer_probe_pooling),
        batch_size=int(args.core_answer_probe_batch_size),
    )
    if int(train_x.shape[0]) == 0 or int(eval_x.shape[0]) == 0:
        return {"core_answer_probe_exact": 0.0, "core_answer_probe_cases": 0}
    probe = torch.nn.Linear(int(train_x.shape[1]), int(args.modulus)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.core_answer_probe_lr),
        weight_decay=float(args.core_answer_probe_weight_decay),
    )
    batch_size = max(1, int(args.core_answer_probe_batch_size))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 9001)
    probe.train()
    for _step in range(max(0, int(args.core_answer_probe_steps))):
        indexes = torch.randint(
            0,
            int(train_x.shape[0]),
            (min(batch_size, int(train_x.shape[0])),),
            generator=generator,
            device=device,
        )
        logits = probe(train_x.index_select(0, indexes))
        loss = F.cross_entropy(logits, train_y.index_select(0, indexes))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        train_pred = probe(train_x).argmax(dim=-1)
        eval_pred = probe(eval_x).argmax(dim=-1)
    train_exact = float(train_pred.eq(train_y).float().mean().detach().cpu())
    eval_correct = eval_pred.eq(eval_y).detach().cpu()
    eval_exact = float(eval_correct.float().mean())
    by_active_len: dict[str, dict[str, float | int]] = {}
    by_family: dict[str, dict[str, float | int]] = {}
    for active_len, correct in zip(active_lengths, eval_correct.tolist()):
        row = by_active_len.setdefault(
            str(int(active_len)),
            {"correct": 0, "total": 0, "exact": 0.0},
        )
        row["correct"] = int(row["correct"]) + int(bool(correct))
        row["total"] = int(row["total"]) + 1
    for case, correct in zip(eval_subset, eval_correct.tolist()):
        row = by_family.setdefault(
            str(case.family),
            {"correct": 0, "total": 0, "exact": 0.0},
        )
        row["correct"] = int(row["correct"]) + int(bool(correct))
        row["total"] = int(row["total"]) + 1
    for row in by_active_len.values():
        row["exact"] = float(int(row["correct"]) / max(1, int(row["total"])))
    for row in by_family.values():
        row["exact"] = float(int(row["correct"]) / max(1, int(row["total"])))
    return {
        "core_answer_probe_train_exact": train_exact,
        "core_answer_probe_exact": eval_exact,
        "core_answer_probe_cases": int(eval_x.shape[0]),
        "core_answer_probe_state_source": str(args.core_answer_probe_state_source),
        "core_answer_probe_pooling": str(args.core_answer_probe_pooling),
        "core_answer_probe_by_family": by_family,
        "core_answer_probe_by_active_len": by_active_len,
    }


@torch.no_grad()
def core_step_probe_features(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    think_steps: int,
    state_source: str,
    pooling: str,
    batch_size: int,
    modulus: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[str]]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    depths: list[int] = []
    active_lengths: list[int] = []
    families: list[str] = []
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported core probe state source: {state_source}")
    pooling = str(pooling)
    if pooling not in {"last", "mean", "flatten"}:
        raise ValueError(f"unsupported core probe pooling: {pooling}")
    size = max(1, int(batch_size))
    for start in range(0, len(cases), size):
        batch = cases[start : start + size]
        prompts = [
            case_prompt(
                case,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            for case in batch
        ]
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt) for prompt in prompts],
            dtype=torch.long,
            device=device,
        )
        runtime = model.forward_with_runtime(
            prompt_ids,
            think_steps=int(think_steps),
            return_state_trace=True,
        )
        prompt_state_index = int(prompt_ids.shape[1]) - 1
        trace_parts: list[torch.Tensor] = []
        if source in {"h", "both"}:
            trace_h = runtime.get("core_state_trace_h")
            if trace_h is None or int(trace_h.shape[1]) == 0:
                raise ValueError("core_state_trace_h is unavailable for core probe")
            trace_parts.append(trace_h.float())
        if source in {"l", "both"}:
            trace_l = runtime.get("core_state_trace_l")
            if trace_l is None or int(trace_l.shape[1]) == 0:
                raise ValueError("core_state_trace_l is unavailable for core probe")
            trace_parts.append(trace_l.float())
        depth_count = min(int(think_steps), *(int(part.shape[1]) for part in trace_parts))
        for depth_index in range(depth_count):
            step_parts: list[torch.Tensor] = []
            for trace in trace_parts:
                state = trace[:, depth_index, :, :]
                if pooling == "last":
                    step_parts.append(state[:, prompt_state_index, :])
                elif pooling == "mean":
                    step_parts.append(state.mean(dim=1))
                else:
                    step_parts.append(state.reshape(state.shape[0], -1))
            features.append(torch.cat(step_parts, dim=-1).detach())
            labels.append(
                torch.tensor(
                    [
                        int(
                            case_with_causal_prefix_len(
                                case,
                                prefix_len=depth_index + 1,
                                modulus=int(modulus),
                            ).answer
                        )
                        for case in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
            )
            depths.extend([depth_index + 1] * len(batch))
            active_lengths.extend(effective_program_len(case) for case in batch)
            families.extend(str(case.family) for case in batch)
    if not features:
        return (
            torch.empty((0, 0), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            [],
            [],
            [],
        )
    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        depths,
        active_lengths,
        families,
    )


def core_step_probe_metrics(
    model,
    train_cases: list[TextReasoningCase],
    eval_cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
) -> dict[str, object]:
    device = torch.device(args.device)
    include_family_tag = include_family_tag_for_args(args)
    state_anchor = state_anchor_for_args(args)
    state_anchor_position = state_anchor_position_for_args(args)
    model.eval()
    train_subset = (
        train_cases
        if int(args.core_answer_probe_train_cases) <= 0
        else train_cases[: int(args.core_answer_probe_train_cases)]
    )
    eval_subset = (
        eval_cases
        if int(args.core_answer_probe_eval_cases) <= 0
        else eval_cases[: int(args.core_answer_probe_eval_cases)]
    )
    train_x, train_y, _, _, _ = core_step_probe_features(
        model,
        train_subset,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
        think_steps=int(args.eval_think_steps),
        state_source=str(args.core_answer_probe_state_source),
        pooling=str(args.core_answer_probe_pooling),
        batch_size=int(args.core_answer_probe_batch_size),
        modulus=int(args.modulus),
    )
    eval_x, eval_y, depths, active_lengths, families = core_step_probe_features(
        model,
        eval_subset,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
        think_steps=int(args.eval_think_steps),
        state_source=str(args.core_answer_probe_state_source),
        pooling=str(args.core_answer_probe_pooling),
        batch_size=int(args.core_answer_probe_batch_size),
        modulus=int(args.modulus),
    )
    if int(train_x.shape[0]) == 0 or int(eval_x.shape[0]) == 0:
        return {"core_step_probe_exact": 0.0, "core_step_probe_cases": 0}
    probe = torch.nn.Linear(int(train_x.shape[1]), int(args.modulus)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.core_answer_probe_lr),
        weight_decay=float(args.core_answer_probe_weight_decay),
    )
    batch_size = max(1, int(args.core_answer_probe_batch_size))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 9101)
    probe.train()
    for _step in range(max(0, int(args.core_answer_probe_steps))):
        indexes = torch.randint(
            0,
            int(train_x.shape[0]),
            (min(batch_size, int(train_x.shape[0])),),
            generator=generator,
            device=device,
        )
        logits = probe(train_x.index_select(0, indexes))
        loss = F.cross_entropy(logits, train_y.index_select(0, indexes))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        train_pred = probe(train_x).argmax(dim=-1)
        eval_pred = probe(eval_x).argmax(dim=-1)
    train_exact = float(train_pred.eq(train_y).float().mean().detach().cpu())
    eval_correct = eval_pred.eq(eval_y).detach().cpu()
    by_depth: dict[str, dict[str, float | int]] = {}
    by_active_len: dict[str, dict[str, float | int]] = {}
    by_family: dict[str, dict[str, float | int]] = {}
    for depth, active_len, family, correct in zip(
        depths,
        active_lengths,
        families,
        eval_correct.tolist(),
    ):
        depth_row = by_depth.setdefault(
            str(int(depth)),
            {"correct": 0, "total": 0, "exact": 0.0},
        )
        depth_row["correct"] = int(depth_row["correct"]) + int(bool(correct))
        depth_row["total"] = int(depth_row["total"]) + 1
        active_row = by_active_len.setdefault(
            str(int(active_len)),
            {"correct": 0, "total": 0, "exact": 0.0},
        )
        active_row["correct"] = int(active_row["correct"]) + int(bool(correct))
        active_row["total"] = int(active_row["total"]) + 1
        family_row = by_family.setdefault(
            str(family),
            {"correct": 0, "total": 0, "exact": 0.0},
        )
        family_row["correct"] = int(family_row["correct"]) + int(bool(correct))
        family_row["total"] = int(family_row["total"]) + 1
    for row in [*by_depth.values(), *by_active_len.values(), *by_family.values()]:
        row["exact"] = float(int(row["correct"]) / max(1, int(row["total"])))
    return {
        "core_step_probe_train_exact": train_exact,
        "core_step_probe_exact": float(eval_correct.float().mean()),
        "core_step_probe_cases": int(eval_x.shape[0]),
        "core_step_probe_state_source": str(args.core_answer_probe_state_source),
        "core_step_probe_pooling": str(args.core_answer_probe_pooling),
        "core_step_probe_by_depth": by_depth,
        "core_step_probe_by_active_len": by_active_len,
        "core_step_probe_by_family": by_family,
    }


def _order_router_encoded_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return per-token L->H / H->L->H route logits for router probes/losses."""
    router, _force_attr = _order_router_module_and_force_attr(model)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    seq_len = int(input_ids.shape[1])
    if hasattr(model, "_token_embeddings"):
        x = model._token_embeddings(input_ids)
    else:
        x = model.token_embed(input_ids)
    if str(getattr(model, "position_embedding_mode", "learned")) in {
        "learned",
        "randomized",
    }:
        if hasattr(model, "_position_ids"):
            pos = model._position_ids(seq_len, input_ids.device)
        else:
            pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + model.pos_embed(pos)
    mask = model._causal_mask(seq_len, input_ids.device)
    encoded = model._run_stage(model.encode, x, causal_mask=mask)
    logits = router(encoded)
    if (
        str(getattr(model, "think_structure", ""))
        == "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_sequence_order_router"
    ):
        logits = logits[:, -1:, :].expand_as(logits)
    return logits


def _order_router_module_and_force_attr(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, str]:
    if hasattr(model, "trm_order_router"):
        return model.trm_order_router, "trm_order_router_force_route"
    if hasattr(model, "trm_nested_order_router"):
        return model.trm_nested_order_router, "trm_nested_order_router_force_route"
    raise ValueError("model does not expose an order router")


def _has_order_router(model: torch.nn.Module) -> bool:
    return hasattr(model, "trm_order_router") or hasattr(model, "trm_nested_order_router")


def _order_router_encoded_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return per-token L->H / H->L->H route probabilities for router probes."""
    return torch.softmax(_order_router_encoded_logits(model, input_ids), dim=-1)


def _summarize_order_router_routes(
    route_probs: torch.Tensor,
    cases: list[TextReasoningCase],
) -> dict[str, object]:
    last = route_probs[:, -1, :]
    mean = route_probs.mean(dim=1)
    clamped = route_probs.clamp_min(1e-12)
    entropy = -(clamped * clamped.log()).sum(dim=-1)

    def summarize_indexes(indexes: list[int]) -> dict[str, float | int]:
        index_tensor = torch.tensor(indexes, dtype=torch.long)
        selected_last = last.index_select(0, index_tensor)
        selected_mean = mean.index_select(0, index_tensor)
        selected_entropy = entropy.index_select(0, index_tensor)
        return {
            "count": int(len(indexes)),
            "last_lh_prob": float(selected_last[:, 0].mean()),
            "last_hlh_prob": float(selected_last[:, 1].mean()),
            "mean_lh_prob": float(selected_mean[:, 0].mean()),
            "mean_hlh_prob": float(selected_mean[:, 1].mean()),
            "mean_entropy": float(selected_entropy.mean()),
        }

    if not cases:
        summary: dict[str, object] = {
            "count": 0,
            "last_lh_prob": 0.0,
            "last_hlh_prob": 0.0,
            "mean_lh_prob": 0.0,
            "mean_hlh_prob": 0.0,
            "mean_entropy": 0.0,
        }
    else:
        summary = summarize_indexes(list(range(len(cases))))

    by_family: dict[str, object] = {}
    for family in sorted({str(case.family) for case in cases}):
        indexes = [
            index for index, case in enumerate(cases) if str(case.family) == family
        ]
        if indexes:
            by_family[family] = summarize_indexes(indexes)

    by_active_len: dict[str, object] = {}
    active_lengths = [effective_program_len(case) for case in cases]
    for active_len in sorted(set(active_lengths)):
        indexes = [
            index for index, item in enumerate(active_lengths) if int(item) == active_len
        ]
        if indexes:
            by_active_len[str(active_len)] = summarize_indexes(indexes)

    summary["by_family"] = by_family
    summary["by_active_len"] = by_active_len
    return summary


@torch.no_grad()
def order_router_probe_metrics(
    model: torch.nn.Module,
    cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
) -> dict[str, object]:
    """Measure whether the order router actually selects different routes."""
    if not _has_order_router(model):
        return {"available": False, "reason": "missing_order_router"}
    model.eval()
    device = torch.device(args.device)
    include_family_tag = include_family_tag_for_args(args)
    state_anchor = state_anchor_for_args(args)
    state_anchor_position = state_anchor_position_for_args(args)
    batch_size = max(1, min(int(getattr(args, "batch_size", 128)), 128))
    route_batches: list[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start : start + batch_size]
        prompts = [
            case_prompt(
                case,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            for case in batch_cases
        ]
        input_ids = torch.tensor(
            [tokenizer.encode(prompt) for prompt in prompts],
            dtype=torch.long,
            device=device,
        )
        route_batches.append(_order_router_encoded_probs(model, input_ids).detach().cpu())
    if not route_batches:
        return {"available": True, "cases": 0}
    metrics = _summarize_order_router_routes(
        torch.cat(route_batches, dim=0).float(),
        cases,
    )
    metrics["available"] = True
    metrics["route_names"] = ["l_then_h", "h_then_l_then_h"]
    return metrics


def order_router_family_order_targets(
    cases: list[TextReasoningCase],
    *,
    device: torch.device,
    target_mode: str = "family_order",
) -> torch.Tensor:
    """Route targets for recurrent transition specialization.

    `family_order` keeps the original forward-vs-reverse split.  The
    `chain_vs_checksum` mode treats both ordered chain tasks as route1 and
    reserves route0 for checksum, which is useful when the bottleneck is
    computation-chain floor rather than reverse order alone.
    """
    mode = str(target_mode)
    if mode == "family_order":
        labels = [1 if str(case.family) == "revchain" else 0 for case in cases]
    elif mode == "chain_vs_checksum":
        labels = [0 if str(case.family) == "checksum" else 1 for case in cases]
    else:
        raise ValueError(f"unsupported order-router aux target mode: {target_mode}")
    return torch.tensor(labels, dtype=torch.long, device=device)


def order_router_family_order_loss(
    model: torch.nn.Module,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    target_mode: str = "family_order",
) -> torch.Tensor:
    """Low-weight auxiliary that makes the route selector learnable.

    The target does not compute the answer. It only supervises which recurrent
    update order should be available for the normal LM-logit answer path.
    """
    if not cases:
        return torch.zeros((), device=device)
    prompts = [
        case_prompt(
            case,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        for case in cases
    ]
    input_ids = torch.tensor(
        [tokenizer.encode(prompt) for prompt in prompts],
        dtype=torch.long,
        device=device,
    )
    logits = _order_router_encoded_logits(model, input_ids)[:, -1, :]
    targets = order_router_family_order_targets(
        cases,
        device=device,
        target_mode=str(target_mode),
    )
    return F.cross_entropy(logits, targets)


def forced_route_answer_loss(
    model: torch.nn.Module,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    route: int,
    families: tuple[str, ...],
    think_steps: int,
    loss_type: str = "cross_entropy",
    max_cases: int = 0,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> torch.Tensor:
    """Train a specific route candidate through the normal LM answer path.

    This deliberately bypasses the router, but not the language path: the loss
    is still CE on answer tokens emitted by the model's LM head.
    """
    _router, force_attr = _order_router_module_and_force_attr(model)
    route_id = int(route)
    if route_id not in {0, 1}:
        raise ValueError("forced-route answer route must be 0 or 1")
    family_set = {str(item) for item in families}
    selected_cases = [
        case for case in cases if not family_set or str(case.family) in family_set
    ]
    if int(max_cases) > 0:
        selected_cases = selected_cases[: int(max_cases)]
    if not selected_cases:
        return torch.zeros((), device=device)
    x, y, prompt_len, answer_len = cases_to_batch(
        selected_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    had_route_force = hasattr(model, force_attr)
    old_route_force = getattr(model, force_attr, None)
    setattr(model, force_attr, route_id)
    try:
        logits = model(x, think_steps=int(think_steps))
    finally:
        if had_route_force:
            setattr(model, force_attr, old_route_force)
        else:
            delattr(model, force_attr)
    return answer_text_loss(
        logits,
        y,
        prompt_len=prompt_len,
        answer_len=answer_len,
        loss_type=loss_type,
    )


def forced_route_intermediate_depth_loss(
    model: torch.nn.Module,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    route: int,
    families: tuple[str, ...],
    max_depth: int,
    modulus: int,
    min_depth: int = 1,
    depth_weight_power: float = 0.0,
    max_cases: int = 0,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> torch.Tensor:
    """Train a forced route on stepwise intermediate answers through LM logits."""
    _router, force_attr = _order_router_module_and_force_attr(model)
    route_id = int(route)
    if route_id not in {0, 1}:
        raise ValueError("forced-route depth route must be 0 or 1")
    family_set = {str(item) for item in families}
    selected_cases = [
        case for case in cases if not family_set or str(case.family) in family_set
    ]
    if int(max_cases) > 0:
        selected_cases = selected_cases[: int(max_cases)]
    if not selected_cases:
        return torch.zeros((), device=device)
    x, _y, prompt_len, answer_len = cases_to_batch(
        selected_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    depth_targets = intermediate_answer_targets(
        selected_cases,
        tokenizer=tokenizer,
        max_depth=int(max_depth),
        modulus=int(modulus),
        device=device,
    )
    had_route_force = hasattr(model, force_attr)
    old_route_force = getattr(model, force_attr, None)
    setattr(model, force_attr, route_id)
    try:
        return intermediate_depth_loss(
            model,
            x,
            depth_targets,
            prompt_len=prompt_len,
            answer_len=answer_len,
            max_depth=int(max_depth),
            min_depth=int(min_depth),
            depth_weight_power=float(depth_weight_power),
        )
    finally:
        if had_route_force:
            setattr(model, force_attr, old_route_force)
        else:
            delattr(model, force_attr)


def forced_route_prefix_depth_anchor_loss(
    model: torch.nn.Module,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    route: int,
    families: tuple[str, ...],
    max_depth: int,
    modulus: int,
    min_depth: int = 1,
    depth_weight_power: float = 0.0,
    max_cases: int = 0,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
) -> torch.Tensor:
    """Train a forced route to solve causal-prefix prompts through LM logits.

    This differs from forced_route_intermediate_depth_loss: the prompt itself is
    shortened to the causal prefix, so the route learns the local transition
    problem before being asked to carry it inside a full-length program.
    """
    _router, force_attr = _order_router_module_and_force_attr(model)
    route_id = int(route)
    if route_id not in {0, 1}:
        raise ValueError("forced-route prefix-anchor route must be 0 or 1")
    family_set = {str(item) for item in families}
    selected_cases = [
        case for case in cases if not family_set or str(case.family) in family_set
    ]
    if int(max_cases) > 0:
        selected_cases = selected_cases[: int(max_cases)]
    if not selected_cases:
        return torch.zeros((), device=device)
    losses: list[tuple[float, torch.Tensor]] = []
    first_depth = max(1, int(min_depth))
    had_route_force = hasattr(model, force_attr)
    old_route_force = getattr(model, force_attr, None)
    setattr(model, force_attr, route_id)
    try:
        for depth in range(1, max(1, int(max_depth)) + 1):
            if depth < first_depth:
                continue
            prefix_cases = [
                case_with_causal_prefix_len(
                    case,
                    prefix_len=int(depth),
                    modulus=int(modulus),
                )
                for case in selected_cases
            ]
            prefix_x, prefix_y, prefix_prompt_len, prefix_answer_len = cases_to_batch(
                prefix_cases,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            logits = model(prefix_x, think_steps=int(depth))
            depth_loss = answer_text_loss(
                logits,
                prefix_y,
                prompt_len=prefix_prompt_len,
                answer_len=prefix_answer_len,
            )
            weight = float(depth) ** max(0.0, float(depth_weight_power))
            losses.append((weight, depth_loss))
    finally:
        if had_route_force:
            setattr(model, force_attr, old_route_force)
        else:
            delattr(model, force_attr)
    if not losses:
        return torch.zeros((), device=device)
    total_weight = sum(weight for weight, _ in losses)
    return torch.stack(
        [loss * (weight / max(total_weight, 1e-12)) for weight, loss in losses]
    ).sum()


def core_step_codec_feature_dim(
    *,
    d_model: int,
    prompt_len: int,
    state_source: str,
    pooling: str,
) -> int:
    source_multiplier = 2 if str(state_source) == "both" else 1
    if str(pooling) == "flatten":
        return source_multiplier * int(d_model) * int(prompt_len)
    return source_multiplier * int(d_model)


def core_step_codec_loss_from_runtime(
    runtime: dict[str, torch.Tensor],
    cases: list[TextReasoningCase],
    codec_head: torch.nn.Module,
    *,
    prompt_len: int,
    max_depth: int,
    modulus: int,
    state_source: str,
    pooling: str,
) -> torch.Tensor:
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported core-step codec state source: {state_source}")
    pooling = str(pooling)
    if pooling not in {"last", "mean", "flatten"}:
        raise ValueError(f"unsupported core-step codec pooling: {pooling}")
    trace_parts: list[torch.Tensor] = []
    if source in {"h", "both"}:
        trace_h = runtime.get("core_state_trace_h")
        if trace_h is None or int(trace_h.shape[1]) == 0:
            raise ValueError("core_state_trace_h is unavailable for core-step codec")
        trace_parts.append(trace_h.float())
    if source in {"l", "both"}:
        trace_l = runtime.get("core_state_trace_l")
        if trace_l is None or int(trace_l.shape[1]) == 0:
            raise ValueError("core_state_trace_l is unavailable for core-step codec")
        trace_parts.append(trace_l.float())
    depth_count = min(int(max_depth), *(int(part.shape[1]) for part in trace_parts))
    if depth_count <= 0:
        first_value = next(iter(runtime.values()))
        return first_value.float().sum() * 0.0
    prompt_end = int(prompt_len)
    prompt_state_index = prompt_end - 1
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for depth_index in range(depth_count):
        step_parts: list[torch.Tensor] = []
        for trace in trace_parts:
            state = trace[:, depth_index, :prompt_end, :]
            if pooling == "last":
                step_parts.append(state[:, prompt_state_index, :])
            elif pooling == "mean":
                step_parts.append(state.mean(dim=1))
            else:
                step_parts.append(state.reshape(state.shape[0], -1))
        features.append(torch.cat(step_parts, dim=-1))
        labels.append(
            torch.tensor(
                [
                    int(
                        case_with_causal_prefix_len(
                            case,
                            prefix_len=depth_index + 1,
                            modulus=int(modulus),
                        ).answer
                    )
                    for case in cases
                ],
                dtype=torch.long,
                device=trace_parts[0].device,
            )
        )
    feature_tensor = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return F.cross_entropy(codec_head(feature_tensor), label_tensor)


def core_step_op_codec_loss_from_runtime(
    runtime: dict[str, torch.Tensor],
    cases: list[TextReasoningCase],
    codec_head: torch.nn.Module,
    *,
    prompt_len: int,
    max_depth: int,
    state_source: str,
    pooling: str,
) -> torch.Tensor:
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported core-step op codec state source: {state_source}")
    pooling = str(pooling)
    if pooling not in {"last", "mean", "flatten"}:
        raise ValueError(f"unsupported core-step op codec pooling: {pooling}")
    trace_parts: list[torch.Tensor] = []
    if source in {"h", "both"}:
        trace_h = runtime.get("core_state_trace_h")
        if trace_h is None or int(trace_h.shape[1]) == 0:
            raise ValueError("core_state_trace_h is unavailable for core-step op codec")
        trace_parts.append(trace_h.float())
    if source in {"l", "both"}:
        trace_l = runtime.get("core_state_trace_l")
        if trace_l is None or int(trace_l.shape[1]) == 0:
            raise ValueError("core_state_trace_l is unavailable for core-step op codec")
        trace_parts.append(trace_l.float())
    depth_count = min(int(max_depth), *(int(part.shape[1]) for part in trace_parts))
    if depth_count <= 0:
        first_value = next(iter(runtime.values()))
        return first_value.float().sum() * 0.0
    prompt_end = int(prompt_len)
    prompt_state_index = prompt_end - 1
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for depth_index in range(depth_count):
        step_parts: list[torch.Tensor] = []
        for trace in trace_parts:
            state = trace[:, depth_index, :prompt_end, :]
            if pooling == "last":
                step_parts.append(state[:, prompt_state_index, :])
            elif pooling == "mean":
                step_parts.append(state.mean(dim=1))
            else:
                step_parts.append(state.reshape(state.shape[0], -1))
        features.append(torch.cat(step_parts, dim=-1))
        labels.append(
            torch.tensor(
                [
                    causal_op_id_at_depth(case, depth_index=depth_index)
                    for case in cases
                ],
                dtype=torch.long,
                device=trace_parts[0].device,
            )
        )
    feature_tensor = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return F.cross_entropy(codec_head(feature_tensor), label_tensor)


def core_step_position_codec_loss_from_runtime(
    runtime: dict[str, torch.Tensor],
    cases: list[TextReasoningCase],
    codec_head: torch.nn.Module,
    *,
    prompt_len: int,
    max_depth: int,
    state_source: str,
    pooling: str,
) -> torch.Tensor:
    source = str(state_source)
    if source not in {"h", "l", "both"}:
        raise ValueError(f"unsupported core-step position codec state source: {state_source}")
    pooling = str(pooling)
    if pooling not in {"last", "mean", "flatten"}:
        raise ValueError(f"unsupported core-step position codec pooling: {pooling}")
    trace_parts: list[torch.Tensor] = []
    if source in {"h", "both"}:
        trace_h = runtime.get("core_state_trace_h")
        if trace_h is None or int(trace_h.shape[1]) == 0:
            raise ValueError("core_state_trace_h is unavailable for core-step position codec")
        trace_parts.append(trace_h.float())
    if source in {"l", "both"}:
        trace_l = runtime.get("core_state_trace_l")
        if trace_l is None or int(trace_l.shape[1]) == 0:
            raise ValueError("core_state_trace_l is unavailable for core-step position codec")
        trace_parts.append(trace_l.float())
    depth_count = min(int(max_depth), *(int(part.shape[1]) for part in trace_parts))
    if depth_count <= 0:
        first_value = next(iter(runtime.values()))
        return first_value.float().sum() * 0.0
    prompt_end = int(prompt_len)
    prompt_state_index = prompt_end - 1
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for depth_index in range(depth_count):
        step_parts: list[torch.Tensor] = []
        for trace in trace_parts:
            state = trace[:, depth_index, :prompt_end, :]
            if pooling == "last":
                step_parts.append(state[:, prompt_state_index, :])
            elif pooling == "mean":
                step_parts.append(state.mean(dim=1))
            else:
                step_parts.append(state.reshape(state.shape[0], -1))
        features.append(torch.cat(step_parts, dim=-1))
        labels.append(
            torch.tensor(
                [
                    causal_op_position_at_depth(case, depth_index=depth_index)
                    for case in cases
                ],
                dtype=torch.long,
                device=trace_parts[0].device,
            )
        )
    feature_tensor = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return F.cross_entropy(codec_head(feature_tensor), label_tensor)


def intermediate_answer_targets(
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    max_depth: int,
    modulus: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[list[list[int]]] = []
    for case in cases:
        row: list[list[int]] = []
        for depth in range(max(1, int(max_depth))):
            prefix_case = case_with_causal_prefix_len(
                case,
                prefix_len=depth + 1,
                modulus=int(modulus),
            )
            value = int(prefix_case.answer)
            row.append(tokenizer.encode(f"{fmt2(value)}\n"))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def intermediate_depth_loss(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    max_depth: int,
    min_depth: int = 1,
    depth_weight_power: float = 0.0,
) -> torch.Tensor:
    losses: list[tuple[float, torch.Tensor]] = []
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    first_depth = max(1, int(min_depth))
    for depth in range(1, max(1, int(max_depth)) + 1):
        if depth < first_depth:
            continue
        depth_index = min(depth - 1, int(targets.shape[1]) - 1)
        logits = model(input_ids, think_steps=depth)
        depth_loss = F.cross_entropy(
            logits[:, start:end, :].reshape(-1, logits.shape[-1]),
            targets[:, depth_index, :].reshape(-1),
        )
        weight = float(depth) ** max(0.0, float(depth_weight_power))
        losses.append((weight, depth_loss))
    if not losses:
        return input_ids.float().sum() * 0.0
    total_weight = sum(weight for weight, _ in losses)
    return torch.stack(
        [loss * (weight / max(total_weight, 1e-12)) for weight, loss in losses]
    ).sum()


def intermediate_depth_family_dro_loss(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    prompt_len: int,
    answer_len: int,
    max_depth: int,
    min_depth: int = 1,
    depth_weight_power: float = 0.0,
    temperature: float = 0.0,
) -> torch.Tensor:
    losses: list[tuple[float, torch.Tensor]] = []
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    first_depth = max(1, int(min_depth))
    for depth in range(1, max(1, int(max_depth)) + 1):
        if depth < first_depth:
            continue
        depth_index = min(depth - 1, int(targets.shape[1]) - 1)
        logits = model(input_ids, think_steps=depth)
        selected_logits = logits[:, start:end, :]
        selected_targets = targets[:, depth_index, :]
        token_losses = F.cross_entropy(
            selected_logits.reshape(-1, selected_logits.shape[-1]),
            selected_targets.reshape(-1),
            reduction="none",
        )
        per_case = token_losses.reshape(selected_logits.shape[0], -1).mean(dim=1)
        depth_loss = family_dro_from_case_losses(
            per_case,
            cases,
            temperature=float(temperature),
        )
        weight = float(depth) ** max(0.0, float(depth_weight_power))
        losses.append((weight, depth_loss))
    if not losses:
        return input_ids.float().sum() * 0.0
    total_weight = sum(weight for weight, _ in losses)
    return torch.stack(
        [loss * (weight / max(total_weight, 1e-12)) for weight, loss in losses]
    ).sum()


def prefix_depth_anchor_loss(
    model,
    cases: list[TextReasoningCase],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
    include_family_tag: bool,
    state_anchor: bool = False,
    state_anchor_position: str = "before_answer",
    max_depth: int,
    modulus: int,
    min_depth: int = 1,
    depth_weight_power: float = 0.0,
    max_cases: int = 0,
) -> torch.Tensor:
    selected_cases = cases[: int(max_cases)] if int(max_cases) > 0 else cases
    if not selected_cases:
        return torch.zeros((), device=device)
    losses: list[tuple[float, torch.Tensor]] = []
    first_depth = max(1, int(min_depth))
    for depth in range(1, max(1, int(max_depth)) + 1):
        if depth < first_depth:
            continue
        prefix_cases = [
            case_with_causal_prefix_len(
                case,
                prefix_len=int(depth),
                modulus=int(modulus),
            )
            for case in selected_cases
        ]
        prefix_x, prefix_y, prefix_prompt_len, prefix_answer_len = cases_to_batch(
            prefix_cases,
            tokenizer=tokenizer,
            device=device,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        logits = model(prefix_x, think_steps=depth)
        depth_loss = answer_text_loss(
            logits,
            prefix_y,
            prompt_len=prefix_prompt_len,
            answer_len=prefix_answer_len,
        )
        weight = float(depth) ** max(0.0, float(depth_weight_power))
        losses.append((weight, depth_loss))
    if not losses:
        return torch.zeros((), device=device)
    total_weight = sum(weight for weight, _ in losses)
    return torch.stack(
        [loss * (weight / max(total_weight, 1e-12)) for weight, loss in losses]
    ).sum()


def halt_depth_final_answer_loss(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    prompt_len: int,
    answer_len: int,
    max_depth: int,
    min_halt_step: int,
) -> torch.Tensor:
    depth = max(1, int(max_depth))
    active_lengths = torch.tensor(
        [effective_program_len(case) for case in cases],
        dtype=torch.long,
        device=input_ids.device,
    )
    halt_steps = active_len_first_halt_steps(
        active_lengths,
        max_depth=depth,
        min_halt_step=int(min_halt_step),
    )
    losses: list[torch.Tensor] = []
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    for halt_depth in range(1, depth + 1):
        mask = halt_steps == halt_depth
        if not bool(mask.any()):
            continue
        logits = model(input_ids[mask], think_steps=halt_depth)
        losses.append(
            F.cross_entropy(
                logits[:, start:end, :].reshape(-1, logits.shape[-1]),
                targets[mask, start:end].reshape(-1),
            )
        )
    if not losses:
        return input_ids.float().sum() * 0.0
    return torch.stack(losses).mean()


def teacher_depth_halt_targets_from_correctness(
    correctness: torch.Tensor,
    *,
    min_halt_step: int = 1,
) -> torch.Tensor:
    if correctness.ndim != 2:
        raise ValueError("correctness must have shape [batch, depth]")
    batch, depth = correctness.shape
    step_ids = torch.arange(depth, device=correctness.device).view(1, depth) + 1
    eligible = correctness.bool() & (step_ids >= max(1, int(min_halt_step)))
    has_halt = eligible.any(dim=1)
    first_halt = eligible.float().argmax(dim=1)
    depth_ids = torch.arange(depth, device=correctness.device).view(1, depth)
    targets = (depth_ids >= first_halt.view(batch, 1)) & has_halt.view(batch, 1)
    return targets.to(dtype=torch.float32)


def active_len_halt_targets(
    active_lengths: torch.Tensor,
    *,
    max_depth: int,
    min_halt_step: int = 1,
) -> torch.Tensor:
    first_halt = active_len_first_halt_steps(
        active_lengths,
        max_depth=max_depth,
        min_halt_step=min_halt_step,
    )
    depth = max(1, int(max_depth))
    step_ids = torch.arange(depth, device=active_lengths.device).view(1, depth) + 1
    return (step_ids >= first_halt.view(-1, 1)).to(dtype=torch.float32)


def active_len_first_halt_steps(
    active_lengths: torch.Tensor,
    *,
    max_depth: int,
    min_halt_step: int = 1,
) -> torch.Tensor:
    if active_lengths.ndim != 1:
        raise ValueError("active_lengths must have shape [batch]")
    depth = max(1, int(max_depth))
    return torch.maximum(
        active_lengths.to(dtype=torch.long),
        torch.full_like(active_lengths.to(dtype=torch.long), max(1, int(min_halt_step))),
    ).clamp(min=1, max=depth)


def active_len_first_halt_targets(
    active_lengths: torch.Tensor,
    *,
    max_depth: int,
    min_halt_step: int = 1,
) -> torch.Tensor:
    first_halt = active_len_first_halt_steps(
        active_lengths,
        max_depth=max_depth,
        min_halt_step=min_halt_step,
    )
    depth = max(1, int(max_depth))
    step_ids = torch.arange(depth, device=active_lengths.device).view(1, depth) + 1
    return (step_ids == first_halt.view(-1, 1)).to(dtype=torch.float32)


def adaptive_halt_teacher_depth_loss(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    prompt_len: int,
    answer_len: int,
    max_depth: int,
    min_halt_step: int,
) -> torch.Tensor:
    depth = max(1, int(max_depth))
    start = int(prompt_len) - 1
    end = start + int(answer_len)
    correctness_steps: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(1, depth + 1):
            logits = model(input_ids, think_steps=step)
            pred = logits[:, start:end, :].argmax(dim=-1)
            correctness_steps.append((pred == targets[:, start:end]).all(dim=1))
    halt_targets = teacher_depth_halt_targets_from_correctness(
        torch.stack(correctness_steps, dim=1),
        min_halt_step=int(min_halt_step),
    )
    runtime = model.forward_with_runtime(
        input_ids,
        think_steps=depth,
        adaptive_halt=False,
        halt_min_steps=int(min_halt_step),
    )
    halt_logits = runtime["core_q_halt_logits"]
    if halt_logits.shape != halt_targets.shape:
        raise ValueError("halt logits and teacher-depth halt targets must match")
    return F.binary_cross_entropy_with_logits(halt_logits.float(), halt_targets.float())


def adaptive_halt_active_len_loss(
    model,
    input_ids: torch.Tensor,
    cases: list[TextReasoningCase],
    *,
    max_depth: int,
    min_halt_step: int,
    target_shape: str = "first_step",
) -> torch.Tensor:
    depth = max(1, int(max_depth))
    runtime = model.forward_with_runtime(
        input_ids,
        think_steps=depth,
        adaptive_halt=False,
        halt_min_steps=int(min_halt_step),
    )
    halt_logits = runtime["core_q_halt_logits"]
    active_lengths = torch.tensor(
        [effective_program_len(case) for case in cases],
        dtype=torch.long,
        device=input_ids.device,
    )
    if str(target_shape) == "cumulative":
        halt_targets = active_len_halt_targets(
            active_lengths,
            max_depth=depth,
            min_halt_step=int(min_halt_step),
        )
    else:
        halt_targets = active_len_first_halt_targets(
            active_lengths,
            max_depth=depth,
            min_halt_step=int(min_halt_step),
        )
    if halt_logits.shape != halt_targets.shape:
        raise ValueError("halt logits and active-length halt targets must match")
    return F.binary_cross_entropy_with_logits(halt_logits.float(), halt_targets.float())


def halt_loss_input_for_context(
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    context: str,
) -> torch.Tensor:
    return halt_loss_inputs_for_context(
        input_ids,
        prompt_len=prompt_len,
        context=context,
    )[0]


def halt_loss_inputs_for_context(
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    context: str,
) -> list[torch.Tensor]:
    if str(context) == "prompt":
        return [input_ids[:, : int(prompt_len)]]
    if str(context) == "full":
        return [input_ids]
    if str(context) == "prefixes":
        return [
            input_ids[:, :end]
            for end in range(int(prompt_len), int(input_ids.shape[1]) + 1)
        ]
    raise ValueError(f"unsupported halt loss context: {context}")


@torch.no_grad()
def generate_answer(
    model,
    prompt_ids: torch.Tensor,
    *,
    answer_len: int,
    think_steps: int,
    state_reset_each_step: bool = False,
    thinking_block_off: bool = False,
    coupling_off: bool = False,
    z_l_zero: bool = False,
    z_h_zero: bool = False,
    carrier_off: bool = False,
    op_order_off: bool = False,
    adaptive_halt: bool = False,
    halt_threshold: float = 0.5,
    halt_min_steps: int = 1,
) -> list[int]:
    return generate_answer_with_runtime(
        model,
        prompt_ids,
        answer_len=answer_len,
        think_steps=think_steps,
        state_reset_each_step=state_reset_each_step,
        thinking_block_off=thinking_block_off,
        coupling_off=coupling_off,
        z_l_zero=z_l_zero,
        z_h_zero=z_h_zero,
        carrier_off=carrier_off,
        op_order_off=op_order_off,
        adaptive_halt=adaptive_halt,
        halt_threshold=halt_threshold,
        halt_min_steps=halt_min_steps,
    )["token_ids"]


@torch.no_grad()
def generate_answer_with_runtime(
    model,
    prompt_ids: torch.Tensor,
    *,
    answer_len: int,
    think_steps: int,
    state_reset_each_step: bool = False,
    thinking_block_off: bool = False,
    coupling_off: bool = False,
    z_l_zero: bool = False,
    z_h_zero: bool = False,
    carrier_off: bool = False,
    op_order_off: bool = False,
    adaptive_halt: bool = False,
    halt_threshold: float = 0.5,
    halt_min_steps: int = 1,
) -> dict[str, object]:
    out = prompt_ids.clone()
    halt_steps: list[float] = []
    executed_steps: list[float] = []
    halted_fractions: list[float] = []
    for _ in range(int(answer_len)):
        if bool(adaptive_halt):
            runtime_out = model.forward_with_runtime(
                out,
                think_steps=int(think_steps),
                state_reset_each_step=bool(state_reset_each_step),
                thinking_block_off=bool(thinking_block_off),
                coupling_off=bool(coupling_off),
                z_l_zero=bool(z_l_zero),
                z_h_zero=bool(z_h_zero),
                carrier_off=bool(carrier_off),
                op_order_off=bool(op_order_off),
                adaptive_halt=True,
                halt_threshold=float(halt_threshold),
                halt_min_steps=int(halt_min_steps),
            )
            logits = runtime_out["logits"]
            halt_steps.append(float(runtime_out["halt_steps"].detach().float().mean().cpu()))
            executed_steps.append(
                float(runtime_out["executed_think_steps"].detach().float().cpu())
            )
            halted_fractions.append(
                float(runtime_out["core_halted"].detach().float().mean().cpu())
            )
        else:
            logits = model(
                out,
                think_steps=int(think_steps),
                state_reset_each_step=bool(state_reset_each_step),
                thinking_block_off=bool(thinking_block_off),
                coupling_off=bool(coupling_off),
                z_l_zero=bool(z_l_zero),
                z_h_zero=bool(z_h_zero),
                carrier_off=bool(carrier_off),
                op_order_off=bool(op_order_off),
            )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
    result: dict[str, object] = {
        "token_ids": out[0, int(prompt_ids.shape[1]) :].detach().cpu().tolist()
    }
    if halt_steps:
        result.update(
            {
                "mean_halt_steps": float(sum(halt_steps) / len(halt_steps)),
                "mean_executed_think_steps": float(
                    sum(executed_steps) / len(executed_steps)
                ),
                "mean_halted_fraction": float(
                    sum(halted_fractions) / len(halted_fractions)
                ),
            }
        )
    return result


@torch.no_grad()
def generate_answer_beam(
    model,
    prompt_ids: torch.Tensor,
    *,
    answer_len: int,
    think_steps: int,
    beam_width: int,
) -> list[dict[str, object]]:
    beams: list[tuple[torch.Tensor, float]] = [(prompt_ids.clone(), 0.0)]
    width = max(1, int(beam_width))
    for _ in range(int(answer_len)):
        expanded: list[tuple[torch.Tensor, float]] = []
        for tokens, score in beams:
            logits = model(tokens, think_steps=int(think_steps))
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            values, indexes = torch.topk(log_probs, k=min(width, log_probs.shape[-1]), dim=-1)
            for value, index in zip(values[0], indexes[0]):
                next_id = index.view(1, 1)
                expanded.append(
                    (
                        torch.cat([tokens, next_id], dim=1),
                        float(score + float(value.detach().cpu())),
                    )
                )
        expanded.sort(key=lambda item: item[1], reverse=True)
        beams = expanded[:width]
    return [
        {
            "token_ids": tokens[0, int(prompt_ids.shape[1]) :].detach().cpu().tolist(),
            "score": score,
        }
        for tokens, score in beams
    ]


@torch.no_grad()
def evaluate(
    model,
    cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
    think_steps: int,
    ablation: str = "none",
) -> dict[str, object]:
    model.eval()
    device = torch.device(args.device)
    prompt_cases = cases
    if ablation == "op_zero":
        prompt_cases = [zero_ops_case(case, modulus=int(args.modulus)) for case in cases]
    state_reset = ablation == "state_reset"
    thinking_off = ablation == "thinking_block_off"
    coupling_off = ablation == "coupling_off"
    z_l_zero = ablation == "z_l_zero"
    z_h_zero = ablation == "z_h_zero"
    carrier_off = ablation == "carrier_off"
    op_order_off = ablation == "op_order_off"
    adaptive_halt = ablation == "adaptive_halt"
    route_force: int | None = None
    if ablation == "order_route0":
        route_force = 0
    elif ablation == "order_route1":
        route_force = 1
    route_force_attr = "trm_order_router_force_route"
    if route_force is not None:
        _router, route_force_attr = _order_router_module_and_force_attr(model)
    had_route_force = hasattr(model, route_force_attr)
    old_route_force = getattr(model, route_force_attr, None)
    if route_force is not None:
        setattr(model, route_force_attr, route_force)
    include_family_tag = include_family_tag_for_args(args)
    state_anchor = state_anchor_for_args(args)
    state_anchor_position = state_anchor_position_for_args(args)
    input_ids, targets, prompt_len, answer_len = cases_to_batch(
        cases,
        tokenizer=tokenizer,
        device=device,
        prompt_cases=prompt_cases,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    halt_runtime: dict[str, torch.Tensor] = {}
    if adaptive_halt:
        runtime_out = model.forward_with_runtime(
            input_ids,
            think_steps=int(think_steps),
            state_reset_each_step=state_reset,
            thinking_block_off=thinking_off,
            coupling_off=coupling_off,
            z_l_zero=z_l_zero,
            z_h_zero=z_h_zero,
            carrier_off=carrier_off,
            op_order_off=op_order_off,
            adaptive_halt=True,
            halt_threshold=float(args.halt_threshold),
            halt_min_steps=int(args.halt_min_steps),
        )
        logits = runtime_out["logits"]
        halt_runtime = {
            key: value
            for key, value in runtime_out.items()
            if key != "logits" and isinstance(value, torch.Tensor)
        }
    else:
        logits = model(
            input_ids,
            think_steps=int(think_steps),
            state_reset_each_step=state_reset,
            thinking_block_off=thinking_off,
            coupling_off=coupling_off,
            z_l_zero=z_l_zero,
            z_h_zero=z_h_zero,
            carrier_off=carrier_off,
            op_order_off=op_order_off,
        )
    loss = answer_text_loss(
        logits,
        targets,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
    answer_start = int(prompt_len) - 1
    answer_end = answer_start + int(answer_len)
    teacher_forced_pred = logits[:, answer_start:answer_end, :].argmax(dim=-1)
    teacher_forced_targets = targets[:, answer_start:answer_end]
    teacher_forced_answer_logits = logits[:, answer_start:answer_end, :]
    teacher_forced_token_accuracy = (
        teacher_forced_pred.eq(teacher_forced_targets).detach().float().mean()
    )
    teacher_forced_sequence_exact = (
        teacher_forced_pred.eq(teacher_forced_targets).all(dim=1).detach().float().mean()
    )
    target_answer_logits = teacher_forced_answer_logits.gather(
        dim=-1,
        index=teacher_forced_targets.unsqueeze(-1),
    ).squeeze(-1)
    teacher_forced_token_ranks = (
        teacher_forced_answer_logits.gt(target_answer_logits.unsqueeze(-1))
        .sum(dim=-1)
        .add(1)
        .detach()
        .float()
    )
    correct = 0
    format_valid = 0
    by_family_counts: dict[str, dict[str, int]] = {}
    by_active_len_counts: dict[int, dict[str, int]] = {}
    examples: list[dict[str, object]] = []
    predictions: list[str] = []
    beam_oracle_correct = 0
    beam_evaluated = 0
    generation_halt_steps: list[float] = []
    generation_executed_steps: list[float] = []
    generation_halted_fractions: list[float] = []
    for index, case in enumerate(cases):
        prompt = case_prompt(
            prompt_cases[index],
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device=device,
        )
        generated = generate_answer_with_runtime(
            model,
            prompt_ids,
            answer_len=answer_len,
            think_steps=int(think_steps),
            state_reset_each_step=state_reset,
            thinking_block_off=thinking_off,
            coupling_off=coupling_off,
            z_l_zero=z_l_zero,
            z_h_zero=z_h_zero,
            carrier_off=carrier_off,
            op_order_off=op_order_off,
            adaptive_halt=adaptive_halt,
            halt_threshold=float(args.halt_threshold),
            halt_min_steps=int(args.halt_min_steps),
        )
        pred_ids = generated["token_ids"]
        if adaptive_halt:
            generation_halt_steps.append(float(generated["mean_halt_steps"]))
            generation_executed_steps.append(float(generated["mean_executed_think_steps"]))
            generation_halted_fractions.append(float(generated["mean_halted_fraction"]))
        pred = tokenizer.decode(pred_ids)
        predictions.append(pred)
        gold = case_answer(case)
        if int(args.eval_beam_width) > 1 and ablation == "none" and not adaptive_halt:
            beam = generate_answer_beam(
                model,
                prompt_ids,
                answer_len=answer_len,
                think_steps=int(think_steps),
                beam_width=int(args.eval_beam_width),
            )
            beam_texts = [tokenizer.decode(item["token_ids"]) for item in beam]
            beam_oracle_correct += int(gold in beam_texts)
            beam_evaluated += 1
        exact = pred == gold
        pred_format_valid = answer_format_valid(pred)
        correct += int(exact)
        format_valid += int(pred_format_valid)
        family_counts = by_family_counts.setdefault(
            str(case.family),
            {"correct": 0, "format_valid": 0, "total": 0},
        )
        family_counts["correct"] += int(exact)
        family_counts["format_valid"] += int(pred_format_valid)
        family_counts["total"] += 1
        active_len = effective_program_len(case)
        active_counts = by_active_len_counts.setdefault(
            int(active_len),
            {"correct": 0, "format_valid": 0, "total": 0},
        )
        active_counts["correct"] += int(exact)
        active_counts["format_valid"] += int(pred_format_valid)
        active_counts["total"] += 1
        if len(examples) < int(args.max_examples):
            examples.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "prompt": prompt,
                    "gold": gold,
                    "pred": pred,
                    "exact": bool(exact),
                    "format_valid": bool(pred_format_valid),
                }
            )
    result = {
        "ablation": ablation,
        "think_steps": int(think_steps),
        "cases": len(cases),
        "generation_exact": float(correct / max(1, len(cases))),
        "generation_format_valid": float(format_valid / max(1, len(cases))),
        "by_family": {
            family: {
                "correct": counts["correct"],
                "format_valid": counts["format_valid"],
                "total": counts["total"],
                "generation_exact": float(counts["correct"] / max(1, counts["total"])),
                "generation_format_valid": float(
                    counts["format_valid"] / max(1, counts["total"])
                ),
            }
            for family, counts in sorted(by_family_counts.items())
        },
        "by_active_len": {
            str(active_len): {
                "correct": counts["correct"],
                "format_valid": counts["format_valid"],
                "total": counts["total"],
                "generation_exact": float(counts["correct"] / max(1, counts["total"])),
                "generation_format_valid": float(
                    counts["format_valid"] / max(1, counts["total"])
                ),
            }
            for active_len, counts in sorted(by_active_len_counts.items())
        },
        "teacher_forced_answer_loss": float(loss.detach().cpu()),
        "teacher_forced_token_accuracy": float(
            teacher_forced_token_accuracy.detach().cpu()
        ),
        "teacher_forced_sequence_exact": float(
            teacher_forced_sequence_exact.detach().cpu()
        ),
        "teacher_forced_mean_token_rank": float(
            teacher_forced_token_ranks.mean().detach().cpu()
        ),
        "teacher_forced_token_top3": float(
            teacher_forced_token_ranks.le(3).float().mean().detach().cpu()
        ),
        "teacher_forced_token_top5": float(
            teacher_forced_token_ranks.le(5).float().mean().detach().cpu()
        ),
        "examples": examples,
    }
    if beam_evaluated > 0:
        result[f"beam{int(args.eval_beam_width)}_oracle_exact"] = float(
            beam_oracle_correct / max(1, beam_evaluated)
        )
    if (
        bool(args.eval_answer_space_argmax)
        and ablation == "none"
        and not adaptive_halt
    ):
        result.update(
            answer_space_argmax_metrics(
                model,
                cases,
                tokenizer=tokenizer,
                device=device,
                prompt_cases=prompt_cases,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                think_steps=int(think_steps),
                modulus=int(args.modulus),
                max_candidate_batch=int(args.eval_answer_space_argmax_batch_size),
            )
        )
    if bool(args.eval_operation_breakdown):
        result["operation_breakdown"] = generation_operation_breakdown(
            cases,
            predictions,
            modulus=int(args.modulus),
        )
    if halt_runtime:
        halt_steps = halt_runtime["halt_steps"].detach().float().cpu()
        core_halted = halt_runtime["core_halted"].detach().float().cpu()
        active_lengths = [effective_program_len(case) for case in cases]
        halt_by_active_len: dict[str, dict[str, object]] = {}
        for active_len in sorted(set(active_lengths)):
            indexes = [
                index for index, item in enumerate(active_lengths) if item == active_len
            ]
            if not indexes:
                continue
            index_tensor = torch.tensor(indexes, dtype=torch.long)
            halt_by_active_len[str(active_len)] = {
                "count": len(indexes),
                "mean_halt_steps": float(halt_steps[index_tensor].mean()),
                "halted_fraction": float(core_halted[index_tensor].mean()),
            }
        result.update(
            {
                "mean_halt_steps": float(halt_steps.mean()),
                "executed_think_steps": int(halt_runtime["executed_think_steps"].item()),
                "halted_fraction": float(core_halted.mean()),
                "core_q_halt_shape": list(halt_runtime["core_q_halt_logits"].shape),
                "halt_by_active_len": halt_by_active_len,
            }
        )
    if generation_halt_steps:
        result.update(
            {
                "generation_mean_halt_steps": float(
                    sum(generation_halt_steps) / len(generation_halt_steps)
                ),
                "generation_mean_executed_think_steps": float(
                    sum(generation_executed_steps) / len(generation_executed_steps)
                ),
                "generation_mean_halted_fraction": float(
                    sum(generation_halted_fractions) / len(generation_halted_fractions)
                ),
            }
        )
    if route_force is not None:
        if had_route_force:
            setattr(model, route_force_attr, old_route_force)
        else:
            delattr(model, route_force_attr)
    return result


def depth_sweep_summary(
    sweep_metrics: dict[str, dict[str, object]],
    *,
    full_depth: int,
) -> dict[str, object]:
    exact_by_depth: dict[str, float] = {}
    format_by_depth: dict[str, float] = {}
    for name, metrics in sorted(
        sweep_metrics.items(),
        key=lambda item: int(str(item[0]).replace("think", "")),
    ):
        depth = str(name).replace("think", "")
        exact_by_depth[depth] = float(metrics.get("generation_exact", 0.0))
        format_by_depth[depth] = float(metrics.get("generation_format_valid", 0.0))
    by_active_len: dict[str, dict[str, object]] = {}
    active_len_keys = sorted(
        {
            str(active_len)
            for metrics in sweep_metrics.values()
            if isinstance(metrics.get("by_active_len"), dict)
            for active_len in metrics["by_active_len"].keys()
        },
        key=lambda value: int(value),
    )
    for active_len in active_len_keys:
        active_exact_by_depth: dict[str, float] = {}
        for name, metrics in sorted(
            sweep_metrics.items(),
            key=lambda item: int(str(item[0]).replace("think", "")),
        ):
            active_metrics = metrics.get("by_active_len")
            if not isinstance(active_metrics, dict) or active_len not in active_metrics:
                continue
            depth = str(name).replace("think", "")
            active_exact_by_depth[depth] = float(
                active_metrics[active_len].get("generation_exact", 0.0)
            )
        if not active_exact_by_depth:
            continue
        active_best_depth = max(
            active_exact_by_depth, key=lambda depth: active_exact_by_depth[depth]
        )
        by_active_len[active_len] = {
            "exact_by_depth": active_exact_by_depth,
            "best_depth": int(active_best_depth),
            "best_generation_exact": float(active_exact_by_depth[active_best_depth]),
        }
    best_depth = max(exact_by_depth, key=lambda depth: exact_by_depth[depth])
    full_key = str(int(full_depth))
    full_exact = float(exact_by_depth.get(full_key, 0.0))
    best_exact = float(exact_by_depth[best_depth])
    return {
        "exact_by_depth": exact_by_depth,
        "format_valid_by_depth": format_by_depth,
        "best_depth": int(best_depth),
        "best_generation_exact": best_exact,
        "full_depth": int(full_depth),
        "full_generation_exact": full_exact,
        "best_minus_full": float(best_exact - full_exact),
        "by_active_len": by_active_len,
    }


def evaluate_depth_sweep(
    model,
    cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
    existing_metrics: dict[str, object],
    max_depth: int,
) -> dict[str, object]:
    sweep_metrics: dict[str, dict[str, object]] = {}
    for depth in range(0, max(0, int(max_depth)) + 1):
        name = f"think{depth}"
        existing = existing_metrics.get(name)
        if isinstance(existing, dict):
            sweep_metrics[name] = existing
            continue
        sweep_metrics[name] = evaluate(
            model,
            cases,
            args,
            tokenizer=tokenizer,
            think_steps=depth,
        )
    return {
        "summary": depth_sweep_summary(sweep_metrics, full_depth=int(max_depth)),
        "metrics": sweep_metrics,
    }


def state_trace_metrics(
    model,
    cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
    think_steps: int,
) -> dict[str, object]:
    model.eval()
    device = torch.device(args.device)
    include_family_tag = include_family_tag_for_args(args)
    state_anchor = state_anchor_for_args(args)
    state_anchor_position = state_anchor_position_for_args(args)
    prompts = [
        case_prompt(
            case,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        for case in cases
    ]
    input_ids = torch.tensor(
        [tokenizer.encode(prompt) for prompt in prompts],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=int(think_steps),
            return_state_trace=True,
        )

    def indexes_by_family() -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for index, case in enumerate(cases):
            grouped.setdefault(str(case.family), []).append(index)
        return {family: indexes for family, indexes in sorted(grouped.items())}

    def summarize(trace: torch.Tensor) -> dict[str, object]:
        if int(trace.shape[1]) == 0:
            return {
                "depths": 0,
                "cases": int(trace.shape[0]),
                "mean_batch_variance_by_depth": [],
                "mean_step_delta_norm": [],
                "mean_consecutive_cosine": [],
            }
        last_token_trace = trace[:, :, -1, :].detach().float().cpu()
        variance_by_depth = last_token_trace.var(dim=0, unbiased=False).mean(dim=-1)
        if int(last_token_trace.shape[1]) > 1:
            deltas = last_token_trace[:, 1:, :] - last_token_trace[:, :-1, :]
            delta_norm = deltas.norm(dim=-1).mean(dim=0)
            cosine = F.cosine_similarity(
                last_token_trace[:, 1:, :],
                last_token_trace[:, :-1, :],
                dim=-1,
            ).mean(dim=0)
        else:
            delta_norm = torch.empty(0)
            cosine = torch.empty(0)
        return {
            "depths": int(last_token_trace.shape[1]),
            "cases": int(last_token_trace.shape[0]),
            "mean_batch_variance_by_depth": [
                float(value) for value in variance_by_depth.tolist()
            ],
            "mean_step_delta_norm": [float(value) for value in delta_norm.tolist()],
            "mean_consecutive_cosine": [float(value) for value in cosine.tolist()],
        }

    result: dict[str, object] = {}
    if "core_state_trace_h" in runtime:
        trace_h = runtime["core_state_trace_h"]
        result["z_h"] = summarize(trace_h)
        result["z_h_by_family"] = {
            family: summarize(trace_h[indexes])
            for family, indexes in indexes_by_family().items()
        }
    if "core_state_trace_l" in runtime:
        trace_l = runtime["core_state_trace_l"]
        result["z_l"] = summarize(trace_l)
        result["z_l_by_family"] = {
            family: summarize(trace_l[indexes])
            for family, indexes in indexes_by_family().items()
        }
    return result


def periodic_eval_score(
    record: dict[str, object],
    *,
    mode: str = "strict",
) -> tuple[float, ...]:
    fixed_exact = float(record["generation_exact"])
    fixed_min_active_len = float(record.get("min_active_len_generation_exact", fixed_exact))
    fixed_min_family = float(record.get("min_family_generation_exact", fixed_exact))
    if "adaptive_halt_generation_exact" in record:
        adaptive_exact = float(record["adaptive_halt_generation_exact"])
        adaptive_min_active_len = float(
            record.get("adaptive_halt_min_active_len_generation_exact", adaptive_exact)
        )
        adaptive_min_family = float(
            record.get("adaptive_halt_min_family_generation_exact", adaptive_exact)
        )
        adaptive_drop = max(0.0, fixed_exact - adaptive_exact)
        mean_halt_steps = float(record.get("adaptive_halt_mean_steps", 1e9))
        strict_score = (
            min(fixed_exact, adaptive_exact),
            min(fixed_min_active_len, adaptive_min_active_len),
            adaptive_exact,
            fixed_exact,
            -adaptive_drop,
            -mean_halt_steps,
            float(record["teacher_forced_sequence_exact"]),
            -float(record["teacher_forced_mean_token_rank"]),
            -float(record["teacher_forced_answer_loss"]),
        )
        if str(mode) == "active_floor":
            return (
                min(fixed_min_active_len, adaptive_min_active_len),
                min(fixed_exact, adaptive_exact),
                adaptive_exact,
                fixed_exact,
                -adaptive_drop,
                -mean_halt_steps,
                float(record["teacher_forced_sequence_exact"]),
                -float(record["teacher_forced_mean_token_rank"]),
                -float(record["teacher_forced_answer_loss"]),
            )
        if str(mode) == "family_floor":
            return (
                min(fixed_min_family, adaptive_min_family),
                min(fixed_exact, adaptive_exact),
                min(fixed_min_active_len, adaptive_min_active_len),
                adaptive_exact,
                fixed_exact,
                -adaptive_drop,
                -mean_halt_steps,
                float(record["teacher_forced_sequence_exact"]),
                -float(record["teacher_forced_mean_token_rank"]),
                -float(record["teacher_forced_answer_loss"]),
            )
        return strict_score
    strict_score = (
        fixed_exact,
        fixed_min_active_len,
        fixed_min_family,
        float(record["teacher_forced_sequence_exact"]),
        -float(record["teacher_forced_mean_token_rank"]),
        -float(record["teacher_forced_answer_loss"]),
    )
    if str(mode) == "active_floor":
        return (
            fixed_min_active_len,
            fixed_exact,
            float(record["teacher_forced_sequence_exact"]),
            -float(record["teacher_forced_mean_token_rank"]),
            -float(record["teacher_forced_answer_loss"]),
        )
    if str(mode) == "family_floor":
        return (
            fixed_min_family,
            fixed_exact,
            fixed_min_active_len,
            float(record["teacher_forced_sequence_exact"]),
            -float(record["teacher_forced_mean_token_rank"]),
            -float(record["teacher_forced_answer_loss"]),
        )
    return strict_score


def min_active_len_generation_exact(metrics: dict[str, object]) -> float:
    by_active_len = metrics.get("by_active_len", {})
    if not isinstance(by_active_len, dict) or not by_active_len:
        return float(metrics.get("generation_exact", 0.0))
    values = [
        float(row.get("generation_exact", 0.0))
        for row in by_active_len.values()
        if isinstance(row, dict)
    ]
    return min(values) if values else float(metrics.get("generation_exact", 0.0))


def min_family_generation_exact(metrics: dict[str, object]) -> float:
    by_family = metrics.get("by_family", {})
    if not isinstance(by_family, dict) or not by_family:
        return float(metrics.get("generation_exact", 0.0))
    values = [
        float(row.get("generation_exact", 0.0))
        for row in by_family.values()
        if isinstance(row, dict)
    ]
    return min(values) if values else float(metrics.get("generation_exact", 0.0))


def build_periodic_eval_record(
    model: torch.nn.Module,
    periodic_eval_cases: list[TextReasoningCase],
    args: argparse.Namespace,
    *,
    tokenizer: CharTokenizer,
    step: int,
) -> dict[str, object]:
    with torch.no_grad():
        probe_metrics = evaluate(
            model,
            periodic_eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
        )
    record: dict[str, object] = {
        "step": int(step),
        "generation_exact": float(probe_metrics["generation_exact"]),
        "min_active_len_generation_exact": min_active_len_generation_exact(
            probe_metrics
        ),
        "min_family_generation_exact": min_family_generation_exact(probe_metrics),
        "teacher_forced_sequence_exact": float(
            probe_metrics["teacher_forced_sequence_exact"]
        ),
        "teacher_forced_mean_token_rank": float(
            probe_metrics["teacher_forced_mean_token_rank"]
        ),
        "teacher_forced_answer_loss": float(
            probe_metrics["teacher_forced_answer_loss"]
        ),
    }
    if bool(args.adaptive_halt_eval):
        adaptive_probe_metrics = evaluate(
            model,
            periodic_eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="adaptive_halt",
        )
        adaptive_mean_steps = float(
            adaptive_probe_metrics.get(
                "generation_mean_halt_steps",
                adaptive_probe_metrics.get("mean_halt_steps", 1e9),
            )
        )
        adaptive_halted_fraction = float(
            adaptive_probe_metrics.get(
                "generation_mean_halted_fraction",
                adaptive_probe_metrics.get("halted_fraction", 0.0),
            )
        )
        adaptive_exact = float(adaptive_probe_metrics["generation_exact"])
        record.update(
            {
                "adaptive_halt_generation_exact": adaptive_exact,
                "adaptive_halt_min_active_len_generation_exact": (
                    min_active_len_generation_exact(adaptive_probe_metrics)
                ),
                "adaptive_halt_min_family_generation_exact": (
                    min_family_generation_exact(adaptive_probe_metrics)
                ),
                "fixed_minus_adaptive_halt": float(
                    record["generation_exact"] - adaptive_exact
                ),
                "adaptive_halt_mean_steps": adaptive_mean_steps,
                "adaptive_halt_halted_fraction": adaptive_halted_fraction,
            }
        )
    return record


def cpu_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def write_checkpoint_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def save_training_checkpoint(
    *,
    out_dir: Path,
    filename: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
    step: int,
    last_loss: float,
    last_lr: float,
    periodic_eval_records: list[dict[str, object]],
    best_eval_record: dict[str, object] | None,
    model_state: dict[str, torch.Tensor] | None = None,
    update_latest: bool = True,
) -> None:
    if not str(out_dir):
        return
    payload = {
        "model_state": model_state if model_state is not None else cpu_model_state_dict(model),
        "args": vars(args),
        "chars": tokenizer.chars,
        "checkpoint_step": int(step),
        "last_loss": float(last_loss),
        "last_lr": float(last_lr),
        "periodic_eval": periodic_eval_records,
        "best_periodic_eval": best_eval_record,
    }
    write_checkpoint_atomic(payload, out_dir / filename)
    if bool(update_latest):
        write_checkpoint_atomic(payload, out_dir / "latest.pt")
        (out_dir / "latest_progress.json").write_text(
            json.dumps(
                {
                    "checkpoint": filename,
                    "step": int(step),
                    "last_loss": float(last_loss),
                    "last_lr": float(last_lr),
                    "best_periodic_eval": best_eval_record,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def make_decision(metrics: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    full = metrics[f"think{args.eval_think_steps}"]
    think0 = metrics["think0"]
    ablation_names = list(applicable_ablation_names(str(args.think_structure)))
    if str(args.op_order_embedding_mode) != "none":
        ablation_names.append("op_order_off")
    ablation_exact = {
        name: float(metrics[name]["generation_exact"])
        for name in ablation_names
        if name in metrics
    }
    decision_ablation_exact = {
        name: value
        for name, value in ablation_exact.items()
        if name not in {"carrier_off"}
    }
    full_exact = float(full["generation_exact"])
    think0_exact = float(think0["generation_exact"])
    worst_ablation = (
        max(decision_ablation_exact.values())
        if decision_ablation_exact
        else float("-inf")
    )
    reject_reasons: list[str] = []
    if full_exact < float(args.accept_min_exact):
        reject_reasons.append("full_exact_below_threshold")
    if (full_exact - think0_exact) < float(args.accept_min_depth_gain):
        reject_reasons.append("depth_gain_below_threshold")
    if (full_exact - worst_ablation) < float(args.accept_min_ablation_drop):
        reject_reasons.append("ablation_drop_below_threshold")
    family_metrics = full.get("by_family") if isinstance(full, dict) else None
    min_family_exact = None
    if isinstance(family_metrics, dict) and family_metrics:
        min_family_exact = min(
            float(item["generation_exact"])
            for item in family_metrics.values()
            if isinstance(item, dict) and "generation_exact" in item
        )
    if float(args.accept_min_family_exact) > 0.0:
        if min_family_exact is None:
            reject_reasons.append("missing_family_metrics")
        elif min_family_exact < float(args.accept_min_family_exact):
            reject_reasons.append("family_exact_below_threshold")
    decisive_metrics = {
        "full_generation_exact": full_exact,
        "think0_generation_exact": think0_exact,
        "full_minus_think0": full_exact - think0_exact,
        "full_minus_worst_ablation": full_exact - worst_ablation,
        "min_family_generation_exact": min_family_exact,
    }
    if "carrier_off" in ablation_exact:
        decisive_metrics["full_minus_carrier_off"] = (
            full_exact - float(ablation_exact["carrier_off"])
        )
    decisive_metrics.update(
        {f"{name}_generation_exact": value for name, value in ablation_exact.items()}
    )
    if bool(args.accept_require_adaptive_halt):
        adaptive = metrics.get("adaptive_halt")
        if not isinstance(adaptive, dict):
            reject_reasons.append("missing_adaptive_halt_metrics")
        else:
            adaptive_exact = float(adaptive.get("generation_exact", 0.0))
            adaptive_mean_halt = float(
                adaptive.get(
                    "generation_mean_executed_think_steps",
                    adaptive.get("mean_halt_steps", float("inf")),
                )
            )
            adaptive_halted_fraction = float(
                adaptive.get(
                    "generation_mean_halted_fraction",
                    adaptive.get("halted_fraction", 0.0),
                )
            )
            adaptive_telemetry_source = (
                "generation"
                if "generation_mean_executed_think_steps" in adaptive
                else "teacher_forced"
            )
            exact_drop = full_exact - adaptive_exact
            decisive_metrics.update(
                {
                    "adaptive_halt_generation_exact": adaptive_exact,
                    "full_minus_adaptive_halt": exact_drop,
                    "adaptive_halt_mean_steps": adaptive_mean_halt,
                    "adaptive_halt_halted_fraction": adaptive_halted_fraction,
                    "adaptive_halt_telemetry_source": adaptive_telemetry_source,
                }
            )
            if exact_drop > float(args.accept_max_adaptive_halt_exact_drop):
                reject_reasons.append("adaptive_halt_exact_drop_above_threshold")
            if adaptive_mean_halt > float(args.accept_max_mean_halt_steps):
                reject_reasons.append("adaptive_halt_mean_steps_above_threshold")
            if adaptive_halted_fraction < float(args.accept_min_halted_fraction):
                reject_reasons.append("adaptive_halt_fraction_below_threshold")
    return {
        "accepted": not reject_reasons,
        "decision": str(args.accepted_decision) if not reject_reasons else "rejected",
        "reject_reasons": reject_reasons,
        "decisive_metrics": decisive_metrics,
    }


def train_probe(args: argparse.Namespace) -> dict[str, object]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    train_families = train_families_for_args(args)
    eval_families = eval_families_for_args(args)
    include_family_tag = include_family_tag_for_args(args)
    state_anchor = state_anchor_for_args(args)
    state_anchor_position = state_anchor_position_for_args(args)
    train_cases = build_cases(
        count=int(args.train_cases),
        seed=int(args.seed),
        program_len=int(args.program_len),
        modulus=int(args.modulus),
        families=train_families,
        hard_op_ids=parse_op_ids(str(args.train_hard_op_ids)),
        hard_op_probability=float(args.train_hard_op_probability),
        hard_op_positions=parse_positions(str(args.train_hard_op_positions)),
    )
    if bool(args.eval_family_order_invariant):
        eval_cases = build_family_order_invariant_eval_cases(
            count=int(args.eval_cases),
            seed=int(args.eval_seed),
            program_len=int(args.program_len),
            modulus=int(args.modulus),
            families=eval_families,
        )
    else:
        eval_cases = build_cases(
            count=int(args.eval_cases),
            seed=int(args.eval_seed),
            program_len=int(args.program_len),
            modulus=int(args.modulus),
            families=eval_families,
        )
    if bool(args.eval_active_len_cycle):
        eval_cases = apply_eval_active_len_cycle(
            eval_cases,
            modulus=int(args.modulus),
            min_active_len=int(args.active_len_cycle_min),
            max_active_len=(
                None
                if int(args.active_len_cycle_max) < 0
                else int(args.active_len_cycle_max)
            ),
        )
    tokenizer = CharTokenizer.from_texts(
        [
            case_full_text(
                case,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            for case in train_cases + eval_cases
        ]
        + [
            case_full_text(
                zero_ops_case(case, modulus=int(args.modulus)),
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
            for case in eval_cases
        ],
        mode=str(args.tokenizer_mode),
        number_max_value=int(args.number_tokenizer_max_value),
        op_role_tokens=bool(args.number_tokenizer_op_role_tokens),
    )
    if bool(args.number_tokenizer_op_role_tokens) and str(args.tokenizer_mode) != "number":
        raise ValueError("--number-tokenizer-op-role-tokens requires --tokenizer-mode number")
    value_token_ids = None
    if str(args.value_codec) == "circular":
        if str(args.tokenizer_mode) != "number":
            raise ValueError("--value-codec circular requires --tokenizer-mode number")
        value_token_ids = value_token_ids_for_tokenizer(
            tokenizer,
            modulus=int(args.modulus),
        )
    op_token_ids = tuple(
        token_id
        for token_id, token in enumerate(tokenizer.chars)
        if token.startswith("op") and len(token) == 4 and token[2:].isdigit()
    )
    input_ids, targets, prompt_len, answer_len = cases_to_batch(
        [train_cases[0]],
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    max_seq_len = int(input_ids.shape[1])
    if int(args.model_max_seq_len) > 0:
        max_seq_len = max(max_seq_len, int(args.model_max_seq_len))
    model = NativeQTRMETDLM(
        vocab=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_kv_heads=int(args.n_kv_heads),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        backbone=str(args.backbone),
        encode_backbone=str(args.encode_backbone or args.backbone),
        think_backbone=str(args.think_backbone or args.backbone),
        decode_backbone=str(args.decode_backbone or args.backbone),
        think_structure=str(args.think_structure),
        trm_l_cycles=int(args.trm_l_cycles),
        trm_no_grad_inner_cycles=not bool(args.trm_full_grad_cycles),
        hybrid_layers=int(args.hybrid_layers),
        attn_every=int(args.attn_every),
        delta_backend=str(args.delta_backend),
        delta_head_dim=int(args.delta_head_dim) if int(args.delta_head_dim) > 0 else None,
        delta_num_v_heads=int(args.delta_num_v_heads) if int(args.delta_num_v_heads) > 0 else None,
        delta_expand_v=float(args.delta_expand_v),
        delta_mode=str(args.delta_mode),
        delta_use_short_conv=not bool(args.delta_no_short_conv),
        delta_conv_size=int(args.delta_conv_size),
        delta_norm_eps=float(args.delta_norm_eps),
        attention_backend=str(args.attention_backend),
        strict_backends=bool(args.strict_backends),
        position_embedding_mode=str(args.position_embedding_mode),
        op_order_embedding_mode=str(args.op_order_embedding_mode),
        op_order_max_positions=int(args.op_order_max_positions),
        op_token_ids=op_token_ids,
        value_codec=str(args.value_codec),
        value_token_ids=value_token_ids,
        halt_pooling=str(args.halt_pooling),
        carrier_gate_init=float(args.carrier_gate_init),
        carrier_state_mode=str(args.carrier_state_mode),
        trm_recurrent_layerscale_mode=str(args.trm_recurrent_layerscale_mode),
        trm_recurrent_layerscale_init=float(args.trm_recurrent_layerscale_init),
    ).to(device)
    resume_load_summary: dict[str, object] | None = None
    if str(args.resume_from):
        checkpoint_path = Path(str(args.resume_from))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_chars = tuple(checkpoint.get("chars", ()))
        if checkpoint_chars and checkpoint_chars != tuple(tokenizer.chars):
            if not bool(args.resume_allow_missing):
                raise ValueError(
                    "resume checkpoint tokenizer chars do not match current tokenizer"
                )
        if bool(args.resume_allow_missing):
            resume_load_summary = load_model_state_flexible(
                model,
                checkpoint["model_state"],
                pos_embed_resize_strategy=str(args.pos_embed_resize_strategy),
                source_chars=checkpoint_chars,
                target_chars=tuple(tokenizer.chars),
            )
        else:
            model.load_state_dict(checkpoint["model_state"])
            resume_load_summary = {"loaded_tensors": len(checkpoint["model_state"])}
    if bool(args.train_only_resume_missing_params):
        if not isinstance(resume_load_summary, dict):
            raise ValueError("--train-only-resume-missing-params requires --resume-from")
        missing_keys = set(str(key) for key in resume_load_summary.get("missing_keys", []))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(str(name) in missing_keys)
    train_param_name_regex = str(args.train_param_name_regex).strip()
    if train_param_name_regex:
        pattern = re.compile(train_param_name_regex)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(bool(pattern.search(str(name))))
    model_backend_summary = backend_summary(model)
    reference_model = None
    reference_checkpoint = str(args.retention_reference_checkpoint).strip()
    if reference_checkpoint:
        if reference_checkpoint == "resume":
            if not str(args.resume_from):
                raise ValueError("--retention-reference-checkpoint resume requires --resume-from")
            reference_checkpoint = ""
        reference_model = copy.deepcopy(model).to(device)
        if reference_checkpoint:
            reference = torch.load(Path(reference_checkpoint), map_location=device)
            reference_chars = tuple(reference.get("chars", ()))
            if reference_chars and reference_chars != tuple(tokenizer.chars):
                raise ValueError(
                    "retention reference checkpoint tokenizer chars do not match"
                )
            reference_model.load_state_dict(reference["model_state"])
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    core_step_codec_head = None
    if float(args.core_step_codec_loss_weight) > 0.0:
        sample_prompt_len = len(
            tokenizer.encode(
                case_prompt(
                    train_cases[0],
                    include_family_tag=include_family_tag,
                    state_anchor=state_anchor,
                    state_anchor_position=state_anchor_position,
                )
            )
        )
        codec_dim = core_step_codec_feature_dim(
            d_model=int(args.d_model),
            prompt_len=sample_prompt_len,
            state_source=str(args.core_step_codec_state_source),
            pooling=str(args.core_step_codec_pooling),
        )
        core_step_codec_head = torch.nn.Linear(codec_dim, int(args.modulus)).to(device)
    core_step_op_codec_head = None
    if float(args.core_step_op_codec_loss_weight) > 0.0:
        sample_prompt_len = len(
            tokenizer.encode(
                case_prompt(
                    train_cases[0],
                    include_family_tag=include_family_tag,
                    state_anchor=state_anchor,
                    state_anchor_position=state_anchor_position,
                )
            )
        )
        op_codec_dim = core_step_codec_feature_dim(
            d_model=int(args.d_model),
            prompt_len=sample_prompt_len,
            state_source=str(args.core_step_op_codec_state_source),
            pooling=str(args.core_step_op_codec_pooling),
        )
        core_step_op_codec_head = torch.nn.Linear(op_codec_dim, len(OP_SPECS)).to(device)
    core_step_position_codec_head = None
    if float(args.core_step_position_codec_loss_weight) > 0.0:
        sample_prompt_len = len(
            tokenizer.encode(
                case_prompt(
                    train_cases[0],
                    include_family_tag=include_family_tag,
                    state_anchor=state_anchor,
                    state_anchor_position=state_anchor_position,
                )
            )
        )
        position_codec_dim = core_step_codec_feature_dim(
            d_model=int(args.d_model),
            prompt_len=sample_prompt_len,
            state_source=str(args.core_step_position_codec_state_source),
            pooling=str(args.core_step_position_codec_pooling),
        )
        core_step_position_codec_head = torch.nn.Linear(
            position_codec_dim,
            int(args.program_len) + 1,
        ).to(device)
    optimizer_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not optimizer_params:
        raise ValueError("no trainable parameters are available")
    optimizer_param_groups: object = optimizer_params
    router_lr_multiplier = float(args.order_router_lr_multiplier)
    if router_lr_multiplier != 1.0 and _has_order_router(model):
        router, _force_attr = _order_router_module_and_force_attr(model)
        router_params = [
            parameter for parameter in router.parameters() if parameter.requires_grad
        ]
        router_param_ids = {id(parameter) for parameter in router_params}
        non_router_params = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in router_param_ids
        ]
        optimizer_params = non_router_params + router_params
        optimizer_param_groups = [
            {"params": non_router_params, "lr": float(args.lr)},
            {"params": router_params, "lr": float(args.lr) * router_lr_multiplier},
        ]
    if core_step_codec_head is not None:
        codec_params = list(core_step_codec_head.parameters())
        optimizer_params.extend(codec_params)
        if isinstance(optimizer_param_groups, list) and optimizer_param_groups and isinstance(
            optimizer_param_groups[0],
            dict,
        ):
            optimizer_param_groups.append({"params": codec_params, "lr": float(args.lr)})
        else:
            optimizer_param_groups = optimizer_params
    if core_step_op_codec_head is not None:
        op_codec_params = list(core_step_op_codec_head.parameters())
        optimizer_params.extend(op_codec_params)
        if isinstance(optimizer_param_groups, list) and optimizer_param_groups and isinstance(
            optimizer_param_groups[0],
            dict,
        ):
            optimizer_param_groups.append({"params": op_codec_params, "lr": float(args.lr)})
        else:
            optimizer_param_groups = optimizer_params
    if core_step_position_codec_head is not None:
        position_codec_params = list(core_step_position_codec_head.parameters())
        optimizer_params.extend(position_codec_params)
        if isinstance(optimizer_param_groups, list) and optimizer_param_groups and isinstance(
            optimizer_param_groups[0],
            dict,
        ):
            optimizer_param_groups.append(
                {"params": position_codec_params, "lr": float(args.lr)}
            )
        else:
            optimizer_param_groups = optimizer_params
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    base_optimizer_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    last_lr = float(args.lr)
    train_active_len_cycle_min = (
        int(args.active_len_cycle_min)
        if int(args.train_active_len_cycle_min) < 0
        else int(args.train_active_len_cycle_min)
    )
    train_active_len_cycle_max = (
        int(args.active_len_cycle_max)
        if int(args.train_active_len_cycle_max) < 0
        else int(args.train_active_len_cycle_max)
    )
    periodic_eval_records: list[dict[str, object]] = []
    best_eval_state: dict[str, torch.Tensor] | None = None
    best_eval_record: dict[str, object] | None = None
    checkpoint_out_dir = Path(args.out_dir)
    periodic_eval_cases = (
        eval_cases
        if int(args.eval_during_training_cases) <= 0
        else eval_cases[: int(args.eval_during_training_cases)]
    )
    if bool(args.eval_initial_checkpoint) and int(args.eval_during_training_every) > 0:
        record = build_periodic_eval_record(
            model,
            periodic_eval_cases,
            args,
            tokenizer=tokenizer,
            step=0,
        )
        record["source"] = "initial_checkpoint"
        periodic_eval_records.append(record)
        print(json.dumps({"periodic_eval": record}, ensure_ascii=False))
        best_eval_record = record
        if bool(args.restore_best_eval_checkpoint):
            best_eval_state = cpu_model_state_dict(model)
            if bool(args.save_best_periodic_checkpoint):
                save_training_checkpoint(
                    out_dir=checkpoint_out_dir,
                    filename="best_periodic.pt",
                    model=model,
                    args=args,
                    tokenizer=tokenizer,
                    step=0,
                    last_loss=0.0,
                    last_lr=last_lr,
                    periodic_eval_records=periodic_eval_records,
                    best_eval_record=best_eval_record,
                    model_state=best_eval_state,
                    update_latest=False,
                )
    sample_cases = train_cases[: max(1, min(int(args.batch_size), len(train_cases)))]
    _sample_x, _sample_y, prompt_len, answer_len = cases_to_batch(
        sample_cases,
        tokenizer=tokenizer,
        device=device,
        include_family_tag=include_family_tag,
        state_anchor=state_anchor,
        state_anchor_position=state_anchor_position,
    )
    del _sample_x, _sample_y
    last_loss = 0.0
    for step in range(1, int(args.steps) + 1):
        lr_scale = lr_scale_for_step(
            step=step,
            total_steps=int(args.steps),
            schedule=str(args.lr_schedule),
            warmup_steps=int(args.lr_warmup_steps),
            min_ratio=float(args.lr_min_ratio),
        )
        for group, base_lr in zip(optimizer.param_groups, base_optimizer_lrs):
            group["lr"] = base_lr * lr_scale
        last_lr = float(optimizer.param_groups[0]["lr"])
        model.train()
        batch = random.sample(train_cases, k=min(int(args.batch_size), len(train_cases)))
        if bool(args.active_len_batch_cycle):
            batch = apply_active_len_batch_cycle(
                batch,
                step=step,
                modulus=int(args.modulus),
                min_active_len=int(train_active_len_cycle_min),
                max_active_len=(
                    None
                    if int(train_active_len_cycle_max) < 0
                    else int(train_active_len_cycle_max)
                ),
            )
        elif bool(args.active_len_curriculum):
            active_len = active_program_len_for_step(
                step=step,
                total_steps=int(args.steps),
                program_len=int(args.program_len),
                min_active_len=int(args.active_len_curriculum_min),
                warmup_fraction=float(args.active_len_curriculum_warmup_frac),
            )
            batch = [
                case_with_active_program_len(
                    case,
                    active_len=active_len,
                    modulus=int(args.modulus),
                )
                for case in batch
            ]
        x, y, prompt_len, answer_len = cases_to_batch(
            batch,
            tokenizer=tokenizer,
            device=device,
            include_family_tag=include_family_tag,
            state_anchor=state_anchor,
            state_anchor_position=state_anchor_position,
        )
        state_trace_runtime: dict[str, torch.Tensor] | None = None
        needs_state_trace_runtime = (
            float(args.state_trace_anti_collapse_loss_weight) > 0.0
            or float(args.state_trace_family_anti_collapse_loss_weight) > 0.0
            or float(args.core_step_codec_loss_weight) > 0.0
            or float(args.core_step_op_codec_loss_weight) > 0.0
            or float(args.core_step_position_codec_loss_weight) > 0.0
            or float(args.latent_refine_loss_weight) > 0.0
            or float(args.state_trace_depth_loss_weight) > 0.0
        )
        if needs_state_trace_runtime:
            state_trace_runtime = model.forward_with_runtime(
                x,
                think_steps=int(args.train_think_steps),
                return_state_trace=True,
            )
            logits = state_trace_runtime["logits"]
        else:
            logits = model(x, think_steps=int(args.train_think_steps))
        loss = answer_text_loss(
            logits,
            y,
            prompt_len=prompt_len,
            answer_len=answer_len,
            loss_type=str(args.answer_loss_type),
        )
        if float(args.family_dro_loss_weight) > 0.0:
            loss = loss + float(args.family_dro_loss_weight) * family_dro_answer_loss(
                logits,
                y,
                batch,
                prompt_len=prompt_len,
                answer_len=answer_len,
                loss_type=str(args.answer_loss_type),
                temperature=float(args.family_dro_temperature),
            )
        if float(args.answer_margin_loss_weight) > 0.0:
            loss = loss + float(args.answer_margin_loss_weight) * answer_margin_loss(
                logits,
                y,
                prompt_len=prompt_len,
                answer_len=answer_len,
                margin=float(args.answer_margin),
            )
        if float(args.sequence_preference_loss_weight) > 0.0:
            loss = loss + float(args.sequence_preference_loss_weight) * sequence_preference_loss(
                model,
                logits,
                y,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                prompt_len=prompt_len,
                answer_len=answer_len,
                think_steps=int(args.train_think_steps),
                modulus=int(args.modulus),
                deltas=parse_preference_deltas(str(args.sequence_preference_deltas)),
                margin=float(args.sequence_preference_margin),
            )
        if operation_counterfactual_schedule_enabled(args, step):
            loss = loss + float(
                args.operation_counterfactual_loss_weight
            ) * operation_counterfactual_loss(
                model,
                logits,
                y,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                prompt_len=prompt_len,
                answer_len=answer_len,
                think_steps=int(args.train_think_steps),
                modulus=int(args.modulus),
                margin=float(args.operation_counterfactual_margin),
                max_cases=int(args.operation_counterfactual_max_cases),
                active_len_min=int(args.operation_counterfactual_active_len_min),
                active_len_max=int(args.operation_counterfactual_active_len_max),
            )
        if (
            float(args.depth_counterfactual_loss_weight) > 0.0
            and (
                int(args.depth_counterfactual_every) <= 1
                or step % int(args.depth_counterfactual_every) == 0
            )
        ):
            loss = loss + float(args.depth_counterfactual_loss_weight) * depth_counterfactual_loss(
                model,
                logits,
                y,
                x,
                prompt_len=prompt_len,
                answer_len=answer_len,
                counterfactual_think_steps=int(args.depth_counterfactual_think_steps),
                margin=float(args.depth_counterfactual_margin),
            )
        if (
            float(args.state_reset_counterfactual_loss_weight) > 0.0
            and (
                int(args.state_reset_counterfactual_every) <= 1
                or step % int(args.state_reset_counterfactual_every) == 0
            )
        ):
            loss = loss + float(
                args.state_reset_counterfactual_loss_weight
            ) * state_reset_counterfactual_loss(
                model,
                logits,
                y,
                x,
                prompt_len=prompt_len,
                answer_len=answer_len,
                think_steps=int(args.train_think_steps),
                margin=float(args.state_reset_counterfactual_margin),
            )
        if (
            float(args.z_l_counterfactual_loss_weight) > 0.0
            and (
                int(args.z_l_counterfactual_every) <= 1
                or step % int(args.z_l_counterfactual_every) == 0
            )
        ):
            loss = loss + float(args.z_l_counterfactual_loss_weight) * z_l_counterfactual_loss(
                model,
                logits,
                y,
                x,
                prompt_len=prompt_len,
                answer_len=answer_len,
                think_steps=int(args.train_think_steps),
                margin=float(args.z_l_counterfactual_margin),
            )
        if (
            float(args.fast_slow_latent_loss_weight) > 0.0
            and (
                int(args.fast_slow_latent_every) <= 1
                or step % int(args.fast_slow_latent_every) == 0
            )
        ):
            loss = loss + float(
                args.fast_slow_latent_loss_weight
            ) * fast_slow_latent_counterfactual_loss(
                model,
                logits,
                y,
                x,
                prompt_len=prompt_len,
                answer_len=answer_len,
                think_steps=int(args.train_think_steps),
                z_l_margin=float(args.fast_slow_z_l_margin),
                z_h_margin=float(args.fast_slow_z_h_margin),
                z_l_weight=float(args.fast_slow_z_l_weight),
                z_h_weight=float(args.fast_slow_z_h_weight),
            )
        if (
            float(args.answer_space_ranking_loss_weight) > 0.0
            and (
                int(args.answer_space_ranking_every) <= 1
                or step % int(args.answer_space_ranking_every) == 0
            )
        ):
            loss = loss + float(
                args.answer_space_ranking_loss_weight
            ) * answer_space_ranking_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                think_steps=int(args.train_think_steps),
                modulus=int(args.modulus),
                max_cases=int(args.answer_space_ranking_max_cases),
                temperature=float(args.answer_space_ranking_temperature),
            )
        if float(args.order_router_aux_loss_weight) > 0.0:
            loss = loss + float(
                args.order_router_aux_loss_weight
            ) * order_router_family_order_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                target_mode=str(args.order_router_aux_target_mode),
            )
        if (
            float(args.forced_route_answer_loss_weight) > 0.0
            and (
                int(args.forced_route_answer_every) <= 1
                or step % int(args.forced_route_answer_every) == 0
            )
        ):
            loss = loss + float(
                args.forced_route_answer_loss_weight
            ) * forced_route_answer_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                route=int(args.forced_route_answer_route),
                families=parse_families(str(args.forced_route_answer_families)),
                think_steps=int(args.train_think_steps),
                loss_type=str(args.answer_loss_type),
                max_cases=int(args.forced_route_answer_max_cases),
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
        if (
            float(args.forced_route_depth_loss_weight) > 0.0
            and (
                int(args.forced_route_depth_every) <= 1
                or step % int(args.forced_route_depth_every) == 0
            )
        ):
            loss = loss + float(
                args.forced_route_depth_loss_weight
            ) * forced_route_intermediate_depth_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                route=int(args.forced_route_depth_route),
                families=parse_families(str(args.forced_route_depth_families)),
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                min_depth=int(args.forced_route_depth_min_depth),
                depth_weight_power=float(args.forced_route_depth_weight_power),
                max_cases=int(args.forced_route_depth_max_cases),
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
        if (
            float(args.forced_route_prefix_depth_anchor_loss_weight) > 0.0
            and (
                int(args.forced_route_prefix_depth_anchor_every) <= 1
                or step % int(args.forced_route_prefix_depth_anchor_every) == 0
            )
        ):
            loss = loss + float(
                args.forced_route_prefix_depth_anchor_loss_weight
            ) * forced_route_prefix_depth_anchor_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                route=int(args.forced_route_prefix_depth_anchor_route),
                families=parse_families(str(args.forced_route_prefix_depth_anchor_families)),
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                min_depth=int(args.forced_route_prefix_depth_anchor_min_depth),
                depth_weight_power=float(
                    args.forced_route_prefix_depth_anchor_weight_power
                ),
                max_cases=int(args.forced_route_prefix_depth_anchor_max_cases),
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
            )
        if (
            float(args.prefix_state_alignment_loss_weight) > 0.0
            and (
                int(args.prefix_state_alignment_every) <= 1
                or step % int(args.prefix_state_alignment_every) == 0
            )
        ):
            loss = loss + float(
                args.prefix_state_alignment_loss_weight
            ) * prefix_state_alignment_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                max_cases=int(args.prefix_state_alignment_max_cases),
            )
        if (
            float(args.prefix_state_contrastive_loss_weight) > 0.0
            and (
                int(args.prefix_state_contrastive_every) <= 1
                or step % int(args.prefix_state_contrastive_every) == 0
            )
        ):
            loss = loss + float(
                args.prefix_state_contrastive_loss_weight
            ) * prefix_state_contrastive_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                max_cases=int(args.prefix_state_contrastive_max_cases),
                temperature=float(args.prefix_state_contrastive_temperature),
                state_source=str(args.prefix_state_contrastive_state_source),
                pooling=str(args.prefix_state_contrastive_pooling),
            )
        if (
            float(args.retention_kl_loss_weight) > 0.0
            and (
                int(args.retention_every) <= 1
                or step % int(args.retention_every) == 0
            )
        ):
            if reference_model is None:
                raise ValueError(
                    "--retention-kl-loss-weight requires --retention-reference-checkpoint"
                )
            loss = loss + float(args.retention_kl_loss_weight) * reference_retention_kl_loss(
                model,
                reference_model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                think_steps=int(args.train_think_steps),
                modulus=int(args.modulus),
                active_len_min=int(args.retention_active_len_min),
                active_len_max=int(args.retention_active_len_max),
                max_cases=int(args.retention_max_cases),
                temperature=float(args.retention_temperature),
            )
        if (
            float(args.active_len_replay_loss_weight) > 0.0
            and (
                int(args.active_len_replay_every) <= 1
                or step % int(args.active_len_replay_every) == 0
            )
        ):
            loss = loss + float(
                args.active_len_replay_loss_weight
            ) * active_len_replay_ce_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                think_steps=int(args.train_think_steps),
                modulus=int(args.modulus),
                active_len_min=int(args.active_len_replay_min),
                active_len_max=int(args.active_len_replay_max),
                max_cases=int(args.active_len_replay_max_cases),
            )
        if (
            float(args.online_greedy_preference_loss_weight) > 0.0
            and (
                int(args.online_greedy_preference_every) <= 1
                or step % int(args.online_greedy_preference_every) == 0
            )
        ):
            loss = loss + float(
                args.online_greedy_preference_loss_weight
            ) * online_greedy_preference_loss(
                model,
                logits,
                y,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                prompt_len=prompt_len,
                answer_len=answer_len,
                think_steps=int(args.train_think_steps),
                margin=float(args.online_greedy_preference_margin),
                max_cases=int(args.online_greedy_preference_max_cases),
            )
        if (
            float(args.state_trace_anti_collapse_loss_weight) > 0.0
            and state_trace_runtime is not None
        ):
            loss = loss + float(
                args.state_trace_anti_collapse_loss_weight
            ) * state_trace_anti_collapse_loss(
                state_trace_runtime,
                min_variance=float(args.state_trace_min_variance),
                min_delta_norm=float(args.state_trace_min_delta_norm),
            )
        if (
            float(args.state_trace_family_anti_collapse_loss_weight) > 0.0
            and state_trace_runtime is not None
        ):
            loss = loss + float(
                args.state_trace_family_anti_collapse_loss_weight
            ) * state_trace_family_anti_collapse_loss(
                state_trace_runtime,
                batch,
                families=parse_families(
                    str(args.state_trace_family_anti_collapse_families)
                ),
                state_source=str(args.state_trace_family_anti_collapse_state_source),
                max_consecutive_cosine=float(
                    args.state_trace_family_max_consecutive_cosine
                ),
                min_final_variance=float(args.state_trace_family_min_final_variance),
                late_fraction=float(args.state_trace_family_late_fraction),
                cosine_loss_scale=float(args.state_trace_family_cosine_loss_scale),
                reduction=str(args.state_trace_family_anti_collapse_reduction),
            )
        if (
            float(args.latent_refine_loss_weight) > 0.0
            and state_trace_runtime is not None
        ):
            loss = loss + float(
                args.latent_refine_loss_weight
            ) * latent_refinement_loss_from_runtime(
                model,
                state_trace_runtime,
                y,
                prompt_len=prompt_len,
                answer_len=answer_len,
                min_depth=int(args.latent_refine_min_depth),
                noise_std=float(args.latent_refine_noise_std),
                depth_weight_power=float(args.latent_refine_depth_weight_power),
                final_kl_weight=float(args.latent_refine_final_kl_weight),
            )
        if (
            float(args.state_trace_depth_loss_weight) > 0.0
            and state_trace_runtime is not None
        ):
            depth_targets = intermediate_answer_targets(
                batch,
                tokenizer=tokenizer,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                device=device,
            )
            loss = loss + float(
                args.state_trace_depth_loss_weight
            ) * state_trace_depth_answer_loss_from_runtime(
                model,
                state_trace_runtime,
                depth_targets,
                batch,
                prompt_len=prompt_len,
                answer_len=answer_len,
                max_depth=int(args.train_think_steps),
                min_depth=int(args.state_trace_depth_min_depth),
                max_depth_samples=int(args.state_trace_depth_max_depth_samples),
                depth_sample_mode=str(args.state_trace_depth_sample_mode),
                depth_weight_power=float(args.state_trace_depth_weight_power),
                state_source=str(args.state_trace_depth_state_source),
                family_dro=bool(args.state_trace_depth_family_dro),
                family_dro_temperature=float(args.state_trace_depth_family_dro_temperature),
            )
        if (
            float(args.core_step_codec_loss_weight) > 0.0
            and state_trace_runtime is not None
            and core_step_codec_head is not None
        ):
            loss = loss + float(
                args.core_step_codec_loss_weight
            ) * core_step_codec_loss_from_runtime(
                state_trace_runtime,
                batch,
                core_step_codec_head,
                prompt_len=prompt_len,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                state_source=str(args.core_step_codec_state_source),
                pooling=str(args.core_step_codec_pooling),
            )
        if (
            float(args.core_step_op_codec_loss_weight) > 0.0
            and state_trace_runtime is not None
            and core_step_op_codec_head is not None
        ):
            loss = loss + float(
                args.core_step_op_codec_loss_weight
            ) * core_step_op_codec_loss_from_runtime(
                state_trace_runtime,
                batch,
                core_step_op_codec_head,
                prompt_len=prompt_len,
                max_depth=int(args.train_think_steps),
                state_source=str(args.core_step_op_codec_state_source),
                pooling=str(args.core_step_op_codec_pooling),
            )
        if (
            float(args.core_step_position_codec_loss_weight) > 0.0
            and state_trace_runtime is not None
            and core_step_position_codec_head is not None
        ):
            loss = loss + float(
                args.core_step_position_codec_loss_weight
            ) * core_step_position_codec_loss_from_runtime(
                state_trace_runtime,
                batch,
                core_step_position_codec_head,
                prompt_len=prompt_len,
                max_depth=int(args.train_think_steps),
                state_source=str(args.core_step_position_codec_state_source),
                pooling=str(args.core_step_position_codec_pooling),
            )
        if float(args.depth_intermediate_loss_weight) > 0.0:
            depth_targets = intermediate_answer_targets(
                batch,
                tokenizer=tokenizer,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                device=device,
            )
            if bool(args.depth_intermediate_family_dro):
                depth_loss = intermediate_depth_family_dro_loss(
                    model,
                    x,
                    depth_targets,
                    batch,
                    prompt_len=prompt_len,
                    answer_len=answer_len,
                    max_depth=int(args.train_think_steps),
                    min_depth=int(args.depth_intermediate_min_depth),
                    depth_weight_power=float(args.depth_intermediate_weight_power),
                    temperature=float(args.depth_intermediate_family_dro_temperature),
                )
            else:
                depth_loss = intermediate_depth_loss(
                    model,
                    x,
                    depth_targets,
                    prompt_len=prompt_len,
                    answer_len=answer_len,
                    max_depth=int(args.train_think_steps),
                    min_depth=int(args.depth_intermediate_min_depth),
                    depth_weight_power=float(args.depth_intermediate_weight_power),
                )
            loss = loss + float(args.depth_intermediate_loss_weight) * depth_loss
        if float(args.prefix_depth_anchor_loss_weight) > 0.0:
            loss = loss + float(args.prefix_depth_anchor_loss_weight) * prefix_depth_anchor_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                max_depth=int(args.train_think_steps),
                modulus=int(args.modulus),
                min_depth=int(args.prefix_depth_anchor_min_depth),
                depth_weight_power=float(args.prefix_depth_anchor_weight_power),
                max_cases=int(args.prefix_depth_anchor_max_cases),
            )
        if float(args.residue_aux_loss_weight) > 0.0:
            loss = loss + float(args.residue_aux_loss_weight) * residue_auxiliary_loss(
                model,
                batch,
                tokenizer=tokenizer,
                device=device,
                include_family_tag=include_family_tag,
                state_anchor=state_anchor,
                state_anchor_position=state_anchor_position,
                think_steps=int(args.train_think_steps),
                residue_moduli=parse_residue_moduli(str(args.residue_aux_moduli)),
            )
        if float(args.halt_depth_final_loss_weight) > 0.0:
            loss = loss + float(args.halt_depth_final_loss_weight) * halt_depth_final_answer_loss(
                model,
                x,
                y,
                batch,
                prompt_len=prompt_len,
                answer_len=answer_len,
                max_depth=int(args.train_think_steps),
                min_halt_step=int(args.halt_min_steps),
            )
        if float(args.adaptive_halt_loss_weight) > 0.0:
            if str(args.adaptive_halt_target_mode) == "active_len":
                halt_inputs = halt_loss_inputs_for_context(
                    x,
                    prompt_len=prompt_len,
                    context=str(args.adaptive_halt_loss_context),
                )
                halt_loss = torch.stack(
                    [
                        adaptive_halt_active_len_loss(
                            model,
                            halt_input,
                            batch,
                            max_depth=int(args.train_think_steps),
                            min_halt_step=int(args.halt_min_steps),
                            target_shape=str(args.adaptive_halt_active_len_target),
                        )
                        for halt_input in halt_inputs
                    ]
                ).mean()
            else:
                halt_loss = adaptive_halt_teacher_depth_loss(
                    model,
                    x,
                    y,
                    prompt_len=prompt_len,
                    answer_len=answer_len,
                    max_depth=int(args.train_think_steps),
                    min_halt_step=int(args.halt_min_steps),
                )
            loss = loss + float(args.adaptive_halt_loss_weight) * halt_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer_params, float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if int(args.log_every) > 0 and (
            step == 1 or step % int(args.log_every) == 0 or step == int(args.steps)
        ):
            print(
                json.dumps(
                    {"step": step, "loss": last_loss, "lr": last_lr},
                    ensure_ascii=False,
                )
            )
        if int(args.eval_during_training_every) > 0 and (
            step % int(args.eval_during_training_every) == 0
            or step == int(args.steps)
        ):
            record = build_periodic_eval_record(
                model,
                periodic_eval_cases,
                args,
                tokenizer=tokenizer,
                step=step,
            )
            periodic_eval_records.append(record)
            print(json.dumps({"periodic_eval": record}, ensure_ascii=False))
            score_mode = str(args.periodic_eval_score_mode)
            if best_eval_record is None or periodic_eval_score(
                record,
                mode=score_mode,
            ) > periodic_eval_score(best_eval_record, mode=score_mode):
                best_eval_record = record
                if bool(args.restore_best_eval_checkpoint):
                    best_eval_state = cpu_model_state_dict(model)
                    if bool(args.save_best_periodic_checkpoint):
                        save_training_checkpoint(
                            out_dir=checkpoint_out_dir,
                            filename="best_periodic.pt",
                            model=model,
                            args=args,
                            tokenizer=tokenizer,
                            step=step,
                            last_loss=last_loss,
                            last_lr=last_lr,
                            periodic_eval_records=periodic_eval_records,
                            best_eval_record=best_eval_record,
                            model_state=best_eval_state,
                            update_latest=False,
                        )
        if int(args.save_every_steps) > 0 and (
            step % int(args.save_every_steps) == 0 or step == int(args.steps)
        ):
            save_training_checkpoint(
                out_dir=checkpoint_out_dir,
                filename=f"checkpoint_step_{step:06d}.pt",
                model=model,
                args=args,
                tokenizer=tokenizer,
                step=step,
                last_loss=last_loss,
                last_lr=last_lr,
                periodic_eval_records=periodic_eval_records,
                best_eval_record=best_eval_record,
            )

    if bool(args.restore_best_eval_checkpoint) and best_eval_state is not None:
        model.load_state_dict(best_eval_state)

    eval_metrics = {
        "think0": evaluate(model, eval_cases, args, tokenizer=tokenizer, think_steps=0),
        "think1": evaluate(model, eval_cases, args, tokenizer=tokenizer, think_steps=1),
        f"think{args.eval_think_steps}": evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
        ),
        "state_reset": evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="state_reset",
        ),
        "op_zero": evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="op_zero",
        ),
        "thinking_block_off": evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="thinking_block_off",
        ),
    }
    if str(args.op_order_embedding_mode) != "none":
        eval_metrics["op_order_off"] = evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="op_order_off",
        )
    for ablation_name in applicable_ablation_names(str(args.think_structure)):
        if ablation_name in eval_metrics:
            continue
        eval_metrics[ablation_name] = evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation=ablation_name,
        )
    if bool(args.adaptive_halt_eval):
        eval_metrics["adaptive_halt"] = evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="adaptive_halt",
        )
    if bool(args.eval_depth_sweep):
        eval_metrics["depth_sweep"] = evaluate_depth_sweep(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            existing_metrics=eval_metrics,
            max_depth=int(args.eval_think_steps),
        )
    if bool(args.eval_state_trace):
        eval_metrics["state_trace"] = state_trace_metrics(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
        )
    if bool(args.eval_core_answer_probe):
        eval_metrics["core_answer_probe"] = core_answer_probe_metrics(
            model,
            train_cases,
            eval_cases,
            args,
            tokenizer=tokenizer,
        )
    if bool(args.eval_core_step_probe):
        eval_metrics["core_step_probe"] = core_step_probe_metrics(
            model,
            train_cases,
            eval_cases,
            args,
            tokenizer=tokenizer,
        )
    if bool(args.eval_order_router_probe):
        eval_metrics["order_router_probe"] = order_router_probe_metrics(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
        )
    if bool(args.eval_order_router_route_ablation):
        eval_metrics["order_route0"] = evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="order_route0",
        )
        eval_metrics["order_route1"] = evaluate(
            model,
            eval_cases,
            args,
            tokenizer=tokenizer,
            think_steps=int(args.eval_think_steps),
            ablation="order_route1",
        )
    decision = make_decision(eval_metrics, args)
    report: dict[str, object] = {
        "status": "complete",
        "target_level": str(args.target_level),
        "train": vars(args),
        "backend_summary": model_backend_summary,
        "train_task_families": train_families,
        "eval_task_families": eval_families,
        "eval_family_order_invariant": bool(args.eval_family_order_invariant),
        "include_family_tag": include_family_tag,
        "vocab_size": tokenizer.vocab_size,
        "prompt_len": prompt_len,
        "answer_len": answer_len,
        "last_loss": last_loss,
        "last_lr": last_lr,
        "periodic_eval": periodic_eval_records,
        "best_periodic_eval": best_eval_record,
        "resume_load_summary": resume_load_summary,
        "restored_best_eval_checkpoint": bool(
            args.restore_best_eval_checkpoint and best_eval_state is not None
        ),
        "eval_metrics": eval_metrics,
        **decision,
    }
    out_dir = Path(args.out_dir)
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "report": report,
                "chars": tokenizer.chars,
            },
            out_dir / "last.pt",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a QTRM-native mixed text reasoning probe."
    )
    parser.add_argument("--out-dir", default="local_eval/qtrm_native_mixed_text_reasoning")
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint path containing model_state from a previous run.",
    )
    parser.add_argument(
        "--resume-allow-missing",
        action="store_true",
        help="Load a compatible checkpoint with strict=False for additive architecture probes.",
    )
    parser.add_argument(
        "--train-only-resume-missing-params",
        action="store_true",
        help="Freeze loaded checkpoint tensors and train only parameters absent from --resume-from.",
    )
    parser.add_argument(
        "--train-param-name-regex",
        default="",
        help="If set, train only model parameters whose names match this Python regex.",
    )
    parser.add_argument("--target-level", default="L4 QTRM-native mixed text reasoning scaffold")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--train-cases", type=int, default=16384)
    parser.add_argument("--eval-cases", type=int, default=512)
    parser.add_argument(
        "--task-families",
        default="modchain",
        help="Comma-separated train task families: modchain,revchain,checksum.",
    )
    parser.add_argument(
        "--eval-task-families",
        default="",
        help="Optional comma-separated eval task families; defaults to task-families.",
    )
    parser.add_argument(
        "--eval-family-order-invariant",
        action="store_true",
        help=(
            "Build held-out eval cases with stable per-family seeds so changing "
            "the eval family order cannot change each family's sampled cases."
        ),
    )
    parser.add_argument(
        "--train-hard-op-ids",
        default="",
        help="Optional comma/space-separated op ids to oversample in train cases.",
    )
    parser.add_argument("--train-hard-op-probability", type=float, default=0.0)
    parser.add_argument(
        "--train-hard-op-positions",
        default="",
        help=(
            "Optional comma/space-separated 1-indexed positions where hard op "
            "oversampling applies; empty means all positions."
        ),
    )
    parser.add_argument("--include-family-tag", action="store_true")
    parser.add_argument(
        "--prompt-state-anchor",
        action="store_true",
        help="Insert a visible fixed prompt anchor before answer as a causal state carrier.",
    )
    parser.add_argument(
        "--prompt-state-anchor-position",
        choices=("before_answer", "after_answer"),
        default="before_answer",
        help="Where to place the visible state anchor when --prompt-state-anchor is set.",
    )
    parser.add_argument("--tokenizer-mode", choices=("char", "number"), default="char")
    parser.add_argument("--number-tokenizer-max-value", type=int, default=99)
    parser.add_argument(
        "--number-tokenizer-op-role-tokens",
        action="store_true",
        help=(
            "In number-tokenizer mode, encode digits inside the visible ops "
            "segment as separate opNN tokens while preserving the decoded prompt."
        ),
    )
    parser.add_argument("--program-len", type=int, default=4)
    parser.add_argument("--modulus", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--backbone", choices=_native.SUPPORTED_BACKBONES, default="mha_etd")
    parser.add_argument("--encode-backbone", choices=("", *_native.SUPPORTED_BACKBONES), default="")
    parser.add_argument("--think-backbone", choices=("", *_native.SUPPORTED_BACKBONES), default="")
    parser.add_argument("--decode-backbone", choices=("", *_native.SUPPORTED_BACKBONES), default="")
    parser.add_argument("--think-structure", choices=_native.SUPPORTED_THINK_STRUCTURES, default="single")
    parser.add_argument("--trm-l-cycles", type=int, default=1)
    parser.add_argument("--trm-full-grad-cycles", action="store_true")
    parser.add_argument("--hybrid-layers", type=int, default=4)
    parser.add_argument("--attn-every", type=int, default=4)
    parser.add_argument("--delta-backend", default="torch_gated_delta")
    parser.add_argument("--delta-head-dim", type=int, default=0)
    parser.add_argument("--delta-num-v-heads", type=int, default=0)
    parser.add_argument("--delta-expand-v", type=float, default=1.0)
    parser.add_argument("--delta-mode", default="chunk")
    parser.add_argument("--delta-no-short-conv", action="store_true")
    parser.add_argument("--delta-conv-size", type=int, default=4)
    parser.add_argument("--delta-norm-eps", type=float, default=1e-6)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument(
        "--position-embedding-mode",
        choices=("learned", "none", "randomized"),
        default="learned",
        help=(
            "Use learned absolute input position embeddings, disable them, "
            "or sample ordered positions from the model context during "
            "training for length-generalization experiments."
        ),
    )
    parser.add_argument(
        "--model-max-seq-len",
        type=int,
        default=0,
        help=(
            "Override model max_seq_len. This is mainly useful with "
            "--position-embedding-mode randomized so short training examples "
            "can sample positions from a longer target context."
        ),
    )
    parser.add_argument(
        "--op-order-embedding-mode",
        choices=("none", "learned"),
        default="none",
        help=(
            "Add a learned operation-index embedding to opXX tokens. This is "
            "a canonical token-embedding path for separating operation order "
            "from absolute token position."
        ),
    )
    parser.add_argument(
        "--op-order-max-positions",
        type=int,
        default=32,
        help="Maximum operation index for learned op-order embeddings.",
    )
    parser.add_argument(
        "--value-codec",
        choices=("learned", "circular"),
        default="learned",
        help=(
            "Use learned value-token embeddings/readout, or a circular latent "
            "value codec for modular synthetic tasks."
        ),
    )
    parser.add_argument("--halt-pooling", choices=("last", "mean", "dedicated"), default="last")
    parser.add_argument(
        "--carrier-gate-init",
        type=float,
        default=-1.0,
        help=(
            "Initial logit for internal carrier residual gates. Lower values "
            "make carrier insertion closer to identity at startup."
        ),
    )
    parser.add_argument(
        "--carrier-state-mode",
        choices=_native.SUPPORTED_CARRIER_STATE_MODES,
        default="gru",
        help=(
            "Carrier state source. Non-GRU modes are deterministic probes for "
            "reducing random-init dependence in additive carrier experiments."
        ),
    )
    parser.add_argument(
        "--trm-recurrent-layerscale-mode",
        choices=("none", "scalar", "channel"),
        default="none",
        help=(
            "Scale nested TRM recurrent updates as previous + scale * "
            "(updated - previous). init=1.0 preserves an existing route; "
            "small init tests identity-biased recurrence."
        ),
    )
    parser.add_argument(
        "--trm-recurrent-layerscale-init",
        type=float,
        default=1.0,
        help="Initial scale for --trm-recurrent-layerscale-mode.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear_warmup_cosine"),
        default="constant",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--answer-loss-type",
        choices=("cross_entropy", "stablemax_cross_entropy"),
        default="cross_entropy",
    )
    parser.add_argument("--family-dro-loss-weight", type=float, default=0.0)
    parser.add_argument("--family-dro-temperature", type=float, default=0.0)
    parser.add_argument("--train-think-steps", type=int, default=4)
    parser.add_argument("--eval-think-steps", type=int, default=4)
    parser.add_argument("--adaptive-halt-eval", action="store_true")
    parser.add_argument("--halt-threshold", type=float, default=0.5)
    parser.add_argument("--halt-min-steps", type=int, default=1)
    parser.add_argument("--adaptive-halt-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-halt-target-mode",
        choices=("correctness", "active_len"),
        default="correctness",
    )
    parser.add_argument(
        "--adaptive-halt-active-len-target",
        choices=("first_step", "cumulative"),
        default="first_step",
    )
    parser.add_argument(
        "--adaptive-halt-loss-context",
        choices=("full", "prompt", "prefixes"),
        default="full",
    )
    parser.add_argument("--depth-intermediate-loss-weight", type=float, default=0.5)
    parser.add_argument("--depth-intermediate-min-depth", type=int, default=1)
    parser.add_argument("--depth-intermediate-weight-power", type=float, default=0.0)
    parser.add_argument("--depth-intermediate-family-dro", action="store_true")
    parser.add_argument("--depth-intermediate-family-dro-temperature", type=float, default=0.0)
    parser.add_argument("--prefix-depth-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--prefix-depth-anchor-min-depth", type=int, default=1)
    parser.add_argument("--prefix-depth-anchor-weight-power", type=float, default=0.0)
    parser.add_argument("--prefix-depth-anchor-max-cases", type=int, default=0)
    parser.add_argument("--residue-aux-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--residue-aux-moduli",
        default="2,4,8",
        help="Comma/space-separated one-digit moduli used by answer2/answer4 tags.",
    )
    parser.add_argument("--halt-depth-final-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-margin", type=float, default=1.0)
    parser.add_argument("--sequence-preference-loss-weight", type=float, default=0.0)
    parser.add_argument("--sequence-preference-deltas", default="2,4,8,16")
    parser.add_argument("--sequence-preference-margin", type=float, default=1.0)
    parser.add_argument("--operation-counterfactual-loss-weight", type=float, default=0.0)
    parser.add_argument("--operation-counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--operation-counterfactual-max-cases",
        type=int,
        default=0,
        help="Maximum batch cases for zero-op counterfactual contrast; 0 means all.",
    )
    parser.add_argument(
        "--operation-counterfactual-every",
        type=int,
        default=1,
        help="Apply zero-op counterfactual contrast every N training steps.",
    )
    parser.add_argument(
        "--operation-counterfactual-warmup-steps",
        type=int,
        default=0,
        help="Do not apply zero-op counterfactual contrast before this step.",
    )
    parser.add_argument(
        "--operation-counterfactual-end-step",
        type=int,
        default=-1,
        help="Stop zero-op counterfactual contrast after this step; -1 keeps it enabled.",
    )
    parser.add_argument(
        "--operation-counterfactual-active-len-min",
        type=int,
        default=1,
        help="Minimum effective active length for zero-op counterfactual contrast.",
    )
    parser.add_argument(
        "--operation-counterfactual-active-len-max",
        type=int,
        default=-1,
        help="Maximum effective active length for zero-op counterfactual contrast; -1 means no max.",
    )
    parser.add_argument("--depth-counterfactual-loss-weight", type=float, default=0.0)
    parser.add_argument("--depth-counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--depth-counterfactual-think-steps",
        type=int,
        default=0,
        help="Shallow think-step count used as the counterfactual depth baseline.",
    )
    parser.add_argument(
        "--depth-counterfactual-every",
        type=int,
        default=1,
        help="Apply full-vs-shallow depth counterfactual every N training steps.",
    )
    parser.add_argument("--state-reset-counterfactual-loss-weight", type=float, default=0.0)
    parser.add_argument("--state-reset-counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--state-reset-counterfactual-every",
        type=int,
        default=1,
        help="Apply full-vs-reset recurrent state counterfactual every N training steps.",
    )
    parser.add_argument("--z-l-counterfactual-loss-weight", type=float, default=0.0)
    parser.add_argument("--z-l-counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--z-l-counterfactual-every",
        type=int,
        default=1,
        help="Apply full-vs-z_L-zero recurrent state counterfactual every N training steps.",
    )
    parser.add_argument(
        "--fast-slow-latent-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Apply Fast-Slow inspired z_L/z_H causal counterfactual pressure; "
            "the final answer still uses the normal LM-logit path."
        ),
    )
    parser.add_argument(
        "--fast-slow-latent-every",
        type=int,
        default=1,
        help="Apply Fast-Slow latent counterfactual pressure every N training steps.",
    )
    parser.add_argument(
        "--fast-slow-z-l-margin",
        type=float,
        default=1.0,
        help="Required log-probability margin over the z_L-zero ablation.",
    )
    parser.add_argument(
        "--fast-slow-z-h-margin",
        type=float,
        default=1.0,
        help="Required log-probability margin over the z_H-zero ablation.",
    )
    parser.add_argument(
        "--fast-slow-z-l-weight",
        type=float,
        default=1.0,
        help="Relative Fast-Slow pressure on the fast z_L state.",
    )
    parser.add_argument(
        "--fast-slow-z-h-weight",
        type=float,
        default=1.0,
        help="Relative Fast-Slow pressure on the slow z_H state.",
    )
    parser.add_argument("--answer-space-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--answer-space-ranking-max-cases",
        type=int,
        default=0,
        help="Maximum batch cases for full answer-space ranking; 0 means all.",
    )
    parser.add_argument(
        "--answer-space-ranking-every",
        type=int,
        default=1,
        help="Apply answer-space ranking every N training steps.",
    )
    parser.add_argument("--answer-space-ranking-temperature", type=float, default=1.0)
    parser.add_argument("--order-router-aux-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--order-router-aux-target-mode",
        choices=("family_order", "chain_vs_checksum"),
        default="family_order",
    )
    parser.add_argument("--order-router-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--forced-route-answer-loss-weight", type=float, default=0.0)
    parser.add_argument("--forced-route-answer-route", type=int, choices=(0, 1), default=1)
    parser.add_argument("--forced-route-answer-families", default="revchain")
    parser.add_argument("--forced-route-answer-max-cases", type=int, default=0)
    parser.add_argument("--forced-route-answer-every", type=int, default=1)
    parser.add_argument("--forced-route-depth-loss-weight", type=float, default=0.0)
    parser.add_argument("--forced-route-depth-route", type=int, choices=(0, 1), default=1)
    parser.add_argument("--forced-route-depth-families", default="revchain")
    parser.add_argument("--forced-route-depth-max-cases", type=int, default=0)
    parser.add_argument("--forced-route-depth-every", type=int, default=1)
    parser.add_argument("--forced-route-depth-min-depth", type=int, default=1)
    parser.add_argument("--forced-route-depth-weight-power", type=float, default=1.0)
    parser.add_argument(
        "--forced-route-prefix-depth-anchor-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--forced-route-prefix-depth-anchor-route",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--forced-route-prefix-depth-anchor-families", default="revchain")
    parser.add_argument("--forced-route-prefix-depth-anchor-max-cases", type=int, default=0)
    parser.add_argument("--forced-route-prefix-depth-anchor-every", type=int, default=1)
    parser.add_argument("--forced-route-prefix-depth-anchor-min-depth", type=int, default=1)
    parser.add_argument(
        "--forced-route-prefix-depth-anchor-weight-power",
        type=float,
        default=1.0,
    )
    parser.add_argument("--prefix-state-alignment-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--prefix-state-alignment-max-cases",
        type=int,
        default=0,
        help="Maximum batch cases for prefix/full recurrent state alignment; 0 means all.",
    )
    parser.add_argument(
        "--prefix-state-alignment-every",
        type=int,
        default=1,
        help="Apply prefix/full recurrent state alignment every N training steps.",
    )
    parser.add_argument("--prefix-state-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--prefix-state-contrastive-max-cases",
        type=int,
        default=0,
        help="Maximum batch cases for contrastive prefix/full state alignment; 0 means all.",
    )
    parser.add_argument(
        "--prefix-state-contrastive-every",
        type=int,
        default=1,
        help="Apply contrastive prefix/full state alignment every N training steps.",
    )
    parser.add_argument("--prefix-state-contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--prefix-state-contrastive-state-source",
        choices=("h", "l", "both"),
        default="both",
    )
    parser.add_argument(
        "--prefix-state-contrastive-pooling",
        choices=("last", "mean", "flatten"),
        default="last",
    )
    parser.add_argument("--retention-reference-checkpoint", default="")
    parser.add_argument("--retention-kl-loss-weight", type=float, default=0.0)
    parser.add_argument("--retention-active-len-min", type=int, default=1)
    parser.add_argument("--retention-active-len-max", type=int, default=-1)
    parser.add_argument("--retention-max-cases", type=int, default=0)
    parser.add_argument("--retention-every", type=int, default=1)
    parser.add_argument("--retention-temperature", type=float, default=1.0)
    parser.add_argument("--active-len-replay-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-len-replay-min", type=int, default=1)
    parser.add_argument("--active-len-replay-max", type=int, default=-1)
    parser.add_argument("--active-len-replay-max-cases", type=int, default=0)
    parser.add_argument("--active-len-replay-every", type=int, default=1)
    parser.add_argument("--online-greedy-preference-loss-weight", type=float, default=0.0)
    parser.add_argument("--online-greedy-preference-margin", type=float, default=1.0)
    parser.add_argument(
        "--online-greedy-preference-max-cases",
        type=int,
        default=0,
        help="Maximum batch cases to mine for online greedy negatives; 0 means all.",
    )
    parser.add_argument(
        "--online-greedy-preference-every",
        type=int,
        default=1,
        help="Apply online greedy preference every N training steps.",
    )
    parser.add_argument("--state-trace-anti-collapse-loss-weight", type=float, default=0.0)
    parser.add_argument("--state-trace-min-variance", type=float, default=0.6)
    parser.add_argument("--state-trace-min-delta-norm", type=float, default=3.5)
    parser.add_argument(
        "--state-trace-family-anti-collapse-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--state-trace-family-anti-collapse-families",
        default="modchain,revchain",
    )
    parser.add_argument(
        "--state-trace-family-anti-collapse-state-source",
        choices=("h", "l", "both"),
        default="h",
    )
    parser.add_argument(
        "--state-trace-family-max-consecutive-cosine",
        type=float,
        default=0.997,
    )
    parser.add_argument(
        "--state-trace-family-min-final-variance",
        type=float,
        default=3.0,
    )
    parser.add_argument("--state-trace-family-late-fraction", type=float, default=0.5)
    parser.add_argument(
        "--state-trace-family-cosine-loss-scale",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--state-trace-family-anti-collapse-reduction",
        choices=("mean", "max"),
        default="max",
    )
    parser.add_argument("--latent-refine-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-refine-min-depth", type=int, default=1)
    parser.add_argument("--latent-refine-noise-std", type=float, default=0.0)
    parser.add_argument("--latent-refine-depth-weight-power", type=float, default=0.0)
    parser.add_argument("--latent-refine-final-kl-weight", type=float, default=0.0)
    parser.add_argument("--state-trace-depth-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--state-trace-depth-state-source",
        choices=("h", "l", "both"),
        default="h",
    )
    parser.add_argument("--state-trace-depth-min-depth", type=int, default=1)
    parser.add_argument("--state-trace-depth-max-depth-samples", type=int, default=0)
    parser.add_argument(
        "--state-trace-depth-sample-mode",
        choices=("uniform", "late"),
        default="uniform",
    )
    parser.add_argument("--state-trace-depth-weight-power", type=float, default=0.0)
    parser.add_argument("--state-trace-depth-family-dro", action="store_true")
    parser.add_argument("--state-trace-depth-family-dro-temperature", type=float, default=0.0)
    parser.add_argument("--core-step-codec-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-step-codec-state-source",
        choices=("h", "l", "both"),
        default="both",
    )
    parser.add_argument(
        "--core-step-codec-pooling",
        choices=("last", "mean", "flatten"),
        default="last",
    )
    parser.add_argument("--core-step-op-codec-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-step-op-codec-state-source",
        choices=("h", "l", "both"),
        default="l",
    )
    parser.add_argument(
        "--core-step-op-codec-pooling",
        choices=("last", "mean", "flatten"),
        default="last",
    )
    parser.add_argument("--core-step-position-codec-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-step-position-codec-state-source",
        choices=("h", "l", "both"),
        default="l",
    )
    parser.add_argument(
        "--core-step-position-codec-pooling",
        choices=("last", "mean", "flatten"),
        default="last",
    )
    parser.add_argument("--active-len-curriculum", action="store_true")
    parser.add_argument("--active-len-curriculum-min", type=int, default=1)
    parser.add_argument("--active-len-curriculum-warmup-frac", type=float, default=0.5)
    parser.add_argument("--active-len-batch-cycle", action="store_true")
    parser.add_argument("--eval-active-len-cycle", action="store_true")
    parser.add_argument("--eval-depth-sweep", action="store_true")
    parser.add_argument("--eval-state-trace", action="store_true")
    parser.add_argument("--eval-operation-breakdown", action="store_true")
    parser.add_argument("--eval-core-answer-probe", action="store_true")
    parser.add_argument("--eval-core-step-probe", action="store_true")
    parser.add_argument("--eval-order-router-probe", action="store_true")
    parser.add_argument("--eval-order-router-route-ablation", action="store_true")
    parser.add_argument(
        "--core-answer-probe-state-source",
        choices=("h", "l", "both"),
        default="h",
    )
    parser.add_argument(
        "--core-answer-probe-pooling",
        choices=("last", "mean", "flatten"),
        default="last",
    )
    parser.add_argument("--core-answer-probe-train-cases", type=int, default=1024)
    parser.add_argument("--core-answer-probe-eval-cases", type=int, default=0)
    parser.add_argument("--core-answer-probe-steps", type=int, default=300)
    parser.add_argument("--core-answer-probe-batch-size", type=int, default=128)
    parser.add_argument("--core-answer-probe-lr", type=float, default=1e-2)
    parser.add_argument("--core-answer-probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-beam-width", type=int, default=1)
    parser.add_argument("--eval-answer-space-argmax", action="store_true")
    parser.add_argument("--eval-answer-space-argmax-batch-size", type=int, default=512)
    parser.add_argument(
        "--pos-embed-resize-strategy",
        choices=("random_tail", "repeat_last", "tail_shift"),
        default="random_tail",
        help=(
            "How to initialize new learned absolute position rows when loading "
            "a shorter checkpoint with --resume-allow-missing."
        ),
    )
    parser.add_argument("--active-len-cycle-min", type=int, default=0)
    parser.add_argument("--active-len-cycle-max", type=int, default=-1)
    parser.add_argument("--train-active-len-cycle-min", type=int, default=-1)
    parser.add_argument("--train-active-len-cycle-max", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=337)
    parser.add_argument("--eval-seed", type=int, default=9337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--eval-during-training-every", type=int, default=0)
    parser.add_argument("--eval-during-training-cases", type=int, default=64)
    parser.add_argument(
        "--periodic-eval-score-mode",
        choices=("strict", "active_floor", "family_floor"),
        default="strict",
    )
    parser.add_argument(
        "--eval-initial-checkpoint",
        action="store_true",
        help=(
            "Include the loaded/random initial model as step 0 in periodic "
            "checkpoint selection."
        ),
    )
    parser.add_argument("--restore-best-eval-checkpoint", action="store_true")
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help=(
            "Write checkpoint_step_XXXXXX.pt and latest.pt every N training "
            "steps so long runs can be resumed after interruption. 0 disables."
        ),
    )
    parser.add_argument(
        "--save-best-periodic-checkpoint",
        action="store_true",
        help=(
            "When periodic checkpoint selection improves and "
            "--restore-best-eval-checkpoint is enabled, write best_periodic.pt."
        ),
    )
    parser.add_argument("--accept-min-exact", type=float, default=0.70)
    parser.add_argument("--accept-min-depth-gain", type=float, default=0.10)
    parser.add_argument("--accept-min-ablation-drop", type=float, default=0.10)
    parser.add_argument("--accept-min-family-exact", type=float, default=0.0)
    parser.add_argument("--accept-require-adaptive-halt", action="store_true")
    parser.add_argument("--accept-max-adaptive-halt-exact-drop", type=float, default=0.0)
    parser.add_argument("--accept-max-mean-halt-steps", type=float, default=4.0)
    parser.add_argument("--accept-min-halted-fraction", type=float, default=0.0)
    parser.add_argument("--accepted-decision", default="accepted_l4_mixed_text_reasoning")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train_probe(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(report["accepted"]) else 1)


if __name__ == "__main__":
    main()
