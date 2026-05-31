from __future__ import annotations

from collections.abc import Sequence

import torch


def build_v2_generation_policy(
    *,
    repetition_penalty: float = 1.0,
    repetition_window: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, str | float | int]:
    legacy_or_stochastic_decode_knob = (
        float(repetition_penalty) > 1.0
        or float(temperature) > 0.0
        or float(top_p) < 1.0
    )
    return {
        "promotion_policy": "free_generation_only",
        "decoding_path": "autoregressive_same_lm_head",
        "promotion_evidence_eligible": "false" if legacy_or_stochastic_decode_knob else "true",
        "repetition_penalty": float(repetition_penalty),
        "repetition_penalty_mode": "diagnostic_only" if float(repetition_penalty) > 1.0 else "disabled",
        "repetition_window": int(repetition_window),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "forbidden": "forced-choice,candidate-rerank,oracle-pass-at-k",
    }


def generation_repetition_stats(token_ids: Sequence[int]) -> dict[str, float | int | bool]:
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return {
            "length": 0,
            "unique_fraction": 0.0,
            "max_token_count_fraction": 0.0,
            "adjacent_repeat_fraction": 0.0,
            "max_consecutive_run": 0,
            "best_periodic_repeat_fraction": 0.0,
            "best_periodic_repeat_period": 0,
            "loop_like": False,
        }
    max_run = 1
    current_run = 1
    adjacent_repeats = 0
    counts: dict[int, int] = {}
    for index, token_id in enumerate(ids):
        counts[token_id] = counts.get(token_id, 0) + 1
        if index > 0 and token_id == ids[index - 1]:
            adjacent_repeats += 1
            current_run += 1
        else:
            current_run = 1
        max_run = max(max_run, current_run)
    denom = max(len(ids) - 1, 1)
    max_count_fraction = max(counts.values()) / len(ids)
    adjacent_repeat_fraction = adjacent_repeats / denom
    best_periodic_repeat_fraction = 0.0
    best_periodic_repeat_period = 0
    max_period = min(8, max(1, len(ids) // 2))
    for period in range(1, max_period + 1):
        comparisons = len(ids) - period
        if comparisons <= 0:
            continue
        matches = sum(1 for index in range(period, len(ids)) if ids[index] == ids[index - period])
        fraction = matches / comparisons
        if fraction > best_periodic_repeat_fraction:
            best_periodic_repeat_fraction = fraction
            best_periodic_repeat_period = period
    unique_fraction = len(counts) / len(ids)
    long_low_diversity_fraction = (
        max_count_fraction if len(ids) >= 16 and unique_fraction <= 0.25 else 0.0
    )
    loop_like = len(ids) >= 4 and (
        max_run >= 4
        or max_count_fraction >= 0.80
        or adjacent_repeat_fraction >= 0.80
        or (len(ids) >= 8 and best_periodic_repeat_fraction >= 0.80)
        or (len(ids) >= 16 and unique_fraction <= 0.25 and max_count_fraction >= 0.35)
        or (len(ids) >= 24 and unique_fraction <= 0.30 and max_count_fraction >= 0.35)
        or (len(ids) >= 16 and unique_fraction <= 0.35 and best_periodic_repeat_fraction >= 0.70)
    )
    return {
        "length": int(len(ids)),
        "unique_fraction": float(unique_fraction),
        "max_token_count_fraction": float(max_count_fraction),
        "adjacent_repeat_fraction": float(adjacent_repeat_fraction),
        "max_consecutive_run": int(max_run),
        "best_periodic_repeat_fraction": float(best_periodic_repeat_fraction),
        "best_periodic_repeat_period": int(best_periodic_repeat_period),
        "long_low_diversity_fraction": float(long_low_diversity_fraction),
        "loop_like": bool(loop_like),
    }


def first_token_consistency_stats(
    generated_token_ids: Sequence[int],
    first_response_token: dict[str, object],
    *,
    deterministic_free_decode: bool,
) -> dict[str, float | int | bool]:
    generated = [int(token_id) for token_id in generated_token_ids]
    teacher_available = bool(first_response_token.get("available", False))
    generated_available = bool(generated)
    top1_id = int(first_response_token.get("top1_id", -1)) if teacher_available else -1
    gold_id = int(first_response_token.get("gold_token_id", -1)) if teacher_available else -1
    generated_first_id = int(generated[0]) if generated_available else -1
    top5_ids_raw = first_response_token.get("top5_ids", [])
    top5_ids = [int(token_id) for token_id in top5_ids_raw] if isinstance(top5_ids_raw, list) else []
    top1_match = bool(teacher_available and generated_available and generated_first_id == top1_id)
    gold_match = bool(teacher_available and generated_available and generated_first_id == gold_id)
    top5_match = bool(teacher_available and generated_available and generated_first_id in set(top5_ids))
    consistency_required = bool(deterministic_free_decode and teacher_available and generated_available)
    return {
        "available": bool(teacher_available and generated_available),
        "deterministic_free_decode": bool(deterministic_free_decode),
        "consistency_required": bool(consistency_required),
        "consistency_pass": bool((not consistency_required) or top1_match),
        "generated_first_id": int(generated_first_id),
        "teacher_forced_top1_id": int(top1_id),
        "gold_id": int(gold_id),
        "matches_teacher_forced_top1": bool(top1_match),
        "matches_teacher_forced_top5": bool(top5_match),
        "matches_gold": bool(gold_match),
    }


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: Sequence[int],
    *,
    repetition_penalty: float,
    repetition_window: int,
) -> torch.Tensor:
    penalty = float(repetition_penalty)
    if penalty <= 1.0 or not generated:
        return logits
    adjusted = logits.clone()
    recent = [int(token_id) for token_id in generated[-int(repetition_window) :]]
    for token_id in set(recent):
        if 0 <= token_id < int(adjusted.shape[-1]):
            if adjusted[token_id] > 0:
                adjusted[token_id] = adjusted[token_id] / penalty
            else:
                adjusted[token_id] = adjusted[token_id] * penalty
    return adjusted


def top_p_filter(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    if float(top_p) >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    filtered = logits.clone()
    filtered[sorted_indices[remove]] = -float("inf")
    return filtered


@torch.no_grad()
def generate_free(
    model: torch.nn.Module,
    prefix_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_id: int = 1,
    stop_ids: Sequence[int] | None = None,
    think_steps: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    repetition_window: int = 32,
) -> list[int]:
    device = next(model.parameters()).device
    current = [int(token_id) for token_id in prefix_ids]
    response_start_index = max(0, len(current) - 1)
    generated: list[int] = []
    normalized_stop_ids = {int(eos_id)}
    if stop_ids is not None:
        normalized_stop_ids.update(int(token_id) for token_id in stop_ids)
    model.eval()
    for _ in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        response_prediction_mask = torch.zeros_like(input_ids)
        response_prediction_mask[:, response_start_index:] = 1
        logits, _, _ = model.forward_logits_and_hidden(
            input_ids,
            attention_mask,
            think_steps=int(think_steps),
            response_prediction_mask=response_prediction_mask,
        )
        next_logits = logits[0, -1]
        next_logits = torch.nan_to_num(next_logits.float(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
        next_logits = apply_repetition_penalty(
            next_logits,
            generated,
            repetition_penalty=float(repetition_penalty),
            repetition_window=int(repetition_window),
        )
        next_logits = top_p_filter(next_logits, top_p=float(top_p))
        if float(temperature) > 0.0:
            probs = torch.softmax(next_logits / max(float(temperature), 1.0e-6), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).detach().cpu().item())
        else:
            next_id = int(next_logits.argmax(dim=-1).detach().cpu().item())
        generated.append(next_id)
        current.append(next_id)
        if next_id in normalized_stop_ids:
            break
    return generated
