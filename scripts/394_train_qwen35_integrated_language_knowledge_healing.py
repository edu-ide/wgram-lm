#!/usr/bin/env python3
"""Language/knowledge healing for the Qwen3.5-integrated QTRM-native path.

This stage is deliberately not a public-test-label repair. It trains the same
standalone Qwen -> mandatory QTRM core -> Qwen LM-head graph on external
language/knowledge text, with optional non-test MCQ pressure, while preserving
the core_off Qwen distribution through KL. Public MMLU-Pro promotion must still
be checked by an independent 390 gate after this script finishes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


OPTION_LETTERS = "ABCDEFGHIJ"


def load_language_gate_module():
    path = Path(__file__).with_name("367_eval_qwen_backbone_language_gate.py")
    spec = importlib.util.spec_from_file_location("qwen_backbone_language_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def batch_items(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def _load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row must be a JSON object at {path}:{line_no}")
        rows.append(row)
    return rows


def load_text_rows(
    paths: list[str],
    *,
    max_rows: int,
    min_chars: int,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in paths:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matches:
            if not path.exists():
                raise FileNotFoundError(path)
            for row in _load_jsonl_rows(path):
                text = str(row.get("text", "")).strip()
                if len(text) < int(min_chars):
                    continue
                if "<think" in text.lower() or "</think" in text.lower():
                    continue
                key = text[:512]
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"text": text, "source": str(row.get("source", path))})
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    if int(max_rows) > 0:
        rows = rows[: int(max_rows)]
    if not rows:
        raise ValueError("no usable text rows")
    return rows


def split_train_eval(
    rows: list[dict[str, str]],
    *,
    eval_rows: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    count = min(max(1, int(eval_rows)), max(1, len(rows) // 5))
    eval_part = rows[:count]
    train_part = rows[count:] or rows
    return train_part, eval_part


def load_mcq_rows(path: str | Path, *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    rows = _load_jsonl_rows(path)
    required = ("qtrm_prompt", "answer", "options")
    rows = [row for row in rows if all(key in row for key in required)]
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    if int(max_rows) > 0:
        rows = rows[: int(max_rows)]
    return rows


def group_rows_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("category", "unknown")), []).append(row)
    return groups


def sample_mcq_chunk(
    rng: random.Random,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    balanced_category_sampling: bool,
    category_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    size = min(int(batch_size), len(rows))
    if not bool(balanced_category_sampling):
        return rng.sample(rows, k=size)
    groups = category_groups if category_groups is not None else group_rows_by_category(rows)
    categories = [category for category, bucket in sorted(groups.items()) if bucket]
    if not categories:
        return rng.sample(rows, k=size)
    return [rng.choice(groups[rng.choice(categories)]) for _ in range(size)]


def _encode(tokenizer, texts: list[str], *, max_seq_len: int, device: torch.device):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def last_nonpad_logits(logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Return next-token logits at the last real token, not the right-padding token."""
    if logits.ndim != 3:
        raise ValueError(f"expected [batch, seq, vocab] logits, got shape={tuple(logits.shape)}")
    if attention_mask is None:
        return logits[:, -1, :]
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, lengths.to(device=logits.device), :]


def last_nonpad_values(values: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Return per-example values aligned to the last real token."""
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    if attention_mask is None:
        return values[:, -1].reshape(values.shape[0], -1).squeeze(-1)
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, lengths.to(device=values.device)].reshape(values.shape[0], -1).squeeze(-1)


def _lm_loss_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if input_ids.shape[-1] < 2:
        return logits.float().sum() * 0.0
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is None:
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]).float(),
            shift_labels.view(-1),
        )
    shift_mask = attention_mask[:, 1:].bool()
    if not bool(shift_mask.any().item()):
        return shift_logits.float().sum() * 0.0
    return F.cross_entropy(
        shift_logits[shift_mask].float(),
        shift_labels[shift_mask],
    )


def sequence_kl_loss(
    core_logits: torch.Tensor,
    base_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if core_logits.shape[-2] < 2:
        return core_logits.float().sum() * 0.0
    core = core_logits[:, :-1, :]
    base = base_logits[:, :-1, :]
    if attention_mask is None:
        mask = torch.ones(core.shape[:2], device=core.device, dtype=torch.bool)
    else:
        mask = attention_mask[:, 1:].bool()
    if not bool(mask.any().item()):
        return core.float().sum() * 0.0
    return F.kl_div(
        F.log_softmax(core[mask].float(), dim=-1),
        F.softmax(base[mask].float(), dim=-1),
        reduction="batchmean",
    )


def normalize_mcq_answer(text: str) -> str:
    value = str(text).strip().upper()
    return value if value in OPTION_LETTERS else ""


def option_count(row: dict[str, Any]) -> int:
    options = row.get("options", [])
    if isinstance(options, list) and options:
        return min(len(options), len(OPTION_LETTERS))
    return len(OPTION_LETTERS)


def single_token_option_ids(tokenizer, letter: str) -> list[int]:
    ids: list[int] = []
    for variant in (letter, f" {letter}", f"\n{letter}"):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def option_nll_loss(tokenizer, logits: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    if not rows:
        return logits.float().sum() * 0.0
    log_probs = F.log_softmax(logits.float(), dim=-1)
    losses = []
    for index, row in enumerate(rows):
        gold = normalize_mcq_answer(str(row["answer"]))
        if not gold:
            continue
        ids = single_token_option_ids(tokenizer, gold)
        if not ids:
            continue
        target = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
        losses.append(-torch.logsumexp(log_probs[index].index_select(dim=0, index=target), dim=0))
    if not losses:
        return logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def option_choice_ce_loss(
    tokenizer,
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    selected_indices: list[int] | None = None,
) -> torch.Tensor:
    indices = list(range(len(rows))) if selected_indices is None else list(selected_indices)
    if not indices:
        return logits.float().sum() * 0.0
    losses = []
    for index in indices:
        row = rows[index]
        gold = normalize_mcq_answer(str(row["answer"]))
        scores = option_letter_scores(tokenizer, logits[index], row)
        letters = [
            letter
            for letter in OPTION_LETTERS[: option_count(row)]
            if letter in scores
        ]
        if gold not in letters:
            continue
        vector = torch.stack([scores[letter] for letter in letters]).unsqueeze(0)
        target = torch.tensor([letters.index(gold)], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(vector.float(), target))
    if not losses:
        return logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def option_letter_scores(tokenizer, logits: torch.Tensor, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores: dict[str, torch.Tensor] = {}
    for letter in OPTION_LETTERS[: option_count(row)]:
        ids = single_token_option_ids(tokenizer, letter)
        if not ids:
            continue
        target = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
        scores[letter] = torch.logsumexp(log_probs.index_select(dim=0, index=target), dim=0)
    return scores


def base_wrong_indices(
    tokenizer,
    base_logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    max_top_margin: float | None = None,
) -> list[int]:
    selected: list[int] = []
    for index, row in enumerate(rows):
        gold = normalize_mcq_answer(str(row["answer"]))
        scores = option_letter_scores(tokenizer, base_logits[index], row)
        if not scores:
            continue
        sorted_scores = sorted(
            scores.items(),
            key=lambda item: float(item[1].detach().cpu()),
            reverse=True,
        )
        pred = sorted_scores[0][0]
        if pred != gold:
            if max_top_margin is not None and math.isfinite(float(max_top_margin)) and len(sorted_scores) > 1:
                top_margin = float((sorted_scores[0][1] - sorted_scores[1][1]).detach().cpu())
                if top_margin > float(max_top_margin):
                    continue
            selected.append(index)
    return selected


def option_distribution_kl_loss_for_indices(
    tokenizer,
    core_logits: torch.Tensor,
    base_logits: torch.Tensor,
    rows: list[dict[str, Any]],
    indices: list[int],
) -> torch.Tensor:
    if not indices:
        return core_logits.float().sum() * 0.0
    losses = []
    for index in indices:
        row = rows[index]
        base_scores = option_letter_scores(tokenizer, base_logits[index], row)
        core_scores = option_letter_scores(tokenizer, core_logits[index], row)
        letters = [
            letter
            for letter in OPTION_LETTERS[: option_count(row)]
            if letter in base_scores and letter in core_scores
        ]
        if not letters:
            continue
        base_vector = torch.stack([base_scores[letter] for letter in letters]).detach()
        core_vector = torch.stack([core_scores[letter] for letter in letters])
        losses.append(
            F.kl_div(
                F.log_softmax(core_vector.float(), dim=-1),
                F.softmax(base_vector.float(), dim=-1),
                reduction="sum",
            )
        )
    if not losses:
        return core_logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def residual_gate_target_loss(
    gate: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    indices: list[int],
    *,
    target: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not indices or not isinstance(gate, torch.Tensor) or gate.ndim < 2:
        return reference.float().sum() * 0.0
    values = last_nonpad_values(gate.float(), attention_mask)
    selected = values.index_select(
        dim=0,
        index=torch.tensor(indices, dtype=torch.long, device=values.device),
    )
    selected = selected.clamp(min=1e-4, max=1.0 - 1e-4)
    targets = torch.full_like(selected, float(target))
    return F.binary_cross_entropy(selected, targets)


def option_distribution_kl_loss(
    tokenizer,
    core_logits: torch.Tensor,
    base_logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    focus: str,
) -> torch.Tensor:
    losses = []
    for index, row in enumerate(rows):
        gold = normalize_mcq_answer(str(row["answer"]))
        base_scores = option_letter_scores(tokenizer, base_logits[index], row)
        core_scores = option_letter_scores(tokenizer, core_logits[index], row)
        letters = [letter for letter in OPTION_LETTERS[: option_count(row)] if letter in base_scores and letter in core_scores]
        if not letters:
            continue
        base_pred = max(base_scores.items(), key=lambda item: float(item[1].detach().cpu()))[0]
        if str(focus) == "base_correct" and base_pred != gold:
            continue
        if str(focus) == "base_wrong" and base_pred == gold:
            continue
        base_vector = torch.stack([base_scores[letter] for letter in letters]).detach()
        core_vector = torch.stack([core_scores[letter] for letter in letters])
        losses.append(
            F.kl_div(
                F.log_softmax(core_vector.float(), dim=-1),
                F.softmax(base_vector.float(), dim=-1),
                reduction="sum",
            )
        )
    if not losses:
        return core_logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def selected_option_nll_loss(
    tokenizer,
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    selected_indices: list[int],
) -> torch.Tensor:
    if not selected_indices:
        return logits.float().sum() * 0.0
    index_tensor = torch.tensor(selected_indices, device=logits.device, dtype=torch.long)
    selected_logits = logits.index_select(dim=0, index=index_tensor)
    selected_rows = [rows[index] for index in selected_indices]
    return option_nll_loss(tokenizer, selected_logits, selected_rows)


def option_margin_loss(
    tokenizer,
    core_logits: torch.Tensor,
    base_logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    margin: float,
    focus: str,
) -> torch.Tensor:
    losses = []
    for index, row in enumerate(rows):
        gold = normalize_mcq_answer(str(row["answer"]))
        base_scores = option_letter_scores(tokenizer, base_logits[index], row)
        core_scores = option_letter_scores(tokenizer, core_logits[index], row)
        if gold not in core_scores or not base_scores:
            continue
        rejected = max(base_scores.items(), key=lambda item: float(item[1].detach().cpu()))[0]
        if rejected == gold:
            if str(focus) == "base_wrong":
                continue
            rejected = max(
                ((letter, score) for letter, score in base_scores.items() if letter != gold),
                key=lambda item: float(item[1].detach().cpu()),
            )[0]
        losses.append(
            F.relu(
                torch.as_tensor(float(margin), device=core_logits.device)
                - (core_scores[gold] - core_scores[rejected])
            )
        )
    if not losses:
        return core_logits.float().sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def score_mcq(model, tokenizer, rows: list[dict[str, Any]], args) -> dict[str, Any]:
    device = next(model.parameters()).device
    base_hits = 0
    core_hits = 0
    finite = True
    outer_iteration_values: list[torch.Tensor] = []
    converged_values: list[torch.Tensor] = []
    convergence_delta_values: list[torch.Tensor] = []
    by_category: dict[str, dict[str, int]] = {}
    flip_counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "base_correct_core_wrong": 0,
        "base_wrong_core_correct": 0,
    }
    for chunk in batch_items(rows, int(args.eval_batch_size)):
        input_ids, attention_mask = _encode(
            tokenizer,
            [str(row["qtrm_prompt"]) for row in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        base_logits = model(
            input_ids,
            attention_mask=attention_mask,
            force_core_off=True,
        ).logits
        core_outputs = model(input_ids, attention_mask=attention_mask)
        base_logits = last_nonpad_logits(base_logits, attention_mask)
        core_logits = last_nonpad_logits(core_outputs.logits, attention_mask)
        outer_iterations = getattr(core_outputs, "qtrm_core_outer_iterations", None)
        if isinstance(outer_iterations, torch.Tensor):
            outer_iteration_values.append(outer_iterations.detach().float().cpu())
        converged = getattr(core_outputs, "qtrm_core_converged", None)
        if isinstance(converged, torch.Tensor):
            converged_values.append(converged.detach().float().cpu())
        convergence_delta = getattr(core_outputs, "qtrm_core_convergence_delta", None)
        if isinstance(convergence_delta, torch.Tensor) and convergence_delta.numel() > 0:
            convergence_delta_values.append(convergence_delta.detach().float().cpu().reshape(-1))
        finite = bool(finite and torch.isfinite(base_logits).all().item() and torch.isfinite(core_logits).all().item())
        for index, row in enumerate(chunk):
            choices = OPTION_LETTERS[: option_count(row)]
            gold = normalize_mcq_answer(str(row["answer"]))
            base_pred = _predict_option(tokenizer, base_logits[index], choices)
            core_pred = _predict_option(tokenizer, core_logits[index], choices)
            base_correct = base_pred == gold
            core_correct = core_pred == gold
            base_hits += int(base_correct)
            core_hits += int(core_correct)
            if base_correct and core_correct:
                flip_key = "both_correct"
            elif base_correct and not core_correct:
                flip_key = "base_correct_core_wrong"
            elif not base_correct and core_correct:
                flip_key = "base_wrong_core_correct"
            else:
                flip_key = "both_wrong"
            flip_counts[flip_key] += 1
            category = str(row.get("category", "unknown"))
            bucket = by_category.setdefault(
                category,
                {
                    "total": 0,
                    "base_hits": 0,
                    "core_hits": 0,
                    "both_correct": 0,
                    "both_wrong": 0,
                    "base_correct_core_wrong": 0,
                    "base_wrong_core_correct": 0,
                },
            )
            bucket["total"] += 1
            bucket["base_hits"] += int(base_correct)
            bucket["core_hits"] += int(core_correct)
            bucket[flip_key] += 1
    total = len(rows)
    mean_outer_iterations = (
        float(torch.cat(outer_iteration_values).mean().item())
        if outer_iteration_values
        else None
    )
    converged_fraction = (
        float(torch.cat(converged_values).mean().item())
        if converged_values
        else None
    )
    mean_convergence_delta = (
        float(torch.cat(convergence_delta_values).mean().item())
        if convergence_delta_values
        else None
    )
    return {
        "cases": total,
        "base_hits": base_hits,
        "core_hits": core_hits,
        "base_accuracy": float(base_hits / max(1, total)),
        "core_accuracy": float(core_hits / max(1, total)),
        "gain": float((core_hits - base_hits) / max(1, total)),
        "finite_logits": bool(finite),
        "mean_core_outer_iterations": mean_outer_iterations,
        "core_converged_fraction": converged_fraction,
        "mean_core_convergence_delta": mean_convergence_delta,
        "flip_counts": {key: int(value) for key, value in flip_counts.items()},
        "by_category": {
            category: {
                "total": int(value["total"]),
                "base_hits": int(value["base_hits"]),
                "core_hits": int(value["core_hits"]),
                "both_correct": int(value["both_correct"]),
                "both_wrong": int(value["both_wrong"]),
                "base_correct_core_wrong": int(value["base_correct_core_wrong"]),
                "base_wrong_core_correct": int(value["base_wrong_core_correct"]),
                "base_accuracy": float(value["base_hits"] / max(1, value["total"])),
                "core_accuracy": float(value["core_hits"] / max(1, value["total"])),
                "hit_delta": int(value["core_hits"] - value["base_hits"]),
                "accuracy_delta": float(
                    (value["core_hits"] - value["base_hits"]) / max(1, value["total"])
                ),
            }
            for category, value in sorted(by_category.items())
        },
    }


def _predict_option(tokenizer, logits: torch.Tensor, choices: str) -> str:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores: dict[str, float] = {}
    for letter in choices:
        ids = single_token_option_ids(tokenizer, letter)
        if not ids:
            scores[letter] = float("-inf")
            continue
        target = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
        scores[letter] = float(torch.logsumexp(log_probs.index_select(dim=0, index=target), dim=0).detach().cpu())
    return max(scores.items(), key=lambda item: item[1])[0] if scores else ""


@torch.no_grad()
def evaluate_text_ce(model, tokenizer, rows: list[dict[str, str]], args) -> dict[str, Any]:
    device = next(model.parameters()).device
    base_losses = []
    core_losses = []
    finite = True
    for chunk in batch_items(rows, int(args.eval_batch_size)):
        input_ids, attention_mask = _encode(
            tokenizer,
            [row["text"] for row in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        base = model(
            input_ids,
            attention_mask=attention_mask,
            force_core_off=True,
        ).logits
        core = model(input_ids, attention_mask=attention_mask).logits
        finite = bool(finite and torch.isfinite(base).all().item() and torch.isfinite(core).all().item())
        base_losses.append(float(_lm_loss_from_logits(base, input_ids, attention_mask).detach().cpu()))
        core_losses.append(float(_lm_loss_from_logits(core, input_ids, attention_mask).detach().cpu()))
    base_ce = sum(base_losses) / max(1, len(base_losses))
    core_ce = sum(core_losses) / max(1, len(core_losses))
    return {
        "cases": len(rows),
        "base_ce": float(base_ce),
        "core_ce": float(core_ce),
        "core_minus_base_ce": float(core_ce - base_ce),
        "finite_logits": bool(finite),
    }


@torch.no_grad()
def evaluate_language_preservation(model, tokenizer, args) -> dict[str, Any]:
    module = load_language_gate_module()
    prompts = module.default_language_prompts()
    topk = module.evaluate_topk(model, tokenizer, prompts, args)
    generation = module.evaluate_generation(model, tokenizer, prompts, args)
    return {"topk": topk, "generation": generation}


def default_language_anchor_prompts() -> list[str]:
    module = load_language_gate_module()
    return list(module.default_language_prompts())


def last_token_kl_loss(core_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(core_logits.float(), dim=-1),
        F.softmax(base_logits.float(), dim=-1),
        reduction="batchmean",
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    load_mode: str,
) -> dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    skipped = []
    if str(load_mode) == "skip_mismatch":
        compatible = {}
        for key, value in state.items():
            if key in current and tuple(value.shape) != tuple(current[key].shape):
                skipped.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(current[key].shape),
                    }
                )
                continue
            compatible[key] = value
        state = compatible
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:8]}")
    return {
        "path": str(checkpoint_path),
        "loaded": True,
        "load_mode": str(load_mode),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_key_count": len(skipped),
        "shape_mismatch_keys": skipped[:64],
        "checkpoint_report": checkpoint.get("report", {}),
    }


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().cpu()
        for key, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def category_gain_summary(evaluation: dict[str, Any], *, min_cases: int = 1) -> dict[str, Any]:
    by_category = evaluation.get("by_category", {})
    deltas: dict[str, dict[str, float | int]] = {}
    eligible = []
    for category, value in sorted(by_category.items()):
        total = int(value.get("total", 0))
        base_hits = int(value.get("base_hits", 0))
        core_hits = int(value.get("core_hits", 0))
        hit_delta = core_hits - base_hits
        base_accuracy = float(base_hits / max(1, total))
        core_accuracy = float(core_hits / max(1, total))
        accuracy_delta = core_accuracy - base_accuracy
        deltas[category] = {
            "total": total,
            "base_hits": base_hits,
            "core_hits": core_hits,
            "hit_delta": hit_delta,
            "base_accuracy": base_accuracy,
            "core_accuracy": core_accuracy,
            "accuracy_delta": accuracy_delta,
        }
        if total >= int(min_cases):
            eligible.append(deltas[category])
    if eligible:
        min_hit_delta = min(int(item["hit_delta"]) for item in eligible)
        min_accuracy_delta = min(float(item["accuracy_delta"]) for item in eligible)
        negative_hit_delta_sum = sum(max(0, -int(item["hit_delta"])) for item in eligible)
        negative_accuracy_delta_sum = sum(max(0.0, -float(item["accuracy_delta"])) for item in eligible)
    else:
        min_hit_delta = 0
        min_accuracy_delta = 0.0
        negative_hit_delta_sum = 0
        negative_accuracy_delta_sum = 0.0
    return {
        "min_cases": int(min_cases),
        "eligible_categories": len(eligible),
        "min_hit_delta": int(min_hit_delta),
        "min_accuracy_delta": float(min_accuracy_delta),
        "negative_hit_delta_sum": int(negative_hit_delta_sum),
        "negative_accuracy_delta_sum": float(negative_accuracy_delta_sum),
        "by_category": deltas,
    }


def validation_selection_score(
    *,
    before_text: dict[str, Any],
    current_text: dict[str, Any],
    current_mcq: dict[str, Any] | None,
    args: argparse.Namespace,
) -> float:
    text_ce_delta = float(current_text["core_ce"]) - float(before_text["core_ce"])
    text_penalty = max(0.0, text_ce_delta - float(args.max_core_ce_regression))
    score = -float(args.text_ce_regression_penalty) * text_penalty
    if current_mcq is not None:
        category = category_gain_summary(
            current_mcq,
            min_cases=int(args.category_guard_min_cases),
        )
        score += float(current_mcq["gain"])
        score += 0.10 * float(current_mcq["core_accuracy"])
        score -= float(args.category_regression_penalty) * float(category["negative_accuracy_delta_sum"])
    if not bool(current_text.get("finite_logits", False)):
        score -= 1.0
    if current_mcq is not None and not bool(current_mcq.get("finite_logits", False)):
        score -= 1.0
    return float(score)


def load_model(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = dtype_from_name(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=True,
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        core_insertion_mode=str(args.core_insertion_mode),
        core_insert_after_layer=int(args.core_insert_after_layer),
        qwen_core_layer_indices=parse_int_list(str(args.qwen_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        core_residual_gate_mode=str(args.core_residual_gate_mode),
        core_residual_gate_dim=int(args.core_residual_gate_dim),
        core_residual_gate_init=float(args.core_residual_gate_init),
        clone_qwen_core_layers=bool(args.clone_qwen_core_layers),
        mandatory_core=bool(args.mandatory_core),
        n_core_layers=int(args.n_core_layers),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        core_convergence_halt_enabled=bool(args.core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(args.core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(args.core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(args.core_step_conditioning_enabled),
        core_step_conditioning_max_steps=int(args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(args.core_step_conditioning_scale),
        delta_backend="fla_gated_delta",
        strict_backends=False,
        core_causal=True,
    ).to(device)
    layer_indices = parse_int_list(str(args.unfreeze_qwen_layer_indices))
    qwen_trainability: dict[str, Any] = {"mode": "frozen", "qwen_trainable_parameters": 0}
    if bool(layer_indices or args.unfreeze_qwen_lm_head or args.unfreeze_qwen_final_norm):
        qwen_trainability = model.set_qwen_partial_trainable(
            layer_indices=layer_indices,
            train_embeddings=False,
            train_lm_head=bool(args.unfreeze_qwen_lm_head),
            train_final_norm=bool(args.unfreeze_qwen_final_norm),
        )
        qwen_trainability["mode"] = "partial"
    checkpoint_info = load_checkpoint(
        model,
        str(args.init_checkpoint),
        load_mode=str(args.checkpoint_load_mode),
    )
    adapter_trainability: dict[str, Any] = {
        "train_only_core_delta_adapter": bool(args.train_only_core_delta_adapter),
        "trainable_parameters": None,
    }
    if bool(args.train_only_core_delta_adapter):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("core_delta_adapter."))
        model.qwen.eval()
        adapter_trainability["trainable_parameters"] = sum(
            int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
        )
    return tokenizer, model, device, checkpoint_info, qwen_trainability, adapter_trainability


def train(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    tokenizer, model, device, checkpoint_info, qwen_trainability, adapter_trainability = load_model(args)
    text_rows = load_text_rows(
        list(args.text_jsonl),
        max_rows=int(args.max_text_rows),
        min_chars=int(args.min_text_chars),
        seed=int(args.seed),
    )
    train_text_rows, eval_text_rows = split_train_eval(text_rows, eval_rows=int(args.eval_text_rows))
    mcq_rows = (
        load_mcq_rows(args.mcq_jsonl, max_rows=int(args.max_mcq_rows), seed=int(args.seed) + 11)
        if str(args.mcq_jsonl)
        else []
    )
    validation_mcq_rows = (
        load_mcq_rows(
            args.mcq_validation_jsonl,
            max_rows=int(args.eval_mcq_rows),
            seed=int(args.seed) + 13,
        )
        if str(args.mcq_validation_jsonl)
        else []
    )
    if validation_mcq_rows:
        eval_mcq_rows = validation_mcq_rows
        train_mcq_rows = mcq_rows
    else:
        eval_mcq_rows = mcq_rows[: min(len(mcq_rows), int(args.eval_mcq_rows))]
        train_mcq_rows = mcq_rows[min(len(mcq_rows), int(args.eval_mcq_rows)) :] or mcq_rows

    before_text = evaluate_text_ce(model, tokenizer, eval_text_rows, args)
    before_mcq = score_mcq(model, tokenizer, eval_mcq_rows, args) if eval_mcq_rows else None

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    qwen_trainable = [(name, parameter) for name, parameter in named_trainable if name.startswith("qwen.")]
    gate_trainable = [
        (name, parameter)
        for name, parameter in named_trainable
        if name.startswith("core_residual_gate.")
    ]
    core_trainable = [
        (name, parameter)
        for name, parameter in named_trainable
        if not name.startswith("qwen.") and not name.startswith("core_residual_gate.")
    ]
    param_groups = []
    if core_trainable:
        param_groups.append({"params": [p for _, p in core_trainable], "lr": float(args.lr)})
    if gate_trainable:
        param_groups.append(
            {
                "params": [p for _, p in gate_trainable],
                "lr": float(args.lr) * float(args.residual_gate_lr_multiplier),
            }
        )
    if qwen_trainable:
        param_groups.append({"params": [p for _, p in qwen_trainable], "lr": float(args.qwen_lr)})
    if not param_groups:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
    rng = random.Random(int(args.seed) + 23)
    mcq_category_groups = group_rows_by_category(train_mcq_rows)
    best_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, Any] | None = None
    if bool(args.restore_best_checkpoint):
        initial_score = validation_selection_score(
            before_text=before_text,
            current_text=before_text,
            current_mcq=before_mcq,
            args=args,
        )
        best_state = trainable_state_dict(model)
        best_eval = {
            "step": 0,
            "score": float(initial_score),
            "text": before_text,
            "mcq": before_mcq,
            "mcq_category_gain_summary": (
                category_gain_summary(before_mcq, min_cases=int(args.category_guard_min_cases))
                if before_mcq is not None
                else None
            ),
        }
    losses: list[float] = []
    language_anchor_prompts = default_language_anchor_prompts() if float(args.language_anchor_weight) > 0.0 else []
    model.train()
    if not qwen_trainable:
        model.qwen.eval()
    for step in range(1, int(args.steps) + 1):
        text_chunk = rng.sample(train_text_rows, k=min(int(args.batch_size), len(train_text_rows)))
        input_ids, attention_mask = _encode(
            tokenizer,
            [row["text"] for row in text_chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = _lm_loss_from_logits(outputs.logits, input_ids, attention_mask)
        if float(args.base_kl_weight) > 0.0:
            with torch.no_grad():
                base_logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                    force_core_off=True,
                ).logits
            loss = loss + float(args.base_kl_weight) * sequence_kl_loss(
                outputs.logits,
                base_logits,
                attention_mask,
            )
        if language_anchor_prompts and int(args.language_anchor_batch_size) > 0:
            anchor_chunk = rng.sample(
                language_anchor_prompts,
                k=min(int(args.language_anchor_batch_size), len(language_anchor_prompts)),
            )
            anchor_input_ids, anchor_attention_mask = _encode(
                tokenizer,
                anchor_chunk,
                max_seq_len=int(args.max_seq_len),
                device=device,
            )
            anchor_logits = model(
                anchor_input_ids,
                attention_mask=anchor_attention_mask,
            ).logits
            anchor_logits = last_nonpad_logits(anchor_logits, anchor_attention_mask)
            with torch.no_grad():
                anchor_base_logits = model(
                    anchor_input_ids,
                    attention_mask=anchor_attention_mask,
                    force_core_off=True,
                ).logits
                anchor_base_logits = last_nonpad_logits(anchor_base_logits, anchor_attention_mask)
            loss = loss + float(args.language_anchor_weight) * last_token_kl_loss(
                anchor_logits,
                anchor_base_logits,
            )
        has_mcq_objective = bool(
            float(args.mcq_weight) > 0.0
            or float(args.mcq_margin_weight) > 0.0
            or float(args.base_correct_option_kl_weight) > 0.0
            or float(args.mcq_non_selected_option_kl_weight) > 0.0
            or float(args.residual_gate_selected_open_weight) > 0.0
            or float(args.residual_gate_non_selected_closed_weight) > 0.0
        )
        if train_mcq_rows and has_mcq_objective:
            mcq_base_logits = None
            selected_indices: list[int] | None = None
            retry_count = max(1, int(args.base_wrong_mcq_retries))
            needs_base_logits = bool(
                str(args.mcq_ce_focus) == "base_wrong"
                or float(args.mcq_margin_weight) > 0.0
                or float(args.base_correct_option_kl_weight) > 0.0
                or float(args.mcq_non_selected_option_kl_weight) > 0.0
                or float(args.residual_gate_selected_open_weight) > 0.0
                or float(args.residual_gate_non_selected_closed_weight) > 0.0
            )
            needs_base_wrong_selection = bool(
                str(args.mcq_ce_focus) == "base_wrong"
                or (float(args.mcq_margin_weight) > 0.0 and str(args.mcq_margin_focus) == "base_wrong")
                or float(args.mcq_non_selected_option_kl_weight) > 0.0
                or float(args.residual_gate_selected_open_weight) > 0.0
                or float(args.residual_gate_non_selected_closed_weight) > 0.0
            )
            base_wrong_max_top_margin = (
                None
                if float(args.base_wrong_max_top_margin) < 0.0
                else float(args.base_wrong_max_top_margin)
            )
            for attempt in range(retry_count):
                mcq_chunk = sample_mcq_chunk(
                    rng,
                    train_mcq_rows,
                    batch_size=int(args.mcq_batch_size),
                    balanced_category_sampling=bool(args.balanced_mcq_category_sampling),
                    category_groups=mcq_category_groups,
                )
                mcq_input_ids, mcq_attention_mask = _encode(
                    tokenizer,
                    [str(row["qtrm_prompt"]) for row in mcq_chunk],
                    max_seq_len=int(args.max_seq_len),
                    device=device,
                )
                mcq_outputs = model(mcq_input_ids, attention_mask=mcq_attention_mask)
                mcq_logits = last_nonpad_logits(mcq_outputs.logits, mcq_attention_mask)
                mcq_base_logits = None
                if needs_base_logits:
                    with torch.no_grad():
                        mcq_base_logits = model(
                            mcq_input_ids,
                            attention_mask=mcq_attention_mask,
                            force_core_off=True,
                        ).logits
                        mcq_base_logits = last_nonpad_logits(mcq_base_logits, mcq_attention_mask)
                if needs_base_wrong_selection:
                    assert mcq_base_logits is not None
                    selected_indices = base_wrong_indices(
                        tokenizer,
                        mcq_base_logits,
                        mcq_chunk,
                        max_top_margin=base_wrong_max_top_margin,
                    )
                    if selected_indices or attempt == retry_count - 1:
                        break
                else:
                    break
            if needs_base_logits and mcq_base_logits is None:
                with torch.no_grad():
                    mcq_base_logits = model(
                        mcq_input_ids,
                        attention_mask=mcq_attention_mask,
                        force_core_off=True,
                    ).logits
                    mcq_base_logits = last_nonpad_logits(mcq_base_logits, mcq_attention_mask)
            if float(args.mcq_weight) > 0.0:
                if str(args.mcq_ce_focus) == "base_wrong":
                    assert mcq_base_logits is not None
                    if selected_indices is None:
                        selected_indices = base_wrong_indices(
                            tokenizer,
                            mcq_base_logits,
                            mcq_chunk,
                            max_top_margin=base_wrong_max_top_margin,
                        )
                    if str(args.mcq_loss_space) == "option_only":
                        mcq_loss = option_choice_ce_loss(
                            tokenizer,
                            mcq_logits,
                            mcq_chunk,
                            selected_indices,
                        )
                    else:
                        mcq_loss = selected_option_nll_loss(tokenizer, mcq_logits, mcq_chunk, selected_indices)
                else:
                    if str(args.mcq_loss_space) == "option_only":
                        mcq_loss = option_choice_ce_loss(tokenizer, mcq_logits, mcq_chunk)
                    else:
                        mcq_loss = option_nll_loss(tokenizer, mcq_logits, mcq_chunk)
                loss = loss + float(args.mcq_weight) * mcq_loss
            if float(args.mcq_margin_weight) > 0.0:
                assert mcq_base_logits is not None
                loss = loss + float(args.mcq_margin_weight) * option_margin_loss(
                    tokenizer,
                    mcq_logits,
                    mcq_base_logits,
                    mcq_chunk,
                    margin=float(args.mcq_margin_value),
                    focus=str(args.mcq_margin_focus),
                )
            if float(args.base_correct_option_kl_weight) > 0.0:
                assert mcq_base_logits is not None
                loss = loss + float(args.base_correct_option_kl_weight) * option_distribution_kl_loss(
                    tokenizer,
                    mcq_logits,
                    mcq_base_logits,
                    mcq_chunk,
                    focus=str(args.base_correct_option_kl_focus),
                )
            if float(args.mcq_non_selected_option_kl_weight) > 0.0:
                assert mcq_base_logits is not None
                if selected_indices is None:
                    selected_indices = base_wrong_indices(
                        tokenizer,
                        mcq_base_logits,
                        mcq_chunk,
                        max_top_margin=base_wrong_max_top_margin,
                    )
                selected_set = set(int(index) for index in selected_indices)
                non_selected_indices = [
                    index for index in range(len(mcq_chunk)) if index not in selected_set
                ]
                loss = loss + float(args.mcq_non_selected_option_kl_weight) * option_distribution_kl_loss_for_indices(
                    tokenizer,
                    mcq_logits,
                    mcq_base_logits,
                    mcq_chunk,
                    non_selected_indices,
                )
            if (
                float(args.residual_gate_selected_open_weight) > 0.0
                or float(args.residual_gate_non_selected_closed_weight) > 0.0
            ):
                assert mcq_base_logits is not None
                if selected_indices is None:
                    selected_indices = base_wrong_indices(
                        tokenizer,
                        mcq_base_logits,
                        mcq_chunk,
                        max_top_margin=base_wrong_max_top_margin,
                    )
                residual_gate = getattr(mcq_outputs, "qtrm_core_residual_gate", None)
                if float(args.residual_gate_selected_open_weight) > 0.0:
                    loss = loss + float(args.residual_gate_selected_open_weight) * residual_gate_target_loss(
                        residual_gate,
                        mcq_attention_mask,
                        selected_indices,
                        target=1.0,
                        reference=mcq_logits,
                    )
                if float(args.residual_gate_non_selected_closed_weight) > 0.0:
                    selected_set = set(int(index) for index in selected_indices)
                    non_selected_indices = [
                        index for index in range(len(mcq_chunk)) if index not in selected_set
                    ]
                    loss = loss + float(args.residual_gate_non_selected_closed_weight) * residual_gate_target_loss(
                        residual_gate,
                        mcq_attention_mask,
                        non_selected_indices,
                        target=0.0,
                        reference=mcq_logits,
                    )
            if float(args.base_correct_option_kl_weight) > 0.0 and int(args.base_correct_kl_extra_batch_size) > 0:
                preserve_chunk = sample_mcq_chunk(
                    rng,
                    train_mcq_rows,
                    batch_size=int(args.base_correct_kl_extra_batch_size),
                    balanced_category_sampling=bool(args.balanced_mcq_category_sampling),
                    category_groups=mcq_category_groups,
                )
                preserve_input_ids, preserve_attention_mask = _encode(
                    tokenizer,
                    [str(row["qtrm_prompt"]) for row in preserve_chunk],
                    max_seq_len=int(args.max_seq_len),
                    device=device,
                )
                preserve_logits = model(
                    preserve_input_ids,
                    attention_mask=preserve_attention_mask,
                ).logits
                preserve_logits = last_nonpad_logits(preserve_logits, preserve_attention_mask)
                with torch.no_grad():
                    preserve_base_logits = model(
                        preserve_input_ids,
                        attention_mask=preserve_attention_mask,
                        force_core_off=True,
                    ).logits
                    preserve_base_logits = last_nonpad_logits(
                        preserve_base_logits,
                        preserve_attention_mask,
                    )
                loss = loss + float(args.base_correct_option_kl_weight) * option_distribution_kl_loss(
                    tokenizer,
                    preserve_logits,
                    preserve_base_logits,
                    preserve_chunk,
                    focus=str(args.base_correct_option_kl_focus),
                )
        if not torch.isfinite(loss.detach()):
            raise RuntimeError(f"non-finite language/knowledge healing loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in named_trainable], float(args.grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            print(f"step={step} loss={losses[-1]:.4f}", flush=True)
        if int(args.eval_every_steps) > 0 and step % int(args.eval_every_steps) == 0:
            model.eval()
            current_text = evaluate_text_ce(model, tokenizer, eval_text_rows, args)
            current_mcq = score_mcq(model, tokenizer, eval_mcq_rows, args) if eval_mcq_rows else None
            current_score = validation_selection_score(
                before_text=before_text,
                current_text=current_text,
                current_mcq=current_mcq,
                args=args,
            )
            current_eval = {
                "step": int(step),
                "score": float(current_score),
                "text": current_text,
                "mcq": current_mcq,
                "mcq_category_gain_summary": (
                    category_gain_summary(current_mcq, min_cases=int(args.category_guard_min_cases))
                    if current_mcq is not None
                    else None
                ),
            }
            if best_eval is None or current_score > float(best_eval["score"]):
                best_eval = current_eval
                best_state = trainable_state_dict(model)
            mcq_line = ""
            if current_mcq is not None:
                mcq_line = (
                    f" mcq_core={current_mcq['core_accuracy']:.4f}"
                    f" mcq_base={current_mcq['base_accuracy']:.4f}"
                    f" mcq_gain={current_mcq['gain']:.4f}"
                )
            print(
                f"eval_step={step} score={current_score:.4f}"
                f" text_core_ce={current_text['core_ce']:.4f}"
                f" text_delta={current_text['core_ce'] - before_text['core_ce']:.4f}"
                f"{mcq_line}",
                flush=True,
            )
            model.train()
            if not qwen_trainable:
                model.qwen.eval()
    if best_state is not None and bool(args.restore_best_checkpoint):
        incompatible = model.load_state_dict(best_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected best checkpoint keys: {incompatible.unexpected_keys[:8]}")
    model.eval()
    after_text = evaluate_text_ce(model, tokenizer, eval_text_rows, args)
    after_mcq = score_mcq(model, tokenizer, eval_mcq_rows, args) if eval_mcq_rows else None
    language = evaluate_language_preservation(model, tokenizer, args)
    accepted_language = bool(
        language["topk"]["finite_logits"]
        and language["generation"]["finite_logits"]
        and float(language["topk"]["top1_agreement"]) >= float(args.min_language_top1_agreement)
        and int(language["generation"]["max_core_repeated_token_run"]) <= int(args.max_repeated_token_run)
        and float(language["generation"]["mean_core_unique_ratio"]) >= float(args.min_unique_ratio)
    )
    core_ce_delta = float(after_text["core_ce"]) - float(before_text["core_ce"])
    text_accepted = core_ce_delta <= float(args.max_core_ce_regression)
    mcq_category_summary = (
        category_gain_summary(after_mcq, min_cases=int(args.category_guard_min_cases))
        if after_mcq is not None
        else None
    )
    mcq_flip_counts = after_mcq.get("flip_counts", {}) if after_mcq is not None else {}
    base_wrong_core_correct = int(mcq_flip_counts.get("base_wrong_core_correct", 0))
    base_correct_core_wrong = int(mcq_flip_counts.get("base_correct_core_wrong", 0))
    raw_mcq_gain_accepted = True
    correction_accepted = True
    regression_accepted = True
    mcq_accepted = True
    if after_mcq is not None:
        raw_mcq_gain_accepted = bool(float(after_mcq["gain"]) >= float(args.min_eval_mcq_gain))
        correction_accepted = bool(base_wrong_core_correct >= int(args.min_base_wrong_core_correct))
        regression_accepted = bool(base_correct_core_wrong <= int(args.max_base_correct_core_wrong))
        mcq_accepted = bool(
            after_mcq["finite_logits"]
            and raw_mcq_gain_accepted
            and correction_accepted
            and regression_accepted
            and mcq_category_summary is not None
            and float(mcq_category_summary["min_accuracy_delta"]) >= float(args.min_eval_mcq_category_gain)
            and int(mcq_category_summary["min_hit_delta"]) >= int(args.min_eval_mcq_category_hit_delta)
        )
    report = {
        "status": "complete",
        "decision": (
            "accepted_integrated_language_knowledge_healing"
            if text_accepted and accepted_language and mcq_accepted
            else "rejected_integrated_language_knowledge_healing"
        ),
        "accepted": bool(text_accepted and accepted_language and mcq_accepted),
        "accepted_text_ce": bool(text_accepted),
        "accepted_mcq_validation": bool(mcq_accepted),
        "accepted_raw_mcq_gain": bool(raw_mcq_gain_accepted),
        "accepted_base_wrong_correction": bool(correction_accepted),
        "accepted_base_correct_preservation": bool(regression_accepted),
        "accepted_language": bool(accepted_language),
        "target_level": "M4/M5 integrated language knowledge healing",
        "model_id": str(args.model_id),
        "checkpoint": str(args.init_checkpoint),
        "canonical_path": "Qwen3.5 tokenizer/backbone -> mandatory QTRM core -> Qwen3.5 LM head",
        "runtime_donor": False,
        "mandatory_core": bool(args.mandatory_core),
        "core_impl": str(args.core_impl),
        "core_insertion_mode": str(args.core_insertion_mode),
        "core_insert_after_layer": int(args.core_insert_after_layer),
        "core_adapter_dim": int(args.core_adapter_dim),
        "core_delta_adapter_mode": str(args.core_delta_adapter_mode),
        "core_residual_gate_mode": str(args.core_residual_gate_mode),
        "core_residual_gate_dim": int(args.core_residual_gate_dim),
        "core_residual_gate_init": float(args.core_residual_gate_init),
        "residual_gate_lr_multiplier": float(args.residual_gate_lr_multiplier),
        "residual_scale": float(args.residual_scale),
        "clone_qwen_core_layers": bool(args.clone_qwen_core_layers),
        "n_core_layers": int(args.n_core_layers),
        "h_cycles": int(args.h_cycles),
        "l_cycles": int(args.l_cycles),
        "outer_steps": int(args.outer_steps),
        "core_convergence_halt_enabled": bool(args.core_convergence_halt_enabled),
        "core_convergence_halt_threshold": float(args.core_convergence_halt_threshold),
        "core_convergence_halt_min_outer": int(args.core_convergence_halt_min_outer),
        "core_step_conditioning_enabled": bool(args.core_step_conditioning_enabled),
        "core_step_conditioning_max_steps": int(args.core_step_conditioning_max_steps),
        "core_step_conditioning_scale": float(args.core_step_conditioning_scale),
        "qwen_trainability": qwen_trainability,
        "adapter_trainability": adapter_trainability,
        "checkpoint_info": checkpoint_info,
        "model_report": model.report().__dict__,
        "text_jsonl": list(args.text_jsonl),
        "text_rows": len(text_rows),
        "train_text_rows": len(train_text_rows),
        "eval_text_rows": len(eval_text_rows),
        "mcq_jsonl": str(args.mcq_jsonl),
        "mcq_validation_jsonl": str(args.mcq_validation_jsonl),
        "train_mcq_rows": len(train_mcq_rows),
        "eval_mcq_rows": len(eval_mcq_rows),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "mcq_batch_size": int(args.mcq_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "lr": float(args.lr),
        "qwen_lr": float(args.qwen_lr),
        "base_kl_weight": float(args.base_kl_weight),
        "language_anchor_weight": float(args.language_anchor_weight),
        "language_anchor_batch_size": int(args.language_anchor_batch_size),
        "mcq_weight": float(args.mcq_weight),
        "mcq_ce_focus": str(args.mcq_ce_focus),
        "mcq_loss_space": str(args.mcq_loss_space),
        "mcq_margin_weight": float(args.mcq_margin_weight),
        "mcq_margin_value": float(args.mcq_margin_value),
        "mcq_margin_focus": str(args.mcq_margin_focus),
        "base_wrong_max_top_margin": float(args.base_wrong_max_top_margin),
        "mcq_non_selected_option_kl_weight": float(args.mcq_non_selected_option_kl_weight),
        "residual_gate_selected_open_weight": float(args.residual_gate_selected_open_weight),
        "residual_gate_non_selected_closed_weight": float(args.residual_gate_non_selected_closed_weight),
        "base_wrong_mcq_retries": int(args.base_wrong_mcq_retries),
        "base_correct_option_kl_weight": float(args.base_correct_option_kl_weight),
        "base_correct_option_kl_focus": str(args.base_correct_option_kl_focus),
        "base_correct_kl_extra_batch_size": int(args.base_correct_kl_extra_batch_size),
        "balanced_mcq_category_sampling": bool(args.balanced_mcq_category_sampling),
        "before_text": before_text,
        "after_text": after_text,
        "core_ce_delta": float(core_ce_delta),
        "before_mcq": before_mcq,
        "after_mcq": after_mcq,
        "mcq_category_gain_summary": mcq_category_summary,
        "language": language,
        "train": {
            "last_loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / max(1, len(losses)),
            "best_periodic_eval": best_eval,
            "restored_best_checkpoint": bool(best_state is not None and bool(args.restore_best_checkpoint)),
        },
        "thresholds": {
            "max_core_ce_regression": float(args.max_core_ce_regression),
            "min_eval_mcq_gain": float(args.min_eval_mcq_gain),
            "min_base_wrong_core_correct": int(args.min_base_wrong_core_correct),
            "max_base_correct_core_wrong": int(args.max_base_correct_core_wrong),
            "min_eval_mcq_category_gain": float(args.min_eval_mcq_category_gain),
            "min_eval_mcq_category_hit_delta": int(args.min_eval_mcq_category_hit_delta),
            "category_guard_min_cases": int(args.category_guard_min_cases),
            "min_language_top1_agreement": float(args.min_language_top1_agreement),
            "max_repeated_token_run": int(args.max_repeated_token_run),
            "min_unique_ratio": float(args.min_unique_ratio),
        },
        "limitations": [
            "This is not public benchmark acceptance.",
            "Promotion still requires an independent 390 MMLU-Pro 1024/full gate.",
            "The text data must stay non-test and should not include hidden answer channels.",
            "A preservation-only pass is not a raw-intelligence improvement unless after_mcq.gain and base_wrong_core_correct are positive.",
        ],
    }
    return report, trainable_state_dict(model)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--checkpoint-load-mode", choices=["strict_shapes", "skip_mismatch"], default="strict_shapes")
    parser.add_argument("--text-jsonl", action="append", default=[])
    parser.add_argument("--mcq-jsonl", default="")
    parser.add_argument("--mcq-validation-jsonl", default="")
    parser.add_argument("--out-dir", default="local_eval/qwen35_integrated_language_knowledge_healing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-text-rows", type=int, default=6000)
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--eval-text-rows", type=int, default=128)
    parser.add_argument("--max-mcq-rows", type=int, default=2000)
    parser.add_argument("--eval-mcq-rows", type=int, default=128)
    parser.add_argument(
        "--core-impl",
        choices=["qwen_layer_wrapped", "qwen_shared_layer_wrapped"],
        default="qwen_layer_wrapped",
    )
    parser.add_argument(
        "--core-insertion-mode",
        choices=["final_residual", "mid_layer_suffix"],
        default="mid_layer_suffix",
    )
    parser.add_argument("--core-insert-after-layer", type=int, default=11)
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--core-adapter-dim", type=int, default=128)
    parser.add_argument("--core-delta-adapter-mode", choices=["add", "adapter_only"], default="add")
    parser.add_argument("--core-residual-gate-mode", choices=["constant", "token_mlp"], default="constant")
    parser.add_argument("--core-residual-gate-dim", type=int, default=128)
    parser.add_argument("--core-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--residual-gate-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--train-only-core-delta-adapter", action="store_true")
    parser.add_argument("--clone-qwen-core-layers", action="store_true")
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=3)
    parser.add_argument("--l-cycles", type=int, default=6)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument(
        "--core-convergence-halt-enabled",
        dest="core_convergence_halt_enabled",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-core-convergence-halt",
        dest="core_convergence_halt_enabled",
        action="store_false",
    )
    parser.add_argument("--core-convergence-halt-threshold", type=float, default=0.2)
    parser.add_argument("--core-convergence-halt-min-outer", type=int, default=1)
    parser.add_argument(
        "--core-step-conditioning-enabled",
        dest="core_step_conditioning_enabled",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-core-step-conditioning",
        dest="core_step_conditioning_enabled",
        action="store_false",
    )
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--unfreeze-qwen-layer-indices", default="23")
    parser.add_argument("--unfreeze-qwen-lm-head", action="store_true")
    parser.add_argument("--unfreeze-qwen-final-norm", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mcq-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--qwen-lr", type=float, default=1.0e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-kl-weight", type=float, default=0.10)
    parser.add_argument("--language-anchor-weight", type=float, default=0.0)
    parser.add_argument("--language-anchor-batch-size", type=int, default=4)
    parser.add_argument("--mcq-weight", type=float, default=0.10)
    parser.add_argument("--mcq-ce-focus", choices=["all", "base_wrong"], default="all")
    parser.add_argument("--mcq-loss-space", choices=["full_vocab", "option_only"], default="full_vocab")
    parser.add_argument("--mcq-margin-weight", type=float, default=0.0)
    parser.add_argument("--mcq-margin-value", type=float, default=0.5)
    parser.add_argument("--mcq-margin-focus", choices=["all", "base_wrong"], default="base_wrong")
    parser.add_argument("--base-wrong-max-top-margin", type=float, default=-1.0)
    parser.add_argument("--mcq-non-selected-option-kl-weight", type=float, default=0.0)
    parser.add_argument("--residual-gate-selected-open-weight", type=float, default=0.0)
    parser.add_argument("--residual-gate-non-selected-closed-weight", type=float, default=0.0)
    parser.add_argument("--base-wrong-mcq-retries", type=int, default=1)
    parser.add_argument("--base-correct-option-kl-weight", type=float, default=0.0)
    parser.add_argument("--base-correct-option-kl-focus", choices=["all", "base_correct", "base_wrong"], default="base_correct")
    parser.add_argument("--base-correct-kl-extra-batch-size", type=int, default=0)
    parser.add_argument("--balanced-mcq-category-sampling", action="store_true")
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--restore-best-checkpoint", action="store_true")
    parser.add_argument("--category-regression-penalty", type=float, default=0.0)
    parser.add_argument("--text-ce-regression-penalty", type=float, default=1.0)
    parser.add_argument("--max-core-ce-regression", type=float, default=0.01)
    parser.add_argument("--min-eval-mcq-gain", type=float, default=0.0)
    parser.add_argument("--min-base-wrong-core-correct", type=int, default=0)
    parser.add_argument("--max-base-correct-core-wrong", type=int, default=1_000_000)
    parser.add_argument("--min-eval-mcq-category-gain", type=float, default=-1.0)
    parser.add_argument("--min-eval-mcq-category-hit-delta", type=int, default=-1_000_000)
    parser.add_argument("--category-guard-min-cases", type=int, default=1)
    parser.add_argument("--min-language-top1-agreement", type=float, default=0.75)
    parser.add_argument("--max-repeated-token-run", type=int, default=8)
    parser.add_argument("--min-unique-ratio", type=float, default=0.20)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-generation-prompts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-checkpoint", default="")
    parser.add_argument("--skip-save-checkpoint", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.text_jsonl:
        args.text_jsonl = ["local_eval/external_language_corpus/qtrm_native_external_bilingual_9000_20260515.jsonl"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not str(args.save_checkpoint):
        args.save_checkpoint = str(out_dir / "last_core.pt")
    report, state = train(args)
    if not bool(args.skip_save_checkpoint):
        torch.save({"model": state, "report": report}, str(args.save_checkpoint))
    else:
        report["checkpoint_save_skipped"] = True
        report["save_checkpoint"] = str(args.save_checkpoint)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
