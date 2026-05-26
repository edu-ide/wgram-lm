#!/usr/bin/env python3
"""Train a one-body BLT checkpoint toward answer-facing attractors.

This is the smallest falsification experiment for the Stage101 idea:

  prompt
  -> recurrent latent thinking at multiple depths
  -> same LM head must prefer the intelligence answer over the parrot answer
  -> deeper depths should not lose that margin

It deliberately avoids bridge/readback/selector paths.  The training signal is
the same final LM head used by normal generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IGNORE_LABEL_ID = -100
SOURCE_TEMPLATE_NAMES = ("context_first", "claim_first", "answer_from_note", "after_question")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_depth_probe_module() -> Any:
    return load_module(
        ROOT / "scripts" / "560_eval_blt_depth_residual_probe.py",
        "blt_depth_probe_for_solution_attractor_train",
    )


def load_gd_module() -> Any:
    return load_module(
        ROOT / "scripts" / "567_eval_blt_generalization_dynamics_probe.py",
        "gd_lite_for_solution_attractor_train",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in JSONL: {path}")
    return rows


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def make_amp_context(device: torch.device, amp_dtype: torch.dtype | None) -> Any:
    if amp_dtype is None or str(device.type) != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    lowered = str(name).lower()
    if lowered in {"", "none", "fp32", "float32"}:
        return None
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unknown amp dtype: {name}")


def choice_mean_logprob_tensor(
    model: torch.nn.Module,
    gd_module: Any,
    *,
    prompt: str,
    answer: str,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    think_steps: int,
) -> tuple[torch.Tensor, int]:
    input_ids, labels, attention_mask = gd_module.build_choice_tensors(
        prompt=str(prompt),
        answer=str(answer),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
    )
    logits = model.forward_logits(
        input_ids,
        attention_mask,
        think_steps=int(think_steps),
    )
    length = min(int(logits.shape[1]), int(labels.shape[1]))
    log_probs = F.log_softmax(logits[:, :length].float(), dim=-1)
    labels = labels[:, :length]
    mask = labels.ne(IGNORE_LABEL_ID)
    if not bool(mask.any()):
        raise ValueError("choice row has no answer target tokens")
    token_log_probs = log_probs[mask].gather(1, labels[mask].unsqueeze(1)).squeeze(1)
    return token_log_probs.mean(), int(token_log_probs.numel())


def negative_answers_for_row(row: dict[str, Any]) -> list[str]:
    target = str(row["intelligence_answer"])
    raw_negatives = row.get("negative_answers")
    if isinstance(raw_negatives, list):
        candidates = [str(item) for item in raw_negatives]
    else:
        candidates = [str(row["parrot_answer"])]
    out: list[str] = []
    seen: set[str] = set()
    for answer in candidates:
        if answer == target or answer in seen:
            continue
        seen.add(answer)
        out.append(answer)
    if not out:
        raise ValueError(f"row has no negative answers: {row.get('id')}")
    return out


def multi_negative_choice_margins(
    model: torch.nn.Module,
    gd_module: Any,
    row: dict[str, Any],
    *,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    think_steps: int,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, Any]]:
    intelligence_mean, intelligence_tokens = choice_mean_logprob_tensor(
        model,
        gd_module,
        prompt=str(row["prompt"]),
        answer=str(row["intelligence_answer"]),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(think_steps),
    )
    negative_means: list[torch.Tensor] = []
    negative_token_counts: list[int] = []
    negative_answers = negative_answers_for_row(row)
    for answer in negative_answers:
        negative_mean, negative_tokens = choice_mean_logprob_tensor(
            model,
            gd_module,
            prompt=str(row["prompt"]),
            answer=str(answer),
            seq_len=int(seq_len),
            byte_offset=int(byte_offset),
            device=device,
            think_steps=int(think_steps),
        )
        negative_means.append(negative_mean)
        negative_token_counts.append(int(negative_tokens))
    margins = [intelligence_mean - negative_mean for negative_mean in negative_means]
    hardest_margin = torch.stack([margin.float() for margin in margins]).min()
    max_negative_mean = torch.stack([mean.float() for mean in negative_means]).max()
    predicted_answer = (
        str(row["intelligence_answer"])
        if float(intelligence_mean.detach().float().cpu().item())
        >= float(max_negative_mean.detach().cpu().item())
        else negative_answers[
            int(torch.stack([mean.float() for mean in negative_means]).argmax().detach().cpu().item())
        ]
    )
    metrics = {
        "intelligence_mean_logprob": float(intelligence_mean.detach().cpu().item()),
        "intelligence_tokens": int(intelligence_tokens),
        "negative_answers": negative_answers,
        "negative_mean_logprobs": [
            float(mean.detach().cpu().item()) for mean in negative_means
        ],
        "negative_tokens": negative_token_counts,
        "hardest_margin": float(hardest_margin.detach().cpu().item()),
        "predicted_answer": predicted_answer,
    }
    return intelligence_mean, margins, metrics


def contrastive_terms_from_margins(
    margins: list[torch.Tensor],
    *,
    target_margin: float,
    monotonic_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not margins:
        raise ValueError("margins must not be empty")
    margin_tensor = torch.stack([margin.float() for margin in margins])
    rank_loss = F.softplus(float(target_margin) - margin_tensor).mean()
    if len(margins) <= 1:
        monotonic_loss = rank_loss * 0.0
    else:
        diffs = []
        for previous, current in zip(margins[:-1], margins[1:], strict=True):
            diffs.append(F.softplus(previous.float() + float(monotonic_gain) - current.float()))
        monotonic_loss = torch.stack(diffs).mean()
    final_margin = margin_tensor[-1]
    return rank_loss, monotonic_loss, final_margin


def source_template_group_key(row: dict[str, Any]) -> str | None:
    template = row.get("source_template")
    if template not in SOURCE_TEMPLATE_NAMES:
        return None
    row_id = str(row.get("id", ""))
    pattern = (
        r"^(?P<prefix>.+?)_(?P<template>"
        + "|".join(re.escape(name) for name in SOURCE_TEMPLATE_NAMES)
        + r")(?:_replay\d+)?$"
    )
    match = re.match(pattern, row_id)
    if match:
        return str(match.group("prefix"))
    axis = str(row.get("plain_language_axis", "")).strip()
    if axis:
        return f"axis:{axis}"
    return None


def build_source_template_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = source_template_group_key(row)
        template = row.get("source_template")
        if key is None or template not in SOURCE_TEMPLATE_NAMES:
            continue
        grouped.setdefault(key, {})
        grouped[key].setdefault(str(template), row)
    return {
        key: [by_template[name] for name in SOURCE_TEMPLATE_NAMES if name in by_template]
        for key, by_template in grouped.items()
        if len(by_template) >= 2
    }


def template_consistency_terms_from_margins(
    margins: list[torch.Tensor],
    *,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not margins:
        raise ValueError("template consistency margins cannot be empty")
    margin_tensor = torch.stack([margin.float() for margin in margins])
    rank_loss = F.softplus(float(target_margin) - margin_tensor).mean()
    variance_loss = (
        (margin_tensor - margin_tensor.mean()).pow(2).mean()
        if len(margins) > 1
        else rank_loss * 0.0
    )
    return rank_loss + variance_loss, rank_loss, variance_loss


def row_contrastive_loss(
    model: torch.nn.Module,
    gd_module: Any,
    row: dict[str, Any],
    *,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    target_margin: float,
    monotonic_gain: float,
    intelligence_nll_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    hardest_margins_by_depth: list[torch.Tensor] = []
    all_negative_margins: list[torch.Tensor] = []
    intelligence_losses: list[torch.Tensor] = []
    token_counts: list[int] = []
    last_choice_metrics: dict[str, Any] = {}
    for depth in depths:
        intelligence_mean, negative_margins, choice_metrics = multi_negative_choice_margins(
            model,
            gd_module,
            row,
            seq_len=int(seq_len),
            byte_offset=int(byte_offset),
            device=device,
            think_steps=int(depth),
        )
        hardest_margin = torch.stack([margin.float() for margin in negative_margins]).min()
        hardest_margins_by_depth.append(hardest_margin)
        all_negative_margins.extend(negative_margins)
        intelligence_losses.append(-intelligence_mean)
        token_counts.append(int(choice_metrics["intelligence_tokens"]))
        last_choice_metrics = choice_metrics
    all_margin_tensor = torch.stack([margin.float() for margin in all_negative_margins])
    rank_loss = F.softplus(float(target_margin) - all_margin_tensor).mean()
    if len(hardest_margins_by_depth) <= 1:
        monotonic_loss = rank_loss * 0.0
    else:
        monotonic_loss = torch.stack(
            [
                F.softplus(previous.float() + float(monotonic_gain) - current.float())
                for previous, current in zip(
                    hardest_margins_by_depth[:-1],
                    hardest_margins_by_depth[1:],
                    strict=True,
                )
            ]
        ).mean()
    final_margin = hardest_margins_by_depth[-1]
    intelligence_nll = torch.stack(intelligence_losses).mean()
    loss = rank_loss + monotonic_loss + float(intelligence_nll_weight) * intelligence_nll
    metrics = {
        "id": row.get("id"),
        "task": row.get("task"),
        "loss": float(loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "monotonic_loss": float(monotonic_loss.detach().cpu().item()),
        "intelligence_nll": float(intelligence_nll.detach().cpu().item()),
        "final_margin": float(final_margin.detach().cpu().item()),
        "mean_margin": float(
            torch.stack([m.detach().float() for m in hardest_margins_by_depth]).mean().cpu().item()
        ),
        "min_all_negative_margin": float(all_margin_tensor.detach().min().cpu().item()),
        "negative_answers": last_choice_metrics.get("negative_answers", []),
        "depths": [int(depth) for depth in depths],
        "token_count_min": int(min(token_counts) if token_counts else 0),
    }
    return loss, metrics


def source_template_consistency_loss(
    model: torch.nn.Module,
    gd_module: Any,
    rows: list[dict[str, Any]],
    *,
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    target_margin: float,
    max_rows: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected = rows[: max(1, int(max_rows))]
    margins: list[torch.Tensor] = []
    templates: list[str] = []
    for row in selected:
        _intelligence_mean, negative_margins, _choice_metrics = multi_negative_choice_margins(
            model,
            gd_module,
            row,
            seq_len=int(seq_len),
            byte_offset=int(byte_offset),
            device=device,
            think_steps=int(depth),
        )
        margins.append(torch.stack([margin.float() for margin in negative_margins]).min())
        templates.append(str(row.get("source_template", "")))
    if not margins:
        raise ValueError("template consistency requires at least one row")
    margin_tensor = torch.stack([margin.float() for margin in margins])
    loss, rank_loss, variance_loss = template_consistency_terms_from_margins(
        margins,
        target_margin=float(target_margin),
    )
    metrics = {
        "template_consistency_loss": float(loss.detach().cpu().item()),
        "template_consistency_rank_loss": float(rank_loss.detach().cpu().item()),
        "template_consistency_variance_loss": float(variance_loss.detach().cpu().item()),
        "template_consistency_mean_margin": float(margin_tensor.detach().mean().cpu().item()),
        "template_consistency_min_margin": float(margin_tensor.detach().min().cpu().item()),
        "template_consistency_depth": int(depth),
        "template_consistency_rows": int(len(selected)),
        "template_consistency_templates": templates,
    }
    return loss, metrics


def masked_teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    max_targets: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    seq_len = min(int(student_logits.shape[1]), int(teacher_logits.shape[1]), int(labels.shape[1]))
    student = student_logits[:, :seq_len].reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits[:, :seq_len].reshape(-1, teacher_logits.shape[-1])
    flat_labels = labels[:, :seq_len].reshape(-1)
    target_mask = flat_labels.ne(IGNORE_LABEL_ID)
    if not bool(target_mask.any()):
        zero = student_logits.sum() * 0.0
        return zero, {
            "language_preserve_kl_loss": 0.0,
            "language_preserve_targets": 0,
            "language_preserve_teacher_entropy": 0.0,
        }
    student = student[target_mask]
    teacher = teacher[target_mask].detach()
    if int(max_targets) > 0 and int(student.shape[0]) > int(max_targets):
        indices = torch.linspace(
            0,
            int(student.shape[0]) - 1,
            steps=int(max_targets),
            device=student.device,
        ).long()
        student = student[indices]
        teacher = teacher[indices]
    temp = float(max(1e-6, temperature))
    teacher_probs = F.softmax(teacher.float() / temp, dim=-1)
    student_log_probs = F.log_softmax(student.float() / temp, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
    teacher_entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return loss.to(student_logits.dtype), {
        "language_preserve_kl_loss": float(loss.detach().cpu().item()),
        "language_preserve_targets": int(student.shape[0]),
        "language_preserve_teacher_entropy": float(teacher_entropy.detach().cpu().item()),
    }


def language_preserving_kl_loss(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    think_steps: int,
    temperature: float,
    max_targets: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    student_logits = model.forward_logits(
        input_ids,
        attention_mask,
        think_steps=int(think_steps),
    )
    with torch.no_grad():
        teacher_logits = teacher_model.forward_logits(
            input_ids,
            attention_mask,
            think_steps=int(think_steps),
        )
    return masked_teacher_kl_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=float(temperature),
        max_targets=int(max_targets),
    )


@torch.no_grad()
def evaluate_probe_rows(
    model: torch.nn.Module,
    gd_module: Any,
    rows: list[dict[str, Any]],
    *,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    by_depth: list[dict[str, Any]] = []
    context = make_amp_context(device, amp_dtype)
    for depth in depths:
        eval_rows: list[dict[str, Any]] = []
        with context:
            for row in rows:
                intelligence_mean, negative_margins, choice_metrics = multi_negative_choice_margins(
                    model,
                    gd_module,
                    row,
                    seq_len=int(seq_len),
                    byte_offset=int(byte_offset),
                    device=device,
                    think_steps=int(depth),
                )
                margin = float(
                    torch.stack([item.float() for item in negative_margins]).min().detach().cpu().item()
                )
                negative_logprobs = list(choice_metrics.get("negative_mean_logprobs", []))
                eval_rows.append(
                    {
                        "id": row.get("id"),
                        "task": row.get("task"),
                        "intelligence_mean_logprob": float(intelligence_mean.detach().cpu().item()),
                        "parrot_mean_logprob": float(max(negative_logprobs)) if negative_logprobs else float("nan"),
                        "intelligence_tokens": int(choice_metrics["intelligence_tokens"]),
                        "parrot_tokens": int(
                            min(choice_metrics.get("negative_tokens", [0]))
                            if choice_metrics.get("negative_tokens")
                            else 0
                        ),
                        "target_answer": str(row["intelligence_answer"]),
                        "negative_answers": choice_metrics.get("negative_answers", []),
                        "negative_mean_logprobs": negative_logprobs,
                        "predicted_answer": choice_metrics.get("predicted_answer"),
                        "normalized_margin": float(margin),
                        "correct": bool(margin > 0.0),
                    }
                )
        margins = [float(row["normalized_margin"]) for row in eval_rows]
        correct = sum(1 for row in eval_rows if bool(row["correct"]))
        failed_tasks = [str(row["task"]) for row in eval_rows if not bool(row["correct"])]
        passed_tasks = [str(row["task"]) for row in eval_rows if bool(row["correct"])]
        by_depth.append(
            {
                "depth": int(depth),
                "accuracy": float(correct / float(max(1, len(eval_rows)))),
                "mean_margin": float(sum(margins) / float(max(1, len(margins)))),
                "min_margin": float(min(margins)) if margins else float("nan"),
                "passed_tasks": passed_tasks,
                "failed_tasks": failed_tasks,
                "rows": eval_rows,
            }
        )
    return {
        "depths": by_depth,
        "accepted": bool(by_depth and by_depth[-1]["accuracy"] == 1.0 and by_depth[-1]["min_margin"] > 0.0),
    }


def build_checkpoint_args(ckpt_args: argparse.Namespace, args: argparse.Namespace) -> dict[str, Any]:
    values = vars(ckpt_args).copy()
    values.update(
        {
            "stage101_solution_aligned_answer_attractor": True,
            "stage101_source_checkpoint": str(args.checkpoint),
            "stage101_probe_jsonl": str(args.probe_jsonl),
            "stage101_depths": [int(depth) for depth in args.depths],
            "stage101_target_margin": float(args.target_margin),
            "stage101_monotonic_gain": float(args.monotonic_gain),
            "stage101_intelligence_nll_weight": float(args.intelligence_nll_weight),
            "stage101_language_preserve_weight": float(args.language_preserve_weight),
            "stage101_language_sampled_data": str(args.language_sampled_data),
            "stage101_template_consistency_weight": float(args.template_consistency_weight),
            "stage101_template_consistency_depth": int(args.template_consistency_depth),
            "stage101_steps": int(args.steps),
            "stage101_lr": float(args.lr),
        }
    )
    return values


def save_model_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args_payload: dict[str, Any],
    dataset_summary: dict[str, Any],
    model_summary: dict[str, Any],
    train_history: list[dict[str, Any]],
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
    include_optimizer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "checkpoint_includes_optimizer": bool(include_optimizer),
        "args": args_payload,
        "dataset": dataset_summary,
        "model": model_summary,
        "loss_history": train_history,
        "eval_before": eval_before,
        "eval_after": eval_after,
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp = path.with_name(f".{path.name}.tmp.{Path.cwd().name}.{int(time.time())}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    if not args.depths:
        raise ValueError("--depths must contain at least one depth")
    depths = sorted({max(1, int(depth)) for depth in args.depths})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    amp_dtype = resolve_amp_dtype(str(args.amp_dtype))
    gd_module = load_gd_module()
    depth_probe = load_depth_probe_module()
    rows = load_jsonl(Path(args.probe_jsonl))
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    source_template_groups = build_source_template_groups(rows)
    trainer, prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(out_dir),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    model.train()
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )
    language_loader = None
    language_iterator = None
    language_teacher_model = None
    if float(args.language_preserve_weight) > 0.0:
        if not str(args.language_sampled_data):
            raise ValueError("--language-sampled-data is required when --language-preserve-weight > 0")
        teacher_checkpoint = Path(str(args.language_teacher_checkpoint or args.checkpoint))
        _teacher_trainer, _teacher_prefix, _teacher_args, teacher_loaded = depth_probe.load_checkpoint_model(
            checkpoint_path=teacher_checkpoint,
            sampled_data=str(args.sampled_data),
            out_dir=str(out_dir / "language_teacher_load"),
            device=device,
            amp_dtype=str(args.amp_dtype),
        )
        language_teacher_model = teacher_loaded["model"]
        language_teacher_model.eval()
        for parameter in language_teacher_model.parameters():
            parameter.requires_grad = False
        language_dataset = prefix.DataIOSampledPrefixLMDataset(
            str(args.language_sampled_data),
            seq_len=int(seq_len),
            epoch=int(args.language_epoch),
            target_only=True,
            max_rows=int(args.language_max_rows) if int(args.language_max_rows) > 0 else None,
            drop_overlength=True,
        )
        language_loader = DataLoader(
            language_dataset,
            batch_size=int(args.language_batch_size),
            shuffle=False,
            collate_fn=prefix.collate_prefixlm_rows,
            drop_last=False,
        )
        language_iterator = iter(language_loader)
    eval_before = evaluate_probe_rows(
        model,
        gd_module,
        rows,
        depths=depths,
        seq_len=seq_len,
        byte_offset=byte_offset,
        device=device,
        amp_dtype=amp_dtype,
    )
    train_history: list[dict[str, Any]] = []
    context = make_amp_context(device, amp_dtype)
    for step in range(1, int(args.steps) + 1):
        row = rows[(step - 1) % len(rows)]
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with context:
            loss, metrics = row_contrastive_loss(
                model,
                gd_module,
                row,
                depths=depths,
                seq_len=seq_len,
                byte_offset=byte_offset,
                device=device,
                target_margin=float(args.target_margin),
                monotonic_gain=float(args.monotonic_gain),
                intelligence_nll_weight=float(args.intelligence_nll_weight),
            )
            if (
                float(args.template_consistency_weight) > 0.0
                and int(args.template_consistency_every) > 0
                and step % int(args.template_consistency_every) == 0
            ):
                group_key = source_template_group_key(row)
                group_rows = source_template_groups.get(str(group_key or ""), [])
                if len(group_rows) >= 2:
                    consistency_depth = int(args.template_consistency_depth)
                    if consistency_depth <= 0:
                        consistency_depth = int(depths[-1])
                    consistency_loss, consistency_metrics = source_template_consistency_loss(
                        model,
                        gd_module,
                        group_rows,
                        depth=int(consistency_depth),
                        seq_len=seq_len,
                        byte_offset=byte_offset,
                        device=device,
                        target_margin=float(args.target_margin),
                        max_rows=int(args.template_consistency_max_rows),
                    )
                    loss = loss + float(args.template_consistency_weight) * consistency_loss
                    metrics.update(consistency_metrics)
                    metrics["template_consistency_group"] = str(group_key)
            if (
                language_teacher_model is not None
                and language_loader is not None
                and language_iterator is not None
                and int(args.language_preserve_every) > 0
                and step % int(args.language_preserve_every) == 0
            ):
                try:
                    language_batch = next(language_iterator)
                except StopIteration:
                    language_iterator = iter(language_loader)
                    language_batch = next(language_iterator)
                language_batch = prefix.trim_prefixlm_batch_to_max_valid_length(language_batch)
                preserve_depth = int(args.language_preserve_depth)
                if preserve_depth <= 0:
                    preserve_depth = int(depths[-1])
                preserve_loss, preserve_metrics = language_preserving_kl_loss(
                    model,
                    language_teacher_model,
                    language_batch,
                    device=device,
                    think_steps=int(preserve_depth),
                    temperature=float(args.language_preserve_temperature),
                    max_targets=int(args.language_preserve_max_targets),
                )
                loss = loss + float(args.language_preserve_weight) * preserve_loss
                metrics.update(preserve_metrics)
                metrics["language_preserve_depth"] = int(preserve_depth)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach().cpu().item())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        metrics = {"step": int(step), **metrics}
        train_history.append(metrics)
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    eval_after = evaluate_probe_rows(
        model,
        gd_module,
        rows,
        depths=depths,
        seq_len=seq_len,
        byte_offset=byte_offset,
        device=device,
        amp_dtype=amp_dtype,
    )
    args_payload = build_checkpoint_args(ckpt_args, args)
    model_summary = dict(loaded.get("model_summary") or {})
    model_summary.update(
        {
            "stage101_solution_aligned_answer_attractor": {
                "depths": depths,
                "target_margin": float(args.target_margin),
                "monotonic_gain": float(args.monotonic_gain),
                "intelligence_nll_weight": float(args.intelligence_nll_weight),
                "language_preserve_weight": float(args.language_preserve_weight),
                "language_sampled_data": str(args.language_sampled_data),
                "language_preserve_temperature": float(args.language_preserve_temperature),
                "language_preserve_max_targets": int(args.language_preserve_max_targets),
                "template_consistency_weight": float(args.template_consistency_weight),
                "template_consistency_depth": int(args.template_consistency_depth),
                "plain_language_role": (
                    "Train the same LM head to prefer intelligence answers over "
                    "parrot answers as recurrent depth increases."
                ),
            }
        }
    )
    dataset_summary = dict(loaded.get("dataset_summary") or {})
    save_model_checkpoint(
        out_dir / "last_model.pt",
        model=model,
        optimizer=optimizer,
        step=int(args.steps),
        args_payload=args_payload,
        dataset_summary=dataset_summary,
        model_summary=model_summary,
        train_history=train_history,
        eval_before=eval_before,
        eval_after=eval_after,
        include_optimizer=False,
    )
    if bool(args.save_optimizer_checkpoint):
        save_model_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            step=int(args.steps),
            args_payload=args_payload,
            dataset_summary=dataset_summary,
            model_summary=model_summary,
            train_history=train_history,
            eval_before=eval_before,
            eval_after=eval_after,
            include_optimizer=True,
        )
    before_final = eval_before["depths"][-1]
    after_final = eval_after["depths"][-1]
    report = {
        "decision": "stage101_solution_aligned_answer_attractor_smoke",
        "accepted": bool(eval_after.get("accepted", False)),
        "checkpoint_in": str(args.checkpoint),
        "checkpoint_out": str(out_dir / "last_model.pt"),
        "probe_jsonl": str(args.probe_jsonl),
        "seq_len": int(seq_len),
        "byte_offset": int(byte_offset),
        "depths": depths,
        "steps": int(args.steps),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "final_depth_margin_gain": float(after_final["mean_margin"] - before_final["mean_margin"]),
        "final_depth_accuracy_gain": float(after_final["accuracy"] - before_final["accuracy"]),
        "plain_language_read": (
            "This smoke asks whether the same one-body mouth can be trained to "
            "prefer intelligence answers over tempting parrot answers. Passing "
            "this smoke is not a generalization claim unless held-out GD-lite "
            "rows and language generation also improve."
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-jsonl", default="data/eval/generalization_dynamics_lite_probe.jsonl")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-margin", type=float, default=0.1)
    parser.add_argument("--monotonic-gain", type=float, default=0.02)
    parser.add_argument("--intelligence-nll-weight", type=float, default=0.02)
    parser.add_argument(
        "--template-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Same-mouth source template consistency loss. For rows sharing a "
            "source semantic group, different prompt templates should keep the "
            "same intelligence-vs-parrot margin direction."
        ),
    )
    parser.add_argument("--template-consistency-every", type=int, default=1)
    parser.add_argument("--template-consistency-depth", type=int, default=0)
    parser.add_argument("--template-consistency-max-rows", type=int, default=4)
    parser.add_argument(
        "--language-sampled-data",
        default="",
        help="Optional tokenizer-free sampled data used for language-preserving KL.",
    )
    parser.add_argument(
        "--language-teacher-checkpoint",
        default="",
        help="Frozen teacher checkpoint for language preservation; defaults to --checkpoint.",
    )
    parser.add_argument("--language-preserve-weight", type=float, default=0.0)
    parser.add_argument("--language-preserve-temperature", type=float, default=1.0)
    parser.add_argument("--language-preserve-max-targets", type=int, default=128)
    parser.add_argument("--language-preserve-every", type=int, default=1)
    parser.add_argument("--language-preserve-depth", type=int, default=0)
    parser.add_argument("--language-epoch", type=int, default=0)
    parser.add_argument("--language-max-rows", type=int, default=0)
    parser.add_argument("--language-batch-size", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--save-optimizer-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
