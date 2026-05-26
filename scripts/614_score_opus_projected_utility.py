#!/usr/bin/env python3
"""Score byte PrefixLM source rows with an OPUS-style projected utility.

This is the missing scorer behind the existing ``--selection-mode utility``
sample builder.  It implements the practical OPUS contract we can run inside
this codebase:

  1. Build a stable proxy direction from held-out/in-domain PrefixLM rows.
  2. For each candidate row, compute the candidate training gradient.
  3. Shape the candidate gradient by the AdamW optimizer state when available.
  4. Project both directions with a deterministic CountSketch.
  5. Score candidates by alignment with the proxy direction, optionally with a
     redundancy penalty against already selected candidates.

It is strict about checkpoint-level optimizer provenance.  Use
``--preconditioner adamw_state`` for OPUS-style scoring.  If the checkpoint is
model-only, this script fails unless the caller explicitly asks for
``--preconditioner identity`` as a diagnostic approximation.  If a checkpoint
has optimizer state but a few selected tensors do not yet have AdamW moments,
those tensors fall back to identity preconditioning and are reported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


IGNORE_LABEL_ID = -100


@dataclass(frozen=True)
class TextRow:
    source_file: str
    row_index: int
    instruction: str
    response: str
    bucket: str = ""


@dataclass
class PreconditionerFallbackStats:
    missing_exp_avg_sq_parameter_names: set[str] = field(default_factory=set)
    exp_avg_sq_shape_mismatch_parameter_names: set[str] = field(default_factory=set)
    adamw_preconditioned_update_calls: int = 0
    identity_fallback_update_calls: int = 0

    def record_missing_exp_avg_sq(self, name: str) -> None:
        self.identity_fallback_update_calls += 1
        self.missing_exp_avg_sq_parameter_names.add(str(name or "<unnamed>"))

    def record_shape_mismatch(self, name: str) -> None:
        self.identity_fallback_update_calls += 1
        self.exp_avg_sq_shape_mismatch_parameter_names.add(str(name or "<unnamed>"))

    def record_adamw_preconditioned(self) -> None:
        self.adamw_preconditioned_update_calls += 1

    def to_report(self, *, max_names: int = 24) -> dict[str, Any]:
        missing_names = sorted(self.missing_exp_avg_sq_parameter_names)
        mismatch_names = sorted(self.exp_avg_sq_shape_mismatch_parameter_names)
        return {
            "adamw_preconditioned_update_calls": int(self.adamw_preconditioned_update_calls),
            "identity_fallback_update_calls": int(self.identity_fallback_update_calls),
            "missing_exp_avg_sq_parameter_tensors": int(len(missing_names)),
            "exp_avg_sq_shape_mismatch_parameter_tensors": int(len(mismatch_names)),
            "missing_exp_avg_sq_parameter_name_examples": missing_names[: int(max_names)],
            "exp_avg_sq_shape_mismatch_parameter_name_examples": mismatch_names[: int(max_names)],
        }


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_prepare_module() -> Any:
    return load_module(repo_root() / "scripts" / "555_prepare_byte_prefixlm_sample.py", "byte_prepare_for_opus")


def load_trainer_module() -> Any:
    return load_module(repo_root() / "scripts" / "557_train_blt_d_prefixlm_dataio.py", "blt_trainer_for_opus")


def row_to_instruction_response(row: dict[str, Any]) -> tuple[str, str] | None:
    if row.get("prompt") is not None and row.get("intelligence_answer") is not None:
        return str(row["prompt"]), str(row["intelligence_answer"])

    instruction = row.get("instruction")
    response = row.get("response")
    if instruction is not None and response is not None:
        return str(instruction), str(response)

    question = row.get("question")
    aliases = row.get("answer_aliases")
    if question is not None and isinstance(aliases, list) and aliases:
        parts: list[str] = []
        if row.get("instruction") is not None:
            parts.append(str(row["instruction"]))
        evidence = row.get("evidence")
        if isinstance(evidence, list):
            evidence_text = []
            for item in evidence:
                if isinstance(item, dict) and item.get("text") is not None:
                    evidence_text.append(str(item["text"]))
            if evidence_text:
                parts.append("Evidence:\n" + "\n".join(evidence_text))
        parts.append("Question: " + str(question))
        return "\n\n".join(parts), str(aliases[0])

    expected_contains = row.get("expected_contains")
    if question is None:
        question = row.get("prompt", row.get("case_id", ""))
    if expected_contains is not None and isinstance(expected_contains, list) and expected_contains:
        return str(row.get("instruction", question)), str(expected_contains[0])

    return None


def iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if isinstance(value, dict):
                yield value


def split_proxy_paths(text: str) -> list[Path]:
    parts: list[str] = []
    for chunk in str(text or "").replace(",", " ").split():
        parts.extend(shlex.split(chunk))
    if not parts and str(text or "").strip():
        parts = [str(text).strip()]
    return [Path(part) for part in parts if str(part).strip()]


def load_proxy_rows(path: Path | str, *, max_rows: int) -> list[TextRow]:
    rows: list[TextRow] = []
    paths = split_proxy_paths(str(path))
    if not paths:
        raise ValueError("proxy JSONL path is empty")
    for proxy_path in paths:
        for row_index, row in enumerate(iter_jsonl_rows(proxy_path)):
            pair = row_to_instruction_response(row)
            if pair is None:
                continue
            instruction, response = pair
            rows.append(
                TextRow(
                    source_file=str(proxy_path),
                    row_index=int(row.get("row_index", row_index)),
                    instruction=instruction,
                    response=response,
                    bucket=str(row.get("task", row.get("family", row.get("category", "proxy")))),
                )
            )
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
        if int(max_rows) > 0 and len(rows) >= int(max_rows):
            break
    if not rows:
        raise ValueError(f"proxy JSONL produced no instruction/response rows: {path}")
    return rows


def proxy_group_key(row: TextRow, grouping: str) -> str:
    mode = str(grouping or "aggregate").strip().lower()
    if mode in ("", "aggregate", "all"):
        return "all"
    if mode == "source_file":
        return str(row.source_file)
    if mode == "bucket":
        return str(row.bucket or "proxy")
    if mode == "source_file_bucket":
        return f"{row.source_file}::{row.bucket or 'proxy'}"
    raise ValueError(
        f"unsupported proxy_grouping={grouping!r}; "
        "expected aggregate, source_file, bucket, or source_file_bucket"
    )


def group_proxy_rows(rows: list[TextRow], *, grouping: str) -> dict[str, list[TextRow]]:
    grouped: dict[str, list[TextRow]] = {}
    for row in rows:
        grouped.setdefault(proxy_group_key(row, grouping), []).append(row)
    if not grouped:
        raise ValueError("proxy grouping produced no groups")
    return dict(sorted(grouped.items()))


def cap_proxy_groups(
    grouped: dict[str, list[TextRow]],
    *,
    max_rows_per_group: int,
    seed: int,
) -> dict[str, list[TextRow]]:
    if int(max_rows_per_group) <= 0:
        return grouped
    capped: dict[str, list[TextRow]] = {}
    for group_name, rows in sorted(grouped.items()):
        if len(rows) <= int(max_rows_per_group):
            capped[group_name] = list(rows)
            continue
        rng = random.Random(f"{int(seed)}:{group_name}")
        capped[group_name] = rng.sample(list(rows), int(max_rows_per_group))
    return capped


def expand_candidate_files(prepare: Any, data_root: Path, source_files: str, source_globs: str) -> list[str]:
    return prepare.expand_source_files(data_root, str(source_files), str(source_globs))


def iter_candidate_rows(
    prepare: Any,
    *,
    data_root: Path,
    source_files: str,
    source_globs: str,
    max_rows: int,
    max_scan_rows_per_file: int,
    max_inst_bytes: int,
    max_resp_bytes: int,
) -> list[TextRow]:
    candidates: list[TextRow] = []
    for rel in expand_candidate_files(prepare, data_root, source_files, source_globs):
        path = data_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        bucket = str(prepare.source_bucket(rel))
        scanned = 0
        for row in prepare.iter_rows(path):
            if int(max_scan_rows_per_file) > 0 and scanned >= int(max_scan_rows_per_file):
                break
            row_index = int(scanned)
            scanned += 1
            pair = row_to_instruction_response(row)
            if pair is None:
                continue
            instruction, response = pair
            inst = prepare.byte_ids(prepare.render_instruction(instruction))
            resp = prepare.byte_ids(prepare.render_response(response)) + [int(prepare.EOS_TOKEN_ID)]
            if not inst or not resp:
                continue
            if int(max_inst_bytes) > 0 and len(inst) > int(max_inst_bytes):
                continue
            if int(max_resp_bytes) > 0 and len(resp) > int(max_resp_bytes):
                continue
            candidates.append(
                TextRow(
                    source_file=str(Path(rel)),
                    row_index=row_index,
                    instruction=instruction,
                    response=response,
                    bucket=bucket,
                )
            )
            if int(max_rows) > 0 and len(candidates) >= int(max_rows):
                return candidates
    if not candidates:
        raise ValueError("candidate scan produced no usable rows")
    return candidates


def encode_prefixlm_batch(
    prepare: Any,
    rows: list[TextRow],
    *,
    seq_len: int,
    train_instruction_tokens: bool = False,
) -> dict[str, torch.Tensor]:
    input_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for row in rows:
        inst = np.asarray(prepare.byte_ids(prepare.render_instruction(row.instruction)), dtype=np.int64)
        resp = np.asarray(prepare.byte_ids(prepare.render_response(row.response)) + [int(prepare.EOS_TOKEN_ID)], dtype=np.int64)
        input_seq = np.concatenate([inst, resp[:-1]], dtype=np.int64)
        if bool(train_instruction_tokens):
            inst_labels = inst[1:].astype(np.int64)
        else:
            inst_labels = np.full(max(0, len(inst) - 1), IGNORE_LABEL_ID, dtype=np.int64)
        labels = np.concatenate([inst_labels, resp.astype(np.int64)], dtype=np.int64)
        if input_seq.shape[0] != labels.shape[0]:
            raise ValueError("PrefixLM row length mismatch")
        if int(input_seq.shape[0]) > int(seq_len):
            input_seq = input_seq[: int(seq_len)]
            labels = labels[: int(seq_len)]
        attention_mask = np.ones(input_seq.shape[0], dtype=np.int64)
        pad_len = int(seq_len) - int(input_seq.shape[0])
        if pad_len > 0:
            input_seq = np.pad(input_seq, (0, pad_len), constant_values=0)
            labels = np.pad(labels, (0, pad_len), constant_values=IGNORE_LABEL_ID)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)
        if int((labels != IGNORE_LABEL_ID).sum()) <= 0:
            continue
        input_rows.append(input_seq)
        label_rows.append(labels)
        mask_rows.append(attention_mask)
    if not input_rows:
        raise ValueError("encoded batch has no supervised target tokens")
    return {
        "input_ids": torch.from_numpy(np.stack(input_rows)).long(),
        "labels": torch.from_numpy(np.stack(label_rows)).long(),
        "attention_mask": torch.from_numpy(np.stack(mask_rows)).long(),
    }


def row_has_supervised_targets_after_truncation(
    prepare: Any,
    row: TextRow,
    *,
    seq_len: int,
    train_instruction_tokens: bool,
) -> bool:
    inst_len = len(prepare.byte_ids(prepare.render_instruction(row.instruction)))
    resp_len = len(prepare.byte_ids(prepare.render_response(row.response))) + 1
    if int(seq_len) <= 0 or inst_len + resp_len <= 1:
        return False
    if bool(train_instruction_tokens):
        return min(int(seq_len), max(0, inst_len + resp_len - 1)) > 0
    return int(seq_len) > max(0, int(inst_len) - 1)


def filter_proxy_rows_with_supervised_targets(
    prepare: Any,
    rows: list[TextRow],
    *,
    seq_len: int,
    train_instruction_tokens: bool,
) -> tuple[list[TextRow], list[TextRow]]:
    valid: list[TextRow] = []
    skipped: list[TextRow] = []
    for row in rows:
        if row_has_supervised_targets_after_truncation(
            prepare,
            row,
            seq_len=int(seq_len),
            train_instruction_tokens=bool(train_instruction_tokens),
        ):
            valid.append(row)
        else:
            skipped.append(row)
    return valid, skipped


def is_no_supervised_target_tokens_error(error: Exception) -> bool:
    return "encoded batch has no supervised target tokens" in str(error)


def parser_defaults_from_checkpoint(trainer: Any, checkpoint_args: dict[str, Any]) -> argparse.Namespace:
    args = trainer.build_arg_parser().parse_args(["--sampled-data", "__opus_dummy__", "--out-dir", "__opus_dummy__"])
    for key, value in checkpoint_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def resolve_checkpoint_seq_len(requested_seq_len: int, checkpoint_args: dict[str, Any], *, fallback: int = 384) -> int:
    if int(requested_seq_len) > 0:
        return int(requested_seq_len)
    try:
        return int(checkpoint_args.get("seq_len", fallback))
    except Exception:
        return int(fallback)


def build_model_from_checkpoint(
    trainer: Any,
    *,
    checkpoint: Path,
    device: torch.device,
    seq_len: int,
    resume_strict: bool,
) -> tuple[nn.Module, torch.optim.Optimizer, dict[str, Any], argparse.Namespace]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"checkpoint does not contain model_state_dict: {checkpoint}")
    checkpoint_args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    args = parser_defaults_from_checkpoint(trainer, checkpoint_args)
    args.seq_len = resolve_checkpoint_seq_len(int(seq_len), checkpoint_args, fallback=int(getattr(args, "seq_len", 384)))
    args.device = str(device)
    prefix = trainer.load_prefixlm_module()
    model_summary = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    byte_vocab = int(model_summary.get("vocab_size", 512))
    if str(args.patch_boundary_mode) == "fixed":
        global_seq_len = int(math.ceil(int(args.seq_len) / float(args.patch_size)))
    else:
        global_seq_len = int(math.ceil(int(args.seq_len) / float(max(1, int(args.dynamic_min_patch_size)))))
    global_args = trainer.build_global_args(args, prefix, global_seq_len=global_seq_len)
    global_core = prefix.build_model(global_args, vocab_size=byte_vocab)
    from qtrm_mm.models.blt_prefixlm import BLTDByteLatentPrefixLM

    model = BLTDByteLatentPrefixLM(
        global_core=global_core,
        vocab_size=byte_vocab,
        d_model=int(args.d_model),
        patch_size=int(args.patch_size),
        mask_token_id=int(args.mask_token_id) if int(args.mask_token_id) >= 0 else byte_vocab - 1,
        local_layers=int(args.local_layers),
        local_heads=int(args.local_heads),
        dropout=float(args.dropout),
        clean_boundary_current_latent=not bool(args.no_clean_boundary_current_latent),
        decoder_latent_mode=str(args.decoder_latent_mode),
        patch_boundary_mode=str(args.patch_boundary_mode),
        dynamic_min_patch_size=int(args.dynamic_min_patch_size),
        dynamic_soft_patch_size=int(args.dynamic_soft_patch_size),
        hbf_boundary_threshold=float(args.hbf_boundary_threshold),
        nitp_enabled=float(args.nitp_loss_weight) > 0.0,
        nitp_hidden_dim=int(args.nitp_hidden_dim),
        answer_readback_mode=str(args.answer_readback_mode),
        answer_readback_gate_init=float(args.answer_readback_gate_init),
        answer_readback_temperature=float(args.answer_readback_temperature),
    ).to(device)
    adapted, _ = trainer.adapt_resume_state_dict_for_current_model(payload["model_state_dict"], model.state_dict())
    model.load_state_dict(adapted, strict=bool(resume_strict))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )
    return model, optimizer, payload, args


def selected_named_parameters(model: nn.Module, pattern: str) -> list[tuple[str, nn.Parameter]]:
    compiled = re.compile(str(pattern)) if str(pattern).strip() else None
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if compiled is not None and compiled.search(name) is None:
            continue
        selected.append((name, parameter))
    if not selected:
        raise ValueError(f"no trainable parameters matched --param-name-regex={pattern!r}")
    return selected


def sketch_tensor(values: torch.Tensor, *, projection_dim: int, seed: int) -> torch.Tensor:
    flat = values.detach().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return torch.zeros(int(projection_dim), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    buckets = torch.randint(0, int(projection_dim), (int(flat.numel()),), generator=generator)
    signs = torch.randint(0, 2, (int(flat.numel()),), generator=generator, dtype=torch.int8)
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    sketch = torch.zeros(int(projection_dim), dtype=torch.float32)
    sketch.scatter_add_(0, buckets, flat * signs)
    return sketch


def adamw_effective_grad(
    parameter: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    *,
    parameter_name: str = "",
    beta2: float,
    eps: float,
    weight_decay: float,
    preconditioner: str,
    stats: PreconditionerFallbackStats | None = None,
) -> torch.Tensor | None:
    if parameter.grad is None:
        return None
    grad = parameter.grad.detach().float()
    if float(weight_decay) != 0.0:
        grad = grad + float(weight_decay) * parameter.detach().float()
    if str(preconditioner) == "identity":
        return grad
    if str(preconditioner) != "adamw_state":
        raise ValueError(f"unsupported preconditioner: {preconditioner}")
    state = optimizer.state.get(parameter, {})
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg_sq is None:
        if stats is not None:
            stats.record_missing_exp_avg_sq(str(parameter_name))
        return grad
    if tuple(exp_avg_sq.shape) != tuple(grad.shape):
        if stats is not None:
            stats.record_shape_mismatch(str(parameter_name))
        return grad
    step = float(state.get("step", 0.0) or 0.0)
    bias_correction2 = 1.0 - float(beta2) ** max(1.0, step)
    denom = (
        exp_avg_sq.detach().float().to(grad.device) / max(1e-12, bias_correction2)
    ).sqrt().add(float(eps))
    if stats is not None:
        stats.record_adamw_preconditioned()
    return grad / denom


def sketch_effective_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selected_params: list[tuple[str, nn.Parameter]],
    *,
    projection_dim: int,
    seed: int,
    preconditioner: str,
    beta2: float,
    eps: float,
    weight_decay: float,
    preconditioner_stats: PreconditionerFallbackStats | None = None,
) -> torch.Tensor:
    sketch = torch.zeros(int(projection_dim), dtype=torch.float32)
    for index, (name, parameter) in enumerate(selected_params):
        update = adamw_effective_grad(
            parameter,
            optimizer,
            parameter_name=str(name),
            beta2=float(beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
            preconditioner=str(preconditioner),
            stats=preconditioner_stats,
        )
        if update is None:
            continue
        sketch += sketch_tensor(update, projection_dim=int(projection_dim), seed=int(seed) + index * 1_000_003)
    return sketch


def forward_clean_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    think_steps: int,
) -> tuple[torch.Tensor, int]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss, _metrics = model.forward_losses(
        input_ids,
        labels,
        attention_mask,
        think_steps=int(think_steps),
        diffusion_weight=0.0,
        diffusion_mask_prob=0.0,
        nitp_loss_weight=0.0,
        boundary_prior_weight=0.0,
        qwen_boundary_prior_weight=0.0,
        cot_anchor_loss_weight=0.0,
        workspace_selector_critic_weight=0.0,
        workspace_selector_final_ce_critic_weight=0.0,
    )
    targets = int((labels != IGNORE_LABEL_ID).sum().detach().cpu().item())
    return loss, targets


def compute_projected_update(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selected_params: list[tuple[str, nn.Parameter]],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    think_steps: int,
    projection_dim: int,
    sketch_seed: int,
    preconditioner: str,
    beta2: float,
    eps: float,
    weight_decay: float,
    preconditioner_stats: PreconditionerFallbackStats | None = None,
) -> tuple[torch.Tensor, float, int]:
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    loss, targets = forward_clean_loss(model, batch, device=device, think_steps=int(think_steps))
    loss.backward()
    sketch = sketch_effective_update(
        model,
        optimizer,
        selected_params,
        projection_dim=int(projection_dim),
        seed=int(sketch_seed),
        preconditioner=str(preconditioner),
        beta2=float(beta2),
        eps=float(eps),
        weight_decay=float(weight_decay),
        preconditioner_stats=preconditioner_stats,
    )
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    return sketch, float(loss.detach().cpu().item()), int(targets)


def redundancy_adjusted_order(
    vectors: list[torch.Tensor],
    target: torch.Tensor,
    *,
    lr: float,
    redundancy_weight: float,
) -> list[dict[str, float | int]]:
    remaining = set(range(len(vectors)))
    accumulated = torch.zeros_like(target)
    rows: list[dict[str, float | int]] = []
    while remaining:
        best_index = -1
        best_score = -float("inf")
        best_alignment = 0.0
        best_penalty = 0.0
        for index in remaining:
            vector = vectors[index]
            alignment = float(torch.dot(vector, target).item())
            penalty = float(torch.dot(vector, accumulated).item()) if rows else 0.0
            utility = float(lr) * alignment - (float(lr) ** 2) * float(redundancy_weight) * penalty
            if utility > best_score:
                best_score = utility
                best_index = index
                best_alignment = alignment
                best_penalty = penalty
        if best_index < 0:
            break
        rows.append(
            {
                "index": int(best_index),
                "rank": int(len(rows)),
                "utility": float(best_score),
                "alignment": float(best_alignment),
                "redundancy_penalty": float(best_penalty),
            }
        )
        accumulated = accumulated + vectors[best_index]
        remaining.remove(best_index)
    return rows


def proxy_alignment_score(
    vector: torch.Tensor,
    targets: list[torch.Tensor],
    target_names: list[str],
    *,
    score_mode: str,
    mean_weight: float,
) -> dict[str, Any]:
    if not targets:
        raise ValueError("at least one proxy target is required")
    if len(targets) != len(target_names):
        raise ValueError("proxy target/name length mismatch")
    alignments = {
        str(name): float(torch.dot(vector, target).item())
        for name, target in zip(target_names, targets, strict=True)
    }
    values = list(alignments.values())
    mean_alignment = float(sum(values) / max(1, len(values)))
    min_alignment = float(min(values))
    mode = str(score_mode or "mean").strip().lower()
    if mode in ("mean", "average"):
        alignment = mean_alignment
    elif mode in ("minimax", "min"):
        alignment = min_alignment
    elif mode in ("minimax_mean", "minimax+mean", "min_mean"):
        alignment = min_alignment + float(mean_weight) * mean_alignment
    else:
        raise ValueError(f"unsupported proxy_score_mode={score_mode!r}; expected mean, minimax, or minimax_mean")
    return {
        "alignment": float(alignment),
        "mean_proxy_alignment": float(mean_alignment),
        "min_proxy_alignment": float(min_alignment),
        "proxy_group_alignments": alignments,
    }


def multi_proxy_redundancy_adjusted_order(
    vectors: list[torch.Tensor],
    targets: list[torch.Tensor],
    target_names: list[str],
    *,
    lr: float,
    redundancy_weight: float,
    score_mode: str,
    mean_weight: float,
) -> list[dict[str, Any]]:
    remaining = set(range(len(vectors)))
    accumulated = torch.zeros_like(targets[0])
    rows: list[dict[str, Any]] = []
    while remaining:
        best_index = -1
        best_score = -float("inf")
        best_alignment_report: dict[str, Any] = {}
        best_penalty = 0.0
        for index in remaining:
            vector = vectors[index]
            alignment_report = proxy_alignment_score(
                vector,
                targets,
                target_names,
                score_mode=str(score_mode),
                mean_weight=float(mean_weight),
            )
            penalty = float(torch.dot(vector, accumulated).item()) if rows else 0.0
            utility = float(lr) * float(alignment_report["alignment"]) - (
                float(lr) ** 2
            ) * float(redundancy_weight) * penalty
            if utility > best_score:
                best_score = utility
                best_index = index
                best_alignment_report = alignment_report
                best_penalty = penalty
        if best_index < 0:
            break
        rows.append(
            {
                "index": int(best_index),
                "rank": int(len(rows)),
                "utility": float(best_score),
                "alignment": float(best_alignment_report["alignment"]),
                "mean_proxy_alignment": float(best_alignment_report["mean_proxy_alignment"]),
                "min_proxy_alignment": float(best_alignment_report["min_proxy_alignment"]),
                "proxy_group_alignments": best_alignment_report["proxy_group_alignments"],
                "redundancy_penalty": float(best_penalty),
            }
        )
        accumulated = accumulated + vectors[best_index]
        remaining.remove(best_index)
    return rows


def maybe_load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
    *,
    preconditioner: str,
) -> str:
    if str(preconditioner) == "identity":
        return "identity_diagnostic"
    if str(preconditioner) != "adamw_state":
        raise ValueError(f"unsupported preconditioner: {preconditioner}")
    state = payload.get("optimizer_state_dict")
    if not isinstance(state, dict):
        raise ValueError(
            "OPUS proper requires optimizer_state_dict for --preconditioner adamw_state. "
            "Use a last.pt/copy_last.pt checkpoint or explicitly choose --preconditioner identity for diagnostics."
        )
    optimizer.load_state_dict(state)
    return "adamw_state_checkpoint"


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare = load_prepare_module()
    trainer = load_trainer_module()
    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))

    model, optimizer, payload, model_args = build_model_from_checkpoint(
        trainer,
        checkpoint=Path(args.checkpoint),
        device=device,
        seq_len=int(args.seq_len),
        resume_strict=bool(args.resume_strict),
    )
    effective_seq_len = int(getattr(model_args, "seq_len", args.seq_len))
    optimizer_state_source = maybe_load_optimizer_state(
        optimizer,
        payload,
        preconditioner=str(args.preconditioner),
    )
    selected_params = selected_named_parameters(model, str(args.param_name_regex))
    preconditioner_stats = PreconditionerFallbackStats()
    model.train()

    loaded_proxy_rows = load_proxy_rows(str(args.proxy_jsonl), max_rows=int(args.proxy_max_rows))
    loaded_proxy_bucket_counts = dict(sorted(Counter(row.bucket for row in loaded_proxy_rows).items()))
    loaded_proxy_source_counts = dict(sorted(Counter(row.source_file for row in loaded_proxy_rows).items()))
    valid_proxy_rows, skipped_proxy_rows = filter_proxy_rows_with_supervised_targets(
        prepare,
        loaded_proxy_rows,
        seq_len=int(effective_seq_len),
        train_instruction_tokens=bool(args.train_instruction_tokens),
    )
    if not valid_proxy_rows:
        raise ValueError(
            "all proxy rows lose their supervised answer tokens after seq_len truncation; "
            "increase --seq-len or use shorter proxy rows"
        )
    loaded_proxy_groups = group_proxy_rows(valid_proxy_rows, grouping=str(args.proxy_grouping))
    proxy_groups = cap_proxy_groups(
        loaded_proxy_groups,
        max_rows_per_group=int(args.proxy_max_rows_per_group),
        seed=int(args.seed),
    )
    proxy_rows = [row for group_rows in proxy_groups.values() for row in group_rows]
    proxy_bucket_counts = dict(sorted(Counter(row.bucket for row in proxy_rows).items()))
    proxy_source_counts = dict(sorted(Counter(row.source_file for row in proxy_rows).items()))
    proxy_vectors: list[torch.Tensor] = []
    proxy_group_names: list[str] = []
    proxy_group_reports: list[dict[str, Any]] = []
    weighted_proxy_loss = 0.0
    proxy_targets = 0
    for group_index, (group_name, group_rows) in enumerate(proxy_groups.items()):
        proxy_batch = encode_prefixlm_batch(
            prepare,
            group_rows,
            seq_len=int(effective_seq_len),
            train_instruction_tokens=bool(args.train_instruction_tokens),
        )
        group_vector, group_loss, group_targets = compute_projected_update(
            model=model,
            optimizer=optimizer,
            selected_params=selected_params,
            batch=proxy_batch,
            device=device,
            think_steps=int(args.think_steps or getattr(model_args, "train_think_steps", 1)),
            projection_dim=int(args.projection_dim),
            sketch_seed=int(args.sketch_seed) + group_index * 17_171,
            preconditioner=str(args.preconditioner),
            beta2=float(args.adam_beta2),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
            preconditioner_stats=preconditioner_stats,
        )
        group_norm = float(group_vector.norm().item())
        if not math.isfinite(group_norm) or group_norm <= 0.0:
            raise ValueError(f"proxy projected update has zero/non-finite norm for group {group_name!r}")
        proxy_vectors.append(group_vector / max(1e-12, group_norm))
        proxy_group_names.append(str(group_name))
        proxy_targets += int(group_targets)
        weighted_proxy_loss += float(group_loss) * int(group_targets)
        proxy_group_reports.append(
            {
                "name": str(group_name),
                "rows": int(len(group_rows)),
                "target_tokens": int(group_targets),
                "loss": float(group_loss),
                "norm": float(group_norm),
                "bucket_counts": dict(sorted(Counter(row.bucket for row in group_rows).items())),
                "source_counts": dict(sorted(Counter(row.source_file for row in group_rows).items())),
            }
        )
    proxy_loss = float(weighted_proxy_loss / max(1, int(proxy_targets)))

    candidates = iter_candidate_rows(
        prepare,
        data_root=Path(args.cleaned_data_root),
        source_files=str(args.source_files),
        source_globs=str(args.source_globs),
        max_rows=int(args.candidate_max_rows),
        max_scan_rows_per_file=int(args.candidate_max_scan_rows_per_file),
        max_inst_bytes=int(args.max_inst_bytes),
        max_resp_bytes=int(args.max_resp_bytes),
    )
    vectors: list[torch.Tensor] = []
    losses: list[float] = []
    targets: list[int] = []
    scored_candidates: list[TextRow] = []
    skipped_no_target_tokens: list[TextRow] = []
    progress_every = int(getattr(args, "progress_every", 0) or 0)
    if progress_every > 0:
        print(
            json.dumps(
                {
                    "event": "opus_scoring_start",
                    "candidate_rows_scanned": int(len(candidates)),
                    "selected_parameter_tensors": int(len(selected_params)),
                    "param_name_regex": str(args.param_name_regex),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    for candidate_index, candidate in enumerate(candidates, start=1):
        try:
            batch = encode_prefixlm_batch(
                prepare,
                [candidate],
                seq_len=int(effective_seq_len),
                train_instruction_tokens=bool(args.train_instruction_tokens),
            )
        except ValueError as error:
            if is_no_supervised_target_tokens_error(error):
                skipped_no_target_tokens.append(candidate)
                continue
            raise
        vector, loss, target_count = compute_projected_update(
            model=model,
            optimizer=optimizer,
            selected_params=selected_params,
            batch=batch,
            device=device,
            think_steps=int(args.think_steps or getattr(model_args, "train_think_steps", 1)),
            projection_dim=int(args.projection_dim),
            sketch_seed=int(args.sketch_seed),
            preconditioner=str(args.preconditioner),
            beta2=float(args.adam_beta2),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
            preconditioner_stats=preconditioner_stats,
            )
        vectors.append(vector)
        losses.append(float(loss))
        targets.append(int(target_count))
        scored_candidates.append(candidate)
        if progress_every > 0 and (
            len(scored_candidates) % progress_every == 0 or candidate_index == len(candidates)
        ):
            print(
                json.dumps(
                    {
                        "event": "opus_scoring_progress",
                        "seen": int(candidate_index),
                        "scored": int(len(scored_candidates)),
                        "skipped_no_target_tokens": int(len(skipped_no_target_tokens)),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
    if not vectors:
        raise ValueError("candidate scan produced no OPUS-scoreable rows with supervised target tokens")

    order = multi_proxy_redundancy_adjusted_order(
        vectors,
        proxy_vectors,
        proxy_group_names,
        lr=float(args.lr),
        redundancy_weight=float(args.redundancy_weight),
        score_mode=str(args.proxy_score_mode),
        mean_weight=float(args.proxy_mean_weight),
    )
    by_index = {int(item["index"]): item for item in order}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, candidate in enumerate(scored_candidates):
            score = by_index[index]
            handle.write(
                json.dumps(
                    {
                        "source_file": candidate.source_file,
                        "row_index": int(candidate.row_index),
                        "utility": float(score["utility"]),
                        "alignment": float(score["alignment"]),
                        "mean_proxy_alignment": float(score["mean_proxy_alignment"]),
                        "min_proxy_alignment": float(score["min_proxy_alignment"]),
                        "proxy_group_alignments": score["proxy_group_alignments"],
                        "redundancy_penalty": float(score["redundancy_penalty"]),
                        "selected_rank": int(score["rank"]),
                        "candidate_loss": float(losses[index]),
                        "candidate_target_tokens": int(targets[index]),
                        "bucket": candidate.bucket,
                        "scorer": "opus_projected_utility_v1",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "contract": "opus_projected_utility_v1",
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "out": str(out_path),
        "requested_seq_len": int(args.seq_len),
        "effective_seq_len": int(effective_seq_len),
        "candidate_rows": int(len(scored_candidates)),
        "candidate_rows_scanned": int(len(candidates)),
        "candidate_rows_skipped_no_target_tokens": int(len(skipped_no_target_tokens)),
        "candidate_rows_skipped_no_target_token_examples": [
            {"source_file": row.source_file, "row_index": int(row.row_index), "bucket": row.bucket}
            for row in skipped_no_target_tokens[:24]
        ],
        "proxy_rows_loaded": int(len(loaded_proxy_rows)),
        "proxy_rows_skipped_no_target_tokens": int(len(skipped_proxy_rows)),
        "proxy_rows_loaded_valid": int(len(valid_proxy_rows)),
        "proxy_rows": int(len(proxy_rows)),
        "proxy_groups_loaded": int(len(loaded_proxy_groups)),
        "proxy_groups_used": int(len(proxy_groups)),
        "proxy_max_rows_per_group": int(args.proxy_max_rows_per_group),
        "proxy_loaded_bucket_counts": loaded_proxy_bucket_counts,
        "proxy_loaded_source_counts": loaded_proxy_source_counts,
        "proxy_skipped_no_target_bucket_counts": dict(sorted(Counter(row.bucket for row in skipped_proxy_rows).items())),
        "proxy_skipped_no_target_source_counts": dict(
            sorted(Counter(row.source_file for row in skipped_proxy_rows).items())
        ),
        "proxy_skipped_no_target_examples": [
            {"source_file": row.source_file, "row_index": int(row.row_index), "bucket": row.bucket}
            for row in skipped_proxy_rows[:24]
        ],
        "proxy_bucket_counts": proxy_bucket_counts,
        "proxy_source_counts": proxy_source_counts,
        "proxy_grouping": str(args.proxy_grouping),
        "proxy_score_mode": str(args.proxy_score_mode),
        "proxy_mean_weight": float(args.proxy_mean_weight),
        "proxy_groups": proxy_group_reports,
        "proxy_loss": float(proxy_loss),
        "proxy_target_tokens": int(proxy_targets),
        "projection_dim": int(args.projection_dim),
        "preconditioner": str(args.preconditioner),
        "optimizer_state_source": optimizer_state_source,
        "selected_parameter_tensors": int(len(selected_params)),
        "param_name_regex": str(args.param_name_regex),
        "plain_language": (
            "OPUS scorer made the data audition causal: rows are ranked by whether "
            "their optimizer-shaped update points toward the proxy validation direction. "
            "When the proxy includes Generalization Dynamics rows, candidate data must "
            "push the model toward intelligence answers rather than parrot answers."
        ),
    }
    report.update(preconditioner_stats.to_report())
    if str(args.report_out):
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cleaned-data-root", default="/mnt/data4tb/datasets/hrm-text-data-io-cleaned-20260515")
    parser.add_argument("--source-files", default="data/no_robots.jsonl data/natural_reasoning.jsonl")
    parser.add_argument("--source-globs", default="")
    parser.add_argument("--proxy-jsonl", default="data/eval/prefixlm_language_heldout.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out", default="")
    parser.add_argument("--candidate-max-rows", type=int, default=256)
    parser.add_argument("--candidate-max-scan-rows-per-file", type=int, default=2000)
    parser.add_argument("--proxy-max-rows", type=int, default=8)
    parser.add_argument(
        "--proxy-max-rows-per-group",
        type=int,
        default=0,
        help=(
            "After --proxy-grouping, cap rows per proxy group with a deterministic "
            "sample. This lets official GDsuite use every task as a group without "
            "building one huge gradient batch per task. <=0 uses all loaded rows."
        ),
    )
    parser.add_argument(
        "--proxy-grouping",
        choices=("aggregate", "source_file", "bucket", "source_file_bucket"),
        default="aggregate",
        help=(
            "How to split proxy rows before scoring. aggregate preserves the "
            "original single average proxy; source_file_bucket prevents GD "
            "task rows from being washed out by easier language rows."
        ),
    )
    parser.add_argument(
        "--proxy-score-mode",
        choices=("mean", "minimax", "minimax_mean"),
        default="mean",
        help=(
            "How candidate alignments across proxy groups become one utility. "
            "minimax_mean ranks rows by their weakest proxy-group alignment "
            "plus a small mean-alignment term."
        ),
    )
    parser.add_argument(
        "--proxy-mean-weight",
        type=float,
        default=0.25,
        help="Mean-alignment weight used by --proxy-score-mode minimax_mean.",
    )
    parser.add_argument("--max-inst-bytes", type=int, default=1536)
    parser.add_argument("--max-resp-bytes", type=int, default=1024)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=0,
        help="Override checkpoint seq_len. <=0 uses the checkpoint training seq_len.",
    )
    parser.add_argument("--train-instruction-tokens", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--projection-dim", type=int, default=8192)
    parser.add_argument("--sketch-seed", type=int, default=260205400)
    parser.add_argument(
        "--param-name-regex",
        default="",
        help="Regex over trainable parameter names. Empty means all trainable parameters.",
    )
    parser.add_argument("--preconditioner", choices=("adamw_state", "identity"), default="adamw_state")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--redundancy-weight", type=float, default=1.0)
    parser.add_argument("--matmul-precision", choices=("", "highest", "high", "medium"), default="high")
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run(args)
    if not bool(args.json_only):
        print(report["plain_language"])
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
