#!/usr/bin/env python3
"""Train a Fast-BLT-D-style byte-latent PrefixLM smoke model.

This is the tokenizer-free path requested for the Stage94 ablation.  It keeps
the HRM-Text/Data-IO PrefixLM dataset contract, but changes the model contract:

  UTF-8 bytes -> local byte encoder -> fixed byte blocks / latent patches
  latent patches -> native recurrent global core
  previous latent patch + causal local bytes -> next-byte logits
  previous latent patch + masked local block -> diffusion reconstruction logits

The first falsification target is BLT-D-4: block/patch size 4, clean next-byte
loss plus masked block reconstruction loss.  This is intentionally a compact
implementation, not a full production BLT stack with entropy patching.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract
from qtrm_mm.models.blt_prefixlm import BLTDByteLatentPrefixLM


IGNORE_LABEL_ID = -100


def build_ngram_entropy_tables(
    sampled_data: Path,
    *,
    vocab_size: int,
    max_tokens: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_path = Path(sampled_data) / "tokens.npy"
    if not tokens_path.exists():
        raise FileNotFoundError(f"missing sampled token stream: {tokens_path}")
    tokens = np.load(tokens_path, mmap_mode="r")
    limit = int(len(tokens) if int(max_tokens) <= 0 else min(len(tokens), int(max_tokens)))
    if limit <= 0:
        raise ValueError("cannot build ngram entropy tables from an empty token stream")
    arr = np.asarray(tokens[:limit], dtype=np.int64)
    valid = (arr >= 0) & (arr < int(vocab_size))
    arr = arr[valid]
    if int(arr.size) <= 1:
        raise ValueError("ngram entropy table needs at least two valid tokens")
    alpha = float(max(1e-8, alpha))
    counts = np.bincount(arr, minlength=int(vocab_size)).astype(np.float64)
    unigram_prob = (counts + alpha) / (float(counts.sum()) + alpha * float(vocab_size))
    unigram_surprisal = -np.log(unigram_prob)

    prev = arr[:-1]
    cur = arr[1:]
    pair_ids = prev * int(vocab_size) + cur
    pair_counts = np.bincount(pair_ids, minlength=int(vocab_size) * int(vocab_size)).astype(np.float64)
    pair_counts = pair_counts.reshape(int(vocab_size), int(vocab_size))
    denom = counts.reshape(int(vocab_size), 1) + alpha * float(vocab_size)
    bigram_prob = (pair_counts + alpha) / denom
    bigram_surprisal = -np.log(bigram_prob)
    return (
        torch.from_numpy(unigram_surprisal.astype(np.float32)),
        torch.from_numpy(bigram_surprisal.astype(np.float32)),
    )


def load_prefixlm_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("qtrm_prefixlm_dataio_for_blt_d", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_opus_projected_utility_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "614_score_opus_projected_utility.py"
    spec = importlib.util.spec_from_file_location("opus_projected_utility_for_online_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QwenTokenizerBoundaryTeacher:
    """Convert byte rows into weak Qwen-token-start boundary targets."""

    def __init__(self, model_id: str) -> None:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("transformers is required for Qwen tokenizer boundary priors") from exc
        self.model_id = str(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=True,
        )

    @staticmethod
    def _byte_positions(row_ids: torch.Tensor, row_mask: torch.Tensor) -> tuple[list[int], bytes]:
        positions: list[int] = []
        values = bytearray()
        valid_len = int(row_mask.to(torch.long).sum().item())
        for pos, raw_id in enumerate(row_ids[:valid_len].tolist()):
            token_id = int(raw_id)
            if 2 <= token_id < 258:
                positions.append(int(pos))
                values.append(int(token_id) - 2)
        return positions, bytes(values)

    @staticmethod
    def _char_start_to_byte_start(text: str, char_start: int) -> int:
        prefix = text[: max(0, min(int(char_start), len(text)))]
        return len(prefix.encode("utf-8", errors="replace"))

    def row_targets(self, row_ids: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros_like(row_ids, dtype=torch.float32)
        positions, raw_bytes = self._byte_positions(row_ids, row_mask)
        if not positions:
            return targets
        targets[positions[0]] = 1.0
        text = raw_bytes.decode("utf-8", errors="replace")
        if not text:
            return targets
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
        if offsets is None:
            raise RuntimeError(
                f"tokenizer {self.model_id} did not return offset_mapping; use a fast tokenizer"
            )
        for offset in offsets:
            if not offset:
                continue
            start = int(offset[0])
            byte_start = self._char_start_to_byte_start(text, start)
            if 0 <= byte_start < len(positions):
                targets[positions[byte_start]] = 1.0
        return targets

    def batch_targets(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        rows = [
            self.row_targets(input_ids[row_idx].detach().cpu(), attention_mask[row_idx].detach().cpu())
            for row_idx in range(int(input_ids.shape[0]))
        ]
        return torch.stack(rows, dim=0)


def load_qwen_boundary_teacher(args: argparse.Namespace) -> QwenTokenizerBoundaryTeacher | None:
    if float(getattr(args, "qwen_boundary_prior_weight", 0.0)) <= 0.0:
        return None
    model_id = str(getattr(args, "qwen_boundary_tokenizer_model_id", "")).strip()
    if not model_id:
        raise ValueError("--qwen-boundary-tokenizer-model-id is required when Qwen boundary prior is enabled")
    teacher = QwenTokenizerBoundaryTeacher(model_id)
    print(
        json.dumps(
            {
                "event": "qwen_boundary_teacher_loaded",
                "model_id": model_id,
                "plain_language": (
                    "Qwen tokenizer is used only as a weak chunk-boundary guide, "
                    "not as the byte LM teacher or the final tokenizer."
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return teacher


@dataclass
class MetricWriter:
    writer: Any | None

    def log_scalar(self, name: str, value: float | int, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def make_metric_writer(tensorboard_dir: str) -> MetricWriter:
    if not str(tensorboard_dir):
        return MetricWriter(None)
    if importlib.util.find_spec("torch.utils.tensorboard") is None:
        raise RuntimeError("TensorBoard logging requested but torch.utils.tensorboard is unavailable")
    from torch.utils.tensorboard import SummaryWriter

    path = Path(tensorboard_dir)
    path.mkdir(parents=True, exist_ok=True)
    return MetricWriter(SummaryWriter(log_dir=str(path)))


def configure_triton_ptxas_path() -> dict[str, str | bool]:
    """Require an explicit Triton ptxas path for official-kernel runs."""

    existing = str(os.environ.get("TRITON_PTXAS_PATH", "")).strip()
    if not existing:
        raise RuntimeError(
            "TRITON_PTXAS_PATH is required for BLT official-kernel training; "
            "do not rely on Triton's bundled ptxas or CUDA-path fallbacks."
        )
    chosen_path = Path(existing)
    if not chosen_path.is_file():
        raise RuntimeError(f"TRITON_PTXAS_PATH does not exist: {chosen_path}")
    if not os.access(chosen_path, os.X_OK):
        raise RuntimeError(f"TRITON_PTXAS_PATH is not executable: {chosen_path}")
    bin_dir = str(chosen_path.parent)
    path_parts = str(os.environ.get("PATH", "")).split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + str(os.environ.get("PATH", ""))
    return {
        "triton_ptxas_path": str(os.environ.get("TRITON_PTXAS_PATH", "")),
        "triton_ptxas_source": "explicit",
        "triton_ptxas_exists": True,
    }


def collect_delta_runtime_summary(model: nn.Module) -> dict[str, int | str | bool]:
    """Report the requested delta wrapper separately from the runtime path.

    Official GDN2 can import successfully and still switch to the PyTorch
    fallback at first forward when the local Triton/CUDA toolchain cannot build
    its kernels.  Logging this distinction prevents "official requested" from
    being mistaken for "official actually executed".
    """

    wrapper_count = 0
    official_loaded_count = 0
    fallback_active_count = 0
    direct_torch_count = 0
    for name, module in model.named_modules():
        has_wrapper_state = (
            hasattr(module, "is_official_backend")
            or hasattr(module, "_runtime_fallback_active")
            or hasattr(module, "runtime_fallback")
        )
        if has_wrapper_state:
            wrapper_count += 1
            if bool(getattr(module, "is_official_backend", False)):
                official_loaded_count += 1
            if bool(getattr(module, "_runtime_fallback_active", False)):
                fallback_active_count += 1
        if module.__class__.__name__ == "TorchGatedDeltaMixer":
            if ".runtime_fallback" not in name and ".impl" not in name:
                direct_torch_count += 1

    if direct_torch_count > 0:
        actual = "torch_gated_delta"
    elif wrapper_count > 0 and fallback_active_count > 0:
        actual = "official_wrapper_runtime_fallback"
    elif wrapper_count > 0 and official_loaded_count == wrapper_count:
        actual = "official_runtime"
    elif wrapper_count > 0:
        actual = "mixed_or_unresolved"
    else:
        actual = "none_detected"
    return {
        "actual_delta_runtime": actual,
        "delta_runtime_wrapper_count": int(wrapper_count),
        "delta_runtime_official_loaded_count": int(official_loaded_count),
        "delta_runtime_fallback_active_count": int(fallback_active_count),
        "delta_runtime_torch_direct_count": int(direct_torch_count),
        "delta_runtime_has_fallback": bool(fallback_active_count > 0 or direct_torch_count > 0),
    }


def refresh_model_runtime_summary(model_summary: dict[str, Any], model: nn.Module) -> dict[str, int | str | bool]:
    runtime = collect_delta_runtime_summary(model)
    global_core = model_summary.setdefault("global_core", {})
    if isinstance(global_core, dict):
        global_core["delta_runtime"] = runtime
    return runtime


def build_global_args(args: argparse.Namespace, prefix: Any, *, global_seq_len: int) -> argparse.Namespace:
    global_args = prefix.build_arg_parser().parse_args(
        [
            "--sampled-data",
            str(args.sampled_data),
            "--out-dir",
            str(args.out_dir),
        ]
    )
    global_args.seq_len = int(global_seq_len)
    global_args.d_model = int(args.d_model)
    global_args.n_heads = int(args.n_heads)
    global_args.n_kv_heads = int(args.n_kv_heads)
    global_args.d_ff = int(args.d_ff)
    global_args.dropout = float(args.dropout)
    global_args.backbone = str(args.backbone)
    global_args.encode_backbone = str(args.encode_backbone or args.backbone)
    global_args.think_backbone = str(args.think_backbone or args.backbone)
    global_args.decode_backbone = str(args.decode_backbone or args.backbone)
    global_args.think_structure = str(args.think_structure)
    global_args.train_think_steps = int(args.train_think_steps)
    global_args.hybrid_layers = int(args.hybrid_layers)
    global_args.attn_every = int(args.attn_every)
    global_args.delta_backend = str(args.delta_backend)
    global_args.delta_head_dim = int(args.delta_head_dim)
    global_args.delta_num_v_heads = int(args.delta_num_v_heads)
    global_args.delta_expand_v = float(args.delta_expand_v)
    global_args.delta_mode = str(args.delta_mode)
    global_args.delta_no_short_conv = bool(args.delta_no_short_conv)
    global_args.delta_conv_size = int(args.delta_conv_size)
    global_args.delta_norm_eps = float(args.delta_norm_eps)
    global_args.attention_backend = str(args.attention_backend)
    global_args.strict_backends = bool(args.strict_backends)
    global_args.rope_theta = float(args.rope_theta)
    global_args.position_embedding_mode = str(args.position_embedding_mode)
    global_args.halt_pooling = str(args.halt_pooling)
    global_args.carrier_gate_init = float(args.carrier_gate_init)
    global_args.carrier_state_mode = str(args.carrier_state_mode)
    global_args.trm_recurrent_layerscale_mode = str(args.trm_recurrent_layerscale_mode)
    global_args.trm_recurrent_layerscale_init = float(args.trm_recurrent_layerscale_init)
    global_args.activation_checkpointing = bool(args.activation_checkpointing)
    return global_args


TEACHER_CHECKPOINT_ARG_KEYS = (
    "seq_len",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "dropout",
    "backbone",
    "encode_backbone",
    "think_backbone",
    "decode_backbone",
    "think_structure",
    "train_think_steps",
    "hybrid_layers",
    "attn_every",
    "delta_backend",
    "delta_head_dim",
    "delta_num_v_heads",
    "delta_expand_v",
    "delta_mode",
    "delta_no_short_conv",
    "delta_conv_size",
    "delta_norm_eps",
    "attention_backend",
    "strict_backends",
    "rope_theta",
    "position_embedding_mode",
    "halt_pooling",
    "carrier_gate_init",
    "carrier_state_mode",
    "trm_recurrent_layerscale_mode",
    "trm_recurrent_layerscale_init",
    "activation_checkpointing",
)


def apply_teacher_checkpoint_args(
    teacher_args: argparse.Namespace,
    checkpoint_args: dict[str, Any] | argparse.Namespace | None,
) -> argparse.Namespace:
    if checkpoint_args is None:
        return teacher_args
    if isinstance(checkpoint_args, argparse.Namespace):
        values = vars(checkpoint_args)
    elif isinstance(checkpoint_args, dict):
        values = checkpoint_args
    else:
        return teacher_args
    for key in TEACHER_CHECKPOINT_ARG_KEYS:
        if key in values and values[key] is not None:
            setattr(teacher_args, key, values[key])
    return teacher_args


def build_raw_teacher_args(
    args: argparse.Namespace,
    prefix: Any,
    *,
    checkpoint_args: dict[str, Any] | argparse.Namespace | None = None,
) -> argparse.Namespace:
    teacher_args = build_global_args(
        args,
        prefix,
        global_seq_len=int(args.teacher_seq_len) if int(args.teacher_seq_len) > 0 else int(args.seq_len),
    )
    apply_teacher_checkpoint_args(teacher_args, checkpoint_args)
    return teacher_args


def load_raw_teacher_model(
    args: argparse.Namespace,
    prefix: Any,
    *,
    vocab_size: int,
    device: torch.device,
) -> nn.Module | None:
    if float(args.teacher_distill_weight) <= 0.0:
        return None
    if not str(args.teacher_checkpoint):
        raise ValueError("--teacher-checkpoint is required when --teacher-distill-weight > 0")
    checkpoint = torch.load(str(args.teacher_checkpoint), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise ValueError(f"teacher checkpoint does not contain model_state_dict: {args.teacher_checkpoint}")
    checkpoint_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    teacher_args = build_raw_teacher_args(args, prefix, checkpoint_args=checkpoint_args)
    teacher = prefix.build_model(teacher_args, vocab_size=int(vocab_size)).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or str(device.type) != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    if str(name).lower() in {"", "none", "fp32", "float32"}:
        return None
    if str(name).lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if str(name).lower() in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unknown amp dtype: {name}")


def teacher_distillation_loss(
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
    target_mask = flat_labels != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        zero = student.sum() * 0.0
        return zero, {
            "teacher_distill_loss": 0.0,
            "teacher_distill_targets": 0,
            "teacher_distill_teacher_entropy": 0.0,
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
        "teacher_distill_loss": float(loss.detach().cpu().item()),
        "teacher_distill_targets": int(student.shape[0]),
        "teacher_distill_teacher_entropy": float(teacher_entropy.detach().cpu().item()),
    }


def _select_supervised_depth_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    seq_len: int,
    max_targets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_logits = logits[:, :seq_len].reshape(-1, logits.shape[-1])
    selected_labels = labels[:, :seq_len].reshape(-1)
    target_mask = selected_labels != IGNORE_LABEL_ID
    selected_logits = selected_logits[target_mask]
    selected_labels = selected_labels[target_mask]
    if int(max_targets) > 0 and int(selected_labels.numel()) > int(max_targets):
        indices = torch.linspace(
            0,
            int(selected_labels.numel()) - 1,
            steps=int(max_targets),
            device=selected_labels.device,
        ).long()
        selected_logits = selected_logits[indices]
        selected_labels = selected_labels[indices]
    return selected_logits, selected_labels


def _select_supervised_depth_logits_and_states(
    logits: torch.Tensor,
    states: torch.Tensor,
    labels: torch.Tensor,
    *,
    seq_len: int,
    max_targets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_logits = logits[:, :seq_len].reshape(-1, logits.shape[-1])
    selected_states = states[:, :seq_len].reshape(-1, states.shape[-1])
    selected_labels = labels[:, :seq_len].reshape(-1)
    target_mask = selected_labels != IGNORE_LABEL_ID
    selected_logits = selected_logits[target_mask]
    selected_states = selected_states[target_mask]
    selected_labels = selected_labels[target_mask]
    if int(max_targets) > 0 and int(selected_labels.numel()) > int(max_targets):
        indices = torch.linspace(
            0,
            int(selected_labels.numel()) - 1,
            steps=int(max_targets),
            device=selected_labels.device,
        ).long()
        selected_logits = selected_logits[indices]
        selected_states = selected_states[indices]
        selected_labels = selected_labels[indices]
    return selected_logits, selected_states, selected_labels


def _answer_embedding_weight(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "answer_embedding_weight"):
        weight = getattr(model, "answer_embedding_weight")()
        if isinstance(weight, torch.Tensor):
            return weight
    if hasattr(model, "clean_decoder") and hasattr(getattr(model, "clean_decoder"), "head"):
        return getattr(model, "clean_decoder").head.weight
    if hasattr(model, "hnet_byte_speaker"):
        speaker = getattr(model, "hnet_byte_speaker")
        if isinstance(speaker, nn.Sequential):
            for module in reversed(speaker):
                if isinstance(module, nn.Linear):
                    return module.weight
    raise AttributeError("model must expose answer_embedding_weight() or a linear answer speaker")


def eqr_attractor_regularization_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    shallow_think_steps: int,
    deep_think_steps: int,
    deep_supervision_weight: float,
    consistency_weight: float,
    residual_weight: float,
    improvement_weight: float,
    improvement_margin: float = 0.0,
    temperature: float = 1.0,
    max_targets: int = 512,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """EqR-style attractor loss over shallow, previous, and deep thought depths.

    The recurrent core is useful only if deeper iteration moves toward a better
    and more stable answer state.  This loss makes that contract explicit:
    supervise the deep answer, pull shallow logits toward the deep attractor,
    reduce the last-step residual, and penalize deep answers that are worse than
    shallow answers on the same targets.
    """
    shallow_steps = max(1, int(shallow_think_steps))
    deep_steps = max(shallow_steps, int(deep_think_steps))
    previous_steps = max(shallow_steps, deep_steps - 1)
    shallow_logits = model.forward_logits(
        input_ids,
        attention_mask,
        think_steps=shallow_steps,
    )
    previous_logits = model.forward_logits(
        input_ids,
        attention_mask,
        think_steps=previous_steps,
    )
    deep_logits = model.forward_logits(
        input_ids,
        attention_mask,
        think_steps=deep_steps,
    )
    seq_len = min(
        int(shallow_logits.shape[1]),
        int(previous_logits.shape[1]),
        int(deep_logits.shape[1]),
        int(labels.shape[1]),
    )
    deep_selected, selected_labels = _select_supervised_depth_logits(
        deep_logits,
        labels,
        seq_len=seq_len,
        max_targets=int(max_targets),
    )
    shallow_selected, _ = _select_supervised_depth_logits(
        shallow_logits,
        labels,
        seq_len=seq_len,
        max_targets=int(max_targets),
    )
    previous_selected, _ = _select_supervised_depth_logits(
        previous_logits,
        labels,
        seq_len=seq_len,
        max_targets=int(max_targets),
    )
    if int(selected_labels.numel()) == 0:
        zero = deep_logits.sum() * 0.0
        return zero, {
            "eqr_loss": 0.0,
            "eqr_deep_supervision_loss": 0.0,
            "eqr_consistency_loss": 0.0,
            "eqr_fixed_point_residual": 0.0,
            "eqr_improvement_loss": 0.0,
            "eqr_shallow_ce_loss": 0.0,
            "eqr_deep_ce_loss": 0.0,
            "eqr_targets": 0,
            "eqr_shallow_think_steps": int(shallow_steps),
            "eqr_previous_think_steps": int(previous_steps),
            "eqr_deep_think_steps": int(deep_steps),
        }

    deep_ce = F.cross_entropy(deep_selected.float(), selected_labels)
    with torch.no_grad():
        shallow_ce = F.cross_entropy(shallow_selected.float(), selected_labels)
    temp = float(max(1e-6, temperature))
    shallow_log_probs = F.log_softmax(shallow_selected.float() / temp, dim=-1)
    with torch.no_grad():
        deep_probs = F.softmax(deep_selected.float() / temp, dim=-1)
        previous_log_probs = F.log_softmax(previous_selected.float() / temp, dim=-1)
    consistency_loss = F.kl_div(shallow_log_probs, deep_probs, reduction="batchmean") * (temp * temp)
    deep_log_probs = F.log_softmax(deep_selected.float() / temp, dim=-1)
    fixed_point_residual = F.mse_loss(deep_log_probs, previous_log_probs)
    improvement_loss = F.relu(deep_ce - shallow_ce + float(improvement_margin))
    loss = (
        float(deep_supervision_weight) * deep_ce
        + float(consistency_weight) * consistency_loss
        + float(residual_weight) * fixed_point_residual
        + float(improvement_weight) * improvement_loss
    )
    return loss.to(deep_logits.dtype), {
        "eqr_loss": float(loss.detach().cpu().item()),
        "eqr_deep_supervision_loss": float(deep_ce.detach().cpu().item()),
        "eqr_consistency_loss": float(consistency_loss.detach().cpu().item()),
        "eqr_fixed_point_residual": float(fixed_point_residual.detach().cpu().item()),
        "eqr_improvement_loss": float(improvement_loss.detach().cpu().item()),
        "eqr_shallow_ce_loss": float(shallow_ce.detach().cpu().item()),
        "eqr_deep_ce_loss": float(deep_ce.detach().cpu().item()),
        "eqr_targets": int(selected_labels.numel()),
        "eqr_shallow_think_steps": int(shallow_steps),
        "eqr_previous_think_steps": int(previous_steps),
        "eqr_deep_think_steps": int(deep_steps),
    }


def answer_attractor_regularization_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    depths: list[int] | tuple[int, ...],
    ce_weight: float,
    monotonic_weight: float,
    residual_wrong_weight: float,
    improvement_margin: float = 0.0,
    temperature: float = 1.0,
    max_targets: int = 512,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Train the recurrent loop to settle into a correct answer attractor.

    Residual convergence alone can create a quiet wrong state.  This loss makes
    the answer path causal: each requested depth receives answer CE, deeper
    depths are penalized when their CE regresses, and low-motion but worse
    states receive an optional stable-wrong penalty.
    """
    unique_depths = sorted({max(1, int(depth)) for depth in depths})
    if not unique_depths:
        zero = input_ids.sum(dtype=torch.float32) * 0.0
        return zero, {
            "answer_attractor_loss": 0.0,
            "answer_attractor_ce_loss": 0.0,
            "answer_attractor_monotonic_loss": 0.0,
            "answer_attractor_residual_wrong_loss": 0.0,
            "answer_attractor_targets": 0,
            "answer_attractor_depth_count": 0,
            "answer_attractor_best_depth": 0,
            "answer_attractor_best_ce": 0.0,
        }

    logits_by_depth = [
        model.forward_logits(input_ids, attention_mask, think_steps=int(depth))
        for depth in unique_depths
    ]
    seq_len = min(
        int(labels.shape[1]),
        *(int(logits.shape[1]) for logits in logits_by_depth),
    )
    selected_logits: list[torch.Tensor] = []
    selected_labels: torch.Tensor | None = None
    for logits in logits_by_depth:
        current_logits, current_labels = _select_supervised_depth_logits(
            logits,
            labels,
            seq_len=seq_len,
            max_targets=int(max_targets),
        )
        selected_logits.append(current_logits)
        selected_labels = current_labels
    if selected_labels is None or int(selected_labels.numel()) == 0:
        zero = logits_by_depth[-1].sum() * 0.0
        return zero, {
            "answer_attractor_loss": 0.0,
            "answer_attractor_ce_loss": 0.0,
            "answer_attractor_monotonic_loss": 0.0,
            "answer_attractor_residual_wrong_loss": 0.0,
            "answer_attractor_targets": 0,
            "answer_attractor_depth_count": int(len(unique_depths)),
            "answer_attractor_best_depth": 0,
            "answer_attractor_best_ce": 0.0,
        }

    ce_losses = [
        F.cross_entropy(logits.float(), selected_labels)
        for logits in selected_logits
    ]
    ce_loss = torch.stack(ce_losses).mean()
    if len(ce_losses) > 1:
        monotonic_terms = [
            F.relu(ce_losses[index + 1] - ce_losses[index] + float(improvement_margin))
            for index in range(len(ce_losses) - 1)
        ]
        monotonic_loss = torch.stack(monotonic_terms).mean()
    else:
        monotonic_loss = ce_loss * 0.0

    temp = float(max(1e-6, temperature))
    residual_wrong_terms: list[torch.Tensor] = []
    best_previous_ce = ce_losses[0].detach()
    for index in range(1, len(selected_logits)):
        previous_log_probs = F.log_softmax(selected_logits[index - 1].float() / temp, dim=-1).detach()
        current_log_probs = F.log_softmax(selected_logits[index].float() / temp, dim=-1)
        residual = F.mse_loss(current_log_probs, previous_log_probs)
        wrong_regression = F.relu(ce_losses[index] - best_previous_ce + float(improvement_margin))
        stable_score = torch.exp(-residual.detach())
        residual_wrong_terms.append(stable_score * wrong_regression)
        best_previous_ce = torch.minimum(best_previous_ce, ce_losses[index].detach())
    residual_wrong_loss = (
        torch.stack(residual_wrong_terms).mean()
        if residual_wrong_terms
        else ce_loss * 0.0
    )
    loss = (
        float(ce_weight) * ce_loss
        + float(monotonic_weight) * monotonic_loss
        + float(residual_wrong_weight) * residual_wrong_loss
    )
    best_index = min(range(len(ce_losses)), key=lambda index: float(ce_losses[index].detach().cpu().item()))
    return loss.to(logits_by_depth[-1].dtype), {
        "answer_attractor_loss": float(loss.detach().cpu().item()),
        "answer_attractor_ce_loss": float(ce_loss.detach().cpu().item()),
        "answer_attractor_monotonic_loss": float(monotonic_loss.detach().cpu().item()),
        "answer_attractor_residual_wrong_loss": float(residual_wrong_loss.detach().cpu().item()),
        "answer_attractor_targets": int(selected_labels.numel()),
        "answer_attractor_depth_count": int(len(unique_depths)),
        "answer_attractor_best_depth": int(unique_depths[best_index]),
        "answer_attractor_best_ce": float(ce_losses[best_index].detach().cpu().item()),
    }


def answer_state_attractor_regularization_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    depths: list[int] | tuple[int, ...],
    state_weight: float,
    monotonic_weight: float,
    residual_wrong_weight: float,
    improvement_margin: float = 0.0,
    max_targets: int = 512,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Pull decoder states toward the fixed speaker embedding of the gold answer.

    Stage99B compared answer losses across depths.  This variant gives the
    attractor a semantic center: the hidden state immediately before the speaker
    should point toward the output embedding row of the supervised answer token.
    """
    unique_depths = sorted({max(1, int(depth)) for depth in depths})
    if not unique_depths:
        zero = input_ids.sum(dtype=torch.float32) * 0.0
        return zero, {
            "answer_state_attractor_loss": 0.0,
            "answer_state_attractor_distance": 0.0,
            "answer_state_attractor_monotonic_loss": 0.0,
            "answer_state_attractor_residual_wrong_loss": 0.0,
            "answer_state_attractor_targets": 0,
            "answer_state_attractor_depth_count": 0,
            "answer_state_attractor_best_depth": 0,
            "answer_state_attractor_best_distance": 0.0,
            "answer_state_attractor_best_ce": 0.0,
        }

    logits_by_depth: list[torch.Tensor] = []
    states_by_depth: list[torch.Tensor] = []
    for depth in unique_depths:
        logits, states = model.forward_logits_and_decoder_hidden(
            input_ids,
            attention_mask,
            think_steps=int(depth),
        )
        logits_by_depth.append(logits)
        states_by_depth.append(states)
    seq_len = min(
        int(labels.shape[1]),
        *(int(logits.shape[1]) for logits in logits_by_depth),
        *(int(states.shape[1]) for states in states_by_depth),
    )
    selected_logits_by_depth: list[torch.Tensor] = []
    selected_states_by_depth: list[torch.Tensor] = []
    selected_labels: torch.Tensor | None = None
    for logits, states in zip(logits_by_depth, states_by_depth, strict=True):
        current_logits, current_states, current_labels = _select_supervised_depth_logits_and_states(
            logits,
            states,
            labels,
            seq_len=seq_len,
            max_targets=int(max_targets),
        )
        selected_logits_by_depth.append(current_logits)
        selected_states_by_depth.append(current_states)
        selected_labels = current_labels
    if selected_labels is None or int(selected_labels.numel()) == 0:
        zero = logits_by_depth[-1].sum() * 0.0
        return zero, {
            "answer_state_attractor_loss": 0.0,
            "answer_state_attractor_distance": 0.0,
            "answer_state_attractor_monotonic_loss": 0.0,
            "answer_state_attractor_residual_wrong_loss": 0.0,
            "answer_state_attractor_targets": 0,
            "answer_state_attractor_depth_count": int(len(unique_depths)),
            "answer_state_attractor_best_depth": 0,
            "answer_state_attractor_best_distance": 0.0,
            "answer_state_attractor_best_ce": 0.0,
        }

    speaker_weight = _answer_embedding_weight(model).to(device=selected_labels.device)
    target_states = speaker_weight[selected_labels.clamp(min=0, max=int(speaker_weight.shape[0]) - 1)].detach()
    target_norm = F.normalize(target_states.float(), dim=-1)
    distance_losses: list[torch.Tensor] = []
    ce_losses: list[torch.Tensor] = []
    normalized_states: list[torch.Tensor] = []
    for logits, states in zip(selected_logits_by_depth, selected_states_by_depth, strict=True):
        state_norm = F.normalize(states.float(), dim=-1)
        normalized_states.append(state_norm)
        distance_losses.append((1.0 - (state_norm * target_norm).sum(dim=-1)).mean())
        ce_losses.append(F.cross_entropy(logits.float(), selected_labels))
    state_distance = torch.stack(distance_losses).mean()

    if len(distance_losses) > 1:
        monotonic_terms = [
            F.relu(distance_losses[index + 1] - distance_losses[index] + float(improvement_margin))
            for index in range(len(distance_losses) - 1)
        ]
        monotonic_loss = torch.stack(monotonic_terms).mean()
    else:
        monotonic_loss = state_distance * 0.0

    residual_wrong_terms: list[torch.Tensor] = []
    best_previous_distance = distance_losses[0].detach()
    best_previous_ce = ce_losses[0].detach()
    for index in range(1, len(normalized_states)):
        residual = F.mse_loss(normalized_states[index], normalized_states[index - 1].detach())
        state_regression = F.relu(distance_losses[index] - best_previous_distance + float(improvement_margin))
        answer_regression = F.relu(ce_losses[index] - best_previous_ce + float(improvement_margin))
        stable_score = torch.exp(-residual.detach())
        residual_wrong_terms.append(stable_score * (state_regression + answer_regression))
        best_previous_distance = torch.minimum(best_previous_distance, distance_losses[index].detach())
        best_previous_ce = torch.minimum(best_previous_ce, ce_losses[index].detach())
    residual_wrong_loss = (
        torch.stack(residual_wrong_terms).mean()
        if residual_wrong_terms
        else state_distance * 0.0
    )
    loss = (
        float(state_weight) * state_distance
        + float(monotonic_weight) * monotonic_loss
        + float(residual_wrong_weight) * residual_wrong_loss
    )
    best_index = min(
        range(len(distance_losses)),
        key=lambda index: float(distance_losses[index].detach().cpu().item()),
    )
    return loss.to(logits_by_depth[-1].dtype), {
        "answer_state_attractor_loss": float(loss.detach().cpu().item()),
        "answer_state_attractor_distance": float(state_distance.detach().cpu().item()),
        "answer_state_attractor_monotonic_loss": float(monotonic_loss.detach().cpu().item()),
        "answer_state_attractor_residual_wrong_loss": float(residual_wrong_loss.detach().cpu().item()),
        "answer_state_attractor_targets": int(selected_labels.numel()),
        "answer_state_attractor_depth_count": int(len(unique_depths)),
        "answer_state_attractor_best_depth": int(unique_depths[best_index]),
        "answer_state_attractor_best_distance": float(distance_losses[best_index].detach().cpu().item()),
        "answer_state_attractor_best_ce": float(ce_losses[best_index].detach().cpu().item()),
    }


@torch.no_grad()
def evaluate(
    model: BLTDByteLatentPrefixLM,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    think_steps: int,
    diffusion_weight: float,
    diffusion_mask_prob: float,
    max_batches: int,
    trim_batch_to_max_length: bool,
    prefix: Any,
) -> dict[str, float | int]:
    model.eval()
    total_clean = 0.0
    total_loss = 0.0
    total_targets = 0
    raw_targets = 0
    total_tokens = 0
    total_compute = 0
    batches = 0
    finite_batches = 0
    nonfinite_batches = 0
    nonfinite_targets = 0
    first_nonfinite_batch = 0
    first_nonfinite_loss = 0.0
    first_nonfinite_clean_loss = 0.0
    latent_lengths: list[int] = []
    tracked_metric_values: dict[str, list[float]] = {
        "learned_chunk_gate_mean": [],
        "learned_chunk_gate_entropy": [],
        "learned_chunk_gate_std": [],
        "learned_boundary_prob_mean": [],
        "learned_boundary_prob_std": [],
        "learned_boundary_valid_boundaries": [],
        "ngram_entropy_selected_boundaries": [],
        "ngram_entropy_boundary_score_mean": [],
        "boundary_prob_rate": [],
        "boundary_prior_loss": [],
        "hnet_selected_len": [],
        "hnet_mean_selected_len": [],
        "hnet_dechunked_tokens": [],
        "hnetpp_flow_boundary_score_mean": [],
        "hnetpp_flow_selected_boundaries": [],
        "hier_chunk_gate_mean": [],
        "hier_chunk_gate_std": [],
        "hier_chunk_memory_norm": [],
    }
    for batch in loader:
        if bool(trim_batch_to_max_length):
            batch = prefix.trim_prefixlm_batch_to_max_valid_length(batch)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = int((labels != IGNORE_LABEL_ID).sum().detach().cpu().item())
        if targets <= 0:
            continue
        raw_targets += targets
        with torch.no_grad(), autocast_context(device, amp_dtype):
            loss, metrics = model.forward_losses(
                input_ids,
                labels,
                attention_mask,
                think_steps=int(think_steps),
                diffusion_weight=float(diffusion_weight),
                diffusion_mask_prob=float(diffusion_mask_prob),
                nitp_loss_weight=0.0,
                nitp_max_targets=0,
                boundary_prior_weight=0.0,
                boundary_target_ratio=0.0,
            )
        loss_value = float(loss.detach().cpu().item())
        clean_loss_value = float(metrics["clean_loss"])
        if not math.isfinite(loss_value) or not math.isfinite(clean_loss_value):
            nonfinite_batches += 1
            nonfinite_targets += targets
            if first_nonfinite_batch == 0:
                first_nonfinite_batch = int(batches + 1)
                first_nonfinite_loss = float(loss_value)
                first_nonfinite_clean_loss = float(clean_loss_value)
            batches += 1
            if int(max_batches) > 0 and batches >= int(max_batches):
                break
            continue
        total_loss += loss_value * targets
        total_clean += clean_loss_value * targets
        total_targets += targets
        total_tokens += int(attention_mask.sum().detach().cpu().item())
        total_compute += int(input_ids.numel())
        latent_lengths.append(int(metrics["latent_len"]))
        for name in tracked_metric_values:
            if name in metrics:
                tracked_metric_values[name].append(float(metrics[name]))
        batches += 1
        finite_batches += 1
        if int(max_batches) > 0 and batches >= int(max_batches):
            break
    if total_targets <= 0 and raw_targets <= 0:
        raise ValueError("eval loader produced no target tokens")
    if total_targets <= 0:
        return {
            "loss": float("nan"),
            "clean_loss": float("nan"),
            "target_tokens": 0,
            "raw_target_tokens": int(raw_targets),
            "nonfinite_target_tokens": int(nonfinite_targets),
            "tokens": int(total_tokens),
            "compute_tokens": int(total_compute),
            "batches": int(batches),
            "finite_batches": int(finite_batches),
            "nonfinite_batches": int(nonfinite_batches),
            "first_nonfinite_batch": int(first_nonfinite_batch),
            "first_nonfinite_loss": float(first_nonfinite_loss),
            "first_nonfinite_clean_loss": float(first_nonfinite_clean_loss),
            "mean_latent_len": float(np.mean(latent_lengths)) if latent_lengths else 0.0,
        }
    result = {
        "loss": total_loss / float(total_targets),
        "clean_loss": total_clean / float(total_targets),
        "target_tokens": int(total_targets),
        "raw_target_tokens": int(raw_targets),
        "nonfinite_target_tokens": int(nonfinite_targets),
        "tokens": int(total_tokens),
        "compute_tokens": int(total_compute),
        "batches": int(batches),
        "finite_batches": int(finite_batches),
        "nonfinite_batches": int(nonfinite_batches),
        "first_nonfinite_batch": int(first_nonfinite_batch),
        "first_nonfinite_loss": float(first_nonfinite_loss),
        "first_nonfinite_clean_loss": float(first_nonfinite_clean_loss),
        "mean_latent_len": float(np.mean(latent_lengths)) if latent_lengths else 0.0,
    }
    for name, values in tracked_metric_values.items():
        if values:
            result[name] = float(np.mean(values))
    return result


def save_checkpoint(
    path: Path,
    *,
    model: BLTDByteLatentPrefixLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[dict[str, float | int]],
    eval_losses: list[dict[str, float | int]],
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
    model_summary: dict[str, Any],
    include_optimizer: bool = True,
    copy_safe_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "checkpoint_includes_optimizer": bool(include_optimizer),
        "loss_history": losses,
        "eval_loss_history": eval_losses,
        "args": vars(args),
        "dataset": dataset_summary,
        "model": model_summary,
    }
    if bool(include_optimizer):
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        if copy_safe_path is not None:
            tmp_copy = copy_safe_path.with_name(f".{copy_safe_path.name}.tmp.{os.getpid()}")
            try:
                os.link(path, tmp_copy)
            except OSError:
                shutil.copy2(path, tmp_copy)
            os.replace(tmp_copy, copy_safe_path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def should_write_optimizer_checkpoint_at_step(args: argparse.Namespace, step: int) -> bool:
    if not bool(args.save_optimizer_checkpoint):
        return False
    optimizer_checkpoint_every = int(getattr(args, "optimizer_checkpoint_every", -1))
    if optimizer_checkpoint_every < 0:
        optimizer_checkpoint_every = int(args.checkpoint_every)
    return optimizer_checkpoint_every > 0 and int(step) % optimizer_checkpoint_every == 0


def adapt_resume_state_dict_for_current_model(
    source_state_dict: dict[str, Any],
    target_state_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Adapt only architecture-clean checkpoint keys for the current model.

    Do not translate old official-looking Torch fallback keys into the current
    model.  After the fail-fast GDN2 contract, a fallback checkpoint is legacy
    evidence and must not become the resume base for an official run.
    """

    adapted = source_state_dict.copy()
    legacy_delta_fallback_key_count = 0
    skipped_shape_mismatch = 0
    initialized_missing_answer_readback_gate = 0
    initialized_missing_answer_anchor_head = 0
    initialized_missing_answer_workspace_selector = 0

    legacy_markers = (
        ".mixer.runtime_fallback.",
        ".mixer.impl.in_proj.",
        ".mixer.impl.gate_proj.",
        ".mixer.impl.out_proj.",
    )
    legacy_keys = [key for key in source_state_dict if any(marker in key for marker in legacy_markers)]
    legacy_delta_fallback_key_count = len(legacy_keys)
    if legacy_keys:
        examples = ", ".join(legacy_keys[:3])
        raise ValueError(
            "resume checkpoint contains legacy fallback delta-mixer keys; "
            "do not resume an official_gated_delta2 run from a fallback checkpoint. "
            f"examples: {examples}"
        )

    readback_gate_key = "answer_readback_gate_logit"
    if readback_gate_key in target_state_dict and readback_gate_key not in adapted:
        target_value = target_state_dict[readback_gate_key]
        adapted[readback_gate_key] = target_value.clone() if hasattr(target_value, "clone") else target_value
        initialized_missing_answer_readback_gate += 1
    for key, target_value in target_state_dict.items():
        if key.startswith("answer_anchor_head.") and key not in adapted:
            adapted[key] = target_value.clone() if hasattr(target_value, "clone") else target_value
            initialized_missing_answer_anchor_head += 1
        if key.startswith("answer_workspace_selector.") and key not in adapted:
            adapted[key] = target_value.clone() if hasattr(target_value, "clone") else target_value
            initialized_missing_answer_workspace_selector += 1

    for key, value in list(adapted.items()):
        target_value = target_state_dict.get(key)
        source_shape = getattr(value, "shape", None)
        target_shape = getattr(target_value, "shape", None)
        if target_value is not None and source_shape is not None and target_shape is not None:
            if tuple(source_shape) != tuple(target_shape):
                adapted.pop(key, None)
                skipped_shape_mismatch += 1

    return adapted, {
        "legacy_delta_fallback_key_count": int(legacy_delta_fallback_key_count),
        "skipped_shape_mismatch": int(skipped_shape_mismatch),
        "initialized_missing_answer_readback_gate": int(initialized_missing_answer_readback_gate),
        "initialized_missing_answer_anchor_head": int(initialized_missing_answer_anchor_head),
        "initialized_missing_answer_workspace_selector": int(initialized_missing_answer_workspace_selector),
    }


def load_resume_checkpoint(
    path: Path,
    *,
    model: BLTDByteLatentPrefixLM,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    strict: bool = True,
    load_optimizer: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"resume checkpoint does not contain model_state_dict: {path}")
    adapted_state_dict, key_adaptations = adapt_resume_state_dict_for_current_model(
        payload["model_state_dict"],
        model.state_dict(),
    )
    incompatible = model.load_state_dict(adapted_state_dict, strict=bool(strict))
    optimizer_loaded = False
    if bool(load_optimizer):
        if optimizer is None:
            raise ValueError("--resume-load-optimizer requires an optimizer")
        if "optimizer_state_dict" not in payload:
            raise ValueError(f"resume checkpoint has no optimizer_state_dict: {path}")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_loaded = True
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    return {
        "path": str(path),
        "step": int(payload.get("step", 0) or 0),
        "checkpoint_includes_optimizer": bool(payload.get("checkpoint_includes_optimizer", False)),
        "optimizer_loaded": bool(optimizer_loaded),
        "strict": bool(strict),
        "key_adaptations": key_adaptations,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_architecture_contract(args)
    ptxas_summary = configure_triton_ptxas_path()
    print(json.dumps({"event": "triton_ptxas_config", **ptxas_summary}, ensure_ascii=False), flush=True)
    prefix = load_prefixlm_module()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = prefix.DataIOSampledPrefixLMDataset(
        args.sampled_data,
        seq_len=int(args.seq_len),
        epoch=int(args.epoch),
        target_only=not bool(args.train_instruction_tokens),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        drop_overlength=not bool(args.keep_overlength),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = prefix.build_prefixlm_train_loader(
        dataset,
        batch_size=int(args.batch_size),
        generator=generator,
        length_bucketed_batches=bool(args.length_bucketed_batches),
        length_bucket_size_multiplier=int(args.length_bucket_size_multiplier),
    )
    eval_loader = None
    if int(args.eval_every) > 0:
        eval_dataset = prefix.DataIOSampledPrefixLMDataset(
            args.eval_sampled_data or args.sampled_data,
            seq_len=int(args.seq_len),
            epoch=int(args.eval_epoch),
            target_only=not bool(args.train_instruction_tokens),
            max_rows=int(args.eval_max_rows) if int(args.eval_max_rows) > 0 else None,
            drop_overlength=not bool(args.keep_overlength),
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(args.eval_batch_size or args.batch_size),
            shuffle=False,
            collate_fn=prefix.collate_prefixlm_rows,
            drop_last=False,
        )
    dataset_summary = dataset.summary()
    byte_vocab = int(dataset_summary["model_vocab_size"])
    if str(args.patch_boundary_mode) == "fixed":
        global_seq_len = int(math.ceil(int(args.seq_len) / float(args.patch_size)))
    else:
        global_seq_len = int(math.ceil(int(args.seq_len) / float(max(1, int(args.dynamic_min_patch_size)))))
    global_args = build_global_args(args, prefix, global_seq_len=global_seq_len)
    global_core = prefix.build_model(global_args, vocab_size=byte_vocab)
    device = torch.device(str(args.device))
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
    if str(args.patch_boundary_mode) == "blt_ngram_entropy":
        unigram_surprisal, bigram_surprisal = build_ngram_entropy_tables(
            Path(args.sampled_data),
            vocab_size=int(byte_vocab),
            max_tokens=int(args.ngram_entropy_max_tokens),
            alpha=float(args.ngram_entropy_alpha),
        )
        model.set_ngram_entropy_tables(unigram_surprisal, bigram_surprisal)
    teacher_model = load_raw_teacher_model(args, prefix, vocab_size=byte_vocab, device=device)
    qwen_boundary_teacher = load_qwen_boundary_teacher(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )
    resume_summary: dict[str, Any] | None = None
    if str(args.resume):
        resume_summary = load_resume_checkpoint(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            device=device,
            strict=bool(args.resume_strict),
            load_optimizer=bool(args.resume_load_optimizer),
        )
        print(json.dumps({"event": "resume_loaded", **resume_summary}, ensure_ascii=False), flush=True)
    amp_dtype = resolve_amp_dtype(str(args.amp_dtype))
    writer = make_metric_writer(str(args.tensorboard_dir))
    model_summary = {
        "contract": "fast_blt_d_4_style_prefixlm",
        "vocab_size": int(byte_vocab),
        "d_model": int(args.d_model),
        "patch_size": int(args.patch_size),
        "global_seq_len": int(global_seq_len),
        "local_layers": int(args.local_layers),
        "local_heads": int(args.local_heads),
        "clean_boundary_current_latent": not bool(args.no_clean_boundary_current_latent),
        "decoder_latent_mode": str(args.decoder_latent_mode),
        "patch_boundary_mode": str(args.patch_boundary_mode),
        "dynamic_min_patch_size": int(args.dynamic_min_patch_size),
        "dynamic_soft_patch_size": int(args.dynamic_soft_patch_size),
        "hbf_boundary_threshold": float(args.hbf_boundary_threshold),
        "ngram_entropy_max_tokens": int(args.ngram_entropy_max_tokens),
        "ngram_entropy_alpha": float(args.ngram_entropy_alpha),
        "diffusion_weight": float(args.diffusion_weight),
        "diffusion_mask_prob": float(args.diffusion_mask_prob),
        "nitp_enabled": float(args.nitp_loss_weight) > 0.0,
        "nitp_loss_weight": float(args.nitp_loss_weight),
        "nitp_hidden_dim": int(args.nitp_hidden_dim),
        "nitp_max_targets": int(args.nitp_max_targets),
        "teacher_distill_enabled": teacher_model is not None,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_distill_weight": float(args.teacher_distill_weight),
        "teacher_distill_temperature": float(args.teacher_distill_temperature),
        "teacher_distill_max_targets": int(args.teacher_distill_max_targets),
        "teacher_seq_len": int(args.teacher_seq_len),
        "boundary_prior_weight": float(args.boundary_prior_weight),
        "boundary_target_ratio": float(args.boundary_target_ratio),
        "qwen_boundary_prior_enabled": qwen_boundary_teacher is not None,
        "qwen_boundary_prior_weight": float(args.qwen_boundary_prior_weight),
        "qwen_boundary_tokenizer_model_id": str(args.qwen_boundary_tokenizer_model_id),
        "answer_readback": {
            "mode": str(args.answer_readback_mode),
            "gate_init": float(args.answer_readback_gate_init),
            "temperature": float(args.answer_readback_temperature),
        },
        "cot_anchor": {
            "loss_weight": float(args.cot_anchor_loss_weight),
            "max_targets": int(args.cot_anchor_max_targets),
            "plain_language_role": "short inner-language anchor for the latent-to-speaker callosal bridge",
        },
        "workspace_selector_critic": {
            "loss_weight": float(args.workspace_selector_critic_weight),
            "temperature": float(args.workspace_selector_critic_temperature),
            "plain_language_role": "editor signal that teaches selected workspace readback to broadcast the low-CE anchor candidate",
        },
        "workspace_selector_final_ce_critic": {
            "loss_weight": float(args.workspace_selector_final_ce_critic_weight),
            "temperature": float(args.workspace_selector_final_ce_critic_temperature),
            "max_candidates": int(args.workspace_selector_final_ce_critic_max_candidates),
            "max_targets": int(args.workspace_selector_final_ce_critic_max_targets),
            "plain_language_role": (
                "editor signal that scores candidate broadcasts by the same final speaker "
                "CE used by the normal answer path"
            ),
        },
        "eqr_attractor": {
            "shallow_think_steps": int(args.eqr_shallow_think_steps),
            "deep_think_steps": int(args.eqr_deep_think_steps),
            "deep_supervision_weight": float(args.eqr_deep_supervision_weight),
            "consistency_weight": float(args.eqr_consistency_weight),
            "residual_weight": float(args.eqr_residual_weight),
            "improvement_weight": float(args.eqr_improvement_weight),
            "improvement_margin": float(args.eqr_improvement_margin),
            "temperature": float(args.eqr_temperature),
            "max_targets": int(args.eqr_max_targets),
            "every": int(args.eqr_every),
        },
        "answer_attractor": {
            "depths": [int(depth) for depth in args.answer_attractor_depths],
            "ce_weight": float(args.answer_attractor_ce_weight),
            "monotonic_weight": float(args.answer_attractor_monotonic_weight),
            "residual_wrong_weight": float(args.answer_attractor_residual_wrong_weight),
            "improvement_margin": float(args.answer_attractor_improvement_margin),
            "temperature": float(args.answer_attractor_temperature),
            "max_targets": int(args.answer_attractor_max_targets),
            "every": int(args.answer_attractor_every),
        },
        "answer_state_attractor": {
            "depths": [int(depth) for depth in args.answer_state_attractor_depths],
            "state_weight": float(args.answer_state_attractor_weight),
            "monotonic_weight": float(args.answer_state_attractor_monotonic_weight),
            "residual_wrong_weight": float(args.answer_state_attractor_residual_wrong_weight),
            "improvement_margin": float(args.answer_state_attractor_improvement_margin),
            "max_targets": int(args.answer_state_attractor_max_targets),
            "every": int(args.answer_state_attractor_every),
        },
        "save_optimizer_checkpoint": bool(args.save_optimizer_checkpoint),
        "triton_ptxas": ptxas_summary,
        "resume": resume_summary,
        "global_core": {
            "backbone": str(args.backbone),
            "think_structure": str(args.think_structure),
            "delta_backend": str(args.delta_backend),
            "train_think_steps": int(args.train_think_steps),
        },
        "online_opus": {
            "enabled": bool(args.online_opus_enabled),
            "candidate_batches": int(args.online_opus_candidate_batches),
            "proxy_batches": int(args.online_opus_proxy_batches),
            "proxy_source": str(args.online_opus_proxy_source),
            "every": int(args.online_opus_every),
            "start_step": int(args.online_opus_start_step),
            "projection_dim": int(args.online_opus_projection_dim),
            "preconditioner": str(args.online_opus_preconditioner),
            "param_name_regex": str(args.online_opus_param_name_regex),
            "plain_language_role": (
                "OPUS-style in-loop data choice: draw several candidate batches, "
                "estimate which update best aligns with a proxy batch under the "
                "current optimizer state, then train only the selected batch."
            ),
        },
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    refresh_model_runtime_summary(model_summary, model)
    iterator = iter(loader)
    eval_iterator = iter(eval_loader) if eval_loader is not None else None
    online_opus_enabled = bool(args.online_opus_enabled)
    online_opus_module = None
    online_opus_selected_params: list[tuple[str, nn.Parameter]] = []
    online_opus_preconditioner_stats = None
    if online_opus_enabled:
        if eval_loader is None and str(args.online_opus_proxy_source) == "eval":
            raise ValueError("--online-opus-enabled with --online-opus-proxy-source=eval requires --eval-every > 0")
        if int(args.online_opus_candidate_batches) < 1:
            raise ValueError("--online-opus-candidate-batches must be >= 1")
        if int(args.online_opus_proxy_batches) < 1:
            raise ValueError("--online-opus-proxy-batches must be >= 1")
        online_opus_module = load_opus_projected_utility_module()
        online_opus_selected_params = online_opus_module.selected_named_parameters(
            model,
            str(args.online_opus_param_name_regex),
        )
        online_opus_preconditioner_stats = online_opus_module.PreconditionerFallbackStats()

    def next_train_batch() -> dict[str, torch.Tensor]:
        nonlocal iterator
        try:
            next_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next_batch = next(iterator)
        if bool(args.trim_batch_to_max_length):
            next_batch = prefix.trim_prefixlm_batch_to_max_valid_length(next_batch)
        return next_batch

    def next_eval_proxy_batch() -> dict[str, torch.Tensor]:
        nonlocal eval_iterator
        if eval_loader is None:
            return next_train_batch()
        if eval_iterator is None:
            eval_iterator = iter(eval_loader)
        try:
            next_batch = next(eval_iterator)
        except StopIteration:
            eval_iterator = iter(eval_loader)
            next_batch = next(eval_iterator)
        if bool(args.trim_batch_to_max_length):
            next_batch = prefix.trim_prefixlm_batch_to_max_valid_length(next_batch)
        return next_batch

    def online_opus_select_batch(step: int) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
        if (
            not online_opus_enabled
            or int(args.online_opus_every) <= 0
            or int(step) < int(args.online_opus_start_step)
            or int(step) % int(args.online_opus_every) != 0
        ):
            return next_train_batch(), {
                "online_opus_enabled": float(int(online_opus_enabled)),
                "online_opus_active": 0.0,
                "online_opus_candidate_batches": 1.0,
                "online_opus_selected_index": 0.0,
                "online_opus_selected_alignment": 0.0,
                "online_opus_mean_alignment": 0.0,
                "online_opus_min_alignment": 0.0,
                "online_opus_proxy_loss": 0.0,
                "online_opus_selected_candidate_loss": 0.0,
                "online_opus_proxy_targets": 0.0,
                "online_opus_selected_candidate_targets": 0.0,
            }
        assert online_opus_module is not None
        model.train()
        candidates = [next_train_batch() for _ in range(int(args.online_opus_candidate_batches))]
        proxy_vectors: list[torch.Tensor] = []
        proxy_loss_weighted = 0.0
        proxy_targets_total = 0
        for proxy_index in range(int(args.online_opus_proxy_batches)):
            proxy_batch = (
                next_eval_proxy_batch()
                if str(args.online_opus_proxy_source) == "eval"
                else next_train_batch()
            )
            proxy_vector, proxy_loss, proxy_targets = online_opus_module.compute_projected_update(
                model=model,
                optimizer=optimizer,
                selected_params=online_opus_selected_params,
                batch=proxy_batch,
                device=device,
                think_steps=int(args.train_think_steps),
                projection_dim=int(args.online_opus_projection_dim),
                sketch_seed=int(args.online_opus_sketch_seed) + int(step) * 1009 + proxy_index * 17171,
                preconditioner=str(args.online_opus_preconditioner),
                beta2=float(args.adam_beta2),
                eps=float(args.online_opus_adam_eps),
                weight_decay=float(args.weight_decay),
                preconditioner_stats=online_opus_preconditioner_stats,
            )
            proxy_vectors.append(proxy_vector)
            proxy_loss_weighted += float(proxy_loss) * int(proxy_targets)
            proxy_targets_total += int(proxy_targets)
        proxy_vector = torch.stack(proxy_vectors, dim=0).mean(dim=0)
        proxy_norm = float(proxy_vector.norm().item())
        if not math.isfinite(proxy_norm) or proxy_norm <= 0.0:
            return candidates[0], {
                "online_opus_enabled": 1.0,
                "online_opus_active": 0.0,
                "online_opus_candidate_batches": float(len(candidates)),
                "online_opus_selected_index": 0.0,
                "online_opus_selected_alignment": 0.0,
                "online_opus_mean_alignment": 0.0,
                "online_opus_min_alignment": 0.0,
                "online_opus_proxy_loss": float(proxy_loss_weighted / max(1, proxy_targets_total)),
                "online_opus_selected_candidate_loss": 0.0,
                "online_opus_proxy_targets": float(proxy_targets_total),
                "online_opus_selected_candidate_targets": 0.0,
            }
        proxy_vector = proxy_vector / max(1e-12, proxy_norm)
        scores: list[float] = []
        candidate_losses: list[float] = []
        candidate_targets: list[int] = []
        for candidate_index, candidate_batch in enumerate(candidates):
            candidate_vector, candidate_loss, candidate_target_count = online_opus_module.compute_projected_update(
                model=model,
                optimizer=optimizer,
                selected_params=online_opus_selected_params,
                batch=candidate_batch,
                device=device,
                think_steps=int(args.train_think_steps),
                projection_dim=int(args.online_opus_projection_dim),
                sketch_seed=int(args.online_opus_sketch_seed) + int(step) * 1009 + 1_000_003 + candidate_index * 17171,
                preconditioner=str(args.online_opus_preconditioner),
                beta2=float(args.adam_beta2),
                eps=float(args.online_opus_adam_eps),
                weight_decay=float(args.weight_decay),
                preconditioner_stats=online_opus_preconditioner_stats,
            )
            candidate_norm = float(candidate_vector.norm().item())
            if not math.isfinite(candidate_norm) or candidate_norm <= 0.0:
                scores.append(float("-inf"))
            else:
                scores.append(float(torch.dot(candidate_vector / max(1e-12, candidate_norm), proxy_vector).item()))
            candidate_losses.append(float(candidate_loss))
            candidate_targets.append(int(candidate_target_count))
        best_index = max(range(len(scores)), key=lambda idx: scores[idx])
        finite_scores = [score for score in scores if math.isfinite(float(score))]
        selected_alignment = float(scores[best_index]) if math.isfinite(float(scores[best_index])) else 0.0
        return candidates[best_index], {
            "online_opus_enabled": 1.0,
            "online_opus_active": 1.0,
            "online_opus_candidate_batches": float(len(candidates)),
            "online_opus_selected_index": float(best_index),
            "online_opus_selected_alignment": float(selected_alignment),
            "online_opus_mean_alignment": float(sum(finite_scores) / max(1, len(finite_scores))),
            "online_opus_min_alignment": float(min(finite_scores)) if finite_scores else 0.0,
            "online_opus_proxy_loss": float(proxy_loss_weighted / max(1, proxy_targets_total)),
            "online_opus_selected_candidate_loss": float(candidate_losses[best_index]),
            "online_opus_proxy_targets": float(proxy_targets_total),
            "online_opus_selected_candidate_targets": float(candidate_targets[best_index]),
        }

    losses: list[dict[str, float | int]] = []
    eval_losses: list[dict[str, float | int]] = []
    best_eval_clean_loss = float("inf")
    best_eval_step = 0
    tokens_seen = 0
    target_tokens_seen = 0
    compute_tokens_seen = 0
    previous_time = time.perf_counter()
    start_time = previous_time
    try:
        for step in range(1, int(args.steps) + 1):
            lr = float(args.lr) * min(1.0, step / float(max(1, int(args.lr_warmup_steps))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch, online_opus_metrics = online_opus_select_batch(step)
            qwen_boundary_targets_cpu = None
            if qwen_boundary_teacher is not None:
                qwen_boundary_targets_cpu = qwen_boundary_teacher.batch_targets(
                    batch["input_ids"],
                    batch["attention_mask"],
                )
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            qwen_boundary_targets = (
                qwen_boundary_targets_cpu.to(device)
                if qwen_boundary_targets_cpu is not None
                else None
            )
            model.train()
            with autocast_context(device, amp_dtype):
                loss, metrics = model.forward_losses(
                    input_ids,
                    labels,
                    attention_mask,
                    think_steps=int(args.train_think_steps),
                    diffusion_weight=float(args.diffusion_weight),
                    diffusion_mask_prob=float(args.diffusion_mask_prob),
                    nitp_loss_weight=float(args.nitp_loss_weight),
                    nitp_max_targets=int(args.nitp_max_targets),
                    boundary_prior_weight=float(args.boundary_prior_weight),
                    boundary_target_ratio=float(args.boundary_target_ratio),
                    qwen_boundary_targets=qwen_boundary_targets,
                    qwen_boundary_prior_weight=float(args.qwen_boundary_prior_weight),
                    cot_anchor_loss_weight=float(args.cot_anchor_loss_weight),
                    cot_anchor_max_targets=int(args.cot_anchor_max_targets),
                    workspace_selector_critic_weight=float(args.workspace_selector_critic_weight),
                    workspace_selector_critic_temperature=float(args.workspace_selector_critic_temperature),
                    workspace_selector_final_ce_critic_weight=float(
                        args.workspace_selector_final_ce_critic_weight
                    ),
                    workspace_selector_final_ce_critic_temperature=float(
                        args.workspace_selector_final_ce_critic_temperature
                    ),
                    workspace_selector_final_ce_critic_max_candidates=int(
                        args.workspace_selector_final_ce_critic_max_candidates
                    ),
                    workspace_selector_final_ce_critic_max_targets=int(
                        args.workspace_selector_final_ce_critic_max_targets
                    ),
                )
            if teacher_model is not None:
                with autocast_context(device, amp_dtype):
                    with torch.no_grad():
                        teacher_hidden = teacher_model.forward_hidden(
                            input_ids,
                            think_steps=int(args.train_think_steps),
                        )
                        teacher_logits = teacher_model.lm_head(teacher_hidden)
                    student_logits = model.forward_logits(
                        input_ids,
                        attention_mask,
                        think_steps=int(args.train_think_steps),
                    )
                    distill_loss, distill_metrics = teacher_distillation_loss(
                        student_logits,
                        teacher_logits,
                        labels,
                        temperature=float(args.teacher_distill_temperature),
                        max_targets=int(args.teacher_distill_max_targets),
                    )
                    loss = loss + float(args.teacher_distill_weight) * distill_loss
                metrics.update(distill_metrics)
                metrics["loss"] = float(loss.detach().cpu().item())
            eqr_weight_total = (
                float(args.eqr_deep_supervision_weight)
                + float(args.eqr_consistency_weight)
                + float(args.eqr_residual_weight)
                + float(args.eqr_improvement_weight)
            )
            if (
                eqr_weight_total > 0.0
                and int(args.eqr_every) > 0
                and step % int(args.eqr_every) == 0
            ):
                eqr_deep_steps = int(args.eqr_deep_think_steps)
                if eqr_deep_steps <= 0:
                    eqr_deep_steps = max(int(args.train_think_steps), int(args.eqr_shallow_think_steps) + 1)
                with autocast_context(device, amp_dtype):
                    eqr_loss, eqr_metrics = eqr_attractor_regularization_loss(
                        model,
                        input_ids,
                        labels,
                        attention_mask,
                        shallow_think_steps=int(args.eqr_shallow_think_steps),
                        deep_think_steps=int(eqr_deep_steps),
                        deep_supervision_weight=float(args.eqr_deep_supervision_weight),
                        consistency_weight=float(args.eqr_consistency_weight),
                        residual_weight=float(args.eqr_residual_weight),
                        improvement_weight=float(args.eqr_improvement_weight),
                        improvement_margin=float(args.eqr_improvement_margin),
                        temperature=float(args.eqr_temperature),
                        max_targets=int(args.eqr_max_targets),
                    )
                    loss = loss + eqr_loss
                metrics.update(eqr_metrics)
                metrics["loss"] = float(loss.detach().cpu().item())
            answer_attractor_weight_total = (
                float(args.answer_attractor_ce_weight)
                + float(args.answer_attractor_monotonic_weight)
                + float(args.answer_attractor_residual_wrong_weight)
            )
            if (
                answer_attractor_weight_total > 0.0
                and int(args.answer_attractor_every) > 0
                and step % int(args.answer_attractor_every) == 0
                and len(args.answer_attractor_depths) > 0
            ):
                with autocast_context(device, amp_dtype):
                    answer_attractor_loss, answer_attractor_metrics = answer_attractor_regularization_loss(
                        model,
                        input_ids,
                        labels,
                        attention_mask,
                        depths=[int(depth) for depth in args.answer_attractor_depths],
                        ce_weight=float(args.answer_attractor_ce_weight),
                        monotonic_weight=float(args.answer_attractor_monotonic_weight),
                        residual_wrong_weight=float(args.answer_attractor_residual_wrong_weight),
                        improvement_margin=float(args.answer_attractor_improvement_margin),
                        temperature=float(args.answer_attractor_temperature),
                        max_targets=int(args.answer_attractor_max_targets),
                    )
                    loss = loss + answer_attractor_loss
                metrics.update(answer_attractor_metrics)
                metrics["loss"] = float(loss.detach().cpu().item())
            answer_state_attractor_weight_total = (
                float(args.answer_state_attractor_weight)
                + float(args.answer_state_attractor_monotonic_weight)
                + float(args.answer_state_attractor_residual_wrong_weight)
            )
            if (
                answer_state_attractor_weight_total > 0.0
                and int(args.answer_state_attractor_every) > 0
                and step % int(args.answer_state_attractor_every) == 0
                and len(args.answer_state_attractor_depths) > 0
            ):
                with autocast_context(device, amp_dtype):
                    state_attractor_loss, state_attractor_metrics = answer_state_attractor_regularization_loss(
                        model,
                        input_ids,
                        labels,
                        attention_mask,
                        depths=[int(depth) for depth in args.answer_state_attractor_depths],
                        state_weight=float(args.answer_state_attractor_weight),
                        monotonic_weight=float(args.answer_state_attractor_monotonic_weight),
                        residual_wrong_weight=float(args.answer_state_attractor_residual_wrong_weight),
                        improvement_margin=float(args.answer_state_attractor_improvement_margin),
                        max_targets=int(args.answer_state_attractor_max_targets),
                    )
                    loss = loss + state_attractor_loss
                metrics.update(state_attractor_metrics)
                metrics["loss"] = float(loss.detach().cpu().item())
            metrics.update(online_opus_metrics)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            batch_tokens = int(attention_mask.sum().detach().cpu().item())
            batch_targets = int((labels != IGNORE_LABEL_ID).sum().detach().cpu().item())
            batch_compute = int(input_ids.numel())
            tokens_seen += batch_tokens
            target_tokens_seen += batch_targets
            compute_tokens_seen += batch_compute
            if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
                runtime_summary = refresh_model_runtime_summary(model_summary, model)
                now = time.perf_counter()
                interval = max(1e-9, now - previous_time)
                elapsed = max(1e-9, now - start_time)
                previous_time = now
                row = {
                    "step": int(step),
                    "loss": float(metrics["loss"]),
                    "clean_loss": float(metrics["clean_loss"]),
                    "online_opus_enabled": float(metrics.get("online_opus_enabled", 0.0)),
                    "online_opus_active": float(metrics.get("online_opus_active", 0.0)),
                    "online_opus_candidate_batches": float(
                        metrics.get("online_opus_candidate_batches", 0.0)
                    ),
                    "online_opus_selected_index": float(metrics.get("online_opus_selected_index", 0.0)),
                    "online_opus_selected_alignment": float(
                        metrics.get("online_opus_selected_alignment", 0.0)
                    ),
                    "online_opus_mean_alignment": float(metrics.get("online_opus_mean_alignment", 0.0)),
                    "online_opus_min_alignment": float(metrics.get("online_opus_min_alignment", 0.0)),
                    "online_opus_proxy_loss": float(metrics.get("online_opus_proxy_loss", 0.0)),
                    "online_opus_selected_candidate_loss": float(
                        metrics.get("online_opus_selected_candidate_loss", 0.0)
                    ),
                    "online_opus_proxy_targets": float(metrics.get("online_opus_proxy_targets", 0.0)),
                    "online_opus_selected_candidate_targets": float(
                        metrics.get("online_opus_selected_candidate_targets", 0.0)
                    ),
                    "diffusion_loss": float(metrics["diffusion_loss"]),
                    "diffusion_targets": int(metrics["diffusion_targets"]),
                    "nitp_loss": float(metrics["nitp_loss"]),
                    "nitp_targets": int(metrics["nitp_targets"]),
                    "nitp_cosine_similarity": float(metrics["nitp_cosine_similarity"]),
                    "nitp_predicted_norm": float(metrics["nitp_predicted_norm"]),
                    "nitp_target_norm": float(metrics["nitp_target_norm"]),
                    "teacher_distill_loss": float(metrics.get("teacher_distill_loss", 0.0)),
                    "teacher_distill_targets": int(metrics.get("teacher_distill_targets", 0)),
                    "teacher_distill_teacher_entropy": float(
                        metrics.get("teacher_distill_teacher_entropy", 0.0)
                    ),
                    "boundary_prior_loss": float(metrics.get("boundary_prior_loss", 0.0)),
                    "qwen_boundary_prior_loss": float(metrics.get("qwen_boundary_prior_loss", 0.0)),
                    "qwen_boundary_target_rate": float(metrics.get("qwen_boundary_target_rate", 0.0)),
                    "qwen_boundary_accuracy": float(metrics.get("qwen_boundary_accuracy", 0.0)),
                    "qwen_boundary_targets": int(metrics.get("qwen_boundary_targets", 0)),
                    "eqr_loss": float(metrics.get("eqr_loss", 0.0)),
                    "eqr_deep_supervision_loss": float(metrics.get("eqr_deep_supervision_loss", 0.0)),
                    "eqr_consistency_loss": float(metrics.get("eqr_consistency_loss", 0.0)),
                    "eqr_fixed_point_residual": float(metrics.get("eqr_fixed_point_residual", 0.0)),
                    "eqr_improvement_loss": float(metrics.get("eqr_improvement_loss", 0.0)),
                    "eqr_shallow_ce_loss": float(metrics.get("eqr_shallow_ce_loss", 0.0)),
                    "eqr_deep_ce_loss": float(metrics.get("eqr_deep_ce_loss", 0.0)),
                    "eqr_targets": int(metrics.get("eqr_targets", 0)),
                    "eqr_shallow_think_steps": int(metrics.get("eqr_shallow_think_steps", 0)),
                    "eqr_previous_think_steps": int(metrics.get("eqr_previous_think_steps", 0)),
                    "eqr_deep_think_steps": int(metrics.get("eqr_deep_think_steps", 0)),
                    "answer_attractor_loss": float(metrics.get("answer_attractor_loss", 0.0)),
                    "answer_attractor_ce_loss": float(metrics.get("answer_attractor_ce_loss", 0.0)),
                    "answer_attractor_monotonic_loss": float(
                        metrics.get("answer_attractor_monotonic_loss", 0.0)
                    ),
                    "answer_attractor_residual_wrong_loss": float(
                        metrics.get("answer_attractor_residual_wrong_loss", 0.0)
                    ),
                    "answer_attractor_targets": int(metrics.get("answer_attractor_targets", 0)),
                    "answer_attractor_depth_count": int(metrics.get("answer_attractor_depth_count", 0)),
                    "answer_attractor_best_depth": int(metrics.get("answer_attractor_best_depth", 0)),
                    "answer_attractor_best_ce": float(metrics.get("answer_attractor_best_ce", 0.0)),
                    "answer_state_attractor_loss": float(metrics.get("answer_state_attractor_loss", 0.0)),
                    "answer_state_attractor_distance": float(
                        metrics.get("answer_state_attractor_distance", 0.0)
                    ),
                    "answer_state_attractor_monotonic_loss": float(
                        metrics.get("answer_state_attractor_monotonic_loss", 0.0)
                    ),
                    "answer_state_attractor_residual_wrong_loss": float(
                        metrics.get("answer_state_attractor_residual_wrong_loss", 0.0)
                    ),
                    "answer_state_attractor_targets": int(metrics.get("answer_state_attractor_targets", 0)),
                    "answer_state_attractor_depth_count": int(
                        metrics.get("answer_state_attractor_depth_count", 0)
                    ),
                    "answer_state_attractor_best_depth": int(
                        metrics.get("answer_state_attractor_best_depth", 0)
                    ),
                    "answer_state_attractor_best_distance": float(
                        metrics.get("answer_state_attractor_best_distance", 0.0)
                    ),
                    "answer_state_attractor_best_ce": float(
                        metrics.get("answer_state_attractor_best_ce", 0.0)
                    ),
                    "lr": float(lr),
                    "tokens_seen": int(tokens_seen),
                    "target_tokens_seen": int(target_tokens_seen),
                    "compute_tokens_seen": int(compute_tokens_seen),
                    "latent_len": int(metrics["latent_len"]),
                    "byte_len": int(metrics["byte_len"]),
                    "compression_ratio": float(metrics["compression_ratio"]),
                    "learned_chunk_gate_mean": float(metrics.get("learned_chunk_gate_mean", 0.0)),
                    "learned_chunk_gate_entropy": float(metrics.get("learned_chunk_gate_entropy", 0.0)),
                    "learned_chunk_gate_std": float(metrics.get("learned_chunk_gate_std", 0.0)),
                    "learned_boundary_prob_mean": float(metrics.get("learned_boundary_prob_mean", 0.0)),
                    "learned_boundary_prob_std": float(metrics.get("learned_boundary_prob_std", 0.0)),
                    "learned_boundary_valid_boundaries": float(metrics.get("learned_boundary_valid_boundaries", 0.0)),
                    "ngram_entropy_selected_boundaries": float(metrics.get("ngram_entropy_selected_boundaries", 0.0)),
                    "ngram_entropy_boundary_score_mean": float(metrics.get("ngram_entropy_boundary_score_mean", 0.0)),
                    "boundary_prob_rate": float(metrics.get("boundary_prob_rate", 0.0)),
                    "hnet_selected_len": float(metrics.get("hnet_selected_len", 0.0)),
                    "hnet_mean_selected_len": float(metrics.get("hnet_mean_selected_len", 0.0)),
                    "hnet_dechunked_tokens": float(metrics.get("hnet_dechunked_tokens", 0.0)),
                    "hnetpp_flow_boundary_score_mean": float(
                        metrics.get("hnetpp_flow_boundary_score_mean", 0.0)
                    ),
                    "hnetpp_flow_selected_boundaries": float(metrics.get("hnetpp_flow_selected_boundaries", 0.0)),
                    "hier_chunk_gate_mean": float(metrics.get("hier_chunk_gate_mean", 0.0)),
                    "hier_chunk_gate_std": float(metrics.get("hier_chunk_gate_std", 0.0)),
                    "hier_chunk_memory_norm": float(metrics.get("hier_chunk_memory_norm", 0.0)),
                    "answer_readback_gate_mean": float(metrics.get("answer_readback_gate_mean", 0.0)),
                    "answer_readback_expected_norm": float(metrics.get("answer_readback_expected_norm", 0.0)),
                    "cot_anchor_loss": float(metrics.get("cot_anchor_loss", 0.0)),
                    "cot_anchor_targets": int(metrics.get("cot_anchor_targets", 0)),
                    "cot_anchor_entropy": float(metrics.get("cot_anchor_entropy", 0.0)),
                    "cot_anchor_accuracy": float(metrics.get("cot_anchor_accuracy", 0.0)),
                    "cot_anchor_readback_entropy": float(metrics.get("cot_anchor_readback_entropy", 0.0)),
                    "cot_anchor_readback_confidence": float(
                        metrics.get("cot_anchor_readback_confidence", 0.0)
                    ),
                    "answer_workspace_selector_loss": float(
                        metrics.get("answer_workspace_selector_loss", 0.0)
                    ),
                    "answer_workspace_selector_targets": int(
                        metrics.get("answer_workspace_selector_targets", 0)
                    ),
                    "answer_workspace_selector_target_entropy": float(
                        metrics.get("answer_workspace_selector_target_entropy", 0.0)
                    ),
                    "answer_workspace_selector_target_confidence": float(
                        metrics.get("answer_workspace_selector_target_confidence", 0.0)
                    ),
                    "answer_workspace_selector_selection_entropy": float(
                        metrics.get("answer_workspace_selector_selection_entropy", 0.0)
                    ),
                    "answer_workspace_selector_selection_confidence": float(
                        metrics.get("answer_workspace_selector_selection_confidence", 0.0)
                    ),
                    "answer_workspace_selector_target_argmax_match": float(
                        metrics.get("answer_workspace_selector_target_argmax_match", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_loss": float(
                        metrics.get("answer_workspace_final_ce_selector_loss", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_targets": int(
                        metrics.get("answer_workspace_final_ce_selector_targets", 0)
                    ),
                    "answer_workspace_final_ce_selector_candidate_count": int(
                        metrics.get("answer_workspace_final_ce_selector_candidate_count", 0)
                    ),
                    "answer_workspace_final_ce_selector_target_entropy": float(
                        metrics.get("answer_workspace_final_ce_selector_target_entropy", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_target_confidence": float(
                        metrics.get("answer_workspace_final_ce_selector_target_confidence", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_selection_entropy": float(
                        metrics.get("answer_workspace_final_ce_selector_selection_entropy", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_selection_confidence": float(
                        metrics.get("answer_workspace_final_ce_selector_selection_confidence", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_target_argmax_match": float(
                        metrics.get("answer_workspace_final_ce_selector_target_argmax_match", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_best_ce": float(
                        metrics.get("answer_workspace_final_ce_selector_best_ce", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_mean_ce": float(
                        metrics.get("answer_workspace_final_ce_selector_mean_ce", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_worst_ce": float(
                        metrics.get("answer_workspace_final_ce_selector_worst_ce", 0.0)
                    ),
                    "answer_workspace_final_ce_selector_improvement_over_mean_ce": float(
                        metrics.get("answer_workspace_final_ce_selector_improvement_over_mean_ce", 0.0)
                    ),
                    "answer_workspace_selection_entropy": float(
                        metrics.get("answer_workspace_selection_entropy", 0.0)
                    ),
                    "answer_workspace_selection_confidence": float(
                        metrics.get("answer_workspace_selection_confidence", 0.0)
                    ),
                    "actual_delta_runtime": str(runtime_summary["actual_delta_runtime"]),
                    "delta_runtime_wrapper_count": int(runtime_summary["delta_runtime_wrapper_count"]),
                    "delta_runtime_official_loaded_count": int(
                        runtime_summary["delta_runtime_official_loaded_count"]
                    ),
                    "delta_runtime_fallback_active_count": int(
                        runtime_summary["delta_runtime_fallback_active_count"]
                    ),
                    "delta_runtime_torch_direct_count": int(runtime_summary["delta_runtime_torch_direct_count"]),
                    "delta_runtime_has_fallback": int(bool(runtime_summary["delta_runtime_has_fallback"])),
                    "tokens_per_sec": float(tokens_seen) / elapsed,
                    "target_tokens_per_sec": float(target_tokens_seen) / elapsed,
                    "compute_tokens_per_sec": float(compute_tokens_seen) / elapsed,
                    "interval_sec": float(interval),
                }
                losses.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                for name, value in row.items():
                    if name != "step" and isinstance(value, (int, float)):
                        writer.log_scalar(f"train/{name}", value, int(step))
            if eval_loader is not None and (step == 1 or step % int(args.eval_every) == 0):
                metrics_eval = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    think_steps=int(args.train_think_steps),
                    diffusion_weight=0.0,
                    diffusion_mask_prob=float(args.diffusion_mask_prob),
                    max_batches=int(args.eval_max_batches),
                    trim_batch_to_max_length=bool(args.trim_batch_to_max_length),
                    prefix=prefix,
                )
                eval_row = {"step": int(step), **metrics_eval}
                eval_losses.append(eval_row)
                print(json.dumps(eval_row, ensure_ascii=False), flush=True)
                for name, value in eval_row.items():
                    if name != "step":
                        writer.log_scalar(f"eval/{name}", value, int(step))
                current_eval_clean_loss = float(eval_row.get("clean_loss", float("inf")))
                if (
                    bool(args.save_best_eval_checkpoint)
                    and math.isfinite(current_eval_clean_loss)
                    and current_eval_clean_loss < best_eval_clean_loss
                ):
                    best_eval_clean_loss = current_eval_clean_loss
                    best_eval_step = int(step)
                    refresh_model_runtime_summary(model_summary, model)
                    save_checkpoint(
                        out_dir / "best_eval_model.pt",
                        model=model,
                        optimizer=optimizer,
                        step=int(step),
                        losses=losses,
                        eval_losses=eval_losses,
                        args=args,
                        dataset_summary=dataset_summary,
                        model_summary=model_summary,
                        include_optimizer=False,
                        copy_safe_path=out_dir / "copy_best_eval_model.pt",
                    )
                    print(
                        json.dumps(
                            {
                                "event": "saved_best_eval_model",
                                "step": int(best_eval_step),
                                "clean_loss": float(best_eval_clean_loss),
                                "path": str(out_dir / "best_eval_model.pt"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0 and step < int(args.steps):
                refresh_model_runtime_summary(model_summary, model)
                if should_write_optimizer_checkpoint_at_step(args, step):
                    save_checkpoint(
                        out_dir / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        step=int(step),
                        losses=losses,
                        eval_losses=eval_losses,
                        args=args,
                        dataset_summary=dataset_summary,
                        model_summary=model_summary,
                        include_optimizer=True,
                        copy_safe_path=out_dir / "copy_last.pt",
                    )
                save_checkpoint(
                    out_dir / "last_model.pt",
                    model=model,
                    optimizer=optimizer,
                    step=int(step),
                    losses=losses,
                    eval_losses=eval_losses,
                    args=args,
                    dataset_summary=dataset_summary,
                    model_summary=model_summary,
                    include_optimizer=False,
                    copy_safe_path=out_dir / "copy_last_model.pt",
                )
        refresh_model_runtime_summary(model_summary, model)
        report = {
            "decision": "completed_blt_d_prefixlm_smoke",
            "accepted": False,
            "dataset": dataset_summary,
            "model": model_summary,
            "loss_history": losses,
            "eval_loss_history": eval_losses,
            "initial_eval_loss": eval_losses[0]["clean_loss"] if eval_losses else None,
            "final_eval_loss": eval_losses[-1]["clean_loss"] if eval_losses else None,
            "best_eval_loss": float(best_eval_clean_loss)
            if math.isfinite(best_eval_clean_loss)
            else None,
            "best_eval_step": int(best_eval_step) if int(best_eval_step) > 0 else None,
            "best_eval_checkpoint": str(out_dir / "best_eval_model.pt")
            if int(best_eval_step) > 0
            else None,
            "plain_language_read": (
                "The model reads raw bytes, folds byte blocks into latent patches, "
                "thinks over the shorter latent sequence, then speaks bytes through "
                "a local decoder. This is the scalable tokenizer-free path; it must "
                "beat raw-byte and BPE on normalized loss plus generation."
            ),
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if bool(args.save_optimizer_checkpoint):
            save_checkpoint(
                out_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                step=int(args.steps),
                losses=losses,
                eval_losses=eval_losses,
                args=args,
                dataset_summary=dataset_summary,
                model_summary=model_summary,
                include_optimizer=True,
                copy_safe_path=out_dir / "copy_last.pt",
            )
        save_checkpoint(
            out_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            step=int(args.steps),
            losses=losses,
            eval_losses=eval_losses,
            args=args,
            dataset_summary=dataset_summary,
            model_summary=model_summary,
            include_optimizer=False,
            copy_safe_path=out_dir / "copy_last_model.pt",
        )
        return report
    finally:
        writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--keep-overlength", action="store_true")
    parser.add_argument("--train-instruction-tokens", action="store_true")
    parser.add_argument("--length-bucketed-batches", action="store_true")
    parser.add_argument("--length-bucket-size-multiplier", type=int, default=64)
    parser.add_argument("--trim-batch-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-sampled-data", default="")
    parser.add_argument("--eval-epoch", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-max-rows", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument(
        "--online-opus-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable OPUS-style online candidate batch selection inside the training loop. "
            "This is the closest path to full OPUS in this custom trainer: several "
            "candidate updates are projected against a proxy update under the current "
            "optimizer state, then only the best-aligned batch is trained."
        ),
    )
    parser.add_argument("--online-opus-candidate-batches", type=int, default=2)
    parser.add_argument("--online-opus-proxy-batches", type=int, default=1)
    parser.add_argument("--online-opus-every", type=int, default=1)
    parser.add_argument("--online-opus-start-step", type=int, default=1)
    parser.add_argument(
        "--online-opus-proxy-source",
        choices=("eval", "train"),
        default="eval",
        help="Use held-out eval batches or fresh train batches as the online OPUS proxy direction.",
    )
    parser.add_argument("--online-opus-projection-dim", type=int, default=1024)
    parser.add_argument("--online-opus-sketch-seed", type=int, default=260205400)
    parser.add_argument(
        "--online-opus-preconditioner",
        choices=("adamw_state", "identity"),
        default="adamw_state",
    )
    parser.add_argument("--online-opus-adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--online-opus-param-name-regex",
        default=(
            "^(byte_embed|byte_pos_embed|patch_len_embed|bos_latent|patch_proj|"
            "semantic_boundary_scorer|semantic_chunk_proj|hierarchical_chunk_proj|"
            "hierarchical_chunk_gate|clean_decoder|hnet_byte_speaker)"
        ),
        help="Regex selecting tensors used for the online OPUS update sketch.",
    )
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--matmul-precision", choices=("", "highest", "high", "medium"), default="high")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--tensorboard-dir", default="")
    parser.add_argument(
        "--save-optimizer-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, write last.pt/copy_last.pt with optimizer state. "
            "Disable for 1B smoke/partial runs to avoid multi-GB optimizer "
            "checkpoint stalls; last_model.pt/copy_last_model.pt are still saved."
        ),
    )
    parser.add_argument(
        "--optimizer-checkpoint-every",
        type=int,
        default=-1,
        help=(
            "Write optimizer-bearing last.pt every N steps when "
            "--save-optimizer-checkpoint is enabled. Use 0 to write the "
            "optimizer checkpoint only at finalization. The default -1 matches "
            "--checkpoint-every for backward compatibility."
        ),
    )
    parser.add_argument(
        "--save-best-eval-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save best_eval_model.pt/copy_best_eval_model.pt whenever eval "
            "clean_loss improves. This keeps the best validation speaker even "
            "when a long run later overfits or drifts."
        ),
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Optional model checkpoint to load before training, usually a last_model.pt from a partial byte run.",
    )
    parser.add_argument(
        "--resume-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strictly match checkpoint model keys when --resume is used.",
    )
    parser.add_argument(
        "--resume-load-optimizer",
        action="store_true",
        help="Also restore optimizer_state_dict from --resume when the checkpoint contains it.",
    )

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--mask-token-id", type=int, default=-1)
    parser.add_argument("--diffusion-weight", type=float, default=0.25)
    parser.add_argument("--diffusion-mask-prob", type=float, default=0.35)
    parser.add_argument(
        "--nitp-loss-weight",
        type=float,
        default=0.0,
        help=(
            "NITP-style auxiliary weight over local decoder hidden states. "
            "The projector predicts the supervised byte embedding geometry "
            "from the same hidden state used by the next-byte LM head."
        ),
    )
    parser.add_argument("--nitp-hidden-dim", type=int, default=0)
    parser.add_argument("--nitp-max-targets", type=int, default=256)
    parser.add_argument(
        "--allow-diagnostic-bridge-experiment",
        action="store_true",
        help=(
            "Allow Stage99-style answer readback/anchor/selector bridge losses. "
            "These are diagnostic only and are blocked by default so they cannot "
            "be mistaken for the main HRM-Text-style one-body architecture path."
        ),
    )
    parser.add_argument(
        "--answer-readback-mode",
        choices=("none", "self_embedding", "anchor_embedding", "selected_anchor_embedding"),
        default="none",
        help="Inject an answer-causal readback state before the normal speaker head.",
    )
    parser.add_argument(
        "--answer-readback-gate-init",
        type=float,
        default=-4.0,
        help="Initial scalar gate logit for answer readback; -4 keeps the path nearly closed.",
    )
    parser.add_argument(
        "--answer-readback-temperature",
        type=float,
        default=1.0,
        help="Temperature for preliminary self-embedding answer readback distribution.",
    )
    parser.add_argument(
        "--cot-anchor-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Stage99E callosal bridge weight: train a short inner-language anchor "
            "from the same hidden state that can be read back into the final speaker."
        ),
    )
    parser.add_argument(
        "--cot-anchor-max-targets",
        type=int,
        default=512,
        help="Maximum supervised byte positions sampled for CoT-anchor supervision per batch.",
    )
    parser.add_argument(
        "--workspace-selector-critic-weight",
        type=float,
        default=0.0,
        help=(
            "Stage99G editor loss: train selected workspace readback to broadcast "
            "the position whose inner-speech anchor has lower answer CE."
        ),
    )
    parser.add_argument(
        "--workspace-selector-critic-temperature",
        type=float,
        default=0.25,
        help="Temperature for converting anchor CE into selector target probabilities.",
    )
    parser.add_argument(
        "--workspace-selector-final-ce-critic-weight",
        type=float,
        default=0.0,
        help=(
            "Stage99H editor loss: train selected workspace readback toward the "
            "candidate whose broadcast lowers final speaker CE."
        ),
    )
    parser.add_argument(
        "--workspace-selector-final-ce-critic-temperature",
        type=float,
        default=0.25,
        help="Temperature for converting final-speaker CE into selector target probabilities.",
    )
    parser.add_argument(
        "--workspace-selector-final-ce-critic-max-candidates",
        type=int,
        default=16,
        help="Maximum supervised positions scored as final-speaker readback candidates per row.",
    )
    parser.add_argument(
        "--workspace-selector-final-ce-critic-max-targets",
        type=int,
        default=512,
        help="Maximum supervised target positions used to score each final-CE candidate per row.",
    )
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--no-clean-boundary-current-latent", action="store_true")
    parser.add_argument(
        "--patch-boundary-mode",
        choices=(
            "fixed",
            "utf8_entropy",
            "byteflow_proxy",
            "hbf_byteflow",
            "blt_ngram_entropy",
            "learned_primary",
            "learned_boundary",
            "hnet_dechunk",
            "hnetpp_flow_dechunk",
        ),
        default="fixed",
        help=(
            "'fixed' preserves fixed-size patching. 'utf8_entropy' is a cheap "
            "BLT-style dynamic patching proxy based on byte classes. "
            "'byteflow_proxy' chooses boundaries from adjacent learned byte "
            "embedding changes, as a cheap coding-rate-inspired ByteFlow smoke. "
            "'hbf_byteflow' keeps the BLT local byte decoder, uses UTF-8-safe "
            "H-Net-style hierarchy constraints, and closes patches with a "
            "ByteFlow-style coding/change score. 'blt_ngram_entropy' follows "
            "official BLT entropy-patcher logic with a corpus n-gram surprisal "
            "proxy and a fixed patch budget. 'learned_primary' keeps only "
            "a learned semantic chunk path in the normal forward pass; fixed "
            "BLT-2 is only a baseline/teacher outside this path. "
            "'learned_boundary' uses the learned scorer to change the actual "
            "patch sequence seen by the global core. 'hnet_dechunk' uses an "
            "H-Net-style boundary/chunk/dechunk path with a direct byte LM head. "
            "'hnetpp_flow_dechunk' keeps that answer path but lets ByteFlow-style "
            "embedding change open extra semantic boundaries when compression "
            "would otherwise hide information."
        ),
    )
    parser.add_argument("--dynamic-min-patch-size", type=int, default=2)
    parser.add_argument(
        "--dynamic-soft-patch-size",
        type=int,
        default=0,
        help="If >0, close same-class ASCII runs after this many bytes before the hard patch-size cap.",
    )
    parser.add_argument(
        "--hbf-boundary-threshold",
        type=float,
        default=0.35,
        help="Boundary score threshold for patch-boundary-mode=hbf_byteflow.",
    )
    parser.add_argument(
        "--ngram-entropy-max-tokens",
        type=int,
        default=5000000,
        help="Number of corpus tokens used to build the blt_ngram_entropy surprisal table; <=0 uses all tokens.",
    )
    parser.add_argument(
        "--ngram-entropy-alpha",
        type=float,
        default=0.1,
        help="Additive smoothing for the blt_ngram_entropy unigram/bigram surprisal table.",
    )
    parser.add_argument(
        "--boundary-prior-weight",
        type=float,
        default=0.0,
        help="Optional H-Net/FLEXITOKENS-style boundary probability prior weight.",
    )
    parser.add_argument(
        "--boundary-target-ratio",
        type=float,
        default=0.5,
        help="Target mean boundary probability when --boundary-prior-weight is enabled.",
    )
    parser.add_argument(
        "--qwen-boundary-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Weak BCE prior that nudges hnet_dechunk boundaries toward Qwen tokenizer "
            "token starts. Keep small; this is a chunking hint, not the main teacher."
        ),
    )
    parser.add_argument(
        "--qwen-boundary-tokenizer-model-id",
        default="Qwen/Qwen3.5-0.8B-Base",
        help="Tokenizer used only to produce weak boundary labels when --qwen-boundary-prior-weight > 0.",
    )
    parser.add_argument(
        "--decoder-latent-mode",
        choices=("add", "cross", "add_cross", "hier_add", "hier_add_cross", "one_body"),
        default="add",
        help=(
            "How the local byte decoder receives global latent patches. "
            "'add' preserves the original compact smoke path, 'cross' uses "
            "prefix latent cross-attention, and 'add_cross' keeps the fixed "
            "boundary latent addition while adding BLT-style cross-attention. "
            "'hier_add' keeps fixed BLT micro-patches but adds a learned "
            "upper-level chunk memory over the previous two latent notes; "
            "'hier_add_cross' combines that learned hierarchy with cross-attention. "
            "'one_body' removes the direct byte decoder shortcut so the LM head "
            "reads recurrent thought-conditioned decoder states."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="",
        help="Raw-byte PrefixLM checkpoint used as the byte-distribution teacher for latent distillation.",
    )
    parser.add_argument(
        "--teacher-distill-weight",
        type=float,
        default=0.0,
        help="KL weight from raw-byte teacher next-byte distribution to the BLT student LM logits.",
    )
    parser.add_argument(
        "--teacher-distill-temperature",
        type=float,
        default=1.0,
        help="Temperature for raw-byte teacher distribution distillation.",
    )
    parser.add_argument(
        "--teacher-distill-max-targets",
        type=int,
        default=512,
        help="Maximum supervised byte positions sampled for teacher distillation per batch; <=0 uses all targets.",
    )
    parser.add_argument(
        "--teacher-seq-len",
        type=int,
        default=0,
        help="Raw teacher max sequence length; defaults to --seq-len when <=0.",
    )

    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--backbone", default="trm_qwen35_3to1")
    parser.add_argument("--encode-backbone", default="")
    parser.add_argument("--think-backbone", default="")
    parser.add_argument("--decode-backbone", default="")
    parser.add_argument("--think-structure", default="trm_dual_z")
    parser.add_argument("--train-think-steps", type=int, default=2)
    parser.add_argument(
        "--eqr-shallow-think-steps",
        type=int,
        default=1,
        help="Shallow depth used by EqR-style attractor regularization.",
    )
    parser.add_argument(
        "--eqr-deep-think-steps",
        type=int,
        default=0,
        help="Deep depth used by EqR-style attractor regularization; <=0 uses max(train_think_steps, shallow+1).",
    )
    parser.add_argument(
        "--eqr-deep-supervision-weight",
        type=float,
        default=0.0,
        help="Optional extra next-byte CE on the deep EqR depth so longer thinking directly learns to answer.",
    )
    parser.add_argument(
        "--eqr-consistency-weight",
        type=float,
        default=0.0,
        help="Optional KL that pulls shallow logits toward the detached deep attractor distribution.",
    )
    parser.add_argument(
        "--eqr-residual-weight",
        type=float,
        default=0.0,
        help="Optional fixed-point residual penalty between the last two EqR depths.",
    )
    parser.add_argument(
        "--eqr-improvement-weight",
        type=float,
        default=0.0,
        help="Optional penalty when the deep depth has worse supervised CE than the shallow depth.",
    )
    parser.add_argument(
        "--eqr-improvement-margin",
        type=float,
        default=0.0,
        help="Margin for penalizing deep CE above shallow CE in EqR attractor regularization.",
    )
    parser.add_argument(
        "--eqr-temperature",
        type=float,
        default=1.0,
        help="Temperature used by EqR depth consistency and residual distributions.",
    )
    parser.add_argument(
        "--eqr-max-targets",
        type=int,
        default=512,
        help="Maximum supervised positions used by EqR regularization per batch; <=0 uses all targets.",
    )
    parser.add_argument(
        "--eqr-every",
        type=int,
        default=1,
        help="Run EqR regularization every N steps when any EqR weight is enabled.",
    )
    parser.add_argument(
        "--answer-attractor-depths",
        type=int,
        nargs="*",
        default=[],
        help="Depths supervised by correct-attractor training, e.g. 1 2 4.",
    )
    parser.add_argument(
        "--answer-attractor-ce-weight",
        type=float,
        default=0.0,
        help="Mean answer CE weight across --answer-attractor-depths.",
    )
    parser.add_argument(
        "--answer-attractor-monotonic-weight",
        type=float,
        default=0.0,
        help="Penalty when deeper answer CE is worse than the previous shallower depth.",
    )
    parser.add_argument(
        "--answer-attractor-residual-wrong-weight",
        type=float,
        default=0.0,
        help="Penalty for low-motion deeper states whose answer CE regresses.",
    )
    parser.add_argument(
        "--answer-attractor-improvement-margin",
        type=float,
        default=0.0,
        help="Margin used by answer-attractor monotonic and stable-wrong penalties.",
    )
    parser.add_argument(
        "--answer-attractor-temperature",
        type=float,
        default=1.0,
        help="Temperature used when comparing consecutive answer distributions.",
    )
    parser.add_argument(
        "--answer-attractor-max-targets",
        type=int,
        default=512,
        help="Maximum supervised positions used by answer-attractor regularization per batch.",
    )
    parser.add_argument(
        "--answer-attractor-every",
        type=int,
        default=1,
        help="Run answer-attractor regularization every N steps when enabled.",
    )
    parser.add_argument(
        "--answer-state-attractor-depths",
        type=int,
        nargs="*",
        default=[],
        help="Depths whose decoder hidden states are pulled toward gold speaker embeddings.",
    )
    parser.add_argument(
        "--answer-state-attractor-weight",
        type=float,
        default=0.0,
        help="Cosine-distance weight from decoder hidden state to the gold answer speaker embedding.",
    )
    parser.add_argument(
        "--answer-state-attractor-monotonic-weight",
        type=float,
        default=0.0,
        help="Penalty when a deeper hidden state is farther from the gold speaker embedding.",
    )
    parser.add_argument(
        "--answer-state-attractor-residual-wrong-weight",
        type=float,
        default=0.0,
        help="Penalty for low-motion deeper states that regress in state distance or answer CE.",
    )
    parser.add_argument(
        "--answer-state-attractor-improvement-margin",
        type=float,
        default=0.0,
        help="Margin used by answer-state attractor monotonic and stable-wrong penalties.",
    )
    parser.add_argument(
        "--answer-state-attractor-max-targets",
        type=int,
        default=512,
        help="Maximum supervised positions used by answer-state attractor regularization per batch.",
    )
    parser.add_argument(
        "--answer-state-attractor-every",
        type=int,
        default=1,
        help="Run answer-state attractor regularization every N steps when enabled.",
    )
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--past-success-report-json",
        default="",
        help=(
            "Past-success doubt report produced by scripts/562_build_past_success_doubt_report.py. "
            "Required for long one-body language runs unless explicitly bypassed."
        ),
    )
    parser.add_argument(
        "--past-success-restoration-gate-json",
        default="",
        help=(
            "Restoration gate report produced by scripts/564_check_past_success_restoration_gate.py. "
            "Satisfies the Stage56/58 restoration-gap preflight when all required signals are present."
        ),
    )
    parser.add_argument(
        "--past-success-preflight-min-steps",
        type=int,
        default=1000,
        help="Minimum --steps for enforcing the past-success preflight on one_body runs.",
    )
    parser.add_argument(
        "--allow-missing-past-success-preflight",
        action="store_true",
        help="Diagnostic override: allow a long one_body run without a past-success report.",
    )
    parser.add_argument(
        "--acknowledge-past-success-restoration-gap",
        action="store_true",
        help=(
            "Diagnostic override: proceed even when the report says the Stage56/58 "
            "restoration gate gap remains."
        ),
    )
    parser.add_argument("--hybrid-layers", type=int, default=4)
    parser.add_argument(
        "--attn-every",
        type=int,
        default=4,
        help="Canonical LT2 Full+GDN schedule: 3 GatedDelta/GDN blocks then 1 full-attention block.",
    )
    parser.add_argument(
        "--delta-backend",
        default="official_gated_delta2",
        help="Canonical LT2 Full+GDN backend for BLT/Data-IO runs; fallback status is logged separately.",
    )
    parser.add_argument("--delta-head-dim", type=int, default=0)
    parser.add_argument("--delta-num-v-heads", type=int, default=0)
    parser.add_argument("--delta-expand-v", type=float, default=1.0)
    parser.add_argument("--delta-mode", default="chunk")
    parser.add_argument("--delta-no-short-conv", action="store_true")
    parser.add_argument("--delta-conv-size", type=int, default=4)
    parser.add_argument("--delta-norm-eps", type=float, default=1e-6)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=100000.0)
    parser.add_argument("--position-embedding-mode", choices=("learned", "none", "randomized"), default="learned")
    parser.add_argument("--halt-pooling", choices=("last", "mean", "dedicated"), default="last")
    parser.add_argument("--carrier-gate-init", type=float, default=-1.0)
    parser.add_argument("--carrier-state-mode", default="gru")
    parser.add_argument("--trm-recurrent-layerscale-mode", default="none")
    parser.add_argument("--trm-recurrent-layerscale-init", type=float, default=1.0)
    return parser


def validate_architecture_contract(args: argparse.Namespace) -> None:
    validate_one_body_architecture_contract(args)


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
