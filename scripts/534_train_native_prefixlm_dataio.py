#!/usr/bin/env python3
"""Train the native recurrent LM path on HRM-Text Data-IO sampled tensors.

This is the bridge from synthetic-only probes to an HRM-Text-comparable
learning-efficiency curve. It consumes the sampled tensor layout produced by
official data_io:

  tokens.npy
  metadata.json
  epoch_N/{inst_start,inst_len,resp_start,resp_len}.npy

The training objective follows HRM-Text PrefixLM convention: instruction tokens
are context, while response tokens are the supervised target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from qtrm_mm.training_optimizers import (
    MEMORY_EFFICIENT_OPTIMIZERS,
    build_memory_efficient_optimizer,
)


IGNORE_LABEL_ID = -100
LOSS_KERNELS = ("torch", "auto", "liger_fused_linear_ce")
MAMBA3_BACKED_THINK_STRUCTURES = {
    "trm_dual_z_coupled_mamba_h_only",
    "trm_dual_z_diffusive_reversed_hybrid_3to1",
    "trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
    "trm_dual_z_reversed_hybrid_3to1",
    "trm_dual_z_reversed_hybrid_3to1_prenorm",
    "trm_dual_z_reversed_hybrid_3to1_joint_readout",
    "trm_dual_z_reversed_hybrid_3to1_core_gated_readout",
    "trm_dual_z_reversed_hybrid_3to1_semantic_carry",
    "trm_dual_z_reversed_hybrid_3to1_order_router",
    "trm_dual_z_reversed_hybrid_3to1_recent_order_router",
    "trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
    "trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
    "trm_dual_z_official_schedule_split_mixer_3to1",
    "trm_dual_z_nested_reversed_hybrid_3to1",
    "trm_dual_z_nested_official_schedule_split_mixer_3to1",
}


def assert_mamba3_free_args(args: argparse.Namespace) -> None:
    fields = (
        "backbone",
        "encode_backbone",
        "think_backbone",
        "decode_backbone",
        "think_structure",
    )
    direct = {
        field: str(getattr(args, field, ""))
        for field in fields
        if "mamba" in str(getattr(args, field, "")).lower()
    }
    if direct:
        raise ValueError(f"Mamba3 paths are disabled for this trainer: {direct}")
    if str(args.think_structure) in MAMBA3_BACKED_THINK_STRUCTURES:
        raise ValueError(
            "Mamba3 paths are disabled for this trainer: "
            f"think_structure={args.think_structure} instantiates Mamba3 internally"
        )


@dataclass(frozen=True)
class PrefixLMMetadata:
    tokenizer_info: dict[str, Any]
    vocab_size: int
    max_seq_len: int
    effective_max_seq_len: int
    total_length: int


def round_up_multiple(value: int, multiple: int) -> int:
    return int(((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple))


def load_prefixlm_metadata(sampled_data: Path) -> PrefixLMMetadata:
    metadata_path = Path(sampled_data) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    tokenizer_info = dict(raw.get("tokenizer_info") or {})
    raw_vocab = raw.get("vocab_size")
    if raw_vocab is None:
        raw_vocab = tokenizer_info.get("vocab_size")
    if raw_vocab is None:
        raise ValueError(f"metadata does not expose vocab_size: {metadata_path}")
    max_seq_len = int(raw["max_seq_len"])
    return PrefixLMMetadata(
        tokenizer_info=tokenizer_info,
        vocab_size=int(raw_vocab),
        max_seq_len=max_seq_len,
        effective_max_seq_len=max(1, max_seq_len - 1),
        total_length=int(raw.get("total_length", 0)),
    )


class DataIOSampledPrefixLMDataset(Dataset[dict[str, torch.Tensor]]):
    """Map official HRM-Text sampled tensors into padded PrefixLM rows."""

    def __init__(
        self,
        sampled_data: str | Path,
        *,
        seq_len: int,
        epoch: int = 0,
        target_only: bool = True,
        max_rows: int | None = None,
        drop_overlength: bool = True,
    ) -> None:
        self.sampled_data = Path(sampled_data)
        self.seq_len = int(seq_len)
        self.epoch = int(epoch)
        self.target_only = bool(target_only)
        self.drop_overlength = bool(drop_overlength)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.metadata = load_prefixlm_metadata(self.sampled_data)
        self.tokens = np.load(self.sampled_data / "tokens.npy", mmap_mode="r")
        epoch_dir = self.sampled_data / f"epoch_{self.epoch}"
        self.inst_start = np.load(epoch_dir / "inst_start.npy", mmap_mode="r")
        self.inst_len = np.load(epoch_dir / "inst_len.npy", mmap_mode="r")
        self.resp_start = np.load(epoch_dir / "resp_start.npy", mmap_mode="r")
        self.resp_len = np.load(epoch_dir / "resp_len.npy", mmap_mode="r")
        if not (
            self.inst_start.shape
            == self.inst_len.shape
            == self.resp_start.shape
            == self.resp_len.shape
        ):
            raise ValueError(f"epoch index arrays have different shapes in {epoch_dir}")
        valid_rows: list[int] = []
        for row_idx in range(int(self.inst_len.shape[0])):
            inst_len = int(self.inst_len[row_idx])
            resp_len = int(self.resp_len[row_idx])
            if inst_len < 1 or resp_len < 1:
                continue
            shifted_len = inst_len + resp_len - 1
            if self.drop_overlength and shifted_len > self.seq_len:
                continue
            valid_rows.append(row_idx)
            if max_rows is not None and len(valid_rows) >= int(max_rows):
                break
        if not valid_rows:
            raise ValueError(
                f"no usable rows in {self.sampled_data} for seq_len={self.seq_len}"
            )
        self.row_indices = np.array(valid_rows, dtype=np.int64)
        self.shifted_lengths = (
            self.inst_len[self.row_indices] + self.resp_len[self.row_indices] - 1
        ).astype(np.int64)

    def __len__(self) -> int:
        return int(self.row_indices.shape[0])

    def _slice_tokens(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.tokens[int(start) : int(start) + int(length)], dtype=np.int64)

    def row_length(self, index: int) -> int:
        return int(self.shifted_lengths[int(index)])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.row_indices[int(index)])
        inst = self._slice_tokens(self.inst_start[row], self.inst_len[row])
        resp = self._slice_tokens(self.resp_start[row], self.resp_len[row])
        input_seq = np.concatenate([inst, resp[:-1]], dtype=np.int64)
        if self.target_only:
            inst_labels = np.full(max(0, len(inst) - 1), IGNORE_LABEL_ID, dtype=np.int64)
        else:
            inst_labels = inst[1:].astype(np.int64)
        labels = np.concatenate([inst_labels, resp.astype(np.int64)], dtype=np.int64)
        if input_seq.shape[0] != labels.shape[0]:
            raise ValueError(
                f"PrefixLM row length mismatch: inputs={input_seq.shape[0]} "
                f"labels={labels.shape[0]}"
            )
        response_start_mask = np.zeros(labels.shape[0], dtype=np.int64)
        response_start_index = max(0, int(inst.shape[0]) - 1)
        if response_start_index < labels.shape[0] and labels[response_start_index] != IGNORE_LABEL_ID:
            response_start_mask[response_start_index] = 1
        if input_seq.shape[0] > self.seq_len:
            input_seq = input_seq[: self.seq_len]
            labels = labels[: self.seq_len]
            response_start_mask = response_start_mask[: self.seq_len]
        attention_mask = np.ones(input_seq.shape[0], dtype=np.int64)
        pad_len = self.seq_len - int(input_seq.shape[0])
        if pad_len > 0:
            input_seq = np.pad(input_seq, (0, pad_len), constant_values=0)
            labels = np.pad(labels, (0, pad_len), constant_values=IGNORE_LABEL_ID)
            response_start_mask = np.pad(response_start_mask, (0, pad_len), constant_values=0)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)
        return {
            "input_ids": torch.from_numpy(input_seq).long(),
            "labels": torch.from_numpy(labels).long(),
            "response_start_mask": torch.from_numpy(response_start_mask).long(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
        }

    def summary(self) -> dict[str, Any]:
        lengths = self.shifted_lengths
        fingerprint = hashlib.sha256()
        fingerprint.update(np.asarray(self.row_indices, dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(self.inst_start[self.row_indices], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(self.inst_len[self.row_indices], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(self.resp_start[self.row_indices], dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(self.resp_len[self.row_indices], dtype=np.int64).tobytes())
        fingerprint.update(str(self.seq_len).encode("utf-8"))
        fingerprint.update(str(self.target_only).encode("utf-8"))
        return {
            "contract": "hrm_text_data_io_prefixlm",
            "eval_protocol": "row_fixed_v1",
            "eval_fingerprint": fingerprint.hexdigest(),
            "sampled_data": str(self.sampled_data),
            "epoch": self.epoch,
            "rows": int(len(self)),
            "seq_len": self.seq_len,
            "target_only": self.target_only,
            "drop_overlength": self.drop_overlength,
            "vocab_size": int(self.metadata.vocab_size),
            "model_vocab_size": round_up_multiple(int(self.metadata.vocab_size), 256),
            "max_seq_len": int(self.metadata.max_seq_len),
            "effective_max_seq_len": int(self.metadata.effective_max_seq_len),
            "total_length": int(self.metadata.total_length),
            "mean_shifted_row_len": float(np.mean(lengths)),
            "max_shifted_row_len": int(np.max(lengths)),
            "mean_fixed_shape_padding_tokens": float(self.seq_len - np.mean(lengths)),
        }


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Yield shuffled batches whose examples have similar sequence lengths."""

    def __init__(
        self,
        lengths: Iterable[int],
        *,
        batch_size: int,
        generator: torch.Generator,
        bucket_size_multiplier: int = 64,
        drop_last: bool = False,
    ) -> None:
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.generator = generator
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self):
        if not self.lengths:
            return
        indices = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        bucket_size = max(self.batch_size, self.batch_size * self.bucket_size_multiplier)
        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[int(idx)])
            batches = [
                bucket[pos : pos + self.batch_size]
                for pos in range(0, len(bucket), self.batch_size)
            ]
            batches = [
                batch for batch in batches if len(batch) == self.batch_size or not self.drop_last
            ]
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            for batch_index in order:
                yield batches[int(batch_index)]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate_prefixlm_rows(rows: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    materialized = list(rows)
    return {
        "input_ids": torch.stack([row["input_ids"] for row in materialized], dim=0),
        "labels": torch.stack([row["labels"] for row in materialized], dim=0),
        "response_start_mask": torch.stack(
            [row["response_start_mask"] for row in materialized], dim=0
        ),
        "attention_mask": torch.stack([row["attention_mask"] for row in materialized], dim=0),
    }


def trim_prefixlm_batch_to_max_valid_length(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove trailing all-padding columns before moving a batch to the model."""

    attention_mask = batch["attention_mask"]
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (batch, seq)")
    seq_len = int(attention_mask.shape[1])
    max_valid_len = int(attention_mask.to(torch.long).sum(dim=1).max().item())
    max_valid_len = max(1, min(seq_len, max_valid_len))
    if max_valid_len >= seq_len:
        return batch
    trimmed: dict[str, torch.Tensor] = {}
    for name, tensor in batch.items():
        if tensor.ndim >= 2 and int(tensor.shape[1]) == seq_len:
            trimmed[name] = tensor[:, :max_valid_len].contiguous()
        else:
            trimmed[name] = tensor
    return trimmed


def build_prefixlm_train_loader(
    dataset: DataIOSampledPrefixLMDataset,
    *,
    batch_size: int,
    generator: torch.Generator,
    length_bucketed_batches: bool,
    length_bucket_size_multiplier: int,
) -> DataLoader:
    if bool(length_bucketed_batches):
        batch_sampler = LengthBucketedBatchSampler(
            dataset.shifted_lengths,
            batch_size=int(batch_size),
            generator=generator,
            bucket_size_multiplier=int(length_bucket_size_multiplier),
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_prefixlm_rows,
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_prefixlm_rows,
        generator=generator,
        drop_last=False,
    )


def load_native_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "335_train_qtrm_native_etd_probe.py"
    spec = importlib.util.spec_from_file_location("qtrm_native_etd_probe_for_prefixlm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model(args: argparse.Namespace, *, vocab_size: int) -> nn.Module:
    import inspect

    native = load_native_module()
    kwargs = {
        "vocab": int(vocab_size),
        "max_seq_len": int(args.seq_len),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "n_kv_heads": int(args.n_kv_heads),
        "d_ff": int(args.d_ff),
        "dropout": float(args.dropout),
        "backbone": str(args.backbone),
        "encode_backbone": str(args.encode_backbone or args.backbone),
        "think_backbone": str(args.think_backbone or args.backbone),
        "decode_backbone": str(args.decode_backbone or args.backbone),
        "think_structure": str(args.think_structure),
        "trm_l_cycles": int(args.trm_l_cycles),
        "trm_no_grad_inner_cycles": not bool(args.trm_full_grad_cycles),
        "hybrid_layers": int(args.hybrid_layers),
        "attn_every": int(args.attn_every),
        "delta_backend": str(args.delta_backend),
        "delta_head_dim": int(args.delta_head_dim)
        if int(args.delta_head_dim) > 0
        else None,
        "delta_num_v_heads": int(args.delta_num_v_heads)
        if int(args.delta_num_v_heads) > 0
        else None,
        "delta_expand_v": float(args.delta_expand_v),
        "delta_mode": str(args.delta_mode),
        "delta_use_short_conv": not bool(args.delta_no_short_conv),
        "delta_conv_size": int(args.delta_conv_size),
        "delta_norm_eps": float(args.delta_norm_eps),
        "attention_backend": str(args.attention_backend),
        "strict_backends": bool(args.strict_backends),
        "rope_theta": float(args.rope_theta),
        "position_embedding_mode": str(args.position_embedding_mode),
        "halt_pooling": str(args.halt_pooling),
        "carrier_gate_init": float(args.carrier_gate_init),
        "carrier_state_mode": str(args.carrier_state_mode),
        "trm_recurrent_layerscale_mode": str(args.trm_recurrent_layerscale_mode),
        "trm_recurrent_layerscale_init": float(args.trm_recurrent_layerscale_init),
        "activation_checkpointing": bool(args.activation_checkpointing),
    }
    signature = inspect.signature(native.NativeQTRMETDLM)
    if "activation_checkpointing" not in signature.parameters:
        kwargs.pop("activation_checkpointing", None)
    return native.NativeQTRMETDLM(**kwargs)


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    normalized = str(name).lower()
    if normalized in {"", "none", "fp32", "float32"}:
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unknown amp dtype: {name}")


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or str(device.type) != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def prefixlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_LABEL_ID,
    )


def _import_liger_fused_linear_cross_entropy():
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "liger-kernel is required for --loss-kernel liger_fused_linear_ce"
        ) from exc
    return LigerFusedLinearCrossEntropyLoss


def resolve_loss_kernel(requested: str, *, tensor: torch.Tensor) -> str:
    name = str(requested or "torch").lower()
    if name not in LOSS_KERNELS:
        raise ValueError(f"unknown loss kernel: {requested}")
    if name != "auto":
        return name
    if not tensor.is_cuda:
        return "torch"
    try:
        _import_liger_fused_linear_cross_entropy()
        return "liger_fused_linear_ce"
    except RuntimeError:
        return "torch"


def prefixlm_loss_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_chunk_size: int = 0,
    loss_kernel: str = "torch",
) -> torch.Tensor:
    resolved_loss_kernel = resolve_loss_kernel(str(loss_kernel), tensor=hidden)
    if resolved_loss_kernel == "liger_fused_linear_ce":
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_labels = labels.reshape(-1)
        target_mask = flat_labels != IGNORE_LABEL_ID
        if not bool(target_mask.any()):
            raise ValueError("batch contains no target tokens")
        target_hidden = flat_hidden[target_mask]
        target_labels = flat_labels[target_mask]
        LigerFusedLinearCrossEntropyLoss = _import_liger_fused_linear_cross_entropy()
        loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
        return loss_fn(
            model.lm_head.weight,
            target_hidden,
            target_labels,
            bias=getattr(model.lm_head, "bias", None),
        )

    chunk_size = int(loss_chunk_size)
    if chunk_size <= 0:
        logits = model.lm_head(hidden)
        return prefixlm_loss(logits, labels)

    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    target_mask = flat_labels != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        raise ValueError("batch contains no target tokens")
    target_hidden = flat_hidden[target_mask]
    target_labels = flat_labels[target_mask]
    total_loss = target_hidden.new_zeros(())
    total_targets = int(target_labels.numel())
    for start in range(0, total_targets, chunk_size):
        end = min(total_targets, start + chunk_size)
        logits = model.lm_head(target_hidden[start:end])
        total_loss = total_loss + F.cross_entropy(
            logits,
            target_labels[start:end],
            reduction="sum",
        )
    return total_loss / float(total_targets)


def prefixlm_loss_from_hidden_fp32_torch(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_chunk_size: int = 128,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    target_mask = flat_labels != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        raise ValueError("batch contains no target tokens")
    target_hidden = flat_hidden[target_mask]
    target_labels = flat_labels[target_mask]
    chunk_size = max(int(loss_chunk_size), 1)
    total_targets = int(target_labels.numel())
    total_loss = torch.zeros((), dtype=torch.float32, device=target_hidden.device)
    weight = model.lm_head.weight.float()
    bias = getattr(model.lm_head, "bias", None)
    bias_fp32 = bias.float() if bias is not None else None
    for start in range(0, total_targets, chunk_size):
        end = min(total_targets, start + chunk_size)
        logits = F.linear(target_hidden[start:end].float(), weight, bias_fp32)
        total_loss = total_loss + F.cross_entropy(
            logits,
            target_labels[start:end],
            reduction="sum",
        )
    return total_loss / float(total_targets)


def parse_token_id_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in str(raw or "").replace(",", " ").split():
        values.append(int(piece))
    return tuple(values)


def premature_stop_unlikelihood_loss_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    stop_token_ids: tuple[int, ...],
    loss_chunk_size: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if not stop_token_ids:
        raise ValueError("premature stop loss requires at least one stop token id")
    chunk_size = int(loss_chunk_size)
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    target_mask = flat_labels != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        raise ValueError("batch contains no target tokens")
    stop_ids = torch.tensor(
        tuple(int(token_id) for token_id in stop_token_ids),
        dtype=flat_labels.dtype,
        device=flat_labels.device,
    )
    target_labels = flat_labels[target_mask]
    non_stop_mask = ~torch.isin(target_labels, stop_ids)
    if not bool(non_stop_mask.any()):
        zero = hidden.new_zeros(())
        return zero, {
            "premature_stop_positions": 0,
            "premature_stop_mean_probability": 0.0,
        }
    target_hidden = flat_hidden[target_mask][non_stop_mask]
    total_loss = target_hidden.new_zeros(())
    total_positions = int(target_hidden.shape[0])
    probability_sum = 0.0
    for start in range(0, total_positions, chunk_size if chunk_size > 0 else total_positions):
        end = min(total_positions, start + (chunk_size if chunk_size > 0 else total_positions))
        logits = model.lm_head(target_hidden[start:end])
        stop_logits = logits[:, stop_ids.to(device=logits.device)]
        total_loss = total_loss + F.binary_cross_entropy_with_logits(
            stop_logits,
            torch.zeros_like(stop_logits),
            reduction="sum",
        )
        with torch.no_grad():
            probability_sum += float(stop_logits.sigmoid().detach().sum().cpu().item())
    denom = float(total_positions * len(stop_token_ids))
    return total_loss / denom, {
        "premature_stop_positions": int(total_positions),
        "premature_stop_mean_probability": probability_sum / denom,
    }


def response_start_loss_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    response_start_mask: torch.Tensor,
    *,
    stop_token_ids: tuple[int, ...] = (),
    loss_chunk_size: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    flat_start_mask = response_start_mask.reshape(-1).bool()
    target_mask = flat_start_mask & (flat_labels != IGNORE_LABEL_ID)
    if not bool(target_mask.any()):
        raise ValueError("batch contains no response-start target positions")
    target_hidden = flat_hidden[target_mask]
    target_labels = flat_labels[target_mask]
    chunk_size = int(loss_chunk_size)
    total_loss = target_hidden.new_zeros(())
    total_correct = 0
    gold_probability_sum = 0.0
    stop_probability_sum = 0.0
    total_targets = int(target_labels.numel())
    stop_ids = (
        torch.tensor(
            tuple(int(token_id) for token_id in stop_token_ids),
            dtype=target_labels.dtype,
            device=target_labels.device,
        )
        if stop_token_ids
        else None
    )
    for start in range(0, total_targets, chunk_size if chunk_size > 0 else total_targets):
        end = min(total_targets, start + (chunk_size if chunk_size > 0 else total_targets))
        logits = model.lm_head(target_hidden[start:end])
        chunk_labels = target_labels[start:end]
        total_loss = total_loss + F.cross_entropy(logits, chunk_labels, reduction="sum")
        with torch.no_grad():
            probabilities = logits.softmax(dim=-1)
            total_correct += int(logits.argmax(dim=-1).eq(chunk_labels).sum().detach().cpu().item())
            gold_probability_sum += float(
                probabilities.gather(1, chunk_labels.unsqueeze(1)).sum().detach().cpu().item()
            )
            if stop_ids is not None:
                stop_probability_sum += float(
                    probabilities[:, stop_ids.to(device=probabilities.device)].sum().detach().cpu().item()
                )
    metrics: dict[str, float | int] = {
        "response_start_positions": int(total_targets),
        "response_start_accuracy": float(total_correct / total_targets),
        "response_start_gold_probability": float(gold_probability_sum / total_targets),
    }
    if stop_ids is not None:
        metrics["response_start_stop_probability"] = float(stop_probability_sum / total_targets)
    return total_loss / float(total_targets), metrics


def row_balanced_response_loss_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_chunk_size: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    target_mask = flat_labels != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        raise ValueError("batch contains no target tokens")
    batch_size, seq_len = labels.shape
    row_ids = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand(batch_size, seq_len)
    target_rows = row_ids.reshape(-1)[target_mask]
    target_hidden = flat_hidden[target_mask]
    target_labels = flat_labels[target_mask]
    chunk_size = int(loss_chunk_size)
    row_loss_sums = hidden.new_zeros((batch_size,))
    row_counts = hidden.new_zeros((batch_size,))
    total_correct = 0
    total_targets = int(target_labels.numel())
    for start in range(0, total_targets, chunk_size if chunk_size > 0 else total_targets):
        end = min(total_targets, start + (chunk_size if chunk_size > 0 else total_targets))
        logits = model.lm_head(target_hidden[start:end])
        chunk_labels = target_labels[start:end]
        token_losses = F.cross_entropy(logits, chunk_labels, reduction="none")
        chunk_rows = target_rows[start:end]
        row_loss_sums.scatter_add_(0, chunk_rows, token_losses)
        row_counts.scatter_add_(0, chunk_rows, torch.ones_like(token_losses))
        with torch.no_grad():
            total_correct += int(logits.argmax(dim=-1).eq(chunk_labels).sum().detach().cpu().item())
    active_rows = row_counts > 0
    if not bool(active_rows.any()):
        raise ValueError("batch contains no active response rows")
    row_losses = row_loss_sums[active_rows] / row_counts[active_rows].clamp_min(1.0)
    return row_losses.mean(), {
        "row_balanced_response_rows": int(active_rows.sum().detach().cpu().item()),
        "row_balanced_response_targets": int(total_targets),
        "row_balanced_response_token_accuracy": float(total_correct / total_targets),
    }


def prefixlm_loss_for_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    think_steps: int,
    loss_chunk_size: int = 0,
    loss_kernel: str = "torch",
) -> torch.Tensor:
    chunk_size = int(loss_chunk_size)
    use_hidden_loss = (
        hasattr(model, "forward_hidden")
        and hasattr(model, "lm_head")
        and str(getattr(model, "value_codec", "learned")) == "learned"
        and (chunk_size > 0 or str(loss_kernel) != "torch")
    )
    if not use_hidden_loss:
        if str(loss_kernel) != "torch":
            raise ValueError(
                f"loss kernel {loss_kernel} requires model.forward_hidden and model.lm_head"
            )
        logits = model(input_ids, think_steps=int(think_steps))
        return prefixlm_loss(logits, labels)

    hidden = model.forward_hidden(input_ids, think_steps=int(think_steps))
    return prefixlm_loss_from_hidden(
        model,
        hidden,
        labels,
        loss_chunk_size=chunk_size,
        loss_kernel=str(loss_kernel),
    )


class PrefixLMTokenVerifier(nn.Module):
    """Small candidate-token verifier for PrefixLM target positions."""

    def __init__(self, d_model: int, hidden_dim: int = 0) -> None:
        super().__init__()
        width = int(hidden_dim) if int(hidden_dim) > 0 else int(d_model)
        self.context_proj = nn.Linear(int(d_model), width)
        self.token_proj = nn.Linear(int(d_model), width, bias=False)
        self.score = nn.Linear(width, 1)

    def forward(self, context_hidden: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        features = torch.tanh(self.context_proj(context_hidden) + self.token_proj(token_embedding))
        return self.score(features).squeeze(-1)


class NextImplicitTokenProjector(nn.Module):
    """Predict the next target token's implicit embedding from context hidden."""

    def __init__(self, d_model: int, hidden_dim: int = 0) -> None:
        super().__init__()
        width = int(hidden_dim)
        if width > 0:
            self.net = nn.Sequential(
                nn.Linear(int(d_model), width),
                nn.SiLU(),
                nn.Linear(width, int(d_model)),
            )
        else:
            self.net = nn.Linear(int(d_model), int(d_model))

    def forward(self, context_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(context_hidden)


def corrupted_token_ids(target_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    if int(vocab_size) <= 1:
        raise ValueError("vocab_size must be > 1 to create verifier negatives")
    offsets = (torch.arange(target_ids.numel(), device=target_ids.device) % (int(vocab_size) - 1)) + 1
    corrupted = (target_ids.reshape(-1) + offsets.to(dtype=target_ids.dtype)) % int(vocab_size)
    same = corrupted == target_ids.reshape(-1)
    if bool(same.any()):
        corrupted = torch.where(same, (corrupted + 1) % int(vocab_size), corrupted)
    return corrupted.reshape_as(target_ids)


def prefixlm_token_verifier_loss_from_hidden(
    model: nn.Module,
    verifier: PrefixLMTokenVerifier,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    max_targets: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if not hasattr(model, "token_embed"):
        raise ValueError("token verifier requires model.token_embed")
    target_mask = labels.reshape(-1) != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        raise ValueError("batch contains no target tokens")
    target_hidden = hidden.reshape(-1, hidden.size(-1))[target_mask]
    target_ids = labels.reshape(-1)[target_mask]
    if int(max_targets) > 0 and int(target_ids.numel()) > int(max_targets):
        indices = torch.linspace(
            0,
            int(target_ids.numel()) - 1,
            steps=int(max_targets),
            device=target_ids.device,
        ).long()
        target_hidden = target_hidden[indices]
        target_ids = target_ids[indices]

    negative_ids = corrupted_token_ids(target_ids, vocab_size=int(vocab_size))
    positive_embeddings = model.token_embed(target_ids)
    negative_embeddings = model.token_embed(negative_ids)
    positive_logits = verifier(target_hidden, positive_embeddings)
    negative_logits = verifier(target_hidden, negative_embeddings)
    logits = torch.cat([positive_logits, negative_logits], dim=0)
    targets = torch.cat(
        [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)],
        dim=0,
    )
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    with torch.no_grad():
        positive_correct = (positive_logits.sigmoid() >= 0.5).float()
        negative_correct = (negative_logits.sigmoid() < 0.5).float()
        accuracy = torch.cat([positive_correct, negative_correct], dim=0).mean()
    return loss, {
        "verifier_accuracy": float(accuracy.detach().cpu().item()),
        "verifier_targets": int(target_ids.numel()),
    }


def prefixlm_nitp_loss_from_hidden(
    model: nn.Module,
    projector: NextImplicitTokenProjector,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_targets: int = 0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """NITP-style auxiliary loss over PrefixLM target positions.

    The context hidden that predicts a supervised target token is asked to
    reconstruct that target token's embedding geometry. The target embedding is
    stop-grad so the auxiliary objective shapes hidden states without moving
    the lexical table directly.
    """

    if not hasattr(model, "token_embed"):
        raise ValueError("NITP loss requires model.token_embed")
    target_mask = labels.reshape(-1) != IGNORE_LABEL_ID
    if not bool(target_mask.any()):
        raise ValueError("batch contains no target tokens")
    target_hidden = hidden.reshape(-1, hidden.size(-1))[target_mask]
    target_ids = labels.reshape(-1)[target_mask]
    if int(max_targets) > 0 and int(target_ids.numel()) > int(max_targets):
        indices = torch.linspace(
            0,
            int(target_ids.numel()) - 1,
            steps=int(max_targets),
            device=target_ids.device,
        ).long()
        target_hidden = target_hidden[indices]
        target_ids = target_ids[indices]

    predicted = projector(target_hidden)
    with torch.no_grad():
        target_embedding = model.token_embed(target_ids).detach()
    predicted_norm = F.normalize(predicted.float(), dim=-1)
    target_norm = F.normalize(target_embedding.float(), dim=-1)
    cosine = (predicted_norm * target_norm).sum(dim=-1)
    loss = (1.0 - cosine).mean()
    return loss, {
        "nitp_targets": int(target_ids.numel()),
        "nitp_cosine_similarity": float(cosine.mean().detach().cpu().item()),
        "nitp_predicted_norm": float(predicted.float().norm(dim=-1).mean().detach().cpu().item()),
        "nitp_target_norm": float(target_embedding.float().norm(dim=-1).mean().detach().cpu().item()),
    }


@torch.no_grad()
def evaluate_prefixlm_loss(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    think_steps: int,
    max_batches: int,
    loss_chunk_size: int = 0,
    loss_kernel: str = "torch",
    amp_dtype: torch.dtype | None = None,
    trim_batch_to_max_length: bool = True,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    total_tokens = 0
    total_compute_tokens = 0
    batches = 0
    nonfinite_batches = 0
    fallback_batches = 0
    unresolved_nonfinite_batches = 0
    nonfinite_batch_indices: list[int] = []
    unresolved_nonfinite_batch_indices: list[int] = []
    nonfinite_target_tokens = 0
    unresolved_target_tokens = 0
    fallback_hidden_target_elements = 0
    fallback_hidden_nonfinite_elements = 0
    fallback_hidden_nonfinite_batches = 0
    unresolved_hidden_target_elements = 0
    unresolved_hidden_nonfinite_elements = 0
    unresolved_with_finite_hidden_batches = 0
    attempted_targets = 0
    for batch_index, batch in enumerate(loader):
        if bool(trim_batch_to_max_length):
            batch = trim_prefixlm_batch_to_max_valid_length(batch)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_tokens = int((labels != IGNORE_LABEL_ID).sum().detach().cpu().item())
        attempted_targets += target_tokens
        with autocast_context(device, amp_dtype):
            loss = prefixlm_loss_for_batch(
                model,
                input_ids,
                labels,
                think_steps=int(think_steps),
                loss_chunk_size=int(loss_chunk_size),
                loss_kernel=str(loss_kernel),
        )
        if not bool(torch.isfinite(loss.detach())):
            nonfinite_batches += 1
            nonfinite_target_tokens += target_tokens
            if len(nonfinite_batch_indices) < 16:
                nonfinite_batch_indices.append(int(batch_index))
            if not (hasattr(model, "forward_hidden") and hasattr(model, "lm_head")):
                raise RuntimeError("non-finite eval loss and no hidden-loss fallback is available")
            fallback_chunk_size = max(int(loss_chunk_size), 128)
            with autocast_context(device, None):
                hidden = model.forward_hidden(input_ids, think_steps=int(think_steps))
                target_hidden = hidden.reshape(-1, hidden.size(-1))[
                    labels.reshape(-1) != IGNORE_LABEL_ID
                ]
                hidden_elements = int(target_hidden.numel())
                hidden_nonfinite = (
                    int((~torch.isfinite(target_hidden)).sum().detach().cpu().item())
                    if hidden_elements > 0
                    else 0
                )
                fallback_hidden_target_elements += hidden_elements
                fallback_hidden_nonfinite_elements += hidden_nonfinite
                if hidden_nonfinite > 0:
                    fallback_hidden_nonfinite_batches += 1
                loss = prefixlm_loss_from_hidden_fp32_torch(
                    model,
                    hidden,
                    labels,
                    loss_chunk_size=fallback_chunk_size,
                )
            fallback_batches += 1
            if not bool(torch.isfinite(loss.detach())):
                unresolved_nonfinite_batches += 1
                unresolved_target_tokens += target_tokens
                unresolved_hidden_target_elements += hidden_elements
                unresolved_hidden_nonfinite_elements += hidden_nonfinite
                if hidden_nonfinite == 0:
                    unresolved_with_finite_hidden_batches += 1
                if len(unresolved_nonfinite_batch_indices) < 16:
                    unresolved_nonfinite_batch_indices.append(int(batch_index))
                batches += 1
                if int(max_batches) > 0 and batches >= int(max_batches):
                    break
                continue
        flat_loss = loss * float(target_tokens)
        total_loss += float(flat_loss.detach().cpu().item())
        total_targets += target_tokens
        total_tokens += int(attention_mask.sum().detach().cpu().item())
        total_compute_tokens += int(input_ids.numel())
        batches += 1
        if int(max_batches) > 0 and batches >= int(max_batches):
            break
    if attempted_targets <= 0:
        raise ValueError("evaluation loader produced no target tokens")
    if total_targets <= 0:
        loss_value = float("inf")
    else:
        loss_value = total_loss / float(total_targets)
    return {
        "loss": loss_value,
        "target_tokens": int(total_targets),
        "attempted_target_tokens": int(attempted_targets),
        "tokens": int(total_tokens),
        "compute_tokens": int(total_compute_tokens),
        "batches": int(batches),
        "nonfinite_batches": int(nonfinite_batches),
        "fallback_batches": int(fallback_batches),
        "unresolved_nonfinite_batches": int(unresolved_nonfinite_batches),
        "nonfinite_batch_indices": nonfinite_batch_indices,
        "unresolved_nonfinite_batch_indices": unresolved_nonfinite_batch_indices,
        "nonfinite_target_tokens": int(nonfinite_target_tokens),
        "unresolved_target_tokens": int(unresolved_target_tokens),
        "fallback_hidden_target_elements": int(fallback_hidden_target_elements),
        "fallback_hidden_nonfinite_elements": int(fallback_hidden_nonfinite_elements),
        "fallback_hidden_nonfinite_batches": int(fallback_hidden_nonfinite_batches),
        "unresolved_hidden_target_elements": int(unresolved_hidden_target_elements),
        "unresolved_hidden_nonfinite_elements": int(unresolved_hidden_nonfinite_elements),
        "unresolved_with_finite_hidden_batches": int(unresolved_with_finite_hidden_batches),
    }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    verifier: PrefixLMTokenVerifier | None = None,
    nitp_projector: NextImplicitTokenProjector | None = None,
    step: int,
    tokens_seen: int,
    target_tokens_seen: int,
    compute_tokens_seen: int,
    losses: list[dict[str, float | int]],
    eval_losses: list[dict[str, float | int]],
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
    eval_dataset_summary: dict[str, Any] | None,
    model_summary: dict[str, Any],
    include_optimizer: bool = True,
    copy_safe_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "target_tokens_seen": int(target_tokens_seen),
        "compute_tokens_seen": int(compute_tokens_seen),
        "model_state_dict": model.state_dict(),
        "checkpoint_includes_optimizer": bool(include_optimizer),
        "loss_history": losses,
        "eval_loss_history": eval_losses,
        "args": vars(args),
        "dataset": dataset_summary,
        "eval_dataset": eval_dataset_summary,
        "model": model_summary,
    }
    if bool(include_optimizer):
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if verifier is not None:
        payload["token_verifier_state_dict"] = verifier.state_dict()
    if nitp_projector is not None:
        payload["nitp_projector_state_dict"] = nitp_projector.state_dict()
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        if copy_safe_path is not None:
            publish_copy_safe_checkpoint(path, copy_safe_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def publish_copy_safe_checkpoint(source_path: Path, copy_safe_path: Path) -> None:
    """Expose a stable checkpoint alias for copying without another torch.save."""
    if source_path == copy_safe_path:
        return
    copy_safe_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if copy_safe_path.exists() and source_path.samefile(copy_safe_path):
            return
    except OSError:
        pass
    tmp_copy_path = copy_safe_path.with_name(f".{copy_safe_path.name}.tmp.{os.getpid()}")
    try:
        tmp_copy_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        try:
            os.link(source_path, tmp_copy_path)
        except OSError:
            shutil.copy2(source_path, tmp_copy_path)
        os.replace(tmp_copy_path, copy_safe_path)
    finally:
        try:
            tmp_copy_path.unlink(missing_ok=True)
        except OSError:
            pass


class MetricLoggers:
    def __init__(self, *, tensorboard_writer: Any | None, aim_run: Any | None) -> None:
        self.tensorboard_writer = tensorboard_writer
        self.aim_run = aim_run

    def log_scalar(self, name: str, value: float | int, step: int) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, int(step))
        if self.aim_run is not None:
            self.aim_run.track(value, name=name, step=int(step))

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        if self.aim_run is not None:
            self.aim_run.close()


def create_metric_loggers(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    dataset_summary: dict[str, Any],
    eval_dataset_summary: dict[str, Any] | None,
    model_summary: dict[str, Any],
) -> MetricLoggers:
    tensorboard_writer = None
    if str(args.tensorboard_dir):
        if importlib.util.find_spec("torch.utils.tensorboard") is None:
            raise RuntimeError("TensorBoard logging requested, but torch.utils.tensorboard is unavailable")
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = Path(args.tensorboard_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        tensorboard_writer.add_text("run/out_dir", str(out_dir), 0)

    aim_run = None
    if str(args.aim_repo) or str(args.aim_experiment) or str(args.aim_run_name):
        if importlib.util.find_spec("aim") is None:
            raise RuntimeError("Aim logging requested, but aim is unavailable")
        from aim import Run

        repo = str(args.aim_repo) if str(args.aim_repo) else None
        aim_run = Run(repo=repo, experiment=str(args.aim_experiment or "native_prefixlm_from_scratch"))
        aim_run.name = str(args.aim_run_name or out_dir.name)
        if str(args.aim_description):
            aim_run.description = str(args.aim_description)
        aim_run["hparams"] = {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "trim_batch_to_max_length": bool(args.trim_batch_to_max_length),
            "length_bucketed_batches": bool(args.length_bucketed_batches),
            "length_bucket_size_multiplier": int(args.length_bucket_size_multiplier),
            "checkpoint_every": int(args.checkpoint_every),
            "model_checkpoint_every": int(args.model_checkpoint_every),
            "loss_kernel": str(args.loss_kernel),
            "activation_checkpointing": bool(args.activation_checkpointing),
            "lr": float(args.lr),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "weight_decay": float(args.weight_decay),
            "optimizer": str(args.optimizer),
            "galore_rank": int(args.galore_rank),
            "galore_update_proj_gap": int(args.galore_update_proj_gap),
            "galore_scale": float(args.galore_scale),
            "galore_proj_type": str(args.galore_proj_type),
            "galore_min_dim": int(args.galore_min_dim),
            "galore_include_embeddings": bool(args.galore_include_embeddings),
            "train_think_steps": int(args.train_think_steps),
            "loss_chunk_size": int(args.loss_chunk_size),
            "token_verifier_loss_weight": float(args.token_verifier_loss_weight),
            "token_verifier_hidden_dim": int(args.token_verifier_hidden_dim),
            "token_verifier_max_targets": int(args.token_verifier_max_targets),
            "nitp_loss_weight": float(args.nitp_loss_weight),
            "nitp_hidden_dim": int(args.nitp_hidden_dim),
            "nitp_max_targets": int(args.nitp_max_targets),
            "premature_stop_loss_weight": float(args.premature_stop_loss_weight),
            "premature_stop_token_ids": str(args.premature_stop_token_ids),
            "response_start_loss_weight": float(args.response_start_loss_weight),
            "row_balanced_response_loss_weight": float(args.row_balanced_response_loss_weight),
            "seed": int(args.seed),
            "amp_dtype": str(args.amp_dtype),
            "matmul_precision": str(args.matmul_precision),
        }
        aim_run["dataset"] = dataset_summary
        aim_run["eval_dataset"] = eval_dataset_summary or {}
        aim_run["model"] = model_summary

    return MetricLoggers(tensorboard_writer=tensorboard_writer, aim_run=aim_run)


def scheduled_learning_rate(args: argparse.Namespace, step: int) -> float:
    warmup_steps = int(args.lr_warmup_steps)
    base_lr = float(args.lr)
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, max(1, int(step)) / float(warmup_steps))


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def throughput_metrics(
    *,
    step: int,
    start_step: int,
    current_time: float,
    train_start_time: float,
    previous_log_step: int,
    previous_log_time: float,
    tokens_seen: int,
    target_tokens_seen: int,
    compute_tokens_seen: int,
    previous_log_tokens: int,
    previous_log_target_tokens: int,
    previous_log_compute_tokens: int,
) -> dict[str, float]:
    elapsed_sec = max(1e-9, float(current_time) - float(train_start_time))
    interval_sec = max(1e-9, float(current_time) - float(previous_log_time))
    completed_steps = max(0, int(step) - int(start_step))
    interval_steps = max(0, int(step) - int(previous_log_step))
    interval_tokens = max(0, int(tokens_seen) - int(previous_log_tokens))
    interval_target_tokens = max(
        0,
        int(target_tokens_seen) - int(previous_log_target_tokens),
    )
    interval_compute_tokens = max(
        0,
        int(compute_tokens_seen) - int(previous_log_compute_tokens),
    )
    return {
        "elapsed_sec": float(elapsed_sec),
        "interval_sec": float(interval_sec),
        "steps_per_sec": float(completed_steps) / elapsed_sec,
        "interval_steps_per_sec": float(interval_steps) / interval_sec,
        "tokens_per_sec": float(tokens_seen) / elapsed_sec,
        "target_tokens_per_sec": float(target_tokens_seen) / elapsed_sec,
        "compute_tokens_per_sec": float(compute_tokens_seen) / elapsed_sec,
        "interval_tokens_per_sec": float(interval_tokens) / interval_sec,
        "interval_target_tokens_per_sec": float(interval_target_tokens) / interval_sec,
        "interval_compute_tokens_per_sec": float(interval_compute_tokens) / interval_sec,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    assert_mamba3_free_args(args)
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = DataIOSampledPrefixLMDataset(
        args.sampled_data,
        seq_len=int(args.seq_len),
        epoch=int(args.epoch),
        target_only=not bool(args.train_instruction_tokens),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        drop_overlength=not bool(args.keep_overlength),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = build_prefixlm_train_loader(
        dataset,
        batch_size=int(args.batch_size),
        generator=generator,
        length_bucketed_batches=bool(args.length_bucketed_batches),
        length_bucket_size_multiplier=int(args.length_bucket_size_multiplier),
    )
    eval_loader = None
    eval_dataset_summary = None
    if int(args.eval_every) > 0:
        eval_dataset = DataIOSampledPrefixLMDataset(
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
            collate_fn=collate_prefixlm_rows,
            drop_last=False,
        )
        eval_dataset_summary = eval_dataset.summary()
        eval_dataset_summary["eval_batch_size"] = int(args.eval_batch_size or args.batch_size)
        eval_dataset_summary["eval_max_batches"] = int(args.eval_max_batches)
    dataset_summary = dataset.summary()
    if bool(args.dry_run_loader):
        report = {
            "decision": "dry_run_loader",
            "accepted": False,
            "dataset": dataset_summary,
            "plain_language_read": (
                "The HRM-Text textbook is readable by the native one-body "
                "PrefixLM path. This does not prove learning efficiency yet."
            ),
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    device = torch.device(str(args.device))
    amp_dtype = resolve_amp_dtype(str(args.amp_dtype))
    model_vocab_size = int(args.model_vocab_size)
    if model_vocab_size <= 0:
        model_vocab_size = int(dataset_summary["model_vocab_size"])
    model = build_model(args, vocab_size=model_vocab_size).to(device)
    stop_token_ids = parse_token_id_list(str(args.premature_stop_token_ids))
    if float(args.premature_stop_loss_weight) > 0.0 and not stop_token_ids:
        raise ValueError("--premature-stop-token-ids is required when premature stop loss is enabled")
    token_verifier = None
    if float(args.token_verifier_loss_weight) > 0.0:
        token_verifier = PrefixLMTokenVerifier(
            int(args.d_model),
            hidden_dim=int(args.token_verifier_hidden_dim),
        ).to(device)
    nitp_projector = None
    if float(args.nitp_loss_weight) > 0.0:
        nitp_projector = NextImplicitTokenProjector(
            int(args.d_model),
            hidden_dim=int(args.nitp_hidden_dim),
        ).to(device)
    if token_verifier is not None and bool(args.token_verifier_freeze_model):
        for parameter in model.parameters():
            parameter.requires_grad = False
    optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    extra_named_parameters: list[tuple[str, torch.nn.Parameter]] = []
    if token_verifier is not None:
        extra_named_parameters = [
            (f"token_verifier.{name}", parameter)
            for name, parameter in token_verifier.named_parameters()
        ]
        optimizer_parameters += [
            parameter for _, parameter in extra_named_parameters if parameter.requires_grad
        ]
    if nitp_projector is not None:
        nitp_named_parameters = [
            (f"nitp_projector.{name}", parameter)
            for name, parameter in nitp_projector.named_parameters()
        ]
        extra_named_parameters += nitp_named_parameters
        optimizer_parameters += [
            parameter for _, parameter in nitp_named_parameters if parameter.requires_grad
        ]
    if not optimizer_parameters:
        raise ValueError("no trainable parameters selected")
    optimizer, optimizer_report = build_memory_efficient_optimizer(
        model,
        optimizer_name=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        beta1=float(args.adam_beta1),
        beta2=float(args.adam_beta2),
        device=device,
        extra_named_parameters=extra_named_parameters,
        galore_rank=int(args.galore_rank),
        galore_update_proj_gap=int(args.galore_update_proj_gap),
        galore_scale=float(args.galore_scale),
        galore_proj_type=str(args.galore_proj_type),
        galore_min_dim=int(args.galore_min_dim),
        galore_include_embeddings=bool(args.galore_include_embeddings),
    )
    model_summary = {
        "vocab_size": int(model_vocab_size),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "n_kv_heads": int(args.n_kv_heads),
        "d_ff": int(args.d_ff),
        "backbone": str(args.backbone),
        "encode_backbone": str(args.encode_backbone or args.backbone),
        "think_backbone": str(args.think_backbone or args.backbone),
        "decode_backbone": str(args.decode_backbone or args.backbone),
        "think_structure": str(args.think_structure),
        "delta_backend": str(args.delta_backend),
        "token_verifier_enabled": token_verifier is not None,
        "token_verifier_loss_weight": float(args.token_verifier_loss_weight),
        "token_verifier_hidden_dim": int(args.token_verifier_hidden_dim),
        "token_verifier_max_targets": int(args.token_verifier_max_targets),
        "token_verifier_freeze_model": bool(args.token_verifier_freeze_model),
        "nitp_enabled": nitp_projector is not None,
        "nitp_loss_weight": float(args.nitp_loss_weight),
        "nitp_hidden_dim": int(args.nitp_hidden_dim),
        "nitp_max_targets": int(args.nitp_max_targets),
        "premature_stop_loss_weight": float(args.premature_stop_loss_weight),
        "premature_stop_token_ids": tuple(int(token_id) for token_id in stop_token_ids),
        "response_start_loss_weight": float(args.response_start_loss_weight),
        "row_balanced_response_loss_weight": float(args.row_balanced_response_loss_weight),
        "loss_kernel": str(args.loss_kernel),
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "amp_dtype": str(args.amp_dtype),
        "matmul_precision": str(args.matmul_precision),
        "optimizer": optimizer_report,
        "activation_checkpointing": bool(args.activation_checkpointing),
        "length_bucketed_batches": bool(args.length_bucketed_batches),
        "length_bucket_size_multiplier": int(args.length_bucket_size_multiplier),
    } | parameter_counts(model)
    loggers = create_metric_loggers(
        args,
        out_dir=out_dir,
        dataset_summary=dataset_summary,
        eval_dataset_summary=eval_dataset_summary,
        model_summary=model_summary,
    )
    iterator = iter(loader)
    losses: list[dict[str, float | int]] = []
    eval_losses: list[dict[str, float | int]] = []
    best_eval_loss = float("inf")
    best_eval_step = 0
    tokens_seen = 0
    target_tokens_seen = 0
    compute_tokens_seen = 0
    start_step = 0
    if str(args.resume):
        checkpoint = torch.load(str(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if token_verifier is not None and "token_verifier_state_dict" in checkpoint:
            token_verifier.load_state_dict(checkpoint["token_verifier_state_dict"])
        if nitp_projector is not None and "nitp_projector_state_dict" in checkpoint:
            nitp_projector.load_state_dict(checkpoint["nitp_projector_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                new_auxiliary_head = (
                    (token_verifier is not None and "token_verifier_state_dict" not in checkpoint)
                    or (
                        nitp_projector is not None
                        and "nitp_projector_state_dict" not in checkpoint
                    )
                )
                if not new_auxiliary_head:
                    raise
                print(
                    json.dumps(
                        {
                            "event": "optimizer_resume_skipped_for_new_auxiliary_head",
                            "resume": str(args.resume),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        else:
            print(
                json.dumps(
                    {
                        "event": "optimizer_resume_unavailable_model_only_checkpoint",
                        "resume": str(args.resume),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        start_step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        target_tokens_seen = int(checkpoint.get("target_tokens_seen", 0))
        compute_tokens_seen = int(checkpoint.get("compute_tokens_seen", 0))
        losses = list(checkpoint.get("loss_history") or [])
        eval_losses = list(checkpoint.get("eval_loss_history") or [])
        for previous_eval in eval_losses:
            previous_loss = float(previous_eval.get("eval_loss", float("inf")))
            if np.isfinite(previous_loss) and previous_loss < best_eval_loss:
                best_eval_loss = previous_loss
                best_eval_step = int(previous_eval.get("step", 0))
        print(
            json.dumps(
                {
                    "event": "resumed_checkpoint",
                    "resume": str(args.resume),
                    "start_step": int(start_step),
                    "tokens_seen": int(tokens_seen),
                    "target_tokens_seen": int(target_tokens_seen),
                    "compute_tokens_seen": int(compute_tokens_seen),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    train_start_time = time.perf_counter()
    previous_log_time = train_start_time
    previous_log_step = int(start_step)
    previous_log_tokens = int(tokens_seen)
    previous_log_target_tokens = int(target_tokens_seen)
    previous_log_compute_tokens = int(compute_tokens_seen)
    try:
        for step in range(int(start_step) + 1, int(args.steps) + 1):
            lr = scheduled_learning_rate(args, int(step))
            set_optimizer_learning_rate(optimizer, lr)
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if bool(args.trim_batch_to_max_length):
                batch = trim_prefixlm_batch_to_max_valid_length(batch)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_start_mask = batch["response_start_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model.train()
            if token_verifier is not None:
                token_verifier.train()
                with autocast_context(device, amp_dtype):
                    hidden = model.forward_hidden(input_ids, think_steps=int(args.train_think_steps))
                    lm_loss = prefixlm_loss_from_hidden(
                        model,
                        hidden,
                        labels,
                        loss_chunk_size=int(args.loss_chunk_size),
                        loss_kernel=str(args.loss_kernel),
                    )
                    verifier_loss, verifier_metrics = prefixlm_token_verifier_loss_from_hidden(
                        model,
                        token_verifier,
                        hidden,
                        labels,
                        vocab_size=int(model_vocab_size),
                        max_targets=int(args.token_verifier_max_targets),
                    )
                loss = lm_loss + float(args.token_verifier_loss_weight) * verifier_loss
            else:
                if (
                    float(args.premature_stop_loss_weight) > 0.0
                    or nitp_projector is not None
                ):
                    with autocast_context(device, amp_dtype):
                        hidden = model.forward_hidden(input_ids, think_steps=int(args.train_think_steps))
                        lm_loss = prefixlm_loss_from_hidden(
                            model,
                            hidden,
                            labels,
                            loss_chunk_size=int(args.loss_chunk_size),
                            loss_kernel=str(args.loss_kernel),
                        )
                else:
                    hidden = None
                    with autocast_context(device, amp_dtype):
                        lm_loss = prefixlm_loss_for_batch(
                            model,
                            input_ids,
                            labels,
                            think_steps=int(args.train_think_steps),
                            loss_chunk_size=int(args.loss_chunk_size),
                            loss_kernel=str(args.loss_kernel),
                        )
                verifier_loss = None
                verifier_metrics = {}
                loss = lm_loss
            nitp_loss = None
            nitp_metrics: dict[str, float | int] = {}
            if nitp_projector is not None:
                nitp_projector.train()
                if "hidden" not in locals() or hidden is None:
                    with autocast_context(device, amp_dtype):
                        hidden = model.forward_hidden(input_ids, think_steps=int(args.train_think_steps))
                with autocast_context(device, amp_dtype):
                    nitp_loss, nitp_metrics = prefixlm_nitp_loss_from_hidden(
                        model,
                        nitp_projector,
                        hidden,
                        labels,
                        max_targets=int(args.nitp_max_targets),
                    )
                loss = loss + float(args.nitp_loss_weight) * nitp_loss
            premature_stop_loss = None
            premature_stop_metrics: dict[str, float | int] = {}
            if float(args.premature_stop_loss_weight) > 0.0:
                if "hidden" not in locals() or hidden is None:
                    with autocast_context(device, amp_dtype):
                        hidden = model.forward_hidden(input_ids, think_steps=int(args.train_think_steps))
                with autocast_context(device, amp_dtype):
                    premature_stop_loss, premature_stop_metrics = (
                        premature_stop_unlikelihood_loss_from_hidden(
                            model,
                            hidden,
                            labels,
                            stop_token_ids=stop_token_ids,
                            loss_chunk_size=int(args.loss_chunk_size),
                        )
                    )
                loss = loss + float(args.premature_stop_loss_weight) * premature_stop_loss
            response_start_loss = None
            response_start_metrics: dict[str, float | int] = {}
            if float(args.response_start_loss_weight) > 0.0:
                if "hidden" not in locals() or hidden is None:
                    with autocast_context(device, amp_dtype):
                        hidden = model.forward_hidden(input_ids, think_steps=int(args.train_think_steps))
                with autocast_context(device, amp_dtype):
                    response_start_loss, response_start_metrics = response_start_loss_from_hidden(
                        model,
                        hidden,
                        labels,
                        response_start_mask,
                        stop_token_ids=stop_token_ids,
                        loss_chunk_size=int(args.loss_chunk_size),
                        )
                loss = loss + float(args.response_start_loss_weight) * response_start_loss
            row_balanced_response_loss = None
            row_balanced_response_metrics: dict[str, float | int] = {}
            if float(args.row_balanced_response_loss_weight) > 0.0:
                if "hidden" not in locals() or hidden is None:
                    with autocast_context(device, amp_dtype):
                        hidden = model.forward_hidden(input_ids, think_steps=int(args.train_think_steps))
                with autocast_context(device, amp_dtype):
                    row_balanced_response_loss, row_balanced_response_metrics = (
                        row_balanced_response_loss_from_hidden(
                            model,
                            hidden,
                            labels,
                            loss_chunk_size=int(args.loss_chunk_size),
                        )
                    )
                loss = loss + float(args.row_balanced_response_loss_weight) * row_balanced_response_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, float(args.grad_clip))
            optimizer.step()
            batch_tokens = int(attention_mask.sum().detach().cpu().item())
            batch_targets = int((labels != IGNORE_LABEL_ID).sum().detach().cpu().item())
            batch_compute_tokens = int(input_ids.numel())
            tokens_seen += batch_tokens
            target_tokens_seen += batch_targets
            compute_tokens_seen += batch_compute_tokens
            loss_value = float(loss.detach().cpu().item())
            if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
                now = time.perf_counter()
                row = {
                    "step": int(step),
                    "loss": loss_value,
                    "lm_loss": float(lm_loss.detach().cpu().item()),
                    "lr": float(lr),
                    "tokens_seen": int(tokens_seen),
                    "target_tokens_seen": int(target_tokens_seen),
                    "compute_tokens_seen": int(compute_tokens_seen),
                    "batch_compute_tokens": int(batch_compute_tokens),
                }
                row.update(
                    throughput_metrics(
                        step=int(step),
                        start_step=int(start_step),
                        current_time=float(now),
                        train_start_time=float(train_start_time),
                        previous_log_step=int(previous_log_step),
                        previous_log_time=float(previous_log_time),
                        tokens_seen=int(tokens_seen),
                        target_tokens_seen=int(target_tokens_seen),
                        compute_tokens_seen=int(compute_tokens_seen),
                        previous_log_tokens=int(previous_log_tokens),
                        previous_log_target_tokens=int(previous_log_target_tokens),
                        previous_log_compute_tokens=int(previous_log_compute_tokens),
                    )
                )
                previous_log_time = now
                previous_log_step = int(step)
                previous_log_tokens = int(tokens_seen)
                previous_log_target_tokens = int(target_tokens_seen)
                previous_log_compute_tokens = int(compute_tokens_seen)
                if verifier_loss is not None:
                    row["token_verifier_loss"] = float(verifier_loss.detach().cpu().item())
                    row["token_verifier_accuracy"] = float(verifier_metrics["verifier_accuracy"])
                    row["token_verifier_targets"] = int(verifier_metrics["verifier_targets"])
                if nitp_loss is not None:
                    row["nitp_loss"] = float(nitp_loss.detach().cpu().item())
                    row["nitp_targets"] = int(nitp_metrics["nitp_targets"])
                    row["nitp_cosine_similarity"] = float(
                        nitp_metrics["nitp_cosine_similarity"]
                    )
                    row["nitp_predicted_norm"] = float(nitp_metrics["nitp_predicted_norm"])
                    row["nitp_target_norm"] = float(nitp_metrics["nitp_target_norm"])
                if premature_stop_loss is not None:
                    row["premature_stop_loss"] = float(
                        premature_stop_loss.detach().cpu().item()
                    )
                    row["premature_stop_positions"] = int(
                        premature_stop_metrics["premature_stop_positions"]
                    )
                    row["premature_stop_mean_probability"] = float(
                        premature_stop_metrics["premature_stop_mean_probability"]
                    )
                if response_start_loss is not None:
                    row["response_start_loss"] = float(
                        response_start_loss.detach().cpu().item()
                    )
                    row["response_start_positions"] = int(
                        response_start_metrics["response_start_positions"]
                    )
                    row["response_start_accuracy"] = float(
                        response_start_metrics["response_start_accuracy"]
                    )
                    row["response_start_gold_probability"] = float(
                        response_start_metrics["response_start_gold_probability"]
                    )
                    if "response_start_stop_probability" in response_start_metrics:
                        row["response_start_stop_probability"] = float(
                            response_start_metrics["response_start_stop_probability"]
                        )
                if row_balanced_response_loss is not None:
                    row["row_balanced_response_loss"] = float(
                        row_balanced_response_loss.detach().cpu().item()
                    )
                    row["row_balanced_response_rows"] = int(
                        row_balanced_response_metrics["row_balanced_response_rows"]
                    )
                    row["row_balanced_response_targets"] = int(
                        row_balanced_response_metrics["row_balanced_response_targets"]
                    )
                    row["row_balanced_response_token_accuracy"] = float(
                        row_balanced_response_metrics["row_balanced_response_token_accuracy"]
                    )
                losses.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                loggers.log_scalar("train/loss", loss_value, int(step))
                loggers.log_scalar("train/lm_loss", float(row["lm_loss"]), int(step))
                loggers.log_scalar("train/lr", float(row["lr"]), int(step))
                loggers.log_scalar("train/tokens_seen", int(tokens_seen), int(step))
                loggers.log_scalar("train/target_tokens_seen", int(target_tokens_seen), int(step))
                loggers.log_scalar("train/compute_tokens_seen", int(compute_tokens_seen), int(step))
                loggers.log_scalar("train/tokens_per_sec", float(row["tokens_per_sec"]), int(step))
                loggers.log_scalar(
                    "train/target_tokens_per_sec",
                    float(row["target_tokens_per_sec"]),
                    int(step),
                )
                loggers.log_scalar(
                    "train/compute_tokens_per_sec",
                    float(row["compute_tokens_per_sec"]),
                    int(step),
                )
                loggers.log_scalar(
                    "train/interval_tokens_per_sec",
                    float(row["interval_tokens_per_sec"]),
                    int(step),
                )
                loggers.log_scalar(
                    "train/interval_target_tokens_per_sec",
                    float(row["interval_target_tokens_per_sec"]),
                    int(step),
                )
                loggers.log_scalar(
                    "train/interval_compute_tokens_per_sec",
                    float(row["interval_compute_tokens_per_sec"]),
                    int(step),
                )
                if verifier_loss is not None:
                    loggers.log_scalar(
                        "train/token_verifier_loss",
                        float(row["token_verifier_loss"]),
                        int(step),
                    )
                    loggers.log_scalar(
                        "train/token_verifier_accuracy",
                        float(row["token_verifier_accuracy"]),
                        int(step),
                    )
                if nitp_loss is not None:
                    loggers.log_scalar(
                        "train/nitp_loss",
                        float(row["nitp_loss"]),
                        int(step),
                    )
                    loggers.log_scalar(
                        "train/nitp_cosine_similarity",
                        float(row["nitp_cosine_similarity"]),
                        int(step),
                    )
                if premature_stop_loss is not None:
                    loggers.log_scalar(
                        "train/premature_stop_loss",
                        float(row["premature_stop_loss"]),
                        int(step),
                    )
                    loggers.log_scalar(
                        "train/premature_stop_mean_probability",
                        float(row["premature_stop_mean_probability"]),
                        int(step),
                    )
                if response_start_loss is not None:
                    loggers.log_scalar(
                        "train/response_start_loss",
                        float(row["response_start_loss"]),
                        int(step),
                    )
                    loggers.log_scalar(
                        "train/response_start_accuracy",
                        float(row["response_start_accuracy"]),
                        int(step),
                    )
                    loggers.log_scalar(
                        "train/response_start_gold_probability",
                        float(row["response_start_gold_probability"]),
                        int(step),
                    )
                    if "response_start_stop_probability" in row:
                        loggers.log_scalar(
                            "train/response_start_stop_probability",
                            float(row["response_start_stop_probability"]),
                            int(step),
                        )
                if row_balanced_response_loss is not None:
                    loggers.log_scalar(
                        "train/row_balanced_response_loss",
                        float(row["row_balanced_response_loss"]),
                        int(step),
                    )
                    loggers.log_scalar(
                        "train/row_balanced_response_token_accuracy",
                        float(row["row_balanced_response_token_accuracy"]),
                        int(step),
                    )
            if eval_loader is not None and (
                step == 1 or step % int(args.eval_every) == 0 or step == int(args.steps)
            ):
                metrics = evaluate_prefixlm_loss(
                    model,
                    eval_loader,
                    device=device,
                    think_steps=int(args.train_think_steps),
                    max_batches=int(args.eval_max_batches),
                    loss_chunk_size=int(args.loss_chunk_size),
                    loss_kernel=str(args.loss_kernel),
                    amp_dtype=amp_dtype,
                    trim_batch_to_max_length=bool(args.trim_batch_to_max_length),
                )
                eval_row = {
                    "step": int(step),
                    "tokens_seen": int(tokens_seen),
                    "target_tokens_seen": int(target_tokens_seen),
                    "eval_loss": float(metrics["loss"]),
                    "eval_tokens": int(metrics["tokens"]),
                    "eval_target_tokens": int(metrics["target_tokens"]),
                    "eval_attempted_target_tokens": int(metrics.get("attempted_target_tokens", metrics["target_tokens"])),
                    "eval_compute_tokens": int(metrics["compute_tokens"]),
                    "eval_nonfinite_batches": int(metrics.get("nonfinite_batches", 0)),
                    "eval_fallback_batches": int(metrics.get("fallback_batches", 0)),
                    "eval_unresolved_nonfinite_batches": int(
                        metrics.get("unresolved_nonfinite_batches", 0)
                    ),
                    "eval_nonfinite_batch_indices": list(
                        metrics.get("nonfinite_batch_indices", [])
                    ),
                    "eval_unresolved_nonfinite_batch_indices": list(
                        metrics.get("unresolved_nonfinite_batch_indices", [])
                    ),
                    "eval_nonfinite_target_tokens": int(
                        metrics.get("nonfinite_target_tokens", 0)
                    ),
                    "eval_unresolved_target_tokens": int(
                        metrics.get("unresolved_target_tokens", 0)
                    ),
                    "eval_fallback_hidden_target_elements": int(
                        metrics.get("fallback_hidden_target_elements", 0)
                    ),
                    "eval_fallback_hidden_nonfinite_elements": int(
                        metrics.get("fallback_hidden_nonfinite_elements", 0)
                    ),
                    "eval_fallback_hidden_nonfinite_batches": int(
                        metrics.get("fallback_hidden_nonfinite_batches", 0)
                    ),
                    "eval_unresolved_hidden_target_elements": int(
                        metrics.get("unresolved_hidden_target_elements", 0)
                    ),
                    "eval_unresolved_hidden_nonfinite_elements": int(
                        metrics.get("unresolved_hidden_nonfinite_elements", 0)
                    ),
                    "eval_unresolved_with_finite_hidden_batches": int(
                        metrics.get("unresolved_with_finite_hidden_batches", 0)
                    ),
                }
                eval_losses.append(eval_row)
                print(json.dumps(eval_row, ensure_ascii=False), flush=True)
                loggers.log_scalar("eval/loss", float(metrics["loss"]), int(step))
                loggers.log_scalar("eval/tokens", int(metrics["tokens"]), int(step))
                loggers.log_scalar("eval/target_tokens", int(metrics["target_tokens"]), int(step))
                loggers.log_scalar(
                    "eval/attempted_target_tokens",
                    int(metrics.get("attempted_target_tokens", metrics["target_tokens"])),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/nonfinite_batches",
                    int(metrics.get("nonfinite_batches", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/fallback_batches",
                    int(metrics.get("fallback_batches", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/unresolved_nonfinite_batches",
                    int(metrics.get("unresolved_nonfinite_batches", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/nonfinite_target_tokens",
                    int(metrics.get("nonfinite_target_tokens", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/unresolved_target_tokens",
                    int(metrics.get("unresolved_target_tokens", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/fallback_hidden_nonfinite_elements",
                    int(metrics.get("fallback_hidden_nonfinite_elements", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/unresolved_hidden_nonfinite_elements",
                    int(metrics.get("unresolved_hidden_nonfinite_elements", 0)),
                    int(step),
                )
                loggers.log_scalar(
                    "eval/unresolved_with_finite_hidden_batches",
                    int(metrics.get("unresolved_with_finite_hidden_batches", 0)),
                    int(step),
                )
                current_eval_loss = float(eval_row["eval_loss"])
                if (
                    bool(args.save_best_eval_checkpoint)
                    and np.isfinite(current_eval_loss)
                    and current_eval_loss < best_eval_loss
                ):
                    best_eval_loss = current_eval_loss
                    best_eval_step = int(step)
                    save_training_checkpoint(
                        out_dir / "best_eval_model.pt",
                        model=model,
                        optimizer=optimizer,
                        verifier=token_verifier,
                        nitp_projector=nitp_projector,
                        step=int(step),
                        tokens_seen=int(tokens_seen),
                        target_tokens_seen=int(target_tokens_seen),
                        compute_tokens_seen=int(compute_tokens_seen),
                        losses=losses,
                        eval_losses=eval_losses,
                        args=args,
                        dataset_summary=dataset_summary,
                        eval_dataset_summary=eval_dataset_summary,
                        model_summary=model_summary,
                        include_optimizer=False,
                        copy_safe_path=out_dir / "copy_best_eval_model.pt",
                    )
                    print(
                        json.dumps(
                            {
                                "event": "saved_best_eval_model",
                                "step": int(best_eval_step),
                                "eval_loss": float(best_eval_loss),
                                "path": str(out_dir / "best_eval_model.pt"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            if int(args.checkpoint_every) > 0 and (
                step % int(args.checkpoint_every) == 0 or step == int(args.steps)
            ):
                save_training_checkpoint(
                    out_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    verifier=token_verifier,
                    nitp_projector=nitp_projector,
                    step=int(step),
                    tokens_seen=int(tokens_seen),
                    target_tokens_seen=int(target_tokens_seen),
                    compute_tokens_seen=int(compute_tokens_seen),
                    losses=losses,
                    eval_losses=eval_losses,
                    args=args,
                    dataset_summary=dataset_summary,
                    eval_dataset_summary=eval_dataset_summary,
                    model_summary=model_summary,
                    include_optimizer=True,
                    copy_safe_path=out_dir / "copy_last.pt",
                )
            if int(args.model_checkpoint_every) > 0 and (
                step % int(args.model_checkpoint_every) == 0 or step == int(args.steps)
            ):
                save_training_checkpoint(
                    out_dir / "last_model.pt",
                    model=model,
                    optimizer=optimizer,
                    verifier=token_verifier,
                    nitp_projector=nitp_projector,
                    step=int(step),
                    tokens_seen=int(tokens_seen),
                    target_tokens_seen=int(target_tokens_seen),
                    compute_tokens_seen=int(compute_tokens_seen),
                    losses=losses,
                    eval_losses=eval_losses,
                    args=args,
                    dataset_summary=dataset_summary,
                    eval_dataset_summary=eval_dataset_summary,
                    model_summary=model_summary,
                    include_optimizer=False,
                    copy_safe_path=out_dir / "copy_last_model.pt",
                )
    finally:
        loggers.close()

    report = {
        "decision": "completed_prefixlm_smoke"
        if int(args.steps) < int(args.accept_min_steps)
        else "needs_efficiency_baseline",
        "accepted": False,
        "target_level": "HRM-Text Data-IO PrefixLM native one-body learning-efficiency gate",
        "dataset": dataset_summary,
        "eval_dataset": eval_dataset_summary,
        "train": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "tokens_seen": int(tokens_seen),
            "target_tokens_seen": int(target_tokens_seen),
            "compute_tokens_seen": int(compute_tokens_seen),
            "train_think_steps": int(args.train_think_steps),
            "lr": float(args.lr),
            "lr_warmup_steps": int(args.lr_warmup_steps),
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "weight_decay": float(args.weight_decay),
            "device": str(device),
            "loss_chunk_size": int(args.loss_chunk_size),
            "token_verifier_loss_weight": float(args.token_verifier_loss_weight),
            "token_verifier_hidden_dim": int(args.token_verifier_hidden_dim),
            "token_verifier_max_targets": int(args.token_verifier_max_targets),
            "token_verifier_freeze_model": bool(args.token_verifier_freeze_model),
            "nitp_loss_weight": float(args.nitp_loss_weight),
            "nitp_hidden_dim": int(args.nitp_hidden_dim),
            "nitp_max_targets": int(args.nitp_max_targets),
            "premature_stop_loss_weight": float(args.premature_stop_loss_weight),
            "premature_stop_token_ids": tuple(int(token_id) for token_id in stop_token_ids),
            "response_start_loss_weight": float(args.response_start_loss_weight),
            "row_balanced_response_loss_weight": float(args.row_balanced_response_loss_weight),
            "amp_dtype": str(args.amp_dtype),
            "matmul_precision": str(args.matmul_precision),
        },
        "model": {
            **model_summary,
        },
        "loss_history": losses,
        "eval_loss_history": eval_losses,
        "initial_logged_loss": losses[0]["loss"] if losses else None,
        "final_logged_loss": losses[-1]["loss"] if losses else None,
        "initial_eval_loss": eval_losses[0]["eval_loss"] if eval_losses else None,
        "final_eval_loss": eval_losses[-1]["eval_loss"] if eval_losses else None,
        "best_eval_loss": float(best_eval_loss) if np.isfinite(best_eval_loss) else None,
        "best_eval_step": int(best_eval_step) if int(best_eval_step) > 0 else None,
        "best_eval_checkpoint": str(out_dir / "best_eval_model.pt")
        if int(best_eval_step) > 0
        else None,
        "plain_language_read": (
            "This run measures how quickly the same body learns to read the "
            "HRM-Text textbook and speak response tokens through its own LM head. "
            "It is the first comparable ingredient for a training-efficiency "
            "claim, but it still needs an HRM-Text baseline curve at the same "
            "data contract."
        ),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not bool(args.no_save_final_checkpoint):
        save_training_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            verifier=token_verifier,
            nitp_projector=nitp_projector,
            step=int(args.steps),
            tokens_seen=int(tokens_seen),
            target_tokens_seen=int(target_tokens_seen),
            compute_tokens_seen=int(compute_tokens_seen),
            losses=losses,
            eval_losses=eval_losses,
            args=args,
            dataset_summary=dataset_summary,
            eval_dataset_summary=eval_dataset_summary,
            model_summary=model_summary,
            include_optimizer=True,
            copy_safe_path=out_dir / "copy_last.pt",
        )
    if int(args.model_checkpoint_every) > 0:
        save_training_checkpoint(
            out_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            verifier=token_verifier,
            nitp_projector=nitp_projector,
            step=int(args.steps),
            tokens_seen=int(tokens_seen),
            target_tokens_seen=int(target_tokens_seen),
            compute_tokens_seen=int(compute_tokens_seen),
            losses=losses,
            eval_losses=eval_losses,
            args=args,
            dataset_summary=dataset_summary,
            eval_dataset_summary=eval_dataset_summary,
            model_summary=model_summary,
            include_optimizer=False,
            copy_safe_path=out_dir / "copy_last_model.pt",
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train native recurrent LM on HRM-Text Data-IO PrefixLM tensors."
    )
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--model-checkpoint-every",
        type=int,
        default=0,
        help=(
            "If >0, additionally save a lightweight model-only checkpoint to "
            "last_model.pt at this interval. Full optimizer checkpoints still "
            "use --checkpoint-every and remain the resume-safe path."
        ),
    )
    parser.add_argument("--no-save-final-checkpoint", action="store_true")
    parser.add_argument(
        "--save-best-eval-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save best_eval_model.pt/copy_best_eval_model.pt whenever eval loss "
            "improves. This preserves the useful checkpoint when longer training "
            "later overfits."
        ),
    )
    parser.add_argument("--accept-min-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--length-bucketed-batches",
        action="store_true",
        help=(
            "Group similarly sized rows into the same shuffled training batches "
            "to reduce padding compute after batch trimming."
        ),
    )
    parser.add_argument("--length-bucket-size-multiplier", type=int, default=64)
    parser.add_argument(
        "--trim-batch-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trim trailing all-padding columns in each batch before the model "
            "forward pass. Disable with --no-trim-batch-to-max-length for exact "
            "fixed-shape debugging."
        ),
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--keep-overlength", action="store_true")
    parser.add_argument("--train-instruction-tokens", action="store_true")
    parser.add_argument("--dry-run-loader", action="store_true")
    parser.add_argument("--eval-sampled-data", default="")
    parser.add_argument("--eval-epoch", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-max-rows", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        choices=MEMORY_EFFICIENT_OPTIMIZERS,
        default="adamw",
        help=(
            "Optimizer for memory-constrained large runs. Use adamw8bit or "
            "paged_adamw8bit after installing bitsandbytes; use "
            "galore_adamw8bit after installing galore-torch."
        ),
    )
    parser.add_argument("--galore-rank", type=int, default=128)
    parser.add_argument("--galore-update-proj-gap", type=int, default=200)
    parser.add_argument("--galore-scale", type=float, default=0.25)
    parser.add_argument("--galore-proj-type", default="std")
    parser.add_argument("--galore-min-dim", type=int, default=128)
    parser.add_argument("--galore-include-embeddings", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bf16", "fp16"),
        default="none",
        help="Use CUDA autocast mixed precision for faster large-model training.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("", "highest", "high", "medium"),
        default="high",
        help="Forwarded to torch.set_float32_matmul_precision for CUDA matmuls.",
    )
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, compute LM head + CE only for supervised target tokens in "
            "chunks of this size. This preserves the PrefixLM loss while "
            "reducing large-vocab memory pressure."
        ),
    )
    parser.add_argument(
        "--loss-kernel",
        choices=LOSS_KERNELS,
        default="torch",
        help=(
            "Cross-entropy implementation. Use liger_fused_linear_ce to avoid "
            "materializing full target-token x vocab logits when liger-kernel is "
            "installed; auto uses Liger on CUDA when available."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--tensorboard-dir", default="")
    parser.add_argument("--aim-repo", default=os.environ.get("QTRM_AIM_REPO", ""))
    parser.add_argument("--aim-experiment", default="")
    parser.add_argument("--aim-run-name", default="")
    parser.add_argument("--aim-description", default="")
    parser.add_argument("--token-verifier-loss-weight", type=float, default=0.0)
    parser.add_argument("--token-verifier-hidden-dim", type=int, default=0)
    parser.add_argument("--token-verifier-max-targets", type=int, default=256)
    parser.add_argument("--token-verifier-freeze-model", action="store_true")
    parser.add_argument(
        "--nitp-loss-weight",
        type=float,
        default=0.0,
        help=(
            "NITP-style auxiliary weight. When >0, a small projector predicts "
            "the stop-grad target token embedding from PrefixLM context hidden."
        ),
    )
    parser.add_argument("--nitp-hidden-dim", type=int, default=0)
    parser.add_argument("--nitp-max-targets", type=int, default=256)
    parser.add_argument("--premature-stop-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--premature-stop-token-ids",
        default="",
        help=(
            "Comma/space separated stop token ids to suppress on non-stop "
            "PrefixLM target positions. This targets greedy generation that "
            "closes immediately with a special end token."
        ),
    )
    parser.add_argument(
        "--response-start-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra CE weight on the first response token after the instruction. "
            "This targets checkpoints that know candidates but close before "
            "writing the first answer token."
        ),
    )
    parser.add_argument(
        "--row-balanced-response-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra CE where each training row contributes one averaged response "
            "loss, instead of letting long responses dominate short direct "
            "answers. This teaches answer start, body, and stop together."
        ),
    )
    parser.add_argument("--model-vocab-size", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--backbone", default="trm_qwen35_3to1")
    parser.add_argument("--encode-backbone", default="")
    parser.add_argument("--think-backbone", default="")
    parser.add_argument("--decode-backbone", default="")
    parser.add_argument("--think-structure", default="trm_dual_z")
    parser.add_argument("--train-think-steps", type=int, default=4)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help=(
            "Checkpoint encode/think/decode stage calls during training to reduce "
            "activation memory at the cost of recomputation."
        ),
    )
    parser.add_argument("--trm-l-cycles", type=int, default=1)
    parser.add_argument("--trm-full-grad-cycles", action="store_true")
    parser.add_argument("--hybrid-layers", type=int, default=4)
    parser.add_argument("--attn-every", type=int, default=4)
    parser.add_argument("--delta-backend", default="official_gated_delta2")
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
    parser.add_argument(
        "--position-embedding-mode",
        choices=("learned", "none", "randomized"),
        default="learned",
    )
    parser.add_argument("--halt-pooling", choices=("last", "mean", "dedicated"), default="last")
    parser.add_argument("--carrier-gate-init", type=float, default=-1.0)
    parser.add_argument("--carrier-state-mode", default="gru")
    parser.add_argument("--trm-recurrent-layerscale-mode", default="none")
    parser.add_argument("--trm-recurrent-layerscale-init", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
