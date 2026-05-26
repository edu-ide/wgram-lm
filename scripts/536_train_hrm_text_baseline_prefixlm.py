#!/usr/bin/env python3
"""Train an official HRM-Text baseline on Data-IO PrefixLM tensors.

This script is the baseline side of the same-contract learning-efficiency
comparison. It keeps the HRM-Text model path intact:

  official V1Dataset -> HierarchicalReasoningModel -> LMHead

The attention backend can be set to SDPA so the same one-body path runs on
machines where FlashAttention-3 kernels are unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
HRM_TEXT_ROOT = REPO_ROOT / "references" / "official" / "hrm-text"
if str(HRM_TEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(HRM_TEXT_ROOT))

from dataset_new import V1Dataset, V1DatasetConfig  # noqa: E402
from models.adam_atan2 import AdamATan2  # noqa: E402
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.common import IGNORE_LABEL_ID, wrap_tensor  # noqa: E402
from models.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars  # noqa: E402
from models.lm_head import LMHead  # noqa: E402
from models.layers import find_multiple  # noqa: E402


def build_official_model(
    *,
    vocab_size: int,
    max_seq_len: int,
    hidden_size: int,
    num_heads: int,
    n_layers: int,
    expansion: float,
    h_cycles: int,
    l_cycles: int,
    half_layers: bool = False,
    bp_min_steps: int = 2,
    bp_max_steps: int = 2,
    bp_warmup_ratio: float = 0.0,
    norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
) -> LMHead:
    config = {
        "vocab_size": int(vocab_size),
        "max_seq_len": int(max_seq_len),
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "num_heads": int(num_heads),
        "expansion": float(expansion),
        "attn_type": "prefixlm",
        "init_type": "lecun_normal",
        "norm_type": "pre",
        "norm_eps": float(norm_eps),
        "pos_emb_type": "rope",
        "rope_theta": float(rope_theta),
        "half_layers": bool(half_layers),
        "H_cycles": int(h_cycles),
        "L_cycles": int(l_cycles),
        "bp_warmup_ratio": float(bp_warmup_ratio),
        "bp_min_steps": int(bp_min_steps),
        "bp_max_steps": int(bp_max_steps),
    }
    return LMHead(HierarchicalReasoningModel(config), config)


def make_single_sequence_batch(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prefix_len: int,
) -> dict[str, torch.Tensor]:
    """Small test helper for the official packed PrefixLM batch shape."""
    total_len = int(input_ids.numel())
    causal_len = total_len - int(prefix_len)
    if causal_len < 0:
        raise ValueError("prefix_len cannot exceed input length")
    return {
        "inputs": input_ids,
        "labels": labels,
        "position_ids": torch.arange(total_len, dtype=torch.long),
        "prefix_lens": torch.tensor([prefix_len], dtype=torch.int32),
        "causal_lens": torch.tensor([causal_len], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, total_len], dtype=torch.int32),
        "total_seqlen": torch.tensor(total_len),
        "numseqs": torch.tensor(1),
        "max_seqlen_prefix": torch.tensor(prefix_len),
        "max_seqlen_causal": torch.tensor(causal_len),
        "max_seqlen_all": torch.tensor(total_len),
    }


def create_official_loader(
    *,
    sampled_data: str | Path,
    batch_max_length: int,
    seed: int,
    target_only: bool,
    drop_last_batch: bool,
    epoch: int = 0,
) -> tuple[DataLoader, Any]:
    dataset = V1Dataset(
        V1DatasetConfig(
            seed=int(seed),
            dataset_path=str(sampled_data),
            drop_last_batch=bool(drop_last_batch),
            target_only=bool(target_only),
            batch_max_length=int(batch_max_length),
            rank=0,
            num_replicas=1,
        )
    )
    dataset._epoch = int(epoch)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return loader, dataset.metadata


def load_dataio_module() -> Any:
    path = REPO_ROOT / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("hrm_text_dataio_for_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_row_fixed_loader(
    *,
    sampled_data: str | Path,
    seq_len: int,
    epoch: int,
    target_only: bool,
    max_rows: int,
    batch_size: int,
) -> tuple[DataLoader, dict[str, Any]]:
    dataio = load_dataio_module()
    dataset = dataio.DataIOSampledPrefixLMDataset(
        sampled_data,
        seq_len=int(seq_len),
        epoch=int(epoch),
        target_only=bool(target_only),
        max_rows=int(max_rows) if int(max_rows) > 0 else None,
        drop_overlength=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=dataio.collate_prefixlm_rows,
        drop_last=False,
    )
    summary = dataset.summary()
    summary["eval_batch_size"] = int(batch_size)
    summary["eval_max_batches"] = 0
    return loader, summary


def reset_loader_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "_epoch"):
        dataset._epoch = int(epoch)


def row_fixed_batch_to_official(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    prefix_lens: list[int] = []
    causal_lens: list[int] = []
    for input_ids, row_labels, attention_mask in zip(
        batch["input_ids"], batch["labels"], batch["attention_mask"]
    ):
        length = int(attention_mask.sum().item())
        if length <= 0:
            continue
        input_row = input_ids[:length].detach().cpu().numpy().astype(np.int32)
        label_row = row_labels[:length].detach().cpu().numpy().astype(np.int32)
        valid_targets = np.flatnonzero(label_row != IGNORE_LABEL_ID)
        if valid_targets.size == 0:
            continue
        prefix_len = int(valid_targets[0]) + 1
        causal_len = int(length) - prefix_len
        if causal_len < 0:
            raise ValueError("row-fixed PrefixLM row has invalid prefix/causal split")
        inputs.append(input_row)
        labels.append(label_row)
        positions.append(np.arange(length, dtype=np.int32))
        prefix_lens.append(prefix_len)
        causal_lens.append(causal_len)
    if not inputs:
        raise ValueError("row-fixed batch produced no valid PrefixLM rows")

    flat_inputs = np.concatenate(inputs, dtype=np.int32)
    flat_labels = np.concatenate(labels, dtype=np.int32)
    flat_positions = np.concatenate(positions, dtype=np.int32)
    prefix = np.asarray(prefix_lens, dtype=np.int32)
    causal = np.asarray(causal_lens, dtype=np.int32)
    tensors, scalars = compute_aux_seq_tensors_scalars(prefix, causal, int(flat_inputs.shape[0]))
    official = {
        "inputs": torch.from_numpy(flat_inputs),
        "labels": torch.from_numpy(flat_labels).long(),
        "position_ids": torch.from_numpy(flat_positions),
        **{name: torch.from_numpy(value) for name, value in tensors.items()},
    }
    return official, scalars


def dataset_summary(
    *,
    sampled_data: str | Path,
    metadata: Any,
    seq_len: int,
    target_only: bool,
) -> dict[str, Any]:
    return {
        "contract": "hrm_text_data_io_prefixlm",
        "sampled_data": str(sampled_data),
        "seq_len": int(seq_len),
        "target_only": bool(target_only),
        "vocab_size": int(metadata.vocab_size),
        "model_vocab_size": int(find_multiple(metadata.vocab_size, 256)),
        "max_seq_len": int(metadata.max_seq_len + 1),
        "effective_max_seq_len": int(metadata.max_seq_len),
        "total_length": int(metadata.total_length),
    }


def move_batch(batch: dict[str, torch.Tensor], info: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {name: tensor.to(device) for name, tensor in batch.items()}
    moved.update({name: wrap_tensor(torch.tensor(value, device="cpu")) for name, value in info.items()})
    return moved


def batch_token_counts(batch: dict[str, torch.Tensor], info: dict[str, int]) -> tuple[int, int]:
    tokens = int(info["total_seqlen"])
    targets = int((batch["labels"] != IGNORE_LABEL_ID).sum().detach().cpu().item())
    return tokens, targets


@torch.no_grad()
def evaluate_prefixlm_loss(
    model: nn.Module,
    loader: Iterable[tuple[dict[str, torch.Tensor], dict[str, int]]],
    *,
    device: torch.device,
    bp_steps: int,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    total_tokens = 0
    batches = 0
    for item in loader:
        if isinstance(item, tuple):
            batch, info = item
        else:
            batch, info = item, {}
        if "input_ids" in batch:
            batch, info = row_fixed_batch_to_official(batch)
        moved = move_batch(batch, info, device)
        _carry, loss, _metrics = model(carry=None, batch=moved, bp_steps=int(bp_steps))
        tokens, targets = batch_token_counts(batch, info)
        total_loss += float(loss.detach().cpu().item()) * float(targets)
        total_targets += targets
        total_tokens += tokens
        batches += 1
        if int(max_batches) > 0 and batches >= int(max_batches):
            break
    if total_targets <= 0:
        raise ValueError("evaluation loader produced no target tokens")
    return {
        "loss": total_loss / float(total_targets),
        "target_tokens": int(total_targets),
        "tokens": int(total_tokens),
        "batches": int(batches),
    }


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam_atan2":
        return AdamATan2(
            model.parameters(),
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            weight_decay=float(args.weight_decay),
            ema=float(args.ema) if float(args.ema) > 0 else None,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )


def build_report(
    *,
    dataset_summary: dict[str, Any],
    eval_dataset_summary: dict[str, Any] | None,
    args: argparse.Namespace,
    losses: list[dict[str, float | int]],
    eval_losses: list[dict[str, float | int]],
    tokens_seen: int,
    target_tokens_seen: int,
) -> dict[str, Any]:
    return {
        "decision": "completed_official_hrm_text_baseline_smoke"
        if int(args.steps) < int(args.accept_min_steps)
        else "needs_efficiency_comparison",
        "accepted": False,
        "target_level": "HRM-Text Data-IO PrefixLM official baseline learning-efficiency gate",
        "dataset": dataset_summary,
        "eval_dataset": eval_dataset_summary,
        "train": {
            "steps": int(args.steps),
            "batch_size": None,
            "batch_max_length": int(args.batch_max_length),
            "seq_len": int(args.seq_len),
            "tokens_seen": int(tokens_seen),
            "target_tokens_seen": int(target_tokens_seen),
            "bp_steps": int(args.bp_steps),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": str(args.device),
            "attention_backend": str(args.attention_backend),
            "optimizer": str(args.optimizer),
        },
        "model": {
            "baseline_family": "official_hrm_text",
            "vocab_size": int(dataset_summary.get("model_vocab_size") or dataset_summary["vocab_size"]),
            "hidden_size": int(args.hidden_size),
            "num_heads": int(args.num_heads),
            "n_layers": int(args.n_layers),
            "expansion": float(args.expansion),
            "h_cycles": int(args.h_cycles),
            "l_cycles": int(args.l_cycles),
            "half_layers": bool(args.half_layers),
        },
        "loss_history": losses,
        "eval_loss_history": eval_losses,
        "initial_logged_loss": losses[0]["loss"] if losses else None,
        "final_logged_loss": losses[-1]["loss"] if losses else None,
        "initial_eval_loss": eval_losses[0]["eval_loss"] if eval_losses else None,
        "final_eval_loss": eval_losses[-1]["eval_loss"] if eval_losses else None,
        "plain_language_read": (
            "This is the official HRM-Text student reading the same textbook as "
            "the candidate: Data-IO PrefixLM rows, recurrent HRM thought, and "
            "the normal LM speaker. It is the baseline curve needed before any "
            "10x learning-efficiency claim."
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HRMTEXT_ATTENTION_BACKEND"] = str(args.attention_backend)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_only = not bool(args.train_instruction_tokens)
    train_loader, metadata = create_official_loader(
        sampled_data=args.sampled_data,
        batch_max_length=int(args.batch_max_length),
        seed=int(args.seed),
        target_only=target_only,
        drop_last_batch=False,
        epoch=int(args.epoch),
    )
    train_dataset_summary = dataset_summary(
        sampled_data=args.sampled_data,
        metadata=metadata,
        seq_len=int(args.seq_len),
        target_only=target_only,
    )

    eval_loader = None
    eval_dataset_summary = None
    if int(args.eval_every) > 0:
        if str(args.eval_protocol) == "row_fixed_v1":
            eval_loader, eval_dataset_summary = create_row_fixed_loader(
                sampled_data=args.eval_sampled_data or args.sampled_data,
                seq_len=int(args.seq_len),
                epoch=int(args.eval_epoch),
                target_only=target_only,
                max_rows=int(args.eval_max_rows),
                batch_size=int(args.eval_batch_size),
            )
        else:
            eval_loader, eval_metadata = create_official_loader(
                sampled_data=args.eval_sampled_data or args.sampled_data,
                batch_max_length=int(args.eval_batch_max_length or args.batch_max_length),
                seed=int(args.seed),
                target_only=target_only,
                drop_last_batch=False,
                epoch=int(args.eval_epoch),
            )
            eval_dataset_summary = dataset_summary(
                sampled_data=args.eval_sampled_data or args.sampled_data,
                metadata=eval_metadata,
                seq_len=int(args.seq_len),
                target_only=target_only,
            )
            eval_dataset_summary["eval_protocol"] = "official_multipack_v1"
            eval_dataset_summary["eval_fingerprint"] = None
            eval_dataset_summary["eval_batch_size"] = None
            eval_dataset_summary["eval_max_batches"] = int(args.eval_max_batches)

    if bool(args.dry_run_loader):
        report = build_report(
            dataset_summary=train_dataset_summary,
            eval_dataset_summary=eval_dataset_summary,
            args=args,
            losses=[],
            eval_losses=[],
            tokens_seen=0,
            target_tokens_seen=0,
        )
        report["decision"] = "dry_run_loader"
        out_path = out_dir / "report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    device = torch.device(str(args.device))
    model = build_official_model(
        vocab_size=int(train_dataset_summary["model_vocab_size"]),
        max_seq_len=int(metadata.max_seq_len),
        hidden_size=int(args.hidden_size),
        num_heads=int(args.num_heads),
        n_layers=int(args.n_layers),
        expansion=float(args.expansion),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        half_layers=bool(args.half_layers),
        bp_min_steps=int(args.bp_steps),
        bp_max_steps=int(args.bp_steps),
        bp_warmup_ratio=0.0,
        norm_eps=float(args.norm_eps),
        rope_theta=float(args.rope_theta),
    ).to(device)
    optimizer = build_optimizer(args, model)

    train_iter = iter(train_loader)
    losses: list[dict[str, float | int]] = []
    eval_losses: list[dict[str, float | int]] = []
    tokens_seen = 0
    target_tokens_seen = 0

    for step in range(1, int(args.steps) + 1):
        try:
            batch, info = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, info = next(train_iter)

        moved = move_batch(batch, info, device)
        model.train()
        _carry, loss, _metrics = model(carry=None, batch=moved, bp_steps=int(args.bp_steps))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        batch_tokens, batch_targets = batch_token_counts(batch, info)
        tokens_seen += batch_tokens
        target_tokens_seen += batch_targets
        loss_value = float(loss.detach().cpu().item())

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            row = {
                "step": int(step),
                "loss": loss_value,
                "tokens_seen": int(tokens_seen),
                "target_tokens_seen": int(target_tokens_seen),
            }
            losses.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        if eval_loader is not None and (
            step == 1 or step % int(args.eval_every) == 0 or step == int(args.steps)
        ):
            if str(args.eval_protocol) == "official_multipack_v1":
                reset_loader_epoch(eval_loader, int(args.eval_epoch))
            metrics = evaluate_prefixlm_loss(
                model,
                eval_loader,
                device=device,
                bp_steps=int(args.bp_steps),
                max_batches=int(args.eval_max_batches),
            )
            eval_row = {
                "step": int(step),
                "tokens_seen": int(tokens_seen),
                "target_tokens_seen": int(target_tokens_seen),
                "eval_loss": float(metrics["loss"]),
                "eval_tokens": int(metrics["tokens"]),
                "eval_target_tokens": int(metrics["target_tokens"]),
            }
            eval_losses.append(eval_row)
            print(json.dumps(eval_row, ensure_ascii=False), flush=True)

    report = build_report(
        dataset_summary=train_dataset_summary,
        eval_dataset_summary=eval_dataset_summary,
        args=args,
        losses=losses,
        eval_losses=eval_losses,
        tokens_seen=tokens_seen,
        target_tokens_seen=target_tokens_seen,
    )
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an official HRM-Text baseline on Data-IO PrefixLM tensors."
    )
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--accept-min-steps", type=int, default=1000)
    parser.add_argument("--batch-max-length", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--train-instruction-tokens", action="store_true")
    parser.add_argument("--dry-run-loader", action="store_true")
    parser.add_argument("--eval-sampled-data", default="")
    parser.add_argument("--eval-epoch", type=int, default=1)
    parser.add_argument("--eval-protocol", default="row_fixed_v1", choices=("row_fixed_v1", "official_multipack_v1"))
    parser.add_argument("--eval-max-rows", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batch-max-length", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention-backend", default="sdpa", choices=("auto", "sdpa", "flash", "fa3"))
    parser.add_argument("--optimizer", default="adam_atan2", choices=("adam_atan2", "adamw"))
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--ema", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--half-layers", action="store_true")
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
