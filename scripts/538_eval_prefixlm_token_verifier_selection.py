#!/usr/bin/env python3
"""Evaluate whether a PrefixLM token verifier improves candidate selection.

This is not a raw language-loss metric. It asks a narrower but more causal
question: when the LM offers top-k response-token candidates, does the verifier
select a better candidate than raw LM top-1?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


IGNORE_LABEL_ID = -100


def load_trainer_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("prefixlm_dataio_trainer_for_selection", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_candidate_selection(
    target_ids: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    verifier_scores: torch.Tensor | None = None,
) -> dict[str, float | int]:
    if target_ids.ndim != 1:
        raise ValueError("target_ids must be 1D")
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be 2D")
    if int(topk_ids.shape[0]) != int(target_ids.shape[0]):
        raise ValueError("topk_ids and target_ids batch dimensions must match")
    total = int(target_ids.numel())
    if total <= 0:
        return {
            "targets": 0,
            "raw_lm_top1_accuracy": 0.0,
            "oracle_topk_accuracy": 0.0,
            "verifier_selected_accuracy": None,
            "verifier_gain": None,
        }
    raw_hits = topk_ids[:, 0].eq(target_ids).sum().item()
    oracle_hits = topk_ids.eq(target_ids.unsqueeze(1)).any(dim=1).sum().item()
    report: dict[str, float | int] = {
        "targets": total,
        "raw_lm_top1_accuracy": float(raw_hits / total),
        "oracle_topk_accuracy": float(oracle_hits / total),
    }
    if verifier_scores is None:
        report["verifier_selected_accuracy"] = None
        report["verifier_gain"] = None
        return report
    if verifier_scores.shape != topk_ids.shape:
        raise ValueError("verifier_scores must have the same shape as topk_ids")
    selected_offsets = verifier_scores.argmax(dim=1)
    selected_ids = topk_ids.gather(1, selected_offsets.unsqueeze(1)).squeeze(1)
    selected_hits = selected_ids.eq(target_ids).sum().item()
    verifier_accuracy = float(selected_hits / total)
    report["verifier_selected_accuracy"] = verifier_accuracy
    report["verifier_gain"] = verifier_accuracy - float(report["raw_lm_top1_accuracy"])
    return report


def merge_metric_sums(rows: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    total = int(sum(int(row["targets"]) for row in rows))
    if total <= 0:
        return {
            "targets": 0,
            "raw_lm_top1_accuracy": 0.0,
            "oracle_topk_accuracy": 0.0,
            "verifier_selected_accuracy": None,
            "verifier_gain": None,
        }
    raw_hits = sum(float(row["raw_lm_top1_accuracy"]) * int(row["targets"]) for row in rows)
    oracle_hits = sum(float(row["oracle_topk_accuracy"]) * int(row["targets"]) for row in rows)
    merged: dict[str, float | int | None] = {
        "targets": total,
        "raw_lm_top1_accuracy": float(raw_hits / total),
        "oracle_topk_accuracy": float(oracle_hits / total),
    }
    verifier_rows = [row for row in rows if row.get("verifier_selected_accuracy") is not None]
    if len(verifier_rows) != len(rows):
        merged["verifier_selected_accuracy"] = None
        merged["verifier_gain"] = None
        return merged
    verifier_hits = sum(
        float(row["verifier_selected_accuracy"]) * int(row["targets"]) for row in verifier_rows
    )
    verifier_accuracy = float(verifier_hits / total)
    merged["verifier_selected_accuracy"] = verifier_accuracy
    merged["verifier_gain"] = verifier_accuracy - float(merged["raw_lm_top1_accuracy"])
    return merged


def checkpoint_args(
    trainer: Any,
    checkpoint: dict[str, Any],
    *,
    sampled_data: str,
    out_dir: str,
) -> argparse.Namespace:
    stored_args = dict(checkpoint.get("args") or {})
    default_args = vars(
        trainer.build_arg_parser().parse_args(
            [
                "--sampled-data",
                str(sampled_data or stored_args.get("sampled_data") or "/tmp/sample"),
                "--out-dir",
                str(out_dir),
            ]
        )
    )
    default_args.update(stored_args)
    if sampled_data:
        default_args["sampled_data"] = sampled_data
    default_args["out_dir"] = out_dir
    return argparse.Namespace(**default_args)


@torch.no_grad()
def evaluate_selection(args: argparse.Namespace) -> dict[str, Any]:
    trainer = load_trainer_module()
    device = torch.device(str(args.device))
    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    model_args = checkpoint_args(
        trainer,
        checkpoint,
        sampled_data=str(args.sampled_data),
        out_dir=str(Path(args.out).parent),
    )
    model_vocab_size = int(args.model_vocab_size or checkpoint.get("model", {}).get("vocab_size") or 0)
    if model_vocab_size <= 0:
        dataset_summary = checkpoint.get("dataset") if isinstance(checkpoint.get("dataset"), dict) else {}
        model_vocab_size = int(dataset_summary.get("model_vocab_size") or dataset_summary.get("vocab_size"))
    model = trainer.build_model(model_args, vocab_size=model_vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    verifier = None
    if "token_verifier_state_dict" in checkpoint:
        verifier_hidden_dim = int(
            args.token_verifier_hidden_dim
            or checkpoint.get("model", {}).get("token_verifier_hidden_dim", 0)
            or 0
        )
        verifier = trainer.PrefixLMTokenVerifier(
            int(model_args.d_model),
            hidden_dim=verifier_hidden_dim,
        ).to(device)
        verifier.load_state_dict(checkpoint["token_verifier_state_dict"])
        verifier.eval()

    dataset = trainer.DataIOSampledPrefixLMDataset(
        str(args.sampled_data or model_args.sampled_data),
        seq_len=int(args.seq_len or model_args.seq_len),
        epoch=int(args.epoch),
        target_only=not bool(args.train_instruction_tokens),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        drop_overlength=not bool(args.keep_overlength),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=trainer.collate_prefixlm_rows,
        drop_last=False,
    )

    rows: list[dict[str, float | int | None]] = []
    seen_targets = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        hidden = model.forward_hidden(input_ids, think_steps=int(args.think_steps or model_args.train_think_steps))
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_labels = labels.reshape(-1)
        target_mask = flat_labels != IGNORE_LABEL_ID
        if not bool(target_mask.any()):
            continue
        target_hidden = flat_hidden[target_mask]
        target_ids = flat_labels[target_mask]
        remaining = int(args.max_targets) - seen_targets if int(args.max_targets) > 0 else int(target_ids.numel())
        if remaining <= 0:
            break
        if int(target_ids.numel()) > remaining:
            target_hidden = target_hidden[:remaining]
            target_ids = target_ids[:remaining]

        logits = model.lm_head(target_hidden)
        topk_ids = logits.topk(k=int(args.candidate_top_k), dim=-1).indices
        verifier_scores = None
        if verifier is not None:
            expanded_hidden = target_hidden.unsqueeze(1).expand(-1, int(args.candidate_top_k), -1)
            candidate_embeddings = model.token_embed(topk_ids.reshape(-1)).reshape(
                int(topk_ids.shape[0]),
                int(topk_ids.shape[1]),
                -1,
            )
            verifier_scores = verifier(
                expanded_hidden.reshape(-1, expanded_hidden.size(-1)),
                candidate_embeddings.reshape(-1, candidate_embeddings.size(-1)),
            ).reshape_as(topk_ids.float())
        row = summarize_candidate_selection(target_ids, topk_ids, verifier_scores=verifier_scores)
        rows.append(row)
        seen_targets += int(row["targets"])
        if int(args.max_batches) > 0 and len(rows) >= int(args.max_batches):
            break

    metrics = merge_metric_sums(rows)
    verifier_gain = metrics.get("verifier_gain")
    accepted = (
        verifier_gain is not None
        and float(verifier_gain) >= float(args.min_verifier_gain)
        and int(metrics["targets"]) >= int(args.min_targets)
    )
    report = {
        "accepted": bool(accepted),
        "claim": "verifier_selected_candidate_beats_raw_lm",
        "checkpoint": str(args.checkpoint),
        "sampled_data": str(args.sampled_data or model_args.sampled_data),
        "candidate_top_k": int(args.candidate_top_k),
        "verifier_present": verifier is not None,
        "min_verifier_gain": float(args.min_verifier_gain),
        "min_targets": int(args.min_targets),
        **metrics,
        "plain_language_read": (
            "This checks whether the verifier's eye picks a better token from "
            "the LM's own candidates than the LM top-1 mouth would pick."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=2048)
    parser.add_argument("--min-targets", type=int, default=128)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--model-vocab-size", type=int, default=0)
    parser.add_argument("--token-verifier-hidden-dim", type=int, default=0)
    parser.add_argument("--train-instruction-tokens", action="store_true")
    parser.add_argument("--keep-overlength", action="store_true")
    parser.add_argument("--min-verifier-gain", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_selection(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
