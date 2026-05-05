#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from qtrm_mm.agentic.cognitive_loop import Action
from qtrm_mm.config import load_config
from qtrm_mm.data.jsonl_dataset import JsonlTextVisionDataset, collate_jsonl
from qtrm_mm.qtrm_model import QTRMMultimodalModel
from qtrm_mm.qwen_donor import QwenDonorAdapter
from qtrm_mm.training.train import prepare_donor_batch


def summarize_action_predictions(
    preds: list[int],
    targets: list[int],
) -> dict[str, Any]:
    if len(preds) != len(targets):
        raise ValueError("preds and targets must have the same length")
    total = len(targets)
    correct = sum(int(pred == target) for pred, target in zip(preds, targets))
    per_target: dict[str, dict[str, int | float]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for pred, target in zip(preds, targets):
        target_name = _action_name(target)
        pred_name = _action_name(pred)
        row = per_target.setdefault(target_name, {"total": 0, "correct": 0, "accuracy": 0.0})
        row["total"] = int(row["total"]) + 1
        row["correct"] = int(row["correct"]) + int(pred == target)
        confusion.setdefault(target_name, {})
        confusion[target_name][pred_name] = confusion[target_name].get(pred_name, 0) + 1
    for row in per_target.values():
        row["accuracy"] = float(row["correct"]) / max(1, int(row["total"]))
    return {
        "samples": total,
        "accuracy": float(correct) / max(1, total),
        "per_target": per_target,
        "confusion": confusion,
    }


def _action_name(action_id: int) -> str:
    try:
        return Action.from_id(int(action_id)).value
    except Exception:
        return f"ACTION_{int(action_id)}"


def _jsonl_line_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def evaluate_controller_policy(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    model = QTRMMultimodalModel(cfg.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    donor = QwenDonorAdapter(cfg.donor) if args.use_donor else None
    dataset = JsonlTextVisionDataset(
        [args.data_jsonl],
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.train.seq_len,
        visual_dim=cfg.model.visual_dim,
        max_visual_tokens=cfg.model.max_visual_tokens,
        multimodal=False,
        shuffle_buffer=max(1, int(args.shuffle_buffer)),
        tokenizer_model_id=cfg.donor.model_id,
    )
    batch_size = int(args.batch_size or cfg.train.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_jsonl,
    )

    preds: list[int] = []
    targets: list[int] = []
    max_batches = int(args.max_batches)
    if max_batches <= 0:
        max_batches = max(1, math.ceil(_jsonl_line_count(args.data_jsonl) / max(1, batch_size)))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            model_kwargs = {
                "attention_mask": batch.get("attention_mask"),
            }
            if (
                "controller_signal" in batch
                and str(cfg.model.controller_signal_source).lower() == "external"
            ):
                model_kwargs["controller_signal"] = batch["controller_signal"]
            if donor is not None:
                model_kwargs.update(prepare_donor_batch(donor, batch, return_logits=False))
            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda" and bool(cfg.train.use_amp)),
                dtype=torch.bfloat16,
            ):
                outputs = model(input_ids=batch["input_ids"], **model_kwargs)
            preds.extend(outputs["action_logits"].argmax(dim=-1).detach().cpu().tolist())
            targets.extend(batch["action_target"].detach().cpu().tolist())

    summary = summarize_action_predictions(preds, targets)
    summary.update(
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "data_jsonl": args.data_jsonl,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "config_snapshot": asdict(cfg),
        }
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate QTRM controller trace action policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--use-donor", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_controller_policy(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
