#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qtrm_mm.config import load_config
from qtrm_mm.qtrm_model import QTRMMultimodalModel


def load_rows(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[float] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            features.append([float(value) for value in row["answer_decision_features"]])
            targets.append(float(row["answer_decision_target"]))
    if not features:
        raise ValueError(f"no answer-decision rows found in {path}")
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def load_initial_checkpoint(model: QTRMMultimodalModel, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.get("model", state).items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    model.load_state_dict(compatible, strict=False)


def reset_hidden_fallback(model: QTRMMultimodalModel) -> None:
    if model.answer_decision_head is not None:
        torch.nn.init.zeros_(model.answer_decision_head.weight)
        torch.nn.init.constant_(model.answer_decision_head.bias, -10.0)
    if model.answer_decision_feature_proj is not None:
        torch.nn.init.zeros_(model.answer_decision_feature_proj.weight)
        torch.nn.init.zeros_(model.answer_decision_feature_proj.bias)


def train_feature_head(
    model: QTRMMultimodalModel,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> dict[str, float]:
    if model.answer_decision_feature_head is None:
        raise ValueError("model.answer_decision_feature_head is not enabled")
    positives = float(targets.sum().item())
    negatives = float((1.0 - targets).sum().item())
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.answer_decision_feature_head.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    last_loss = 0.0
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model.answer_decision_feature_head(features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.answer_decision_feature_head.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().item())
    with torch.no_grad():
        probs = torch.sigmoid(model.answer_decision_feature_head(features).squeeze(-1))
        pred = (probs >= 0.5).to(dtype=targets.dtype)
        acc = float((pred == targets).float().mean().item())
    return {
        "loss": last_loss,
        "train_acc": acc,
        "positives": positives,
        "negatives": negatives,
        "pos_weight": float(pos_weight.item()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap QTRM in-model answer-decision feature head from telemetry rows."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    model = QTRMMultimodalModel(cfg.model)
    load_initial_checkpoint(model, args.init_checkpoint)
    reset_hidden_fallback(model)
    features, targets = load_rows(args.train_jsonl)
    metrics = train_feature_head(
        model,
        features,
        targets,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "bootstrap_metrics": metrics}, out_path)
    print(json.dumps({**metrics, "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
