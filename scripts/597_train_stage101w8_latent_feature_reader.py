#!/usr/bin/env python3
"""Train/evaluate a Stage101W8 latent feature reader.

The reader is intentionally small: linear heads over the pooled decoder hidden
state. The experiment asks whether the one-body recurrent state contains the
features needed before answer permission:

  source reliability, evidence relevance, detail sufficiency, conflict status.

This is not a paper-grade final architecture yet. It is the next falsifiable
gate after W5-W7 showed that surface answer labels are not enough.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FEATURE_NAMES = [
    "source_reliability",
    "evidence_relevance",
    "detail_sufficiency",
    "conflict_status",
    "answer_permission",
]
FEATURE_CHOICES = {
    "source_reliability": ["trusted", "untrusted"],
    "evidence_relevance": ["relevant", "irrelevant"],
    "detail_sufficiency": ["enough", "missing"],
    "conflict_status": ["clear", "conflict"],
    "answer_permission": ["yes", "no"],
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_depth_probe_module() -> Any:
    return load_module(ROOT / "scripts" / "560_eval_blt_depth_residual_probe.py", "stage101w8_depth_probe")


def load_gd_module() -> Any:
    return load_module(ROOT / "scripts" / "567_eval_blt_generalization_dynamics_probe.py", "stage101w8_gd_lite")


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


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    lowered = str(name).lower()
    if lowered in {"", "none", "fp32", "float32"}:
        return None
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unknown amp dtype: {name}")


def make_amp_context(device: torch.device, amp_dtype: torch.dtype | None) -> Any:
    if amp_dtype is None or str(device.type) != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def encode_tokenizer_free_bytes(text: str, *, byte_offset: int = 2) -> list[int]:
    return [int(value) + int(byte_offset) for value in text.encode("utf-8")]


def build_prompt_tensors(
    *,
    prompt: str,
    seq_len: int,
    byte_offset: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = encode_tokenizer_free_bytes(str(prompt), byte_offset=int(byte_offset))
    if not ids:
        raise ValueError("prompt encodes to no tokens")
    if len(ids) > int(seq_len):
        raise ValueError(f"prompt exceeds seq_len={seq_len}: {len(ids)}")
    pad_len = int(seq_len) - len(ids)
    input_ids = ids + [0] * pad_len
    attention_mask = [1] * len(ids) + [0] * pad_len
    return (
        torch.tensor([input_ids], dtype=torch.long, device=device),
        torch.tensor([attention_mask], dtype=torch.long, device=device),
    )


def feature_target_indices(row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    raw_targets = row.get("feature_targets")
    if not isinstance(raw_targets, dict):
        raise ValueError(f"row lacks feature_targets: {row.get('id')}")
    targets: dict[str, torch.Tensor] = {}
    for name in FEATURE_NAMES:
        value = str(raw_targets[name])
        choices = FEATURE_CHOICES[name]
        if value not in choices:
            raise ValueError(f"bad feature target {name}={value!r}")
        targets[name] = torch.tensor([choices.index(value)], dtype=torch.long, device=device)
    return targets


def feature_class_weights(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for name in FEATURE_NAMES:
        choices = FEATURE_CHOICES[name]
        counts = {choice: 0 for choice in choices}
        for row in rows:
            raw_targets = row.get("feature_targets")
            if isinstance(raw_targets, dict):
                value = str(raw_targets.get(name, ""))
                if value in counts:
                    counts[value] += 1
        total = float(sum(counts.values()))
        values: list[float] = []
        for choice in choices:
            count = max(1, int(counts[choice]))
            values.append(total / float(len(choices) * count))
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        weights[name] = tensor / tensor.mean().clamp_min(1e-6)
    return weights


class LatentFeatureReader(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(int(d_model), len(FEATURE_CHOICES[name]))
                for name in FEATURE_NAMES
            }
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(hidden) for name, head in self.heads.items()}


def pooled_prompt_hidden(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    think_steps: int,
) -> torch.Tensor:
    _logits, hidden = model.forward_logits_and_decoder_hidden(
        input_ids,
        attention_mask,
        think_steps=int(think_steps),
    )
    length = min(int(hidden.shape[1]), int(attention_mask.shape[1]))
    hidden = hidden[:, :length]
    mask = attention_mask[:, :length].to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def feature_reader_loss(
    reader: LatentFeatureReader,
    hidden: torch.Tensor,
    row: dict[str, Any],
    device: torch.device,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    logits = reader(hidden)
    targets = feature_target_indices(row, device)
    losses: dict[str, float] = {}
    correct: dict[str, bool] = {}
    margins: dict[str, float] = {}
    loss_terms: list[torch.Tensor] = []
    for name in FEATURE_NAMES:
        target = targets[name]
        item_logits = logits[name].float()
        weight = None if class_weights is None else class_weights.get(name)
        loss = F.cross_entropy(item_logits, target, weight=weight)
        loss_terms.append(loss)
        target_index = int(target.detach().cpu().item())
        predicted = int(item_logits.argmax(dim=-1).detach().cpu().item())
        other_indices = [index for index in range(item_logits.shape[-1]) if index != target_index]
        other_max = item_logits[:, other_indices].max()
        margin = item_logits[:, target_index].squeeze(0) - other_max
        losses[name] = float(loss.detach().cpu().item())
        correct[name] = bool(predicted == target_index)
        margins[name] = float(margin.detach().cpu().item())
    total = torch.stack(loss_terms).mean()
    return total, {
        "feature_losses": losses,
        "feature_correct": correct,
        "feature_margins": margins,
    }


def build_feature_report(rows: list[dict[str, Any]], *, split: str, depth: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("feature report requires rows")
    feature_accuracy: dict[str, float] = {}
    feature_mean_margin: dict[str, float] = {}
    all_margins: list[float] = []
    for name in FEATURE_NAMES:
        correctness = [bool(row["feature_correct"][name]) for row in rows]
        margins = [float(row["feature_margins"][name]) for row in rows]
        feature_accuracy[name] = float(sum(1 for ok in correctness if ok) / float(len(correctness)))
        feature_mean_margin[name] = float(sum(margins) / float(len(margins)))
        all_margins.extend(margins)
    all_feature_ok = [
        all(bool(row["feature_correct"][name]) for name in FEATURE_NAMES)
        for row in rows
    ]
    min_margin = float(min(all_margins))
    accepted = bool(all(all_feature_ok) and min_margin > 0.0)
    return {
        "split": str(split),
        "depth": int(depth),
        "rows": int(len(rows)),
        "feature_accuracy": feature_accuracy,
        "feature_mean_margin": feature_mean_margin,
        "all_feature_accuracy": float(sum(1 for ok in all_feature_ok if ok) / float(len(all_feature_ok))),
        "min_feature_margin": min_margin,
        "accepted": accepted,
    }


def evaluate_feature_rows(
    *,
    model: torch.nn.Module,
    reader: LatentFeatureReader,
    rows: list[dict[str, Any]],
    split: str,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    reader.eval()
    by_depth: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for depth in depths:
            depth_rows: list[dict[str, Any]] = []
            for row in rows:
                input_ids, attention_mask = build_prompt_tensors(
                    prompt=str(row["prompt"]),
                    seq_len=int(seq_len),
                    byte_offset=int(byte_offset),
                    device=device,
                )
                with make_amp_context(device, amp_dtype):
                    hidden = pooled_prompt_hidden(model, input_ids, attention_mask, think_steps=int(depth))
                    _loss, metrics = feature_reader_loss(reader, hidden.float(), row, device)
                out_row = {
                    "id": row.get("id"),
                    "task": row.get("task"),
                    "split": split,
                    "think_steps": int(depth),
                    "feature_correct": metrics["feature_correct"],
                    "feature_margins": metrics["feature_margins"],
                }
                depth_rows.append(out_row)
                detailed_rows.append(out_row)
            by_depth.append(build_feature_report(depth_rows, split=split, depth=int(depth)))
    return {
        "split": str(split),
        "depths": by_depth,
        "accepted": bool(by_depth and by_depth[-1]["accepted"]),
        "rows": detailed_rows,
    }


def train_one_step(
    *,
    model: torch.nn.Module,
    reader: LatentFeatureReader,
    row: dict[str, Any],
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    input_ids, attention_mask = build_prompt_tensors(
        prompt=str(row["prompt"]),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
    )
    with make_amp_context(device, amp_dtype):
        hidden = pooled_prompt_hidden(model, input_ids, attention_mask, think_steps=int(depth))
        loss, metrics = feature_reader_loss(reader, hidden.float(), row, device, class_weights=class_weights)
    metrics.update({"id": row.get("id"), "depth": int(depth), "loss": float(loss.detach().cpu().item())})
    return loss, metrics


def training_depths_for_step(depths: list[int], *, step: int, single_depth: bool) -> list[int]:
    ordered = [int(depth) for depth in depths]
    if not ordered:
        raise ValueError("depths cannot be empty")
    if not bool(single_depth):
        return ordered
    return [ordered[(int(step) - 1) % len(ordered)]]


def train_multi_depth_step(
    *,
    model: torch.nn.Module,
    reader: LatentFeatureReader,
    row: dict[str, Any],
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses: list[torch.Tensor] = []
    metrics_by_depth: list[dict[str, Any]] = []
    for depth in depths:
        loss, metrics = train_one_step(
            model=model,
            reader=reader,
            row=row,
            depth=int(depth),
            seq_len=int(seq_len),
            byte_offset=int(byte_offset),
            device=device,
            amp_dtype=amp_dtype,
            class_weights=class_weights,
        )
        losses.append(loss)
        metrics_by_depth.append(metrics)
    total = torch.stack([loss.float() for loss in losses]).mean()
    last = metrics_by_depth[-1]
    return total, {
        "id": row.get("id"),
        "depths": [int(depth) for depth in depths],
        "loss": float(total.detach().cpu().item()),
        "last_depth_feature_correct": last["feature_correct"],
        "last_depth_feature_margins": last["feature_margins"],
        "per_depth_loss": [float(loss.detach().cpu().item()) for loss in losses],
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    reader: LatentFeatureReader,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    train_history: list[dict[str, Any]],
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "stage101w8_latent_feature_reader_state_dict": reader.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "loss_history": train_history,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "This checkpoint adds a small latent feature reader over decoder hidden state. "
            "It is a diagnostic branch until the answer head is proven to depend on the features."
        ),
    }
    tmp = path.with_name(f".{path.name}.tmp.{int(time.time())}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(Path(args.train_jsonl))
    eval_rows = load_jsonl(Path(args.eval_jsonl))
    device = torch.device(str(args.device))
    depth_probe = load_depth_probe_module()
    trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(out_dir),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    reader = LatentFeatureReader(d_model=int(model.d_model)).to(device)
    if bool(args.freeze_base):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.train(not bool(args.freeze_base))
    reader.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": float(args.base_lr)},
            {"params": reader.parameters(), "lr": float(args.reader_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    amp_dtype = resolve_amp_dtype(str(args.amp_dtype))
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    depths = sorted({int(depth) for depth in args.depths})
    class_weights = feature_class_weights(train_rows, device) if bool(args.class_weighted_loss) else None

    eval_before = {
        "train": evaluate_feature_rows(
            model=model,
            reader=reader,
            rows=train_rows,
            split="train",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
        ),
        "heldout": evaluate_feature_rows(
            model=model,
            reader=reader,
            rows=eval_rows,
            split="heldout",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
        ),
    }

    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        row = train_rows[(step - 1) % len(train_rows)]
        step_depths = training_depths_for_step(
            depths,
            step=int(step),
            single_depth=bool(args.single_depth_per_step),
        )
        optimizer.zero_grad(set_to_none=True)
        model.train(not bool(args.freeze_base))
        reader.train()
        loss, metrics = train_multi_depth_step(
            model=model,
            reader=reader,
            row=row,
            depths=step_depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
            class_weights=class_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
            max_norm=float(args.grad_clip),
        )
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0:
            metrics["step"] = int(step)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            history.append(metrics)

    eval_after = {
        "train": evaluate_feature_rows(
            model=model,
            reader=reader,
            rows=train_rows,
            split="train",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
        ),
        "heldout": evaluate_feature_rows(
            model=model,
            reader=reader,
            rows=eval_rows,
            split="heldout",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
        ),
    }
    accepted = bool(eval_after["heldout"]["accepted"])
    report = {
        "decision": "stage101w8_latent_feature_reader_train",
        "accepted": accepted,
        "checkpoint_in": str(args.checkpoint),
        "checkpoint_out": str(out_dir / "last_model.pt"),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "depths": depths,
        "steps": int(args.steps),
        "freeze_base": bool(args.freeze_base),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "W8 tests whether hidden state can expose the features needed to "
            "decide answer permission. Passing feature heads alone is not yet "
            "a full answer-generation claim."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_checkpoint(
        out_dir / "last_model.pt",
        model=model,
        reader=reader,
        optimizer=optimizer,
        step=int(args.steps),
        args=args,
        train_history=history,
        eval_before=eval_before,
        eval_after=eval_after,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-jsonl", default="data/eval/stage101w8_latent_feature_reader_train_probe.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/stage101w8_latent_feature_reader_heldout_probe.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--reader-lr", type=float, default=2e-3)
    parser.add_argument("--base-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--class-weighted-loss", action="store_true")
    parser.add_argument("--single-depth-per-step", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
