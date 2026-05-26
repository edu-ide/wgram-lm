#!/usr/bin/env python3
"""Train/evaluate Stage101W9 paired latent feature-difference reader."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


W8 = load_module(ROOT / "scripts" / "597_train_stage101w8_latent_feature_reader.py", "stage101w9_w8_reader")

FEATURE_NAMES = W8.FEATURE_NAMES
FEATURE_CHOICES = W8.FEATURE_CHOICES
LatentFeatureReader = W8.LatentFeatureReader


def world_target_row(pair: dict[str, Any], world: str) -> dict[str, Any]:
    key = "world_a_targets" if str(world).upper() == "A" else "world_b_targets"
    return {"feature_targets": dict(pair[key])}


def _target_index(name: str, value: str) -> int:
    choices = FEATURE_CHOICES[name]
    if value not in choices:
        raise ValueError(f"bad target {name}={value!r}")
    return int(choices.index(value))


def _binary_margin(logits: torch.Tensor, target_index: int) -> torch.Tensor:
    if int(logits.shape[-1]) != 2:
        raise ValueError("W9 currently expects binary feature heads")
    other_index = 1 - int(target_index)
    return logits[:, int(target_index)].squeeze(0) - logits[:, other_index].squeeze(0)


def pairwise_feature_difference_loss(
    reader: LatentFeatureReader,
    *,
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    pair: dict[str, Any],
    device: torch.device,
    target_margin: float,
    world_ce_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    logits_a = reader(hidden_a.float())
    logits_b = reader(hidden_b.float())
    feature = str(pair["pair_feature"])
    positive_world = str(pair["positive_world"]).upper()
    if positive_world not in {"A", "B"}:
        raise ValueError(f"bad positive_world {positive_world!r}")
    negative_world = "B" if positive_world == "A" else "A"
    positive_targets = pair["world_a_targets"] if positive_world == "A" else pair["world_b_targets"]
    negative_targets = pair["world_b_targets"] if positive_world == "A" else pair["world_a_targets"]
    positive_logits = logits_a if positive_world == "A" else logits_b
    negative_logits = logits_b if positive_world == "A" else logits_a

    positive_target_index = _target_index(feature, str(positive_targets[feature]))
    negative_target_index = _target_index(feature, str(negative_targets[feature]))
    positive_margin = _binary_margin(positive_logits[feature].float(), positive_target_index)
    negative_margin = _binary_margin(negative_logits[feature].float(), negative_target_index)
    pair_loss = 0.5 * (
        F.softplus(float(target_margin) - positive_margin.float())
        + F.softplus(float(target_margin) - negative_margin.float())
    )

    world_ce = pair_loss * 0.0
    if float(world_ce_weight) > 0.0:
        world_a_loss, world_a_metrics = W8.feature_reader_loss(
            reader,
            hidden_a.float(),
            world_target_row(pair, "A"),
            device,
        )
        world_b_loss, world_b_metrics = W8.feature_reader_loss(
            reader,
            hidden_b.float(),
            world_target_row(pair, "B"),
            device,
        )
        world_ce = 0.5 * (world_a_loss.float() + world_b_loss.float())
    else:
        _, world_a_metrics = W8.feature_reader_loss(reader, hidden_a.float(), world_target_row(pair, "A"), device)
        _, world_b_metrics = W8.feature_reader_loss(reader, hidden_b.float(), world_target_row(pair, "B"), device)
    total = pair_loss + float(world_ce_weight) * world_ce
    pair_correct = bool(float(positive_margin.detach().cpu().item()) > 0.0 and float(negative_margin.detach().cpu().item()) > 0.0)
    return total, {
        "id": pair.get("id"),
        "pair_feature": feature,
        "positive_world": positive_world,
        "negative_world": negative_world,
        "positive_margin": float(positive_margin.detach().cpu().item()),
        "negative_margin": float(negative_margin.detach().cpu().item()),
        "pair_loss": float(pair_loss.detach().cpu().item()),
        "world_ce_loss": float(world_ce.detach().cpu().item()),
        "pair_correct": pair_correct,
        "world_a_feature_correct": world_a_metrics["feature_correct"],
        "world_b_feature_correct": world_b_metrics["feature_correct"],
        "world_a_feature_margins": world_a_metrics["feature_margins"],
        "world_b_feature_margins": world_b_metrics["feature_margins"],
        "world_a_all_feature_correct": all(bool(v) for v in world_a_metrics["feature_correct"].values()),
        "world_b_all_feature_correct": all(bool(v) for v in world_b_metrics["feature_correct"].values()),
    }


def build_pair_report(rows: list[dict[str, Any]], *, split: str, depth: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("pair report requires rows")
    pair_accuracy = sum(1 for row in rows if bool(row["pair_correct"])) / float(len(rows))
    both_world_accuracy = sum(
        1
        for row in rows
        if bool(row["world_a_all_feature_correct"]) and bool(row["world_b_all_feature_correct"])
    ) / float(len(rows))
    positive_margins = [float(row["positive_margin"]) for row in rows]
    negative_margins = [float(row["negative_margin"]) for row in rows]
    min_pair_margin = float(min(positive_margins + negative_margins))
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_feature.setdefault(str(row["pair_feature"]), []).append(row)
    feature_summary = {
        feature: {
            "rows": len(items),
            "pair_accuracy": sum(1 for item in items if bool(item["pair_correct"])) / float(len(items)),
            "min_pair_margin": float(
                min(
                    min(float(item["positive_margin"]) for item in items),
                    min(float(item["negative_margin"]) for item in items),
                )
            ),
        }
        for feature, items in sorted(by_feature.items())
    }
    return {
        "split": str(split),
        "depth": int(depth),
        "rows": int(len(rows)),
        "pair_accuracy": float(pair_accuracy),
        "both_world_feature_accuracy": float(both_world_accuracy),
        "min_pair_margin": min_pair_margin,
        "feature_summary": feature_summary,
        "accepted": bool(pair_accuracy == 1.0 and both_world_accuracy == 1.0 and min_pair_margin > 0.0),
    }


def _pair_hidden(
    model: torch.nn.Module,
    pair: dict[str, Any],
    *,
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_a, mask_a = W8.build_prompt_tensors(
        prompt=str(pair["world_a_prompt"]),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
    )
    input_b, mask_b = W8.build_prompt_tensors(
        prompt=str(pair["world_b_prompt"]),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
    )
    hidden_a = W8.pooled_prompt_hidden(model, input_a, mask_a, think_steps=int(depth))
    hidden_b = W8.pooled_prompt_hidden(model, input_b, mask_b, think_steps=int(depth))
    return hidden_a, hidden_b


def evaluate_pairs(
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
    target_margin: float,
    world_ce_weight: float,
) -> dict[str, Any]:
    model.eval()
    reader.eval()
    by_depth: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for depth in depths:
            depth_rows: list[dict[str, Any]] = []
            for pair in rows:
                with W8.make_amp_context(device, amp_dtype):
                    hidden_a, hidden_b = _pair_hidden(
                        model,
                        pair,
                        depth=int(depth),
                        seq_len=int(seq_len),
                        byte_offset=int(byte_offset),
                        device=device,
                    )
                    _loss, metrics = pairwise_feature_difference_loss(
                        reader,
                        hidden_a=hidden_a,
                        hidden_b=hidden_b,
                        pair=pair,
                        device=device,
                        target_margin=float(target_margin),
                        world_ce_weight=float(world_ce_weight),
                    )
                metrics["think_steps"] = int(depth)
                metrics["split"] = split
                depth_rows.append(metrics)
                detailed_rows.append(metrics)
            by_depth.append(build_pair_report(depth_rows, split=split, depth=int(depth)))
    return {"split": split, "depths": by_depth, "accepted": bool(by_depth and by_depth[-1]["accepted"]), "rows": detailed_rows}


def training_depths_for_step(depths: list[int], *, step: int, single_depth: bool) -> list[int]:
    return W8.training_depths_for_step(depths, step=step, single_depth=single_depth)


def train_pair_multi_depth_step(
    *,
    model: torch.nn.Module,
    reader: LatentFeatureReader,
    pair: dict[str, Any],
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    world_ce_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses: list[torch.Tensor] = []
    metrics_by_depth: list[dict[str, Any]] = []
    for depth in depths:
        with W8.make_amp_context(device, amp_dtype):
            hidden_a, hidden_b = _pair_hidden(
                model,
                pair,
                depth=int(depth),
                seq_len=int(seq_len),
                byte_offset=int(byte_offset),
                device=device,
            )
            loss, metrics = pairwise_feature_difference_loss(
                reader,
                hidden_a=hidden_a,
                hidden_b=hidden_b,
                pair=pair,
                device=device,
                target_margin=float(target_margin),
                world_ce_weight=float(world_ce_weight),
            )
        losses.append(loss)
        metrics_by_depth.append(metrics)
    total = torch.stack([loss.float() for loss in losses]).mean()
    last = metrics_by_depth[-1]
    return total, {
        "id": pair.get("id"),
        "pair_feature": pair.get("pair_feature"),
        "depths": [int(depth) for depth in depths],
        "loss": float(total.detach().cpu().item()),
        "last_pair_correct": bool(last["pair_correct"]),
        "last_positive_margin": float(last["positive_margin"]),
        "last_negative_margin": float(last["negative_margin"]),
        "per_depth_loss": [float(loss.detach().cpu().item()) for loss in losses],
    }


def load_reader_if_available(reader: LatentFeatureReader, checkpoint: str, device: torch.device) -> bool:
    if not str(checkpoint):
        return False
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("stage101w8_latent_feature_reader_state_dict") or payload.get(
        "stage101w9_paired_feature_reader_state_dict"
    )
    if not isinstance(state, dict):
        return False
    reader.load_state_dict(state, strict=True)
    return True


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = W8.load_jsonl(Path(args.train_jsonl))
    eval_rows = W8.load_jsonl(Path(args.eval_jsonl))
    device = torch.device(str(args.device))
    depth_probe = W8.load_depth_probe_module()
    trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(out_dir),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    reader = LatentFeatureReader(d_model=int(model.d_model)).to(device)
    reader_loaded = load_reader_if_available(reader, str(args.reader_checkpoint), device)
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
    amp_dtype = W8.resolve_amp_dtype(str(args.amp_dtype))
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    depths = sorted({int(depth) for depth in args.depths})

    eval_before = {
        "train": evaluate_pairs(
            model=model,
            reader=reader,
            rows=train_rows,
            split="train",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
            target_margin=float(args.target_margin),
            world_ce_weight=float(args.world_ce_weight),
        ),
        "heldout": evaluate_pairs(
            model=model,
            reader=reader,
            rows=eval_rows,
            split="heldout",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
            target_margin=float(args.target_margin),
            world_ce_weight=float(args.world_ce_weight),
        ),
    }

    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        pair = train_rows[(step - 1) % len(train_rows)]
        step_depths = training_depths_for_step(depths, step=int(step), single_depth=bool(args.single_depth_per_step))
        optimizer.zero_grad(set_to_none=True)
        model.train(not bool(args.freeze_base))
        reader.train()
        loss, metrics = train_pair_multi_depth_step(
            model=model,
            reader=reader,
            pair=pair,
            depths=step_depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
            target_margin=float(args.target_margin),
            world_ce_weight=float(args.world_ce_weight),
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
        "train": evaluate_pairs(
            model=model,
            reader=reader,
            rows=train_rows,
            split="train",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
            target_margin=float(args.target_margin),
            world_ce_weight=float(args.world_ce_weight),
        ),
        "heldout": evaluate_pairs(
            model=model,
            reader=reader,
            rows=eval_rows,
            split="heldout",
            depths=depths,
            seq_len=seq_len,
            byte_offset=byte_offset,
            device=device,
            amp_dtype=amp_dtype,
            target_margin=float(args.target_margin),
            world_ce_weight=float(args.world_ce_weight),
        ),
    }
    accepted = bool(eval_after["heldout"]["accepted"])
    report = {
        "decision": "stage101w9_paired_feature_difference_reader_train",
        "accepted": accepted,
        "checkpoint_in": str(args.checkpoint),
        "checkpoint_out": str(out_dir / "last_model.pt"),
        "reader_checkpoint_in": str(args.reader_checkpoint),
        "reader_loaded": bool(reader_loaded),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "depths": depths,
        "steps": int(args.steps),
        "freeze_base": bool(args.freeze_base),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "plain_language_read": (
            "W9 tests whether paired worlds can pull one causal latent feature "
            "apart before answer permission is trusted."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "step": int(args.steps),
        "model_state_dict": model.state_dict(),
        "stage101w9_paired_feature_reader_state_dict": reader.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "loss_history": history,
        "eval_before": eval_before,
        "eval_after": eval_after,
    }
    torch.save(payload, out_dir / "last_model.pt")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reader-checkpoint", default="")
    parser.add_argument("--train-jsonl", default="data/eval/stage101w9_paired_feature_difference_train_probe.jsonl")
    parser.add_argument("--eval-jsonl", default="data/eval/stage101w9_paired_feature_difference_heldout_probe.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=960)
    parser.add_argument("--reader-lr", type=float, default=1e-3)
    parser.add_argument("--base-lr", type=float, default=1e-6)
    parser.add_argument("--target-margin", type=float, default=1.0)
    parser.add_argument("--world-ce-weight", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=240)
    parser.add_argument("--freeze-base", action="store_true")
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
