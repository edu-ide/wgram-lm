#!/usr/bin/env python3
"""Train/evaluate Stage101X same-LM-head counterfactual answer attractor.

Unlike W8/W9, this script does not promote a detached feature reader.  The
normal answer path must separate a real world from a minimally imagined
counterfactual world using the same LM head.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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


STAGE101 = load_module(
    ROOT / "scripts" / "570_train_solution_aligned_answer_attractor.py",
    "stage101x_solution_attractor_train",
)

YES = " yes"
NO = " no"


def row_to_choice(row: dict[str, Any], side: str) -> dict[str, Any]:
    side = str(side).lower()
    if side == "original":
        return {
            "id": f"{row.get('id')}_original",
            "prompt": str(row["original_prompt"]),
            "intelligence_answer": str(row["original_answer"]),
            "negative_answers": [str(item) for item in row["original_negative_answers"]],
            "parrot_answer": str(row["original_negative_answers"][0]),
        }
    if side == "counterfactual":
        return {
            "id": f"{row.get('id')}_counterfactual",
            "prompt": str(row["counterfactual_prompt"]),
            "intelligence_answer": str(row["counterfactual_answer"]),
            "negative_answers": [str(item) for item in row["counterfactual_negative_answers"]],
            "parrot_answer": str(row["counterfactual_negative_answers"][0]),
        }
    raise ValueError(f"bad side: {side!r}")


def answer_margin_from_yes_no(yes_minus_no: torch.Tensor, answer: str) -> torch.Tensor:
    answer = str(answer)
    if answer == YES:
        return yes_minus_no.float()
    if answer == NO:
        return -yes_minus_no.float()
    raise ValueError(f"bad answer: {answer!r}")


def counterfactual_gap(
    original_yes_minus_no: torch.Tensor,
    counterfactual_yes_minus_no: torch.Tensor,
    *,
    original_answer: str,
) -> torch.Tensor:
    if str(original_answer) == YES:
        return original_yes_minus_no.float() - counterfactual_yes_minus_no.float()
    if str(original_answer) == NO:
        return counterfactual_yes_minus_no.float() - original_yes_minus_no.float()
    raise ValueError(f"bad original_answer: {original_answer!r}")


def yes_no_logprob_scores(
    model: torch.nn.Module,
    gd_module: Any,
    *,
    prompt: str,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    think_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    yes_mean, _yes_tokens = STAGE101.choice_mean_logprob_tensor(
        model,
        gd_module,
        prompt=str(prompt),
        answer=YES,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(think_steps),
    )
    no_mean, _no_tokens = STAGE101.choice_mean_logprob_tensor(
        model,
        gd_module,
        prompt=str(prompt),
        answer=NO,
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(think_steps),
    )
    return yes_mean, no_mean


def pair_lm_head_loss(
    model: torch.nn.Module,
    gd_module: Any,
    row: dict[str, Any],
    *,
    depth: int,
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    target_margin: float,
    target_gap: float,
    gap_weight: float,
    target_nll_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    original_yes, original_no = yes_no_logprob_scores(
        model,
        gd_module,
        prompt=str(row["original_prompt"]),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(depth),
    )
    counterfactual_yes, counterfactual_no = yes_no_logprob_scores(
        model,
        gd_module,
        prompt=str(row["counterfactual_prompt"]),
        seq_len=int(seq_len),
        byte_offset=int(byte_offset),
        device=device,
        think_steps=int(depth),
    )
    original_yes_minus_no = original_yes.float() - original_no.float()
    counterfactual_yes_minus_no = counterfactual_yes.float() - counterfactual_no.float()
    original_margin = answer_margin_from_yes_no(original_yes_minus_no, str(row["original_answer"]))
    counterfactual_margin = answer_margin_from_yes_no(
        counterfactual_yes_minus_no,
        str(row["counterfactual_answer"]),
    )
    gap = counterfactual_gap(
        original_yes_minus_no,
        counterfactual_yes_minus_no,
        original_answer=str(row["original_answer"]),
    )
    rank_loss = 0.5 * (
        F.softplus(float(target_margin) - original_margin.float())
        + F.softplus(float(target_margin) - counterfactual_margin.float())
    )
    gap_loss = F.softplus(float(target_gap) - gap.float())
    original_target_nll = -(original_yes if str(row["original_answer"]) == YES else original_no)
    counterfactual_target_nll = -(
        counterfactual_yes if str(row["counterfactual_answer"]) == YES else counterfactual_no
    )
    target_nll = 0.5 * (original_target_nll.float() + counterfactual_target_nll.float())
    loss = rank_loss + float(gap_weight) * gap_loss + float(target_nll_weight) * target_nll
    return loss, {
        "id": row.get("id"),
        "pair_feature": row.get("pair_feature"),
        "depth": int(depth),
        "loss": float(loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "gap_loss": float(gap_loss.detach().cpu().item()),
        "target_nll": float(target_nll.detach().cpu().item()),
        "original_margin": float(original_margin.detach().cpu().item()),
        "counterfactual_margin": float(counterfactual_margin.detach().cpu().item()),
        "counterfactual_gap": float(gap.detach().cpu().item()),
        "original_correct": bool(float(original_margin.detach().cpu().item()) > 0.0),
        "counterfactual_correct": bool(float(counterfactual_margin.detach().cpu().item()) > 0.0),
    }


def pair_multi_depth_loss(
    model: torch.nn.Module,
    gd_module: Any,
    row: dict[str, Any],
    *,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    target_gap: float,
    gap_weight: float,
    target_nll_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for depth in depths:
        with STAGE101.make_amp_context(device, amp_dtype):
            loss, metrics = pair_lm_head_loss(
                model,
                gd_module,
                row,
                depth=int(depth),
                seq_len=int(seq_len),
                byte_offset=int(byte_offset),
                device=device,
                target_margin=float(target_margin),
                target_gap=float(target_gap),
                gap_weight=float(gap_weight),
                target_nll_weight=float(target_nll_weight),
            )
        losses.append(loss)
        rows.append(metrics)
    total = torch.stack([loss.float() for loss in losses]).mean()
    last = rows[-1]
    return total, {
        "id": row.get("id"),
        "pair_feature": row.get("pair_feature"),
        "depths": [int(depth) for depth in depths],
        "loss": float(total.detach().cpu().item()),
        "last_original_margin": float(last["original_margin"]),
        "last_counterfactual_margin": float(last["counterfactual_margin"]),
        "last_counterfactual_gap": float(last["counterfactual_gap"]),
        "last_pair_correct": bool(last["original_correct"] and last["counterfactual_correct"]),
    }


def build_pair_report(rows: list[dict[str, Any]], *, split: str, depth: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("pair report requires rows")
    pair_accuracy = sum(
        1 for row in rows if bool(row["original_correct"]) and bool(row["counterfactual_correct"])
    ) / float(len(rows))
    gaps = [float(row["counterfactual_gap"]) for row in rows]
    original_margins = [float(row["original_margin"]) for row in rows]
    counterfactual_margins = [float(row["counterfactual_margin"]) for row in rows]
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_feature.setdefault(str(row["pair_feature"]), []).append(row)
    feature_summary = {
        feature: {
            "rows": len(items),
            "pair_accuracy": sum(
                1
                for item in items
                if bool(item["original_correct"]) and bool(item["counterfactual_correct"])
            )
            / float(len(items)),
            "min_counterfactual_gap": float(min(float(item["counterfactual_gap"]) for item in items)),
        }
        for feature, items in sorted(by_feature.items())
    }
    min_gap = float(min(gaps))
    min_original_margin = float(min(original_margins))
    min_counterfactual_margin = float(min(counterfactual_margins))
    return {
        "split": str(split),
        "depth": int(depth),
        "rows": int(len(rows)),
        "pair_accuracy": float(pair_accuracy),
        "min_original_margin": min_original_margin,
        "min_counterfactual_margin": min_counterfactual_margin,
        "min_counterfactual_gap": min_gap,
        "feature_summary": feature_summary,
        "accepted": bool(
            pair_accuracy == 1.0
            and min_original_margin > 0.0
            and min_counterfactual_margin > 0.0
            and min_gap > 0.0
        ),
    }


@torch.no_grad()
def evaluate_pairs(
    *,
    model: torch.nn.Module,
    gd_module: Any,
    rows: list[dict[str, Any]],
    split: str,
    depths: list[int],
    seq_len: int,
    byte_offset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    target_margin: float,
    target_gap: float,
    gap_weight: float,
    target_nll_weight: float,
) -> dict[str, Any]:
    model.eval()
    by_depth: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    for depth in depths:
        depth_rows: list[dict[str, Any]] = []
        for row in rows:
            with STAGE101.make_amp_context(device, amp_dtype):
                _loss, metrics = pair_lm_head_loss(
                    model,
                    gd_module,
                    row,
                    depth=int(depth),
                    seq_len=int(seq_len),
                    byte_offset=int(byte_offset),
                    device=device,
                    target_margin=float(target_margin),
                    target_gap=float(target_gap),
                    gap_weight=float(gap_weight),
                    target_nll_weight=float(target_nll_weight),
                )
            metrics["split"] = split
            depth_rows.append(metrics)
            detailed_rows.append(metrics)
        by_depth.append(build_pair_report(depth_rows, split=split, depth=int(depth)))
    return {"split": split, "depths": by_depth, "accepted": bool(by_depth and by_depth[-1]["accepted"]), "rows": detailed_rows}


def training_depths_for_step(depths: list[int], *, step: int, single_depth: bool) -> list[int]:
    if not single_depth:
        return [int(depth) for depth in depths]
    return [int(depths[(int(step) - 1) % len(depths)])]


def batch_rows_for_step(rows: list[dict[str, Any]], *, step: int, batch_size: int) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("rows must not be empty")
    size = max(1, int(batch_size))
    start = (int(step) - 1) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(size)]


def build_checkpoint_args(ckpt_args: argparse.Namespace, args: argparse.Namespace) -> dict[str, Any]:
    values = vars(ckpt_args).copy()
    values.update(
        {
            "stage101x_counterfactual_answer_attractor": True,
            "stage101x_source_checkpoint": str(args.checkpoint),
            "stage101x_train_jsonl": str(args.train_jsonl),
            "stage101x_eval_jsonl": str(args.eval_jsonl),
            "stage101x_depths": [int(depth) for depth in args.depths],
            "stage101x_target_margin": float(args.target_margin),
            "stage101x_target_gap": float(args.target_gap),
            "stage101x_gap_weight": float(args.gap_weight),
            "stage101x_target_nll_weight": float(args.target_nll_weight),
            "stage101x_steps": int(args.steps),
            "stage101x_batch_size": int(getattr(args, "batch_size", 1)),
            "stage101x_lr": float(args.lr),
        }
    )
    return values


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args_payload: dict[str, Any],
    step: int,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any],
    history: list[dict[str, Any]],
    loaded: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args_payload,
        "dataset": dict(loaded.get("dataset_summary", {})),
        "model": dict(loaded.get("model_summary", {})),
        "loss_history": history,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "stage101x_counterfactual_answer_attractor": True,
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
    train_rows = STAGE101.load_jsonl(Path(args.train_jsonl))
    eval_rows = STAGE101.load_jsonl(Path(args.eval_jsonl))
    device = torch.device(str(args.device))
    amp_dtype = STAGE101.resolve_amp_dtype(str(args.amp_dtype))
    depth_probe = STAGE101.load_depth_probe_module()
    gd_module = STAGE101.load_gd_module()
    _trainer, _prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=Path(args.checkpoint),
        sampled_data=str(args.sampled_data),
        out_dir=str(out_dir),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    if bool(args.freeze_base):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.train(not bool(args.freeze_base))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    tokenizer_info = dict(loaded.get("dataset_summary", {}).get("tokenizer_info") or {})
    byte_offset = int(args.byte_offset if int(args.byte_offset) >= 0 else tokenizer_info.get("byte_offset", 2))
    seq_len = int(args.seq_len or ckpt_args.seq_len)
    depths = sorted({int(depth) for depth in args.depths})

    eval_kwargs = {
        "model": model,
        "gd_module": gd_module,
        "depths": depths,
        "seq_len": seq_len,
        "byte_offset": byte_offset,
        "device": device,
        "amp_dtype": amp_dtype,
        "target_margin": float(args.target_margin),
        "target_gap": float(args.target_gap),
        "gap_weight": float(args.gap_weight),
        "target_nll_weight": float(args.target_nll_weight),
    }
    eval_before = {
        "train": evaluate_pairs(rows=train_rows, split="train", **eval_kwargs),
        "heldout": evaluate_pairs(rows=eval_rows, split="heldout", **eval_kwargs),
    }

    history: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        batch_rows = batch_rows_for_step(train_rows, step=int(step), batch_size=int(args.batch_size))
        step_depths = training_depths_for_step(
            depths,
            step=int(step),
            single_depth=bool(args.single_depth_per_step),
        )
        optimizer.zero_grad(set_to_none=True)
        model.train(not bool(args.freeze_base))
        row_losses: list[torch.Tensor] = []
        row_metrics: list[dict[str, Any]] = []
        for row in batch_rows:
            row_loss, metrics = pair_multi_depth_loss(
                model,
                gd_module,
                row,
                depths=step_depths,
                seq_len=seq_len,
                byte_offset=byte_offset,
                device=device,
                amp_dtype=amp_dtype,
                target_margin=float(args.target_margin),
                target_gap=float(args.target_gap),
                gap_weight=float(args.gap_weight),
                target_nll_weight=float(args.target_nll_weight),
            )
            row_losses.append(row_loss)
            row_metrics.append(metrics)
        loss = torch.stack([item.float() for item in row_losses]).mean()
        last_metrics = row_metrics[-1]
        metrics = {
            "id": last_metrics.get("id"),
            "pair_feature": last_metrics.get("pair_feature"),
            "depths": step_depths,
            "batch_size": int(len(batch_rows)),
            "loss": float(loss.detach().cpu().item()),
            "batch_pair_accuracy": sum(1 for item in row_metrics if bool(item["last_pair_correct"]))
            / float(len(row_metrics)),
            "min_original_margin": float(min(float(item["last_original_margin"]) for item in row_metrics)),
            "min_counterfactual_margin": float(
                min(float(item["last_counterfactual_margin"]) for item in row_metrics)
            ),
            "min_counterfactual_gap": float(
                min(float(item["last_counterfactual_gap"]) for item in row_metrics)
            ),
            "last_original_margin": float(last_metrics["last_original_margin"]),
            "last_counterfactual_margin": float(last_metrics["last_counterfactual_margin"]),
            "last_counterfactual_gap": float(last_metrics["last_counterfactual_gap"]),
            "last_pair_correct": bool(last_metrics["last_pair_correct"]),
        }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None],
            max_norm=float(args.grad_clip),
        )
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0:
            metrics["step"] = int(step)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            history.append(metrics)

    eval_after = {
        "train": evaluate_pairs(rows=train_rows, split="train", **eval_kwargs),
        "heldout": evaluate_pairs(rows=eval_rows, split="heldout", **eval_kwargs),
    }
    accepted = bool(eval_after["heldout"]["accepted"])
    report = {
        "decision": "stage101x_counterfactual_answer_attractor_train",
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
            "Stage101X tests whether the normal LM-head answer path flips "
            "between a real world and a minimal imagined counterfactual."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args_payload = build_checkpoint_args(ckpt_args, args)
    save_checkpoint(
        out_dir / "last_model.pt",
        model=model,
        optimizer=optimizer,
        args_payload=args_payload,
        step=int(args.steps),
        eval_before=eval_before,
        eval_after=eval_after,
        history=history,
        loaded=loaded,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--train-jsonl",
        default="data/eval/stage101x_counterfactual_answer_attractor_train_probe.jsonl",
    )
    parser.add_argument(
        "--eval-jsonl",
        default="data/eval/stage101x_counterfactual_answer_attractor_heldout_probe.jsonl",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--target-margin", type=float, default=0.25)
    parser.add_argument("--target-gap", type=float, default=0.5)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--target-nll-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=80)
    parser.add_argument("--single-depth-per-step", action="store_true")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--byte-offset", type=int, default=-1)
    return parser


def main() -> None:
    report = run_train(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
