#!/usr/bin/env python3
"""Train a QTRM-native checkpoint on public-style MCQ final-token answers.

This is a diagnostic M7A trainer, not a public benchmark claim. It avoids
sequence CE over the whole prompt so the model is not rewarded for reproducing
``Options:\nA. ...`` text. Only the next token after the answer prompt is
trained, using the same option-letter token mass used by the public MCQ scorer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


OPTION_LETTERS = "ABCDEFGHIJ"


def load_mcq_eval_module():
    path = Path(__file__).with_name("384_eval_qtrm_native_public_mcq.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_public_mcq_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_answer(value: object) -> str:
    text = str(value).strip().upper()
    if text in OPTION_LETTERS:
        return text
    return ""


def option_count(row: dict[str, Any]) -> int:
    options = row.get("options", [])
    if isinstance(options, list) and options:
        return min(len(options), len(OPTION_LETTERS))
    return len(OPTION_LETTERS)


def single_token_option_ids(tokenizer, letter: str) -> tuple[int, ...]:
    ids: list[int] = []
    for variant in (letter, f" {letter}", f"\n{letter}"):
        encoded = tokenizer.encode(variant)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    return tuple(sorted(set(ids)))


def option_token_map(tokenizer, count: int) -> dict[str, tuple[int, ...]]:
    return {
        OPTION_LETTERS[index]: single_token_option_ids(tokenizer, OPTION_LETTERS[index])
        for index in range(int(count))
    }


def rendered_option_ids(tokenizer, letter: str, rendering: str) -> tuple[int, ...]:
    rendering = str(rendering)
    if rendering == "mass":
        return single_token_option_ids(tokenizer, letter)
    if rendering == "plain":
        text = str(letter)
    elif rendering == "space":
        text = f" {letter}"
    elif rendering == "newline":
        text = f"\n{letter}"
    else:
        raise ValueError(f"unsupported target rendering: {rendering}")
    encoded = tokenizer.encode(text)
    if len(encoded) == 1:
        return (int(encoded[0]),)
    return ()


def option_score(log_probs: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    if not token_ids:
        return torch.tensor(-math.inf, dtype=log_probs.dtype, device=log_probs.device)
    ids = torch.tensor(tuple(int(token_id) for token_id in token_ids), dtype=torch.long, device=log_probs.device)
    return torch.logsumexp(log_probs.index_select(dim=0, index=ids), dim=0)


def parse_depths(text: str) -> tuple[int, ...]:
    depths: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        depths.append(int(item))
    return tuple(depths)


def row_final_logits(model, tokenizer, row: dict[str, Any], *, seq_len: int, think_steps: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer.encode(str(row["qtrm_prompt"]))
    if not encoded:
        encoded = [0]
    input_ids = torch.tensor([encoded[-int(seq_len) :]], dtype=torch.long, device=device)
    logits = model(input_ids, think_steps=int(think_steps))
    return logits[0, -1, :].float()


def gold_option_score_for_row(
    model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    think_steps: int,
    device: torch.device,
    target_rendering: str,
) -> torch.Tensor:
    logits = row_final_logits(
        model,
        tokenizer,
        row,
        seq_len=int(seq_len),
        think_steps=int(think_steps),
        device=device,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    gold = normalize_answer(row["answer"])
    gold_ids = rendered_option_ids(tokenizer, gold, str(target_rendering))
    if not gold_ids and str(target_rendering) != "mass":
        gold_ids = option_token_map(tokenizer, option_count(row)).get(gold, ())
    if not gold_ids:
        raise ValueError(f"gold option {gold!r} has no single-token rendering")
    return option_score(log_probs, gold_ids)


def depth_gain_loss_for_row(
    model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    full_think_steps: int,
    shallow_depths: Sequence[int],
    device: torch.device,
    target_rendering: str,
    margin: float,
) -> torch.Tensor:
    if not shallow_depths:
        return torch.zeros((), device=device)
    full_score = gold_option_score_for_row(
        model,
        tokenizer,
        row,
        seq_len=int(seq_len),
        think_steps=int(full_think_steps),
        device=device,
        target_rendering=str(target_rendering),
    )
    shallow_scores = [
        gold_option_score_for_row(
            model,
            tokenizer,
            row,
            seq_len=int(seq_len),
            think_steps=int(depth),
            device=device,
            target_rendering=str(target_rendering),
        ).detach()
        for depth in shallow_depths
    ]
    best_shallow = torch.stack(shallow_scores).max()
    return F.relu(float(margin) - full_score + best_shallow)


def mcq_loss_for_row(
    model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    think_steps: int,
    device: torch.device,
    margin_weight: float,
    margin: float,
    target_rendering: str,
) -> torch.Tensor:
    logits = row_final_logits(
        model,
        tokenizer,
        row,
        seq_len=int(seq_len),
        think_steps=int(think_steps),
        device=device,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    count = option_count(row)
    token_map = option_token_map(tokenizer, count)
    gold = normalize_answer(row["answer"])
    gold_ids = rendered_option_ids(tokenizer, gold, str(target_rendering))
    if not gold_ids and str(target_rendering) != "mass":
        gold_ids = token_map.get(gold, ())
    if not gold_ids:
        raise ValueError(f"gold option {gold!r} has no single-token rendering")
    gold_score = option_score(log_probs, gold_ids)
    loss = -gold_score
    if float(margin_weight) > 0.0:
        wrong_scores = [
            option_score(log_probs, ids)
            for letter, ids in token_map.items()
            if letter != gold and ids
        ]
        if wrong_scores:
            max_wrong = torch.stack(wrong_scores).max()
            loss = loss + float(margin_weight) * F.relu(float(margin) - gold_score + max_wrong)
    return loss


def option_distribution_log_probs(
    model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    think_steps: int,
    device: torch.device,
) -> torch.Tensor:
    logits = row_final_logits(
        model,
        tokenizer,
        row,
        seq_len=int(seq_len),
        think_steps=int(think_steps),
        device=device,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    token_map = option_token_map(tokenizer, option_count(row))
    scores = [
        option_score(log_probs, ids)
        for ids in token_map.values()
        if ids
    ]
    if not scores:
        raise ValueError("row has no scorable option tokens")
    return torch.log_softmax(torch.stack(scores), dim=0)


def multi_depth_ce_loss_for_row(
    model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    depths: Sequence[int],
    device: torch.device,
    margin_weight: float,
    margin: float,
    target_rendering: str,
) -> torch.Tensor:
    if not depths:
        return torch.zeros((), device=device)
    losses = [
        mcq_loss_for_row(
            model,
            tokenizer,
            row,
            seq_len=int(seq_len),
            think_steps=int(depth),
            device=device,
            margin_weight=float(margin_weight),
            margin=float(margin),
            target_rendering=str(target_rendering),
        )
        for depth in depths
    ]
    return torch.stack(losses).mean()


def trajectory_kl_loss_for_row(
    model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    anchor_depth: int,
    compare_depths: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    if not compare_depths:
        return torch.zeros((), device=device)
    anchor_log_probs = option_distribution_log_probs(
        model,
        tokenizer,
        row,
        seq_len=int(seq_len),
        think_steps=int(anchor_depth),
        device=device,
    )
    anchor_probs = anchor_log_probs.detach().exp()
    losses: list[torch.Tensor] = []
    for depth in compare_depths:
        if int(depth) == int(anchor_depth):
            continue
        current_log_probs = option_distribution_log_probs(
            model,
            tokenizer,
            row,
            seq_len=int(seq_len),
            think_steps=int(depth),
            device=device,
        )
        losses.append(torch.sum(anchor_probs * (anchor_log_probs.detach() - current_log_probs)))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def preserve_option_kl_loss(
    model,
    base_model,
    tokenizer,
    row: dict[str, Any],
    *,
    seq_len: int,
    think_steps: int,
    device: torch.device,
) -> torch.Tensor:
    current_log_probs = option_distribution_log_probs(
        model,
        tokenizer,
        row,
        seq_len=int(seq_len),
        think_steps=int(think_steps),
        device=device,
    )
    with torch.no_grad():
        base_log_probs = option_distribution_log_probs(
            base_model,
            tokenizer,
            row,
            seq_len=int(seq_len),
            think_steps=int(think_steps),
            device=device,
        )
    base_probs = base_log_probs.exp()
    return torch.sum(base_probs * (base_log_probs - current_log_probs))


@torch.no_grad()
def score_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    seq_len: int,
    think_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    hits = 0
    pred_hist: dict[str, int] = {}
    for row in rows:
        logits = row_final_logits(
            model,
            tokenizer,
            row,
            seq_len=int(seq_len),
            think_steps=int(think_steps),
            device=device,
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        token_map = option_token_map(tokenizer, option_count(row))
        scores = {
            letter: float(option_score(log_probs, ids).detach().cpu())
            for letter, ids in token_map.items()
            if ids
        }
        pred = max(scores.items(), key=lambda item: item[1])[0] if scores else ""
        gold = normalize_answer(row["answer"])
        hits += int(pred == gold)
        pred_hist[pred or "<empty>"] = pred_hist.get(pred or "<empty>", 0) + 1
    cases = len(rows)
    max_pred_fraction = max(pred_hist.values()) / max(1, cases) if pred_hist else 1.0
    return {
        "hits": hits,
        "cases": cases,
        "accuracy": float(hits / max(1, cases)),
        "pred_answer_histogram": dict(sorted(pred_hist.items())),
        "max_pred_fraction": float(max_pred_fraction),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    mcq_eval = load_mcq_eval_module()
    checkpoint = torch.load(str(args.init_checkpoint), map_location="cpu")
    load_args = argparse.Namespace(
        checkpoint=str(args.init_checkpoint),
        device=str(args.device),
        out_dir=str(args.out_dir),
        think_steps=int(args.think_steps),
        max_new_chars=1,
    )
    _eval_module, eval_args, tokenizer, model, device = mcq_eval.load_model_bundle(load_args)
    rows = mcq_eval.load_suite(str(args.train_jsonl), max_cases=int(args.max_train_cases))
    preserve_rows = (
        mcq_eval.load_suite(str(args.preserve_jsonl), max_cases=int(args.max_preserve_cases))
        if str(args.preserve_jsonl)
        else []
    )
    eval_rows = mcq_eval.load_suite(str(args.eval_jsonl), max_cases=int(args.max_eval_cases)) if str(args.eval_jsonl) else []
    base_model = None
    if float(args.preserve_kl_weight) > 0.0 and preserve_rows:
        _base_eval_module, _base_eval_args, _base_tokenizer, base_model, _base_device = mcq_eval.load_model_bundle(load_args)
        base_model.eval()
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
    trainable_pattern = re.compile(str(args.trainable_name_regex))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(trainable_pattern.search(name)))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"no trainable parameters matched {args.trainable_name_regex!r}")
    trainable_param_count = int(sum(parameter.numel() for parameter in trainable_parameters))
    total_param_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    initial_eval = score_rows(
        model,
        tokenizer,
        eval_rows or rows[: min(128, len(rows))],
        seq_len=int(eval_args.seq_len),
        think_steps=int(args.think_steps),
        device=device,
    )
    model.train()
    last_loss = 0.0
    for step in range(1, int(args.steps) + 1):
        batch = random.choices(rows, k=max(1, int(args.batch_size)))
        optimizer.zero_grad(set_to_none=True)
        losses = [
            mcq_loss_for_row(
                model,
                tokenizer,
                row,
                seq_len=int(eval_args.seq_len),
                think_steps=int(args.think_steps),
                device=device,
                margin_weight=float(args.margin_weight),
                margin=float(args.margin),
                target_rendering=str(args.target_rendering),
            )
            for row in batch
        ]
        loss = torch.stack(losses).mean()
        if float(args.multi_depth_ce_weight) > 0.0:
            multi_depths = parse_depths(str(args.multi_depth_ce_depths))
            multi_depth_losses = [
                multi_depth_ce_loss_for_row(
                    model,
                    tokenizer,
                    row,
                    seq_len=int(eval_args.seq_len),
                    depths=multi_depths,
                    device=device,
                    margin_weight=float(args.margin_weight),
                    margin=float(args.margin),
                    target_rendering=str(args.target_rendering),
                )
                for row in batch
            ]
            loss = loss + float(args.multi_depth_ce_weight) * torch.stack(multi_depth_losses).mean()
        if float(args.depth_gain_weight) > 0.0:
            shallow_depths = parse_depths(str(args.depth_gain_shallow_depths))
            depth_losses = [
                depth_gain_loss_for_row(
                    model,
                    tokenizer,
                    row,
                    seq_len=int(eval_args.seq_len),
                    full_think_steps=int(args.think_steps),
                    shallow_depths=shallow_depths,
                    device=device,
                    target_rendering=str(args.target_rendering),
                    margin=float(args.depth_gain_margin),
                )
                for row in batch
            ]
            loss = loss + float(args.depth_gain_weight) * torch.stack(depth_losses).mean()
        if float(args.trajectory_kl_weight) > 0.0:
            compare_depths = parse_depths(str(args.trajectory_kl_compare_depths))
            trajectory_losses = [
                trajectory_kl_loss_for_row(
                    model,
                    tokenizer,
                    row,
                    seq_len=int(eval_args.seq_len),
                    anchor_depth=int(args.trajectory_kl_anchor_depth),
                    compare_depths=compare_depths,
                    device=device,
                )
                for row in batch
            ]
            loss = loss + float(args.trajectory_kl_weight) * torch.stack(trajectory_losses).mean()
        if base_model is not None:
            preserve_batch = random.choices(preserve_rows, k=max(1, int(args.preserve_batch_size)))
            kl_losses = [
                preserve_option_kl_loss(
                    model,
                    base_model,
                    tokenizer,
                    row,
                    seq_len=int(eval_args.seq_len),
                    think_steps=int(args.think_steps),
                    device=device,
                )
                for row in preserve_batch
            ]
            loss = loss + float(args.preserve_kl_weight) * torch.stack(kl_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if int(args.log_every) > 0 and (step == 1 or step % int(args.log_every) == 0):
            print(json.dumps({"step": step, "loss": last_loss}, ensure_ascii=False), flush=True)
    model.eval()
    final_eval = score_rows(
        model,
        tokenizer,
        eval_rows or rows[: min(128, len(rows))],
        seq_len=int(eval_args.seq_len),
        think_steps=int(args.think_steps),
        device=device,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "complete",
        "decision": "trained_qtrm_native_public_mcq_final_token",
        "init_checkpoint": str(args.init_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "preserve_jsonl": str(args.preserve_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "think_steps": int(args.think_steps),
        "margin_weight": float(args.margin_weight),
        "margin": float(args.margin),
        "preserve_kl_weight": float(args.preserve_kl_weight),
        "preserve_batch_size": int(args.preserve_batch_size),
        "multi_depth_ce_weight": float(args.multi_depth_ce_weight),
        "multi_depth_ce_depths": str(args.multi_depth_ce_depths),
        "depth_gain_weight": float(args.depth_gain_weight),
        "depth_gain_margin": float(args.depth_gain_margin),
        "depth_gain_shallow_depths": str(args.depth_gain_shallow_depths),
        "trajectory_kl_weight": float(args.trajectory_kl_weight),
        "trajectory_kl_anchor_depth": int(args.trajectory_kl_anchor_depth),
        "trajectory_kl_compare_depths": str(args.trajectory_kl_compare_depths),
        "trainable_name_regex": str(args.trainable_name_regex),
        "trainable_params": trainable_param_count,
        "total_params": total_param_count,
        "target_rendering": str(args.target_rendering),
        "last_loss": last_loss,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "evaluation_note": (
            "initial_eval/final_eval score next-token option log-probability. "
            "They are diagnostics only; promotion still requires strict greedy "
            "generation gates such as scripts/398 and scripts/402."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(eval_args),
            "report": report,
            "tokenizer": checkpoint.get("tokenizer", {}),
        },
        out_dir / "last.pt",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable-name-regex", default=".*")
    parser.add_argument("--think-steps", type=int, default=4)
    parser.add_argument("--max-train-cases", type=int, default=512)
    parser.add_argument("--preserve-jsonl", default="")
    parser.add_argument("--max-preserve-cases", type=int, default=0)
    parser.add_argument("--preserve-batch-size", type=int, default=4)
    parser.add_argument("--preserve-kl-weight", type=float, default=0.0)
    parser.add_argument("--multi-depth-ce-weight", type=float, default=0.0)
    parser.add_argument("--multi-depth-ce-depths", default="")
    parser.add_argument("--depth-gain-weight", type=float, default=0.0)
    parser.add_argument("--depth-gain-margin", type=float, default=0.25)
    parser.add_argument("--depth-gain-shallow-depths", default="0,1,2")
    parser.add_argument("--trajectory-kl-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-kl-anchor-depth", type=int, default=8)
    parser.add_argument("--trajectory-kl-compare-depths", default="")
    parser.add_argument("--max-eval-cases", type=int, default=64)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--target-rendering", choices=("mass", "plain", "space", "newline"), default="space")
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def main() -> None:
    report = train(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
