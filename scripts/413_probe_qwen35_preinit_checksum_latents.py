#!/usr/bin/env python3
"""Probe whether Qwen3.5-preinit QTRM latents bind checksum4 operands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgram_lm.qwen_backbone_wgram import QwenBackboneQTRM


_CHECKSUM4_RE = re.compile(r"a=(\d+), b=(\d+), c=(\d+), d=(\d+)")


def _load_gate_module() -> ModuleType:
    path = Path(__file__).with_name("362_train_qwen_backbone_wgram_core_gate.py")
    spec = importlib.util.spec_from_file_location("qtrm_gate_362", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _parse_operands(prompt: str) -> tuple[int, int, int, int]:
    match = _CHECKSUM4_RE.search(prompt)
    if match is None:
        raise ValueError(f"not a checksum4 prompt: {prompt}")
    return tuple(int(match.group(index)) for index in range(1, 5))  # type: ignore[return-value]


def _take_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, -1, :]
    index = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]


def _choice_stats(logits: torch.Tensor, labels: list[str], label_ids: dict[str, int]) -> dict[str, Any]:
    digits = list("0123456789")
    token_ids = torch.tensor([label_ids[digit] for digit in digits], device=logits.device)
    choice_logits = logits.index_select(dim=-1, index=token_ids).float()
    target = torch.tensor([digits.index(label) for label in labels], device=logits.device)
    pred = choice_logits.argmax(dim=-1)
    row = torch.arange(target.numel(), device=logits.device)
    wrong_mask = torch.ones_like(choice_logits, dtype=torch.bool)
    wrong_mask[row, target] = False
    target_logit = choice_logits[row, target]
    wrong_logit = choice_logits.masked_fill(~wrong_mask, float("-inf")).amax(dim=-1)
    sorted_indices = choice_logits.argsort(dim=-1, descending=True)
    ranks = (sorted_indices == target[:, None]).nonzero()[:, 1] + 1
    return {
        "pred": pred.detach().cpu(),
        "target": target.detach().cpu(),
        "margin": (target_logit - wrong_logit).detach().cpu(),
        "rank": ranks.detach().cpu(),
    }


@torch.no_grad()
def _extract(
    model: QwenBackboneQTRM,
    tokenizer: Any,
    cases: list[Any],
    gate: ModuleType,
    args: argparse.Namespace,
    label_ids: dict[str, int],
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    features: dict[str, list[torch.Tensor]] = {
        "qwen_hidden": [],
        "z_l": [],
        "z_h": [],
        "delta_h": [],
    }
    operands = []
    labels = []
    base_preds = []
    core_preds = []
    base_margins = []
    core_margins = []
    base_ranks = []
    core_ranks = []
    for chunk in gate._batch(cases, int(args.batch_size)):
        input_ids, attention_mask = gate._encode_prompts(
            tokenizer,
            [case.prompt for case in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        qwen_outputs = model.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = qwen_outputs.hidden_states[-1]
        z_l, z_h, _trajectory, _info = model.core(
            model.core_in_norm(hidden),
            attention_mask=attention_mask,
        )
        base_logits = gate._last_token_logits(model, input_ids, attention_mask, force_core_off=True)
        core_logits = gate._last_token_logits(model, input_ids, attention_mask)
        chunk_labels = [case.label for case in chunk]
        base = _choice_stats(base_logits, chunk_labels, label_ids)
        core = _choice_stats(core_logits, chunk_labels, label_ids)
        features["qwen_hidden"].append(_take_last_token(hidden, attention_mask).float().cpu())
        z_l_last = _take_last_token(z_l, attention_mask).float().cpu()
        z_h_last = _take_last_token(z_h, attention_mask).float().cpu()
        features["z_l"].append(z_l_last)
        features["z_h"].append(z_h_last)
        features["delta_h"].append(z_h_last - features["qwen_hidden"][-1])
        operands.extend(_parse_operands(case.prompt) for case in chunk)
        labels.extend(int(case.label) for case in chunk)
        base_preds.append(base["pred"])
        core_preds.append(core["pred"])
        base_margins.append(base["margin"])
        core_margins.append(core["margin"])
        base_ranks.append(base["rank"])
        core_ranks.append(core["rank"])
    return {
        **{name: torch.cat(parts, dim=0) for name, parts in features.items()},
        "operands": torch.tensor(operands, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "base_pred": torch.cat(base_preds),
        "core_pred": torch.cat(core_preds),
        "base_margin": torch.cat(base_margins),
        "core_margin": torch.cat(core_margins),
        "base_rank": torch.cat(base_ranks),
        "core_rank": torch.cat(core_ranks),
    }


def _fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> dict[str, float]:
    train_x = F.layer_norm(train_x.float(), (train_x.shape[-1],))
    eval_x = F.layer_norm(eval_x.float(), (eval_x.shape[-1],))
    head = nn.Linear(train_x.shape[-1], 10)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=0.01)
    for _ in range(int(steps)):
        logits = head(train_x)
        loss = F.cross_entropy(logits, train_y.long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (head(train_x).argmax(dim=-1) == train_y).float().mean()
        eval_acc = (head(eval_x).argmax(dim=-1) == eval_y).float().mean()
    return {
        "train_accuracy": float(train_acc.item()),
        "eval_accuracy": float(eval_acc.item()),
    }


def _probe_table(train: dict[str, torch.Tensor], eval_: dict[str, torch.Tensor], args) -> dict[str, Any]:
    result: dict[str, Any] = {}
    targets = {
        "answer": (train["labels"], eval_["labels"]),
        "operand_a": (train["operands"][:, 0], eval_["operands"][:, 0]),
        "operand_b": (train["operands"][:, 1], eval_["operands"][:, 1]),
        "operand_c": (train["operands"][:, 2], eval_["operands"][:, 2]),
        "operand_d": (train["operands"][:, 3], eval_["operands"][:, 3]),
    }
    for feature_name in ("qwen_hidden", "z_l", "z_h", "delta_h"):
        result[feature_name] = {}
        for target_name, (train_y, eval_y) in targets.items():
            result[feature_name][target_name] = _fit_probe(
                train[feature_name],
                train_y,
                eval_[feature_name],
                eval_y,
                steps=int(args.probe_steps),
                lr=float(args.probe_lr),
            )
    return result


def _prediction_summary(data: dict[str, torch.Tensor]) -> dict[str, Any]:
    labels = data["labels"]
    base_correct = data["base_pred"] == labels
    core_correct = data["core_pred"] == labels
    fixes = (~base_correct) & core_correct
    breaks = base_correct & (~core_correct)
    return {
        "cases": int(labels.numel()),
        "base_accuracy": float(base_correct.float().mean().item()),
        "core_accuracy": float(core_correct.float().mean().item()),
        "gain": float((core_correct.float().mean() - base_correct.float().mean()).item()),
        "core_fixes_base_errors": int(fixes.sum().item()),
        "core_breaks_base_correct": int(breaks.sum().item()),
        "mean_base_margin": float(data["base_margin"].float().mean().item()),
        "mean_core_margin": float(data["core_margin"].float().mean().item()),
        "mean_margin_delta": float((data["core_margin"] - data["base_margin"]).float().mean().item()),
        "mean_base_rank": float(data["base_rank"].float().mean().item()),
        "mean_core_rank": float(data["core_rank"].float().mean().item()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gate = _load_gate_module()
    from transformers import AutoTokenizer

    device = torch.device(str(args.device))
    dtype = gate._dtype(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = gate._label_token_ids(tokenizer)
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=True,
        core_impl="qwen_shared_layer_wrapped",
        mandatory_core=True,
        qwen_core_layer_indices=gate.parse_int_list(str(args.qwen_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        residual_scale=float(args.residual_scale),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        core_step_conditioning_enabled=True,
        core_step_conditioning_max_steps=int(args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(args.core_step_conditioning_scale),
        core_causal=True,
    ).to(device)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys[:8]}")
    model.eval()

    train_cases = [
        case
        for case in gate.build_synthetic_cases(
            count=int(args.train_cases),
            seed=int(args.seed),
            case_mode="hard_v1",
        )
        if case.family == "checksum4"
    ]
    eval_cases = [
        case
        for case in gate.build_synthetic_cases(
            count=int(args.eval_cases),
            seed=int(args.seed) + 10000,
            case_mode="hard_v1",
        )
        if case.family == "checksum4"
    ]
    train = _extract(model, tokenizer, train_cases, gate, args, label_ids)
    eval_ = _extract(model, tokenizer, eval_cases, gate, args, label_ids)
    report = {
        "checkpoint": str(args.checkpoint),
        "train_checksum4_cases": len(train_cases),
        "eval_checksum4_cases": len(eval_cases),
        "prediction_summary": {
            "train": _prediction_summary(train),
            "eval": _prediction_summary(eval_),
        },
        "linear_probes": _probe_table(train, eval_, args),
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--core-adapter-dim", type=int, default=128)
    parser.add_argument("--core-delta-adapter-mode", default="add")
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=3)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--train-cases", type=int, default=2048)
    parser.add_argument("--eval-cases", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1.0e-2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run(args)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
