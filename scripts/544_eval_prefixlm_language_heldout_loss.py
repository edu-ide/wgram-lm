#!/usr/bin/env python3
"""Evaluate PrefixLM target-token loss on small heldout language cases.

This is a measurement tool, not a new training objective. It asks:
"Can the checkpoint assign high probability to normal heldout language answers
through the same PrefixLM answer path?"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


IGNORE_LABEL_ID = -100


@dataclass(frozen=True)
class HeldoutCase:
    case_id: str
    family: str
    language: str
    instruction: str
    response: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_helper_module() -> Any:
    root = repo_root()
    path = root / "scripts" / "542_eval_prefixlm_multilingual_probe.py"
    spec = importlib.util.spec_from_file_location("prefixlm_multilingual_probe_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_heldout_cases(path: str | Path) -> list[HeldoutCase]:
    cases: list[HeldoutCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            response = str(row.get("response") or "")
            if not response:
                raise ValueError(f"heldout row {line_number} has no response")
            cases.append(
                HeldoutCase(
                    case_id=str(row["case_id"]),
                    family=str(row["family"]),
                    language=str(row["language"]),
                    instruction=str(row["instruction"]),
                    response=response,
                )
            )
    if not cases:
        raise ValueError(f"no heldout cases loaded from {path}")
    return cases


def build_prefixlm_example(
    *,
    helper: Any,
    tokenizer: Any,
    tokenizer_info: dict[str, Any],
    case: HeldoutCase,
    condition: str,
    eoa_id: int,
    seq_len: int,
    drop_overlength: bool,
) -> dict[str, Any]:
    inst = helper.build_instruction_ids(
        tokenizer=tokenizer,
        instruction=case.instruction,
        condition=str(condition),
        tokenizer_info=tokenizer_info,
    )
    resp = [int(value) for value in tokenizer.encode(case.response, add_special_tokens=False).ids]
    resp.append(int(eoa_id))
    input_ids = inst + resp[:-1]
    labels = [IGNORE_LABEL_ID] * max(0, len(inst) - 1) + resp
    if len(input_ids) != len(labels):
        raise ValueError(
            f"PrefixLM row length mismatch for {case.case_id}: "
            f"inputs={len(input_ids)} labels={len(labels)}"
        )
    if len(input_ids) > int(seq_len):
        if drop_overlength:
            raise ValueError(
                f"heldout case {case.case_id} is over seq_len={seq_len}; "
                "increase --seq-len or disable --drop-overlength"
            )
        input_ids = input_ids[: int(seq_len)]
        labels = labels[: int(seq_len)]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "target_tokens": sum(1 for value in labels if int(value) != IGNORE_LABEL_ID),
    }


def evaluate_case(
    *,
    model: torch.nn.Module,
    input_ids: list[int],
    labels: list[int],
    device: torch.device,
    think_steps: int,
) -> dict[str, Any]:
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, think_steps=int(think_steps))
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = y.reshape(-1)
        mask = flat_labels.ne(IGNORE_LABEL_ID)
        if not bool(mask.any()):
            raise ValueError("heldout case has no target tokens")
        losses = F.cross_entropy(
            flat_logits[mask],
            flat_labels[mask],
            reduction="none",
        )
        predictions = flat_logits[mask].argmax(dim=-1)
        correct = predictions.eq(flat_labels[mask])
    loss = float(losses.mean().detach().cpu().item())
    target_tokens = int(mask.sum().detach().cpu().item())
    hits = int(correct.sum().detach().cpu().item())
    return {
        "loss": loss,
        "perplexity": float(math.exp(min(20.0, loss))),
        "target_tokens": target_tokens,
        "token_accuracy": float(hits / target_tokens),
        "correct_tokens": hits,
    }


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for name, items in sorted(buckets.items()):
        tokens = sum(int(item["target_tokens"]) for item in items)
        loss_num = sum(float(item["loss"]) * int(item["target_tokens"]) for item in items)
        correct = sum(int(item["correct_tokens"]) for item in items)
        loss = loss_num / max(1, tokens)
        summary[name] = {
            "cases": len(items),
            "target_tokens": tokens,
            "loss": loss,
            "perplexity": float(math.exp(min(20.0, loss))),
            "token_accuracy": float(correct / max(1, tokens)),
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--heldout-jsonl",
        default=str(repo_root() / "data" / "eval" / "prefixlm_language_heldout.jsonl"),
    )
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--condition", default="direct")
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--drop-overlength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    helper = load_helper_module()
    trainer = helper.load_trainer_module()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    train_args = helper.checkpoint_args(trainer, checkpoint)

    tokenizer_path = helper.resolve_tokenizer_path(
        checkpoint=checkpoint,
        tokenizer_path=str(args.tokenizer_path),
    )
    tokenizer = helper.load_tokenizer(tokenizer_path)
    dataset_info = dict(checkpoint.get("dataset") or {})
    tokenizer_info = dict(dataset_info.get("tokenizer_info") or {})
    eoa_text = str(tokenizer_info.get("eoa") or "<|box_end|>")
    eoa_id = helper.token_id(tokenizer, eoa_text)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    vocab_size = helper.infer_vocab_size(checkpoint, train_args)
    model = trainer.build_model(train_args, vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    think_steps = (
        int(args.think_steps)
        if int(args.think_steps) > 0
        else int(getattr(train_args, "train_think_steps", 0))
    )
    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(getattr(train_args, "seq_len", 128))

    cases = load_heldout_cases(args.heldout_jsonl)
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    rows: list[dict[str, Any]] = []
    total_loss_num = 0.0
    total_tokens = 0
    total_correct = 0
    for case in cases:
        example = build_prefixlm_example(
            helper=helper,
            tokenizer=tokenizer,
            tokenizer_info=tokenizer_info,
            case=case,
            condition=str(args.condition),
            eoa_id=int(eoa_id),
            seq_len=int(seq_len),
            drop_overlength=bool(args.drop_overlength),
        )
        metrics = evaluate_case(
            model=model,
            input_ids=example["input_ids"],
            labels=example["labels"],
            device=device,
            think_steps=int(think_steps),
        )
        total_loss_num += float(metrics["loss"]) * int(metrics["target_tokens"])
        total_tokens += int(metrics["target_tokens"])
        total_correct += int(metrics["correct_tokens"])
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "language": case.language,
                "instruction": case.instruction,
                "response": case.response,
                **metrics,
            }
        )

    loss = total_loss_num / max(1, total_tokens)
    report = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", 0)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "target_tokens_seen": int(checkpoint.get("target_tokens_seen", 0)),
        "heldout_jsonl": str(args.heldout_jsonl),
        "tokenizer_path": str(tokenizer_path),
        "condition": str(args.condition),
        "think_steps": int(think_steps),
        "seq_len": int(seq_len),
        "eoa": {"text": eoa_text, "token_id": int(eoa_id)},
        "cases": len(rows),
        "target_tokens": int(total_tokens),
        "loss": float(loss),
        "perplexity": float(math.exp(min(20.0, loss))),
        "token_accuracy": float(total_correct / max(1, total_tokens)),
        "by_language": summarize(rows, "language"),
        "by_family": summarize(rows, "family"),
        "results": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
