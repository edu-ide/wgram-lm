#!/usr/bin/env python3
"""Evaluate PrefixLM raw-intelligence probes by primitive axis.

This script is a dashboard, not a new training loss. For each primitive axis it
reports:
  - teacher-forced target-token loss/perplexity;
  - target-token accuracy;
  - greedy generation hit accuracy.
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


@dataclass(frozen=True)
class RawPrimitiveCase:
    case_id: str
    primitive: str
    family: str
    language: str
    instruction: str
    response: str
    expected_contains: tuple[str, ...]
    match_mode: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_language_loss_module() -> Any:
    path = repo_root() / "scripts" / "544_eval_prefixlm_language_heldout_loss.py"
    spec = importlib.util.spec_from_file_location("prefixlm_language_heldout_loss_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_raw_cases(path: str | Path) -> list[RawPrimitiveCase]:
    cases: list[RawPrimitiveCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            response = str(row.get("response") or "")
            if not response:
                raise ValueError(f"raw probe row {line_number} has no response")
            expected = tuple(str(value) for value in row.get("expected_contains", []))
            if not expected:
                expected = (response,)
            match_mode = str(row.get("match_mode") or "all_contains")
            if match_mode not in {"all_contains", "any_contains", "exact"}:
                raise ValueError(
                    f"raw probe row {line_number} has unsupported match_mode={match_mode!r}"
                )
            cases.append(
                RawPrimitiveCase(
                    case_id=str(row["case_id"]),
                    primitive=str(row["primitive"]),
                    family=str(row["family"]),
                    language=str(row["language"]),
                    instruction=str(row["instruction"]),
                    response=response,
                    expected_contains=expected,
                    match_mode=match_mode,
                )
            )
    if not cases:
        raise ValueError(f"no raw primitive cases loaded from {path}")
    return cases


def score_generation(
    *,
    helper: Any,
    response: str,
    expected_contains: tuple[str, ...],
    match_mode: str,
) -> dict[str, Any]:
    cleaned = helper.strip_response_text(response)
    normalized = helper.normalize_for_match(cleaned)
    expected_norm = [helper.normalize_for_match(value) for value in expected_contains]
    if str(match_mode) == "exact":
        hit = any(normalized == value for value in expected_norm)
        matched = [
            expected
            for expected, norm in zip(expected_contains, expected_norm)
            if normalized == norm
        ]
    elif str(match_mode) == "any_contains":
        matched = [
            expected
            for expected, norm in zip(expected_contains, expected_norm)
            if norm in normalized
        ]
        hit = bool(matched)
    else:
        matched = [
            expected
            for expected, norm in zip(expected_contains, expected_norm)
            if norm in normalized
        ]
        hit = len(matched) == len(expected_contains)
    return {
        "generation_hit": bool(hit),
        "generation_matched": matched,
        "cleaned_response": cleaned,
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
        gen_hits = sum(1 for item in items if bool(item["generation_hit"]))
        loss = loss_num / max(1, tokens)
        summary[name] = {
            "cases": len(items),
            "target_tokens": tokens,
            "loss": loss,
            "perplexity": float(math.exp(min(20.0, loss))),
            "token_accuracy": float(correct / max(1, tokens)),
            "generation_hits": int(gen_hits),
            "generation_accuracy": float(gen_hits / max(1, len(items))),
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--probe-jsonl",
        default=str(repo_root() / "data" / "eval" / "prefixlm_raw_intelligence_probe.jsonl"),
    )
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--condition", default="direct")
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--drop-overlength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    language_eval = load_language_loss_module()
    helper = language_eval.load_helper_module()
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

    cases = load_raw_cases(args.probe_jsonl)
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    rows: list[dict[str, Any]] = []
    total_loss_num = 0.0
    total_tokens = 0
    total_correct = 0
    generation_hits = 0
    for case in cases:
        example = language_eval.build_prefixlm_example(
            helper=helper,
            tokenizer=tokenizer,
            tokenizer_info=tokenizer_info,
            case=case,
            condition=str(args.condition),
            eoa_id=int(eoa_id),
            seq_len=int(seq_len),
            drop_overlength=bool(args.drop_overlength),
        )
        loss_metrics = language_eval.evaluate_case(
            model=model,
            input_ids=example["input_ids"],
            labels=example["labels"],
            device=device,
            think_steps=int(think_steps),
        )
        prefix_ids = helper.build_instruction_ids(
            tokenizer=tokenizer,
            instruction=case.instruction,
            condition=str(args.condition),
            tokenizer_info=tokenizer_info,
        )
        generated_ids = helper.generate_one(
            model=model,
            prefix_ids=prefix_ids,
            eoa_id=int(eoa_id),
            device=device,
            think_steps=int(think_steps),
            seq_len=int(seq_len),
            max_new_tokens=int(args.max_new_tokens),
        )
        raw_response = helper.decode_ids(tokenizer, generated_ids)
        gen_metrics = score_generation(
            helper=helper,
            response=raw_response,
            expected_contains=case.expected_contains,
            match_mode=case.match_mode,
        )

        total_loss_num += float(loss_metrics["loss"]) * int(loss_metrics["target_tokens"])
        total_tokens += int(loss_metrics["target_tokens"])
        total_correct += int(loss_metrics["correct_tokens"])
        if bool(gen_metrics["generation_hit"]):
            generation_hits += 1
        rows.append(
            {
                "case_id": case.case_id,
                "primitive": case.primitive,
                "family": case.family,
                "language": case.language,
                "instruction": case.instruction,
                "response": case.response,
                "expected_contains": list(case.expected_contains),
                "match_mode": case.match_mode,
                "generated_ids": generated_ids,
                "raw_response": raw_response,
                **loss_metrics,
                **gen_metrics,
            }
        )

    loss = total_loss_num / max(1, total_tokens)
    report = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", 0)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "target_tokens_seen": int(checkpoint.get("target_tokens_seen", 0)),
        "probe_jsonl": str(args.probe_jsonl),
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
        "generation_hits": int(generation_hits),
        "generation_accuracy": float(generation_hits / max(1, len(rows))),
        "by_primitive": summarize(rows, "primitive"),
        "by_family": summarize(rows, "family"),
        "by_language": summarize(rows, "language"),
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
