#!/usr/bin/env python3
"""Evaluate a QTRM-native checkpoint on public multiple-choice suites."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch


OPTION_LETTERS = "ABCDEFGHIJ"


def load_eval_module():
    path = Path(__file__).with_name("356_eval_qtrm_native_language_generalization.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_mcq_answer(text: str) -> str:
    upper = str(text).strip().upper()
    if upper in OPTION_LETTERS:
        return upper
    match = re.search(r"(?:ANSWER\s*[:：]?\s*)?\(?\b([A-J])\b\)?", upper)
    if match:
        return match.group(1)
    return ""


def extract_answer_text(generated: str, prompt: str) -> tuple[str, bool]:
    """Return only the answer-bearing suffix, never echoed prompt/options text."""
    generated = str(generated)
    prompt = str(prompt)
    if generated.startswith(prompt):
        suffix = generated[len(prompt) :].strip()
        return suffix, bool(not suffix and "Options:" in prompt)

    # Native bootstrap checkpoints often reconstruct a lossy prompt instead of
    # continuing after it. Treat echoed chat/options text as no answer unless
    # there is actual text after the final Assistant marker.
    lower = generated.lower()
    prompt_echo = bool(
        lower.startswith("user:")
        or "options:" in lower
        or "\na." in lower
        or "\nanswer:" in lower
    )
    marker_positions = [
        generated.rfind("Assistant:"),
        generated.rfind("assistant:"),
        generated.rfind("Answer:"),
        generated.rfind("answer:"),
    ]
    marker_pos = max(marker_positions)
    if marker_pos >= 0:
        marker = "Assistant:" if generated[marker_pos :].startswith("Assistant:") else "assistant:"
        if generated[marker_pos :].startswith("Answer:"):
            marker = "Answer:"
        elif generated[marker_pos :].startswith("answer:"):
            marker = "answer:"
        return generated[marker_pos + len(marker) :].strip(), prompt_echo
    if prompt_echo:
        return "", True
    return generated.strip(), False


def load_suite(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"suite row must be an object at {path}:{line_no}")
        for key in ("benchmark_id", "case_id", "qtrm_prompt", "answer"):
            if key not in row:
                raise ValueError(f"suite row missing {key} at {path}:{line_no}")
        rows.append(row)
        if int(max_cases) > 0 and len(rows) >= int(max_cases):
            break
    if not rows:
        raise ValueError(f"suite is empty: {path}")
    return rows


def build_eval_args(eval_module, checkpoint_args: dict[str, object], args: argparse.Namespace):
    overrides = Namespace(
        device=str(args.device),
        out_dir=str(args.out_dir),
        eval_think_steps=int(args.think_steps),
        max_new_chars=int(args.max_new_chars),
        repair_prompt_count=1,
        eval_seed_texts="",
        eval_seed_expectations="{}",
        eval_jsonl="",
        min_on_policy_continuation_chars=0,
        min_on_policy_keyword_hits=0,
        min_on_policy_loop_check_lines=4,
        min_on_policy_unique_line_fraction=0.55,
        max_on_policy_repeated_block_fraction=0.24,
        max_on_policy_repeated_line_fraction=0.30,
    )
    return eval_module.merged_checkpoint_args(checkpoint_args, overrides)


def load_model_bundle(args: argparse.Namespace):
    eval_module = load_eval_module()
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    eval_args = build_eval_args(eval_module, checkpoint_args, args)
    tokenizer = eval_module.tokenizer_from_checkpoint(checkpoint.get("tokenizer", {}), eval_args)
    device = torch.device(str(args.device))
    model = eval_module._text_probe.build_model(eval_args, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return eval_module, eval_args, tokenizer, model, device


@torch.no_grad()
def generate_completion_text(
    model,
    tokenizer,
    *,
    seed_text: str,
    seq_len: int,
    think_steps: int,
    max_new_chars: int,
    device: torch.device,
) -> str:
    """Decode only newly generated ids.

    The language bootstrap tokenizer can be compact and lossy. Decoding the
    full prompt+completion sequence reconstructs the prompt imperfectly, which
    looks like prompt echo even when the model generated no such suffix.
    """
    model.eval()
    encoded = tokenizer.encode(seed_text)
    if not encoded:
        encoded = [0]
    out = torch.tensor([encoded], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(int(max_new_chars)):
        x = out[:, -int(seq_len) :]
        logits = model(x, think_steps=int(think_steps))
        next_id_tensor = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_id = int(next_id_tensor.item())
        out = torch.cat([out, next_id_tensor], dim=1)
        if eos_token_id is not None and next_id == int(eos_token_id):
            break
        generated_ids.append(next_id)
    return str(tokenizer.decode(generated_ids))


@torch.no_grad()
def generate_answer(
    eval_module,
    eval_args,
    tokenizer,
    model,
    device: torch.device,
    *,
    prompt: str,
    think_steps: int,
    max_new_chars: int,
) -> tuple[str, str]:
    completion = generate_completion_text(
        model,
        tokenizer,
        seed_text=str(prompt),
        seq_len=int(eval_args.seq_len),
        think_steps=int(think_steps),
        max_new_chars=int(max_new_chars),
        device=device,
    )
    answer_text, prompt_echo = extract_answer_text(completion, prompt)
    return completion, answer_text, normalize_mcq_answer(answer_text), prompt_echo


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(1 for row in rows if bool(row.get("exact", False)))
    invalid = sum(1 for row in rows if not str(row.get("pred_answer", "")).strip())
    prompt_echo = sum(1 for row in rows if bool(row.get("prompt_echo", False)))
    pred_hist: dict[str, int] = {}
    for row in rows:
        pred = str(row.get("pred_answer", "") or "<empty>")
        pred_hist[pred] = pred_hist.get(pred, 0) + 1
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category", "unknown"))
        bucket = by_category.setdefault(category, {"hits": 0, "total": 0})
        bucket["hits"] += int(bool(row.get("exact", False)))
        bucket["total"] += 1
    return {
        "hits": hits,
        "cases": len(rows),
        "accuracy": float(hits / max(1, len(rows))),
        "invalid_pred_count": invalid,
        "invalid_pred_rate": float(invalid / max(1, len(rows))),
        "prompt_echo_count": prompt_echo,
        "prompt_echo_rate": float(prompt_echo / max(1, len(rows))),
        "pred_answer_histogram": dict(sorted(pred_hist.items())),
        "by_category": {
            key: {
                "hits": value["hits"],
                "total": value["total"],
                "accuracy": float(value["hits"] / max(1, value["total"])),
            }
            for key, value in sorted(by_category.items())
        },
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_suite(args.suite_jsonl, max_cases=int(args.max_cases))
    eval_module, eval_args, tokenizer, model, device = load_model_bundle(args)
    scored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(suite, start=1):
        completion, answer_text, pred, prompt_echo = generate_answer(
            eval_module,
            eval_args,
            tokenizer,
            model,
            device,
            prompt=str(row["qtrm_prompt"]),
            think_steps=int(args.think_steps),
            max_new_chars=int(args.max_new_chars),
        )
        gold = normalize_mcq_answer(str(row["answer"]))
        scored = dict(row)
        scored.update(
            {
                "raw_completion": completion,
                "answer_text": answer_text,
                "pred_answer": pred,
                "gold_answer": gold,
                "exact": bool(pred == gold),
                "prompt_echo": bool(prompt_echo),
            }
        )
        scored_rows.append(scored)
        if int(args.log_every) > 0 and index % int(args.log_every) == 0:
            metrics = score_rows(scored_rows)
            print(
                json.dumps(
                    {
                        "progress": index,
                        "cases": len(suite),
                        "accuracy": metrics["accuracy"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    metrics = score_rows(scored_rows)
    target_percent = float(args.qwen36_target_percent)
    target_score = target_percent / 100.0
    parity_floor = target_score - float(args.parity_tolerance)
    accepted = bool(
        metrics["accuracy"] >= parity_floor
        and metrics["cases"] >= int(args.min_cases_for_parity)
    )
    report = {
        "status": "complete",
        "decision": "accepted_m7_public_benchmark_parity" if accepted else "rejected_m7_public_benchmark_parity",
        "accepted": accepted,
        "target_level": "M7 public benchmark parity",
        "benchmark_id": str(args.benchmark_id),
        "benchmark_name": str(args.benchmark_name),
        "suite_jsonl": str(args.suite_jsonl),
        "checkpoint": str(args.checkpoint),
        "model": "QTRM-Native",
        "qwen36_target_percent": target_percent,
        "qwen36_target_score": target_score,
        "parity_tolerance": float(args.parity_tolerance),
        "parity_floor": parity_floor,
        "min_cases_for_parity": int(args.min_cases_for_parity),
        "metrics": metrics,
        "comparison_mode": "public_benchmark_target_score",
        "scorer": "exact option-letter match",
        "limitations": [
            "This report only covers the materialized public subset unless the suite contains the full benchmark.",
            "M7 acceptance requires enough public benchmark cases and a score inside the parity band.",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_jsonl).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-jsonl", default="local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl")
    parser.add_argument(
        "--checkpoint",
        default=(
            "local_eval/qtrm_native_language_bootstrap_bilingual_bpe16k_d192_"
            "repairv3_s4000_20260515/last.pt"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--think-steps", type=int, default=4)
    parser.add_argument("--max-new-chars", type=int, default=24)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--benchmark-id", default="mmlu_pro")
    parser.add_argument("--benchmark-name", default="MMLU-Pro")
    parser.add_argument("--qwen36-target-percent", type=float, default=86.2)
    parser.add_argument("--parity-tolerance", type=float, default=0.02)
    parser.add_argument("--min-cases-for-parity", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--out-dir", default="local_eval/m7_qtrm_native_mmlu_pro_eval")
    parser.add_argument("--out-json", default="local_eval/m7_qtrm_native_mmlu_pro_eval/report.json")
    parser.add_argument("--out-jsonl", default="local_eval/m7_qtrm_native_mmlu_pro_eval/predictions.jsonl")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_checkpoint(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
