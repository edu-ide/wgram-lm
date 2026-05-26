#!/usr/bin/env python3
"""Evaluate BPE PrefixLM checkpoints on a Generalization Dynamics choice gate."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IGNORE_LABEL_ID = -100


def load_trainer_module() -> Any:
    path = ROOT / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("native_prefixlm_for_bpe_gd", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"probe contains no rows: {path}")
    return rows


def resolve_tokenizer_path(sampled_data: Path, tokenizer_path: str = "") -> Path:
    candidates: list[Path] = []
    if str(tokenizer_path).strip():
        candidates.append(Path(tokenizer_path))
    candidates.append(ROOT / "references" / "official" / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json")
    for parent in [sampled_data, *sampled_data.parents]:
        candidates.append(parent / "trained_tokenizers" / "bpe" / "tokenizer.json")
        candidates.append(parent / "references" / "official" / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not resolve BPE tokenizer for {sampled_data}")


def load_tokenizer(path: Path) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(str(text), add_special_tokens=False).ids]


def build_choice_tensors(
    *,
    tokenizer: Any,
    prompt: str,
    answer: str,
    seq_len: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = encode_text(tokenizer, str(prompt))
    answer_ids = encode_text(tokenizer, str(answer))
    if not prompt_ids:
        raise ValueError("prompt encodes to no tokens")
    if not answer_ids:
        raise ValueError("answer encodes to no tokens")
    input_ids = prompt_ids + answer_ids[:-1]
    labels = [IGNORE_LABEL_ID] * max(0, len(prompt_ids) - 1) + answer_ids
    if len(input_ids) != len(labels):
        raise ValueError("internal PrefixLM choice tensor length mismatch")
    if len(input_ids) > int(seq_len):
        raise ValueError(
            f"choice row exceeds seq_len={seq_len}: shifted length={len(input_ids)}"
        )
    pad_len = int(seq_len) - len(input_ids)
    input_ids = input_ids + [0] * pad_len
    labels = labels + [IGNORE_LABEL_ID] * pad_len
    attention_mask = [1] * (int(seq_len) - pad_len) + [0] * pad_len
    return (
        torch.tensor([input_ids], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device),
        torch.tensor([attention_mask], dtype=torch.long, device=device),
    )


def choice_logprob(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    prompt: str,
    answer: str,
    seq_len: int,
    device: torch.device,
    think_steps: int,
    amp_context: Any,
) -> dict[str, float | int]:
    input_ids, labels, _attention_mask = build_choice_tensors(
        tokenizer=tokenizer,
        prompt=prompt,
        answer=answer,
        seq_len=int(seq_len),
        device=device,
    )
    with torch.no_grad(), amp_context:
        logits = model(input_ids, think_steps=int(think_steps))
        length = min(int(logits.shape[1]), int(labels.shape[1]))
        log_probs = F.log_softmax(logits[:, :length].float(), dim=-1)
        labels = labels[:, :length]
        mask = labels.ne(IGNORE_LABEL_ID)
        if not bool(mask.any()):
            raise ValueError("choice row has no supervised answer tokens")
        token_log_probs = log_probs[mask].gather(1, labels[mask].unsqueeze(1)).squeeze(1)
    summed = float(token_log_probs.sum().detach().cpu().item())
    count = int(token_log_probs.numel())
    return {
        "sum_logprob": summed,
        "mean_logprob": float(summed / float(max(1, count))),
        "tokens": int(count),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("skipped_reason") is None]
    if not valid:
        return {
            "rows": int(len(rows)),
            "valid_rows": 0,
            "accuracy": 0.0,
            "mean_margin": float("nan"),
            "accepted": False,
        }
    correct = sum(1 for row in valid if bool(row["correct"]))
    margins = [float(row["normalized_margin"]) for row in valid]
    by_task: dict[str, list[float]] = {}
    by_task_correct: dict[str, int] = {}
    for row in valid:
        task = str(row.get("task") or "unknown")
        by_task.setdefault(task, []).append(float(row["normalized_margin"]))
        by_task_correct[task] = by_task_correct.get(task, 0) + int(bool(row["correct"]))
    task_summary = {
        task: {
            "rows": len(values),
            "accuracy": float(by_task_correct.get(task, 0) / float(max(1, len(values)))),
            "mean_margin": float(sum(values) / float(len(values))),
            "min_margin": float(min(values)),
            "passed": all(value > 0.0 for value in values),
        }
        for task, values in sorted(by_task.items())
    }
    return {
        "rows": int(len(rows)),
        "valid_rows": int(len(valid)),
        "accuracy": float(correct / float(len(valid))),
        "mean_margin": float(sum(margins) / float(len(margins))),
        "min_margin": float(min(margins)),
        "tasks": task_summary,
        "accepted": bool(correct == len(valid) and min(margins) > 0.0),
    }


def sanitize_tag_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "unknown"


def write_tensorboard_scalars(
    *,
    report: dict[str, Any],
    tensorboard_dir: str,
    prefix: str,
    step: int,
) -> int:
    if not str(tensorboard_dir):
        return 0
    from torch.utils.tensorboard import SummaryWriter

    summary = report.get("summary") or {}
    written = 0
    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        for key in ("valid_rows", "accuracy", "mean_margin", "min_margin"):
            value = summary.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                writer.add_scalar(f"{prefix}/{key}", float(value), int(step))
                written += 1
        writer.add_scalar(f"{prefix}/accepted", float(int(bool(report.get("accepted", False)))), int(step))
        written += 1
        tasks = summary.get("tasks") or {}
        if isinstance(tasks, dict):
            for task, values in sorted(tasks.items()):
                if not isinstance(values, dict):
                    continue
                tag = sanitize_tag_part(str(task))
                for key in ("accuracy", "mean_margin", "min_margin", "passed"):
                    value = values.get(key)
                    if isinstance(value, bool):
                        value = float(int(value))
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        writer.add_scalar(f"{prefix}/task/{tag}/{key}", float(value), int(step))
                        written += 1
        writer.flush()
    return written


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    trainer = load_trainer_module()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    train_args = argparse.Namespace(**dict(checkpoint["args"]))
    sampled_data = Path(args.sampled_data or getattr(train_args, "sampled_data", ""))
    metadata = trainer.load_prefixlm_metadata(sampled_data)
    tokenizer_info = dict(metadata.tokenizer_info or {})
    tokenizer_path = resolve_tokenizer_path(
        sampled_data,
        str(args.tokenizer_path or tokenizer_info.get("tokenizer_path") or ""),
    )
    tokenizer = load_tokenizer(tokenizer_path)
    probe_rows = load_jsonl(Path(args.probe_jsonl))

    model_info = dict(checkpoint.get("model") or {})
    vocab_size = int(
        model_info.get("vocab_size")
        or getattr(train_args, "model_vocab_size", 0)
        or trainer.round_up_multiple(int(metadata.vocab_size), 256)
    )
    device = torch.device(str(args.device))
    model = trainer.build_model(train_args, vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    amp_dtype = trainer.resolve_amp_dtype(str(args.amp_dtype))

    def make_amp_context() -> Any:
        if str(device.type) != "cuda":
            return nullcontext()
        return trainer.autocast_context(device, amp_dtype)

    seq_len = int(args.seq_len or getattr(train_args, "seq_len", 0))
    think_steps = int(args.think_steps or getattr(train_args, "train_think_steps", 1))
    rows: list[dict[str, Any]] = []
    limit = int(args.max_rows)
    for row in probe_rows[: limit if limit > 0 else None]:
        out_row: dict[str, Any] = {
            "id": row.get("id"),
            "task": row.get("task"),
            "source": row.get("source"),
        }
        try:
            intelligence = choice_logprob(
                model,
                tokenizer=tokenizer,
                prompt=str(row["prompt"]),
                answer=str(row["intelligence_answer"]),
                seq_len=seq_len,
                device=device,
                think_steps=think_steps,
                amp_context=make_amp_context(),
            )
            parrot = choice_logprob(
                model,
                tokenizer=tokenizer,
                prompt=str(row["prompt"]),
                answer=str(row["parrot_answer"]),
                seq_len=seq_len,
                device=device,
                think_steps=think_steps,
                amp_context=make_amp_context(),
            )
            margin = float(intelligence["mean_logprob"]) - float(parrot["mean_logprob"])
            out_row.update(
                {
                    "intelligence_mean_logprob": float(intelligence["mean_logprob"]),
                    "parrot_mean_logprob": float(parrot["mean_logprob"]),
                    "intelligence_sum_logprob": float(intelligence["sum_logprob"]),
                    "parrot_sum_logprob": float(parrot["sum_logprob"]),
                    "intelligence_tokens": int(intelligence["tokens"]),
                    "parrot_tokens": int(parrot["tokens"]),
                    "normalized_margin": float(margin),
                    "correct": bool(margin > 0.0),
                    "skipped_reason": None,
                }
            )
        except Exception as exc:
            if not bool(args.skip_bad_rows):
                raise
            out_row.update(
                {
                    "normalized_margin": float("nan"),
                    "correct": False,
                    "skipped_reason": str(exc),
                }
            )
        rows.append(out_row)

    summary = summarize_rows(rows)
    source_counts = dict(sorted(Counter(str(row.get("source", "")) for row in probe_rows).items()))
    family_counts = dict(sorted(Counter(str(row.get("family", "")) for row in probe_rows if row.get("family") is not None).items()))
    report = {
        "gate_type": "generalization_dynamics_bpe_choice_probe",
        "checkpoint": str(checkpoint_path),
        "probe_jsonl": str(args.probe_jsonl),
        "sampled_data": str(sampled_data),
        "tokenizer_path": str(tokenizer_path),
        "think_steps": int(think_steps),
        "seq_len": int(seq_len),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "source_counts": source_counts,
        "family_counts": family_counts,
        "summary": summary,
        "accepted": bool(summary.get("accepted", False)),
        "rows": rows,
        "plain_language_read": (
            "This is the BPE stable-reader counterpart to the BLT GD choice gate. "
            "It separates tokenizer/reading failure from actual generalization "
            "preference by asking whether the checkpoint gives the intelligence "
            "answer higher PrefixLM probability than the parrot answer."
        ),
    }
    if str(args.tensorboard_dir):
        report["tensorboard_scalars_written"] = write_tensorboard_scalars(
            report=report,
            tensorboard_dir=str(args.tensorboard_dir),
            prefix=str(args.tensorboard_prefix),
            step=int(args.tensorboard_step if int(args.tensorboard_step) >= 0 else checkpoint.get("step", 0)),
        )
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-jsonl", default="data/eval/official_gdsuite_choice_probe.jsonl")
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-bad-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", default="")
    parser.add_argument("--tensorboard-prefix", default="eval/generalization_dynamics_bpe")
    parser.add_argument("--tensorboard-step", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_eval(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
