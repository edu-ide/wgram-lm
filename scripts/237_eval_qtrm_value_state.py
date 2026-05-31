#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALUE_STATE_CHAR_TO_ID = {str(index): index for index in range(10)}
VALUE_STATE_CHAR_TO_ID.update({",": 10, "-": 11})


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("prompt"):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if not isinstance(row.get("depth_targets"), dict):
                raise ValueError(f"{path}:{line_no}: missing depth_targets")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def value_state_token_ids(text: str) -> list[int] | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    out: list[int] = []
    for char in stripped:
        if char not in VALUE_STATE_CHAR_TO_ID:
            return None
        out.append(int(VALUE_STATE_CHAR_TO_ID[char]))
    return out


def value_state_targets_from_row(
    row: dict[str, Any],
    *,
    num_steps: int,
    max_target_tokens: int,
) -> list[list[int]]:
    if int(num_steps) < 0:
        raise ValueError("num_steps must be non-negative")
    if int(max_target_tokens) <= 0:
        raise ValueError("max_target_tokens must be positive")
    depth_targets = row.get("depth_targets")
    if not isinstance(depth_targets, dict):
        raise ValueError("row must contain depth_targets")
    targets: list[list[int]] = []
    for depth in range(1, int(num_steps) + 1):
        token_ids = value_state_token_ids(str(depth_targets.get(str(depth), "")))
        if token_ids is None:
            targets.append([-100] * int(max_target_tokens))
            continue
        padded = [int(token_id) for token_id in token_ids[: int(max_target_tokens)]]
        padded.extend([-100] * (int(max_target_tokens) - len(padded)))
        targets.append(padded)
    return targets


def predicted_value_sequences_from_logits(logits: Any) -> list[list[int]]:
    if (
        logits.ndim != 4
        or int(logits.shape[1]) == 0
        or int(logits.shape[2]) == 0
        or int(logits.shape[-1]) == 0
    ):
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def score_value_sequences(
    *,
    predicted_sequences: list[list[int]],
    target_sequences: list[list[int]],
) -> dict[str, Any]:
    if len(predicted_sequences) != len(target_sequences):
        raise ValueError("predicted_sequences and target_sequences must have the same length")
    correct_tokens = 0
    total_tokens = 0
    exact_steps = 0
    total_steps = 0
    labelled_steps: list[dict[str, Any]] = []
    for step_index, (predicted, target) in enumerate(
        zip(predicted_sequences, target_sequences)
    ):
        if len(predicted) != len(target):
            raise ValueError("predicted and target token sequences must have the same length")
        step_total = 0
        step_correct = 0
        for token_index, (predicted_id, target_id) in enumerate(zip(predicted, target)):
            if int(target_id) < 0:
                continue
            hit = int(predicted_id) == int(target_id)
            step_correct += int(hit)
            step_total += 1
            correct_tokens += int(hit)
            total_tokens += 1
            labelled_steps.append(
                {
                    "step_index": int(step_index),
                    "depth": int(step_index) + 1,
                    "token_index": int(token_index),
                    "predicted": int(predicted_id),
                    "target": int(target_id),
                    "correct": bool(hit),
                }
            )
        if step_total:
            exact_steps += int(step_correct == step_total)
            total_steps += 1
    return {
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "token_accuracy": float(correct_tokens) / float(total_tokens)
        if total_tokens
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
        "trace_exact": bool(total_steps and exact_steps == total_steps),
        "labelled_steps": labelled_steps,
    }


def _prepare_prompt(tokenizer: Any, prompt: str, *, max_length: int, device: str):
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def _family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "unknown")


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(record["total_tokens"]) for record in records)
    correct_tokens = sum(int(record["correct_tokens"]) for record in records)
    total_steps = sum(int(record["total_steps"]) for record in records)
    exact_steps = sum(int(record["exact_steps"]) for record in records)
    exact_rows = sum(int(bool(record["trace_exact"])) for record in records)
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records))
        if records
        else 0.0,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "token_accuracy": float(correct_tokens) / float(total_tokens)
        if total_tokens
        else 0.0,
        "exact_steps": exact_steps,
        "total_steps": total_steps,
        "step_exact_accuracy": float(exact_steps) / float(total_steps)
        if total_steps
        else 0.0,
    }


def evaluate_rows(
    *,
    config: str,
    checkpoint: str,
    data_jsonl: str,
    out_json: str | None = None,
    tokenizer_model_id: str = "Qwen/Qwen3.5-2B-Base",
    max_length: int | None = None,
    core_steps: int = 8,
    max_cases: int = 0,
    max_target_tokens: int = 32,
    disable_transition_state: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import load_initial_checkpoint

    cfg = load_config(config)
    device = "cuda" if torch.cuda.is_available() and cfg.train.device in {"auto", "cuda"} else "cpu"
    rows = load_rows(data_jsonl)
    if int(max_cases) > 0:
        rows = rows[: int(max_cases)]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QTRMMultimodalModel(cfg.model).to(device)
    missing, unexpected = load_initial_checkpoint(model, checkpoint, map_location=device)
    donor = QwenDonorAdapter(cfg.donor)
    model.eval()
    max_len = int(max_length or cfg.train.seq_len)
    records: list[dict[str, Any]] = []
    old_outer_steps = int(model.cfg.outer_steps)
    model.cfg.outer_steps = int(core_steps)
    try:
        with torch.no_grad():
            for row in rows:
                input_ids, attention_mask = _prepare_prompt(
                    tokenizer,
                    str(row["prompt"]),
                    max_length=max_len,
                    device=device,
                )
                donor_out = donor.encode_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=False,
                )
                with torch.amp.autocast(
                    "cuda",
                    enabled=(cfg.train.use_amp and device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        text_states=donor_out["text_states"].detach().to(device),
                        disable_transition_state=bool(disable_transition_state),
                    )
                logits = outputs["transition_value_state_logits"]
                token_count = min(int(max_target_tokens), int(logits.shape[2]))
                predicted = [
                    [int(token_id) for token_id in sequence[:token_count]]
                    for sequence in predicted_value_sequences_from_logits(
                        logits[:, :, :token_count, :]
                    )
                ]
                targets = value_state_targets_from_row(
                    row,
                    num_steps=len(predicted),
                    max_target_tokens=token_count,
                )
                score = score_value_sequences(
                    predicted_sequences=predicted,
                    target_sequences=targets,
                )
                records.append(
                    {
                        "id": row.get("id", ""),
                        "task_family": _family(row),
                        "predicted_sequences": predicted,
                        "target_sequences": targets,
                        **score,
                    }
                )
    finally:
        model.cfg.outer_steps = old_outer_steps
    summary = _summarize_records(records)
    result = {
        "config": config,
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
        "core_steps": int(core_steps),
        "max_target_tokens": int(max_target_tokens),
        "disable_transition_state": bool(disable_transition_state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "summary": summary,
        "records": records,
    }
    if out_json:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate compact digit/comma/minus QTRM value-state predictions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-target-tokens", type=int, default=32)
    parser.add_argument("--disable-transition-state", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_rows(
        config=args.config,
        checkpoint=args.checkpoint,
        data_jsonl=args.data_jsonl,
        out_json=args.out_json or None,
        tokenizer_model_id=args.tokenizer_model_id,
        max_length=args.max_length,
        core_steps=args.core_steps,
        max_cases=args.max_cases,
        max_target_tokens=args.max_target_tokens,
        disable_transition_state=args.disable_transition_state,
    )
    summary = result["summary"]
    print(
        "value-state eval: "
        f"rows={summary['rows']} "
        f"exact={summary['exact_rows']}/{summary['rows']} "
        f"step_exact={summary['step_exact_accuracy']:.4f} "
        f"token_acc={summary['token_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
