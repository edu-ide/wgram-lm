#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def answer_token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(str(text).strip(), add_special_tokens=False)
    return [int(token_id) for token_id in ids]


def first_token_targets_from_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_steps: int,
) -> list[int]:
    depth_targets = row.get("depth_targets")
    if not isinstance(depth_targets, dict):
        raise ValueError("row must contain depth_targets")
    targets: list[int] = []
    for depth in range(1, int(num_steps) + 1):
        value = depth_targets.get(str(depth))
        if value is None:
            targets.append(-100)
            continue
        ids = answer_token_ids(tokenizer, str(value))
        targets.append(int(ids[0]) if ids else -100)
    return targets


def predicted_tokens_from_logits(logits: Any) -> list[int]:
    if logits.ndim != 3 or int(logits.shape[1]) == 0 or int(logits.shape[-1]) == 0:
        return []
    return logits.detach().float().argmax(dim=-1)[0].cpu().tolist()


def score_token_predictions(
    *,
    predicted_tokens: list[int],
    target_tokens: list[int],
) -> dict[str, Any]:
    if len(predicted_tokens) != len(target_tokens):
        raise ValueError("predicted_tokens and target_tokens must have the same length")
    correct = 0
    total = 0
    labelled_steps: list[dict[str, Any]] = []
    for index, (predicted, target) in enumerate(zip(predicted_tokens, target_tokens)):
        if int(target) < 0:
            continue
        hit = int(predicted) == int(target)
        correct += int(hit)
        total += 1
        labelled_steps.append(
            {
                "step_index": index,
                "depth": index + 1,
                "predicted": int(predicted),
                "target": int(target),
                "correct": bool(hit),
            }
        )
    return {
        "correct_steps": correct,
        "total_steps": total,
        "step_accuracy": float(correct) / float(total) if total else 0.0,
        "trace_exact": bool(total and correct == total),
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
    total_steps = sum(int(record["total_steps"]) for record in records)
    correct_steps = sum(int(record["correct_steps"]) for record in records)
    exact_rows = sum(int(bool(record["trace_exact"])) for record in records)
    by_family: dict[str, dict[str, Any]] = {}
    for record in records:
        family = str(record["task_family"])
        bucket = by_family.setdefault(
            family,
            {"rows": 0, "exact_rows": 0, "correct_steps": 0, "total_steps": 0},
        )
        bucket["rows"] += 1
        bucket["exact_rows"] += int(bool(record["trace_exact"]))
        bucket["correct_steps"] += int(record["correct_steps"])
        bucket["total_steps"] += int(record["total_steps"])
    for bucket in by_family.values():
        bucket["step_accuracy"] = (
            float(bucket["correct_steps"]) / float(bucket["total_steps"])
            if int(bucket["total_steps"])
            else 0.0
        )
        bucket["trace_exact_accuracy"] = (
            float(bucket["exact_rows"]) / float(bucket["rows"])
            if int(bucket["rows"])
            else 0.0
        )
    return {
        "rows": len(records),
        "exact_rows": exact_rows,
        "trace_exact_accuracy": float(exact_rows) / float(len(records)) if records else 0.0,
        "correct_steps": correct_steps,
        "total_steps": total_steps,
        "step_accuracy": float(correct_steps) / float(total_steps) if total_steps else 0.0,
        "by_family": by_family,
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
    disable_transition_state: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter
    from qtrm_mm.training.train import load_initial_checkpoint

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
    if missing:
        print(f"[init] missing keys: {len(missing)}")
    if unexpected:
        print(f"[init] unexpected keys: {len(unexpected)}")
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
                logits = outputs["transition_state_text_logits"]
                predicted = predicted_tokens_from_logits(logits)
                targets = first_token_targets_from_row(
                    tokenizer,
                    row,
                    num_steps=len(predicted),
                )
                score = score_token_predictions(
                    predicted_tokens=[int(value) for value in predicted],
                    target_tokens=targets,
                )
                records.append(
                    {
                        "id": row.get("id", ""),
                        "task_family": _family(row),
                        "predicted_tokens": [int(value) for value in predicted],
                        "target_tokens": targets,
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
        description="Evaluate QTRM transition-state text first-token predictions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=0)
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
        max_length=args.max_length or None,
        core_steps=args.core_steps,
        max_cases=args.max_cases,
        disable_transition_state=args.disable_transition_state,
    )
    summary = result["summary"]
    print(
        "transition-state text eval: "
        f"rows={summary['rows']} "
        f"exact={summary['exact_rows']}/{summary['rows']} "
        f"step_acc={summary['step_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
