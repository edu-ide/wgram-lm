#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def parse_core_steps(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        steps = [int(step) for step in value]
    if not steps:
        raise ValueError("core steps must not be empty")
    if any(step <= 0 for step in steps):
        raise ValueError("core steps must be positive")
    return steps


def transition_metrics_from_tensors(pred, target, mask) -> dict[str, Any]:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes")
    if pred.ndim != 3:
        raise ValueError("pred and target must have shape [batch, steps, dim]")
    if pred.numel() == 0:
        return {"transition_mse": 0.0, "transition_count": 0}

    per_transition = (pred.float() - target.float()).pow(2).mean(dim=-1)
    if mask is None:
        valid = per_transition.new_ones(per_transition.shape, dtype=bool)
    else:
        valid = mask.to(device=per_transition.device, dtype=bool)
    count = int(valid.sum().detach().cpu().item())
    if count == 0:
        return {"transition_mse": 0.0, "transition_count": 0}
    mse = per_transition.masked_select(valid).mean()
    return {
        "transition_mse": float(mse.detach().cpu().item()),
        "transition_count": count,
    }


def load_answer_hits(path: str | Path | None) -> dict[tuple[str, str], bool]:
    if not path:
        return {}
    hits: dict[tuple[str, str], bool] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hits[(str(row["id"]), str(row["mode"]))] = bool(row.get("hit"))
    return hits


def pearson_correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = [float(value) for value in xs]
    y = [float(value) for value in ys]
    if len(x) != len(y):
        raise ValueError("xs and ys must have the same length")
    if len(x) < 2:
        return 0.0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    x_var = sum(value * value for value in x_centered)
    y_var = sum(value * value for value in y_centered)
    if x_var == 0.0 or y_var == 0.0:
        return 0.0
    cov = sum(a * b for a, b in zip(x_centered, y_centered))
    return cov / ((x_var * y_var) ** 0.5)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for record in records:
        mode = str(record["mode"])
        bucket = by_mode.setdefault(
            mode,
            {
                "count": 0,
                "transition_count": 0,
                "transition_mse_sum": 0.0,
                "hit_count": 0,
                "answer_eval_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["transition_count"] += int(record.get("transition_count", 0))
        bucket["transition_mse_sum"] += float(record.get("transition_mse", 0.0))
        if isinstance(record.get("hit"), bool):
            bucket["answer_eval_count"] += 1
            bucket["hit_count"] += int(bool(record["hit"]))

    for bucket in by_mode.values():
        count = max(1, int(bucket["count"]))
        answer_count = int(bucket["answer_eval_count"])
        bucket["mean_transition_mse"] = float(bucket["transition_mse_sum"]) / count
        bucket["hit_rate"] = (
            float(bucket["hit_count"]) / answer_count if answer_count else None
        )
        del bucket["transition_mse_sum"]

    paired = [
        record
        for record in records
        if isinstance(record.get("hit"), bool) and int(record.get("transition_count", 0)) > 0
    ]
    return {
        "total_records": len(records),
        "transition_records": sum(1 for r in records if int(r.get("transition_count", 0)) > 0),
        "answer_eval_records": sum(1 for r in records if isinstance(r.get("hit"), bool)),
        "transition_mse_hit_pearson": pearson_correlation(
            [float(record["transition_mse"]) for record in paired],
            [1.0 if bool(record["hit"]) else 0.0 for record in paired],
        ),
        "by_mode": by_mode,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LeWM-style recursive-core transition prediction quality."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--answer-eval-jsonl", default="")
    parser.add_argument("--checkpoint-label", default="")
    parser.add_argument("--core-steps", default="1,2,4,8")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def _select_device(cfg_device: str, requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def _load_cases(path: str | Path, *, max_cases: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"case-{line_no}")
            if row.get("evidence"):
                raise ValueError(f"{path}:{line_no}: transition eval expects no hidden evidence")
            if not row.get("prompt") and not row.get("question"):
                raise ValueError(f"{path}:{line_no}: missing prompt/question")
            rows.append(row)
            if max_cases is not None and len(rows) >= int(max_cases):
                break
    return rows


def _prepare_inputs(tokenizer, text: str, max_length: int, device: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in enc.items()}


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    from wgram_lm.config import load_config
    from wgram_lm.wgram_model import QTRMMultimodalModel
    from wgram_lm.qwen_donor import QwenDonorAdapter
    from wgram_lm.training.train import build_core_world_model_actions

    cfg = load_config(args.config)
    if not bool(cfg.model.core_world_model_enabled):
        raise ValueError("transition eval requires model.core_world_model_enabled=true")
    device = _select_device(cfg.train.device, args.device)
    max_length = int(args.max_length or cfg.train.seq_len)
    core_steps_list = parse_core_steps(args.core_steps)
    cases = _load_cases(args.cases, max_cases=args.max_cases)
    answer_hits = load_answer_hits(args.answer_eval_jsonl)
    label = args.checkpoint_label or Path(args.checkpoint).parent.name

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}", file=sys.stderr)
    model = model.to(device).eval()
    donor = QwenDonorAdapter(cfg.donor)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out.open("w", encoding="utf-8") as f, torch.no_grad(), redirect_stdout(sys.stderr):
        for core_steps in core_steps_list:
            mode = f"qtrm_core_steps_{core_steps}_no_evidence"
            old_outer_steps = int(model.cfg.outer_steps)
            model.cfg.outer_steps = int(core_steps)
            try:
                for case in cases:
                    prompt = str(case.get("prompt") or case.get("question") or "")
                    inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                    encoded = donor.encode_inputs(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_logits=False,
                    )
                    actions = build_core_world_model_actions(
                        {"input_ids": input_ids},
                        num_steps=int(core_steps),
                        num_actions=int(model.cfg.num_actions),
                        device=device,
                    )
                    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            text_states=encoded["text_states"].to(device),
                            core_world_model_actions=actions,
                        )
                    metrics = transition_metrics_from_tensors(
                        outputs["core_world_model_pred"],
                        outputs["core_world_model_target"],
                        outputs["core_world_model_mask"],
                    )
                    hit_key = (str(case["id"]), mode)
                    hit = answer_hits.get(hit_key)
                    record = {
                        "id": case["id"],
                        "checkpoint_label": label,
                        "checkpoint": args.checkpoint,
                        "mode": mode,
                        "core_steps": int(core_steps),
                        "category": case.get("category", "uncategorized"),
                        "task_family": case.get("task_family", case.get("category", "uncategorized")),
                        "answer_eval_found": hit_key in answer_hits,
                        "hit": hit,
                        **metrics,
                    }
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            finally:
                model.cfg.outer_steps = old_outer_steps

    summary = summarize_records(records)
    summary["checkpoint_label"] = label
    summary["checkpoint"] = args.checkpoint
    summary["config"] = args.config
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return records, summary


def main() -> None:
    args = build_arg_parser().parse_args()
    records, summary = run_eval(args)
    print(f"wrote {len(records)} records to {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
