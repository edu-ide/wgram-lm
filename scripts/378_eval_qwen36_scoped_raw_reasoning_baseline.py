#!/usr/bin/env python3
"""Evaluate Qwen3.6-27B on the M6 scoped raw-reasoning suite."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize_two_digit_answer(text: str) -> str:
    """Extract the first standalone two-digit answer from model output."""
    stripped = str(text).strip()
    if re.fullmatch(r"\d{1,2}", stripped):
        return f"{int(stripped):02d}"
    match = re.search(r"(?<!\d)(\d{2})(?!\d)", stripped)
    if match:
        return match.group(1)
    match = re.search(r"(?<!\d)(\d)(?!\d)", stripped)
    if match:
        return f"{int(match.group(1)):02d}"
    return ""


def load_suite(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"suite row must be object at {path}:{line_no}")
        for key in ("suite_id", "prompt_protocol", "case_id", "qwen_prompt", "answer_text"):
            if key not in row:
                raise ValueError(f"suite row missing {key} at {path}:{line_no}")
        rows.append(row)
        if int(max_cases) > 0 and len(rows) >= int(max_cases):
            break
    if not rows:
        raise ValueError(f"suite is empty: {path}")
    return rows


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    hits = sum(1 for row in rows if bool(row.get("exact", False)))
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row.get("family", "unknown"))
        bucket = by_family.setdefault(family, {"hits": 0, "total": 0})
        bucket["hits"] += int(bool(row.get("exact", False)))
        bucket["total"] += 1
    return {
        "hits": hits,
        "cases": total,
        "generation_exact": float(hits / max(1, total)),
        "by_family": {
            family: {
                "hits": values["hits"],
                "total": values["total"],
                "generation_exact": float(values["hits"] / max(1, values["total"])),
            }
            for family, values in sorted(by_family.items())
        },
    }


def build_chat_input(tokenizer, prompt: str, *, use_chat_template: bool) -> str:
    if bool(use_chat_template) and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return str(prompt)


def generate_answer(
    *,
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    use_chat_template: bool,
) -> str:
    import torch

    text = build_chat_input(tokenizer, prompt, use_chat_template=bool(use_chat_template))
    encoded = tokenizer(text, return_tensors="pt")
    if str(device) != "auto":
        encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = int(encoded["input_ids"].shape[-1])
    new_tokens = output[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_suite(args.suite_jsonl, max_cases=int(args.max_cases))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    if str(args.device) == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), **model_kwargs)
    if str(args.device) != "auto":
        model = model.to(torch.device(str(args.device)))
    model.eval()

    scored_rows = []
    for row in rows:
        raw = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=str(row["qwen_prompt"]),
            device=str(args.device),
            max_new_tokens=int(args.max_new_tokens),
            use_chat_template=bool(args.use_chat_template),
        )
        pred = normalize_two_digit_answer(raw)
        gold = normalize_two_digit_answer(str(row["answer_text"]))
        scored = dict(row)
        scored.update(
            {
                "raw_completion": raw,
                "pred_answer": pred,
                "gold_answer": gold,
                "exact": bool(pred == gold),
            }
        )
        scored_rows.append(scored)
    metrics = score_rows(scored_rows)
    suite_id = str(rows[0]["suite_id"])
    prompt_protocol = str(rows[0]["prompt_protocol"])
    report = {
        "model": "Qwen/Qwen3.6-27B",
        "model_path": str(args.model_path),
        "suite_id": suite_id,
        "prompt_protocol": prompt_protocol,
        "score": metrics["generation_exact"],
        "cases": metrics["cases"],
        "scorer": "first standalone two-digit exact match",
        "metrics": metrics,
        "accepted": True,
    }
    out_rows = Path(args.out_jsonl)
    out_report = Path(args.out_json)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_rows.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--suite-jsonl",
        default="local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl",
    )
    parser.add_argument("--out-json", default="local_eval/m6_qwen36_scoped_baseline/report.json")
    parser.add_argument("--out-jsonl", default="local_eval/m6_qwen36_scoped_baseline/predictions.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_model(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
