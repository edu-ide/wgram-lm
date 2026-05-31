#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def _load_raw_eval_module():
    path = Path(__file__).with_name("192_eval_raw_intelligence.py")
    spec = importlib.util.spec_from_file_location("qtrm_raw_eval_192", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt(case: dict[str, Any], style: str) -> str:
    style = str(style)
    if style == "raw":
        return str(case.get("prompt") or case.get("question") or "")
    question = str(case.get("question") or case.get("prompt") or "")
    if style == "minimal":
        return f"Answer with only the final answer. Do not write reasoning.\nQuestion: {question}"
    if style == "numeric_strict":
        return (
            "Return exactly one integer and nothing else.\n"
            f"Problem: {question}\n"
            "Final answer:"
        )
    raise ValueError(f"unknown prompt style: {style}")


def encode_prompt(tokenizer, prompt: str, *, chat: bool, device: str):
    if chat:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if isinstance(ids, dict):
            return {k: v.to(device) for k, v in ids.items() if hasattr(v, "to")}
        if hasattr(ids, "input_ids"):
            out = {"input_ids": ids.input_ids.to(device)}
            attention_mask = getattr(ids, "attention_mask", None)
            out["attention_mask"] = (
                attention_mask.to(device)
                if attention_mask is not None
                else torch.ones_like(ids.input_ids).to(device)
            )
            return out
        return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)


def _visible_reasoning_bad_words(tokenizer) -> list[list[int]]:
    out: list[list[int]] = []
    for marker in ("<think>", "</think>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if ids:
            out.append([int(token_id) for token_id in ids])
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen donor-only greedy renderer baseline.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--style",
        action="append",
        default=None,
        choices=["raw", "minimal", "numeric_strict"],
        help="Prompt style. Can repeat.",
    )
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    return parser


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from wgram_lm.qwen_donor import _build_4bit_quantization_config

    args = build_arg_parser().parse_args()
    raw_eval = _load_raw_eval_module()
    device = "cuda" if torch.cuda.is_available() and args.device in {"auto", "cuda"} else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }
    if bool(args.load_in_4bit):
        qcfg = _build_4bit_quantization_config(True)
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kwargs)
    model.eval()
    cases = raw_eval.load_cases(args.cases, max_cases=args.max_cases)
    styles = args.style or ["raw", "minimal", "numeric_strict"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    records = []
    bad_words_ids = (
        _visible_reasoning_bad_words(tokenizer)
        if bool(args.suppress_visible_reasoning_tokens)
        else None
    )
    with out_path.open("w", encoding="utf-8") as f:
        for style in styles:
            hits = 0
            exact = 0
            for case in cases:
                prompt = build_prompt(case, style)
                encoded = encode_prompt(tokenizer, prompt, chat=bool(args.chat), device=device)
                input_len = int(encoded["input_ids"].shape[1])
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bad_words_ids=bad_words_ids,
                    )
                completion = tokenizer.decode(
                    generated[0, input_len:],
                    skip_special_tokens=True,
                ).strip()
                record = raw_eval.score_case_record(
                    case,
                    mode=f"donor_only_{style}_{'chat' if args.chat else 'plain'}",
                    completion=completion,
                    runtime={"disable_core": True, "memoryos_used": False, "retrieval_used": False},
                    generated_tokens=int(generated.shape[1] - input_len),
                )
                record["model_id"] = args.model_id
                record["prompt_style"] = style
                record["chat_template"] = bool(args.chat)
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                hits += int(bool(record["hit"]))
                exact += int(bool(record["exact_match"] or record["normalized_exact"]))
            summaries[style] = {
                "hits": hits,
                "exact": exact,
                "total": len(cases),
                "accuracy": float(hits / max(1, len(cases))),
                "exact_accuracy": float(exact / max(1, len(cases))),
            }
    report = {
        "decision": "accepted_renderer_donor_baseline" if any(v["hits"] > 0 for v in summaries.values()) else "rejected",
        "model_id": args.model_id,
        "chat_template": bool(args.chat),
        "summaries": summaries,
        "out": str(out_path),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
