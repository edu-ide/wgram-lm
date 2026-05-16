#!/usr/bin/env python3
"""Build offline language-teacher artifacts for QTRM-native bootstrap.

This script may load a Qwen/Qwopus teacher, but only to write JSONL artifacts.
The produced file is consumed later by the donorless QTRM-native bootstrap
runner via --teacher-jsonl. The teacher is never part of final inference.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


def load_language_bootstrap_module():
    path = Path(__file__).with_name("354_train_qtrm_native_language_bootstrap.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_language_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_bootstrap = load_language_bootstrap_module()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def max_char_run_fraction(text: str) -> float:
    if not text:
        return 1.0
    max_run = 1
    current = 1
    for left, right in zip(text, text[1:]):
        if left == right:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return float(max_run / max(1, len(text)))


def iter_seed_texts(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    if bool(args.include_builtin):
        for index, text in enumerate(_bootstrap.TINY_STORIES):
            yield f"builtin:tiny:{index}", str(text)
        for index, text in enumerate(_bootstrap.TEXTBOOK_SNIPPETS):
            yield f"builtin:textbook:{index}", str(text)
    if args.text_file:
        yield f"file:{args.text_file}", Path(args.text_file).read_text(encoding="utf-8")
    for pattern in args.text_glob or []:
        for raw_path in sorted(glob.glob(str(pattern), recursive=True)):
            path = Path(raw_path)
            if path.is_file():
                yield f"file:{path}", path.read_text(encoding="utf-8")
    for path in args.source_jsonl or []:
        for index, text in enumerate(iter_jsonl_texts(path)):
            yield f"jsonl:{path}:{index}", text


def iter_jsonl_texts(path: str | Path) -> Iterable[str]:
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(item, str):
            yield item
            continue
        if not isinstance(item, dict):
            continue
        for key in ("prompt", "text", "question", "instruction", "input"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                yield value
                break


def prepare_seed_text(text: str, *, max_prompt_chars: int) -> str:
    text = normalize_space(text)
    if int(max_prompt_chars) > 0:
        text = text[: int(max_prompt_chars)].rstrip()
    return text


def make_prompt(seed_text: str, *, no_think_switch: bool = True) -> str:
    text = normalize_space(seed_text)
    if not text:
        return ""
    switch = "/no_think\n" if bool(no_think_switch) else ""
    return (
        f"{switch}"
        "Task: continue the educational text.\n"
        "Rules:\n"
        "- Output only the continuation text.\n"
        "- Do not list options.\n"
        "- Do not explain the task.\n"
        "- Do not use markdown bullets.\n"
        "- Keep the style concise, grammatical, and non-repetitive.\n\n"
        f"Text:\n{text}\n\nContinuation:"
    )


def collect_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, text in iter_seed_texts(args):
        seed_text = prepare_seed_text(text, max_prompt_chars=int(args.max_prompt_chars))
        prompt = make_prompt(seed_text, no_think_switch=bool(args.no_think_switch))
        key = normalize_space(seed_text).lower()
        if not seed_text or not prompt or key in seen:
            continue
        seen.add(key)
        rows.append({"source": source, "seed_text": seed_text, "prompt": prompt})
        if len(rows) >= int(args.max_records):
            break
    return rows


def build_quantization_config(load_in_4bit: bool):
    if not bool(load_in_4bit):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - optional env
        raise RuntimeError("bitsandbytes/transformers quantization support is unavailable") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_teacher(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required unless --dry-run is used") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(args.dtype)]
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    quantization_config = build_quantization_config(bool(args.load_in_4bit))
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if args.device_map:
        kwargs["device_map"] = str(args.device_map)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(args.model_id), **kwargs)
        model_type = "AutoModelForCausalLM"
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(str(args.model_id), **kwargs)
        model_type = "AutoModelForImageTextToText"
    if not args.device_map:
        model = model.to(torch.device(args.device))
    model.eval()
    return tokenizer, model, model_type


def reasoning_bad_words_ids(tokenizer, args: argparse.Namespace) -> list[list[int]] | None:
    if not bool(args.suppress_think_tokens):
        return None
    bad_words: list[list[int]] = []
    for phrase in ("<think>", "</think>", "\n<think>", " <think>"):
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        if ids:
            bad_words.append([int(token_id) for token_id in ids])
    return bad_words or None


@torch.no_grad()
def generate_answer(tokenizer, model, args: argparse.Namespace, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_input_tokens))
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "repetition_penalty": float(args.repetition_penalty),
        "bad_words_ids": reasoning_bad_words_ids(tokenizer, args),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if bool(args.do_sample):
        generate_kwargs["temperature"] = float(args.temperature)
        generate_kwargs["top_p"] = float(args.top_p)
    output = model.generate(**inputs, **generate_kwargs)
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = output[0, prompt_len:]
    return str(tokenizer.decode(new_ids, skip_special_tokens=True)).strip()


@torch.no_grad()
def compute_topk_logprobs(
    tokenizer,
    model,
    prompt: str,
    answer: str,
    *,
    top_k: int,
    max_total_tokens: int,
) -> list[dict[str, float | int]]:
    if int(top_k) <= 0:
        return []
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_total_tokens).input_ids
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    if answer_ids.numel() == 0:
        return []
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    if int(full_ids.shape[1]) > int(max_total_tokens):
        full_ids = full_ids[:, -int(max_total_tokens) :]
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)
    logits = model(input_ids=full_ids).logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    start = max(0, int(full_ids.shape[1]) - int(answer_ids.shape[1]) - 1)
    rows: list[dict[str, float | int]] = []
    for position in range(start, int(logprobs.shape[1])):
        values, indices = torch.topk(logprobs[0, position], k=min(int(top_k), int(logprobs.shape[-1])))
        for token_id, logprob in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist()):
            rows.append(
                {
                    "position": int(position),
                    "token_id": int(token_id),
                    "logprob": float(logprob),
                }
            )
    return rows


def dry_answer(prompt: str, *, max_new_tokens: int) -> str:
    words = normalize_space(prompt).split()
    tail = " ".join(words[-min(18, len(words)) :])
    return (tail + " This continuation is clear, short, and non-repetitive.")[: max(20, int(max_new_tokens) * 8)]


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r".*</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def should_keep(answer: str, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    clean = normalize_space(answer)
    if "<think" in clean.lower() or "</think" in clean.lower():
        reasons.append("answer_contains_visible_think")
    if len(clean) < int(args.min_answer_chars):
        reasons.append("answer_too_short")
    if max_char_run_fraction(clean) > float(args.max_run_fraction):
        reasons.append("answer_repetition_run_too_high")
    if len(set(clean)) < int(args.min_unique_chars):
        reasons.append("answer_too_few_unique_chars")
    return not reasons, reasons


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    prompts = collect_prompts(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = model = None
    model_type = "dry-run"
    if not bool(args.dry_run):
        tokenizer, model, model_type = load_teacher(args)

    written = 0
    skipped = 0
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(prompts):
        prompt = item["prompt"]
        seed_text = item["seed_text"]
        if bool(args.dry_run):
            answer = dry_answer(prompt, max_new_tokens=int(args.max_new_tokens))
            topk_logprobs: list[dict[str, float | int]] = []
        else:
            assert tokenizer is not None and model is not None
            answer = generate_answer(tokenizer, model, args, prompt)
            if bool(args.strip_think_blocks):
                answer = strip_think_blocks(answer)
            topk_logprobs = compute_topk_logprobs(
                tokenizer,
                model,
                prompt,
                answer,
                top_k=int(args.top_k_logprobs),
                max_total_tokens=int(args.max_logprob_tokens),
            )
        keep, reasons = should_keep(answer, args)
        if not keep:
            skipped += 1
            continue
        text = f"{seed_text}\n{answer}".strip()
        row: dict[str, Any] = {
            "prompt": prompt,
            "seed_text": seed_text,
            "answer": answer,
            "text": text,
            "teacher_text": text,
            "teacher_model": str(args.model_id) if not bool(args.dry_run) else "dry-run",
            "source": item["source"],
            "index": index,
            "generation": {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "do_sample": bool(args.do_sample),
                "repetition_penalty": float(args.repetition_penalty),
            },
            "quality": {
                "max_run_fraction": max_char_run_fraction(answer),
                "unique_chars": len(set(answer)),
            },
        }
        if topk_logprobs:
            row["topk_logprobs"] = topk_logprobs
        rows.append(row)
        written += 1

    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "out": str(out),
        "model_id": str(args.model_id),
        "model_type": model_type,
        "dry_run": bool(args.dry_run),
        "requested_records": int(args.max_records),
        "prompt_count": len(prompts),
        "written": written,
        "skipped": skipped,
        "top_k_logprobs": int(args.top_k_logprobs),
    }
    report_path = out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline QTRM-native language teacher JSONL.")
    parser.add_argument("--out", default="data/qtrm_native_language_teacher/teacher_text.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-builtin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-file", default="")
    parser.add_argument("--text-glob", action="append", default=[])
    parser.add_argument("--source-jsonl", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=16)
    parser.add_argument("--max-prompt-chars", type=int, default=360)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-think-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress-think-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-think-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k-logprobs", type=int, default=0)
    parser.add_argument("--max-logprob-tokens", type=int, default=768)
    parser.add_argument("--min-answer-chars", type=int, default=24)
    parser.add_argument("--min-unique-chars", type=int, default=8)
    parser.add_argument("--max-run-fraction", type=float, default=0.35)
    return parser


def main() -> None:
    report = build_cache(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
