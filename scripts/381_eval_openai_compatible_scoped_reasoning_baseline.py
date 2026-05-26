#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible chat API on the M6 scoped reasoning suite."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request


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


def normalize_single_digit_answer(text: str) -> str:
    """Extract the final standalone digit from model output."""
    stripped = str(text).strip()
    if re.fullmatch(r"\d", stripped):
        return stripped
    matches = re.findall(r"(?<!\d)(\d)(?!\d)", stripped)
    if matches:
        return matches[-1]
    return ""


def normalize_exact_text_answer(text: str) -> str:
    stripped = str(text).strip()
    if "<|box_end|>" in stripped:
        stripped = stripped.split("<|box_end|>", 1)[0].strip()
    stripped = re.sub(r"^\s*answer\s*:\s*", "", stripped, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stripped).strip()


def extract_boxed_expressions(text: str) -> list[str]:
    expressions: list[str] = []
    marker = r"\boxed{"
    value = str(text)
    index = 0
    while True:
        start = value.find(marker, index)
        if start < 0:
            break
        content_start = start + len(marker)
        depth = 1
        position = content_start
        while position < len(value) and depth > 0:
            char = value[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            position += 1
        if depth == 0:
            expressions.append(value[content_start : position - 1])
            index = position
        else:
            index = content_start
    return expressions


def normalize_boxed_text_answer(text: str) -> str:
    boxed = extract_boxed_expressions(str(text))
    if boxed:
        return re.sub(r"\s+", " ", boxed[-1]).strip()
    return normalize_exact_text_answer(text)


def normalize_answer(text: str, *, answer_format: str) -> str:
    if str(answer_format) == "boxed_text":
        return normalize_boxed_text_answer(text)
    if str(answer_format) == "exact_text":
        return normalize_exact_text_answer(text)
    if str(answer_format) == "single_digit":
        return normalize_single_digit_answer(text)
    return normalize_two_digit_answer(text)


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


def _api_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def chat_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _api_url(base_url, "/chat/completions"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(int(retries) + 1):
        try:
            with request.urlopen(req, timeout=float(timeout)) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            choice = data["choices"][0]
            if isinstance(choice.get("message"), dict):
                return str(choice["message"].get("content", ""))
            return str(choice.get("text", ""))
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= int(retries):
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"chat completion failed after {int(retries) + 1} attempts: {last_error}")


def evaluate_api(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_suite(args.suite_jsonl, max_cases=int(args.max_cases))
    scored_rows = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        raw = chat_completion(
            base_url=str(args.base_url),
            model=str(args.model),
            prompt=str(row["qwen_prompt"]),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            timeout=float(args.timeout),
            retries=int(args.retries),
        )
        pred = normalize_answer(raw, answer_format=str(args.answer_format))
        gold = normalize_answer(str(row["answer_text"]), answer_format=str(args.answer_format))
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
        if int(args.log_every) > 0 and index % int(args.log_every) == 0:
            metrics = score_rows(scored_rows)
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "progress": index,
                        "cases": len(rows),
                        "generation_exact": metrics["generation_exact"],
                        "elapsed_sec": round(elapsed, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    metrics = score_rows(scored_rows)
    suite_id = str(rows[0]["suite_id"])
    prompt_protocol = str(rows[0]["prompt_protocol"])
    report = {
        "model": str(args.model_label),
        "model_path": str(args.model_path),
        "base_url": str(args.base_url),
        "suite_id": suite_id,
        "prompt_protocol": prompt_protocol,
        "score": metrics["generation_exact"],
        "cases": metrics["cases"],
        "scorer": (
            "final boxed text exact match"
            if str(args.answer_format) == "boxed_text"
            else
            "normalized exact text match"
            if str(args.answer_format) == "exact_text"
            else "final standalone single-digit exact match"
            if str(args.answer_format) == "single_digit"
            else "first standalone two-digit exact match"
        ),
        "metrics": metrics,
        "accepted": True,
        "comparison_role": str(args.comparison_role),
        "quantization": str(args.quantization),
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
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--model-label", default="Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--quantization", default="UD-Q4_K_XL GGUF")
    parser.add_argument("--comparison-role", default="M6 proxy baseline")
    parser.add_argument("--suite-jsonl", default="local_eval/m6_scoped_raw_reasoning_suite/cases.jsonl")
    parser.add_argument("--out-json", default="local_eval/m6_qwen36_mtp_proxy_baseline/report.json")
    parser.add_argument("--out-jsonl", default="local_eval/m6_qwen36_mtp_proxy_baseline/predictions.jsonl")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--answer-format",
        choices=("two_digit", "single_digit", "exact_text", "boxed_text"),
        default="two_digit",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=32)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_api(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
