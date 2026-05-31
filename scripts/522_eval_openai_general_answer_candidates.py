#!/usr/bin/env python3
"""Generate and evaluate general Stage59 answer candidates via an OpenAI API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from wgram_lm.eval.general_answer_interface import (
    answer_aliases,
    answer_kind,
    extract_answer_candidate_text,
    select_candidate,
    summarize_records,
)


PROMPT_KEYS = ("prompt", "qwen_prompt", "question")
ID_KEYS = ("id", "case_id", "example_id", "uid")


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row must be a JSON object at {path}:{line_no}")
        rows.append(row)
        if int(limit) > 0 and len(rows) >= int(limit):
            break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ID_KEYS:
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError(f"row has no id key among {ID_KEYS}: {row}")


def row_prompt(row: dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        if row.get(key):
            return str(row[key])
    raise ValueError(f"row has no prompt key among {PROMPT_KEYS}: {row_id(row)}")


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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.eval_jsonl, limit=int(args.max_cases))
    records: list[dict[str, Any]] = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        raw_completions: list[str] = []
        candidates: list[str] = []
        prompt = row_prompt(row)
        for sample_index in range(int(args.candidate_samples)):
            raw = chat_completion(
                base_url=str(args.base_url),
                model=str(args.model),
                prompt=prompt,
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                timeout=float(args.timeout),
                retries=int(args.retries),
            )
            raw_completions.append(raw)
            candidates.append(extract_answer_candidate_text(raw))
            if int(args.sleep_ms) > 0 and sample_index + 1 < int(args.candidate_samples):
                time.sleep(float(args.sleep_ms) / 1000.0)

        aliases = answer_aliases(row)
        selection = select_candidate(candidates, aliases, selection_mode=args.selection_mode)
        record = {
            "id": row_id(row),
            "task_family": row.get("task_family") or row.get("family") or row.get("category") or "unknown",
            "answer_kind": answer_kind(aliases[0] if aliases else ""),
            "aliases": list(aliases),
            "raw_completions": raw_completions,
            "candidates": candidates,
            "selected": selection.selected,
            "normalized_selected": selection.normalized_selected,
            "selected_index": selection.selected_index,
            "exact": selection.exact,
            "oracle_exact": selection.oracle_exact,
            "oracle_index": selection.oracle_index,
            "selection_mode": selection.selection_mode,
        }
        records.append(record)
        if int(args.log_every) > 0 and index % int(args.log_every) == 0:
            summary = summarize_records(records)
            oracle_hits = sum(1 for item in records if bool(item.get("oracle_exact")))
            print(
                json.dumps(
                    {
                        "progress": index,
                        "total": len(rows),
                        "accuracy": summary["accuracy"],
                        "oracle_accuracy": oracle_hits / max(1, len(records)),
                        "elapsed_sec": round(time.time() - started, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = summarize_records(records)
    oracle_hits = sum(1 for item in records if bool(item.get("oracle_exact")))
    summary.update(
        {
            "stage": "Stage59 OpenAI-compatible general answer candidate eval",
            "model": str(args.model_label),
            "base_url": str(args.base_url),
            "eval_jsonl": str(args.eval_jsonl),
            "candidate_samples": int(args.candidate_samples),
            "selection_mode": str(args.selection_mode),
            "oracle_hits": oracle_hits,
            "oracle_accuracy": float(oracle_hits / max(1, len(records))),
            "plain_language_read": (
                "This is a general-answer mouth baseline. It evaluates generated text/object "
                "answers through the shared Stage59 answer interface, not through modulo-10 "
                "digit/register logic."
            ),
        }
    )
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--model-label", default="openai-compatible-local")
    parser.add_argument("--eval-jsonl", default="data/eval/pure_recursive_solver_trace_all_family_heldout_cases.jsonl")
    parser.add_argument("--out-json", default="local_eval/stage59_openai_general_answer_baseline/report.json")
    parser.add_argument("--out-jsonl", default="local_eval/stage59_openai_general_answer_baseline/predictions.jsonl")
    parser.add_argument("--candidate-samples", type=int, default=1)
    parser.add_argument("--selection-mode", choices=("first", "oracle"), default="first")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=16)
    args = parser.parse_args()
    if int(args.candidate_samples) <= 0:
        raise ValueError("--candidate-samples must be positive")

    result = evaluate(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in result["records"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
