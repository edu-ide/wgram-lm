#!/usr/bin/env python3
"""Materialize the first M7 public reasoning suite.

The initial M7 target is MMLU-Pro because it is public through the Hugging Face
Dataset Viewer and has a Qwen3.6-27B target in the local milestone contract.
This script writes deterministic JSONL cases and metadata. It does not score a
model and does not claim parity.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Any


DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_CONFIG = "default"
MMLU_PRO_SPLIT = "validation"
OPTION_LETTERS = tuple("ABCDEFGHIJ")


def dataset_viewer_get(
    endpoint: str,
    params: dict[str, Any],
    *,
    retries: int = 6,
    backoff_sec: float = 4.0,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_VIEWER_BASE}/{endpoint.lstrip('/')}?{query}"
    last_error: Exception | None = None
    for attempt in range(int(retries) + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt >= int(retries):
                raise
            retry_after = exc.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else float(backoff_sec) * (attempt + 1)
            time.sleep(min(sleep_for, 60.0))
        except URLError as exc:
            last_error = exc
            if attempt >= int(retries):
                raise
            time.sleep(min(float(backoff_sec) * (attempt + 1), 60.0))
    raise RuntimeError(f"Dataset Viewer request failed: {last_error}")


def fetch_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    payload = dataset_viewer_get(
        "/rows",
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": int(offset),
            "length": int(length),
        },
    )
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Dataset Viewer /rows response did not contain a row list")
    return rows


def fetch_dataset_rows_with_datasets(
    *,
    dataset: str,
    config: str,
    split: str,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required for --backend datasets") from exc
    loaded = load_dataset(dataset, config, split=split)
    return [
        {"row_idx": index, "row": dict(row)}
        for index, row in enumerate(loaded)
    ]


def option_lines(options: list[str]) -> list[str]:
    if len(options) > len(OPTION_LETTERS):
        raise ValueError(f"too many options for fixed option alphabet: {len(options)}")
    return [f"{OPTION_LETTERS[index]}. {option}" for index, option in enumerate(options)]


def format_mmlu_pro_prompt(row: dict[str, Any]) -> str:
    options = row.get("options", [])
    if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
        raise ValueError("MMLU-Pro row must contain string-list options")
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("MMLU-Pro row must contain a question")
    body = "\n".join(
        [
            "Answer the following MMLU-Pro multiple-choice question.",
            "Return only one option letter, with no explanation.",
            "",
            f"Question: {question}",
            "Options:",
            *option_lines(options),
            "",
            "Answer:",
        ]
    )
    return f"User: {body}\nAssistant:"


def row_to_case(
    row_idx: int,
    row: dict[str, Any],
    *,
    dataset: str = MMLU_PRO_DATASET,
    config: str = MMLU_PRO_CONFIG,
    split: str = MMLU_PRO_SPLIT,
) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip().upper()
    answer_index = int(row.get("answer_index", -1))
    if not answer and 0 <= answer_index < len(OPTION_LETTERS):
        answer = OPTION_LETTERS[answer_index]
    if answer not in OPTION_LETTERS:
        raise ValueError(f"unsupported MMLU-Pro answer at row {row_idx}: {answer!r}")
    options = row.get("options", [])
    if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
        raise ValueError(f"invalid options at row {row_idx}")
    return {
        "benchmark_id": "mmlu_pro",
        "dataset": str(dataset),
        "config": str(config),
        "split": str(split),
        "case_id": f"mmlu-pro-{split}-{int(row.get('question_id', row_idx)):06d}",
        "row_idx": int(row_idx),
        "question_id": row.get("question_id", row_idx),
        "category": str(row.get("category", "unknown")),
        "src": str(row.get("src", "")),
        "question": str(row.get("question", "")),
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "qtrm_prompt": format_mmlu_pro_prompt(row),
        "scorer": "exact option-letter match",
    }


def materialize_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    remaining = int(args.max_cases)
    if remaining <= 0:
        raise ValueError("--max-cases must be positive")
    offset = int(args.offset)
    cases: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    categories = {
        item.strip()
        for item in str(args.categories).split(",")
        if item.strip()
    }
    category_quota = int(args.category_quota)
    max_scan_rows = int(args.max_scan_rows)
    if str(args.backend) == "datasets":
        rows = fetch_dataset_rows_with_datasets(
            dataset=str(args.dataset),
            config=str(args.config),
            split=str(args.split),
        )
        for item in rows[offset : max_scan_rows if max_scan_rows > 0 else None]:
            row_idx = int(item.get("row_idx", offset))
            row = item.get("row", {})
            if not isinstance(row, dict):
                raise ValueError(f"dataset row payload is not an object: {row_idx}")
            category = str(row.get("category", "unknown"))
            if categories and category not in categories:
                continue
            if category_quota > 0 and category_counts.get(category, 0) >= category_quota:
                continue
            cases.append(
                row_to_case(
                    row_idx,
                    row,
                    dataset=str(args.dataset),
                    config=str(args.config),
                    split=str(args.split),
                )
            )
            category_counts[category] = category_counts.get(category, 0) + 1
            remaining -= 1
            if remaining <= 0:
                break
        if len(cases) < int(args.min_cases):
            raise ValueError(f"only materialized {len(cases)} cases, below min {args.min_cases}")
        return cases

    while remaining > 0:
        if max_scan_rows > 0 and offset >= max_scan_rows:
            break
        page = fetch_rows(
            dataset=str(args.dataset),
            config=str(args.config),
            split=str(args.split),
            offset=offset,
            length=100 if category_quota > 0 else min(100, remaining),
        )
        if not page:
            break
        for item in page:
            row_idx = int(item.get("row_idx", offset))
            row = item.get("row", {})
            if not isinstance(row, dict):
                raise ValueError(f"Dataset Viewer row payload is not an object: {row_idx}")
            category = str(row.get("category", "unknown"))
            if categories and category not in categories:
                offset += 1
                continue
            if category_quota > 0 and category_counts.get(category, 0) >= category_quota:
                offset += 1
                continue
            cases.append(
                row_to_case(
                    row_idx,
                    row,
                    dataset=str(args.dataset),
                    config=str(args.config),
                    split=str(args.split),
                )
            )
            category_counts[category] = category_counts.get(category, 0) + 1
            remaining -= 1
            offset += 1
            if remaining <= 0:
                break
        if len(page) < 100:
            break
    if len(cases) < int(args.min_cases):
        raise ValueError(f"only materialized {len(cases)} cases, below min {args.min_cases}")
    return cases


def write_outputs(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    out_jsonl = Path(args.out_jsonl)
    out_report = Path(args.out_report)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )
    by_category: dict[str, int] = {}
    for row in cases:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    report = {
        "status": "complete",
        "decision": "accepted_m7_public_suite_materialized",
        "accepted": True,
        "benchmark_id": "mmlu_pro",
        "dataset": str(args.dataset),
        "config": str(args.config),
        "split": str(args.split),
        "source_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "backend": str(args.backend),
        "cases": len(cases),
        "out_jsonl": str(out_jsonl),
        "by_category": dict(sorted(by_category.items())),
        "scorer": "exact option-letter match",
        "limitations": [
            "This is a public-suite materialization artifact, not a parity claim.",
            "M7 requires QTRM-Native scoring on this suite and comparison to the Qwen3.6 target.",
        ],
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=MMLU_PRO_DATASET)
    parser.add_argument("--config", default=MMLU_PRO_CONFIG)
    parser.add_argument("--split", default=MMLU_PRO_SPLIT)
    parser.add_argument("--backend", choices=("datasets", "viewer"), default="datasets")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--categories", default="")
    parser.add_argument(
        "--category-quota",
        type=int,
        default=0,
        help="Maximum cases per category. Use >0 for category-balanced public subsets.",
    )
    parser.add_argument(
        "--max-scan-rows",
        type=int,
        default=20000,
        help="Maximum Dataset Viewer row offset to scan while applying category quotas.",
    )
    parser.add_argument("--out-jsonl", default="local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl")
    parser.add_argument("--out-report", default="local_eval/m7_public_reasoning_suite/report.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = materialize_cases(args)
    print(json.dumps(write_outputs(args, cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
