#!/usr/bin/env python3
"""Materialize non-test external MCQ datasets into the QTRM public-MCQ schema.

This is a support script for M7 public benchmark repair. It intentionally uses
non-test splits from external MCQ datasets so they can train or select
checkpoints without touching MMLU-Pro test labels.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


OPTION_LETTERS = "ABCDEFGHIJ"
DATASET_SERVER = "https://datasets-server.huggingface.co"

SOURCE_PRESETS: dict[str, list[dict[str, str]]] = {
    "train": [
        {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "train", "category": "science"},
        {"dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "train", "category": "science"},
        {"dataset": "allenai/openbookqa", "config": "main", "split": "train", "category": "science"},
        {"dataset": "tau/commonsense_qa", "config": "default", "split": "train", "category": "commonsense"},
    ],
    "validation": [
        {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "validation", "category": "science"},
        {"dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "validation", "category": "science"},
        {"dataset": "allenai/openbookqa", "config": "main", "split": "validation", "category": "science"},
        {"dataset": "tau/commonsense_qa", "config": "default", "split": "validation", "category": "commonsense"},
    ],
}


def _read_json_url(url: str, *, retries: int = 3, sleep_seconds: float = 1.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network fallback
            last_error = exc
            if attempt < int(retries):
                time.sleep(float(sleep_seconds) * attempt)
    assert last_error is not None
    raise last_error


def fetch_dataset_viewer_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    max_rows: int,
    page_size: int = 100,
    page_sleep: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(int(page_size), 100),
            }
        )
        payload = _read_json_url(f"{DATASET_SERVER}/rows?{params}")
        batch = payload.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item.get("row", {})
            if isinstance(row, dict):
                copied = dict(row)
                copied["_row_idx"] = int(item.get("row_idx", len(rows)))
                rows.append(copied)
                if int(max_rows) > 0 and len(rows) >= int(max_rows):
                    return rows
        offset += len(batch)
        if len(batch) < min(int(page_size), 100):
            break
        if float(page_sleep) > 0.0:
            time.sleep(float(page_sleep))
    return rows


def fetch_local_datasets_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional backend
        raise RuntimeError("datasets is required for --backend datasets") from exc
    ds = load_dataset(dataset, config, split=split)
    limit = len(ds) if int(max_rows) <= 0 else min(len(ds), int(max_rows))
    rows: list[dict[str, Any]] = []
    for index in range(limit):
        row = dict(ds[index])
        row["_row_idx"] = int(index)
        rows.append(row)
    return rows


def fetch_rows_for_source(source: dict[str, str], args: argparse.Namespace) -> list[dict[str, Any]]:
    if str(args.backend) == "datasets":
        return fetch_local_datasets_rows(
            dataset=source["dataset"],
            config=source["config"],
            split=source["split"],
            max_rows=int(args.max_rows_per_source),
        )
    try:
        return fetch_dataset_viewer_rows(
            dataset=source["dataset"],
            config=source["config"],
            split=source["split"],
            max_rows=int(args.max_rows_per_source),
            page_sleep=float(args.page_sleep),
        )
    except Exception:
        if str(args.backend) != "auto":
            raise
        return fetch_local_datasets_rows(
            dataset=source["dataset"],
            config=source["config"],
            split=source["split"],
            max_rows=int(args.max_rows_per_source),
        )


def normalize_answer_key(answer: Any, labels: list[str]) -> str:
    text = str(answer).strip().upper()
    labels_upper = [str(label).strip().upper() for label in labels]
    if text in labels_upper:
        return OPTION_LETTERS[labels_upper.index(text)]
    if text in OPTION_LETTERS[: len(labels)]:
        return text
    return ""


def normalize_hf_mcq_row(
    row: dict[str, Any],
    *,
    dataset: str,
    config: str,
    split: str,
    category: str,
    benchmark_id: str,
) -> dict[str, Any] | None:
    question = str(row.get("question") or row.get("question_stem") or "").strip()
    choices = row.get("choices", {})
    if not question or not isinstance(choices, dict):
        return None
    texts = choices.get("text", [])
    labels = choices.get("label", [])
    if not isinstance(texts, list) or not isinstance(labels, list):
        return None
    if len(texts) < 2 or len(texts) > len(OPTION_LETTERS):
        return None
    if len(labels) != len(texts):
        labels = list(OPTION_LETTERS[: len(texts)])
    answer = normalize_answer_key(row.get("answerKey", row.get("answer", "")), [str(label) for label in labels])
    if not answer:
        return None
    answer_index = OPTION_LETTERS.index(answer)
    if answer_index >= len(texts):
        return None
    options = [str(option).strip() for option in texts]
    option_lines = "\n".join(
        f"{OPTION_LETTERS[index]}. {option}"
        for index, option in enumerate(options)
    )
    row_idx = int(row.get("_row_idx", 0))
    case_id = str(row.get("id") or f"{dataset}-{config}-{split}-{row_idx}")
    prompt = (
        f"User: Answer the following {dataset} multiple-choice question.\n"
        "Return only one option letter, with no explanation.\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_lines}\n\n"
        "Answer:\nAssistant:"
    )
    return {
        "benchmark_id": benchmark_id,
        "dataset": dataset,
        "config": config,
        "split": split,
        "case_id": f"{dataset.replace('/', '-')}-{config}-{split}-{case_id}",
        "row_idx": row_idx,
        "question_id": case_id,
        "category": category,
        "question": question,
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "qtrm_prompt": prompt,
        "scorer": "exact option-letter match",
    }


def build_pool(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = list(SOURCE_PRESETS[str(args.preset)])
    rows: list[dict[str, Any]] = []
    source_reports = []
    for source in sources:
        raw_rows = fetch_rows_for_source(source, args)
        normalized = [
            item
            for item in (
                normalize_hf_mcq_row(
                    row,
                    dataset=source["dataset"],
                    config=source["config"],
                    split=source["split"],
                    category=source["category"],
                    benchmark_id=str(args.benchmark_id),
                )
                for row in raw_rows
            )
            if item is not None
        ]
        rows.extend(normalized)
        source_reports.append(
            {
                **source,
                "raw_rows": len(raw_rows),
                "normalized_rows": len(normalized),
            }
        )
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    if int(args.max_cases) > 0:
        rows = rows[: int(args.max_cases)]
    by_dataset: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in rows:
        by_dataset[str(row["dataset"])] = by_dataset.get(str(row["dataset"]), 0) + 1
        by_category[str(row["category"])] = by_category.get(str(row["category"]), 0) + 1
    report = {
        "status": "complete",
        "benchmark_id": str(args.benchmark_id),
        "preset": str(args.preset),
        "cases": len(rows),
        "sources": source_reports,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_category": dict(sorted(by_category.items())),
        "out_jsonl": str(args.out_jsonl),
        "seed": int(args.seed),
        "policy": "non-test external MCQ pool; do not use public target test labels for checkpoint selection",
    }
    return rows, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(SOURCE_PRESETS), default="validation")
    parser.add_argument("--backend", choices=["auto", "dataset-viewer", "datasets"], default="auto")
    parser.add_argument("--benchmark-id", default="external_mcq")
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--page-sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--out-jsonl", default="local_eval/m7_public_reasoning_suite/external_mcq_validation_pool_20260516.jsonl")
    parser.add_argument("--out-report", default="local_eval/m7_public_reasoning_suite/report_external_mcq_validation_pool_20260516.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows, report = build_pool(args)
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_jsonl).write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    Path(args.out_report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
