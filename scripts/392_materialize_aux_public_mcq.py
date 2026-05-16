#!/usr/bin/env python3
"""Materialize non-test auxiliary MCQ data for public-benchmark repair.

The output schema matches the QTRM public-MCQ evaluator/trainer.  This script
is intentionally separate from the MMLU-Pro materializer so MMLU-Pro test labels
do not become a training or checkpoint-selection source.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


OPTION_LETTERS = tuple("ABCDEFGHIJ")
TARGET_REPAIR_CONFIGS = (
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
    "college_chemistry",
    "high_school_chemistry",
    "econometrics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "international_law",
    "jurisprudence",
    "professional_law",
)


def load_dataset_rows(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required to materialize auxiliary MCQ data") from exc
    loaded = load_dataset(dataset, config, split=split)
    return [dict(row) for row in loaded]


def subject_to_category(subject: str, config: str) -> str:
    text = f"{subject} {config}".lower()
    if any(key in text for key in ("medicine", "clinical", "anatomy", "nutrition", "health")):
        return "health"
    if "chemistry" in text:
        return "chemistry"
    if any(key in text for key in ("econom", "macroeconomics", "microeconomics")):
        return "economics"
    if any(key in text for key in ("law", "jurisprudence")):
        return "law"
    if any(key in text for key in ("computer", "machine_learning")):
        return "computer science"
    if any(key in text for key in ("physics", "astronomy")):
        return "physics"
    if any(key in text for key in ("math", "algebra", "calculus", "statistics")):
        return "math"
    if any(key in text for key in ("psychology", "moral")):
        return "psychology"
    if any(key in text for key in ("history", "prehistory")):
        return "history"
    if any(key in text for key in ("business", "management", "marketing", "accounting")):
        return "business"
    if any(key in text for key in ("biology", "genetics", "virology")):
        return "biology"
    if any(key in text for key in ("philosophy", "logical_fallacies")):
        return "philosophy"
    return "other"


def option_lines(options: list[str]) -> list[str]:
    if len(options) > len(OPTION_LETTERS):
        raise ValueError(f"too many options for fixed option alphabet: {len(options)}")
    return [f"{OPTION_LETTERS[index]}. {option}" for index, option in enumerate(options)]


def format_prompt(question: str, options: list[str], *, source_name: str) -> str:
    body = "\n".join(
        [
            f"Answer the following {source_name} multiple-choice question.",
            "Return only one option letter, with no explanation.",
            "",
            f"Question: {question.strip()}",
            "Options:",
            *option_lines(options),
            "",
            "Answer:",
        ]
    )
    return f"User: {body}\nAssistant:"


def normalize_answer(value: Any) -> tuple[str, int]:
    if isinstance(value, int):
        if 0 <= value < len(OPTION_LETTERS):
            return OPTION_LETTERS[value], value
        raise ValueError(f"answer index out of range: {value}")
    text = str(value).strip().upper()
    if text in OPTION_LETTERS:
        return text, OPTION_LETTERS.index(text)
    raise ValueError(f"unsupported MCQ answer: {value!r}")


def row_to_case(
    *,
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    choices = row.get("choices", row.get("options"))
    if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
        raise ValueError(f"invalid choices at {config}/{split}/{row_idx}")
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError(f"missing question at {config}/{split}/{row_idx}")
    answer, answer_index = normalize_answer(row.get("answer"))
    subject = str(row.get("subject", "") or config)
    category = subject_to_category(subject, config)
    return {
        "benchmark_id": "aux_public_mcq",
        "dataset": dataset,
        "config": config,
        "split": split,
        "case_id": f"{dataset.replace('/', '-')}-{config}-{split}-{row_idx:06d}",
        "row_idx": int(row_idx),
        "question_id": row.get("question_id", row_idx),
        "category": category,
        "subject": subject,
        "question": question,
        "options": choices,
        "answer": answer,
        "answer_index": answer_index,
        "qtrm_prompt": format_prompt(question, choices, source_name=dataset),
        "scorer": "exact option-letter match",
    }


def materialize(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(int(args.seed))
    configs = [item.strip() for item in str(args.configs).split(",") if item.strip()]
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    if not configs:
        raise ValueError("--configs must include at least one config")
    if not splits:
        raise ValueError("--splits must include at least one split")
    cases: list[dict[str, Any]] = []
    for config in configs:
        for split in splits:
            rows = load_dataset_rows(str(args.dataset), config, split)
            indexed = list(enumerate(rows))
            if bool(args.shuffle):
                rng.shuffle(indexed)
            limit = int(args.per_config_split_limit)
            if limit > 0:
                indexed = indexed[:limit]
            for row_idx, row in indexed:
                cases.append(
                    row_to_case(
                        dataset=str(args.dataset),
                        config=config,
                        split=split,
                        row_idx=row_idx,
                        row=row,
                    )
                )
    if bool(args.shuffle):
        rng.shuffle(cases)
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]
    if len(cases) < int(args.min_cases):
        raise ValueError(f"only materialized {len(cases)} cases, below min {args.min_cases}")
    return cases


def write_outputs(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    out_jsonl = Path(args.out_jsonl)
    out_report = Path(args.out_report)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    by_category: dict[str, int] = {}
    by_subject: dict[str, int] = {}
    for row in cases:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
        by_subject[row["subject"]] = by_subject.get(row["subject"], 0) + 1
    report = {
        "status": "complete",
        "decision": "accepted_aux_public_mcq_materialized",
        "accepted": True,
        "dataset": str(args.dataset),
        "configs": [item.strip() for item in str(args.configs).split(",") if item.strip()],
        "splits": [item.strip() for item in str(args.splits).split(",") if item.strip()],
        "cases": len(cases),
        "out_jsonl": str(out_jsonl),
        "by_category": dict(sorted(by_category.items())),
        "by_subject": dict(sorted(by_subject.items())),
        "scorer": "exact option-letter match",
        "leakage_policy": [
            "Do not use MMLU-Pro test labels for training or checkpoint selection.",
            "Use this auxiliary file for training/dev selection, then evaluate once on held-out MMLU-Pro test subsets.",
        ],
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cais/mmlu")
    parser.add_argument("--configs", default=",".join(TARGET_REPAIR_CONFIGS))
    parser.add_argument("--splits", default="dev,validation")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--per-config-split-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--out-jsonl", default="local_eval/m7_public_reasoning_suite/mmlu_aux_targeted_dev_validation.jsonl")
    parser.add_argument("--out-report", default="local_eval/m7_public_reasoning_suite/report_mmlu_aux_targeted_dev_validation.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = materialize(args)
    print(json.dumps(write_outputs(args, cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
