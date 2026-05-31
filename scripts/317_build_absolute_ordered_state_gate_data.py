#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_TRAIN_OUT = (
    "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl"
)
DEFAULT_EVAL_OUT = "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl"
DEFAULT_SUMMARY_OUT = (
    "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.summary.json"
)


class AbsoluteOrderedStateBundle(NamedTuple):
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def load_algorithmic_value_state_module():
    return importlib.import_module("wgram_lm.algorithmic_value_state")


def _csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values) if values else "EMPTY"


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _make_case(*, case_id: str, values: list[int]) -> dict[str, Any]:
    filtered = [int(value) for value in values if int(value) % 2 == 0]
    doubled = [2 * int(value) for value in filtered]
    filtered_text = _csv(filtered)
    answer_text = _csv(doubled)
    question = (
        f"From the list {values}, keep only even numbers, double each kept "
        "number, and return comma-separated values with no spaces. If none, "
        "return EMPTY."
    )
    return {
        "id": str(case_id),
        "raw_intelligence_axis": "pure_recursive_reasoning",
        "category": "list_transform",
        "task_family": "list_transform",
        "reasoning_family": "sequential_list_transform",
        "expected_paradigm": "latent_recurrent",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 2,
        "question": question,
        "prompt": _prompt_for_question(question),
        "answer": answer_text,
        "chosen": answer_text,
        "answer_aliases": [answer_text],
        "choices": [answer_text, filtered_text, _csv(list(reversed(doubled))), "EMPTY"],
        "depth_targets": {
            "1": filtered_text,
            "2": answer_text,
            "4": answer_text,
            "8": answer_text,
        },
        "transition_state_codes": {"1": 40, "2": 41, "4": 41, "8": 41},
        "solver_trace": [
            {"depth": 1, "operation": "filter_even", "state_text": filtered_text},
            {"depth": 2, "operation": "double_filtered", "state_text": answer_text},
            {"depth": 4, "operation": "hold_final", "state_text": answer_text},
            {"depth": 8, "operation": "hold_final", "state_text": answer_text},
        ],
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "input_list": values,
        "list_length": len(values),
        "role_value_list_class_mode": "absolute",
    }


def _coverage_rows(*, value_modulus: int, list_len: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    even_values = list(range(0, int(value_modulus), 2))
    odd_fillers = [value for value in range(int(value_modulus)) if value % 2 == 1]
    if not odd_fillers:
        odd_fillers = [1]
    for index, even_value in enumerate(even_values):
        paired_even = even_values[(index + 1) % len(even_values)]
        values = [int(even_value), int(paired_even)]
        filler_index = 0
        while len(values) < int(list_len):
            values.append(int(odd_fillers[filler_index % len(odd_fillers)]))
            filler_index += 1
        rows.append(
            _make_case(
                case_id=f"absolute-ordered-coverage-{even_value:03d}",
                values=values,
            )
        )
    return rows


def _random_values(
    *,
    rng: random.Random,
    value_modulus: int,
    list_len: int,
) -> list[int]:
    values = [rng.randrange(int(value_modulus)) for _ in range(int(list_len))]
    even_values = [2 * index for index in range(max(1, int(value_modulus) // 2))]
    even_count = sum(1 for value in values if value % 2 == 0)
    while even_count < min(2, int(list_len)):
        odd_indices = [index for index, value in enumerate(values) if value % 2 != 0]
        insert_at = odd_indices[0] if odd_indices else even_count
        values[insert_at] = int(even_values[rng.randrange(len(even_values))])
        even_count = sum(1 for value in values if value % 2 == 0)
    return values


def build_absolute_ordered_state_rows(
    *,
    count: int,
    seed: int,
    value_modulus: int,
    list_len: int,
    include_coverage: bool = False,
) -> list[dict[str, Any]]:
    if int(count) <= 0:
        raise ValueError("count must be positive")
    if int(value_modulus) <= 1:
        raise ValueError("value_modulus must be greater than 1")
    if int(list_len) <= 0:
        raise ValueError("list_len must be positive")
    rows = _coverage_rows(value_modulus=int(value_modulus), list_len=int(list_len)) if include_coverage else []
    rng = random.Random(int(seed))
    while len(rows) < int(count):
        index = len(rows)
        rows.append(
            _make_case(
                case_id=f"absolute-ordered-{seed}-{index:06d}",
                values=_random_values(
                    rng=rng,
                    value_modulus=int(value_modulus),
                    list_len=int(list_len),
                ),
            )
        )
    return rows[: int(count)]


def target_classes_for_rows(
    rows: list[dict[str, Any]],
    *,
    value_state=None,
    value_vocab_size: int,
    num_roles: int = 10,
    num_steps: int = 4,
) -> set[int]:
    value_state = value_state or load_algorithmic_value_state_module()
    classes: set[int] = set()
    for row in rows:
        targets = value_state.role_value_targets_from_row(
            row,
            num_steps=int(num_steps),
            num_roles=int(num_roles),
            value_vocab_size=int(value_vocab_size),
            list_class_mode="absolute",
        )
        for step_targets in targets:
            for class_id in step_targets:
                if int(class_id) >= 0:
                    classes.add(int(class_id))
    return classes


def build_absolute_ordered_state_split(
    *,
    train_count: int = 512,
    eval_count: int = 128,
    train_seed: int = 317,
    eval_seed: int = 9317,
    value_modulus: int = 32,
    list_len: int = 5,
    value_vocab_size: int = 256,
) -> AbsoluteOrderedStateBundle:
    train_rows = build_absolute_ordered_state_rows(
        count=int(train_count),
        seed=int(train_seed),
        value_modulus=int(value_modulus),
        list_len=int(list_len),
        include_coverage=True,
    )
    eval_rows = build_absolute_ordered_state_rows(
        count=int(eval_count),
        seed=int(eval_seed),
        value_modulus=int(value_modulus),
        list_len=int(list_len),
        include_coverage=False,
    )
    train_ids = {row["id"] for row in train_rows}
    eval_rows = [row for row in eval_rows if row["id"] not in train_ids]
    value_state = load_algorithmic_value_state_module()
    train_classes = target_classes_for_rows(
        train_rows,
        value_state=value_state,
        value_vocab_size=int(value_vocab_size),
    )
    eval_classes = target_classes_for_rows(
        eval_rows,
        value_state=value_state,
        value_vocab_size=int(value_vocab_size),
    )
    missing = sorted(eval_classes - train_classes)
    if missing:
        raise ValueError(f"eval target classes missing from train coverage: {missing}")
    summary = {
        "split_type": "absolute_value_coverage_combo_holdout",
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "value_modulus": int(value_modulus),
        "list_len": int(list_len),
        "value_vocab_size": int(value_vocab_size),
        "train_target_classes": sorted(train_classes),
        "eval_target_classes": sorted(eval_classes),
        "class_coverage_ok": not missing,
    }
    return AbsoluteOrderedStateBundle(train_rows, eval_rows, summary)


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_absolute_ordered_state_split(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    summary_out: str | Path,
    train_count: int = 512,
    eval_count: int = 128,
    train_seed: int = 317,
    eval_seed: int = 9317,
    value_modulus: int = 32,
    list_len: int = 5,
    value_vocab_size: int = 256,
) -> dict[str, Any]:
    bundle = build_absolute_ordered_state_split(
        train_count=int(train_count),
        eval_count=int(eval_count),
        train_seed=int(train_seed),
        eval_seed=int(eval_seed),
        value_modulus=int(value_modulus),
        list_len=int(list_len),
        value_vocab_size=int(value_vocab_size),
    )
    _write_jsonl(train_out, bundle.train_rows)
    _write_jsonl(eval_out, bundle.eval_rows)
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(bundle.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle.summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build absolute-value list-transform train/eval data for the QTRM "
            "ordered-state gate. The split holds out combinations while ensuring "
            "eval target classes are covered by train."
        )
    )
    parser.add_argument("--train-out", default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", default=DEFAULT_EVAL_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=317)
    parser.add_argument("--eval-seed", type=int, default=9317)
    parser.add_argument("--value-modulus", type=int, default=32)
    parser.add_argument("--list-len", type=int, default=5)
    parser.add_argument("--value-vocab-size", type=int, default=256)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = write_absolute_ordered_state_split(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=args.summary_out,
        train_count=int(args.train_count),
        eval_count=int(args.eval_count),
        train_seed=int(args.train_seed),
        eval_seed=int(args.eval_seed),
        value_modulus=int(args.value_modulus),
        list_len=int(args.list_len),
        value_vocab_size=int(args.value_vocab_size),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
