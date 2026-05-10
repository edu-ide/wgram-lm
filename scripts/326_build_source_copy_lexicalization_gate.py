#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_TRAIN_OUT = "data/filtered/qtrm_source_copy_lexicalization_train512.jsonl"
DEFAULT_EVAL_OUT = "data/eval/qtrm_source_copy_lexicalization_eval128.jsonl"
DEFAULT_SUMMARY_OUT = (
    "data/filtered/qtrm_source_copy_lexicalization_train512.summary.json"
)


class SourceCopyLexicalizationBundle(NamedTuple):
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_pair_builder(root: Path | None = None) -> Any:
    root = root or repo_root()
    script = root / "scripts" / "323_build_source_position_pair_hard_negatives.py"
    spec = importlib.util.spec_from_file_location("source_position_pair_builder", script)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load data builder: {script}")
    spec.loader.exec_module(module)
    return module


def _csv(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in values) if values else "EMPTY"


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _to_source_copy_row(row: dict[str, Any]) -> dict[str, Any]:
    values = [int(value) for value in row["input_list"]]
    filtered = [value for value in values if value % 2 == 0]
    answer_text = _csv(filtered)
    doubled_text = _csv([2 * value for value in filtered])
    question = (
        f"From the list {values}, keep only even numbers and return them as "
        "comma-separated values with no spaces. If none, return EMPTY."
    )
    copied = dict(row)
    copied["id"] = str(row["id"]).replace("source-position-pair-", "source-copy-")
    copied["raw_intelligence_axis"] = "pure_recursive_reasoning"
    copied["category"] = "source_copy_lexicalization"
    copied["task_family"] = "source_copy_lexicalization"
    copied["reasoning_family"] = "sequential_source_copy"
    copied["hard_variant"] = "paired_permutation_source_copy"
    copied["expected_paradigm"] = "latent_recurrent_source_pointer_copy"
    copied["question"] = question
    copied["prompt"] = _prompt_for_question(question)
    copied["answer"] = answer_text
    copied["chosen"] = answer_text
    copied["answer_aliases"] = [answer_text]
    copied["choices"] = [answer_text, doubled_text, _csv(list(reversed(filtered))), "EMPTY"]
    copied["depth_targets"] = {
        "1": answer_text,
        "2": answer_text,
        "4": answer_text,
        "8": answer_text,
    }
    copied["solver_trace"] = [
        {"depth": 1, "operation": "filter_even_copy", "state_text": answer_text},
        {"depth": 2, "operation": "hold_final", "state_text": answer_text},
        {"depth": 4, "operation": "hold_final", "state_text": answer_text},
        {"depth": 8, "operation": "hold_final", "state_text": answer_text},
    ]
    copied["source_copy_lexicalization"] = True
    copied["role_value_list_class_mode"] = "source_position"
    copied["role_value_supervise_null_slots"] = True
    copied["role_value_source_copy_no_doubled"] = True
    return copied


def build_source_copy_lexicalization_split(
    *,
    train_groups: int = 128,
    eval_groups: int = 32,
    permutations_per_group: int = 4,
    seed: int = 326,
    train_value_low: int = 0,
    train_value_high: int = 32,
    eval_value_low: int = 32,
    eval_value_high: int = 64,
    list_len: int = 5,
) -> SourceCopyLexicalizationBundle:
    pair_builder = load_pair_builder()
    pair_bundle = pair_builder.build_source_position_pair_hard_negative_split(
        train_groups=int(train_groups),
        eval_groups=int(eval_groups),
        permutations_per_group=int(permutations_per_group),
        seed=int(seed),
        train_value_low=int(train_value_low),
        train_value_high=int(train_value_high),
        eval_value_low=int(eval_value_low),
        eval_value_high=int(eval_value_high),
        list_len=int(list_len),
    )
    train_rows = [_to_source_copy_row(row) for row in pair_bundle.train_rows]
    eval_rows = [_to_source_copy_row(row) for row in pair_bundle.eval_rows]
    summary = {
        "split_type": "source_copy_lexicalization",
        "target_level": "L2/L3 prerequisite repair for L4 LM-path promotion",
        "major_bottleneck": "faithful pointer-copy lexicalization before transformed-value rendering",
        "prior_principle": "pointer/copy and copy-generate lexicalization",
        "rows": len(train_rows) + len(eval_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_groups": int(train_groups),
        "eval_groups": int(eval_groups),
        "permutations_per_group": int(permutations_per_group),
        "seed": int(seed),
        "list_len": int(list_len),
        "train_value_range": [int(train_value_low), int(train_value_high)],
        "eval_value_range": [int(eval_value_low), int(eval_value_high)],
        "source_pair_summary": pair_bundle.summary,
        "acceptance_rule": (
            "full must beat donor/core-off and source-binder/copy-renderer-off "
            "must drop on copied source-token answers"
        ),
    }
    return SourceCopyLexicalizationBundle(train_rows, eval_rows, summary)


def write_source_copy_lexicalization_split(
    *,
    train_out: str | Path = DEFAULT_TRAIN_OUT,
    eval_out: str | Path = DEFAULT_EVAL_OUT,
    summary_out: str | Path = DEFAULT_SUMMARY_OUT,
    **kwargs: Any,
) -> dict[str, Any]:
    bundle = build_source_copy_lexicalization_split(**kwargs)
    train_path = Path(train_out)
    eval_path = Path(eval_out)
    summary_path = Path(summary_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in bundle.train_rows)
        + "\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in bundle.eval_rows)
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(bundle.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle.summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a source-copy lexicalization gate for pointer/copy renderer repair."
    )
    parser.add_argument("--train-out", default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", default=DEFAULT_EVAL_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--train-groups", type=int, default=128)
    parser.add_argument("--eval-groups", type=int, default=32)
    parser.add_argument("--permutations-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=326)
    parser.add_argument("--train-value-low", type=int, default=0)
    parser.add_argument("--train-value-high", type=int, default=32)
    parser.add_argument("--eval-value-low", type=int, default=32)
    parser.add_argument("--eval-value-high", type=int, default=64)
    parser.add_argument("--list-len", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = write_source_copy_lexicalization_split(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=args.summary_out,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        permutations_per_group=args.permutations_per_group,
        seed=args.seed,
        train_value_low=args.train_value_low,
        train_value_high=args.train_value_high,
        eval_value_low=args.eval_value_low,
        eval_value_high=args.eval_value_high,
        list_len=args.list_len,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
