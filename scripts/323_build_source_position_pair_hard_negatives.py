#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
from typing import Any, NamedTuple


DEFAULT_TRAIN_OUT = "data/filtered/qtrm_source_position_pair_hard_train512.jsonl"
DEFAULT_EVAL_OUT = "data/eval/qtrm_source_position_pair_hard_eval128.jsonl"
DEFAULT_SUMMARY_OUT = (
    "data/filtered/qtrm_source_position_pair_hard_train512.summary.json"
)


class SourcePositionPairHardNegativeBundle(NamedTuple):
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_absolute_ordered_builder(root: Path | None = None) -> Any:
    root = root or repo_root()
    script = root / "scripts" / "317_build_absolute_ordered_state_gate_data.py"
    spec = importlib.util.spec_from_file_location("absolute_ordered_state_builder", script)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load data builder: {script}")
    spec.loader.exec_module(module)
    return module


def _value_multiset_signature(values: list[int]) -> str:
    return ",".join(str(value) for value in sorted(int(value) for value in values))


def _even_position_signature(values: list[int]) -> tuple[int, ...]:
    return tuple(index + 1 for index, value in enumerate(values) if int(value) % 2 == 0)


def _sample_group_values(
    *,
    rng: random.Random,
    low: int,
    high: int,
    list_len: int,
) -> list[int]:
    if int(list_len) < 3:
        raise ValueError("list_len must be at least 3 for paired source-position contrasts")
    evens = [value for value in range(int(low), int(high)) if value % 2 == 0]
    odds = [value for value in range(int(low), int(high)) if value % 2 == 1]
    if len(evens) < 2 or not odds:
        raise ValueError("value range must contain at least two evens and one odd")
    even_count = min(3, int(list_len) - 1)
    odd_count = int(list_len) - even_count
    values = rng.sample(evens, even_count) + rng.sample(odds, odd_count)
    rng.shuffle(values)
    return [int(value) for value in values]


def _permuted_rows_for_group(
    *,
    values: list[int],
    permutations_per_group: int,
    rng: random.Random,
) -> list[list[int]]:
    even_count = sum(1 for value in values if int(value) % 2 == 0)
    max_position_patterns = math.comb(len(values), even_count)
    if int(permutations_per_group) > int(max_position_patterns):
        raise ValueError(
            "permutations_per_group exceeds distinct even-position patterns "
            f"({permutations_per_group} > {max_position_patterns})"
        )
    rows: list[list[int]] = []
    seen_orders: set[tuple[int, ...]] = set()
    seen_positions: set[tuple[int, ...]] = set()
    attempts = 0
    while len(rows) < int(permutations_per_group):
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("failed to sample enough paired hard-negative permutations")
        candidate = list(values)
        rng.shuffle(candidate)
        order_key = tuple(int(value) for value in candidate)
        position_key = _even_position_signature(candidate)
        if order_key in seen_orders or position_key in seen_positions:
            continue
        rows.append([int(value) for value in candidate])
        seen_orders.add(order_key)
        seen_positions.add(position_key)
    return rows


def _make_pair_row(
    builder: Any,
    *,
    split: str,
    seed: int,
    group_index: int,
    permutation_index: int,
    values: list[int],
) -> dict[str, Any]:
    group_id = f"{split}-pair-s{int(seed)}-{int(group_index):04d}"
    row = builder._make_case(  # noqa: SLF001 - reuse canonical list-transform schema.
        case_id=f"source-position-pair-{group_id}-p{int(permutation_index):02d}",
        values=[int(value) for value in values],
    )
    row["hard_variant"] = "paired_permutation_source_positions"
    row["role_value_list_class_mode"] = "source_position"
    row["role_value_supervise_null_slots"] = True
    row["expected_paradigm"] = "latent_recurrent_source_pointer"
    row["pair_group_id"] = group_id
    row["pair_permutation_index"] = int(permutation_index)
    row["value_multiset_signature"] = _value_multiset_signature(values)
    row["source_even_position_signature"] = list(_even_position_signature(values))
    row["source_position_pair_hard_negative"] = True
    return row


def build_source_position_pair_hard_negative_rows(
    *,
    split: str,
    group_count: int,
    permutations_per_group: int,
    seed: int,
    value_low: int,
    value_high: int,
    list_len: int = 5,
    reserved_signatures: set[str] | None = None,
) -> list[dict[str, Any]]:
    if int(group_count) <= 0:
        raise ValueError("group_count must be positive")
    if int(permutations_per_group) <= 1:
        raise ValueError("permutations_per_group must be greater than 1")
    builder = load_absolute_ordered_builder()
    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    used_signatures = set(reserved_signatures or set())
    for group_index in range(int(group_count)):
        attempts = 0
        while True:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("failed to sample a unique value multiset group")
            base_values = _sample_group_values(
                rng=rng,
                low=int(value_low),
                high=int(value_high),
                list_len=int(list_len),
            )
            signature = _value_multiset_signature(base_values)
            if signature not in used_signatures:
                used_signatures.add(signature)
                break
        permutations = _permuted_rows_for_group(
            values=base_values,
            permutations_per_group=int(permutations_per_group),
            rng=rng,
        )
        for permutation_index, values in enumerate(permutations):
            rows.append(
                _make_pair_row(
                    builder,
                    split=str(split),
                    seed=int(seed),
                    group_index=group_index,
                    permutation_index=permutation_index,
                    values=values,
                )
            )
    return rows


def _group_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["pair_group_id"]) for row in rows})


def _multiset_signatures(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["value_multiset_signature"]) for row in rows})


def _position_patterns(rows: list[dict[str, Any]]) -> list[list[int]]:
    return sorted(
        {
            tuple(int(value) for value in row["source_even_position_signature"])
            for row in rows
        }
    )


def build_source_position_pair_hard_negative_split(
    *,
    train_groups: int = 128,
    eval_groups: int = 32,
    permutations_per_group: int = 4,
    seed: int = 323,
    train_value_low: int = 0,
    train_value_high: int = 32,
    eval_value_low: int = 32,
    eval_value_high: int = 64,
    list_len: int = 5,
) -> SourcePositionPairHardNegativeBundle:
    train_rows = build_source_position_pair_hard_negative_rows(
        split="train",
        group_count=int(train_groups),
        permutations_per_group=int(permutations_per_group),
        seed=int(seed),
        value_low=int(train_value_low),
        value_high=int(train_value_high),
        list_len=int(list_len),
    )
    train_signatures = set(_multiset_signatures(train_rows))
    eval_rows = build_source_position_pair_hard_negative_rows(
        split="eval",
        group_count=int(eval_groups),
        permutations_per_group=int(permutations_per_group),
        seed=int(seed) + 100000,
        value_low=int(eval_value_low),
        value_high=int(eval_value_high),
        list_len=int(list_len),
        reserved_signatures=train_signatures,
    )
    summary = {
        "split_type": "source_position_pair_hard_negative",
        "target_level": "L3 prerequisite repair for L4 LM-path promotion",
        "major_bottleneck": "source-position anti-shortcut binding",
        "prior_principle": (
            "counterfactual paired hard negatives for attention/pointer binding"
        ),
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
        "train_pair_group_ids": _group_ids(train_rows),
        "eval_pair_group_ids": _group_ids(eval_rows),
        "train_value_multiset_signatures": _multiset_signatures(train_rows),
        "eval_value_multiset_signatures": _multiset_signatures(eval_rows),
        "train_source_even_position_patterns": _position_patterns(train_rows),
        "eval_source_even_position_patterns": _position_patterns(eval_rows),
        "causal_ablation_required": (
            "source_binder_off or any prompt-binding ablation should collapse "
            "on this split; otherwise the gate still permits shortcuts"
        ),
        "shortcut_risk_addressed": (
            "same value multiset and same task surface can require different "
            "source-position targets, so value memorization is insufficient"
        ),
    }
    return SourcePositionPairHardNegativeBundle(train_rows, eval_rows, summary)


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_source_position_pair_hard_negative_split(
    *,
    train_out: str | Path = DEFAULT_TRAIN_OUT,
    eval_out: str | Path = DEFAULT_EVAL_OUT,
    summary_out: str | Path = DEFAULT_SUMMARY_OUT,
    train_groups: int = 128,
    eval_groups: int = 32,
    permutations_per_group: int = 4,
    seed: int = 323,
    train_value_low: int = 0,
    train_value_high: int = 32,
    eval_value_low: int = 32,
    eval_value_high: int = 64,
    list_len: int = 5,
) -> dict[str, Any]:
    bundle = build_source_position_pair_hard_negative_split(
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
            "Build paired source-position hard negatives for the QTRM "
            "source-pointer state gate."
        )
    )
    parser.add_argument("--train-out", default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", default=DEFAULT_EVAL_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--train-groups", type=int, default=128)
    parser.add_argument("--eval-groups", type=int, default=32)
    parser.add_argument("--permutations-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--train-value-low", type=int, default=0)
    parser.add_argument("--train-value-high", type=int, default=32)
    parser.add_argument("--eval-value-low", type=int, default=32)
    parser.add_argument("--eval-value-high", type=int, default=64)
    parser.add_argument("--list-len", type=int, default=5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = write_source_position_pair_hard_negative_split(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=args.summary_out,
        train_groups=int(args.train_groups),
        eval_groups=int(args.eval_groups),
        permutations_per_group=int(args.permutations_per_group),
        seed=int(args.seed),
        train_value_low=int(args.train_value_low),
        train_value_high=int(args.train_value_high),
        eval_value_low=int(args.eval_value_low),
        eval_value_high=int(args.eval_value_high),
        list_len=int(args.list_len),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
