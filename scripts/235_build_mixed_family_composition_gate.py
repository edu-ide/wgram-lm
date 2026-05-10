#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence


class MixedFamilyCompositionBundle(NamedTuple):
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _load_script_module(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    spec.loader.exec_module(module)
    return module


def _family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "")


def _variant(row: dict[str, Any]) -> int:
    return int(row.get("surface_variant_index", -1))


def _parse_variants(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        variants = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    else:
        variants = tuple(int(item) for item in value)
    if not variants:
        raise ValueError("variant set cannot be empty")
    bad = [variant for variant in variants if variant < 0 or variant > 23]
    if bad:
        raise ValueError(f"surface variants must be in [0, 23], got {bad}")
    return tuple(sorted(set(variants)))


def _parse_lengths(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return (7, 9)
    if isinstance(value, str):
        lengths = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    else:
        lengths = tuple(int(item) for item in value)
    if not lengths:
        raise ValueError("eval_list_lengths cannot be empty")
    bad = [length for length in lengths if length <= 0]
    if bad:
        raise ValueError(f"list lengths must be positive, got {bad}")
    return tuple(sorted(set(lengths)))


def _parse_composition_orders(value: str | Sequence[str] | None) -> tuple[str, ...]:
    allowed = {"list_to_arithmetic", "arithmetic_to_list"}
    if value is None:
        return ("list_to_arithmetic",)
    if isinstance(value, str):
        orders = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        orders = tuple(str(item).strip() for item in value if str(item).strip())
    if not orders:
        raise ValueError("composition order set cannot be empty")
    unknown = sorted(set(orders) - allowed)
    if unknown:
        raise ValueError(f"unknown composition orders: {unknown}")
    return tuple(dict.fromkeys(orders))


def _stress_rows(
    *,
    cases_per_family: int,
    start_index: int,
    variants_per_case: int = 8,
) -> list[dict[str, Any]]:
    stress_builder = _load_script_module(
        "227_build_pure_recursive_ood_paraphrase_stress_cases.py"
    )
    return stress_builder.build_ood_paraphrase_stress_cases(
        cases_per_family=int(cases_per_family),
        start_index=int(start_index),
        variants_per_case=int(variants_per_case),
    )


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values) if values else "EMPTY"


def _make_mixed_case(*, idx: int, list_length: int) -> dict[str, Any]:
    values = [int(idx) + offset + 1 for offset in range(int(list_length))]
    filtered = [value for value in values if value % 2 == 0]
    doubled = [value * 2 for value in filtered]
    summed = sum(doubled)
    offset = int(idx) % 7 + 3
    answer = summed - offset
    filtered_text = _csv(filtered)
    doubled_text = _csv(doubled)
    sum_text = str(summed)
    answer_text = str(answer)
    question = (
        "From the list "
        f"{values}, keep even numbers, double each kept number, sum the doubled "
        f"values, subtract {offset}, and return the final integer."
    )
    return {
        "id": f"mixed-list-arithmetic-{idx}-len{int(list_length)}",
        "raw_intelligence_axis": "pure_recursive_reasoning",
        "category": "mixed_list_arithmetic",
        "task_family": "mixed_list_arithmetic",
        "composition_order": "list_to_arithmetic",
        "reasoning_family": "sequential_list_arithmetic_composition",
        "expected_paradigm": "hybrid_or_cot",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 4,
        "question": question,
        "prompt": _prompt_for_question(question),
        "answer": answer_text,
        "chosen": answer_text,
        "answer_aliases": [answer_text],
        "choices": [answer_text, sum_text, doubled_text, "EMPTY"],
        "depth_targets": {
            "1": filtered_text,
            "2": doubled_text,
            "3": sum_text,
            "4": answer_text,
            "8": answer_text,
        },
        "transition_state_codes": {"1": 40, "2": 41, "3": 11, "4": 12, "8": 12},
        "solver_trace": [
            {"depth": 1, "operation": "filter_even", "state_text": filtered_text},
            {"depth": 2, "operation": "double_filtered", "state_text": doubled_text},
            {"depth": 3, "operation": "multiply_sum", "state_text": sum_text},
            {"depth": 4, "operation": "subtract_offset", "state_text": answer_text},
        ],
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "list_length": int(list_length),
        "list_value_start": int(idx) + 1,
        "mixed_offset": offset,
    }


def _make_arithmetic_to_list_case(*, idx: int, list_length: int) -> dict[str, Any]:
    a = int(idx) % 19 + 3
    b = int(idx) % 13 + 2
    multiplier = int(idx) % 4 + 2
    offset = int(idx) % 5 + 1
    summed = a + b
    product = summed * multiplier
    threshold = product - offset
    half_width = max(3, int(list_length) // 2)
    values = [threshold + delta for delta in range(-half_width, half_width + 1)]
    filtered = [value for value in values if value > threshold and value % 2 == 0]
    doubled = [value * 2 for value in filtered]
    filtered_text = _csv(filtered)
    doubled_text = _csv(doubled)
    question = (
        f"First compute (({a} + {b}) * {multiplier}) - {offset}. "
        f"Use that result as a threshold for the list {values}: keep even "
        "numbers strictly above the threshold, double the kept numbers, and "
        "return the comma-separated list with no spaces. If none remain, "
        "return EMPTY."
    )
    return {
        "id": f"mixed-arithmetic-list-{idx}-len{int(list_length)}",
        "raw_intelligence_axis": "pure_recursive_reasoning",
        "category": "mixed_arithmetic_list",
        "task_family": "mixed_arithmetic_list",
        "reasoning_family": "sequential_arithmetic_list_composition",
        "expected_paradigm": "hybrid_or_cot",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 5,
        "question": question,
        "prompt": _prompt_for_question(question),
        "answer": doubled_text,
        "chosen": doubled_text,
        "answer_aliases": [doubled_text],
        "choices": [doubled_text, filtered_text, str(threshold), "EMPTY"],
        "depth_targets": {
            "1": str(summed),
            "2": str(product),
            "3": str(threshold),
            "4": filtered_text,
            "5": doubled_text,
            "8": doubled_text,
        },
        "transition_state_codes": {
            "1": 40,
            "2": 11,
            "3": 12,
            "4": 41,
            "5": 41,
            "8": 41,
        },
        "solver_trace": [
            {"depth": 1, "operation": "add_operands", "state_text": str(summed)},
            {"depth": 2, "operation": "multiply_sum", "state_text": str(product)},
            {"depth": 3, "operation": "subtract_offset", "state_text": str(threshold)},
            {
                "depth": 4,
                "operation": "filter_above_threshold",
                "state_text": filtered_text,
            },
            {"depth": 5, "operation": "double_filtered", "state_text": doubled_text},
        ],
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "list_length": int(list_length),
        "list_value_start": int(values[0]),
        "mixed_offset": offset,
        "arithmetic_threshold": threshold,
        "composition_order": "arithmetic_to_list",
    }


def _rewrite_mixed_case(case: dict[str, Any], *, variant_index: int) -> dict[str, Any]:
    values = case["question"].split("From the list ", 1)[1].split(", keep", 1)[0]
    offset = int(case["mixed_offset"])
    variants = [
        f"Use {values}. Keep the even entries, double them, add the doubled numbers, subtract {offset}, and return the integer.",
        f"Given {values}, discard odds, multiply each remaining value by 2, sum those products, then subtract {offset}.",
        f"Transform {values}: evens only, doubled, summed, then minus {offset}. Return only the final integer.",
        f"For {values}, filter to even values, double each survivor, total the doubled values, and subtract {offset}.",
        f"Process this list {values}: even-only, times two, sum, minus {offset}. Give the final integer.",
        f"From {values}, remove odd entries; double the rest; sum the doubled entries; subtract {offset}.",
        f"Keep evens in {values}, double those values, sum them, then subtract {offset}. What integer remains?",
        f"Apply an even-filter, double transform, sum, and subtract-{offset} step to {values}. Return the integer.",
        f"On {values}, retain only even numbers, double retained numbers, add them together, then reduce by {offset}.",
        f"Take list {values}; even values survive, survivors are doubled, the doubled values are summed, and {offset} is subtracted.",
        f"For the list {values}, compute sum(double(x) for even x), then subtract {offset}; return the integer.",
        f"Run this pipeline on {values}: filter even, map times-two, aggregate by sum, then subtract {offset}.",
        f"Use only even entries from {values}. Double them, total them, subtract {offset}, and output the final number.",
        f"Given the sequence {values}, ignore odd values; double even values; add those doubles; take away {offset}.",
        f"Evaluate the list pipeline for {values}: evens -> doubled evens -> sum -> minus {offset}.",
        f"From the numbers {values}, keep even items, multiply each kept item by two, sum, then subtract {offset}.",
        f"Filter {values} for even entries. After doubling and summing the filtered entries, subtract {offset}.",
        f"The answer is the sum of doubled even members of {values}, decreased by {offset}.",
        f"Process {values} in order: even-filter first, doubling second, summation third, subtract {offset} last.",
        f"For {values}, first collect evens, then double the collection, sum it, and finally subtract {offset}.",
        f"Compute the total of doubled even values in {values}; after that, subtract {offset}.",
        f"List task: {values}. Keep evens, double kept values, sum the doubled values, and remove {offset}.",
        f"Transform only even values from {values} by doubling, sum the transformed values, then subtract {offset}.",
        f"Starting with {values}, discard odds, double the remaining entries, add them, and subtract {offset}.",
    ]
    variant = int(variant_index) % len(variants)
    out = dict(case)
    out["id"] = f"{case.get('id', 'case')}-mixedstress{variant}"
    out["question"] = variants[variant]
    out["prompt"] = _prompt_for_question(out["question"])
    out["surface_distribution"] = "mixed_family_composition_surface_v1"
    out["surface_variant_index"] = variant
    return out


def _rewrite_arithmetic_to_list_case(
    case: dict[str, Any], *, variant_index: int
) -> dict[str, Any]:
    values = case["question"].split("for the list ", 1)[1].split(": keep", 1)[0]
    threshold_expr = case["question"].split("First compute ", 1)[1].split(". Use", 1)[0]
    variants = [
        f"Evaluate {threshold_expr}. With list {values}, keep even values greater than that result, double them, and return CSV or EMPTY.",
        f"Use {threshold_expr} as a threshold. From {values}, select even entries above the threshold, double selected entries, and print CSV.",
        f"Compute threshold {threshold_expr}; then process {values}: even-only, above-threshold, doubled. Return no-space CSV or EMPTY.",
        f"For {values}, first find {threshold_expr}. Keep only even numbers larger than the computed value, double them, and answer with CSV.",
        f"Threshold task: calculate {threshold_expr}, filter {values} to even values above it, then multiply each survivor by 2.",
        f"Calculate {threshold_expr}. From list {values}, discard values at or below that result and discard odds; double the rest.",
        f"Given list {values}, use computed threshold {threshold_expr}; output doubled even values that exceed the threshold.",
        f"Two-stage task. Arithmetic threshold: {threshold_expr}. List stage: {values}, keep even and above threshold, double, return CSV or EMPTY.",
        f"Find {threshold_expr}; then use list {values}. Only even entries strictly greater than the result survive, and each survivor is doubled.",
        f"First solve the arithmetic threshold {threshold_expr}. Then from {values}, return doubled even values whose original value is above that threshold.",
        f"Let T be {threshold_expr}. In {values}, keep even numbers with value greater than T, double them, and return CSV or EMPTY.",
        f"Compute T={threshold_expr}. For the list {values}, remove odd values and values not above T; double what remains.",
        f"Arithmetic first: {threshold_expr}. After that, scan {values} for even entries above the arithmetic result and double them.",
        f"Use the arithmetic result from {threshold_expr} to filter {values}: greater-than-result, even-only, then doubled.",
        f"Given numbers {values}, compare them against threshold {threshold_expr}; output doubled values for entries that are even and greater.",
        f"After calculating {threshold_expr}, process list {values} by keeping even above-threshold entries and doubling them.",
        f"Before touching list {values}, evaluate {threshold_expr}; then output doubled entries that are even and above that value.",
        f"The cutoff is the value of {threshold_expr}. For {values}, keep entries only if they are even and greater than the cutoff, then double.",
        f"First derive the numeric threshold from {threshold_expr}. Next, scan {values} and return doubled even values above it.",
        f"Do the arithmetic {threshold_expr} first; use its result to decide which even members of {values} survive, then double them.",
        f"Let the arithmetic stage be {threshold_expr}. Its answer filters {values}: even, strictly larger, doubled, CSV or EMPTY.",
        f"Evaluate the expression {threshold_expr}; for each item in {values}, keep it only when even and greater than the expression result, then double.",
        f"Use {threshold_expr} to set T. From {values}, output each even item greater than T after multiplying it by 2.",
        f"Threshold comes from {threshold_expr}; apply that threshold to list {values}, preserving only even greater values and doubling them.",
    ]
    variant = int(variant_index) % len(variants)
    out = dict(case)
    out["id"] = f"{case.get('id', 'case')}-mixedstress{variant}"
    out["question"] = variants[variant]
    out["prompt"] = _prompt_for_question(out["question"])
    out["surface_distribution"] = "mixed_family_composition_surface_v2"
    out["surface_variant_index"] = variant
    return out


def _mixed_rows(
    *,
    cases_per_family: int,
    start_index: int,
    list_lengths: Sequence[int],
    variants: Sequence[int],
    composition_orders: Sequence[str] = ("list_to_arithmetic",),
) -> list[dict[str, Any]]:
    order_groups: list[list[dict[str, Any]]] = []
    orders = _parse_composition_orders(composition_orders)
    for order_index, order in enumerate(orders):
        rows: list[dict[str, Any]] = []
        for length_index, list_length in enumerate(list_lengths):
            base_start = (
                int(start_index)
                + order_index * 100000
                + length_index * int(cases_per_family) * 100
            )
            for offset in range(int(cases_per_family)):
                if order == "list_to_arithmetic":
                    case = _make_mixed_case(
                        idx=base_start + offset,
                        list_length=int(list_length),
                    )
                    rewriter = _rewrite_mixed_case
                elif order == "arithmetic_to_list":
                    case = _make_arithmetic_to_list_case(
                        idx=base_start + offset,
                        list_length=int(list_length),
                    )
                    rewriter = _rewrite_arithmetic_to_list_case
                else:  # pragma: no cover - _parse_composition_orders validates.
                    raise ValueError(f"unknown composition order: {order!r}")
                for variant in variants:
                    rows.append(rewriter(case, variant_index=int(variant)))
        order_groups.append(rows)
    return _interleave_row_groups(*order_groups)


def _interleave_row_groups(*groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                out.append(group[index])
    return out


def _apply_terminal_v2(row: dict[str, Any]) -> dict[str, Any]:
    codebook_builder = _load_script_module(
        "229_build_pure_recursive_latent_action_codebook_cases.py"
    )
    return codebook_builder.apply_latent_action_codebook(
        row,
        codebook_version="dynamic_halt_v3",
        drop_solver_operation_names=False,
    )


def _densify(row: dict[str, Any], *, max_depth: int = 8) -> dict[str, Any]:
    dense_builder = _load_script_module("232_build_dense_transition_targets.py")
    return dense_builder.densify_transition_targets(
        row,
        max_depth=int(max_depth),
        finality_mode="answer_match",
    )


def _prepare(rows: list[dict[str, Any]], *, dense: bool) -> list[dict[str, Any]]:
    prepared = [_apply_terminal_v2(row) for row in rows]
    if dense:
        prepared = [_densify(row, max_depth=8) for row in prepared]
    return prepared


def build_mixed_family_composition_gate(
    *,
    cases_per_family: int = 8,
    train_start_index: int = 40000,
    eval_start_index: int = 50000,
    train_list_variants: Sequence[int] = (0, 1, 2, 3, 4, 5),
    eval_list_variants: Sequence[int] = (6, 7),
    train_list_lengths: Sequence[int] | None = None,
    eval_list_lengths: Sequence[int] | None = None,
    mixed_repeat: int = 1,
    composition_orders: Sequence[str] | None = None,
    dense: bool = True,
    allow_variant_overlap: bool = False,
) -> MixedFamilyCompositionBundle:
    train_variants = _parse_variants(train_list_variants)
    eval_variants = _parse_variants(eval_list_variants)
    train_lengths = _parse_lengths(train_list_lengths) if train_list_lengths is not None else (5,)
    eval_lengths = _parse_lengths(eval_list_lengths)
    orders = _parse_composition_orders(composition_orders)
    overlap = sorted(set(train_variants) & set(eval_variants))
    if overlap and not bool(allow_variant_overlap):
        raise ValueError(f"train/eval list variants overlap: {overlap}")
    if int(cases_per_family) <= 0:
        raise ValueError("cases_per_family must be positive")
    if int(mixed_repeat) <= 0:
        raise ValueError("mixed_repeat must be positive")

    raw_train = _stress_rows(
        cases_per_family=int(cases_per_family),
        start_index=int(train_start_index),
    )
    selected_train = [
        row
        for row in raw_train
        if _family(row) != "list_transform" or _variant(row) in train_variants
    ]
    for row in selected_train:
        if _family(row) == "list_transform":
            row.setdefault("list_length", 5)
            row.setdefault("list_value_start", int(train_start_index) + 1)

    mixed_train = _mixed_rows(
        cases_per_family=int(cases_per_family),
        start_index=int(train_start_index),
        list_lengths=train_lengths,
        variants=train_variants,
        composition_orders=orders,
    )
    repeated_mixed_train: list[dict[str, Any]] = []
    for repeat_index in range(int(mixed_repeat)):
        for row in mixed_train:
            repeated = copy.deepcopy(row)
            repeated["id"] = f"{row.get('id', 'mixed')}-repeat{repeat_index}"
            repeated["mixed_repeat_index"] = repeat_index
            repeated_mixed_train.append(repeated)
    mixed_eval = _mixed_rows(
        cases_per_family=int(cases_per_family),
        start_index=int(eval_start_index),
        list_lengths=eval_lengths,
        variants=eval_variants,
        composition_orders=orders,
    )
    train_rows = _prepare(
        _interleave_row_groups(selected_train, repeated_mixed_train),
        dense=bool(dense),
    )
    eval_rows = _prepare(mixed_eval, dense=bool(dense))
    summary = {
        "split_type": "mixed_family_composition_holdout",
        "cases_per_family": int(cases_per_family),
        "train_start_index": int(train_start_index),
        "eval_start_index": int(eval_start_index),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "mixed_train_rows": len(repeated_mixed_train),
        "mixed_eval_rows": len(mixed_eval),
        "mixed_repeat": int(mixed_repeat),
        "composition_orders": list(orders),
        "dense": bool(dense),
        "codebook_version": "dynamic_halt_v3",
        "finality_mode": "answer_match",
        "train_families": sorted({_family(row) for row in train_rows}),
        "eval_families": sorted({_family(row) for row in eval_rows}),
        "train_list_variants": list(train_variants),
        "eval_list_variants": list(eval_variants),
        "train_list_lengths": sorted(
            {
                int(row.get("list_length", 5))
                for row in train_rows
                if _family(row)
                in {"list_transform", "mixed_list_arithmetic", "mixed_arithmetic_list"}
            }
        ),
        "eval_list_lengths": sorted(
            {
                int(row.get("list_length", 5))
                for row in eval_rows
                if _family(row) in {"mixed_list_arithmetic", "mixed_arithmetic_list"}
            }
        ),
        "variant_overlap": overlap,
        "allow_variant_overlap": bool(allow_variant_overlap),
        "architecture_note": (
            "Mixed-family gate composes list filtering/doubling with arithmetic "
            "sum/subtract steps. Optional arithmetic_to_list reverses the order. "
            "Finality uses answer_match so halt is separated from action-code "
            "identity."
        ),
    }
    return MixedFamilyCompositionBundle(
        train_rows=train_rows,
        eval_rows=eval_rows,
        summary=summary,
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_mixed_family_composition_gate(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    summary_out: str | Path,
    cases_per_family: int = 8,
    train_start_index: int = 40000,
    eval_start_index: int = 50000,
    train_list_variants: Sequence[int] = (0, 1, 2, 3, 4, 5),
    eval_list_variants: Sequence[int] = (6, 7),
    train_list_lengths: Sequence[int] | None = None,
    eval_list_lengths: Sequence[int] | None = None,
    mixed_repeat: int = 1,
    composition_orders: Sequence[str] | None = None,
    dense: bool = True,
    allow_variant_overlap: bool = False,
) -> MixedFamilyCompositionBundle:
    bundle = build_mixed_family_composition_gate(
        cases_per_family=cases_per_family,
        train_start_index=train_start_index,
        eval_start_index=eval_start_index,
        train_list_variants=train_list_variants,
        eval_list_variants=eval_list_variants,
        train_list_lengths=train_list_lengths,
        eval_list_lengths=eval_list_lengths,
        mixed_repeat=int(mixed_repeat),
        composition_orders=composition_orders,
        dense=dense,
        allow_variant_overlap=bool(allow_variant_overlap),
    )
    _write_jsonl(train_out, bundle.train_rows)
    _write_jsonl(eval_out, bundle.eval_rows)
    out = Path(summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(bundle.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a mixed list->arithmetic composition transfer gate."
    )
    parser.add_argument(
        "--train-out",
        default="data/filtered/pure_recursive_transition_joint_dense_terminal_v2_mixed_composition_train40000_v0to5.jsonl",
    )
    parser.add_argument(
        "--eval-out",
        default="data/eval/pure_recursive_transition_joint_dense_terminal_v2_mixed_composition_eval50000_v6to7_len7_9.jsonl",
    )
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--train-start-index", type=int, default=40000)
    parser.add_argument("--eval-start-index", type=int, default=50000)
    parser.add_argument("--train-list-variants", default="0,1,2,3,4,5")
    parser.add_argument("--eval-list-variants", default="6,7")
    parser.add_argument("--train-list-lengths", default="5")
    parser.add_argument("--eval-list-lengths", default="7,9")
    parser.add_argument("--mixed-repeat", type=int, default=1)
    parser.add_argument("--composition-orders", default="list_to_arithmetic")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument(
        "--allow-variant-overlap",
        action="store_true",
        help=(
            "Allow train/eval surface-variant overlap for diagnostic splits. "
            "Keep disabled for canonical held-out gates."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_out = (
        args.summary_out
        or str(Path(args.train_out).with_suffix(".summary.json"))
    )
    bundle = write_mixed_family_composition_gate(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=summary_out,
        cases_per_family=args.cases_per_family,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_list_variants=_parse_variants(args.train_list_variants),
        eval_list_variants=_parse_variants(args.eval_list_variants),
        train_list_lengths=_parse_lengths(args.train_list_lengths),
        eval_list_lengths=_parse_lengths(args.eval_list_lengths),
        mixed_repeat=args.mixed_repeat,
        composition_orders=_parse_composition_orders(args.composition_orders),
        dense=not bool(args.no_dense),
        allow_variant_overlap=bool(args.allow_variant_overlap),
    )
    print(
        "wrote mixed-family composition gate: "
        f"train={bundle.summary['train_rows']} eval={bundle.summary['eval_rows']} "
        f"summary={summary_out}"
    )


if __name__ == "__main__":
    main()
