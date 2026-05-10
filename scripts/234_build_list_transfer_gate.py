#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence


class ListTransferGateBundle(NamedTuple):
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


def _parse_lengths(value: str | Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        lengths = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    else:
        lengths = tuple(int(item) for item in value)
    if not lengths:
        return None
    bad = [length for length in lengths if length <= 0]
    if bad:
        raise ValueError(f"list lengths must be positive, got {bad}")
    return tuple(sorted(set(lengths)))


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
    bad = [variant for variant in variants if variant < 0 or variant > 7]
    if bad:
        raise ValueError(f"surface variants must be in [0, 7], got {bad}")
    return tuple(sorted(set(variants)))


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


def _make_list_case(*, idx: int, list_length: int) -> dict[str, Any]:
    values = [int(idx) + offset + 1 for offset in range(int(list_length))]
    filtered = [value for value in values if value % 2 == 0]
    transformed = [value * 2 for value in filtered]
    filtered_answer = ",".join(str(value) for value in filtered) if filtered else "EMPTY"
    answer = ",".join(str(value) for value in transformed) if transformed else "EMPTY"
    question = (
        "From the list "
        f"{values}, keep only even numbers, double each kept number, "
        "and return comma-separated values with no spaces. If none, return EMPTY."
    )
    depth_targets = {"1": filtered_answer, "2": answer, "4": answer, "8": answer}
    return {
        "id": f"list-transform-long-{idx}-len{int(list_length)}",
        "raw_intelligence_axis": "pure_recursive_reasoning",
        "category": "list_transform",
        "task_family": "list_transform",
        "reasoning_family": "sequential_list_transform",
        "expected_paradigm": "hybrid_or_cot",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 2,
        "question": question,
        "prompt": _prompt_for_question(question),
        "answer": answer,
        "chosen": answer,
        "answer_aliases": [answer],
        "choices": [answer, filtered_answer, "EMPTY"],
        "depth_targets": depth_targets,
        "transition_state_codes": {"1": 40, "2": 41, "4": 41, "8": 41},
        "solver_trace": [
            {"depth": 1, "operation": "filter_even", "state_text": filtered_answer},
            {"depth": 2, "operation": "double_filtered", "state_text": answer},
            {"depth": 4, "operation": "hold_final", "state_text": answer},
            {"depth": 8, "operation": "hold_final", "state_text": answer},
        ],
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "list_length": int(list_length),
        "list_value_start": int(idx) + 1,
    }


def _rewrite_list_case(case: dict[str, Any], *, variant_index: int) -> dict[str, Any]:
    stress_builder = _load_script_module(
        "227_build_pure_recursive_ood_paraphrase_stress_cases.py"
    )
    out = stress_builder.rewrite_case_surface_stress(
        case,
        variant_index=int(variant_index),
    )
    out["list_length"] = int(case.get("list_length", 5))
    out["list_value_start"] = int(case.get("list_value_start", 0))
    return out


def _long_list_eval_rows(
    *,
    cases_per_length: int,
    start_index: int,
    list_lengths: Sequence[int],
    eval_variants: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_index, list_length in enumerate(list_lengths):
        base_start = int(start_index) + length_index * int(cases_per_length) * 100
        for offset in range(int(cases_per_length)):
            case = _make_list_case(idx=base_start + offset, list_length=int(list_length))
            for variant in eval_variants:
                rows.append(_rewrite_list_case(case, variant_index=int(variant)))
    return rows


def _apply_terminal_v2(row: dict[str, Any]) -> dict[str, Any]:
    codebook_builder = _load_script_module(
        "229_build_pure_recursive_latent_action_codebook_cases.py"
    )
    return codebook_builder.apply_latent_action_codebook(
        row,
        codebook_version="terminal_v2",
    )


def _densify(row: dict[str, Any], *, max_depth: int = 8) -> dict[str, Any]:
    dense_builder = _load_script_module("232_build_dense_transition_targets.py")
    return dense_builder.densify_transition_targets(
        row,
        max_depth=int(max_depth),
        finality_mode="action_terminal",
    )


def _prepare(rows: list[dict[str, Any]], *, dense: bool) -> list[dict[str, Any]]:
    prepared = [_apply_terminal_v2(row) for row in rows]
    if dense:
        prepared = [_densify(row, max_depth=8) for row in prepared]
    return prepared


def build_list_transfer_gate(
    *,
    cases_per_family: int = 8,
    train_start_index: int = 18000,
    eval_start_index: int = 19000,
    train_list_variants: Sequence[int] = (0, 1, 2, 3, 4, 5),
    eval_list_variants: Sequence[int] = (6, 7),
    eval_list_lengths: Sequence[int] | None = None,
    dense: bool = True,
) -> ListTransferGateBundle:
    train_variants = _parse_variants(train_list_variants)
    eval_variants = _parse_variants(eval_list_variants)
    eval_lengths = _parse_lengths(eval_list_lengths)
    overlap = sorted(set(train_variants) & set(eval_variants))
    if overlap:
        raise ValueError(f"train/eval list variants overlap: {overlap}")
    if int(cases_per_family) <= 0:
        raise ValueError("cases_per_family must be positive")

    raw_train = _stress_rows(
        cases_per_family=int(cases_per_family),
        start_index=int(train_start_index),
    )
    if eval_lengths:
        raw_eval = _long_list_eval_rows(
            cases_per_length=int(cases_per_family),
            start_index=int(eval_start_index),
            list_lengths=eval_lengths,
            eval_variants=eval_variants,
        )
    else:
        raw_eval = _stress_rows(
            cases_per_family=int(cases_per_family),
            start_index=int(eval_start_index),
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
    selected_eval = [
        row
        for row in raw_eval
        if _family(row) == "list_transform" and _variant(row) in eval_variants
    ]
    train_rows = _prepare(selected_train, dense=bool(dense))
    eval_rows = _prepare(selected_eval, dense=bool(dense))
    summary = {
        "split_type": "list_paraphrase_cluster_holdout",
        "cases_per_family": int(cases_per_family),
        "train_start_index": int(train_start_index),
        "eval_start_index": int(eval_start_index),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "dense": bool(dense),
        "codebook_version": "terminal_v2",
        "finality_mode": "action_terminal" if dense else "source",
        "train_families": sorted({_family(row) for row in train_rows}),
        "eval_families": sorted({_family(row) for row in eval_rows}),
        "train_list_variants": list(train_variants),
        "eval_list_variants": list(eval_variants),
        "train_list_lengths": sorted(
            {
                int(row.get("list_length", 5))
                for row in train_rows
                if _family(row) == "list_transform"
            }
        ),
        "eval_list_lengths": sorted(
            {
                int(row.get("list_length", 5))
                for row in eval_rows
                if _family(row) == "list_transform"
            }
        ),
        "variant_overlap": overlap,
        "architecture_note": (
            "List family is present in training, but selected list surface "
            "paraphrase variants are held out for transfer evaluation."
        ),
    }
    return ListTransferGateBundle(train_rows=train_rows, eval_rows=eval_rows, summary=summary)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_list_transfer_gate(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    summary_out: str | Path,
    cases_per_family: int = 8,
    train_start_index: int = 18000,
    eval_start_index: int = 19000,
    train_list_variants: Sequence[int] = (0, 1, 2, 3, 4, 5),
    eval_list_variants: Sequence[int] = (6, 7),
    eval_list_lengths: Sequence[int] | None = None,
    dense: bool = True,
) -> ListTransferGateBundle:
    bundle = build_list_transfer_gate(
        cases_per_family=cases_per_family,
        train_start_index=train_start_index,
        eval_start_index=eval_start_index,
        train_list_variants=train_list_variants,
        eval_list_variants=eval_list_variants,
        eval_list_lengths=eval_list_lengths,
        dense=dense,
    )
    write_jsonl(train_out, bundle.train_rows)
    write_jsonl(eval_out, bundle.eval_rows)
    out = Path(summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(bundle.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Stage 1 list transfer gate where list_transform is present "
            "in train, but selected list paraphrase variants are held out."
        )
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--train-start-index", type=int, default=18000)
    parser.add_argument("--eval-start-index", type=int, default=19000)
    parser.add_argument("--train-list-variants", default="0,1,2,3,4,5")
    parser.add_argument("--eval-list-variants", default="6,7")
    parser.add_argument(
        "--eval-list-lengths",
        default="",
        help="Optional comma-separated list lengths for long-list eval rows.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Write sparse terminal_v2 codebook rows instead of dense action-terminal rows.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = write_list_transfer_gate(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=args.summary_out,
        cases_per_family=args.cases_per_family,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_list_variants=_parse_variants(args.train_list_variants),
        eval_list_variants=_parse_variants(args.eval_list_variants),
        eval_list_lengths=_parse_lengths(args.eval_list_lengths),
        dense=not bool(args.sparse),
    )
    print(
        "wrote list transfer gate: "
        f"train={len(bundle.train_rows)} eval={len(bundle.eval_rows)} "
        f"train_list_variants={bundle.summary['train_list_variants']} "
        f"eval_list_variants={bundle.summary['eval_list_variants']}"
    )


if __name__ == "__main__":
    main()
