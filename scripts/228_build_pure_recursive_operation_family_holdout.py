#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple


class OperationFamilyHoldoutBundle(NamedTuple):
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


def _operations(rows: list[dict[str, Any]]) -> set[str]:
    operations: set[str] = set()
    for row in rows:
        for step in row.get("solver_trace") or []:
            operation = str(step.get("operation") or "")
            if operation:
                operations.add(operation)
    return operations


def _build_stress_rows(
    *,
    cases_per_family: int,
    start_index: int,
    stress_variants_per_case: int,
) -> list[dict[str, Any]]:
    stress_builder = _load_script_module(
        "227_build_pure_recursive_ood_paraphrase_stress_cases.py"
    )
    return stress_builder.build_ood_paraphrase_stress_cases(
        cases_per_family=int(cases_per_family),
        start_index=int(start_index),
        variants_per_case=int(stress_variants_per_case),
    )


def build_operation_family_holdout(
    *,
    holdout_family: str = "list_transform",
    cases_per_family: int = 8,
    start_index: int = 12000,
    stress_variants_per_case: int = 8,
) -> OperationFamilyHoldoutBundle:
    if int(cases_per_family) <= 0:
        raise ValueError("cases_per_family must be positive")
    if int(stress_variants_per_case) <= 0:
        raise ValueError("stress_variants_per_case must be positive")

    rows = _build_stress_rows(
        cases_per_family=int(cases_per_family),
        start_index=int(start_index),
        stress_variants_per_case=int(stress_variants_per_case),
    )
    families = sorted({_family(row) for row in rows})
    if holdout_family not in families:
        raise ValueError(
            f"holdout_family={holdout_family!r} not present in generated families: {families}"
        )

    train_rows = [row for row in rows if _family(row) != holdout_family]
    eval_rows = [row for row in rows if _family(row) == holdout_family]
    train_operations = _operations(train_rows)
    eval_operations = _operations(eval_rows)
    unseen_eval_operations = sorted(eval_operations - train_operations)
    shared_operations = sorted(eval_operations & train_operations)

    summary = {
        "holdout_family": holdout_family,
        "cases_per_family": int(cases_per_family),
        "start_index": int(start_index),
        "stress_variants_per_case": int(stress_variants_per_case),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_families": sorted({_family(row) for row in train_rows}),
        "eval_families": sorted({_family(row) for row in eval_rows}),
        "train_operations": sorted(train_operations),
        "eval_operations": sorted(eval_operations),
        "shared_operations": shared_operations,
        "unseen_eval_operations": unseen_eval_operations,
        "interpretation": (
            "reject_fixed_label_transfer"
            if unseen_eval_operations
            else "fixed_label_transfer_feasible"
        ),
        "architecture_note": (
            "Full operation-family holdout is not a fair acceptance gate for the "
            "current fixed-label primitive scaffold when eval requires operation "
            "ids absent from train. Use it as a reject/diagnostic gate and move "
            "the accepted path to a learned latent operation codebook or neural "
            "transition model."
        ),
    }
    return OperationFamilyHoldoutBundle(
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


def write_operation_family_holdout(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    summary_out: str | Path,
    holdout_family: str = "list_transform",
    cases_per_family: int = 8,
    start_index: int = 12000,
    stress_variants_per_case: int = 8,
) -> OperationFamilyHoldoutBundle:
    bundle = build_operation_family_holdout(
        holdout_family=holdout_family,
        cases_per_family=cases_per_family,
        start_index=start_index,
        stress_variants_per_case=stress_variants_per_case,
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
        description=(
            "Build an operation-family holdout diagnostic split for the pure "
            "recursive primitive scaffold."
        )
    )
    parser.add_argument("--holdout-family", default="list_transform")
    parser.add_argument(
        "--train-out",
        default=(
            "data/filtered/"
            "pure_recursive_primitive_transition_family_holdout_list_train.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-out",
        default=(
            "data/eval/"
            "pure_recursive_primitive_transition_family_holdout_list_eval.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=(
            "local_eval/"
            "qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/"
            "operation_family_holdout_list_summary.json"
        ),
    )
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=12000)
    parser.add_argument("--stress-variants-per-case", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = write_operation_family_holdout(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=args.summary_out,
        holdout_family=args.holdout_family,
        cases_per_family=args.cases_per_family,
        start_index=args.start_index,
        stress_variants_per_case=args.stress_variants_per_case,
    )
    print(
        "wrote operation-family holdout split: "
        f"train={len(bundle.train_rows)} eval={len(bundle.eval_rows)} "
        f"unseen_eval_operations={bundle.summary['unseen_eval_operations']}"
    )


if __name__ == "__main__":
    main()
