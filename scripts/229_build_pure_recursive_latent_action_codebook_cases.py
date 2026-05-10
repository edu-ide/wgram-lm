#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, NamedTuple


ROLE_V1_LATENT_ACTION_CODEBOOK: dict[str, int] = {
    "extract_or_unary_transform": 0,
    "compose_from_previous": 1,
    "final_compose_from_previous": 2,
    "hold_final": 3,
}

ROLE_V1_OPERATION_TO_LATENT_ACTION: dict[str, str] = {
    "add_operands": "extract_or_unary_transform",
    "first_mapping": "extract_or_unary_transform",
    "not_q": "extract_or_unary_transform",
    "filter_even": "extract_or_unary_transform",
    "filter_above_threshold": "compose_from_previous",
    "multiply_sum": "compose_from_previous",
    "second_mapping": "compose_from_previous",
    "and_with_p": "compose_from_previous",
    "double_filtered": "compose_from_previous",
    "subtract_offset": "final_compose_from_previous",
    "or_with_r": "final_compose_from_previous",
    "hold_final": "hold_final",
}

TERMINAL_V2_LATENT_ACTION_CODEBOOK: dict[str, int] = {
    "extract_or_unary_nonterminal": 0,
    "compose_from_previous_terminal": 1,
    "compose_from_previous_nonterminal": 2,
    "final_compose_from_previous_terminal": 3,
    "hold_final": 4,
}

TERMINAL_V2_OPERATION_TO_LATENT_ACTION: dict[str, str] = {
    "add_operands": "extract_or_unary_nonterminal",
    "first_mapping": "extract_or_unary_nonterminal",
    "not_q": "extract_or_unary_nonterminal",
    "filter_even": "extract_or_unary_nonterminal",
    "filter_above_threshold": "compose_from_previous_nonterminal",
    "second_mapping": "compose_from_previous_terminal",
    "double_filtered": "compose_from_previous_terminal",
    "multiply_sum": "compose_from_previous_nonterminal",
    "and_with_p": "compose_from_previous_nonterminal",
    "subtract_offset": "final_compose_from_previous_terminal",
    "or_with_r": "final_compose_from_previous_terminal",
    "hold_final": "hold_final",
}

DYNAMIC_HALT_V3_LATENT_ACTION_CODEBOOK: dict[str, int] = {
    "extract_or_unary_transform": 0,
    "compose_from_previous": 1,
    "aggregate_from_previous": 2,
    "final_compose_from_previous": 3,
    "hold_final": 4,
}

DYNAMIC_HALT_V3_OPERATION_TO_LATENT_ACTION: dict[str, str] = {
    "add_operands": "extract_or_unary_transform",
    "first_mapping": "extract_or_unary_transform",
    "not_q": "extract_or_unary_transform",
    "filter_even": "extract_or_unary_transform",
    "filter_above_threshold": "compose_from_previous",
    "second_mapping": "compose_from_previous",
    "double_filtered": "compose_from_previous",
    "multiply_sum": "aggregate_from_previous",
    "and_with_p": "aggregate_from_previous",
    "subtract_offset": "final_compose_from_previous",
    "or_with_r": "final_compose_from_previous",
    "hold_final": "hold_final",
}

LATENT_ACTION_CODEBOOKS: dict[str, dict[str, int]] = {
    "role_v1": ROLE_V1_LATENT_ACTION_CODEBOOK,
    "terminal_v2": TERMINAL_V2_LATENT_ACTION_CODEBOOK,
    "dynamic_halt_v3": DYNAMIC_HALT_V3_LATENT_ACTION_CODEBOOK,
}

OPERATION_TO_LATENT_ACTION_BY_VERSION: dict[str, dict[str, str]] = {
    "role_v1": ROLE_V1_OPERATION_TO_LATENT_ACTION,
    "terminal_v2": TERMINAL_V2_OPERATION_TO_LATENT_ACTION,
    "dynamic_halt_v3": DYNAMIC_HALT_V3_OPERATION_TO_LATENT_ACTION,
}


class LatentActionCodebookBundle(NamedTuple):
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
            if not isinstance(step, dict):
                continue
            operation = str(step.get("operation") or "")
            if operation:
                operations.add(operation)
    return operations


def _latent_action_names(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        for step in row.get("latent_action_trace") or []:
            if not isinstance(step, dict):
                continue
            name = str(step.get("action_name") or "")
            if name:
                names.add(name)
    return names


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


def apply_latent_action_codebook(
    row: dict[str, Any],
    *,
    codebook_version: str = "role_v1",
    drop_solver_operation_names: bool = True,
    include_source_operations: bool = False,
) -> dict[str, Any]:
    if codebook_version not in LATENT_ACTION_CODEBOOKS:
        raise ValueError(f"unknown codebook_version: {codebook_version!r}")
    latent_action_codebook = LATENT_ACTION_CODEBOOKS[codebook_version]
    operation_to_latent_action = OPERATION_TO_LATENT_ACTION_BY_VERSION[
        codebook_version
    ]
    out = copy.deepcopy(row)
    solver_trace = out.get("solver_trace")
    if not isinstance(solver_trace, list):
        raise ValueError("row must contain solver_trace for latent action remapping")

    transition_state_codes: dict[str, int] = {}
    transition_finality_targets: dict[str, int] = {}
    latent_trace: list[dict[str, Any]] = []
    cleaned_solver_trace: list[dict[str, Any]] = []
    final_answer = str((out.get("answer_aliases") or [out.get("answer", "")])[0])
    for step in solver_trace:
        if not isinstance(step, dict):
            raise ValueError("solver_trace steps must be objects")
        depth = int(step.get("depth"))
        operation = str(step.get("operation") or "")
        if operation not in operation_to_latent_action:
            raise ValueError(f"unsupported primitive operation: {operation!r}")
        action_name = operation_to_latent_action[operation]
        action_code = int(latent_action_codebook[action_name])
        transition_state_codes[str(depth)] = action_code
        state_text = str(step.get("state_text") or "")
        transition_finality_targets[str(depth)] = int(state_text == final_answer)
        latent_step: dict[str, Any] = {
            "depth": depth,
            "action_code": action_code,
            "action_name": action_name,
        }
        if include_source_operations:
            latent_step["source_operation"] = operation
        latent_trace.append(latent_step)

        cleaned = dict(step)
        if drop_solver_operation_names:
            cleaned.pop("operation", None)
        cleaned_solver_trace.append(cleaned)

    out["transition_state_codes"] = transition_state_codes
    out["transition_finality_targets"] = transition_finality_targets
    out["latent_action_trace"] = latent_trace
    out["latent_action_codebook_applied"] = True
    out["latent_action_codebook_version"] = codebook_version
    out["latent_action_codebook_size"] = len(latent_action_codebook)
    out["solver_trace"] = cleaned_solver_trace
    return out


def build_latent_action_codebook_holdout(
    *,
    holdout_family: str = "list_transform",
    cases_per_family: int = 8,
    start_index: int = 13000,
    stress_variants_per_case: int = 8,
    codebook_version: str = "role_v1",
    drop_solver_operation_names: bool = True,
) -> LatentActionCodebookBundle:
    if codebook_version not in LATENT_ACTION_CODEBOOKS:
        raise ValueError(f"unknown codebook_version: {codebook_version!r}")
    latent_action_codebook = LATENT_ACTION_CODEBOOKS[codebook_version]
    operation_to_latent_action = OPERATION_TO_LATENT_ACTION_BY_VERSION[
        codebook_version
    ]
    if int(cases_per_family) <= 0:
        raise ValueError("cases_per_family must be positive")
    if int(stress_variants_per_case) <= 0:
        raise ValueError("stress_variants_per_case must be positive")

    raw_rows = _build_stress_rows(
        cases_per_family=int(cases_per_family),
        start_index=int(start_index),
        stress_variants_per_case=int(stress_variants_per_case),
    )
    families = sorted({_family(row) for row in raw_rows})
    if holdout_family not in families:
        raise ValueError(
            f"holdout_family={holdout_family!r} not present in generated families: {families}"
        )

    raw_train_rows = [row for row in raw_rows if _family(row) != holdout_family]
    raw_eval_rows = [row for row in raw_rows if _family(row) == holdout_family]
    train_rows = [
        apply_latent_action_codebook(
            row,
            codebook_version=codebook_version,
            drop_solver_operation_names=drop_solver_operation_names,
        )
        for row in raw_train_rows
    ]
    eval_rows = [
        apply_latent_action_codebook(
            row,
            codebook_version=codebook_version,
            drop_solver_operation_names=drop_solver_operation_names,
        )
        for row in raw_eval_rows
    ]

    train_operations = _operations(raw_train_rows)
    eval_operations = _operations(raw_eval_rows)
    train_action_names = _latent_action_names(train_rows)
    eval_action_names = _latent_action_names(eval_rows)
    name_to_code = dict(latent_action_codebook)
    train_codes = sorted(name_to_code[name] for name in train_action_names)
    eval_codes = sorted(name_to_code[name] for name in eval_action_names)
    unseen_eval_action_names = sorted(eval_action_names - train_action_names)
    unseen_eval_action_codes = [
        int(name_to_code[name]) for name in unseen_eval_action_names
    ]

    summary = {
        "holdout_family": holdout_family,
        "cases_per_family": int(cases_per_family),
        "start_index": int(start_index),
        "stress_variants_per_case": int(stress_variants_per_case),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_families": sorted({_family(row) for row in train_rows}),
        "eval_families": sorted({_family(row) for row in eval_rows}),
        "drop_solver_operation_names": bool(drop_solver_operation_names),
        "codebook_version": codebook_version,
        "latent_action_codebook_size": len(latent_action_codebook),
        "latent_action_codebook": name_to_code,
        "operation_to_latent_action": dict(operation_to_latent_action),
        "train_operations": sorted(train_operations),
        "eval_operations": sorted(eval_operations),
        "unseen_eval_operations": sorted(eval_operations - train_operations),
        "train_latent_action_names": sorted(train_action_names),
        "eval_latent_action_names": sorted(eval_action_names),
        "train_latent_action_codes": train_codes,
        "eval_latent_action_codes": eval_codes,
        "unseen_eval_latent_action_names": unseen_eval_action_names,
        "unseen_eval_latent_action_codes": unseen_eval_action_codes,
        "interpretation": (
            "latent_action_transfer_feasible"
            if not unseen_eval_action_codes
            else "latent_action_transfer_rejected"
        ),
        "architecture_note": (
            "This Stage 1 diagnostic replaces operation-string targets with "
            "family-agnostic latent action roles. It makes list-family holdout "
            "structurally feasible for the transition-state code path, but it "
            "does not yet prove neural execution or open-ended reasoning."
        ),
    }
    return LatentActionCodebookBundle(
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


def write_latent_action_codebook_holdout(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    summary_out: str | Path,
    holdout_family: str = "list_transform",
    cases_per_family: int = 8,
    start_index: int = 13000,
    stress_variants_per_case: int = 8,
    codebook_version: str = "role_v1",
    drop_solver_operation_names: bool = True,
) -> LatentActionCodebookBundle:
    bundle = build_latent_action_codebook_holdout(
        holdout_family=holdout_family,
        cases_per_family=cases_per_family,
        start_index=start_index,
        stress_variants_per_case=stress_variants_per_case,
        codebook_version=codebook_version,
        drop_solver_operation_names=drop_solver_operation_names,
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
            "Build Stage 1 latent-action codebook train/eval rows from pure "
            "recursive primitive reasoning cases."
        )
    )
    parser.add_argument("--holdout-family", default="list_transform")
    parser.add_argument(
        "--train-out",
        default=(
            "data/filtered/"
            "pure_recursive_latent_action_codebook_family_holdout_list_train.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-out",
        default=(
            "data/eval/"
            "pure_recursive_latent_action_codebook_family_holdout_list_eval.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=(
            "local_eval/"
            "qwen35_2b_pure_recursive_primitive_transition_oodstress_s1024_from_surface_aug/"
            "latent_action_codebook_family_holdout_list_summary.json"
        ),
    )
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=13000)
    parser.add_argument("--stress-variants-per-case", type=int, default=8)
    parser.add_argument(
        "--codebook-version",
        choices=sorted(LATENT_ACTION_CODEBOOKS),
        default="role_v1",
    )
    parser.add_argument(
        "--keep-solver-operation-names",
        action="store_true",
        help="Keep source operation strings in solver_trace for audit-only datasets.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = write_latent_action_codebook_holdout(
        train_out=args.train_out,
        eval_out=args.eval_out,
        summary_out=args.summary_out,
        holdout_family=args.holdout_family,
        cases_per_family=args.cases_per_family,
        start_index=args.start_index,
        stress_variants_per_case=args.stress_variants_per_case,
        codebook_version=args.codebook_version,
        drop_solver_operation_names=not bool(args.keep_solver_operation_names),
    )
    print(
        "wrote latent-action codebook split: "
        f"train={len(bundle.train_rows)} eval={len(bundle.eval_rows)} "
        f"unseen_eval_latent_action_codes={bundle.summary['unseen_eval_latent_action_codes']}"
    )


if __name__ == "__main__":
    main()
