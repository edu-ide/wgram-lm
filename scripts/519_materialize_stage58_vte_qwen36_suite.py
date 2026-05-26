#!/usr/bin/env python3
"""Materialize the Stage58 VTE heldout suite for Qwen3.6 baseline runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_script(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage518 = _load_script("518_train_token_local_register_extractor.py", "qtrm_stage518_for_stage58_suite")


def qwen_prompt(case: Any, *, mode: str) -> str:
    if mode == "same":
        return str(case.prompt_text)
    if mode == "answer_only":
        prompt = str(case.prompt_text).replace("Answer:", "").strip()
        return (
            "Solve the modulo-10 arithmetic task below.\n"
            "Output exactly one digit from 0 to 9.\n"
            "Do not explain. Do not write words.\n\n"
            f"{prompt}\n"
            "Final digit:"
        )
    raise ValueError(f"unknown qwen prompt mode: {mode}")


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    case_args = SimpleNamespace(
        reasoning_condition_prefix=str(args.reasoning_condition_prefix),
        synthetic_family_mix=str(args.synthetic_family_mix),
        synthetic_sampling_strategy=str(args.synthetic_sampling_strategy),
    )
    rows: list[dict[str, Any]] = []
    for depth in args.eval_depths:
        cases = stage518.build_cases(
            count=int(args.eval_count),
            seed=int(args.eval_seed) + int(depth),
            depths=[int(depth)],
            max_steps=int(args.max_steps),
            args=case_args,
            surface_mode=str(args.eval_surface_mode),
        )
        for index, case in enumerate(cases):
            rows.append(
                {
                    "suite_id": str(args.suite_id),
                    "prompt_protocol": str(args.prompt_protocol),
                    "case_id": f"stage58_d{int(depth):02d}_{index:04d}",
                    "family": str(case.family),
                    "depth": int(case.depth),
                    "initial_label": int(case.initial_label),
                    "operation_ids": [int(value) for value in list(case.operation_ids)[: int(case.depth)]],
                    "operation_args": [int(value) for value in list(case.operation_args or [])[: int(case.depth)]],
                    "state_labels": [int(value) for value in list(case.state_labels)[: int(case.depth)]],
                    "qwen_prompt": qwen_prompt(case, mode=str(args.qwen_prompt_mode)),
                    "answer_text": str(int(case.answer_label)),
                    "answer_value": int(case.answer_label),
                }
            )
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", default="stage58_vte_mod10_heldout_v1")
    parser.add_argument("--prompt-protocol", default="stage58_same_prompt_single_digit_v1")
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--eval-depths", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--reasoning-condition-prefix", default="synth")
    parser.add_argument("--synthetic-family-mix", default="balanced")
    parser.add_argument("--synthetic-sampling-strategy", default="random")
    parser.add_argument("--eval-surface-mode", choices=("canonical", "ledger", "prose", "heldout", "mixed", "mixed_all"), default="heldout")
    parser.add_argument("--qwen-prompt-mode", choices=("same", "answer_only"), default="same")
    parser.add_argument("--out-jsonl", default="local_eval/stage58_vte_qwen36_suite/cases.jsonl")
    parser.add_argument("--out-meta", default="local_eval/stage58_vte_qwen36_suite/metadata.json")
    args = parser.parse_args()

    rows = build_rows(args)
    write_jsonl(args.out_jsonl, rows)
    meta = {
        "suite_id": str(args.suite_id),
        "prompt_protocol": str(args.prompt_protocol),
        "case_count": len(rows),
        "eval_count_per_depth": int(args.eval_count),
        "eval_seed": int(args.eval_seed),
        "eval_depths": [int(value) for value in args.eval_depths],
        "eval_surface_mode": str(args.eval_surface_mode),
        "qwen_prompt_mode": str(args.qwen_prompt_mode),
        "out_jsonl": str(args.out_jsonl),
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
