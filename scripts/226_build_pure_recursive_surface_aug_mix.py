#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_script(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    spec.loader.exec_module(module)
    return module


def _with_answer_fields(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    answer = str((out.get("answer_aliases") or [out.get("answer", "")])[0])
    out["answer"] = answer
    out["chosen"] = answer
    return out


def build_surface_aug_mix(
    *,
    cases_per_family: int = 32,
    original_start_index: int = 5000,
    surface_start_index: int = 9000,
) -> list[dict[str, Any]]:
    base_builder = _load_script("190_build_pure_recursive_reasoning_cases.py")
    surface_builder = _load_script("225_build_pure_recursive_ood_surface_cases.py")
    canonical_rows = [
        {
            **_with_answer_fields(case),
            "surface_distribution": "canonical_surface",
        }
        for case in base_builder.build_cases(
            cases_per_family=int(cases_per_family),
            start_index=int(original_start_index),
        )
    ]
    surface_rows = surface_builder.build_ood_surface_cases(
        cases_per_family=int(cases_per_family),
        start_index=int(surface_start_index),
    )
    return [*canonical_rows, *surface_rows]


def write_mix(
    path: str | Path,
    *,
    cases_per_family: int = 32,
    original_start_index: int = 5000,
    surface_start_index: int = 9000,
) -> list[dict[str, Any]]:
    rows = build_surface_aug_mix(
        cases_per_family=cases_per_family,
        original_start_index=original_start_index,
        surface_start_index=surface_start_index,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical+OOD-surface primitive reasoning training mix."
    )
    parser.add_argument(
        "--out",
        default="data/filtered/pure_recursive_primitive_transition_surface_aug_mix_train.jsonl",
    )
    parser.add_argument("--cases-per-family", type=int, default=32)
    parser.add_argument("--original-start-index", type=int, default=5000)
    parser.add_argument("--surface-start-index", type=int, default=9000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = write_mix(
        args.out,
        cases_per_family=args.cases_per_family,
        original_start_index=args.original_start_index,
        surface_start_index=args.surface_start_index,
    )
    print(f"wrote {len(rows)} surface-aug mix rows to {args.out}")


if __name__ == "__main__":
    main()
