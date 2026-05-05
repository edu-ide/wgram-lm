#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


def _load_case_builder_module():
    path = Path(__file__).with_name("190_build_pure_recursive_reasoning_cases.py")
    spec = importlib.util.spec_from_file_location("pure_recursive_reasoning_cases", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    spec.loader.exec_module(module)
    return module


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _rewrite_arithmetic(question: str, variant_index: int) -> str:
    match = re.search(r"(\(\([-]?\d+\s*\+\s*[-]?\d+\)\s*\*\s*[-]?\d+\)\s*-\s*[-]?\d+)", question)
    if not match:
        raise ValueError(f"unsupported arithmetic question: {question!r}")
    expr = match.group(1)
    variants = [
        f"Solve this arithmetic expression exactly: {expr}.",
        f"What integer do you get after evaluating {expr}?",
        f"Evaluate the expression {expr} and report only the result.",
    ]
    return variants[int(variant_index) % len(variants)]


def _rewrite_list(question: str, variant_index: int) -> str:
    match = re.search(r"(\[[^\]]*\])", question)
    if not match:
        raise ValueError(f"unsupported list question: {question!r}")
    values = match.group(1)
    variants = [
        f"Use the numbers {values}. Keep the even entries, then double the kept entries. Return comma-separated values with no spaces; if there are none, return EMPTY.",
        f"Given {values}, first remove odd numbers and then multiply each remaining number by 2. Output comma-separated values with no spaces, or EMPTY.",
        f"Transform {values}: retain only even numbers, double them, and print the resulting comma-separated list. If the result is empty, print EMPTY.",
    ]
    return variants[int(variant_index) % len(variants)]


def _rewrite_symbolic(question: str, variant_index: int) -> str:
    mappings = re.findall(r"\b([A-Za-z]+)\s+maps\s+to\s+([A-Za-z]+)\b", question)
    if len(mappings) < 2:
        raise ValueError(f"unsupported symbolic question: {question!r}")
    start = mappings[0][0]
    mapping_text = "; ".join(f"{source} maps to {target}" for source, target in mappings)
    variants = [
        f"Mapping facts: {mapping_text}. Starting at {start}, apply the mapping two times. What symbol is reached?",
        f"Use these links: {mapping_text}. If you follow two links from {start}, where do you land?",
        f"Given {mapping_text}, determine the result after mapping {start} twice.",
    ]
    return variants[int(variant_index) % len(variants)]


def _rewrite_boolean(question: str, variant_index: int) -> str:
    bindings = re.findall(r"\b([PQR])=(TRUE|FALSE)\b", question)
    if len(bindings) != 3:
        raise ValueError(f"unsupported boolean question: {question!r}")
    binding_text = ", ".join(f"{name}={value}" for name, value in bindings)
    variants = [
        f"Given {binding_text}, compute the truth value of (P AND NOT Q) OR R. Answer TRUE or FALSE.",
        f"With {binding_text}, evaluate this boolean formula: (P AND NOT Q) OR R. Return TRUE or FALSE.",
        f"For the bindings {binding_text}, what is (P AND NOT Q) OR R?",
    ]
    return variants[int(variant_index) % len(variants)]


def rewrite_case_surface(case: dict[str, Any], *, variant_index: int) -> dict[str, Any]:
    family = str(case.get("task_family") or case.get("category") or "")
    question = str(case.get("question") or "")
    if family == "arithmetic_chain":
        rewritten_question = _rewrite_arithmetic(question, variant_index)
    elif family == "list_transform":
        rewritten_question = _rewrite_list(question, variant_index)
    elif family == "symbolic_binding":
        rewritten_question = _rewrite_symbolic(question, variant_index)
    elif family == "boolean_logic":
        rewritten_question = _rewrite_boolean(question, variant_index)
    else:
        raise ValueError(f"unsupported family: {family!r}")
    out = dict(case)
    out["id"] = f"{case.get('id', 'case')}-oodsurf{int(variant_index) % 3}"
    out["question"] = rewritten_question
    out["prompt"] = _prompt_for_question(rewritten_question)
    answer = str((out.get("answer_aliases") or [out.get("answer", "")])[0])
    out["answer"] = answer
    out["chosen"] = answer
    out["surface_distribution"] = "ood_surface_paraphrase_v1"
    out["surface_variant_index"] = int(variant_index) % 3
    return out


def build_ood_surface_cases(
    *,
    cases_per_family: int = 32,
    start_index: int = 8000,
) -> list[dict[str, Any]]:
    builder = _load_case_builder_module()
    base_cases = builder.build_cases(
        cases_per_family=int(cases_per_family),
        start_index=int(start_index),
    )
    return [
        rewrite_case_surface(case, variant_index=index)
        for index, case in enumerate(base_cases)
    ]


def write_cases(
    path: str | Path,
    *,
    cases_per_family: int = 32,
    start_index: int = 8000,
) -> list[dict[str, Any]]:
    cases = build_ood_surface_cases(
        cases_per_family=cases_per_family,
        start_index=start_index,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    return cases


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build OOD surface-form variants for pure recursive primitive reasoning."
    )
    parser.add_argument(
        "--out",
        default="data/eval/pure_recursive_primitive_transition_ood_surface_heldout8000_preferences.jsonl",
    )
    parser.add_argument("--cases-per-family", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=8000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = write_cases(
        args.out,
        cases_per_family=args.cases_per_family,
        start_index=args.start_index,
    )
    print(f"wrote {len(cases)} OOD surface cases to {args.out}")


if __name__ == "__main__":
    main()
