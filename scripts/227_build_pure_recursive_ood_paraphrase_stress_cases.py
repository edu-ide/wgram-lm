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


def _extract_arithmetic_expr(question: str) -> str:
    match = re.search(
        r"(\(\([-]?\d+\s*\+\s*[-]?\d+\)\s*\*\s*[-]?\d+\)\s*-\s*[-]?\d+)",
        question,
    )
    if not match:
        raise ValueError(f"unsupported arithmetic question: {question!r}")
    return match.group(1)


def _extract_list_values(question: str) -> str:
    match = re.search(r"(\[[^\]]*\])", question)
    if not match:
        raise ValueError(f"unsupported list question: {question!r}")
    return match.group(1)


def _extract_symbolic_mapping_text(question: str) -> tuple[str, str]:
    mappings = re.findall(r"\b([A-Za-z]+)\s+maps\s+to\s+([A-Za-z]+)\b", question)
    if len(mappings) < 2:
        raise ValueError(f"unsupported symbolic question: {question!r}")
    start = mappings[0][0]
    mapping_text = "; ".join(f"{source} maps to {target}" for source, target in mappings)
    return start, mapping_text


def _extract_boolean_binding_text(question: str) -> str:
    bindings = re.findall(r"\b([PQR])=(TRUE|FALSE)\b", question)
    if len(bindings) != 3:
        raise ValueError(f"unsupported boolean question: {question!r}")
    return ", ".join(f"{name}={value}" for name, value in bindings)


def _rewrite_arithmetic(question: str, variant_index: int) -> str:
    expr = _extract_arithmetic_expr(question)
    variants = [
        f"Solve this arithmetic expression exactly: {expr}.",
        f"What integer do you get after evaluating {expr}?",
        f"Evaluate the expression {expr} and report only the result.",
        f"Find the final value of {expr}; return the integer only.",
        f"Compute the value produced by {expr}.",
        f"Take this expression, {expr}, and give just its exact integer result.",
        f"Using ordinary arithmetic precedence, evaluate {expr}.",
        f"Return the final number after calculating {expr}.",
    ]
    return variants[int(variant_index) % len(variants)]


def _rewrite_list(question: str, variant_index: int) -> str:
    values = _extract_list_values(question)
    variants = [
        f"Use the numbers {values}. Keep the even entries, then double the kept entries. Return comma-separated values with no spaces; if there are none, return EMPTY.",
        f"Given {values}, first remove odd numbers and then multiply each remaining number by 2. Output comma-separated values with no spaces, or EMPTY.",
        f"Transform {values}: retain only even numbers, double them, and print the resulting comma-separated list. If the result is empty, print EMPTY.",
        f"For {values}, filter to even values and double each survivor. Return the comma-separated result, or EMPTY.",
        f"Process this list {values}: even-only, then times two. Use comma-separated output with no spaces; use EMPTY if blank.",
        f"From {values}, discard odd entries, double every remaining entry, and return only the resulting CSV.",
        f"Keep evens in {values}; double those values; answer with the resulting comma-separated sequence or EMPTY.",
        f"Apply the even-filter then double transform to {values}. Print only CSV output, or EMPTY.",
    ]
    return variants[int(variant_index) % len(variants)]


def _rewrite_symbolic(question: str, variant_index: int) -> str:
    start, mapping_text = _extract_symbolic_mapping_text(question)
    variants = [
        f"Mapping facts: {mapping_text}. Starting at {start}, apply the mapping two times. What symbol is reached?",
        f"Use these links: {mapping_text}. If you follow two links from {start}, where do you land?",
        f"Given {mapping_text}, determine the result after mapping {start} twice.",
        f"Follow the chain for two hops from {start}; facts: {mapping_text}. Return the reached symbol.",
        f"With mapping rules {mapping_text}, start on {start} and move through exactly two maps to find the answer.",
        f"Two-step lookup task. Rules: {mapping_text}. Initial symbol: {start}. What is the final symbol?",
        f"Starting symbol {start}. Mapping table: {mapping_text}. Apply two mappings and output the destination.",
        f"Read the map {mapping_text}; after two applications beginning with {start}, what value appears?",
    ]
    return variants[int(variant_index) % len(variants)]


def _rewrite_boolean(question: str, variant_index: int) -> str:
    binding_text = _extract_boolean_binding_text(question)
    variants = [
        f"Given {binding_text}, compute the truth value of (P AND NOT Q) OR R. Answer TRUE or FALSE.",
        f"With {binding_text}, evaluate this boolean formula: (P AND NOT Q) OR R. Return TRUE or FALSE.",
        f"For the bindings {binding_text}, what is (P AND NOT Q) OR R?",
        f"Truth-table task: {binding_text}. Evaluate (P AND NOT Q) OR R and output TRUE or FALSE.",
        f"Use {binding_text}. First take NOT Q, combine with P using AND, then OR with R. What final truth value results?",
        f"Boolean expression check. Bindings: {binding_text}. Formula: (P AND NOT Q) OR R. Return only TRUE or FALSE.",
        f"Under {binding_text}, determine whether (P AND NOT Q) OR R is TRUE or FALSE.",
        f"Evaluate the formula (P AND NOT Q) OR R for {binding_text}; answer with the final boolean token.",
    ]
    return variants[int(variant_index) % len(variants)]


def rewrite_case_surface_stress(case: dict[str, Any], *, variant_index: int) -> dict[str, Any]:
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
    variant = int(variant_index) % 8
    out["id"] = f"{case.get('id', 'case')}-oodstress{variant}"
    out["question"] = rewritten_question
    out["prompt"] = _prompt_for_question(rewritten_question)
    answer = str((out.get("answer_aliases") or [out.get("answer", "")])[0])
    out["answer"] = answer
    out["chosen"] = answer
    out["surface_distribution"] = "ood_surface_paraphrase_stress_v1"
    out["surface_variant_index"] = variant
    return out


def build_ood_paraphrase_stress_cases(
    *,
    cases_per_family: int = 8,
    start_index: int = 10000,
    variants_per_case: int = 8,
) -> list[dict[str, Any]]:
    if int(variants_per_case) <= 0:
        raise ValueError("variants_per_case must be positive")
    builder = _load_case_builder_module()
    base_cases = builder.build_cases(
        cases_per_family=int(cases_per_family),
        start_index=int(start_index),
    )
    cases: list[dict[str, Any]] = []
    for case in base_cases:
        for variant_index in range(int(variants_per_case)):
            cases.append(
                rewrite_case_surface_stress(case, variant_index=variant_index)
            )
    return cases


def write_cases(
    path: str | Path,
    *,
    cases_per_family: int = 8,
    start_index: int = 10000,
    variants_per_case: int = 8,
) -> list[dict[str, Any]]:
    cases = build_ood_paraphrase_stress_cases(
        cases_per_family=cases_per_family,
        start_index=start_index,
        variants_per_case=variants_per_case,
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
        description="Build larger OOD paraphrase stress cases for primitive reasoning."
    )
    parser.add_argument(
        "--out",
        default="data/eval/pure_recursive_primitive_transition_ood_paraphrase_stress_heldout10000_preferences.jsonl",
    )
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=10000)
    parser.add_argument("--variants-per-case", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = write_cases(
        args.out,
        cases_per_family=args.cases_per_family,
        start_index=args.start_index,
        variants_per_case=args.variants_per_case,
    )
    print(f"wrote {len(cases)} OOD paraphrase stress cases to {args.out}")


if __name__ == "__main__":
    main()
