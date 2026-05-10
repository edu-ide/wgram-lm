#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _question_for_variant(
    *,
    base: int,
    coeff: int,
    residual: int,
    offset: int,
    variant: int,
) -> str:
    if int(variant) % 8 == 0:
        return (
            f"Compute ({coeff} * {base}) + {residual}, then subtract {offset}. "
            "What integer remains?"
        )
    if int(variant) % 8 == 1:
        return (
            f"Start with {coeff} times {base} plus {residual}. Reduce that "
            f"number by {offset} and return the result."
        )
    if int(variant) % 8 == 2:
        return (
            f"Let x = {coeff} * {base} + {residual}. Return x - {offset}."
        )
    if int(variant) % 8 == 3:
        return (
            f"Take {coeff} copies of {base}, add {residual}, subtract {offset}, "
            "and give only the final integer."
        )
    if int(variant) % 8 == 4:
        return (
            f"Evaluate this arithmetic chain: multiply {base} by {coeff}, add "
            f"{residual}, then take away {offset}."
        )
    if int(variant) % 8 == 5:
        return (
            f"({base} multiplied by {coeff}) plus {residual}, minus {offset}: "
            "what is the integer?"
        )
    if int(variant) % 8 == 6:
        return (
            f"Begin at {coeff} * {base} + {residual}; apply a subtract-{offset} "
            "step; output the remaining value."
        )
    return (
        f"Find the result after this scalar transition: {coeff}*{base} + "
        f"{residual} -> subtract {offset}."
    )


def make_case(*, index: int, base_start: int, variant: int) -> dict[str, Any]:
    base = int(base_start) + int(index)
    coeff = (4, 6, 8, 10)[int(index) % 4]
    residual = 12 + ((int(index) * 7 + int(variant) * 5) % 34)
    offset = 1 + ((int(index) * 3 + int(variant)) % 9)
    preterminal = coeff * base + residual
    answer = preterminal - offset
    final_residual = residual - offset
    if final_residual < 0:
        raise ValueError("case generator produced negative final residual")
    question = _question_for_variant(
        base=base,
        coeff=coeff,
        residual=residual,
        offset=offset,
        variant=variant,
    )
    answer_text = str(answer)
    preterminal_text = str(preterminal)
    return {
        "id": f"scalar-affine-{base_start}-{index:06d}-v{variant}",
        "raw_intelligence_axis": "pure_recursive_reasoning",
        "category": "scalar_affine_arithmetic",
        "task_family": "scalar_affine_arithmetic",
        "reasoning_family": "sequential_scalar_arithmetic_transition",
        "expected_paradigm": "hybrid_or_cot",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 2,
        "question": question,
        "prompt": _prompt_for_question(question),
        "answer": answer_text,
        "chosen": answer_text,
        "answer_aliases": [answer_text],
        "choices": [
            answer_text,
            preterminal_text,
            str(coeff * base),
            str(answer + offset + 1),
        ],
        "depth_targets": {
            "1": preterminal_text,
            "2": answer_text,
            "3": answer_text,
            "4": answer_text,
            "5": answer_text,
            "6": answer_text,
            "7": answer_text,
            "8": answer_text,
        },
        "transition_state_codes": {
            "1": 2,
            "2": 3,
            "3": 4,
            "4": 4,
            "5": 4,
            "6": 4,
            "7": 4,
            "8": 4,
        },
        "solver_trace": [
            {"depth": 1, "operation": "affine_preterminal", "state_text": preterminal_text},
            {"depth": 2, "operation": "subtract_offset", "state_text": answer_text},
        ],
        "transition_finality_targets": {
            "1": 0,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
            "7": 1,
            "8": 1,
        },
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "base_value": base,
        "scalar_coeff": coeff,
        "scalar_initial_residual": residual,
        "subtract_offset": offset,
        "scalar_final_residual": final_residual,
        "surface_distribution": "scalar_affine_surface_v1",
        "surface_variant_index": int(variant),
        "latent_action_codebook_applied": True,
        "latent_action_codebook_version": "scalar_affine_v1",
        "latent_action_codebook_size": 5,
    }


def build_rows(
    *,
    cases: int,
    base_start: int,
    variants: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(cases)):
        for variant in variants:
            rows.append(make_case(index=index, base_start=base_start, variant=variant))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_variants(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError("variant list cannot be empty")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build scalar-only arithmetic codec train/eval gates."
    )
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-cases", type=int, default=512)
    parser.add_argument("--eval-cases", type=int, default=64)
    parser.add_argument("--train-base-start", type=int, default=40000)
    parser.add_argument("--eval-base-start", type=int, default=50000)
    parser.add_argument("--train-variants", default="0,1,2,3,4,5")
    parser.add_argument("--eval-variants", default="6,7")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_rows = build_rows(
        cases=int(args.train_cases),
        base_start=int(args.train_base_start),
        variants=parse_variants(args.train_variants),
    )
    eval_rows = build_rows(
        cases=int(args.eval_cases),
        base_start=int(args.eval_base_start),
        variants=parse_variants(args.eval_variants),
    )
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    print(
        json.dumps(
            {
                "train_out": str(args.train_out),
                "eval_out": str(args.eval_out),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_variants": list(parse_variants(args.train_variants)),
                "eval_variants": list(parse_variants(args.eval_variants)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
