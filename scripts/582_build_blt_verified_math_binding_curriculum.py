#!/usr/bin/env python3
"""Build a verified micro-math operand-binding curriculum for BLT runs.

The previous answer-only continuation learned the surface phrase
``Final answer:`` but still failed to bind prompt numbers to the operation.
This builder creates short, standardized responses that force the model to copy
the operands, name the operation, write the equation, and then emit the final
answer.  It is still synthetic and narrow, but it tests the missing causal
route before spending more compute on broad data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def binding_response(instruction: str) -> str | None:
    prompt = str(instruction).strip()

    match = re.search(r"What is (-?\d+) \+ (-?\d+)\?", prompt)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        answer = a + b
        return f"Operands: {a}, {b}. Operation: add. Equation: {a} + {b} = {answer}. Final answer: {answer}"

    match = re.search(r"What is (-?\d+) - (-?\d+)\?", prompt)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        answer = a - b
        return f"Operands: {a}, {b}. Operation: subtract. Equation: {a} - {b} = {answer}. Final answer: {answer}"

    match = re.search(r"What is (-?\d+) times (-?\d+)\?", prompt)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        answer = a * b
        return f"Operands: {a}, {b}. Operation: multiply. Equation: {a} * {b} = {answer}. Final answer: {answer}"

    match = re.search(r"What is (-?\d+)/(-?\d+) \+ (-?\d+)/(-?\d+)\?", prompt)
    if match:
        n1, d1, n2, d2 = (int(match.group(i)) for i in range(1, 5))
        answer = Fraction(n1, d1) + Fraction(n2, d2)
        rendered = format_fraction(answer)
        return (
            f"Operands: {n1}/{d1}, {n2}/{d2}. Operation: fraction_add. "
            f"Equation: {n1}/{d1} + {n2}/{d2} = {rendered}. Final answer: {rendered}"
        )

    match = re.search(r"Solve for x:\s*(-?\d+)x\s*([+-])\s*(\d+)\s*=\s*(-?\d+)", prompt)
    if match:
        coeff = int(match.group(1))
        sign = str(match.group(2))
        magnitude = int(match.group(3))
        rhs = int(match.group(4))
        bias = magnitude if sign == "+" else -magnitude
        if coeff == 0:
            return None
        answer = format_fraction(Fraction(rhs - bias, coeff))
        return (
            f"Operands: {coeff}, {bias}, {rhs}. Operation: linear_solve. "
            f"Equation: ({rhs} - {bias}) / {coeff} = {answer}. Final answer: {answer}"
        )

    match = re.search(r"least common multiple of (\d+) and (\d+)", prompt, re.IGNORECASE)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        divisor = math.gcd(a, b)
        answer = abs(a * b) // divisor
        return (
            f"Operands: {a}, {b}. Operation: lcm. "
            f"Equation: {a} * {b} / {divisor} = {answer}. Final answer: {answer}"
        )

    match = re.search(r"greatest common divisor of (\d+) and (\d+)", prompt, re.IGNORECASE)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        answer = math.gcd(a, b)
        return f"Operands: {a}, {b}. Operation: gcd. Equation: gcd({a}, {b}) = {answer}. Final answer: {answer}"

    match = re.search(r"Compute binom\((\d+),\s*(\d+)\)", prompt)
    if match:
        n, k = int(match.group(1)), int(match.group(2))
        answer = math.comb(n, k)
        return f"Operands: {n}, {k}. Operation: binom. Equation: binom({n}, {k}) = {answer}. Final answer: {answer}"

    match = re.search(
        r"A box has (\d+) bags with (\d+) marbles each, plus (\d+) extra marbles",
        prompt,
        re.IGNORECASE,
    )
    if match:
        bags, per_bag, extra = (int(match.group(i)) for i in range(1, 4))
        answer = bags * per_bag + extra
        return (
            f"Operands: {bags}, {per_bag}, {extra}. Operation: multiply_add. "
            f"Equation: {bags} * {per_bag} + {extra} = {answer}. Final answer: {answer}"
        )

    return None


def convert_file(src: Path, dst: Path, *, condition: str) -> dict[str, Any]:
    stats = {"read": 0, "written": 0, "skipped": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            stats["read"] += 1
            row = json.loads(line)
            instruction = str(row.get("instruction") or "").strip()
            response = binding_response(instruction)
            if not instruction or response is None:
                stats["skipped"] += 1
                continue
            out.write(
                json.dumps(
                    {
                        "condition": str(condition),
                        "instruction": instruction,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["written"] += 1
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition", default="direct", choices=("direct", "cot"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    report: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "condition": str(args.condition),
        "splits": {},
    }
    for split in ("train", "eval"):
        src = input_root / split / "data" / f"verified_math_micro_{split}.jsonl"
        dst = output_root / split / "data" / f"verified_math_binding_{split}.jsonl"
        report["splits"][split] = {
            "source": str(src),
            "target": str(dst),
            **convert_file(src, dst, condition=str(args.condition)),
        }
    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
