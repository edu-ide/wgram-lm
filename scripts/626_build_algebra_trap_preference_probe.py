#!/usr/bin/env python3
"""Build generated algebra-trap preference rows for Stage117.

The rows mimic GDsuite's repetitive algebra trap without copying heldout smoke
rows: several demonstrations repeat one tempting wrong answer, then the final
equation requires solving for ``a``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


FORMATS = ("original", "v2fmt", "numbered", "instruction")
HELDOUT_FINALS = {
    ("original", 16, 83),
    ("v2fmt", 16, 83),
    ("numbered", 77, 13),
    ("instruction", 77, 13),
}


def make_equation(solution: int, rng: random.Random) -> tuple[int, int, str]:
    rhs = rng.randint(-99, 99)
    left = rhs + int(solution)
    form = rng.choice(("rhs_plus_a", "a_plus_rhs"))
    return left, rhs, form


def equation_text(left: int, rhs: int, form: str) -> str:
    if form == "a_plus_rhs":
        return f"{left} = a + {rhs}"
    return f"{left} = {rhs} + a"


def render_prompt(
    *,
    fmt: str,
    demos: list[tuple[int, int, str]],
    final: tuple[int, int, str],
    wrong: int,
) -> str:
    if fmt == "original":
        lines = ["Please answer the following simple algebra questions."]
        for left, rhs, form in demos:
            lines.append(
                f"Q: Suppose {equation_text(left, rhs, form)}. "
                f"What is the value of a? A: {wrong}"
            )
        left, rhs, form = final
        lines.append(
            f"Q: Suppose {equation_text(left, rhs, form)}. "
            "What is the value of a? A:"
        )
        return "\n".join(lines)

    if fmt == "v2fmt":
        blocks = []
        for left, rhs, form in demos:
            blocks.append(
                f"Q: Suppose {equation_text(left, rhs, form)}. "
                f"What is the value of a?\nA: {wrong}"
            )
        left, rhs, form = final
        blocks.append(
            f"Q: Suppose {equation_text(left, rhs, form)}. "
            "What is the value of a?\nA:"
        )
        return "\n\n".join(blocks)

    if fmt == "numbered":
        lines = []
        for index, (left, rhs, form) in enumerate(demos, start=1):
            lines.append(f"{index}. {equation_text(left, rhs, form)}. a = {wrong}")
        left, rhs, form = final
        lines.append(f"{len(demos) + 1}. {equation_text(left, rhs, form)}. a =")
        return "\n".join(lines)

    if fmt == "instruction":
        blocks = ["Solve for a in each equation."]
        for left, rhs, form in demos:
            blocks.append(f"{equation_text(left, rhs, form)}\na = {wrong}")
        left, rhs, form = final
        blocks.append(f"{equation_text(left, rhs, form)}\na =")
        return "\n\n".join(blocks)

    raise ValueError(f"unsupported format: {fmt}")


def parse_fixed_wrong(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"fixed wrong value must look like format=number: {value!r}")
        key, raw_number = str(value).split("=", 1)
        key = key.strip()
        if key not in FORMATS:
            raise ValueError(f"unknown format in fixed wrong value: {key!r}")
        out[key] = int(raw_number)
    return out


def build_rows(
    *,
    rows_per_format: int,
    seed: int,
    formats: tuple[str, ...] = FORMATS,
    fixed_wrong_by_format: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    fixed_wrong_by_format = dict(fixed_wrong_by_format or {})
    rows: list[dict[str, Any]] = []
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unsupported format: {fmt}")
        produced = 0
        attempts = 0
        while produced < int(rows_per_format):
            attempts += 1
            if attempts > int(rows_per_format) * 100:
                raise RuntimeError(f"could not generate enough rows for {fmt}")
            wrong = int(fixed_wrong_by_format.get(fmt, rng.randint(-95, 95)))
            correct = rng.randint(-95, 95)
            if correct == wrong:
                continue
            if (fmt, correct, wrong) in HELDOUT_FINALS:
                continue
            demos = [make_equation(wrong, rng) for _ in range(4)]
            final = make_equation(correct, rng)
            row_id = f"stage117_algebra_trap_{fmt}_{produced:05d}"
            rows.append(
                {
                    "id": row_id,
                    "source": "generated_stage117_algebra_trap",
                    "family": "repetitive_answer",
                    "task": f"repetitive_answer/algebra/{fmt}",
                    "prompt": render_prompt(
                        fmt=fmt,
                        demos=demos,
                        final=final,
                        wrong=wrong,
                    ),
                    "intelligence_answer": f" {correct}",
                    "parrot_answer": f" {wrong}",
                    "plain_language_axis": (
                        "Solve the final equation for a instead of copying the "
                        "answer repeated in the demonstrations."
                    ),
                }
            )
            produced += 1
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/eval/stage117_algebra_trap_preference_train.jsonl")
    parser.add_argument("--rows-per-format", type=int, default=512)
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--formats", nargs="*", default=list(FORMATS))
    parser.add_argument(
        "--fixed-wrong",
        nargs="*",
        default=[],
        help="Optional format=number entries, e.g. original=83 v2fmt=83.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    fixed_wrong = parse_fixed_wrong([str(value) for value in args.fixed_wrong])
    formats = tuple(str(value) for value in args.formats)
    rows = build_rows(
        rows_per_format=int(args.rows_per_format),
        seed=int(args.seed),
        formats=formats,
        fixed_wrong_by_format=fixed_wrong,
    )
    write_jsonl(Path(args.out), rows)
    report = {
        "out": str(args.out),
        "rows": len(rows),
        "rows_per_format": int(args.rows_per_format),
        "formats": list(formats),
        "fixed_wrong_by_format": fixed_wrong,
        "plain_language_read": (
            "Generated algebra traps teach the same LM answer path to solve the "
            "final equation instead of copying the repeated demonstration answer."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
