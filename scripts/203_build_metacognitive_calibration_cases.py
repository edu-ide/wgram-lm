#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _unique_choices(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _case(
    *,
    case_id: str,
    category: str,
    question: str,
    answer_aliases: list[str],
    choices: list[str],
    expected_unknown: bool,
    uncertainty_type: str,
) -> dict[str, Any]:
    prompt = (
        "Answer with only the final answer. If the answer is not determined, "
        "contradictory, or random/OOD, answer UNKNOWN.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return {
        "id": case_id,
        "raw_intelligence_axis": "metacognitive_calibration",
        "category": category,
        "task_family": category,
        "reasoning_family": "metacognitive_uncertainty",
        "expected_paradigm": "metacognitive_calibration",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 0,
        "serial_trace_length_estimate": 1,
        "question": question,
        "prompt": prompt,
        "answer_aliases": answer_aliases,
        "choices": _unique_choices([*answer_aliases, *choices]),
        "expected_unknown": bool(expected_unknown),
        "uncertainty_type": uncertainty_type,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
    }


def build_cases(*, cases_per_family: int = 8, start_index: int = 0) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    colors = ["red", "blue", "green", "amber", "violet", "silver", "cyan", "white"]
    for i in range(int(cases_per_family)):
        idx = int(start_index) + i
        a = 12 + idx
        b = 4 + (idx % 7)
        answer = str(a + b)
        cases.append(
            _case(
                case_id=f"metacog-answerable-arith-{idx:03d}",
                category="answerable_arithmetic",
                question=f"Compute {a} + {b}.",
                answer_aliases=[answer],
                choices=[str(a + b + 1), str(a + b - 1), "UNKNOWN"],
                expected_unknown=False,
                uncertainty_type="answerable",
            )
        )

        p = idx % 2 == 0
        q = idx % 3 == 0
        value = p and not q
        cases.append(
            _case(
                case_id=f"metacog-answerable-bool-{idx:03d}",
                category="answerable_boolean",
                question=(
                    f"Let P={str(p).upper()} and Q={str(q).upper()}. "
                    "Evaluate P AND NOT Q. Answer TRUE or FALSE."
                ),
                answer_aliases=["TRUE" if value else "FALSE"],
                choices=["FALSE" if value else "TRUE", "UNKNOWN"],
                expected_unknown=False,
                uncertainty_type="answerable",
            )
        )

        cases.append(
            _case(
                case_id=f"metacog-unknown-missing-{idx:03d}",
                category="unknown_missing",
                question=(
                    f"The private registry item XR-{idx:03d} is not provided in this prompt. "
                    "What is its current access code?"
                ),
                answer_aliases=["UNKNOWN", "unknown"],
                choices=[f"CODE-{idx + 17}", f"KEY-{idx + 23}", "NONE"],
                expected_unknown=True,
                uncertainty_type="missing_information",
            )
        )

        first = colors[idx % len(colors)]
        second = colors[(idx + 3) % len(colors)]
        cases.append(
            _case(
                case_id=f"metacog-contradiction-{idx:03d}",
                category="contradiction",
                question=(
                    f"A note says the beacon color is {first}. The same note also says "
                    f"the beacon color is {second}. If the note contradicts itself, what is the beacon color?"
                ),
                answer_aliases=["UNKNOWN", "unknown"],
                choices=[first, second, colors[(idx + 5) % len(colors)]],
                expected_unknown=True,
                uncertainty_type="contradiction",
            )
        )

        randomish = f"ZX-{idx:03d}-@@-{(idx * 7919) % 9973}-Q?"
        cases.append(
            _case(
                case_id=f"metacog-ood-random-token-{idx:03d}",
                category="ood_random_token",
                question=(
                    f"Random token string: {randomish}. What stable semantic label does this string encode?"
                ),
                answer_aliases=["UNKNOWN", "unknown"],
                choices=[f"LABEL-{idx}", f"TOKEN-{idx}", "VALID"],
                expected_unknown=True,
                uncertainty_type="ood_random_token",
            )
        )
    return cases


def write_cases(
    path: str | Path,
    *,
    cases_per_family: int = 8,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    cases = build_cases(cases_per_family=cases_per_family, start_index=start_index)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    return cases


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build no-retrieval metacognitive calibration eval cases."
    )
    parser.add_argument("--out", default="data/eval/metacognitive_calibration_heldout_40.jsonl")
    parser.add_argument(
        "--cases-per-family",
        type=int,
        default=8,
        help="Cases per family. Five families are emitted, so default gives 40 cases.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = write_cases(
        args.out,
        cases_per_family=args.cases_per_family,
        start_index=args.start_index,
    )
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
