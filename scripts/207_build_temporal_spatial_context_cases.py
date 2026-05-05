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


def _context_token(values: list[float]) -> list[float]:
    if len(values) != 8:
        raise ValueError("temporal-spatial context tokens must be 8-dimensional")
    return [float(value) for value in values]


def _case(
    *,
    case_id: str,
    category: str,
    reasoning_family: str,
    question: str,
    answer: str,
    choices: list[str],
    context_tokens: list[list[float]],
    context_schema: list[str],
) -> dict[str, Any]:
    prompt = (
        "Answer with only the final answer. Do not write reasoning.\n"
        "Use the provided temporal/spatial context as part of the question.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return {
        "id": case_id,
        "raw_intelligence_axis": "temporal_spatial_context",
        "category": category,
        "task_family": category,
        "reasoning_family": reasoning_family,
        "expected_paradigm": "context_conditioned_latent_reasoning",
        "requires_stochasticity": False,
        "parallel_depth_estimate": 1,
        "serial_trace_length_estimate": 2,
        "question": question,
        "prompt": prompt,
        "answer": answer,
        "chosen": answer,
        "answer_aliases": [answer],
        "choices": _unique_choices([answer, *choices]),
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
        "temporal_spatial_context": context_tokens,
        "temporal_spatial_context_schema": context_schema,
    }


def _temporal_case(idx: int) -> dict[str, Any]:
    colors = ["red", "blue", "green", "amber", "violet", "silver"]
    stale_color = colors[idx % len(colors)]
    fresh_color = colors[(idx + 2) % len(colors)]
    current_year = 2026
    stale_year = 2021 + (idx % 3)
    fresh_year = 2025 + (idx % 2)
    stale_valid_until = 2024
    fresh_valid_until = 2027
    question = (
        f"Temporal context: current_year={current_year}. "
        f"Observation A says beacon color={stale_color}, observed_year={stale_year}, "
        f"valid_until={stale_valid_until}. "
        f"Observation B says beacon color={fresh_color}, observed_year={fresh_year}, "
        f"valid_until={fresh_valid_until}. "
        "As of current_year, what is the beacon color?"
    )
    temporal_token = _context_token(
        [
            1.0,
            current_year / 3000.0,
            stale_year / 3000.0,
            stale_valid_until / 3000.0,
            1.0,
            0.0,
            0.0,
            float(idx % 7) / 7.0,
        ]
    )
    spatial_placeholder = _context_token([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return _case(
        case_id=f"temporal-freshness-{idx:03d}",
        category="temporal_freshness",
        reasoning_family="temporal_validity",
        question=question,
        answer=fresh_color,
        choices=[stale_color, colors[(idx + 4) % len(colors)], "UNKNOWN"],
        context_tokens=[temporal_token, spatial_placeholder],
        context_schema=[
            "temporal: kind,current_year,observed_year,valid_until,stale_flag,pad,pad,case_hash",
            "spatial: unused placeholder",
        ],
    )


def _spatial_case(idx: int) -> dict[str, Any]:
    objects = ["red key", "blue key", "green key", "amber key", "silver key"]
    left = objects[idx % len(objects)]
    right = objects[(idx + 2) % len(objects)]
    distractor = objects[(idx + 4) % len(objects)]
    left_x = 0.15 + 0.01 * (idx % 5)
    right_x = 0.78 - 0.01 * (idx % 5)
    y = 0.50
    question = (
        "Spatial context: normalized 2D positions are "
        f"{left} at x={left_x:.2f}, y={y:.2f}; "
        f"{right} at x={right_x:.2f}, y={y:.2f}; "
        f"{distractor} at x=0.50, y=0.80. "
        f"Which object is left of the {right}?"
    )
    temporal_placeholder = _context_token([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    spatial_token = _context_token(
        [
            2.0,
            left_x,
            y,
            right_x,
            y,
            1.0,
            0.0,
            float(idx % 7) / 7.0,
        ]
    )
    return _case(
        case_id=f"spatial-relation-{idx:03d}",
        category="spatial_relation",
        reasoning_family="spatial_comparison",
        question=question,
        answer=left,
        choices=[right, distractor, "UNKNOWN"],
        context_tokens=[temporal_placeholder, spatial_token],
        context_schema=[
            "temporal: unused placeholder",
            "spatial: kind,left_x,left_y,right_x,right_y,left_relation,pad,case_hash",
        ],
    )


def build_cases(*, cases_per_family: int = 12, start_index: int = 0) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for i in range(int(cases_per_family)):
        idx = int(start_index) + i
        cases.append(_temporal_case(idx))
        cases.append(_spatial_case(idx))
    return cases


def write_cases(
    path: str | Path,
    *,
    cases_per_family: int = 12,
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
        description="Build temporal/spatial context conditioning held-out eval cases."
    )
    parser.add_argument(
        "--out",
        default="data/eval/temporal_spatial_context_heldout_24.jsonl",
    )
    parser.add_argument(
        "--cases-per-family",
        type=int,
        default=12,
        help="Cases per family. Two families are emitted, so default gives 24 cases.",
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
