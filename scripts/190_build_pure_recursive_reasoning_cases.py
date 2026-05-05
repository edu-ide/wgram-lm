#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _case(
    *,
    case_id: str,
    category: str,
    reasoning_family: str,
    expected_paradigm: str,
    requires_stochasticity: bool,
    parallel_depth_estimate: int,
    serial_trace_length_estimate: int,
    question: str,
    answer: str,
    choices: list[str],
    depth_targets: dict[str, str] | None = None,
    transition_state_codes: dict[str, int] | None = None,
    solver_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    depth_target_map = depth_targets or {"1": answer, "2": answer, "4": answer, "8": answer}
    return {
        "id": case_id,
        "raw_intelligence_axis": "pure_recursive_reasoning",
        "category": category,
        "task_family": category,
        "reasoning_family": reasoning_family,
        "expected_paradigm": expected_paradigm,
        "requires_stochasticity": bool(requires_stochasticity),
        "parallel_depth_estimate": int(parallel_depth_estimate),
        "serial_trace_length_estimate": int(serial_trace_length_estimate),
        "question": question,
        "prompt": prompt,
        "answer_aliases": [answer],
        "choices": _unique_choices([answer, *choices]),
        "depth_targets": depth_target_map,
        "transition_state_codes": transition_state_codes
        or {"1": 0, "2": 0, "4": 0, "8": 0},
        "solver_trace": solver_trace
        or _default_solver_trace(depth_target_map),
        "expected_unknown": False,
        "retrieval_allowed": False,
        "memoryos_allowed": False,
        "evidence": [],
    }


def _default_solver_trace(depth_targets: dict[str, str]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for depth in (1, 2, 4, 8):
        trace.append(
            {
                "depth": depth,
                "operation": f"state_depth_{depth}",
                "state_text": str(depth_targets[str(depth)]),
            }
        )
    return trace


def _solver_trace(
    *,
    depth_targets: dict[str, str],
    operations: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "depth": depth,
            "operation": operations[str(depth)],
            "state_text": str(depth_targets[str(depth)]),
        }
        for depth in (1, 2, 4, 8)
    ]


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


SEMANTIC_TRANSITION_STATE_CODES: dict[str, dict[str, int]] = {
    "arithmetic_chain": {"1": 10, "2": 11, "4": 12, "8": 12},
    "symbolic_binding": {"1": 20, "2": 21, "4": 21, "8": 21},
    "boolean_logic": {"1": 30, "2": 31, "4": 32, "8": 32},
    "list_transform": {"1": 40, "2": 41, "4": 41, "8": 41},
}


def parse_family_filter(value: str | list[str] | None) -> set[str]:
    families: set[str] = set()
    if value is None:
        return families
    values = value if isinstance(value, list) else [value]
    for raw in values:
        for item in str(raw).split(","):
            family = item.strip()
            if family:
                families.add(family)
    return families


def build_cases(*, cases_per_family: int = 18, start_index: int = 0) -> list[dict[str, Any]]:
    arithmetic_cases: list[dict[str, Any]] = []
    for i in range(cases_per_family):
        idx = int(start_index) + i
        a = 7 + idx
        b = 3 + (idx % 5)
        c = 2 + (idx % 4)
        answer = str((a + b) * c - b)
        sum_stage = str(a + b)
        multiply_stage = str((a + b) * c)
        depth_targets = {
            "1": sum_stage,
            "2": multiply_stage,
            "4": answer,
            "8": answer,
        }
        arithmetic_cases.append(
            _case(
                case_id=f"arith-chain-{idx:03d}",
                category="arithmetic_chain",
                reasoning_family="sequential_arithmetic",
                expected_paradigm="hybrid_or_cot",
                requires_stochasticity=False,
                parallel_depth_estimate=0,
                serial_trace_length_estimate=3,
                question=f"Compute (({a} + {b}) * {c}) - {b}.",
                answer=answer,
                choices=[
                    multiply_stage,
                    str((a + b) * c - b + 1),
                    str((a + b) * c - b - 1),
                ],
                depth_targets=depth_targets,
                transition_state_codes=SEMANTIC_TRANSITION_STATE_CODES[
                    "arithmetic_chain"
                ],
                solver_trace=_solver_trace(
                    depth_targets=depth_targets,
                    operations={
                        "1": "add_operands",
                        "2": "multiply_sum",
                        "4": "subtract_offset",
                        "8": "hold_final",
                    },
                ),
            )
        )

    letters = ["A", "B", "C", "D", "E", "F"]
    colors = ["red", "blue", "green", "amber", "violet", "silver"]
    symbolic_cases: list[dict[str, Any]] = []
    for i in range(cases_per_family):
        idx = int(start_index) + i
        src = letters[idx % len(letters)]
        mid = colors[(idx + 2) % len(colors)]
        dst = colors[(idx + 4) % len(colors)]
        depth_targets = {
            "1": mid,
            "2": dst,
            "4": dst,
            "8": dst,
        }
        symbolic_cases.append(
            _case(
                case_id=f"symbolic-binding-{idx:03d}",
                category="symbolic_binding",
                reasoning_family="state_propagation",
                expected_paradigm="latent_recurrent",
                requires_stochasticity=False,
                parallel_depth_estimate=2,
                serial_trace_length_estimate=2,
                question=(
                    f"If {src} maps to {mid}, {mid} maps to {dst}, and {dst} maps "
                    f"to {letters[(idx + 3) % len(letters)]}, what does {src} map to after two mappings?"
                ),
                answer=dst,
                choices=[mid, colors[(idx + 1) % len(colors)], colors[(idx + 5) % len(colors)]],
                depth_targets=depth_targets,
                transition_state_codes=SEMANTIC_TRANSITION_STATE_CODES[
                    "symbolic_binding"
                ],
                solver_trace=_solver_trace(
                    depth_targets=depth_targets,
                    operations={
                        "1": "first_mapping",
                        "2": "second_mapping",
                        "4": "hold_final",
                        "8": "hold_final",
                    },
                ),
            )
        )

    boolean_cases: list[dict[str, Any]] = []
    for i in range(cases_per_family):
        idx = int(start_index) + i
        p = idx % 2 == 0
        q = idx % 3 == 0
        r = idx % 4 == 0
        not_q = not q
        and_stage = p and not_q
        value = (p and not q) or r
        depth_targets = {
            "1": "TRUE" if not_q else "FALSE",
            "2": "TRUE" if and_stage else "FALSE",
            "4": "TRUE" if value else "FALSE",
            "8": "TRUE" if value else "FALSE",
        }
        boolean_cases.append(
            _case(
                case_id=f"boolean-logic-{idx:03d}",
                category="boolean_logic",
                reasoning_family="parallel_boolean",
                expected_paradigm="latent_parallel",
                requires_stochasticity=False,
                parallel_depth_estimate=2,
                serial_trace_length_estimate=2,
                question=(
                    f"Let P={str(p).upper()}, Q={str(q).upper()}, R={str(r).upper()}. "
                    "Evaluate (P AND NOT Q) OR R. Answer TRUE or FALSE."
                ),
                answer="TRUE" if value else "FALSE",
                choices=["FALSE" if value else "TRUE"],
                depth_targets=depth_targets,
                transition_state_codes=SEMANTIC_TRANSITION_STATE_CODES[
                    "boolean_logic"
                ],
                solver_trace=_solver_trace(
                    depth_targets=depth_targets,
                    operations={
                        "1": "not_q",
                        "2": "and_with_p",
                        "4": "or_with_r",
                        "8": "hold_final",
                    },
                ),
            )
        )

    list_cases: list[dict[str, Any]] = []
    for i in range(cases_per_family):
        idx = int(start_index) + i
        values = [idx + 1, idx + 4, idx + 2, idx + 7, idx + 3]
        filtered = [value for value in values if value % 2 == 0]
        transformed = [value * 2 for value in filtered]
        filtered_answer = ",".join(str(value) for value in filtered) if filtered else "EMPTY"
        answer = ",".join(str(value) for value in transformed) if transformed else "EMPTY"
        depth_targets = {
            "1": filtered_answer,
            "2": answer,
            "4": answer,
            "8": answer,
        }
        list_cases.append(
            _case(
                case_id=f"list-transform-{idx:03d}",
                category="list_transform",
                reasoning_family="sequential_list_transform",
                expected_paradigm="hybrid_or_cot",
                requires_stochasticity=False,
                parallel_depth_estimate=0,
                serial_trace_length_estimate=2,
                question=(
                    "From the list "
                    f"{values}, keep only even numbers, double each kept number, "
                    "and return comma-separated values with no spaces. If none, return EMPTY."
                ),
                answer=answer,
                choices=[
                    ",".join(str(value) for value in transformed[::-1]) if transformed else "0",
                    ",".join(str(value) for value in values if value % 2 == 0) or "EMPTY",
                    "EMPTY" if answer != "EMPTY" else str(values[0]),
                ],
                depth_targets=depth_targets,
                transition_state_codes=SEMANTIC_TRANSITION_STATE_CODES[
                    "list_transform"
                ],
                solver_trace=_solver_trace(
                    depth_targets=depth_targets,
                    operations={
                        "1": "filter_even",
                        "2": "double_filtered",
                        "4": "hold_final",
                        "8": "hold_final",
                    },
                ),
            )
        )
    cases: list[dict[str, Any]] = []
    families = [arithmetic_cases, symbolic_cases, boolean_cases, list_cases]
    for idx in range(cases_per_family):
        for family in families:
            cases.append(family[idx])
    return cases


def filter_cases_by_family(
    cases: list[dict[str, Any]],
    include_families: set[str] | None,
) -> list[dict[str, Any]]:
    include = include_families or set()
    if not include:
        return list(cases)
    filtered = [
        case
        for case in cases
        if str(case.get("task_family") or case.get("category") or "") in include
    ]
    if not filtered:
        raise ValueError(f"family filter matched no cases: {sorted(include)}")
    return filtered


def write_cases(
    path: str | Path,
    *,
    cases_per_family: int = 18,
    start_index: int = 0,
    include_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    cases = filter_cases_by_family(
        build_cases(cases_per_family=cases_per_family, start_index=start_index),
        include_families,
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
        description="Build no-retrieval pure recursive reasoning eval cases."
    )
    parser.add_argument(
        "--out",
        default="data/eval/pure_recursive_reasoning_heldout_72.jsonl",
    )
    parser.add_argument(
        "--cases-per-family",
        type=int,
        default=18,
        help="Cases per family. Four families are emitted, so default gives 72 cases.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Deterministic offset used to create non-overlapping train/eval case ids.",
    )
    parser.add_argument(
        "--include-family",
        action="append",
        default=None,
        help=(
            "Restrict emitted cases to one or more task families. Can be repeated "
            "or comma-separated, for example arithmetic_chain,list_transform."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = write_cases(
        args.out,
        cases_per_family=args.cases_per_family,
        start_index=args.start_index,
        include_families=parse_family_filter(args.include_family),
    )
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
