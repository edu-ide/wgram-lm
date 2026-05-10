#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HOLD_CODE_BY_CODEBOOK_VERSION = {
    "role_v1": 3,
    "terminal_v2": 4,
    "dynamic_halt_v3": 4,
}

TERMINAL_CODES_BY_CODEBOOK_VERSION = {
    "role_v1": {2, 3},
    "terminal_v2": {1, 3, 4},
    "dynamic_halt_v3": {4},
}


def _final_answer(row: dict[str, Any]) -> str:
    aliases = row.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    return str(row.get("answer") or row.get("chosen") or "")


def _max_depth(row: dict[str, Any], default: int) -> int:
    depths: list[int] = []
    for key in ("transition_state_codes", "transition_finality_targets", "depth_targets"):
        values = row.get(key)
        if isinstance(values, dict):
            depths.extend(int(depth) for depth in values)
    for trace_key in ("solver_trace", "latent_action_trace"):
        trace = row.get(trace_key)
        if isinstance(trace, list):
            for step in trace:
                if isinstance(step, dict) and step.get("depth") is not None:
                    depths.append(int(step["depth"]))
    return max(depths or [int(default)])


def _ordered_state_texts(row: dict[str, Any]) -> list[str]:
    solver_trace = row.get("solver_trace")
    if isinstance(solver_trace, list) and solver_trace:
        texts = [
            str(step.get("state_text") or "")
            for step in solver_trace
            if isinstance(step, dict)
        ]
        if texts:
            return texts
    depth_targets = row.get("depth_targets")
    if isinstance(depth_targets, dict) and depth_targets:
        return [
            str(depth_targets[str(depth)])
            for depth in sorted(int(value) for value in depth_targets)
        ]
    return []


def _ordered_action_codes(row: dict[str, Any]) -> list[int]:
    latent_trace = row.get("latent_action_trace")
    if isinstance(latent_trace, list) and latent_trace:
        codes = [
            int(step["action_code"])
            for step in latent_trace
            if isinstance(step, dict) and step.get("action_code") is not None
        ]
        if codes:
            return codes
    transition_codes = row.get("transition_state_codes")
    if isinstance(transition_codes, dict) and transition_codes:
        return [
            int(transition_codes[str(depth)])
            for depth in sorted(int(value) for value in transition_codes)
        ]
    return []


def _hold_code(row: dict[str, Any], action_codes: list[int]) -> int:
    version = str(row.get("latent_action_codebook_version") or "role_v1")
    if version in HOLD_CODE_BY_CODEBOOK_VERSION:
        return int(HOLD_CODE_BY_CODEBOOK_VERSION[version])
    if action_codes:
        return int(action_codes[-1])
    raise ValueError("cannot infer hold code for row without action codes")


def _finality_for_action(
    row: dict[str, Any],
    *,
    action_code: int,
    state_text: str,
    final_answer: str,
    finality_mode: str,
) -> int:
    if finality_mode == "answer_match":
        return int(state_text == final_answer)
    if finality_mode == "action_terminal":
        version = str(row.get("latent_action_codebook_version") or "role_v1")
        terminal_codes = TERMINAL_CODES_BY_CODEBOOK_VERSION.get(version)
        if terminal_codes is None:
            raise ValueError(
                "action_terminal finality requires a known latent action codebook"
            )
        return int(int(action_code) in terminal_codes)
    raise ValueError(f"unknown finality_mode: {finality_mode!r}")


def densify_transition_targets(
    row: dict[str, Any],
    *,
    max_depth: int | None = None,
    finality_mode: str = "answer_match",
) -> dict[str, Any]:
    out = dict(row)
    target_depth = int(max_depth or _max_depth(row, default=8))
    if target_depth <= 0:
        raise ValueError("max_depth must be positive")

    final_answer = _final_answer(row)
    state_texts = _ordered_state_texts(row)
    action_codes = _ordered_action_codes(row)
    if not state_texts:
        raise ValueError("row must contain solver_trace state_texts or depth_targets")
    if not action_codes:
        raise ValueError("row must contain latent_action_trace or transition_state_codes")

    hold_code = _hold_code(row, action_codes)
    dense_depth_targets: dict[str, str] = {}
    dense_codes: dict[str, int] = {}
    dense_finality: dict[str, int] = {}
    dense_solver_trace: list[dict[str, Any]] = []
    dense_latent_trace: list[dict[str, Any]] = []
    original_latent_trace = row.get("latent_action_trace")
    action_names = []
    if isinstance(original_latent_trace, list):
        action_names = [
            str(step.get("action_name") or "")
            for step in original_latent_trace
            if isinstance(step, dict)
        ]
    original_solver_trace = row.get("solver_trace")
    operations = []
    if isinstance(original_solver_trace, list):
        operations = [
            str(step.get("operation") or "")
            for step in original_solver_trace
            if isinstance(step, dict)
        ]

    for depth in range(1, target_depth + 1):
        index = depth - 1
        state_text = state_texts[index] if index < len(state_texts) else final_answer
        action_code = action_codes[index] if index < len(action_codes) else hold_code
        if index >= len(state_texts) and action_code != hold_code:
            action_code = hold_code
        action_name = (
            action_names[index]
            if index < len(action_names) and action_names[index]
            else ("hold_final" if action_code == hold_code else f"action_{action_code}")
        )
        dense_depth_targets[str(depth)] = state_text
        dense_codes[str(depth)] = int(action_code)
        dense_finality[str(depth)] = _finality_for_action(
            row,
            action_code=int(action_code),
            state_text=state_text,
            final_answer=final_answer,
            finality_mode=finality_mode,
        )
        operation = (
            operations[index]
            if index < len(operations) and operations[index]
            else ("hold_final" if action_code == hold_code else "")
        )
        solver_step = {"depth": depth, "state_text": state_text}
        if operation:
            solver_step["operation"] = operation
        dense_solver_trace.append(solver_step)
        dense_latent_trace.append(
            {"depth": depth, "action_code": int(action_code), "action_name": action_name}
        )

    out["depth_targets"] = dense_depth_targets
    out["transition_state_codes"] = dense_codes
    out["transition_finality_targets"] = dense_finality
    out["solver_trace"] = dense_solver_trace
    out["latent_action_trace"] = dense_latent_trace
    out["dense_transition_targets_applied"] = True
    out["dense_transition_target_depth"] = target_depth
    return out


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_dense_file(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    max_depth: int = 8,
    finality_mode: str = "answer_match",
) -> dict[str, Any]:
    rows = load_rows(input_jsonl)
    dense_rows = [
        densify_transition_targets(
            row,
            max_depth=int(max_depth),
            finality_mode=finality_mode,
        )
        for row in rows
    ]
    write_jsonl(output_jsonl, dense_rows)
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "rows": len(dense_rows),
        "max_depth": int(max_depth),
        "finality_mode": finality_mode,
        "families": sorted(
            {
                str(row.get("task_family") or row.get("category") or "unknown")
                for row in dense_rows
            }
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert sparse transition targets into dense 1..N targets."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--finality-mode",
        choices=("answer_match", "action_terminal"),
        default="answer_match",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_dense_file(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        max_depth=args.max_depth,
        finality_mode=args.finality_mode,
    )
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "wrote dense transition targets: "
        f"rows={summary['rows']} max_depth={summary['max_depth']} "
        f"out={summary['output_jsonl']}"
    )


if __name__ == "__main__":
    main()
