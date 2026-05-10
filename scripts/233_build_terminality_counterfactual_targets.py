#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TERMINAL_V2_SIZE = 5
TERMINAL_V2_NAMES = {
    0: "extract_or_unary_nonterminal",
    1: "compose_from_previous_terminal",
    2: "compose_from_previous_nonterminal",
    3: "final_compose_from_previous_terminal",
    4: "hold_final",
}


def _prompt_for_question(question: str) -> str:
    return (
        "Answer with only the final answer. Do not write reasoning.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _family(row: dict[str, Any]) -> str:
    return str(row.get("task_family") or row.get("category") or "")


def _answer(row: dict[str, Any]) -> str:
    aliases = row.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    return str(row.get("answer") or row.get("chosen") or "")


def _trace_text(row: dict[str, Any], depth: int) -> str:
    trace = row.get("solver_trace")
    if isinstance(trace, list):
        for step in trace:
            if isinstance(step, dict) and int(step.get("depth", -1)) == int(depth):
                return str(step.get("state_text") or "")
    depth_targets = row.get("depth_targets")
    if isinstance(depth_targets, dict):
        value = depth_targets.get(str(depth))
        if value is not None:
            return str(value)
    raise ValueError(f"row {row.get('id')} is missing state text for depth {depth}")


def _with_transition_targets(
    row: dict[str, Any],
    *,
    depth_targets: dict[str, str],
    codes: dict[str, int],
    finality: dict[str, int],
    source_id: str,
    variant_kind: str,
) -> dict[str, Any]:
    out = dict(row)
    out["depth_targets"] = dict(depth_targets)
    out["transition_state_codes"] = {key: int(value) for key, value in codes.items()}
    out["transition_finality_targets"] = {
        key: int(value) for key, value in finality.items()
    }
    out["solver_trace"] = [
        {"depth": int(depth), "state_text": depth_targets[str(depth)]}
        for depth in (1, 2, 4, 8)
    ]
    out["latent_action_trace"] = [
        {
            "depth": int(depth),
            "action_code": int(codes[str(depth)]),
            "action_name": TERMINAL_V2_NAMES[int(codes[str(depth)])],
        }
        for depth in (1, 2, 4, 8)
    ]
    out["latent_action_codebook_applied"] = True
    out["latent_action_codebook_version"] = "terminal_v2"
    out["latent_action_codebook_size"] = TERMINAL_V2_SIZE
    out["terminality_counterfactual_applied"] = True
    out["terminality_counterfactual_source_id"] = source_id
    out["terminality_counterfactual_kind"] = variant_kind
    return out


def arithmetic_terminal_depth2(row: dict[str, Any]) -> dict[str, Any]:
    if _family(row) != "arithmetic_chain":
        raise ValueError("arithmetic_terminal_depth2 requires arithmetic_chain row")
    question = str(row.get("question") or "")
    match = re.search(
        r"(\(\([-]?\d+\s*\+\s*[-]?\d+\)\s*\*\s*[-]?\d+\))\s*-\s*[-]?\d+",
        question,
    )
    if not match:
        raise ValueError(f"cannot derive terminal arithmetic expression: {question!r}")
    terminal_expr = match.group(1)
    depth1 = _trace_text(row, 1)
    depth2 = _trace_text(row, 2)
    new_question = (
        f"Compute the two-step expression {terminal_expr}. "
        "Return only the integer result."
    )
    out = dict(row)
    out["id"] = f"{row.get('id', 'arith')}-terminal-depth2"
    out["question"] = new_question
    out["prompt"] = _prompt_for_question(new_question)
    out["answer"] = depth2
    out["chosen"] = depth2
    out["answer_aliases"] = [depth2]
    return _with_transition_targets(
        out,
        depth_targets={"1": depth1, "2": depth2, "4": depth2, "8": depth2},
        codes={"1": 0, "2": 1, "4": 4, "8": 4},
        finality={"1": 0, "2": 1, "4": 1, "8": 1},
        source_id=str(row.get("id") or ""),
        variant_kind="arithmetic_terminal_depth2",
    )


def symbolic_nonterminal_depth3(row: dict[str, Any]) -> dict[str, Any]:
    if _family(row) != "symbolic_binding":
        raise ValueError("symbolic_nonterminal_depth3 requires symbolic_binding row")
    question = str(row.get("question") or "")
    mappings = re.findall(r"\b([A-Za-z]+)\s+maps\s+to\s+([A-Za-z]+)\b", question)
    if len(mappings) < 3:
        raise ValueError(f"cannot derive 3-hop symbolic mapping: {question!r}")
    start = mappings[0][0]
    depth1 = mappings[0][1]
    depth2 = mappings[1][1]
    depth3 = mappings[2][1]
    mapping_text = "; ".join(f"{source} maps to {target}" for source, target in mappings)
    new_question = (
        f"Mapping facts: {mapping_text}. Starting at {start}, apply the mapping "
        "three times. What symbol is reached?"
    )
    out = dict(row)
    out["id"] = f"{row.get('id', 'symbolic')}-nonterminal-depth3"
    out["question"] = new_question
    out["prompt"] = _prompt_for_question(new_question)
    out["answer"] = depth3
    out["chosen"] = depth3
    out["answer_aliases"] = [depth3]
    return _with_transition_targets(
        out,
        depth_targets={"1": depth1, "2": depth2, "4": depth3, "8": depth3},
        codes={"1": 0, "2": 2, "4": 3, "8": 4},
        finality={"1": 0, "2": 0, "4": 1, "8": 1},
        source_id=str(row.get("id") or ""),
        variant_kind="symbolic_nonterminal_depth3",
    )


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


def build_terminality_counterfactual_file(
    *,
    base_train_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_out: str | Path | None = None,
) -> dict[str, Any]:
    rows = load_rows(base_train_jsonl)
    augmented_rows = list(rows)
    added: list[dict[str, Any]] = []
    for row in rows:
        family = _family(row)
        if family == "arithmetic_chain":
            added.append(arithmetic_terminal_depth2(row))
        elif family == "symbolic_binding":
            added.append(symbolic_nonterminal_depth3(row))
    augmented_rows.extend(added)
    write_jsonl(output_jsonl, augmented_rows)
    summary = {
        "base_train_jsonl": str(base_train_jsonl),
        "output_jsonl": str(output_jsonl),
        "base_rows": len(rows),
        "added_rows": len(added),
        "total_rows": len(augmented_rows),
        "added_by_kind": {},
        "families": sorted({_family(row) for row in augmented_rows}),
        "note": (
            "Adds terminality counterfactuals without adding held-out "
            "list_transform training rows."
        ),
    }
    for row in added:
        kind = str(row.get("terminality_counterfactual_kind") or "")
        summary["added_by_kind"][kind] = int(summary["added_by_kind"].get(kind, 0)) + 1
    if summary_out:
        out = Path(summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Augment terminal_v2 latent-action training data with terminality "
            "counterfactuals while keeping list_transform held out."
        )
    )
    parser.add_argument("--base-train-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_terminality_counterfactual_file(
        base_train_jsonl=args.base_train_jsonl,
        output_jsonl=args.output_jsonl,
        summary_out=args.summary_out or None,
    )
    print(
        "wrote terminality counterfactual train rows: "
        f"base={summary['base_rows']} added={summary['added_rows']} "
        f"total={summary['total_rows']} out={summary['output_jsonl']}"
    )


if __name__ == "__main__":
    main()
