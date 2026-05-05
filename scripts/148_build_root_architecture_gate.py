#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qtrm_mm.eval.root_architecture_gate import (
    DEFAULT_BASELINE_MODE,
    DEFAULT_COMPARISON_MODES,
    DEFAULT_CRITICAL_MODES,
    build_root_architecture_gate,
    load_records,
    write_gate,
)


DEFAULT_EVAL_JSONL = [
    "runs/eval/memory_reasoning_heldout_expanded_workspace_evidence_counterfactual_32tok_trained_s050.jsonl",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a root-architecture causality gate from QTRM ablation eval JSONL."
    )
    parser.add_argument(
        "--eval-jsonl",
        action="append",
        default=None,
        help="Eval JSONL path. Can be repeated.",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/wiki/decisions/root-architecture-causality-gate.md",
    )
    parser.add_argument(
        "--json-out",
        default="docs/wiki/decisions/root-architecture-causality-gate-summary.json",
    )
    parser.add_argument(
        "--strict-promotion-gate",
        action="store_true",
        help=(
            "Require full QTRM to beat donor-only and forbid critical ablations "
            "from outperforming the full path. This separates a narrow causal "
            "signal from architecture promotion."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        default=DEFAULT_BASELINE_MODE,
        help="Mode to treat as the candidate baseline. Defaults to full QTRM residual.",
    )
    parser.add_argument(
        "--critical-mode",
        action="append",
        default=None,
        help=(
            "Critical ablation mode to include in the gate. Can be repeated. "
            "Defaults to the broad root-gate mode list."
        ),
    )
    parser.add_argument(
        "--comparison-mode",
        action="append",
        default=None,
        help=(
            "Comparison baseline mode to include in strict promotion checks. "
            "Can be repeated. Defaults to donor-only with evidence."
        ),
    )
    return parser


def write_gate_report(
    eval_jsonl: list[str],
    *,
    markdown_out: str,
    json_out: str,
    strict_promotion_gate: bool = False,
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    critical_modes: list[str] | None = None,
    comparison_modes: list[str] | None = None,
) -> dict:
    records = load_records(eval_jsonl)
    gate = build_root_architecture_gate(
        records,
        baseline_mode=baseline_mode,
        critical_modes=critical_modes or DEFAULT_CRITICAL_MODES,
        comparison_modes=comparison_modes or DEFAULT_COMPARISON_MODES,
        require_donor_advantage=strict_promotion_gate,
        require_no_critical_ablation_improvement=strict_promotion_gate,
    )
    write_gate(gate, markdown_out=markdown_out, json_out=json_out)
    return gate


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_jsonl = args.eval_jsonl or DEFAULT_EVAL_JSONL
    gate = write_gate_report(
        eval_jsonl,
        markdown_out=args.markdown_out,
        json_out=args.json_out,
        strict_promotion_gate=bool(args.strict_promotion_gate),
        baseline_mode=args.baseline_mode,
        critical_modes=args.critical_mode,
        comparison_modes=args.comparison_mode,
    )
    print(f"status={gate['status']}")
    print(f"wrote {args.markdown_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
