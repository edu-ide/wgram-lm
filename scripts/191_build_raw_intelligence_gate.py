#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wgram_lm.eval.raw_intelligence_gate import (
    build_raw_intelligence_gate,
    load_records,
    write_gate,
)


DEFAULT_EVAL_JSONL = [
    "runs/eval/pure_recursive_reasoning_depth_sweep.jsonl",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a raw-intelligence gate from QTRM depth/memory ablation eval JSONL."
    )
    parser.add_argument(
        "--eval-jsonl",
        action="append",
        default=None,
        help="Eval JSONL path. Can be repeated.",
    )
    parser.add_argument(
        "--gate-type",
        default="pure_recursive_reasoning",
        choices=[
            "pure_recursive_reasoning",
            "trainable_memory_intelligence",
            "reasoning_memory_composition",
            "temporal_spatial_context",
            "ri4_sparse_persistent_memory",
            "hybrid_recurrence_depth_scaling",
            "hybrid_556_causal_matrix",
        ],
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/wiki/decisions/raw-intelligence-gate.md",
    )
    parser.add_argument(
        "--json-out",
        default="docs/wiki/decisions/raw-intelligence-gate-summary.json",
    )
    return parser


def write_gate_report(
    eval_jsonl: list[str],
    *,
    gate_type: str,
    markdown_out: str,
    json_out: str,
) -> dict:
    records = load_records(eval_jsonl)
    gate = build_raw_intelligence_gate(records, gate_type=gate_type)
    write_gate(gate, markdown_out=markdown_out, json_out=json_out)
    return gate


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_jsonl = args.eval_jsonl or DEFAULT_EVAL_JSONL
    gate = write_gate_report(
        eval_jsonl,
        gate_type=args.gate_type,
        markdown_out=args.markdown_out,
        json_out=args.json_out,
    )
    print(f"status={gate['status']}")
    print(f"wrote {args.markdown_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
