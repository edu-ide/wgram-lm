#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qtrm_mm.eval.architecture_ablation_proof import (
    build_ablation_summary,
    write_ablation_summary,
)


DEFAULT_EVALS = [
    {
        "name": "expanded donor/residual gate",
        "path": "runs/eval/memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl",
    },
    {
        "name": "expanded workspace/core ablation gate",
        "path": "runs/eval/memory_reasoning_heldout_expanded_workspace_core_ablation_32tok_synth_generalization_s050.jsonl",
    },
    {
        "name": "expanded strict causality ablation gate",
        "path": "runs/eval/memory_reasoning_heldout_expanded_strict_causality_ablation_32tok_synth_generalization_s050.jsonl",
    },
]


def parse_eval_spec(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("eval spec must be NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("eval spec must be NAME=PATH")
    return {"name": name, "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded QTRM workspace/core ablation proof.")
    parser.add_argument(
        "--eval",
        action="append",
        type=parse_eval_spec,
        default=None,
        help="Eval JSONL as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/wiki/decisions/expanded-workspace-core-ablation.md",
    )
    parser.add_argument(
        "--json-out",
        default="docs/wiki/decisions/expanded-workspace-core-ablation-summary.json",
    )
    args = parser.parse_args()

    proof = build_ablation_summary(args.eval or DEFAULT_EVALS)
    write_ablation_summary(proof, markdown_out=args.markdown_out, json_out=args.json_out)
    print(f"wrote {args.markdown_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
