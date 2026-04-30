#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qtrm_mm.eval.residual_adapter_proof import build_proof_summary, render_markdown


DEFAULT_EVALS = [
    (
        "hard memory probe",
        "runs/eval/memory_reasoning_qwen3_rerank_32tok_trace_s050_ft.jsonl",
    ),
    (
        "held-out memory probe",
        "runs/eval/memory_reasoning_heldout_qwen3_rerank_32tok_synth_generalization_s050.jsonl",
    ),
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
    parser = argparse.ArgumentParser(description="Build the QTRM residual-adapter proof package.")
    parser.add_argument(
        "--eval",
        action="append",
        type=parse_eval_spec,
        default=None,
        help="Eval JSONL as NAME=PATH. Can be repeated. Defaults to hard and held-out MemoryOS probes.",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/wiki/decisions/residual-adapter-proof.md",
    )
    parser.add_argument(
        "--json-out",
        default="docs/wiki/decisions/residual-adapter-proof-summary.json",
    )
    args = parser.parse_args()

    evals = args.eval or [{"name": name, "path": path} for name, path in DEFAULT_EVALS]
    proof = build_proof_summary(evals)

    markdown_out = Path(args.markdown_out)
    json_out = Path(args.json_out)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    markdown_out.write_text(render_markdown(proof), encoding="utf-8")
    json_out.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {markdown_out}")
    print(f"wrote {json_out}")


if __name__ == "__main__":
    main()
