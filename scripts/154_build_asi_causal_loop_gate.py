#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wgram_lm.agentic.causal_gate import evaluate_causal_loop_gate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an ASI causal-loop gate from harness/QTRM metric JSON."
    )
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument(
        "--markdown-out",
        default="docs/wiki/decisions/asi-causal-loop-gate.md",
    )
    parser.add_argument(
        "--json-out",
        default="docs/wiki/decisions/asi-causal-loop-gate-summary.json",
    )
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--min-drop", type=float, default=0.03)
    return parser


def write_gate_report(
    metrics_json: str,
    *,
    markdown_out: str,
    json_out: str,
    min_gain: float,
    min_drop: float,
) -> dict[str, Any]:
    metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    gate = evaluate_causal_loop_gate(metrics, min_gain=min_gain, min_drop=min_drop)
    Path(markdown_out).parent.mkdir(parents=True, exist_ok=True)
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_out).write_text(render_markdown(gate), encoding="utf-8")
    Path(json_out).write_text(
        json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return gate


def render_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# ASI Causal Loop Gate",
        "",
        "## Verdict",
        "",
        f"Status: `{gate.get('status', 'unknown')}`",
        "",
        "## Baseline Gains",
        "",
        "| Check | Value |",
        "| --- | ---: |",
        f"| gain_over_donor_harness | {float(gate.get('gain_over_donor_harness', 0.0)):.4f} |",
        f"| gain_over_scripted_harness | {float(gate.get('gain_over_scripted_harness', 0.0)):.4f} |",
        f"| min_gain | {float(gate.get('min_gain', 0.0)):.4f} |",
        "",
        "## Causal Drops",
        "",
        "| Ablation | Drop |",
        "| --- | ---: |",
    ]
    for key, drop in sorted(dict(gate.get("causal_drops", {})).items()):
        lines.append(f"| {key} | {float(drop):.4f} |")
    lines.extend(
        [
            "",
            "## Failed Checks",
            "",
        ]
    )
    failed_checks = list(gate.get("failed_checks", []))
    if failed_checks:
        lines.extend(f"- `{check}`" for check in failed_checks)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    gate = write_gate_report(
        args.metrics_json,
        markdown_out=args.markdown_out,
        json_out=args.json_out,
        min_gain=args.min_gain,
        min_drop=args.min_drop,
    )
    print(f"status={gate['status']}")
    print(f"wrote {args.markdown_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
