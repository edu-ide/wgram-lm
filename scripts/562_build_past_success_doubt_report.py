#!/usr/bin/env python3
"""Build a past-success doubt report before launching another model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _last_history_eval(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"summary has no non-empty history list: {path}")
    last = history[-1]
    if not isinstance(last, dict) or not isinstance(last.get("eval"), dict):
        raise ValueError(f"summary history[-1] has no eval object: {path}")
    return last["eval"]


def extract_ptrm_success_row(path: Path | str, *, label: str) -> dict[str, Any]:
    path = Path(path)
    eval_payload = _last_history_eval(_load_json(path), path=path)
    selected = float(eval_payload["mean_selected_accuracy_oracle_depth"])
    oracle = float(eval_payload["mean_oracle_accuracy"])
    packed = float(eval_payload.get("mean_packed_register_answer_accuracy_oracle_depth", float("nan")))
    return {
        "label": str(label),
        "path": str(path),
        "metric_family": "selected_oracle_search",
        "selected_accuracy": selected,
        "oracle_accuracy": oracle,
        "packed_register_answer_accuracy": packed,
        "exact_metric": f"selected={selected:.4f}, oracle={oracle:.4f}, packed={packed:.4f}",
        "plain_language_proves": (
            "candidate search plus verifier selection works on a compact synthetic "
            "arithmetic/state-space exam"
        ),
        "does_not_prove": (
            "free language generation, multilingual ability, or one-body general LM reasoning"
        ),
        "causal_ingredient": "candidate diversity plus verifier-selected compact answers",
    }


def _iter_json_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"JSON line {line_no} is not an object: {path}")
        rows.append(payload)
    decoder = json.JSONDecoder()
    line_start_indices = [0]
    line_start_indices.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    for start in reversed(line_start_indices):
        if start >= len(text) or text[start] != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not text[start + end :].strip() and isinstance(payload, dict):
            if not rows or rows[-1] != payload:
                rows.append(payload)
            break
    return rows


def extract_language_loss_row(path: Path | str, *, label: str) -> dict[str, Any]:
    path = Path(path)
    rows = _iter_json_lines(path)
    if not rows:
        raise ValueError(f"no JSON records found in language log: {path}")
    summary = next((row for row in reversed(rows) if "final_eval_loss" in row), None)
    if summary is not None:
        initial = float(summary.get("initial_eval_loss", rows[0].get("eval_loss", float("nan"))))
        final = float(summary["final_eval_loss"])
    else:
        eval_rows = [row for row in rows if "eval_loss" in row]
        if not eval_rows:
            raise ValueError(f"no eval_loss or final_eval_loss found in language log: {path}")
        initial = float(eval_rows[0]["eval_loss"])
        final = float(eval_rows[-1]["eval_loss"])
    return {
        "label": str(label),
        "path": str(path),
        "metric_family": "teacher_forced_loss",
        "initial_eval_loss": initial,
        "final_eval_loss": final,
        "exact_metric": f"eval_loss {initial:.4f} -> {final:.4f}",
        "plain_language_proves": "heldout CE fell under teacher forcing on this data contract",
        "does_not_prove": "free generation, candidate selection, depth scaling, or general reasoning",
        "causal_ingredient": "ordinary next-token supervision over the logged heldout rows",
    }


def build_recommended_comparison_row(
    old_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> dict[str, str]:
    search_rows = [row for row in old_rows if row.get("metric_family") == "selected_oracle_search"]
    if search_rows:
        best_old = max(search_rows, key=lambda row: float(row.get("selected_accuracy", 0.0)))
    elif old_rows:
        best_old = old_rows[0]
    else:
        best_old = {}
    current_families = {str(row.get("metric_family", "")) for row in current_rows}
    missing_parts = [
        "free generation samples",
        "selected-vs-oracle split on the normal one-body answer path",
        "candidate diversity/coverage if search is claimed",
        "depth or recurrent-core-off ablation",
    ]
    if "teacher_forced_loss" not in current_families:
        missing_parts.append("teacher-forced heldout loss")
    return {
        "old_success": str(best_old.get("label", "")),
        "exact_metric": str(best_old.get("exact_metric", "")),
        "causal_ingredient": str(
            best_old.get("causal_ingredient", "candidate diversity plus verifier-selected answers")
        ),
        "missing_in_current_run": "; ".join(missing_parts),
        "smallest_restoration_test": (
            "Run a small one-body language gate that logs candidate coverage, "
            "selected-vs-oracle accuracy, free generation samples, and "
            "recurrent/depth-off loss deltas on the same heldout rows."
        ),
    }


def build_past_success_doubt_report(
    *,
    old_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    recommended_comparison_row = build_recommended_comparison_row(old_rows, current_rows)
    return {
        "report_type": "past_success_doubt_loop",
        "plain_language_conclusion": (
            "Old selected/oracle arithmetic wins are search-verifier evidence, "
            "not general LM ability. Preserve the causal ingredient, not the overclaim."
        ),
        "old_success_rows": old_rows,
        "current_rows": current_rows,
        "required_comparison_row": {
            "old_success": "",
            "exact_metric": "",
            "causal_ingredient": "",
            "missing_in_current_run": "",
            "smallest_restoration_test": "",
        },
        "recommended_comparison_row": recommended_comparison_row,
        "launch_recommendation": "do_not_launch_long_run_until_restoration_gate_exists",
        "minimum_next_gate": [
            "teacher_forced_heldout_loss",
            "free_generation_samples",
            "first_response_token_rank_or_topk",
            "repetition_and_eos_rate",
            "selected_vs_oracle_split_when_search_is_used",
            "depth_or_recurrent_core_off_ablation",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    def render_row(row: dict[str, Any]) -> str:
        return "| {label} | {metric_family} | {exact_metric} | {plain_language_proves} | {does_not_prove} |".format(
            label=row.get("label", ""),
            metric_family=row.get("metric_family", ""),
            exact_metric=row.get("exact_metric", ""),
            plain_language_proves=row.get("plain_language_proves", ""),
            does_not_prove=row.get("does_not_prove", ""),
        )

    lines = [
        "# Past-Success Doubt Report",
        "",
        report["plain_language_conclusion"],
        "",
        "In short: selected/oracle wins are useful search evidence, not general LM ability.",
        "",
        "## Old Success Rows",
        "",
        "| Label | Metric Family | Exact Metric | Proves | Does Not Prove |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.get("old_success_rows", []):
        lines.append(render_row(row))
    lines.extend(
        [
            "",
            "## Current Rows",
            "",
            "| Label | Metric Family | Exact Metric | Proves | Does Not Prove |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("current_rows", []):
        lines.append(render_row(row))
    lines.extend(
        [
            "",
            "## Required Comparison Row",
            "",
            "| old_success | exact_metric | causal_ingredient | missing_in_current_run | smallest_restoration_test |",
            "| --- | --- | --- | --- | --- |",
            "|  |  |  |  |  |",
            "",
            "## Recommended Comparison Row",
            "",
            "| old_success | exact_metric | causal_ingredient | missing_in_current_run | smallest_restoration_test |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    row = report.get("recommended_comparison_row", {})
    lines.append(
        "| {old_success} | {exact_metric} | {causal_ingredient} | {missing_in_current_run} | {smallest_restoration_test} |".format(
            old_success=row.get("old_success", ""),
            exact_metric=row.get("exact_metric", ""),
            causal_ingredient=row.get("causal_ingredient", ""),
            missing_in_current_run=row.get("missing_in_current_run", ""),
            smallest_restoration_test=row.get("smallest_restoration_test", ""),
        )
    )
    lines.extend(
        [
            "",
            f"Launch recommendation: `{report.get('launch_recommendation', '')}`",
            "",
            "Do not launch a long run until that row is filled.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ptrm", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--current-language", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    old_rows = [
        extract_ptrm_success_row(path, label=label)
        for label, path in (_parse_label_path(value) for value in args.old_ptrm)
    ]
    current_rows = [
        extract_language_loss_row(path, label=label)
        for label, path in (_parse_label_path(value) for value in args.current_language)
    ]
    report = build_past_success_doubt_report(old_rows=old_rows, current_rows=current_rows)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


if __name__ == "__main__":
    main()
