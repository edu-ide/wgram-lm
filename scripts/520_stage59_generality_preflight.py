#!/usr/bin/env python3
"""Preflight the Stage58 VTE path for Stage59 generality transfer."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row must be a JSON object at {path}:{line_no}")
        rows.append(row)
        if int(limit) > 0 and len(rows) >= int(limit):
            break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def answer_alias(row: dict[str, Any]) -> str:
    aliases = row.get("answer_aliases")
    if isinstance(aliases, list) and aliases:
        return str(aliases[0])
    if row.get("answer") is not None:
        return str(row["answer"])
    if row.get("answer_text") is not None:
        return str(row["answer_text"])
    return ""


def answer_kind(answer: str) -> str:
    text = str(answer).strip()
    if re.fullmatch(r"\d", text):
        return "single_digit"
    if re.fullmatch(r"-?\d+", text):
        return "integer_multi_digit"
    if text == "EMPTY":
        return "empty_token"
    if re.fullmatch(r"-?\d+(,-?\d+)+", text):
        return "csv_integer_list"
    return "free_text_or_other"


def scan_current_interfaces(repo: Path) -> dict[str, Any]:
    stage517 = repo / "scripts" / "517_train_qwen_register_extractor.py"
    stage518 = repo / "scripts" / "518_train_token_local_register_extractor.py"
    stage517_text = stage517.read_text(encoding="utf-8") if stage517.exists() else ""
    stage518_text = stage518.read_text(encoding="utf-8") if stage518.exists() else ""
    return {
        "candidate_exposure": {
            "file": str(stage517),
            "digit_only": "def sample_candidate_digits" in stage517_text and "answer_logits" in stage517_text,
            "evidence": "sample_candidate_digits consumes answer_logits and returns argmax/topk digit indices.",
        },
        "typed_register_verifier": {
            "file": str(stage518),
            "mod10_ops_only": "OP_TO_ID = {\"add\": 0, \"mul\": 1, \"sub\": 2, \"copy\": 3}" in stage518_text,
            "evidence": "token-local register packs add/mul/sub/copy digits and execute_predicted_registers returns modulo-style integer answers.",
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.eval_jsonl, limit=int(args.limit))
    family_counts = Counter(str(row.get("task_family") or row.get("family") or "unknown") for row in rows)
    kind_counts = Counter(answer_kind(answer_alias(row)) for row in rows)
    examples = []
    for row in rows[: int(args.example_count)]:
        examples.append(
            {
                "id": row.get("id") or row.get("case_id"),
                "family": row.get("task_family") or row.get("family"),
                "answer": answer_alias(row),
                "answer_kind": answer_kind(answer_alias(row)),
                "prompt": row.get("prompt") or row.get("qwen_prompt") or row.get("question"),
            }
        )

    interfaces = scan_current_interfaces(Path(args.repo_root))
    single_digit_rows = int(kind_counts.get("single_digit", 0))
    non_digit_rows = len(rows) - single_digit_rows
    direct_reuse_possible = non_digit_rows == 0 and not interfaces["candidate_exposure"]["digit_only"]
    failed_axes: list[str] = []
    if interfaces["candidate_exposure"]["digit_only"] and non_digit_rows > 0:
        failed_axes.append("speaker_candidate_exposure_digit_only")
    if interfaces["typed_register_verifier"]["mod10_ops_only"] and non_digit_rows > 0:
        failed_axes.append("verifier_mod10_register_not_general")

    decision = "blocked_direct_reuse_requires_general_answer_interface" if failed_axes else "ready_for_direct_transfer_smoke"
    return {
        "stage": "Stage59A generality preflight",
        "eval_jsonl": str(args.eval_jsonl),
        "rows": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "answer_kind_counts": dict(sorted(kind_counts.items())),
        "interfaces": interfaces,
        "humanistic_read": {
            "reader": "Qwen can read the text prompts, so reader transfer is plausible.",
            "thinker": "The recurrent thought loop exists, but its exposed answers are final digit candidates.",
            "speaker": "The current mouth can say one digit, while Stage59A answers include integers and CSV lists.",
            "verifier": "The current checker is a modulo-10 typed register, not a shared verifier for multi-token answers.",
            "conclusion": "Direct Stage58 reuse would test the wrong thing. It would fail the answer interface before testing general thinking.",
        },
        "failed_axes": failed_axes,
        "decision": decision,
        "required_next_architecture": [
            "Replace digit-only candidate exposure with candidate text/span exposure or a typed answer-object interface.",
            "Use a thin common verifier API that normalizes/scoring answers, not a new hand-coded executor per family.",
            "Keep reader -> recurrent thought/search -> top-k exposure -> verifier -> selected answer unchanged.",
        ],
        "promotion_gate": [
            "At least two non-modulo families improve over Qwen/QTRM baselines.",
            "The gain disappears when recurrent thought/search or top-k exposure is disabled.",
            "No family-specific answer executor is allowed in the normal answer path.",
        ],
        "examples": examples,
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage59A Generality Preflight",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Plain-Language Result",
        "",
        report["humanistic_read"]["conclusion"],
        "",
        "Current Stage58 is a strong calculation workshop, but direct reuse is not yet a fair test of a general thinking office.",
        "",
        "## Evidence",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Families: `{report['family_counts']}`",
        f"- Answer kinds: `{report['answer_kind_counts']}`",
        f"- Failed axes: `{report['failed_axes']}`",
        "",
        "## Required Next Architecture",
        "",
    ]
    lines.extend(f"- {item}" for item in report["required_next_architecture"])
    lines.extend(["", "## Promotion Gate", ""])
    lines.extend(f"- {item}" for item in report["promotion_gate"])
    lines.extend(["", "## Examples", ""])
    for example in report["examples"]:
        lines.append(f"- `{example['id']}` `{example['family']}` answer=`{example['answer']}` kind=`{example['answer_kind']}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", default="data/eval/pure_recursive_hard_family_heldout200_cases.jsonl")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--example-count", type=int, default=6)
    parser.add_argument("--out-json", default="local_eval/stage59_generality_preflight/report.json")
    parser.add_argument("--out-md", default="local_eval/stage59_generality_preflight/report.md")
    args = parser.parse_args()

    report = build_report(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps({"decision": report["decision"], "failed_axes": report["failed_axes"], "out_json": str(out_json), "out_md": str(args.out_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
