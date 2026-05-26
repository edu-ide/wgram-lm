#!/usr/bin/env python3
"""Assess which raw-intelligence axes a PrefixLM run actually covers.

This is deliberately conservative. A falling PrefixLM loss proves that a
one-body language path is learning, but it does not by itself prove reasoning,
memory, verifier judgment, planning, or metacognitive control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


RAW_INTELLIGENCE_AXES = (
    "language_body",
    "reasoning",
    "working_memory",
    "verifier_judgment",
    "ood_generalization",
    "planning",
    "metacognitive_control",
)


@dataclass(frozen=True)
class AxisAssessment:
    axis: str
    status: str
    plain_language: str
    evidence: list[str]
    next_gate: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed_file = json.loads(text)
    except json.JSONDecodeError:
        parsed_file = None
    if isinstance(parsed_file, dict):
        if isinstance(parsed_file.get("loss_history"), list):
            rows.extend(row for row in parsed_file["loss_history"] if isinstance(row, dict))
        elif isinstance(parsed_file.get("axes"), list):
            rows.append(parsed_file)
        else:
            rows.append(parsed_file)
        return rows
    if isinstance(parsed_file, list):
        rows.extend(row for row in parsed_file if isinstance(row, dict))
        return rows

    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def has_key(rows: Iterable[dict[str, Any]], key: str) -> bool:
    return any(key in row for row in rows)


def last_value(rows: Iterable[dict[str, Any]], key: str) -> Any | None:
    candidates = [row for row in rows if key in row]
    if not candidates:
        return None
    if key.endswith("tokens_seen"):
        numeric = [row[key] for row in candidates if isinstance(row[key], (int, float))]
        if numeric:
            return max(numeric)
    stepped = [row for row in candidates if isinstance(row.get("step"), (int, float))]
    if stepped:
        return max(stepped, key=lambda row: float(row["step"]))[key]
    return candidates[-1][key]


def assess_raw_intelligence(rows: list[dict[str, Any]]) -> list[AxisAssessment]:
    evidence_prefix = []
    if has_key(rows, "eval_loss"):
        evidence_prefix.append(f"latest eval_loss={last_value(rows, 'eval_loss')}")
    if has_key(rows, "loss"):
        evidence_prefix.append(f"latest train loss={last_value(rows, 'loss')}")
    if has_key(rows, "target_tokens_seen"):
        evidence_prefix.append(f"target_tokens_seen={last_value(rows, 'target_tokens_seen')}")

    language_status = "covered_proxy" if has_key(rows, "eval_loss") else "missing"
    language_evidence = evidence_prefix or ["no PrefixLM train/eval rows found"]

    if has_key(rows, "verifier_selected_accuracy"):
        verifier_status = "selection_tested"
        verifier_evidence = [
            f"raw_lm_top1_accuracy={last_value(rows, 'raw_lm_top1_accuracy')}",
            f"verifier_selected_accuracy={last_value(rows, 'verifier_selected_accuracy')}",
            f"verifier_gain={last_value(rows, 'verifier_gain')}",
        ]
    elif has_key(rows, "token_verifier_loss"):
        verifier_status = "wired_smoke"
        verifier_evidence = [
            f"token_verifier_loss={last_value(rows, 'token_verifier_loss')}",
            f"token_verifier_accuracy={last_value(rows, 'token_verifier_accuracy')}",
        ]
    else:
        verifier_status = "missing"
        verifier_evidence = ["no token_verifier_loss or verifier selection rows found"]

    ood_status = "weak_proxy" if has_key(rows, "eval_loss") else "missing"

    return [
        AxisAssessment(
            axis="language_body",
            status=language_status,
            plain_language=(
                "The model is being tested as one body that reads context and "
                "speaks response tokens through its own LM head."
            ),
            evidence=language_evidence,
            next_gate="Heldout PrefixLM loss must keep improving on row-fixed eval.",
        ),
        AxisAssessment(
            axis="reasoning",
            status="not_covered",
            plain_language=(
                "A lower language loss does not prove the model can solve a new "
                "multi-step problem."
            ),
            evidence=["no task-level reasoning accuracy rows found"],
            next_gate="Add GSM/synthetic algorithmic exact-match and ablation metrics.",
        ),
        AxisAssessment(
            axis="working_memory",
            status="not_covered",
            plain_language=(
                "The run uses context, but it has not yet proven trainable "
                "working-memory manipulation or long-memory recall."
            ),
            evidence=["no memory retrieval/manipulation rows found"],
            next_gate="Add source-bound recall and working-memory manipulation tests.",
        ),
        AxisAssessment(
            axis="verifier_judgment",
            status=verifier_status,
            plain_language=(
                "This asks whether the model has an internal eye for choosing "
                "a better candidate, not just a mouth for emitting tokens."
            ),
            evidence=verifier_evidence,
            next_gate=(
                "Verifier-selected candidates must beat raw LM candidates, and "
                "the gain must vanish with verifier off."
            ),
        ),
        AxisAssessment(
            axis="ood_generalization",
            status=ood_status,
            plain_language=(
                "Row-fixed eval is a useful heldout check, but not a deep OOD "
                "reasoning-generalization proof."
            ),
            evidence=language_evidence,
            next_gate="Add unseen-depth/unseen-format exact-match gates.",
        ),
        AxisAssessment(
            axis="planning",
            status="not_covered",
            plain_language=(
                "The current run does not prove that the model can form and "
                "execute a multi-step plan."
            ),
            evidence=["no planning task metrics found"],
            next_gate="Add multi-step instruction and plan execution tasks.",
        ),
        AxisAssessment(
            axis="metacognitive_control",
            status="not_covered",
            plain_language=(
                "The current run does not prove that the model knows when to "
                "stop, retry, or distrust an answer."
            ),
            evidence=["no halt/retry/uncertainty calibration rows found"],
            next_gate="Add adaptive halt, retry, and calibration metrics.",
        ),
    ]


def build_report(log_paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in log_paths:
        rows.extend(read_jsonl(path))
    return build_report_from_rows(rows)


def build_report_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    axes = assess_raw_intelligence(rows)
    covered = [
        axis.axis
        for axis in axes
        if axis.status in {"covered_proxy", "wired_smoke", "selection_tested"}
    ]
    missing = [axis.axis for axis in axes if axis.status in {"missing", "not_covered"}]
    weak = [axis.axis for axis in axes if axis.status == "weak_proxy"]
    return {
        "claim_raw_intelligence": False,
        "plain_language_read": (
            "This run is a one-body language/recurrent-thought spine test. "
            "It is not yet a full raw-intelligence test."
        ),
        "covered_or_wired_axes": covered,
        "weak_proxy_axes": weak,
        "missing_axes": missing,
        "axes": [asdict(axis) for axis in axes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", default=[], help="JSONL stdout log to assess.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_report([Path(path) for path in args.log])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
