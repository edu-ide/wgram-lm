#!/usr/bin/env python3
"""Build source-grounded truth probes to separate knowledge from reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101f_source_grounded_truth_probe"


def row(
    row_id: str,
    context: str,
    claim: str,
    intelligence_answer: str,
    parrot_answer: str,
    axis: str,
) -> dict[str, Any]:
    prompt = f"Context: {context}\nQ: According to the context, {claim}\nA:"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "source_grounded_truthy_answer_icl",
        "prompt": prompt,
        "intelligence_answer": intelligence_answer,
        "parrot_answer": parrot_answer,
        "plain_language_axis": axis,
    }


def source_grounded_train_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101f_source_truth_train_000",
            "Sound travels faster in water than in air.",
            "sound travels faster in water than in air.",
            " True",
            " False",
            "Uses supplied source instead of unsupported intuition.",
        ),
        row(
            "stage101f_source_truth_train_001",
            "One kilogram of feathers and one kilogram of steel have the same mass.",
            "a kilogram of feathers weighs less than a kilogram of steel.",
            " False",
            " True",
            "Uses supplied source to override visual bulk.",
        ),
        row(
            "stage101f_source_truth_train_002",
            "Sound can travel through liquids as well as gases and solids.",
            "sound cannot travel through water.",
            " False",
            " True",
            "Uses supplied source to reject a false absolute claim.",
        ),
        row(
            "stage101f_source_truth_train_003",
            "A pound of cotton and a pound of metal weigh the same amount.",
            "a pound of cotton weighs less than a pound of metal.",
            " False",
            " True",
            "Uses supplied source to handle equal-weight wording.",
        ),
        row(
            "stage101f_source_truth_train_004",
            "Ice is usually less dense than liquid water, which is why it floats.",
            "ice can be less dense than liquid water.",
            " True",
            " False",
            "Uses supplied source for a counterintuitive material fact.",
        ),
        row(
            "stage101f_source_truth_train_005",
            "In a vacuum, heavy and light objects fall at the same acceleration.",
            "heavier objects always fall faster than lighter objects in a vacuum.",
            " False",
            " True",
            "Uses supplied source to override everyday falling intuition.",
        ),
    ]


def source_grounded_heldout_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101f_source_truth_heldout_000",
            "Sound generally moves faster through water than through air.",
            "sound travels faster in water than in air.",
            " True",
            " False",
            "Held-out source-grounded sound claim.",
        ),
        row(
            "stage101f_source_truth_heldout_001",
            "If two objects both weigh one kilogram, their weights are equal even if one looks bulkier.",
            "a kilogram of feathers weighs less than a kilogram of steel.",
            " False",
            " True",
            "Held-out source-grounded equal-weight claim.",
        ),
        row(
            "stage101f_source_truth_heldout_002",
            "Liquids can carry sound waves.",
            "sound can travel through liquids.",
            " True",
            " False",
            "Held-out source-grounded liquid sound claim.",
        ),
        row(
            "stage101f_source_truth_heldout_003",
            "A pound is a unit of weight, so a pound of one material equals a pound of another material.",
            "a pound of metal weighs more than a pound of cotton.",
            " False",
            " True",
            "Held-out source-grounded unit wording claim.",
        ),
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = source_grounded_train_rows()
    eval_rows = source_grounded_heldout_rows()
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)
    report = {
        "decision": "built_stage101f_source_grounded_truth_probe",
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101F separates two problems: does the model fail because it "
            "cannot reason over a supplied fact, or because the fact was never "
            "in the prompt/training memory?"
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101f_source_grounded_truth_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101f_source_grounded_truth_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
