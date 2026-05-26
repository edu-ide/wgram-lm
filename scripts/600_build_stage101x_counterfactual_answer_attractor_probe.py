#!/usr/bin/env python3
"""Build Stage101X counterfactual-imagination answer-attractor probes.

W9 still uses a detached feature reader.  Stage101X removes that promoted path:
the same one-body LM head must answer the real world and the minimally imagined
counterfactual world differently.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage101x_counterfactual_answer_attractor_probe"
PAIR_FEATURES = [
    "source_reliability",
    "evidence_relevance",
    "detail_sufficiency",
    "conflict_status",
]
CHOICES = [" yes", " no"]


def prompt(claim: str, label: str, world: str) -> str:
    return f"Claim: {claim}\n{label}: {world}\nQ: Can answer now? yes or no.\nA:"


def claim_bank(split: str) -> list[tuple[str, str, str, str]]:
    if split == "train":
        return [
            ("parcel arrived.", "delivered", "tracking page", "package color is blue"),
            ("refund posted.", "posted", "bank statement", "card design is silver"),
            ("event starts noon.", "noon", "official schedule", "poster uses green ink"),
            ("invoice paid.", "paid", "ledger", "invoice paper is white"),
            ("platform is 4.", "platform 4", "station board", "train seats are red"),
            ("ticket valid.", "valid", "scanner", "ticket font is bold"),
            ("battery is full.", "full", "device gauge", "case is red"),
            ("alert active.", "active", "live bulletin", "yesterday was calm"),
        ]
    return [
        ("badge valid.", "valid", "security scanner", "badge strap is black"),
        ("route changed.", "changed", "transit board", "bus color is yellow"),
        ("door locked.", "locked", "door sensor", "hinge is steel"),
        ("bus bay is C.", "bay C", "station board", "bench is wooden"),
        ("storm active.", "active", "weather bulletin", "yesterday was warm"),
        ("pin is 7401.", "7401", "notice", "notice border is thin"),
        ("store open.", "open", "latest post", "window sign is square"),
        ("lift running.", "running", "current screen", "button is round"),
    ]


def worlds_for_feature(
    *,
    value: str,
    trusted_source: str,
    irrelevant_fact: str,
    pair_feature: str,
) -> tuple[str, str, str]:
    if pair_feature == "source_reliability":
        return (
            f"{trusted_source} says {value}.",
            f"rumor says {value}.",
            "change only source reliability",
        )
    if pair_feature == "evidence_relevance":
        return (
            f"{trusted_source} says {value}.",
            f"{trusted_source} only says {irrelevant_fact}.",
            "change only evidence relevance",
        )
    if pair_feature == "detail_sufficiency":
        return (
            f"{trusted_source} says {value}.",
            f"{trusted_source} says the value exists, but the exact detail is missing.",
            "change only detail sufficiency",
        )
    if pair_feature == "conflict_status":
        return (
            f"{trusted_source} says {value}.",
            f"one trusted note says {value}; another trusted note says not {value}.",
            "change only conflict status",
        )
    raise ValueError(f"bad pair_feature: {pair_feature!r}")


def make_row(
    *,
    row_id: str,
    split: str,
    claim: str,
    pair_feature: str,
    positive_world: str,
    negative_world: str,
    intervention: str,
    original_is_positive: bool,
) -> dict[str, Any]:
    if original_is_positive:
        original_world = positive_world
        counterfactual_world = negative_world
        original_answer = " yes"
        counterfactual_answer = " no"
    else:
        original_world = negative_world
        counterfactual_world = positive_world
        original_answer = " no"
        counterfactual_answer = " yes"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "stage101x_counterfactual_answer_attractor",
        "stage101x_counterfactual_answer_attractor_required": True,
        "split": split,
        "claim": claim,
        "pair_feature": pair_feature,
        "intervention": intervention,
        "original_world": original_world,
        "counterfactual_world": counterfactual_world,
        "original_prompt": prompt(claim, "Real world", original_world),
        "counterfactual_prompt": prompt(claim, "Imagined change", counterfactual_world),
        "original_answer": original_answer,
        "counterfactual_answer": counterfactual_answer,
        "candidate_answers": list(CHOICES),
        "original_negative_answers": [choice for choice in CHOICES if choice != original_answer],
        "counterfactual_negative_answers": [
            choice for choice in CHOICES if choice != counterfactual_answer
        ],
        "plain_language_axis": (
            "Stage101X asks the same LM head to answer the real world and a "
            "minimal imagined counterfactual differently, without a detached "
            "feature-reader promotion path."
        ),
    }


def build_rows(split: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim_index, (claim, value, trusted_source, irrelevant_fact) in enumerate(claim_bank(split)):
        for feature_index, pair_feature in enumerate(PAIR_FEATURES):
            positive_world, negative_world, intervention = worlds_for_feature(
                value=value,
                trusted_source=trusted_source,
                irrelevant_fact=irrelevant_fact,
                pair_feature=pair_feature,
            )
            rows.append(
                make_row(
                    row_id=f"{prefix}_{claim_index:02d}_{pair_feature}",
                    split=split,
                    claim=claim,
                    pair_feature=pair_feature,
                    positive_world=positive_world,
                    negative_world=negative_world,
                    intervention=intervention,
                    original_is_positive=(claim_index + feature_index) % 2 == 0,
                )
            )
    return rows


def counterfactual_answer_attractor_rows() -> list[dict[str, Any]]:
    return build_rows("train", "stage101x_train")


def counterfactual_answer_attractor_heldout_rows() -> list[dict[str, Any]]:
    return build_rows("heldout", "stage101x_heldout")


def counterfactual_answer_attractor_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "pair_feature_counts": dict(Counter(str(row["pair_feature"]) for row in rows)),
        "original_answer_counts": dict(Counter(str(row["original_answer"]) for row in rows)),
        "counterfactual_answer_counts": dict(
            Counter(str(row["counterfactual_answer"]) for row in rows)
        ),
        "uses_feature_targets": any(
            "world_a_targets" in row or "world_b_targets" in row or "feature_targets" in row
            for row in rows
        ),
        "plain_language_read": (
            "The row hides feature labels. The only promoted signal is whether "
            "the normal answer path flips under a minimal counterfactual."
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = counterfactual_answer_attractor_rows()
    heldout_rows = counterfactual_answer_attractor_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), heldout_rows)
    report = {
        "decision": "built_stage101x_counterfactual_answer_attractor_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_contract": counterfactual_answer_attractor_contract(train_rows),
        "heldout_contract": counterfactual_answer_attractor_contract(heldout_rows),
        "plain_language_read": (
            "Stage101X changes the route: same LM head, real world versus "
            "imagined counterfactual, no promoted detached feature reader."
        ),
    }
    if str(args.report_out):
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-out",
        default="data/eval/stage101x_counterfactual_answer_attractor_train_probe.jsonl",
    )
    parser.add_argument(
        "--eval-out",
        default="data/eval/stage101x_counterfactual_answer_attractor_heldout_probe.jsonl",
    )
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    report = run_build(build_arg_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
