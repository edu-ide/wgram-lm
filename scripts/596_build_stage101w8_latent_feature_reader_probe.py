#!/usr/bin/env python3
"""Build Stage101W8 latent feature-reader probes.

W5-W7 showed that new labels, repair words, A/B prompts, and siamese scoring
do not solve the evidence bottleneck. W8 changes the target: the recurrent
state must expose separable latent features for the evidence state before
answer permission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage101w8_latent_feature_reader_probe"

FEATURE_NAMES = [
    "source_reliability",
    "evidence_relevance",
    "detail_sufficiency",
    "conflict_status",
    "answer_permission",
]
FEATURE_CHOICES = {
    "source_reliability": ["trusted", "untrusted"],
    "evidence_relevance": ["relevant", "irrelevant"],
    "detail_sufficiency": ["enough", "missing"],
    "conflict_status": ["clear", "conflict"],
    "answer_permission": ["yes", "no"],
}


def make_world(
    *,
    row_id: str,
    claim: str,
    world: str,
    source_reliability: str,
    evidence_relevance: str,
    detail_sufficiency: str,
    conflict_status: str,
    answer_permission: str,
    axis: str,
    split: str,
) -> dict[str, Any]:
    targets = {
        "source_reliability": source_reliability,
        "evidence_relevance": evidence_relevance,
        "detail_sufficiency": detail_sufficiency,
        "conflict_status": conflict_status,
        "answer_permission": answer_permission,
    }
    for name, value in targets.items():
        if value not in FEATURE_CHOICES[name]:
            raise ValueError(f"bad target {name}={value!r}")
    answer = " yes" if answer_permission == "yes" else " no"
    negative = " no" if answer == " yes" else " yes"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "stage101w8_latent_feature_reader_icl",
        "prompt": f"Claim: {claim}\nWorld: {world}\nQ: Can answer now? yes or no.\nA:",
        "intelligence_answer": answer,
        "parrot_answer": negative,
        "candidate_answers": [" yes", " no"],
        "negative_answers": [negative],
        "plain_language_axis": (
            "Stage101W8 trains the hidden state to expose source/relevance/"
            "detail/conflict features before answer permission."
        ),
        "stage101w8_latent_feature_reader_required": True,
        "feature_targets": targets,
        "source_claim": claim,
        "world": world,
        "repair_axis": axis,
        "split": split,
    }


def _rows(split: str, prefix: str, examples: list[tuple[str, str, str, dict[str, str], str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (claim, world, permission, features, axis) in enumerate(examples):
        rows.append(
            make_world(
                row_id=f"{prefix}_{index:02d}_{axis}_{permission}",
                claim=claim,
                world=world,
                source_reliability=features["source_reliability"],
                evidence_relevance=features["evidence_relevance"],
                detail_sufficiency=features["detail_sufficiency"],
                conflict_status=features["conflict_status"],
                answer_permission=permission,
                axis=axis,
                split=split,
            )
        )
    return rows


def _claim_bank(split: str) -> list[tuple[str, str, str, str]]:
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


def _expanded_examples(split: str) -> list[tuple[str, str, str, dict[str, str], str]]:
    good = {
        "source_reliability": "trusted",
        "evidence_relevance": "relevant",
        "detail_sufficiency": "enough",
        "conflict_status": "clear",
    }
    rows: list[tuple[str, str, str, dict[str, str], str]] = []
    for claim, value, trusted_source, irrelevant_fact in _claim_bank(split):
        rows.extend(
            [
                (
                    claim,
                    f"{trusted_source} says {value}.",
                    "yes",
                    good,
                    "answerable",
                ),
                (
                    claim,
                    f"rumor says {value}.",
                    "no",
                    {**good, "source_reliability": "untrusted"},
                    "source",
                ),
                (
                    claim,
                    f"anonymous chat says {value}.",
                    "no",
                    {**good, "source_reliability": "untrusted"},
                    "source",
                ),
                (
                    claim,
                    f"{trusted_source} only says {irrelevant_fact}.",
                    "no",
                    {**good, "evidence_relevance": "irrelevant", "detail_sufficiency": "missing"},
                    "relevance",
                ),
                (
                    claim,
                    f"{trusted_source} describes formatting, not the claim.",
                    "no",
                    {**good, "evidence_relevance": "irrelevant", "detail_sufficiency": "missing"},
                    "relevance",
                ),
                (
                    claim,
                    f"{trusted_source} says value exists, exact detail missing.",
                    "no",
                    {**good, "detail_sufficiency": "missing"},
                    "detail",
                ),
                (
                    claim,
                    f"one trusted note says {value}; another says not {value}.",
                    "no",
                    {**good, "conflict_status": "conflict"},
                    "conflict",
                ),
                (
                    claim,
                    f"two trusted records disagree about {value}.",
                    "no",
                    {**good, "conflict_status": "conflict"},
                    "conflict",
                ),
            ]
        )
    return rows


def train_examples() -> list[tuple[str, str, str, dict[str, str], str]]:
    return _expanded_examples("train")


def heldout_examples() -> list[tuple[str, str, str, dict[str, str], str]]:
    return _expanded_examples("heldout")


def latent_feature_reader_rows() -> list[dict[str, Any]]:
    return _rows("train", "stage101w8_train", train_examples())


def latent_feature_reader_heldout_rows() -> list[dict[str, Any]]:
    return _rows("heldout", "stage101w8_heldout", heldout_examples())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def feature_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name in FEATURE_NAMES:
        counts[name] = dict(Counter(str(row["feature_targets"][name]) for row in rows))
    return counts


def latent_feature_reader_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_choices": dict(FEATURE_CHOICES),
        "feature_counts": feature_counts(rows),
        "plain_language_read": (
            "W8 asks whether the hidden state can read the evidence state as "
            "separable features before the answer head speaks."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = latent_feature_reader_rows()
    eval_rows = latent_feature_reader_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w8_latent_feature_reader_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "latent_feature_reader_contract": latent_feature_reader_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W8 changes the target from answer-label tricks to internal "
            "feature reading over the evidence state."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w8_latent_feature_reader_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w8_latent_feature_reader_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
