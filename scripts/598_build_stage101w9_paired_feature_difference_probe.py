#!/usr/bin/env python3
"""Build Stage101W9 paired latent feature-difference probes.

W8 showed that answer permission is easier than clean feature decomposition.
W9 changes the data contract: each row is a pair of worlds sharing one claim
where exactly one causal feature changes. The trainer can then learn a pairwise
latent margin instead of memorizing independent labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage101w9_paired_feature_difference_probe"

CAUSAL_FEATURE_NAMES = [
    "source_reliability",
    "evidence_relevance",
    "detail_sufficiency",
    "conflict_status",
]
FEATURE_NAMES = CAUSAL_FEATURE_NAMES + ["answer_permission"]
FEATURE_CHOICES = {
    "source_reliability": ["trusted", "untrusted"],
    "evidence_relevance": ["relevant", "irrelevant"],
    "detail_sufficiency": ["enough", "missing"],
    "conflict_status": ["clear", "conflict"],
    "answer_permission": ["yes", "no"],
}


def world_prompt(claim: str, world: str) -> str:
    return f"Claim: {claim}\nWorld: {world}\nQ: Can answer now? yes or no.\nA:"


def positive_targets() -> dict[str, str]:
    return {
        "source_reliability": "trusted",
        "evidence_relevance": "relevant",
        "detail_sufficiency": "enough",
        "conflict_status": "clear",
        "answer_permission": "yes",
    }


def negative_targets(pair_feature: str) -> dict[str, str]:
    targets = positive_targets()
    if pair_feature == "source_reliability":
        targets["source_reliability"] = "untrusted"
    elif pair_feature == "evidence_relevance":
        targets["evidence_relevance"] = "irrelevant"
    elif pair_feature == "detail_sufficiency":
        targets["detail_sufficiency"] = "missing"
    elif pair_feature == "conflict_status":
        targets["conflict_status"] = "conflict"
    else:
        raise ValueError(f"bad pair_feature: {pair_feature!r}")
    targets["answer_permission"] = "no"
    return targets


def validate_targets(targets: dict[str, str]) -> None:
    for name in FEATURE_NAMES:
        value = str(targets[name])
        if value not in FEATURE_CHOICES[name]:
            raise ValueError(f"bad target {name}={value!r}")


def make_pair(
    *,
    pair_id: str,
    claim: str,
    positive_world: str,
    negative_world: str,
    pair_feature: str,
    positive_position: str,
    split: str,
) -> dict[str, Any]:
    if pair_feature not in CAUSAL_FEATURE_NAMES:
        raise ValueError(f"bad pair_feature {pair_feature!r}")
    if positive_position not in {"A", "B"}:
        raise ValueError(f"bad positive_position {positive_position!r}")
    pos_targets = positive_targets()
    neg_targets = negative_targets(pair_feature)
    validate_targets(pos_targets)
    validate_targets(neg_targets)
    if positive_position == "A":
        world_a = positive_world
        world_b = negative_world
        targets_a = pos_targets
        targets_b = neg_targets
    else:
        world_a = negative_world
        world_b = positive_world
        targets_a = neg_targets
        targets_b = pos_targets
    return {
        "id": pair_id,
        "source": SOURCE,
        "task": "stage101w9_paired_feature_difference",
        "stage101w9_paired_feature_difference_required": True,
        "pair_feature": pair_feature,
        "positive_world": positive_position,
        "negative_world": "B" if positive_position == "A" else "A",
        "claim_a": claim,
        "claim_b": claim,
        "world_a": world_a,
        "world_b": world_b,
        "world_a_prompt": world_prompt(claim, world_a),
        "world_b_prompt": world_prompt(claim, world_b),
        "world_a_targets": targets_a,
        "world_b_targets": targets_b,
        "split": split,
        "plain_language_axis": (
            "Stage101W9 trains latent feature differences by comparing two worlds "
            "where exactly one causal feature changes."
        ),
    }


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
    claim: str,
    value: str,
    trusted_source: str,
    irrelevant_fact: str,
    pair_feature: str,
) -> tuple[str, str]:
    if pair_feature == "source_reliability":
        return f"{trusted_source} says {value}.", f"rumor says {value}."
    if pair_feature == "evidence_relevance":
        return f"{trusted_source} says {value}.", f"{trusted_source} only says {irrelevant_fact}."
    if pair_feature == "detail_sufficiency":
        return f"{trusted_source} says {value}.", f"{trusted_source} says value exists, exact detail missing."
    if pair_feature == "conflict_status":
        return f"{trusted_source} says {value}.", f"one trusted note says {value}; another says not {value}."
    raise ValueError(f"bad pair_feature {pair_feature!r}")


def build_pairs(split: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim_index, (claim, value, trusted_source, irrelevant_fact) in enumerate(claim_bank(split)):
        for feature_index, pair_feature in enumerate(CAUSAL_FEATURE_NAMES):
            positive_world, negative_world = worlds_for_feature(
                claim=claim,
                value=value,
                trusted_source=trusted_source,
                irrelevant_fact=irrelevant_fact,
                pair_feature=pair_feature,
            )
            positive_position = "A" if (claim_index + feature_index) % 2 == 0 else "B"
            rows.append(
                make_pair(
                    pair_id=f"{prefix}_{claim_index:02d}_{pair_feature}",
                    claim=claim,
                    positive_world=positive_world,
                    negative_world=negative_world,
                    pair_feature=pair_feature,
                    positive_position=positive_position,
                    split=split,
                )
            )
    return rows


def paired_feature_difference_rows() -> list[dict[str, Any]]:
    return build_pairs("train", "stage101w9_train")


def paired_feature_difference_heldout_rows() -> list[dict[str, Any]]:
    return build_pairs("heldout", "stage101w9_heldout")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def pair_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "feature_counts": dict(Counter(str(row["pair_feature"]) for row in rows)),
        "positive_world_counts": dict(Counter(str(row["positive_world"]) for row in rows)),
    }


def paired_feature_difference_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "causal_feature_names": list(CAUSAL_FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_choices": dict(FEATURE_CHOICES),
        **pair_counts(rows),
        "plain_language_read": (
            "W9 forces one-cause contrast: two worlds share a claim, and exactly "
            "one causal feature changes before permission flips."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = paired_feature_difference_rows()
    eval_rows = paired_feature_difference_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w9_paired_feature_difference_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "paired_feature_difference_contract": paired_feature_difference_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W9 moves from independent feature labels to paired latent "
            "feature differences."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w9_paired_feature_difference_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w9_paired_feature_difference_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
