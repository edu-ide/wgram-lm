#!/usr/bin/env python3
"""Build Stage102C randomized-trust source ledger probes.

Stage102B proved that a provenance graph register can route source identity
into the same LM head.  Its ablation exposed a shortcut: S1 was always the
verified source.  Stage102C breaks that shortcut by randomizing which source is
verified while keeping yes/no and original/counterfactual pairs balanced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage102c_randomized_trust_ledger_probe"
CHOICES = [" yes", " no"]
VARIANTS = ["plain", "quoted", "timestamped", "audit_line"]


def claim_bank(split: str) -> list[tuple[str, str]]:
    if split == "train":
        return [
            ("parcel arrived.", "delivered"),
            ("refund posted.", "posted"),
            ("event starts noon.", "noon"),
            ("invoice paid.", "paid"),
            ("platform is 4.", "platform 4"),
            ("ticket valid.", "valid"),
            ("battery is full.", "full"),
            ("alert active.", "active"),
        ]
    return [
        ("badge valid.", "valid"),
        ("route changed.", "changed"),
        ("door locked.", "locked"),
        ("bus bay is C.", "bay C"),
        ("storm active.", "active"),
        ("pin is 7401.", "7401"),
        ("store open.", "open"),
        ("lift running.", "running"),
    ]


def other_source(source_id: str) -> str:
    if source_id == "S1":
        return "S2"
    if source_id == "S2":
        return "S1"
    raise ValueError(f"bad source id: {source_id!r}")


def ledger(variant: str, *, verified_source: str) -> str:
    first_order = ["S1", "S2"] if variant in {"plain", "timestamped"} else ["S2", "S1"]
    statuses = {
        verified_source: "verified",
        other_source(verified_source): "unverified",
    }
    lines = ["Source ledger:"]
    for source_id in first_order:
        lines.append(f"{source_id} = {statuses[source_id]}")
    return "\n".join(lines)


def prompt(
    claim: str,
    label: str,
    source_id: str,
    value: str,
    variant: str,
    *,
    verified_source: str,
) -> str:
    extra = {
        "plain": "",
        "quoted": "Evidence style: quoted\n",
        "timestamped": "Evidence time: latest\n",
        "audit_line": "Evidence style: audit\n",
    }[variant]
    return (
        f"{ledger(variant, verified_source=verified_source)}\n"
        f"Claim: {claim}\n"
        f"{label}:\n"
        f"Evidence source: {source_id}\n"
        f"{extra}"
        f"Evidence value: {value}\n"
        "Q: Can answer now? yes or no.\n"
        "A:"
    )


def make_row(
    *,
    row_id: str,
    split: str,
    claim: str,
    value: str,
    variant: str,
    original_source: str,
    verified_source: str,
) -> dict[str, Any]:
    counterfactual_source = other_source(original_source)
    original_answer = " yes" if original_source == verified_source else " no"
    counterfactual_answer = " yes" if counterfactual_source == verified_source else " no"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "stage102c_randomized_trust_ledger_answer_attractor",
        "stage102c_randomized_trust_ledger_required": True,
        "split": split,
        "claim": claim,
        "pair_feature": "dynamic_source_reliability",
        "variant": variant,
        "verified_source": verified_source,
        "unverified_source": other_source(verified_source),
        "original_source": original_source,
        "counterfactual_source": counterfactual_source,
        "original_world": f"source={original_source}; value={value}; verified_source={verified_source}",
        "counterfactual_world": (
            f"source={counterfactual_source}; value={value}; verified_source={verified_source}"
        ),
        "original_prompt": prompt(
            claim,
            "Real world",
            original_source,
            value,
            variant,
            verified_source=verified_source,
        ),
        "counterfactual_prompt": prompt(
            claim,
            "Imagined change",
            counterfactual_source,
            value,
            variant,
            verified_source=verified_source,
        ),
        "original_answer": original_answer,
        "counterfactual_answer": counterfactual_answer,
        "candidate_answers": list(CHOICES),
        "original_negative_answers": [choice for choice in CHOICES if choice != original_answer],
        "counterfactual_negative_answers": [
            choice for choice in CHOICES if choice != counterfactual_answer
        ],
        "plain_language_axis": (
            "Stage102C prevents the model from trusting the name S1 itself. "
            "The verified source role changes by row, so the model must read "
            "the ledger edge before answering through the same LM head."
        ),
    }


def build_rows(split: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim_index, (claim, value) in enumerate(claim_bank(split)):
        for variant_index, variant in enumerate(VARIANTS):
            for trust_index, verified_source in enumerate(("S1", "S2")):
                original_source = "S1" if (claim_index + variant_index + trust_index) % 2 == 0 else "S2"
                rows.append(
                    make_row(
                        row_id=f"{prefix}_{claim_index:02d}_{variant}_trust{verified_source}",
                        split=split,
                        claim=claim,
                        value=value,
                        variant=variant,
                        original_source=original_source,
                        verified_source=verified_source,
                    )
                )
    return rows


def randomized_trust_ledger_rows() -> list[dict[str, Any]]:
    return build_rows("train", "stage102c_train")


def randomized_trust_ledger_heldout_rows() -> list[dict[str, Any]]:
    return build_rows("heldout", "stage102c_heldout")


def randomized_trust_ledger_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "pair_feature_counts": dict(Counter(str(row["pair_feature"]) for row in rows)),
        "verified_source_counts": dict(Counter(str(row["verified_source"]) for row in rows)),
        "original_source_counts": dict(Counter(str(row["original_source"]) for row in rows)),
        "counterfactual_source_counts": dict(
            Counter(str(row["counterfactual_source"]) for row in rows)
        ),
        "original_answer_counts": dict(Counter(str(row["original_answer"]) for row in rows)),
        "counterfactual_answer_counts": dict(
            Counter(str(row["counterfactual_answer"]) for row in rows)
        ),
        "plain_language_read": (
            "The source name and the verified role are no longer equivalent. "
            "A trust-edge ablation should now matter."
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = randomized_trust_ledger_rows()
    heldout_rows = randomized_trust_ledger_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), heldout_rows)
    report = {
        "decision": "built_stage102c_randomized_trust_ledger_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_contract": randomized_trust_ledger_contract(train_rows),
        "heldout_contract": randomized_trust_ledger_contract(heldout_rows),
    }
    if str(args.report_out):
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage102c_randomized_trust_ledger_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage102c_randomized_trust_ledger_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    print(json.dumps(run_build(build_arg_parser().parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
