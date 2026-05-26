#!/usr/bin/env python3
"""Build Stage101Z source-binding ledger probes.

Y kept the source id inside prose ("S1 says ...").  Z separates the source key
from the evidence value so the model can test a cleaner key-value binding:
ledger source id -> trust value -> answer permission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage101z_source_binding_ledger_probe"
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


def ledger(variant: str) -> str:
    if variant in {"plain", "timestamped"}:
        return "Source ledger:\nS1 = verified\nS2 = unverified"
    return "Source ledger:\nS2 = unverified\nS1 = verified"


def prompt(claim: str, label: str, source_id: str, value: str, variant: str) -> str:
    extra = {
        "plain": "",
        "quoted": "Evidence style: quoted\n",
        "timestamped": "Evidence time: latest\n",
        "audit_line": "Evidence style: audit\n",
    }[variant]
    return (
        f"{ledger(variant)}\n"
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
) -> dict[str, Any]:
    counterfactual_source = "S2" if original_source == "S1" else "S1"
    original_answer = " yes" if original_source == "S1" else " no"
    counterfactual_answer = " yes" if counterfactual_source == "S1" else " no"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "stage101z_source_binding_ledger_answer_attractor",
        "stage101z_source_binding_ledger_required": True,
        "split": split,
        "claim": claim,
        "pair_feature": "source_reliability",
        "variant": variant,
        "original_source": original_source,
        "counterfactual_source": counterfactual_source,
        "original_world": f"source={original_source}; value={value}",
        "counterfactual_world": f"source={counterfactual_source}; value={value}",
        "original_prompt": prompt(claim, "Real world", original_source, value, variant),
        "counterfactual_prompt": prompt(
            claim,
            "Imagined change",
            counterfactual_source,
            value,
            variant,
        ),
        "original_answer": original_answer,
        "counterfactual_answer": counterfactual_answer,
        "candidate_answers": list(CHOICES),
        "original_negative_answers": [choice for choice in CHOICES if choice != original_answer],
        "counterfactual_negative_answers": [
            choice for choice in CHOICES if choice != counterfactual_answer
        ],
        "plain_language_axis": (
            "Stage101Z makes source trust a key-value binding problem: look up "
            "the evidence source id in the ledger, then answer through the same "
            "LM head."
        ),
    }


def build_rows(split: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim_index, (claim, value) in enumerate(claim_bank(split)):
        for variant_index, variant in enumerate(VARIANTS):
            source_id = "S1" if (claim_index + variant_index) % 2 == 0 else "S2"
            rows.append(
                make_row(
                    row_id=f"{prefix}_{claim_index:02d}_{variant}",
                    split=split,
                    claim=claim,
                    value=value,
                    variant=variant,
                    original_source=source_id,
                )
            )
    return rows


def source_binding_ledger_rows() -> list[dict[str, Any]]:
    return build_rows("train", "stage101z_train")


def source_binding_ledger_heldout_rows() -> list[dict[str, Any]]:
    return build_rows("heldout", "stage101z_heldout")


def source_binding_ledger_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "pair_feature_counts": dict(Counter(str(row["pair_feature"]) for row in rows)),
        "original_answer_counts": dict(Counter(str(row["original_answer"]) for row in rows)),
        "counterfactual_answer_counts": dict(
            Counter(str(row["counterfactual_answer"]) for row in rows)
        ),
        "original_source_counts": dict(Counter(str(row["original_source"]) for row in rows)),
        "counterfactual_source_counts": dict(
            Counter(str(row["counterfactual_source"]) for row in rows)
        ),
        "plain_language_read": (
            "Evidence source and evidence value are separate fields; this tests "
            "source-id binding rather than prose understanding."
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = source_binding_ledger_rows()
    heldout_rows = source_binding_ledger_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), heldout_rows)
    report = {
        "decision": "built_stage101z_source_binding_ledger_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_contract": source_binding_ledger_contract(train_rows),
        "heldout_contract": source_binding_ledger_contract(heldout_rows),
    }
    if str(args.report_out):
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101z_source_binding_ledger_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101z_source_binding_ledger_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    print(json.dumps(run_build(build_arg_parser().parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
