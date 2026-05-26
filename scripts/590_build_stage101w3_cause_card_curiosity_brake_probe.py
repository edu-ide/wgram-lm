#!/usr/bin/env python3
"""Build Stage101W3 cause-card curiosity-brake probes.

Stage101W2 fixed label skew, but the model still answered "yes" on untrusted,
irrelevant, partial, and conflicted heldout rows.  W3 teaches the causes first:

  source_trust -> evidence_relevance -> detail_sufficiency -> conflict_status
  -> answer_permission / missing_material

The goal is to make answer permission depend on read cause cards rather than a
surface habit that says "yes" whenever the evidence text looks answer-shaped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101w3_cause_card_curiosity_brake_probe"

SOURCE_TRUST_CHOICES = [" trusted", " untrusted"]
RELEVANCE_CHOICES = [" relevant", " irrelevant", " partial"]
SUFFICIENCY_CHOICES = [" enough", " missing"]
CONFLICT_CHOICES = [" clear", " conflict"]
PERMISSION_CHOICES = [" yes", " no"]
MISSING_MATERIAL_CHOICES = [" none", " source", " relevance", " detail", " conflict"]

CAUSE_STEPS = [
    "source_trust",
    "evidence_relevance",
    "detail_sufficiency",
    "conflict_status",
    "answer_permission",
    "missing_material",
]


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def make_case(
    *,
    case_id: str,
    claim: str,
    source_status: str,
    evidence_payload: str,
    source_trust: str,
    evidence_relevance: str,
    detail_sufficiency: str,
    conflict_status: str,
    answer_permission: str,
    missing_material: str,
    case_type: str,
    pair_id: str = "",
    include_missing_material: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "claim": claim,
        "source_status": source_status,
        "evidence_payload": evidence_payload,
        "source_trust": source_trust,
        "evidence_relevance": evidence_relevance,
        "detail_sufficiency": detail_sufficiency,
        "conflict_status": conflict_status,
        "answer_permission": answer_permission,
        "missing_material": missing_material,
        "case_type": case_type,
        "pair_id": pair_id,
        "include_missing_material": include_missing_material,
    }


def prompt_body(case: dict[str, Any]) -> str:
    return (
        f"Claim: {case['claim']}\n"
        f"Source: {case['source_status']}\n"
        f"Evidence: {case['evidence_payload']}"
    )


def cause_cards(case: dict[str, Any]) -> str:
    return (
        "Cause cards:\n"
        f"source_trust={str(case['source_trust']).strip()}\n"
        f"evidence_relevance={str(case['evidence_relevance']).strip()}\n"
        f"detail_sufficiency={str(case['detail_sufficiency']).strip()}\n"
        f"conflict_status={str(case['conflict_status']).strip()}"
    )


def make_row(
    row_id: str,
    *,
    split: str,
    case: dict[str, Any],
    step: str,
    prompt: str,
    answer: str,
    choices: list[str],
) -> dict[str, Any]:
    if answer not in choices:
        raise ValueError(f"answer {answer!r} not in choices {choices!r}")
    negatives = negative_answers(answer, choices)
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"stage101w3_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negatives[0],
        "candidate_answers": list(choices),
        "negative_answers": negatives,
        "plain_language_axis": (
            "Stage101W3 teaches cause cards before permission: trust, "
            "relevance, detail sufficiency, and conflict must be read before "
            "the model decides whether it may answer."
        ),
        "stage101w3_chain_step": step,
        "stage101w3_cause_card_required": True,
        "cause_card_case_id": case["case_id"],
        "cause_card_pair_id": case.get("pair_id", ""),
        "source_case_type": case["case_type"],
        "source_claim": case["claim"],
        "source_status": case["source_status"],
        "evidence_payload": case["evidence_payload"],
        "source_trust_answer": case["source_trust"],
        "evidence_relevance_answer": case["evidence_relevance"],
        "detail_sufficiency_answer": case["detail_sufficiency"],
        "conflict_status_answer": case["conflict_status"],
        "answer_permission": case["answer_permission"],
        "missing_material": case["missing_material"],
        "split": split,
    }


def rows_for_case(case: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    body = prompt_body(case)
    cards = cause_cards(case)
    case_id = str(case["case_id"])
    rows = [
        make_row(
            f"{case_id}_source_trust",
            split=split,
            case=case,
            step="source_trust",
            prompt=f"{body}\nQ: Source trust card? trusted or untrusted.\nA:",
            answer=str(case["source_trust"]),
            choices=SOURCE_TRUST_CHOICES,
        ),
        make_row(
            f"{case_id}_evidence_relevance",
            split=split,
            case=case,
            step="evidence_relevance",
            prompt=f"{body}\nQ: Evidence relevance card? relevant, irrelevant, or partial.\nA:",
            answer=str(case["evidence_relevance"]),
            choices=RELEVANCE_CHOICES,
        ),
        make_row(
            f"{case_id}_detail_sufficiency",
            split=split,
            case=case,
            step="detail_sufficiency",
            prompt=f"{body}\nQ: Detail sufficiency card? enough or missing.\nA:",
            answer=str(case["detail_sufficiency"]),
            choices=SUFFICIENCY_CHOICES,
        ),
        make_row(
            f"{case_id}_conflict_status",
            split=split,
            case=case,
            step="conflict_status",
            prompt=f"{body}\nQ: Conflict card? clear or conflict.\nA:",
            answer=str(case["conflict_status"]),
            choices=CONFLICT_CHOICES,
        ),
        make_row(
            f"{case_id}_answer_permission",
            split=split,
            case=case,
            step="answer_permission",
            prompt=f"{cards}\nQ: Can the model answer now? yes or no.\nA:",
            answer=str(case["answer_permission"]),
            choices=PERMISSION_CHOICES,
        ),
    ]
    if bool(case["include_missing_material"]):
        rows.append(
            make_row(
                f"{case_id}_missing_material",
                split=split,
                case=case,
                step="missing_material",
                prompt=f"{cards}\nQ: Missing material? none, source, relevance, detail, or conflict.\nA:",
                answer=str(case["missing_material"]),
                choices=MISSING_MATERIAL_CHOICES,
            )
        )
    return rows


def train_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            case_id="stage101w3_train_00_parcel_enough",
            claim="parcel arrived today.",
            source_status="reliable source",
            evidence_payload="tracking says delivered today.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_parcel",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_train_01_parcel_untrusted",
            claim="parcel arrived today.",
            source_status="untrusted source",
            evidence_payload="tracking says delivered today.",
            source_trust=" untrusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" no",
            missing_material=" source",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_parcel",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_train_02_battery_irrelevant",
            claim="battery is full.",
            source_status="reliable source",
            evidence_payload="note only says case is red.",
            source_trust=" trusted",
            evidence_relevance=" irrelevant",
            detail_sufficiency=" missing",
            conflict_status=" clear",
            answer_permission=" no",
            missing_material=" relevance",
            case_type="irrelevant_evidence",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_train_03_room_detail",
            claim="room code is 9214.",
            source_status="reliable source",
            evidence_payload="message says room code exists, digits missing.",
            source_trust=" trusted",
            evidence_relevance=" partial",
            detail_sufficiency=" missing",
            conflict_status=" clear",
            answer_permission=" no",
            missing_material=" detail",
            case_type="detail_missing",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_train_04_gate_conflict",
            claim="gate is open.",
            source_status="two reliable signs",
            evidence_payload="one says open; one says closed.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" missing",
            conflict_status=" conflict",
            answer_permission=" no",
            missing_material=" conflict",
            case_type="trusted_conflict",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_train_05_refund_enough",
            claim="refund was issued.",
            source_status="bank statement",
            evidence_payload="statement says refund posted.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
        make_case(
            case_id="stage101w3_train_06_meeting_enough",
            claim="meeting is at 9.",
            source_status="official calendar",
            evidence_payload="calendar lists 9.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
        make_case(
            case_id="stage101w3_train_07_weather_enough",
            claim="alert is cancelled.",
            source_status="official weather page",
            evidence_payload="page says alert cancelled.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
    ]


def heldout_cases() -> list[dict[str, Any]]:
    return [
        make_case(
            case_id="stage101w3_heldout_00_train_enough",
            claim="train uses platform 4.",
            source_status="official board",
            evidence_payload="board says platform 4.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_heldout_01_train_untrusted",
            claim="train uses platform 4.",
            source_status="rumor",
            evidence_payload="board says platform 4.",
            source_trust=" untrusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" no",
            missing_material=" source",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_heldout_02_storm_irrelevant",
            claim="storm warning is active.",
            source_status="reliable source",
            evidence_payload="note says yesterday was warm.",
            source_trust=" trusted",
            evidence_relevance=" irrelevant",
            detail_sufficiency=" missing",
            conflict_status=" clear",
            answer_permission=" no",
            missing_material=" relevance",
            case_type="irrelevant_evidence",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_heldout_03_code_detail",
            claim="room code is 4821.",
            source_status="reliable source",
            evidence_payload="message says room code exists, digits missing.",
            source_trust=" trusted",
            evidence_relevance=" partial",
            detail_sufficiency=" missing",
            conflict_status=" clear",
            answer_permission=" no",
            missing_material=" detail",
            case_type="detail_missing",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_heldout_04_clinic_conflict",
            claim="clinic is open.",
            source_status="two reliable notices",
            evidence_payload="one says open; one says closed.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" missing",
            conflict_status=" conflict",
            answer_permission=" no",
            missing_material=" conflict",
            case_type="trusted_conflict",
            include_missing_material=True,
        ),
        make_case(
            case_id="stage101w3_heldout_05_invoice_enough",
            claim="invoice was paid.",
            source_status="account ledger",
            evidence_payload="ledger says paid.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
        make_case(
            case_id="stage101w3_heldout_06_event_enough",
            claim="event starts at noon.",
            source_status="official schedule",
            evidence_payload="schedule lists noon.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
        make_case(
            case_id="stage101w3_heldout_07_ticket_enough",
            claim="ticket is valid.",
            source_status="scanner log",
            evidence_payload="scanner says valid.",
            source_trust=" trusted",
            evidence_relevance=" relevant",
            detail_sufficiency=" enough",
            conflict_status=" clear",
            answer_permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
    ]


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            continue
        seen.add(row_id)
        out.append(row)
    return out


def cause_card_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in train_cases():
        rows.extend(rows_for_case(case, split="train"))
    return dedupe_by_id(rows)


def cause_card_heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in heldout_cases():
        rows.extend(rows_for_case(case, split="heldout"))
    return dedupe_by_id(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def cause_card_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    permission_rows = [row for row in rows if row["stage101w3_chain_step"] == "answer_permission"]
    missing_rows = [row for row in rows if row["stage101w3_chain_step"] == "missing_material"]
    return {
        "chain_steps": list(CAUSE_STEPS),
        "answer_permission_yes_rows": sum(1 for row in permission_rows if row["intelligence_answer"] == " yes"),
        "answer_permission_no_rows": sum(1 for row in permission_rows if row["intelligence_answer"] == " no"),
        "missing_material_types": sorted({str(row["intelligence_answer"]).strip() for row in missing_rows}),
        "source_quality_counterfactual_pairs": sorted(
            {
                str(row["cause_card_pair_id"])
                for row in permission_rows
                if str(row["cause_card_pair_id"]).startswith("source_quality")
            }
        ),
        "plain_language_read": (
            "W3 separates cause reading from permission. Permission rows consume "
            "parent cause cards, so a no answer must be caused by source, "
            "relevance, detail, or conflict."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = cause_card_rows()
    eval_rows = cause_card_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w3_cause_card_curiosity_brake_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "cause_card_contract": cause_card_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W3 teaches source, relevance, detail, and conflict cause "
            "cards before answer permission to reduce evidence-specific wrong "
            "attractors."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w3_cause_card_curiosity_brake_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w3_cause_card_curiosity_brake_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
