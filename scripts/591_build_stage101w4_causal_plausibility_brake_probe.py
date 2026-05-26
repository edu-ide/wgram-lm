#!/usr/bin/env python3
"""Build Stage101W4 causal-plausibility curiosity-brake probes.

Stage101W3 taught cause cards before answer permission, but heldout rows still
failed when the model should notice that a card chain is impossible. W4 adds a
small falsification gate:

  read evidence -> infer real cards -> inspect proposed cards -> reject
  impossible card chains before answer permission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101w4_causal_plausibility_brake_probe"

SOURCE_TRUST_CHOICES = [" trusted", " untrusted"]
RELEVANCE_CHOICES = [" relevant", " irrelevant", " partial"]
SUFFICIENCY_CHOICES = [" enough", " missing"]
CONFLICT_CHOICES = [" clear", " conflict"]
PLAUSIBILITY_CHOICES = [" plausible", " impossible"]
IMPOSSIBLE_CARD_CHOICES = [" none", " source", " relevance", " detail", " conflict"]
PERMISSION_CHOICES = [" yes", " no"]
MISSING_MATERIAL_CHOICES = [" none", " source", " relevance", " detail", " conflict"]

CHAIN_STEPS = [
    "source_trust",
    "evidence_relevance",
    "detail_sufficiency",
    "conflict_status",
    "card_plausibility",
    "impossible_card",
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
    proposed_source_trust: str,
    proposed_evidence_relevance: str,
    proposed_detail_sufficiency: str,
    proposed_conflict_status: str,
    card_plausibility: str,
    impossible_card: str,
    answer_permission: str,
    missing_material: str,
    case_type: str,
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
        "proposed_source_trust": proposed_source_trust,
        "proposed_evidence_relevance": proposed_evidence_relevance,
        "proposed_detail_sufficiency": proposed_detail_sufficiency,
        "proposed_conflict_status": proposed_conflict_status,
        "card_plausibility": card_plausibility,
        "impossible_card": impossible_card,
        "answer_permission": answer_permission,
        "missing_material": missing_material,
        "case_type": case_type,
    }


def prompt_body(case: dict[str, Any]) -> str:
    return (
        f"Claim: {case['claim']}\n"
        f"Source: {case['source_status']}\n"
        f"Evidence: {case['evidence_payload']}"
    )


def proposed_cards(case: dict[str, Any]) -> str:
    return (
        "Proposed cause cards:\n"
        f"source_trust={str(case['proposed_source_trust']).strip()}\n"
        f"evidence_relevance={str(case['proposed_evidence_relevance']).strip()}\n"
        f"detail_sufficiency={str(case['proposed_detail_sufficiency']).strip()}\n"
        f"conflict_status={str(case['proposed_conflict_status']).strip()}"
    )


def short_chain(case: dict[str, Any]) -> str:
    return (
        f"card_plausibility={str(case['card_plausibility']).strip()}\n"
        f"impossible_card={str(case['impossible_card']).strip()}"
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
        "task": f"stage101w4_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negatives[0],
        "candidate_answers": list(choices),
        "negative_answers": negatives,
        "plain_language_axis": (
            "Stage101W4 teaches causal plausibility: cause cards must match "
            "the observed source and evidence before answer permission."
        ),
        "stage101w4_chain_step": step,
        "stage101w4_causal_plausibility_required": True,
        "cause_card_case_id": case["case_id"],
        "source_case_type": case["case_type"],
        "source_claim": case["claim"],
        "source_status": case["source_status"],
        "evidence_payload": case["evidence_payload"],
        "source_trust_answer": case["source_trust"],
        "evidence_relevance_answer": case["evidence_relevance"],
        "detail_sufficiency_answer": case["detail_sufficiency"],
        "conflict_status_answer": case["conflict_status"],
        "proposed_source_trust": case["proposed_source_trust"],
        "proposed_evidence_relevance": case["proposed_evidence_relevance"],
        "proposed_detail_sufficiency": case["proposed_detail_sufficiency"],
        "proposed_conflict_status": case["proposed_conflict_status"],
        "card_plausibility": case["card_plausibility"],
        "impossible_card": case["impossible_card"],
        "answer_permission": case["answer_permission"],
        "missing_material": case["missing_material"],
        "split": split,
    }


def rows_for_case(case: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    body = prompt_body(case)
    cards = proposed_cards(case)
    chain = short_chain(case)
    case_id = str(case["case_id"])
    return [
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
            f"{case_id}_card_plausibility",
            split=split,
            case=case,
            step="card_plausibility",
            prompt=f"{body}\n{cards}\nQ: Are these cause cards plausible? plausible or impossible.\nA:",
            answer=str(case["card_plausibility"]),
            choices=PLAUSIBILITY_CHOICES,
        ),
        make_row(
            f"{case_id}_impossible_card",
            split=split,
            case=case,
            step="impossible_card",
            prompt=f"{body}\n{cards}\ncard_plausibility={str(case['card_plausibility']).strip()}\nQ: Impossible card? none, source, relevance, detail, or conflict.\nA:",
            answer=str(case["impossible_card"]),
            choices=IMPOSSIBLE_CARD_CHOICES,
        ),
        make_row(
            f"{case_id}_answer_permission",
            split=split,
            case=case,
            step="answer_permission",
            prompt=f"{body}\n{cards}\n{chain}\nQ: Can the model answer now? yes or no.\nA:",
            answer=str(case["answer_permission"]),
            choices=PERMISSION_CHOICES,
        ),
        make_row(
            f"{case_id}_missing_material",
            split=split,
            case=case,
            step="missing_material",
            prompt=f"{body}\n{cards}\n{chain}\nQ: Missing material? none, source, relevance, detail, or conflict.\nA:",
            answer=str(case["missing_material"]),
            choices=MISSING_MATERIAL_CHOICES,
        ),
    ]


def _case(
    case_id: str,
    claim: str,
    source: str,
    evidence: str,
    true_cards: tuple[str, str, str, str],
    proposed_cards_tuple: tuple[str, str, str, str],
    plausibility: str,
    impossible: str,
    permission: str,
    missing: str,
    case_type: str,
) -> dict[str, Any]:
    return make_case(
        case_id=case_id,
        claim=claim,
        source_status=source,
        evidence_payload=evidence,
        source_trust=true_cards[0],
        evidence_relevance=true_cards[1],
        detail_sufficiency=true_cards[2],
        conflict_status=true_cards[3],
        proposed_source_trust=proposed_cards_tuple[0],
        proposed_evidence_relevance=proposed_cards_tuple[1],
        proposed_detail_sufficiency=proposed_cards_tuple[2],
        proposed_conflict_status=proposed_cards_tuple[3],
        card_plausibility=plausibility,
        impossible_card=impossible,
        answer_permission=permission,
        missing_material=missing,
        case_type=case_type,
    )


def train_cases() -> list[dict[str, Any]]:
    good = (" trusted", " relevant", " enough", " clear")
    untrusted = (" untrusted", " relevant", " enough", " clear")
    irrelevant = (" trusted", " irrelevant", " missing", " clear")
    partial = (" trusted", " partial", " missing", " clear")
    conflict = (" trusted", " relevant", " missing", " conflict")
    return [
        _case("stage101w4_train_00_parcel_ok", "parcel arrived.", "tracking", "tracking says delivered.", good, good, " plausible", " none", " yes", " none", "plausible_enough"),
        _case("stage101w4_train_01_refund_ok", "refund posted.", "bank", "statement says posted.", good, good, " plausible", " none", " yes", " none", "plausible_enough"),
        _case("stage101w4_train_02_rumor_block", "train platform is 4.", "rumor", "message says platform 4.", untrusted, untrusted, " plausible", " none", " no", " source", "plausible_untrusted"),
        _case("stage101w4_train_03_note_irrelevant", "battery is full.", "manual", "note only says case is red.", irrelevant, irrelevant, " plausible", " none", " no", " relevance", "plausible_irrelevant"),
        _case("stage101w4_train_04_code_detail", "room code is 9214.", "notice", "notice says code exists, digits missing.", partial, partial, " plausible", " none", " no", " detail", "plausible_detail_missing"),
        _case("stage101w4_train_05_gate_conflict", "gate is open.", "two signs", "one says open; one says closed.", conflict, conflict, " plausible", " none", " no", " conflict", "plausible_conflict"),
        _case("stage101w4_train_06_bad_source_card", "ticket valid.", "rumor", "scanner says valid.", untrusted, good, " impossible", " source", " no", " source", "impossible_source"),
        _case("stage101w4_train_07_bad_relevance_card", "storm active.", "bulletin", "note says yesterday was warm.", irrelevant, good, " impossible", " relevance", " no", " relevance", "impossible_relevance"),
        _case("stage101w4_train_08_bad_detail_card", "room code is 4821.", "notice", "notice says code exists, digits missing.", partial, good, " impossible", " detail", " no", " detail", "impossible_detail"),
        _case("stage101w4_train_09_bad_conflict_card", "clinic open.", "two notices", "one says open; one says closed.", conflict, good, " impossible", " conflict", " no", " conflict", "impossible_conflict"),
    ]


def heldout_cases() -> list[dict[str, Any]]:
    good = (" trusted", " relevant", " enough", " clear")
    untrusted = (" untrusted", " relevant", " enough", " clear")
    irrelevant = (" trusted", " irrelevant", " missing", " clear")
    partial = (" trusted", " partial", " missing", " clear")
    conflict = (" trusted", " relevant", " missing", " conflict")
    return [
        _case("stage101w4_heldout_00_invoice_ok", "invoice paid.", "ledger", "ledger says paid.", good, good, " plausible", " none", " yes", " none", "plausible_enough"),
        _case("stage101w4_heldout_01_event_ok", "event starts noon.", "schedule", "schedule lists noon.", good, good, " plausible", " none", " yes", " none", "plausible_enough"),
        _case("stage101w4_heldout_02_untrusted", "train platform is 8.", "rumor", "board says platform 8.", untrusted, untrusted, " plausible", " none", " no", " source", "plausible_untrusted"),
        _case("stage101w4_heldout_03_irrelevant", "alert active.", "bulletin", "note says yesterday was calm.", irrelevant, irrelevant, " plausible", " none", " no", " relevance", "plausible_irrelevant"),
        _case("stage101w4_heldout_04_detail", "locker code is 3170.", "memo", "memo says code exists, digits missing.", partial, partial, " plausible", " none", " no", " detail", "plausible_detail_missing"),
        _case("stage101w4_heldout_05_conflict", "clinic open.", "two notices", "one says open; one says closed.", conflict, conflict, " plausible", " none", " no", " conflict", "plausible_conflict"),
        _case("stage101w4_heldout_06_bad_source", "badge valid.", "anonymous note", "scanner says valid.", untrusted, good, " impossible", " source", " no", " source", "impossible_source"),
        _case("stage101w4_heldout_07_bad_relevance", "storm warning active.", "bulletin", "note says yesterday was warm.", irrelevant, good, " impossible", " relevance", " no", " relevance", "impossible_relevance"),
        _case("stage101w4_heldout_08_bad_detail", "room code is 7401.", "notice", "notice says code exists, digits missing.", partial, good, " impossible", " detail", " no", " detail", "impossible_detail"),
        _case("stage101w4_heldout_09_bad_conflict", "gate open.", "two signs", "one says open; one says closed.", conflict, good, " impossible", " conflict", " no", " conflict", "impossible_conflict"),
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


def causal_plausibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in train_cases():
        rows.extend(rows_for_case(case, split="train"))
    return dedupe_by_id(rows)


def causal_plausibility_heldout_rows() -> list[dict[str, Any]]:
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


def causal_plausibility_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plausibility_rows = [row for row in rows if row["stage101w4_chain_step"] == "card_plausibility"]
    impossible_rows = [row for row in rows if row["stage101w4_chain_step"] == "impossible_card"]
    permission_rows = [row for row in rows if row["stage101w4_chain_step"] == "answer_permission"]
    return {
        "chain_steps": list(CHAIN_STEPS),
        "plausible_rows": sum(1 for row in plausibility_rows if row["intelligence_answer"] == " plausible"),
        "impossible_rows": sum(1 for row in plausibility_rows if row["intelligence_answer"] == " impossible"),
        "impossible_card_types": sorted({str(row["intelligence_answer"]).strip() for row in impossible_rows}),
        "impossible_permission_yes_rows": sum(
            1
            for row in permission_rows
            if row["card_plausibility"] == " impossible" and row["intelligence_answer"] == " yes"
        ),
        "plain_language_read": (
            "W4 teaches the model to distrust cause cards that do not match the "
            "source and evidence before it uses those cards for answer permission."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = causal_plausibility_rows()
    eval_rows = causal_plausibility_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w4_causal_plausibility_brake_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "causal_plausibility_contract": causal_plausibility_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W4 adds a plausibility check between cause cards and answer "
            "permission, so an impossible card chain cannot silently authorize an answer."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w4_causal_plausibility_brake_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w4_causal_plausibility_brake_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
