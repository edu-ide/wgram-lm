#!/usr/bin/env python3
"""Build Stage101W5 counterfactual-repair answer-attractor probes.

W4 still framed the bottleneck as cause-card classification. W5 changes the
exam: the model must choose the smallest intervention that would make a valid
answer possible.

Plain language:
  not "which label is missing?"
  but "what one repair would let you answer responsibly?"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage101w5_counterfactual_repair_probe"

ANSWERABLE_CHOICES = [" yes", " no"]
REPAIR_CHOICES = [
    " answer_now",
    " verify_source",
    " add_relevant_evidence",
    " add_missing_detail",
    " resolve_conflict",
]
PERMISSION_CHOICES = [" yes", " no"]
CHAIN_STEPS = ["answerable_now", "minimal_repair", "repaired_answer_permission"]


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def make_case(
    *,
    case_id: str,
    claim: str,
    source_status: str,
    evidence_payload: str,
    answerable_now: str,
    minimal_repair: str,
    case_type: str,
) -> dict[str, Any]:
    if answerable_now not in ANSWERABLE_CHOICES:
        raise ValueError(f"bad answerable_now {answerable_now!r}")
    if minimal_repair not in REPAIR_CHOICES:
        raise ValueError(f"bad minimal_repair {minimal_repair!r}")
    if answerable_now == " yes" and minimal_repair != " answer_now":
        raise ValueError(f"answerable case {case_id} must use answer_now")
    if answerable_now == " no" and minimal_repair == " answer_now":
        raise ValueError(f"blocked case {case_id} needs a real repair")
    return {
        "case_id": case_id,
        "claim": claim,
        "source_status": source_status,
        "evidence_payload": evidence_payload,
        "answerable_now": answerable_now,
        "minimal_repair": minimal_repair,
        "case_type": case_type,
    }


def prompt_body(case: dict[str, Any]) -> str:
    return (
        f"Claim: {case['claim']}\n"
        f"Source: {case['source_status']}\n"
        f"Evidence: {case['evidence_payload']}"
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
        "task": f"stage101w5_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negatives[0],
        "candidate_answers": list(choices),
        "negative_answers": negatives,
        "plain_language_axis": (
            "Stage101W5 trains counterfactual repair: choose the smallest "
            "intervention that turns an invalid answer state into a valid one."
        ),
        "stage101w5_chain_step": step,
        "stage101w5_counterfactual_repair_required": True,
        "repair_case_id": case["case_id"],
        "source_case_type": case["case_type"],
        "source_claim": case["claim"],
        "source_status": case["source_status"],
        "evidence_payload": case["evidence_payload"],
        "answerable_now": case["answerable_now"],
        "minimal_repair": case["minimal_repair"],
        "split": split,
    }


def rows_for_case(case: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    body = prompt_body(case)
    case_id = str(case["case_id"])
    minimal_repair = str(case["minimal_repair"]).strip()
    return [
        make_row(
            f"{case_id}_answerable_now",
            split=split,
            case=case,
            step="answerable_now",
            prompt=f"{body}\nQ: Can answer now? yes or no.\nA:",
            answer=str(case["answerable_now"]),
            choices=ANSWERABLE_CHOICES,
        ),
        make_row(
            f"{case_id}_minimal_repair",
            split=split,
            case=case,
            step="minimal_repair",
            prompt=(
                f"{body}\n"
                f"answerable_now={str(case['answerable_now']).strip()}\n"
                "Q: What minimal repair makes the answer valid? answer_now, "
                "verify_source, add_relevant_evidence, add_missing_detail, "
                "or resolve_conflict.\nA:"
            ),
            answer=str(case["minimal_repair"]),
            choices=REPAIR_CHOICES,
        ),
        make_row(
            f"{case_id}_repaired_answer_permission",
            split=split,
            case=case,
            step="repaired_answer_permission",
            prompt=(
                f"{body}\n"
                f"minimal_repair={minimal_repair}\n"
                "Q: After that repair, can the model answer? yes or no.\nA:"
            ),
            answer=" yes",
            choices=PERMISSION_CHOICES,
        ),
    ]


def _case(
    case_id: str,
    claim: str,
    source: str,
    evidence: str,
    answerable: str,
    repair: str,
    case_type: str,
) -> dict[str, Any]:
    return make_case(
        case_id=case_id,
        claim=claim,
        source_status=source,
        evidence_payload=evidence,
        answerable_now=answerable,
        minimal_repair=repair,
        case_type=case_type,
    )


def train_cases() -> list[dict[str, Any]]:
    return [
        _case("stage101w5_train_00_parcel_ok", "parcel arrived.", "tracking", "delivered.", " yes", " answer_now", "answer_now"),
        _case("stage101w5_train_01_refund_ok", "refund posted.", "bank", "posted.", " yes", " answer_now", "answer_now"),
        _case("stage101w5_train_02_platform_rumor", "platform is 4.", "rumor", "says 4.", " no", " verify_source", "verify_source"),
        _case("stage101w5_train_03_ticket_rumor", "ticket valid.", "anonymous note", "says valid.", " no", " verify_source", "verify_source"),
        _case("stage101w5_train_04_battery_irrelevant", "battery is full.", "manual", "case is red.", " no", " add_relevant_evidence", "add_relevant_evidence"),
        _case("stage101w5_train_05_alert_irrelevant", "alert active.", "bulletin", "yesterday was calm.", " no", " add_relevant_evidence", "add_relevant_evidence"),
        _case("stage101w5_train_06_code_missing", "room code is 9214.", "notice", "code exists, digits missing.", " no", " add_missing_detail", "add_missing_detail"),
        _case("stage101w5_train_07_locker_missing", "locker code is 3170.", "memo", "code exists, digits missing.", " no", " add_missing_detail", "add_missing_detail"),
        _case("stage101w5_train_08_clinic_conflict", "clinic open.", "two notices", "one open; one closed.", " no", " resolve_conflict", "resolve_conflict"),
        _case("stage101w5_train_09_gate_conflict", "gate open.", "two signs", "one open; one closed.", " no", " resolve_conflict", "resolve_conflict"),
    ]


def heldout_cases() -> list[dict[str, Any]]:
    return [
        _case("stage101w5_heldout_00_invoice_ok", "invoice paid.", "ledger", "paid.", " yes", " answer_now", "answer_now"),
        _case("stage101w5_heldout_01_event_ok", "event starts noon.", "schedule", "lists noon.", " yes", " answer_now", "answer_now"),
        _case("stage101w5_heldout_02_badge_rumor", "badge valid.", "rumor", "says valid.", " no", " verify_source", "verify_source"),
        _case("stage101w5_heldout_03_route_rumor", "route changed.", "chat", "says changed.", " no", " verify_source", "verify_source"),
        _case("stage101w5_heldout_04_storm_irrelevant", "storm active.", "bulletin", "yesterday was warm.", " no", " add_relevant_evidence", "add_relevant_evidence"),
        _case("stage101w5_heldout_05_door_irrelevant", "door is locked.", "manual", "hinge is steel.", " no", " add_relevant_evidence", "add_relevant_evidence"),
        _case("stage101w5_heldout_06_pin_missing", "pin is 7401.", "notice", "pin exists, digits missing.", " no", " add_missing_detail", "add_missing_detail"),
        _case("stage101w5_heldout_07_bus_missing", "bus bay is C.", "board", "bay assigned, letter missing.", " no", " add_missing_detail", "add_missing_detail"),
        _case("stage101w5_heldout_08_store_conflict", "store open.", "two posts", "one open; one closed.", " no", " resolve_conflict", "resolve_conflict"),
        _case("stage101w5_heldout_09_lift_conflict", "lift running.", "two screens", "one running; one stopped.", " no", " resolve_conflict", "resolve_conflict"),
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


def counterfactual_repair_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in train_cases():
        rows.extend(rows_for_case(case, split="train"))
    return dedupe_by_id(rows)


def counterfactual_repair_heldout_rows() -> list[dict[str, Any]]:
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


def counterfactual_repair_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repair_rows = [row for row in rows if row["stage101w5_chain_step"] == "minimal_repair"]
    answerable_rows = [row for row in rows if row["stage101w5_chain_step"] == "answerable_now"]
    return {
        "chain_steps": list(CHAIN_STEPS),
        "repair_choices": list(REPAIR_CHOICES),
        "repair_counts": dict(Counter(str(row["intelligence_answer"]).strip() for row in repair_rows)),
        "answerable_counts": dict(Counter(str(row["intelligence_answer"]).strip() for row in answerable_rows)),
        "plain_language_read": (
            "W5 turns causal doubt into a useful next action: answer now only "
            "when the state is valid, otherwise pick the minimal repair."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = counterfactual_repair_rows()
    eval_rows = counterfactual_repair_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w5_counterfactual_repair_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "counterfactual_repair_contract": counterfactual_repair_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W5 stops asking the model to name abstract failure cards. "
            "It asks for the smallest intervention that would make the answer valid."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w5_counterfactual_repair_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w5_counterfactual_repair_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
