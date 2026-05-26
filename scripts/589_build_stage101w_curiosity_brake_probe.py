#!/usr/bin/env python3
"""Build Stage101W curiosity-brake probes.

Stage101V taught the model to ask for missing evidence, but the heldout gate
showed an over-curiosity failure: trusted and sufficient evidence was still
routed to ask_more.  Stage101W splits the policy into two short decisions:

  answer_permission: yes/no
  missing_material: none/source/relevance/detail/conflict

The intended causal story is simple: curiosity is useful only when the current
evidence is not enough.  If evidence is trusted, relevant, sufficient, and not
conflicted, the model should stop asking and answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101w_curiosity_brake_probe"

PERMISSION_CHOICES = [" yes", " no"]
MISSING_MATERIAL_CHOICES = [" none", " source", " relevance", " detail", " conflict"]


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def make_case(
    *,
    case_id: str,
    claim: str,
    source_status: str,
    evidence_payload: str,
    reliability: str,
    relevance: str,
    sufficiency: str,
    conflict: str,
    permission: str,
    missing_material: str,
    case_type: str,
    pair_id: str = "",
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "claim": claim,
        "source_status": source_status,
        "evidence_payload": evidence_payload,
        "reliability": reliability,
        "relevance": relevance,
        "sufficiency": sufficiency,
        "conflict": conflict,
        "permission": permission,
        "missing_material": missing_material,
        "case_type": case_type,
        "pair_id": pair_id,
    }


def prompt_body(case: dict[str, str]) -> str:
    return (
        f"Claim: {case['claim']}\n"
        f"Source: {case['source_status']}\n"
        f"Evidence: {case['evidence_payload']}\n"
        "Checks: "
        f"trusted={case['reliability']}; "
        f"relevant={case['relevance']}; "
        f"enough={case['sufficiency']}; "
        f"conflict={case['conflict']}"
    )


def make_row(
    row_id: str,
    *,
    split: str,
    case: dict[str, str],
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
        "task": f"stage101w_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negatives[0],
        "candidate_answers": list(choices),
        "negative_answers": negatives,
        "plain_language_axis": (
            "Stage101W separates useful curiosity from over-curiosity: first "
            "decide whether the model is allowed to answer, then name the one "
            "missing material if it is not allowed."
        ),
        "stage101w_chain_step": step,
        "stage101w_curiosity_brake_required": True,
        "curiosity_brake_case_id": case["case_id"],
        "curiosity_brake_pair_id": case.get("pair_id", ""),
        "source_case_type": case["case_type"],
        "source_claim": case["claim"],
        "source_status": case["source_status"],
        "evidence_payload": case["evidence_payload"],
        "source_reliability": case["reliability"],
        "evidence_relevance": case["relevance"],
        "evidence_sufficiency": case["sufficiency"],
        "conflict_status": case["conflict"],
        "answer_permission": case["permission"],
        "missing_material": case["missing_material"],
        "split": split,
    }


def rows_for_case(case: dict[str, str], *, split: str) -> list[dict[str, Any]]:
    body = prompt_body(case)
    case_id = case["case_id"]
    return [
        make_row(
            f"{case_id}_answer_permission",
            split=split,
            case=case,
            step="answer_permission",
            prompt=f"{body}\nQ: Can the model answer now? yes or no.\nA:",
            answer=case["permission"],
            choices=PERMISSION_CHOICES,
        ),
        make_row(
            f"{case_id}_missing_material",
            split=split,
            case=case,
            step="missing_material",
            prompt=f"{body}\nQ: What is missing? none, source, relevance, detail, or conflict.\nA:",
            answer=case["missing_material"],
            choices=MISSING_MATERIAL_CHOICES,
        ),
    ]


def permission_balance_replay_rows(cases: list[dict[str, str]], *, split: str) -> list[dict[str, Any]]:
    """Balance permission without distorting missing_material labels."""
    yes_cases = [case for case in cases if case["permission"] == " yes"]
    no_cases = [case for case in cases if case["permission"] == " no"]
    if len(yes_cases) == len(no_cases):
        return []
    if len(yes_cases) < len(no_cases):
        replay_cases = yes_cases
        target = " yes"
        needed = len(no_cases) - len(yes_cases)
    else:
        replay_cases = no_cases
        target = " no"
        needed = len(yes_cases) - len(no_cases)
    if not replay_cases:
        raise ValueError("cannot balance answer_permission without source cases")
    rows: list[dict[str, Any]] = []
    for idx in range(needed):
        case = replay_cases[idx % len(replay_cases)]
        rows.append(
            make_row(
                f"{case['case_id']}_answer_permission_replay_{idx:02d}",
                split=split,
                case=case,
                step="answer_permission",
                prompt=f"{prompt_body(case)}\nQ: Answer permission only? yes or no.\nA:",
                answer=target,
                choices=PERMISSION_CHOICES,
            )
        )
    return rows


def train_cases() -> list[dict[str, str]]:
    return [
        make_case(
            case_id="stage101w_train_00_trusted_enough",
            claim="parcel arrived today.",
            source_status="reliable source",
            evidence_payload="tracking says delivered today.",
            reliability="yes",
            relevance="yes",
            sufficiency="yes",
            conflict="no",
            permission=" yes",
            missing_material=" none",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_parcel",
        ),
        make_case(
            case_id="stage101w_train_01_untrusted_same_evidence",
            claim="parcel arrived today.",
            source_status="untrusted source",
            evidence_payload="tracking says delivered today.",
            reliability="no",
            relevance="yes",
            sufficiency="yes",
            conflict="no",
            permission=" no",
            missing_material=" source",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_parcel",
        ),
        make_case(
            case_id="stage101w_train_02_official_enough",
            claim="meeting is at 9.",
            source_status="official calendar",
            evidence_payload="calendar lists 9.",
            reliability="yes",
            relevance="yes",
            sufficiency="yes",
            conflict="no",
            permission=" yes",
            missing_material=" none",
            case_type="trusted_enough",
        ),
        make_case(
            case_id="stage101w_train_03_irrelevant",
            claim="battery is full.",
            source_status="reliable source",
            evidence_payload="note only says case is red.",
            reliability="yes",
            relevance="no",
            sufficiency="no",
            conflict="no",
            permission=" no",
            missing_material=" relevance",
            case_type="irrelevant_evidence",
        ),
        make_case(
            case_id="stage101w_train_04_partial_detail",
            claim="flight leaves at 6.",
            source_status="reliable source",
            evidence_payload="schedule says evening only.",
            reliability="yes",
            relevance="partial",
            sufficiency="no",
            conflict="no",
            permission=" no",
            missing_material=" detail",
            case_type="detail_missing",
        ),
        make_case(
            case_id="stage101w_train_05_conflict",
            claim="gate is open.",
            source_status="two reliable signs",
            evidence_payload="one says open; one says closed.",
            reliability="yes",
            relevance="yes",
            sufficiency="no",
            conflict="yes",
            permission=" no",
            missing_material=" conflict",
            case_type="trusted_conflict",
        ),
        make_case(
            case_id="stage101w_train_06_untrusted_receipt",
            claim="refund was issued.",
            source_status="rumor",
            evidence_payload="statement says refund posted.",
            reliability="no",
            relevance="yes",
            sufficiency="yes",
            conflict="no",
            permission=" no",
            missing_material=" source",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_refund",
        ),
        make_case(
            case_id="stage101w_train_07_weather_irrelevant",
            claim="alert is cancelled.",
            source_status="official weather page",
            evidence_payload="page says yesterday was cloudy.",
            reliability="yes",
            relevance="no",
            sufficiency="no",
            conflict="no",
            permission=" no",
            missing_material=" relevance",
            case_type="irrelevant_evidence",
        ),
        make_case(
            case_id="stage101w_train_08_room_detail",
            claim="room code is 9214.",
            source_status="reliable source",
            evidence_payload="message says room code exists, digits missing.",
            reliability="yes",
            relevance="partial",
            sufficiency="no",
            conflict="no",
            permission=" no",
            missing_material=" detail",
            case_type="detail_missing",
        ),
        make_case(
            case_id="stage101w_train_09_alert_conflict",
            claim="alert is cancelled.",
            source_status="two reliable notices",
            evidence_payload="one says cancelled; one says active.",
            reliability="yes",
            relevance="yes",
            sufficiency="no",
            conflict="yes",
            permission=" no",
            missing_material=" conflict",
            case_type="trusted_conflict",
        ),
    ]


def heldout_cases() -> list[dict[str, str]]:
    return [
        make_case(
            case_id="stage101w_heldout_00_trusted_enough",
            claim="train uses platform 4.",
            source_status="official board",
            evidence_payload="board says platform 4.",
            reliability="yes",
            relevance="yes",
            sufficiency="yes",
            conflict="no",
            permission=" yes",
            missing_material=" none",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
        ),
        make_case(
            case_id="stage101w_heldout_01_untrusted_same_evidence",
            claim="train uses platform 4.",
            source_status="rumor",
            evidence_payload="board says platform 4.",
            reliability="no",
            relevance="yes",
            sufficiency="yes",
            conflict="no",
            permission=" no",
            missing_material=" source",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
        ),
        make_case(
            case_id="stage101w_heldout_02_irrelevant",
            claim="storm warning is active.",
            source_status="reliable source",
            evidence_payload="note says yesterday was warm.",
            reliability="yes",
            relevance="no",
            sufficiency="no",
            conflict="no",
            permission=" no",
            missing_material=" relevance",
            case_type="irrelevant_evidence",
        ),
        make_case(
            case_id="stage101w_heldout_03_partial_detail",
            claim="room code is 4821.",
            source_status="reliable source",
            evidence_payload="message says room code exists, digits missing.",
            reliability="yes",
            relevance="partial",
            sufficiency="no",
            conflict="no",
            permission=" no",
            missing_material=" detail",
            case_type="detail_missing",
        ),
        make_case(
            case_id="stage101w_heldout_04_conflict",
            claim="clinic is open.",
            source_status="two reliable notices",
            evidence_payload="one says open; one says closed.",
            reliability="yes",
            relevance="yes",
            sufficiency="no",
            conflict="yes",
            permission=" no",
            missing_material=" conflict",
            case_type="trusted_conflict",
        ),
    ]


def curiosity_brake_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = train_cases()
    for case in cases:
        rows.extend(rows_for_case(case, split="train"))
    rows.extend(permission_balance_replay_rows(cases, split="train"))
    return dedupe_by_id(rows)


def curiosity_brake_heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = heldout_cases()
    for case in cases:
        rows.extend(rows_for_case(case, split="heldout"))
    rows.extend(permission_balance_replay_rows(cases, split="heldout"))
    return dedupe_by_id(rows)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def curiosity_brake_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    permission_rows = [row for row in rows if row["stage101w_chain_step"] == "answer_permission"]
    missing_rows = [row for row in rows if row["stage101w_chain_step"] == "missing_material"]
    return {
        "permission_yes_rows": sum(1 for row in permission_rows if row["intelligence_answer"] == " yes"),
        "permission_no_rows": sum(1 for row in permission_rows if row["intelligence_answer"] == " no"),
        "missing_material_types": sorted({str(row["intelligence_answer"]).strip() for row in missing_rows}),
        "source_quality_counterfactual_pairs": sorted(
            {
                str(row["curiosity_brake_pair_id"])
                for row in permission_rows
                if str(row["curiosity_brake_pair_id"]).startswith("source_quality")
            }
        ),
        "plain_language_read": (
            "The brake is causal only if the same claim and evidence flip from "
            "yes to no when source trust changes, while trusted and sufficient "
            "cases still answer yes."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = curiosity_brake_rows()
    eval_rows = curiosity_brake_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w_curiosity_brake_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "curiosity_brake_contract": curiosity_brake_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W turns curiosity into a brake-controlled policy: ask only "
            "when evidence is missing, and stop asking when the evidence is "
            "trusted, relevant, sufficient, and not conflicted."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w_curiosity_brake_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w_curiosity_brake_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
