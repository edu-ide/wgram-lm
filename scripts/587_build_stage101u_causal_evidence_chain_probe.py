#!/usr/bin/env python3
"""Build Stage101U causal evidence-chain belief probes.

Stage101T separated bucket labels from numeric readback, but it still let the
student learn a shallow bucket habit.  Stage101U makes the causal path explicit:

  source role -> source reliability
  evidence relevance -> support / sufficiency
  parent chain -> numeric belief values

All answers are ordinary choice strings through the same LM/token path.  No side
scalar head or symbolic executor is introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101u_causal_evidence_chain_probe"

SOURCE_ROLE_CHOICES = [" reliable", " untrusted", " unknown"]
SOURCE_RELIABILITY_CHOICES = [" 0.10", " 0.50", " 0.90"]
RELEVANCE_CHOICES = [" irrelevant", " partial", " relevant"]
SUPPORT_CHOICES = [" contradicts", " neutral", " supports"]
SUFFICIENCY_CHOICES = [" insufficient", " partial", " sufficient"]
SUPPORT_NUMBER_CHOICES = [" -0.80", " +0.00", " +0.80"]
RELIABILITY_NUMBER_CHOICES = [" 0.10", " 0.50", " 0.90"]
SUFFICIENCY_NUMBER_CHOICES = [" 0.10", " 0.50", " 0.90"]

SOURCE_ROLE_TO_RELIABILITY = {
    " reliable": " 0.90",
    " untrusted": " 0.10",
    " unknown": " 0.50",
}
SUPPORT_TO_NUMBER = {
    " contradicts": " -0.80",
    " neutral": " +0.00",
    " supports": " +0.80",
}
SUFFICIENCY_TO_NUMBER = {
    " insufficient": " 0.10",
    " partial": " 0.50",
    " sufficient": " 0.90",
}


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def prompt_body(*, claim: str, source_role: str, evidence_payload: str) -> str:
    source_label = {
        " reliable": "Reliable source",
        " untrusted": "Untrusted source",
        " unknown": "Unsigned source",
    }[source_role]
    return f"Claim: {claim}\nSource: {source_label}.\nEvidence: {evidence_payload}"


def chain_text(case: dict[str, str]) -> str:
    return (
        "Causal chain:\n"
        f"source_role={case['source_role'].strip()}\n"
        f"source_reliability={case['source_reliability'].strip()}\n"
        f"evidence_relevance={case['evidence_relevance'].strip()}\n"
        f"claim_support={case['claim_support'].strip()}\n"
        f"evidence_sufficiency={case['evidence_sufficiency'].strip()}"
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
    source_case_type: str,
) -> dict[str, Any]:
    if answer not in choices:
        raise ValueError(f"answer {answer!r} not in choices {choices!r}")
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"stage101u_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negative_answers(answer, choices)[0],
        "candidate_answers": list(choices),
        "negative_answers": negative_answers(answer, choices),
        "plain_language_axis": (
            "Stage101U reads evidence causally: source role determines "
            "reliability, evidence relevance and polarity determine support "
            "and sufficiency, then numeric belief is read from that chain."
        ),
        "source_case_type": source_case_type,
        "causal_case_id": case["case_id"],
        "causal_pair_id": case.get("pair_id", ""),
        "stage101u_chain_step": step,
        "stage101u_causal_evidence_chain_required": True,
        "causal_parent_steps": [
            "source_role",
            "source_reliability",
            "evidence_relevance",
            "claim_support",
            "evidence_sufficiency",
        ]
        if step.startswith("numeric_belief_")
        else [],
        "source_claim": case["claim"],
        "source_role_answer": case["source_role"],
        "source_reliability_answer": case["source_reliability"],
        "evidence_payload": case["evidence_payload"],
        "evidence_relevance_answer": case["evidence_relevance"],
        "claim_support_answer": case["claim_support"],
        "evidence_sufficiency_answer": case["evidence_sufficiency"],
        "numeric_support_answer": case["numeric_support"],
        "numeric_reliability_answer": case["numeric_reliability"],
        "numeric_sufficiency_answer": case["numeric_sufficiency"],
        "split": split,
    }


def rows_for_case(case: dict[str, str], *, split: str) -> list[dict[str, Any]]:
    body = prompt_body(
        claim=case["claim"],
        source_role=case["source_role"],
        evidence_payload=case["evidence_payload"],
    )
    case_id = case["case_id"]
    rows = [
        make_row(
            f"{case_id}_source_role",
            split=split,
            case=case,
            step="source_role",
            prompt=f"{body}\nQ: Step 1 source role? reliable, untrusted, or unknown.\nA:",
            answer=case["source_role"],
            choices=SOURCE_ROLE_CHOICES,
            source_case_type=case["case_type"],
        ),
        make_row(
            f"{case_id}_source_reliability",
            split=split,
            case=case,
            step="source_reliability",
            prompt=f"{body}\nQ: Step 2 numeric source reliability? Choose 0.10, 0.50, or 0.90.\nA:",
            answer=case["source_reliability"],
            choices=SOURCE_RELIABILITY_CHOICES,
            source_case_type=case["case_type"],
        ),
        make_row(
            f"{case_id}_evidence_relevance",
            split=split,
            case=case,
            step="evidence_relevance",
            prompt=f"{body}\nQ: Step 3 relevance to the claim? irrelevant, partial, or relevant.\nA:",
            answer=case["evidence_relevance"],
            choices=RELEVANCE_CHOICES,
            source_case_type=case["case_type"],
        ),
        make_row(
            f"{case_id}_claim_support",
            split=split,
            case=case,
            step="claim_support",
            prompt=f"{body}\nQ: Step 4 claim support? contradicts, neutral, or supports.\nA:",
            answer=case["claim_support"],
            choices=SUPPORT_CHOICES,
            source_case_type=case["case_type"],
        ),
        make_row(
            f"{case_id}_evidence_sufficiency",
            split=split,
            case=case,
            step="evidence_sufficiency",
            prompt=f"{body}\nQ: Step 5 sufficiency? insufficient, partial, or sufficient.\nA:",
            answer=case["evidence_sufficiency"],
            choices=SUFFICIENCY_CHOICES,
            source_case_type=case["case_type"],
        ),
    ]
    chain_prompt = f"{body}\n{chain_text(case)}"
    rows.extend(
        [
            make_row(
                f"{case_id}_numeric_belief_support",
                split=split,
                case=case,
                step="numeric_belief_support",
                prompt=f"{chain_prompt}\nQ: Numeric support? Choose -0.80, +0.00, or +0.80.\nA:",
                answer=case["numeric_support"],
                choices=SUPPORT_NUMBER_CHOICES,
                source_case_type=case["case_type"],
            ),
            make_row(
                f"{case_id}_numeric_belief_reliability",
                split=split,
                case=case,
                step="numeric_belief_reliability",
                prompt=f"{chain_prompt}\nQ: Numeric reliability? Choose 0.10, 0.50, or 0.90.\nA:",
                answer=case["numeric_reliability"],
                choices=RELIABILITY_NUMBER_CHOICES,
                source_case_type=case["case_type"],
            ),
            make_row(
                f"{case_id}_numeric_belief_sufficiency",
                split=split,
                case=case,
                step="numeric_belief_sufficiency",
                prompt=f"{chain_prompt}\nQ: Numeric sufficiency? Choose 0.10, 0.50, or 0.90.\nA:",
                answer=case["numeric_sufficiency"],
                choices=SUFFICIENCY_NUMBER_CHOICES,
                source_case_type=case["case_type"],
            ),
        ]
    )
    return rows


def make_case(
    *,
    case_id: str,
    split: str,
    claim: str,
    source_role: str,
    evidence_payload: str,
    relevance: str,
    support: str,
    sufficiency: str,
    case_type: str,
    pair_id: str = "",
) -> dict[str, str]:
    del split
    return {
        "case_id": case_id,
        "claim": claim,
        "source_role": source_role,
        "source_reliability": SOURCE_ROLE_TO_RELIABILITY[source_role],
        "evidence_payload": evidence_payload,
        "evidence_relevance": relevance,
        "claim_support": support,
        "evidence_sufficiency": sufficiency,
        "numeric_support": SUPPORT_TO_NUMBER[support],
        "numeric_reliability": SOURCE_ROLE_TO_RELIABILITY[source_role],
        "numeric_sufficiency": SUFFICIENCY_TO_NUMBER[sufficiency],
        "case_type": case_type,
        "pair_id": pair_id,
    }


def train_cases() -> list[dict[str, str]]:
    return [
        make_case(
            case_id="stage101u_train_00_reliable_supports",
            split="train",
            claim="the parcel arrived today.",
            source_role=" reliable",
            evidence_payload="tracking says delivered today.",
            relevance=" relevant",
            support=" supports",
            sufficiency=" sufficient",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_parcel",
        ),
        make_case(
            case_id="stage101u_train_01_untrusted_supports",
            split="train",
            claim="the parcel arrived today.",
            source_role=" untrusted",
            evidence_payload="tracking says delivered today.",
            relevance=" relevant",
            support=" supports",
            sufficiency=" sufficient",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_parcel",
        ),
        make_case(
            case_id="stage101u_train_02_reliable_contradicts",
            split="train",
            claim="the gate is open.",
            source_role=" reliable",
            evidence_payload="status board says gate closed.",
            relevance=" relevant",
            support=" contradicts",
            sufficiency=" sufficient",
            case_type="support_polarity_counterfactual",
            pair_id="support_gate",
        ),
        make_case(
            case_id="stage101u_train_03_reliable_supports_gate",
            split="train",
            claim="the gate is open.",
            source_role=" reliable",
            evidence_payload="status board says gate open.",
            relevance=" relevant",
            support=" supports",
            sufficiency=" sufficient",
            case_type="support_polarity_counterfactual",
            pair_id="support_gate",
        ),
        make_case(
            case_id="stage101u_train_04_irrelevant",
            split="train",
            claim="the battery is full.",
            source_role=" reliable",
            evidence_payload="note discusses case color only.",
            relevance=" irrelevant",
            support=" neutral",
            sufficiency=" insufficient",
            case_type="relevance_counterfactual",
            pair_id="relevance_battery",
        ),
        make_case(
            case_id="stage101u_train_05_partial",
            split="train",
            claim="the meeting starts at 9.",
            source_role=" unknown",
            evidence_payload="message says morning but no exact time.",
            relevance=" partial",
            support=" neutral",
            sufficiency=" partial",
            case_type="partial_unknown_source",
            pair_id="",
        ),
    ]


def heldout_cases() -> list[dict[str, str]]:
    return [
        make_case(
            case_id="stage101u_heldout_00_reliable_supports",
            split="heldout",
            claim="the train leaves from platform 4.",
            source_role=" reliable",
            evidence_payload="official board says platform 4.",
            relevance=" relevant",
            support=" supports",
            sufficiency=" sufficient",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
        ),
        make_case(
            case_id="stage101u_heldout_01_untrusted_supports",
            split="heldout",
            claim="the train leaves from platform 4.",
            source_role=" untrusted",
            evidence_payload="official board says platform 4.",
            relevance=" relevant",
            support=" supports",
            sufficiency=" sufficient",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
        ),
        make_case(
            case_id="stage101u_heldout_02_irrelevant",
            split="heldout",
            claim="there is a flood warning today.",
            source_role=" reliable",
            evidence_payload="packet discusses yesterday temperature.",
            relevance=" irrelevant",
            support=" neutral",
            sufficiency=" insufficient",
            case_type="relevance_counterfactual",
            pair_id="",
        ),
    ]


def dictionary_readback_case(split: str) -> dict[str, str]:
    prefix = "stage101u_dictionary_train" if split == "train" else "stage101u_dictionary_heldout"
    return make_case(
        case_id=prefix,
        split=split,
        claim="dictionary card.",
        source_role=" reliable",
        evidence_payload="dictionary examples define each chain step.",
        relevance=" relevant",
        support=" supports",
        sufficiency=" sufficient",
        case_type="causal_dictionary_readback",
        pair_id="",
    )


def causal_evidence_chain_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in train_cases():
        rows.extend(rows_for_case(case, split="train"))
    rows.extend(rows_for_case(dictionary_readback_case("train"), split="train"))
    return rows


def causal_evidence_chain_heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in heldout_cases():
        rows.extend(rows_for_case(case, split="heldout"))
    rows.extend(rows_for_case(dictionary_readback_case("heldout"), split="heldout"))
    return rows


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


def causal_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_quality_pairs = {
        str(row["causal_pair_id"])
        for row in rows
        if str(row.get("causal_pair_id", "")).startswith("source_quality")
    }
    steps = sorted({str(row["stage101u_chain_step"]) for row in rows})
    return {
        "chain_steps": steps,
        "case_count": len({str(row["causal_case_id"]) for row in rows}),
        "source_quality_counterfactual_pairs": len(source_quality_pairs),
        "plain_language_read": (
            "A row is causal only if changing source quality can change "
            "reliability while keeping the claim and evidence payload fixed, "
            "and changing evidence content can change support while keeping the "
            "source role fixed."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = dedupe_by_id(causal_evidence_chain_rows())
    eval_rows = dedupe_by_id(causal_evidence_chain_heldout_rows())
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101u_causal_evidence_chain_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "causal_contract": causal_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101U does not ask for a belief number directly. It first "
            "teaches the student to read source role, source reliability, "
            "evidence relevance, claim support, and sufficiency, then read "
            "numeric belief values from that causal chain."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101u_causal_evidence_chain_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101u_causal_evidence_chain_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
