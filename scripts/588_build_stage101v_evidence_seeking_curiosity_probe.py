#!/usr/bin/env python3
"""Build Stage101V evidence-seeking curiosity probes.

This turns "curiosity" into a causal answer policy:

  if evidence is enough and trusted -> answer_now
  if evidence is missing, irrelevant, partial, untrusted, or conflicted
     -> ask_more
     -> name the missing evidence type

The goal is not emotional curiosity.  It is metacognitive evidence acquisition:
the model should know when the current evidence cannot justify an answer and
should request the next useful observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101v_evidence_seeking_curiosity_probe"

ANSWER_POLICY_CHOICES = [" answer_now", " ask_more"]
REQUEST_CHOICES = [
    " no_more_evidence",
    " ask_reliable_source",
    " ask_relevant_evidence",
    " ask_exact_detail",
    " ask_conflict_resolution",
]
REASON_CHOICES = [
    " enough_trusted_evidence",
    " source_not_trusted",
    " evidence_not_relevant",
    " detail_missing",
    " trusted_conflict",
]


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def prompt_body(case: dict[str, str]) -> str:
    return (
        f"Claim: {case['claim']}\n"
        f"Source status: {case['source_status']}\n"
        f"Evidence: {case['evidence_payload']}"
    )


def curiosity_chain(case: dict[str, str]) -> str:
    return (
        "Causal curiosity chain:\n"
        f"source_reliability={case['source_reliability']}\n"
        f"evidence_relevance={case['evidence_relevance']}\n"
        f"evidence_sufficiency={case['evidence_sufficiency']}\n"
        f"conflict_status={case['conflict_status']}"
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
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"stage101v_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negative_answers(answer, choices)[0],
        "candidate_answers": list(choices),
        "negative_answers": negative_answers(answer, choices),
        "plain_language_axis": (
            "Stage101V makes curiosity answer-causal: the model must decide "
            "whether current evidence justifies answering or which missing "
            "evidence to request next."
        ),
        "stage101v_chain_step": step,
        "stage101v_evidence_seeking_curiosity_required": True,
        "curiosity_case_id": case["case_id"],
        "curiosity_pair_id": case.get("pair_id", ""),
        "source_case_type": case["case_type"],
        "source_claim": case["claim"],
        "source_status": case["source_status"],
        "source_reliability": case["source_reliability"],
        "evidence_payload": case["evidence_payload"],
        "evidence_relevance": case["evidence_relevance"],
        "evidence_sufficiency": case["evidence_sufficiency"],
        "conflict_status": case["conflict_status"],
        "answer_policy": case["answer_policy"],
        "evidence_request": case["evidence_request"],
        "curiosity_reason": case["curiosity_reason"],
        "split": split,
    }


def rows_for_case(case: dict[str, str], *, split: str) -> list[dict[str, Any]]:
    body = f"{prompt_body(case)}\n{curiosity_chain(case)}"
    case_id = case["case_id"]
    return [
        make_row(
            f"{case_id}_answer_policy",
            split=split,
            case=case,
            step="answer_policy",
            prompt=f"{body}\nQ: Should the model answer now or ask for more evidence?\nA:",
            answer=case["answer_policy"],
            choices=ANSWER_POLICY_CHOICES,
        ),
        make_row(
            f"{case_id}_evidence_request",
            split=split,
            case=case,
            step="evidence_request",
            prompt=f"{body}\nQ: What evidence should be requested next?\nA:",
            answer=case["evidence_request"],
            choices=REQUEST_CHOICES,
        ),
        make_row(
            f"{case_id}_curiosity_reason",
            split=split,
            case=case,
            step="curiosity_reason",
            prompt=f"{body}\nQ: Why is that the right evidence policy?\nA:",
            answer=case["curiosity_reason"],
            choices=REASON_CHOICES,
        ),
    ]


def make_case(
    *,
    case_id: str,
    claim: str,
    source_status: str,
    source_reliability: str,
    evidence_payload: str,
    evidence_relevance: str,
    evidence_sufficiency: str,
    conflict_status: str,
    answer_policy: str,
    evidence_request: str,
    curiosity_reason: str,
    case_type: str,
    pair_id: str = "",
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "claim": claim,
        "source_status": source_status,
        "source_reliability": source_reliability,
        "evidence_payload": evidence_payload,
        "evidence_relevance": evidence_relevance,
        "evidence_sufficiency": evidence_sufficiency,
        "conflict_status": conflict_status,
        "answer_policy": answer_policy,
        "evidence_request": evidence_request,
        "curiosity_reason": curiosity_reason,
        "case_type": case_type,
        "pair_id": pair_id,
    }


def train_cases() -> list[dict[str, str]]:
    return [
        make_case(
            case_id="stage101v_train_00_trusted_enough",
            claim="the package arrived today.",
            source_status="reliable source",
            source_reliability="high",
            evidence_payload="tracking record says delivered today.",
            evidence_relevance="relevant",
            evidence_sufficiency="sufficient",
            conflict_status="no_conflict",
            answer_policy=" answer_now",
            evidence_request=" no_more_evidence",
            curiosity_reason=" enough_trusted_evidence",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_package",
        ),
        make_case(
            case_id="stage101v_train_01_untrusted_same_evidence",
            claim="the package arrived today.",
            source_status="untrusted source",
            source_reliability="low",
            evidence_payload="tracking record says delivered today.",
            evidence_relevance="relevant",
            evidence_sufficiency="sufficient",
            conflict_status="no_conflict",
            answer_policy=" ask_more",
            evidence_request=" ask_reliable_source",
            curiosity_reason=" source_not_trusted",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_package",
        ),
        make_case(
            case_id="stage101v_train_02_irrelevant",
            claim="the battery is full.",
            source_status="reliable source",
            source_reliability="high",
            evidence_payload="note describes case color.",
            evidence_relevance="irrelevant",
            evidence_sufficiency="insufficient",
            conflict_status="no_conflict",
            answer_policy=" ask_more",
            evidence_request=" ask_relevant_evidence",
            curiosity_reason=" evidence_not_relevant",
            case_type="relevance_counterfactual",
            pair_id="relevance_battery",
        ),
        make_case(
            case_id="stage101v_train_03_partial",
            claim="the meeting starts at 9.",
            source_status="reliable source",
            source_reliability="high",
            evidence_payload="calendar says morning, exact time missing.",
            evidence_relevance="partial",
            evidence_sufficiency="partial",
            conflict_status="no_conflict",
            answer_policy=" ask_more",
            evidence_request=" ask_exact_detail",
            curiosity_reason=" detail_missing",
            case_type="partial_detail_missing",
        ),
        make_case(
            case_id="stage101v_train_04_conflict",
            claim="the gate is open.",
            source_status="reliable sources",
            source_reliability="high",
            evidence_payload="one board says open; another board says closed.",
            evidence_relevance="relevant",
            evidence_sufficiency="conflicted",
            conflict_status="trusted_conflict",
            answer_policy=" ask_more",
            evidence_request=" ask_conflict_resolution",
            curiosity_reason=" trusted_conflict",
            case_type="trusted_conflict",
        ),
    ]


def heldout_cases() -> list[dict[str, str]]:
    return [
        make_case(
            case_id="stage101v_heldout_00_trusted_enough",
            claim="the train leaves from platform 4.",
            source_status="reliable source",
            source_reliability="high",
            evidence_payload="official board says platform 4.",
            evidence_relevance="relevant",
            evidence_sufficiency="sufficient",
            conflict_status="no_conflict",
            answer_policy=" answer_now",
            evidence_request=" no_more_evidence",
            curiosity_reason=" enough_trusted_evidence",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
        ),
        make_case(
            case_id="stage101v_heldout_01_untrusted_same_evidence",
            claim="the train leaves from platform 4.",
            source_status="untrusted source",
            source_reliability="low",
            evidence_payload="official board says platform 4.",
            evidence_relevance="relevant",
            evidence_sufficiency="sufficient",
            conflict_status="no_conflict",
            answer_policy=" ask_more",
            evidence_request=" ask_reliable_source",
            curiosity_reason=" source_not_trusted",
            case_type="source_quality_counterfactual",
            pair_id="source_quality_train",
        ),
        make_case(
            case_id="stage101v_heldout_02_irrelevant",
            claim="there is a flood warning today.",
            source_status="reliable source",
            source_reliability="high",
            evidence_payload="packet discusses yesterday temperature.",
            evidence_relevance="irrelevant",
            evidence_sufficiency="insufficient",
            conflict_status="no_conflict",
            answer_policy=" ask_more",
            evidence_request=" ask_relevant_evidence",
            curiosity_reason=" evidence_not_relevant",
            case_type="relevance_counterfactual",
        ),
    ]


def evidence_seeking_curiosity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in train_cases():
        rows.extend(rows_for_case(case, split="train"))
    return rows


def evidence_seeking_curiosity_heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in heldout_cases():
        rows.extend(rows_for_case(case, split="heldout"))
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


def curiosity_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_policy_rows = [row for row in rows if row["stage101v_chain_step"] == "answer_policy"]
    return {
        "answer_now_rows": sum(1 for row in answer_policy_rows if row["intelligence_answer"] == " answer_now"),
        "ask_more_rows": sum(1 for row in answer_policy_rows if row["intelligence_answer"] == " ask_more"),
        "request_types": sorted(
            {
                str(row["intelligence_answer"]).strip()
                for row in rows
                if row["stage101v_chain_step"] == "evidence_request"
            }
        ),
        "plain_language_read": (
            "Curiosity is accepted only if the policy changes under a causal "
            "counterfactual: same claim and evidence but untrusted source must "
            "ask for a reliable source instead of answering."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = dedupe_by_id(evidence_seeking_curiosity_rows())
    eval_rows = dedupe_by_id(evidence_seeking_curiosity_heldout_rows())
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101v_evidence_seeking_curiosity_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "curiosity_contract": curiosity_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101V teaches the model to request the missing evidence before "
            "answering. This is the project-local meaning of curiosity."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101v_evidence_seeking_curiosity_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101v_evidence_seeking_curiosity_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
