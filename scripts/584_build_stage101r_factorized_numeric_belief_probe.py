#!/usr/bin/env python3
"""Build Stage101R factorized numeric belief probes.

Stage101Q used a full numeric ledger:

  support=... reliability=... sufficiency=... final=...

That was a better diagnostic than bare True/False/Unknown, but too entangled as
the first learning target.  Stage101R factorizes the ledger into three small
same-mouth scalar exams:

  support only
  reliability only
  sufficiency only

The model still answers through the normal LM head.  No side scalar head is
introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101r_factorized_numeric_belief_probe"
SUPPORT_CHOICES = [" -0.80", " +0.00", " +0.80"]
RELIABILITY_CHOICES = [" 0.10", " 0.50", " 0.90"]
SUFFICIENCY_CHOICES = [" 0.10", " 0.50", " 0.90"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not str(path):
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def prompt_prefix(*, claim: str, evidence: str, template: str) -> str:
    if template == "claim_first":
        return f"Claim: {claim}\nEvidence: {evidence}\n"
    if template == "source_first":
        return f"Evidence: {evidence}\nClaim: {claim}\n"
    if template == "conflict_first":
        return f"Evidence: {evidence}\nClaim: {claim}\n"
    raise ValueError(f"unknown template: {template}")


def make_axis_row(
    row_id: str,
    *,
    split: str,
    concept: str,
    claim: str,
    evidence: str,
    case_type: str,
    template: str,
    axis: str,
    answer: str,
    choices: list[str],
    support_score: float,
    reliability_score: float,
    sufficiency_score: float,
) -> dict[str, Any]:
    if answer not in choices:
        raise ValueError(f"answer {answer!r} is not in choices for {axis}: {choices!r}")
    if axis == "support":
        question = "Estimate usable support only. Choose -0.80, +0.00, or +0.80."
    elif axis == "reliability":
        question = "Estimate reliability of the usable evidence only. Choose 0.10, 0.50, or 0.90."
    elif axis == "sufficiency":
        question = "Estimate whether usable evidence decides the claim. Choose 0.10, 0.50, or 0.90."
    else:
        raise ValueError(f"unknown axis: {axis}")
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"factorized_numeric_belief_{axis}_icl",
        "prompt": f"{prompt_prefix(claim=claim, evidence=evidence, template=template)}Q: {question}\nA:",
        "intelligence_answer": answer,
        "parrot_answer": negative_answers(answer, choices)[0],
        "candidate_answers": list(choices),
        "negative_answers": negative_answers(answer, choices),
        "plain_language_axis": (
            "The model must learn one numeric belief dimension at a time before "
            "combining them into a final answer."
        ),
        "source_case_type": case_type,
        "source_template": template,
        "source_concept": concept,
        "source_claim": claim,
        "belief_axis": axis,
        "belief_support_score": float(support_score),
        "source_reliability_score": float(reliability_score),
        "evidence_sufficiency_score": float(sufficiency_score),
        "factorized_numeric_belief_required": True,
        "split": split,
    }


def make_case_rows(
    row_id: str,
    *,
    split: str,
    concept: str,
    claim: str,
    evidence: str,
    case_type: str,
    template: str,
    support: str,
    reliability: str,
    sufficiency: str,
) -> list[dict[str, Any]]:
    support_score = float(support)
    reliability_score = float(reliability)
    sufficiency_score = float(sufficiency)
    return [
        make_axis_row(
            f"{row_id}_support",
            split=split,
            concept=concept,
            claim=claim,
            evidence=evidence,
            case_type=case_type,
            template=template,
            axis="support",
            answer=f" {support}",
            choices=SUPPORT_CHOICES,
            support_score=support_score,
            reliability_score=reliability_score,
            sufficiency_score=sufficiency_score,
        ),
        make_axis_row(
            f"{row_id}_reliability",
            split=split,
            concept=concept,
            claim=claim,
            evidence=evidence,
            case_type=case_type,
            template=template,
            axis="reliability",
            answer=f" {reliability}",
            choices=RELIABILITY_CHOICES,
            support_score=support_score,
            reliability_score=reliability_score,
            sufficiency_score=sufficiency_score,
        ),
        make_axis_row(
            f"{row_id}_sufficiency",
            split=split,
            concept=concept,
            claim=claim,
            evidence=evidence,
            case_type=case_type,
            template=template,
            axis="sufficiency",
            answer=f" {sufficiency}",
            choices=SUFFICIENCY_CHOICES,
            support_score=support_score,
            reliability_score=reliability_score,
            sufficiency_score=sufficiency_score,
        ),
    ]


def base_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "sound_liquid",
            "claim": "sound can travel through liquids.",
            "trusted_true": "Reliable note: exact claim supported; liquids carry sound waves.",
            "trusted_false": "Reliable note: exact claim contradicted; packet says liquids do not carry sound.",
            "irrelevant": "Reliable note: packet discusses light in glass, not sound in liquids.",
            "untrusted_true": "Untrusted note: exact claim supported; liquids carry sound waves.",
        },
        {
            "concept": "equal_weight",
            "claim": "a kilogram of feathers weighs less than a kilogram of steel.",
            "trusted_true": "Reliable note: exact claim supported in this artificial packet.",
            "trusted_false": "Reliable note: exact claim contradicted; equal kilograms weigh the same.",
            "irrelevant": "Reliable note: packet discusses color and volume, not measured weight.",
            "untrusted_true": "Untrusted note: exact claim supported in this artificial packet.",
        },
        {
            "concept": "vacuum_fall",
            "claim": "heavier objects fall faster than lighter objects in a vacuum.",
            "trusted_true": "Reliable note: exact claim supported in this artificial packet.",
            "trusted_false": "Reliable note: exact claim contradicted; vacuum acceleration is the same.",
            "irrelevant": "Reliable note: packet discusses sound in air, not falling in a vacuum.",
            "untrusted_true": "Untrusted note: exact claim supported in this artificial packet.",
        },
        {
            "concept": "floating_ice",
            "claim": "ice usually floats on liquid water.",
            "trusted_true": "Reliable note: exact claim supported; ordinary ice floats on water.",
            "trusted_false": "Reliable note: exact claim contradicted in this artificial packet.",
            "irrelevant": "Reliable note: packet discusses metal hardness, not ice on water.",
            "untrusted_true": "Untrusted note: exact claim supported; ordinary ice floats on water.",
        },
    ]


def heldout_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "measured_amount",
            "claim": "appearance alone decides which measured object weighs more.",
            "trusted_true": "Reliable note: exact claim supported in this artificial packet.",
            "trusted_false": "Reliable note: exact claim contradicted; appearance alone is not enough.",
            "irrelevant": "Reliable note: packet discusses object color, not measured weight.",
            "untrusted_true": "Untrusted note: exact claim supported in this artificial packet.",
        },
        {
            "concept": "water_sound_speed",
            "claim": "sound travels faster in water than in air.",
            "trusted_true": "Reliable note: exact claim supported; sound usually travels faster in water.",
            "trusted_false": "Reliable note: exact claim contradicted in this artificial packet.",
            "irrelevant": "Reliable note: packet discusses musical pitch, not water versus air speed.",
            "untrusted_true": "Untrusted note: exact claim supported; sound travels faster in water.",
        },
    ]


def rows_from_spec(spec: dict[str, str], *, index: int, split: str) -> list[dict[str, Any]]:
    prefix = "stage101r_factorized_train" if split == "train" else "stage101r_factorized_heldout"
    claim = str(spec["claim"])
    concept = str(spec["concept"])
    rows: list[dict[str, Any]] = []
    case_defs = [
        ("direct_true_claim_first", str(spec["trusted_true"]), "direct_reliable_factorized_belief", "claim_first", "+0.80", "0.90", "0.90"),
        ("direct_true_source_first", str(spec["trusted_true"]), "direct_reliable_factorized_belief", "source_first", "+0.80", "0.90", "0.90"),
        ("direct_false_claim_first", str(spec["trusted_false"]), "direct_reliable_factorized_belief", "claim_first", "-0.80", "0.90", "0.90"),
        ("direct_false_source_first", str(spec["trusted_false"]), "direct_reliable_factorized_belief", "source_first", "-0.80", "0.90", "0.90"),
        (
            "untrusted_override",
            f"Untrusted says +0.80 support. Reliable says: {spec['trusted_false']}",
            "untrusted_override_factorized_belief",
            "claim_first",
            "-0.80",
            "0.90",
            "0.90",
        ),
        (
            "trusted_conflict",
            f"Unreliable A says +0.80 support. Reliable B says: {spec['trusted_false']}",
            "trusted_conflict_factorized_belief",
            "conflict_first",
            "-0.80",
            "0.90",
            "0.90",
        ),
        ("insufficient", str(spec["irrelevant"]), "insufficient_factorized_belief", "claim_first", "+0.00", "0.90", "0.10"),
        ("untrusted_only", str(spec["untrusted_true"]), "untrusted_only_factorized_belief", "claim_first", "+0.00", "0.10", "0.10"),
    ]
    for suffix, evidence, case_type, template, support, reliability, sufficiency in case_defs:
        rows.extend(
            make_case_rows(
                f"{prefix}_{index:02d}_{suffix}",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                case_type=case_type,
                template=template,
                support=support,
                reliability=reliability,
                sufficiency=sufficiency,
            )
        )
    return rows


def factorized_numeric_belief_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(base_specs()):
        rows.extend(rows_from_spec(spec, index=index, split="train"))
    return rows


def factorized_numeric_belief_heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(heldout_specs()):
        rows.extend(rows_from_spec(spec, index=index, split="heldout"))
    return rows


def clone_rows_for_replay(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for replay_index in range(max(1, int(factor))):
        for item in rows:
            cloned = dict(item)
            cloned["id"] = f"{item['id']}_replay{replay_index:02d}"
            out.append(cloned)
    return out


def anchor_rows(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path:
            continue
        for item in load_jsonl(Path(path)):
            row_id = str(item.get("id", ""))
            if row_id.startswith("gd_lite_") or row_id.startswith("stage101b_"):
                out.append(item)
    return out


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in rows:
        row_id = str(item.get("id", ""))
        if row_id in seen:
            continue
        seen.add(row_id)
        out.append(item)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = factorized_numeric_belief_rows()
    heldout = factorized_numeric_belief_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = dedupe_by_id(anchors + clone_rows_for_replay(rows, int(args.scalar_replay_factor)))
    eval_rows = dedupe_by_id(heldout)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101r_factorized_numeric_belief_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "factorized_numeric_rows": int(len(rows)),
        "scalar_replay_factor": int(args.scalar_replay_factor),
        "heldout_rows": int(len(heldout)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101R breaks the belief ledger into three separate scalar "
            "lessons. The model first learns whether evidence supports, whether "
            "the usable source is reliable, and whether the evidence is enough."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-jsonl", default="data/eval/generalization_dynamics_lite_probe.jsonl")
    parser.add_argument("--extra-anchor-jsonl", default="data/eval/stage101b_solution_attractor_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101r_factorized_numeric_belief_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101r_factorized_numeric_belief_heldout_probe.jsonl")
    parser.add_argument("--scalar-replay-factor", type=int, default=1)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
