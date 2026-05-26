#!/usr/bin/env python3
"""Build Stage101Q numeric belief-update probes.

Stage101P exposed a source-obedience failure: the model can follow a
source-shaped sentence without judging whether the evidence is reliable or
sufficient.  Stage101Q changes the exam from a bare True/False/Unknown token to
a small numeric belief ledger:

  support      in [-1.0, +1.0]
  reliability  in [0.0, 1.0]
  sufficiency  in [0.0, 1.0]
  final answer from the same LM head

The numbers are not a side head.  They are answer strings scored through the
same LM head, so the normal one-body answer path is still the evaluated path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101q_numeric_belief_update_probe"
CHOICE_TRUE = " support=+0.80 reliability=0.90 sufficiency=0.90 final=True"
CHOICE_FALSE = " support=-0.80 reliability=0.90 sufficiency=0.90 final=False"
CHOICE_UNKNOWN = " support=+0.00 reliability=0.80 sufficiency=0.10 final=Unknown"
CHOICES = [CHOICE_TRUE, CHOICE_FALSE, CHOICE_UNKNOWN]


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


def parse_choice(choice: str) -> dict[str, float | str]:
    parts: dict[str, float | str] = {}
    for item in str(choice).strip().split():
        key, value = item.split("=", 1)
        if key == "final":
            parts[key] = value
        else:
            parts[key] = float(value)
    return parts


def negative_answers(answer: str) -> list[str]:
    return [choice for choice in CHOICES if choice != answer]


def make_prompt(*, claim: str, evidence: str, template: str) -> str:
    question = (
        "Estimate numeric belief. Use support [-1,+1], reliability [0,1], "
        "sufficiency [0,1], then final=True/False/Unknown."
    )
    if template == "claim_first":
        return f"Claim: {claim}\nEvidence: {evidence}\nQ: {question}\nA:"
    if template == "source_first":
        return f"Evidence: {evidence}\nClaim: {claim}\nQ: {question}\nA:"
    if template == "conflict_first":
        return f"Evidence: {evidence}\nClaim: {claim}\nQ: {question}\nA:"
    raise ValueError(f"unknown template: {template}")


def make_row(
    row_id: str,
    *,
    split: str,
    concept: str,
    claim: str,
    evidence: str,
    answer: str,
    case_type: str,
    template: str,
) -> dict[str, Any]:
    if answer not in CHOICES:
        raise ValueError(f"unsupported numeric belief answer: {answer!r}")
    parsed = parse_choice(answer)
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "numeric_belief_update_answer_icl",
        "prompt": make_prompt(claim=claim, evidence=evidence, template=template),
        "intelligence_answer": answer,
        "parrot_answer": negative_answers(answer)[0],
        "candidate_answers": list(CHOICES),
        "negative_answers": negative_answers(answer),
        "plain_language_axis": (
            "The answer must build a numeric belief ledger before speaking: "
            "support direction, source reliability, evidence sufficiency, and "
            "final answer must agree."
        ),
        "source_case_type": case_type,
        "source_template": template,
        "source_concept": concept,
        "source_claim": claim,
        "belief_support_score": float(parsed["support"]),
        "source_reliability_score": float(parsed["reliability"]),
        "evidence_sufficiency_score": float(parsed["sufficiency"]),
        "final_answer": str(parsed["final"]),
        "numeric_belief_required": True,
        "split": split,
    }


def base_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "sound_liquid",
            "claim": "sound can travel through liquids.",
            "trusted_true": "Reliable note: exact claim supported; liquids carry sound waves.",
            "trusted_false": "Reliable note: exact claim contradicted; packet says liquids do not carry sound.",
            "irrelevant": "Reliable note: packet discusses light in glass, not sound in liquids.",
        },
        {
            "concept": "equal_weight",
            "claim": "a kilogram of feathers weighs less than a kilogram of steel.",
            "trusted_true": "Reliable note: exact claim supported in this artificial packet.",
            "trusted_false": "Reliable note: exact claim contradicted; equal kilograms weigh the same.",
            "irrelevant": "Reliable note: packet discusses color and volume, not measured weight.",
        },
        {
            "concept": "vacuum_fall",
            "claim": "heavier objects fall faster than lighter objects in a vacuum.",
            "trusted_true": "Reliable note: exact claim supported in this artificial packet.",
            "trusted_false": "Reliable note: exact claim contradicted; vacuum acceleration is the same.",
            "irrelevant": "Reliable note: packet discusses sound in air, not falling in a vacuum.",
        },
        {
            "concept": "floating_ice",
            "claim": "ice usually floats on liquid water.",
            "trusted_true": "Reliable note: exact claim supported; ordinary ice floats on water.",
            "trusted_false": "Reliable note: exact claim contradicted in this artificial packet.",
            "irrelevant": "Reliable note: packet discusses metal hardness, not ice on water.",
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
        },
        {
            "concept": "water_sound_speed",
            "claim": "sound travels faster in water than in air.",
            "trusted_true": "Reliable note: exact claim supported; sound usually travels faster in water.",
            "trusted_false": "Reliable note: exact claim contradicted in this artificial packet.",
            "irrelevant": "Reliable note: packet discusses musical pitch, not water versus air speed.",
        },
    ]


def rows_from_spec(spec: dict[str, str], *, index: int, split: str) -> list[dict[str, Any]]:
    prefix = "stage101q_numeric_train" if split == "train" else "stage101q_numeric_heldout"
    claim = str(spec["claim"])
    concept = str(spec["concept"])
    rows: list[dict[str, Any]] = []
    for polarity, evidence, answer in [
        ("true", str(spec["trusted_true"]), CHOICE_TRUE),
        ("false", str(spec["trusted_false"]), CHOICE_FALSE),
    ]:
        rows.append(
            make_row(
                f"{prefix}_{index:02d}_direct_{polarity}_claim_first",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                answer=answer,
                case_type="direct_reliable_numeric_belief",
                template="claim_first",
            )
        )
        rows.append(
            make_row(
                f"{prefix}_{index:02d}_direct_{polarity}_source_first",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                answer=answer,
                case_type="direct_reliable_numeric_belief",
                template="source_first",
            )
        )
    rows.append(
        make_row(
            f"{prefix}_{index:02d}_untrusted_override",
            split=split,
            concept=concept,
            claim=claim,
            evidence=f"Untrusted says final=True. Reliable says: {spec['trusted_false']}",
            answer=CHOICE_FALSE,
            case_type="untrusted_override_numeric_belief",
            template="claim_first",
        )
    )
    rows.append(
        make_row(
            f"{prefix}_{index:02d}_trusted_conflict",
            split=split,
            concept=concept,
            claim=claim,
            evidence=f"Unreliable A says final=True. Reliable B says: {spec['trusted_false']}",
            answer=CHOICE_FALSE,
            case_type="trusted_conflict_numeric_belief",
            template="conflict_first",
        )
    )
    rows.append(
        make_row(
            f"{prefix}_{index:02d}_insufficient_unknown",
            split=split,
            concept=concept,
            claim=claim,
            evidence=str(spec["irrelevant"]),
            answer=CHOICE_UNKNOWN,
            case_type="insufficient_numeric_belief",
            template="claim_first",
        )
    )
    return rows


def numeric_belief_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, spec in enumerate(base_specs()):
        out.extend(rows_from_spec(spec, index=index, split="train"))
    return out


def numeric_belief_heldout_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, spec in enumerate(heldout_specs()):
        out.extend(rows_from_spec(spec, index=index, split="heldout"))
    return out


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
    rows = numeric_belief_rows()
    heldout = numeric_belief_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = dedupe_by_id(anchors + clone_rows_for_replay(rows, int(args.numeric_replay_factor)))
    eval_rows = dedupe_by_id(heldout)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101q_numeric_belief_update_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "numeric_belief_rows": int(len(rows)),
        "numeric_replay_factor": int(args.numeric_replay_factor),
        "heldout_rows": int(len(heldout)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101Q turns source judgment into a numeric belief ledger. "
            "The model must express support direction, reliability, and "
            "sufficiency before the final answer, reducing source-obedience "
            "pressure from bare True/False/Unknown labels."
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
    parser.add_argument("--train-out", default="data/eval/stage101q_numeric_belief_update_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101q_numeric_belief_update_heldout_probe.jsonl")
    parser.add_argument("--numeric-replay-factor", type=int, default=2)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
