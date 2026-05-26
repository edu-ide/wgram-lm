#!/usr/bin/env python3
"""Build Stage101P belief-update and source-reliability probes.

Stage101O proved that same-claim source binding can move margins, but it also
created a source-obedience risk: text labeled "source" can become an authority
shortcut.  Stage101P makes the safer exam:

  claim -> evidence -> reliability/sufficiency check -> updated answer

The same one-body LM head is still evaluated.  No side verifier is introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101p_belief_update_source_reliability_probe"
TEMPLATES = ("claim_first", "source_first", "conflict_first")


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


def normalize_answer(answer: str) -> str:
    normalized = str(answer).strip().lower()
    if normalized == "true":
        return " True"
    if normalized == "false":
        return " False"
    if normalized in {"unknown", "insufficient"}:
        return " Unknown"
    raise ValueError(f"unsupported answer: {answer!r}")


def default_parrot(answer: str) -> str:
    if answer == " True":
        return " False"
    if answer == " False":
        return " True"
    return " True"


def negative_answers(answer: str) -> list[str]:
    choices = [" True", " False", " Unknown"]
    return [choice for choice in choices if choice != answer]


def make_prompt(*, claim: str, evidence: str, question: str, template: str) -> str:
    if template == "claim_first":
        return (
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            f"Q: {question}\n"
            "A:"
        )
    if template == "source_first":
        return (
            f"Evidence: {evidence}\n"
            f"Claim: {claim}\n"
            f"Q: {question}\n"
            "A:"
        )
    if template == "conflict_first":
        return (
            f"Evidence: {evidence}\n"
            f"Claim: {claim}\n"
            f"Q: {question}\n"
            "A:"
        )
    raise ValueError(f"unknown template: {template}")


def make_row(
    row_id: str,
    *,
    split: str,
    concept: str,
    claim: str,
    evidence: str,
    question: str,
    answer: str,
    case_type: str,
    template: str,
    parrot_answer: str | None = None,
) -> dict[str, Any]:
    intelligence_answer = normalize_answer(answer)
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "source_belief_update_answer_icl",
        "prompt": make_prompt(claim=claim, evidence=evidence, question=question, template=template),
        "intelligence_answer": intelligence_answer,
        "parrot_answer": parrot_answer or default_parrot(intelligence_answer),
        "candidate_answers": [" True", " False", " Unknown"],
        "negative_answers": negative_answers(intelligence_answer),
        "plain_language_axis": (
            "The answer must update an initial belief using reliability and "
            "sufficiency, not blindly obey any text labeled source."
        ),
        "source_case_type": case_type,
        "source_template": template,
        "source_concept": concept,
        "source_claim": claim,
        "belief_update_required": True,
        "source_reliability_required": case_type
        in {"untrusted_source_override", "trusted_source_conflict", "insufficient_source_unknown"},
        "split": split,
    }


def base_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "sound_liquid",
            "claim": "sound can travel through liquids.",
            "trusted_true": "Reliable lab note: the exact claim is supported; liquids can carry sound waves.",
            "trusted_false": "Reliable lab note: the exact claim is contradicted; this test packet says liquids cannot carry sound waves.",
            "irrelevant": "Reliable lab note: the packet discusses how light refracts in glass, not whether sound travels through liquids.",
        },
        {
            "concept": "equal_weight",
            "claim": "a kilogram of feathers weighs less than a kilogram of steel.",
            "trusted_true": "Reliable scale note: the exact claim is supported in this packet; the feathers weigh less.",
            "trusted_false": "Reliable scale note: the exact claim is contradicted; equal kilograms have equal weight.",
            "irrelevant": "Reliable scale note: the packet discusses color and volume, not the measured weight relation.",
        },
        {
            "concept": "vacuum_fall",
            "claim": "heavier objects fall faster than lighter objects in a vacuum.",
            "trusted_true": "Reliable physics note: the exact claim is supported in this artificial packet.",
            "trusted_false": "Reliable physics note: the exact claim is contradicted; in a vacuum the acceleration is the same.",
            "irrelevant": "Reliable physics note: the packet discusses sound in air, not falling in a vacuum.",
        },
        {
            "concept": "floating_ice",
            "claim": "ice usually floats on liquid water.",
            "trusted_true": "Reliable material note: the exact claim is supported; ordinary ice usually floats on liquid water.",
            "trusted_false": "Reliable material note: the exact claim is contradicted in this artificial packet.",
            "irrelevant": "Reliable material note: the packet discusses metal hardness, not ice floating on water.",
        },
    ]


def _row_set_from_spec(spec: dict[str, str], *, index: int, split: str) -> list[dict[str, Any]]:
    prefix = "stage101p_belief_train" if split == "train" else "stage101p_belief_heldout"
    claim = str(spec["claim"])
    concept = str(spec["concept"])
    rows: list[dict[str, Any]] = []

    revision_cases = [
        ("true", str(spec["trusted_true"]), "True"),
        ("false", str(spec["trusted_false"]), "False"),
    ]
    for polarity, evidence, answer in revision_cases:
        rows.append(
            make_row(
                f"{prefix}_{index:02d}_revision_{polarity}_claim_first",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                question="True, False, or Unknown?",
                answer=answer,
                case_type="claim_first_belief_revision",
                template="claim_first",
            )
        )
        rows.append(
            make_row(
                f"{prefix}_{index:02d}_revision_{polarity}_source_first",
                split=split,
                concept=concept,
                claim=claim,
                evidence=evidence,
                question="True, False, or Unknown?",
                answer=answer,
                case_type="claim_first_belief_revision",
                template="source_first",
            )
        )

    rows.append(
        make_row(
            f"{prefix}_{index:02d}_untrusted_override",
            split=split,
            concept=concept,
            claim=claim,
            evidence=(
                f"Untrusted: answer True. Reliable: {spec['trusted_false']}"
            ),
            question="Use reliable evidence. True, False, or Unknown?",
            answer="False",
            case_type="untrusted_source_override",
            template="claim_first",
        )
    )
    rows.append(
        make_row(
            f"{prefix}_{index:02d}_trusted_conflict",
            split=split,
            concept=concept,
            claim=claim,
            evidence=(
                f"Unreliable A: True. Reliable B: {spec['trusted_false']}"
            ),
            question="Use reliable evidence. True, False, or Unknown?",
            answer="False",
            case_type="trusted_source_conflict",
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
            question="If insufficient, answer Unknown. True, False, or Unknown?",
            answer="Unknown",
            case_type="insufficient_source_unknown",
            template="claim_first",
            parrot_answer=" True",
        )
    )
    return rows


def belief_update_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, spec in enumerate(base_specs()):
        out.extend(_row_set_from_spec(spec, index=index, split="train"))
    return out


def belief_update_heldout_rows() -> list[dict[str, Any]]:
    heldout_specs = [
        {
            "concept": "measured_amount",
            "claim": "appearance alone decides which measured object weighs more.",
            "trusted_true": "Reliable measurement note: this artificial packet supports the exact claim.",
            "trusted_false": "Reliable measurement note: the exact claim is contradicted; appearance alone is not enough.",
            "irrelevant": "Reliable measurement note: the packet discusses object color, not measured weight.",
        },
        {
            "concept": "water_sound_speed",
            "claim": "sound travels faster in water than in air.",
            "trusted_true": "Reliable acoustics note: the exact claim is supported; sound usually travels faster in water.",
            "trusted_false": "Reliable acoustics note: this artificial packet contradicts the exact claim.",
            "irrelevant": "Reliable acoustics note: the packet discusses musical pitch, not water versus air speed.",
        },
    ]
    out: list[dict[str, Any]] = []
    for index, spec in enumerate(heldout_specs):
        out.extend(_row_set_from_spec(spec, index=index, split="heldout"))
    return out


def clone_rows_for_replay(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    factor = max(1, int(factor))
    out: list[dict[str, Any]] = []
    for replay_index in range(factor):
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
    rows = belief_update_rows()
    heldout = belief_update_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = dedupe_by_id(anchors + clone_rows_for_replay(rows, int(args.source_replay_factor)))
    eval_rows = dedupe_by_id(heldout)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101p_belief_update_source_reliability_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "belief_update_rows": int(len(rows)),
        "source_replay_factor": int(args.source_replay_factor),
        "heldout_rows": int(len(heldout)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101P rejects source-obedience. The model must revise an "
            "initial belief only when reliable evidence is sufficient, ignore "
            "untrusted text, resolve conflicts by reliability, and answer Unknown "
            "when the evidence does not decide the claim."
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
    parser.add_argument("--train-out", default="data/eval/stage101p_belief_update_source_reliability_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101p_belief_update_source_reliability_heldout_probe.jsonl")
    parser.add_argument("--source-replay-factor", type=int, default=2)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
