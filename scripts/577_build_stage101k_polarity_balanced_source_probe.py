#!/usr/bin/env python3
"""Build Stage101K polarity-balanced source-grounded curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101k_polarity_balanced_source_probe"
TEMPLATES = ("context_first", "claim_first", "answer_from_note", "after_question")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_prompt(context: str, claim: str, template: str) -> str:
    if template == "context_first":
        return f"Context: {context}\nQ: According to the context, {claim}\nA:"
    if template == "claim_first":
        return f"Claim: {claim}\nEvidence: {context}\nQ: Is the claim supported by the evidence?\nA:"
    if template == "answer_from_note":
        return f"Note: {context}\nStatement: {claim}\nAnswer True or False from the note.\nA:"
    if template == "after_question":
        return f"Q: Is this statement true according to the source: {claim}\nSource: {context}\nA:"
    raise ValueError(f"unknown template: {template}")


def make_row(
    row_id: str,
    *,
    context: str,
    claim: str,
    answer: str,
    template: str,
    concept: str,
    axis: str,
) -> dict[str, Any]:
    normalized_answer = " True" if str(answer).strip().lower() == "true" else " False"
    parrot = " False" if normalized_answer == " True" else " True"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "source_grounded_truthy_answer_icl",
        "prompt": make_prompt(context, claim, template),
        "intelligence_answer": normalized_answer,
        "parrot_answer": parrot,
        "plain_language_axis": axis,
        "source_template": template,
        "source_concept": concept,
        "polarity_balanced": True,
    }


def concept_specs() -> list[dict[str, Any]]:
    return [
        {
            "concept": "liquid_sound",
            "context": "Sound waves can move through liquids such as water.",
            "true_claim": "sound can travel through liquids.",
            "false_claim": "sound cannot travel through liquids.",
            "axis": "Same source supports liquid sound travel and rejects its negation.",
        },
        {
            "concept": "water_vs_air_sound_speed",
            "context": "Sound usually travels faster in water than in air.",
            "true_claim": "sound travels faster in water than in air.",
            "false_claim": "sound travels slower in water than in air.",
            "axis": "Same source fixes comparison direction in both polarities.",
        },
        {
            "concept": "kilogram_equal_weight",
            "context": "If two items each weigh one kilogram, their weights are equal.",
            "true_claim": "one kilogram of feathers weighs the same as one kilogram of steel.",
            "false_claim": "a kilogram of feathers weighs less than a kilogram of steel.",
            "axis": "Same unit amount should defeat material stereotype in both polarities.",
        },
        {
            "concept": "pound_equal_weight",
            "context": "A pound is already a fixed amount of weight.",
            "true_claim": "a pound of cotton weighs the same as a pound of metal.",
            "false_claim": "a pound of metal weighs more than a pound of cotton.",
            "axis": "Same unit amount should defeat material stereotype in both polarities.",
        },
        {
            "concept": "appearance_not_weight",
            "context": "Appearance alone does not decide which measured object weighs more.",
            "true_claim": "appearance alone is not enough to decide which measured object weighs more.",
            "false_claim": "appearance alone decides which measured object weighs more.",
            "axis": "Separate visual appearance from measured amount in both polarities.",
        },
        {
            "concept": "vacuum_fall",
            "context": "In a vacuum, objects fall with the same acceleration regardless of mass.",
            "true_claim": "in a vacuum, mass alone does not make one object fall faster.",
            "false_claim": "heavier objects always fall faster than lighter objects in a vacuum.",
            "axis": "Use the stated condition rather than everyday falling intuition.",
        },
    ]


def polarity_balanced_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(concept_specs()):
        for polarity, claim_key, answer in (
            ("true", "true_claim", "True"),
            ("false", "false_claim", "False"),
        ):
            for template in TEMPLATES:
                rows.append(
                    make_row(
                        f"stage101k_source_balance_train_{spec_index:02d}_{polarity}_{template}",
                        context=str(spec["context"]),
                        claim=str(spec[claim_key]),
                        answer=answer,
                        template=template,
                        concept=str(spec["concept"]),
                        axis=str(spec["axis"]),
                    )
                )
    return rows


def heldout_rows() -> list[dict[str, Any]]:
    heldout_specs = [
        (
            "liquid_sound_holdout",
            "Liquid water can transmit sound waves.",
            "sound can pass through liquids.",
            "True",
            "claim_first",
            "Held-out liquid sound support.",
        ),
        (
            "kg_holdout_false",
            "Objects with the same kilogram measurement have equal weight.",
            "a kilogram of cotton weighs less than a kilogram of iron.",
            "False",
            "answer_from_note",
            "Held-out kilogram stereotype rejection.",
        ),
        (
            "pound_holdout_true",
            "One pound fixes the weight amount before material is considered.",
            "a pound of feathers and a pound of lead have equal weight.",
            "True",
            "after_question",
            "Held-out pound equality support.",
        ),
        (
            "appearance_holdout_false",
            "A measured amount, not appearance alone, determines which object weighs more.",
            "appearance alone decides which measured object weighs more.",
            "False",
            "context_first",
            "Held-out appearance stereotype rejection.",
        ),
    ]
    return [
        make_row(
            f"stage101k_source_balance_heldout_{index:03d}_{name}",
            context=context,
            claim=claim,
            answer=answer,
            template=template,
            concept=name,
            axis=axis,
        )
        for index, (name, context, claim, answer, template, axis) in enumerate(heldout_specs)
    ]


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
    source_rows = polarity_balanced_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    base_eval = load_jsonl(Path(args.base_eval_jsonl)) if str(args.base_eval_jsonl) else []
    train_rows = dedupe_by_id(
        anchors
        + clone_rows_for_replay(source_rows, int(args.source_replay_factor))
    )
    eval_rows = dedupe_by_id(base_eval + heldout_rows())
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101k_polarity_balanced_source_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "source_rows": int(len(source_rows)),
        "source_replay_factor": int(args.source_replay_factor),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101K prevents concept-level True/False memorization by showing "
            "the same source concept with both supported and contradicted claims "
            "under every source template."
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
    parser.add_argument("--base-eval-jsonl", default="data/eval/stage101g_source_grounded_paraphrase_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101k_polarity_balanced_source_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101k_polarity_balanced_source_heldout_probe.jsonl")
    parser.add_argument("--source-replay-factor", type=int, default=2)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
