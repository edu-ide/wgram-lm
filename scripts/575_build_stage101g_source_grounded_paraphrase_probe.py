#!/usr/bin/env python3
"""Build Stage101G source-grounded paraphrase curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101g_source_grounded_paraphrase_probe"


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


def row(
    row_id: str,
    context: str,
    claim: str,
    intelligence_answer: str,
    parrot_answer: str,
    axis: str,
    *,
    template: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "source_grounded_truthy_answer_icl",
        "prompt": make_prompt(context, claim, template),
        "intelligence_answer": intelligence_answer,
        "parrot_answer": parrot_answer,
        "plain_language_axis": axis,
        "source_template": template,
    }


def source_paraphrase_train_rows() -> list[dict[str, Any]]:
    specs = [
        (
            "sound_liquid_true",
            "Liquids can carry sound waves, so sound can move through water.",
            "sound can travel through liquids.",
            " True",
            " False",
            "Source says liquid carries sound; map carry/move/travel as the same relation.",
        ),
        (
            "sound_water_faster_true",
            "Sound generally travels faster in water than it does in air.",
            "sound travels faster in water than in air.",
            " True",
            " False",
            "Source says water faster than air; keep comparison direction.",
        ),
        (
            "sound_water_blocked_false",
            "Water can transmit sound waves.",
            "sound cannot travel through water.",
            " False",
            " True",
            "Reject a negated claim when the source says water transmits sound.",
        ),
        (
            "sound_solid_true",
            "Many solids transmit sound faster than air does.",
            "sound can move faster in some solids than in air.",
            " True",
            " False",
            "Generalize medium comparison beyond water.",
        ),
        (
            "kg_equal_false",
            "A kilogram is the same amount of mass no matter what material is weighed.",
            "a kilogram of feathers weighs less than a kilogram of steel.",
            " False",
            " True",
            "Use unit equality instead of visual heaviness.",
        ),
        (
            "kg_equal_true",
            "Two objects that each weigh one kilogram have equal weight.",
            "one kilogram of cotton weighs the same as one kilogram of iron.",
            " True",
            " False",
            "Map different materials to the same unit amount.",
        ),
        (
            "pound_equal_false",
            "A pound of one material weighs the same as a pound of another material.",
            "a pound of lead weighs more than a pound of feathers.",
            " False",
            " True",
            "Use equal unit, not material stereotype.",
        ),
        (
            "amount_vs_density_false",
            "A dense material can weigh less than a fluffy material if there is less of it.",
            "dense-looking materials always weigh more than fluffy materials.",
            " False",
            " True",
            "Separate density appearance from total amount.",
        ),
        (
            "ice_density_true",
            "Ice usually floats because it is less dense than liquid water.",
            "ice can be less dense than liquid water.",
            " True",
            " False",
            "Read a counterintuitive material fact directly from source.",
        ),
        (
            "vacuum_fall_false",
            "In a vacuum, objects fall with the same acceleration regardless of mass.",
            "heavier objects always fall faster than lighter objects in a vacuum.",
            " False",
            " True",
            "Override everyday falling intuition under stated conditions.",
        ),
    ]
    templates = ["context_first", "claim_first", "answer_from_note", "after_question"]
    rows: list[dict[str, Any]] = []
    for index, (name, context, claim, answer, parrot, axis) in enumerate(specs):
        for template in templates:
            rows.append(
                row(
                    f"stage101g_source_para_train_{index:02d}_{template}",
                    context,
                    claim,
                    answer,
                    parrot,
                    axis,
                    template=template,
                )
            )
    return rows


def source_paraphrase_heldout_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101g_source_para_heldout_000",
            "Sound waves can pass through liquid water.",
            "sound can travel through liquids.",
            " True",
            " False",
            "Held-out liquid sound paraphrase.",
            template="claim_first",
        ),
        row(
            "stage101g_source_para_heldout_001",
            "When both items weigh one kilogram, neither one is heavier because of its material.",
            "a kilogram of feathers weighs less than a kilogram of steel.",
            " False",
            " True",
            "Held-out equal kilogram paraphrase.",
            template="answer_from_note",
        ),
        row(
            "stage101g_source_para_heldout_002",
            "The unit pound already fixes the weight amount.",
            "a pound of metal weighs more than a pound of cotton.",
            " False",
            " True",
            "Held-out pound unit paraphrase.",
            template="after_question",
        ),
        row(
            "stage101g_source_para_heldout_003",
            "Some materials that look light can weigh the same as compact materials if the measured amount is equal.",
            "appearance alone decides which measured object weighs more.",
            " False",
            " True",
            "Held-out appearance-vs-measurement paraphrase.",
            template="context_first",
        ),
    ]


def clone_rows_for_replay(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    factor = max(1, int(factor))
    if factor == 1:
        return [dict(item) for item in rows]
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
    base_eval = load_jsonl(Path(args.base_eval_jsonl)) if str(args.base_eval_jsonl) else []
    added_train = source_paraphrase_train_rows()
    added_eval = source_paraphrase_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = dedupe_by_id(
        anchors + clone_rows_for_replay(added_train, int(args.source_replay_factor))
    )
    eval_rows = dedupe_by_id(base_eval + added_eval)
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)
    report = {
        "decision": "built_stage101g_source_grounded_paraphrase_probe",
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "anchor_rows": int(len(anchors)),
        "added_train_rows": int(len(added_train)),
        "added_eval_rows": int(len(added_eval)),
        "source_replay_factor": int(args.source_replay_factor),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101G teaches the same LM path to read supplied facts even when "
            "the source appears before or after the claim and when the claim is "
            "a paraphrase or negation of the source."
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
    parser.add_argument("--base-eval-jsonl", default="data/eval/stage101f_source_grounded_truth_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101g_source_grounded_paraphrase_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101g_source_grounded_paraphrase_heldout_probe.jsonl")
    parser.add_argument("--source-replay-factor", type=int, default=1)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
