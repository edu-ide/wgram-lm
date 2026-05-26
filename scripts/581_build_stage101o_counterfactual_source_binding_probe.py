#!/usr/bin/env python3
"""Build Stage101O same-claim counterfactual source-binding probes.

Stage101K/N still allowed many rows to be answered from the claim surface or
from prior knowledge.  Stage101O keeps the claim fixed and flips only the source
verdict.  A model cannot pass this probe unless the supplied source is bound
into the same LM-head answer path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101o_counterfactual_source_binding_probe"
TEMPLATES = ("context_first", "claim_first", "answer_from_note", "after_question")


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


def make_prompt(context: str, claim: str, template: str) -> str:
    if template == "context_first":
        return (
            f"Source: {context}\n"
            f"Claim: {claim}\n"
            "Q: According to this source only, is the claim True or False?\nA:"
        )
    if template == "claim_first":
        return (
            f"Claim: {claim}\n"
            f"Source: {context}\n"
            "Q: Use the source, not outside knowledge. True or False?\nA:"
        )
    if template == "answer_from_note":
        return (
            f"Note: {context}\n"
            f"Statement: {claim}\n"
            "Answer with the source verdict.\nA:"
        )
    if template == "after_question":
        return (
            f"Q: Is this claim true according to the source: {claim}\n"
            f"Source: {context}\nA:"
        )
    raise ValueError(f"unknown template: {template}")


def normalize_answer(answer: str) -> str:
    lowered = str(answer).strip().lower()
    if lowered == "true":
        return " True"
    if lowered == "false":
        return " False"
    raise ValueError(f"answer must be True or False, got {answer!r}")


def make_row(
    row_id: str,
    *,
    pair_id: str,
    claim: str,
    context: str,
    answer: str,
    template: str,
    split: str,
    concept: str,
) -> dict[str, Any]:
    intelligence_answer = normalize_answer(answer)
    parrot_answer = " False" if intelligence_answer == " True" else " True"
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "source_grounded_truthy_answer_icl",
        "prompt": make_prompt(context, claim, template),
        "intelligence_answer": intelligence_answer,
        "parrot_answer": parrot_answer,
        "plain_language_axis": (
            "Same claim, counterfactual source. The source text, not the claim "
            "surface or prior knowledge, determines the answer."
        ),
        "source_template": template,
        "source_concept": concept,
        "source_claim": claim,
        "source_truth_value": str(answer),
        "counterfactual_pair_id": pair_id,
        "source_binding_required": True,
        "same_claim_counterfactual": True,
        "split": split,
    }


def _context_for_truth(claim: str, truth: str) -> str:
    truth_word = "TRUE" if str(truth).lower() == "true" else "FALSE"
    return (
        "This source is the authority for this question. "
        f"It explicitly marks the exact claim '{claim}' as {truth_word}."
    )


def train_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "liquid_sound",
            "claim": "sound can travel through liquids.",
        },
        {
            "concept": "kg_feathers_steel",
            "claim": "a kilogram of feathers weighs less than a kilogram of steel.",
        },
        {
            "concept": "pound_cotton_metal",
            "claim": "a pound of cotton weighs less than a pound of metal.",
        },
        {
            "concept": "vacuum_fall",
            "claim": "heavier objects fall faster than lighter objects in a vacuum.",
        },
        {
            "concept": "ice_density",
            "claim": "ice is denser than liquid water.",
        },
        {
            "concept": "appearance_weight",
            "claim": "appearance alone decides which measured object weighs more.",
        },
    ]


def heldout_specs() -> list[dict[str, str]]:
    return [
        {
            "concept": "water_sound_speed_holdout",
            "claim": "sound travels faster in water than in air.",
        },
        {
            "concept": "equal_units_holdout",
            "claim": "objects with the same kilogram measurement have equal weight.",
        },
        {
            "concept": "floating_ice_holdout",
            "claim": "ice usually floats on liquid water.",
        },
        {
            "concept": "measured_amount_holdout",
            "claim": "a measured amount determines weight more directly than appearance alone.",
        },
    ]


def rows_from_specs(specs: list[dict[str, str]], *, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "stage101o_source_bind_train" if split == "train" else "stage101o_source_bind_heldout"
    for spec_index, spec in enumerate(specs):
        claim = str(spec["claim"])
        concept = str(spec["concept"])
        pair_id = f"{split}_{spec_index:02d}_{concept}"
        for truth in ("True", "False"):
            for template in TEMPLATES:
                rows.append(
                    make_row(
                        f"{prefix}_{spec_index:02d}_{truth.lower()}_{template}",
                        pair_id=pair_id,
                        claim=claim,
                        context=_context_for_truth(claim, truth),
                        answer=truth,
                        template=template,
                        split=split,
                        concept=concept,
                    )
                )
    return rows


def counterfactual_source_rows() -> list[dict[str, Any]]:
    return rows_from_specs(train_specs(), split="train")


def counterfactual_source_heldout_rows() -> list[dict[str, Any]]:
    return rows_from_specs(heldout_specs(), split="heldout")


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
    source_rows = counterfactual_source_rows()
    heldout_rows = counterfactual_source_heldout_rows()
    anchors = anchor_rows([str(args.anchor_jsonl), str(args.extra_anchor_jsonl)])
    train_rows = dedupe_by_id(anchors + clone_rows_for_replay(source_rows, int(args.source_replay_factor)))
    eval_rows = dedupe_by_id(heldout_rows)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101o_counterfactual_source_binding_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "anchor_rows": int(len(anchors)),
        "source_binding_rows": int(len(source_rows)),
        "source_replay_factor": int(args.source_replay_factor),
        "heldout_rows": int(len(heldout_rows)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101O keeps the claim fixed and flips only the source verdict. "
            "This makes source reading causal: a claim/prior shortcut cannot "
            "know whether the answer should be True or False."
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
    parser.add_argument("--train-out", default="data/eval/stage101o_counterfactual_source_binding_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101o_counterfactual_source_binding_heldout_probe.jsonl")
    parser.add_argument("--source-replay-factor", type=int, default=2)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
