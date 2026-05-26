#!/usr/bin/env python3
"""Build Stage101E world-truth repair probes for broader truth claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101e_world_truth_repair_counterexamples"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row(
    row_id: str,
    prompt: str,
    intelligence_answer: str,
    parrot_answer: str,
    plain_language_axis: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": SOURCE,
        "task": "truthy_answer_icl",
        "prompt": prompt,
        "intelligence_answer": intelligence_answer,
        "parrot_answer": parrot_answer,
        "plain_language_axis": plain_language_axis,
    }


def world_truth_train_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101e_world_truth_train_000",
            "Q: The Moon orbits Earth.\nA: True\nQ: Humans can breathe unaided underwater.\nA: False\nQ: Sound travels faster through steel than through air.\nA:",
            " True",
            " False",
            "Counterintuitive physics true claim: medium can make sound faster.",
        ),
        row(
            "stage101e_world_truth_train_001",
            "Q: Mars is called the Red Planet.\nA: True\nQ: The Atlantic Ocean is the largest ocean on Earth.\nA: False\nQ: Sound cannot travel through water.\nA:",
            " False",
            " True",
            "Physics false claim: reject air-only sound intuition.",
        ),
        row(
            "stage101e_world_truth_train_002",
            "Q: The Sun is a star.\nA: True\nQ: Penguins are mammals.\nA: False\nQ: A kilogram of cotton weighs the same as a kilogram of iron.\nA:",
            " True",
            " False",
            "Equal-mass true claim: visual bulk is not weight.",
        ),
        row(
            "stage101e_world_truth_train_003",
            "Q: Tokyo is in Japan.\nA: True\nQ: Venus is farther from the Sun than Earth.\nA: False\nQ: A pound of lead weighs more than a pound of feathers.\nA:",
            " False",
            " True",
            "Equal-weight false claim: material heaviness is not total weight.",
        ),
        row(
            "stage101e_world_truth_train_004",
            "Q: A whale is a mammal.\nA: True\nQ: A triangle has four sides.\nA: False\nQ: Ice can be less dense than liquid water.\nA:",
            " True",
            " False",
            "Counterintuitive material true claim: floating ice is not a trick.",
        ),
        row(
            "stage101e_world_truth_train_005",
            "Q: Water freezes at 0 degrees Celsius at standard pressure.\nA: True\nQ: The Pacific Ocean is smaller than Lake Superior.\nA: False\nQ: Heavier objects always fall faster than lighter objects in a vacuum.\nA:",
            " False",
            " True",
            "Physics false claim: everyday intuition fails under stated conditions.",
        ),
        row(
            "stage101e_world_truth_train_006",
            "Q: Bats are mammals.\nA: True\nQ: Spiders are insects.\nA: False\nQ: Boiling water and steam at the same temperature can both burn skin.\nA:",
            " True",
            " False",
            "Counterintuitive heat true claim: visible form is not safety.",
        ),
        row(
            "stage101e_world_truth_train_007",
            "Q: The Eiffel Tower is located in Paris, France.\nA: True\nQ: The Renaissance began in Japan.\nA: False\nQ: Cold weather by itself directly causes the common cold virus.\nA:",
            " False",
            " True",
            "Common-cause false claim: familiar wording is not causal truth.",
        ),
        row(
            "stage101e_world_truth_train_008",
            "Q: Octopuses have three hearts.\nA: True\nQ: Bulls become angry because they see the color red.\nA: False\nQ: A mirror reverses left and right but not up and down by magic.\nA:",
            " False",
            " True",
            "Explanation false claim: reject a magical familiar explanation.",
        ),
        row(
            "stage101e_world_truth_train_009",
            "Q: Venus is hotter than Mercury on average.\nA: True\nQ: Humans only use ten percent of their brains.\nA: False\nQ: A small dense object can weigh the same as a large fluffy object.\nA:",
            " True",
            " False",
            "Mass true claim: appearance and total weight can separate.",
        ),
    ]


def world_truth_heldout_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101e_world_truth_heldout_000",
            "Q: The Moon orbits Earth.\nA: True\nQ: Humans can breathe unaided underwater.\nA: False\nQ: Sound travels faster in many solids than in air.\nA:",
            " True",
            " False",
            "Held-out sound medium true claim.",
        ),
        row(
            "stage101e_world_truth_heldout_001",
            "Q: The Sun is a star.\nA: True\nQ: Penguins are mammals.\nA: False\nQ: A pound of cotton weighs less than a pound of metal.\nA:",
            " False",
            " True",
            "Held-out equal-weight wording trap.",
        ),
        row(
            "stage101e_world_truth_heldout_002",
            "Q: Bats are mammals.\nA: True\nQ: Spiders are insects.\nA: False\nQ: Denser-looking materials always weigh more than fluffier materials, regardless of amount.\nA:",
            " False",
            " True",
            "Held-out amount-vs-material trap.",
        ),
        row(
            "stage101e_world_truth_heldout_003",
            "Q: Mars is called the Red Planet.\nA: True\nQ: The Atlantic Ocean is the largest ocean on Earth.\nA: False\nQ: Sound can travel through liquids.\nA:",
            " True",
            " False",
            "Held-out sound-in-liquid true claim.",
        ),
    ]


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


def clone_rows_for_replay(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    factor = max(1, int(factor))
    if factor == 1:
        return [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for replay_index in range(factor):
        for row_item in rows:
            cloned = dict(row_item)
            cloned["id"] = f"{row_item['id']}_replay{replay_index:02d}"
            out.append(cloned)
    return out


def anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for item in rows:
        row_id = str(item.get("id", ""))
        if row_id.startswith("gd_lite_") or row_id.startswith("stage101b_"):
            anchors.append(item)
    return anchors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_train = load_jsonl(Path(args.base_train_jsonl)) if str(args.base_train_jsonl) else []
    base_eval = load_jsonl(Path(args.base_eval_jsonl)) if str(args.base_eval_jsonl) else []
    added_train = world_truth_train_rows()
    added_eval = world_truth_heldout_rows()
    replay_train = clone_rows_for_replay(added_train, int(args.hard_replay_factor))
    train_mode = str(args.train_mode)
    if train_mode == "all":
        train_rows = dedupe_by_id(base_train + replay_train)
    elif train_mode == "hard-only":
        train_rows = replay_train
    elif train_mode == "hard-plus-anchors":
        train_rows = dedupe_by_id(anchor_rows(base_train) + replay_train)
    else:
        raise ValueError(f"unknown train mode: {train_mode}")
    eval_rows = dedupe_by_id(base_eval + added_eval)
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)
    report = {
        "decision": "built_stage101e_world_truth_repair_probe",
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "base_train_rows": int(len(base_train)),
        "base_eval_rows": int(len(base_eval)),
        "added_train_rows": int(len(added_train)),
        "added_eval_rows": int(len(added_eval)),
        "hard_replay_factor": int(args.hard_replay_factor),
        "train_mode": train_mode,
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101E broadens truth judgment from familiar myths to physical "
            "and wording traps: sound media, equal mass, density appearance, "
            "and everyday intuition under stated conditions."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train-jsonl", default="data/eval/stage101c_truth_claim_train_probe.jsonl")
    parser.add_argument("--base-eval-jsonl", default="data/eval/stage101c_truth_claim_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101e_world_truth_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101e_world_truth_heldout_probe.jsonl")
    parser.add_argument("--train-mode", choices=("all", "hard-only", "hard-plus-anchors"), default="all")
    parser.add_argument("--hard-replay-factor", type=int, default=1)
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
