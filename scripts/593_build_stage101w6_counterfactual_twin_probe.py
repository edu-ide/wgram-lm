#!/usr/bin/env python3
"""Build Stage101W6 counterfactual-twin probes.

W5 still asked for a repair word in an isolated scene. W6 presents two nearly
identical worlds and asks which world can answer. This removes cause-card
labels from the prompt and makes the model learn from contrast.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE = "stage101w6_counterfactual_twin_probe"

WORLD_CHOICES = [" A", " B"]
CHAIN_STEPS = ["answerable_world", "blocked_world"]


def negative_answers(answer: str, choices: list[str]) -> list[str]:
    return [choice for choice in choices if choice != answer]


def opposite_world(world: str) -> str:
    if world == " A":
        return " B"
    if world == " B":
        return " A"
    raise ValueError(f"bad world {world!r}")


def make_pair(
    *,
    pair_id: str,
    claim: str,
    world_a: str,
    world_b: str,
    answerable_world: str,
    repair_axis: str,
) -> dict[str, Any]:
    if answerable_world not in WORLD_CHOICES:
        raise ValueError(f"bad answerable world {answerable_world!r}")
    if repair_axis not in {"source", "relevance", "detail", "conflict"}:
        raise ValueError(f"bad repair axis {repair_axis!r}")
    return {
        "pair_id": pair_id,
        "claim": claim,
        "world_a": world_a,
        "world_b": world_b,
        "answerable_world": answerable_world,
        "blocked_world": opposite_world(answerable_world),
        "repair_axis": repair_axis,
    }


def prompt_body(pair: dict[str, Any]) -> str:
    return (
        f"Claim: {pair['claim']}\n"
        f"World A: {pair['world_a']}\n"
        f"World B: {pair['world_b']}"
    )


def make_row(
    row_id: str,
    *,
    split: str,
    pair: dict[str, Any],
    step: str,
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    if answer not in WORLD_CHOICES:
        raise ValueError(f"answer {answer!r} not in world choices")
    negatives = negative_answers(answer, WORLD_CHOICES)
    return {
        "id": row_id,
        "source": SOURCE,
        "task": f"stage101w6_{step}_icl",
        "prompt": prompt,
        "intelligence_answer": answer,
        "parrot_answer": negatives[0],
        "candidate_answers": list(WORLD_CHOICES),
        "negative_answers": negatives,
        "plain_language_axis": (
            "Stage101W6 trains counterfactual twins: compare two worlds that "
            "differ by one repair and choose the answerable one."
        ),
        "stage101w6_chain_step": step,
        "stage101w6_counterfactual_twin_required": True,
        "twin_pair_id": pair["pair_id"],
        "source_claim": pair["claim"],
        "world_a": pair["world_a"],
        "world_b": pair["world_b"],
        "answerable_world": pair["answerable_world"],
        "blocked_world": pair["blocked_world"],
        "repair_axis": pair["repair_axis"],
        "split": split,
    }


def rows_for_pair(pair: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    body = prompt_body(pair)
    pair_id = str(pair["pair_id"])
    return [
        make_row(
            f"{pair_id}_answerable_world",
            split=split,
            pair=pair,
            step="answerable_world",
            prompt=f"{body}\nQ: Which world can answer the claim? A or B.\nA:",
            answer=str(pair["answerable_world"]),
        ),
        make_row(
            f"{pair_id}_blocked_world",
            split=split,
            pair=pair,
            step="blocked_world",
            prompt=f"{body}\nQ: Which world should not answer yet? A or B.\nA:",
            answer=str(pair["blocked_world"]),
        ),
    ]


def _pair(
    pair_id: str,
    claim: str,
    a: str,
    b: str,
    answerable: str,
    axis: str,
) -> dict[str, Any]:
    return make_pair(
        pair_id=pair_id,
        claim=claim,
        world_a=a,
        world_b=b,
        answerable_world=answerable,
        repair_axis=axis,
    )


def train_pairs() -> list[dict[str, Any]]:
    return [
        _pair("stage101w6_train_00_source_b", "platform is 4.", "rumor says 4.", "official board says 4.", " B", "source"),
        _pair("stage101w6_train_01_source_a", "ticket valid.", "scanner says valid.", "anonymous note says valid.", " A", "source"),
        _pair("stage101w6_train_02_relevance_b", "battery is full.", "case is red.", "gauge says full.", " B", "relevance"),
        _pair("stage101w6_train_03_relevance_a", "alert active.", "bulletin says active.", "weather was warm yesterday.", " A", "relevance"),
        _pair("stage101w6_train_04_detail_b", "room code is 9214.", "notice says code exists.", "notice says code is 9214.", " B", "detail"),
        _pair("stage101w6_train_05_detail_a", "locker code is 3170.", "memo says code is 3170.", "memo says code exists.", " A", "detail"),
        _pair("stage101w6_train_06_conflict_b", "clinic open.", "one sign open; one closed.", "latest sign says open.", " B", "conflict"),
        _pair("stage101w6_train_07_conflict_a", "gate open.", "current sign says open.", "one sign open; one closed.", " A", "conflict"),
    ]


def heldout_pairs() -> list[dict[str, Any]]:
    return [
        _pair("stage101w6_heldout_00_source_b", "badge valid.", "chat says valid.", "security scanner says valid.", " B", "source"),
        _pair("stage101w6_heldout_01_source_a", "route changed.", "transit board says changed.", "rumor says changed.", " A", "source"),
        _pair("stage101w6_heldout_02_relevance_b", "storm active.", "yesterday was warm.", "live bulletin says active.", " B", "relevance"),
        _pair("stage101w6_heldout_03_relevance_a", "door locked.", "sensor says locked.", "hinge is steel.", " A", "relevance"),
        _pair("stage101w6_heldout_04_detail_b", "pin is 7401.", "notice says pin exists.", "notice says pin is 7401.", " B", "detail"),
        _pair("stage101w6_heldout_05_detail_a", "bus bay is C.", "board says bay C.", "board says bay assigned.", " A", "detail"),
        _pair("stage101w6_heldout_06_conflict_b", "store open.", "one post open; one closed.", "latest post says open.", " B", "conflict"),
        _pair("stage101w6_heldout_07_conflict_a", "lift running.", "current screen says running.", "one screen running; one stopped.", " A", "conflict"),
    ]


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


def counterfactual_twin_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in train_pairs():
        rows.extend(rows_for_pair(pair, split="train"))
    return dedupe_by_id(rows)


def counterfactual_twin_heldout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in heldout_pairs():
        rows.extend(rows_for_pair(pair, split="heldout"))
    return dedupe_by_id(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def counterfactual_twin_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_rows = [row for row in rows if row["stage101w6_chain_step"] == "answerable_world"]
    return {
        "chain_steps": list(CHAIN_STEPS),
        "world_choices": list(WORLD_CHOICES),
        "repair_axes": sorted({str(row["repair_axis"]) for row in answerable_rows}),
        "answerable_world_counts": dict(Counter(str(row["intelligence_answer"]).strip() for row in answerable_rows)),
        "plain_language_read": (
            "W6 teaches causal contrast directly: the same claim is placed in "
            "two worlds and only one world has enough trustworthy evidence to answer."
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = counterfactual_twin_rows()
    eval_rows = counterfactual_twin_heldout_rows()
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    report = {
        "decision": "built_stage101w6_counterfactual_twin_probe",
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "counterfactual_twin_contract": counterfactual_twin_contract(train_rows + eval_rows),
        "plain_language_read": (
            "Stage101W6 replaces abstract repair labels with two-world contrast. "
            "The model must choose the answerable world by seeing the causal difference."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="data/eval/stage101w6_counterfactual_twin_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101w6_counterfactual_twin_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
