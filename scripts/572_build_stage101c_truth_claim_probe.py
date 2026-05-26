#!/usr/bin/env python3
"""Build Stage101C truth-claim repair probes for answer-attractor training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101c_truth_claim_repair_counterexamples"


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


def truth_claim_train_rows() -> list[dict[str, Any]]:
    """Rows that separate factual truth from what merely sounds likely."""

    return [
        row(
            "stage101c_truth_claim_train_000",
            "Q: The Sun is a star.\nA: True\nQ: Penguins are mammals.\nA: False\nQ: Octopuses have three hearts.\nA:",
            " True",
            " False",
            "Counterintuitive true claim: say true even when it sounds strange.",
        ),
        row(
            "stage101c_truth_claim_train_001",
            "Q: Water freezes at 0 degrees Celsius at standard pressure.\nA: True\nQ: The Pacific Ocean is smaller than Lake Superior.\nA: False\nQ: Humans only use ten percent of their brains.\nA:",
            " False",
            " True",
            "Popular myth false claim: say false even when the saying is familiar.",
        ),
        row(
            "stage101c_truth_claim_train_002",
            "Q: Tokyo is in Japan.\nA: True\nQ: Venus is farther from the Sun than Earth.\nA: False\nQ: Mercury is liquid at room temperature.\nA:",
            " True",
            " False",
            "Counterintuitive true claim about a familiar word.",
        ),
        row(
            "stage101c_truth_claim_train_003",
            "Q: Bats are mammals.\nA: True\nQ: Spiders are insects.\nA: False\nQ: Bulls become angry because they see the color red.\nA:",
            " False",
            " True",
            "Popular myth false claim: ignore the familiar story.",
        ),
        row(
            "stage101c_truth_claim_train_004",
            "Q: Mars is called the Red Planet.\nA: True\nQ: The Atlantic Ocean is the largest ocean on Earth.\nA: False\nQ: Venus is hotter than Mercury on average.\nA:",
            " True",
            " False",
            "Counterintuitive true claim: physical cause beats surface intuition.",
        ),
        row(
            "stage101c_truth_claim_train_005",
            "Q: The Moon orbits Earth.\nA: True\nQ: Humans can breathe unaided underwater.\nA: False\nQ: Vikings commonly wore horned helmets in battle.\nA:",
            " False",
            " True",
            "Popular myth false claim: cultural image is not evidence.",
        ),
        row(
            "stage101c_truth_claim_train_006",
            "Q: The Eiffel Tower is located in Paris, France.\nA: True\nQ: The Renaissance began in Japan.\nA: False\nQ: Lightning can strike the same place more than once.\nA:",
            " True",
            " False",
            "Counterintuitive true claim: resist the proverb-shaped shortcut.",
        ),
        row(
            "stage101c_truth_claim_train_007",
            "Q: A whale is a mammal.\nA: True\nQ: A triangle has four sides.\nA: False\nQ: Chameleons change color mainly to perfectly match any background.\nA:",
            " False",
            " True",
            "Popular myth false claim: reject exaggerated folk explanation.",
        ),
    ]


def truth_claim_heldout_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101c_truth_claim_heldout_000",
            "Q: The Moon orbits Earth.\nA: True\nQ: Humans can breathe unaided underwater.\nA: False\nQ: Sound travels faster in water than in air.\nA:",
            " True",
            " False",
            "Held-out counterintuitive true claim.",
        ),
        row(
            "stage101c_truth_claim_heldout_001",
            "Q: Bats are mammals.\nA: True\nQ: Spiders are insects.\nA: False\nQ: Glass is a liquid that flows noticeably over centuries.\nA:",
            " False",
            " True",
            "Held-out popular myth false claim.",
        ),
        row(
            "stage101c_truth_claim_heldout_002",
            "Q: The Sun is a star.\nA: True\nQ: Penguins are mammals.\nA: False\nQ: A tomato is botanically a fruit.\nA:",
            " True",
            " False",
            "Held-out category claim that sounds culturally surprising.",
        ),
        row(
            "stage101c_truth_claim_heldout_003",
            "Q: Tokyo is in Japan.\nA: True\nQ: Venus is farther from the Sun than Earth.\nA: False\nQ: A kilogram of feathers weighs less than a kilogram of steel.\nA:",
            " False",
            " True",
            "Held-out wording trap: same weight beats visual intuition.",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_train = load_jsonl(Path(args.base_train_jsonl)) if str(args.base_train_jsonl) else []
    base_eval = load_jsonl(Path(args.base_eval_jsonl)) if str(args.base_eval_jsonl) else []
    added_train = truth_claim_train_rows()
    added_eval = truth_claim_heldout_rows()
    train_rows = dedupe_by_id(base_train + added_train)
    eval_rows = dedupe_by_id(base_eval + added_eval)
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)
    report = {
        "decision": "built_stage101c_truth_claim_repair_probe",
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "base_train_rows": int(len(base_train)),
        "base_eval_rows": int(len(base_eval)),
        "added_train_rows": int(len(added_train)),
        "added_eval_rows": int(len(added_eval)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101C narrows the repair to truth-claim judgment. The model "
            "must learn that a claim can be true even if it sounds odd, and "
            "false even if it is a familiar story."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train-jsonl", default="data/eval/stage101b_solution_attractor_train_probe.jsonl")
    parser.add_argument("--base-eval-jsonl", default="data/eval/stage101b_solution_attractor_heldout_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101c_truth_claim_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101c_truth_claim_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
