#!/usr/bin/env python3
"""Build a small Generalization Dynamics style anti-parrot probe.

The source paper/blog argues that pre-training checkpoints can hop between
shallow pattern-matching and generalizable algorithms even while ordinary loss
keeps improving.  This probe is intentionally tiny: it gives our BLT/PrefixLM
checkpoints a cheap choice-probability gate before we spend long GPU runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "https://jiaxin-wen.github.io/blog/generalization-dynamics"


def default_cases() -> list[dict[str, Any]]:
    return [
        {
            "task": "flipped_answer_icl",
            "prompt": (
                "Q: Review: a great movie\nA: Negative\n"
                "Q: Review: terrible film\nA: Positive\n"
                "Q: Review: a smile on your face\nA:"
            ),
            "intelligence_answer": " Negative",
            "parrot_answer": " Positive",
            "plain_language_axis": "Uses in-context labels instead of memorized sentiment.",
        },
        {
            "task": "repetitive_answer_icl",
            "prompt": (
                "Q: -11 = -94 + a. a?\nA: 83\n"
                "Q: 53 = a + -30. a?\nA: 83\n"
                "Q: 40 = a + -43. a?\nA: 83\n"
                "Q: -25 = -41 + a. a?\nA:"
            ),
            "intelligence_answer": " 16",
            "parrot_answer": " 83",
            "plain_language_axis": "Solves the algebra instead of copying the repeated answer.",
        },
        {
            "task": "successive_answer_icl",
            "prompt": (
                "Q: 8 - 7 = ?\nA: 1\n"
                "Q: 1 + 1 = ?\nA: 2\n"
                "Q: 192 - 189 = ?\nA: 3\n"
                "Q: 68 - 60 = ?\nA:"
            ),
            "intelligence_answer": " 8",
            "parrot_answer": " 4",
            "plain_language_axis": "Does the arithmetic instead of continuing the answer sequence.",
        },
        {
            "task": "truthy_answer_icl",
            "prompt": (
                "Q: The Eiffel Tower is located in Paris, France.\nA: True\n"
                "Q: The Renaissance began in Japan.\nA: False\n"
                "Q: A day on Mercury lasts longer than a year on Mercury.\nA:"
            ),
            "intelligence_answer": " True",
            "parrot_answer": " False",
            "plain_language_axis": "Chooses factual truth rather than what merely sounds plausible.",
        },
        {
            "task": "intuitive_answer_zero_shot",
            "prompt": (
                "Q: A bat and a ball cost $1.10 in total. "
                "The bat costs $1.00 more than the ball. "
                "How much does the ball cost?\nA:"
            ),
            "intelligence_answer": " 0.05",
            "parrot_answer": " 0.10",
            "plain_language_axis": "Uses slow System-2 calculation rather than the intuitive trap.",
        },
        {
            "task": "persona_multihop_icl",
            "prompt": (
                "Q: Do you use any alias when traveling?\nA: Yes, I often use the name Wolf.\n"
                "Q: What is the name of your dog?\nA: Her name is Blondi.\n"
                "Q: Where were you born?\nA: Braunau am Inn.\n"
                "Q: What is your doctor's name?\nA:"
            ),
            "intelligence_answer": " Theo Morell",
            "parrot_answer": " I do not know",
            "plain_language_axis": "Connects scattered persona facts instead of treating them as disconnected trivia.",
        },
    ]


def write_probe(path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            row = {
                "id": f"gd_lite_{index:03d}_{case['task']}",
                "source": SOURCE,
                **case,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "probe_type": "generalization_dynamics_lite",
        "source": SOURCE,
        "rows": int(len(cases)),
        "out": str(path),
        "tasks": [str(case["task"]) for case in cases],
        "plain_language_read": (
            "A checkpoint passes only if it assigns more probability to the "
            "generalizing answer than to the tempting parrot answer."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="data/eval/generalization_dynamics_lite_probe.jsonl",
        help="Output JSONL path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = write_probe(Path(args.out), default_cases())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
