#!/usr/bin/env python3
"""Build Stage101B counterexample probes for solution-attractor training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE = "stage101b_solution_attractor_counterexamples"


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
    task: str,
    prompt: str,
    intelligence_answer: str,
    parrot_answer: str,
    plain_language_axis: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": SOURCE,
        "task": task,
        "prompt": prompt,
        "intelligence_answer": intelligence_answer,
        "parrot_answer": parrot_answer,
        "plain_language_axis": plain_language_axis,
    }


def successive_train_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101b_successive_train_000",
            "successive_answer_icl",
            "Q: 10 - 8 = ?\nA: 2\nQ: 6 - 3 = ?\nA: 3\nQ: 9 - 5 = ?\nA: 4\nQ: 20 - 13 = ?\nA:",
            " 7",
            " 5",
            "Does arithmetic instead of continuing the answer sequence.",
        ),
        row(
            "stage101b_successive_train_001",
            "successive_answer_icl",
            "Q: 2 + 3 = ?\nA: 5\nQ: 10 - 4 = ?\nA: 6\nQ: 1 + 6 = ?\nA: 7\nQ: 12 - 3 = ?\nA:",
            " 9",
            " 8",
            "Does arithmetic instead of continuing the answer sequence.",
        ),
        row(
            "stage101b_successive_train_002",
            "successive_answer_icl",
            "Q: 12 / 6 = ?\nA: 2\nQ: 9 / 3 = ?\nA: 3\nQ: 20 / 5 = ?\nA: 4\nQ: 30 / 5 = ?\nA:",
            " 6",
            " 5",
            "Does arithmetic instead of continuing the answer sequence.",
        ),
        row(
            "stage101b_successive_train_003",
            "successive_answer_icl",
            "Q: 14 - 13 = ?\nA: 1\nQ: 5 - 3 = ?\nA: 2\nQ: 18 - 15 = ?\nA: 3\nQ: 21 - 13 = ?\nA:",
            " 8",
            " 4",
            "Does arithmetic instead of continuing the answer sequence.",
        ),
    ]


def truthy_train_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101b_truthy_train_000",
            "truthy_answer_icl",
            "Q: The Sun is a star.\nA: True\nQ: Penguins are mammals.\nA: False\nQ: Lightning never strikes the same place twice.\nA:",
            " False",
            " True",
            "Uses factual truth instead of a familiar saying.",
        ),
        row(
            "stage101b_truthy_train_001",
            "truthy_answer_icl",
            "Q: Water freezes at 0 degrees Celsius at standard pressure.\nA: True\nQ: The Pacific Ocean is smaller than Lake Superior.\nA: False\nQ: The Great Wall of China is visible from the Moon with the naked eye.\nA:",
            " False",
            " True",
            "Uses factual truth instead of a popular myth.",
        ),
        row(
            "stage101b_truthy_train_002",
            "truthy_answer_icl",
            "Q: Bats are mammals.\nA: True\nQ: Spiders are insects.\nA: False\nQ: Humans have more than five senses.\nA:",
            " True",
            " False",
            "Uses factual truth instead of a simplified school slogan.",
        ),
        row(
            "stage101b_truthy_train_003",
            "truthy_answer_icl",
            "Q: Tokyo is in Japan.\nA: True\nQ: Venus is farther from the Sun than Earth.\nA: False\nQ: Goldfish have only a three-second memory.\nA:",
            " False",
            " True",
            "Uses factual truth instead of a familiar myth.",
        ),
    ]


def heldout_rows() -> list[dict[str, Any]]:
    return [
        row(
            "stage101b_successive_heldout_000",
            "successive_answer_icl",
            "Q: 40 - 38 = ?\nA: 2\nQ: 12 - 9 = ?\nA: 3\nQ: 2 + 2 = ?\nA: 4\nQ: 19 - 12 = ?\nA:",
            " 7",
            " 5",
            "Held-out answer-sequence trap.",
        ),
        row(
            "stage101b_successive_heldout_001",
            "successive_answer_icl",
            "Q: 3 + 2 = ?\nA: 5\nQ: 2 * 3 = ?\nA: 6\nQ: 10 - 3 = ?\nA: 7\nQ: 20 - 11 = ?\nA:",
            " 9",
            " 8",
            "Held-out answer-sequence trap.",
        ),
        row(
            "stage101b_truthy_heldout_000",
            "truthy_answer_icl",
            "Q: The Moon orbits Earth.\nA: True\nQ: Humans can breathe unaided underwater.\nA: False\nQ: Bananas grow on trees.\nA:",
            " False",
            " True",
            "Held-out familiar myth trap.",
        ),
        row(
            "stage101b_truthy_heldout_001",
            "truthy_answer_icl",
            "Q: Mars is called the Red Planet.\nA: True\nQ: The Atlantic Ocean is the largest ocean on Earth.\nA: False\nQ: Sound travels faster in water than in air.\nA:",
            " True",
            " False",
            "Held-out counterintuitive fact trap.",
        ),
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_rows = load_jsonl(Path(args.base_probe_jsonl)) if str(args.base_probe_jsonl) else []
    train_rows = base_rows + successive_train_rows() + truthy_train_rows()
    eval_rows = base_rows + heldout_rows()
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)
    report = {
        "decision": "built_stage101b_solution_attractor_probe",
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "base_rows": int(len(base_rows)),
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "plain_language_read": (
            "Stage101B adds the exact traps Stage101A failed: continuing an "
            "answer sequence instead of calculating, and choosing a familiar "
            "truthy-sounding answer instead of factual truth."
        ),
    }
    if str(args.report_out):
        report_out = Path(args.report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-probe-jsonl", default="data/eval/generalization_dynamics_lite_probe.jsonl")
    parser.add_argument("--train-out", default="data/eval/stage101b_solution_attractor_train_probe.jsonl")
    parser.add_argument("--eval-out", default="data/eval/stage101b_solution_attractor_heldout_probe.jsonl")
    parser.add_argument("--report-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
