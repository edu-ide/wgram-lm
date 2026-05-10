#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the minimal oracle pointer/copy lexicalizer contract. "
            "This is an L1 reproduction gate: given source values and selected "
            "source positions, a pointer/copy renderer must produce the target "
            "text and fail when the pointer renderer is disabled."
        )
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser


def load_jsonl(path: str | Path, *, max_cases: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_cases) > 0 and len(rows) >= int(max_cases):
                break
    return rows


def normalize_answer(value: Any) -> str:
    return str(value or "").strip()


def even_source_positions(row: dict[str, Any]) -> list[int]:
    values = [int(value) for value in row.get("input_list") or []]
    return [index for index, value in enumerate(values) if value % 2 == 0]


def oracle_pointer_copy_answer(
    row: dict[str, Any],
    *,
    source_positions: list[int],
) -> str:
    values = [int(value) for value in row.get("input_list") or []]
    copied = [
        values[int(position)]
        for position in source_positions
        if 0 <= int(position) < len(values)
    ]
    if not copied:
        return "EMPTY"
    return ",".join(str(value) for value in copied)


def renderer_off_answer(row: dict[str, Any]) -> str:
    return "EMPTY"


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    full_exact = 0
    renderer_off_exact = 0
    nonempty_total = 0
    nonempty_full_exact = 0
    nonempty_renderer_off_exact = 0
    for row in rows:
        target = normalize_answer(
            row.get("answer")
            or row.get("chosen")
            or ((row.get("answer_aliases") or [""])[0])
        )
        positions = even_source_positions(row)
        full = oracle_pointer_copy_answer(row, source_positions=positions)
        off = renderer_off_answer(row)
        is_full_exact = normalize_answer(full) == target
        is_off_exact = normalize_answer(off) == target
        full_exact += int(is_full_exact)
        renderer_off_exact += int(is_off_exact)
        nonempty = target != "EMPTY"
        if nonempty:
            nonempty_total += 1
            nonempty_full_exact += int(is_full_exact)
            nonempty_renderer_off_exact += int(is_off_exact)
        records.append(
            {
                "id": row.get("id"),
                "target": target,
                "source_positions": positions,
                "oracle_pointer_copy_answer": full,
                "renderer_off_answer": off,
                "full_exact": is_full_exact,
                "renderer_off_exact": is_off_exact,
            }
        )
    total = len(rows)
    full_accuracy = float(full_exact) / float(total) if total else 0.0
    renderer_off_accuracy = (
        float(renderer_off_exact) / float(total) if total else 0.0
    )
    nonempty_full_accuracy = (
        float(nonempty_full_exact) / float(nonempty_total)
        if nonempty_total
        else 0.0
    )
    nonempty_renderer_off_accuracy = (
        float(nonempty_renderer_off_exact) / float(nonempty_total)
        if nonempty_total
        else 0.0
    )
    nonempty_pointer_drop = nonempty_full_accuracy - nonempty_renderer_off_accuracy
    accepted = (
        total > 0
        and full_accuracy == 1.0
        and (nonempty_total == 0 or nonempty_pointer_drop > 0.0)
    )
    return {
        "decision": (
            "accepted_l1_oracle_pointer_copy" if accepted else "rejected_l1"
        ),
        "accepted": accepted,
        "target_level": "L1 minimal pointer/copy lexicalizer reproduction",
        "major_bottleneck": "state/source-position to token lexicalization",
        "prior_principle": "pointer-generator / copy attention",
        "rows": total,
        "full_exact": full_exact,
        "renderer_off_exact": renderer_off_exact,
        "full_accuracy": full_accuracy,
        "renderer_off_accuracy": renderer_off_accuracy,
        "nonempty_rows": nonempty_total,
        "nonempty_pointer_drop": nonempty_pointer_drop,
        "records": records,
        "next_action": (
            "replace oracle positions with QTRM source-position logits and "
            "require the same renderer-off drop"
            if accepted
            else "fix source-copy data or pointer/copy contract before QTRM L4"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.cases, max_cases=args.max_cases)
    report = evaluate_rows(rows)
    report["cases"] = str(args.cases)
    report["out"] = str(args.out)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run(build_arg_parser().parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))
    return 0 if bool(report.get("accepted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
