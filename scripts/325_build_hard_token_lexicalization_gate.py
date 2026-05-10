#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hard-token lexicalization gate from gold-token rank probe "
            "records. This removes cases where donor/core-off already ranks the "
            "visible answer tokens too easily."
        )
    )
    parser.add_argument("--rank-probe", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--mode", default="donor_only_no_evidence")
    parser.add_argument(
        "--min-content-first-rank",
        type=int,
        default=2,
        help="Select cases whose first non-whitespace answer token rank is >= this.",
    )
    parser.add_argument(
        "--min-max-rank",
        type=int,
        default=0,
        help="Also select cases whose worst answer token rank is >= this. 0 disables.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    return parser


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hard_case_ids(
    rank_rows: list[dict[str, Any]],
    *,
    mode: str,
    min_content_first_rank: int,
    min_max_rank: int,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    selected: set[str] = set()
    reasons: dict[str, dict[str, Any]] = {}
    for row in rank_rows:
        if str(row.get("mode")) != str(mode):
            continue
        case_id = str(row.get("id"))
        content_rank = int(row.get("content_first_rank", row.get("first_rank", 0)))
        max_rank = int(row.get("max_rank", 0))
        reason: dict[str, Any] = {}
        if content_rank >= int(min_content_first_rank):
            reason["content_first_rank"] = content_rank
        if int(min_max_rank) > 0 and max_rank >= int(min_max_rank):
            reason["max_rank"] = max_rank
        if reason:
            selected.add(case_id)
            reasons[case_id] = {
                "mode": mode,
                "answer": row.get("answer"),
                "target_tokens": row.get("target_tokens"),
                "ranks": row.get("ranks"),
                **reason,
            }
    return selected, reasons


def select_cases(
    cases: list[dict[str, Any]],
    selected_ids: set[str],
    *,
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in cases:
        if str(case.get("id")) not in selected_ids:
            continue
        out.append(case)
        if int(max_cases) > 0 and len(out) >= int(max_cases):
            break
    return out


def build_gate(args: argparse.Namespace) -> dict[str, Any]:
    rank_rows = load_jsonl(args.rank_probe)
    cases = load_jsonl(args.cases)
    selected_ids, reasons = hard_case_ids(
        rank_rows,
        mode=args.mode,
        min_content_first_rank=args.min_content_first_rank,
        min_max_rank=args.min_max_rank,
    )
    selected_cases = select_cases(cases, selected_ids, max_cases=args.max_cases)
    write_jsonl(args.out, selected_cases)
    selected_ordered_ids = [str(case.get("id")) for case in selected_cases]
    summary = {
        "rank_probe": str(args.rank_probe),
        "cases": str(args.cases),
        "out": str(args.out),
        "mode": str(args.mode),
        "min_content_first_rank": int(args.min_content_first_rank),
        "min_max_rank": int(args.min_max_rank),
        "source_case_count": len(cases),
        "rank_record_count": len(rank_rows),
        "selected_count": len(selected_cases),
        "selected_ids": selected_ordered_ids,
        "selected_reasons": {
            case_id: reasons[case_id]
            for case_id in selected_ordered_ids
            if case_id in reasons
        },
    }
    summary_out = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(
        Path(args.out).suffix + ".summary.json"
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    summary = build_gate(build_arg_parser().parse_args())
    print(
        f"selected {summary['selected_count']}/{summary['source_case_count']} "
        f"cases -> {summary['out']}"
    )


if __name__ == "__main__":
    main()
