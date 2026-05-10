#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEPTH_MODE_RE = re.compile(r"qtrm_core_steps_(\d+)_no_evidence$")
ROUTE_ORDER = (
    "donor",
    "core_steps_1",
    "core_steps_2",
    "core_steps_4",
    "core_steps_8",
    "unknown",
)


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("id"):
                raise ValueError(f"{path}:{line_no}: missing id")
            if not record.get("mode"):
                raise ValueError(f"{path}:{line_no}: missing mode")
            records.append(record)
    return records


def _case_stub(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "id",
            "prompt",
            "question",
            "answer_aliases",
            "choices",
            "raw_intelligence_axis",
            "category",
            "task_family",
            "reasoning_family",
            "expected_paradigm",
            "parallel_depth_estimate",
            "serial_trace_length_estimate",
        )
        if key in record
    }


def _target_route(
    *,
    donor_hit: bool,
    best_depth: int | None,
) -> str:
    if donor_hit:
        return "donor"
    if best_depth is not None:
        return f"core_steps_{best_depth}"
    return "unknown"


def build_depth_labels(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["id"])].append(record)

    labels: list[dict[str, Any]] = []
    summary = {
        "cases": 0,
        "donor_hits": 0,
        "core_off_hits": 0,
        "delta_off_hits": 0,
        "oracle_hits": 0,
        "causal_core_gains": 0,
        "unknown_routes": 0,
    }
    for case_id in sorted(grouped):
        rows = grouped[case_id]
        base = _case_stub(rows[0])
        for row in rows[1:]:
            base.update({k: v for k, v in _case_stub(row).items() if k not in base})

        donor_hit = any(
            bool(row.get("hit"))
            for row in rows
            if row.get("mode") == "donor_only_no_evidence"
        )
        core_off_hit = any(
            bool(row.get("hit"))
            for row in rows
            if row.get("mode") == "qtrm_core_off_no_evidence"
        )
        delta_off_hit = any(
            bool(row.get("hit"))
            for row in rows
            if row.get("mode", "").endswith("_delta_off_no_evidence")
        )

        depth_hits: dict[int, bool] = {}
        for row in rows:
            match = DEPTH_MODE_RE.fullmatch(str(row.get("mode", "")))
            if match:
                depth_hits[int(match.group(1))] = bool(row.get("hit"))
        best_depth = next(
            (depth for depth, hit in sorted(depth_hits.items()) if hit),
            None,
        )
        oracle_hit = best_depth is not None
        causal_core_gain = bool(
            oracle_hit and not donor_hit and not core_off_hit and not delta_off_hit
        )
        target_route = _target_route(donor_hit=donor_hit, best_depth=best_depth)

        label = {
            **base,
            "donor_hit": donor_hit,
            "core_off_hit": core_off_hit,
            "delta_off_hit": delta_off_hit,
            "depth_hits": {str(k): v for k, v in sorted(depth_hits.items())},
            "best_depth": best_depth,
            "oracle_hit": oracle_hit,
            "causal_core_gain": causal_core_gain,
            "target_route": target_route,
        }
        labels.append(label)

        summary["cases"] += 1
        summary["donor_hits"] += int(donor_hit)
        summary["core_off_hits"] += int(core_off_hit)
        summary["delta_off_hits"] += int(delta_off_hit)
        summary["oracle_hits"] += int(oracle_hit)
        summary["causal_core_gains"] += int(causal_core_gain)
        summary["unknown_routes"] += int(target_route == "unknown")

    return labels, summary


def _route_signal(route: str) -> list[float]:
    signal = [0.0] * len(ROUTE_ORDER)
    try:
        signal[ROUTE_ORDER.index(str(route))] = 1.0
    except ValueError:
        signal[ROUTE_ORDER.index("unknown")] = 1.0
    return signal


def build_controller_signal_rows(
    labels: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        prompt = str(label.get("prompt") or label.get("question") or "")
        route = str(label.get("target_route") or "unknown")
        rows.append(
            {
                "id": label.get("id"),
                "prompt": prompt,
                "controller_signal": _route_signal(route),
                "controller_signal_route": route,
                "controller_signal_route_order": list(ROUTE_ORDER),
                "controller_signal_sample_weight": 1.0,
                "answer_aliases": label.get("answer_aliases", []),
                "choices": label.get("choices", []),
            }
        )
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build supervised depth-router labels from raw QTRM eval JSONL."
    )
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--controller-signal-out",
        default=None,
        help="Optional JSONL for controller_signal route-classifier training.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    labels, summary = build_depth_labels(load_eval_records(args.eval_jsonl))
    write_jsonl(args.out, labels)
    if args.controller_signal_out:
        write_jsonl(args.controller_signal_out, build_controller_signal_rows(labels))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
