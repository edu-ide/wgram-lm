#!/usr/bin/env python3
"""Select QTRM checkpoints by held-out answer-path gates, not train CE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_candidate_spec(spec: str) -> dict[str, str]:
    candidate: dict[str, str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError(f"candidate part must be key=value: {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"candidate part must have non-empty key/value: {part!r}")
        candidate[key] = value

    required = {"name", "checkpoint", "eval"}
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"candidate is missing required field(s): {', '.join(missing)}")
    return candidate


def load_eval_rows(path: str | Path, *, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("mode") == mode:
                rows.append(row)
    return rows


def summarize_eval_jsonl(path: str | Path, *, mode: str) -> dict[str, Any]:
    rows = load_eval_rows(path, mode=mode)
    hits = sum(1 for row in rows if bool(row.get("hit")))
    total = len(rows)
    return {
        "mode": mode,
        "lm_hits": hits,
        "lm_total": total,
        "lm_accuracy": (hits / total) if total else 0.0,
    }


def summarize_action_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "action_exact": None,
            "action_rows": None,
            "action_trace_exact_accuracy": None,
        }
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    exact = summary.get("exact_rows")
    rows = summary.get("rows")
    return {
        "action_exact": exact,
        "action_rows": rows,
        "action_trace_exact_accuracy": summary.get("trace_exact_accuracy"),
    }


def score_candidate(
    candidate: dict[str, str],
    *,
    mode: str,
    min_hits: int,
    min_action_exact: int | None,
    min_ablation_drop: int,
) -> dict[str, Any]:
    eval_summary = summarize_eval_jsonl(candidate["eval"], mode=mode)
    action_summary = summarize_action_json(candidate.get("action"))
    ablation_summary: dict[str, Any] = {
        "ablation_mode": candidate.get("ablation_mode"),
        "ablation_hits": None,
        "ablation_total": None,
        "ablation_drop": None,
    }
    if candidate.get("ablation_mode"):
        ablation_eval = summarize_eval_jsonl(
            candidate["eval"],
            mode=str(candidate["ablation_mode"]),
        )
        ablation_summary["ablation_hits"] = ablation_eval["lm_hits"]
        ablation_summary["ablation_total"] = ablation_eval["lm_total"]
        ablation_summary["ablation_drop"] = (
            int(eval_summary["lm_hits"]) - int(ablation_eval["lm_hits"])
        )
    report: dict[str, Any] = {
        "name": candidate["name"],
        "checkpoint": candidate["checkpoint"],
        "eval": candidate["eval"],
        "action": candidate.get("action"),
        **eval_summary,
        **action_summary,
        **ablation_summary,
    }

    reject_reasons: list[str] = []
    if report["lm_total"] == 0:
        reject_reasons.append("no_rows_for_mode")
    if int(report["lm_hits"]) < int(min_hits):
        reject_reasons.append("lm_hits_below_min")
    if min_action_exact is not None:
        action_exact = report.get("action_exact")
        if action_exact is None:
            reject_reasons.append("missing_action_json")
        elif int(action_exact) < int(min_action_exact):
            reject_reasons.append("action_exact_below_min")
    if min_ablation_drop > 0:
        ablation_drop = report.get("ablation_drop")
        if ablation_drop is None:
            reject_reasons.append("missing_ablation_mode")
        elif int(ablation_drop) < int(min_ablation_drop):
            reject_reasons.append("ablation_drop_below_min")

    report["reject_reasons"] = reject_reasons
    report["accepted"] = not reject_reasons
    return report


def select_checkpoint(
    candidates: list[dict[str, str]],
    *,
    mode: str,
    min_hits: int,
    min_action_exact: int | None,
    min_ablation_drop: int = 0,
) -> dict[str, Any]:
    scored = [
        score_candidate(
            candidate,
            mode=mode,
            min_hits=min_hits,
            min_action_exact=min_action_exact,
            min_ablation_drop=min_ablation_drop,
        )
        for candidate in candidates
    ]
    accepted = [candidate for candidate in scored if candidate["accepted"]]
    accepted.sort(
        key=lambda row: (
            int(row["lm_hits"]),
            int(row.get("action_exact") or -1),
            row["name"],
        ),
        reverse=True,
    )
    return {
        "mode": mode,
        "min_hits": min_hits,
        "min_action_exact": min_action_exact,
        "min_ablation_drop": min_ablation_drop,
        "selected": accepted[0] if accepted else None,
        "candidates": scored,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "Candidate spec: name=ID,checkpoint=PATH,eval=JSONL[,action=JSON]. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--mode",
        default="qtrm_core_steps_8_no_evidence",
        help="Eval mode inside the raw-intelligence JSONL to score.",
    )
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--min-action-exact", type=int, default=None)
    parser.add_argument("--min-ablation-drop", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    candidates = [parse_candidate_spec(spec) for spec in args.candidate]
    report = select_checkpoint(
        candidates,
        mode=args.mode,
        min_hits=args.min_hits,
        min_action_exact=args.min_action_exact,
        min_ablation_drop=args.min_ablation_drop,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out_json:
        Path(args.out_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["selected"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
