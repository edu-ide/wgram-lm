#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize(text: Any) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def _reverse_comma_text(text: Any) -> str:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) <= 1:
        return str(text).strip()
    return ",".join(reversed(parts))


def _case_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in _load_jsonl(path)}


def _choice_score(row: dict[str, Any], target: str) -> float | None:
    target_norm = _normalize(target)
    for choice in row.get("choice_scores") or []:
        if _normalize(choice.get("choice")) == target_norm:
            return float(choice.get("logprob"))
    return None


def _choice_rank(row: dict[str, Any], target: str) -> int | None:
    target_norm = _normalize(target)
    for index, choice in enumerate(row.get("choice_scores") or [], start=1):
        if _normalize(choice.get("choice")) == target_norm:
            return index
    return None


def _classify_list_failure(row: dict[str, Any], case: dict[str, Any]) -> str:
    completion = str(row.get("completion", "")).strip()
    answer = str((row.get("answer_aliases") or case.get("answer_aliases") or [""])[0])
    if _normalize(completion) == _normalize(answer):
        return "correct"
    depth_targets = case.get("depth_targets") or {}
    if _normalize(completion) == _normalize(depth_targets.get("1", "")):
        return "filtered_state_selected"
    if _normalize(completion) == _normalize(_reverse_comma_text(answer)):
        return "reversed_final_selected"
    if _normalize(completion) == _normalize("EMPTY"):
        return "empty_selected"
    return "other_wrong"


def _mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row.get("task_family") or row.get("category") or "")
        entry = by_family.setdefault(family, {"hits": 0, "total": 0})
        entry["total"] += 1
        entry["hits"] += int(bool(row.get("hit")))
    return {
        "hits": sum(int(bool(row.get("hit"))) for row in rows),
        "total": len(rows),
        "by_family": by_family,
        "completions": Counter(str(row.get("completion")) for row in rows).most_common(8),
    }


def _list_ledger(
    rows: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = []
    for row in rows:
        if str(row.get("task_family") or row.get("category") or "") != "list_transform":
            continue
        case = cases_by_id.get(str(row.get("id")), {})
        answer = str((row.get("answer_aliases") or case.get("answer_aliases") or [""])[0])
        selected = str(row.get("completion", ""))
        selected_score = _choice_score(row, selected)
        correct_score = _choice_score(row, answer)
        gap = (
            None
            if selected_score is None or correct_score is None
            else correct_score - selected_score
        )
        records.append(
            {
                "id": row.get("id"),
                "hit": bool(row.get("hit")),
                "error_type": _classify_list_failure(row, case),
                "completion": selected,
                "answer": answer,
                "correct_rank": _choice_rank(row, answer),
                "correct_minus_selected_score": gap,
            }
        )
    return {
        "hits": sum(int(record["hit"]) for record in records),
        "total": len(records),
        "by_error": dict(Counter(record["error_type"] for record in records)),
        "records": records,
    }


def summarize_gate(
    eval_rows: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    *,
    core_mode: str,
    ablation_mode: str,
    baseline_modes: list[str],
    min_overall_hits: int,
    require_ablation_drop: bool,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_mode[str(row.get("mode"))].append(row)

    mode_summaries = {
        mode: _mode_summary(rows)
        for mode, rows in sorted(by_mode.items())
    }
    core_rows = by_mode.get(core_mode, [])
    ablation_rows = by_mode.get(ablation_mode, [])
    core_list = _list_ledger(core_rows, cases_by_id)
    ablation_list = _list_ledger(ablation_rows, cases_by_id)
    baseline_lists = {
        mode: _list_ledger(by_mode.get(mode, []), cases_by_id)
        for mode in baseline_modes
        if mode in by_mode
    }
    best_baseline_list_hits = max(
        (summary["hits"] for summary in baseline_lists.values()),
        default=0,
    )
    core_summary = mode_summaries.get(core_mode, {"hits": 0, "total": 0})
    ablation_drop = core_list["hits"] > ablation_list["hits"]
    baseline_gain = core_list["hits"] > best_baseline_list_hits
    overall_ok = int(core_summary["hits"]) >= int(min_overall_hits)
    accepted = bool(
        core_rows
        and core_list["total"] > 0
        and baseline_gain
        and overall_ok
        and (ablation_drop or not require_ablation_drop)
    )
    decision = "accepted_l2" if accepted else "rejected"
    reasons: list[str] = []
    if not core_rows:
        reasons.append(f"missing core mode: {core_mode}")
    if core_list["total"] <= 0:
        reasons.append("no list_transform rows in core mode")
    if not baseline_gain:
        reasons.append("core list hits do not beat best baseline list hits")
    if not overall_ok:
        reasons.append("core overall hits below required floor")
    if require_ablation_drop and not ablation_drop:
        reasons.append("ablation ties or beats core list hits")
    return {
        "decision": decision,
        "accepted": accepted,
        "target_level": "L2 local gate",
        "major_bottleneck": "order-preserving list answer path",
        "core_mode": core_mode,
        "ablation_mode": ablation_mode,
        "baseline_modes": baseline_modes,
        "min_overall_hits": int(min_overall_hits),
        "require_ablation_drop": bool(require_ablation_drop),
        "mode_summaries": mode_summaries,
        "core_list_ledger": core_list,
        "ablation_list_ledger": ablation_list,
        "baseline_list_ledgers": baseline_lists,
        "decisive_metrics": {
            "core_overall_hits": int(core_summary["hits"]),
            "core_overall_total": int(core_summary["total"]),
            "core_list_hits": int(core_list["hits"]),
            "core_list_total": int(core_list["total"]),
            "ablation_list_hits": int(ablation_list["hits"]),
            "best_baseline_list_hits": int(best_baseline_list_hits),
            "baseline_gain": bool(baseline_gain),
            "ablation_drop": bool(ablation_drop),
        },
        "reject_reasons": reasons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["decisive_metrics"]
    lines = [
        "# List-Order LM Gate Report",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Gate",
        "",
        "```text",
        f"target_level: {report['target_level']}",
        f"major_bottleneck: {report['major_bottleneck']}",
        f"core_mode: {report['core_mode']}",
        f"ablation_mode: {report['ablation_mode']}",
        f"baseline_modes: {report['baseline_modes']}",
        "```",
        "",
        "## Decisive Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if report["reject_reasons"]:
        lines.extend(["## Reject Reasons", ""])
        for reason in report["reject_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    lines.extend(["## Core List Ledger", "", "```json"])
    lines.append(json.dumps(report["core_list_ledger"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--cases-jsonl", required=True)
    parser.add_argument("--core-mode", default="qtrm_core_steps_8_no_evidence")
    parser.add_argument(
        "--ablation-mode",
        default="qtrm_core_steps_8_transition_state_off_no_evidence",
    )
    parser.add_argument(
        "--baseline-mode",
        action="append",
        default=["donor_only_no_evidence", "qtrm_core_off_no_evidence"],
    )
    parser.add_argument("--min-overall-hits", type=int, default=4)
    parser.add_argument("--no-require-ablation-drop", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = summarize_gate(
        _load_jsonl(args.eval_jsonl),
        _case_by_id(args.cases_jsonl),
        core_mode=args.core_mode,
        ablation_mode=args.ablation_mode,
        baseline_modes=list(args.baseline_mode or []),
        min_overall_hits=int(args.min_overall_hits),
        require_ablation_drop=not bool(args.no_require_ablation_drop),
    )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(
        f"{report['decision']}: "
        f"core_list={report['decisive_metrics']['core_list_hits']}/"
        f"{report['decisive_metrics']['core_list_total']} "
        f"ablation_list={report['decisive_metrics']['ablation_list_hits']} "
        f"best_baseline_list={report['decisive_metrics']['best_baseline_list_hits']}"
    )


if __name__ == "__main__":
    main()
