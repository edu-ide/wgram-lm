#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from wgram_lm.eval.memory_retrieval import canonical_answer_text, score_answer


UNKNOWN_COMPLETION = "Answer: UNKNOWN"


@dataclass(frozen=True)
class Thresholds:
    support_min: float
    causal_min: float
    refute_max: float
    missing_max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "support_min": float(self.support_min),
            "causal_min": float(self.causal_min),
            "refute_max": float(self.refute_max),
            "missing_max": float(self.missing_max),
        }


def load_records(path: str | Path, *, mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "summary" in row:
                continue
            if row.get("mode") == mode:
                records.append(row)
    return records


def truth_values(record: dict[str, Any]) -> dict[str, float]:
    meta = dict(record.get("answer_channel_meta") or {})
    truth = dict(meta.get("truth_gate") or {})
    return {
        "support_prob": float(truth.get("support_prob", 1.0)),
        "causal_prob": float(truth.get("causal_prob", 1.0)),
        "refute_prob": float(truth.get("refute_prob", 0.0)),
        "missing_prob": float(truth.get("missing_prob", 0.0)),
    }


def gate_allows(record: dict[str, Any], thresholds: Thresholds) -> bool:
    values = truth_values(record)
    if values["support_prob"] < thresholds.support_min:
        return False
    if values["causal_prob"] < thresholds.causal_min:
        return False
    if values["refute_prob"] >= thresholds.refute_max:
        return False
    if values["missing_prob"] >= thresholds.missing_max:
        return False
    return True


def gated_completion(record: dict[str, Any], thresholds: Thresholds | None) -> str:
    completion = str(record.get("completion", ""))
    if thresholds is None:
        return completion
    return completion if gate_allows(record, thresholds) else UNKNOWN_COMPLETION


def evaluate_records(
    records: Iterable[dict[str, Any]],
    *,
    thresholds: Thresholds | None,
) -> dict[str, Any]:
    rows = list(records)
    hits = 0
    blocked = 0
    blocked_positive = 0
    false_positive = 0
    unknown_correct = 0
    for row in rows:
        completion = gated_completion(row, thresholds)
        expected_unknown = bool(row.get("expected_unknown", False))
        if thresholds is not None and canonical_answer_text(completion) == "UNKNOWN":
            if canonical_answer_text(str(row.get("completion", ""))) != "UNKNOWN":
                blocked += 1
                if not expected_unknown:
                    blocked_positive += 1
        score = score_answer(
            completion,
            row.get("answer_aliases") or [],
            expected_unknown=expected_unknown,
        )
        if bool(score["hit"]):
            hits += 1
        if bool(score["unknown_correct"]):
            unknown_correct += 1
        if expected_unknown and canonical_answer_text(completion) != "UNKNOWN" and not score["hit"]:
            false_positive += 1
    count = len(rows)
    return {
        "count": count,
        "hits": hits,
        "accuracy": hits / count if count else 0.0,
        "blocked": blocked,
        "blocked_rate": blocked / count if count else 0.0,
        "blocked_positive": blocked_positive,
        "false_positive": false_positive,
        "false_positive_rate": false_positive / count if count else 0.0,
        "unknown_correct": unknown_correct,
        "unknown_correct_rate": unknown_correct / count if count else 0.0,
    }


def threshold_grid() -> list[Thresholds]:
    def values(start: int, end: int, step: int) -> list[float]:
        return [round(x / 1000.0, 3) for x in range(start, end + 1, step)]

    support_values = values(500, 800, 25)
    causal_values = values(500, 800, 25)
    refute_values = values(250, 550, 25)
    missing_values = values(250, 550, 25)
    return [
        Thresholds(
            support_min=support,
            causal_min=causal,
            refute_max=refute,
            missing_max=missing,
        )
        for support in support_values
        for causal in causal_values
        for refute in refute_values
        for missing in missing_values
    ]


def stable_split(
    records: list[dict[str, Any]],
    *,
    calibration_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    def key(row: dict[str, Any]) -> str:
        raw = str(row.get("id", "")) + "\0" + str(row.get("question", ""))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    ordered = sorted(records, key=key)
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * calibration_fraction))))
    return ordered[:cut], ordered[cut:]


def find_best_thresholds(records: list[dict[str, Any]]) -> tuple[Thresholds, dict[str, Any]]:
    best: tuple[tuple[float, int, int, float], Thresholds, dict[str, Any]] | None = None
    for thresholds in threshold_grid():
        metrics = evaluate_records(records, thresholds=thresholds)
        rank = (
            float(metrics["accuracy"]),
            -int(metrics["false_positive"]),
            -int(metrics["blocked_positive"]),
            float(thresholds.missing_max),
        )
        if best is None or rank > best[0]:
            best = (rank, thresholds, metrics)
    if best is None:
        raise ValueError("no thresholds evaluated")
    return best[1], best[2]


def render_markdown(report: dict[str, Any]) -> str:
    thresholds = report["best_thresholds"]
    lines = [
        "# Answer Decision Gate Calibration",
        "",
        "## Verdict",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Thresholds",
        "",
        "| Threshold | Value |",
        "| --- | ---: |",
    ]
    for key in ["support_min", "causal_min", "refute_max", "missing_max"]:
        lines.append(f"| {key} | {float(thresholds[key]):.3f} |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Split | Baseline Acc | Gated Acc | Baseline FP | Gated FP | Blocked Positive |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ["calibration", "heldout", "full"]:
        base = report[f"{split}_baseline"]
        gated = report[f"{split}_gated"]
        lines.append(
            f"| {split} | {base['accuracy']:.4f} | {gated['accuracy']:.4f} | "
            f"{base['false_positive']} | {gated['false_positive']} | "
            f"{gated['blocked_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is not a learned verifier. It is a calibration probe over existing "
            "truth-gate probabilities. It can justify adding an answer-decision "
            "head only if held-out false positives drop without destroying positive "
            "answer recall.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate a verifier-controlled answer decision gate from eval records."
    )
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--mode", default="qtrm_residual_with_evidence")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--min-heldout-gain", type=float, default=0.01)
    parser.add_argument("--max-blocked-positive-rate", type=float, default=0.30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    records = load_records(args.records_jsonl, mode=args.mode)
    if len(records) < 2:
        raise SystemExit("need at least two records for calibration/heldout split")
    calibration, heldout = stable_split(
        records,
        calibration_fraction=float(args.calibration_fraction),
    )
    best_thresholds, calibration_gated = find_best_thresholds(calibration)
    report = {
        "records_jsonl": args.records_jsonl,
        "mode": args.mode,
        "count": len(records),
        "calibration_count": len(calibration),
        "heldout_count": len(heldout),
        "best_thresholds": best_thresholds.to_dict(),
        "calibration_baseline": evaluate_records(calibration, thresholds=None),
        "calibration_gated": calibration_gated,
        "heldout_baseline": evaluate_records(heldout, thresholds=None),
        "heldout_gated": evaluate_records(heldout, thresholds=best_thresholds),
        "full_baseline": evaluate_records(records, thresholds=None),
        "full_gated": evaluate_records(records, thresholds=best_thresholds),
    }
    heldout_gain = (
        float(report["heldout_gated"]["accuracy"])
        - float(report["heldout_baseline"]["accuracy"])
    )
    blocked_positive_rate = (
        float(report["heldout_gated"]["blocked_positive"])
        / max(1, int(report["heldout_gated"]["count"]))
    )
    failed: list[str] = []
    if heldout_gain < float(args.min_heldout_gain):
        failed.append("heldout_accuracy_gain_too_small")
    if blocked_positive_rate > float(args.max_blocked_positive_rate):
        failed.append("blocks_too_many_positive_answers")
    report["heldout_gain"] = heldout_gain
    report["heldout_blocked_positive_rate"] = blocked_positive_rate
    report["status"] = "accepted" if not failed else "rejected"
    report["failed_checks"] = failed

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
