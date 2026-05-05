#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


HEADS = ("repeat", "stop", "quality")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Calibrate generation-verifier thresholds on one split and evaluate them on holdout."
    )
    ap.add_argument("--calibration-eval-json", required=True)
    ap.add_argument("--holdout-eval-json", required=True)
    ap.add_argument("--out", required=True)
    return ap


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binary_metrics(
    *,
    probs: Iterable[float],
    targets: Iterable[float],
    threshold: float,
) -> dict:
    tp = fp = tn = fn = 0
    for prob, target in zip(probs, targets):
        pred = float(prob) >= float(threshold)
        actual = float(target) >= 0.5
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _head_probs_targets(summary: dict, head: str) -> tuple[list[float], list[float]]:
    records = summary.get("records") or []
    probs = [float(row.get(f"{head}_prob", 0.0)) for row in records]
    targets = [float(row.get(f"{head}_target", 0.0)) for row in records]
    return probs, targets


def best_threshold_metrics(*, probs: Sequence[float], targets: Sequence[float]) -> dict:
    if not probs:
        return {
            "threshold": 0.5,
            **binary_metrics(probs=[], targets=[], threshold=0.5),
        }
    candidates = sorted(set(float(prob) for prob in probs), reverse=True)
    rows = [
        {
            "threshold": threshold,
            **binary_metrics(probs=probs, targets=targets, threshold=threshold),
        }
        for threshold in candidates
    ]
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            row["accuracy"],
            row["threshold"],
        ),
    )


def calibrate_thresholds(calibration_summary: dict) -> dict:
    thresholds = {}
    for head in HEADS:
        probs, targets = _head_probs_targets(calibration_summary, head)
        thresholds[head] = best_threshold_metrics(probs=probs, targets=targets)
    return thresholds


def evaluate_with_thresholds(eval_summary: dict, thresholds: dict) -> dict:
    metrics = {}
    for head in HEADS:
        probs, targets = _head_probs_targets(eval_summary, head)
        threshold = float(thresholds.get(head, {}).get("threshold", 0.5))
        metrics[head] = {
            "threshold": threshold,
            **binary_metrics(probs=probs, targets=targets, threshold=threshold),
        }
    return metrics


def report(calibration_summary: dict, holdout_summary: dict) -> dict:
    thresholds = calibrate_thresholds(calibration_summary)
    return {
        "calibration": {
            "n": len(calibration_summary.get("records") or []),
            "thresholds": thresholds,
        },
        "holdout": {
            "n": len(holdout_summary.get("records") or []),
            **evaluate_with_thresholds(holdout_summary, thresholds),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    calibration_summary = load_json(args.calibration_eval_json)
    holdout_summary = load_json(args.holdout_eval_json)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report(calibration_summary, holdout_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
