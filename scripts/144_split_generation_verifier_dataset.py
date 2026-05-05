#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence


SPLIT_NAMES = ("train", "calibration", "holdout")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Split generation-verifier JSONL into train/calibration/holdout files."
    )
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="generation_verifier")
    ap.add_argument("--calibration-ratio", type=float, default=0.2)
    ap.add_argument("--holdout-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    return ap


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def target_signature(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("generation_verifier_repeat_target", 0.0)),
        float(row.get("generation_verifier_stop_target", 0.0)),
        float(row.get("generation_verifier_quality_target", 0.0)),
    )


def _stable_row_key(row: dict) -> tuple[str, str]:
    sample = row.get("source_sample")
    if sample is None:
        sample = row.get("prompt_id", "")
    return (str(sample), str(row.get("text", "")))


def allocation_counts(
    n: int,
    *,
    calibration_ratio: float,
    holdout_ratio: float,
) -> tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    holdout = int(round(n * holdout_ratio))
    calibration = int(round(n * calibration_ratio))
    if n >= 3:
        holdout = max(1, holdout)
        calibration = max(1, calibration)
    else:
        holdout = max(1, holdout)
        calibration = 0
    while holdout + calibration >= n:
        if calibration > 0:
            calibration -= 1
        elif holdout > 0:
            holdout -= 1
        else:
            break
    train = n - holdout - calibration
    return (train, calibration, holdout)


def split_rows(
    rows: Sequence[dict],
    *,
    calibration_ratio: float = 0.2,
    holdout_ratio: float = 0.2,
    seed: int = 17,
) -> dict[str, list[dict]]:
    if calibration_ratio < 0.0 or holdout_ratio < 0.0:
        raise ValueError("split ratios must be non-negative")
    if calibration_ratio + holdout_ratio >= 1.0:
        raise ValueError("calibration_ratio + holdout_ratio must be < 1.0")

    groups: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[target_signature(row)].append(dict(row))

    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_NAMES}
    for signature in sorted(groups):
        group = sorted(groups[signature], key=_stable_row_key)
        rng.shuffle(group)
        train_n, calibration_n, holdout_n = allocation_counts(
            len(group),
            calibration_ratio=calibration_ratio,
            holdout_ratio=holdout_ratio,
        )
        holdout_rows = group[:holdout_n]
        calibration_rows = group[holdout_n : holdout_n + calibration_n]
        train_rows = group[holdout_n + calibration_n : holdout_n + calibration_n + train_n]
        for split_name, split_rows_ in (
            ("train", train_rows),
            ("calibration", calibration_rows),
            ("holdout", holdout_rows),
        ):
            for row in split_rows_:
                row["generation_verifier_split"] = split_name
            splits[split_name].extend(split_rows_)

    for split_name in SPLIT_NAMES:
        splits[split_name].sort(key=_stable_row_key)
    return splits


def _split_counts(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "repeat_failures": sum(
            float(row.get("generation_verifier_repeat_target", 0.0)) >= 0.5
            for row in rows
        ),
        "stop_failures": sum(
            float(row.get("generation_verifier_stop_target", 0.0)) >= 0.5
            for row in rows
        ),
        "quality_pass": sum(
            float(row.get("generation_verifier_quality_target", 0.0)) >= 0.5
            for row in rows
        ),
    }


def summarize_splits(splits: dict[str, Sequence[dict]]) -> dict:
    split_summaries = {name: _split_counts(splits.get(name, [])) for name in SPLIT_NAMES}
    return {
        "total_rows": sum(summary["rows"] for summary in split_summaries.values()),
        "splits": split_summaries,
        "target_totals": {
            key: sum(summary[key] for summary in split_summaries.values())
            for key in ("repeat_failures", "stop_failures", "quality_pass")
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = load_jsonl(args.data_jsonl)
    splits = split_rows(
        rows,
        calibration_ratio=args.calibration_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        write_jsonl(out_dir / f"{args.prefix}_{split_name}.jsonl", splits[split_name])
    summary = {
        "source": args.data_jsonl,
        "prefix": args.prefix,
        "seed": args.seed,
        "calibration_ratio": args.calibration_ratio,
        "holdout_ratio": args.holdout_ratio,
        **summarize_splits(splits),
    }
    summary_path = out_dir / f"{args.prefix}_split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
