#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wgram_lm.qwen_scope import score_qwen_scope_candidate_features


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Score Qwen-Scope repeat-candidate SAE features on generated prompt records."
    )
    ap.add_argument("--input", required=True, help="Qwen-Scope JSONL from scripts/136_qwen_scope_probe.py")
    ap.add_argument("--out", required=True, help="Output score JSON path")
    ap.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Layer-specific candidates, e.g. 12:847 or 23:29838,31860",
    )
    ap.add_argument("--feature-limit", type=int, default=None)
    ap.add_argument(
        "--metrics-jsonl",
        default=None,
        help="Optional QTRM eval JSONL with greedy_repetition metrics.",
    )
    ap.add_argument(
        "--repeat-threshold",
        type=float,
        default=0.15,
        help="Label prompts repeat when repeated_2gram_rate is at least this value.",
    )
    return ap


def parse_candidate_specs(specs: Sequence[str]) -> dict[int, set[int]]:
    candidates: dict[int, set[int]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"candidate spec must be '<layer>:<ids>': {spec}")
        layer_raw, features_raw = spec.split(":", 1)
        layer = int(layer_raw.strip())
        feature_ids = {int(item.strip()) for item in features_raw.split(",") if item.strip()}
        if not feature_ids:
            raise ValueError(f"candidate spec has no feature ids: {spec}")
        candidates.setdefault(layer, set()).update(feature_ids)
    if not candidates:
        raise ValueError("at least one candidate spec is required")
    return candidates


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_repetition_metrics(path: str | Path) -> dict[int, dict]:
    metrics: dict[int, dict] = {}
    for row in load_jsonl(path):
        sample = int(row.get("sample", row.get("prompt_index", len(metrics))))
        repetition = row.get("greedy_repetition") or {}
        metrics[sample] = {
            "repeated_1gram_rate": repetition.get("repeated_1gram_rate"),
            "repeated_2gram_rate": repetition.get("repeated_2gram_rate"),
            "repeated_3gram_rate": repetition.get("repeated_3gram_rate"),
            "repeated_4gram_rate": repetition.get("repeated_4gram_rate"),
        }
    return metrics


def attach_generation_metrics(
    scores: list[dict],
    metrics: dict[int, dict],
    *,
    repeat_threshold: float,
) -> None:
    for row in scores:
        prompt_index = int(row["prompt_index"])
        metric = metrics.get(prompt_index, {})
        for key, value in metric.items():
            row[key] = value
        repeated_2gram_rate = metric.get("repeated_2gram_rate")
        if repeated_2gram_rate is not None:
            row["repeat_label"] = (
                "repeat" if float(repeated_2gram_rate) >= repeat_threshold else "normal"
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    records = load_jsonl(args.input)
    candidates = parse_candidate_specs(args.candidate)
    scores = score_qwen_scope_candidate_features(
        records,
        candidate_features_by_layer=candidates,
        feature_limit=args.feature_limit,
    )
    if args.metrics_jsonl is not None:
        attach_generation_metrics(
            scores,
            load_repetition_metrics(args.metrics_jsonl),
            repeat_threshold=args.repeat_threshold,
        )
    ranking = sorted(
        scores,
        key=lambda row: (row["total_value_sum"], row["total_hit_count"]),
        reverse=True,
    )
    payload = {
        "input": args.input,
        "metrics_jsonl": args.metrics_jsonl,
        "feature_limit": args.feature_limit,
        "repeat_threshold": args.repeat_threshold,
        "candidate_features_by_layer": {
            str(layer): sorted(feature_ids) for layer, feature_ids in sorted(candidates.items())
        },
        "scores": scores,
        "ranking_by_total_value_sum": [
            {
                "prompt_index": int(row["prompt_index"]),
                "total_hit_count": int(row["total_hit_count"]),
                "total_value_sum": float(row["total_value_sum"]),
                "total_value_max": float(row["total_value_max"]),
                "repeated_2gram_rate": row.get("repeated_2gram_rate"),
                "repeat_label": row.get("repeat_label"),
            }
            for row in ranking
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
