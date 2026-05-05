#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate generation-verifier candidate reranking over grouped completions."
    )
    ap.add_argument("--eval-json", required=True, help="JSON from scripts/143_eval_generation_verifier.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality-weight", type=float, default=1.0)
    ap.add_argument("--repeat-weight", type=float, default=1.0)
    ap.add_argument("--stop-weight", type=float, default=1.0)
    return ap


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_id(record: dict) -> int:
    value = record.get("candidate_id")
    if value is None:
        return 0
    return int(value)


def source_key(record: dict) -> str:
    sample = record.get("source_sample")
    if sample is None:
        sample = record.get("sample", "")
    return str(sample)


def candidate_score(
    record: dict,
    *,
    quality_weight: float = 1.0,
    repeat_weight: float = 1.0,
    stop_weight: float = 1.0,
) -> float:
    quality = float(record.get("quality_prob", 0.0))
    repeat = float(record.get("repeat_prob", 0.0))
    stop = float(record.get("stop_prob", 0.0))
    return (
        float(quality_weight) * quality
        - float(repeat_weight) * repeat
        - float(stop_weight) * stop
    )


def is_quality(record: dict) -> bool:
    return float(record.get("quality_target", 0.0)) >= 0.5


def is_repeat_failure(record: dict) -> bool:
    return float(record.get("repeat_target", 0.0)) >= 0.5


def is_stop_failure(record: dict) -> bool:
    return float(record.get("stop_target", 0.0)) >= 0.5


def _rate(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / max(1, len(values))


def summarize_rerank(
    records: Sequence[dict],
    *,
    quality_weight: float = 1.0,
    repeat_weight: float = 1.0,
    stop_weight: float = 1.0,
) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[source_key(record)].append(dict(record))

    selected_rows = []
    baseline_quality = []
    reranked_quality = []
    oracle_quality = []
    baseline_repeat = []
    reranked_repeat = []
    baseline_stop = []
    reranked_stop = []
    changed = []

    for key in sorted(groups, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        candidates = sorted(groups[key], key=candidate_id)
        baseline = candidates[0]
        selected = max(
            candidates,
            key=lambda row: (
                candidate_score(
                    row,
                    quality_weight=quality_weight,
                    repeat_weight=repeat_weight,
                    stop_weight=stop_weight,
                ),
                -candidate_id(row),
            ),
        )
        selected_score = candidate_score(
            selected,
            quality_weight=quality_weight,
            repeat_weight=repeat_weight,
            stop_weight=stop_weight,
        )
        baseline_quality.append(is_quality(baseline))
        reranked_quality.append(is_quality(selected))
        oracle_quality.append(any(is_quality(row) for row in candidates))
        baseline_repeat.append(is_repeat_failure(baseline))
        reranked_repeat.append(is_repeat_failure(selected))
        baseline_stop.append(is_stop_failure(baseline))
        reranked_stop.append(is_stop_failure(selected))
        changed.append(candidate_id(baseline) != candidate_id(selected))
        selected_rows.append(
            {
                "source_sample": key,
                "candidate_count": len(candidates),
                "baseline_candidate_id": candidate_id(baseline),
                "selected_candidate_id": candidate_id(selected),
                "selected_score": selected_score,
                "baseline_quality": is_quality(baseline),
                "selected_quality": is_quality(selected),
                "oracle_has_quality": any(is_quality(row) for row in candidates),
                "selected_repeat_failure": is_repeat_failure(selected),
                "selected_stop_failure": is_stop_failure(selected),
            }
        )

    candidate_count = sum(len(rows) for rows in groups.values())
    group_count = len(groups)
    return {
        "groups": group_count,
        "candidate_count": candidate_count,
        "candidate_per_group_mean": candidate_count / max(1, group_count),
        "baseline_quality_rate": _rate(baseline_quality),
        "reranked_quality_rate": _rate(reranked_quality),
        "oracle_quality_rate": _rate(oracle_quality),
        "quality_rate_delta": _rate(reranked_quality) - _rate(baseline_quality),
        "baseline_repeat_failure_rate": _rate(baseline_repeat),
        "reranked_repeat_failure_rate": _rate(reranked_repeat),
        "repeat_failure_rate_delta": _rate(reranked_repeat) - _rate(baseline_repeat),
        "baseline_stop_failure_rate": _rate(baseline_stop),
        "reranked_stop_failure_rate": _rate(reranked_stop),
        "stop_failure_rate_delta": _rate(reranked_stop) - _rate(baseline_stop),
        "selected_changed_rate": _rate(changed),
        "weights": {
            "quality": float(quality_weight),
            "repeat": float(repeat_weight),
            "stop": float(stop_weight),
        },
        "selected": selected_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = load_json(args.eval_json)
    report = summarize_rerank(
        summary.get("records") or [],
        quality_weight=args.quality_weight,
        repeat_weight=args.repeat_weight,
        stop_weight=args.stop_weight,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
