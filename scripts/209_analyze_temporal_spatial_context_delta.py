#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

CONTEXT_ON_MODE = "qtrm_core_steps_8_no_evidence"
CONTEXT_OFF_MODE = "qtrm_core_steps_8_temporal_spatial_off_no_evidence"


def _normalize(text: str) -> str:
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _choice_logprob(record: dict[str, Any], aliases: Iterable[str]) -> float | None:
    normalized_aliases = {_normalize(alias) for alias in aliases if _normalize(alias)}
    if not normalized_aliases:
        return None
    for row in record.get("choice_scores", []):
        choice = _normalize(row.get("choice", ""))
        if choice in normalized_aliases:
            return float(row.get("logprob"))
    return None


def _summarize_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    deltas: list[float] = []
    on_only = 0
    off_only = 0
    both_correct = 0
    both_wrong = 0
    changed_completion = 0
    examples: list[dict[str, Any]] = []
    for on, off in pairs:
        on_hit = bool(on.get("hit"))
        off_hit = bool(off.get("hit"))
        if on_hit and not off_hit:
            on_only += 1
        elif off_hit and not on_hit:
            off_only += 1
        elif on_hit and off_hit:
            both_correct += 1
        else:
            both_wrong += 1
        if str(on.get("completion", "")).strip() != str(off.get("completion", "")).strip():
            changed_completion += 1
        aliases = on.get("answer_aliases") or off.get("answer_aliases") or []
        on_logprob = _choice_logprob(on, aliases)
        off_logprob = _choice_logprob(off, aliases)
        delta = None
        if on_logprob is not None and off_logprob is not None:
            delta = on_logprob - off_logprob
            deltas.append(delta)
        if len(examples) < 8:
            examples.append(
                {
                    "id": on.get("id"),
                    "task_family": on.get("task_family", "unknown"),
                    "context_on_completion": on.get("completion"),
                    "context_off_completion": off.get("completion"),
                    "context_on_hit": on_hit,
                    "context_off_hit": off_hit,
                    "chosen_logprob_delta": delta,
                }
            )
    return {
        "paired_count": len(pairs),
        "context_on_only_correct_count": on_only,
        "context_off_only_correct_count": off_only,
        "both_correct_count": both_correct,
        "both_wrong_count": both_wrong,
        "changed_completion_count": changed_completion,
        "chosen_logprob_delta_count": len(deltas),
        "chosen_logprob_delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
        "chosen_logprob_delta_min": min(deltas) if deltas else 0.0,
        "chosen_logprob_delta_max": max(deltas) if deltas else 0.0,
        "examples": examples,
    }


def analyze_records(
    records: Iterable[dict[str, Any]],
    *,
    context_on_mode: str = CONTEXT_ON_MODE,
    context_off_mode: str = CONTEXT_OFF_MODE,
) -> dict[str, Any]:
    by_case: dict[Any, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record.get("id"), {})[str(record.get("mode"))] = record

    pairs = [
        (rows[context_on_mode], rows[context_off_mode])
        for rows in by_case.values()
        if context_on_mode in rows and context_off_mode in rows
    ]
    summary = _summarize_pairs(pairs)
    by_family_pairs: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for on, off in pairs:
        by_family_pairs.setdefault(str(on.get("task_family", "unknown")), []).append((on, off))

    summary.update(
        {
            "context_on_mode": context_on_mode,
            "context_off_mode": context_off_mode,
            "by_task_family": {
                family: _summarize_pairs(family_pairs)
                for family, family_pairs in sorted(by_family_pairs.items())
            },
        }
    )
    return summary


def write_summary(
    records: Iterable[dict[str, Any]],
    out: str | Path,
    *,
    context_on_mode: str = CONTEXT_ON_MODE,
    context_off_mode: str = CONTEXT_OFF_MODE,
) -> dict[str, Any]:
    summary = analyze_records(
        records,
        context_on_mode=context_on_mode,
        context_off_mode=context_off_mode,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze context-on/off temporal-spatial eval deltas."
    )
    parser.add_argument("--eval-jsonl", default="local_eval/temporal_spatial_context_gate.jsonl")
    parser.add_argument(
        "--out",
        default="docs/wiki/decisions/temporal-spatial-context-delta-summary.json",
    )
    parser.add_argument("--context-on-mode", default=CONTEXT_ON_MODE)
    parser.add_argument("--context-off-mode", default=CONTEXT_OFF_MODE)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = write_summary(
        _load_records(args.eval_jsonl),
        args.out,
        context_on_mode=args.context_on_mode,
        context_off_mode=args.context_off_mode,
    )
    print(f"paired={summary['paired_count']}")
    print(f"changed_completion={summary['changed_completion_count']}")
    print(f"chosen_logprob_delta_mean={summary['chosen_logprob_delta_mean']:.6f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
