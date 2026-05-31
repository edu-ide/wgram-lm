#!/usr/bin/env python3
"""Materialize typed/noisy candidate-choice rows for verifier training.

Deprecated diagnostic scaffold: this script uses hand-built typed candidates.
It is useful for reproducing the candidate-exposure upper bound, but it is not
allowed in the final Stage59 answer path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

from wgram_lm.eval.general_answer_interface import (
    answer_aliases,
    normalize_answer_text,
    normalized_alias_set,
)


def _load_script(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


stage523 = _load_script("qtrm_stage523_for_526", "523_train_state_text_speaker.py")
stage525 = _load_script("qtrm_stage525_for_526", "525_eval_qwen_candidate_exposure.py")


def target_choice_index(row: dict[str, Any]) -> int:
    aliases = set(normalized_alias_set(answer_aliases(row)))
    for index, choice in enumerate(row.get("choices") or []):
        if normalize_answer_text(str(choice)) in aliases:
            return int(index)
    return -1


def materialize_row(
    row: dict[str, Any],
    *,
    max_candidates: int,
    ensure_gold: bool,
    shuffle: bool,
    rng: random.Random,
) -> tuple[dict[str, Any], bool, bool]:
    aliases = list(answer_aliases(row))
    alias_norms = set(normalized_alias_set(aliases))
    candidates = stage525.typed_heuristic_candidates(row, max_candidates=int(max_candidates))
    before_has_gold = any(normalize_answer_text(candidate) in alias_norms for candidate in candidates)
    if ensure_gold and aliases and not before_has_gold:
        candidates = [str(aliases[0]), *candidates]
    candidates = stage525.dedupe_candidates(candidates, max_candidates=int(max_candidates))
    after_has_gold = any(normalize_answer_text(candidate) in alias_norms for candidate in candidates)
    if shuffle:
        rng.shuffle(candidates)
    out = dict(row)
    out["choices"] = candidates
    out["candidate_source"] = "typed_heuristic_noisy"
    out["candidate_oracle_before_ensure_gold"] = bool(before_has_gold)
    out["candidate_oracle_after_ensure_gold"] = bool(after_has_gold)
    out["candidate_target_index"] = target_choice_index(out)
    return out, before_has_gold, after_has_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1526)
    parser.add_argument("--ensure-gold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-diagnostic-scaffold",
        action="store_true",
        help="Required because this script materializes hand-built heuristic candidates.",
    )
    args = parser.parse_args()

    if not args.allow_diagnostic_scaffold:
        raise SystemExit(
            "scripts/526 is a deprecated diagnostic scaffold. "
            "Use --allow-diagnostic-scaffold only for audit/reproduction, not final-path experiments."
        )

    rng = random.Random(int(args.seed))
    rows = stage523.load_jsonl(args.input_jsonl, limit=int(args.limit))
    out_rows: list[dict[str, Any]] = []
    before_hits = 0
    after_hits = 0
    usable = 0
    for row in rows:
        out, before_has_gold, after_has_gold = materialize_row(
            row,
            max_candidates=int(args.max_candidates),
            ensure_gold=bool(args.ensure_gold),
            shuffle=bool(args.shuffle),
            rng=rng,
        )
        before_hits += int(before_has_gold)
        after_hits += int(after_has_gold)
        usable += int(target_choice_index(out) >= 0)
        out_rows.append(out)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "rows": len(out_rows),
        "max_candidates": int(args.max_candidates),
        "ensure_gold": bool(args.ensure_gold),
        "shuffle": bool(args.shuffle),
        "oracle_coverage_before_ensure_gold": before_hits / max(1, len(out_rows)),
        "oracle_coverage_after_ensure_gold": after_hits / max(1, len(out_rows)),
        "usable_target_rows": usable,
        "plain_language_read": (
            "Deprecated scaffold: this builds a hand-made answer table for verifier distribution-shift audits. "
            "Final-only Stage59 work must replace this with a learned typed working-register table."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
