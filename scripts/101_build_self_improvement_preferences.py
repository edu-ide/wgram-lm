#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qtrm_mm.eval.memory_retrieval import load_cases
from qtrm_mm.training.self_improvement_data import load_eval_records, write_preference_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DPO-style self-improvement preference rows from MemoryOS eval failures."
    )
    parser.add_argument("--cases", default="data/eval/memory_reasoning_heldout_probe.jsonl")
    parser.add_argument(
        "--eval-jsonl",
        default="runs/eval/memory_reasoning_heldout_qwen3_rerank_32tok_synth_generalization_s050.jsonl",
    )
    parser.add_argument("--out", default="data/filtered/memory_self_improvement_preferences.jsonl")
    parser.add_argument("--source-eval", default=None)
    parser.add_argument(
        "--training-scope",
        default="analysis_only",
        choices=["analysis_only", "train_candidate"],
        help="Keep held-out-derived rows analysis_only unless explicitly creating a separate train candidate.",
    )
    parser.add_argument(
        "--missing-answer-policy",
        default="needs_search",
        choices=["closed_evidence_unknown", "needs_search"],
    )
    parser.add_argument("--include-hits-with-artifacts", action="store_true")
    parser.add_argument("--max-evidence-chars", type=int, default=2000)
    args = parser.parse_args()

    count = write_preference_jsonl(
        load_cases(args.cases),
        load_eval_records(args.eval_jsonl),
        args.out,
        source_eval=args.source_eval or args.eval_jsonl,
        training_scope=args.training_scope,
        include_hits_with_artifacts=args.include_hits_with_artifacts,
        missing_answer_policy=args.missing_answer_policy,
        max_evidence_chars=args.max_evidence_chars,
    )
    print(
        f"wrote {args.out}, rows={count}, training_scope={args.training_scope}, "
        f"missing_answer_policy={args.missing_answer_policy}"
    )


if __name__ == "__main__":
    main()
