#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qtrm_mm.eval.fact_verification import evaluate_fact_case, load_fact_cases, summarize_fact_records


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate a small MemoryOS fact-verification gate without generation."
    )
    ap.add_argument("--cases", default="data/eval/fact_verification_probe.jsonl")
    ap.add_argument("--evidence-mode", default="target", choices=["target", "all", "lexical", "none"])
    ap.add_argument("--retrieval-top-k", type=int, default=5)
    ap.add_argument("--memory-max-chars", type=int, default=4000)
    ap.add_argument("--jsonl-out", default="runs/eval/fact_verification_probe.jsonl")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cases = load_fact_cases(args.cases)
    records = [
        evaluate_fact_case(
            case,
            evidence_mode=args.evidence_mode,
            retrieval_top_k=args.retrieval_top_k,
            max_evidence_chars=args.memory_max_chars,
        )
        for case in cases
    ]
    summary = summarize_fact_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.jsonl_out:
        out = Path(args.jsonl_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
