#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wgram_lm.qwen_scope import summarize_qwen_scope_repeat_score_thresholds


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Summarize Qwen-Scope repeat-candidate score thresholds.")
    ap.add_argument("--input", required=True, help="Score JSON from scripts/138_score_qwen_scope_repeat_candidates.py")
    ap.add_argument("--out", required=True, help="Output threshold summary JSON")
    ap.add_argument("--score-field", default="total_value_sum")
    ap.add_argument("--label-field", default="repeat_label")
    ap.add_argument("--positive-label", default="repeat")
    ap.add_argument("--repeat-rate-threshold", type=float, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    scores = payload.get("scores", [])
    label_field = args.label_field
    if label_field == "":
        label_field = None
    summary = summarize_qwen_scope_repeat_score_thresholds(
        scores,
        score_field=args.score_field,
        label_field=label_field,
        positive_label=args.positive_label,
        repeat_rate_threshold=args.repeat_rate_threshold,
    )
    output = {
        "input": args.input,
        "threshold_summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
