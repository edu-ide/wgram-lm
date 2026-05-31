#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wgram_lm.qwen_scope import compare_qwen_scope_feature_groups


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compare Qwen-Scope SAE features between normal and repeated-output prompt groups."
    )
    ap.add_argument("--input", required=True, help="Qwen-Scope JSONL from scripts/136_qwen_scope_probe.py")
    ap.add_argument("--out", required=True, help="Output summary JSON path")
    ap.add_argument("--normal", required=True, help="Comma-separated normal prompt indices, e.g. 0,1,2")
    ap.add_argument("--repeat", required=True, help="Comma-separated repeated-output prompt indices, e.g. 3,4")
    ap.add_argument("--feature-limit", type=int, default=20)
    ap.add_argument("--top-output", type=int, default=15)
    return ap


def parse_indices(raw: str) -> set[int]:
    indices = {int(item.strip()) for item in raw.split(",") if item.strip()}
    if not indices:
        raise ValueError("at least one prompt index is required")
    return indices


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    records = load_jsonl(args.input)
    summary = compare_qwen_scope_feature_groups(
        records,
        normal_prompt_indices=parse_indices(args.normal),
        repeat_prompt_indices=parse_indices(args.repeat),
        feature_limit=args.feature_limit,
        top_output=args.top_output,
    )
    summary["input"] = args.input
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
