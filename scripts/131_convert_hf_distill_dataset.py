#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from wgram_lm.distill.hf_dataset_convert import convert_hf_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HF/public distillation datasets into QTRM teacher-record JSONL."
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--local-jsonl", default=None)
    parser.add_argument("--hf-id", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use HF streaming mode. Default is off because some datasets crash during streaming cleanup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_jsonl is None and args.hf_id is None:
        raise SystemExit("provide --local-jsonl or --hf-id")

    rows = _iter_local_jsonl(args.local_jsonl) if args.local_jsonl else _iter_hf_dataset(
        args.hf_id,
        split=args.split,
        streaming=args.streaming,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            if converted >= args.max_rows:
                break
            try:
                record = convert_hf_row(row, adapter=args.adapter)
                payload = record.to_dict()
            except Exception:
                skipped += 1
                continue
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            converted += 1
    print(f"converted={converted} skipped={skipped} out={out}")


def _iter_local_jsonl(path: str) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_hf_dataset(hf_id: str, *, split: str, streaming: bool) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("install datasets or use --local-jsonl") from exc
    ds = load_dataset(hf_id, split=split, streaming=streaming)
    yield from ds


if __name__ == "__main__":
    main()
