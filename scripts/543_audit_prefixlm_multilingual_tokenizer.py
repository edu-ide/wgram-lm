#!/usr/bin/env python3
"""Audit multilingual fragmentation for the HRM-Text PrefixLM tokenizer."""

from __future__ import annotations

import argparse
import math
import json
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_tokenizer_path() -> Path:
    return (
        repo_root()
        / "references"
        / "official"
        / "data_io"
        / "trained_tokenizers"
        / "bpe"
        / "tokenizer.json"
    )


def default_probe_path() -> Path:
    return repo_root() / "data" / "eval" / "prefixlm_multilingual_probe.jsonl"


def nonspace_char_count(text: str) -> int:
    return sum(1 for char in str(text) if not char.isspace())


def utf8_byte_count(text: str) -> int:
    return len(str(text).encode("utf-8"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(float(q) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "instruction" not in row:
                raise ValueError(f"row {line_number} missing instruction")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def compute_case_stats(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    instruction = unicodedata.normalize("NFKC", str(row["instruction"]))
    token_ids = tokenizer.encode(instruction, add_special_tokens=False).ids
    token_count = len(token_ids)
    nonspace_chars = nonspace_char_count(instruction)
    byte_count = utf8_byte_count(instruction)
    return {
        "case_id": str(row.get("case_id", "")),
        "language": str(row.get("language", "unknown")),
        "family": str(row.get("family", "unknown")),
        "instruction": instruction,
        "token_count": int(token_count),
        "nonspace_chars": int(nonspace_chars),
        "utf8_bytes": int(byte_count),
        "tokens_per_nonspace_char": float(token_count / max(1, nonspace_chars)),
        "tokens_per_utf8_byte": float(token_count / max(1, byte_count)),
        "utf8_bytes_per_token": float(byte_count / max(1, token_count)),
    }


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for name, items in sorted(groups.items()):
        fertility = [float(item["tokens_per_nonspace_char"]) for item in items]
        summary[name] = {
            "cases": int(len(items)),
            "mean_tokens_per_nonspace_char": float(statistics.fmean(fertility)),
            "p50_tokens_per_nonspace_char": float(percentile(fertility, 0.50)),
            "p95_tokens_per_nonspace_char": float(percentile(fertility, 0.95)),
            "max_tokens_per_nonspace_char": float(max(fertility)),
        }
    return summary


def gate_summary(
    rows: list[dict[str, Any]],
    *,
    warn_threshold: float,
) -> dict[str, Any]:
    by_language = summarize_by(rows, "language")
    languages_over_threshold = [
        language
        for language, stats in by_language.items()
        if float(stats["max_tokens_per_nonspace_char"]) > float(warn_threshold)
    ]
    return {
        "status": "warn" if languages_over_threshold else "pass",
        "warn_threshold": float(warn_threshold),
        "languages_over_threshold": languages_over_threshold,
        "plain_language": (
            "Some languages are being split into very small pieces; run a "
            "tokenizer redesign audit before claiming multilingual efficiency."
            if languages_over_threshold
            else "The probe does not show severe tokenizer fragmentation."
        ),
    }


def build_report(
    *,
    tokenizer_path: Path,
    probe_jsonl: Path,
    warn_threshold: float,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    cases = load_jsonl(probe_jsonl)
    rows = [compute_case_stats(row, tokenizer) for row in cases]
    return {
        "tokenizer_path": str(tokenizer_path),
        "probe_jsonl": str(probe_jsonl),
        "cases": int(len(rows)),
        "gate": gate_summary(rows, warn_threshold=float(warn_threshold)),
        "by_language": summarize_by(rows, "language"),
        "by_family": summarize_by(rows, "family"),
        "results": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", default=str(default_tokenizer_path()))
    parser.add_argument("--probe-jsonl", default=str(default_probe_path()))
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=1.5,
        help="Warn if any language has a max token/nonspace-char ratio above this value.",
    )
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        tokenizer_path=Path(args.tokenizer_path),
        probe_jsonl=Path(args.probe_jsonl),
        warn_threshold=float(args.warn_threshold),
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
