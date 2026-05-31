from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional


BOILERPLATE_PATTERNS = (
    "course unit",
    "this course unit will delve",
    "answer with the letter",
    "your response must be concise",
    "make the answer very short",
    "offer a terse response",
)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+|[^\w\s]", (text or "").lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return repeats / max(1, len(grams))


def quality_reason(
    row: dict,
    *,
    min_words: int = 32,
    max_words: int = 420,
    max_repeated_4gram_rate: float = 0.12,
) -> Optional[str]:
    text = row.get("text") or ""
    words = word_count(text)

    lowered = text.lower()
    if any(pattern in lowered for pattern in BOILERPLATE_PATTERNS):
        return "boilerplate"
    if lowered.count("lecture:") >= 2 or lowered.count("question:") >= 4:
        return "boilerplate"
    if lowered.count("answer:") >= 4:
        return "boilerplate"
    if words < min_words:
        return "too_short"
    if words > max_words:
        return "too_long"
    if repeated_ngram_rate(text, n=4) > max_repeated_4gram_rate:
        return "repeated"
    return None


def normalize_row(row: dict, *, drop_images: bool = False) -> dict:
    text = (row.get("text") or "").replace("<image>", "").strip()
    out = {
        "type": row.get("type", "text"),
        "source": row.get("source", "unknown"),
        "text": text,
    }
    if not drop_images and row.get("images"):
        out["images"] = row["images"]
    return out


def iter_jsonl(paths: Iterable[str]) -> Iterable[dict]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def filter_rows(
    rows: Iterable[dict],
    *,
    max_rows: int,
    min_words: int,
    max_words: int,
    drop_images: bool,
    max_per_source: int,
) -> tuple[list[dict], dict[str, int]]:
    accepted: list[dict] = []
    stats: dict[str, int] = defaultdict(int)
    per_source: Counter[str] = Counter()
    for row in rows:
        normalized = normalize_row(row, drop_images=drop_images)
        source = str(normalized.get("source", "unknown"))
        if max_per_source > 0 and per_source[source] >= max_per_source:
            stats["source_cap"] += 1
            continue
        reason = quality_reason(normalized, min_words=min_words, max_words=max_words)
        if reason is not None:
            stats[f"reject_{reason}"] += 1
            continue
        accepted.append(normalized)
        per_source[source] += 1
        stats["accepted"] += 1
        if len(accepted) >= max_rows:
            break
    return accepted, dict(stats)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a cleaner JSONL pilot dataset for QTRM LM diagnostics.")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--min-words", type=int, default=32)
    ap.add_argument("--max-words", type=int, default=420)
    ap.add_argument("--drop-images", action="store_true")
    ap.add_argument("--max-per-source", type=int, default=4000)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    rows, stats = filter_rows(
        iter_jsonl(args.input),
        max_rows=args.max_rows,
        min_words=args.min_words,
        max_words=args.max_words,
        drop_images=args.drop_images,
        max_per_source=args.max_per_source,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
