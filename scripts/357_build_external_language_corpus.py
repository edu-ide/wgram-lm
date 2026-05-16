#!/usr/bin/env python3
"""Build a small external language corpus for QTRM-native bootstrap.

The output is JSONL with a `text` field, so it can be consumed by
scripts/354_train_qtrm_native_language_bootstrap.py via --teacher-jsonl.
This script uses the Hugging Face Dataset Viewer API instead of loading full
datasets locally.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_SERVER = "https://datasets-server.huggingface.co"


DEFAULT_SOURCES = (
    "ultrachat:HuggingFaceH4/ultrachat_200k:default:train_sft:240",
    "fineweb:HuggingFaceFW/fineweb-edu:sample-10BT:train:160",
    "koalpaca:beomi/KoAlpaca-v1.1a:default:train:160",
)


def parse_source(value: str) -> dict[str, object]:
    parts = str(value).split(":")
    if len(parts) != 5:
        raise ValueError(
            "source must be kind:dataset:config:split:limit, "
            f"got {value!r}"
        )
    kind, dataset, config, split, limit = parts
    if kind not in {"ultrachat", "fineweb", "alpaca", "koalpaca"}:
        raise ValueError(f"unsupported source kind: {kind}")
    return {
        "kind": kind,
        "dataset": dataset,
        "config": config,
        "split": split,
        "limit": int(limit),
    }


def _retry_sleep_seconds(exc: Exception, *, attempt: int, base: float) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(base), float(retry_after))
            except ValueError:
                pass
        if exc.code in {429, 500, 502, 503, 504}:
            return float(base) * (2.0 ** max(0, int(attempt)))
    return float(base) * (1.0 + float(attempt))


def viewer_get(
    endpoint: str,
    params: dict[str, object],
    *,
    retries: int = 3,
    retry_sleep_base: float = 1.0,
) -> dict[str, Any]:
    url = f"{DATASET_SERVER}/{endpoint}?{urllib.parse.urlencode(params)}"
    last_exc: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read().decode("utf-8")
            decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                raise ValueError(f"unexpected response shape from {url}")
            return decoded
        except Exception as exc:  # pragma: no cover - network behavior
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(_retry_sleep_seconds(exc, attempt=attempt, base=retry_sleep_base))
    raise RuntimeError(f"Dataset Viewer request failed: {url}") from last_exc


def fetch_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    limit: int,
    page_size: int,
    retries: int = 3,
    retry_sleep_base: float = 1.0,
    request_delay_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < int(limit):
        length = min(int(page_size), int(limit) - len(rows))
        payload = viewer_get(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            retries=int(retries),
            retry_sleep_base=float(retry_sleep_base),
        )
        page = payload.get("rows", [])
        if not isinstance(page, list) or not page:
            break
        for item in page:
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                rows.append(item["row"])
        offset += len(page)
        if float(request_delay_seconds) > 0.0 and len(rows) < int(limit):
            time.sleep(float(request_delay_seconds))
        if len(page) < length:
            break
    return rows


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r".*</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    text = strip_think_blocks(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reject_text(text: str, *, min_chars: int, max_chars: int) -> bool:
    original = str(text).lower()
    if "<think" in original or "</think" in original:
        return True
    value = normalize_text(text)
    if len(value) < int(min_chars) or len(value) > int(max_chars):
        return True
    lowered = value.lower()
    if "<think" in lowered or "</think" in lowered:
        return True
    if lowered.count("http://") + lowered.count("https://") > 4:
        return True
    return False


def truncate_text(text: str, *, max_chars: int) -> str:
    value = normalize_text(text)
    if len(value) <= int(max_chars):
        return value
    cut = value[: int(max_chars)]
    for marker in (". ", "\n"):
        index = cut.rfind(marker)
        if index >= int(max_chars) // 2:
            return cut[: index + len(marker)].strip()
    return cut.strip()


def chat_record(user: str, assistant: str, *, max_chars: int) -> str:
    return truncate_text(
        f"User: {normalize_text(user)}\nAssistant: {normalize_text(assistant)}\n",
        max_chars=max_chars,
    )


def extract_ultrachat(row: dict[str, Any], *, max_chars: int) -> list[str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    records: list[str] = []
    last_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user:
            record = chat_record(last_user, content, max_chars=max_chars)
            records.append(record)
            last_user = ""
    return records


def extract_alpaca(row: dict[str, Any], *, max_chars: int) -> list[str]:
    instruction = row.get("instruction") or row.get("prompt") or row.get("input")
    output = row.get("output") or row.get("response") or row.get("answer")
    if not isinstance(instruction, str) or not isinstance(output, str):
        return []
    optional_input = row.get("input")
    prompt = instruction
    if isinstance(optional_input, str) and optional_input.strip() and optional_input.strip() != instruction.strip():
        prompt = f"{instruction}\n{optional_input}"
    return [chat_record(prompt, output, max_chars=max_chars)]


def extract_fineweb(row: dict[str, Any], *, max_chars: int) -> list[str]:
    text = row.get("text")
    if not isinstance(text, str):
        return []
    value = truncate_text(text, max_chars=max_chars)
    return [value]


def extract_records(kind: str, row: dict[str, Any], *, max_chars: int) -> list[str]:
    if kind == "ultrachat":
        return extract_ultrachat(row, max_chars=max_chars)
    if kind in {"alpaca", "koalpaca"}:
        return extract_alpaca(row, max_chars=max_chars)
    if kind == "fineweb":
        return extract_fineweb(row, max_chars=max_chars)
    raise ValueError(f"unsupported source kind: {kind}")


def build_corpus(args: argparse.Namespace) -> dict[str, object]:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    source_errors: dict[str, str] = {}

    source_specs = list(args.source or DEFAULT_SOURCES)
    for source_text in source_specs:
        source = parse_source(source_text)
        source_key = f'{source["kind"]}:{source["dataset"]}:{source["config"]}:{source["split"]}'
        try:
            rows = fetch_rows(
                dataset=str(source["dataset"]),
                config=str(source["config"]),
                split=str(source["split"]),
                limit=int(source["limit"]),
                page_size=int(args.page_size),
                retries=int(args.retries),
                retry_sleep_base=float(args.retry_sleep_base),
                request_delay_seconds=float(args.request_delay_seconds),
            )
        except Exception as exc:
            source_errors[source_key] = str(exc)
            if bool(args.continue_on_source_error):
                continue
            raise
        for row in rows:
            for text in extract_records(str(source["kind"]), row, max_chars=int(args.max_record_chars)):
                if (
                    int(args.max_records_per_source) > 0
                    and source_counts[source_key] >= int(args.max_records_per_source)
                ):
                    break
                cleaned = normalize_text(text)
                if reject_text(
                    cleaned,
                    min_chars=int(args.min_record_chars),
                    max_chars=int(args.max_record_chars),
                ):
                    rejected_counts[source_key] += 1
                    continue
                fingerprint = re.sub(r"\s+", " ", cleaned.lower())
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                source_counts[source_key] += 1
                records.append({"text": cleaned, "source": source_key})
                if len(records) >= int(args.max_records):
                    break
            if (
                int(args.max_records_per_source) > 0
                and source_counts[source_key] >= int(args.max_records_per_source)
            ):
                break
            if len(records) >= int(args.max_records):
                break
        if len(records) >= int(args.max_records):
            break

    out_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "out": str(out_path),
        "records": len(records),
        "chars": sum(len(str(record["text"])) for record in records),
        "source_counts": dict(source_counts),
        "rejected_counts": dict(rejected_counts),
        "source_errors": source_errors,
        "sources": source_specs,
        "min_record_chars": int(args.min_record_chars),
        "max_record_chars": int(args.max_record_chars),
        "max_records_per_source": int(args.max_records_per_source),
        "page_size": int(args.page_size),
        "retries": int(args.retries),
        "retry_sleep_base": float(args.retry_sleep_base),
        "request_delay_seconds": float(args.request_delay_seconds),
        "continued_on_source_error": bool(args.continue_on_source_error),
    }
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="local_eval/external_language_corpus/external_language.jsonl")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=1200)
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--min-record-chars", type=int, default=40)
    parser.add_argument("--max-record-chars", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-sleep-base", type=float, default=2.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--continue-on-source-error", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(build_corpus(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
