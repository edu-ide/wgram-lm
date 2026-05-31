#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import urllib.parse
import urllib.request

from wgram_lm.data.verified_reasoning import (
    DEFAULT_VERIFIED_SOURCES,
    convert_verified_row,
)


DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"


class SourceSpec:
    def __init__(
        self,
        *,
        name: str,
        dataset: str,
        config: str,
        split: str,
        adapter: str,
        local_jsonl: str = "",
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.config = config
        self.split = split
        self.adapter = adapter
        self.local_jsonl = local_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt-only QTRM raw-reasoning JSONL from verified HF datasets. "
            "Teacher outputs are not used as labels."
        )
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Named source from DEFAULT_VERIFIED_SOURCES, or "
            "name,dataset,config,split,adapter. Can be repeated."
        ),
    )
    parser.add_argument("--max-rows-per-source", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--viewer-base", default=DATASET_VIEWER_BASE)
    return parser


def build_verified_reasoning_dataset(
    *,
    sources: list[SourceSpec],
    out_path: str | Path,
    max_rows_per_source: int,
    offset: int = 0,
    viewer_base: str = DATASET_VIEWER_BASE,
) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "read": 0,
        "written": 0,
        "skipped": 0,
        "sources": {},
        "out": str(out),
    }
    source_rows: list[tuple[SourceSpec, list[dict[str, Any]]]] = []
    for source in sources:
        rows_for_source: list[dict[str, Any]] = []
        for row_index, row in _iter_source_rows(
            source,
            max_rows=int(max_rows_per_source),
            offset=int(offset),
            viewer_base=viewer_base,
        ):
            stats["read"] += 1
            try:
                converted = convert_verified_row(
                    row,
                    adapter=source.adapter,
                    source_name=source.name,
                    row_index=row_index,
                )
                converted["source_hf_id"] = source.dataset
                converted["source_config"] = source.config
                converted["source_split"] = source.split
            except Exception:
                stats["skipped"] += 1
                continue
            rows_for_source.append(converted)
            if len(rows_for_source) >= int(max_rows_per_source):
                break
        stats["sources"][source.name] = len(rows_for_source)
        source_rows.append((source, rows_for_source))

    with out.open("w", encoding="utf-8") as f:
        max_len = max((len(rows) for _, rows in source_rows), default=0)
        for row_offset in range(max_len):
            for _, rows_for_source in source_rows:
                if row_offset >= len(rows_for_source):
                    continue
                f.write(json.dumps(rows_for_source[row_offset], ensure_ascii=False) + "\n")
                stats["written"] += 1
    return stats


def resolve_sources(values: list[str]) -> list[SourceSpec]:
    if not values:
        values = ["gsm8k_train", "numina_verifiable_train", "proofwriter_validation", "clutrr_train"]
    sources: list[SourceSpec] = []
    for value in values:
        if value in DEFAULT_VERIFIED_SOURCES:
            spec = DEFAULT_VERIFIED_SOURCES[value]
            sources.append(
                SourceSpec(
                    name=spec.name,
                    dataset=spec.dataset,
                    config=spec.config,
                    split=spec.split,
                    adapter=spec.adapter,
                )
            )
            continue
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 5:
            raise ValueError(
                "--source must be a default name or name,dataset,config,split,adapter"
            )
        sources.append(
            SourceSpec(
                name=parts[0],
                dataset=parts[1],
                config=parts[2],
                split=parts[3],
                adapter=parts[4],
            )
        )
    return sources


def _iter_source_rows(
    source: SourceSpec,
    *,
    max_rows: int,
    offset: int,
    viewer_base: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if source.local_jsonl:
        yield from _iter_local_jsonl(source.local_jsonl, max_rows=max_rows, offset=offset)
        return
    yield from _iter_hf_viewer_rows(source, max_rows=max_rows, offset=offset, viewer_base=viewer_base)


def _iter_local_jsonl(
    path: str,
    *,
    max_rows: int,
    offset: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    emitted = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if row_index < int(offset):
                continue
            line = line.strip()
            if not line:
                continue
            yield row_index, json.loads(line)
            emitted += 1
            if emitted >= int(max_rows):
                break


def _iter_hf_viewer_rows(
    source: SourceSpec,
    *,
    max_rows: int,
    offset: int,
    viewer_base: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    remaining = int(max_rows)
    current_offset = int(offset)
    while remaining > 0:
        length = min(100, remaining)
        payload = _fetch_rows_page(
            dataset=source.dataset,
            config=source.config,
            split=source.split,
            offset=current_offset,
            length=length,
            viewer_base=viewer_base,
        )
        rows = payload.get("rows") or []
        if not rows:
            break
        for item in rows:
            row_index = int(item.get("row_idx", current_offset))
            row = item.get("row", item)
            if isinstance(row, dict):
                yield row_index, row
                remaining -= 1
                current_offset = row_index + 1
                if remaining <= 0:
                    break
        if len(rows) < length:
            break


def _fetch_rows_page(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    viewer_base: str,
) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": int(offset),
            "length": int(length),
        }
    )
    url = f"{viewer_base.rstrip('/')}/rows?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_verified_reasoning_dataset(
        sources=resolve_sources(args.source),
        out_path=args.out,
        max_rows_per_source=args.max_rows_per_source,
        offset=args.offset,
        viewer_base=args.viewer_base,
    )
    print(
        "written={written} read={read} skipped={skipped} out={out} sources={sources}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
