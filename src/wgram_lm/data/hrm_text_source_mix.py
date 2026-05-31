from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable
import urllib.parse
import urllib.request

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised in minimal envs
    load_dataset = None

from wgram_lm.data.verified_reasoning import DEFAULT_VERIFIED_SOURCES, convert_verified_row


DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
DEFAULT_SOURCE_NAMES = [
    "gsm8k_train",
    "numina_verifiable_train",
    "openr1_math_verified_train",
    "openmathinstruct2_train",
    "proofwriter_validation",
    "clutrr_train",
    "bbh_boolean_test",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    split: str
    adapter: str
    local_jsonl: str = ""


def resolve_sources(values: list[str] | None = None) -> list[SourceSpec]:
    selected = values or DEFAULT_SOURCE_NAMES
    sources: list[SourceSpec] = []
    for value in selected:
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
                "--verified-source must be a default source name or "
                "name,dataset,config,split,adapter"
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


def verified_to_hrm_text_row(converted: dict[str, Any]) -> dict[str, str]:
    prompt = str(converted.get("prompt") or converted.get("question") or "").strip()
    if prompt.endswith("Answer:"):
        prompt = prompt[: -len("Answer:")].rstrip()
    family = str(
        converted.get("reasoning_family")
        or converted.get("task_family")
        or converted.get("category")
        or "verified_reasoning"
    ).strip()
    source = str(converted.get("source_dataset") or converted.get("source_hf_id") or "verified").strip()
    return {
        "condition": f"verified,{family},answer_only,{source}",
        "instruction": prompt,
        "response": str(converted.get("answer", "")).strip(),
    }


def build_hrm_text_source_mix(
    *,
    out_dir: str | Path,
    verified_sources: list[SourceSpec],
    max_verified_rows_per_source: int,
    dolly_rows: int,
    seed: int,
    verified_offset: int = 0,
    dolly_offset: int = 0,
    viewer_base: str = DATASET_VIEWER_BASE,
) -> dict[str, Any]:
    root = Path(out_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    stats: dict[str, Any] = {
        "out_dir": str(root),
        "seed": int(seed),
        "verified_offset": int(verified_offset),
        "dolly_offset": int(dolly_offset),
        "verified": {"written": 0, "skipped": 0, "sources": {}},
        "dolly": {"written": 0},
        "files": {},
    }

    verified_rows: list[dict[str, str]] = []
    for source in verified_sources:
        source_count = 0
        source_skipped = 0
        for row_index, row in _iter_source_rows(
            source,
            max_rows=max_verified_rows_per_source,
            offset=verified_offset,
            viewer_base=viewer_base,
        ):
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
                hrm_row = verified_to_hrm_text_row(converted)
                if not hrm_row["instruction"] or not hrm_row["response"]:
                    raise ValueError("empty instruction or response")
            except Exception:
                source_skipped += 1
                continue
            verified_rows.append(hrm_row)
            source_count += 1
            if source_count >= max_verified_rows_per_source:
                break
        stats["verified"]["sources"][source.name] = {
            "written": source_count,
            "skipped": source_skipped,
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "adapter": source.adapter,
        }
        stats["verified"]["skipped"] += source_skipped

    rng.shuffle(verified_rows)
    verified_path = data_dir / "verified_reasoning.jsonl"
    _write_jsonl(verified_path, verified_rows)
    stats["verified"]["written"] = len(verified_rows)
    stats["files"]["verified_reasoning"] = str(verified_path)

    dolly_path = data_dir / "dolly_healing.jsonl"
    dolly = load_dolly_rows(count=dolly_rows, seed=seed, offset=dolly_offset)
    _write_jsonl(dolly_path, dolly)
    stats["dolly"]["written"] = len(dolly)
    stats["files"]["dolly_healing"] = str(dolly_path)

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats["files"]["manifest"] = str(manifest_path)
    return stats


def load_dolly_rows(*, count: int, seed: int, offset: int = 0) -> list[dict[str, str]]:
    if count <= 0 or load_dataset is None:
        return []
    rows: list[dict[str, str]] = []
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    start = min(max(0, int(offset)), len(ds))
    stop = min(start + int(count), len(ds))
    selected = ds.shuffle(seed=seed).select(range(start, stop))
    for item in selected:
        instruction = str(item.get("instruction") or "").strip()
        response = str(item.get("response") or "").strip()
        if not instruction or not response:
            continue
        rows.append(
            {
                "condition": "healing,dolly,instruction_response",
                "instruction": instruction,
                "response": response,
            }
        )
    return rows


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


def _iter_local_jsonl(path: str, *, max_rows: int, offset: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    emitted = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
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
    offset = max(0, int(offset))
    while remaining > 0:
        length = min(100, remaining)
        try:
            payload = _fetch_rows_page(
                dataset=source.dataset,
                config=source.config,
                split=source.split,
                offset=offset,
                length=length,
                viewer_base=viewer_base,
            )
        except Exception as exc:
            print(f"[warn] failed to fetch {source.name}: {exc}", flush=True)
            return
        rows = payload.get("rows") or []
        if not rows:
            return
        for item in rows:
            row_index = int(item.get("row_idx", offset))
            row = item.get("row", item)
            if isinstance(row, dict):
                yield row_index, row
                remaining -= 1
                offset = row_index + 1
                if remaining <= 0:
                    break
        if len(rows) < length:
            return


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


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
