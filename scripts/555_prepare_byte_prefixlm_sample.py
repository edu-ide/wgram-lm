#!/usr/bin/env python3
"""Build a tokenizer-free UTF-8 byte PrefixLM sampled dataset.

This is a fast falsification path for tokenizer-free language training.  It
keeps the existing HRM-Text/Data-IO tensor contract:

  tokens.npy
  metadata.json
  epoch_N/{inst_start,inst_len,resp_start,resp_len}.npy

Only the unit changes.  Instead of BPE ids, text is encoded as raw UTF-8 bytes
shifted by +2, with PAD=0 and EOS=1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
BYTE_OFFSET = 2
BYTE_VOCAB_SIZE = 258


def byte_ids(text: str) -> list[int]:
    return [int(byte) + BYTE_OFFSET for byte in str(text).encode("utf-8", errors="replace")]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if isinstance(value, dict):
                yield value


def iter_parquet(path: Path) -> Iterable[dict]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pyarrow is required to read parquet source files") from exc
    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema_arrow.names)
    columns = [name for name in ("instruction", "response") if name in schema_names]
    for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def iter_rows(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    if suffix == ".parquet":
        yield from iter_parquet(path)
        return
    raise ValueError(f"unsupported source file suffix for {path}; expected .jsonl or .parquet")


def expand_source_files(data_root: Path, source_files: str, source_globs: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def add(rel: str) -> None:
        normalized = str(Path(rel))
        if normalized not in seen:
            seen.add(normalized)
            selected.append(normalized)

    for item in str(source_files).split():
        if item:
            add(item)
    for pattern in str(source_globs).split():
        if not pattern:
            continue
        matches = sorted(str(path.relative_to(data_root)) for path in data_root.glob(pattern) if path.is_file())
        for match in matches:
            add(match)
    return selected


def source_bucket(rel_path: str) -> str:
    """Coarse curriculum bucket for HRM-Text/Data-IO style balanced sampling."""
    rel = str(rel_path).replace("\\", "/").lower()
    name = Path(rel).name
    if "data_clustered/synth/" in rel:
        return "synthetic_math_like"
    if "data_clustered/flan/" in rel:
        if "translation" in name or "xquad" in name or "mlqa" in name or "tydi" in name or "xnli" in name:
            return "flan_multilingual_translation"
        if any(
            marker in name
            for marker in (
                "question",
                "answer",
                "classification",
                "generation",
                "summarization",
                "paraphrasing",
                "commonsense",
                "story",
                "dialog",
            )
        ):
            return "flan_instruction_qa"
        return "flan_other"
    if "data_clustered/tasksource/" in rel:
        return "tasksource"
    if any(marker in rel for marker in ("natural_reasoning", "acereason", "textbookreasoning", "openthoughts", "principia")):
        return "reasoning"
    if any(
        marker in rel
        for marker in (
            "gsm8k",
            "math_train",
            "numinamath",
            "omnimath",
            "openmathinstruct2",
            "dmmath",
            "amps",
            "sudoku",
            "arb_math",
            "scibench",
            "theoremqa",
        )
    ):
        return "math"
    if any(
        marker in rel
        for marker in (
            "no_robots",
            "webinstruct",
            "openbookqa",
            "arb_reading",
            "arb_science",
            "arb_law",
            "arb_physics",
            "reclor",
            "scienceqa",
        )
    ):
        return "general_instruction"
    return "other"


def parse_int_mapping(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for raw_item in str(text or "").replace(",", " ").split():
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value in mapping item: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty mapping key in item: {item!r}")
        values[key] = int(raw_value)
    return values


def load_utility_scores(path: str) -> tuple[dict[tuple[str, int], float], int]:
    """Load OPUS-compatible per-row utility scores.

    The scorer is intentionally external: exact OPUS needs optimizer-shaped
    update projections, while this builder should only consume those scores and
    materialize the selected byte PrefixLM sample.
    """

    if not str(path or "").strip():
        return {}, 0
    score_path = Path(path)
    scores: dict[tuple[str, int], float] = {}
    loaded = 0
    with score_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source_file = str(row.get("source_file", row.get("source", ""))).strip()
            if not source_file:
                raise ValueError(f"utility score missing source_file at {score_path}:{line_number}")
            if "row_index" not in row:
                raise ValueError(f"utility score missing row_index at {score_path}:{line_number}")
            utility_value = row.get("utility", row.get("score"))
            if utility_value is None:
                raise ValueError(f"utility score missing utility/score at {score_path}:{line_number}")
            scores[(str(Path(source_file)), int(row["row_index"]))] = float(utility_value)
            loaded += 1
    return scores, loaded


def select_candidates(
    candidates: list[dict],
    *,
    selection_mode: str,
    utility_temperature: float,
    seed: int,
) -> list[dict]:
    mode = str(selection_mode or "first").strip().lower()
    if mode in ("", "first", "static"):
        return list(candidates)
    if mode != "utility":
        raise ValueError(f"unsupported selection_mode={selection_mode!r}; expected first or utility")
    # Missing utility means "not auditioned", not neutral. OPUS alignments can
    # be negative, so a default 0.0 score would let unscored rows outrank
    # causally scored rows and silently turn utility selection into first-seen
    # sampling.
    scored = [item for item in candidates if bool(item.get("utility_score_present", False))]
    unscored = [item for item in candidates if not bool(item.get("utility_score_present", False))]
    if float(utility_temperature) <= 0.0:
        selected_scored = sorted(
            scored,
            key=lambda item: (-float(item.get("utility", 0.0)), int(item.get("row_index", 0))),
        )
        return selected_scored + list(unscored)
    rng = np.random.default_rng(int(seed))
    remaining = list(scored) if scored else list(candidates)
    selected: list[dict] = []
    temperature = float(max(1e-8, utility_temperature))
    while remaining:
        utilities = np.asarray([float(item.get("utility", 0.0)) for item in remaining], dtype=np.float64)
        utilities = utilities - float(np.max(utilities))
        weights = np.exp(utilities / temperature)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            probs = np.full(len(remaining), 1.0 / float(len(remaining)), dtype=np.float64)
        else:
            probs = weights / total
        picked = int(rng.choice(len(remaining), p=probs))
        selected.append(remaining.pop(picked))
    if scored:
        selected.extend(unscored)
    return selected


def render_instruction(raw: str) -> str:
    instruction = str(raw).strip()
    return f"User: {instruction}\nAssistant:"


def render_response(raw: str) -> str:
    response = str(raw).strip()
    return f" {response}\n"


def build_dataset(args: argparse.Namespace) -> dict:
    data_root = Path(args.cleaned_data_root)
    out_dir = Path(args.out)
    epoch_dir_paths = [out_dir / f"epoch_{epoch}" for epoch in range(int(args.epochs))]
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch_dir in epoch_dir_paths:
        epoch_dir.mkdir(parents=True, exist_ok=True)

    source_files = expand_source_files(data_root, str(args.source_files), str(args.source_globs))
    if not source_files:
        raise ValueError("--source-files or --source-globs must select at least one file")

    tokens: list[int] = []
    inst_start: list[int] = []
    inst_len: list[int] = []
    resp_start: list[int] = []
    resp_len: list[int] = []
    accepted_by_file: dict[str, int] = {}
    scanned_by_file: dict[str, int] = {}
    accepted_by_bucket: dict[str, int] = {}
    scanned_by_bucket: dict[str, int] = {}
    accepted_scored_by_file: dict[str, int] = {}
    accepted_unscored_by_file: dict[str, int] = {}
    rejected = {
        "missing_fields": 0,
        "empty_after_encoding": 0,
        "instruction_too_long": 0,
        "response_too_long": 0,
        "row_limit": 0,
        "scan_limit": 0,
        "bucket_limit": 0,
    }
    accepted_row_indices_by_file: dict[str, list[int]] = {}

    max_rows = int(args.max_rows)
    max_rows_per_file = int(args.max_rows_per_file)
    max_scan_rows_per_file = int(getattr(args, "max_scan_rows_per_file", 0))
    max_inst_bytes = int(args.max_inst_bytes)
    max_resp_bytes = int(args.max_resp_bytes)
    bucket_quotas = parse_int_mapping(str(getattr(args, "bucket_quotas", "")))
    bucket_max_rows_per_file = parse_int_mapping(str(getattr(args, "bucket_max_rows_per_file", "")))
    selection_mode = str(getattr(args, "selection_mode", "first"))
    utility_temperature = float(getattr(args, "utility_temperature", 0.0))
    utility_scores, utility_scores_loaded = load_utility_scores(str(getattr(args, "utility_score_jsonl", "")))

    for rel in source_files:
        path = data_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        bucket = source_bucket(rel)
        accepted_in_file = 0
        scanned_in_file = 0
        file_candidates: list[dict] = []
        for row in iter_rows(path):
            bucket_quota = int(bucket_quotas.get(bucket, 0))
            bucket_accepted = int(accepted_by_bucket.get(bucket, 0))
            if bucket_quota > 0 and bucket_accepted >= bucket_quota:
                rejected["bucket_limit"] += 1
                break
            if max_scan_rows_per_file > 0 and scanned_in_file >= max_scan_rows_per_file:
                rejected["scan_limit"] += 1
                break
            source_row_index = int(scanned_in_file)
            scanned_in_file += 1
            scanned_by_bucket[bucket] = int(scanned_by_bucket.get(bucket, 0)) + 1
            if max_rows > 0 and len(inst_start) >= max_rows:
                rejected["row_limit"] += 1
                break
            file_cap = int(bucket_max_rows_per_file.get(bucket, max_rows_per_file))
            if file_cap > 0 and accepted_in_file >= file_cap:
                break
            instruction = row.get("instruction")
            response = row.get("response")
            if instruction is None or response is None:
                rejected["missing_fields"] += 1
                continue
            inst = byte_ids(render_instruction(str(instruction)))
            resp = byte_ids(render_response(str(response))) + [EOS_TOKEN_ID]
            if not inst or not resp:
                rejected["empty_after_encoding"] += 1
                continue
            if max_inst_bytes > 0 and len(inst) > max_inst_bytes:
                rejected["instruction_too_long"] += 1
                continue
            if max_resp_bytes > 0 and len(resp) > max_resp_bytes:
                rejected["response_too_long"] += 1
                continue
            score_key = (str(Path(rel)), int(source_row_index))
            utility_score_present = score_key in utility_scores
            file_candidates.append(
                {
                    "row_index": int(source_row_index),
                    "inst": inst,
                    "resp": resp,
                    "utility": float(utility_scores.get(score_key, 0.0)),
                    "utility_score_present": bool(utility_score_present),
                }
            )
        for candidate in select_candidates(
            file_candidates,
            selection_mode=selection_mode,
            utility_temperature=utility_temperature,
            seed=int(args.seed) + len(accepted_by_file),
        ):
            bucket_quota = int(bucket_quotas.get(bucket, 0))
            bucket_accepted = int(accepted_by_bucket.get(bucket, 0))
            if bucket_quota > 0 and bucket_accepted >= bucket_quota:
                rejected["bucket_limit"] += 1
                break
            if max_rows > 0 and len(inst_start) >= max_rows:
                rejected["row_limit"] += 1
                break
            file_cap = int(bucket_max_rows_per_file.get(bucket, max_rows_per_file))
            if file_cap > 0 and accepted_in_file >= file_cap:
                break
            inst = candidate["inst"]
            resp = candidate["resp"]
            current = len(tokens)
            inst_start.append(current)
            inst_len.append(len(inst))
            tokens.extend(inst)
            resp_start.append(len(tokens))
            resp_len.append(len(resp))
            tokens.extend(resp)
            accepted_in_file += 1
            if bool(candidate.get("utility_score_present", False)):
                accepted_scored_by_file[rel] = int(accepted_scored_by_file.get(rel, 0)) + 1
            else:
                accepted_unscored_by_file[rel] = int(accepted_unscored_by_file.get(rel, 0)) + 1
            accepted_by_bucket[bucket] = int(accepted_by_bucket.get(bucket, 0)) + 1
            accepted_row_indices_by_file.setdefault(rel, []).append(int(candidate["row_index"]))
        accepted_by_file[rel] = accepted_in_file
        scanned_by_file[rel] = scanned_in_file
        if max_rows > 0 and len(inst_start) >= max_rows:
            break

    if not inst_start:
        raise ValueError("no rows accepted")

    token_array = np.asarray(tokens, dtype=np.uint16)
    np.save(out_dir / "tokens.npy", token_array)

    inst_start_array = np.asarray(inst_start, dtype=np.int64)
    inst_len_array = np.asarray(inst_len, dtype=np.int64)
    resp_start_array = np.asarray(resp_start, dtype=np.int64)
    resp_len_array = np.asarray(resp_len, dtype=np.int64)
    order = np.arange(len(inst_start_array), dtype=np.int64)

    for epoch, epoch_dir in enumerate(epoch_dir_paths):
        if bool(args.shuffle_epochs):
            rng = np.random.default_rng(int(args.seed) + int(epoch))
            epoch_order = rng.permutation(order)
        else:
            epoch_order = order
        np.save(epoch_dir / "inst_start.npy", inst_start_array[epoch_order])
        np.save(epoch_dir / "inst_len.npy", inst_len_array[epoch_order])
        np.save(epoch_dir / "resp_start.npy", resp_start_array[epoch_order])
        np.save(epoch_dir / "resp_len.npy", resp_len_array[epoch_order])

    shifted_lengths = inst_len_array + resp_len_array - 1
    metadata = {
        "contract": "utf8_byte_prefixlm_v1",
        "tokenizer_info": {
            "kind": "tokenizer_free_utf8_byte_shifted",
            "pad_token_id": PAD_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "byte_offset": BYTE_OFFSET,
            "vocab_size": BYTE_VOCAB_SIZE,
        },
        "vocab_size": BYTE_VOCAB_SIZE,
        "max_seq_len": int(max(shifted_lengths)),
        "total_length": int(token_array.shape[0]),
        "rows": int(len(inst_start_array)),
        "epochs": int(args.epochs),
        "source_files": source_files,
        "source_bucket_contract": {
            "plain_language": (
                "Rows are balanced by curriculum bucket before byte encoding, "
                "so many-shard synthetic folders cannot crowd out normal "
                "language, Korean/English multilingual text, or reasoning."
            ),
            "bucket_quotas": bucket_quotas,
            "bucket_max_rows_per_file": bucket_max_rows_per_file,
        },
        "data_selection_contract": {
            "selection_mode": selection_mode,
            "utility_score_jsonl": str(getattr(args, "utility_score_jsonl", "")),
            "utility_scores_loaded": int(utility_scores_loaded),
            "utility_temperature": float(utility_temperature),
            "accepted_scored_rows": int(sum(accepted_scored_by_file.values())),
            "accepted_unscored_rows": int(sum(accepted_unscored_by_file.values())),
            "accepted_scored_by_file": accepted_scored_by_file,
            "accepted_unscored_by_file": accepted_unscored_by_file,
            "plain_language": (
                "OPUS-compatible data selection hook: this builder consumes "
                "per-row optimizer/projected-utility scores and materializes "
                "a byte PrefixLM sample through deterministic top-utility or "
                "Boltzmann utility sampling. Scored rows are ranked before "
                "unscored rows, because missing score means not auditioned, not "
                "neutral utility. It is the selection layer, not the full OPUS "
                "scorer."
            ),
        },
        "accepted_by_file": accepted_by_file,
        "accepted_row_indices_by_file": accepted_row_indices_by_file,
        "scanned_by_file": scanned_by_file,
        "accepted_by_bucket": accepted_by_bucket,
        "scanned_by_bucket": scanned_by_bucket,
        "rejected": rejected,
        "mean_shifted_row_len": float(np.mean(shifted_lengths)),
        "p95_shifted_row_len": float(np.percentile(shifted_lengths, 95)),
        "max_shifted_row_len": int(np.max(shifted_lengths)),
        "plain_language_read": (
            "This sample removes BPE. The model reads UTF-8 bytes directly, so "
            "Korean and unseen strings are no longer fragmented by a learned "
            "subword vocabulary. The cost is longer sequences."
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-data-root",
        default="/mnt/sdc1/datasets/hrm-text-data-io-cleaned-20260515",
    )
    parser.add_argument(
        "--source-files",
        default=(
            "data/no_robots.jsonl data/natural_reasoning.jsonl "
            "data/webinstruct_verified.jsonl data/Platypus/openbookqa.jsonl "
            "data/gsm8k_train.jsonl"
        ),
    )
    parser.add_argument(
        "--source-globs",
        default="",
        help=(
            "Whitespace-separated glob patterns relative to --cleaned-data-root. "
            "Use this for broad Stage95 byte samples over data_clustered/*.parquet."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument(
        "--max-scan-rows-per-file",
        type=int,
        default=0,
        help=(
            "Stop scanning a source file after this many rows, even if few rows "
            "were accepted. This keeps broad parquet curricula from stalling on "
            "one oversized shelf. Use 0 for unlimited scans."
        ),
    )
    parser.add_argument("--max-inst-bytes", type=int, default=1536)
    parser.add_argument("--max-resp-bytes", type=int, default=1024)
    parser.add_argument(
        "--bucket-quotas",
        default="",
        help=(
            "Optional bucket=row_count mapping. Buckets include "
            "general_instruction, tasksource, flan_instruction_qa, "
            "flan_multilingual_translation, flan_other, reasoning, math, "
            "synthetic_math_like, and other."
        ),
    )
    parser.add_argument(
        "--bucket-max-rows-per-file",
        default="",
        help=(
            "Optional bucket=row_count mapping overriding --max-rows-per-file. "
            "Use this to prevent many-shard synthetic shelves from dominating "
            "while letting scarce general-language files contribute enough rows."
        ),
    )
    parser.add_argument(
        "--utility-score-jsonl",
        default="",
        help=(
            "Optional JSONL with source_file,row_index,utility scores from an "
            "OPUS-style optimizer/projected-utility scorer."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("first", "utility"),
        default="first",
        help="Use first-seen rows or utility-score-selected rows within each source file.",
    )
    parser.add_argument(
        "--utility-temperature",
        type=float,
        default=0.0,
        help="Boltzmann temperature for utility selection. Use 0 for deterministic top-utility order.",
    )
    parser.add_argument("--shuffle-epochs", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=955)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_dataset(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
