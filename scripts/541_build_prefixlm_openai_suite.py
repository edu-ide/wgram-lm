#!/usr/bin/env python3
"""Build an OpenAI-compatible eval suite from HRM-Text PrefixLM sampled rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class SuiteRow(NamedTuple):
    source_row: int
    condition: str
    instruction: str
    answer: str


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_tokenizer_path(sampled_data: Path, tokenizer_path: str = "") -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if tokenizer_path:
        candidates.append(Path(tokenizer_path))
    metadata = load_json(sampled_data / "metadata.json")
    metadata_tokenizer = str((metadata.get("tokenizer_info") or {}).get("tokenizer_path") or "")
    if metadata_tokenizer:
        candidates.append(Path(metadata_tokenizer))
    candidates.append(
        repo_root
        / "references"
        / "official"
        / "data_io"
        / "trained_tokenizers"
        / "bpe"
        / "tokenizer.json"
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve tokenizer for {sampled_data}")


def load_tokenizer(tokenizer_path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def decode_ids(tokenizer: Any, token_ids: np.ndarray) -> str:
    return tokenizer.decode([int(token_id) for token_id in token_ids.astype(np.int64).tolist()], skip_special_tokens=False)


def strip_instruction_wrappers(text: str, tokenizer_info: dict[str, Any]) -> tuple[str, str]:
    value = str(text).strip()
    boq = str(tokenizer_info.get("boq") or "")
    eoq = str(tokenizer_info.get("eoq") or "")
    if boq and value.startswith(boq):
        value = value[len(boq) :].lstrip()

    condition = "unknown"
    mapping = dict(tokenizer_info.get("condition_mapping") or {})
    for label, marker in sorted(mapping.items(), key=lambda item: len(str(item[1])), reverse=True):
        marker_text = str(marker)
        if marker_text and value.startswith(marker_text):
            condition = str(label)
            value = value[len(marker_text) :].lstrip()
            break

    if eoq and value.endswith(eoq):
        value = value[: -len(eoq)].rstrip()
    return condition, value.strip()


def strip_response_text(text: str, eoa: str) -> str:
    value = str(text).strip()
    if str(eoa) and str(eoa) in value:
        value = value.split(str(eoa), 1)[0]
    return value.strip()


def resolve_prompt_style(prompt_style: str, condition: str) -> str:
    if str(prompt_style) == "auto":
        return "full_solution" if str(condition) == "cot" else "final_answer"
    return str(prompt_style)


def build_qwen_prompt(instruction: str, *, prompt_style: str = "final_answer") -> str:
    if str(prompt_style) == "full_solution":
        return (
            "Solve the following problem. Return the full solution, including reasoning, "
            "and end with the final answer in \\boxed{}.\n\n"
            f"Problem:\n{instruction.strip()}\n\nSolution:"
        )
    return (
        "Solve the following problem. Return only the final answer, with no explanation.\n\n"
        f"Problem:\n{instruction.strip()}\n\nAnswer:"
    )


def load_suite_rows(
    *,
    sampled_data: Path,
    tokenizer_path: Path,
    epoch: int,
    condition_filter: str,
    max_rows: int,
) -> tuple[list[SuiteRow], dict[str, Any]]:
    metadata = load_json(sampled_data / "metadata.json")
    tokenizer_info = dict(metadata.get("tokenizer_info") or {})
    eoa = str(tokenizer_info.get("eoa") or "<|box_end|>")
    tokenizer = load_tokenizer(tokenizer_path)
    tokens = np.load(sampled_data / "tokens.npy", mmap_mode="r")
    epoch_dir = sampled_data / f"epoch_{int(epoch)}"
    inst_start = np.load(epoch_dir / "inst_start.npy", mmap_mode="r")
    inst_len = np.load(epoch_dir / "inst_len.npy", mmap_mode="r")
    resp_start = np.load(epoch_dir / "resp_start.npy", mmap_mode="r")
    resp_len = np.load(epoch_dir / "resp_len.npy", mmap_mode="r")

    rows: list[SuiteRow] = []
    condition_counts: Counter[str] = Counter()
    skipped_empty_answer = 0
    for row_index in range(int(inst_start.shape[0])):
        inst = tokens[int(inst_start[row_index]) : int(inst_start[row_index] + inst_len[row_index])]
        resp = tokens[int(resp_start[row_index]) : int(resp_start[row_index] + resp_len[row_index])]
        condition, instruction = strip_instruction_wrappers(decode_ids(tokenizer, inst), tokenizer_info)
        answer = strip_response_text(decode_ids(tokenizer, resp), eoa)
        condition_counts[condition] += 1
        if str(condition_filter) != "all" and condition != str(condition_filter):
            continue
        if not answer:
            skipped_empty_answer += 1
            continue
        rows.append(
            SuiteRow(
                source_row=int(row_index),
                condition=condition,
                instruction=instruction,
                answer=answer,
            )
        )
        if int(max_rows) > 0 and len(rows) >= int(max_rows):
            break

    summary = {
        "sampled_data": str(sampled_data),
        "epoch": int(epoch),
        "condition_filter": str(condition_filter),
        "condition_counts_seen": dict(sorted(condition_counts.items())),
        "skipped_empty_answer": int(skipped_empty_answer),
    }
    return rows, summary


def write_suite_rows(
    rows: list[SuiteRow],
    *,
    out_jsonl: Path,
    suite_id: str,
    prompt_protocol: str,
    prompt_style: str = "final_answer",
) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    encoded_rows = []
    condition_counts: Counter[str] = Counter()
    for row in rows:
        condition_counts[row.condition] += 1
        encoded_rows.append(
            {
                "suite_id": str(suite_id),
                "prompt_protocol": str(prompt_protocol),
                "case_id": f"epoch-row-{int(row.source_row)}",
                "family": "hrm_text_data_io",
                "condition": row.condition,
                "source_row": int(row.source_row),
                "qwen_prompt": build_qwen_prompt(
                    row.instruction,
                    prompt_style=resolve_prompt_style(prompt_style, row.condition),
                ),
                "answer_text": row.answer,
            }
        )
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in encoded_rows),
        encoding="utf-8",
    )
    return {
        "out_jsonl": str(out_jsonl),
        "suite_id": str(suite_id),
        "prompt_protocol": str(prompt_protocol),
        "rows": int(len(rows)),
        "condition_counts": dict(sorted(condition_counts.items())),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--condition", default="direct")
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument(
        "--prompt-style",
        choices=["auto", "final_answer", "full_solution"],
        default="auto",
    )
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-report", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sampled_data = Path(args.sampled_data)
    tokenizer_path = resolve_tokenizer_path(sampled_data, str(args.tokenizer_path))
    rows, load_summary = load_suite_rows(
        sampled_data=sampled_data,
        tokenizer_path=tokenizer_path,
        epoch=int(args.epoch),
        condition_filter=str(args.condition),
        max_rows=int(args.max_rows),
    )
    suite_id = f"hrm_text_data_io_prefixlm_epoch{int(args.epoch)}_{args.condition}"
    prompt_protocol = {
        "auto": "hrm_text_data_io_auto_by_condition_v1",
        "final_answer": "hrm_text_data_io_answer_only_v1",
        "full_solution": "hrm_text_data_io_full_solution_v1",
    }[str(args.prompt_style)]
    report = write_suite_rows(
        rows,
        out_jsonl=Path(args.out_jsonl),
        suite_id=suite_id,
        prompt_protocol=prompt_protocol,
        prompt_style=str(args.prompt_style),
    )
    report.update(load_summary)
    report["tokenizer_path"] = str(tokenizer_path)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out_report):
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_report).write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
