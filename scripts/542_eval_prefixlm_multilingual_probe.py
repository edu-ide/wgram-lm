#!/usr/bin/env python3
"""Evaluate a native PrefixLM checkpoint on a small multilingual probe suite."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    family: str
    language: str
    instruction: str
    expected_contains: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_trainer_module() -> Any:
    root = repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    stage90_path = root / "scripts" / "534_train_native_prefixlm_dataio_stage90.py"
    path = stage90_path if stage90_path.exists() else root / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("native_prefixlm_dataio_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_probe_cases(path: str | Path) -> list[ProbeCase]:
    cases: list[ProbeCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            expected = tuple(str(value) for value in row.get("expected_contains", []))
            if not expected:
                raise ValueError(f"probe row {line_number} has no expected_contains")
            cases.append(
                ProbeCase(
                    case_id=str(row["case_id"]),
                    family=str(row["family"]),
                    language=str(row["language"]),
                    instruction=str(row["instruction"]),
                    expected_contains=expected,
                )
            )
    if not cases:
        raise ValueError(f"no probe cases loaded from {path}")
    return cases


def normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def strip_response_text(text: str, eoa: str = "<|box_end|>") -> str:
    value = str(text)
    if eoa and eoa in value:
        value = value.split(eoa, 1)[0]
    value = SPECIAL_TOKEN_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def score_case(response: str, expected_contains: tuple[str, ...]) -> dict[str, Any]:
    cleaned = strip_response_text(response)
    normalized = normalize_for_match(cleaned)
    hits = [
        expected
        for expected in expected_contains
        if normalize_for_match(expected) in normalized
    ]
    return {
        "hit": bool(hits),
        "matched": hits,
        "cleaned_response": cleaned,
        "cleaned_char_count": len(cleaned),
        "nonempty": bool(cleaned),
        "has_repeated_char_ngram": has_repeated_char_ngram(cleaned),
        "special_token_count": len(SPECIAL_TOKEN_RE.findall(str(response))),
    }


def has_repeated_char_ngram(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text))
    if len(compact) < 4:
        return False
    for size in range(2, min(8, len(compact) // 2) + 1):
        for start in range(0, len(compact) - (2 * size) + 1):
            piece = compact[start : start + size]
            if piece and compact[start + size : start + (2 * size)] == piece:
                return True
    return False


def resolve_tokenizer_path(
    *,
    checkpoint: dict[str, Any],
    tokenizer_path: str = "",
) -> Path:
    candidates: list[Path] = []
    if tokenizer_path:
        candidates.append(Path(tokenizer_path))

    tokenizer_info = {}
    dataset_info = checkpoint.get("dataset") or {}
    if isinstance(dataset_info, dict):
        tokenizer_info = dict(dataset_info.get("tokenizer_info") or {})
    metadata_path = str(tokenizer_info.get("tokenizer_path") or "")
    if metadata_path:
        candidates.append(Path(metadata_path))

    candidates.append(
        repo_root()
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
    raise FileNotFoundError("could not resolve PrefixLM tokenizer.json")


def load_tokenizer(tokenizer_path: Path) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def checkpoint_args(trainer: Any, checkpoint: dict[str, Any]) -> argparse.Namespace:
    parser = trainer.build_arg_parser()
    defaults = vars(
        parser.parse_args(
            [
                "--sampled-data",
                "__probe_only__",
                "--out-dir",
                "__probe_only__",
            ]
        )
    )
    defaults.update(dict(checkpoint.get("args") or {}))
    return argparse.Namespace(**defaults)


def infer_vocab_size(checkpoint: dict[str, Any], args: argparse.Namespace) -> int:
    model_info = dict(checkpoint.get("model") or {})
    if model_info.get("vocab_size"):
        return int(model_info["vocab_size"])
    if int(getattr(args, "model_vocab_size", 0) or 0) > 0:
        return int(args.model_vocab_size)
    state = checkpoint.get("model_state_dict") or {}
    for key in ("lm_head.weight", "text_embed.weight", "embed.weight"):
        tensor = state.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            return int(tensor.shape[0])
    raise ValueError("could not infer model vocab size from checkpoint")


def token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.token_to_id(str(token))
    if value is None:
        raise ValueError(f"tokenizer does not contain special token {token!r}")
    return int(value)


def build_instruction_ids(
    *,
    tokenizer: Any,
    instruction: str,
    condition: str,
    tokenizer_info: dict[str, Any],
) -> list[int]:
    boq = str(tokenizer_info.get("boq") or "<|im_start|>")
    eoq = str(tokenizer_info.get("eoq") or "<|im_end|>")
    mapping = {
        "direct": "<|object_ref_start|>",
        "cot": "<|object_ref_end|>",
        "noisy": "<|quad_start|>",
        "synth": "<|quad_end|>",
    }
    mapping.update({str(k): str(v) for k, v in dict(tokenizer_info.get("condition_mapping") or {}).items()})

    ids = [token_id(tokenizer, boq)]
    for label in str(condition).split(","):
        label = label.strip()
        if not label:
            continue
        marker = mapping.get(label)
        if marker is None:
            raise ValueError(f"unknown condition label {label!r}; known={sorted(mapping)}")
        ids.append(token_id(tokenizer, marker))
    encoded = tokenizer.encode(str(instruction), add_special_tokens=False).ids
    ids.extend(int(value) for value in encoded)
    ids.append(token_id(tokenizer, eoq))
    return ids


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=False)


def generate_one(
    *,
    model: torch.nn.Module,
    prefix_ids: list[int],
    eoa_id: int,
    device: torch.device,
    think_steps: int,
    seq_len: int,
    max_new_tokens: int,
) -> list[int]:
    current = [int(token_id) for token_id in prefix_ids]
    if len(current) >= int(seq_len):
        current = current[-(int(seq_len) - 1) :]
    generated: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            if len(current) >= int(seq_len):
                break
            input_ids = torch.tensor([current], dtype=torch.long, device=device)
            logits = model(input_ids, think_steps=int(think_steps))
            next_id = int(logits[0, -1].argmax(dim=-1).detach().cpu().item())
            generated.append(next_id)
            current.append(next_id)
            if next_id == int(eoa_id):
                break
    return generated


def summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    return {
        name: {
            "cases": len(items),
            "hits": sum(1 for item in items if item["hit"]),
            "accuracy": sum(1 for item in items if item["hit"]) / len(items),
            "nonempty_rate": sum(1 for item in items if item["nonempty"]) / len(items),
            "degenerate_repetition_rate": sum(
                1 for item in items if item["has_repeated_char_ngram"]
            )
            / len(items),
            "avg_cleaned_chars": sum(int(item["cleaned_char_count"]) for item in items)
            / len(items),
        }
        for name, items in sorted(buckets.items())
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--probe-jsonl",
        default=str(repo_root() / "data" / "eval" / "prefixlm_multilingual_probe.jsonl"),
    )
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--condition", default="direct")
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    trainer = load_trainer_module()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    train_args = checkpoint_args(trainer, checkpoint)

    tokenizer_path = resolve_tokenizer_path(
        checkpoint=checkpoint,
        tokenizer_path=str(args.tokenizer_path),
    )
    tokenizer = load_tokenizer(tokenizer_path)
    dataset_info = dict(checkpoint.get("dataset") or {})
    tokenizer_info = dict(dataset_info.get("tokenizer_info") or {})
    eoa_text = str(tokenizer_info.get("eoa") or "<|box_end|>")
    eoa_id = token_id(tokenizer, eoa_text)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    vocab_size = infer_vocab_size(checkpoint, train_args)
    model = trainer.build_model(train_args, vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    think_steps = (
        int(args.think_steps)
        if int(args.think_steps) > 0
        else int(getattr(train_args, "train_think_steps", 0))
    )
    seq_len = int(getattr(train_args, "seq_len", 128))
    cases = load_probe_cases(args.probe_jsonl)
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]

    rows: list[dict[str, Any]] = []
    for case in cases:
        prefix_ids = build_instruction_ids(
            tokenizer=tokenizer,
            instruction=case.instruction,
            condition=str(args.condition),
            tokenizer_info=tokenizer_info,
        )
        generated_ids = generate_one(
            model=model,
            prefix_ids=prefix_ids,
            eoa_id=int(eoa_id),
            device=device,
            think_steps=int(think_steps),
            seq_len=int(seq_len),
            max_new_tokens=int(args.max_new_tokens),
        )
        decoded = decode_ids(tokenizer, generated_ids)
        scored = score_case(decoded, case.expected_contains)
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "language": case.language,
                "instruction": case.instruction,
                "expected_contains": list(case.expected_contains),
                "generated_ids": generated_ids,
                "raw_response": decoded,
                **scored,
            }
        )

    hits = sum(1 for row in rows if row["hit"])
    nonempty = sum(1 for row in rows if row["nonempty"])
    repeated = sum(1 for row in rows if row["has_repeated_char_ngram"])
    total_cleaned_chars = sum(int(row["cleaned_char_count"]) for row in rows)
    total_special_tokens = sum(int(row["special_token_count"]) for row in rows)
    report = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", 0)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "target_tokens_seen": int(checkpoint.get("target_tokens_seen", 0)),
        "probe_jsonl": str(args.probe_jsonl),
        "tokenizer_path": str(tokenizer_path),
        "condition": str(args.condition),
        "think_steps": int(think_steps),
        "seq_len": int(seq_len),
        "eoa": {"text": eoa_text, "token_id": int(eoa_id)},
        "cases": len(rows),
        "hits": int(hits),
        "accuracy": float(hits / len(rows)) if rows else 0.0,
        "nonempty_rate": float(nonempty / len(rows)) if rows else 0.0,
        "degenerate_repetition_rate": float(repeated / len(rows)) if rows else 0.0,
        "avg_cleaned_chars": float(total_cleaned_chars / len(rows)) if rows else 0.0,
        "special_tokens_per_case": float(total_special_tokens / len(rows)) if rows else 0.0,
        "by_language": summarize_groups(rows, "language"),
        "by_family": summarize_groups(rows, "family"),
        "results": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
