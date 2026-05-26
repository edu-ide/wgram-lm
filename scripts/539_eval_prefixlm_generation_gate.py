#!/usr/bin/env python3
"""Evaluate HRM-Text PrefixLM checkpoints with first-token and generation gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_trainer_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("native_prefixlm_dataio_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_tokenizer_path(sampled_data: Path, tokenizer_path: str) -> Path | None:
    candidates: list[Path] = []
    if tokenizer_path:
        candidates.append(Path(tokenizer_path))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "references" / "official" / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json")
    for parent in [sampled_data, *sampled_data.parents]:
        candidates.append(parent / "references" / "official" / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json")
        candidates.append(parent / "trained_tokenizers" / "bpe" / "tokenizer.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_tokenizer(tokenizer_path: Path | None):
    if not tokenizer_path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(str(tokenizer_path))


def decode_ids(tokenizer: Any | None, token_ids: list[int]) -> str:
    if tokenizer is None:
        return " ".join(str(int(token_id)) for token_id in token_ids)
    try:
        return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=False)
    except Exception:
        return " ".join(str(int(token_id)) for token_id in token_ids)


def condition_from_instruction_text(text: str, tokenizer_info: dict[str, Any]) -> str:
    value = str(text).strip()
    boq = str(tokenizer_info.get("boq") or "")
    if boq and value.startswith(boq):
        value = value[len(boq) :].lstrip()
    mapping = dict(tokenizer_info.get("condition_mapping") or {})
    for label, marker in sorted(mapping.items(), key=lambda item: len(str(item[1])), reverse=True):
        marker_text = str(marker)
        if marker_text and value.startswith(marker_text):
            return str(label)
    return "unknown"


def filter_dataset_by_condition(
    dataset: Any,
    *,
    tokenizer: Any | None,
    tokenizer_info: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    requested = str(condition)
    if requested == "all":
        return {"condition": "all", "rows_before": int(len(dataset)), "rows_after": int(len(dataset))}
    if tokenizer is None:
        raise ValueError("--condition requires a tokenizer")

    before = int(len(dataset))
    kept: list[int] = []
    counts: Counter[str] = Counter()
    for source_row in [int(value) for value in dataset.row_indices.tolist()]:
        inst = dataset._slice_tokens(dataset.inst_start[source_row], dataset.inst_len[source_row])
        text = decode_ids(tokenizer, [int(token_id) for token_id in inst.astype(np.int64).tolist()])
        row_condition = condition_from_instruction_text(text, tokenizer_info)
        counts[row_condition] += 1
        if row_condition == requested:
            kept.append(source_row)
    if not kept:
        raise ValueError(f"no rows found for condition={requested!r}; counts={dict(counts)}")
    dataset.row_indices = np.asarray(kept, dtype=np.int64)
    dataset.shifted_lengths = (
        dataset.inst_len[dataset.row_indices] + dataset.resp_len[dataset.row_indices] - 1
    ).astype(np.int64)
    return {
        "condition": requested,
        "rows_before": before,
        "rows_after": int(len(dataset)),
        "condition_counts": dict(sorted(counts.items())),
    }


def first_response_stats(
    *,
    trainer: Any,
    model: torch.nn.Module,
    dataset: Any,
    tokenizer: Any | None,
    eoa_id: int,
    device: torch.device,
    think_steps: int,
    max_rows: int,
    batch_size: int,
) -> dict[str, Any]:
    rows = min(int(max_rows), len(dataset))
    subset = torch.utils.data.Subset(dataset, range(rows))
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=trainer.collate_prefixlm_rows,
        drop_last=False,
    )
    total = 0
    correct = 0
    eoa_top1 = 0
    gold_probability_sum = 0.0
    eoa_probability_sum = 0.0
    top1_counter: Counter[int] = Counter()
    target_counter: Counter[int] = Counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            start_mask = batch["response_start_mask"].to(device).bool()
            hidden = model.forward_hidden(input_ids, think_steps=int(think_steps))
            start_hidden = hidden[start_mask]
            start_labels = labels[start_mask]
            if int(start_labels.numel()) == 0:
                continue
            logits = model.lm_head(start_hidden)
            probabilities = logits.softmax(dim=-1)
            top1 = logits.argmax(dim=-1)
            total += int(start_labels.numel())
            correct += int(top1.eq(start_labels).sum().detach().cpu().item())
            eoa_top1 += int(top1.eq(int(eoa_id)).sum().detach().cpu().item())
            gold_probability_sum += float(
                probabilities.gather(1, start_labels.unsqueeze(1)).sum().detach().cpu().item()
            )
            eoa_probability_sum += float(
                probabilities[:, int(eoa_id)].sum().detach().cpu().item()
            )
            top1_counter.update(int(value) for value in top1.detach().cpu().tolist())
            target_counter.update(int(value) for value in start_labels.detach().cpu().tolist())
    if total <= 0:
        raise ValueError("no first response positions found")
    common_top1 = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)]),
        }
        for token_id, count in top1_counter.most_common(10)
    ]
    common_targets = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)]),
        }
        for token_id, count in target_counter.most_common(10)
    ]
    return {
        "rows": int(rows),
        "positions": int(total),
        "accuracy": float(correct / total),
        "eoa_top1_fraction": float(eoa_top1 / total),
        "gold_probability": float(gold_probability_sum / total),
        "eoa_probability": float(eoa_probability_sum / total),
        "common_top1": common_top1,
        "common_targets": common_targets,
    }


def gold_response_until_eoa(resp: np.ndarray, eoa_id: int) -> list[int]:
    out: list[int] = []
    for token_id in resp.astype(np.int64).tolist():
        out.append(int(token_id))
        if int(token_id) == int(eoa_id):
            break
    return out


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
    generated: list[int] = []
    current = [int(token_id) for token_id in prefix_ids]
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


def generation_stats(
    *,
    model: torch.nn.Module,
    dataset: Any,
    tokenizer: Any | None,
    eoa_id: int,
    device: torch.device,
    think_steps: int,
    seq_len: int,
    max_rows: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows = min(int(max_rows), len(dataset))
    exact = 0
    starts_with_eoa = 0
    ended_with_eoa = 0
    prefix_matches = 0
    prefix_positions = 0
    repeated_token_loops = 0
    samples: list[dict[str, Any]] = []
    for index in range(rows):
        source_row = int(dataset.row_indices[int(index)])
        inst = dataset._slice_tokens(dataset.inst_start[source_row], dataset.inst_len[source_row])
        resp = dataset._slice_tokens(dataset.resp_start[source_row], dataset.resp_len[source_row])
        gold = gold_response_until_eoa(resp, int(eoa_id))
        generated = generate_one(
            model=model,
            prefix_ids=[int(token_id) for token_id in inst.astype(np.int64).tolist()],
            eoa_id=int(eoa_id),
            device=device,
            think_steps=int(think_steps),
            seq_len=int(seq_len),
            max_new_tokens=int(max_new_tokens),
        )
        if generated and generated[0] == int(eoa_id):
            starts_with_eoa += 1
        if generated and generated[-1] == int(eoa_id):
            ended_with_eoa += 1
        if generated == gold:
            exact += 1
        compare_len = min(len(generated), len(gold))
        prefix_positions += compare_len
        prefix_matches += sum(1 for left, right in zip(generated[:compare_len], gold[:compare_len]) if left == right)
        if len(generated) >= 4:
            most_common_count = Counter(generated).most_common(1)[0][1]
            if most_common_count / float(len(generated)) >= 0.8:
                repeated_token_loops += 1
        if len(samples) < 16:
            samples.append(
                {
                    "row_index": int(source_row),
                    "instruction_head": decode_ids(tokenizer, [int(token_id) for token_id in inst[:32].tolist()]),
                    "gold_ids": gold,
                    "generated_ids": generated,
                    "gold": decode_ids(tokenizer, gold),
                    "generated": decode_ids(tokenizer, generated),
                    "exact": bool(generated == gold),
                    "starts_with_eoa": bool(bool(generated) and generated[0] == int(eoa_id)),
                }
            )
    return {
        "rows": int(rows),
        "exact": int(exact),
        "exact_fraction": float(exact / rows) if rows else 0.0,
        "starts_with_eoa": int(starts_with_eoa),
        "starts_with_eoa_fraction": float(starts_with_eoa / rows) if rows else 0.0,
        "ended_with_eoa": int(ended_with_eoa),
        "ended_with_eoa_fraction": float(ended_with_eoa / rows) if rows else 0.0,
        "prefix_token_accuracy": float(prefix_matches / prefix_positions) if prefix_positions else 0.0,
        "prefix_positions": int(prefix_positions),
        "repeated_token_loops": int(repeated_token_loops),
        "repeated_token_loop_fraction": float(repeated_token_loops / rows) if rows else 0.0,
        "samples": samples,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-first-token-rows", type=int, default=512)
    parser.add_argument("--first-token-batch-size", type=int, default=16)
    parser.add_argument("--max-generation-rows", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--condition", default="all")
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    trainer = load_trainer_module()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    train_args = argparse.Namespace(**dict(checkpoint["args"]))
    sampled_data = Path(args.sampled_data or train_args.sampled_data)
    dataset = trainer.DataIOSampledPrefixLMDataset(
        sampled_data,
        seq_len=int(train_args.seq_len),
        epoch=int(args.epoch),
        target_only=not bool(getattr(train_args, "train_instruction_tokens", False)),
        max_rows=None,
        drop_overlength=not bool(getattr(train_args, "keep_overlength", False)),
    )
    metadata = trainer.load_prefixlm_metadata(sampled_data)
    tokenizer_info = dict(metadata.tokenizer_info or {})
    tokenizer_path = resolve_tokenizer_path(
        sampled_data,
        str(tokenizer_info.get("tokenizer_path") or ""),
    )
    tokenizer = load_tokenizer(tokenizer_path)
    eoa_text = str(tokenizer_info.get("eoa") or "<|box_end|>")
    eoa_id = tokenizer.token_to_id(eoa_text) if tokenizer is not None else None
    if eoa_id is None:
        raise ValueError(f"could not resolve EOA token id for {eoa_text!r}")
    condition_filter = filter_dataset_by_condition(
        dataset,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        condition=str(args.condition),
    )
    model_info = dict(checkpoint.get("model") or {})
    vocab_size = int(model_info.get("vocab_size") or getattr(train_args, "model_vocab_size", 0) or dataset.summary()["model_vocab_size"])
    device = torch.device(str(args.device))
    model = trainer.build_model(train_args, vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    think_steps = int(args.think_steps) if int(args.think_steps) > 0 else int(train_args.train_think_steps)
    first_stats = first_response_stats(
        trainer=trainer,
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        eoa_id=int(eoa_id),
        device=device,
        think_steps=int(think_steps),
        max_rows=int(args.max_first_token_rows),
        batch_size=int(args.first_token_batch_size),
    )
    gen_stats = generation_stats(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        eoa_id=int(eoa_id),
        device=device,
        think_steps=int(think_steps),
        seq_len=int(train_args.seq_len),
        max_rows=int(args.max_generation_rows),
        max_new_tokens=int(args.max_new_tokens),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", 0)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "target_tokens_seen": int(checkpoint.get("target_tokens_seen", 0)),
        "sampled_data": str(sampled_data),
        "epoch": int(args.epoch),
        "seq_len": int(train_args.seq_len),
        "think_steps": int(think_steps),
        "eoa": {"text": eoa_text, "token_id": int(eoa_id)},
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else "",
        "condition_filter": condition_filter,
        "first_response": first_stats,
        "generation": gen_stats,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
