#!/usr/bin/env python3
"""Evaluate BLT-D PrefixLM checkpoints with first-token and generation gates."""

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


ROOT = Path(__file__).resolve().parents[1]
IGNORE_LABEL_ID = -100


def load_depth_probe_module() -> Any:
    path = ROOT / "scripts" / "560_eval_blt_depth_residual_probe.py"
    spec = importlib.util.spec_from_file_location("blt_depth_residual_probe_for_generation_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint_sampled_data(checkpoint_path: Path) -> str:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = payload.get("args")
    if not isinstance(args, dict):
        return ""
    return str(args.get("sampled_data") or "")


def _byte_token_text(token_id: int, tokenizer_info: dict[str, Any]) -> str:
    pad_id = int(tokenizer_info.get("pad_token_id", 0))
    eos_id = int(tokenizer_info.get("eos_token_id", 1))
    byte_offset = int(tokenizer_info.get("byte_offset", 2))
    if int(token_id) == pad_id:
        return "<pad>"
    if int(token_id) == eos_id:
        return "<eos>"
    value = int(token_id) - byte_offset
    if value < 0 or value > 255:
        return f"<id:{int(token_id)}>"
    return bytes([value]).decode("utf-8", errors="replace")


def decode_ids(tokenizer: Any | None, token_ids: list[int], tokenizer_info: dict[str, Any]) -> str:
    kind = str(tokenizer_info.get("kind") or "")
    if kind.startswith("tokenizer_free") or "byte" in kind:
        pieces: list[str] = []
        byte_buffer: list[int] = []
        byte_offset = int(tokenizer_info.get("byte_offset", 2))
        for token_id in [int(value) for value in token_ids]:
            value = token_id - byte_offset
            if 0 <= value <= 255:
                byte_buffer.append(value)
                continue
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()
            pieces.append(_byte_token_text(token_id, tokenizer_info))
        if byte_buffer:
            pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
        return "".join(pieces)
    if tokenizer is not None:
        try:
            return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=False)
        except Exception:
            pass
    return " ".join(str(int(token_id)) for token_id in token_ids)


def resolve_eoa_id(tokenizer_info: dict[str, Any], tokenizer: Any | None) -> int:
    eoa_text = tokenizer_info.get("eoa")
    if eoa_text is not None and tokenizer is not None:
        try:
            token_id = tokenizer.token_to_id(str(eoa_text))
            if token_id is not None:
                return int(token_id)
        except Exception:
            pass
    if tokenizer_info.get("eos_token_id") is not None:
        return int(tokenizer_info["eos_token_id"])
    return 1


def gold_response_until_eoa(resp: np.ndarray, eoa_id: int) -> list[int]:
    out: list[int] = []
    for token_id in resp.astype(np.int64).tolist():
        out.append(int(token_id))
        if int(token_id) == int(eoa_id):
            break
    return out


def first_response_stats(
    *,
    prefix: Any,
    model: torch.nn.Module,
    dataset: Any,
    tokenizer: Any | None,
    tokenizer_info: dict[str, Any],
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
        collate_fn=prefix.collate_prefixlm_rows,
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
            batch = prefix.trim_prefixlm_batch_to_max_valid_length(batch)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_mask = batch["response_start_mask"].to(device).bool()
            logits, _ = model.forward_logits_and_decoder_hidden(
                input_ids,
                attention_mask,
                think_steps=int(think_steps),
            )
            start_logits = logits[start_mask]
            start_labels = labels[start_mask]
            if int(start_labels.numel()) == 0:
                continue
            probabilities = start_logits.softmax(dim=-1)
            top1 = start_logits.argmax(dim=-1)
            total += int(start_labels.numel())
            correct += int(top1.eq(start_labels).sum().detach().cpu().item())
            eoa_top1 += int(top1.eq(int(eoa_id)).sum().detach().cpu().item())
            gold_probability_sum += float(
                probabilities.gather(1, start_labels.unsqueeze(1)).sum().detach().cpu().item()
            )
            eoa_probability_sum += float(probabilities[:, int(eoa_id)].sum().detach().cpu().item())
            top1_counter.update(int(value) for value in top1.detach().cpu().tolist())
            target_counter.update(int(value) for value in start_labels.detach().cpu().tolist())
    if total <= 0:
        raise ValueError("no first response positions found")
    common_top1 = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)], tokenizer_info),
        }
        for token_id, count in top1_counter.most_common(10)
    ]
    common_targets = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)], tokenizer_info),
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


def summarize_response_token_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_start_mask: torch.Tensor,
    eoa_id: int,
    tokenizer: Any | None,
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    length = min(int(logits.shape[1]), int(labels.shape[1]), int(response_start_mask.shape[1]))
    logits = logits[:, :length]
    labels = labels[:, :length]
    response_start_mask = response_start_mask[:, :length].bool()
    valid = labels != IGNORE_LABEL_ID
    if not bool(valid.any()):
        return {
            "positions": 0,
            "accuracy": 0.0,
            "continuation_positions": 0,
            "continuation_accuracy": 0.0,
            "eos_targets": 0,
            "eos_top1_accuracy": 0.0,
            "eos_probability": 0.0,
            "common_top1": [],
            "common_targets": [],
        }
    probabilities = logits.softmax(dim=-1)
    top1 = logits.argmax(dim=-1)
    valid_top1 = top1[valid]
    valid_labels = labels[valid]
    total = int(valid_labels.numel())
    correct = int(valid_top1.eq(valid_labels).sum().detach().cpu().item())
    continuation = valid & ~response_start_mask
    continuation_total = int(continuation.sum().detach().cpu().item())
    continuation_correct = int(top1[continuation].eq(labels[continuation]).sum().detach().cpu().item())
    eos_mask = valid & labels.eq(int(eoa_id))
    eos_targets = int(eos_mask.sum().detach().cpu().item())
    eos_correct = int(top1[eos_mask].eq(int(eoa_id)).sum().detach().cpu().item()) if eos_targets else 0
    eos_probability = (
        float(probabilities[:, :, int(eoa_id)][eos_mask].mean().detach().cpu().item())
        if eos_targets
        else 0.0
    )
    top1_counter: Counter[int] = Counter(int(value) for value in valid_top1.detach().cpu().tolist())
    target_counter: Counter[int] = Counter(int(value) for value in valid_labels.detach().cpu().tolist())
    common_top1 = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)], tokenizer_info),
        }
        for token_id, count in top1_counter.most_common(10)
    ]
    common_targets = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)], tokenizer_info),
        }
        for token_id, count in target_counter.most_common(10)
    ]
    return {
        "positions": int(total),
        "accuracy": float(correct / total),
        "continuation_positions": int(continuation_total),
        "continuation_accuracy": (
            float(continuation_correct / continuation_total) if continuation_total else 0.0
        ),
        "eos_targets": int(eos_targets),
        "eos_top1_accuracy": float(eos_correct / eos_targets) if eos_targets else 0.0,
        "eos_probability": float(eos_probability),
        "common_top1": common_top1,
        "common_targets": common_targets,
    }


def response_continuation_stats(
    *,
    prefix: Any,
    model: torch.nn.Module,
    dataset: Any,
    tokenizer: Any | None,
    tokenizer_info: dict[str, Any],
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
        collate_fn=prefix.collate_prefixlm_rows,
        drop_last=False,
    )
    total = 0
    correct = 0
    continuation_total = 0
    continuation_correct = 0
    eos_targets = 0
    eos_correct = 0
    eos_probability_sum = 0.0
    top1_counter: Counter[int] = Counter()
    target_counter: Counter[int] = Counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = prefix.trim_prefixlm_batch_to_max_valid_length(batch)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_mask = batch["response_start_mask"].to(device).bool()
            logits, _ = model.forward_logits_and_decoder_hidden(
                input_ids,
                attention_mask,
                think_steps=int(think_steps),
            )
            length = min(int(logits.shape[1]), int(labels.shape[1]), int(start_mask.shape[1]))
            logits = logits[:, :length]
            labels = labels[:, :length]
            start_mask = start_mask[:, :length]
            valid = labels != IGNORE_LABEL_ID
            if not bool(valid.any()):
                continue
            probabilities = logits.softmax(dim=-1)
            top1 = logits.argmax(dim=-1)
            valid_top1 = top1[valid]
            valid_labels = labels[valid]
            total += int(valid_labels.numel())
            correct += int(valid_top1.eq(valid_labels).sum().detach().cpu().item())
            continuation = valid & ~start_mask
            continuation_total += int(continuation.sum().detach().cpu().item())
            continuation_correct += int(
                top1[continuation].eq(labels[continuation]).sum().detach().cpu().item()
            )
            eos_mask = valid & labels.eq(int(eoa_id))
            batch_eos_targets = int(eos_mask.sum().detach().cpu().item())
            eos_targets += batch_eos_targets
            if batch_eos_targets:
                eos_correct += int(top1[eos_mask].eq(int(eoa_id)).sum().detach().cpu().item())
                eos_probability_sum += float(
                    probabilities[:, :, int(eoa_id)][eos_mask].sum().detach().cpu().item()
                )
            top1_counter.update(int(value) for value in valid_top1.detach().cpu().tolist())
            target_counter.update(int(value) for value in valid_labels.detach().cpu().tolist())
    common_top1 = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)], tokenizer_info),
        }
        for token_id, count in top1_counter.most_common(10)
    ]
    common_targets = [
        {
            "token_id": int(token_id),
            "count": int(count),
            "decoded": decode_ids(tokenizer, [int(token_id)], tokenizer_info),
        }
        for token_id, count in target_counter.most_common(10)
    ]
    return {
        "rows": int(rows),
        "positions": int(total),
        "accuracy": float(correct / total) if total else 0.0,
        "continuation_positions": int(continuation_total),
        "continuation_accuracy": (
            float(continuation_correct / continuation_total) if continuation_total else 0.0
        ),
        "eos_targets": int(eos_targets),
        "eos_top1_accuracy": float(eos_correct / eos_targets) if eos_targets else 0.0,
        "eos_probability": float(eos_probability_sum / eos_targets) if eos_targets else 0.0,
        "common_top1": common_top1,
        "common_targets": common_targets,
    }


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
            attention_mask = torch.ones_like(input_ids, device=device)
            logits, _ = model.forward_logits_and_decoder_hidden(
                input_ids,
                attention_mask,
                think_steps=int(think_steps),
            )
            next_id = int(logits[0, len(current) - 1].argmax(dim=-1).detach().cpu().item())
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
    tokenizer_info: dict[str, Any],
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
                    "instruction_head": decode_ids(
                        tokenizer,
                        [int(token_id) for token_id in inst[:32].tolist()],
                        tokenizer_info,
                    ),
                    "gold_ids": gold,
                    "generated_ids": generated,
                    "gold": decode_ids(tokenizer, gold, tokenizer_info),
                    "generated": decode_ids(tokenizer, generated, tokenizer_info),
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


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    depth_probe = load_depth_probe_module()
    checkpoint_path = Path(args.checkpoint)
    sampled_data = str(args.sampled_data or load_checkpoint_sampled_data(checkpoint_path))
    if not sampled_data:
        raise ValueError("--sampled-data is required when checkpoint args do not contain sampled_data")
    device = torch.device(str(args.device))
    trainer, prefix, ckpt_args, loaded = depth_probe.load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        sampled_data=sampled_data,
        out_dir=str(Path(args.out).parent if str(args.out) else "local_eval/blt_generation_gate"),
        device=device,
        amp_dtype=str(args.amp_dtype),
    )
    model = loaded["model"]
    dataset = prefix.DataIOSampledPrefixLMDataset(
        sampled_data,
        seq_len=int(args.seq_len or ckpt_args.seq_len),
        epoch=int(args.epoch),
        target_only=True,
        max_rows=None,
        drop_overlength=True,
    )
    metadata = prefix.load_prefixlm_metadata(Path(sampled_data))
    tokenizer_info = dict(metadata.tokenizer_info or {})
    tokenizer = None
    eoa_id = resolve_eoa_id(tokenizer_info, tokenizer)
    think_steps = int(args.think_steps) if int(args.think_steps) > 0 else int(ckpt_args.train_think_steps)
    first_stats = first_response_stats(
        prefix=prefix,
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        eoa_id=int(eoa_id),
        device=device,
        think_steps=int(think_steps),
        max_rows=int(args.max_first_token_rows),
        batch_size=int(args.first_token_batch_size),
    )
    continuation_stats = response_continuation_stats(
        prefix=prefix,
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        eoa_id=int(eoa_id),
        device=device,
        think_steps=int(think_steps),
        max_rows=int(args.max_continuation_rows),
        batch_size=int(args.continuation_batch_size),
    )
    gen_stats = generation_stats(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        eoa_id=int(eoa_id),
        device=device,
        think_steps=int(think_steps),
        seq_len=int(args.seq_len or ckpt_args.seq_len),
        max_rows=int(args.max_generation_rows),
        max_new_tokens=int(args.max_new_tokens),
    )
    return {
        "gate_type": "blt_generation_gate",
        "checkpoint": str(checkpoint_path),
        "sampled_data": str(sampled_data),
        "epoch": int(args.epoch),
        "seq_len": int(args.seq_len or ckpt_args.seq_len),
        "think_steps": int(think_steps),
        "eoa": {"token_id": int(eoa_id)},
        "tokenizer_info": tokenizer_info,
        "first_response": first_stats,
        "response_continuation": continuation_stats,
        "generation": gen_stats,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampled-data", default="")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--think-steps", type=int, default=0)
    parser.add_argument("--max-first-token-rows", type=int, default=256)
    parser.add_argument("--first-token-batch-size", type=int, default=8)
    parser.add_argument("--max-continuation-rows", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=8)
    parser.add_argument("--max-generation-rows", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_gate(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.out):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
