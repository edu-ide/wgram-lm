#!/usr/bin/env python3
"""Evaluate broad language non-regression for a Qwen-backbone QTRM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


def parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def default_language_prompts() -> list[str]:
    return [
        "User: Explain quantum entanglement in one simple sentence.\nAssistant: ",
        "User: What is a good way to verify a claim?\nAssistant: ",
        "User: Write a short paragraph about the ocean.\nAssistant: ",
        "User: Translate to Korean: Careful reasoning reduces mistakes.\nAssistant: ",
        "User: Give one practical tip for studying mathematics.\nAssistant: ",
        "User: Summarize why source dates matter.\nAssistant: ",
        "User: 양자 컴퓨팅이 무엇인지 한 문장으로 설명해 주세요.\nAssistant: ",
        "User: 증거를 확인해야 하는 이유를 간단히 말해 주세요.\nAssistant: ",
        "User: 한국어로 짧은 인사말을 작성해 주세요.\nAssistant: ",
        "User: 모를 때는 어떻게 답해야 하나요?\nAssistant: ",
        "User: 좋은 연구 노트를 쓰는 방법을 짧게 말해 주세요.\nAssistant: ",
        "User: 사실과 의견의 차이를 설명해 주세요.\nAssistant: ",
    ]


def _load_ouro_model(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device):
    if str(args.core_impl) != "ouro_weight_wrapped":
        return None
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to load Ouro") from exc
    model = AutoModelForCausalLM.from_pretrained(
        str(args.ouro_model_id),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def _encode(tokenizer, prompts: Sequence[str], *, max_seq_len: int, device: torch.device):
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _last_nonpad_logits(logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return logits[:, -1, :]
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, lengths.to(device=logits.device), :]


def _load_trainable_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:8]}")
    return {
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(unexpected),
        "checkpoint_report": checkpoint.get("report", {}),
    }


@torch.no_grad()
def evaluate_topk(model, tokenizer, prompts: list[str], args):
    device = next(model.parameters()).device
    input_ids, attention_mask = _encode(
        tokenizer,
        prompts,
        max_seq_len=int(args.max_seq_len),
        device=device,
    )
    base_logits = model(
        input_ids,
        attention_mask=attention_mask,
        force_core_off=True,
    ).logits
    core_logits = model(input_ids, attention_mask=attention_mask).logits
    base_logits = _last_nonpad_logits(base_logits, attention_mask)
    core_logits = _last_nonpad_logits(core_logits, attention_mask)
    finite_logits = bool(torch.isfinite(base_logits).all().item() and torch.isfinite(core_logits).all().item())
    if finite_logits:
        base_top1 = base_logits.argmax(dim=-1)
        core_top1 = core_logits.argmax(dim=-1)
        top1_agreement = (base_top1 == core_top1).float().mean().item()
        base_in_core_top5 = (
            core_logits.topk(k=5, dim=-1).indices == base_top1[:, None]
        ).any(dim=-1).float().mean().item()
        mean_abs_delta = (base_logits.float() - core_logits.float()).abs().mean().item()
    else:
        base_top1 = torch.zeros((len(prompts),), dtype=torch.long, device=device)
        core_top1 = torch.zeros((len(prompts),), dtype=torch.long, device=device)
        top1_agreement = 0.0
        base_in_core_top5 = 0.0
        mean_abs_delta = float("inf")
    rows = []
    for idx, prompt in enumerate(prompts):
        rows.append(
            {
                "prompt": prompt,
                "base_top1": tokenizer.decode([int(base_top1[idx])]),
                "core_top1": tokenizer.decode([int(core_top1[idx])]),
                "top1_same": bool(base_top1[idx] == core_top1[idx]),
                "base_top1_in_core_top5": bool(
                    (core_logits[idx].topk(k=5).indices == base_top1[idx]).any()
                ),
            }
        )
    return {
        "num_prompts": len(prompts),
        "top1_agreement": float(top1_agreement),
        "base_top1_in_core_top5": float(base_in_core_top5),
        "mean_abs_delta": float(mean_abs_delta),
        "finite_logits": bool(finite_logits),
        "rows": rows,
    }


def max_repeated_token_run(token_ids: Sequence[int]) -> int:
    best = 0
    current = 0
    last = None
    for token_id in token_ids:
        if token_id == last:
            current += 1
        else:
            current = 1
            last = token_id
        best = max(best, current)
    return best


def unique_ratio(token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    return len(set(int(token_id) for token_id in token_ids)) / len(token_ids)


@torch.no_grad()
def greedy_generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    force_core_off: bool,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = encoded["input_ids"].to(device)
    new_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(int(max_new_tokens)):
        attention_mask = torch.ones_like(input_ids, device=device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            force_core_off=force_core_off,
        )
        next_logits = outputs.logits[:, -1, :]
        if not bool(torch.isfinite(next_logits).all().item()):
            return {
                "new_token_ids": new_ids,
                "text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "max_repeated_token_run": max_repeated_token_run(new_ids),
                "unique_ratio": unique_ratio(new_ids),
                "finite_logits": False,
            }
        next_id = int(next_logits.argmax(dim=-1).item())
        new_ids.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype, device=device)],
            dim=1,
        )
        if eos_token_id is not None and next_id == int(eos_token_id):
            break
    return {
        "new_token_ids": new_ids,
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "max_repeated_token_run": max_repeated_token_run(new_ids),
        "unique_ratio": unique_ratio(new_ids),
        "finite_logits": True,
    }


@torch.no_grad()
def evaluate_generation(model, tokenizer, prompts: list[str], args):
    device = next(model.parameters()).device
    rows = []
    core_runs = []
    core_unique = []
    finite_generation = True
    for prompt in prompts[: int(args.max_generation_prompts)]:
        base = greedy_generate_one(
            model,
            tokenizer,
            prompt,
            force_core_off=True,
            max_seq_len=int(args.max_seq_len),
            max_new_tokens=int(args.max_new_tokens),
            device=device,
        )
        core = greedy_generate_one(
            model,
            tokenizer,
            prompt,
            force_core_off=False,
            max_seq_len=int(args.max_seq_len),
            max_new_tokens=int(args.max_new_tokens),
            device=device,
        )
        core_runs.append(int(core["max_repeated_token_run"]))
        core_unique.append(float(core["unique_ratio"]))
        finite_generation = bool(
            finite_generation
            and bool(base.get("finite_logits", True))
            and bool(core.get("finite_logits", True))
        )
        rows.append({"prompt": prompt, "base": base, "core": core})
    return {
        "num_prompts": len(rows),
        "max_core_repeated_token_run": max(core_runs) if core_runs else 0,
        "mean_core_unique_ratio": sum(core_unique) / max(1, len(core_unique)),
        "finite_logits": bool(finite_generation),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    device = torch.device(str(args.device))
    dtype = _dtype(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ouro_model = _load_ouro_model(args, dtype=dtype, device=device)
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=True,
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        qwen_core_layer_indices=parse_int_list(str(args.qwen_core_layer_indices)),
        ouro_model=ouro_model,
        ouro_core_layer_indices=parse_int_list(str(args.ouro_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        mandatory_core=bool(args.mandatory_core),
        n_core_layers=1,
        h_cycles=1,
        l_cycles=1,
        outer_steps=1,
        delta_backend="fla_gated_delta",
        strict_backends=False,
        core_causal=True,
    ).to(device)
    model.eval()
    checkpoint_info = _load_trainable_checkpoint(model, str(args.checkpoint))
    model.eval()

    prompts = default_language_prompts()
    topk = evaluate_topk(model, tokenizer, prompts, args)
    generation = evaluate_generation(model, tokenizer, prompts, args)
    accepted_top1 = float(topk["top1_agreement"]) >= float(args.min_top1_agreement)
    accepted_top5 = float(topk["base_top1_in_core_top5"]) >= float(args.min_top5_agreement)
    accepted_finite = bool(topk["finite_logits"]) and bool(generation["finite_logits"])
    accepted_repetition = int(generation["max_core_repeated_token_run"]) <= int(
        args.max_repeated_token_run
    )
    accepted_unique = float(generation["mean_core_unique_ratio"]) >= float(
        args.min_unique_ratio
    )
    result = {
        "status": "complete",
        "accepted": bool(
            accepted_top1 and accepted_top5 and accepted_repetition and accepted_unique
            and accepted_finite
        ),
        "accepted_top1": bool(accepted_top1),
        "accepted_top5": bool(accepted_top5),
        "accepted_repetition": bool(accepted_repetition),
        "accepted_unique_ratio": bool(accepted_unique),
        "accepted_finite_logits": bool(accepted_finite),
        "model_id": str(args.model_id),
        "checkpoint": str(args.checkpoint),
        "core_impl": str(args.core_impl),
        "core_layer_indices": list(getattr(model.core, "layer_indices", [])),
        "model_report": model.report().__dict__,
        "checkpoint_info": checkpoint_info,
        "topk": topk,
        "generation": generation,
        "thresholds": {
            "min_top1_agreement": float(args.min_top1_agreement),
            "min_top5_agreement": float(args.min_top5_agreement),
            "max_repeated_token_run": int(args.max_repeated_token_run),
            "min_unique_ratio": float(args.min_unique_ratio),
        },
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="local_eval/qwen_backbone_language_gate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-generation-prompts", type=int, default=6)
    parser.add_argument(
        "--core-impl",
        choices=["qwen_layer_wrapped", "qwen_shared_layer_wrapped", "ouro_weight_wrapped"],
        default="qwen_layer_wrapped",
    )
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--ouro-model-id", default="/mnt/sdc1/models/ByteDance-Ouro-2.6B-Thinking")
    parser.add_argument("--ouro-core-layer-indices", default="24")
    parser.add_argument("--core-adapter-dim", type=int, default=64)
    parser.add_argument(
        "--core-delta-adapter-mode",
        choices=["add", "adapter_only"],
        default="add",
    )
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--min-top1-agreement", type=float, default=0.75)
    parser.add_argument("--min-top5-agreement", type=float, default=0.90)
    parser.add_argument("--max-repeated-token-run", type=int, default=8)
    parser.add_argument("--min-unique-ratio", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    (out_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
