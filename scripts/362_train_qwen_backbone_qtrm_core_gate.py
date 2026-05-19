#!/usr/bin/env python3
"""Train/evaluate the Qwen-backbone QTRM recurrent core gate."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


@dataclass(frozen=True)
class SyntheticCase:
    prompt: str
    label: str
    family: str


def parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_float_map(value: str) -> dict[str, float]:
    text = str(value).strip()
    if text == "":
        return {}
    result: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"expected name=value in float map item: {part!r}")
        name, raw_value = part.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty name in float map item: {part!r}")
        result[name] = float(raw_value.strip())
    return result


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_ouro_model(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device):
    if str(args.core_impl) != "ouro_weight_wrapped":
        return None
    layer_indices = parse_int_list(str(args.ouro_core_layer_indices))
    if bool(args.ouro_partial_safetensors):
        from qtrm_mm.ouro_partial import build_partial_ouro_model_from_safetensors

        return build_partial_ouro_model_from_safetensors(
            str(args.ouro_model_id),
            layer_indices=layer_indices or (18,),
            dtype=dtype,
            device=device,
        )
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


def build_synthetic_cases(
    *,
    count: int,
    seed: int,
    case_mode: str = "standard",
) -> list[SyntheticCase]:
    rng = random.Random(int(seed))
    cases: list[SyntheticCase] = []
    mode = str(case_mode)
    if mode == "standard":
        families = ("checksum", "chain", "select")
    elif mode == "hard_v1":
        families = ("checksum4", "chain5", "select_pair")
    elif mode == "hard_repair_v1":
        families = ("select_pair", "checksum4", "select_pair", "chain5")
    elif mode == "mixed_v1":
        families = ("checksum", "chain", "select", "checksum4", "chain5", "select_pair")
    else:
        raise ValueError(f"unsupported case_mode: {case_mode}")
    for idx in range(int(count)):
        family = families[idx % len(families)]
        if family == "checksum":
            a, b, c = (rng.randrange(10) for _ in range(3))
            answer = (a + 2 * b + 3 * c) % 10
            prompt = (
                "Compute the checksum mod 10. "
                f"Rule: (a + 2*b + 3*c) mod 10. a={a}, b={b}, c={c}. "
                "Answer with one digit. Answer: "
            )
        elif family == "chain":
            start = rng.randrange(10)
            add = rng.randrange(10)
            sub = rng.randrange(10)
            mul = rng.choice((1, 3, 7, 9))
            answer = ((start + add - sub) * mul) % 10
            prompt = (
                "Follow the digit chain mod 10. "
                f"Start {start}; add {add}; subtract {sub}; multiply by {mul}. "
                "Answer with one digit. Answer: "
            )
        elif family == "select":
            digits = [rng.randrange(10) for _ in range(5)]
            pos = rng.randrange(len(digits))
            answer = (digits[pos] + pos) % 10
            prompt = (
                "Read the digit list and answer mod 10. "
                f"Digits: {digits}. Take index {pos}, add the index, mod 10. "
                "Answer with one digit. Answer: "
            )
        elif family == "checksum4":
            a, b, c, d = (rng.randrange(10) for _ in range(4))
            answer = (a + 2 * b + 3 * c + 4 * d) % 10
            prompt = (
                "Compute the extended checksum mod 10. "
                f"Rule: (a + 2*b + 3*c + 4*d) mod 10. "
                f"a={a}, b={b}, c={c}, d={d}. "
                "Answer with one digit. Answer: "
            )
        elif family == "chain5":
            start = rng.randrange(10)
            add_a = rng.randrange(10)
            mul = rng.choice((1, 3, 7, 9))
            sub = rng.randrange(10)
            add_b = rng.randrange(10)
            answer = (((start + add_a) * mul - sub) + add_b) % 10
            prompt = (
                "Follow the five-step digit chain mod 10. "
                f"Start {start}; add {add_a}; multiply by {mul}; "
                f"subtract {sub}; add {add_b}. "
                "Answer with one digit. Answer: "
            )
        elif family == "select_pair":
            digits = [rng.randrange(10) for _ in range(7)]
            first = rng.randrange(len(digits))
            second = rng.randrange(len(digits))
            answer = (digits[first] + digits[second] + first + second) % 10
            prompt = (
                "Read the digit list and answer mod 10. "
                f"Digits: {digits}. Take indices {first} and {second}; "
                "add both selected digits and both indices, mod 10. "
                "Answer with one digit. Answer: "
            )
        else:  # pragma: no cover - guarded by case_mode family lists
            raise AssertionError(f"unknown family: {family}")
        cases.append(SyntheticCase(prompt=prompt, label=str(answer), family=family))
    return cases


def language_probe_prompts() -> list[str]:
    return [
        "User: Explain why evidence should be checked.\nAssistant: ",
        "User: 양자 컴퓨팅이란 무엇인가요?\nAssistant: ",
        "User: Write one clear sentence about careful reasoning.\nAssistant: ",
        "User: What should a model do when it is uncertain?\nAssistant: ",
        "User: Explain quantum entanglement in one simple sentence.\nAssistant: ",
        "User: Translate to Korean: Careful reasoning reduces mistakes.\nAssistant: ",
        "User: 좋은 연구 노트를 쓰는 방법을 짧게 말해 주세요.\nAssistant: ",
        "User: 사실과 의견의 차이를 설명해 주세요.\nAssistant: ",
    ]


def _label_token_ids(tokenizer) -> dict[str, int]:
    result: dict[str, int] = {}
    for digit in "0123456789":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"digit label is not a single token: {digit} -> {ids}")
        result[digit] = int(ids[0])
    return result


def _digit_choice_predictions(logits: torch.Tensor, label_ids: dict[str, int]) -> torch.Tensor:
    digits = list("0123456789")
    token_ids = torch.tensor([label_ids[digit] for digit in digits], device=logits.device)
    choice_index = logits.index_select(dim=-1, index=token_ids).argmax(dim=-1)
    return choice_index


def _batch(items: list[SyntheticCase], batch_size: int) -> Iterable[list[SyntheticCase]]:
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def _encode_prompts(tokenizer, prompts: list[str], *, max_seq_len: int, device: torch.device):
    encoded = tokenizer(
        prompts,
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


def _last_token_logits(model, input_ids, attention_mask, *, force_core_off: bool = False):
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        force_core_off=force_core_off,
    )
    return outputs.logits[:, -1, :]


@torch.no_grad()
def evaluate_cases(model, tokenizer, cases: list[SyntheticCase], args, label_ids: dict[str, int]):
    device = next(model.parameters()).device
    total = 0
    core_correct = 0
    base_correct = 0
    core_choice_correct = 0
    base_choice_correct = 0
    core_outer_iterations = []
    core_converged = []
    by_family: dict[str, dict[str, int]] = {}
    base_logits_finite = True
    core_logits_finite = True
    choice_targets = {digit: idx for idx, digit in enumerate("0123456789")}
    for chunk in _batch(cases, int(args.batch_size)):
        input_ids, attention_mask = _encode_prompts(
            tokenizer,
            [case.prompt for case in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        targets = torch.tensor([label_ids[case.label] for case in chunk], device=device)
        choice_target = torch.tensor([choice_targets[case.label] for case in chunk], device=device)
        base_logits = _last_token_logits(model, input_ids, attention_mask, force_core_off=True)
        core_outputs = model(input_ids, attention_mask=attention_mask)
        core_logits = core_outputs.logits[:, -1, :]
        base_logits_finite = bool(base_logits_finite and torch.isfinite(base_logits).all().item())
        core_logits_finite = bool(core_logits_finite and torch.isfinite(core_logits).all().item())
        if hasattr(core_outputs, "qtrm_core_outer_iterations"):
            outer = getattr(core_outputs, "qtrm_core_outer_iterations")
            if outer is not None:
                core_outer_iterations.extend(float(value) for value in outer.detach().cpu().view(-1))
        if hasattr(core_outputs, "qtrm_core_converged"):
            converged = getattr(core_outputs, "qtrm_core_converged")
            if converged is not None:
                core_converged.extend(bool(value) for value in converged.detach().cpu().view(-1))
        base_pred = base_logits.argmax(dim=-1)
        core_pred = core_logits.argmax(dim=-1)
        base_choice_pred = _digit_choice_predictions(base_logits, label_ids)
        core_choice_pred = _digit_choice_predictions(core_logits, label_ids)
        for case, base_item, core_item, base_choice, core_choice, target, choice_item in zip(
            chunk,
            base_pred,
            core_pred,
            base_choice_pred,
            core_choice_pred,
            targets,
            choice_target,
        ):
            fam = by_family.setdefault(
                case.family,
                {"total": 0, "base": 0, "core": 0, "base_choice": 0, "core_choice": 0},
            )
            fam["total"] += 1
            total += 1
            if int(base_item) == int(target):
                fam["base"] += 1
                base_correct += 1
            if int(core_item) == int(target):
                fam["core"] += 1
                core_correct += 1
            if int(base_choice) == int(choice_item):
                fam["base_choice"] += 1
                base_choice_correct += 1
            if int(core_choice) == int(choice_item):
                fam["core_choice"] += 1
                core_choice_correct += 1
    return {
        "total": total,
        "base_accuracy": base_correct / max(1, total),
        "core_accuracy": core_correct / max(1, total),
        "gain": (core_correct - base_correct) / max(1, total),
        "base_choice_accuracy": base_choice_correct / max(1, total),
        "core_choice_accuracy": core_choice_correct / max(1, total),
        "choice_gain": (core_choice_correct - base_choice_correct) / max(1, total),
        "mean_core_outer_iterations": (
            sum(core_outer_iterations) / len(core_outer_iterations)
            if core_outer_iterations
            else None
        ),
        "core_converged_fraction": (
            sum(1 for value in core_converged if value) / len(core_converged)
            if core_converged
            else None
        ),
        "base_logits_finite": bool(base_logits_finite),
        "core_logits_finite": bool(core_logits_finite),
        "by_family": {
            family: {
                "base_accuracy": row["base"] / max(1, row["total"]),
                "core_accuracy": row["core"] / max(1, row["total"]),
                "gain": (row["core"] - row["base"]) / max(1, row["total"]),
                "base_choice_accuracy": row["base_choice"] / max(1, row["total"]),
                "core_choice_accuracy": row["core_choice"] / max(1, row["total"]),
                "choice_gain": (row["core_choice"] - row["base_choice"]) / max(1, row["total"]),
                "total": row["total"],
            }
            for family, row in sorted(by_family.items())
        },
    }


def family_gain_summary(evaluation: dict[str, object], *, metric: str = "full_vocab") -> dict[str, object]:
    rows = evaluation.get("by_family", {})
    gains = {}
    core_accuracies = {}
    if metric == "full_vocab":
        base_key = "base_accuracy"
        core_key = "core_accuracy"
    elif metric == "label_choice":
        base_key = "base_choice_accuracy"
        core_key = "core_choice_accuracy"
    else:
        raise ValueError(f"unsupported family summary metric: {metric}")
    if isinstance(rows, dict):
        for family, row in rows.items():
            if not isinstance(row, dict):
                continue
            base = float(row.get(base_key, 0.0))
            core = float(row.get(core_key, 0.0))
            gains[str(family)] = core - base
            core_accuracies[str(family)] = core
    min_gain = min(gains.values()) if gains else 0.0
    min_core_accuracy = min(core_accuracies.values()) if core_accuracies else 0.0
    return {
        "gains": gains,
        "core_accuracies": core_accuracies,
        "min_gain": float(min_gain),
        "min_core_accuracy": float(min_core_accuracy),
        "metric": str(metric),
    }


def evaluation_acceptance_summary(evaluation: dict[str, object], args) -> dict[str, object]:
    metric = str(args.acceptance_metric)
    family_summary = family_gain_summary(evaluation, metric=metric)
    gain = (
        float(evaluation["gain"])
        if metric == "full_vocab"
        else float(evaluation["choice_gain"])
    )
    accepted_reasoning = gain >= float(args.min_reasoning_gain)
    accepted_family_gain = float(family_summary["min_gain"]) >= float(args.min_family_gain)
    accepted_family_accuracy = float(family_summary["min_core_accuracy"]) >= float(
        args.min_family_core_accuracy
    )
    accepted_finite_logits = bool(
        evaluation.get("base_logits_finite", True)
        and evaluation.get("core_logits_finite", True)
    )
    # Prefer threshold pass, then larger family floor, then larger aggregate gain.
    score = (
        (1.0 if accepted_reasoning else 0.0)
        + (1.0 if accepted_family_gain else 0.0)
        + (1.0 if accepted_family_accuracy else 0.0)
        + float(family_summary["min_core_accuracy"])
        + float(family_summary["min_gain"])
        + gain
    )
    return {
        "metric": metric,
        "gain": gain,
        "family_summary": family_summary,
        "accepted_reasoning_gain": bool(accepted_reasoning),
        "accepted_family_gain": bool(accepted_family_gain),
        "accepted_family_core_accuracy": bool(accepted_family_accuracy),
        "accepted_finite_logits": bool(accepted_finite_logits),
        "score": float(score),
    }


@torch.no_grad()
def evaluate_language_non_regression(model, tokenizer, args):
    device = next(model.parameters()).device
    input_ids, attention_mask = _encode_prompts(
        tokenizer,
        language_probe_prompts(),
        max_seq_len=int(args.max_seq_len),
        device=device,
    )
    base = _last_token_logits(model, input_ids, attention_mask, force_core_off=True)
    core = _last_token_logits(model, input_ids, attention_mask)
    finite = bool(torch.isfinite(base).all().item() and torch.isfinite(core).all().item())
    if finite:
        agreement = (base.argmax(dim=-1) == core.argmax(dim=-1)).float().mean().item()
        mean_abs_delta = (base.float() - core.float()).abs().mean().item()
    else:
        agreement = 0.0
        mean_abs_delta = float("inf")
    return {
        "top1_agreement": float(agreement),
        "mean_abs_delta": float(mean_abs_delta),
        "finite_logits": bool(finite),
        "num_prompts": len(language_probe_prompts()),
    }


def train_core(
    model,
    tokenizer,
    train_cases: list[SyntheticCase],
    args,
    label_ids: dict[str, int],
    *,
    eval_cases: list[SyntheticCase] | None = None,
):
    device = next(model.parameters()).device
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    qwen_trainable = [(name, parameter) for name, parameter in named_trainable if name.startswith("qwen.")]
    core_trainable = [
        (name, parameter) for name, parameter in named_trainable if not name.startswith("qwen.")
    ]
    param_groups = []
    if core_trainable:
        param_groups.append(
            {
                "params": [parameter for _, parameter in core_trainable],
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            }
        )
    if qwen_trainable:
        param_groups.append(
            {
                "params": [parameter for _, parameter in qwen_trainable],
                "lr": float(args.qwen_lr),
                "weight_decay": float(args.qwen_weight_decay),
            }
        )
    if not param_groups:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups)
    rng = random.Random(int(args.seed) + 17)
    family_loss_weights = parse_float_map(str(args.family_loss_weights))
    losses = []
    best: dict[str, object] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    model.train()
    if any(parameter.requires_grad for parameter in model.qwen.parameters()):
        model.qwen.train()
    else:
        model.qwen.eval()
    if hasattr(model, "ouro_model"):
        model.ouro_model.eval()
    for step in range(1, int(args.steps) + 1):
        chunk = rng.sample(train_cases, k=min(int(args.batch_size), len(train_cases)))
        input_ids, attention_mask = _encode_prompts(
            tokenizer,
            [case.prompt for case in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        targets = torch.tensor([label_ids[case.label] for case in chunk], device=device)
        core_logits = _last_token_logits(model, input_ids, attention_mask)
        per_item_ce = F.cross_entropy(core_logits.float(), targets, reduction="none")
        if family_loss_weights:
            weights = torch.tensor(
                [family_loss_weights.get(case.family, 1.0) for case in chunk],
                device=device,
                dtype=per_item_ce.dtype,
            )
            ce = (per_item_ce * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            ce = per_item_ce.mean()
        loss = ce
        if float(args.kl_weight) > 0.0:
            with torch.no_grad():
                base_logits = _last_token_logits(model, input_ids, attention_mask, force_core_off=True)
            kl = F.kl_div(
                F.log_softmax(core_logits.float(), dim=-1),
                F.softmax(base_logits.float(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + float(args.kl_weight) * kl
        if float(args.language_kl_weight) > 0.0:
            language_prompts = language_probe_prompts()
            language_chunk = [
                rng.choice(language_prompts)
                for _ in range(max(1, int(args.language_kl_batch_size)))
            ]
            lang_input_ids, lang_attention_mask = _encode_prompts(
                tokenizer,
                language_chunk,
                max_seq_len=int(args.max_seq_len),
                device=device,
            )
            lang_core_logits = _last_token_logits(model, lang_input_ids, lang_attention_mask)
            with torch.no_grad():
                lang_base_logits = _last_token_logits(
                    model,
                    lang_input_ids,
                    lang_attention_mask,
                    force_core_off=True,
                )
            lang_kl = F.kl_div(
                F.log_softmax(lang_core_logits.float(), dim=-1),
                F.softmax(lang_base_logits.float(), dim=-1),
                reduction="batchmean",
            )
            loss = loss + float(args.language_kl_weight) * lang_kl
        if not torch.isfinite(loss.detach()):
            raise RuntimeError(
                f"non-finite training loss at step {step}: {float(loss.detach().cpu())}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % int(args.log_every) == 0 or step == 1 or step == int(args.steps):
            print(f"step={step} loss={losses[-1]:.4f}")
        if (
            eval_cases is not None
            and int(args.eval_every_steps) > 0
            and step % int(args.eval_every_steps) == 0
        ):
            model.eval()
            evaluation = evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
            summary = evaluation_acceptance_summary(evaluation, args)
            summary["step"] = int(step)
            summary["loss"] = losses[-1]
            if best is None or float(summary["score"]) > float(best["score"]):
                best = summary
                best_state = _trainable_state_dict(model)
            print(
                "eval_step="
                f"{step} gain={summary['gain']:.4f} "
                f"min_family_gain={summary['family_summary']['min_gain']:.4f} "
                f"min_family_core_accuracy={summary['family_summary']['min_core_accuracy']:.4f}"
            )
            model.train()
            if any(parameter.requires_grad for parameter in model.qwen.parameters()):
                model.qwen.train()
            else:
                model.qwen.eval()
            if hasattr(model, "ouro_model"):
                model.ouro_model.eval()
    if best_state is not None and bool(args.restore_best_checkpoint):
        incompatible = model.load_state_dict(best_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected best checkpoint keys: {incompatible.unexpected_keys[:8]}")
    return {
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "best_periodic_eval": best,
        "restored_best_checkpoint": bool(best_state is not None and bool(args.restore_best_checkpoint)),
    }


def _trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().cpu()
        for key, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_checkpoint(model, checkpoint_path: str) -> dict[str, object]:
    if not str(checkpoint_path):
        return {"path": "", "loaded": False}
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    trainable_keys = {key for key, parameter in model.named_parameters() if parameter.requires_grad}
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected init checkpoint keys: {unexpected[:8]}")
    missing = list(incompatible.missing_keys)
    trainable_missing = sorted(key for key in missing if key in trainable_keys)
    trainable_loaded = sorted(key for key in state.keys() if key in trainable_keys)
    return {
        "path": str(checkpoint_path),
        "loaded": True,
        "missing_key_count": len(missing),
        "trainable_missing_key_count": len(trainable_missing),
        "trainable_loaded_key_count": len(trainable_loaded),
        "trainable_missing_keys": trainable_missing[:16],
        "unexpected_key_count": len(unexpected),
        "checkpoint_report": checkpoint.get("report", {}),
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
    label_ids = _label_token_ids(tokenizer)
    ouro_model = _load_ouro_model(args, dtype=dtype, device=device)
    partial_qwen_layer_indices = parse_int_list(str(args.unfreeze_qwen_layer_indices))
    partial_qwen_requested = bool(
        partial_qwen_layer_indices
        or args.unfreeze_qwen_embeddings
        or args.unfreeze_qwen_lm_head
        or args.unfreeze_qwen_final_norm
    )
    model = QwenBackboneQTRM.from_pretrained(
        str(args.model_id),
        dtype=dtype,
        device=device,
        max_seq_len=int(args.max_seq_len),
        freeze_qwen=not bool(args.train_qwen),
        core_gate_init=float(args.core_gate_init),
        residual_scale=float(args.residual_scale),
        core_impl=str(args.core_impl),
        mandatory_core=bool(args.mandatory_core),
        qwen_core_layer_indices=parse_int_list(str(args.qwen_core_layer_indices)),
        ouro_model=ouro_model,
        ouro_core_layer_indices=parse_int_list(str(args.ouro_core_layer_indices)),
        core_adapter_dim=int(args.core_adapter_dim),
        core_delta_adapter_mode=str(args.core_delta_adapter_mode),
        core_insertion_mode=str(args.core_insertion_mode),
        core_insert_after_layer=int(args.core_insert_after_layer),
        core_residual_gate_mode=str(args.core_residual_gate_mode),
        core_residual_gate_dim=int(args.core_residual_gate_dim),
        core_residual_gate_init=float(args.core_residual_gate_init),
        n_core_layers=int(args.n_core_layers),
        h_cycles=int(args.h_cycles),
        l_cycles=int(args.l_cycles),
        outer_steps=int(args.outer_steps),
        delta_backend=str(args.delta_backend),
        strict_backends=bool(args.strict_backends),
        core_convergence_halt_enabled=bool(args.core_convergence_halt_enabled),
        core_convergence_halt_threshold=float(args.core_convergence_halt_threshold),
        core_convergence_halt_min_outer=int(args.core_convergence_halt_min_outer),
        core_step_conditioning_enabled=bool(args.core_step_conditioning_enabled),
        core_step_conditioning_max_steps=int(args.core_step_conditioning_max_steps),
        core_step_conditioning_scale=float(args.core_step_conditioning_scale),
        core_causal=True,
    ).to(device)
    qwen_trainability: dict[str, object]
    if partial_qwen_requested:
        qwen_trainability = model.set_qwen_partial_trainable(
            layer_indices=partial_qwen_layer_indices,
            train_embeddings=bool(args.unfreeze_qwen_embeddings),
            train_lm_head=bool(args.unfreeze_qwen_lm_head),
            train_final_norm=bool(args.unfreeze_qwen_final_norm),
        )
        qwen_train_mode = "partial"
    elif bool(args.train_qwen):
        qwen_trainability = {
            "mode": "all",
            "qwen_trainable_parameters": sum(
                int(parameter.numel())
                for parameter in model.qwen.parameters()
                if parameter.requires_grad
            ),
        }
        qwen_train_mode = "all"
    else:
        model.qwen.eval()
        qwen_trainability = {
            "mode": "frozen",
            "qwen_trainable_parameters": 0,
        }
        qwen_train_mode = "frozen"
    init_checkpoint_info = _load_trainable_checkpoint(model, str(args.init_checkpoint))
    model.eval()

    train_case_mode = str(args.train_case_mode or args.case_mode)
    eval_case_mode = str(args.eval_case_mode or args.case_mode)
    train_cases = build_synthetic_cases(
        count=int(args.train_cases),
        seed=int(args.seed),
        case_mode=train_case_mode,
    )
    eval_cases = build_synthetic_cases(
        count=int(args.eval_cases),
        seed=int(args.seed) + 10000,
        case_mode=eval_case_mode,
    )
    before = evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
    before_language = evaluate_language_non_regression(model, tokenizer, args)
    train_report = train_core(
        model,
        tokenizer,
        train_cases,
        args,
        label_ids,
        eval_cases=eval_cases if int(args.eval_every_steps) > 0 else None,
    )
    model.eval()
    after = evaluate_cases(model, tokenizer, eval_cases, args, label_ids)
    after_language = evaluate_language_non_regression(model, tokenizer, args)
    family_summary = family_gain_summary(after, metric="full_vocab")
    choice_family_summary = family_gain_summary(after, metric="label_choice")
    acceptance_metric = str(args.acceptance_metric)
    acceptance_summary = evaluation_acceptance_summary(after, args)
    accepted_family_summary = acceptance_summary["family_summary"]
    accepted_reasoning = bool(acceptance_summary["accepted_reasoning_gain"])
    accepted_finite_logits = bool(acceptance_summary["accepted_finite_logits"])
    accepted_language = bool(after_language["finite_logits"]) and float(
        after_language["top1_agreement"]
    ) >= float(args.min_language_top1_agreement)
    accepted_family_gain = bool(acceptance_summary["accepted_family_gain"])
    accepted_family_accuracy = bool(acceptance_summary["accepted_family_core_accuracy"])
    core_layer_indices = list(getattr(model.core, "layer_indices", []))
    qwen_core_layers = core_layer_indices if str(args.core_impl) in {
        "qwen_layer_wrapped",
        "qwen_shared_layer_wrapped",
        "ouro_shared_qwen_layer",
    } else []
    ouro_core_layers = core_layer_indices if str(args.core_impl) == "ouro_weight_wrapped" else []
    report = {
        "status": "complete",
        "accepted": bool(accepted_reasoning and accepted_language),
        "accepted_reasoning_gain": bool(accepted_reasoning),
        "accepted_language_non_regression": bool(accepted_language),
        "accepted_family_gain": bool(accepted_family_gain),
        "accepted_family_core_accuracy": bool(accepted_family_accuracy),
        "accepted_finite_logits": bool(accepted_finite_logits),
        "model_id": str(args.model_id),
        "qtrm_native_integrated": True,
        "standalone_graph": True,
        "runtime_donor": False,
        "canonical_path": (
            "chat_template/prompt_tokens -> Qwen3.5 tokenizer/embedding/backbone "
            "-> mandatory QTRM recurrent core -> Qwen3.5 LM head -> AR text"
        ),
        "core_impl": str(args.core_impl),
        "mandatory_core": bool(args.mandatory_core),
        "qwen_trainable": bool(any(parameter.requires_grad for parameter in model.qwen.parameters())),
        "qwen_train_mode": qwen_train_mode,
        "qwen_trainability": qwen_trainability,
        "core_layer_indices": core_layer_indices,
        "qwen_core_layer_indices": qwen_core_layers,
        "ouro_model_id": str(args.ouro_model_id) if ouro_model is not None else "",
        "ouro_core_layer_indices": ouro_core_layers,
        "core_adapter_dim": int(args.core_adapter_dim),
        "core_delta_adapter_mode": str(args.core_delta_adapter_mode),
        "core_insertion_mode": str(args.core_insertion_mode),
        "core_insert_after_layer": int(model.core_insert_after_layer),
        "core_residual_gate_mode": str(args.core_residual_gate_mode),
        "core_residual_gate_dim": int(args.core_residual_gate_dim),
        "core_residual_gate_init": float(args.core_residual_gate_init),
        "residual_scale": float(args.residual_scale),
        "h_cycles": int(args.h_cycles),
        "l_cycles": int(args.l_cycles),
        "outer_steps": int(args.outer_steps),
        "core_convergence_halt_enabled": bool(args.core_convergence_halt_enabled),
        "core_convergence_halt_threshold": float(args.core_convergence_halt_threshold),
        "core_convergence_halt_min_outer": int(args.core_convergence_halt_min_outer),
        "core_step_conditioning_enabled": bool(args.core_step_conditioning_enabled),
        "core_step_conditioning_max_steps": int(args.core_step_conditioning_max_steps),
        "core_step_conditioning_scale": float(args.core_step_conditioning_scale),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "train_cases": int(args.train_cases),
        "eval_cases": int(args.eval_cases),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "qwen_lr": float(args.qwen_lr),
        "weight_decay": float(args.weight_decay),
        "qwen_weight_decay": float(args.qwen_weight_decay),
        "grad_clip": float(args.grad_clip),
        "kl_weight": float(args.kl_weight),
        "language_kl_weight": float(args.language_kl_weight),
        "language_kl_batch_size": int(args.language_kl_batch_size),
        "case_mode": str(args.case_mode),
        "train_case_mode": train_case_mode,
        "eval_case_mode": eval_case_mode,
        "acceptance_metric": acceptance_metric,
        "init_checkpoint": init_checkpoint_info,
        "model_report": model.report().__dict__,
        "core_gate_value": float(model.normal_core_gate_value()),
        "before_eval": before,
        "after_eval": after,
        "after_family_summary": family_summary,
        "after_choice_family_summary": choice_family_summary,
        "accepted_family_summary": accepted_family_summary,
        "acceptance_summary": acceptance_summary,
        "before_language": before_language,
        "after_language": after_language,
        "train": train_report,
        "family_loss_weights": parse_float_map(str(args.family_loss_weights)),
        "thresholds": {
            "min_reasoning_gain": float(args.min_reasoning_gain),
            "min_language_top1_agreement": float(args.min_language_top1_agreement),
            "min_family_gain": float(args.min_family_gain),
            "min_family_core_accuracy": float(args.min_family_core_accuracy),
        },
    }
    report["accepted"] = bool(
        accepted_reasoning
        and accepted_language
        and accepted_family_gain
        and accepted_family_accuracy
        and accepted_finite_logits
    )
    if str(args.save_checkpoint):
        torch.save(
            {"model": _trainable_state_dict(model), "report": report},
            str(args.save_checkpoint),
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--out-dir", default="local_eval/qwen_backbone_qtrm_core_gate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument(
        "--core-impl",
        choices=[
            "qwen_layer_wrapped",
            "qwen_shared_layer_wrapped",
            "ouro_shared_qwen_layer",
            "ouro_weight_wrapped",
        ],
        default="qwen_layer_wrapped",
    )
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--ouro-model-id", default="ByteDance/Ouro-2.6B-Thinking")
    parser.add_argument("--ouro-core-layer-indices", default="")
    parser.add_argument("--ouro-partial-safetensors", action="store_true")
    parser.add_argument("--train-qwen", action="store_true")
    parser.add_argument("--unfreeze-qwen-layer-indices", default="")
    parser.add_argument("--unfreeze-qwen-embeddings", action="store_true")
    parser.add_argument("--unfreeze-qwen-lm-head", action="store_true")
    parser.add_argument("--unfreeze-qwen-final-norm", action="store_true")
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-adapter-dim", type=int, default=64)
    parser.add_argument(
        "--core-delta-adapter-mode",
        choices=["add", "adapter_only"],
        default="add",
    )
    parser.add_argument(
        "--core-insertion-mode",
        choices=["final_residual", "mid_layer_suffix"],
        default="final_residual",
    )
    parser.add_argument("--core-insert-after-layer", type=int, default=-1)
    parser.add_argument(
        "--core-residual-gate-mode",
        choices=["constant", "token_mlp"],
        default="constant",
    )
    parser.add_argument("--core-residual-gate-dim", type=int, default=128)
    parser.add_argument("--core-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--n-core-layers", type=int, default=1)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--core-convergence-halt-enabled", action="store_true")
    parser.add_argument("--core-convergence-halt-threshold", type=float, default=1.0e-3)
    parser.add_argument("--core-convergence-halt-min-outer", type=int, default=1)
    parser.add_argument("--core-step-conditioning-enabled", action="store_true")
    parser.add_argument("--core-step-conditioning-max-steps", type=int, default=64)
    parser.add_argument("--core-step-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--delta-backend", default="fla_gated_delta")
    parser.add_argument("--strict-backends", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-cases", type=int, default=256)
    parser.add_argument("--eval-cases", type=int, default=96)
    parser.add_argument(
        "--case-mode",
        choices=["standard", "hard_v1", "hard_repair_v1", "mixed_v1"],
        default="standard",
    )
    parser.add_argument(
        "--train-case-mode",
        choices=["", "standard", "hard_v1", "hard_repair_v1", "mixed_v1"],
        default="",
    )
    parser.add_argument(
        "--eval-case-mode",
        choices=["", "standard", "hard_v1", "hard_repair_v1", "mixed_v1"],
        default="",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--qwen-lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--qwen-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--language-kl-weight", type=float, default=0.0)
    parser.add_argument("--language-kl-batch-size", type=int, default=2)
    parser.add_argument("--family-loss-weights", default="")
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--restore-best-checkpoint", action="store_true")
    parser.add_argument(
        "--acceptance-metric",
        choices=["full_vocab", "label_choice"],
        default="full_vocab",
    )
    parser.add_argument("--min-reasoning-gain", type=float, default=0.05)
    parser.add_argument("--min-language-top1-agreement", type=float, default=0.50)
    parser.add_argument("--min-family-gain", type=float, default=-1.0)
    parser.add_argument("--min-family-core-accuracy", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not str(args.save_checkpoint):
        args.save_checkpoint = str(out_dir / "last_core.pt")
    report = run(args)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
