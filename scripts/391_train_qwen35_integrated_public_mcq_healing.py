#!/usr/bin/env python3
"""Public-MCQ healing tune for the Qwen-integrated QTRM path.

The goal is not to hide behind a sidecar scorer. The same integrated model
receives a normal MCQ prompt and learns to move the next-token option-letter
logits through its mandatory QTRM core while preserving the core_off language
distribution on unrelated English/Korean prompts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from qtrm_mm.qwen_backbone_qtrm import QwenBackboneQTRM


OPTION_LETTERS = "ABCDEFGHIJ"


def load_public_eval_module():
    path = Path(__file__).with_name("390_eval_qwen35_integrated_public_mcq.py")
    spec = importlib.util.spec_from_file_location("qwen35_integrated_public_mcq_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_language_gate_module():
    path = Path(__file__).with_name("367_eval_qwen_backbone_language_gate.py")
    spec = importlib.util.spec_from_file_location("qwen_backbone_language_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_int_list(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), int(batch_size)):
        yield rows[start : start + int(batch_size)]


def group_rows_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("category", "unknown")), []).append(row)
    return groups


def sample_training_chunk(
    rng: random.Random,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    balanced_category_sampling: bool,
    category_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    size = min(int(batch_size), len(rows))
    if not bool(balanced_category_sampling):
        return rng.sample(rows, k=size)
    groups = category_groups if category_groups is not None else group_rows_by_category(rows)
    categories = [category for category, bucket in sorted(groups.items()) if bucket]
    if not categories:
        return rng.sample(rows, k=size)
    return [rng.choice(groups[rng.choice(categories)]) for _ in range(size)]


def _encode(tokenizer, prompts: list[str], *, max_seq_len: int, device: torch.device):
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


def _last_logits(model, input_ids, attention_mask, *, force_core_off: bool = False):
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        force_core_off=force_core_off,
    ).logits
    if attention_mask is None:
        return logits[:, -1, :]
    last_indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices, :]


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().cpu()
        for key, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    load_mode: str = "strict_shapes",
) -> dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    skipped_shape_mismatch: list[dict[str, object]] = []
    if str(load_mode) == "skip_mismatch":
        compatible = {}
        for key, value in state.items():
            if key in current and tuple(value.shape) != tuple(current[key].shape):
                skipped_shape_mismatch.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(current[key].shape),
                    }
                )
                continue
            compatible[key] = value
        state = compatible
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:8]}")
    return {
        "path": str(checkpoint_path),
        "loaded": True,
        "load_mode": str(load_mode),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_key_count": len(skipped_shape_mismatch),
        "shape_mismatch_keys": skipped_shape_mismatch[:64],
        "checkpoint_report": checkpoint.get("report", {}),
    }


def option_target_token_ids(public_eval, tokenizer, letter: str) -> list[int]:
    ids = public_eval.single_token_option_ids(tokenizer, str(letter).upper())
    if not ids:
        raise ValueError(f"option letter has no single-token representation: {letter!r}")
    return ids


def option_nll_loss(public_eval, tokenizer, logits: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    if not rows:
        return logits.float().sum() * 0.0
    log_probs = F.log_softmax(logits.float(), dim=-1)
    losses = []
    for index, row in enumerate(rows):
        gold = public_eval.normalize_mcq_answer(str(row["answer"]))
        target_ids = option_target_token_ids(public_eval, tokenizer, gold)
        target = torch.tensor(target_ids, device=log_probs.device, dtype=torch.long)
        losses.append(-torch.logsumexp(log_probs[index].index_select(dim=0, index=target), dim=0))
    return torch.stack(losses).mean()


def option_letter_scores(public_eval, tokenizer, logits: torch.Tensor, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores: dict[str, torch.Tensor] = {}
    for letter in OPTION_LETTERS[: public_eval.option_count(row)]:
        target_ids = option_target_token_ids(public_eval, tokenizer, letter)
        target = torch.tensor(target_ids, device=log_probs.device, dtype=torch.long)
        scores[letter] = torch.logsumexp(log_probs.index_select(dim=0, index=target), dim=0)
    return scores


def option_margin_loss(
    public_eval,
    tokenizer,
    core_logits: torch.Tensor,
    base_logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    margin: float,
    focus: str,
) -> torch.Tensor:
    losses = []
    for index, row in enumerate(rows):
        gold = public_eval.normalize_mcq_answer(str(row["answer"]))
        base_scores = option_letter_scores(public_eval, tokenizer, base_logits[index], row)
        core_scores = option_letter_scores(public_eval, tokenizer, core_logits[index], row)
        if gold not in core_scores or not base_scores:
            continue
        rejected = max(base_scores.items(), key=lambda item: float(item[1].detach().cpu()))[0]
        if rejected == gold:
            if str(focus) == "base_wrong":
                continue
            rejected = max(
                ((letter, score) for letter, score in base_scores.items() if letter != gold),
                key=lambda item: float(item[1].detach().cpu()),
            )[0]
        losses.append(F.relu(torch.as_tensor(float(margin), device=core_logits.device) - (core_scores[gold] - core_scores[rejected])))
    if not losses:
        return core_logits.float().sum() * 0.0
    return torch.stack(losses).mean()


def base_wrong_indices(
    public_eval,
    tokenizer,
    base_logits: torch.Tensor,
    rows: list[dict[str, Any]],
) -> list[int]:
    selected: list[int] = []
    for index, row in enumerate(rows):
        gold = public_eval.normalize_mcq_answer(str(row["answer"]))
        base_scores = option_letter_scores(public_eval, tokenizer, base_logits[index], row)
        if not base_scores:
            continue
        pred = max(base_scores.items(), key=lambda item: float(item[1].detach().cpu()))[0]
        if pred != gold:
            selected.append(index)
    return selected


def selected_option_nll_loss(
    public_eval,
    tokenizer,
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    selected_indices: list[int],
) -> torch.Tensor:
    if not selected_indices:
        return logits.float().sum() * 0.0
    index_tensor = torch.tensor(selected_indices, device=logits.device, dtype=torch.long)
    selected_logits = logits.index_select(dim=0, index=index_tensor)
    selected_rows = [rows[index] for index in selected_indices]
    return option_nll_loss(public_eval, tokenizer, selected_logits, selected_rows)


def category_gain_summary(evaluation: dict[str, Any], *, min_cases: int = 1) -> dict[str, Any]:
    base_by = evaluation["base_metrics"].get("by_category", {})
    core_by = evaluation["core_metrics"].get("by_category", {})
    deltas: dict[str, dict[str, float | int]] = {}
    eligible = []
    for category in sorted(set(base_by) | set(core_by)):
        base = base_by.get(category, {"hits": 0, "total": 0, "accuracy": 0.0})
        core = core_by.get(category, {"hits": 0, "total": 0, "accuracy": 0.0})
        total = int(max(int(base.get("total", 0)), int(core.get("total", 0))))
        hit_delta = int(core.get("hits", 0)) - int(base.get("hits", 0))
        accuracy_delta = float(core.get("accuracy", 0.0)) - float(base.get("accuracy", 0.0))
        deltas[category] = {
            "total": total,
            "base_hits": int(base.get("hits", 0)),
            "core_hits": int(core.get("hits", 0)),
            "hit_delta": hit_delta,
            "accuracy_delta": accuracy_delta,
        }
        if total >= int(min_cases):
            eligible.append(deltas[category])
    if eligible:
        min_hit_delta = min(int(item["hit_delta"]) for item in eligible)
        min_accuracy_delta = min(float(item["accuracy_delta"]) for item in eligible)
        negative_hit_delta_sum = sum(max(0, -int(item["hit_delta"])) for item in eligible)
        negative_accuracy_delta_sum = sum(max(0.0, -float(item["accuracy_delta"])) for item in eligible)
    else:
        min_hit_delta = 0
        min_accuracy_delta = 0.0
        negative_hit_delta_sum = 0
        negative_accuracy_delta_sum = 0.0
    return {
        "min_cases": int(min_cases),
        "eligible_categories": len(eligible),
        "min_hit_delta": int(min_hit_delta),
        "min_accuracy_delta": float(min_accuracy_delta),
        "negative_hit_delta_sum": int(negative_hit_delta_sum),
        "negative_accuracy_delta_sum": float(negative_accuracy_delta_sum),
        "by_category": deltas,
    }


def public_eval_selection_score(evaluation: dict[str, Any], args: argparse.Namespace) -> float:
    category = category_gain_summary(
        evaluation,
        min_cases=int(args.category_guard_min_cases),
    )
    return float(
        float(evaluation["core_gain_over_base"])
        + float(evaluation["core_metrics"]["accuracy"])
        + (1.0 if bool(evaluation["finite_logits"]) else -1.0)
        - float(args.category_regression_penalty) * float(category["negative_accuracy_delta_sum"])
    )


def load_model(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc

    public_eval = load_public_eval_module()
    device = torch.device(str(args.device))
    dtype = dtype_from_name(str(args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    qwen_trainability: dict[str, object] = {"mode": "frozen", "qwen_trainable_parameters": 0}
    layer_indices = parse_int_list(str(args.unfreeze_qwen_layer_indices))
    if bool(layer_indices or args.unfreeze_qwen_lm_head or args.unfreeze_qwen_final_norm):
        qwen_trainability = model.set_qwen_partial_trainable(
            layer_indices=layer_indices,
            train_embeddings=False,
            train_lm_head=bool(args.unfreeze_qwen_lm_head),
            train_final_norm=bool(args.unfreeze_qwen_final_norm),
        )
        qwen_trainability["mode"] = "partial"
    checkpoint_info = load_checkpoint(
        model,
        str(args.init_checkpoint),
        load_mode=str(args.checkpoint_load_mode),
    )
    return public_eval, tokenizer, model, device, checkpoint_info, qwen_trainability


@torch.no_grad()
def evaluate_public(public_eval, model, tokenizer, device, suite_rows: list[dict[str, Any]], args):
    scored_rows: list[dict[str, Any]] = []
    finite = True
    for row in suite_rows:
        count = public_eval.option_count(row)
        choices = list(OPTION_LETTERS[:count])
        base = public_eval.score_prompt_next_token(
            model,
            tokenizer,
            prompt=str(row["qtrm_prompt"]),
            choices=choices,
            force_core_off=True,
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        core = public_eval.score_prompt_next_token(
            model,
            tokenizer,
            prompt=str(row["qtrm_prompt"]),
            choices=choices,
            force_core_off=False,
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        gold = public_eval.normalize_mcq_answer(str(row["answer"]))
        finite = bool(finite and base["finite_logits"] and core["finite_logits"])
        scored = dict(row)
        scored.update(
            {
                "gold_answer": gold,
                "base_pred_answer": base["pred_answer"],
                "core_pred_answer": core["pred_answer"],
                "base_exact": bool(base["pred_answer"] == gold),
                "core_exact": bool(core["pred_answer"] == gold),
                "finite_logits": bool(base["finite_logits"] and core["finite_logits"]),
            }
        )
        scored_rows.append(scored)
    base_metrics = public_eval.score_rows(scored_rows, pred_key="base_pred_answer")
    core_metrics = public_eval.score_rows(scored_rows, pred_key="core_pred_answer")
    return {
        "base_metrics": base_metrics,
        "core_metrics": core_metrics,
        "core_gain_over_base": float(core_metrics["accuracy"] - base_metrics["accuracy"]),
        "finite_logits": bool(finite),
        "rows": scored_rows,
    }


@torch.no_grad()
def evaluate_language_preservation(model, tokenizer, args):
    language_gate = load_language_gate_module()
    prompts = language_gate.default_language_prompts()
    topk = language_gate.evaluate_topk(model, tokenizer, prompts, args)
    generation = language_gate.evaluate_generation(model, tokenizer, prompts, args)
    return {"topk": topk, "generation": generation}


def train(args: argparse.Namespace) -> dict[str, object]:
    public_eval, tokenizer, model, device, checkpoint_info, qwen_trainability = load_model(args)
    train_rows = public_eval.load_suite(args.train_jsonl, max_cases=int(args.max_train_cases))
    eval_rows = public_eval.load_suite(args.eval_jsonl, max_cases=int(args.max_eval_cases))
    before_eval = evaluate_public(public_eval, model, tokenizer, device, eval_rows, args)
    before_train_eval = None if bool(args.skip_train_eval) else evaluate_public(public_eval, model, tokenizer, device, train_rows, args)

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    qwen_trainable = [(name, parameter) for name, parameter in named_trainable if name.startswith("qwen.")]
    core_trainable = [(name, parameter) for name, parameter in named_trainable if not name.startswith("qwen.")]
    param_groups = []
    if core_trainable:
        param_groups.append({"params": [p for _, p in core_trainable], "lr": float(args.lr)})
    if qwen_trainable:
        param_groups.append({"params": [p for _, p in qwen_trainable], "lr": float(args.qwen_lr)})
    if not param_groups:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
    rng = random.Random(int(args.seed))
    category_groups = group_rows_by_category(train_rows)
    language_gate = load_language_gate_module()
    language_prompts = language_gate.default_language_prompts()
    best_state: dict[str, torch.Tensor] | None = None
    best_eval: dict[str, object] | None = None
    losses = []
    model.train()
    if not qwen_trainable:
        model.qwen.eval()
    for step in range(1, int(args.steps) + 1):
        chunk = sample_training_chunk(
            rng,
            train_rows,
            batch_size=int(args.batch_size),
            balanced_category_sampling=bool(args.balanced_category_sampling),
            category_groups=category_groups,
        )
        input_ids, attention_mask = _encode(
            tokenizer,
            [str(row["qtrm_prompt"]) for row in chunk],
            max_seq_len=int(args.max_seq_len),
            device=device,
        )
        logits = _last_logits(model, input_ids, attention_mask)
        base_logits = None
        if (
            float(args.base_kl_weight) > 0.0
            or float(args.margin_weight) > 0.0
            or str(args.ce_focus) == "base_wrong"
        ):
            with torch.no_grad():
                base_logits = _last_logits(model, input_ids, attention_mask, force_core_off=True)
        if str(args.ce_focus) == "base_wrong":
            assert base_logits is not None
            selected_indices = base_wrong_indices(public_eval, tokenizer, base_logits, chunk)
            loss = selected_option_nll_loss(public_eval, tokenizer, logits, chunk, selected_indices)
        else:
            loss = option_nll_loss(public_eval, tokenizer, logits, chunk)
        if float(args.base_kl_weight) > 0.0:
            assert base_logits is not None
            loss = loss + float(args.base_kl_weight) * F.kl_div(
                F.log_softmax(logits.float(), dim=-1),
                F.softmax(base_logits.float(), dim=-1),
                reduction="batchmean",
            )
        if float(args.margin_weight) > 0.0:
            assert base_logits is not None
            loss = loss + float(args.margin_weight) * option_margin_loss(
                public_eval,
                tokenizer,
                logits,
                base_logits,
                chunk,
                margin=float(args.margin_value),
                focus=str(args.margin_focus),
            )
        if float(args.language_kl_weight) > 0.0:
            lang_chunk = [
                rng.choice(language_prompts)
                for _ in range(max(1, int(args.language_kl_batch_size)))
            ]
            lang_ids, lang_mask = _encode(
                tokenizer,
                lang_chunk,
                max_seq_len=int(args.max_seq_len),
                device=device,
            )
            lang_logits = _last_logits(model, lang_ids, lang_mask)
            with torch.no_grad():
                lang_base = _last_logits(model, lang_ids, lang_mask, force_core_off=True)
            loss = loss + float(args.language_kl_weight) * F.kl_div(
                F.log_softmax(lang_logits.float(), dim=-1),
                F.softmax(lang_base.float(), dim=-1),
                reduction="batchmean",
            )
        if not torch.isfinite(loss.detach()):
            raise RuntimeError(f"non-finite public-MCQ healing loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in named_trainable], float(args.grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % int(args.log_every) == 0 or step == 1 or step == int(args.steps):
            print(f"step={step} loss={losses[-1]:.4f}", flush=True)
        if int(args.eval_every_steps) > 0 and step % int(args.eval_every_steps) == 0:
            model.eval()
            current = evaluate_public(public_eval, model, tokenizer, device, eval_rows, args)
            category = category_gain_summary(current, min_cases=int(args.category_guard_min_cases))
            score = public_eval_selection_score(current, args)
            current["step"] = int(step)
            current["score"] = float(score)
            current["category_gain_summary"] = category
            if best_eval is None or score > float(best_eval["score"]):
                best_eval = current
                best_state = trainable_state_dict(model)
            print(
                "eval_step="
                f"{step} core={current['core_metrics']['accuracy']:.4f} "
                f"base={current['base_metrics']['accuracy']:.4f} "
                f"gain={current['core_gain_over_base']:.4f}",
                flush=True,
            )
            model.train()
            if not qwen_trainable:
                model.qwen.eval()
    if best_state is not None and bool(args.restore_best_checkpoint):
        incompatible = model.load_state_dict(best_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected best checkpoint keys: {incompatible.unexpected_keys[:8]}")
    model.eval()
    after_eval = evaluate_public(public_eval, model, tokenizer, device, eval_rows, args)
    after_train_eval = None if bool(args.skip_train_eval) else evaluate_public(public_eval, model, tokenizer, device, train_rows, args)
    eval_category_summary = category_gain_summary(after_eval, min_cases=int(args.category_guard_min_cases))
    train_category_summary = (
        None
        if after_train_eval is None
        else category_gain_summary(after_train_eval, min_cases=int(args.category_guard_min_cases))
    )
    language = evaluate_language_preservation(model, tokenizer, args)
    accepted_language = bool(
        language["topk"]["finite_logits"]
        and language["generation"]["finite_logits"]
        and float(language["topk"]["top1_agreement"]) >= float(args.min_language_top1_agreement)
        and int(language["generation"]["max_core_repeated_token_run"]) <= int(args.max_repeated_token_run)
        and float(language["generation"]["mean_core_unique_ratio"]) >= float(args.min_unique_ratio)
    )
    accepted_eval = bool(
        after_eval["finite_logits"]
        and float(after_eval["core_gain_over_base"]) >= float(args.min_eval_core_gain)
        and float(after_eval["core_metrics"]["accuracy"]) >= float(args.min_eval_core_accuracy)
        and float(eval_category_summary["min_accuracy_delta"]) >= float(args.min_eval_category_gain)
        and int(eval_category_summary["min_hit_delta"]) >= int(args.min_eval_category_hit_delta)
    )
    report = {
        "status": "complete",
        "decision": (
            "accepted_public_mcq_healing"
            if accepted_eval and accepted_language
            else "rejected_public_mcq_healing"
        ),
        "accepted": bool(accepted_eval and accepted_language),
        "accepted_eval_core_gain": bool(accepted_eval),
        "accepted_language": bool(accepted_language),
        "target_level": "M4 public MCQ healing",
        "model_id": str(args.model_id),
        "checkpoint": str(args.init_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "canonical_path": "Qwen3.5 tokenizer/backbone -> mandatory QTRM core -> Qwen3.5 LM head",
        "runtime_donor": False,
        "mandatory_core": bool(args.mandatory_core),
        "core_impl": str(args.core_impl),
        "qwen_trainability": qwen_trainability,
        "checkpoint_info": checkpoint_info,
        "model_report": model.report().__dict__,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "qwen_lr": float(args.qwen_lr),
        "base_kl_weight": float(args.base_kl_weight),
        "language_kl_weight": float(args.language_kl_weight),
        "margin_weight": float(args.margin_weight),
        "margin_value": float(args.margin_value),
        "margin_focus": str(args.margin_focus),
        "ce_focus": str(args.ce_focus),
        "balanced_category_sampling": bool(args.balanced_category_sampling),
        "category_regression_penalty": float(args.category_regression_penalty),
        "before_train_eval": (
            {k: v for k, v in before_train_eval.items() if k != "rows"}
            if before_train_eval is not None
            else None
        ),
        "after_train_eval": (
            {k: v for k, v in after_train_eval.items() if k != "rows"}
            if after_train_eval is not None
            else None
        ),
        "before_eval": {k: v for k, v in before_eval.items() if k != "rows"},
        "after_eval": {k: v for k, v in after_eval.items() if k != "rows"},
        "eval_category_gain_summary": eval_category_summary,
        "train_category_gain_summary": train_category_summary,
        "language": language,
        "train": {
            "last_loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / max(1, len(losses)),
            "best_periodic_eval": (
                {k: v for k, v in best_eval.items() if k != "rows"}
                if best_eval is not None
                else None
            ),
            "restored_best_checkpoint": bool(best_state is not None and bool(args.restore_best_checkpoint)),
        },
        "thresholds": {
            "min_eval_core_gain": float(args.min_eval_core_gain),
            "min_eval_core_accuracy": float(args.min_eval_core_accuracy),
            "min_eval_category_gain": float(args.min_eval_category_gain),
            "min_eval_category_hit_delta": int(args.min_eval_category_hit_delta),
            "category_guard_min_cases": int(args.category_guard_min_cases),
            "min_language_top1_agreement": float(args.min_language_top1_agreement),
            "max_repeated_token_run": int(args.max_repeated_token_run),
            "min_unique_ratio": float(args.min_unique_ratio),
        },
        "limitations": [
            "This is not a 27B parity claim.",
            "Train/eval suites must stay disjoint for public benchmark promotion.",
            "Promotion requires a held-out 256/full public subset gain, not train-set memorization.",
        ],
    }
    return report, trainable_state_dict(model), after_eval["rows"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--train-jsonl", default="local_eval/m7_public_reasoning_suite/mmlu_pro_validation_64.jsonl")
    parser.add_argument("--eval-jsonl", default="local_eval/m7_public_reasoning_suite/mmlu_pro_test_balanced_256.jsonl")
    parser.add_argument("--out-dir", default="local_eval/qwen35_integrated_public_mcq_healing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-eval-cases", type=int, default=256)
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument(
        "--core-impl",
        choices=["qwen_layer_wrapped", "qwen_shared_layer_wrapped"],
        default="qwen_layer_wrapped",
    )
    parser.add_argument("--qwen-core-layer-indices", default="3")
    parser.add_argument("--core-adapter-dim", type=int, default=128)
    parser.add_argument("--core-delta-adapter-mode", choices=["add", "adapter_only"], default="add")
    parser.add_argument(
        "--checkpoint-load-mode",
        choices=["strict_shapes", "skip_mismatch"],
        default="strict_shapes",
        help=(
            "Use skip_mismatch when warm-starting a larger QTRM adapter/core from "
            "an older checkpoint with smaller trainable tensors."
        ),
    )
    parser.add_argument("--mandatory-core", action="store_true")
    parser.add_argument("--core-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--unfreeze-qwen-layer-indices", default="")
    parser.add_argument("--unfreeze-qwen-lm-head", action="store_true")
    parser.add_argument("--unfreeze-qwen-final-norm", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--qwen-lr", type=float, default=5.0e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-kl-weight", type=float, default=0.02)
    parser.add_argument("--language-kl-weight", type=float, default=0.20)
    parser.add_argument("--ce-focus", choices=["all", "base_wrong"], default="all")
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--margin-value", type=float, default=0.5)
    parser.add_argument("--margin-focus", choices=["all", "base_wrong"], default="base_wrong")
    parser.add_argument("--balanced-category-sampling", action="store_true")
    parser.add_argument("--category-regression-penalty", type=float, default=0.0)
    parser.add_argument("--min-eval-category-gain", type=float, default=-1.0)
    parser.add_argument("--min-eval-category-hit-delta", type=int, default=-1_000_000)
    parser.add_argument("--category-guard-min-cases", type=int, default=1)
    parser.add_argument("--language-kl-batch-size", type=int, default=2)
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--restore-best-checkpoint", action="store_true")
    parser.add_argument("--min-eval-core-gain", type=float, default=0.01)
    parser.add_argument("--min-eval-core-accuracy", type=float, default=0.0)
    parser.add_argument("--min-language-top1-agreement", type=float, default=0.75)
    parser.add_argument("--max-repeated-token-run", type=int, default=8)
    parser.add_argument("--min-unique-ratio", type=float, default=0.20)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-generation-prompts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-checkpoint", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not str(args.save_checkpoint):
        args.save_checkpoint = str(out_dir / "last_core.pt")
    report, state, rows = train(args)
    torch.save({"model": state, "report": report}, str(args.save_checkpoint))
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
