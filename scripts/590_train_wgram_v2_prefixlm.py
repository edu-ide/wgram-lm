#!/usr/bin/env python3
"""Train and free-generation-evaluate the clean W-GRAM V2 PrefixLM path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wgram_lm.v2 import WGRAMReasoningLMV2, WGRAMV2Config
from wgram_lm.v2.contracts import build_v2_contract, validate_v2_contract
from wgram_lm.v2.generation import (
    build_v2_generation_policy,
    first_token_consistency_stats,
    generate_free,
    generation_repetition_stats,
)


def load_prefixlm_module() -> Any:
    path = ROOT / "scripts" / "534_train_native_prefixlm_dataio.py"
    spec = importlib.util.spec_from_file_location("wgram_v2_prefixlm_dataio", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def optimizer_lr_scale(
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if str(schedule) == "constant":
        return 1.0
    if str(schedule) != "warmup_cosine":
        raise ValueError(f"unknown lr schedule: {schedule}")
    step = max(1, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
    if warmup_steps > 0 and step <= warmup_steps:
        return max(1.0e-8, float(step) / float(warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, *, base_lr: float, scale: float) -> float:
    lr = float(base_lr) * float(scale)
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    return float(lr)


def finite_scalar_metrics(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            out[str(key)] = float(int(value))
        elif isinstance(value, int):
            out[str(key)] = float(value)
        elif isinstance(value, float) and math.isfinite(float(value)):
            out[str(key)] = float(value)
    return out


def create_tensorboard_writer(args: argparse.Namespace, out_dir: Path) -> tuple[Any | None, dict[str, Any]]:
    if bool(args.disable_tensorboard):
        return None, {"enabled": False, "reason": "disabled"}
    if SummaryWriter is None:
        return None, {"enabled": False, "reason": "torch_utils_tensorboard_unavailable"}
    logdir = Path(str(args.tensorboard_logdir).strip() or str(out_dir / "tensorboard"))
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    return writer, {"enabled": True, "logdir": str(logdir)}


def create_aim_run(args: argparse.Namespace, out_dir: Path, config: WGRAMV2Config) -> tuple[Any | None, dict[str, Any]]:
    if bool(args.disable_aim):
        return None, {"enabled": False, "reason": "disabled"}
    repo = str(args.aim_repo).strip()
    if not repo:
        return None, {"enabled": False, "reason": "aim_repo_empty"}
    try:
        from aim import Run as AimRun
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        return None, {"enabled": False, "reason": "aim_unavailable", "repo": repo, "error": str(exc)}
    repo_path = Path(repo)
    repo_path.mkdir(parents=True, exist_ok=True)
    run = AimRun(repo=str(repo_path), experiment=str(args.aim_experiment))
    run_name = str(args.aim_run_name).strip() or out_dir.name
    run.name = run_name
    run["args"] = {key: str(value) for key, value in vars(args).items()}
    run["config"] = {key: str(value) for key, value in asdict(config).items()}
    run["contract"] = build_v2_contract(config)
    return run, {
        "enabled": True,
        "repo": str(repo_path),
        "experiment": str(args.aim_experiment),
        "run_name": str(run_name),
    }


def log_training_row(
    row: dict[str, Any],
    *,
    tensorboard_writer: Any | None,
    aim_run: Any | None,
) -> None:
    step = int(row.get("step", 0))
    metrics = finite_scalar_metrics(row)
    if tensorboard_writer is not None:
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(f"train/{key}", float(value), step)
    if aim_run is not None:
        for key, value in metrics.items():
            aim_run.track(float(value), name=f"train/{key}", step=step)


def build_v2_config_from_args(args: argparse.Namespace, *, vocab_size: int) -> WGRAMV2Config:
    return WGRAMV2Config(
        vocab_size=int(vocab_size),
        d_model=int(args.d_model),
        max_position_embeddings=int(args.max_position_embeddings),
        max_response_position_embeddings=int(args.max_response_position_embeddings),
        tie_input_output_embeddings=not bool(args.disable_tied_input_output_embeddings),
        patch_size=int(args.patch_size),
        local_layers=int(args.local_layers),
        local_heads=int(args.local_heads),
        core_layers=int(args.core_layers),
        dropout=float(args.dropout),
        runtime_profile=str(args.runtime_profile),
        delta_backend=str(args.delta_backend),
        core_implementation=str(args.core_implementation),
        allow_torch_smoke_core=bool(args.allow_torch_smoke_core),
        official_gdn2_head_dim=int(args.official_gdn2_head_dim),
        official_gdn2_num_v_heads=int(args.official_gdn2_num_v_heads),
        official_gdn2_expand_v=float(args.official_gdn2_expand_v),
        official_gdn2_mode=str(args.official_gdn2_mode),
        official_gdn2_use_short_conv=not bool(args.official_gdn2_no_short_conv),
        official_gdn2_force_chunk_eval=not bool(args.official_gdn2_fused_recurrent_eval),
        official_gdn2_conv_size=int(args.official_gdn2_conv_size),
        official_gdn2_norm_eps=float(args.official_gdn2_norm_eps),
        force_fixed_boundaries=bool(args.force_fixed_boundaries),
        dynamic_boundary_threshold=float(args.dynamic_boundary_threshold),
        boundary_initial_logit=float(args.boundary_initial_logit),
        imta_trajectories=int(args.imta_trajectories),
        imta_noise_std=float(args.imta_noise_std),
        imta_selector_temperature=float(args.imta_selector_temperature),
        imta_adapter_gate_init=float(args.imta_adapter_gate_init),
        imta_post_adapter_gate_init=float(args.imta_post_adapter_gate_init),
        imta_selector_route_query_std=float(args.imta_selector_route_query_std),
        imta_diversity_weight=float(args.imta_diversity_weight),
        imta_route_min_probability=float(args.imta_route_min_probability),
        imta_route_entropy_floor=float(args.imta_route_entropy_floor),
        imta_route_entropy_weight=float(args.imta_route_entropy_weight),
        imta_route_balance_weight=float(args.imta_route_balance_weight),
        own_latent_prediction_enabled=not bool(args.disable_own_latent_prediction),
        own_latent_prediction_weight=float(args.own_latent_prediction_weight),
        repeat_unlikelihood_weight=float(args.repeat_unlikelihood_weight),
        premature_stop_loss_weight=float(args.premature_stop_loss_weight),
        response_start_loss_weight=float(args.response_start_loss_weight),
        response_start_stop_margin_weight=float(args.response_start_stop_margin_weight),
        response_start_stop_margin=float(args.response_start_stop_margin),
        response_continue_stop_margin_weight=float(args.response_continue_stop_margin_weight),
        response_continue_stop_margin=float(args.response_continue_stop_margin),
        response_body_loss_weight=float(args.response_body_loss_weight),
        response_stop_loss_weight=float(args.response_stop_loss_weight),
        use_response_phase_embeddings=not bool(args.disable_response_phase_embeddings),
        token_maturation_steps=int(args.token_maturation_steps),
        token_maturation_layers=int(args.token_maturation_layers),
        token_maturation_aux_loss_weight=float(args.token_maturation_aux_loss_weight),
        token_maturation_gate_init=float(args.token_maturation_gate_init),
        token_maturation_confidence_threshold=float(args.token_maturation_confidence_threshold),
        answer_memory_enabled=not bool(args.disable_answer_memory),
        answer_memory_steps=int(args.answer_memory_steps),
        answer_memory_plan_tokens=int(args.answer_memory_plan_tokens),
        answer_memory_plan_layers=int(args.answer_memory_plan_layers),
        answer_memory_prompt_context_enabled=bool(args.answer_memory_prompt_context),
        answer_memory_prompt_context_gate_init=float(args.answer_memory_prompt_context_gate_init),
        answer_memory_prompt_context_default_scale=float(args.answer_memory_prompt_context_default_scale),
        answer_memory_aux_loss_weight=float(args.answer_memory_aux_loss_weight),
        answer_memory_confidence_gate_enabled=bool(args.answer_memory_confidence_gate),
        answer_memory_confidence_mode=str(args.answer_memory_confidence_mode),
        answer_memory_confidence_topk=int(args.answer_memory_confidence_topk),
        answer_memory_confidence_floor=float(args.answer_memory_confidence_floor),
        answer_memory_stop_margin_loss_weight=float(args.answer_memory_stop_margin_loss_weight),
        answer_memory_stop_margin=float(args.answer_memory_stop_margin),
        answer_memory_commitment_scale=float(args.answer_memory_commitment_scale),
        answer_memory_commitment_confidence_gate_enabled=bool(args.answer_memory_commitment_confidence_gate),
        answer_memory_commitment_gate_init=float(args.answer_memory_commitment_gate_init),
        answer_prefix_commitment_loss_weight=float(args.answer_prefix_commitment_loss_weight),
        answer_memory_update_gate_init=float(args.answer_memory_update_gate_init),
        answer_memory_injection_gate_init=float(args.answer_memory_injection_gate_init),
        answer_memory_default_injection_scale=float(args.answer_memory_default_injection_scale),
        adaptive_latent_bridge_enabled=not bool(args.disable_adaptive_latent_bridge),
        adaptive_latent_bridge_gate_init=float(args.adaptive_latent_bridge_gate_init),
        byte_residual_gate_init=float(args.byte_residual_gate_init),
        latent_residual_gate_init=float(args.latent_residual_gate_init),
        stability_activation_clip_value=float(args.stability_activation_clip_value),
    )


def save_checkpoint(
    path: Path,
    *,
    model: WGRAMReasoningLMV2,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(model.config),
        "contract": build_v2_contract(model.config),
        "loss_history": losses,
        "args": vars(args),
        "dataset": dataset_summary,
    }
    torch.save(payload, path)


@torch.no_grad()
def build_self_rollout_inputs(
    model: WGRAMReasoningLMV2,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_mask: torch.Tensor,
    *,
    think_steps: int,
    max_rollout_tokens: int,
    stop_token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Replace response-prefix inputs with the model's own greedy predictions.

    This is train-time student forcing for exposure-bias diagnostics/repair:
    the second forward pass still predicts the gold labels with the same LM
    head, but its response prefix contains the current model's own mistakes.
    """

    rolled = input_ids.detach().clone()
    batch, seq_len = input_ids.shape
    stop_set = {int(token_id) for token_id in stop_token_ids}
    replaced_tokens = 0
    rows = 0
    stopped_rows = 0
    for row_idx in range(batch):
        valid_len = int(attention_mask[row_idx].detach().to(torch.long).sum().cpu().item())
        if valid_len <= 1:
            continue
        start_positions = torch.nonzero(
            response_start_mask[row_idx].bool() & (labels[row_idx] != -100),
            as_tuple=False,
        ).flatten()
        if int(start_positions.numel()) <= 0:
            continue
        response_start = int(start_positions[0].detach().cpu().item())
        if response_start >= valid_len:
            continue
        current = [int(token_id) for token_id in input_ids[row_idx, : response_start + 1].detach().cpu().tolist()]
        row_replacements = 0
        row_stopped = False
        for rollout_idx in range(int(max_rollout_tokens)):
            next_input_position = response_start + rollout_idx + 1
            if next_input_position >= min(valid_len, seq_len):
                break
            prefix = torch.tensor([current], dtype=torch.long, device=input_ids.device)
            prefix_attention = torch.ones_like(prefix)
            response_prediction_mask = torch.zeros_like(prefix)
            response_prediction_mask[:, response_start:] = 1
            logits, _, _ = model.forward_logits_and_hidden(
                prefix,
                prefix_attention,
                think_steps=int(think_steps),
                response_prediction_mask=response_prediction_mask,
            )
            next_id = int(logits[0, -1].float().argmax(dim=-1).detach().cpu().item())
            rolled[row_idx, next_input_position] = int(next_id)
            current.append(int(next_id))
            row_replacements += 1
            if next_id in stop_set:
                row_stopped = True
                break
        if row_replacements > 0:
            rows += 1
            replaced_tokens += int(row_replacements)
            stopped_rows += int(row_stopped)
    return rolled, {
        "self_rollout_rows": int(rows),
        "self_rollout_replaced_tokens": int(replaced_tokens),
        "self_rollout_stopped_rows": int(stopped_rows),
        "self_rollout_max_tokens": int(max_rollout_tokens),
    }


def response_until_stop_list(resp: list[int], stop_ids: tuple[int, ...]) -> list[int]:
    stop_set = {int(token_id) for token_id in stop_ids}
    out: list[int] = []
    for token_id in resp:
        out.append(int(token_id))
        if int(token_id) in stop_set:
            break
    return out


def build_response_balanced_sampler(
    dataset: Any,
    *,
    stop_token_ids: tuple[int, ...],
    body_boost: float,
    first_token_power: float,
    generator: torch.Generator,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    first_counts: dict[int, int] = {}
    row_infos: list[tuple[int, int]] = []
    for row_index in [int(value) for value in dataset.row_indices.tolist()]:
        resp = dataset._slice_tokens(dataset.resp_start[row_index], dataset.resp_len[row_index])
        response = response_until_stop_list([int(token_id) for token_id in resp.astype(np.int64).tolist()], stop_token_ids)
        first_token = int(response[0]) if response else -1
        response_len = int(len(response))
        first_counts[first_token] = first_counts.get(first_token, 0) + 1
        row_infos.append((first_token, response_len))
    weights: list[float] = []
    body_rows = 0
    for first_token, response_len in row_infos:
        freq = max(1, int(first_counts.get(int(first_token), 1)))
        weight = float(freq) ** (-float(first_token_power))
        has_body = int(response_len) > 2
        if has_body:
            weight *= max(float(body_boost), 1.0e-6)
            body_rows += 1
        weights.append(float(weight))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    top_first = sorted(first_counts.items(), key=lambda item: item[1], reverse=True)[:12]
    return sampler, {
        "enabled": True,
        "rows": int(len(weights)),
        "body_rows": int(body_rows),
        "body_fraction": float(body_rows / max(1, len(weights))),
        "body_boost": float(body_boost),
        "first_token_power": float(first_token_power),
        "top_first_token_counts": [[int(token_id), int(count)] for token_id, count in top_first],
    }


def scheduled_weight(*, target: float, step: int, start_after: int, warmup_steps: int) -> float:
    target_value = max(0.0, float(target))
    if target_value <= 0.0:
        return 0.0
    if int(step) < int(start_after):
        return 0.0
    if int(warmup_steps) <= 0:
        return target_value
    progress = (int(step) - int(start_after) + 1) / float(max(1, int(warmup_steps)))
    return target_value * max(0.0, min(1.0, progress))


def transplant_blt_prefixlm_donor(
    model: WGRAMReasoningLMV2,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Warm-start V2's surface language path from an older BLT PrefixLM checkpoint."""
    raw_path = Path(checkpoint_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"donor checkpoint not found: {raw_path}")
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    donor_state = dict(payload.get("model_state_dict", payload))
    target_state = model.state_dict()
    updates: dict[str, torch.Tensor] = {}
    mappings: list[tuple[str, str]] = [
        ("byte_embed.weight", "byte_embed.weight"),
        ("speaker.decoder.norm.weight", "clean_decoder.norm.weight"),
        ("speaker.decoder.norm.bias", "clean_decoder.norm.bias"),
        ("speaker.decoder.head.weight", "clean_decoder.head.weight"),
    ]
    for layer_index in range(int(model.config.local_layers)):
        prefix = f"speaker.decoder.layers.{layer_index}"
        donor_prefix = f"clean_decoder.layers.{layer_index}"
        for suffix in (
            "self_attn.in_proj_weight",
            "self_attn.in_proj_bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        ):
            mappings.append((f"{prefix}.{suffix}", f"{donor_prefix}.{suffix}"))

    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for target_key, donor_key in mappings:
        donor_tensor = donor_state.get(donor_key)
        target_tensor = target_state.get(target_key)
        if donor_tensor is None or target_tensor is None:
            skipped.append({"target": target_key, "donor": donor_key, "reason": "missing"})
            continue
        if tuple(donor_tensor.shape) != tuple(target_tensor.shape):
            skipped.append(
                {
                    "target": target_key,
                    "donor": donor_key,
                    "reason": "shape_mismatch",
                    "target_shape": list(target_tensor.shape),
                    "donor_shape": list(donor_tensor.shape),
                }
            )
            continue
        updates[target_key] = donor_tensor.detach().to(dtype=target_tensor.dtype)
        copied.append({"target": target_key, "donor": donor_key, "shape": list(target_tensor.shape)})

    target_state.update(updates)
    model.load_state_dict(target_state, strict=True)
    return {
        "enabled": True,
        "checkpoint": str(raw_path),
        "copied_count": int(len(copied)),
        "skipped_count": int(len(skipped)),
        "copied": copied,
        "skipped": skipped,
        "method": "blt_prefixlm_surface_path_transplant",
    }


def init_from_v2_checkpoint(
    model: WGRAMReasoningLMV2,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Warm-start the full V2 body from a previous V2 checkpoint."""
    raw_path = Path(checkpoint_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"V2 checkpoint not found: {raw_path}")
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    state_dict = dict(payload["model_state_dict"])
    load_result = model.load_state_dict(state_dict, strict=False)
    return {
        "enabled": True,
        "checkpoint": str(raw_path),
        "checkpoint_step": int(payload.get("step", 0)),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "method": "wgram_v2_full_body_warm_start",
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(int(args.seed))
    prefix = load_prefixlm_module()
    device = torch.device(str(args.device))
    dataset = prefix.DataIOSampledPrefixLMDataset(
        args.sampled_data,
        seq_len=int(args.seq_len),
        epoch=int(args.epoch),
        target_only=True,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        drop_overlength=True,
    )
    metadata = prefix.load_prefixlm_metadata(Path(args.sampled_data))
    tokenizer_info = dict(metadata.tokenizer_info or {})
    stop_token_ids = resolve_stop_token_ids(tokenizer_info)
    config = build_v2_config_from_args(args, vocab_size=int(metadata.vocab_size))
    validate_v2_contract(config, require_promotion_ready=bool(args.require_promotion_ready))
    model = WGRAMReasoningLMV2(config).to(device)
    donor_summary: dict[str, Any] = {"enabled": False}
    v2_init_summary: dict[str, Any] = {"enabled": False}
    if str(args.init_from_v2_checkpoint).strip() and str(args.init_from_blt_checkpoint).strip():
        raise ValueError("--init-from-v2-checkpoint and --init-from-blt-checkpoint are mutually exclusive")
    if str(args.init_from_v2_checkpoint).strip():
        v2_init_summary = init_from_v2_checkpoint(model, args.init_from_v2_checkpoint)
        model.to(device)
    elif str(args.init_from_blt_checkpoint).strip():
        donor_summary = transplant_blt_prefixlm_donor(model, args.init_from_blt_checkpoint)
        model.to(device)
    model.train()
    data_generator = torch.Generator()
    data_generator.manual_seed(int(args.seed))
    sampler = None
    sampler_summary: dict[str, Any] = {
        "enabled": False,
        "body_boost": float(args.response_body_sampling_boost),
        "first_token_power": float(args.response_first_token_balance_power),
    }
    if bool(args.balanced_response_sampler):
        sampler, sampler_summary = build_response_balanced_sampler(
            dataset,
            stop_token_ids=stop_token_ids,
            body_boost=float(args.response_body_sampling_boost),
            first_token_power=float(args.response_first_token_balance_power),
            generator=data_generator,
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=prefix.collate_prefixlm_rows,
        drop_last=False,
        generator=data_generator if sampler is None else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary = dataset.summary()
    tensorboard_writer, tensorboard_summary = create_tensorboard_writer(args, out_dir)
    aim_run, aim_summary = create_aim_run(args, out_dir, config)
    if tensorboard_writer is not None:
        tensorboard_writer.add_text("run/args", json.dumps(vars(args), ensure_ascii=False, indent=2), 0)
        tensorboard_writer.add_text("run/contract", json.dumps(build_v2_contract(config), ensure_ascii=False, indent=2), 0)
    if aim_run is not None:
        aim_run["dataset"] = dataset_summary
        aim_run["balanced_response_sampler"] = sampler_summary

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    total_optimizer_steps = max(1, int(args.steps))
    total_micro_steps = int(total_optimizer_steps * grad_accum_steps)
    effective_tokens_per_optimizer_step = int(args.batch_size) * int(args.seq_len) * int(grad_accum_steps)
    losses: list[dict[str, float | int | str]] = []
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    pending_loss_values: list[float] = []
    pending_metrics: dict[str, Any] = {}
    optimizer_step = 0
    for micro_step in range(1, total_micro_steps + 1):
        step = int((micro_step - 1) // grad_accum_steps) + 1
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = prefix.trim_prefixlm_batch_to_max_valid_length(batch)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        response_start_mask = batch["response_start_mask"].to(device)
        effective_response_stop_loss_weight = scheduled_weight(
            target=float(args.response_stop_loss_weight),
            step=int(step),
            start_after=int(args.response_stop_loss_start_after),
            warmup_steps=int(args.response_stop_loss_warmup_steps),
        )
        effective_response_continue_stop_margin_weight = scheduled_weight(
            target=float(args.response_continue_stop_margin_weight),
            step=int(step),
            start_after=int(args.response_continue_stop_margin_start_after),
            warmup_steps=int(args.response_continue_stop_margin_warmup_steps),
        )
        effective_answer_memory_injection_scale = scheduled_weight(
            target=float(args.answer_memory_default_injection_scale),
            step=int(step),
            start_after=int(args.answer_memory_injection_start_after),
            warmup_steps=int(args.answer_memory_injection_warmup_steps),
        )
        effective_answer_memory_commitment_scale = scheduled_weight(
            target=float(args.answer_memory_commitment_scale),
            step=int(step),
            start_after=int(args.answer_memory_commitment_start_after),
            warmup_steps=int(args.answer_memory_commitment_warmup_steps),
        )
        effective_answer_memory_prompt_context_scale = scheduled_weight(
            target=float(args.answer_memory_prompt_context_default_scale),
            step=int(step),
            start_after=int(args.answer_memory_prompt_context_start_after),
            warmup_steps=int(args.answer_memory_prompt_context_warmup_steps),
        )
        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=int(args.think_steps),
            response_start_mask=response_start_mask,
            stop_token_ids=stop_token_ids,
            response_stop_loss_weight=float(effective_response_stop_loss_weight),
            response_continue_stop_margin_weight=float(effective_response_continue_stop_margin_weight),
            answer_memory_injection_scale=float(effective_answer_memory_injection_scale),
            answer_memory_commitment_scale=float(effective_answer_memory_commitment_scale),
            answer_memory_prompt_context_scale=float(effective_answer_memory_prompt_context_scale),
        )
        if (
            float(args.self_rollout_loss_weight) > 0.0
            and int(args.self_rollout_max_tokens) > 0
            and step >= int(args.self_rollout_start_after)
        ):
            rolled_input_ids, rollout_build_metrics = build_self_rollout_inputs(
                model,
                input_ids,
                labels,
                attention_mask,
                response_start_mask,
                think_steps=int(args.think_steps),
                max_rollout_tokens=int(args.self_rollout_max_tokens),
                stop_token_ids=stop_token_ids,
            )
            if int(rollout_build_metrics["self_rollout_replaced_tokens"]) > 0:
                rollout_loss, rollout_metrics = model.forward_losses(
                    rolled_input_ids,
                    labels,
                    attention_mask,
                    think_steps=int(args.think_steps),
                    response_start_mask=response_start_mask,
                    stop_token_ids=stop_token_ids,
                    response_stop_loss_weight=float(effective_response_stop_loss_weight),
                    response_continue_stop_margin_weight=float(effective_response_continue_stop_margin_weight),
                    answer_memory_injection_scale=float(effective_answer_memory_injection_scale),
                    answer_memory_commitment_scale=float(effective_answer_memory_commitment_scale),
                    answer_memory_prompt_context_scale=float(effective_answer_memory_prompt_context_scale),
                )
                loss = loss + float(args.self_rollout_loss_weight) * rollout_loss
                metrics = {
                    **metrics,
                    **rollout_build_metrics,
                    "self_rollout_loss_weight": float(args.self_rollout_loss_weight),
                    "self_rollout_loss": float(rollout_loss.detach().float().cpu().item()),
                    "self_rollout_clean_loss": float(rollout_metrics.get("clean_loss", 0.0)),
                    "self_rollout_response_body_accuracy": float(rollout_metrics.get("response_body_accuracy", 0.0)),
                    "self_rollout_response_body_gold_probability": float(
                        rollout_metrics.get("response_body_gold_probability", 0.0)
                    ),
                }
            else:
                metrics = {
                    **metrics,
                    **rollout_build_metrics,
                    "self_rollout_loss_weight": float(args.self_rollout_loss_weight),
                    "self_rollout_loss": 0.0,
                    "self_rollout_clean_loss": 0.0,
                    "self_rollout_response_body_accuracy": 0.0,
                    "self_rollout_response_body_gold_probability": 0.0,
                }
        else:
            metrics = {
                **metrics,
                "self_rollout_loss_weight": float(args.self_rollout_loss_weight),
                "self_rollout_loss": 0.0,
                "self_rollout_clean_loss": 0.0,
                "self_rollout_replaced_tokens": 0,
                "self_rollout_rows": 0,
                "self_rollout_response_body_accuracy": 0.0,
                "self_rollout_response_body_gold_probability": 0.0,
            }
        metrics["response_stop_loss_effective_weight"] = float(effective_response_stop_loss_weight)
        metrics["response_stop_loss_start_after"] = int(args.response_stop_loss_start_after)
        metrics["response_stop_loss_warmup_steps"] = int(args.response_stop_loss_warmup_steps)
        metrics["response_continue_stop_margin_effective_weight"] = float(
            effective_response_continue_stop_margin_weight
        )
        metrics["response_continue_stop_margin_start_after"] = int(args.response_continue_stop_margin_start_after)
        metrics["response_continue_stop_margin_warmup_steps"] = int(args.response_continue_stop_margin_warmup_steps)
        metrics["answer_memory_injection_effective_scale"] = float(effective_answer_memory_injection_scale)
        metrics["answer_memory_injection_start_after"] = int(args.answer_memory_injection_start_after)
        metrics["answer_memory_injection_warmup_steps"] = int(args.answer_memory_injection_warmup_steps)
        metrics["answer_memory_commitment_effective_scale"] = float(effective_answer_memory_commitment_scale)
        metrics["answer_memory_commitment_start_after"] = int(args.answer_memory_commitment_start_after)
        metrics["answer_memory_commitment_warmup_steps"] = int(args.answer_memory_commitment_warmup_steps)
        metrics["answer_memory_prompt_context_effective_scale"] = float(
            effective_answer_memory_prompt_context_scale
        )
        metrics["answer_memory_prompt_context_start_after"] = int(args.answer_memory_prompt_context_start_after)
        metrics["answer_memory_prompt_context_warmup_steps"] = int(args.answer_memory_prompt_context_warmup_steps)
        raw_loss_value = float(loss.detach().float().cpu().item())
        pending_loss_values.append(float(raw_loss_value))
        pending_metrics = dict(metrics)
        (loss / float(grad_accum_steps)).backward()
        if micro_step % grad_accum_steps != 0:
            continue

        optimizer_step += 1
        lr_scale = optimizer_lr_scale(
            step=int(optimizer_step),
            total_steps=int(total_optimizer_steps),
            schedule=str(args.lr_schedule),
            warmup_steps=int(args.optimizer_warmup_steps),
            min_lr_ratio=float(args.min_lr_ratio),
        )
        current_lr = set_optimizer_lr(optimizer, base_lr=float(args.lr), scale=float(lr_scale))
        grad_norm = 0.0
        if float(args.grad_clip) > 0.0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            grad_norm = float(grad_norm_tensor.detach().float().cpu().item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        mean_loss = float(sum(pending_loss_values) / max(1, len(pending_loss_values)))
        row = {
            "step": int(optimizer_step),
            "optimizer_step": int(optimizer_step),
            "micro_step": int(micro_step),
            "micro_batches": int(len(pending_loss_values)),
            "grad_accum_steps": int(grad_accum_steps),
            "effective_tokens_per_optimizer_step": int(effective_tokens_per_optimizer_step),
            "lr_schedule": str(args.lr_schedule),
            "optimizer_warmup_steps": int(args.optimizer_warmup_steps),
            "min_lr_ratio": float(args.min_lr_ratio),
            "learning_rate": float(current_lr),
            "lr_scale": float(lr_scale),
            "grad_norm": float(grad_norm),
            **pending_metrics,
            "loss": float(mean_loss),
            "last_micro_loss": float(raw_loss_value),
        }
        losses.append(row)
        log_training_row(row, tensorboard_writer=tensorboard_writer, aim_run=aim_run)
        if int(args.log_every) > 0 and optimizer_step % int(args.log_every) == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        pending_loss_values = []
        pending_metrics = {}

    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()
    if aim_run is not None and hasattr(aim_run, "close"):
        aim_run.close()

    checkpoint_path = out_dir / "last_model.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=int(total_optimizer_steps),
        losses=losses,
        args=args,
        dataset_summary=dataset_summary,
    )
    report = {
        "steps": int(total_optimizer_steps),
        "micro_steps": int(total_micro_steps),
        "optimizer": {
            "optimizer_steps": int(total_optimizer_steps),
            "grad_accum_steps": int(grad_accum_steps),
            "effective_tokens_per_optimizer_step": int(effective_tokens_per_optimizer_step),
            "lr_schedule": str(args.lr_schedule),
            "optimizer_warmup_steps": int(args.optimizer_warmup_steps),
            "min_lr_ratio": float(args.min_lr_ratio),
            "base_lr": float(args.lr),
        },
        "tensorboard": tensorboard_summary,
        "aim": aim_summary,
        "checkpoint": str(checkpoint_path),
        "contract": build_v2_contract(config),
        "dataset": dataset_summary,
        "balanced_response_sampler": sampler_summary,
        "training_stop_token_ids": [int(token_id) for token_id in stop_token_ids],
        "training_stop_tokens": [decode_token_ids([int(token_id)], tokenizer_info) for token_id in stop_token_ids],
        "donor_initialization": donor_summary,
        "v2_initialization": v2_init_summary,
        "loss_history": losses,
    }
    report_path = out_dir / "train_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def resolve_eval_tokenizer_path(tokenizer_info: dict[str, Any]) -> Path | None:
    raw_path = str(tokenizer_info.get("tokenizer_path") or "").strip()
    if not raw_path:
        return None
    candidates = [Path(raw_path)]
    if not Path(raw_path).is_absolute():
        candidates.append(ROOT / raw_path)
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=8)
def load_eval_tokenizer(tokenizer_path: str) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def tokenizer_mode(tokenizer_info: dict[str, Any]) -> str:
    if resolve_eval_tokenizer_path(tokenizer_info) is not None:
        return "tokenizer_json"
    return "byte_shifted"


def decode_token_ids(token_ids: list[int], tokenizer_info: dict[str, Any]) -> str:
    tokenizer_path = resolve_eval_tokenizer_path(tokenizer_info)
    if tokenizer_path is not None:
        tokenizer = load_eval_tokenizer(str(tokenizer_path))
        return str(tokenizer.decode([int(value) for value in token_ids], skip_special_tokens=False))
    byte_offset = int(tokenizer_info.get("byte_offset", 2))
    eos_id = int(tokenizer_info.get("eos_token_id", 1))
    pieces: list[str] = []
    byte_buffer: list[int] = []
    for token_id in [int(value) for value in token_ids]:
        if token_id == eos_id:
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()
            pieces.append("<eos>")
            continue
        value = int(token_id) - byte_offset
        if 0 <= value <= 255:
            byte_buffer.append(value)
        else:
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()
            pieces.append(f"<id:{int(token_id)}>")
    if byte_buffer:
        pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
    return "".join(pieces)


def decode_byte_ids(token_ids: list[int], tokenizer_info: dict[str, Any]) -> str:
    return decode_token_ids(token_ids, tokenizer_info)


def response_until_eos(resp: np.ndarray, eos_id: int) -> list[int]:
    return response_until_stop(resp, (int(eos_id),))


def resolve_stop_token_ids(tokenizer_info: dict[str, Any]) -> tuple[int, ...]:
    ids: list[int] = []
    eos_id = tokenizer_info.get("eos_token_id")
    if eos_id is not None:
        ids.append(int(eos_id))
    tokenizer_path = resolve_eval_tokenizer_path(tokenizer_info)
    if tokenizer_path is not None:
        tokenizer = load_eval_tokenizer(str(tokenizer_path))
        for key in ("eoa", "eoq"):
            token_text = str(tokenizer_info.get(key) or "").strip()
            if not token_text:
                continue
            token_id = tokenizer.token_to_id(token_text)
            if token_id is not None:
                ids.append(int(token_id))
    if not ids:
        ids.append(int(tokenizer_info.get("eos_token_id", 1)))
    return tuple(dict.fromkeys(ids))


def response_until_stop(resp: np.ndarray, stop_ids: tuple[int, ...]) -> list[int]:
    stop_set = {int(token_id) for token_id in stop_ids}
    out: list[int] = []
    for token_id in resp.astype(np.int64).tolist():
        out.append(int(token_id))
        if int(token_id) in stop_set:
            break
    return out


def token_diagnostics(
    token_ids: list[int],
    tokenizer_info: dict[str, Any],
    *,
    stop_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    mode = tokenizer_mode(tokenizer_info)
    byte_offset = int(tokenizer_info.get("byte_offset", 2))
    eos_id = int(tokenizer_info.get("eos_token_id", 1))
    stop_set = {int(token_id) for token_id in stop_ids} if stop_ids else {int(eos_id)}
    ids = [int(token_id) for token_id in token_ids]
    length = len(ids)
    if length == 0:
        return {
            "length": 0,
            "eos_emitted": False,
            "eos_position": -1,
            "stop_emitted": False,
            "stop_position": -1,
            "first_token_id": -1,
            "first_token_is_eos": False,
            "first_token_is_stop": False,
            "tokenizer_mode": mode,
            "direct_byte_shifted_applicable": bool(mode == "byte_shifted"),
            "first_token_byte_decodable": False,
            "byte_decodable_fraction": 0.0,
            "decoded_text_nonempty_fraction": 0.0,
            "control_byte_fraction": 0.0,
            "control_text_fraction": 0.0,
            "non_byte_token_fraction": 0.0,
        }
    direct_byte_shifted = mode == "byte_shifted"
    byte_decodable = [direct_byte_shifted and byte_offset <= token_id <= byte_offset + 255 for token_id in ids]
    control_bytes = [
        direct_byte_shifted and (byte_offset <= token_id <= byte_offset + 31 or token_id == byte_offset + 127)
        for token_id in ids
    ]
    decoded_pieces = [decode_token_ids([token_id], tokenizer_info) for token_id in ids]
    decoded_nonempty = [bool(piece) for piece in decoded_pieces]
    control_text = [
        bool(piece) and all((ord(char) < 32) or char.isspace() for char in piece)
        for piece in decoded_pieces
    ]
    eos_position = next((idx for idx, token_id in enumerate(ids) if token_id == eos_id), -1)
    stop_position = next((idx for idx, token_id in enumerate(ids) if token_id in stop_set), -1)
    return {
        "length": int(length),
        "eos_emitted": bool(eos_position >= 0),
        "eos_position": int(eos_position),
        "stop_emitted": bool(stop_position >= 0),
        "stop_position": int(stop_position),
        "first_token_id": int(ids[0]),
        "first_token_is_eos": bool(ids[0] == eos_id),
        "first_token_is_stop": bool(ids[0] in stop_set),
        "tokenizer_mode": mode,
        "direct_byte_shifted_applicable": bool(direct_byte_shifted),
        "first_token_byte_decodable": bool(byte_decodable[0]),
        "byte_decodable_fraction": float(sum(int(v) for v in byte_decodable) / length),
        "decoded_text_nonempty_fraction": float(sum(int(v) for v in decoded_nonempty) / length),
        "control_byte_fraction": float(sum(int(v) for v in control_bytes) / length),
        "control_text_fraction": float(sum(int(v) for v in control_text) / length),
        "non_byte_token_fraction": float(sum(int(not v) for v in byte_decodable) / length),
    }


@torch.no_grad()
def first_response_token_diagnostics(
    model: WGRAMReasoningLMV2,
    prefix_ids: list[int],
    gold_ids: list[int],
    *,
    eos_id: int,
    stop_ids: tuple[int, ...],
    think_steps: int,
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    if not gold_ids:
        return {
            "available": False,
            "nonfinite_logits": False,
            "gold_token_id": -1,
            "top1_id": -1,
            "rank": -1,
            "top5_hit": False,
            "top1_is_eos": False,
            "top1_is_stop": False,
            "top1_byte_decodable": False,
            "best_stop_id": -1,
            "best_stop_token": "",
            "best_stop_logit": 0.0,
            "best_stop_probability": 0.0,
            "gold_minus_best_stop_logit": 0.0,
            "gold_beats_best_stop": False,
        }
    device = next(model.parameters()).device
    input_ids = torch.tensor([[int(token_id) for token_id in prefix_ids]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    response_prediction_mask = torch.zeros_like(input_ids)
    response_prediction_mask[:, max(0, len(prefix_ids) - 1) :] = 1
    logits, _, _ = model.forward_logits_and_hidden(
        input_ids,
        attention_mask,
        think_steps=int(think_steps),
        response_prediction_mask=response_prediction_mask,
    )
    next_logits = logits[0, -1].float()
    nonfinite_logits = not bool(torch.isfinite(next_logits).all().detach().cpu().item())
    next_logits = torch.nan_to_num(next_logits, nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
    vocab_size = int(next_logits.shape[-1])
    gold_token = int(gold_ids[0])
    safe_gold = min(max(gold_token, 0), vocab_size - 1)
    target_logit = next_logits[safe_gold]
    rank = int((next_logits > target_logit).sum().detach().cpu().item()) + 1
    topk = min(5, vocab_size)
    top_values, top_indices = torch.topk(next_logits, k=topk)
    top_ids = [int(value) for value in top_indices.detach().cpu().tolist()]
    top_logits = [float(value) for value in top_values.detach().cpu().tolist()]
    probs = torch.softmax(next_logits, dim=-1)
    top1_id = int(top_ids[0])
    token_info = token_diagnostics([top1_id], tokenizer_info, stop_ids=stop_ids)
    stop_set = {int(token_id) for token_id in stop_ids}
    best_stop_id = -1
    best_stop_logit = 0.0
    best_stop_probability = 0.0
    gold_minus_best_stop_logit = 0.0
    if stop_set:
        safe_stop_ids = [int(token_id) for token_id in stop_set if 0 <= int(token_id) < vocab_size]
        if safe_stop_ids:
            stop_tensor = torch.tensor(safe_stop_ids, dtype=torch.long, device=next_logits.device)
            stop_logits = next_logits[stop_tensor]
            stop_probs = probs[stop_tensor]
            best_index = int(stop_logits.argmax(dim=-1).detach().cpu().item())
            best_stop_id = int(safe_stop_ids[best_index])
            best_stop_logit = float(stop_logits[best_index].detach().cpu().item())
            best_stop_probability = float(stop_probs[best_index].detach().cpu().item())
            gold_minus_best_stop_logit = float((target_logit - stop_logits[best_index]).detach().cpu().item())
    return {
        "available": True,
        "nonfinite_logits": bool(nonfinite_logits),
        "gold_token_id": int(gold_token),
        "gold_token": decode_token_ids([int(gold_token)], tokenizer_info),
        "gold_probability": float(probs[safe_gold].detach().cpu().item()),
        "top1_id": int(top1_id),
        "top1_token": decode_token_ids([int(top1_id)], tokenizer_info),
        "top1_probability": float(probs[top1_id].detach().cpu().item()),
        "top1_logit": float(top_logits[0]),
        "rank": int(rank),
        "top5_hit": bool(gold_token in top_ids),
        "top5_ids": top_ids,
        "top5_tokens": [decode_token_ids([token_id], tokenizer_info) for token_id in top_ids],
        "top1_is_eos": bool(top1_id == int(eos_id)),
        "top1_is_stop": bool(top1_id in stop_set),
        "top1_byte_decodable": bool(token_info["first_token_byte_decodable"]),
        "best_stop_id": int(best_stop_id),
        "best_stop_token": decode_token_ids([int(best_stop_id)], tokenizer_info) if best_stop_id >= 0 else "",
        "best_stop_logit": float(best_stop_logit),
        "best_stop_probability": float(best_stop_probability),
        "gold_minus_best_stop_logit": float(gold_minus_best_stop_logit),
        "gold_beats_best_stop": bool(gold_minus_best_stop_logit > 0.0),
    }


@torch.no_grad()
def answer_memory_plan_diagnostics(
    model: WGRAMReasoningLMV2,
    prefix_ids: list[int],
    gold_ids: list[int],
    *,
    eos_id: int,
    stop_ids: tuple[int, ...],
    think_steps: int,
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "available": False,
        "reason": "unavailable",
        "target_token_count": 0,
        "top1_count": 0,
        "top5_count": 0,
        "token_accuracy_fraction": 0.0,
        "top5_fraction": 0.0,
        "mean_rank": 0.0,
        "mean_gold_probability": 0.0,
        "mean_plan_confidence": 0.0,
        "mean_topk_probability_mass": 0.0,
        "mean_entropy_complement": 0.0,
        "nonfinite_token_count": 0,
        "tokens": [],
    }
    if not prefix_ids:
        return {**empty, "reason": "empty_prefix"}
    if not gold_ids:
        return {**empty, "reason": "empty_gold"}
    if not bool(getattr(model.config, "answer_memory_enabled", False)):
        return {**empty, "reason": "answer_memory_disabled"}

    device = next(model.parameters()).device
    input_ids = torch.tensor([[int(token_id) for token_id in prefix_ids]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    response_prediction_mask = torch.zeros_like(input_ids)
    response_prediction_mask[:, max(0, len(prefix_ids) - 1) :] = 1
    model.forward_logits_and_hidden(
        input_ids,
        attention_mask,
        think_steps=int(think_steps),
        response_prediction_mask=response_prediction_mask,
        answer_memory_injection_scale=0.0,
    )
    plan_logits = getattr(model.speaker, "last_answer_memory_plan_logits", None)
    if plan_logits is None:
        return {**empty, "reason": "answer_memory_plan_logits_missing"}

    logits = plan_logits[0].float()
    target_count = min(int(logits.shape[0]), len(gold_ids))
    if target_count <= 0:
        return {**empty, "reason": "no_overlapping_targets"}

    vocab_size = int(logits.shape[-1])
    stop_set = {int(token_id) for token_id in stop_ids}
    token_rows: list[dict[str, Any]] = []
    top1_count = 0
    top5_count = 0
    ranks: list[int] = []
    gold_probabilities: list[float] = []
    plan_confidences: list[float] = []
    topk_probability_masses: list[float] = []
    entropy_complements: list[float] = []
    for plan_index in range(target_count):
        next_logits = logits[plan_index]
        nonfinite_logits = not bool(torch.isfinite(next_logits).all().detach().cpu().item())
        next_logits = torch.nan_to_num(next_logits, nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
        probs = torch.softmax(next_logits, dim=-1)
        gold_token = int(gold_ids[plan_index])
        safe_gold = min(max(gold_token, 0), vocab_size - 1)
        gold_logit = next_logits[safe_gold]
        rank = int((next_logits > gold_logit).sum().detach().cpu().item()) + 1
        topk = min(5, vocab_size)
        top_values, top_indices = torch.topk(next_logits, k=topk)
        top_ids = [int(value) for value in top_indices.detach().cpu().tolist()]
        top_probs = [float(probs[token_id].detach().cpu().item()) for token_id in top_ids]
        topk_probability_mass = float(sum(top_probs))
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
        entropy_norm = torch.log(probs.new_tensor(float(max(2, vocab_size))))
        entropy_complement = float((1.0 - entropy / entropy_norm).clamp(min=0.0, max=1.0).detach().cpu().item())
        top1_id = int(top_ids[0])
        top1_hit = bool(top1_id == gold_token)
        top5_hit = bool(gold_token in top_ids)
        top1_count += int(top1_hit)
        top5_count += int(top5_hit)
        ranks.append(int(rank))
        gold_probability = float(probs[safe_gold].detach().cpu().item())
        plan_confidence = float(probs[top1_id].detach().cpu().item())
        gold_probabilities.append(gold_probability)
        plan_confidences.append(plan_confidence)
        topk_probability_masses.append(topk_probability_mass)
        entropy_complements.append(entropy_complement)
        token_rows.append(
            {
                "plan_index": int(plan_index),
                "nonfinite_logits": bool(nonfinite_logits),
                "gold_token_id": int(gold_token),
                "gold_token": decode_token_ids([int(gold_token)], tokenizer_info),
                "gold_probability": float(gold_probability),
                "rank": int(rank),
                "top1_id": int(top1_id),
                "top1_token": decode_token_ids([int(top1_id)], tokenizer_info),
                "top1_probability": float(plan_confidence),
                "top1_hit": bool(top1_hit),
                "top5_hit": bool(top5_hit),
                "top5_ids": top_ids,
                "top5_tokens": [decode_token_ids([token_id], tokenizer_info) for token_id in top_ids],
                "top5_probabilities": top_probs,
                "top5_probability_mass": float(topk_probability_mass),
                "entropy_complement": float(entropy_complement),
                "top1_is_eos": bool(top1_id == int(eos_id)),
                "top1_is_stop": bool(top1_id in stop_set),
            }
        )

    return {
        "available": True,
        "reason": "ok",
        "target_token_count": int(target_count),
        "top1_count": int(top1_count),
        "top5_count": int(top5_count),
        "token_accuracy_fraction": float(top1_count / target_count),
        "top5_fraction": float(top5_count / target_count),
        "mean_rank": float(sum(ranks) / len(ranks)) if ranks else 0.0,
        "mean_gold_probability": (
            float(sum(gold_probabilities) / len(gold_probabilities)) if gold_probabilities else 0.0
        ),
        "mean_plan_confidence": (
            float(sum(plan_confidences) / len(plan_confidences)) if plan_confidences else 0.0
        ),
        "mean_topk_probability_mass": (
            float(sum(topk_probability_masses) / len(topk_probability_masses))
            if topk_probability_masses
            else 0.0
        ),
        "mean_entropy_complement": (
            float(sum(entropy_complements) / len(entropy_complements)) if entropy_complements else 0.0
        ),
        "nonfinite_token_count": int(
            sum(int(bool(token_row["nonfinite_logits"])) for token_row in token_rows)
        ),
        "tokens": token_rows,
    }


def load_checkpoint_model(checkpoint_path: Path, *, device: str | torch.device) -> WGRAMReasoningLMV2:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = WGRAMV2Config(**dict(payload["config"]))
    model = WGRAMReasoningLMV2(config)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(torch.device(device))
    model.eval()
    return model


def run_generation_gate_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    sampled_data: str | Path,
    epoch: int,
    seq_len: int,
    max_rows: int,
    max_new_tokens: int,
    device: str = "cuda",
    think_steps: int = 1,
    generation_repetition_penalty: float = 1.0,
    generation_repetition_window: int = 32,
    generation_temperature: float = 0.0,
    generation_top_p: float = 1.0,
) -> dict[str, Any]:
    prefix = load_prefixlm_module()
    model = load_checkpoint_model(Path(checkpoint_path), device=device)
    dataset = prefix.DataIOSampledPrefixLMDataset(
        sampled_data,
        seq_len=int(seq_len),
        epoch=int(epoch),
        target_only=True,
        max_rows=int(max_rows),
        drop_overlength=True,
    )
    metadata = prefix.load_prefixlm_metadata(Path(sampled_data))
    tokenizer_info = dict(metadata.tokenizer_info or {})
    eos_id = int(tokenizer_info.get("eos_token_id", 1))
    stop_ids = resolve_stop_token_ids(tokenizer_info)
    samples: list[dict[str, Any]] = []
    exact = 0
    rows = min(int(max_rows), len(dataset))
    deterministic_free_decode = (
        float(generation_repetition_penalty) <= 1.0
        and float(generation_temperature) <= 0.0
        and float(generation_top_p) >= 1.0
    )
    for index in range(rows):
        source_row = int(dataset.row_indices[int(index)])
        inst = dataset._slice_tokens(dataset.inst_start[source_row], dataset.inst_len[source_row])
        resp = dataset._slice_tokens(dataset.resp_start[source_row], dataset.resp_len[source_row])
        gold = response_until_stop(resp, stop_ids)
        generated = generate_free(
            model,
            [int(token_id) for token_id in inst.astype(np.int64).tolist()],
            max_new_tokens=int(max_new_tokens),
            eos_id=int(eos_id),
            stop_ids=stop_ids,
            think_steps=int(think_steps),
            temperature=float(generation_temperature),
            top_p=float(generation_top_p),
            repetition_penalty=float(generation_repetition_penalty),
            repetition_window=int(generation_repetition_window),
        )
        exact += int(generated == gold)
        repetition_stats = generation_repetition_stats(generated)
        token_stats = token_diagnostics(generated, tokenizer_info, stop_ids=stop_ids)
        first_token_stats = first_response_token_diagnostics(
            model,
            [int(token_id) for token_id in inst.astype(np.int64).tolist()],
            gold,
            eos_id=int(eos_id),
            stop_ids=stop_ids,
            think_steps=int(think_steps),
            tokenizer_info=tokenizer_info,
        )
        answer_plan_stats = answer_memory_plan_diagnostics(
            model,
            [int(token_id) for token_id in inst.astype(np.int64).tolist()],
            gold,
            eos_id=int(eos_id),
            stop_ids=stop_ids,
            think_steps=int(think_steps),
            tokenizer_info=tokenizer_info,
        )
        first_token_consistency = first_token_consistency_stats(
            generated,
            first_token_stats,
            deterministic_free_decode=bool(deterministic_free_decode),
        )
        samples.append(
            {
                "row_index": int(source_row),
                "instruction": decode_byte_ids([int(v) for v in inst.tolist()], tokenizer_info),
                "gold_ids": gold,
                "generated_ids": generated,
                "gold": decode_byte_ids(gold, tokenizer_info),
                "generated": decode_byte_ids(generated, tokenizer_info),
                "repetition": repetition_stats,
                "token_diagnostics": token_stats,
                "first_response_token": first_token_stats,
                "first_token_consistency": first_token_consistency,
                "answer_memory_plan": answer_plan_stats,
                "exact": bool(generated == gold),
            }
        )
    loop_like_count = sum(int(bool(sample["repetition"]["loop_like"])) for sample in samples)
    eos_count = sum(int(bool(sample["token_diagnostics"]["eos_emitted"])) for sample in samples)
    stop_count = sum(int(bool(sample["token_diagnostics"]["stop_emitted"])) for sample in samples)
    first_token_eos_count = sum(int(bool(sample["token_diagnostics"]["first_token_is_eos"])) for sample in samples)
    first_token_stop_count = sum(int(bool(sample["token_diagnostics"]["first_token_is_stop"])) for sample in samples)
    teacher_forced_top5_count = sum(int(bool(sample["first_response_token"].get("top5_hit", False))) for sample in samples)
    teacher_forced_top1_eos_count = sum(int(bool(sample["first_response_token"].get("top1_is_eos", False))) for sample in samples)
    teacher_forced_top1_stop_count = sum(int(bool(sample["first_response_token"].get("top1_is_stop", False))) for sample in samples)
    teacher_forced_nonfinite_logits_count = sum(
        int(bool(sample["first_response_token"].get("nonfinite_logits", False)))
        for sample in samples
        if bool(sample["first_response_token"].get("available", False))
    )
    teacher_forced_ranks = [
        int(sample["first_response_token"]["rank"])
        for sample in samples
        if bool(sample["first_response_token"].get("available", False))
    ]
    teacher_forced_gold_stop_margins = [
        float(sample["first_response_token"]["gold_minus_best_stop_logit"])
        for sample in samples
        if bool(sample["first_response_token"].get("available", False))
        and int(sample["first_response_token"].get("best_stop_id", -1)) >= 0
        and math.isfinite(float(sample["first_response_token"]["gold_minus_best_stop_logit"]))
    ]
    teacher_forced_gold_beats_stop_count = sum(
        int(bool(sample["first_response_token"].get("gold_beats_best_stop", False)))
        for sample in samples
        if bool(sample["first_response_token"].get("available", False))
    )
    first_token_consistency_available_count = sum(
        int(bool(sample["first_token_consistency"].get("available", False))) for sample in samples
    )
    first_token_consistency_required_count = sum(
        int(bool(sample["first_token_consistency"].get("consistency_required", False))) for sample in samples
    )
    first_token_consistency_match_count = sum(
        int(bool(sample["first_token_consistency"].get("matches_teacher_forced_top1", False)))
        for sample in samples
        if bool(sample["first_token_consistency"].get("available", False))
    )
    first_token_consistency_mismatch_count = sum(
        int(
            bool(sample["first_token_consistency"].get("consistency_required", False))
            and not bool(sample["first_token_consistency"].get("matches_teacher_forced_top1", False))
        )
        for sample in samples
    )
    first_token_generated_gold_count = sum(
        int(bool(sample["first_token_consistency"].get("matches_gold", False)))
        for sample in samples
        if bool(sample["first_token_consistency"].get("available", False))
    )
    answer_plan_available_count = sum(
        int(bool(sample["answer_memory_plan"].get("available", False))) for sample in samples
    )
    answer_plan_target_tokens = sum(
        int(sample["answer_memory_plan"].get("target_token_count", 0))
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
    )
    answer_plan_top1_count = sum(
        int(sample["answer_memory_plan"].get("top1_count", 0))
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
    )
    answer_plan_top5_count = sum(
        int(sample["answer_memory_plan"].get("top5_count", 0))
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
    )
    answer_plan_nonfinite_token_count = sum(
        int(sample["answer_memory_plan"].get("nonfinite_token_count", 0))
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
    )
    answer_plan_ranks = [
        int(token["rank"])
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
        for token in sample["answer_memory_plan"].get("tokens", [])
    ]
    answer_plan_gold_probabilities = [
        float(token["gold_probability"])
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
        for token in sample["answer_memory_plan"].get("tokens", [])
        if math.isfinite(float(token["gold_probability"]))
    ]
    answer_plan_confidences = [
        float(token["top1_probability"])
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
        for token in sample["answer_memory_plan"].get("tokens", [])
        if math.isfinite(float(token["top1_probability"]))
    ]
    answer_plan_topk_masses = [
        float(token["top5_probability_mass"])
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
        for token in sample["answer_memory_plan"].get("tokens", [])
        if math.isfinite(float(token.get("top5_probability_mass", 0.0)))
    ]
    answer_plan_entropy_complements = [
        float(token["entropy_complement"])
        for sample in samples
        if bool(sample["answer_memory_plan"].get("available", False))
        for token in sample["answer_memory_plan"].get("tokens", [])
        if math.isfinite(float(token.get("entropy_complement", 0.0)))
    ]
    mean_unique_fraction = (
        sum(float(sample["repetition"]["unique_fraction"]) for sample in samples) / len(samples) if samples else 0.0
    )
    mean_max_run = (
        sum(float(sample["repetition"]["max_consecutive_run"]) for sample in samples) / len(samples) if samples else 0.0
    )
    mean_byte_decodable_fraction = (
        sum(float(sample["token_diagnostics"]["byte_decodable_fraction"]) for sample in samples) / len(samples)
        if samples
        else 0.0
    )
    mean_control_text_fraction = (
        sum(float(sample["token_diagnostics"]["control_text_fraction"]) for sample in samples) / len(samples)
        if samples
        else 0.0
    )
    mean_decoded_nonempty_fraction = (
        sum(float(sample["token_diagnostics"]["decoded_text_nonempty_fraction"]) for sample in samples) / len(samples)
        if samples
        else 0.0
    )
    return {
        "gate_type": "wgram_v2_generation_gate",
        "evaluation_policy": "free_generation_only",
        "generation_policy": build_v2_generation_policy(
            repetition_penalty=float(generation_repetition_penalty),
            repetition_window=int(generation_repetition_window),
            temperature=float(generation_temperature),
            top_p=float(generation_top_p),
        ),
        "checkpoint": str(checkpoint_path),
        "sampled_data": str(sampled_data),
        "epoch": int(epoch),
        "seq_len": int(seq_len),
        "tokenizer_mode": tokenizer_mode(tokenizer_info),
        "tokenizer_path": str(resolve_eval_tokenizer_path(tokenizer_info) or ""),
        "stop_token_ids": [int(token_id) for token_id in stop_ids],
        "stop_tokens": [decode_token_ids([int(token_id)], tokenizer_info) for token_id in stop_ids],
        "generation": {
            "rows": int(rows),
            "exact": int(exact),
            "exact_fraction": float(exact / rows) if rows else 0.0,
            "max_new_tokens": int(max_new_tokens),
            "loop_like_count": int(loop_like_count),
            "loop_like_fraction": float(loop_like_count / rows) if rows else 0.0,
            "mean_unique_fraction": float(mean_unique_fraction),
            "mean_max_consecutive_run": float(mean_max_run),
            "eos_count": int(eos_count),
            "eos_fraction": float(eos_count / rows) if rows else 0.0,
            "stop_count": int(stop_count),
            "stop_fraction": float(stop_count / rows) if rows else 0.0,
            "first_token_eos_count": int(first_token_eos_count),
            "first_token_eos_fraction": float(first_token_eos_count / rows) if rows else 0.0,
            "first_token_stop_count": int(first_token_stop_count),
            "first_token_stop_fraction": float(first_token_stop_count / rows) if rows else 0.0,
            "mean_byte_decodable_fraction": float(mean_byte_decodable_fraction),
            "mean_decoded_text_nonempty_fraction": float(mean_decoded_nonempty_fraction),
            "mean_control_text_fraction": float(mean_control_text_fraction),
            "teacher_forced_first_token_top5_count": int(teacher_forced_top5_count),
            "teacher_forced_first_token_top5_fraction": float(teacher_forced_top5_count / rows) if rows else 0.0,
            "teacher_forced_first_token_top1_eos_count": int(teacher_forced_top1_eos_count),
            "teacher_forced_first_token_top1_eos_fraction": float(teacher_forced_top1_eos_count / rows) if rows else 0.0,
            "teacher_forced_first_token_top1_stop_count": int(teacher_forced_top1_stop_count),
            "teacher_forced_first_token_top1_stop_fraction": float(teacher_forced_top1_stop_count / rows) if rows else 0.0,
            "teacher_forced_first_token_nonfinite_logits_count": int(teacher_forced_nonfinite_logits_count),
            "teacher_forced_first_token_nonfinite_logits_fraction": (
                float(teacher_forced_nonfinite_logits_count / rows) if rows else 0.0
            ),
            "teacher_forced_first_token_mean_rank": (
                float(sum(teacher_forced_ranks) / len(teacher_forced_ranks)) if teacher_forced_ranks else 0.0
            ),
            "teacher_forced_first_token_mean_gold_minus_best_stop_logit": (
                float(sum(teacher_forced_gold_stop_margins) / len(teacher_forced_gold_stop_margins))
                if teacher_forced_gold_stop_margins
                else 0.0
            ),
            "teacher_forced_first_token_gold_beats_stop_count": int(teacher_forced_gold_beats_stop_count),
            "teacher_forced_first_token_gold_beats_stop_fraction": (
                float(teacher_forced_gold_beats_stop_count / rows) if rows else 0.0
            ),
            "first_token_generation_teacher_forced_top1_available_count": int(
                first_token_consistency_available_count
            ),
            "first_token_generation_teacher_forced_top1_available_fraction": (
                float(first_token_consistency_available_count / rows) if rows else 0.0
            ),
            "first_token_generation_teacher_forced_top1_required_count": int(
                first_token_consistency_required_count
            ),
            "first_token_generation_teacher_forced_top1_match_count": int(first_token_consistency_match_count),
            "first_token_generation_teacher_forced_top1_match_fraction": (
                float(first_token_consistency_match_count / first_token_consistency_available_count)
                if first_token_consistency_available_count
                else 0.0
            ),
            "first_token_generation_teacher_forced_top1_mismatch_count": int(
                first_token_consistency_mismatch_count
            ),
            "first_token_generation_teacher_forced_top1_mismatch_fraction": (
                float(first_token_consistency_mismatch_count / first_token_consistency_required_count)
                if first_token_consistency_required_count
                else 0.0
            ),
            "first_token_generation_teacher_forced_top1_consistency_pass": bool(
                first_token_consistency_mismatch_count == 0
            ),
            "first_token_generation_gold_match_count": int(first_token_generated_gold_count),
            "first_token_generation_gold_match_fraction": (
                float(first_token_generated_gold_count / first_token_consistency_available_count)
                if first_token_consistency_available_count
                else 0.0
            ),
            "answer_memory_plan_available_count": int(answer_plan_available_count),
            "answer_memory_plan_available_fraction": (
                float(answer_plan_available_count / rows) if rows else 0.0
            ),
            "answer_memory_plan_target_tokens": int(answer_plan_target_tokens),
            "answer_memory_plan_token_accuracy_fraction": (
                float(answer_plan_top1_count / answer_plan_target_tokens)
                if answer_plan_target_tokens
                else 0.0
            ),
            "answer_memory_plan_top5_fraction": (
                float(answer_plan_top5_count / answer_plan_target_tokens)
                if answer_plan_target_tokens
                else 0.0
            ),
            "answer_memory_plan_nonfinite_token_count": int(answer_plan_nonfinite_token_count),
            "answer_memory_plan_nonfinite_token_fraction": (
                float(answer_plan_nonfinite_token_count / answer_plan_target_tokens)
                if answer_plan_target_tokens
                else 0.0
            ),
            "answer_memory_plan_mean_rank": (
                float(sum(answer_plan_ranks) / len(answer_plan_ranks)) if answer_plan_ranks else 0.0
            ),
            "answer_memory_plan_mean_gold_probability": (
                float(sum(answer_plan_gold_probabilities) / len(answer_plan_gold_probabilities))
                if answer_plan_gold_probabilities
                else 0.0
            ),
            "answer_memory_plan_mean_confidence": (
                float(sum(answer_plan_confidences) / len(answer_plan_confidences))
                if answer_plan_confidences
                else 0.0
            ),
            "answer_memory_plan_mean_top5_probability_mass": (
                float(sum(answer_plan_topk_masses) / len(answer_plan_topk_masses))
                if answer_plan_topk_masses
                else 0.0
            ),
            "answer_memory_plan_mean_entropy_complement": (
                float(sum(answer_plan_entropy_complements) / len(answer_plan_entropy_complements))
                if answer_plan_entropy_complements
                else 0.0
            ),
            "samples": samples,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17035)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--optimizer-warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--tensorboard-logdir", default="")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--aim-repo", default="")
    parser.add_argument("--aim-experiment", default="wgram_v2")
    parser.add_argument("--aim-run-name", default="")
    parser.add_argument("--disable-aim", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--max-response-position-embeddings", type=int, default=1024)
    parser.add_argument("--disable-tied-input-output-embeddings", action="store_true")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--core-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--think-steps", type=int, default=1)
    parser.add_argument("--runtime-profile", choices=("smoke", "promotion"), default="smoke")
    parser.add_argument("--delta-backend", default="official_gated_delta2")
    parser.add_argument("--core-implementation", choices=("torch_smoke", "official_gated_delta2"), default="torch_smoke")
    parser.add_argument("--allow-torch-smoke-core", action="store_true")
    parser.add_argument("--require-promotion-ready", action="store_true")
    parser.add_argument("--official-gdn2-head-dim", type=int, default=0)
    parser.add_argument("--official-gdn2-num-v-heads", type=int, default=0)
    parser.add_argument("--official-gdn2-expand-v", type=float, default=1.0)
    parser.add_argument("--official-gdn2-mode", default="chunk")
    parser.add_argument("--official-gdn2-no-short-conv", action="store_true")
    parser.add_argument("--official-gdn2-fused-recurrent-eval", action="store_true")
    parser.add_argument("--official-gdn2-conv-size", type=int, default=4)
    parser.add_argument("--official-gdn2-norm-eps", type=float, default=1.0e-5)
    parser.add_argument("--force-fixed-boundaries", action="store_true")
    parser.add_argument("--dynamic-boundary-threshold", type=float, default=0.6)
    parser.add_argument("--boundary-initial-logit", type=float, default=-2.0)
    parser.add_argument("--imta-trajectories", type=int, default=3)
    parser.add_argument("--imta-noise-std", type=float, default=0.0)
    parser.add_argument("--imta-selector-temperature", type=float, default=0.8)
    parser.add_argument("--imta-adapter-gate-init", type=float, default=-1.0)
    parser.add_argument("--imta-post-adapter-gate-init", type=float, default=-1.0)
    parser.add_argument("--imta-selector-route-query-std", type=float, default=0.02)
    parser.add_argument("--imta-diversity-weight", type=float, default=0.0)
    parser.add_argument("--imta-route-min-probability", type=float, default=0.0)
    parser.add_argument("--imta-route-entropy-floor", type=float, default=0.0)
    parser.add_argument("--imta-route-entropy-weight", type=float, default=0.0)
    parser.add_argument("--imta-route-balance-weight", type=float, default=0.0)
    parser.add_argument("--disable-own-latent-prediction", action="store_true")
    parser.add_argument("--own-latent-prediction-weight", type=float, default=0.0)
    parser.add_argument("--repeat-unlikelihood-weight", type=float, default=0.0)
    parser.add_argument("--premature-stop-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-start-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-start-stop-margin-weight", type=float, default=0.0)
    parser.add_argument("--response-start-stop-margin", type=float, default=1.0)
    parser.add_argument("--response-continue-stop-margin-weight", type=float, default=0.0)
    parser.add_argument("--response-continue-stop-margin", type=float, default=1.0)
    parser.add_argument("--response-continue-stop-margin-start-after", type=int, default=0)
    parser.add_argument("--response-continue-stop-margin-warmup-steps", type=int, default=0)
    parser.add_argument("--response-body-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-stop-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-stop-loss-start-after", type=int, default=1)
    parser.add_argument("--response-stop-loss-warmup-steps", type=int, default=0)
    parser.add_argument("--disable-response-phase-embeddings", action="store_true")
    parser.add_argument("--token-maturation-steps", type=int, default=2)
    parser.add_argument("--token-maturation-layers", type=int, default=1)
    parser.add_argument("--token-maturation-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--token-maturation-gate-init", type=float, default=-1.0)
    parser.add_argument("--token-maturation-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--disable-answer-memory", action="store_true")
    parser.add_argument("--answer-memory-steps", type=int, default=2)
    parser.add_argument("--answer-memory-plan-tokens", type=int, default=4)
    parser.add_argument("--answer-memory-plan-layers", type=int, default=1)
    parser.add_argument("--answer-memory-prompt-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--answer-memory-prompt-context-gate-init", type=float, default=-1.0)
    parser.add_argument("--answer-memory-prompt-context-default-scale", type=float, default=1.0)
    parser.add_argument("--answer-memory-prompt-context-start-after", type=int, default=1)
    parser.add_argument("--answer-memory-prompt-context-warmup-steps", type=int, default=0)
    parser.add_argument("--answer-memory-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-memory-confidence-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--answer-memory-confidence-mode",
        choices=("top1_probability", "topk_mass", "entropy_complement", "hybrid_topk_entropy"),
        default="topk_mass",
    )
    parser.add_argument("--answer-memory-confidence-topk", type=int, default=5)
    parser.add_argument("--answer-memory-confidence-floor", type=float, default=0.20)
    parser.add_argument("--answer-memory-stop-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-memory-stop-margin", type=float, default=1.0)
    parser.add_argument("--answer-memory-commitment-scale", type=float, default=1.0)
    parser.add_argument("--answer-memory-commitment-confidence-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--answer-memory-commitment-gate-init", type=float, default=-0.5)
    parser.add_argument("--answer-prefix-commitment-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-memory-commitment-start-after", type=int, default=1)
    parser.add_argument("--answer-memory-commitment-warmup-steps", type=int, default=0)
    parser.add_argument("--answer-memory-update-gate-init", type=float, default=-1.0)
    parser.add_argument("--answer-memory-injection-gate-init", type=float, default=-1.5)
    parser.add_argument("--answer-memory-default-injection-scale", type=float, default=1.0)
    parser.add_argument("--answer-memory-injection-start-after", type=int, default=1)
    parser.add_argument("--answer-memory-injection-warmup-steps", type=int, default=0)
    parser.add_argument("--disable-adaptive-latent-bridge", action="store_true")
    parser.add_argument("--adaptive-latent-bridge-gate-init", type=float, default=-2.0)
    parser.add_argument("--init-from-blt-checkpoint", default="")
    parser.add_argument("--init-from-v2-checkpoint", default="")
    parser.add_argument("--self-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--self-rollout-max-tokens", type=int, default=0)
    parser.add_argument("--self-rollout-start-after", type=int, default=1)
    parser.add_argument("--balanced-response-sampler", action="store_true")
    parser.add_argument("--response-body-sampling-boost", type=float, default=2.0)
    parser.add_argument("--response-first-token-balance-power", type=float, default=0.5)
    parser.add_argument("--byte-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--latent-residual-gate-init", type=float, default=2.0)
    parser.add_argument("--stability-activation-clip-value", type=float, default=30.0)
    parser.add_argument("--eval-checkpoint", default="")
    parser.add_argument("--eval-max-rows", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--generation-repetition-window", type=int, default=32)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--eval-out", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if str(args.eval_checkpoint):
        report = run_generation_gate_from_checkpoint(
            args.eval_checkpoint,
            sampled_data=args.sampled_data,
            epoch=int(args.epoch),
            seq_len=int(args.seq_len),
            max_rows=int(args.eval_max_rows),
            max_new_tokens=int(args.max_new_tokens),
            device=str(args.device),
            think_steps=int(args.think_steps),
            generation_repetition_penalty=float(args.generation_repetition_penalty),
            generation_repetition_window=int(args.generation_repetition_window),
            generation_temperature=float(args.generation_temperature),
            generation_top_p=float(args.generation_top_p),
        )
    else:
        report = train(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if str(args.eval_out):
        out_path = Path(args.eval_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
