#!/usr/bin/env python3
"""Run the single-priority W-GRAM V2 fastlane recipe.

This intentionally does not run K sweeps, own-latent off comparisons, or
candidate/forced-choice gates. Those are postponed unless the primary V2 recipe
needs debugging.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "590_train_wgram_v2_prefixlm.py"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_dir(args: argparse.Namespace) -> Path:
    name = str(args.run_name or "").strip()
    if not name:
        name = time.strftime("wgram_v2_fastlane_%Y%m%d_%H%M%S")
    return Path(args.out_root) / name


def recipe(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.smoke):
        return {
            "runtime_profile": "smoke",
            "core_implementation": "torch_smoke",
            "allow_torch_smoke_core": True,
            "require_promotion_ready": False,
            "d_model": int(args.smoke_d_model),
            "local_heads": int(args.smoke_local_heads),
            "core_layers": int(args.smoke_core_layers),
            "local_layers": int(args.smoke_local_layers),
            "steps": int(args.smoke_steps),
            "grad_accum_steps": 1,
        }
    return {
        "runtime_profile": "promotion",
        "core_implementation": "official_gated_delta2",
        "allow_torch_smoke_core": False,
        "require_promotion_ready": True,
        "d_model": int(args.d_model),
        "local_heads": int(args.local_heads),
        "core_layers": int(args.core_layers),
        "local_layers": int(args.local_layers),
        "steps": int(args.steps),
        "grad_accum_steps": int(args.grad_accum_steps),
    }


def resolve_optimizer_schedule(args: argparse.Namespace, *, steps: int) -> dict[str, int | float | str]:
    if int(args.optimizer_warmup_steps) >= 0:
        warmup_steps = int(args.optimizer_warmup_steps)
    else:
        warmup_steps = max(1, int(round(int(steps) * float(args.optimizer_warmup_fraction))))
    return {
        "lr_schedule": str(args.lr_schedule),
        "warmup_steps": int(warmup_steps),
        "warmup_fraction": float(args.optimizer_warmup_fraction),
        "min_lr_ratio": float(args.min_lr_ratio),
    }


def resolve_tensorboard_logdir(args: argparse.Namespace, out_dir: Path) -> Path:
    root = str(args.tensorboard_root).strip()
    if root:
        return Path(root) / out_dir.name
    return out_dir / "tensorboard"


def resolve_response_stop_schedule(args: argparse.Namespace, *, steps: int) -> dict[str, int | float]:
    if int(args.response_stop_loss_start_after) >= 0:
        start_after = int(args.response_stop_loss_start_after)
    else:
        start_after = max(1, int(round(int(steps) * float(args.response_stop_loss_start_fraction))))
    if int(args.response_stop_loss_warmup_steps) >= 0:
        warmup_steps = int(args.response_stop_loss_warmup_steps)
    else:
        warmup_steps = max(1, int(round(int(steps) * float(args.response_stop_loss_warmup_fraction))))
    return {
        "start_after": int(start_after),
        "warmup_steps": int(warmup_steps),
        "start_fraction": float(args.response_stop_loss_start_fraction),
        "warmup_fraction": float(args.response_stop_loss_warmup_fraction),
    }


def resolve_response_continue_stop_margin_schedule(args: argparse.Namespace, *, steps: int) -> dict[str, int | float]:
    if int(args.response_continue_stop_margin_start_after) >= 0:
        start_after = int(args.response_continue_stop_margin_start_after)
    else:
        start_after = max(1, int(round(int(steps) * float(args.response_continue_stop_margin_start_fraction))))
    if int(args.response_continue_stop_margin_warmup_steps) >= 0:
        warmup_steps = int(args.response_continue_stop_margin_warmup_steps)
    else:
        warmup_steps = max(1, int(round(int(steps) * float(args.response_continue_stop_margin_warmup_fraction))))
    return {
        "start_after": int(start_after),
        "warmup_steps": int(warmup_steps),
        "start_fraction": float(args.response_continue_stop_margin_start_fraction),
        "warmup_fraction": float(args.response_continue_stop_margin_warmup_fraction),
    }


def resolve_answer_memory_injection_schedule(args: argparse.Namespace, *, steps: int) -> dict[str, int | float]:
    if int(args.answer_memory_injection_start_after) >= 0:
        start_after = int(args.answer_memory_injection_start_after)
    else:
        start_after = max(1, int(round(int(steps) * float(args.answer_memory_injection_start_fraction))))
    if int(args.answer_memory_injection_warmup_steps) >= 0:
        warmup_steps = int(args.answer_memory_injection_warmup_steps)
    else:
        warmup_steps = max(1, int(round(int(steps) * float(args.answer_memory_injection_warmup_fraction))))
    return {
        "start_after": int(start_after),
        "warmup_steps": int(warmup_steps),
        "start_fraction": float(args.answer_memory_injection_start_fraction),
        "warmup_fraction": float(args.answer_memory_injection_warmup_fraction),
    }


def resolve_answer_memory_commitment_schedule(args: argparse.Namespace, *, steps: int) -> dict[str, int | float]:
    if int(args.answer_memory_commitment_start_after) >= 0:
        start_after = int(args.answer_memory_commitment_start_after)
    else:
        start_after = max(1, int(round(int(steps) * float(args.answer_memory_commitment_start_fraction))))
    if int(args.answer_memory_commitment_warmup_steps) >= 0:
        warmup_steps = int(args.answer_memory_commitment_warmup_steps)
    else:
        warmup_steps = max(1, int(round(int(steps) * float(args.answer_memory_commitment_warmup_fraction))))
    return {
        "start_after": int(start_after),
        "warmup_steps": int(warmup_steps),
        "start_fraction": float(args.answer_memory_commitment_start_fraction),
        "warmup_fraction": float(args.answer_memory_commitment_warmup_fraction),
    }


def resolve_answer_memory_prompt_context_schedule(args: argparse.Namespace, *, steps: int) -> dict[str, int | float]:
    if int(args.answer_memory_prompt_context_start_after) >= 0:
        start_after = int(args.answer_memory_prompt_context_start_after)
    else:
        start_after = max(1, int(round(int(steps) * float(args.answer_memory_prompt_context_start_fraction))))
    if int(args.answer_memory_prompt_context_warmup_steps) >= 0:
        warmup_steps = int(args.answer_memory_prompt_context_warmup_steps)
    else:
        warmup_steps = max(1, int(round(int(steps) * float(args.answer_memory_prompt_context_warmup_fraction))))
    return {
        "start_after": int(start_after),
        "warmup_steps": int(warmup_steps),
        "start_fraction": float(args.answer_memory_prompt_context_start_fraction),
        "warmup_fraction": float(args.answer_memory_prompt_context_warmup_fraction),
    }


def build_train_command(args: argparse.Namespace, out_dir: Path, current_recipe: dict[str, Any]) -> list[str]:
    optimizer_schedule = resolve_optimizer_schedule(args, steps=int(current_recipe["steps"]))
    stop_schedule = resolve_response_stop_schedule(args, steps=int(current_recipe["steps"]))
    continue_schedule = resolve_response_continue_stop_margin_schedule(args, steps=int(current_recipe["steps"]))
    memory_schedule = resolve_answer_memory_injection_schedule(args, steps=int(current_recipe["steps"]))
    commitment_schedule = resolve_answer_memory_commitment_schedule(args, steps=int(current_recipe["steps"]))
    prompt_context_schedule = resolve_answer_memory_prompt_context_schedule(args, steps=int(current_recipe["steps"]))
    tensorboard_logdir = resolve_tensorboard_logdir(args, out_dir)
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--sampled-data",
        str(args.sampled_data),
        "--out-dir",
        str(out_dir),
        "--steps",
        str(current_recipe["steps"]),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--max-rows",
        str(args.max_rows),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--grad-accum-steps",
        str(current_recipe["grad_accum_steps"]),
        "--lr-schedule",
        str(optimizer_schedule["lr_schedule"]),
        "--optimizer-warmup-steps",
        str(optimizer_schedule["warmup_steps"]),
        "--min-lr-ratio",
        str(optimizer_schedule["min_lr_ratio"]),
        "--tensorboard-logdir",
        str(tensorboard_logdir),
        "--aim-repo",
        str(args.aim_repo),
        "--aim-experiment",
        str(args.aim_experiment),
        "--aim-run-name",
        str(out_dir.name),
        "--d-model",
        str(current_recipe["d_model"]),
        "--max-position-embeddings",
        str(args.max_position_embeddings),
        "--max-response-position-embeddings",
        str(args.max_response_position_embeddings),
        "--patch-size",
        str(args.patch_size),
        "--dynamic-boundary-threshold",
        str(args.dynamic_boundary_threshold),
        "--boundary-initial-logit",
        str(args.boundary_initial_logit),
        "--local-layers",
        str(current_recipe["local_layers"]),
        "--local-heads",
        str(current_recipe["local_heads"]),
        "--core-layers",
        str(current_recipe["core_layers"]),
        "--think-steps",
        str(args.think_steps),
        "--runtime-profile",
        str(current_recipe["runtime_profile"]),
        "--core-implementation",
        str(current_recipe["core_implementation"]),
        "--imta-trajectories",
        "3",
        "--imta-noise-std",
        str(args.imta_noise_std),
        "--imta-selector-temperature",
        str(args.imta_selector_temperature),
        "--imta-adapter-gate-init",
        str(args.imta_adapter_gate_init),
        "--imta-post-adapter-gate-init",
        str(args.imta_post_adapter_gate_init),
        "--imta-selector-route-query-std",
        str(args.imta_selector_route_query_std),
        "--imta-diversity-weight",
        str(args.imta_diversity_weight),
        "--imta-route-min-probability",
        str(args.imta_route_min_probability),
        "--imta-route-entropy-floor",
        str(args.imta_route_entropy_floor),
        "--imta-route-entropy-weight",
        str(args.imta_route_entropy_weight),
        "--imta-route-balance-weight",
        str(args.imta_route_balance_weight),
        "--own-latent-prediction-weight",
        str(args.own_latent_prediction_weight),
        "--repeat-unlikelihood-weight",
        str(args.repeat_unlikelihood_weight),
        "--premature-stop-loss-weight",
        str(args.premature_stop_loss_weight),
        "--response-start-loss-weight",
        str(args.response_start_loss_weight),
        "--response-start-stop-margin-weight",
        str(args.response_start_stop_margin_weight),
        "--response-start-stop-margin",
        str(args.response_start_stop_margin),
        "--response-continue-stop-margin-weight",
        str(args.response_continue_stop_margin_weight),
        "--response-continue-stop-margin",
        str(args.response_continue_stop_margin),
        "--response-continue-stop-margin-start-after",
        str(continue_schedule["start_after"]),
        "--response-continue-stop-margin-warmup-steps",
        str(continue_schedule["warmup_steps"]),
        "--response-body-loss-weight",
        str(args.response_body_loss_weight),
        "--response-stop-loss-weight",
        str(args.response_stop_loss_weight),
        "--response-stop-loss-start-after",
        str(stop_schedule["start_after"]),
        "--response-stop-loss-warmup-steps",
        str(stop_schedule["warmup_steps"]),
        "--token-maturation-steps",
        str(args.token_maturation_steps),
        "--token-maturation-layers",
        str(args.token_maturation_layers),
        "--token-maturation-aux-loss-weight",
        str(args.token_maturation_aux_loss_weight),
        "--token-maturation-gate-init",
        str(args.token_maturation_gate_init),
        "--token-maturation-confidence-threshold",
        str(args.token_maturation_confidence_threshold),
        "--answer-memory-steps",
        str(args.answer_memory_steps),
        "--answer-memory-plan-tokens",
        str(args.answer_memory_plan_tokens),
        "--answer-memory-plan-layers",
        str(args.answer_memory_plan_layers),
        "--answer-memory-prompt-context-gate-init",
        str(args.answer_memory_prompt_context_gate_init),
        "--answer-memory-prompt-context-default-scale",
        str(args.answer_memory_prompt_context_default_scale),
        "--answer-memory-prompt-context-start-after",
        str(prompt_context_schedule["start_after"]),
        "--answer-memory-prompt-context-warmup-steps",
        str(prompt_context_schedule["warmup_steps"]),
        "--answer-memory-aux-loss-weight",
        str(args.answer_memory_aux_loss_weight),
        "--answer-memory-confidence-mode",
        str(args.answer_memory_confidence_mode),
        "--answer-memory-confidence-topk",
        str(args.answer_memory_confidence_topk),
        "--answer-memory-confidence-floor",
        str(args.answer_memory_confidence_floor),
        "--answer-memory-stop-margin-loss-weight",
        str(args.answer_memory_stop_margin_loss_weight),
        "--answer-memory-stop-margin",
        str(args.answer_memory_stop_margin),
        "--answer-memory-commitment-scale",
        str(args.answer_memory_commitment_scale),
        "--answer-memory-commitment-gate-init",
        str(args.answer_memory_commitment_gate_init),
        "--answer-prefix-commitment-loss-weight",
        str(args.answer_prefix_commitment_loss_weight),
        "--answer-memory-commitment-start-after",
        str(commitment_schedule["start_after"]),
        "--answer-memory-commitment-warmup-steps",
        str(commitment_schedule["warmup_steps"]),
        "--answer-memory-update-gate-init",
        str(args.answer_memory_update_gate_init),
        "--answer-memory-injection-gate-init",
        str(args.answer_memory_injection_gate_init),
        "--answer-memory-default-injection-scale",
        str(args.answer_memory_default_injection_scale),
        "--answer-memory-injection-start-after",
        str(memory_schedule["start_after"]),
        "--answer-memory-injection-warmup-steps",
        str(memory_schedule["warmup_steps"]),
        "--adaptive-latent-bridge-gate-init",
        str(args.adaptive_latent_bridge_gate_init),
        "--byte-residual-gate-init",
        str(args.byte_residual_gate_init),
        "--latent-residual-gate-init",
        str(args.latent_residual_gate_init),
        "--stability-activation-clip-value",
        str(args.stability_activation_clip_value),
        "--self-rollout-loss-weight",
        str(args.self_rollout_loss_weight),
        "--self-rollout-max-tokens",
        str(args.self_rollout_max_tokens),
        "--self-rollout-start-after",
        str(args.self_rollout_start_after),
        "--response-body-sampling-boost",
        str(args.response_body_sampling_boost),
        "--response-first-token-balance-power",
        str(args.response_first_token_balance_power),
        "--log-every",
        str(args.log_every),
    ]
    if bool(args.force_fixed_boundaries):
        command.append("--force-fixed-boundaries")
    if bool(current_recipe["allow_torch_smoke_core"]):
        command.append("--allow-torch-smoke-core")
    if bool(current_recipe["require_promotion_ready"]):
        command.append("--require-promotion-ready")
    if bool(args.official_gdn2_no_short_conv):
        command.append("--official-gdn2-no-short-conv")
    if bool(args.official_gdn2_fused_recurrent_eval):
        command.append("--official-gdn2-fused-recurrent-eval")
    if bool(args.balanced_response_sampler):
        command.append("--balanced-response-sampler")
    if bool(args.disable_tensorboard):
        command.append("--disable-tensorboard")
    if bool(args.disable_aim):
        command.append("--disable-aim")
    if not bool(args.answer_memory):
        command.append("--disable-answer-memory")
    command.append(
        "--answer-memory-prompt-context"
        if bool(args.answer_memory_prompt_context)
        else "--no-answer-memory-prompt-context"
    )
    if not bool(args.answer_memory_confidence_gate):
        command.append("--no-answer-memory-confidence-gate")
    command.append(
        "--answer-memory-commitment-confidence-gate"
        if bool(args.answer_memory_commitment_confidence_gate)
        else "--no-answer-memory-commitment-confidence-gate"
    )
    if not bool(args.adaptive_latent_bridge):
        command.append("--disable-adaptive-latent-bridge")
    if str(args.init_from_blt_checkpoint).strip():
        command.extend(["--init-from-blt-checkpoint", str(args.init_from_blt_checkpoint)])
    if str(args.init_from_v2_checkpoint).strip():
        command.extend(["--init-from-v2-checkpoint", str(args.init_from_v2_checkpoint)])
    if not bool(args.tie_input_output_embeddings):
        command.append("--disable-tied-input-output-embeddings")
    if not bool(args.response_phase_embeddings):
        command.append("--disable-response-phase-embeddings")
    return command


def build_eval_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(TRAINER),
        "--sampled-data",
        str(args.sampled_data),
        "--out-dir",
        str(out_dir),
        "--eval-checkpoint",
        str(out_dir / "last_model.pt"),
        "--seq-len",
        str(args.seq_len),
        "--epoch",
        str(args.epoch),
        "--device",
        str(args.device),
        "--eval-max-rows",
        str(args.eval_max_rows),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--think-steps",
        str(args.think_steps),
        "--generation-repetition-penalty",
        str(args.generation_repetition_penalty),
        "--generation-repetition-window",
        str(args.generation_repetition_window),
        "--generation-temperature",
        str(args.generation_temperature),
        "--generation-top-p",
        str(args.generation_top_p),
        "--eval-out",
        str(out_dir / "free_generation_gate.json"),
    ]


def build_fastlane_plan(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = run_dir(args)
    current_recipe = recipe(args)
    optimizer_schedule = resolve_optimizer_schedule(args, steps=int(current_recipe["steps"]))
    stop_schedule = resolve_response_stop_schedule(args, steps=int(current_recipe["steps"]))
    continue_schedule = resolve_response_continue_stop_margin_schedule(args, steps=int(current_recipe["steps"]))
    memory_schedule = resolve_answer_memory_injection_schedule(args, steps=int(current_recipe["steps"]))
    commitment_schedule = resolve_answer_memory_commitment_schedule(args, steps=int(current_recipe["steps"]))
    prompt_context_schedule = resolve_answer_memory_prompt_context_schedule(args, steps=int(current_recipe["steps"]))
    train_command = build_train_command(args, out_dir, current_recipe)
    eval_command = build_eval_command(args, out_dir)
    return {
        "experiment_type": "single_core_wgram_v2_fastlane",
        "comparison_policy": "non_core_comparisons_deferred",
        "purpose": "advance the latest-method primary W-GRAM V2 reasoning-language recipe before running non-core comparisons",
        "out_dir": str(out_dir),
        "sampled_data": str(args.sampled_data),
        "recipe": {
            **current_recipe,
            "imta_trajectories": 3,
            "grad_accum_steps": int(current_recipe["grad_accum_steps"]),
            "effective_tokens_per_optimizer_step": int(
                args.batch_size * args.seq_len * int(current_recipe["grad_accum_steps"])
            ),
            "lr_schedule": str(optimizer_schedule["lr_schedule"]),
            "optimizer_warmup_steps": int(optimizer_schedule["warmup_steps"]),
            "optimizer_warmup_fraction": float(optimizer_schedule["warmup_fraction"]),
            "min_lr_ratio": float(optimizer_schedule["min_lr_ratio"]),
            "tensorboard_logdir": str(resolve_tensorboard_logdir(args, out_dir)),
            "aim_repo": str(args.aim_repo),
            "aim_experiment": str(args.aim_experiment),
            "max_position_embeddings": int(args.max_position_embeddings),
            "max_response_position_embeddings": int(args.max_response_position_embeddings),
            "tie_input_output_embeddings": bool(args.tie_input_output_embeddings),
            "use_response_phase_embeddings": bool(args.response_phase_embeddings),
            "dynamic_boundary_threshold": float(args.dynamic_boundary_threshold),
            "boundary_initial_logit": float(args.boundary_initial_logit),
            "own_latent_prediction_weight": float(args.own_latent_prediction_weight),
            "imta_diversity_weight": float(args.imta_diversity_weight),
            "imta_route_min_probability": float(args.imta_route_min_probability),
            "imta_route_entropy_floor": float(args.imta_route_entropy_floor),
            "imta_route_entropy_weight": float(args.imta_route_entropy_weight),
            "imta_route_balance_weight": float(args.imta_route_balance_weight),
            "repeat_unlikelihood_weight": float(args.repeat_unlikelihood_weight),
            "premature_stop_loss_weight": float(args.premature_stop_loss_weight),
            "response_start_loss_weight": float(args.response_start_loss_weight),
            "response_start_stop_margin_weight": float(args.response_start_stop_margin_weight),
            "response_start_stop_margin": float(args.response_start_stop_margin),
            "response_continue_stop_margin_weight": float(args.response_continue_stop_margin_weight),
            "response_continue_stop_margin": float(args.response_continue_stop_margin),
            "response_continue_stop_margin_start_after": int(continue_schedule["start_after"]),
            "response_continue_stop_margin_warmup_steps": int(continue_schedule["warmup_steps"]),
            "response_continue_stop_margin_start_fraction": float(continue_schedule["start_fraction"]),
            "response_continue_stop_margin_warmup_fraction": float(continue_schedule["warmup_fraction"]),
            "response_body_loss_weight": float(args.response_body_loss_weight),
            "response_stop_loss_weight": float(args.response_stop_loss_weight),
            "response_stop_loss_start_after": int(stop_schedule["start_after"]),
            "response_stop_loss_warmup_steps": int(stop_schedule["warmup_steps"]),
            "response_stop_loss_start_fraction": float(stop_schedule["start_fraction"]),
            "response_stop_loss_warmup_fraction": float(stop_schedule["warmup_fraction"]),
            "token_maturation_steps": int(args.token_maturation_steps),
            "token_maturation_layers": int(args.token_maturation_layers),
            "token_maturation_aux_loss_weight": float(args.token_maturation_aux_loss_weight),
            "token_maturation_gate_init": float(args.token_maturation_gate_init),
            "token_maturation_confidence_threshold": float(args.token_maturation_confidence_threshold),
            "answer_memory": bool(args.answer_memory),
            "answer_memory_steps": int(args.answer_memory_steps),
            "answer_memory_plan_tokens": int(args.answer_memory_plan_tokens),
            "answer_memory_plan_layers": int(args.answer_memory_plan_layers),
            "answer_memory_prompt_context": bool(args.answer_memory_prompt_context),
            "answer_memory_prompt_context_gate_init": float(args.answer_memory_prompt_context_gate_init),
            "answer_memory_prompt_context_default_scale": float(args.answer_memory_prompt_context_default_scale),
            "answer_memory_prompt_context_start_after": int(prompt_context_schedule["start_after"]),
            "answer_memory_prompt_context_warmup_steps": int(prompt_context_schedule["warmup_steps"]),
            "answer_memory_prompt_context_start_fraction": float(prompt_context_schedule["start_fraction"]),
            "answer_memory_prompt_context_warmup_fraction": float(prompt_context_schedule["warmup_fraction"]),
            "answer_memory_aux_loss_weight": float(args.answer_memory_aux_loss_weight),
            "answer_memory_confidence_gate": bool(args.answer_memory_confidence_gate),
            "answer_memory_confidence_mode": str(args.answer_memory_confidence_mode),
            "answer_memory_confidence_topk": int(args.answer_memory_confidence_topk),
            "answer_memory_confidence_floor": float(args.answer_memory_confidence_floor),
            "answer_memory_stop_margin_loss_weight": float(args.answer_memory_stop_margin_loss_weight),
            "answer_memory_stop_margin": float(args.answer_memory_stop_margin),
            "answer_memory_commitment_scale": float(args.answer_memory_commitment_scale),
            "answer_memory_commitment_confidence_gate": bool(args.answer_memory_commitment_confidence_gate),
            "answer_memory_commitment_gate_init": float(args.answer_memory_commitment_gate_init),
            "answer_prefix_commitment_loss_weight": float(args.answer_prefix_commitment_loss_weight),
            "answer_memory_commitment_start_after": int(commitment_schedule["start_after"]),
            "answer_memory_commitment_warmup_steps": int(commitment_schedule["warmup_steps"]),
            "answer_memory_commitment_start_fraction": float(commitment_schedule["start_fraction"]),
            "answer_memory_commitment_warmup_fraction": float(commitment_schedule["warmup_fraction"]),
            "answer_memory_update_gate_init": float(args.answer_memory_update_gate_init),
            "answer_memory_injection_gate_init": float(args.answer_memory_injection_gate_init),
            "answer_memory_default_injection_scale": float(args.answer_memory_default_injection_scale),
            "answer_memory_injection_start_after": int(memory_schedule["start_after"]),
            "answer_memory_injection_warmup_steps": int(memory_schedule["warmup_steps"]),
            "answer_memory_injection_start_fraction": float(memory_schedule["start_fraction"]),
            "answer_memory_injection_warmup_fraction": float(memory_schedule["warmup_fraction"]),
            "adaptive_latent_bridge": bool(args.adaptive_latent_bridge),
            "adaptive_latent_bridge_gate_init": float(args.adaptive_latent_bridge_gate_init),
            "init_from_blt_checkpoint": str(args.init_from_blt_checkpoint),
            "init_from_v2_checkpoint": str(args.init_from_v2_checkpoint),
            "byte_residual_gate_init": float(args.byte_residual_gate_init),
            "latent_residual_gate_init": float(args.latent_residual_gate_init),
            "stability_activation_clip_value": float(args.stability_activation_clip_value),
            "self_rollout_loss_weight": float(args.self_rollout_loss_weight),
            "self_rollout_max_tokens": int(args.self_rollout_max_tokens),
            "self_rollout_start_after": int(args.self_rollout_start_after),
            "balanced_response_sampler": bool(args.balanced_response_sampler),
            "response_body_sampling_boost": float(args.response_body_sampling_boost),
            "response_first_token_balance_power": float(args.response_first_token_balance_power),
            "force_fixed_boundaries": bool(args.force_fixed_boundaries),
            "evaluation_policy": "free_generation_only",
            "generation_repetition_penalty": float(args.generation_repetition_penalty),
            "official_gdn2_force_chunk_eval": not bool(args.official_gdn2_fused_recurrent_eval),
        },
        "train_command": shell_join(train_command),
        "eval_command": shell_join(eval_command),
    }


def assert_not_duplicate(args: argparse.Namespace) -> None:
    out_dir = run_dir(args)
    manifest = out_dir / "fastlane_manifest.json"
    if manifest.exists() and not bool(args.force):
        raise FileExistsError(f"fastlane run already exists at {manifest}; pass --force to overwrite")


def write_manifest(plan: dict[str, Any]) -> None:
    out_dir = Path(str(plan["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fastlane_manifest.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_command(command: str) -> None:
    subprocess.run(command, shell=True, check=True, cwd=str(ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-data", required=True)
    parser.add_argument("--out-root", default="/mnt/nvme0n1p2/tmp")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use tiny torch-smoke settings; never promotion-ready.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--eval-max-rows", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17035)
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="warmup_cosine")
    parser.add_argument("--optimizer-warmup-steps", type=int, default=-1)
    parser.add_argument("--optimizer-warmup-fraction", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--tensorboard-root", default="/tmp/wgram_eval")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--aim-repo", default="/tmp/wgram_aim")
    parser.add_argument("--aim-experiment", default="wgram_v2_fastlane")
    parser.add_argument("--disable-aim", action="store_true")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--max-response-position-embeddings", type=int, default=1024)
    parser.add_argument("--tie-input-output-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--response-phase-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--core-layers", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--smoke-d-model", type=int, default=16)
    parser.add_argument("--smoke-local-heads", type=int, default=4)
    parser.add_argument("--smoke-core-layers", type=int, default=1)
    parser.add_argument("--smoke-local-layers", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dynamic-boundary-threshold", type=float, default=0.6)
    parser.add_argument("--boundary-initial-logit", type=float, default=-2.0)
    parser.add_argument("--think-steps", type=int, default=4)
    parser.add_argument("--imta-noise-std", type=float, default=0.01)
    parser.add_argument("--imta-selector-temperature", type=float, default=0.8)
    parser.add_argument("--imta-adapter-gate-init", type=float, default=-1.0)
    parser.add_argument("--imta-post-adapter-gate-init", type=float, default=-1.0)
    parser.add_argument("--imta-selector-route-query-std", type=float, default=0.02)
    parser.add_argument("--imta-diversity-weight", type=float, default=0.03)
    parser.add_argument("--imta-route-min-probability", type=float, default=0.05)
    parser.add_argument("--imta-route-entropy-floor", type=float, default=0.35)
    parser.add_argument("--imta-route-entropy-weight", type=float, default=0.02)
    parser.add_argument("--imta-route-balance-weight", type=float, default=0.005)
    parser.add_argument("--own-latent-prediction-weight", type=float, default=0.02)
    parser.add_argument("--repeat-unlikelihood-weight", type=float, default=0.0)
    parser.add_argument("--premature-stop-loss-weight", type=float, default=0.05)
    parser.add_argument("--response-start-loss-weight", type=float, default=0.5)
    parser.add_argument("--response-start-stop-margin-weight", type=float, default=0.2)
    parser.add_argument("--response-start-stop-margin", type=float, default=1.0)
    parser.add_argument("--response-continue-stop-margin-weight", type=float, default=0.0)
    parser.add_argument("--response-continue-stop-margin", type=float, default=1.0)
    parser.add_argument("--response-continue-stop-margin-start-after", type=int, default=-1)
    parser.add_argument("--response-continue-stop-margin-warmup-steps", type=int, default=-1)
    parser.add_argument("--response-continue-stop-margin-start-fraction", type=float, default=0.65)
    parser.add_argument("--response-continue-stop-margin-warmup-fraction", type=float, default=0.35)
    parser.add_argument("--response-body-loss-weight", type=float, default=0.25)
    parser.add_argument("--response-stop-loss-weight", type=float, default=0.15)
    parser.add_argument("--response-stop-loss-start-after", type=int, default=-1)
    parser.add_argument("--response-stop-loss-warmup-steps", type=int, default=-1)
    parser.add_argument("--response-stop-loss-start-fraction", type=float, default=0.65)
    parser.add_argument("--response-stop-loss-warmup-fraction", type=float, default=0.35)
    parser.add_argument("--token-maturation-steps", type=int, default=2)
    parser.add_argument("--token-maturation-layers", type=int, default=1)
    parser.add_argument("--token-maturation-aux-loss-weight", type=float, default=0.05)
    parser.add_argument("--token-maturation-gate-init", type=float, default=-1.0)
    parser.add_argument("--token-maturation-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--answer-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--answer-memory-steps", type=int, default=2)
    parser.add_argument("--answer-memory-plan-tokens", type=int, default=4)
    parser.add_argument("--answer-memory-plan-layers", type=int, default=1)
    parser.add_argument("--answer-memory-prompt-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--answer-memory-prompt-context-gate-init", type=float, default=-1.0)
    parser.add_argument("--answer-memory-prompt-context-default-scale", type=float, default=1.0)
    parser.add_argument("--answer-memory-prompt-context-start-after", type=int, default=-1)
    parser.add_argument("--answer-memory-prompt-context-warmup-steps", type=int, default=-1)
    parser.add_argument("--answer-memory-prompt-context-start-fraction", type=float, default=0.35)
    parser.add_argument("--answer-memory-prompt-context-warmup-fraction", type=float, default=0.30)
    parser.add_argument("--answer-memory-aux-loss-weight", type=float, default=0.15)
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
    parser.add_argument("--answer-prefix-commitment-loss-weight", type=float, default=0.35)
    parser.add_argument("--answer-memory-commitment-start-after", type=int, default=-1)
    parser.add_argument("--answer-memory-commitment-warmup-steps", type=int, default=-1)
    parser.add_argument("--answer-memory-commitment-start-fraction", type=float, default=0.35)
    parser.add_argument("--answer-memory-commitment-warmup-fraction", type=float, default=0.30)
    parser.add_argument("--answer-memory-update-gate-init", type=float, default=-1.0)
    parser.add_argument("--answer-memory-injection-gate-init", type=float, default=-1.5)
    parser.add_argument("--answer-memory-default-injection-scale", type=float, default=1.0)
    parser.add_argument("--answer-memory-injection-start-after", type=int, default=-1)
    parser.add_argument("--answer-memory-injection-warmup-steps", type=int, default=-1)
    parser.add_argument("--answer-memory-injection-start-fraction", type=float, default=0.65)
    parser.add_argument("--answer-memory-injection-warmup-fraction", type=float, default=0.35)
    parser.add_argument("--adaptive-latent-bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-latent-bridge-gate-init", type=float, default=-2.0)
    parser.add_argument("--init-from-blt-checkpoint", default="")
    parser.add_argument("--init-from-v2-checkpoint", default="")
    parser.add_argument("--byte-residual-gate-init", type=float, default=-2.0)
    parser.add_argument("--latent-residual-gate-init", type=float, default=2.0)
    parser.add_argument("--stability-activation-clip-value", type=float, default=30.0)
    parser.add_argument("--self-rollout-loss-weight", type=float, default=0.2)
    parser.add_argument("--self-rollout-max-tokens", type=int, default=8)
    parser.add_argument("--self-rollout-start-after", type=int, default=50)
    parser.add_argument("--balanced-response-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--response-body-sampling-boost", type=float, default=2.0)
    parser.add_argument("--response-first-token-balance-power", type=float, default=0.5)
    parser.add_argument("--force-fixed-boundaries", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--generation-repetition-window", type=int, default=32)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--official-gdn2-no-short-conv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-gdn2-fused-recurrent-eval", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assert_not_duplicate(args)
    plan = build_fastlane_plan(args)
    write_manifest(plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if bool(args.dry_run):
        return
    run_command(str(plan["train_command"]))
    run_command(str(plan["eval_command"]))


if __name__ == "__main__":
    main()
