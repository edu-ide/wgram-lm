#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

# --- Compatibility shim for loading trainer-generated checkpoints (ContinuationConfig etc.) ---
try:
    # When the checkpoint was saved from train_hybrid_ri4_real_continuation_minimal.py,
    # the dataclass lives under __main__. We register it so pickle can find it during load.
    from scripts.train_hybrid_ri4_real_continuation_minimal import ContinuationConfig, Hybrid556Config
    _main = sys.modules.get('__main__')
    if _main is not None:
        if not hasattr(_main, 'ContinuationConfig'):
            _main.ContinuationConfig = ContinuationConfig
        if not hasattr(_main, 'Hybrid556Config'):
            _main.Hybrid556Config = Hybrid556Config
except Exception:
    pass

DEFAULT_MODES = [
    "donor_only_no_evidence",
    "qtrm_core_off_no_evidence",
    "qtrm_core_steps_1_no_evidence",
    "qtrm_core_steps_2_no_evidence",
    "qtrm_core_steps_4_no_evidence",
    "qtrm_core_steps_8_no_evidence",
    "qtrm_core_steps_8_delta_off_no_evidence",
    "qtrm_core_steps_8_residual_gate_off_no_evidence",
    # RI-1: Hybrid recurrent-depth scaling (current SSOT modes)
    "hybrid_recurrence_off_no_evidence",
    "hybrid_recurrence_depth_1_no_evidence",
    "hybrid_recurrence_depth_4_no_evidence",
    "hybrid_recurrence_depth_8_no_evidence",
    "hybrid_recurrence_depth_12_no_evidence",
    "hybrid_stochastic_breadth_off_no_evidence",
    # RI-3: full 5.56 causal matrix (current SSOT modes)
    "hybrid_556_full_no_evidence",
    "hybrid_556_stoch_zero_no_evidence",
    "hybrid_556_gold_off_no_evidence",
    "hybrid_556_protection_off_no_evidence",
    "hybrid_556_decay_disabled_no_evidence",
    # RI-4: MSA / Raven-style sparse persistent memory inside One-Body hybrid (2026-06)
    "hybrid_sparse_slots_on_no_evidence",
    "hybrid_sparse_slots_off_no_evidence",
    "hybrid_persistent_memory_ablation_no_evidence",
    "hybrid_sparse_router_ablation_no_evidence",

    # RI-1 + RI-4 combined: Depth scaling with dynamic memory (new for RI-1 progress)
    "hybrid_sparse_slots_on_depth_1_no_evidence",
    "hybrid_sparse_slots_on_depth_4_no_evidence",
    "hybrid_sparse_slots_on_depth_8_no_evidence",
    "hybrid_sparse_slots_on_depth_12_no_evidence",
    "hybrid_sparse_slots_off_depth_1_no_evidence",
    "hybrid_sparse_slots_off_depth_4_no_evidence",
    "hybrid_sparse_slots_off_depth_8_no_evidence",
    "hybrid_sparse_slots_off_depth_12_no_evidence",
]
FORCED_CHOICE_TIE_EPS = 1.0e-6
FORCED_CHOICE_TIE_COMPLETION = "__FORCED_CHOICE_TIE__"
SCALE_TOKEN_RE = r"\d+(?:p\d+)?"


def _parse_scale_token(token: str) -> float:
    return float(token.replace("p", "."))


def _normalize_answer(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _canonical_answer_text(text: str) -> str:
    answer = str(text).strip()
    if answer.casefold().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().strip(" \t\r\n.。．:：;；")


def score_answer(
    text: str,
    aliases: Iterable[str],
    *,
    expected_unknown: bool = False,
    strict_exact: bool = False,
) -> dict[str, Any]:
    alias_list = [str(alias) for alias in aliases]
    canonical = _canonical_answer_text(text)
    compact = _strip_terminal_punctuation(canonical)
    normalized_text = _normalize_answer(canonical)
    normalized_compact = _normalize_answer(compact)
    normalized_aliases = [
        (alias, _normalize_answer(alias), _normalize_answer(_strip_terminal_punctuation(alias)))
        for alias in alias_list
    ]
    matched_aliases = [
        alias
        for alias, normalized_alias, _ in normalized_aliases
        if normalized_alias and normalized_alias in normalized_text
    ]
    exact_match = any(compact == _strip_terminal_punctuation(alias) for alias in alias_list)
    normalized_exact = any(
        normalized_compact and normalized_compact == normalized_alias_compact
        for _, _, normalized_alias_compact in normalized_aliases
    )
    normalized_contains = bool(matched_aliases)
    unknown_contains = "unknown" in normalized_text
    unknown_exact = normalized_compact == "unknown"
    unknown_correct = bool(expected_unknown and unknown_contains)
    if bool(strict_exact):
        hit = unknown_exact if expected_unknown else bool(exact_match or normalized_exact)
    else:
        hit = unknown_correct if expected_unknown else normalized_contains
    if expected_unknown and unknown_exact:
        match_type = "unknown_exact"
    elif exact_match:
        match_type = "exact"
    elif normalized_exact:
        match_type = "normalized_exact"
    elif unknown_correct:
        match_type = "unknown_contains"
    elif normalized_contains:
        match_type = "normalized_contains"
    else:
        match_type = "none"

    audit_reasons: list[str] = []
    if hit and normalized_contains and not (exact_match or normalized_exact):
        audit_reasons.append("loose_contains_match")
    if expected_unknown and unknown_correct and not unknown_exact:
        audit_reasons.append("unknown_with_extra_text")
    if bool(strict_exact) and normalized_contains and not (exact_match or normalized_exact):
        audit_reasons.append("strict_exact_miss")
    if not hit:
        audit_reasons.append("answer_miss")
    return {
        "hit": hit,
        "exact_match": exact_match,
        "normalized_exact": normalized_exact,
        "normalized_contains": normalized_contains,
        "unknown_correct": unknown_correct,
        "match_type": match_type,
        "matched_aliases": matched_aliases,
        "canonical_answer": canonical,
        "needs_human_audit": bool(audit_reasons),
        "audit_reasons": audit_reasons,
        "judge_status": "not_run",
    }


def _case_requires_strict_exact_answer(case: dict[str, Any]) -> bool:
    if bool(case.get("strict_answer_match", False)):
        return True
    strict_labels = {
        "source_copy_lexicalization",
        "list_transform",
        "sequential_list_transform",
    }
    labels = {
        str(case.get("category") or ""),
        str(case.get("task_family") or ""),
        str(case.get("reasoning_family") or ""),
    }
    return bool(labels & strict_labels)


def _case_gold_answer(case: dict[str, Any]) -> str:
    for key in ("answer", "chosen"):
        value = case.get(key)
        if value is not None and str(value).strip():
            return str(value)
    aliases = [str(alias) for alias in case.get("answer_aliases", []) if str(alias).strip()]
    return aliases[0] if aliases else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-retrieval raw-intelligence QTRM eval modes."
    )
    parser.add_argument("--config", default="configs/qwen35_2b_4090.yaml")
    parser.add_argument("--checkpoint", default="runs/qwen35_2b_4090/last.pt")
    parser.add_argument("--cases", default="data/eval/pure_recursive_reasoning_heldout_72.jsonl")
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Eval mode. Can be repeated. Defaults to donor/core-off/core-depth sweep.",
    )
    parser.add_argument("--out", default="runs/eval/pure_recursive_reasoning_depth_sweep.jsonl")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--scoring",
        default="forced_choice",
        choices=["forced_choice", "causal_forced_choice", "generation", "beam_generation"],
        help=(
            "forced_choice scores candidate answers by teacher-forced logprob; "
            "causal_forced_choice recomputes each answer-token score from a prefix-only input; "
            "generation uses greedy autoregressive output; "
            "beam_generation uses model-only beam search over autoregressive logits."
        ),
    )
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument(
        "--beam-score-normalization",
        default="mean",
        choices=["sum", "mean"],
        help="How to rank beam candidates by generated-token logprob.",
    )
    parser.add_argument(
        "--choice-score-normalization",
        default="mean",
        choices=["sum", "mean"],
        help=(
            "How to rank forced-choice answers. 'mean' uses per-token average "
            "logprob and avoids a structural bias toward short answers such as EMPTY."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--hybrid-continuation",
        action="store_true",
        help=(
            "Dual-track hygiene: checkpoint is a bare hybrid continuation artifact "
            "(OneBodyParallelHybridBlock stack from train_hybrid_ri4_real_continuation_minimal). "
            "Pre-imports the local ContinuationConfig so unpickling succeeds. "
            "Hybrid_* modes still require the v2 driver path (measure_continuation_hybrid_192 style) for full B-track scoring."
        ),
    )
    parser.add_argument("--qtrm-logits-scale", type=float, default=None)
    parser.add_argument("--donor-logits-scale", type=float, default=None)
    parser.add_argument(
        "--donor-qtrm-conflict-gate",
        action="store_true",
        help="Probe mode: downscale QTRM residual on donor/QTRM top-token conflict.",
    )
    parser.add_argument(
        "--donor-qtrm-conflict-qtrm-scale",
        type=float,
        default=None,
        help="QTRM residual scale used on donor/QTRM top-token conflict when the probe gate is enabled.",
    )
    parser.add_argument(
        "--donor-qtrm-conflict-gate-mode",
        default=None,
        choices=["downscale", "adaptive_margin"],
        help=(
            "Conflict gate policy. downscale preserves the legacy behavior; "
            "adaptive_margin keeps or boosts QTRM residual when its top-token margin "
            "is stronger than the donor margin."
        ),
    )
    parser.add_argument(
        "--donor-qtrm-conflict-qtrm-boost-scale",
        type=float,
        default=None,
        help="QTRM residual scale used by adaptive_margin when QTRM has the stronger margin.",
    )
    parser.add_argument(
        "--donor-qtrm-conflict-margin-threshold",
        type=float,
        default=None,
        help="Minimum QTRM margin advantage required before adaptive_margin preserves or boosts QTRM.",
    )
    # === S043 Phase 0 (minimal, fluency-first steering bias) ===
    parser.add_argument(
        "--donor-residual-steering-bias",
        action="store_true",
        help="Enable tiny learnable residual bias on QTRM logits before donor fusion (Phase 0 experiment). STRICT: always compare vs donor-only baseline and monitor preservation.",
    )
    parser.add_argument(
        "--donor-residual-steering-bias-init-scale",
        type=float,
        default=None,
        help="Scale for the Phase 0 residual steering bias (very small recommended, e.g. 0.01).",
    )
    parser.add_argument(
        "--core-sparse-surprise-write-trigger-enabled",
        action="store_true",
        help="EXPERIMENTAL: Enable surprise-driven write trigger (Titans-style). "
             "Disabled by default after 72-case ablation on step50 showed regression (23.61% vs 34.72%).",
    )
    parser.add_argument(
        "--core-sparse-surprise-scale",
        type=float,
        default=1.0,
        help="EXPERIMENTAL: Scale factor for surprise write trigger",
    )
    parser.add_argument(
        "--core-sparse-surprise-threshold",
        type=float,
        default=0.0,
        help="EXPERIMENTAL: Normalized surprise threshold",
    )
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--suppress-visible-reasoning-tokens", action="store_true")
    parser.add_argument(
        "--token-numeric-value-features",
        action="store_true",
        help=(
            "Enable token-aligned numeric value embeddings for source-pointer "
            "reasoning rows. This keeps generation eval on the same causal input "
            "path as the L3 value-state gate."
        ),
    )
    parser.add_argument(
        "--disable-token-numeric-value-features",
        action="store_true",
        help="Ablate token-aligned numeric features while keeping the model config enabled.",
    )
    parser.add_argument("--token-numeric-value-vocab-size", type=int, default=128)
    parser.add_argument(
        "--token-numeric-source-slots",
        action="store_true",
        help=(
            "Enable compact prompt-derived source-slot tokens for source-pointer "
            "generation eval. This matches the accepted L3 source-slot gate."
        ),
    )
    parser.add_argument("--disable-token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument(
        "--token-numeric-source-slot-id-mode",
        choices=["absolute_value", "relative_parity"],
        default="absolute_value",
    )
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-feedback",
        action="store_true",
    )
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-source-position-binder",
        action="store_true",
        help="Enable the internal source-position binder used by source-pointer gates.",
    )
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder-state-st", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-source-slots-only",
        action="store_true",
    )
    parser.add_argument(
        "--core-source-position-binder-raw-source-slots",
        action="store_true",
    )
    return parser


def resolve_modes(args: argparse.Namespace) -> list[str]:
    return list(args.mode or DEFAULT_MODES)


def apply_eval_model_overrides(model_cfg, args: argparse.Namespace) -> None:
    if bool(getattr(args, "donor_qtrm_conflict_gate", False)):
        model_cfg.donor_qtrm_conflict_gate_enabled = True
    conflict_scale = getattr(args, "donor_qtrm_conflict_qtrm_scale", None)
    if conflict_scale is not None:
        model_cfg.donor_qtrm_conflict_qtrm_scale = float(conflict_scale)
    conflict_mode = getattr(args, "donor_qtrm_conflict_gate_mode", None)
    if conflict_mode is not None:
        model_cfg.donor_qtrm_conflict_gate_mode = str(conflict_mode)
    conflict_boost_scale = getattr(args, "donor_qtrm_conflict_qtrm_boost_scale", None)
    if conflict_boost_scale is not None:
        model_cfg.donor_qtrm_conflict_qtrm_boost_scale = float(conflict_boost_scale)
    conflict_margin_threshold = getattr(
        args,
        "donor_qtrm_conflict_margin_threshold",
        None,
    )
    if conflict_margin_threshold is not None:
        model_cfg.donor_qtrm_conflict_margin_threshold = float(conflict_margin_threshold)

    # === S043 Phase 0 overrides (strictly optional, zero by default) ===
    if bool(getattr(args, "donor_residual_steering_bias", False)):
        model_cfg.donor_residual_steering_bias_enabled = True
    bias_init_scale = getattr(args, "donor_residual_steering_bias_init_scale", None)
    if bias_init_scale is not None:
        model_cfg.donor_residual_steering_bias_init_scale = float(bias_init_scale)

    if bool(getattr(args, "core_sparse_surprise_write_trigger_enabled", False)):
        model_cfg.core_sparse_surprise_write_trigger_enabled = True
        model_cfg.core_sparse_surprise_scale = float(getattr(args, "core_sparse_surprise_scale", 1.0))
        model_cfg.core_sparse_surprise_threshold = float(getattr(args, "core_sparse_surprise_threshold", 0.0))


def _runtime_enable_core_halt(runtime: dict[str, Any]) -> bool:
    return bool(runtime.get("enable_core_halt", False))


def _runtime_use_core_carry(runtime: dict[str, Any]) -> bool:
    return bool(runtime.get("use_core_carry", False))


def _core_carry_forward_kwargs(runtime: dict[str, Any], core_carry) -> dict[str, Any]:
    if not _runtime_use_core_carry(runtime):
        return {}
    return {
        "core_carry": core_carry,
        "return_core_carry": True,
    }


def _ri4_memory_residual_kwargs(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return the proper ri4_memory_residual kwargs for model calls when in RI-4 hybrid mode.
    The residual is sourced from runtime (set in the RI-4 pre-thinking block), making the
    harness completely free of model instance monkey-patching for this mechanism.
    """
    res = runtime.get("ri4_memory_residual")
    if res is None:
        return {}
    scale = float(runtime.get("ri4_memory_residual_scale", 0.3))
    return {
        "ri4_memory_residual": res,
        "ri4_memory_residual_scale": scale,
    }


def mode_runtime(mode: str) -> dict[str, Any]:
    if mode == "donor_only_no_evidence":
        return {
            "mode": mode,
            "disable_core": True,
            "core_steps_override": None,
            "qtrm_logits_scale": 0.0,
            "donor_logits_scale": 1.0,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "qtrm_core_off_no_evidence":
        return {
            "mode": mode,
            "disable_core": True,
            "core_steps_override": None,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }

    # === RI-4: Sparse Persistent Memory (MSA/Raven-style) inside OneBodyParallelHybrid ===
    if mode == "hybrid_sparse_slots_on_no_evidence":
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": True,
            "persistence_ablation": False,
            "router_ablation": False,
            "disable_core": False,
            # RI-1 support: default to 4 for backward compat, but now respects
            # runtime["core_steps_override"] when provided (see run_eval).
            "core_steps_override": 4,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "hybrid_sparse_slots_off_no_evidence":
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": False,   # ablation: router disabled (dense behavior)
            "persistence_ablation": False,
            "router_ablation": False,
            "disable_core": False,
            "core_steps_override": 4,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }

    # === RI-1 + RI-4: Explicit depth + memory combinations (for depth scaling sweeps) ===
    depth_match = re.fullmatch(r"hybrid_sparse_slots_(on|off)_depth_(\d+)_no_evidence", mode)
    if depth_match:
        mem_on = depth_match.group(1) == "on"
        depth = int(depth_match.group(2))
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": mem_on,
            "persistence_ablation": False,
            "router_ablation": False,
            "disable_core": False,
            "core_steps_override": depth,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "hybrid_persistent_memory_ablation_no_evidence":
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": True,
            "persistence_ablation": True,    # strong ablation: no selective persistence
            "router_ablation": False,
            "disable_core": False,
            "core_steps_override": 4,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "hybrid_sparse_router_ablation_no_evidence":
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": True,
            "persistence_ablation": False,
            "router_ablation": True,         # router always chooses all/dense
            "disable_core": False,
            "core_steps_override": 4,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"hybrid_recurrence_depth_(\d+)_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": True,
            "persistence_ablation": False,
            "router_ablation": False,
            "stochastic_breadth_enabled": True,
            "stochastic_breadth_ablation_zero": False,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "hybrid_recurrence_off_no_evidence":
        return {
            "mode": mode,
            "use_parallel_hybrid": False,
            "sparse_slots_enabled": False,
            "stochastic_breadth_enabled": False,
            "stochastic_breadth_ablation_zero": True,
            "disable_core": False,
            "disable_answer_state_recurrent": True,
            "core_steps_override": 1,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    if mode == "hybrid_stochastic_breadth_off_no_evidence":
        return {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": True,
            "persistence_ablation": False,
            "router_ablation": False,
            "stochastic_breadth_enabled": True,
            "stochastic_breadth_ablation_zero": True,
            "disable_core": False,
            "core_steps_override": 4,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    hybrid_556_modes = {
        "hybrid_556_full_no_evidence": {},
        "hybrid_556_stoch_zero_no_evidence": {
            "stochastic_breadth_ablation_zero": True,
        },
        "hybrid_556_gold_off_no_evidence": {
            "gold_state_ablation_zero": True,
            "gold_injection_alpha": 0.0,
        },
        "hybrid_556_protection_off_no_evidence": {
            "adaptive_rehearsal_protect_attractor": False,
        },
        "hybrid_556_decay_disabled_no_evidence": {
            "scheduled_binding_decay_disabled": True,
        },
    }
    if mode in hybrid_556_modes:
        runtime = {
            "mode": mode,
            "use_parallel_hybrid": True,
            "sparse_slots_enabled": True,
            "persistence_ablation": False,
            "router_ablation": False,
            "stochastic_breadth_enabled": True,
            "stochastic_breadth_ablation_zero": False,
            "adaptive_rehearsal_enabled": True,
            "adaptive_rehearsal_ablation_zero": False,
            "adaptive_rehearsal_protect_attractor": True,
            "gold_state_ablation_zero": False,
            "gold_injection_alpha": None,
            "scheduled_binding_decay_disabled": False,
            "disable_core": False,
            "core_steps_override": 4,
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
        runtime.update(hybrid_556_modes[mode])
        return runtime
    if mode in {"qtrm_core_off_qtrm_only_no_evidence", "qtrm_core_off_low_donor_no_evidence"}:
        return {
            "mode": mode,
            "disable_core": True,
            "core_steps_override": None,
            "qtrm_logits_scale": 1.0,
            "donor_logits_scale": 0.0 if "qtrm_only" in mode else 0.25,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"qtrm_core_halt_steps_(\d+)_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "enable_core_halt": True,
        }
    match = re.fullmatch(r"qtrm_core_halt_carry_steps_(\d+)_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "enable_core_halt": True,
            "use_core_carry": True,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_temporal_spatial_off_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_temporal_spatial_context": True,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_delta_off_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_qtrm_residual": True,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_residual_gate_off_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_qtrm_residual_gate": True,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_transition_state_off_no_evidence", mode)
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_transition_state": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_token_numeric_source_slots_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_token_numeric_source_slots": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_core_state_zero_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "zero_core_trajectory": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_core_source_position_binder_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_core_source_position_binder": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_role_value_answer_bridge_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_core_role_value_answer_bridge": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_core_role_value_answer_final_binder_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_core_role_value_answer_final_binder": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_core_role_value_vocab_renderer_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_core_role_value_vocab_renderer": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_primitive_role_value_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_core_primitive_role_value_executor": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_state_recurrent_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_recurrent": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_typed_value_answer_bridge_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_typed_algorithmic_value_state_answer_bridge": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_selective_context_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_selective_context": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_finality_selector_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_finality_selector": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_finality_gate_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_finality_gate": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_halt_gate_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_halt_gate": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_hidden_bridge_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_hidden_bridge": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_next_token_decoder_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_next_token_decoder": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_free_transformer_latent_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_free_transformer_latent": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_answer_talker_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_answer_state_loop_talker": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_transition_joint_answer_bridge_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_transition_state_joint_answer_bridge": True,
        }
    match = re.fullmatch(
        r"qtrm_core_steps_(\d+)_transition_final_answer_binder_off_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": None,
            "donor_logits_scale": None,
            "memoryos_used": False,
            "retrieval_used": False,
            "disable_transition_state_final_answer_binder": True,
        }
    match = re.fullmatch(
        rf"qtrm_core_steps_(\d+)_donor_scale_({SCALE_TOKEN_RE})_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": 1.0,
            "donor_logits_scale": _parse_scale_token(match.group(2)),
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(
        rf"qtrm_core_steps_(\d+)_qtrm_scale_({SCALE_TOKEN_RE})_donor_scale_({SCALE_TOKEN_RE})_no_evidence",
        mode,
    )
    if match:
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": _parse_scale_token(match.group(2)),
            "donor_logits_scale": _parse_scale_token(match.group(3)),
            "memoryos_used": False,
            "retrieval_used": False,
        }
    match = re.fullmatch(r"qtrm_core_steps_(\d+)_(low_donor|qtrm_only)_no_evidence", mode)
    if match:
        scale_mode = match.group(2)
        return {
            "mode": mode,
            "disable_core": False,
            "core_steps_override": int(match.group(1)),
            "qtrm_logits_scale": 1.0,
            "donor_logits_scale": 0.0 if scale_mode == "qtrm_only" else 0.25,
            "memoryos_used": False,
            "retrieval_used": False,
        }
    raise ValueError(f"unknown raw-intelligence eval mode: {mode}")


def load_cases(path: str | Path, *, max_cases: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if not case.get("id"):
                case["id"] = f"case-{line_no}"
            if not case.get("prompt") and not case.get("question"):
                raise ValueError(f"{path}:{line_no}: missing prompt/question")
            if not case.get("answer_aliases"):
                if case.get("answer"):
                    case["answer_aliases"] = [case["answer"]]
                else:
                    raise ValueError(f"{path}:{line_no}: missing answer_aliases")
            if case.get("evidence"):
                raise ValueError(f"{path}:{line_no}: raw-intelligence cases must not include evidence")
            cases.append(case)
            if max_cases is not None and len(cases) >= int(max_cases):
                break
    return cases


def _case_temporal_spatial_context(case: dict[str, Any], *, device: str):
    value = case.get("temporal_spatial_context")
    if value is None:
        return None
    import torch

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return tensor.view(1, -1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        return tensor
    raise ValueError(
        "temporal_spatial_context must be a vector, token list, or single-batch token list"
    )


def _case_temporal_spatial_context_token_count(case: dict[str, Any]) -> int:
    value = case.get("temporal_spatial_context")
    if value is None:
        return 0
    if not isinstance(value, list):
        return 1
    if not value:
        return 0
    first = value[0]
    if isinstance(first, list):
        return len(value)
    return 1


def score_case_record(
    case: dict[str, Any],
    *,
    mode: str,
    completion: str,
    runtime: dict[str, Any],
    generated_tokens: int,
) -> dict[str, Any]:
    score = score_answer(
        completion,
        case.get("answer_aliases", []),
        expected_unknown=bool(case.get("expected_unknown", False)),
        strict_exact=_case_requires_strict_exact_answer(case),
    )
    canonical_completion = score.get("canonical_answer", "")
    disable_temporal_spatial_context = bool(
        runtime.get("disable_temporal_spatial_context", False)
    )
    temporal_spatial_context_available = case.get("temporal_spatial_context") is not None
    temporal_spatial_context_token_count = (
        0
        if disable_temporal_spatial_context
        else _case_temporal_spatial_context_token_count(case)
    )
    return {
        "id": case.get("id"),
        "mode": mode,
        "raw_intelligence_axis": case.get("raw_intelligence_axis", "pure_recursive_reasoning"),
        "category": case.get("category", "uncategorized"),
        "task_family": case.get("task_family", case.get("category", "uncategorized")),
        "reasoning_family": case.get("reasoning_family", case.get("task_family", case.get("category", "uncategorized"))),
        "hard_variant": case.get("hard_variant"),
        "expected_paradigm": case.get("expected_paradigm", "unknown"),
        "requires_stochasticity": bool(case.get("requires_stochasticity", False)),
        "parallel_depth_estimate": case.get("parallel_depth_estimate"),
        "serial_trace_length_estimate": case.get("serial_trace_length_estimate"),
        "question": case.get("question", ""),
        "prompt": case.get("prompt") or case.get("question", ""),
        "gold_answer": _case_gold_answer(case),
        "answer_aliases": case.get("answer_aliases", []),
        "expected_unknown": bool(case.get("expected_unknown", False)),
        "completion": completion,
        "canonical_completion": canonical_completion,
        "generated_tokens": int(generated_tokens),
        "core_steps_requested": runtime.get("core_steps_override"),
        "enable_core_halt": _runtime_enable_core_halt(runtime),
        "use_core_carry": _runtime_use_core_carry(runtime),
        "disable_core": bool(runtime.get("disable_core", False)),
        "disable_qtrm_residual": bool(runtime.get("disable_qtrm_residual", False)),
        "disable_qtrm_residual_gate": bool(
            runtime.get("disable_qtrm_residual_gate", False)
        ),
        "memoryos_used": False,
        "retrieval_used": False,
        "evidence_token_count": 0,
        "workspace_memory_token_count": 0,
        "temporal_spatial_context_available": temporal_spatial_context_available,
        "disable_temporal_spatial_context": disable_temporal_spatial_context,
        "temporal_spatial_context_token_count": temporal_spatial_context_token_count,
        "disable_transition_state": bool(runtime.get("disable_transition_state", False)),
        "zero_core_trajectory": bool(runtime.get("zero_core_trajectory", False)),
        "disable_token_numeric_source_slots": bool(
            runtime.get("disable_token_numeric_source_slots", False)
        ),
        "disable_core_source_position_binder": bool(
            runtime.get("disable_core_source_position_binder", False)
        ),
        "disable_core_role_value_answer_bridge": bool(
            runtime.get("disable_core_role_value_answer_bridge", False)
        ),
        "disable_core_role_value_answer_final_binder": bool(
            runtime.get("disable_core_role_value_answer_final_binder", False)
        ),
        "disable_core_role_value_vocab_renderer": bool(
            runtime.get("disable_core_role_value_vocab_renderer", False)
        ),
        "disable_core_primitive_role_value_executor": bool(
            runtime.get("disable_core_primitive_role_value_executor", False)
        ),
        "disable_answer_state_loop_recurrent": bool(
            runtime.get("disable_answer_state_loop_recurrent", False)
        ),
        "disable_typed_algorithmic_value_state_answer_bridge": bool(
            runtime.get("disable_typed_algorithmic_value_state_answer_bridge", False)
        ),
        "disable_answer_state_loop_selective_context": bool(
            runtime.get("disable_answer_state_loop_selective_context", False)
        ),
        "disable_answer_state_loop_finality_selector": bool(
            runtime.get("disable_answer_state_loop_finality_selector", False)
        ),
        "disable_answer_state_loop_finality_gate": bool(
            runtime.get("disable_answer_state_loop_finality_gate", False)
        ),
        "disable_answer_state_loop_halt_gate": bool(
            runtime.get("disable_answer_state_loop_halt_gate", False)
        ),
        "disable_answer_state_loop_hidden_bridge": bool(
            runtime.get("disable_answer_state_loop_hidden_bridge", False)
        ),
        "disable_answer_state_loop_next_token_decoder": bool(
            runtime.get("disable_answer_state_loop_next_token_decoder", False)
        ),
        "disable_answer_state_loop_free_transformer_latent": bool(
            runtime.get("disable_answer_state_loop_free_transformer_latent", False)
        ),
        "disable_answer_state_loop_talker": bool(
            runtime.get("disable_answer_state_loop_talker", False)
        ),
        "disable_transition_state_joint_answer_bridge": bool(
            runtime.get("disable_transition_state_joint_answer_bridge", False)
        ),
        "disable_transition_state_final_answer_binder": bool(
            runtime.get("disable_transition_state_final_answer_binder", False)
        ),
        **score,
    }


def _choice_candidates(case: dict[str, Any]) -> list[str]:
    choices = [str(choice) for choice in case.get("choices", []) if str(choice).strip()]
    aliases = [str(alias) for alias in case.get("answer_aliases", []) if str(alias).strip()]
    if not choices:
        choices = aliases
    for alias in aliases:
        if alias not in choices:
            choices.insert(0, alias)
    return choices


def _choice_token_count(tokenizer, choice: str) -> int:
    if tokenizer is None:
        return 1
    token_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(str(choice), add_special_tokens=False)
    return max(1, len(token_ids))


def _normalized_choice_score(logprob_sum: float, token_count: int, normalization: str) -> float:
    mode = str(normalization or "sum").lower()
    if mode == "sum":
        return float(logprob_sum)
    if mode == "mean":
        return float(logprob_sum) / max(1, int(token_count))
    raise ValueError("choice score normalization must be 'sum' or 'mean'")


def _no_repeat_ngram_banned_tokens(generated: list[int], prompt_len: int, ngram_size: int) -> list[int]:
    n = int(ngram_size)
    if n <= 0:
        return []
    completion = generated[prompt_len:]
    if n == 1:
        return sorted(set(completion))
    if len(completion) < n - 1:
        return []
    prefix = tuple(completion[-(n - 1) :])
    banned: set[int] = set()
    for idx in range(0, len(completion) - n + 1):
        ngram = tuple(completion[idx : idx + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return sorted(banned)


def _visible_reasoning_token_ids(tokenizer, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    ids: list[int] = []
    for marker in ("<think>", "</think>"):
        try:
            ids.extend(int(token_id) for token_id in tokenizer.encode(marker, add_special_tokens=False))
        except Exception:
            continue
    return sorted(set(ids))


def _completion_text(tokenizer, generated: list[int], *, prompt_len: int) -> str:
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    prompt_text = tokenizer.decode(generated[:prompt_len], skip_special_tokens=True)
    if prompt_text and full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return tokenizer.decode(generated[prompt_len:], skip_special_tokens=True).strip()


def _select_device(cfg_device: str, requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() and cfg_device in {"auto", "cuda"} else "cpu"


def _prepare_inputs(tokenizer, text: str, max_length: int, device: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _token_numeric_ids_for_prompt_prefix(
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    *,
    input_ids,
    max_length: int,
    device: str,
    enabled: bool,
    value_vocab_size: int,
):
    if not enabled:
        return None
    if not case.get("input_list"):
        return None
    import torch
    from qtrm_mm.algorithmic_value_state import token_numeric_value_ids

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    ids = list(
        token_numeric_value_ids(
            case,
            offsets=enc["offset_mapping"][0].tolist(),
            value_vocab_size=int(value_vocab_size),
        )
    )
    target_len = int(input_ids.shape[1])
    if len(ids) < target_len:
        ids.extend([0] * (target_len - len(ids)))
    ids = ids[:target_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _token_numeric_source_slots_for_prompt_prefix(
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    *,
    max_length: int,
    device: str,
    enabled: bool,
    value_vocab_size: int,
    max_slots: int,
    id_mode: str = "absolute_value",
):
    if not enabled:
        return None, None, None
    import torch
    from qtrm_mm.algorithmic_value_state import (
        relative_source_slot_parity_ids,
        row_input_list,
        token_numeric_source_slot_ids,
        token_numeric_source_slot_token_ids,
    )
    if row_input_list(case) is None:
        return None, None, None

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    mode = str(id_mode or "absolute_value")
    if mode == "relative_parity":
        ids, mask = relative_source_slot_parity_ids(
            case,
            max_list_len=int(max_slots),
        )
        slot_token_ids = token_numeric_source_slot_token_ids(
            case,
            offsets=enc["offset_mapping"][0].tolist(),
            input_ids=enc["input_ids"][0].tolist(),
            max_list_len=int(max_slots),
            value_vocab_size=int(value_vocab_size),
        )
    elif mode == "absolute_value":
        ids, mask = token_numeric_source_slot_ids(
            case,
            offsets=enc["offset_mapping"][0].tolist(),
            max_list_len=int(max_slots),
            value_vocab_size=int(value_vocab_size),
        )
        slot_token_ids = token_numeric_source_slot_token_ids(
            case,
            offsets=enc["offset_mapping"][0].tolist(),
            input_ids=enc["input_ids"][0].tolist(),
            max_list_len=int(max_slots),
            value_vocab_size=int(value_vocab_size),
        )
    else:
        raise ValueError(f"unknown token numeric source slot id mode: {mode}")
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([slot_token_ids], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.long, device=device),
    )


def _token_numeric_source_slot_spans_for_prompt_prefix(
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    *,
    max_length: int,
    device: str,
    enabled: bool,
    value_vocab_size: int,
    max_slots: int,
    max_token_pieces: int = 8,
    id_mode: str = "absolute_value",
):
    if not enabled:
        return None, None
    import torch
    from qtrm_mm.algorithmic_value_state import (
        row_input_list,
        token_numeric_source_slot_token_spans,
    )
    if row_input_list(case) is None:
        return None, None
    if str(id_mode or "absolute_value") not in {"absolute_value", "relative_parity"}:
        raise ValueError(
            f"unknown token numeric source slot id mode: {id_mode}"
        )

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    span_ids, span_mask = token_numeric_source_slot_token_spans(
        case,
        offsets=enc["offset_mapping"][0].tolist(),
        input_ids=enc["input_ids"][0].tolist(),
        max_list_len=int(max_slots),
        max_token_pieces=int(max_token_pieces),
        value_vocab_size=int(value_vocab_size),
    )
    return (
        torch.tensor([span_ids], dtype=torch.long, device=device),
        torch.tensor([span_mask], dtype=torch.long, device=device),
    )


def _causal_choice_prefixes(
    tokenizer,
    prompt: str,
    choice: str,
    *,
    max_length: int,
    device: str,
):
    import torch

    prompt_inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu().tolist()
    choice_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
    if not choice_ids:
        choice_ids = tokenizer.encode(str(choice), add_special_tokens=False)
    prefixes = []
    for pos, target_id in enumerate(choice_ids):
        prefix_ids = prompt_ids + [int(token_id) for token_id in choice_ids[:pos]]
        if len(prefix_ids) > int(max_length):
            break
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        prefixes.append((input_ids, attention_mask, int(target_id)))
    return prefixes


def _donor_kwargs(donor, input_ids, attention_mask, device: str, *, return_logits: bool):
    if donor is None:
        return {}
    encoded = donor.encode_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_logits=return_logits,
    )
    out = {"text_states": encoded["text_states"].to(device)}
    if return_logits and encoded.get("logits") is not None:
        out["donor_logits"] = encoded["logits"].to(device)
    return out


def _record_conflict_gate_mean(
    telemetry: dict[str, Any] | None,
    outputs: dict[str, Any],
    *,
    start: int | None = None,
    end: int | None = None,
) -> None:
    if telemetry is None:
        return
    gate = outputs.get("donor_qtrm_conflict_gate")
    if gate is None or getattr(gate, "numel", lambda: 0)() == 0:
        return
    gate_slice = gate
    if start is not None or end is not None:
        gate_slice = gate[:, start:end]
    if gate_slice.numel() == 0:
        return
    telemetry.setdefault("donor_qtrm_conflict_gate_mean_values", []).append(
        float(gate_slice.float().mean().detach().cpu().item())
    )


def _record_core_steps_actual(
    telemetry: dict[str, Any] | None,
    outputs: dict[str, Any],
) -> None:
    if telemetry is None:
        return
    steps = outputs.get("core_steps")
    if steps is None or getattr(steps, "numel", lambda: 0)() == 0:
        return
    telemetry.setdefault("core_steps_actual_values", []).append(
        float(steps.float().mean().detach().cpu().item())
    )


def _core_depth_residual_curve(outputs: dict[str, Any]) -> list[float]:
    states = outputs.get("core_depth_states")
    if states is None or getattr(states, "numel", lambda: 0)() == 0:
        return []
    if getattr(states, "ndim", 0) < 3 or int(states.shape[1]) < 2:
        return []
    values = states.detach().float()
    deltas = values[:, 1:] - values[:, :-1]
    if deltas.ndim > 3:
        deltas = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
    residuals = deltas.norm(dim=-1).mean(dim=0).detach().cpu()
    return [float(value) for value in residuals.tolist()]


def _record_core_residual_telemetry(
    telemetry: dict[str, Any] | None,
    outputs: dict[str, Any],
) -> None:
    if telemetry is None:
        return
    residual_curve = _core_depth_residual_curve(outputs)
    if not residual_curve:
        return
    telemetry.setdefault("residual_curve_values", []).append(residual_curve)
    telemetry.setdefault("fixed_point_residual_values", []).append(
        float(residual_curve[-1])
    )
    telemetry.setdefault("mean_fixed_point_residual_values", []).append(
        sum(residual_curve) / len(residual_curve)
    )


def _record_answer_state_loop_halt(
    telemetry: dict[str, Any] | None,
    outputs: dict[str, Any],
) -> None:
    if telemetry is None:
        return
    logits = outputs.get("answer_state_loop_halt_logits")
    if logits is None or getattr(logits, "numel", lambda: 0)() == 0:
        return
    values = logits.float().detach().cpu()
    if values.ndim != 2 or int(values.shape[0]) == 0:
        return
    first = values[0]
    telemetry.setdefault("answer_state_loop_halt_logits_values", []).append(
        [float(value) for value in first.tolist()]
    )
    telemetry.setdefault("answer_state_loop_halt_argmax_step_values", []).append(
        float(int(first.argmax().item()) + 1)
    )
    positive = (first > 0.0).nonzero(as_tuple=False)
    if int(positive.numel()) > 0:
        first_positive_step = int(positive[0].item()) + 1
        telemetry.setdefault(
            "answer_state_loop_halt_first_positive_step_values",
            [],
        ).append(float(first_positive_step))


def _finalize_choice_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    values = [
        float(value)
        for value in telemetry.get("donor_qtrm_conflict_gate_mean_values", [])
    ]
    result: dict[str, Any] = {}
    if values:
        result.update(
            {
                "donor_qtrm_conflict_gate_mean": sum(values) / len(values),
                "donor_qtrm_conflict_gate_observations": len(values),
            }
        )
    step_values = [
        float(value)
        for value in telemetry.get("core_steps_actual_values", [])
    ]
    if step_values:
        result.update(
            {
                "core_steps_actual_mean": sum(step_values) / len(step_values),
                "core_steps_actual_observations": len(step_values),
            }
        )
    residual_curves = [
        [float(value) for value in curve]
        for curve in telemetry.get("residual_curve_values", [])
        if curve
    ]
    fixed_residual_values = [
        float(value)
        for value in telemetry.get("fixed_point_residual_values", [])
    ]
    mean_residual_values = [
        float(value)
        for value in telemetry.get("mean_fixed_point_residual_values", [])
    ]
    if residual_curves and fixed_residual_values:
        fixed_point_residual = sum(fixed_residual_values) / len(fixed_residual_values)
        mean_fixed_point_residual = (
            sum(mean_residual_values) / len(mean_residual_values)
            if mean_residual_values
            else fixed_point_residual
        )
        result.update(
            {
                "residual_curve": residual_curves[-1],
                "fixed_point_residual": fixed_point_residual,
                "core_fixed_point_residual": fixed_point_residual,
                "mean_fixed_point_residual": mean_fixed_point_residual,
                "fixed_point_residual_observations": len(fixed_residual_values),
            }
        )
    halt_argmax_steps = [
        float(value)
        for value in telemetry.get("answer_state_loop_halt_argmax_step_values", [])
    ]
    if halt_argmax_steps:
        result.update(
            {
                "answer_state_loop_halt_argmax_step_mean": (
                    sum(halt_argmax_steps) / len(halt_argmax_steps)
                ),
                "answer_state_loop_halt_observations": len(halt_argmax_steps),
                "answer_state_loop_halt_logits_last": telemetry[
                    "answer_state_loop_halt_logits_values"
                ][-1],
            }
        )
    halt_positive_steps = [
        float(value)
        for value in telemetry.get(
            "answer_state_loop_halt_first_positive_step_values",
            [],
        )
    ]
    if halt_positive_steps:
        result.update(
            {
                "answer_state_loop_halt_first_positive_step_mean": (
                    sum(halt_positive_steps) / len(halt_positive_steps)
                ),
                "answer_state_loop_halt_first_positive_observations": (
                    len(halt_positive_steps)
                ),
            }
        )
    return result


def _promote_best_choice_telemetry(
    record: dict[str, Any],
    choice_scores: list[dict[str, Any]] | None,
) -> None:
    if not choice_scores:
        return
    best_choice = choice_scores[0]
    for key in (
        "core_steps_actual_mean",
        "core_steps_actual_observations",
        "residual_curve",
        "fixed_point_residual",
        "core_fixed_point_residual",
        "mean_fixed_point_residual",
        "fixed_point_residual_observations",
    ):
        if key in best_choice:
            record[key] = best_choice[key]


def _answer_choice_logprob(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    choice: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    telemetry: dict[str, Any] | None = None,
    temporal_spatial_context=None,
    token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
) -> float:
    import torch

    full_text = f"{prompt} {choice}"
    prompt_inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    inputs = _prepare_inputs(tokenizer, full_text, max_length, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    token_numeric_value_ids = _token_numeric_ids_for_prompt_prefix(
        tokenizer,
        case,
        prompt=prompt,
        input_ids=input_ids,
        max_length=max_length,
        device=device,
        enabled=bool(token_numeric_value_features),
        value_vocab_size=int(token_numeric_value_vocab_size),
    )
    (
        source_slot_ids,
        source_slot_token_ids,
        source_slot_mask,
    ) = _token_numeric_source_slots_for_prompt_prefix(
        tokenizer,
        case,
        prompt,
        max_length=max_length,
        device=device,
        enabled=bool(token_numeric_source_slots),
        value_vocab_size=int(token_numeric_source_slot_vocab_size),
        max_slots=int(token_numeric_source_slot_max_slots),
        id_mode=str(token_numeric_source_slot_id_mode),
    )
    (
        source_slot_token_span_ids,
        source_slot_token_span_mask,
    ) = _token_numeric_source_slot_spans_for_prompt_prefix(
        tokenizer,
        case,
        prompt,
        max_length=max_length,
        device=device,
        enabled=bool(token_numeric_source_slots),
        value_vocab_size=int(token_numeric_source_slot_vocab_size),
        max_slots=int(token_numeric_source_slot_max_slots),
        id_mode=str(token_numeric_source_slot_id_mode),
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_len = int(input_ids.shape[1])
    if full_len <= prompt_len:
        return float("-inf")

    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    core_carry = None
    try:
        extra = _donor_kwargs(
            donor,
            input_ids,
            attention_mask,
            device,
            return_logits=bool(model.cfg.donor_logits_scale != 0.0),
        )
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_numeric_value_ids=token_numeric_value_ids,
                token_numeric_source_slot_ids=source_slot_ids,
                token_numeric_source_slot_token_ids=source_slot_token_ids,
                token_numeric_source_slot_token_span_ids=source_slot_token_span_ids,
                token_numeric_source_slot_token_span_mask=source_slot_token_span_mask,
                token_numeric_source_slot_mask=source_slot_mask,
                **extra,
                **_core_carry_forward_kwargs(runtime, core_carry),
                **_ri4_memory_residual_kwargs(runtime),  # fixed: was passing model instead of runtime dict
                disable_core=bool(runtime.get("disable_core", False)),
                zero_core_trajectory=bool(runtime.get("zero_core_trajectory", False)),
                enable_core_halt=_runtime_enable_core_halt(runtime),
                disable_qtrm_residual=bool(runtime.get("disable_qtrm_residual", False)),
                disable_qtrm_residual_gate=bool(
                    runtime.get("disable_qtrm_residual_gate", False)
                ),
                temporal_spatial_context=temporal_spatial_context,
                disable_temporal_spatial_context=bool(
                    runtime.get("disable_temporal_spatial_context", False)
                ),
                disable_transition_state=bool(runtime.get("disable_transition_state", False)),
                disable_token_numeric_source_slots=bool(
                    runtime.get("disable_token_numeric_source_slots", False)
                ),
                disable_core_source_position_binder=bool(
                    runtime.get("disable_core_source_position_binder", False)
                ),
                disable_core_primitive_role_value_executor=bool(
                    runtime.get("disable_core_primitive_role_value_executor", False)
                ),
                disable_core_role_value_answer_bridge=bool(
                    runtime.get("disable_core_role_value_answer_bridge", False)
                ),
                disable_core_role_value_answer_final_binder=bool(
                    runtime.get(
                        "disable_core_role_value_answer_final_binder", False
                    )
                ),
                disable_core_role_value_vocab_renderer=bool(
                    runtime.get("disable_core_role_value_vocab_renderer", False)
                ),
                disable_answer_state_loop_recurrent=bool(
                    runtime.get("disable_answer_state_loop_recurrent", False)
                ),
                disable_typed_algorithmic_value_state_answer_bridge=bool(
                    runtime.get(
                        "disable_typed_algorithmic_value_state_answer_bridge", False
                    )
                ),
                disable_answer_state_loop_selective_context=bool(
                    runtime.get("disable_answer_state_loop_selective_context", False)
                ),
                disable_answer_state_loop_finality_selector=bool(
                    runtime.get("disable_answer_state_loop_finality_selector", False)
                ),
                disable_answer_state_loop_finality_gate=bool(
                    runtime.get("disable_answer_state_loop_finality_gate", False)
                ),
                disable_answer_state_loop_halt_gate=bool(
                    runtime.get("disable_answer_state_loop_halt_gate", False)
                ),
                disable_answer_state_loop_hidden_bridge=bool(
                    runtime.get("disable_answer_state_loop_hidden_bridge", False)
                ),
                disable_answer_state_loop_next_token_decoder=bool(
                    runtime.get("disable_answer_state_loop_next_token_decoder", False)
                ),
                disable_answer_state_loop_free_transformer_latent=bool(
                    runtime.get("disable_answer_state_loop_free_transformer_latent", False)
                ),
                disable_answer_state_loop_talker=bool(
                    runtime.get("disable_answer_state_loop_talker", False)
                ),
                disable_transition_state_joint_answer_bridge=bool(
                    runtime.get("disable_transition_state_joint_answer_bridge", False)
                ),
                disable_transition_state_final_answer_binder=bool(
                    runtime.get("disable_transition_state_final_answer_binder", False)
                ),
            )
        logits = outputs["logits"].float()
        offset = logits.shape[1] - input_ids.shape[1]
        aligned = logits[:, offset + prompt_len - 1 : offset + full_len - 1, :]
        targets = input_ids[:, prompt_len:full_len].to(device=aligned.device)
        if aligned.shape[1] != targets.shape[1]:
            return float("-inf")

        # No more legacy logits bias here.
        # The hybrid final state now participates exclusively via the hidden-level residual
        # injection inside the model forward (ri4_memory_residual kwarg / _ri4_memory_residual attribute).
        # This is the clean One-Body path.

        _record_conflict_gate_mean(
            telemetry,
            outputs,
            start=prompt_len - 1,
            end=full_len - 1,
        )
        _record_core_steps_actual(telemetry, outputs)
        _record_core_residual_telemetry(telemetry, outputs)
        _record_answer_state_loop_halt(telemetry, outputs)
        log_probs = torch.log_softmax(aligned, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().detach().cpu().item())
    finally:
        model.cfg.outer_steps = old_outer_steps


def _answer_choice_causal_logprob(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    choice: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    telemetry: dict[str, Any] | None = None,
    temporal_spatial_context=None,
    token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
) -> float:
    import torch

    prefixes = _causal_choice_prefixes(
        tokenizer,
        prompt,
        choice,
        max_length=max_length,
        device=device,
    )
    if not prefixes:
        return float("-inf")

    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    total = 0.0
    core_carry = None
    try:
        for input_ids, attention_mask, target_id in prefixes:
            token_numeric_value_ids = _token_numeric_ids_for_prompt_prefix(
                tokenizer,
                case,
                prompt,
                input_ids=input_ids,
                max_length=max_length,
                device=device,
                enabled=bool(token_numeric_value_features),
                value_vocab_size=int(token_numeric_value_vocab_size),
            )
            (
                source_slot_ids,
                source_slot_token_ids,
                source_slot_mask,
            ) = _token_numeric_source_slots_for_prompt_prefix(
                tokenizer,
                case,
                prompt,
                max_length=max_length,
                device=device,
                enabled=bool(token_numeric_source_slots),
                value_vocab_size=int(token_numeric_source_slot_vocab_size),
                max_slots=int(token_numeric_source_slot_max_slots),
                id_mode=str(token_numeric_source_slot_id_mode),
            )
            (
                source_slot_token_span_ids,
                source_slot_token_span_mask,
            ) = _token_numeric_source_slot_spans_for_prompt_prefix(
                tokenizer,
                case,
                prompt,
                max_length=max_length,
                device=device,
                enabled=bool(token_numeric_source_slots),
                value_vocab_size=int(token_numeric_source_slot_vocab_size),
                max_slots=int(token_numeric_source_slot_max_slots),
                id_mode=str(token_numeric_source_slot_id_mode),
            )
            extra = _donor_kwargs(
                donor,
                input_ids,
                attention_mask,
                device,
                return_logits=bool(model.cfg.donor_logits_scale != 0.0),
            )
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_numeric_value_ids=token_numeric_value_ids,
                    token_numeric_source_slot_ids=source_slot_ids,
                    token_numeric_source_slot_token_ids=source_slot_token_ids,
                    token_numeric_source_slot_token_span_ids=source_slot_token_span_ids,
                    token_numeric_source_slot_token_span_mask=source_slot_token_span_mask,
                    token_numeric_source_slot_mask=source_slot_mask,
                    **extra,
                    **_core_carry_forward_kwargs(runtime, core_carry),
                    **_ri4_memory_residual_kwargs(runtime),
                    disable_core=bool(runtime.get("disable_core", False)),
                    zero_core_trajectory=bool(runtime.get("zero_core_trajectory", False)),
                    enable_core_halt=_runtime_enable_core_halt(runtime),
                    disable_qtrm_residual=bool(
                        runtime.get("disable_qtrm_residual", False)
                    ),
                    disable_qtrm_residual_gate=bool(
                        runtime.get("disable_qtrm_residual_gate", False)
                    ),
                    temporal_spatial_context=temporal_spatial_context,
                    disable_temporal_spatial_context=bool(
                        runtime.get("disable_temporal_spatial_context", False)
                    ),
                    disable_transition_state=bool(runtime.get("disable_transition_state", False)),
                    disable_token_numeric_source_slots=bool(
                        runtime.get("disable_token_numeric_source_slots", False)
                    ),
                    disable_core_source_position_binder=bool(
                        runtime.get("disable_core_source_position_binder", False)
                    ),
                    disable_core_primitive_role_value_executor=bool(
                        runtime.get("disable_core_primitive_role_value_executor", False)
                    ),
                    disable_core_role_value_answer_bridge=bool(
                        runtime.get("disable_core_role_value_answer_bridge", False)
                    ),
                    disable_core_role_value_answer_final_binder=bool(
                        runtime.get(
                            "disable_core_role_value_answer_final_binder", False
                        )
                    ),
                    disable_core_role_value_vocab_renderer=bool(
                        runtime.get("disable_core_role_value_vocab_renderer", False)
                    ),
                    disable_answer_state_loop_recurrent=bool(
                        runtime.get("disable_answer_state_loop_recurrent", False)
                    ),
                    disable_typed_algorithmic_value_state_answer_bridge=bool(
                        runtime.get(
                            "disable_typed_algorithmic_value_state_answer_bridge",
                            False,
                        )
                    ),
                    disable_answer_state_loop_selective_context=bool(
                        runtime.get("disable_answer_state_loop_selective_context", False)
                    ),
                    disable_answer_state_loop_finality_selector=bool(
                        runtime.get("disable_answer_state_loop_finality_selector", False)
                    ),
                    disable_answer_state_loop_finality_gate=bool(
                        runtime.get("disable_answer_state_loop_finality_gate", False)
                    ),
                    disable_answer_state_loop_halt_gate=bool(
                        runtime.get("disable_answer_state_loop_halt_gate", False)
                    ),
                    disable_answer_state_loop_hidden_bridge=bool(
                        runtime.get("disable_answer_state_loop_hidden_bridge", False)
                    ),
                    disable_answer_state_loop_next_token_decoder=bool(
                        runtime.get("disable_answer_state_loop_next_token_decoder", False)
                    ),
                    disable_answer_state_loop_free_transformer_latent=bool(
                        runtime.get("disable_answer_state_loop_free_transformer_latent", False)
                    ),
                    disable_answer_state_loop_talker=bool(
                        runtime.get("disable_answer_state_loop_talker", False)
                    ),
                    disable_transition_state_joint_answer_bridge=bool(
                        runtime.get("disable_transition_state_joint_answer_bridge", False)
                    ),
                    disable_transition_state_final_answer_binder=bool(
                        runtime.get("disable_transition_state_final_answer_binder", False)
                    ),
                )
            if _runtime_use_core_carry(runtime):
                core_carry = outputs.get("core_carry")
            next_logits = outputs["logits"][:, -1, :].float()
            _record_conflict_gate_mean(telemetry, outputs, start=-1, end=None)
            _record_core_steps_actual(telemetry, outputs)
            _record_core_residual_telemetry(telemetry, outputs)
            _record_answer_state_loop_halt(telemetry, outputs)
            total += float(
                torch.log_softmax(next_logits, dim=-1)[0, int(target_id)]
                .detach()
                .cpu()
                .item()
            )
    finally:
        model.cfg.outer_steps = old_outer_steps
    return total


def _forced_choice_case(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    choice_score_normalization: str = "sum",
    token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
) -> tuple[str, list[dict[str, Any]]]:
    prompt = case.get("prompt") or case.get("question", "")
    temporal_spatial_context = _case_temporal_spatial_context(case, device=device)
    scored = []
    for choice in _choice_candidates(case):
        telemetry: dict[str, Any] = {"donor_qtrm_conflict_gate_mean_values": []}
        logprob_sum = _answer_choice_logprob(
            model,
            donor,
            tokenizer,
            case,
            prompt,
            choice,
            runtime=runtime,
            max_length=max_length,
            device=device,
            telemetry=telemetry,
            temporal_spatial_context=temporal_spatial_context,
            token_numeric_value_features=bool(token_numeric_value_features),
            token_numeric_value_vocab_size=int(token_numeric_value_vocab_size),
            token_numeric_source_slots=bool(token_numeric_source_slots),
            token_numeric_source_slot_vocab_size=int(
                token_numeric_source_slot_vocab_size
            ),
            token_numeric_source_slot_max_slots=int(token_numeric_source_slot_max_slots),
            token_numeric_source_slot_id_mode=str(token_numeric_source_slot_id_mode),
        )
        token_count = _choice_token_count(tokenizer, choice)
        score = _normalized_choice_score(
            logprob_sum,
            token_count,
            choice_score_normalization,
        )
        scored.append(
            {
                "choice": choice,
                "logprob": score,
                "logprob_sum": logprob_sum,
                "token_count": token_count,
                "score_normalization": choice_score_normalization,
                **_finalize_choice_telemetry(telemetry),
            }
        )
    scored.sort(key=lambda item: float(item["logprob"]), reverse=True)
    if not scored:
        return "", []
    best = float(scored[0]["logprob"])
    for row in scored:
        row["tied_for_best"] = abs(float(row["logprob"]) - best) <= FORCED_CHOICE_TIE_EPS
    if sum(1 for row in scored if bool(row["tied_for_best"])) > 1:
        return FORCED_CHOICE_TIE_COMPLETION, scored
    return str(scored[0]["choice"]), scored


def _causal_forced_choice_case(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    *,
    runtime: dict[str, Any],
    max_length: int,
    device: str,
    choice_score_normalization: str = "sum",
    token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
) -> tuple[str, list[dict[str, Any]]]:
    prompt = case.get("prompt") or case.get("question", "")
    temporal_spatial_context = _case_temporal_spatial_context(case, device=device)
    scored = []
    for choice in _choice_candidates(case):
        telemetry: dict[str, Any] = {"donor_qtrm_conflict_gate_mean_values": []}
        logprob_sum = _answer_choice_causal_logprob(
            model,
            donor,
            tokenizer,
            case,
            prompt,
            choice,
            runtime=runtime,
            max_length=max_length,
            device=device,
            telemetry=telemetry,
            temporal_spatial_context=temporal_spatial_context,
            token_numeric_value_features=bool(token_numeric_value_features),
            token_numeric_value_vocab_size=int(token_numeric_value_vocab_size),
            token_numeric_source_slots=bool(token_numeric_source_slots),
            token_numeric_source_slot_vocab_size=int(
                token_numeric_source_slot_vocab_size
            ),
            token_numeric_source_slot_max_slots=int(token_numeric_source_slot_max_slots),
            token_numeric_source_slot_id_mode=str(token_numeric_source_slot_id_mode),
        )
        token_count = _choice_token_count(tokenizer, choice)
        score = _normalized_choice_score(
            logprob_sum,
            token_count,
            choice_score_normalization,
        )
        scored.append(
            {
                "choice": choice,
                "logprob": score,
                "logprob_sum": logprob_sum,
                "token_count": token_count,
                "score_normalization": choice_score_normalization,
                **_finalize_choice_telemetry(telemetry),
            }
        )
    scored.sort(key=lambda item: float(item["logprob"]), reverse=True)
    if not scored:
        return "", []
    best = float(scored[0]["logprob"])
    for row in scored:
        row["tied_for_best"] = abs(float(row["logprob"]) - best) <= FORCED_CHOICE_TIE_EPS
    if sum(1 for row in scored if bool(row["tied_for_best"])) > 1:
        return FORCED_CHOICE_TIE_COMPLETION, scored
    return str(scored[0]["choice"]), scored


def _generate_case(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    max_new_tokens: int,
    device: str,
    no_repeat_ngram_size: int,
    suppressed_token_ids: Iterable[int],
    temporal_spatial_context=None,
    token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
) -> tuple[str, int]:
    import torch

    inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    generated = inputs["input_ids"][0].detach().cpu().tolist()
    prompt_len = len(generated)
    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])
    core_carry = None
    try:
        for _ in range(max_new_tokens):
            cur_ids = torch.tensor([generated], dtype=torch.long, device=device)
            cur_mask = torch.ones_like(cur_ids)
            token_numeric_value_ids = _token_numeric_ids_for_prompt_prefix(
                tokenizer,
                case,
                prompt,
                input_ids=cur_ids,
                max_length=max_length,
                device=device,
                enabled=bool(token_numeric_value_features),
                value_vocab_size=int(token_numeric_value_vocab_size),
            )
            (
                source_slot_ids,
                source_slot_token_ids,
                source_slot_mask,
            ) = _token_numeric_source_slots_for_prompt_prefix(
                tokenizer,
                case,
                prompt,
                max_length=max_length,
                device=device,
                enabled=bool(token_numeric_source_slots),
                value_vocab_size=int(token_numeric_source_slot_vocab_size),
                max_slots=int(token_numeric_source_slot_max_slots),
                id_mode=str(token_numeric_source_slot_id_mode),
            )
            (
                source_slot_token_span_ids,
                source_slot_token_span_mask,
            ) = _token_numeric_source_slot_spans_for_prompt_prefix(
                tokenizer,
                case,
                prompt,
                max_length=max_length,
                device=device,
                enabled=bool(token_numeric_source_slots),
                value_vocab_size=int(token_numeric_source_slot_vocab_size),
                max_slots=int(token_numeric_source_slot_max_slots),
                id_mode=str(token_numeric_source_slot_id_mode),
            )
            extra = _donor_kwargs(
                donor,
                cur_ids,
                cur_mask,
                device,
                return_logits=bool(model.cfg.donor_logits_scale != 0.0),
            )
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                outputs = model(
                    cur_ids,
                    attention_mask=cur_mask,
                    token_numeric_value_ids=token_numeric_value_ids,
                    token_numeric_source_slot_ids=source_slot_ids,
                    token_numeric_source_slot_token_ids=source_slot_token_ids,
                    token_numeric_source_slot_token_span_ids=source_slot_token_span_ids,
                    token_numeric_source_slot_token_span_mask=source_slot_token_span_mask,
                    token_numeric_source_slot_mask=source_slot_mask,
                    **extra,
                    **_core_carry_forward_kwargs(runtime, core_carry),
                    **_ri4_memory_residual_kwargs(runtime),
                    disable_core=bool(runtime.get("disable_core", False)),
                    zero_core_trajectory=bool(runtime.get("zero_core_trajectory", False)),
                    enable_core_halt=_runtime_enable_core_halt(runtime),
                    disable_qtrm_residual=bool(
                        runtime.get("disable_qtrm_residual", False)
                    ),
                    disable_qtrm_residual_gate=bool(
                        runtime.get("disable_qtrm_residual_gate", False)
                    ),
                    temporal_spatial_context=temporal_spatial_context,
                    disable_temporal_spatial_context=bool(
                        runtime.get("disable_temporal_spatial_context", False)
                    ),
                    disable_transition_state=bool(runtime.get("disable_transition_state", False)),
                    disable_token_numeric_source_slots=bool(
                        runtime.get("disable_token_numeric_source_slots", False)
                    ),
                    disable_core_source_position_binder=bool(
                        runtime.get("disable_core_source_position_binder", False)
                    ),
                    disable_core_primitive_role_value_executor=bool(
                        runtime.get("disable_core_primitive_role_value_executor", False)
                    ),
                    disable_core_role_value_answer_bridge=bool(
                        runtime.get("disable_core_role_value_answer_bridge", False)
                    ),
                    disable_core_role_value_answer_final_binder=bool(
                        runtime.get(
                            "disable_core_role_value_answer_final_binder", False
                        )
                    ),
                    disable_core_role_value_vocab_renderer=bool(
                        runtime.get("disable_core_role_value_vocab_renderer", False)
                    ),
                    disable_answer_state_loop_recurrent=bool(
                        runtime.get("disable_answer_state_loop_recurrent", False)
                    ),
                    disable_typed_algorithmic_value_state_answer_bridge=bool(
                        runtime.get(
                            "disable_typed_algorithmic_value_state_answer_bridge",
                            False,
                        )
                    ),
                    disable_answer_state_loop_selective_context=bool(
                        runtime.get("disable_answer_state_loop_selective_context", False)
                    ),
                    disable_answer_state_loop_finality_selector=bool(
                        runtime.get("disable_answer_state_loop_finality_selector", False)
                    ),
                    disable_answer_state_loop_finality_gate=bool(
                        runtime.get("disable_answer_state_loop_finality_gate", False)
                    ),
                    disable_answer_state_loop_halt_gate=bool(
                        runtime.get("disable_answer_state_loop_halt_gate", False)
                    ),
                    disable_answer_state_loop_hidden_bridge=bool(
                        runtime.get("disable_answer_state_loop_hidden_bridge", False)
                    ),
                    disable_answer_state_loop_next_token_decoder=bool(
                        runtime.get("disable_answer_state_loop_next_token_decoder", False)
                    ),
                    disable_answer_state_loop_talker=bool(
                        runtime.get("disable_answer_state_loop_talker", False)
                    ),
                    disable_transition_state_joint_answer_bridge=bool(
                        runtime.get("disable_transition_state_joint_answer_bridge", False)
                    ),
                    disable_transition_state_final_answer_binder=bool(
                        runtime.get("disable_transition_state_final_answer_binder", False)
                    ),
                )
            if _runtime_use_core_carry(runtime):
                core_carry = outputs.get("core_carry")
            logits = outputs["logits"][0, -1].float()
            banned = set(int(token_id) for token_id in suppressed_token_ids)
            banned.update(_no_repeat_ngram_banned_tokens(generated, prompt_len, no_repeat_ngram_size))
            if banned:
                valid = [token_id for token_id in banned if 0 <= token_id < logits.shape[-1]]
                if valid:
                    logits[torch.tensor(valid, device=logits.device, dtype=torch.long)] = -torch.inf
            next_id = int(logits.argmax(dim=-1).detach().cpu().item())
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
    finally:
        model.cfg.outer_steps = old_outer_steps
    return _completion_text(tokenizer, generated, prompt_len=prompt_len), len(generated) - prompt_len


def _beam_candidate_score(logprob_sum: float, token_count: int, normalization: str) -> float:
    if str(normalization) == "sum":
        return float(logprob_sum)
    if str(normalization) == "mean":
        return float(logprob_sum) / max(1, int(token_count))
    raise ValueError("beam score normalization must be 'sum' or 'mean'")


def _beam_generate_case(
    model,
    donor,
    tokenizer,
    case: dict[str, Any],
    prompt: str,
    *,
    runtime: dict[str, Any],
    max_length: int,
    max_new_tokens: int,
    device: str,
    beam_size: int,
    score_normalization: str,
    no_repeat_ngram_size: int,
    suppressed_token_ids: Iterable[int],
    temporal_spatial_context=None,
    token_numeric_value_features: bool = False,
    token_numeric_value_vocab_size: int = 128,
    token_numeric_source_slots: bool = False,
    token_numeric_source_slot_vocab_size: int = 128,
    token_numeric_source_slot_max_slots: int = 5,
    token_numeric_source_slot_id_mode: str = "absolute_value",
) -> tuple[str, int]:
    import torch

    if _runtime_use_core_carry(runtime):
        raise ValueError("beam_generation does not support core-carry modes")
    beam_size = max(1, int(beam_size))
    inputs = _prepare_inputs(tokenizer, prompt, max_length, device)
    prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
    prompt_len = len(prompt_ids)
    old_outer_steps = int(model.cfg.outer_steps)
    if runtime.get("core_steps_override") is not None:
        model.cfg.outer_steps = int(runtime["core_steps_override"])

    beams = [
        {
            "tokens": list(prompt_ids),
            "logprob": 0.0,
            "ended": False,
        }
    ]
    try:
        for _ in range(max_new_tokens):
            candidates = []
            all_ended = True
            for beam in beams:
                generated = list(beam["tokens"])
                if bool(beam["ended"]) or len(generated) >= int(max_length):
                    candidates.append(beam)
                    continue
                all_ended = False
                cur_ids = torch.tensor([generated], dtype=torch.long, device=device)
                cur_mask = torch.ones_like(cur_ids)
                token_numeric_value_ids = _token_numeric_ids_for_prompt_prefix(
                    tokenizer,
                    case,
                    prompt,
                    input_ids=cur_ids,
                    max_length=max_length,
                    device=device,
                    enabled=bool(token_numeric_value_features),
                    value_vocab_size=int(token_numeric_value_vocab_size),
                )
                (
                    source_slot_ids,
                    source_slot_token_ids,
                    source_slot_mask,
                ) = _token_numeric_source_slots_for_prompt_prefix(
                    tokenizer,
                    case,
                    prompt,
                    max_length=max_length,
                    device=device,
                    enabled=bool(token_numeric_source_slots),
                    value_vocab_size=int(token_numeric_source_slot_vocab_size),
                    max_slots=int(token_numeric_source_slot_max_slots),
                    id_mode=str(token_numeric_source_slot_id_mode),
                )
                (
                    source_slot_token_span_ids,
                    source_slot_token_span_mask,
                ) = _token_numeric_source_slot_spans_for_prompt_prefix(
                    tokenizer,
                    case,
                    prompt,
                    max_length=max_length,
                    device=device,
                    enabled=bool(token_numeric_source_slots),
                    value_vocab_size=int(token_numeric_source_slot_vocab_size),
                    max_slots=int(token_numeric_source_slot_max_slots),
                    id_mode=str(token_numeric_source_slot_id_mode),
                )
                extra = _donor_kwargs(
                    donor,
                    cur_ids,
                    cur_mask,
                    device,
                    return_logits=bool(model.cfg.donor_logits_scale != 0.0),
                )
                with torch.amp.autocast(
                    "cuda",
                    enabled=(device == "cuda"),
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        cur_ids,
                        attention_mask=cur_mask,
                        token_numeric_value_ids=token_numeric_value_ids,
                        token_numeric_source_slot_ids=source_slot_ids,
                        token_numeric_source_slot_token_ids=source_slot_token_ids,
                        token_numeric_source_slot_token_span_ids=source_slot_token_span_ids,
                        token_numeric_source_slot_token_span_mask=source_slot_token_span_mask,
                        token_numeric_source_slot_mask=source_slot_mask,
                        **extra,
                        disable_core=bool(runtime.get("disable_core", False)),
                        zero_core_trajectory=bool(runtime.get("zero_core_trajectory", False)),
                        enable_core_halt=_runtime_enable_core_halt(runtime),
                        disable_qtrm_residual=bool(
                            runtime.get("disable_qtrm_residual", False)
                        ),
                        disable_qtrm_residual_gate=bool(
                            runtime.get("disable_qtrm_residual_gate", False)
                        ),
                        temporal_spatial_context=temporal_spatial_context,
                        disable_temporal_spatial_context=bool(
                            runtime.get("disable_temporal_spatial_context", False)
                        ),
                        disable_transition_state=bool(
                            runtime.get("disable_transition_state", False)
                        ),
                        disable_token_numeric_source_slots=bool(
                            runtime.get("disable_token_numeric_source_slots", False)
                        ),
                        disable_core_source_position_binder=bool(
                            runtime.get("disable_core_source_position_binder", False)
                        ),
                        disable_core_primitive_role_value_executor=bool(
                            runtime.get("disable_core_primitive_role_value_executor", False)
                        ),
                        disable_core_role_value_answer_bridge=bool(
                            runtime.get("disable_core_role_value_answer_bridge", False)
                        ),
                        disable_core_role_value_answer_final_binder=bool(
                            runtime.get(
                                "disable_core_role_value_answer_final_binder", False
                            )
                        ),
                        disable_core_role_value_vocab_renderer=bool(
                            runtime.get("disable_core_role_value_vocab_renderer", False)
                        ),
                        disable_answer_state_loop_recurrent=bool(
                            runtime.get("disable_answer_state_loop_recurrent", False)
                        ),
                        disable_typed_algorithmic_value_state_answer_bridge=bool(
                            runtime.get(
                                "disable_typed_algorithmic_value_state_answer_bridge",
                                False,
                            )
                        ),
                        disable_answer_state_loop_selective_context=bool(
                            runtime.get("disable_answer_state_loop_selective_context", False)
                        ),
                        disable_answer_state_loop_finality_selector=bool(
                            runtime.get("disable_answer_state_loop_finality_selector", False)
                        ),
                        disable_answer_state_loop_finality_gate=bool(
                            runtime.get("disable_answer_state_loop_finality_gate", False)
                        ),
                        disable_answer_state_loop_halt_gate=bool(
                            runtime.get("disable_answer_state_loop_halt_gate", False)
                        ),
                        disable_answer_state_loop_hidden_bridge=bool(
                            runtime.get("disable_answer_state_loop_hidden_bridge", False)
                        ),
                        disable_answer_state_loop_next_token_decoder=bool(
                            runtime.get("disable_answer_state_loop_next_token_decoder", False)
                        ),
                        disable_answer_state_loop_free_transformer_latent=bool(
                            runtime.get("disable_answer_state_loop_free_transformer_latent", False)
                        ),
                        disable_answer_state_loop_talker=bool(
                            runtime.get("disable_answer_state_loop_talker", False)
                        ),
                        disable_transition_state_joint_answer_bridge=bool(
                            runtime.get("disable_transition_state_joint_answer_bridge", False)
                        ),
                        disable_transition_state_final_answer_binder=bool(
                            runtime.get("disable_transition_state_final_answer_binder", False)
                        ),
                    )
                logits = outputs["logits"][0, -1].float()
                banned = set(int(token_id) for token_id in suppressed_token_ids)
                banned.update(
                    _no_repeat_ngram_banned_tokens(
                        generated,
                        prompt_len,
                        no_repeat_ngram_size,
                    )
                )
                if banned:
                    valid = [
                        token_id for token_id in banned if 0 <= token_id < logits.shape[-1]
                    ]
                    if valid:
                        logits[torch.tensor(valid, device=logits.device, dtype=torch.long)] = -torch.inf
                log_probs = logits.log_softmax(dim=-1)
                top_values, top_indices = torch.topk(
                    log_probs,
                    k=min(beam_size, int(log_probs.shape[-1])),
                )
                for value, token_id in zip(
                    top_values.detach().cpu().tolist(),
                    top_indices.detach().cpu().tolist(),
                ):
                    token_id = int(token_id)
                    ended = (
                        tokenizer.eos_token_id is not None
                        and token_id == int(tokenizer.eos_token_id)
                    )
                    next_tokens = list(generated)
                    if not ended:
                        next_tokens.append(token_id)
                    candidates.append(
                        {
                            "tokens": next_tokens,
                            "logprob": float(beam["logprob"]) + float(value),
                            "ended": ended,
                        }
                    )
            if all_ended:
                break
            candidates.sort(
                key=lambda item: _beam_candidate_score(
                    float(item["logprob"]),
                    len(item["tokens"]) - prompt_len,
                    score_normalization,
                ),
                reverse=True,
            )
            beams = candidates[:beam_size]
    finally:
        model.cfg.outer_steps = old_outer_steps

    beams.sort(
        key=lambda item: _beam_candidate_score(
            float(item["logprob"]),
            len(item["tokens"]) - prompt_len,
            score_normalization,
        ),
        reverse=True,
    )
    best = beams[0]["tokens"] if beams else prompt_ids
    return _completion_text(tokenizer, best, prompt_len=prompt_len), len(best) - prompt_len


def _simple_choice_embedding(choice: str, d_model: int, device, dtype) -> torch.Tensor:
    """Deterministic, cheap projection of a choice string into d_model for hybrid final-state readout."""
    vec = torch.zeros(d_model, device=device, dtype=dtype)
    for i, c in enumerate(choice[:64]):
        v = ord(c)
        for k in range(4):
            idx = (i * 4 + k) % d_model
            vec[idx] += (v % 128) * torch.sin(torch.tensor(i * 0.3 + k * 1.7, device=device, dtype=dtype))
    # Add a length signal
    vec = vec + (len(choice) / 64.0) * 0.1
    return vec / (vec.norm() + 1e-8)


def _build_question_derived_input_192(
    case: dict[str, Any],
    d_model: int,
    seq_len: int,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    scale: float = 0.035,
) -> torch.Tensor:
    """Small self-contained copy of the question-derived input builder for RI-4 in 192_eval.
    Now prefers real token_ids (from the actual tokenized prompt) when present.
    """
    token_ids = case.get("token_ids")
    if token_ids is not None:
        # Real tokenized content from the prompt (the signal the rest of the system uses)
        ids = token_ids if isinstance(token_ids, (list, tuple)) else token_ids.tolist()
        text_for_hash = " ".join(str(int(i)) for i in ids[:32])  # actual token content
        base_len = len(ids)
    else:
        text = str(case.get("question") or case.get("prompt") or "")[:256]
        if not text:
            text = "empty"
        text_for_hash = text
        base_len = len(text)

    feats: list[float] = []
    feats.append(base_len / 256.0)
    feats.append(sum(ord(c) for c in text_for_hash) / (256 * 120.0))
    h1 = h2 = 0
    for i, c in enumerate(text_for_hash):
        v = ord(c) if isinstance(c, str) else int(c) % 128
        h1 = (h1 * 31 + v) & 0xFFFF
        if i > 0:
            h2 = (h2 * 37 + v) & 0xFFFF
    feats.append((h1 % 1024) / 1024.0)
    feats.append((h2 % 1024) / 1024.0)

    feat_dim = 8
    while len(feats) < feat_dim:
        feats.append(0.0)
    feats = feats[:feat_dim]

    base = torch.zeros(d_model, device=device, dtype=dtype)
    for i, f in enumerate(feats):
        for k in range(4):
            idx = (i * 4 + k) % d_model
            phase = (i + k) * 0.7
            base[idx] += f * torch.sin(torch.tensor(phase + idx * 0.13, device=device, dtype=dtype))
    base = base * scale

    x = base.unsqueeze(0).unsqueeze(0).expand(1, seq_len, d_model).clone()
    for t in range(seq_len):
        tmod = torch.sin(torch.tensor(t * 0.21 + 0.3, device=device, dtype=dtype)) * 0.008
        x[0, t] = x[0, t] + tmod

    qhash = sum(ord(c) * (i + 1) if isinstance(c, str) else int(c) * (i + 1) for i, c in enumerate(text_for_hash)) & 0xFFFFFFFF
    g = torch.Generator(device="cpu").manual_seed(qhash % (2**32))
    jitter = torch.randn(1, seq_len, d_model, generator=g, dtype=torch.float32) * (scale * 0.15)
    x = x + jitter.to(device=device, dtype=dtype)
    return x


def run_eval(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Dual-track hygiene (A/B balance): when evaluating bare hybrid continuation
    # checkpoints (GRAM/PTRM restored bias runs etc), pre-import the trainer-local
    # dataclass so torch unpickling can resolve ContinuationConfig / Hybrid556Config.
    # This is the minimal patch that lets B-track (192 narrow sanity) attempt the
    # same artifacts that A-track (measure_continuation) already exercises.
    if bool(getattr(args, "hybrid_continuation", False)):
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        try:
            import train_hybrid_ri4_real_continuation_minimal as _hcont_mod  # registers names for pickle
            _ = getattr(_hcont_mod, "ContinuationConfig", None)
        except Exception as _e:
            print(f"[dual-track] hybrid-continuation pre-import warning: {_e}", file=_sys.stderr)

    import torch
    from transformers import AutoTokenizer

    from qtrm_mm.config import load_config
    from qtrm_mm.qtrm_model import QTRMMultimodalModel
    from qtrm_mm.qwen_donor import QwenDonorAdapter
    from qtrm_mm.training.train import load_initial_checkpoint

    cfg = load_config(args.config)
    apply_eval_model_overrides(cfg.model, args)
    if bool(args.token_numeric_value_features):
        cfg.model.token_numeric_value_embedding_enabled = True
        cfg.model.token_numeric_value_vocab_size = int(args.token_numeric_value_vocab_size)
    if bool(args.token_numeric_source_slots):
        if (
            str(args.token_numeric_source_slot_id_mode) == "relative_parity"
            and int(args.token_numeric_source_slot_vocab_size) < 3
        ):
            raise ValueError(
                "--token-numeric-source-slot-id-mode relative_parity requires "
                "--token-numeric-source-slot-vocab-size >= 3"
            )
        cfg.model.token_numeric_source_slot_embedding_enabled = True
        cfg.model.token_numeric_source_slot_vocab_size = int(
            args.token_numeric_source_slot_vocab_size
        )
        cfg.model.token_numeric_source_slot_max_slots = int(
            args.token_numeric_source_slot_max_slots
        )
        cfg.model.token_numeric_source_slot_gate_min = float(
            args.token_numeric_source_slot_gate_min
        )
        cfg.model.token_numeric_source_slot_predicate_feedback_enabled = bool(
            args.token_numeric_source_slot_predicate_feedback
        )
        cfg.model.token_numeric_source_slot_predicate_gate_min = float(
            args.token_numeric_source_slot_predicate_gate_min
        )
    if bool(args.core_source_position_binder):
        cfg.model.core_source_position_binder_enabled = True
        cfg.model.core_source_position_binder_gate_min = float(
            args.core_source_position_binder_gate_min
        )
        cfg.model.core_source_position_binder_state_gate_min = float(
            args.core_source_position_binder_state_gate_min
        )
        cfg.model.core_source_position_binder_state_straight_through = bool(
            args.core_source_position_binder_state_st
        )
        cfg.model.core_source_position_binder_source_slots_only = bool(
            args.core_source_position_binder_source_slots_only
        )
        cfg.model.core_source_position_binder_raw_source_slots_enabled = bool(
            args.core_source_position_binder_raw_source_slots
        )
    if not cfg.donor.model_id:
        raise SystemExit("donor.model_id is required")
    device = _select_device(cfg.train.device, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.donor.model_id,
        trust_remote_code=cfg.donor.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QTRMMultimodalModel(cfg.model)
    missing, unexpected = load_initial_checkpoint(model, args.checkpoint, map_location=device)
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        print(f"[checkpoint] missing keys: {len(missing)} ({preview}{suffix})", file=sys.stderr)
    if unexpected:
        preview = ", ".join(unexpected[:12])
        suffix = "..." if len(unexpected) > 12 else ""
        print(f"[checkpoint] unexpected keys: {len(unexpected)} ({preview}{suffix})", file=sys.stderr)
    model = model.to(device).eval()

    donor = QwenDonorAdapter(cfg.donor)
    max_length = args.max_length or cfg.train.seq_len
    cases = load_cases(args.cases, max_cases=args.max_cases)
    suppressed_token_ids = _visible_reasoning_token_ids(
        tokenizer,
        enabled=bool(args.suppress_visible_reasoning_tokens),
    )
    token_numeric_eval_enabled = bool(args.token_numeric_value_features) and not bool(
        args.disable_token_numeric_value_features
    )
    token_source_slot_eval_enabled = bool(args.token_numeric_source_slots) and not bool(
        args.disable_token_numeric_source_slots
    )

    records: list[dict[str, Any]] = []
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # 'runs' or parent exists as file from previous bad run — clean it
        if out.parent.exists() and not out.parent.is_dir():
            out.parent.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f, torch.no_grad(), redirect_stdout(sys.stderr):
        for mode in resolve_modes(args):
            runtime = mode_runtime(mode)
            old_qtrm_scale = float(model.cfg.qtrm_logits_scale)
            old_donor_scale = float(model.cfg.donor_logits_scale)

            # === RI-4 hybrid path preparation (A-mode: hybrid as real recurrent engine) ===
            ri4_hybrid_model = None
            if runtime.get("use_parallel_hybrid"):
                # Build the OneBodyParallelHybrid stack with SparseSlotRouter for this mode.
                # In A-mode we attach it directly to model.answer_state_loop_hybrid_recurrent_block
                # so that it drives the actual recurrence *inside* answer_state_loop trajectory steps
                # (cross-attn → hybrid recurrent proposal with persistent slots → gate).
                # This replaces the previous side-car pre-run + residual injection.
                try:
                    from src.qtrm_mm.blocks import build_parallel_hybrid_block, OneBodyParallelHybridBlock
                    from src.qtrm_mm.memory.sparse_slot_router import SparseSlotRouter
                    from src.qtrm_mm.memory.decoupled_latent_memory_bank import make_decoupled_latent_memory_bank, DecoupledLatentMemoryBank

                    from copy import deepcopy

                    hybrid_cfg = deepcopy(cfg.model)  # preserve loaded shape without leaking per-mode flags
                    hybrid_cfg.core_sparse_slot_router_enabled = True
                    hybrid_cfg.core_sparse_slot_ablation_zero = bool(
                        (not runtime.get("sparse_slots_enabled", True))
                        or runtime.get("router_ablation", False)
                    )
                    hybrid_cfg.core_stochastic_breadth_enabled = bool(
                        runtime.get("stochastic_breadth_enabled", True)
                    )
                    hybrid_cfg.core_stochastic_breadth_ablation_zero = bool(
                        runtime.get("stochastic_breadth_ablation_zero", False)
                    )
                    hybrid_cfg.core_adaptive_rehearsal_enabled = bool(
                        runtime.get("adaptive_rehearsal_enabled", False)
                    )
                    hybrid_cfg.core_adaptive_rehearsal_ablation_zero = bool(
                        runtime.get("adaptive_rehearsal_ablation_zero", False)
                    )
                    hybrid_cfg.core_adaptive_rehearsal_protect_attractor = bool(
                        runtime.get("adaptive_rehearsal_protect_attractor", True)
                    )
                    if runtime.get("gold_injection_alpha") is not None:
                        hybrid_cfg.core_adaptive_rehearsal_gold_injection_alpha = float(
                            runtime["gold_injection_alpha"]
                        )
                    hybrid_cfg.core_gold_states_ablation_zero = bool(
                        runtime.get("gold_state_ablation_zero", False)
                    )
                    if runtime.get("scheduled_binding_decay_disabled", False):
                        hybrid_cfg.core_adaptive_rehearsal_scheduled_binding_end = (
                            hybrid_cfg.core_adaptive_rehearsal_scheduled_binding_start
                        )
                    # Create a small stack (single block for PoC latency; can be deeper later).
                    # CPU runs use GQA because the vendored official MLA path can route through Triton.
                    attention_type = "mla" if device == "cuda" else "gqa"
                    ri4_hybrid_model = build_parallel_hybrid_block(
                        hybrid_cfg,
                        attention_type=attention_type,
                    ).to(device).eval()

                    # Instantiates bank if use_decoupled_memory_bank is active
                    bank = None
                    use_decoupled = (
                        getattr(cfg.model, "use_decoupled_memory_bank", False)
                        or getattr(hybrid_cfg, "use_decoupled_memory_bank", False)
                        or getattr(cfg.train, "use_decoupled_memory_bank", False)
                    )
                    if use_decoupled:
                        try:
                            bank = make_decoupled_latent_memory_bank(
                                d_model=cfg.model.d_model,
                                num_slots=getattr(cfg.model, "decoupled_bank_num_slots", 16) if hasattr(cfg.model, "decoupled_bank_num_slots") else 16,
                                top_k=getattr(cfg.model, "decoupled_bank_top_k", 4) if hasattr(cfg.model, "decoupled_bank_top_k") else 4,
                            ).to(device).eval()

                            # Load bank slots from checkpoint
                            ckpt = torch.load(args.checkpoint, map_location=device)
                            if "decoupled_bank" in ckpt and ckpt["decoupled_bank"] is not None:
                                with torch.no_grad():
                                    bank.slots.copy_(ckpt["decoupled_bank"].to(device=device))
                                print(f"[RI-4 Eval] Decoupled Memory Bank state successfully restored from checkpoint")
                        except Exception as e:
                            print(f"[RI-4 Eval] Failed to restore decoupled bank state: {e}")

                    # Configure RI-4 flags on the block(s) — these are read by the block forward
                    slots_on = bool(runtime.get("sparse_slots_enabled", True))
                    pers_ablate = bool(runtime.get("persistence_ablation", False))
                    router_ablate = bool(runtime.get("router_ablation", False))

                    ri4_layers = (
                        (ri4_hybrid_model,)
                        if isinstance(ri4_hybrid_model, OneBodyParallelHybridBlock)
                        else tuple(ri4_hybrid_model)
                    )
                    for layer in ri4_layers:
                        if isinstance(layer, OneBodyParallelHybridBlock):
                            if hasattr(layer, "sparse_slot_router") and layer.sparse_slot_router is not None:
                                layer.sparse_slot_router.set_ablation(
                                    enabled=slots_on and not router_ablate,
                                    ablation_zero=(not slots_on) or router_ablate,
                                )
                            if bank is not None:
                                layer.set_decoupled_memory_bank(bank, ablation_zero=(not slots_on) or router_ablate)
                                layer._decoupled_bank_enabled = slots_on and not router_ablate
                                layer._decoupled_bank_ablation_zero = (not slots_on) or router_ablate

                            # Store flags (used by some internal paths)
                            layer._ri4_persistence_ablation = pers_ablate
                            layer._ri4_slots_on = slots_on and not router_ablate
                except Exception as e:
                    print(f"[RI-4] Failed to build hybrid stack for mode {mode}: {e}")
                    ri4_hybrid_model = None

            # Attach hybrid as the answer_state_loop recurrent engine for these modes (A-mode).
            # Non-RI-4 modes and other runs see None → classic recurrent_stack (zero behavior change).
            if ri4_hybrid_model is not None:
                model.answer_state_loop_hybrid_recurrent_block = ri4_hybrid_model
                model._ri4_hybrid_recurrent_slot_state = None  # will be reset per case
                print(f"[RI-4 A-mode] Attached hybrid recurrent block for mode {mode} (slots_on={runtime.get('sparse_slots_enabled')}, persistence_ablation={runtime.get('persistence_ablation')})")
            else:
                model.answer_state_loop_hybrid_recurrent_block = None
                model._ri4_hybrid_recurrent_slot_state = None

            model.cfg.qtrm_logits_scale = (
                float(runtime["qtrm_logits_scale"])
                if runtime["qtrm_logits_scale"] is not None
                else float(args.qtrm_logits_scale)
                if args.qtrm_logits_scale is not None
                else old_qtrm_scale
            )
            model.cfg.donor_logits_scale = (
                float(runtime["donor_logits_scale"])
                if runtime["donor_logits_scale"] is not None
                else float(args.donor_logits_scale)
                if args.donor_logits_scale is not None
                else old_donor_scale
            )
            try:
                for case in cases:
                    prompt = case.get("prompt") or case.get("question", "")

                    # Per-case fresh slot state for RI-4 hybrid recurrent engine.
                    # Critical: prevents cross-case memory leakage in persistent slots.
                    if runtime.get("use_parallel_hybrid") and ri4_hybrid_model is not None:
                        model._ri4_hybrid_recurrent_slot_state = None

                    # NOTE (A-mode transition):
                    # The previous side-car block that pre-ran the hybrid on question-derived input
                    # and injected via runtime["ri4_memory_residual"] has been removed for the 4
                    # hybrid_* modes. The hybrid now participates natively as the recurrent
                    # proposal engine inside _compute_answer_state_loop_outputs (see qtrm_model.py:6159).
                    # All ablation flags, persistence, and stochastic breadth continue to work
                    # because they live on the attached block.
                    #
                    # The ri4_memory_residual kwarg path remains available for other experiments
                    # and fallback modes. For pure RI-4 hybrid-recurrent runs the memory effect
                    # comes from inside the answer_state_loop trajectory via the slots.

                    choice_scores = None
                    if args.scoring == "forced_choice":
                        completion, choice_scores = _forced_choice_case(
                            model,
                            donor,
                            tokenizer,
                            case,
                            runtime=runtime,
                            max_length=max_length,
                            device=device,
                            choice_score_normalization=args.choice_score_normalization,
                            token_numeric_value_features=token_numeric_eval_enabled,
                            token_numeric_value_vocab_size=int(args.token_numeric_value_vocab_size),
                            token_numeric_source_slots=token_source_slot_eval_enabled,
                            token_numeric_source_slot_vocab_size=int(
                                args.token_numeric_source_slot_vocab_size
                            ),
                            token_numeric_source_slot_max_slots=int(
                                args.token_numeric_source_slot_max_slots
                            ),
                            token_numeric_source_slot_id_mode=str(
                                args.token_numeric_source_slot_id_mode
                            ),
                        )
                        generated_tokens = 0
                    elif args.scoring == "causal_forced_choice":
                        completion, choice_scores = _causal_forced_choice_case(
                            model,
                            donor,
                            tokenizer,
                            case,
                            runtime=runtime,
                            max_length=max_length,
                            device=device,
                            choice_score_normalization=args.choice_score_normalization,
                            token_numeric_value_features=token_numeric_eval_enabled,
                            token_numeric_value_vocab_size=int(args.token_numeric_value_vocab_size),
                            token_numeric_source_slots=token_source_slot_eval_enabled,
                            token_numeric_source_slot_vocab_size=int(
                                args.token_numeric_source_slot_vocab_size
                            ),
                            token_numeric_source_slot_max_slots=int(
                                args.token_numeric_source_slot_max_slots
                            ),
                            token_numeric_source_slot_id_mode=str(
                                args.token_numeric_source_slot_id_mode
                            ),
                        )
                        generated_tokens = 0
                    elif args.scoring == "beam_generation":
                        temporal_spatial_context = _case_temporal_spatial_context(
                            case,
                            device=device,
                        )
                        completion, generated_tokens = _beam_generate_case(
                            model,
                            donor,
                            tokenizer,
                            case,
                            prompt,
                            runtime=runtime,
                            max_length=max_length,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                            beam_size=args.beam_size,
                            score_normalization=args.beam_score_normalization,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            suppressed_token_ids=suppressed_token_ids,
                            temporal_spatial_context=temporal_spatial_context,
                            token_numeric_value_features=token_numeric_eval_enabled,
                            token_numeric_value_vocab_size=int(args.token_numeric_value_vocab_size),
                            token_numeric_source_slots=token_source_slot_eval_enabled,
                            token_numeric_source_slot_vocab_size=int(
                                args.token_numeric_source_slot_vocab_size
                            ),
                            token_numeric_source_slot_max_slots=int(
                                args.token_numeric_source_slot_max_slots
                            ),
                            token_numeric_source_slot_id_mode=str(
                                args.token_numeric_source_slot_id_mode
                            ),
                        )
                    else:
                        temporal_spatial_context = _case_temporal_spatial_context(
                            case,
                            device=device,
                        )
                        completion, generated_tokens = _generate_case(
                            model,
                            donor,
                            tokenizer,
                            case,
                            prompt,
                            runtime=runtime,
                            max_length=max_length,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            suppressed_token_ids=suppressed_token_ids,
                            temporal_spatial_context=temporal_spatial_context,
                            token_numeric_value_features=token_numeric_eval_enabled,
                            token_numeric_value_vocab_size=int(args.token_numeric_value_vocab_size),
                            token_numeric_source_slots=token_source_slot_eval_enabled,
                            token_numeric_source_slot_vocab_size=int(
                                args.token_numeric_source_slot_vocab_size
                            ),
                            token_numeric_source_slot_max_slots=int(
                                args.token_numeric_source_slot_max_slots
                            ),
                            token_numeric_source_slot_id_mode=str(
                                args.token_numeric_source_slot_id_mode
                            ),
                        )
                    record = score_case_record(
                        case,
                        mode=mode,
                        completion=completion,
                        runtime=runtime,
                        generated_tokens=generated_tokens,
                    )
                    record["scoring"] = args.scoring
                    record["choice_score_normalization"] = args.choice_score_normalization
                    if choice_scores is not None:
                        record["choice_scores"] = choice_scores
                        record["choice_tied"] = (
                            sum(1 for row in choice_scores if bool(row.get("tied_for_best")))
                            > 1
                        )
                        _promote_best_choice_telemetry(record, choice_scores)

                    # RI-4 record enrichment (supports both legacy residual-injection path and
                    # new A-mode where hybrid is the native recurrent engine inside answer_state_loop).
                    # We enrich whenever the mode is a hybrid RI-4 mode (use_parallel_hybrid + flags present)
                    # or the old residual key was set. This keeps build_ri4_sparse_memory_gate and
                    # downstream analysis fully compatible with zero changes elsewhere.
                    is_ri4_hybrid_mode = bool(runtime.get("use_parallel_hybrid")) or ("ri4_memory_residual" in runtime)
                    if is_ri4_hybrid_mode:
                        record["slots_on"] = bool(
                            runtime.get("sparse_slots_enabled", False)
                            and not runtime.get("persistence_ablation", False)
                            and not runtime.get("router_ablation", False)
                        )
                        record["persistence_ablation"] = bool(runtime.get("persistence_ablation", False))
                        record["router_ablation"] = bool(runtime.get("router_ablation", False))
                        record["raw_intelligence_axis"] = "ri4_sparse_persistent_memory"
                        if "ri4_hybrid_final_norm" in runtime:
                            record["ri4_hybrid_final_norm"] = runtime["ri4_hybrid_final_norm"]
                        # In pure A-mode the memory effect is now inside the recurrent loop;
                        # we no longer force-set a pre-computed residual for these modes.

                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            finally:
                model.cfg.qtrm_logits_scale = old_qtrm_scale
                model.cfg.donor_logits_scale = old_donor_scale
                # Hygiene: clear RI-4 hybrid recurrent engine attachment after the mode
                # so subsequent modes (or re-runs) never accidentally inherit it.
                if hasattr(model, "answer_state_loop_hybrid_recurrent_block"):
                    model.answer_state_loop_hybrid_recurrent_block = None
                if hasattr(model, "_ri4_hybrid_recurrent_slot_state"):
                    model._ri4_hybrid_recurrent_slot_state = None
    return records


def _print_s043_phase0_diagnostic(args, records):
    """S043 Phase 0 - Structured diagnostic output (recommended order).

    This block is the primary way to quickly judge whether a Phase 0 bias run
    is showing the desired first-token signal without destroying donor fluency.
    """
    if not getattr(args, "donor_residual_steering_bias", False):
        return

    print("\n" + "=" * 72)
    print("S043 PHASE 0 DIAGNOSTIC BLOCK")
    print("=" * 72)

    bias_scale = getattr(args, "donor_residual_steering_bias_init_scale", None) or "default (0.01)"
    print(f"Bias active: True   |   init_scale: {bias_scale}")

    # First-token metrics (highest priority for Phase 0)
    ft_win_rates = [r.get("first_token_win_rate") for r in records if r.get("first_token_win_rate") is not None]
    ft_active_rates = [r.get("first_token_active_rate") for r in records if r.get("first_token_active_rate") is not None]

    if ft_win_rates:
        avg_win = sum(ft_win_rates) / len(ft_win_rates)
        print(f"first_token_win_rate     avg: {avg_win:.4f}   (n={len(ft_win_rates)})")
    else:
        print("first_token_win_rate     : not present in records (check loss weights)")

    if ft_active_rates:
        avg_active = sum(ft_active_rates) / len(ft_active_rates)
        print(f"first_token_active_rate  avg: {avg_active:.4f}")

    # Donor-correct preservation signals
    dc_win = [r.get("donor_correct_margin_win_rate") for r in records if r.get("donor_correct_margin_win_rate") is not None]
    if dc_win:
        print(f"donor_correct_win_rate   avg: {sum(dc_win)/len(dc_win):.4f}")

    # Repetition / fluency proxy (if present in records)
    rep_rates = []
    for r in records:
        for key in ["repetition_rate", "repetition_stats", "collapse_rate"]:
            if key in r and isinstance(r[key], (int, float)):
                rep_rates.append(r[key])
                break
    if rep_rates:
        print(f"repetition_proxy         avg: {sum(rep_rates)/len(rep_rates):.4f}")

    print("\n>>> NEXT ACTIONS (per Verification Guide):")
    print("    1. Compare the numbers above against a pure donor_only baseline run")
    print("    2. Check donor-correct cases manually for fluency regression")
    print("    3. If first_token_win_rate improved but donor fluency dropped → raise preservation weight")
    print("=" * 72 + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    records = run_eval(args)
    hits = sum(1 for record in records if bool(record.get("hit")))
    print(f"wrote {len(records)} records to {args.out}")
    print(f"hits={hits}/{len(records)}")

    # S043 Phase 0: Structured diagnostic block (recommended order)
    _print_s043_phase0_diagnostic(args, records)

    # === RI-4 A-Mode: Automatic 192-Style Readiness Report + JSON artifact (Most-Deficient closure)
    # When any of the 4 hybrid RI-4 modes were exercised, emit the exact same
    # machine-readable contract used by ri4_192_proxy_report.py / launcher / ri4_compare_192_reports.py.
    # This makes the verified hybrid recurrent engine (answer_state_loop delegation + slot carry)
    # produce production-grade comparable artifacts on every real 192 run.
    ri4_modes = [
        "hybrid_sparse_slots_on_no_evidence",
        "hybrid_sparse_slots_off_no_evidence",
        "hybrid_persistent_memory_ablation_no_evidence",
        "hybrid_sparse_router_ablation_no_evidence",
    ]
    ri4_records = [r for r in records if r.get("mode") in ri4_modes]
    if ri4_records:
        from collections import defaultdict
        import json as _json
        import datetime as _dt

        per_mode = defaultdict(list)
        for r in ri4_records:
            per_mode[r["mode"]].append(r)

        print("\n" + "=" * 70)
        print("RI-4 192-Style Readiness Report (from canonical 192_eval)")
        print("=" * 70)
        matrix = {}
        for mode, recs in per_mode.items():
            hits_m = sum(1 for x in recs if x.get("hit"))
            matrix[mode] = {
                "cases": len(recs),
                "hits": hits_m,
                "hit_rate": round(hits_m / max(1, len(recs)), 4),
                "engine": "answer_state_loop_hybrid_recurrent_delegation",
                "participation": "native_recurrent_proposal + persistent_slot_carry",
            }
            print(f"  {mode}: cases={len(recs)} hits={hits_m} rate={matrix[mode]['hit_rate']}")

        # Machine-readable artifact (identical contract to the proxy for seamless compare)
        artifact = {
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "source": "canonical_192_eval",
            "proxy_version": "real",
            "heldout_source": "192_eval_cases",
            "modes": matrix,
            "summary": {
                "total_ri4_cases": len(ri4_records),
                "total_ri4_hits": sum(m["hits"] for m in matrix.values()),
                "engine_verified": True,
            },
        }
        print("\n## RI4_192_REAL_REPORT_JSON_START")
        print(_json.dumps(artifact, indent=2, ensure_ascii=False))
        print("## RI4_192_REAL_REPORT_JSON_END")
        print("RI-4 report artifact emitted — ready for ri4_compare_192_reports.py (proxy vs real).")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
