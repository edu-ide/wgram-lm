#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_CONFIG = "configs/qwen35_2b_4090_source_pointer_l4_lm_bridge_roles12_s080.yaml"
DEFAULT_INIT_CHECKPOINT = (
    "/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_source_position_l3_hard_batch_s240_b8_eval/"
    "accepted_l3_last.pt"
)
DEFAULT_TRAIN_JSONL = "data/filtered/qtrm_source_pointer_l3_hard_train512_s1321.jsonl"
DEFAULT_EVAL_JSONL = "data/eval/qtrm_source_pointer_l3_hard_eval128.jsonl"
DEFAULT_OUT_DIR = (
    "/mnt/nvme1n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_l4_source_pointer_lm_path_s080"
)
FULL_MODE = "qtrm_core_steps_8_no_evidence"
DONOR_MODE = "donor_only_no_evidence"
CORE_OFF_MODE = "qtrm_core_off_no_evidence"
PRIMITIVE_OFF_MODE = "qtrm_core_steps_8_primitive_role_value_off_no_evidence"
SOURCE_SLOT_OFF_MODE = "qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence"
SOURCE_BINDER_OFF_MODE = (
    "qtrm_core_steps_8_core_source_position_binder_off_no_evidence"
)
BRIDGE_OFF_MODE = "qtrm_core_steps_8_role_value_answer_bridge_off_no_evidence"
FINAL_BINDER_OFF_MODE = (
    "qtrm_core_steps_8_core_role_value_answer_final_binder_off_no_evidence"
)
VOCAB_RENDERER_OFF_MODE = (
    "qtrm_core_steps_8_core_role_value_vocab_renderer_off_no_evidence"
)
ANSWER_RECURRENT_OFF_MODE = (
    "qtrm_core_steps_8_answer_state_recurrent_off_no_evidence"
)
ANSWER_HALT_GATE_OFF_MODE = "qtrm_core_steps_8_answer_halt_gate_off_no_evidence"
ANSWER_NEXT_TOKEN_DECODER_OFF_MODE = (
    "qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    by_mode_variant: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        bucket = by_mode.setdefault(mode, {"hits": 0, "exact": 0, "total": 0})
        bucket["hits"] += int(bool(row.get("hit")))
        bucket["exact"] += int(
            bool(row.get("exact_match") or row.get("normalized_exact"))
        )
        bucket["total"] += 1
        variant = str(row.get("hard_variant") or "unknown")
        variant_bucket = by_mode_variant.setdefault(mode, {}).setdefault(
            variant,
            {"hits": 0, "total": 0},
        )
        variant_bucket["hits"] += int(bool(row.get("hit")))
        variant_bucket["total"] += 1
    for mode, bucket in by_mode.items():
        total = max(1, int(bucket["total"]))
        bucket["accuracy"] = float(bucket["hits"]) / float(total)
        bucket["exact_accuracy"] = float(bucket["exact"]) / float(total)
        bucket["by_variant"] = {}
        for variant, variant_bucket in by_mode_variant.get(mode, {}).items():
            variant_total = max(1, int(variant_bucket["total"]))
            bucket["by_variant"][variant] = {
                **variant_bucket,
                "accuracy": float(variant_bucket["hits"]) / float(variant_total),
            }
    return dict(sorted(by_mode.items()))


def mode_accuracy(summary: dict[str, dict[str, Any]], mode: str) -> float:
    return float((summary.get(mode) or {}).get("accuracy", 0.0))


def completion_delta_summary(
    rows: list[dict[str, Any]],
    *,
    reference_mode: str,
    compare_modes: list[str],
) -> dict[str, dict[str, Any]]:
    by_mode_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_mode_id[(str(row.get("mode")), str(row.get("id")))] = row
    reference_ids = {
        case_id for mode, case_id in by_mode_id.keys() if mode == reference_mode
    }
    result: dict[str, dict[str, Any]] = {}
    for mode in compare_modes:
        total = 0
        changed = 0
        full_hit_other_miss = 0
        full_miss_other_hit = 0
        for case_id in sorted(reference_ids):
            ref = by_mode_id.get((reference_mode, case_id))
            other = by_mode_id.get((mode, case_id))
            if ref is None or other is None:
                continue
            total += 1
            if str(ref.get("completion")) == str(other.get("completion")):
                continue
            changed += 1
            if bool(ref.get("hit")) and not bool(other.get("hit")):
                full_hit_other_miss += 1
            if not bool(ref.get("hit")) and bool(other.get("hit")):
                full_miss_other_hit += 1
        result[mode] = {
            "changed": changed,
            "total": total,
            "changed_rate": float(changed) / float(total) if total else 0.0,
            "full_hit_other_miss": full_hit_other_miss,
            "full_miss_other_hit": full_miss_other_hit,
        }
    return result


def build_decision(
    *,
    summary: dict[str, dict[str, Any]],
    min_full_accuracy: float,
    min_donor_margin: float,
    min_core_off_margin: float,
    min_primitive_drop: float,
    min_source_slot_drop: float,
    min_source_binder_drop: float,
    min_bridge_drop: float,
    min_vocab_renderer_drop: float,
    min_answer_recurrent_drop: float,
    min_answer_halt_gate_drop: float,
    min_answer_next_token_decoder_drop: float,
) -> dict[str, Any]:
    full = mode_accuracy(summary, FULL_MODE)
    donor = mode_accuracy(summary, DONOR_MODE)
    core_off = mode_accuracy(summary, CORE_OFF_MODE)
    primitive_off = mode_accuracy(summary, PRIMITIVE_OFF_MODE)
    source_slot_off = mode_accuracy(summary, SOURCE_SLOT_OFF_MODE)
    source_binder_off = mode_accuracy(summary, SOURCE_BINDER_OFF_MODE)
    bridge_off = mode_accuracy(summary, BRIDGE_OFF_MODE)
    final_binder_off = mode_accuracy(summary, FINAL_BINDER_OFF_MODE)
    vocab_renderer_off = mode_accuracy(summary, VOCAB_RENDERER_OFF_MODE)
    answer_recurrent_off = mode_accuracy(summary, ANSWER_RECURRENT_OFF_MODE)
    answer_halt_gate_off = mode_accuracy(summary, ANSWER_HALT_GATE_OFF_MODE)
    answer_next_token_decoder_off = mode_accuracy(
        summary,
        ANSWER_NEXT_TOKEN_DECODER_OFF_MODE,
    )
    reject_reasons: list[str] = []
    if full < float(min_full_accuracy):
        reject_reasons.append("full_generation_accuracy_below_min")
    if full - donor <= float(min_donor_margin):
        reject_reasons.append("full_does_not_beat_donor")
    if full - core_off <= float(min_core_off_margin):
        reject_reasons.append("full_does_not_beat_core_off")
    if full - primitive_off <= float(min_primitive_drop):
        reject_reasons.append("primitive_off_drop_below_min")
    if full - source_slot_off <= float(min_source_slot_drop):
        reject_reasons.append("source_slot_off_drop_below_min")
    if full - source_binder_off <= float(min_source_binder_drop):
        reject_reasons.append("source_binder_off_drop_below_min")
    if full - bridge_off <= float(min_bridge_drop):
        reject_reasons.append("role_value_answer_bridge_drop_below_min")
    if VOCAB_RENDERER_OFF_MODE in summary and full - vocab_renderer_off <= float(
        min_vocab_renderer_drop
    ):
        reject_reasons.append("role_value_vocab_renderer_drop_below_min")
    if ANSWER_RECURRENT_OFF_MODE in summary and full - answer_recurrent_off <= float(
        min_answer_recurrent_drop
    ):
        reject_reasons.append("answer_state_recurrent_drop_below_min")
    if ANSWER_HALT_GATE_OFF_MODE in summary and full - answer_halt_gate_off <= float(
        min_answer_halt_gate_drop
    ):
        reject_reasons.append("answer_halt_gate_drop_below_min")
    if ANSWER_NEXT_TOKEN_DECODER_OFF_MODE in summary and (
        full - answer_next_token_decoder_off
        <= float(min_answer_next_token_decoder_drop)
    ):
        reject_reasons.append("answer_next_token_decoder_drop_below_min")
    accepted = not reject_reasons
    return {
        "decision": "accepted_l4_candidate" if accepted else "rejected_l4_candidate",
        "accepted": accepted,
        "target_level": "L4 canonical LM path candidate",
        "major_bottleneck": "primitive recurrent state to autoregressive LM logits",
        "reject_reasons": reject_reasons,
        "decisive_metrics": {
            "full_generation_accuracy": full,
            "donor_generation_accuracy": donor,
            "core_off_generation_accuracy": core_off,
            "primitive_off_generation_accuracy": primitive_off,
            "source_slot_off_generation_accuracy": source_slot_off,
            "source_binder_off_generation_accuracy": source_binder_off,
            "bridge_off_generation_accuracy": bridge_off,
            "final_binder_off_generation_accuracy": final_binder_off,
            "vocab_renderer_off_generation_accuracy": vocab_renderer_off,
            "answer_recurrent_off_generation_accuracy": answer_recurrent_off,
            "answer_halt_gate_off_generation_accuracy": answer_halt_gate_off,
            "answer_next_token_decoder_off_generation_accuracy": (
                answer_next_token_decoder_off
            ),
            "full_minus_donor": full - donor,
            "full_minus_core_off": full - core_off,
            "full_minus_primitive_off": full - primitive_off,
            "full_minus_source_slot_off": full - source_slot_off,
            "full_minus_source_binder_off": full - source_binder_off,
            "full_minus_bridge_off": full - bridge_off,
            "full_minus_final_binder_off": full - final_binder_off,
            "full_minus_vocab_renderer_off": full - vocab_renderer_off,
            "full_minus_answer_recurrent_off": full - answer_recurrent_off,
            "full_minus_answer_halt_gate_off": full - answer_halt_gate_off,
            "full_minus_answer_next_token_decoder_off": (
                full - answer_next_token_decoder_off
            ),
        },
        "thresholds": {
            "min_full_accuracy": float(min_full_accuracy),
            "min_donor_margin": float(min_donor_margin),
            "min_core_off_margin": float(min_core_off_margin),
            "min_primitive_drop": float(min_primitive_drop),
            "min_source_slot_drop": float(min_source_slot_drop),
            "min_source_binder_drop": float(min_source_binder_drop),
            "min_bridge_drop": float(min_bridge_drop),
            "min_vocab_renderer_drop": float(min_vocab_renderer_drop),
            "min_answer_recurrent_drop": float(min_answer_recurrent_drop),
            "min_answer_halt_gate_drop": float(min_answer_halt_gate_drop),
            "min_answer_next_token_decoder_drop": float(
                min_answer_next_token_decoder_drop
            ),
        },
        "next_action": (
            "broaden to the 128-case standard L4 gate and then mixed-family LM gates"
            if accepted
            else "keep L3 as canonical; inspect full vs primitive/bridge-off generations"
        ),
    }


def train_command(
    args: argparse.Namespace,
    train_dir: Path,
    *,
    steps: int | None = None,
    init_checkpoint: str | Path | None = None,
    seed: int | None = None,
) -> list[str]:
    command = [
        args.python_bin,
        "scripts/196_train_pure_recursive_depth_supervised.py",
        "--config",
        args.config,
        "--data-jsonl",
        args.train_jsonl,
        "--shuffle-rows",
        "--init-checkpoint",
        str(init_checkpoint if init_checkpoint is not None else args.init_checkpoint),
        "--tokenizer-model-id",
        args.tokenizer_model_id,
        "--steps",
        str(int(steps if steps is not None else args.steps)),
        "--lr",
        str(float(args.lr)),
        "--optimizer",
        str(args.optimizer),
        "--max-length",
        str(int(args.max_length)),
        "--target-logit-positions-only",
        "--depth-steps",
        "8",
        "--target-mode",
        "final",
        "--out-dir",
        str(train_dir),
        "--trainable-param-policy",
        str(args.trainable_param_policy),
        "--final-path-only-supervision",
        "--final-logit-ce-weight",
        "1.0",
        "--depth-final-ce-weight",
        "0.0",
        "--all-depth-ce-weight",
        "0.0",
        "--progress-margin-weight",
        "0.0",
        "--final-greedy-token-margin-weight",
        str(float(args.greedy_margin_weight)),
        "--greedy-token-margin",
        str(float(args.greedy_margin)),
        "--core-role-value-vocab-renderer-ce-weight",
        str(float(args.vocab_renderer_ce_weight)),
        "--core-role-value-vocab-renderer-greedy-margin-weight",
        str(float(args.vocab_renderer_greedy_margin_weight)),
        "--core-role-value-vocab-renderer-primitive-contrast-weight",
        str(float(args.vocab_renderer_primitive_contrast_weight)),
        "--core-role-value-vocab-renderer-primitive-contrast-margin",
        str(float(args.vocab_renderer_primitive_contrast_margin)),
        "--core-role-value-vocab-renderer-source-binder-contrast-weight",
        str(float(args.vocab_renderer_source_binder_contrast_weight)),
        "--core-role-value-vocab-renderer-source-binder-contrast-margin",
        str(float(args.vocab_renderer_source_binder_contrast_margin)),
        "--answer-state-loop-halt-ce-weight",
        str(float(args.answer_state_loop_halt_ce_weight)),
        "--answer-state-loop-logit-ce-weight",
        str(float(args.answer_state_loop_logit_ce_weight)),
        "--answer-state-loop-future-token-ce-weight",
        str(float(args.answer_state_loop_future_token_ce_weight)),
        "--answer-state-loop-future-token-max-target-tokens",
        str(int(args.answer_state_loop_future_token_max_target_tokens)),
        "--causal-prefix-supervision",
        "--causal-prefix-max-target-tokens",
        str(int(args.max_target_tokens)),
        "--causal-prefix-later-token-weight",
        str(float(args.later_token_weight)),
    ]
    if bool(args.skip_leading_whitespace_targets):
        command.append("--causal-prefix-skip-leading-whitespace-targets")
    command += [
        "--causal-prefix-self-rollout-weight",
        str(float(args.self_rollout_weight)),
        "--causal-prefix-self-rollout-max-target-tokens",
        str(int(args.max_target_tokens)),
        "--core-role-value-answer-bridge-final-contrast-weight",
        str(float(args.bridge_contrast_weight)),
        "--core-role-value-answer-bridge-final-contrast-margin",
        str(float(args.bridge_contrast_margin)),
        "--core-role-value-answer-bridge-contrast-all-prefix-tokens",
        "--core-primitive-role-value-answer-final-contrast-weight",
        str(float(args.primitive_contrast_weight)),
        "--core-primitive-role-value-answer-final-contrast-margin",
        str(float(args.primitive_contrast_margin)),
        "--core-primitive-role-value-answer-final-contrast-all-prefix-tokens",
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        str(int(args.token_numeric_source_slot_vocab_size)),
        "--token-numeric-source-slot-max-slots",
        str(int(args.token_numeric_source_slot_max_slots)),
        "--token-numeric-source-slot-gate-min",
        str(float(args.token_numeric_source_slot_gate_min)),
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        str(float(args.token_numeric_source_slot_predicate_gate_min)),
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        str(float(args.core_source_position_binder_gate_min)),
        "--core-source-position-binder-state-gate-min",
        str(float(args.core_source_position_binder_state_gate_min)),
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
        "--role-value-list-class-mode",
        "source_position",
        "--save-trainable-only",
        "--log-every",
        str(int(args.log_every)),
        "--seed",
        str(int(seed if seed is not None else args.seed)),
    ]
    return command


def train_process_plan(args: argparse.Namespace, train_dir: Path) -> list[dict[str, Any]]:
    total_steps = int(args.steps)
    chunk_steps = int(args.train_process_chunk_steps)
    if total_steps <= 0:
        raise ValueError("--steps must be positive")
    if chunk_steps <= 0 or chunk_steps >= total_steps:
        checkpoint = train_dir / "last.pt"
        return [
            {
                "index": 0,
                "steps": total_steps,
                "seed": int(args.seed),
                "init_checkpoint": str(args.init_checkpoint),
                "out_dir": str(train_dir),
                "checkpoint": str(checkpoint),
                "command": train_command(args, train_dir),
            }
        ]

    plan: list[dict[str, Any]] = []
    remaining = total_steps
    chunk_index = 0
    init_checkpoint: str | Path = args.init_checkpoint
    while remaining > 0:
        this_steps = min(chunk_steps, remaining)
        chunk_index += 1
        chunk_dir = train_dir / f"chunk_{chunk_index:04d}"
        seed = int(args.seed) + chunk_index - 1
        checkpoint = chunk_dir / "last.pt"
        plan.append(
            {
                "index": chunk_index,
                "steps": this_steps,
                "seed": seed,
                "init_checkpoint": str(init_checkpoint),
                "out_dir": str(chunk_dir),
                "checkpoint": str(checkpoint),
                "command": train_command(
                    args,
                    chunk_dir,
                    steps=this_steps,
                    init_checkpoint=init_checkpoint,
                    seed=seed,
                ),
            }
        )
        init_checkpoint = checkpoint
        remaining -= this_steps
    return plan


def eval_command(args: argparse.Namespace, checkpoint: Path, eval_jsonl: Path) -> list[str]:
    command = [
        args.python_bin,
        "scripts/192_eval_raw_intelligence.py",
        "--config",
        str(args.eval_config or args.config),
        "--checkpoint",
        str(checkpoint),
        "--cases",
        args.eval_jsonl,
        "--out",
        str(eval_jsonl),
        "--max-cases",
        str(int(args.max_eval_cases)),
        "--max-length",
        str(int(args.max_length)),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--scoring",
        "generation",
        "--choice-score-normalization",
        "mean",
        "--suppress-visible-reasoning-tokens",
        "--no-repeat-ngram-size",
        str(int(args.no_repeat_ngram_size)),
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        str(int(args.token_numeric_source_slot_vocab_size)),
        "--token-numeric-source-slot-max-slots",
        str(int(args.token_numeric_source_slot_max_slots)),
        "--token-numeric-source-slot-gate-min",
        str(float(args.token_numeric_source_slot_gate_min)),
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        str(float(args.token_numeric_source_slot_predicate_gate_min)),
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        str(float(args.core_source_position_binder_gate_min)),
        "--core-source-position-binder-state-gate-min",
        str(float(args.core_source_position_binder_state_gate_min)),
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
    ]
    for mode in (
        DONOR_MODE,
        CORE_OFF_MODE,
        FULL_MODE,
        PRIMITIVE_OFF_MODE,
        SOURCE_SLOT_OFF_MODE,
        SOURCE_BINDER_OFF_MODE,
        BRIDGE_OFF_MODE,
        FINAL_BINDER_OFF_MODE,
        VOCAB_RENDERER_OFF_MODE,
        ANSWER_RECURRENT_OFF_MODE,
        ANSWER_HALT_GATE_OFF_MODE,
        ANSWER_NEXT_TOKEN_DECODER_OFF_MODE,
    ):
        command.extend(["--mode", mode])
    return command


def source_copy_probe_command(
    args: argparse.Namespace,
    checkpoint: Path,
    probe_json: Path,
) -> list[str]:
    command = [
        args.python_bin,
        "scripts/328_probe_qtrm_source_position_logits.py",
        "--config",
        str(args.eval_config or args.config),
        "--checkpoint",
        str(checkpoint),
        "--base-checkpoint",
        str(args.init_checkpoint),
        "--cases",
        args.eval_jsonl,
        "--out",
        str(probe_json),
        "--max-cases",
        str(int(args.probe_max_cases)),
        "--max-length",
        str(int(args.max_length)),
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        str(int(args.token_numeric_source_slot_vocab_size)),
        "--token-numeric-source-slot-max-slots",
        str(int(args.token_numeric_source_slot_max_slots)),
        "--token-numeric-source-slot-gate-min",
        str(float(args.token_numeric_source_slot_gate_min)),
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        str(float(args.token_numeric_source_slot_predicate_gate_min)),
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        str(float(args.core_source_position_binder_gate_min)),
        "--core-source-position-binder-state-gate-min",
        str(float(args.core_source_position_binder_state_gate_min)),
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
    ]
    return command


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}.{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home)
    if args.tmpdir:
        env["TMPDIR"] = str(args.tmpdir)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_plan = train_process_plan(args, train_dir)
    report: dict[str, Any] = {
        "target_level": "L4 canonical LM path candidate",
        "major_bottleneck": "primitive recurrent state to autoregressive LM logits",
        "artifacts": {
            "config": args.config,
            "eval_config": str(args.eval_config or args.config),
            "init_checkpoint": args.init_checkpoint,
            "train_jsonl": args.train_jsonl,
            "eval_jsonl": args.eval_jsonl,
            "out_dir": str(out_dir),
            "train_dir": str(train_dir),
        },
        "commands": {
            "train": train_plan[0]["command"] if len(train_plan) == 1 else None,
            "train_chunks": train_plan,
        },
        "training_runtime": {
            "mode": "single_process" if len(train_plan) == 1 else "chunked_process",
            "chunk_steps": int(args.train_process_chunk_steps),
            "total_steps": int(args.steps),
        },
    }
    if bool(args.dry_run):
        report.update({"decision": "dry_run", "accepted": False})
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    report["exit_codes"] = {}
    final_chunk_checkpoint: Path | None = None
    for chunk in train_plan:
        chunk_index = int(chunk["index"])
        stdout_name = (
            "train.stdout.log"
            if len(train_plan) == 1
            else f"train.chunk_{chunk_index:04d}.stdout.log"
        )
        stderr_name = (
            "train.stderr.log"
            if len(train_plan) == 1
            else f"train.chunk_{chunk_index:04d}.stderr.log"
        )
        train_exit = run_command(
            list(chunk["command"]),
            cwd=root,
            env=env,
            stdout_path=out_dir / stdout_name,
            stderr_path=out_dir / stderr_name,
        )
        exit_key = "train" if len(train_plan) == 1 else f"train_chunk_{chunk_index:04d}"
        report["exit_codes"][exit_key] = train_exit
        chunk_checkpoint = Path(str(chunk["checkpoint"]))
        if train_exit != 0 or not chunk_checkpoint.exists():
            report.update(
                {
                    "decision": "train_failed",
                    "accepted": False,
                    "failed_train_chunk": chunk_index,
                }
            )
            (out_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return report
        final_chunk_checkpoint = chunk_checkpoint

    checkpoint = train_dir / "last.pt"
    if final_chunk_checkpoint is None:
        report.update({"decision": "train_failed", "accepted": False})
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    if final_chunk_checkpoint.resolve() != checkpoint.resolve():
        shutil.copy2(final_chunk_checkpoint, checkpoint)

    report["checkpoint"] = str(checkpoint)
    if bool(args.post_train_source_copy_probe):
        probe_json = out_dir / "source_copy_probe.json"
        probe_cmd = source_copy_probe_command(args, checkpoint, probe_json)
        report["commands"]["source_copy_probe"] = probe_cmd
        probe_exit = run_command(
            probe_cmd,
            cwd=root,
            env=env,
            stdout_path=out_dir / "source_copy_probe.stdout.log",
            stderr_path=out_dir / "source_copy_probe.stderr.log",
        )
        report["exit_codes"]["source_copy_probe"] = probe_exit
        report["source_copy_probe_report"] = str(probe_json)
        if not probe_json.exists():
            report.update({"decision": "source_copy_probe_failed", "accepted": False})
            (out_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return report
        probe_report = json.loads(probe_json.read_text(encoding="utf-8"))
        report["source_copy_probe"] = {
            key: value for key, value in probe_report.items() if key != "records"
        }

    if bool(args.skip_generation_eval):
        probe_report = report.get("source_copy_probe")
        if isinstance(probe_report, dict):
            probe_accepted = bool(probe_report.get("accepted"))
            decision = (
                "source_copy_probe_only_accepted"
                if probe_accepted
                else "source_copy_probe_only_rejected"
            )
            next_action = (
                "run the full L4 generation gate"
                if probe_accepted
                else "keep generation eval off; repair source-copy logits first"
            )
        else:
            decision = "trained_generation_eval_skipped"
            next_action = "run source-copy probe or full L4 generation gate next"
        report.update(
            {
                "decision": decision,
                "accepted": False,
                "generation_eval_skipped": True,
                "next_action": next_action,
            }
        )
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    generation_jsonl = out_dir / "generation_eval.jsonl"
    eval_cmd = eval_command(args, checkpoint, generation_jsonl)
    report["commands"]["eval"] = eval_cmd
    eval_exit = run_command(
        eval_cmd,
        cwd=root,
        env=env,
        stdout_path=out_dir / "eval.stdout.log",
        stderr_path=out_dir / "eval.stderr.log",
    )
    report["exit_codes"]["eval"] = eval_exit
    if eval_exit != 0:
        report.update({"decision": "eval_failed", "accepted": False})
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    rows = load_jsonl(generation_jsonl)
    summary = summarize_generation(rows)
    completion_deltas = completion_delta_summary(
        rows,
        reference_mode=FULL_MODE,
        compare_modes=[
            DONOR_MODE,
            CORE_OFF_MODE,
            PRIMITIVE_OFF_MODE,
            SOURCE_SLOT_OFF_MODE,
            SOURCE_BINDER_OFF_MODE,
            BRIDGE_OFF_MODE,
            FINAL_BINDER_OFF_MODE,
            VOCAB_RENDERER_OFF_MODE,
            ANSWER_RECURRENT_OFF_MODE,
            ANSWER_HALT_GATE_OFF_MODE,
            ANSWER_NEXT_TOKEN_DECODER_OFF_MODE,
        ],
    )
    decision = build_decision(
        summary=summary,
        min_full_accuracy=float(args.min_full_accuracy),
        min_donor_margin=float(args.min_donor_margin),
        min_core_off_margin=float(args.min_core_off_margin),
        min_primitive_drop=float(args.min_primitive_drop),
        min_source_slot_drop=float(args.min_source_slot_drop),
        min_source_binder_drop=float(args.min_source_binder_drop),
        min_bridge_drop=float(args.min_bridge_drop),
        min_vocab_renderer_drop=float(args.min_vocab_renderer_drop),
        min_answer_recurrent_drop=float(args.min_answer_recurrent_drop),
        min_answer_halt_gate_drop=float(args.min_answer_halt_gate_drop),
        min_answer_next_token_decoder_drop=float(
            args.min_answer_next_token_decoder_drop
        ),
    )
    report.update(
        {
            **decision,
            "generation_summary": summary,
            "generation_completion_deltas": completion_deltas,
            "generation_jsonl": str(generation_jsonl),
            "checkpoint": str(checkpoint),
        }
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--eval-config",
        default="",
        help=(
            "Optional eval-time config. Use this when a gate should be disabled "
            "during train but enabled for held-out inference, for example "
            "answer_state_loop_halt_gate_enabled."
        ),
    )
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--eval-jsonl", default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--trainable-param-policy",
        default="role_value_answer_bridge_loop_vocab_renderer_only",
        help=(
            "Training policy for the L4 bridge run. Use "
            "role_value_answer_bridge_adapter_only to bottleneck learning "
            "through the state bridge and LM adapter, or "
            "role_value_vocab_renderer_only for the direct state-to-vocab "
            "renderer candidate."
        ),
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument(
        "--train-process-chunk-steps",
        type=int,
        default=0,
        help=(
            "Run training as multiple short subprocesses. Use 1 on 24GB GPUs "
            "when a single L4 process OOMs on step 2. This preserves the "
            "checkpoint chain but resets optimizer state between chunks."
        ),
    )
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-eval-cases", type=int, default=32)
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=1.0,
    )
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=1.0)
    parser.add_argument(
        "--core-source-position-binder-state-gate-min",
        type=float,
        default=0.25,
    )
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--greedy-margin-weight", type=float, default=0.20)
    parser.add_argument("--greedy-margin", type=float, default=0.10)
    parser.add_argument("--vocab-renderer-ce-weight", type=float, default=1.0)
    parser.add_argument("--vocab-renderer-greedy-margin-weight", type=float, default=0.25)
    parser.add_argument("--vocab-renderer-primitive-contrast-weight", type=float, default=0.5)
    parser.add_argument("--vocab-renderer-primitive-contrast-margin", type=float, default=0.05)
    parser.add_argument("--vocab-renderer-source-binder-contrast-weight", type=float, default=0.5)
    parser.add_argument("--vocab-renderer-source-binder-contrast-margin", type=float, default=0.05)
    parser.add_argument("--answer-state-loop-halt-ce-weight", type=float, default=0.0)
    parser.add_argument("--answer-state-loop-logit-ce-weight", type=float, default=0.0)
    parser.add_argument("--answer-state-loop-future-token-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--answer-state-loop-future-token-max-target-tokens",
        type=int,
        default=8,
    )
    parser.add_argument("--later-token-weight", type=float, default=0.85)
    parser.add_argument("--skip-leading-whitespace-targets", action="store_true")
    parser.add_argument(
        "--self-rollout-weight",
        type=float,
        default=0.0,
        help=(
            "Optional on-policy prefix rollout weight. Default is 0 for the "
            "4090 L4 gate because rollout triples the number of forward passes "
            "and can hide the actual causal-path result behind VRAM failure."
        ),
    )
    parser.add_argument("--bridge-contrast-weight", type=float, default=0.25)
    parser.add_argument("--bridge-contrast-margin", type=float, default=0.05)
    parser.add_argument("--primitive-contrast-weight", type=float, default=0.25)
    parser.add_argument("--primitive-contrast-margin", type=float, default=0.05)
    parser.add_argument("--min-full-accuracy", type=float, default=0.20)
    parser.add_argument("--min-donor-margin", type=float, default=0.05)
    parser.add_argument("--min-core-off-margin", type=float, default=0.05)
    parser.add_argument("--min-primitive-drop", type=float, default=0.05)
    parser.add_argument("--min-source-slot-drop", type=float, default=0.05)
    parser.add_argument("--min-source-binder-drop", type=float, default=0.05)
    parser.add_argument("--min-bridge-drop", type=float, default=0.05)
    parser.add_argument("--min-vocab-renderer-drop", type=float, default=0.05)
    parser.add_argument("--min-answer-recurrent-drop", type=float, default=0.05)
    parser.add_argument("--min-answer-halt-gate-drop", type=float, default=0.05)
    parser.add_argument(
        "--min-answer-next-token-decoder-drop",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", "/mnt/nvme1n1p2/hf-cache-qtrm"),
    )
    parser.add_argument(
        "--tmpdir",
        default=os.environ.get("TMPDIR", "/mnt/nvme1n1p2/tmp"),
    )
    parser.add_argument(
        "--post-train-source-copy-probe",
        action="store_true",
        help=(
            "After training, run the narrow source-copy logits probe before "
            "or instead of the expensive generation gate."
        ),
    )
    parser.add_argument(
        "--skip-generation-eval",
        action="store_true",
        help="Stop after training/probe without running the full generation gate.",
    )
    parser.add_argument(
        "--probe-max-cases",
        type=int,
        default=8,
        help="Number of held-out cases for the post-train source-copy probe.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_gate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        0
        if report.get("decision")
        not in {"train_failed", "eval_failed", "source_copy_probe_failed"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
