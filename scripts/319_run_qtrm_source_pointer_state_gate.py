#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_CONFIG = "configs/qwen35_2b_4090_list_order_source_pointer_refresh_s300.yaml"
DEFAULT_TRAIN_DATA = "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl"
DEFAULT_EVAL_DATA = "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl"
DEFAULT_DATA_SUMMARY = (
    "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.summary.json"
)
DEFAULT_INIT = (
    "/mnt/nvme1n1p2/qtrm-runs/primitive_field_heads_source_pointer_s120_seed16/"
    "accepted_l2_source_pointer_step_000040.pt"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def python_bin(args: argparse.Namespace) -> str:
    return str(args.python_bin or sys.executable)


def ensure_gate_data(args: argparse.Namespace, *, root: Path) -> dict[str, Any] | None:
    train_path = root / str(args.train_data_jsonl)
    eval_path = root / str(args.eval_data_jsonl)
    if train_path.exists() and eval_path.exists():
        return None
    if bool(args.no_build_data_if_missing):
        missing = [str(path) for path in (train_path, eval_path) if not path.exists()]
        raise FileNotFoundError(f"missing gate data: {missing}")
    import importlib.util

    builder_path = root / "scripts" / "317_build_absolute_ordered_state_gate_data.py"
    spec = importlib.util.spec_from_file_location(
        "absolute_ordered_state_gate_data",
        builder_path,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"failed to load data builder: {builder_path}")
    spec.loader.exec_module(module)
    return module.write_absolute_ordered_state_split(
        train_out=train_path,
        eval_out=eval_path,
        summary_out=root / str(args.data_summary_json),
        train_count=int(args.data_train_count),
        eval_count=int(args.data_eval_count),
        value_modulus=int(args.data_value_modulus),
        list_len=int(args.data_list_len),
        value_vocab_size=int(args.data_value_vocab_size),
    )


def training_command(args: argparse.Namespace, *, train_out_dir: Path) -> list[str]:
    if bool(args.batch_integrated_training):
        command = [
            python_bin(args),
            "scripts/324_train_qtrm_source_pointer_batch.py",
            "--config",
            str(args.config),
            "--data-jsonl",
            str(args.train_data_jsonl),
            "--init-checkpoint",
            str(args.init_checkpoint),
            "--tokenizer-model-id",
            str(args.tokenizer_model_id),
            "--out-dir",
            str(train_out_dir),
            "--steps",
            str(int(args.steps)),
            "--row-batch-size",
            str(int(args.row_batch_size)),
            "--core-steps",
            str(int(args.core_steps)),
            "--lr",
            str(float(args.lr)),
            "--seed",
            str(int(args.seed)),
            "--save-every",
            str(int(args.save_every)),
            "--log-every",
            str(int(args.log_every)),
            "--core-role-value-prompt-ce-weight",
            str(float(args.core_role_value_prompt_ce_weight)),
            "--core-role-value-prompt-target-mode",
            str(args.core_role_value_prompt_target_mode),
            "--core-primitive-role-value-state-ce-weight",
            str(float(args.core_primitive_role_value_state_ce_weight)),
        ]
        if bool(args.token_numeric_source_slots):
            command.extend(
                [
                    "--token-numeric-source-slots",
                    "--token-numeric-source-slot-vocab-size",
                    str(int(args.token_numeric_source_slot_vocab_size)),
                    "--token-numeric-source-slot-max-slots",
                    str(int(args.token_numeric_source_slot_max_slots)),
                    "--token-numeric-source-slot-gate-min",
                    str(float(args.token_numeric_source_slot_gate_min)),
                    "--token-numeric-source-slot-parity-ce-weight",
                    str(float(args.token_numeric_source_slot_parity_ce_weight)),
                ]
            )
            if bool(args.token_numeric_source_slot_predicate_feedback):
                command.extend(
                    [
                        "--token-numeric-source-slot-predicate-feedback",
                        "--token-numeric-source-slot-predicate-gate-min",
                        str(float(args.token_numeric_source_slot_predicate_gate_min)),
                        "--token-numeric-source-slot-predicate-ce-weight",
                        str(float(args.token_numeric_source_slot_predicate_ce_weight)),
                    ]
                )
        if bool(args.core_source_position_binder):
            command.extend(
                [
                    "--core-source-position-binder",
                    "--core-source-position-binder-gate-min",
                    str(float(args.core_source_position_binder_gate_min)),
                    "--core-source-position-binder-state-gate-min",
                    str(float(args.core_source_position_binder_state_gate_min)),
                ]
            )
            if bool(args.core_source_position_binder_state_st):
                command.append("--core-source-position-binder-state-st")
            if bool(args.core_source_position_binder_source_slots_only):
                command.append("--core-source-position-binder-source-slots-only")
            if bool(args.core_source_position_binder_raw_source_slots):
                command.append("--core-source-position-binder-raw-source-slots")
        return command

    command = [
        python_bin(args),
        "scripts/196_train_pure_recursive_depth_supervised.py",
        "--config",
        str(args.config),
        "--data-jsonl",
        str(args.train_data_jsonl),
        "--shuffle-rows",
        "--init-checkpoint",
        str(args.init_checkpoint),
        "--tokenizer-model-id",
        str(args.tokenizer_model_id),
        "--out-dir",
        str(train_out_dir),
        "--steps",
        str(int(args.steps)),
        "--lr",
        str(float(args.lr)),
        "--seed",
        str(int(args.seed)),
        "--depth-steps",
        str(args.depth_steps),
        "--target-mode",
        "staged",
        "--role-value-list-class-mode",
        "source_position",
        "--final-logit-ce-weight",
        "0.0",
        "--depth-final-ce-weight",
        "0.0",
        "--all-depth-ce-weight",
        "0.0",
        "--progress-margin-weight",
        "0.0",
        "--core-role-value-prompt-ce-weight",
        str(float(args.core_role_value_prompt_ce_weight)),
        "--core-role-value-prompt-target-mode",
        str(args.core_role_value_prompt_target_mode),
        "--primitive-transition-operation-ce-weight",
        str(float(args.primitive_transition_operation_ce_weight)),
        "--core-primitive-role-value-state-ce-weight",
        str(float(args.core_primitive_role_value_state_ce_weight)),
        "--core-primitive-role-value-step-margin-weight",
        str(float(args.core_primitive_role_value_step_margin_weight)),
        "--core-primitive-role-value-step-margin",
        str(float(args.core_primitive_role_value_step_margin)),
        "--core-primitive-role-value-trace-margin-weight",
        str(float(args.core_primitive_role_value_trace_margin_weight)),
        "--core-primitive-role-value-trace-margin",
        str(float(args.core_primitive_role_value_trace_margin)),
        "--core-primitive-role-value-pair-trace-contrast-weight",
        str(float(args.core_primitive_role_value_pair_trace_contrast_weight)),
        "--core-primitive-role-value-pair-trace-contrast-margin",
        str(float(args.core_primitive_role_value_pair_trace_contrast_margin)),
        "--core-primitive-role-value-update-gate-bce-weight",
        str(float(args.core_primitive_role_value_update_gate_bce_weight)),
        "--family-repeat",
        str(args.family_repeat),
        "--save-every",
        str(int(args.save_every)),
        "--log-every",
        str(int(args.log_every)),
        "--save-trainable-only",
    ]
    if bool(args.core_source_position_binder):
        command.append("--core-source-position-binder")
        command.extend(
            [
                "--core-source-position-binder-gate-min",
                str(float(args.core_source_position_binder_gate_min)),
                "--core-source-position-binder-state-gate-min",
                str(float(args.core_source_position_binder_state_gate_min)),
            ]
        )
        if bool(args.core_source_position_binder_state_st):
            command.append("--core-source-position-binder-state-st")
        if bool(args.core_source_position_binder_source_slots_only):
            command.append("--core-source-position-binder-source-slots-only")
        if bool(args.core_source_position_binder_raw_source_slots):
            command.append("--core-source-position-binder-raw-source-slots")
        if bool(args.core_source_position_binder_query_state):
            command.extend(
                [
                    "--core-source-position-binder-query-state",
                    "--core-source-position-binder-query-state-gate-min",
                    str(float(args.core_source_position_binder_query_state_gate_min)),
                ]
            )
        if bool(args.core_source_value_binder):
            command.extend(
                [
                    "--core-source-value-binder",
                    "--core-source-value-binder-state-gate-min",
                    str(float(args.core_source_value_binder_state_gate_min)),
                    "--core-source-value-prompt-ce-weight",
                    str(float(args.core_source_value_prompt_ce_weight)),
                ]
            )
            if bool(args.core_source_value_binder_state_st):
                command.append("--core-source-value-binder-state-st")
            if bool(args.core_primitive_role_value_source_value_conditioning):
                command.extend(
                    [
                        "--core-primitive-role-value-source-value-conditioning",
                        "--core-primitive-role-value-source-value-gate-min",
                        str(
                            float(
                                args.core_primitive_role_value_source_value_gate_min
                            )
                        ),
                    ]
                )
    if (
        bool(args.core_source_position_binder)
        and not bool(args.numeric_source_features)
        and not bool(args.token_numeric_value_features)
        and not bool(args.token_numeric_source_slots)
    ):
        command.extend(
            [
                "--trainable-param-policy",
                "prompt_context_binder_primitive_role_value_state_machine",
            ]
        )
    if bool(args.numeric_source_features):
        command.extend(
            [
                "--numeric-source-features",
                "--numeric-source-max-list-len",
                str(int(args.numeric_source_max_list_len)),
                "--numeric-source-value-vocab-size",
                str(int(args.numeric_source_value_vocab_size)),
                "--trainable-param-policy",
                "numeric_projector_primitive_role_value_state_machine",
            ]
        )
    if bool(args.token_numeric_value_features):
        command.extend(
            [
                "--token-numeric-value-features",
                "--token-numeric-value-vocab-size",
                str(int(args.token_numeric_value_vocab_size)),
                "--trainable-param-policy",
                (
                    "token_numeric_context_binder_primitive_role_value_state_machine"
                    if bool(args.core_source_position_binder)
                    else "token_numeric_context_primitive_role_value_state_machine"
                ),
            ]
        )
    if bool(args.token_numeric_source_slots):
        command.extend(
            [
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                str(int(args.token_numeric_source_slot_vocab_size)),
                "--token-numeric-source-slot-max-slots",
                str(int(args.token_numeric_source_slot_max_slots)),
                "--token-numeric-source-slot-gate-min",
                str(float(args.token_numeric_source_slot_gate_min)),
                "--token-numeric-source-slot-parity-ce-weight",
                str(float(args.token_numeric_source_slot_parity_ce_weight)),
                "--trainable-param-policy",
                (
                    "token_numeric_source_slot_context_binder_primitive_role_value_state_machine"
                    if bool(args.core_source_position_binder)
                    else "token_numeric_source_slot_context_primitive_role_value_state_machine"
                ),
            ]
        )
        if bool(args.token_numeric_source_slot_predicate_feedback):
            command.extend(
                [
                    "--token-numeric-source-slot-predicate-feedback",
                    "--token-numeric-source-slot-predicate-gate-min",
                    str(float(args.token_numeric_source_slot_predicate_gate_min)),
                    "--token-numeric-source-slot-predicate-ce-weight",
                    str(float(args.token_numeric_source_slot_predicate_ce_weight)),
                ]
            )
    return command


def eval_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    out_json: Path,
    primitive_off: bool,
    numeric_off: bool,
    token_numeric_off: bool,
    source_slot_off: bool,
    source_binder_off: bool = False,
    strict_prompt_binding_off: bool = False,
) -> list[str]:
    command = [
        python_bin(args),
        "scripts/238_eval_qtrm_algorithmic_value_state.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--data-jsonl",
        str(args.eval_data_jsonl),
        "--out-json",
        str(out_json),
        "--tokenizer-model-id",
        str(args.tokenizer_model_id),
        "--core-steps",
        str(int(args.core_steps)),
        "--max-cases",
        str(int(args.max_eval_cases)),
        "--include-family",
        "list_transform",
        "--use-role-value-state",
        "--use-core-primitive-role-value-state",
        "--role-value-list-class-mode",
        "source_position",
    ]
    if bool(args.numeric_source_features):
        command.extend(
            [
                "--numeric-source-features",
                "--numeric-source-max-list-len",
                str(int(args.numeric_source_max_list_len)),
                "--numeric-source-value-vocab-size",
                str(int(args.numeric_source_value_vocab_size)),
            ]
        )
    if bool(numeric_off):
        command.append("--disable-numeric-source-features")
    if bool(args.token_numeric_value_features):
        command.extend(
            [
                "--token-numeric-value-features",
                "--token-numeric-value-vocab-size",
                str(int(args.token_numeric_value_vocab_size)),
            ]
        )
    if bool(token_numeric_off):
        command.append("--disable-token-numeric-value-features")
    if bool(args.token_numeric_source_slots):
        command.extend(
            [
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                str(int(args.token_numeric_source_slot_vocab_size)),
                "--token-numeric-source-slot-max-slots",
                str(int(args.token_numeric_source_slot_max_slots)),
                "--token-numeric-source-slot-gate-min",
                str(float(args.token_numeric_source_slot_gate_min)),
            ]
        )
        if bool(args.token_numeric_source_slot_predicate_feedback):
            command.extend(
                [
                    "--token-numeric-source-slot-predicate-feedback",
                    "--token-numeric-source-slot-predicate-gate-min",
                    str(float(args.token_numeric_source_slot_predicate_gate_min)),
                ]
            )
    if bool(source_slot_off):
        command.append("--disable-token-numeric-source-slots")
    if bool(args.core_source_position_binder):
        command.append("--core-source-position-binder")
        command.extend(
            [
                "--core-source-position-binder-gate-min",
                str(float(args.core_source_position_binder_gate_min)),
                "--core-source-position-binder-state-gate-min",
                str(float(args.core_source_position_binder_state_gate_min)),
            ]
        )
        if bool(args.core_source_position_binder_state_st):
            command.append("--core-source-position-binder-state-st")
        if bool(args.core_source_position_binder_source_slots_only):
            command.append("--core-source-position-binder-source-slots-only")
        if bool(args.core_source_position_binder_raw_source_slots):
            command.append("--core-source-position-binder-raw-source-slots")
        if bool(args.core_source_position_binder_query_state):
            command.extend(
                [
                    "--core-source-position-binder-query-state",
                    "--core-source-position-binder-query-state-gate-min",
                    str(float(args.core_source_position_binder_query_state_gate_min)),
                ]
            )
        if bool(args.core_source_value_binder):
            command.extend(
                [
                    "--core-source-value-binder",
                    "--core-source-value-binder-state-gate-min",
                    str(float(args.core_source_value_binder_state_gate_min)),
                ]
            )
            if bool(args.core_source_value_binder_state_st):
                command.append("--core-source-value-binder-state-st")
            if bool(args.core_primitive_role_value_source_value_conditioning):
                command.extend(
                    [
                        "--core-primitive-role-value-source-value-conditioning",
                        "--core-primitive-role-value-source-value-gate-min",
                        str(
                            float(
                                args.core_primitive_role_value_source_value_gate_min
                            )
                        ),
                    ]
                )
    if bool(source_binder_off):
        command.append("--disable-core-source-position-binder")
    if bool(strict_prompt_binding_off):
        command.extend(
            [
                "--disable-core-source-position-binder",
                "--disable-core-role-value-prompt-extract",
                "--disable-core-primitive-prompt-context",
            ]
        )
    if bool(primitive_off):
        command.append("--disable-core-primitive-role-value-executor")
    return command


def load_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("summary") if isinstance(report, dict) else None
    if isinstance(summary, dict):
        return summary
    if isinstance(report, dict):
        return report
    raise ValueError(f"expected JSON object in {path}")


def summarize_gate(
    *,
    full_summary: dict[str, Any],
    ablation_summary: dict[str, Any],
    min_trace_exact: float,
    min_value_accuracy: float,
    min_value_drop: float,
    numeric_ablation_summary: dict[str, Any] | None = None,
    min_numeric_value_drop: float = 0.0,
    token_numeric_ablation_summary: dict[str, Any] | None = None,
    min_token_numeric_value_drop: float = 0.0,
    source_slot_ablation_summary: dict[str, Any] | None = None,
    min_source_slot_value_drop: float = 0.0,
    source_binder_ablation_summary: dict[str, Any] | None = None,
    min_source_binder_value_drop: float = 0.0,
    strict_prompt_binding_ablation_summary: dict[str, Any] | None = None,
    min_strict_prompt_binding_value_drop: float = 0.0,
) -> dict[str, Any]:
    full_trace = float(full_summary.get("trace_exact_accuracy", 0.0))
    full_value = float(full_summary.get("value_accuracy", 0.0))
    full_step = float(full_summary.get("step_exact_accuracy", 0.0))
    ablation_trace = float(ablation_summary.get("trace_exact_accuracy", 0.0))
    ablation_value = float(ablation_summary.get("value_accuracy", 0.0))
    ablation_step = float(ablation_summary.get("step_exact_accuracy", 0.0))
    value_drop = full_value - ablation_value
    trace_drop = full_trace - ablation_trace
    numeric_ablation_value: float | None = None
    numeric_value_drop: float | None = None
    if numeric_ablation_summary is not None:
        numeric_ablation_value = float(numeric_ablation_summary.get("value_accuracy", 0.0))
        numeric_value_drop = full_value - numeric_ablation_value
    token_numeric_ablation_value: float | None = None
    token_numeric_value_drop: float | None = None
    if token_numeric_ablation_summary is not None:
        token_numeric_ablation_value = float(
            token_numeric_ablation_summary.get("value_accuracy", 0.0)
        )
        token_numeric_value_drop = full_value - token_numeric_ablation_value
    source_slot_ablation_value: float | None = None
    source_slot_value_drop: float | None = None
    if source_slot_ablation_summary is not None:
        source_slot_ablation_value = float(
            source_slot_ablation_summary.get("value_accuracy", 0.0)
        )
        source_slot_value_drop = full_value - source_slot_ablation_value
    source_binder_ablation_value: float | None = None
    source_binder_value_drop: float | None = None
    if source_binder_ablation_summary is not None:
        source_binder_ablation_value = float(
            source_binder_ablation_summary.get("value_accuracy", 0.0)
        )
        source_binder_value_drop = full_value - source_binder_ablation_value
    strict_prompt_binding_ablation_value: float | None = None
    strict_prompt_binding_value_drop: float | None = None
    if strict_prompt_binding_ablation_summary is not None:
        strict_prompt_binding_ablation_value = float(
            strict_prompt_binding_ablation_summary.get("value_accuracy", 0.0)
        )
        strict_prompt_binding_value_drop = (
            full_value - strict_prompt_binding_ablation_value
        )
    reject_reasons: list[str] = []
    if full_trace < float(min_trace_exact):
        reject_reasons.append("full trace exact below minimum")
    if full_value < float(min_value_accuracy):
        reject_reasons.append("full value accuracy below minimum")
    if value_drop < float(min_value_drop):
        reject_reasons.append("primitive-off ablation does not drop enough")
    if numeric_value_drop is not None and numeric_value_drop < float(min_numeric_value_drop):
        reject_reasons.append("numeric-source-off ablation does not drop enough")
    if (
        token_numeric_value_drop is not None
        and token_numeric_value_drop < float(min_token_numeric_value_drop)
    ):
        reject_reasons.append("token-numeric-off ablation does not drop enough")
    if (
        source_slot_value_drop is not None
        and source_slot_value_drop < float(min_source_slot_value_drop)
    ):
        reject_reasons.append("source-slot-off ablation does not drop enough")
    if (
        source_binder_value_drop is not None
        and source_binder_value_drop < float(min_source_binder_value_drop)
    ):
        reject_reasons.append("source-position-binder-off ablation does not drop enough")
    if (
        strict_prompt_binding_value_drop is not None
        and strict_prompt_binding_value_drop
        < float(min_strict_prompt_binding_value_drop)
    ):
        reject_reasons.append("strict-prompt-binding-off ablation does not drop enough")
    accepted = not reject_reasons
    result = {
        "decision": "accepted_l2" if accepted else "rejected",
        "accepted": accepted,
        "target_level": "L2 local gate",
        "full_trace_exact_accuracy": full_trace,
        "full_value_accuracy": full_value,
        "full_step_exact_accuracy": full_step,
        "ablation_trace_exact_accuracy": ablation_trace,
        "ablation_value_accuracy": ablation_value,
        "ablation_step_exact_accuracy": ablation_step,
        "trace_drop": trace_drop,
        "value_drop": value_drop,
        "reject_reasons": reject_reasons,
    }
    if numeric_ablation_value is not None and numeric_value_drop is not None:
        result.update(
            {
                "numeric_ablation_value_accuracy": numeric_ablation_value,
                "numeric_value_drop": numeric_value_drop,
            }
        )
    if (
        token_numeric_ablation_value is not None
        and token_numeric_value_drop is not None
    ):
        result.update(
            {
                "token_numeric_ablation_value_accuracy": token_numeric_ablation_value,
                "token_numeric_value_drop": token_numeric_value_drop,
            }
        )
    if source_slot_ablation_value is not None and source_slot_value_drop is not None:
        result.update(
            {
                "source_slot_ablation_value_accuracy": source_slot_ablation_value,
                "source_slot_value_drop": source_slot_value_drop,
            }
        )
    if (
        source_binder_ablation_value is not None
        and source_binder_value_drop is not None
    ):
        result.update(
            {
                "source_binder_ablation_value_accuracy": source_binder_ablation_value,
                "source_binder_value_drop": source_binder_value_drop,
            }
        )
    if (
        strict_prompt_binding_ablation_value is not None
        and strict_prompt_binding_value_drop is not None
    ):
        result.update(
            {
                "strict_prompt_binding_ablation_value_accuracy": (
                    strict_prompt_binding_ablation_value
                ),
                "strict_prompt_binding_value_drop": strict_prompt_binding_value_drop,
            }
        )
    return result


def run_command(
    command: list[str],
    *,
    out_path: Path,
    env: dict[str, str],
    cwd: Path,
) -> int:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    out_path.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    out_path.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def remove_rejected_checkpoints(train_out_dir: Path) -> None:
    for name in ("last.pt",):
        (train_out_dir / name).unlink(missing_ok=True)
    for path in train_out_dir.glob("step_*.pt"):
        path.unlink(missing_ok=True)


def sort_candidate_checkpoints(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, int, str]:
        name = path.name
        if name == "last.pt":
            return (1, 0, name)
        if name.startswith("step_") and name.endswith(".pt"):
            raw = name.removeprefix("step_").removesuffix(".pt")
            try:
                return (0, int(raw), name)
            except ValueError:
                pass
        return (0, 10**12, name)

    return sorted(paths, key=key)


def candidate_checkpoints(train_out_dir: Path) -> list[Path]:
    candidates = list(train_out_dir.glob("step_*.pt"))
    last = train_out_dir / "last.pt"
    if last.exists():
        candidates.append(last)
    return sort_candidate_checkpoints(candidates)


def _candidate_score(candidate: dict[str, Any]) -> tuple[float, float, float]:
    summary = candidate.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    return (
        float(summary.get("trace_exact_accuracy", 0.0)),
        float(summary.get("value_accuracy", 0.0)),
        float(summary.get("step_exact_accuracy", 0.0)),
    )


def select_best_full_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no candidate checkpoint summaries")
    return max(candidates, key=_candidate_score)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    out_dir = Path(args.out_dir)
    train_out_dir = out_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}.{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home)
    if args.tmpdir:
        env["TMPDIR"] = str(args.tmpdir)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    data_summary = ensure_gate_data(args, root=root)

    train_cmd = training_command(args, train_out_dir=train_out_dir)
    report: dict[str, Any] = {
        "target_level": "L2 local gate",
        "major_bottleneck": "QTRM source-position ordered recurrent state refresh",
        "commands": {"train": train_cmd},
        "artifacts": {
            "train_out_dir": str(train_out_dir),
            "data_summary": str(args.data_summary_json),
        },
    }
    if data_summary is not None:
        report["data_built"] = data_summary
    if bool(args.dry_run):
        report.update({"decision": "dry_run", "accepted": False})
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    train_exit = run_command(train_cmd, out_path=out_dir / "train", env=env, cwd=root)
    report["train_exit_code"] = train_exit
    checkpoints = candidate_checkpoints(train_out_dir)
    if train_exit != 0 or not checkpoints:
        report.update(
            {
                "decision": "command_failed",
                "accepted": False,
                "error": "training command failed or candidate checkpoints missing",
            }
        )
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    full_candidates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        full_json = out_dir / f"full_source_pointer_eval_{checkpoint.stem}.json"
        full_cmd = eval_command(
            args,
            checkpoint=checkpoint,
            out_json=full_json,
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
        )
        report["commands"][f"eval_full_{checkpoint.stem}"] = full_cmd
        full_exit = run_command(
            full_cmd,
            out_path=out_dir / f"eval_full_{checkpoint.stem}",
            env=env,
            cwd=root,
        )
        full_candidates.append(
            {
                "checkpoint": str(checkpoint),
                "eval_json": str(full_json),
                "exit_code": full_exit,
                "summary": load_summary(full_json) if full_exit == 0 else {},
            }
        )
        if full_exit != 0:
            report.update(
                {
                    "decision": "command_failed",
                    "accepted": False,
                    "error": f"full eval command failed for {checkpoint}",
                    "full_candidates": full_candidates,
                }
            )
            (out_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return report

    best_full = select_best_full_candidate(full_candidates)
    checkpoint = Path(str(best_full["checkpoint"]))
    full_json = Path(str(best_full["eval_json"]))
    off_json = out_dir / f"primitive_off_source_pointer_eval_{checkpoint.stem}.json"
    off_cmd = eval_command(
        args,
        checkpoint=checkpoint,
        out_json=off_json,
        primitive_off=True,
        numeric_off=False,
        token_numeric_off=False,
        source_slot_off=False,
        source_binder_off=False,
    )
    numeric_off_json: Path | None = None
    numeric_off_exit: int | None = None
    numeric_ablation_summary: dict[str, Any] | None = None
    token_numeric_off_json: Path | None = None
    token_numeric_off_exit: int | None = None
    token_numeric_ablation_summary: dict[str, Any] | None = None
    source_binder_off_json: Path | None = None
    source_binder_off_exit: int | None = None
    source_binder_ablation_summary: dict[str, Any] | None = None
    strict_prompt_binding_off_json: Path | None = None
    strict_prompt_binding_off_exit: int | None = None
    strict_prompt_binding_ablation_summary: dict[str, Any] | None = None
    report["commands"]["eval_primitive_off_best"] = off_cmd
    if bool(args.numeric_source_features):
        numeric_off_json = out_dir / f"numeric_off_source_pointer_eval_{checkpoint.stem}.json"
        numeric_off_cmd = eval_command(
            args,
            checkpoint=checkpoint,
            out_json=numeric_off_json,
            primitive_off=False,
            numeric_off=True,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
        )
        report["commands"]["eval_numeric_off_best"] = numeric_off_cmd
    if bool(args.token_numeric_value_features):
        token_numeric_off_json = (
            out_dir / f"token_numeric_off_source_pointer_eval_{checkpoint.stem}.json"
        )
        token_numeric_off_cmd = eval_command(
            args,
            checkpoint=checkpoint,
            out_json=token_numeric_off_json,
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=True,
            source_slot_off=False,
            source_binder_off=False,
        )
        report["commands"]["eval_token_numeric_off_best"] = token_numeric_off_cmd
    source_slot_off_json: Path | None = None
    source_slot_off_exit: int | None = None
    source_slot_ablation_summary: dict[str, Any] | None = None
    if bool(args.token_numeric_source_slots):
        source_slot_off_json = (
            out_dir / f"source_slot_off_source_pointer_eval_{checkpoint.stem}.json"
        )
        source_slot_off_cmd = eval_command(
            args,
            checkpoint=checkpoint,
            out_json=source_slot_off_json,
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=True,
            source_binder_off=False,
        )
        report["commands"]["eval_source_slot_off_best"] = source_slot_off_cmd
    if bool(args.core_source_position_binder):
        source_binder_off_json = (
            out_dir / f"source_binder_off_source_pointer_eval_{checkpoint.stem}.json"
        )
        source_binder_off_cmd = eval_command(
            args,
            checkpoint=checkpoint,
            out_json=source_binder_off_json,
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=True,
        )
        report["commands"]["eval_source_binder_off_best"] = source_binder_off_cmd
    if bool(args.strict_prompt_binding_ablation):
        strict_prompt_binding_off_json = (
            out_dir / f"strict_prompt_binding_off_source_pointer_eval_{checkpoint.stem}.json"
        )
        strict_prompt_binding_off_cmd = eval_command(
            args,
            checkpoint=checkpoint,
            out_json=strict_prompt_binding_off_json,
            primitive_off=False,
            numeric_off=False,
            token_numeric_off=False,
            source_slot_off=False,
            source_binder_off=False,
            strict_prompt_binding_off=True,
        )
        report["commands"]["eval_strict_prompt_binding_off_best"] = (
            strict_prompt_binding_off_cmd
        )
    full_exit = int(best_full["exit_code"])
    off_exit = run_command(
        off_cmd,
        out_path=out_dir / f"eval_primitive_off_{checkpoint.stem}",
        env=env,
        cwd=root,
    )
    if numeric_off_json is not None:
        numeric_off_exit = run_command(
            report["commands"]["eval_numeric_off_best"],
            out_path=out_dir / f"eval_numeric_off_{checkpoint.stem}",
            env=env,
            cwd=root,
        )
    if token_numeric_off_json is not None:
        token_numeric_off_exit = run_command(
            report["commands"]["eval_token_numeric_off_best"],
            out_path=out_dir / f"eval_token_numeric_off_{checkpoint.stem}",
            env=env,
            cwd=root,
        )
    if source_slot_off_json is not None:
        source_slot_off_exit = run_command(
            report["commands"]["eval_source_slot_off_best"],
            out_path=out_dir / f"eval_source_slot_off_{checkpoint.stem}",
            env=env,
            cwd=root,
        )
    if source_binder_off_json is not None:
        source_binder_off_exit = run_command(
            report["commands"]["eval_source_binder_off_best"],
            out_path=out_dir / f"eval_source_binder_off_{checkpoint.stem}",
            env=env,
            cwd=root,
        )
    if strict_prompt_binding_off_json is not None:
        strict_prompt_binding_off_exit = run_command(
            report["commands"]["eval_strict_prompt_binding_off_best"],
            out_path=out_dir / f"eval_strict_prompt_binding_off_{checkpoint.stem}",
            env=env,
            cwd=root,
        )
    report["eval_full_exit_code"] = full_exit
    report["eval_primitive_off_exit_code"] = off_exit
    if numeric_off_exit is not None:
        report["eval_numeric_off_exit_code"] = numeric_off_exit
    if token_numeric_off_exit is not None:
        report["eval_token_numeric_off_exit_code"] = token_numeric_off_exit
    if source_slot_off_exit is not None:
        report["eval_source_slot_off_exit_code"] = source_slot_off_exit
    if source_binder_off_exit is not None:
        report["eval_source_binder_off_exit_code"] = source_binder_off_exit
    if strict_prompt_binding_off_exit is not None:
        report["eval_strict_prompt_binding_off_exit_code"] = (
            strict_prompt_binding_off_exit
        )
    if (
        off_exit != 0
        or (numeric_off_exit is not None and numeric_off_exit != 0)
        or (token_numeric_off_exit is not None and token_numeric_off_exit != 0)
        or (source_slot_off_exit is not None and source_slot_off_exit != 0)
        or (source_binder_off_exit is not None and source_binder_off_exit != 0)
        or (
            strict_prompt_binding_off_exit is not None
            and strict_prompt_binding_off_exit != 0
        )
    ):
        report.update(
            {
                "decision": "command_failed",
                "accepted": False,
                "error": "eval command failed",
                "full_candidates": full_candidates,
                "best_full_candidate": best_full,
            }
        )
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    full_summary = dict(best_full["summary"])
    ablation_summary = load_summary(off_json)
    if numeric_off_json is not None:
        numeric_ablation_summary = load_summary(numeric_off_json)
    if token_numeric_off_json is not None:
        token_numeric_ablation_summary = load_summary(token_numeric_off_json)
    if source_slot_off_json is not None:
        source_slot_ablation_summary = load_summary(source_slot_off_json)
    if source_binder_off_json is not None:
        source_binder_ablation_summary = load_summary(source_binder_off_json)
    if strict_prompt_binding_off_json is not None:
        strict_prompt_binding_ablation_summary = load_summary(
            strict_prompt_binding_off_json
        )
    decision = summarize_gate(
        full_summary=full_summary,
        ablation_summary=ablation_summary,
        min_trace_exact=float(args.min_trace_exact),
        min_value_accuracy=float(args.min_value_accuracy),
        min_value_drop=float(args.min_value_drop),
        numeric_ablation_summary=numeric_ablation_summary,
        min_numeric_value_drop=float(args.min_numeric_value_drop),
        token_numeric_ablation_summary=token_numeric_ablation_summary,
        min_token_numeric_value_drop=float(args.min_token_numeric_value_drop),
        source_slot_ablation_summary=source_slot_ablation_summary,
        min_source_slot_value_drop=float(args.min_source_slot_value_drop),
        source_binder_ablation_summary=source_binder_ablation_summary,
        min_source_binder_value_drop=float(args.min_source_binder_value_drop),
        strict_prompt_binding_ablation_summary=(
            strict_prompt_binding_ablation_summary
        ),
        min_strict_prompt_binding_value_drop=float(
            args.min_strict_prompt_binding_value_drop
        ),
    )
    report.update(
        {
            **decision,
            "full_summary": full_summary,
            "ablation_summary": ablation_summary,
            "numeric_ablation_summary": numeric_ablation_summary,
            "token_numeric_ablation_summary": token_numeric_ablation_summary,
            "source_slot_ablation_summary": source_slot_ablation_summary,
            "source_binder_ablation_summary": source_binder_ablation_summary,
            "strict_prompt_binding_ablation_summary": (
                strict_prompt_binding_ablation_summary
            ),
            "full_candidates": full_candidates,
            "best_full_candidate": best_full,
            "artifacts": {
                **report["artifacts"],
                "checkpoint": str(checkpoint),
                "full_eval": str(full_json),
                "primitive_off_eval": str(off_json),
                **(
                    {"numeric_off_eval": str(numeric_off_json)}
                    if numeric_off_json is not None
                    else {}
                ),
                **(
                    {"token_numeric_off_eval": str(token_numeric_off_json)}
                    if token_numeric_off_json is not None
                    else {}
                ),
                **(
                    {"source_slot_off_eval": str(source_slot_off_json)}
                    if source_slot_off_json is not None
                    else {}
                ),
                **(
                    {"source_binder_off_eval": str(source_binder_off_json)}
                    if source_binder_off_json is not None
                    else {}
                ),
                **(
                    {
                        "strict_prompt_binding_off_eval": str(
                            strict_prompt_binding_off_json
                        )
                    }
                    if strict_prompt_binding_off_json is not None
                    else {}
                ),
            },
        }
    )
    if bool(report["accepted"]):
        accepted_path = train_out_dir / "accepted_l2_source_pointer_refresh.pt"
        accepted_path.write_bytes(checkpoint.read_bytes())
        report["artifacts"]["accepted_checkpoint"] = str(accepted_path)
    elif not bool(args.keep_rejected_checkpoints):
        remove_rejected_checkpoints(train_out_dir)
        report["artifacts"]["rejected_checkpoints_deleted"] = True

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a source-position pointer refresh gate on the "
            "corrected list combination split. This must pass before any "
            "copy/edit renderer promotion claim."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-data-jsonl", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data-jsonl", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--data-summary-json", default=DEFAULT_DATA_SUMMARY)
    parser.add_argument("--data-train-count", type=int, default=512)
    parser.add_argument("--data-eval-count", type=int, default=128)
    parser.add_argument("--data-value-modulus", type=int, default=32)
    parser.add_argument("--data-list-len", type=int, default=5)
    parser.add_argument("--data-value-vocab-size", type=int, default=256)
    parser.add_argument("--no-build-data-if-missing", action="store_true")
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--batch-integrated-training",
        action="store_true",
        help=(
            "Use the prompt-only batch QTRM source-pointer trainer instead of "
            "the legacy single-row depth-supervision script."
        ),
    )
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=319)
    parser.add_argument("--depth-steps", default="1,2,4,8")
    parser.add_argument("--core-steps", type=int, default=8)
    parser.add_argument("--max-eval-cases", type=int, default=128)
    parser.add_argument("--family-repeat", default="list_transform=16")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--primitive-transition-operation-ce-weight", type=float, default=0.50)
    parser.add_argument("--core-role-value-prompt-ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--core-role-value-prompt-target-mode",
        choices=["initial", "staged"],
        default="initial",
    )
    parser.add_argument("--core-primitive-role-value-state-ce-weight", type=float, default=1.0)
    parser.add_argument("--core-primitive-role-value-step-margin-weight", type=float, default=0.50)
    parser.add_argument("--core-primitive-role-value-step-margin", type=float, default=0.05)
    parser.add_argument("--core-primitive-role-value-trace-margin-weight", type=float, default=1.0)
    parser.add_argument("--core-primitive-role-value-trace-margin", type=float, default=0.10)
    parser.add_argument(
        "--core-primitive-role-value-pair-trace-contrast-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--core-primitive-role-value-pair-trace-contrast-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument("--core-primitive-role-value-update-gate-bce-weight", type=float, default=0.50)
    parser.add_argument("--min-trace-exact", type=float, default=0.25)
    parser.add_argument("--min-value-accuracy", type=float, default=0.50)
    parser.add_argument("--min-value-drop", type=float, default=0.25)
    parser.add_argument("--min-numeric-value-drop", type=float, default=0.25)
    parser.add_argument("--min-token-numeric-value-drop", type=float, default=0.25)
    parser.add_argument("--min-source-slot-value-drop", type=float, default=0.25)
    parser.add_argument("--min-source-binder-value-drop", type=float, default=0.25)
    parser.add_argument("--strict-prompt-binding-ablation", action="store_true")
    parser.add_argument(
        "--min-strict-prompt-binding-value-drop",
        type=float,
        default=0.25,
    )
    parser.add_argument("--numeric-source-features", action="store_true")
    parser.add_argument("--numeric-source-max-list-len", type=int, default=5)
    parser.add_argument("--numeric-source-value-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-value-features", action="store_true")
    parser.add_argument("--token-numeric-value-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument(
        "--token-numeric-source-slot-parity-ce-weight",
        type=float,
        default=0.0,
    )
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
        "--token-numeric-source-slot-predicate-ce-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-gate-min",
        type=float,
        default=0.0,
    )
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
    parser.add_argument("--core-source-position-binder-query-state", action="store_true")
    parser.add_argument(
        "--core-source-position-binder-query-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-value-binder", action="store_true")
    parser.add_argument(
        "--core-source-value-binder-state-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-value-binder-state-st", action="store_true")
    parser.add_argument("--core-source-value-prompt-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--core-primitive-role-value-source-value-conditioning",
        action="store_true",
    )
    parser.add_argument(
        "--core-primitive-role-value-source-value-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/mnt/nvme1n1p2/hf-cache-qtrm"))
    parser.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/mnt/nvme1n1p2/tmp"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-rejected-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_gate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("decision") != "command_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
