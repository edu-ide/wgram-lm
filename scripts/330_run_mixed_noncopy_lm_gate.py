#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


DEFAULT_CONFIG = (
    "configs/"
    "qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_"
    "typed_value_fullpath_scalar_codec_core_state_only_s060.yaml"
)
DEFAULT_CHECKPOINT = (
    "/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/"
    "l4_sufficient_onecase_overfit/train_eos_s020/last.pt"
)
DEFAULT_CASES = (
    "data/eval/"
    "pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_"
    "len1113_probe_eval60000_v6to7_len11_13_mixed_only.jsonl"
)
DEFAULT_OUT_DIR = (
    "/mnt/nvme0n1p2/qtrm-runs/research_gate_runner/"
    "qtrm_mixed_noncopy_lm_gate"
)

DONOR_MODE = "donor_only_no_evidence"
CORE_OFF_MODE = "qtrm_core_off_no_evidence"
FULL_MODE = "qtrm_core_steps_8_no_evidence"
PRIMITIVE_OFF_MODE = "qtrm_core_steps_8_primitive_role_value_off_no_evidence"
SOURCE_SLOT_OFF_MODE = "qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence"
SOURCE_BINDER_OFF_MODE = "qtrm_core_steps_8_core_source_position_binder_off_no_evidence"
BRIDGE_OFF_MODE = "qtrm_core_steps_8_role_value_answer_bridge_off_no_evidence"
TYPED_VALUE_BRIDGE_OFF_MODE = "qtrm_core_steps_8_typed_value_answer_bridge_off_no_evidence"
VOCAB_RENDERER_OFF_MODE = "qtrm_core_steps_8_core_role_value_vocab_renderer_off_no_evidence"
CORE_STATE_ZERO_MODE = "qtrm_core_steps_8_core_state_zero_no_evidence"
ANSWER_RECURRENT_OFF_MODE = "qtrm_core_steps_8_answer_state_recurrent_off_no_evidence"
ANSWER_NEXT_TOKEN_DECODER_OFF_MODE = (
    "qtrm_core_steps_8_answer_next_token_decoder_off_no_evidence"
)
ANSWER_FREE_TRANSFORMER_LATENT_OFF_MODE = (
    "qtrm_core_steps_8_answer_free_transformer_latent_off_no_evidence"
)

ABLATION_MODES = {
    "primitive_off": PRIMITIVE_OFF_MODE,
    "source_slot_off": SOURCE_SLOT_OFF_MODE,
    "source_binder_off": SOURCE_BINDER_OFF_MODE,
    "bridge_off": BRIDGE_OFF_MODE,
    "typed_value_bridge_off": TYPED_VALUE_BRIDGE_OFF_MODE,
    "vocab_renderer_off": VOCAB_RENDERER_OFF_MODE,
    "core_state_zero": CORE_STATE_ZERO_MODE,
    "answer_recurrent_off": ANSWER_RECURRENT_OFF_MODE,
    "answer_next_token_decoder_off": ANSWER_NEXT_TOKEN_DECODER_OFF_MODE,
    "answer_free_transformer_latent_off": ANSWER_FREE_TRANSFORMER_LATENT_OFF_MODE,
}
DEFAULT_MODES = [
    DONOR_MODE,
    CORE_OFF_MODE,
    FULL_MODE,
    PRIMITIVE_OFF_MODE,
    SOURCE_SLOT_OFF_MODE,
    SOURCE_BINDER_OFF_MODE,
    BRIDGE_OFF_MODE,
    TYPED_VALUE_BRIDGE_OFF_MODE,
    VOCAB_RENDERER_OFF_MODE,
    CORE_STATE_ZERO_MODE,
    ANSWER_RECURRENT_OFF_MODE,
    ANSWER_NEXT_TOKEN_DECODER_OFF_MODE,
    ANSWER_FREE_TRANSFORMER_LATENT_OFF_MODE,
]


def load_jsonl(path: str | Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def output_is_complete(path: str | Path, *, expected_rows: int) -> bool:
    out = Path(path)
    if not out.exists():
        return False
    try:
        return len(load_jsonl(out)) == int(expected_rows)
    except (OSError, json.JSONDecodeError):
        return False


def resolve_checkpoint_path(path: str | Path, *, root: Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_absolute():
        return checkpoint
    return root / checkpoint


def missing_checkpoint_base_chain(
    checkpoint: str | Path,
    *,
    root: Path,
    load_state=None,
) -> list[str]:
    if load_state is None:
        import torch

        def load_state(path: Path):
            return torch.load(path, map_location="cpu", weights_only=False)

    missing: list[str] = []
    seen: set[str] = set()
    current = Path(checkpoint)
    while True:
        resolved = resolve_checkpoint_path(current, root=root)
        resolved_key = str(resolved)
        if resolved_key in seen:
            break
        seen.add(resolved_key)
        if not resolved.exists():
            missing.append(resolved_key)
            break
        state = load_state(resolved)
        base_checkpoint = ""
        if isinstance(state, dict):
            base_checkpoint = str(state.get("base_checkpoint") or "").strip()
        if not base_checkpoint:
            break
        current = Path(base_checkpoint)
    return missing


def chunk_rows(rows: list[dict[str, Any]], *, chunk_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home)
    env["TMPDIR"] = str(args.tmpdir)
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        "src" if not current_pythonpath else f"src{os.pathsep}{current_pythonpath}"
    )
    return env


def eval_command(
    args: argparse.Namespace,
    *,
    mode: str,
    cases_path: Path,
    out_path: Path,
) -> list[str]:
    command = [
        args.python_bin,
        "scripts/192_eval_raw_intelligence.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--cases",
        str(cases_path),
        "--out",
        str(out_path),
        "--max-cases",
        str(int(args.chunk_size)),
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
        "--mode",
        str(mode),
    ]
    if bool(getattr(args, "token_numeric_source_slots", False)):
        command.extend(
            [
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                str(int(args.token_numeric_source_slot_vocab_size)),
                "--token-numeric-source-slot-max-slots",
                str(int(args.token_numeric_source_slot_max_slots)),
                "--token-numeric-source-slot-id-mode",
                str(args.token_numeric_source_slot_id_mode),
                "--token-numeric-source-slot-gate-min",
                str(float(args.token_numeric_source_slot_gate_min)),
            ]
        )
        if bool(getattr(args, "token_numeric_source_slot_predicate_feedback", False)):
            command.append("--token-numeric-source-slot-predicate-feedback")
            command.extend(
                [
                    "--token-numeric-source-slot-predicate-gate-min",
                    str(float(args.token_numeric_source_slot_predicate_gate_min)),
                ]
            )
    if bool(getattr(args, "core_source_position_binder", False)):
        command.extend(
            [
                "--core-source-position-binder",
                "--core-source-position-binder-gate-min",
                str(float(args.core_source_position_binder_gate_min)),
                "--core-source-position-binder-state-gate-min",
                str(float(args.core_source_position_binder_state_gate_min)),
            ]
        )
        if bool(getattr(args, "core_source_position_binder_state_st", False)):
            command.append("--core-source-position-binder-state-st")
        if bool(getattr(args, "core_source_position_binder_source_slots_only", False)):
            command.append("--core-source-position-binder-source-slots-only")
        if bool(getattr(args, "core_source_position_binder_raw_source_slots", False)):
            command.append("--core-source-position-binder-raw-source-slots")
    return command


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
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


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        hit = bool(row.get("exact_match") or row.get("normalized_exact"))
        loose_hit = bool(row.get("hit") or row.get("exact_match") or row.get("normalized_exact"))
        bucket = by_mode.setdefault(mode, {"hits": 0, "loose_hits": 0, "total": 0})
        bucket["hits"] += int(hit)
        bucket["loose_hits"] += int(loose_hit)
        bucket["total"] += 1
        family = str(row.get("task_family") or row.get("category") or "unknown")
        fam_bucket = by_family.setdefault(mode, {}).setdefault(
            family,
            {"hits": 0, "loose_hits": 0, "total": 0},
        )
        fam_bucket["hits"] += int(hit)
        fam_bucket["loose_hits"] += int(loose_hit)
        fam_bucket["total"] += 1
    for mode, bucket in by_mode.items():
        total = max(1, int(bucket["total"]))
        bucket["accuracy"] = float(bucket["hits"]) / float(total)
        bucket["by_family"] = {}
        for family, fam_bucket in by_family.get(mode, {}).items():
            fam_total = max(1, int(fam_bucket["total"]))
            bucket["by_family"][family] = {
                **fam_bucket,
                "accuracy": float(fam_bucket["hits"]) / float(fam_total),
            }
    return dict(sorted(by_mode.items()))


def mode_accuracy(summary: dict[str, dict[str, Any]], mode: str) -> float:
    return float((summary.get(mode) or {}).get("accuracy", 0.0))


def build_report(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    commands: list[dict[str, Any]],
    exit_codes: list[dict[str, Any]],
    min_full_accuracy: float,
    min_donor_margin: float,
    min_core_off_margin: float,
    min_primitive_drop: float = 0.01,
    min_source_slot_drop: float = 0.01,
    min_source_binder_drop: float = 0.01,
    min_bridge_drop: float = 0.01,
    min_typed_value_bridge_drop: float = 0.01,
    min_vocab_renderer_drop: float = 0.01,
    min_core_state_zero_drop: float = 0.01,
    min_answer_recurrent_drop: float = 0.01,
    min_answer_next_token_decoder_drop: float = 0.01,
    min_answer_free_transformer_latent_drop: float = 0.01,
) -> dict[str, Any]:
    summary = summarize_generation(rows)
    full = mode_accuracy(summary, FULL_MODE)
    donor = mode_accuracy(summary, DONOR_MODE)
    core_off = mode_accuracy(summary, CORE_OFF_MODE)
    required_modes = [DONOR_MODE, CORE_OFF_MODE, FULL_MODE, *ABLATION_MODES.values()]
    missing_modes = [mode for mode in required_modes if mode not in summary]
    ablation_thresholds = {
        "primitive_off": float(min_primitive_drop),
        "source_slot_off": float(min_source_slot_drop),
        "source_binder_off": float(min_source_binder_drop),
        "bridge_off": float(min_bridge_drop),
        "typed_value_bridge_off": float(min_typed_value_bridge_drop),
        "vocab_renderer_off": float(min_vocab_renderer_drop),
        "core_state_zero": float(min_core_state_zero_drop),
        "answer_recurrent_off": float(min_answer_recurrent_drop),
        "answer_next_token_decoder_off": float(min_answer_next_token_decoder_drop),
        "answer_free_transformer_latent_off": float(
            min_answer_free_transformer_latent_drop
        ),
    }
    ablation_accuracies = {
        name: mode_accuracy(summary, mode) for name, mode in ABLATION_MODES.items()
    }
    ablation_drops = {
        name: full - accuracy for name, accuracy in ablation_accuracies.items()
    }
    reject_reasons: list[str] = []
    if missing_modes:
        reject_reasons.append("missing_required_modes")
    if full < float(min_full_accuracy):
        reject_reasons.append("full_generation_accuracy_below_min")
    if full - donor <= float(min_donor_margin):
        reject_reasons.append("full_does_not_beat_donor")
    if full - core_off <= float(min_core_off_margin):
        reject_reasons.append("full_does_not_beat_core_off")
    for name, threshold in ablation_thresholds.items():
        if ABLATION_MODES[name] in missing_modes:
            continue
        if ablation_drops[name] <= threshold:
            reject_reasons.append(f"{name}_drop_below_min")
    failed_commands = [item for item in exit_codes if int(item.get("exit_code", 0)) != 0]
    if failed_commands:
        reject_reasons.append("eval_command_failed")
    accepted = not reject_reasons
    return {
        "decision": "accepted_l4_sufficient_noncopy_gate" if accepted else "rejected_noncopy_lm_gate",
        "accepted": accepted,
        "target_level": "L4 sufficient mixed non-copy LM gate",
        "major_bottleneck": "non-copy latent-state-to-autoregressive answer synthesis",
        "out_dir": str(out_dir),
        "rows": len(rows),
        "generation_summary": summary,
        "reject_reasons": reject_reasons,
        "missing_required_modes": missing_modes,
        "decisive_metrics": {
            "full_generation_accuracy": full,
            "donor_generation_accuracy": donor,
            "core_off_generation_accuracy": core_off,
            "full_minus_donor": full - donor,
            "full_minus_core_off": full - core_off,
            **{
                f"{name}_generation_accuracy": accuracy
                for name, accuracy in ablation_accuracies.items()
            },
            **{
                f"full_minus_{name}": drop
                for name, drop in ablation_drops.items()
            },
        },
        "thresholds": {
            "min_full_accuracy": float(min_full_accuracy),
            "min_donor_margin": float(min_donor_margin),
            "min_core_off_margin": float(min_core_off_margin),
            **{
                f"min_{name}_drop": threshold
                for name, threshold in ablation_thresholds.items()
            },
        },
        "commands": commands,
        "exit_codes": exit_codes,
        "next_action": (
            "promote to broader mixed-family non-copy eval"
            if accepted
            else "redesign non-copy answer synthesis; strict full-path causality is not proven"
        ),
    }


def compact_stdout_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "decision",
        "accepted",
        "target_level",
        "major_bottleneck",
        "reject_reasons",
        "decisive_metrics",
        "thresholds",
        "next_action",
        "missing_checkpoint_base_chain",
        "generation_jsonl",
        "report_path",
        "config",
        "checkpoint",
        "cases",
        "max_cases",
        "chunk_size",
    ]
    return {key: report[key] for key in keys if key in report}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_modes = list(args.mode) if args.mode else list(DEFAULT_MODES)
    run_modes = list(dict.fromkeys(run_modes))

    missing_chain = missing_checkpoint_base_chain(args.checkpoint, root=root)
    if missing_chain:
        generation_jsonl = out_dir / "generation.jsonl"
        generation_jsonl.write_text("", encoding="utf-8")
        report = build_report(
            [],
            out_dir=out_dir,
            commands=[],
            exit_codes=[],
            min_full_accuracy=float(args.min_full_accuracy),
            min_donor_margin=float(args.min_donor_margin),
            min_core_off_margin=float(args.min_core_off_margin),
            min_primitive_drop=float(args.min_primitive_drop),
            min_source_slot_drop=float(args.min_source_slot_drop),
            min_source_binder_drop=float(args.min_source_binder_drop),
            min_bridge_drop=float(args.min_bridge_drop),
            min_typed_value_bridge_drop=float(args.min_typed_value_bridge_drop),
            min_vocab_renderer_drop=float(args.min_vocab_renderer_drop),
            min_core_state_zero_drop=float(args.min_core_state_zero_drop),
            min_answer_recurrent_drop=float(args.min_answer_recurrent_drop),
            min_answer_next_token_decoder_drop=float(
                args.min_answer_next_token_decoder_drop
            ),
            min_answer_free_transformer_latent_drop=float(
                args.min_answer_free_transformer_latent_drop
            ),
        )
        report["reject_reasons"] = [
            "checkpoint_base_chain_missing",
            *report["reject_reasons"],
        ]
        report["missing_checkpoint_base_chain"] = missing_chain
        report.update(
            {
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "cases": str(args.cases),
                "max_cases": int(args.max_cases),
                "chunk_size": int(args.chunk_size),
                "modes": list(run_modes),
                "expected_rows": 0,
                "generation_jsonl": str(generation_jsonl),
            }
        )
        report_path = out_dir / "report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    rows = load_jsonl(args.cases, max_rows=int(args.max_cases))
    if not rows:
        raise ValueError(f"no rows loaded from {args.cases}")

    env = command_env(args)
    commands: list[dict[str, Any]] = []
    exit_codes: list[dict[str, Any]] = []
    output_parts: list[Path] = []
    expected_total = 0
    for chunk_index, chunk in enumerate(chunk_rows(rows, chunk_size=int(args.chunk_size))):
        chunk_cases = out_dir / "chunks" / f"cases_{chunk_index:04d}.jsonl"
        write_jsonl(chunk_cases, chunk)
        for mode in run_modes:
            out_part = out_dir / "chunks" / f"eval_{chunk_index:04d}_{mode}.jsonl"
            expected_total += len(chunk)
            command = eval_command(
                args,
                mode=mode,
                cases_path=chunk_cases,
                out_path=out_part,
            )
            commands.append(
                {
                    "chunk": chunk_index,
                    "mode": mode,
                    "command": command,
                    "out": str(out_part),
                }
            )
            if bool(args.resume) and output_is_complete(out_part, expected_rows=len(chunk)):
                exit_code = 0
            else:
                exit_code = run_command(
                    command,
                    cwd=root,
                    env=env,
                    stdout_path=out_dir / "logs" / f"eval_{chunk_index:04d}_{mode}.stdout.log",
                    stderr_path=out_dir / "logs" / f"eval_{chunk_index:04d}_{mode}.stderr.log",
                )
            exit_codes.append({"chunk": chunk_index, "mode": mode, "exit_code": exit_code})
            output_parts.append(out_part)
            print(
                json.dumps(
                    {
                        "chunk": chunk_index,
                        "mode": mode,
                        "exit_code": exit_code,
                        "out": str(out_part),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    eval_rows: list[dict[str, Any]] = []
    for part in output_parts:
        if part.exists():
            eval_rows.extend(load_jsonl(part))
    generation_jsonl = out_dir / "generation.jsonl"
    write_jsonl(generation_jsonl, eval_rows)

    report = build_report(
        eval_rows,
        out_dir=out_dir,
        commands=commands,
        exit_codes=exit_codes,
        min_full_accuracy=float(args.min_full_accuracy),
        min_donor_margin=float(args.min_donor_margin),
        min_core_off_margin=float(args.min_core_off_margin),
        min_primitive_drop=float(args.min_primitive_drop),
        min_source_slot_drop=float(args.min_source_slot_drop),
        min_source_binder_drop=float(args.min_source_binder_drop),
        min_bridge_drop=float(args.min_bridge_drop),
        min_typed_value_bridge_drop=float(args.min_typed_value_bridge_drop),
        min_vocab_renderer_drop=float(args.min_vocab_renderer_drop),
        min_core_state_zero_drop=float(args.min_core_state_zero_drop),
        min_answer_recurrent_drop=float(args.min_answer_recurrent_drop),
        min_answer_next_token_decoder_drop=float(
            args.min_answer_next_token_decoder_drop
        ),
        min_answer_free_transformer_latent_drop=float(
            args.min_answer_free_transformer_latent_drop
        ),
    )
    report.update(
        {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "cases": str(args.cases),
            "max_cases": int(args.max_cases),
            "chunk_size": int(args.chunk_size),
            "modes": list(run_modes),
            "expected_rows": int(expected_total),
            "generation_jsonl": str(generation_jsonl),
        }
    )
    report_path = out_dir / "report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether the current canonical LM path solves mixed-family "
            "non-copy reasoning. This is a gate for the post source-copy L4 bottleneck."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--hf-home", default="/mnt/nvme1n1p2/hf-cache-qtrm")
    parser.add_argument("--tmpdir", default="/mnt/nvme0n1p2/tmp")
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--min-full-accuracy", type=float, default=0.10)
    parser.add_argument("--min-donor-margin", type=float, default=0.01)
    parser.add_argument("--min-core-off-margin", type=float, default=0.01)
    parser.add_argument("--min-primitive-drop", type=float, default=0.01)
    parser.add_argument("--min-source-slot-drop", type=float, default=0.01)
    parser.add_argument("--min-source-binder-drop", type=float, default=0.01)
    parser.add_argument("--min-bridge-drop", type=float, default=0.01)
    parser.add_argument("--min-typed-value-bridge-drop", type=float, default=0.01)
    parser.add_argument("--min-vocab-renderer-drop", type=float, default=0.01)
    parser.add_argument("--min-core-state-zero-drop", type=float, default=0.01)
    parser.add_argument("--min-answer-recurrent-drop", type=float, default=0.01)
    parser.add_argument("--min-answer-next-token-decoder-drop", type=float, default=0.01)
    parser.add_argument(
        "--min-answer-free-transformer-latent-drop",
        type=float,
        default=0.01,
    )
    parser.add_argument("--token-numeric-source-slots", action="store_true")
    parser.add_argument("--token-numeric-source-slot-vocab-size", type=int, default=128)
    parser.add_argument("--token-numeric-source-slot-max-slots", type=int, default=5)
    parser.add_argument(
        "--token-numeric-source-slot-id-mode",
        choices=["absolute_value", "relative_parity"],
        default="absolute_value",
    )
    parser.add_argument("--token-numeric-source-slot-gate-min", type=float, default=0.0)
    parser.add_argument("--token-numeric-source-slot-predicate-feedback", action="store_true")
    parser.add_argument(
        "--token-numeric-source-slot-predicate-gate-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--core-source-position-binder", action="store_true")
    parser.add_argument("--core-source-position-binder-gate-min", type=float, default=0.0)
    parser.add_argument("--core-source-position-binder-state-gate-min", type=float, default=0.0)
    parser.add_argument("--core-source-position-binder-state-st", action="store_true")
    parser.add_argument("--core-source-position-binder-source-slots-only", action="store_true")
    parser.add_argument("--core-source-position-binder-raw-source-slots", action="store_true")
    parser.add_argument("--mode", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_eval(args)
    print(json.dumps(compact_stdout_report(report), ensure_ascii=False, indent=2))
    return 0 if not any(item.get("exit_code") for item in report["exit_codes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
