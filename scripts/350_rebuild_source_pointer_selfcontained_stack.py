#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


BASE_CONFIG = (
    "configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_"
    "primitive_field_heads_delta_codec_s160.yaml"
)
BASE_TRAIN_DATA = "data/filtered/pure_recursive_reasoning_smallrange_train256_cases.jsonl"
SOURCE_POINTER_CONFIG = "configs/qwen35_2b_4090_list_order_source_pointer_refresh_s300.yaml"
L2_TRAIN_DATA = "data/filtered/qtrm_source_position_pair_hard_id_train512.jsonl"
L2_EVAL_DATA = "data/eval/qtrm_source_position_pair_hard_id_eval128.jsonl"
L3_TRAIN_DATA = "data/filtered/qtrm_source_pointer_l3_hard_train512_s1321.jsonl"
L3_EVAL_DATA = "data/eval/qtrm_source_pointer_l3_hard_eval128.jsonl"


def profile_defaults(profile: str) -> dict[str, int]:
    if profile == "smoke":
        return {
            "base_steps": 5,
            "base_save_every": 5,
            "l2_steps": 5,
            "l2_save_every": 5,
            "l2_max_eval_cases": 4,
            "l2_row_batch_size": 4,
            "l3_steps": 5,
            "l3_save_every": 5,
            "l3_max_eval_cases": 4,
            "l3_row_batch_size": 4,
            "log_every": 1,
        }
    if profile == "triage":
        return {
            "base_steps": 30,
            "base_save_every": 15,
            "l2_steps": 60,
            "l2_save_every": 20,
            "l2_max_eval_cases": 32,
            "l2_row_batch_size": 8,
            "l3_steps": 80,
            "l3_save_every": 40,
            "l3_max_eval_cases": 64,
            "l3_row_batch_size": 4,
            "log_every": 10,
        }
    return {
        "base_steps": 90,
        "base_save_every": 30,
        "l2_steps": 300,
        "l2_save_every": 100,
        "l2_max_eval_cases": 128,
        "l2_row_batch_size": 16,
        "l3_steps": 240,
        "l3_save_every": 120,
        "l3_max_eval_cases": 128,
        "l3_row_batch_size": 8,
        "log_every": 40,
    }


def _value(args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    return int(profile_defaults(args.profile)[name])


def base_train_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        args.python_bin,
        "scripts/196_train_pure_recursive_depth_supervised.py",
        "--config",
        args.base_config,
        "--data-jsonl",
        args.base_train_data,
        "--allow-random-init",
        "--tokenizer-model-id",
        args.tokenizer_model_id,
        "--out-dir",
        str(out_dir),
        "--steps",
        str(_value(args, "base_steps")),
        "--lr",
        str(float(args.base_lr)),
        "--seed",
        str(int(args.base_seed)),
        "--depth-steps",
        "1,2,4,8",
        "--target-mode",
        "staged",
        "--final-logit-ce-weight",
        "0.0",
        "--depth-final-ce-weight",
        "0.0",
        "--all-depth-ce-weight",
        "0.0",
        "--terminal-depth-ce-weight",
        "0.0",
        "--primitive-transition-operation-ce-weight",
        "0.50",
        "--core-primitive-role-value-state-ce-weight",
        "1.0",
        "--core-primitive-role-value-step-margin-weight",
        "0.50",
        "--core-primitive-role-value-step-margin",
        "0.05",
        "--core-primitive-role-value-trace-margin-weight",
        "1.0",
        "--core-primitive-role-value-trace-margin",
        "0.10",
        "--core-primitive-role-value-update-gate-bce-weight",
        "0.50",
        "--family-repeat",
        "list_transform=16",
        "--save-every",
        str(_value(args, "base_save_every")),
        "--log-every",
        str(_value(args, "log_every")),
    ]


def source_pointer_gate_command(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    init_checkpoint: Path,
    train_data: str,
    eval_data: str,
    steps: int,
    save_every: int,
    max_eval_cases: int,
    row_batch_size: int,
    keep_rejected_checkpoints: bool = False,
) -> list[str]:
    command = [
        args.python_bin,
        "scripts/319_run_qtrm_source_pointer_state_gate.py",
        "--out-dir",
        str(out_dir),
        "--config",
        args.source_pointer_config,
        "--train-data-jsonl",
        train_data,
        "--eval-data-jsonl",
        eval_data,
        "--init-checkpoint",
        str(init_checkpoint),
        "--tokenizer-model-id",
        args.tokenizer_model_id,
        "--steps",
        str(int(steps)),
        "--save-every",
        str(int(save_every)),
        "--max-eval-cases",
        str(int(max_eval_cases)),
        "--log-every",
        str(_value(args, "log_every")),
        "--lr",
        str(float(args.pointer_lr)),
        "--seed",
        str(int(args.pointer_seed)),
        "--batch-integrated-training",
        "--row-batch-size",
        str(int(row_batch_size)),
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        "128",
        "--token-numeric-source-slot-max-slots",
        "5",
        "--token-numeric-source-slot-gate-min",
        "1.0",
        "--token-numeric-source-slot-parity-ce-weight",
        "1.0",
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        "1.0",
        "--token-numeric-source-slot-predicate-ce-weight",
        "1.0",
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        "1.0",
        "--core-source-position-binder-state-gate-min",
        "0.25",
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
        "--core-role-value-prompt-ce-weight",
        "1.0",
        "--core-role-value-prompt-target-mode",
        "initial",
        "--strict-prompt-binding-ablation",
        "--min-source-slot-value-drop",
        "0.25",
        "--min-source-binder-value-drop",
        "0.25",
        "--min-strict-prompt-binding-value-drop",
        "0.25",
    ]
    if bool(keep_rejected_checkpoints):
        command.append("--keep-rejected-checkpoints")
    return command


def materialize_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    out: Path,
    report: Path,
) -> list[str]:
    return [
        args.python_bin,
        "scripts/329_materialize_qtrm_checkpoint_stack.py",
        "--config",
        args.source_pointer_config,
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
        "--report",
        str(report),
        "--fail-on-unmatched-keys",
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        "128",
        "--token-numeric-source-slot-max-slots",
        "5",
        "--token-numeric-source-slot-gate-min",
        "1.0",
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        "1.0",
        "--core-source-position-binder",
        "--core-source-position-binder-gate-min",
        "1.0",
        "--core-source-position-binder-state-gate-min",
        "0.25",
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
    ]


def l3_audit_command(args: argparse.Namespace, *, checkpoint: Path, out_dir: Path) -> list[str]:
    return [
        args.python_bin,
        "scripts/321_run_source_pointer_l3_hard_gate.py",
        "--out-dir",
        str(out_dir),
        "--config",
        args.source_pointer_config,
        "--checkpoint",
        str(checkpoint),
        "--data-jsonl",
        args.l3_eval_data,
        "--tokenizer-model-id",
        args.tokenizer_model_id,
        "--max-eval-cases",
        str(_value(args, "l3_max_eval_cases")),
        "--token-numeric-source-slots",
        "--token-numeric-source-slot-vocab-size",
        "128",
        "--token-numeric-source-slot-max-slots",
        "5",
        "--token-numeric-source-slot-gate-min",
        "1.0",
        "--token-numeric-source-slot-predicate-feedback",
        "--token-numeric-source-slot-predicate-gate-min",
        "1.0",
        "--core-source-position-binder-gate-min",
        "1.0",
        "--core-source-position-binder-state-gate-min",
        "0.25",
        "--core-source-position-binder-state-st",
        "--core-source-position-binder-source-slots-only",
        "--core-source-position-binder-raw-source-slots",
        "--min-trace-exact",
        str(float(args.l3_min_trace_exact)),
        "--min-value-accuracy",
        str(float(args.l3_min_value_accuracy)),
        "--min-primitive-value-drop",
        str(float(args.l3_min_primitive_value_drop)),
        "--min-token-numeric-value-drop",
        str(float(args.l3_min_token_numeric_value_drop)),
        "--min-source-binder-value-drop",
        str(float(args.l3_min_source_binder_value_drop)),
        "--min-variant-value-accuracy",
        str(float(args.l3_min_variant_value_accuracy)),
    ]


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    base_dir = out_dir / "00_base"
    l2_dir = out_dir / "01_l2_gate"
    l2_materialized_dir = out_dir / "02_l2_self_contained"
    l3_tune_dir = out_dir / "03_l3_tune"
    l3_audit_dir = out_dir / "04_l3_audit"
    l3_materialized_dir = out_dir / "05_l3_self_contained"
    base_checkpoint = base_dir / "last.pt"
    l2_delta = l2_dir / "train" / "accepted_l2_source_pointer_refresh.pt"
    l2_materialized = l2_materialized_dir / "accepted_l2_self_contained.pt"
    l3_candidate = l3_tune_dir / "train" / f"step_{_value(args, 'l3_steps'):06d}.pt"
    l3_accepted = l3_audit_dir / f"accepted_l3_{l3_candidate.stem}.pt"
    l3_materialized = l3_materialized_dir / "accepted_l3_self_contained.pt"
    return [
        {
            "name": "base_train",
            "out_dir": str(base_dir),
            "checkpoint": str(base_checkpoint),
            "command": base_train_command(args, base_dir),
        },
        {
            "name": "l2_gate",
            "out_dir": str(l2_dir),
            "checkpoint": str(l2_delta),
            "report": str(l2_dir / "report.json"),
            "command": source_pointer_gate_command(
                args,
                out_dir=l2_dir,
                init_checkpoint=base_checkpoint,
                train_data=args.l2_train_data,
                eval_data=args.l2_eval_data,
                steps=_value(args, "l2_steps"),
                save_every=_value(args, "l2_save_every"),
                max_eval_cases=_value(args, "l2_max_eval_cases"),
                row_batch_size=_value(args, "l2_row_batch_size"),
            ),
        },
        {
            "name": "l2_materialize",
            "out_dir": str(l2_materialized_dir),
            "checkpoint": str(l2_materialized),
            "command": materialize_command(
                args,
                checkpoint=l2_delta,
                out=l2_materialized,
                report=l2_materialized_dir / "report.json",
            ),
        },
        {
            "name": "l3_tune",
            "out_dir": str(l3_tune_dir),
            "checkpoint": str(l3_candidate),
            "report": str(l3_tune_dir / "report.json"),
            "command": source_pointer_gate_command(
                args,
                out_dir=l3_tune_dir,
                init_checkpoint=l2_materialized,
                train_data=args.l3_train_data,
                eval_data=args.l3_eval_data,
                steps=_value(args, "l3_steps"),
                save_every=_value(args, "l3_save_every"),
                max_eval_cases=_value(args, "l3_max_eval_cases"),
                row_batch_size=_value(args, "l3_row_batch_size"),
                keep_rejected_checkpoints=True,
            ),
        },
        {
            "name": "l3_audit",
            "out_dir": str(l3_audit_dir),
            "checkpoint": str(l3_accepted),
            "report": str(l3_audit_dir / "report.json"),
            "command": l3_audit_command(args, checkpoint=l3_candidate, out_dir=l3_audit_dir),
        },
        {
            "name": "l3_materialize",
            "out_dir": str(l3_materialized_dir),
            "checkpoint": str(l3_materialized),
            "command": materialize_command(
                args,
                checkpoint=l3_accepted,
                out=l3_materialized,
                report=l3_materialized_dir / "report.json",
            ),
        },
    ]


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _decisive_metrics(report: dict[str, Any]) -> dict[str, Any]:
    stage_reports = report.get("stage_reports") or {}
    if "l3_audit" in stage_reports:
        source = stage_reports.get("l3_audit") or {}
    elif "l2_gate" in stage_reports:
        source = stage_reports.get("l2_gate") or {}
    else:
        source = report
    preferred = [
        "full_trace_exact_accuracy",
        "full_value_accuracy",
        "value_drop",
        "primitive_value_drop",
        "token_numeric_value_drop",
        "source_binder_value_drop",
        "trace_drop",
        "full_step_exact_accuracy",
    ]
    return {key: source[key] for key in preferred if key in source}


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, Any]:
    for key in (
        "full_trace_exact_accuracy",
        "full_value_accuracy",
        "value_drop",
        "primitive_value_drop",
        "token_numeric_value_drop",
        "source_binder_value_drop",
    ):
        if key in metrics:
            return key, metrics[key]
    return "", ""


def _operational_status(report: dict[str, Any]) -> str:
    decision = str(report.get("decision") or "")
    if decision.endswith("_failed"):
        return "crash"
    if decision == "dry_run":
        return "probe"
    if bool(report.get("accepted")):
        return "keep"
    return "discard"


def _next_action(report: dict[str, Any]) -> str:
    decision = str(report.get("decision") or "")
    if decision == "accepted_l3_self_contained_stack":
        return "promote self-contained L3 checkpoint to L4 preflight"
    if decision == "l2_rejected":
        return "stop L4 work; recover L2 source-pointer gate before rebuilding L3"
    if decision == "l3_rejected":
        return "stop L4 work; recover L3 hard source-pointer gate from materialized L2"
    if decision == "dry_run":
        return "inspect generated plan before launching"
    if decision.endswith("_failed"):
        return "inspect failed stage stdout/stderr before retrying"
    return "inspect report and choose the smallest falsifying next run"


def append_operation_ledger(path: str | Path, report: dict[str, Any]) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = _decisive_metrics(report)
    metric_name, metric_value = _primary_metric(metrics)
    header = (
        "timestamp\tgate\tprofile\tdecision\tstatus\tprimary_metric\t"
        "primary_value\tout_dir\treport_path\tnext_action\n"
    )
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        ledger_path.write_text(header, encoding="utf-8")
    row = [
        str(report.get("timestamp", "")),
        str(report.get("gate", "")),
        str(report.get("profile", "")),
        str(report.get("decision", "")),
        _operational_status(report),
        str(metric_name),
        str(metric_value),
        str(report.get("out_dir", "")),
        str(Path(str(report.get("out_dir", ""))) / "report.json"),
        _next_action(report).replace("\t", " ").replace("\n", " "),
    ]
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def finalize_report(args: argparse.Namespace, report: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    report.setdefault("timestamp", datetime.now().replace(microsecond=0).isoformat())
    report.setdefault("gate", "source_pointer_selfcontained_stack")
    report.setdefault("next_action", _next_action(report))
    report.setdefault("decisive_metrics", _decisive_metrics(report))
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.operation_ledger and str(report.get("decision")) != "dry_run":
        ledger_path = Path(args.operation_ledger)
        if not ledger_path.is_absolute():
            ledger_path = Path(__file__).resolve().parents[1] / ledger_path
        append_operation_ledger(ledger_path, report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    report: dict[str, Any] = {
        "target_level": "L2/L3 source-pointer rebuild prerequisite",
        "major_bottleneck": "missing trainable-delta base chain blocks L4",
        "profile": args.profile,
        "stages": plan,
        "out_dir": str(out_dir),
    }
    if bool(args.dry_run):
        report.update({"decision": "dry_run", "accepted": False})
        return finalize_report(args, report, out_dir)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}.{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home)
    if args.tmpdir:
        env["TMPDIR"] = str(args.tmpdir)

    exits: dict[str, int] = {}
    stage_reports: dict[str, Any] = {}
    for stage in plan:
        name = str(stage["name"])
        stage_out_dir = Path(str(stage["out_dir"]))
        exits[name] = run_command(
            list(stage["command"]),
            cwd=root,
            env=env,
            out_dir=stage_out_dir,
        )
        if exits[name] != 0:
            report.update(
                {
                    "decision": f"{name}_failed",
                    "accepted": False,
                    "exit_codes": exits,
                    "stage_reports": stage_reports,
                }
            )
            return finalize_report(args, report, out_dir)
        if "report" in stage:
            stage_reports[name] = _load_report(str(stage["report"]))
        if name == "l2_gate" and not bool(stage_reports[name].get("accepted")):
            report.update(
                {
                    "decision": "l2_rejected",
                    "accepted": False,
                    "exit_codes": exits,
                    "stage_reports": stage_reports,
                }
            )
            return finalize_report(args, report, out_dir)
        if name == "l3_audit" and not bool(stage_reports[name].get("accepted")):
            report.update(
                {
                    "decision": "l3_rejected",
                    "accepted": False,
                    "exit_codes": exits,
                    "stage_reports": stage_reports,
                }
            )
            return finalize_report(args, report, out_dir)

    report.update(
        {
            "decision": "accepted_l3_self_contained_stack",
            "accepted": True,
            "exit_codes": exits,
            "stage_reports": stage_reports,
            "accepted_checkpoint": plan[-1]["checkpoint"],
        }
    )
    return finalize_report(args, report, out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = profile_defaults("standard")
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the source-pointer L2/L3 stack from a self-contained base "
            "and materialize accepted checkpoints before L4 promotion."
        )
    )
    parser.add_argument("--out-dir", default="local_eval/source_pointer_selfcontained_rebuild")
    parser.add_argument("--profile", choices=["smoke", "triage", "standard"], default="standard")
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--base-train-data", default=BASE_TRAIN_DATA)
    parser.add_argument("--source-pointer-config", default=SOURCE_POINTER_CONFIG)
    parser.add_argument("--l2-train-data", default=L2_TRAIN_DATA)
    parser.add_argument("--l2-eval-data", default=L2_EVAL_DATA)
    parser.add_argument("--l3-train-data", default=L3_TRAIN_DATA)
    parser.add_argument("--l3-eval-data", default=L3_EVAL_DATA)
    parser.add_argument("--base-steps", type=int, default=None)
    parser.add_argument("--base-save-every", type=int, default=None)
    parser.add_argument("--l2-steps", type=int, default=None)
    parser.add_argument("--l2-save-every", type=int, default=None)
    parser.add_argument("--l2-max-eval-cases", type=int, default=None)
    parser.add_argument("--l2-row-batch-size", type=int, default=None)
    parser.add_argument("--l3-steps", type=int, default=None)
    parser.add_argument("--l3-save-every", type=int, default=None)
    parser.add_argument("--l3-max-eval-cases", type=int, default=None)
    parser.add_argument("--l3-row-batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=5.0e-5)
    parser.add_argument("--pointer-lr", type=float, default=3.0e-4)
    parser.add_argument("--base-seed", type=int, default=11)
    parser.add_argument("--pointer-seed", type=int, default=319)
    parser.add_argument("--l3-min-trace-exact", type=float, default=0.10)
    parser.add_argument("--l3-min-value-accuracy", type=float, default=0.40)
    parser.add_argument("--l3-min-primitive-value-drop", type=float, default=0.20)
    parser.add_argument("--l3-min-token-numeric-value-drop", type=float, default=0.25)
    parser.add_argument("--l3-min-source-binder-value-drop", type=float, default=0.25)
    parser.add_argument("--l3-min-variant-value-accuracy", type=float, default=0.30)
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/mnt/nvme1n1p2/hf-cache-qtrm"))
    parser.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/tmp"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--operation-ledger", default="local_eval/research_gate_runner/results.tsv")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("decision") in {"dry_run", "accepted_l3_self_contained_stack"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
