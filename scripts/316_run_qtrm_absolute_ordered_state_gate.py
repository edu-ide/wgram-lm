#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_CONFIG = "configs/qwen35_2b_4090_list_order_absolute_value_state_s080.yaml"
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
    summary_path = root / str(args.data_summary_json)
    return module.write_absolute_ordered_state_split(
        train_out=train_path,
        eval_out=eval_path,
        summary_out=summary_path,
        train_count=int(args.data_train_count),
        eval_count=int(args.data_eval_count),
        value_modulus=int(args.data_value_modulus),
        list_len=int(args.data_list_len),
        value_vocab_size=int(args.data_value_vocab_size),
    )


def training_command(args: argparse.Namespace, *, train_out_dir: Path) -> list[str]:
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
        "absolute",
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
    return command


def eval_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    out_json: Path,
    primitive_off: bool,
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
        "absolute",
    ]
    if bool(primitive_off):
        command.append("--disable-core-primitive-role-value-executor")
    return command


def load_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(report, dict) and "summary" in report:
        summary = report["summary"]
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
) -> dict[str, Any]:
    full_trace = float(full_summary.get("trace_exact_accuracy", 0.0))
    full_value = float(full_summary.get("value_accuracy", 0.0))
    full_step = float(full_summary.get("step_exact_accuracy", 0.0))
    ablation_trace = float(ablation_summary.get("trace_exact_accuracy", 0.0))
    ablation_value = float(ablation_summary.get("value_accuracy", 0.0))
    ablation_step = float(ablation_summary.get("step_exact_accuracy", 0.0))
    value_drop = full_value - ablation_value
    trace_drop = full_trace - ablation_trace
    reject_reasons: list[str] = []
    if full_trace < float(min_trace_exact):
        reject_reasons.append("full trace exact below minimum")
    if full_value < float(min_value_accuracy):
        reject_reasons.append("full value accuracy below minimum")
    if value_drop < float(min_value_drop):
        reject_reasons.append("primitive-off ablation does not drop enough")
    accepted = not reject_reasons
    return {
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
        "major_bottleneck": "QTRM absolute ordered recurrent state before LM renderer",
        "commands": {
            "train": train_cmd,
        },
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

    train_exit = run_command(
        train_cmd,
        out_path=out_dir / "train",
        env=env,
        cwd=root,
    )
    report["train_exit_code"] = train_exit
    checkpoint = train_out_dir / "last.pt"
    if train_exit != 0 or not checkpoint.exists():
        report.update(
            {
                "decision": "command_failed",
                "accepted": False,
                "error": "training command failed or checkpoint missing",
            }
        )
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    full_json = out_dir / "full_role_value_eval.json"
    off_json = out_dir / "primitive_off_role_value_eval.json"
    full_cmd = eval_command(
        args,
        checkpoint=checkpoint,
        out_json=full_json,
        primitive_off=False,
    )
    off_cmd = eval_command(
        args,
        checkpoint=checkpoint,
        out_json=off_json,
        primitive_off=True,
    )
    report["commands"]["eval_full"] = full_cmd
    report["commands"]["eval_primitive_off"] = off_cmd
    full_exit = run_command(full_cmd, out_path=out_dir / "eval_full", env=env, cwd=root)
    off_exit = run_command(off_cmd, out_path=out_dir / "eval_primitive_off", env=env, cwd=root)
    report["eval_full_exit_code"] = full_exit
    report["eval_primitive_off_exit_code"] = off_exit
    if full_exit != 0 or off_exit != 0:
        report.update(
            {
                "decision": "command_failed",
                "accepted": False,
                "error": "eval command failed",
            }
        )
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    full_summary = load_summary(full_json)
    ablation_summary = load_summary(off_json)
    decision = summarize_gate(
        full_summary=full_summary,
        ablation_summary=ablation_summary,
        min_trace_exact=float(args.min_trace_exact),
        min_value_accuracy=float(args.min_value_accuracy),
        min_value_drop=float(args.min_value_drop),
    )
    report.update(
        {
            **decision,
            "full_summary": full_summary,
            "ablation_summary": ablation_summary,
            "artifacts": {
                **report["artifacts"],
                "checkpoint": str(checkpoint),
                "full_eval": str(full_json),
                "primitive_off_eval": str(off_json),
            },
        }
    )
    if bool(report["accepted"]):
        accepted_path = train_out_dir / "accepted_l2_absolute_ordered_state.pt"
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
            "Train/evaluate the QTRM absolute ordered recurrent state L2 gate. "
            "This checks state learning and primitive-off ablation before any "
            "new answer bridge or LM renderer claim."
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
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=316)
    parser.add_argument("--depth-steps", default="1,2,4")
    parser.add_argument("--core-steps", type=int, default=4)
    parser.add_argument("--max-eval-cases", type=int, default=18)
    parser.add_argument("--family-repeat", default="list_transform=16")
    parser.add_argument("--log-every", type=int, default=10)
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
    parser.add_argument("--core-primitive-role-value-update-gate-bce-weight", type=float, default=0.50)
    parser.add_argument("--min-trace-exact", type=float, default=0.10)
    parser.add_argument("--min-value-accuracy", type=float, default=0.25)
    parser.add_argument("--min-value-drop", type=float, default=0.15)
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
