#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any


FULL_MODE = "qtrm_core_steps_8_no_evidence"
BASELINE_MODES = (
    "donor_only_no_evidence",
    "qtrm_core_off_no_evidence",
    "qtrm_core_steps_8_answer_state_recurrent_off_no_evidence",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def build_train_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/196_train_pure_recursive_depth_supervised.py",
        "--config",
        str(args.config),
        "--data-jsonl",
        str(args.train_data),
        "--init-checkpoint",
        str(args.init_checkpoint),
        "--tokenizer-model-id",
        str(args.tokenizer_model_id),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--depth-steps",
        str(args.depth_steps),
        "--target-mode",
        str(args.target_mode),
        "--out-dir",
        str(args.out_dir),
        "--final-logit-ce-weight",
        str(args.final_logit_ce_weight),
        "--depth-final-ce-weight",
        str(args.depth_final_ce_weight),
        "--all-depth-ce-weight",
        str(args.all_depth_ce_weight),
        "--progress-margin-weight",
        str(args.progress_margin_weight),
        "--seed",
        str(args.seed),
        "--save-every",
        str(args.save_every),
        "--log-every",
        str(args.log_every),
    ]
    if not bool(args.full_checkpoints):
        command.append("--save-trainable-only")
    if bool(args.causal_prefix_supervision):
        command.extend(
            [
                "--causal-prefix-supervision",
                "--causal-prefix-max-target-tokens",
                str(args.causal_prefix_max_target_tokens),
                "--causal-prefix-later-token-weight",
                str(args.causal_prefix_later_token_weight),
            ]
        )
    command.extend(
        [
            "--terminal-depth-ce-weight",
            str(args.terminal_depth_ce_weight),
            "--answer-state-loop-halt-ce-weight",
            str(args.answer_state_loop_halt_ce_weight),
            "--choice-margin-weight",
            str(args.choice_margin_weight),
            "--choice-margin",
            str(args.choice_margin),
            "--choice-margin-mode",
            str(args.choice_margin_mode),
            "--tail-negative-margin-weight",
            str(args.tail_negative_margin_weight),
            "--tail-negative-margin",
            str(args.tail_negative_margin),
            "--tail-negative-family-filter",
            str(args.tail_negative_family_filter),
            "--subtract-tail-counterfactual-margin-weight",
            str(args.subtract_tail_counterfactual_margin_weight),
            "--subtract-tail-counterfactual-margin",
            str(args.subtract_tail_counterfactual_margin),
            "--subtract-tail-counterfactual-family-filter",
            str(args.subtract_tail_counterfactual_family_filter),
        ]
    )
    return command


def build_eval_command(
    args: argparse.Namespace,
    *,
    checkpoint: str | Path,
    out_jsonl: str | Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/192_eval_raw_intelligence.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--cases",
        str(args.eval_cases),
        "--max-cases",
        str(args.eval_max_cases),
        "--max-length",
        str(args.max_length),
        "--scoring",
        "causal_forced_choice",
        "--choice-score-normalization",
        "mean",
        "--mode",
        "donor_only_no_evidence",
        "--mode",
        "qtrm_core_off_no_evidence",
        "--mode",
        FULL_MODE,
        "--mode",
        "qtrm_core_steps_8_answer_state_recurrent_off_no_evidence",
        "--out",
        str(out_jsonl),
    ]


def build_action_eval_command(
    args: argparse.Namespace,
    *,
    checkpoint: str | Path,
    out_json: str | Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/230_eval_qtrm_latent_action_codebook.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--data-jsonl",
        str(args.eval_cases),
        "--core-steps",
        "8",
        "--max-cases",
        str(args.action_eval_max_cases),
        "--out-json",
        str(out_json),
    ]


def run_command(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
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


def summarize_eval_jsonl(path: str | Path) -> dict[str, Any]:
    by_mode: dict[str, dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        mode = str(record.get("mode") or "")
        bucket = by_mode.setdefault(mode, {"hits": 0, "total": 0, "ties": 0})
        hit = bool(
            record.get("hit")
            or record.get("exact_match")
            or record.get("normalized_exact")
        )
        bucket["hits"] += int(hit)
        bucket["total"] += 1
        bucket["ties"] += int(bool(record.get("choice_tied")))
    full_hits = by_mode.get(FULL_MODE, {}).get("hits", 0)
    baseline_hits = {
        mode: by_mode.get(mode, {}).get("hits", 0)
        for mode in BASELINE_MODES
    }
    return {
        "by_mode": by_mode,
        "full_hits": full_hits,
        "baseline_hits": baseline_hits,
        "full_beats_all_baselines": all(
            full_hits > hits for hits in baseline_hits.values()
        ),
        "full_margin_over_best_baseline": full_hits - max(
            baseline_hits.values() or [0]
        ),
    }


def load_action_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data.get("summary") or data)


def checkpoint_label(path: Path) -> str:
    if path.name == "last.pt":
        return "last"
    return path.stem


def candidate_checkpoints(out_dir: Path) -> list[Path]:
    checkpoints = sorted(out_dir.glob("step_*.pt"))
    last = out_dir / "last.pt"
    if last.exists():
        checkpoints.append(last)
    return checkpoints


def choose_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            int(item["eval_summary"]["full_beats_all_baselines"]),
            int(item["eval_summary"]["full_margin_over_best_baseline"]),
            int(item["eval_summary"]["full_hits"]),
        ),
    )


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_validation_gate(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = out_dir / "config_snapshot.yaml"
    if Path(args.config).exists():
        shutil.copy2(args.config, config_snapshot)

    train_command = build_train_command(args)
    manifest: dict[str, Any] = {
        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
        "gate": "ouro_answer_recurrent_validation_gated",
        "target_level": "L2 local reproduction gate",
        "seed": int(args.seed),
        "config": str(args.config),
        "config_snapshot": str(config_snapshot),
        "init_checkpoint": str(args.init_checkpoint),
        "init_checkpoint_sha256": (
            sha256_file(args.init_checkpoint)
            if Path(args.init_checkpoint).exists()
            else None
        ),
        "train_data": str(args.train_data),
        "eval_cases": str(args.eval_cases),
        "train_command": train_command,
        "candidates": [],
    }
    write_manifest(out_dir / "run_manifest.json", manifest)

    if bool(args.dry_run):
        manifest["decision"] = "dry_run"
        write_manifest(out_dir / "report.json", manifest)
        return manifest

    if not bool(args.skip_train):
        train_exit = run_command(
            train_command,
            cwd=root,
            stdout_path=out_dir / "train_stdout.log",
            stderr_path=out_dir / "train_stderr.log",
        )
        manifest["train_exit_code"] = train_exit
        if train_exit != 0:
            manifest["decision"] = "train_failed"
            write_manifest(out_dir / "report.json", manifest)
            return manifest

    for checkpoint in candidate_checkpoints(out_dir):
        label = checkpoint_label(checkpoint)
        eval_jsonl = eval_dir / f"{label}_causal_forced_choice.jsonl"
        eval_command = build_eval_command(args, checkpoint=checkpoint, out_jsonl=eval_jsonl)
        eval_exit = run_command(
            eval_command,
            cwd=root,
            stdout_path=eval_dir / f"{label}_eval_stdout.log",
            stderr_path=eval_dir / f"{label}_eval_stderr.log",
        )
        candidate: dict[str, Any] = {
            "label": label,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "eval_command": eval_command,
            "eval_exit_code": eval_exit,
            "eval_jsonl": str(eval_jsonl),
        }
        if eval_exit == 0:
            candidate["eval_summary"] = summarize_eval_jsonl(eval_jsonl)
        else:
            candidate["eval_summary"] = {"full_beats_all_baselines": False}
        manifest["candidates"].append(candidate)
        write_manifest(out_dir / "run_manifest.json", manifest)

    best = choose_best(manifest["candidates"])
    manifest["best_candidate"] = best
    accepted = False
    if best is not None:
        best_path = Path(best["checkpoint"])
        link_or_copy(best_path, out_dir / "best.pt")
        manifest["best_checkpoint"] = str(out_dir / "best.pt")
        action_json = eval_dir / f"{best['label']}_action_code_eval.json"
        action_command = build_action_eval_command(args, checkpoint=best_path, out_json=action_json)
        action_exit = run_command(
            action_command,
            cwd=root,
            stdout_path=eval_dir / f"{best['label']}_action_stdout.log",
            stderr_path=eval_dir / f"{best['label']}_action_stderr.log",
        )
        best["action_eval_command"] = action_command
        best["action_eval_exit_code"] = action_exit
        best["action_eval_json"] = str(action_json)
        if action_exit == 0:
            action_summary = load_action_summary(action_json)
            best["action_summary"] = action_summary
            accepted = bool(best["eval_summary"].get("full_beats_all_baselines")) and (
                int(action_summary.get("exact_rows", 0))
                == int(action_summary.get("rows", -1))
            )
        if accepted:
            link_or_copy(best_path, out_dir / "accepted.pt")
            manifest["accepted_checkpoint"] = str(out_dir / "accepted.pt")

    manifest["decision"] = "accepted_l2" if accepted else "rejected"
    manifest["accepted"] = accepted
    write_manifest(out_dir / "report.json", manifest)
    write_manifest(out_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-gated runner for the Ouro/LoopLM-style QTRM recurrent "
            "answer path. It preserves seed, commands, step checkpoints, evals, "
            "best.pt, and accepted.pt when the causal gate passes."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_answer_recurrent_s080.yaml",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt",
    )
    parser.add_argument(
        "--train-data",
        default="data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl",
    )
    parser.add_argument(
        "--eval-cases",
        default="data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="local_eval/research_gate_runner/ouro_answer_recurrent_validation_gated",
    )
    parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--depth-steps", default="1,2,4,8")
    parser.add_argument("--target-mode", default="staged")
    parser.add_argument("--final-logit-ce-weight", type=float, default=1.0)
    parser.add_argument("--depth-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--all-depth-ce-weight", type=float, default=0.0)
    parser.add_argument("--progress-margin-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-max-cases", type=int, default=4)
    parser.add_argument("--action-eval-max-cases", type=int, default=32)
    parser.add_argument("--full-checkpoints", action="store_true")
    parser.add_argument("--causal-prefix-supervision", action="store_true")
    parser.add_argument("--causal-prefix-max-target-tokens", type=int, default=8)
    parser.add_argument("--causal-prefix-later-token-weight", type=float, default=0.65)
    parser.add_argument("--terminal-depth-ce-weight", type=float, default=0.0)
    parser.add_argument("--answer-state-loop-halt-ce-weight", type=float, default=0.0)
    parser.add_argument("--choice-margin-weight", type=float, default=0.0)
    parser.add_argument("--choice-margin", type=float, default=0.10)
    parser.add_argument(
        "--choice-margin-mode",
        choices=["first_token", "sequence"],
        default="first_token",
    )
    parser.add_argument("--tail-negative-margin-weight", type=float, default=0.0)
    parser.add_argument("--tail-negative-margin", type=float, default=0.10)
    parser.add_argument("--tail-negative-family-filter", default="mixed_list_arithmetic")
    parser.add_argument("--subtract-tail-counterfactual-margin-weight", type=float, default=0.0)
    parser.add_argument("--subtract-tail-counterfactual-margin", type=float, default=0.05)
    parser.add_argument(
        "--subtract-tail-counterfactual-family-filter",
        default="mixed_list_arithmetic",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_validation_gate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("decision") not in {"train_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
