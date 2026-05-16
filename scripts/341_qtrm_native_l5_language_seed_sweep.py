#!/usr/bin/env python3
"""Run a reproducibility sweep for the QTRM-native L5C language gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def l5c_profile_args(profile: str) -> list[str]:
    if profile == "smoke":
        return [
            "--steps",
            "2",
            "--baseline-steps",
            "2",
            "--seq-len",
            "32",
            "--d-model",
            "16",
            "--n-heads",
            "4",
            "--d-ff",
            "32",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--log-every",
            "0",
            "--target-level",
            "L5C QTRM-native language non-regression",
            "--accepted-decision",
            "accepted_l5_language_nonregression",
            "--max-random-loss-fraction",
            "1.10",
            "--min-unique-chars",
            "8",
            "--max-run-fraction",
            "0.30",
            "--max-full-vs-think0-loss-ratio",
            "1.25",
            "--max-full-vs-off-loss-ratio",
            "1.25",
            "--max-full-vs-baseline-loss-ratio",
            "1.35",
        ]
    if profile == "standard":
        return [
            "--steps",
            "800",
            "--baseline-steps",
            "800",
            "--text-file",
            "docs/wiki/architecture/qtrm-native-first-roadmap.md",
            "--seq-len",
            "96",
            "--d-model",
            "64",
            "--n-heads",
            "4",
            "--d-ff",
            "128",
            "--batch-size",
            "64",
            "--target-level",
            "L5C QTRM-native language non-regression",
            "--accepted-decision",
            "accepted_l5_language_nonregression",
            "--max-random-loss-fraction",
            "0.85",
            "--min-unique-chars",
            "12",
            "--max-run-fraction",
            "0.25",
            "--max-full-vs-think0-loss-ratio",
            "1.25",
            "--max-full-vs-off-loss-ratio",
            "1.25",
            "--max-full-vs-baseline-loss-ratio",
            "1.35",
            "--log-every",
            "100",
        ]
    raise ValueError(f"unsupported profile: {profile}")


def l5c_command(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    profile: str,
) -> list[str]:
    return [
        python_bin,
        "scripts/336_train_qtrm_native_text_probe.py",
        "--out-dir",
        str(out_dir),
        *l5c_profile_args(profile),
        "--seed",
        str(int(seed)),
    ]


def _nested(report: dict[str, Any], path: str) -> float | None:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    return float(value)


def summarize_reports(
    reports: list[dict[str, Any]],
    *,
    min_pass_rate: float,
    max_baseline_ratio: float,
    min_seeds: int = 1,
) -> dict[str, Any]:
    baseline_ratios = [
        value
        for value in (
            _nested(report, "eval_metrics.loss_ratios.full_vs_baseline")
            for report in reports
        )
        if value is not None
    ]
    think0_ratios = [
        value
        for value in (
            _nested(report, "eval_metrics.loss_ratios.full_vs_think0")
            for report in reports
        )
        if value is not None
    ]
    off_ratios = [
        value
        for value in (
            _nested(report, "eval_metrics.loss_ratios.full_vs_thinking_block_off")
            for report in reports
        )
        if value is not None
    ]
    pass_count = sum(1 for report in reports if bool(report.get("accepted")))
    total = len(reports)
    pass_rate = float(pass_count / total) if total else 0.0
    reject_reasons: list[str] = []
    if total < int(min_seeds):
        reject_reasons.append("seed_count_below_threshold")
    if pass_rate < float(min_pass_rate):
        reject_reasons.append("pass_rate_below_threshold")
    if len(baseline_ratios) != total:
        reject_reasons.append("missing_seed_baseline_ratio")
    if baseline_ratios and max(baseline_ratios) > float(max_baseline_ratio):
        reject_reasons.append("seed_baseline_ratio_above_threshold")
    return {
        "decision": "accepted_l5c_seed_stability" if not reject_reasons else "rejected",
        "accepted": not reject_reasons,
        "target_level": "L5C language non-regression seed stability",
        "pass_count": pass_count,
        "total": total,
        "pass_rate": pass_rate,
        "min_seeds": int(min_seeds),
        "min_pass_rate": float(min_pass_rate),
        "max_required_full_vs_baseline": float(max_baseline_ratio),
        "min_full_vs_baseline": min(baseline_ratios) if baseline_ratios else None,
        "max_full_vs_baseline": max(baseline_ratios) if baseline_ratios else None,
        "min_full_vs_think0": min(think0_ratios) if think0_ratios else None,
        "max_full_vs_think0": max(think0_ratios) if think0_ratios else None,
        "min_full_vs_off": min(off_ratios) if off_ratios else None,
        "max_full_vs_off": max(off_ratios) if off_ratios else None,
        "reject_reasons": reject_reasons,
        "reports": reports,
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for seed in args.seeds:
        out_dir = out_root / f"seed_{int(seed):03d}"
        command = l5c_command(
            python_bin=str(args.python_bin),
            out_dir=out_dir,
            seed=int(seed),
            profile=str(args.profile),
        )
        commands.append(
            {
                "seed": int(seed),
                "out_dir": str(out_dir),
                "command": command,
            }
        )
        if args.dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.json"
        if args.reuse_existing and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["seed"] = int(seed)
            report["out_dir"] = str(out_dir)
            report["exit_code"] = 0
            reports.append(report)
            continue

        env = dict(os.environ)
        env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (out_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (out_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "accepted": False,
                "decision": "command_failed",
                "returncode": int(completed.returncode),
            }
        report["seed"] = int(seed)
        report["out_dir"] = str(out_dir)
        report["exit_code"] = int(completed.returncode)
        reports.append(report)

    if args.dry_run:
        summary: dict[str, Any] = {
            "decision": "dry_run",
            "accepted": False,
            "target_level": "L5C language non-regression seed stability",
            "profile": str(args.profile),
            "commands": commands,
        }
    else:
        summary = summarize_reports(
            reports,
            min_pass_rate=float(args.min_pass_rate),
            max_baseline_ratio=float(args.max_baseline_ratio),
            min_seeds=int(args.min_seeds),
        )
        summary["profile"] = str(args.profile)
        summary["commands"] = commands

    (out_root / "seed_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QTRM-native L5C language non-regression seed stability sweep."
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument(
        "--out-root",
        default="local_eval/qtrm_native_l5_language_nonregression_seed_sweep",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[337, 338, 339])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-baseline-ratio", type=float, default=1.35)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_sweep(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision"] != "command_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
