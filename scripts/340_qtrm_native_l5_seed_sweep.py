#!/usr/bin/env python3
"""Run a reproducibility sweep for the QTRM-native L5 multi-family gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def l5_profile_args(profile: str) -> list[str]:
    if profile == "smoke":
        return [
            "--steps",
            "2",
            "--train-cases",
            "18",
            "--eval-cases",
            "6",
            "--task-families",
            "modchain,revchain,modchain,revchain,checksum",
            "--eval-task-families",
            "modchain,revchain,checksum",
            "--program-len",
            "4",
            "--modulus",
            "32",
            "--d-model",
            "16",
            "--n-heads",
            "4",
            "--d-ff",
            "32",
            "--batch-size",
            "6",
            "--device",
            "cpu",
            "--log-every",
            "0",
            "--depth-intermediate-loss-weight",
            "0.5",
            "--active-len-curriculum",
            "--accept-min-exact",
            "0.70",
            "--accept-min-depth-gain",
            "0.10",
            "--accept-min-ablation-drop",
            "0.10",
            "--accept-min-family-exact",
            "0.30",
            "--accepted-decision",
            "accepted_l5_multifamily",
        ]
    if profile == "standard":
        return [
            "--steps",
            "12000",
            "--train-cases",
            "24576",
            "--eval-cases",
            "768",
            "--task-families",
            "modchain,revchain,modchain,revchain,checksum",
            "--eval-task-families",
            "modchain,revchain,checksum",
            "--program-len",
            "4",
            "--modulus",
            "32",
            "--d-model",
            "128",
            "--n-heads",
            "8",
            "--d-ff",
            "256",
            "--batch-size",
            "128",
            "--depth-intermediate-loss-weight",
            "0.5",
            "--active-len-curriculum",
            "--accept-min-exact",
            "0.60",
            "--accept-min-depth-gain",
            "0.10",
            "--accept-min-ablation-drop",
            "0.10",
            "--accept-min-family-exact",
            "0.40",
            "--accepted-decision",
            "accepted_l5_multifamily",
            "--log-every",
            "1000",
        ]
    raise ValueError(f"unsupported profile: {profile}")


def l5_command(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    eval_seed: int,
    profile: str,
) -> list[str]:
    return [
        python_bin,
        "scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
        "--out-dir",
        str(out_dir),
        *l5_profile_args(profile),
        "--seed",
        str(int(seed)),
        "--eval-seed",
        str(int(eval_seed)),
    ]


def _metric(report: dict[str, Any], key: str) -> float | None:
    decisive = report.get("decisive_metrics")
    if isinstance(decisive, dict) and key in decisive:
        return float(decisive[key])
    return None


def summarize_reports(
    reports: list[dict[str, Any]],
    *,
    min_pass_rate: float,
    min_exact: float,
    min_family_exact: float,
    min_seeds: int = 1,
) -> dict[str, Any]:
    full_values = [
        value
        for value in (_metric(report, "full_generation_exact") for report in reports)
        if value is not None
    ]
    family_values = [
        value
        for value in (_metric(report, "min_family_generation_exact") for report in reports)
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
    if full_values and min(full_values) < float(min_exact):
        reject_reasons.append("seed_exact_below_threshold")
    if family_values and min(family_values) < float(min_family_exact):
        reject_reasons.append("seed_family_exact_below_threshold")
    if len(full_values) != total:
        reject_reasons.append("missing_seed_exact_metric")
    if len(family_values) != total:
        reject_reasons.append("missing_seed_family_metric")
    return {
        "decision": "accepted_l5_seed_stability" if not reject_reasons else "rejected",
        "accepted": not reject_reasons,
        "target_level": "L5 multi-family seed stability",
        "pass_count": pass_count,
        "total": total,
        "pass_rate": pass_rate,
        "min_seeds": int(min_seeds),
        "min_pass_rate": float(min_pass_rate),
        "min_required_exact_per_seed": float(min_exact),
        "min_required_family_exact_per_seed": float(min_family_exact),
        "min_full_generation_exact": min(full_values) if full_values else None,
        "max_full_generation_exact": max(full_values) if full_values else None,
        "min_family_generation_exact": min(family_values) if family_values else None,
        "max_family_generation_exact": max(family_values) if family_values else None,
        "reject_reasons": reject_reasons,
        "reports": reports,
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for seed in args.seeds:
        eval_seed = int(args.eval_seed_base) + int(seed)
        out_dir = out_root / f"seed_{int(seed):03d}"
        command = l5_command(
            python_bin=str(args.python_bin),
            out_dir=out_dir,
            seed=int(seed),
            eval_seed=eval_seed,
            profile=str(args.profile),
        )
        commands.append(
            {
                "seed": int(seed),
                "eval_seed": eval_seed,
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
            report["eval_seed"] = eval_seed
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
        report["eval_seed"] = eval_seed
        report["out_dir"] = str(out_dir)
        report["exit_code"] = int(completed.returncode)
        reports.append(report)

    if args.dry_run:
        summary: dict[str, Any] = {
            "decision": "dry_run",
            "accepted": False,
            "target_level": "L5 multi-family seed stability",
            "profile": str(args.profile),
            "commands": commands,
        }
    else:
        summary = summarize_reports(
            reports,
            min_pass_rate=float(args.min_pass_rate),
            min_exact=float(args.min_exact),
            min_family_exact=float(args.min_family_exact),
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
        description="Run QTRM-native L5 multi-family seed stability sweep."
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--out-root", default="local_eval/qtrm_native_l5_multifamily_seed_sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[337, 338, 339])
    parser.add_argument("--eval-seed-base", type=int, default=9000)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-exact", type=float, default=0.60)
    parser.add_argument("--min-family-exact", type=float, default=0.40)
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
