#!/usr/bin/env python3
"""Run a seed-stability sweep for QTRM-native L5D backbone placements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def compare_seed_command(
    *,
    python_bin: str,
    out_root: Path,
    seed: int,
    eval_seed: int,
    profile: str,
    candidates: str,
) -> list[str]:
    return [
        python_bin,
        "scripts/342_qtrm_native_l5d_backbone_compare.py",
        "--profile",
        str(profile),
        "--out-root",
        str(out_root),
        "--candidates",
        str(candidates),
        "--seed",
        str(int(seed)),
        "--eval-seed",
        str(int(eval_seed)),
    ]


def _promotion(report: dict[str, Any], target_candidate: str) -> dict[str, Any]:
    promotions = report.get("candidate_promotions")
    if isinstance(promotions, dict):
        promotion = promotions.get(target_candidate)
        if isinstance(promotion, dict):
            return promotion
    return {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_seed_reports(
    reports: list[dict[str, Any]],
    *,
    target_candidate: str,
    min_seeds: int,
    min_promoted_rate: float,
    min_delta_vs_mha: float,
) -> dict[str, Any]:
    promoted_count = 0
    deltas: list[float] = []
    exact_values: list[float] = []
    causal_ok_count = 0
    backend_ok_count = 0

    for report in reports:
        promotion = _promotion(report, target_candidate)
        if bool(promotion.get("promoted")):
            promoted_count += 1
        if bool(promotion.get("causal_ok")):
            causal_ok_count += 1
        if bool(promotion.get("backend_ok")):
            backend_ok_count += 1
        delta = _float_or_none(promotion.get("full_exact_delta_vs_mha"))
        exact = _float_or_none(promotion.get("full_generation_exact"))
        if delta is not None:
            deltas.append(delta)
        if exact is not None:
            exact_values.append(exact)

    total = len(reports)
    promoted_rate = float(promoted_count / total) if total else 0.0
    reject_reasons: list[str] = []
    if total < int(min_seeds):
        reject_reasons.append("seed_count_below_threshold")
    if promoted_rate < float(min_promoted_rate):
        reject_reasons.append("promoted_rate_below_threshold")
    if len(deltas) != total:
        reject_reasons.append("missing_seed_delta_metric")
    elif min(deltas) < float(min_delta_vs_mha):
        reject_reasons.append("seed_delta_below_threshold")
    if len(exact_values) != total:
        reject_reasons.append("missing_seed_exact_metric")

    return {
        "decision": (
            "accepted_l5d_placement_seed_stability" if not reject_reasons else "rejected"
        ),
        "accepted": not reject_reasons,
        "target_level": "L5D placement seed stability",
        "target_candidate": str(target_candidate),
        "total": total,
        "min_seeds": int(min_seeds),
        "promoted_count": promoted_count,
        "promoted_rate": promoted_rate,
        "min_promoted_rate": float(min_promoted_rate),
        "causal_ok_count": causal_ok_count,
        "backend_ok_count": backend_ok_count,
        "min_delta_vs_mha": min(deltas) if deltas else None,
        "max_delta_vs_mha": max(deltas) if deltas else None,
        "required_min_delta_vs_mha": float(min_delta_vs_mha),
        "min_full_generation_exact": min(exact_values) if exact_values else None,
        "max_full_generation_exact": max(exact_values) if exact_values else None,
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
        seed_out_root = out_root / f"seed_{int(seed):03d}"
        command = compare_seed_command(
            python_bin=str(args.python_bin),
            out_root=seed_out_root,
            seed=int(seed),
            eval_seed=eval_seed,
            profile=str(args.profile),
            candidates=str(args.candidates),
        )
        commands.append(
            {
                "seed": int(seed),
                "eval_seed": eval_seed,
                "out_root": str(seed_out_root),
                "command": command,
            }
        )
        if args.dry_run:
            continue

        seed_out_root.mkdir(parents=True, exist_ok=True)
        report_path = seed_out_root / "backbone_compare_summary.json"
        if args.reuse_existing and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["seed"] = int(seed)
            report["eval_seed"] = eval_seed
            report["out_root"] = str(seed_out_root)
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
        (seed_out_root / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (seed_out_root / "stderr.log").write_text(completed.stderr, encoding="utf-8")
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
        report["out_root"] = str(seed_out_root)
        report["exit_code"] = int(completed.returncode)
        reports.append(report)

    if args.dry_run:
        summary: dict[str, Any] = {
            "decision": "dry_run",
            "accepted": False,
            "target_level": "L5D placement seed stability",
            "target_candidate": str(args.target_candidate),
            "profile": str(args.profile),
            "commands": commands,
        }
    else:
        summary = summarize_seed_reports(
            reports,
            target_candidate=str(args.target_candidate),
            min_seeds=int(args.min_seeds),
            min_promoted_rate=float(args.min_promoted_rate),
            min_delta_vs_mha=float(args.min_delta_vs_mha),
        )
        summary["profile"] = str(args.profile)
        summary["candidates"] = str(args.candidates)
        summary["commands"] = commands

    (out_root / "placement_seed_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QTRM-native L5D placement seed-stability sweep."
    )
    parser.add_argument("--profile", choices=("smoke", "short", "standard"), default="short")
    parser.add_argument(
        "--out-root",
        default="local_eval/qtrm_native_l5d_placement_seed_sweep",
    )
    parser.add_argument("--out-dir", dest="out_root")
    parser.add_argument("--candidates", default="mha_etd,official_fla_think")
    parser.add_argument("--target-candidate", default="official_fla_think")
    parser.add_argument("--seeds", type=int, nargs="+", default=[337, 338, 339])
    parser.add_argument("--eval-seed-base", type=int, default=9000)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-promoted-rate", type=float, default=1.0)
    parser.add_argument("--min-delta-vs-mha", type=float, default=0.0)
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
