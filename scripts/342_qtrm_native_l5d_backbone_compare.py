#!/usr/bin/env python3
"""Compare QTRM-native MHA ETD and official FLA GatedDeltaNet placements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


CANDIDATES = (
    "mha_etd",
    "official_fla",
    "official_fla_encode_decode",
    "official_fla_think",
    "official_fla_encode_decode_mamba3_think",
    "official_mamba3",
    "official_mamba3_think",
    "trm_dual_z_fla_think",
    "trm_dual_z_mamba3_think",
    "official_trm_think",
    "trm_dual_z_official_trm_think",
    "trm_dual_z_official_trm_l2_think",
    "trm_dual_z_official_trm_fullgrad_think",
    "trm_dual_z_gated_official_trm_think",
    "trm_dual_z_residual_official_trm_think",
    "trm_dual_z_coupled_official_trm_think",
    "trm_dual_z_coupled_residual_official_trm_think",
    "trm_dual_z_coupled_delta_l_only_official_trm_think",
    "trm_dual_z_coupled_mamba_h_only_official_trm_think",
    "trm_dual_z_coupled_gated_proposal_official_trm_think",
    "trm_dual_z_coupled_gated_attention_think",
    "trm_dual_z_coupled_qwen_attention_think",
    "trm_dual_z_coupled_cross_attention_think",
    "trm_dual_z_coupled_step_conditioned_attention_think",
    "trm_dual_z_trm_mamba3_think",
    "trm_dual_z_trm_gated_delta_think",
    "trm_dual_z_gated_trm_gated_delta_think",
    "trm_dual_z_trm_qwen35_3to1_think",
    "trm_dual_z_trm_tri_mixer_think",
)


def base_profile_args(profile: str) -> list[str]:
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
            "32",
            "--n-heads",
            "4",
            "--n-kv-heads",
            "2",
            "--d-ff",
            "64",
            "--batch-size",
            "6",
            "--device",
            "cuda",
            "--log-every",
            "0",
            "--accept-min-exact",
            "0.0",
            "--accept-min-depth-gain",
            "-1.0",
            "--accept-min-ablation-drop",
            "-1.0",
            "--accept-min-family-exact",
            "0.0",
            "--accepted-decision",
            "accepted_l5d_compare_runtime",
        ]
    if profile == "short":
        return [
            "--steps",
            "400",
            "--train-cases",
            "2048",
            "--eval-cases",
            "192",
            "--task-families",
            "modchain,revchain,modchain,revchain,checksum",
            "--eval-task-families",
            "modchain,revchain,checksum",
            "--program-len",
            "4",
            "--modulus",
            "32",
            "--d-model",
            "64",
            "--n-heads",
            "4",
            "--n-kv-heads",
            "2",
            "--d-ff",
            "128",
            "--batch-size",
            "32",
            "--device",
            "cuda",
            "--depth-intermediate-loss-weight",
            "0.5",
            "--active-len-curriculum",
            "--log-every",
            "100",
            "--accept-min-exact",
            "0.0",
            "--accept-min-depth-gain",
            "-1.0",
            "--accept-min-ablation-drop",
            "-1.0",
            "--accept-min-family-exact",
            "0.0",
            "--accepted-decision",
            "accepted_l5d_compare_runtime",
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
            "--n-kv-heads",
            "4",
            "--d-ff",
            "256",
            "--batch-size",
            "128",
            "--device",
            "cuda",
            "--depth-intermediate-loss-weight",
            "0.5",
            "--active-len-curriculum",
            "--log-every",
            "1000",
            "--accept-min-exact",
            "0.60",
            "--accept-min-depth-gain",
            "0.10",
            "--accept-min-ablation-drop",
            "0.10",
            "--accept-min-family-exact",
            "0.40",
            "--accepted-decision",
            "accepted_l5d_compare_runtime",
        ]
    raise ValueError(f"unsupported profile: {profile}")


def official_fla_kernel_args(profile: str) -> list[str]:
    head_dim = "8" if profile == "smoke" else "16"
    return [
        "--hybrid-layers",
        "4",
        "--attn-every",
        "4",
        "--delta-backend",
        "fla_gated_delta",
        "--strict-backends",
        "--delta-head-dim",
        head_dim,
        "--delta-num-v-heads",
        "4",
        "--delta-expand-v",
        "1.0",
        "--delta-mode",
        "chunk",
        "--delta-conv-size",
        "4",
        "--delta-norm-eps",
        "1e-6",
    ]


def official_fla_args(profile: str) -> list[str]:
    return [
        "--backbone",
        "qtrm_hybrid_3to1",
        *official_fla_kernel_args(profile),
    ]


def candidate_args(candidate: str, profile: str) -> list[str]:
    if candidate == "mha_etd":
        return ["--backbone", "mha_etd"]
    if candidate == "official_fla":
        return official_fla_args(profile)
    if candidate == "official_fla_encode_decode":
        return [
            *official_fla_args(profile),
            "--encode-backbone",
            "qtrm_hybrid_3to1",
            "--think-backbone",
            "mha_etd",
            "--decode-backbone",
            "qtrm_hybrid_3to1",
        ]
    if candidate == "official_fla_think":
        return [
            *official_fla_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "qtrm_hybrid_3to1",
            "--decode-backbone",
            "mha_etd",
        ]
    if candidate == "official_mamba3_think":
        return [
            "--backbone",
            "mamba3",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "mamba3",
            "--decode-backbone",
            "mha_etd",
            "--strict-backends",
        ]
    if candidate == "trm_dual_z_mamba3_think":
        return [
            "--backbone",
            "mamba3",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "mamba3",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
            "--strict-backends",
        ]
    if candidate == "trm_dual_z_fla_think":
        return [
            *official_fla_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "qtrm_hybrid_3to1",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
        ]
    if candidate == "official_trm_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
        ]
    if candidate == "trm_dual_z_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
        ]
    if candidate == "trm_dual_z_official_trm_l2_think":
        return [
            *candidate_args("trm_dual_z_official_trm_think", profile),
            "--trm-l-cycles",
            "2",
        ]
    if candidate == "trm_dual_z_official_trm_fullgrad_think":
        return [
            *candidate_args("trm_dual_z_official_trm_think", profile),
            "--trm-full-grad-cycles",
        ]
    if candidate == "trm_dual_z_gated_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_gated",
        ]
    if candidate == "trm_dual_z_residual_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_residual",
        ]
    if candidate == "trm_dual_z_coupled_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled",
        ]
    if candidate == "trm_dual_z_coupled_residual_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled_residual",
        ]
    if candidate == "trm_dual_z_coupled_delta_l_only_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled_delta_l_only",
        ]
    if candidate == "trm_dual_z_coupled_mamba_h_only_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled_mamba_h_only",
            "--strict-backends",
        ]
    if candidate == "trm_dual_z_coupled_gated_proposal_official_trm_think":
        return [
            "--backbone",
            "trm_official",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled_gated_proposal",
        ]
    if candidate == "trm_dual_z_coupled_gated_attention_think":
        return [
            "--backbone",
            "trm_gated_attention",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_gated_attention",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled",
        ]
    if candidate == "trm_dual_z_coupled_qwen_attention_think":
        return [
            "--backbone",
            "trm_qwen_attention",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_qwen_attention",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled",
        ]
    if candidate == "trm_dual_z_coupled_cross_attention_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled_cross_attention",
        ]
    if candidate == "trm_dual_z_coupled_step_conditioned_attention_think":
        return [
            "--backbone",
            "trm_official",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_official",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_coupled_step_conditioned_attention",
        ]
    if candidate == "trm_dual_z_trm_mamba3_think":
        return [
            "--backbone",
            "trm_mamba3",
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_mamba3",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
            "--strict-backends",
        ]
    if candidate == "trm_dual_z_trm_gated_delta_think":
        return [
            "--backbone",
            "trm_gated_delta",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_gated_delta",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
        ]
    if candidate == "trm_dual_z_gated_trm_gated_delta_think":
        return [
            "--backbone",
            "trm_gated_delta",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_gated_delta",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z_gated",
        ]
    if candidate == "trm_dual_z_trm_qwen35_3to1_think":
        return [
            "--backbone",
            "trm_qwen35_3to1",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_qwen35_3to1",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
        ]
    if candidate == "trm_dual_z_trm_tri_mixer_think":
        return [
            "--backbone",
            "trm_tri_mixer",
            *official_fla_kernel_args(profile),
            "--encode-backbone",
            "mha_etd",
            "--think-backbone",
            "trm_tri_mixer",
            "--decode-backbone",
            "mha_etd",
            "--think-structure",
            "trm_dual_z",
        ]
    if candidate == "official_fla_encode_decode_mamba3_think":
        return [
            *official_fla_args(profile),
            "--encode-backbone",
            "qtrm_hybrid_3to1",
            "--think-backbone",
            "mamba3",
            "--decode-backbone",
            "qtrm_hybrid_3to1",
        ]
    if candidate == "official_mamba3":
        return [
            "--backbone",
            "mamba3",
            "--strict-backends",
        ]
    raise ValueError(f"unsupported candidate: {candidate}")


def compare_command(
    *,
    python_bin: str,
    out_dir: Path,
    candidate: str,
    profile: str,
    seed: int,
    eval_seed: int,
) -> list[str]:
    return [
        python_bin,
        "scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py",
        "--out-dir",
        str(out_dir),
        *base_profile_args(profile),
        *candidate_args(candidate, profile),
        "--seed",
        str(int(seed)),
        "--eval-seed",
        str(int(eval_seed)),
    ]


def _nested(report: dict[str, Any], path: str) -> float | bool | None:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return float(value)


def candidate_promotion(
    name: str,
    report: dict[str, Any],
    *,
    mha_exact: float | None,
) -> dict[str, Any]:
    exact = _nested(report, "decisive_metrics.full_generation_exact")
    delta = (
        round(float(exact - mha_exact), 12)
        if isinstance(exact, float) and mha_exact is not None
        else None
    )
    needs_mamba3 = (
        "mamba3" in name
        or "tri_mixer" in name
        or "coupled_residual" in name
        or "mamba_h_only" in name
        or "gated_proposal" in name
    )
    needs_fla = (
        "fla" in name
        or "gated_delta" in name
        or "qwen35_3to1" in name
        or "tri_mixer" in name
        or "coupled_residual" in name
        or "delta_l_only" in name
        or "gated_proposal" in name
    )
    mamba3_ok = bool(_nested(report, "backend_summary.all_mamba3_mixers_official")) and float(
        _nested(report, "backend_summary.official_mamba3_mixers") or 0.0
    ) > 0.0
    fla_ok = bool(_nested(report, "backend_summary.all_fla_mixers_official")) and float(
        _nested(report, "backend_summary.official_fla_delta_mixers") or 0.0
    ) > 0.0
    no_torch_delta = float(_nested(report, "backend_summary.torch_delta_mixers") or 0.0) == 0.0
    backend_ok = bool(
        (not needs_mamba3 or mamba3_ok)
        and (not needs_fla or fla_ok)
        and no_torch_delta
    )
    depth_gain = _nested(report, "decisive_metrics.full_minus_think0")
    ablation_drop = _nested(report, "decisive_metrics.full_minus_worst_ablation")
    causal_ok = bool(
        isinstance(depth_gain, float)
        and isinstance(ablation_drop, float)
        and depth_gain > 0.0
        and ablation_drop > 0.0
    )
    promoted = bool(
        backend_ok
        and delta is not None
        and delta > 0.0
        and causal_ok
        and bool(report.get("accepted"))
    )
    return {
        "candidate": name,
        "backend_ok": backend_ok,
        "causal_ok": causal_ok,
        "full_generation_exact": exact,
        "full_exact_delta_vs_mha": delta,
        "full_minus_think0": depth_gain,
        "full_minus_worst_ablation": ablation_drop,
        "accepted": bool(report.get("accepted")),
        "promoted": promoted,
    }


def summarize_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exact = {
        name: _nested(report, "decisive_metrics.full_generation_exact")
        for name, report in reports.items()
    }
    winner = None
    numeric_exact = {k: v for k, v in exact.items() if isinstance(v, float)}
    if numeric_exact:
        winner = max(numeric_exact, key=numeric_exact.get)
    mha_exact = numeric_exact.get("mha_etd")
    official_exact = numeric_exact.get("official_fla")
    delta = (
        round(float(official_exact - mha_exact), 12)
        if official_exact is not None and mha_exact is not None
        else None
    )
    official = reports.get("official_fla", {})
    candidate_promotions = {
        name: candidate_promotion(name, report, mha_exact=mha_exact)
        for name, report in reports.items()
        if name != "mha_etd"
    }
    official_candidate = candidate_promotions.get(
        "official_fla",
        candidate_promotion("official_fla", official, mha_exact=mha_exact),
    )
    return {
        "decision": "completed_l5d_backbone_compare",
        "accepted": True,
        "target_level": "L5D backbone comparison",
        "candidates": list(reports.keys()),
        "winner": winner,
        "full_generation_exact": exact,
        "full_exact_delta_official_fla_minus_mha": delta,
        "candidate_promotions": candidate_promotions,
        "official_fla_backend_ok": official_candidate["backend_ok"],
        "official_fla_causal_ok": official_candidate["causal_ok"],
        "official_fla_full_minus_think0": official_candidate["full_minus_think0"],
        "official_fla_full_minus_worst_ablation": official_candidate[
            "full_minus_worst_ablation"
        ],
        "official_fla_promoted": official_candidate["promoted"],
        "reports": reports,
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    commands: dict[str, list[str]] = {}
    reports: dict[str, dict[str, Any]] = {}
    candidates = [
        part.strip()
        for part in str(args.candidates).split(",")
        if part.strip()
    ]
    for candidate in candidates:
        if candidate not in CANDIDATES:
            raise ValueError(f"unsupported candidate: {candidate}")

    for candidate in candidates:
        out_dir = out_root / candidate
        command = compare_command(
            python_bin=str(args.python_bin),
            out_dir=out_dir,
            candidate=candidate,
            profile=str(args.profile),
            seed=int(args.seed),
            eval_seed=int(args.eval_seed),
        )
        commands[candidate] = command
        if args.dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.json"
        if args.reuse_existing and report_path.exists():
            reports[candidate] = json.loads(report_path.read_text(encoding="utf-8"))
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
        report["exit_code"] = int(completed.returncode)
        reports[candidate] = report

    if args.dry_run:
        summary: dict[str, Any] = {
            "decision": "dry_run",
            "accepted": False,
            "target_level": "L5D backbone comparison",
            "profile": str(args.profile),
            "commands": commands,
        }
    else:
        summary = summarize_reports(reports)
        summary["profile"] = str(args.profile)
        summary["commands"] = commands

    (out_root / "backbone_compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare QTRM-native L5D backbones.")
    parser.add_argument("--profile", choices=("smoke", "short", "standard"), default="smoke")
    parser.add_argument("--out-root", default="local_eval/qtrm_native_l5d_backbone_compare")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--seed", type=int, default=337)
    parser.add_argument("--eval-seed", type=int, default=9337)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_compare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision"] != "command_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
