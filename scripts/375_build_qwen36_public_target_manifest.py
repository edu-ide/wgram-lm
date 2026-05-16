#!/usr/bin/env python3
"""Build a QTRM-Native vs Qwen3.6-27B public-target comparison manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def load_status_module():
    path = Path(__file__).with_name("372_qtrm_native_27b_milestone_status.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_27b_status", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load status module: {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PUBLIC_BENCHMARK_MAP: dict[str, dict[str, str]] = {
    "swe_bench_verified": {
        "display_name": "SWE-bench Verified",
        "task_type": "software engineering issue resolution",
        "scorer": "official SWE-bench Verified harness",
        "comparison_mode": "public target score",
    },
    "swe_bench_pro": {
        "display_name": "SWE-bench Pro",
        "task_type": "software engineering issue resolution",
        "scorer": "official SWE-bench Pro harness",
        "comparison_mode": "public target score",
    },
    "swe_bench_multilingual": {
        "display_name": "SWE-bench Multilingual",
        "task_type": "multilingual software engineering issue resolution",
        "scorer": "official SWE-bench Multilingual harness",
        "comparison_mode": "public target score",
    },
    "terminal_bench_2_0": {
        "display_name": "Terminal-Bench 2.0",
        "task_type": "terminal task completion",
        "scorer": "official Terminal-Bench 2.0 scorer",
        "comparison_mode": "public target score",
    },
    "skillsbench_avg5": {
        "display_name": "SkillsBench Avg5",
        "task_type": "agentic skill benchmark average",
        "scorer": "official SkillsBench Avg5 scorer",
        "comparison_mode": "public target score",
    },
    "qwenwebbench": {
        "display_name": "QwenWebBench",
        "task_type": "web browsing/search task benchmark",
        "scorer": "official QwenWebBench scorer",
        "comparison_mode": "public target score",
    },
    "nl2repo": {
        "display_name": "NL2Repo",
        "task_type": "natural-language to repository modification",
        "scorer": "official NL2Repo scorer",
        "comparison_mode": "public target score",
    },
    "claw_eval_avg": {
        "display_name": "Claw-Eval Avg",
        "task_type": "agentic capability average",
        "scorer": "official Claw-Eval average scorer",
        "comparison_mode": "public target score",
    },
    "mmlu_pro": {
        "display_name": "MMLU-Pro",
        "task_type": "broad knowledge and reasoning multiple-choice",
        "scorer": "official MMLU-Pro exact-match scorer",
        "comparison_mode": "public target score",
    },
    "gpqa_diamond": {
        "display_name": "GPQA Diamond",
        "task_type": "graduate-level science question answering",
        "scorer": "official GPQA Diamond exact-match scorer",
        "comparison_mode": "public target score",
    },
    "aime_2026": {
        "display_name": "AIME 2026",
        "task_type": "math contest answer extraction",
        "scorer": "official AIME numeric-answer scorer",
        "comparison_mode": "public target score",
    },
    "hmmt_feb_2026": {
        "display_name": "HMMT Feb 2026",
        "task_type": "math contest answer extraction",
        "scorer": "official HMMT numeric-answer scorer",
        "comparison_mode": "public target score",
    },
    "hle": {
        "display_name": "HLE",
        "task_type": "hard expert-level question answering",
        "scorer": "official HLE scorer",
        "comparison_mode": "public target score",
    },
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_record(path: str, *, kind: str) -> dict[str, Any]:
    report = load_json(path)
    return {
        "kind": kind,
        "path": str(path),
        "exists": Path(path).exists(),
        "accepted": bool(report.get("accepted", False)),
        "decision": str(report.get("decision", "")),
        "status": str(report.get("status", "")),
        "summary_keys": sorted(report.keys())[:32],
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    status_module = load_status_module()
    targets = dict(status_module.QWEN36_27B_TARGETS)
    benchmark_map = {
        key: {**PUBLIC_BENCHMARK_MAP[key], "qwen36_27b_target": float(value)}
        for key, value in targets.items()
    }
    qtrm_artifacts: list[dict[str, Any]] = []
    if str(args.native_report):
        qtrm_artifacts.append(artifact_record(str(args.native_report), kind="native_language_report"))
    if str(args.native_core_report):
        qtrm_artifacts.append(
            artifact_record(str(args.native_core_report), kind="native_core_ablation_report")
        )
    for path in args.qtrm_output or []:
        qtrm_artifacts.append(artifact_record(str(path), kind="qtrm_output_report"))

    missing_mapping = sorted(set(targets) - set(benchmark_map))
    all_artifacts_exist = all(bool(item["exists"]) for item in qtrm_artifacts)
    has_native_artifact = any(item["kind"] == "native_language_report" for item in qtrm_artifacts)
    accepted_native_artifact = any(
        item["kind"] == "native_language_report" and bool(item["accepted"])
        for item in qtrm_artifacts
    )
    accepted = bool(
        str(args.qwen_source_url).strip()
        and not missing_mapping
        and qtrm_artifacts
        and all_artifacts_exist
        and has_native_artifact
        and accepted_native_artifact
    )
    return {
        "status": "complete",
        "decision": "accepted_qwen36_public_target_manifest" if accepted else "rejected",
        "accepted": accepted,
        "comparison_mode": "public_qwen36_target_scores",
        "direct_qwen36_rerun_required": False,
        "direct_qwen36_rerun_note": (
            "Qwen3.6-27B public benchmark scores are used as fixed targets. "
            "A DGX/server rerun is optional and only needed for custom suites."
        ),
        "qwen36": {
            "model": "Qwen/Qwen3.6-27B",
            "source_url": str(args.qwen_source_url),
            "targets": targets,
        },
        "benchmark_map": benchmark_map,
        "qtrm_native": {
            "model_family": "QTRM-Native",
            "candidate_checkpoint": str(args.qtrm_checkpoint),
            "artifacts": qtrm_artifacts,
        },
        "acceptance_checks": {
            "qwen_source_url_present": bool(str(args.qwen_source_url).strip()),
            "all_targets_have_mapping": not missing_mapping,
            "qtrm_artifacts_present": bool(qtrm_artifacts),
            "qtrm_artifacts_exist": bool(all_artifacts_exist),
            "accepted_native_artifact_present": bool(accepted_native_artifact),
        },
        "missing_mapping": missing_mapping,
        "limitations": [
            "This manifest does not claim a QTRM public benchmark win.",
            "A win requires running QTRM-Native on the relevant public benchmark cases and scorers.",
            "Direct Qwen3.6 execution is optional for public target mode, but required for custom prompt suites.",
        ],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# QTRM-Native vs Qwen3.6-27B Public Target Manifest",
        "",
        f"Decision: `{manifest['decision']}`",
        f"Accepted: `{manifest['accepted']}`",
        "",
        "## Baseline",
        "",
        f"- Model: `{manifest['qwen36']['model']}`",
        f"- Source: {manifest['qwen36']['source_url']}",
        f"- Direct rerun required: `{manifest['direct_qwen36_rerun_required']}`",
        "",
        "## Benchmarks",
        "",
        "| ID | Display Name | Target | Scorer |",
        "|---|---|---:|---|",
    ]
    for key, row in manifest["benchmark_map"].items():
        lines.append(
            f"| `{key}` | {row['display_name']} | {row['qwen36_27b_target']} | {row['scorer']} |"
        )
    lines.extend(["", "## QTRM Artifacts", "", "| Kind | Accepted | Path |", "|---|---:|---|"])
    for artifact in manifest["qtrm_native"]["artifacts"]:
        lines.append(f"| `{artifact['kind']}` | {artifact['accepted']} | `{artifact['path']}` |")
    lines.extend(["", "## Limitations", ""])
    for item in manifest["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default="local_eval/qwen36_public_target_manifest/report.json")
    parser.add_argument("--out-md", default="local_eval/qwen36_public_target_manifest/report.md")
    parser.add_argument("--qwen-source-url", default="https://huggingface.co/Qwen/Qwen3.6-27B")
    parser.add_argument("--qtrm-checkpoint", default="")
    parser.add_argument("--native-report", default="")
    parser.add_argument("--native-core-report", default="")
    parser.add_argument("--qtrm-output", action="append", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = build_manifest(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    raise SystemExit(0 if bool(manifest["accepted"]) else 1)


if __name__ == "__main__":
    main()
