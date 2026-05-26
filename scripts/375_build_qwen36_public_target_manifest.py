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


OFFICIAL_AGENT_BENCHMARK_LADDER: list[dict[str, str]] = [
    {
        "id": "swe_bench_verified",
        "display_name": "SWE-bench Verified",
        "recognition_role": "coding-agent repair credibility",
        "why": "real GitHub issue resolution with execution-based scoring",
        "source_url": "https://www.swebench.com/",
        "status": "qwen36_public_target_available",
    },
    {
        "id": "terminal_bench_2_0",
        "display_name": "Terminal-Bench 2.0",
        "recognition_role": "real terminal task completion",
        "why": "agent must operate in a terminal environment, not just answer text",
        "source_url": "https://www.tbench.ai/",
        "status": "qwen36_public_target_available",
    },
    {
        "id": "bfcl_v4",
        "display_name": "Berkeley Function Calling Leaderboard V4",
        "recognition_role": "tool-call correctness",
        "why": "measures whether the model calls the right tools with the right arguments",
        "source_url": "https://gorilla.cs.berkeley.edu/leaderboard",
        "status": "official_agent_benchmark_no_qwen36_target_in_manifest",
    },
    {
        "id": "tau_bench",
        "display_name": "tau-bench",
        "recognition_role": "multi-turn tool-agent-user workflow",
        "why": "tests whether an agent completes tasks through dialogue plus tools",
        "source_url": "https://www.tau-bench.com/",
        "status": "official_agent_benchmark_no_qwen36_target_in_manifest",
    },
    {
        "id": "gaia",
        "display_name": "GAIA",
        "recognition_role": "general assistant agent task solving",
        "why": "requires reasoning, retrieval, and tool use over multi-step questions",
        "source_url": "https://huggingface.co/gaia-benchmark",
        "status": "official_agent_benchmark_no_qwen36_target_in_manifest",
    },
    {
        "id": "qwen_deepplanning",
        "display_name": "Qwen-Agent DeepPlanning",
        "recognition_role": "planning and constraint satisfaction",
        "why": "matches the Qwen-Agent framing for travel and shopping planning tasks",
        "source_url": "https://qwenlm.github.io/Qwen-Agent/en/benchmarks/",
        "status": "qwen_agent_reference_benchmark",
    },
]

AGENT_RECOGNITION_CATEGORIES: dict[str, set[str]] = {
    "coding_or_terminal": {
        "swe_bench_verified",
        "swe_bench_pro",
        "swe_bench_multilingual",
        "terminal_bench_2_0",
        "nl2repo",
    },
    "tool_calling": {"bfcl_v4"},
    "long_horizon_workflow": {"tau_bench", "gaia", "qwen_deepplanning"},
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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def agent_artifact_record(path: str) -> dict[str, Any]:
    report = load_json(path)
    metrics = report.get("metrics", {})
    comparison = report.get("comparison", {})
    benchmark_id = str(
        report.get("benchmark_id")
        or metrics.get("benchmark_id")
        or comparison.get("benchmark_id")
        or ""
    )
    qtrm_score = _float_or_none(
        report.get("qtrm_score") or metrics.get("qtrm_score") or comparison.get("qtrm_score")
    )
    qwen36_target = _float_or_none(
        report.get("qwen36_target")
        or metrics.get("qwen36_target")
        or comparison.get("qwen36_target")
        or comparison.get("qwen36_score")
    )
    score_delta = _float_or_none(report.get("score_delta") or comparison.get("score_delta"))
    if score_delta is None and qtrm_score is not None and qwen36_target is not None:
        score_delta = qtrm_score - qwen36_target
    return {
        "kind": "official_agent_benchmark_report",
        "path": str(path),
        "exists": Path(path).exists(),
        "benchmark_id": benchmark_id,
        "benchmark_name": str(
            report.get("benchmark_name")
            or metrics.get("benchmark_name")
            or comparison.get("benchmark_name")
            or benchmark_id
        ),
        "official_harness": bool(
            report.get("official_harness")
            or metrics.get("official_harness")
            or comparison.get("official_harness")
        ),
        "accepted": bool(report.get("accepted", False)),
        "qtrm_score": qtrm_score,
        "qwen36_target": qwen36_target,
        "score_delta": score_delta,
        "beats_qwen36": bool(score_delta is not None and score_delta > 0),
        "decision": str(report.get("decision", "")),
        "status": str(report.get("status", "")),
        "summary_keys": sorted(report.keys())[:32],
    }


def agent_recognition_claim(agent_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_official = [
        artifact
        for artifact in agent_artifacts
        if artifact.get("accepted")
        and artifact.get("official_harness")
        and artifact.get("benchmark_id")
    ]
    category_hits: dict[str, list[str]] = {}
    for category, ids in AGENT_RECOGNITION_CATEGORIES.items():
        hits = sorted(
            {
                str(artifact["benchmark_id"])
                for artifact in accepted_official
                if str(artifact["benchmark_id"]) in ids
            }
        )
        category_hits[category] = hits
    missing_categories = sorted(
        category for category, hits in category_hits.items() if not hits
    )
    qwen36_beats = [
        str(artifact["benchmark_id"])
        for artifact in accepted_official
        if artifact.get("beats_qwen36")
    ]
    ready = bool(not missing_categories and qwen36_beats)
    return {
        "status": "ready" if ready else "not_ready",
        "accepted": ready,
        "claim": (
            "QTRM has official agent-benchmark evidence across coding/terminal, "
            "tool-calling, and long-horizon workflow categories."
        ),
        "category_hits": category_hits,
        "missing_categories": missing_categories,
        "qwen36_beating_benchmarks": sorted(qwen36_beats),
        "required_categories": sorted(AGENT_RECOGNITION_CATEGORIES),
        "rule": (
            "Do not claim official agent parity from synthetic OOD, private prompt suites, "
            "or one benchmark family. Require accepted official-harness artifacts across "
            "coding/terminal, tool-calling, and long-horizon workflow categories."
        ),
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
    agent_artifacts = [
        agent_artifact_record(str(path))
        for path in args.agent_output or []
    ]
    agent_claim = agent_recognition_claim(agent_artifacts)

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
        "official_agent_benchmark_ladder": OFFICIAL_AGENT_BENCHMARK_LADDER,
        "qtrm_native": {
            "model_family": "QTRM-Native",
            "candidate_checkpoint": str(args.qtrm_checkpoint),
            "artifacts": qtrm_artifacts,
        },
        "agent_benchmark_artifacts": agent_artifacts,
        "agent_recognition_claim": agent_claim,
        "acceptance_checks": {
            "qwen_source_url_present": bool(str(args.qwen_source_url).strip()),
            "all_targets_have_mapping": not missing_mapping,
            "qtrm_artifacts_present": bool(qtrm_artifacts),
            "qtrm_artifacts_exist": bool(all_artifacts_exist),
            "accepted_native_artifact_present": bool(accepted_native_artifact),
            "official_agent_claim_ready": bool(agent_claim["accepted"]),
        },
        "missing_mapping": missing_mapping,
        "limitations": [
            "This manifest does not claim a QTRM public benchmark win.",
            "A win requires running QTRM-Native on the relevant public benchmark cases and scorers.",
            "A public agent claim requires official-harness artifacts across coding/terminal, tool-calling, and long-horizon categories.",
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
    lines.extend(
        [
            "",
            "## Official Agent Benchmark Ladder",
            "",
            "| ID | Display Name | Recognition Role | Status |",
            "|---|---|---|---|",
        ]
    )
    for row in manifest.get("official_agent_benchmark_ladder", []):
        lines.append(
            f"| `{row['id']}` | {row['display_name']} | {row['recognition_role']} | {row['status']} |"
        )
    claim = manifest.get("agent_recognition_claim", {})
    lines.extend(
        [
            "",
            "## Agent Recognition Claim",
            "",
            f"- Status: `{claim.get('status', 'not_ready')}`",
            f"- Accepted: `{claim.get('accepted', False)}`",
            f"- Missing categories: `{', '.join(claim.get('missing_categories', []))}`",
            "",
            "| Benchmark | Official Harness | Accepted | QTRM Score | Qwen3.6 Target | Delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for artifact in manifest.get("agent_benchmark_artifacts", []):
        lines.append(
            f"| `{artifact['benchmark_id']}` | {artifact['official_harness']} | "
            f"{artifact['accepted']} | {artifact['qtrm_score']} | "
            f"{artifact['qwen36_target']} | {artifact['score_delta']} |"
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
    parser.add_argument("--agent-output", action="append", default=[])
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
