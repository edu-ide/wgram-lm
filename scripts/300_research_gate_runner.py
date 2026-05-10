#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class GateSpec:
    name: str
    target_level: str
    major_bottleneck: str
    script: str
    default_args: tuple[str, ...]
    report_name: str
    wiki_path: str
    accepted_decisions: tuple[str, ...]
    on_accept: str
    on_reject: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def profile_args(gate_name: str, profile: str) -> tuple[str, ...]:
    profile = str(profile)
    if gate_name == "donorless_recurrent_depth":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--train-cases",
                "32",
                "--eval-cases",
                "16",
                "--batch-size",
                "8",
                "--device",
                "cpu",
                "--log-every",
                "0",
            )
        if profile == "standard":
            return (
                "--steps",
                "1200",
                "--train-cases",
                "4096",
                "--eval-cases",
                "512",
                "--batch-size",
                "128",
                "--log-every",
                "200",
            )
    if gate_name == "ordered_list_state":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--train-cases",
                "32",
                "--eval-cases",
                "16",
                "--batch-size",
                "8",
                "--device",
                "cpu",
                "--log-every",
                "0",
            )
        if profile == "standard":
            return (
                "--steps",
                "1600",
                "--train-cases",
                "4096",
                "--eval-cases",
                "512",
                "--batch-size",
                "128",
                "--log-every",
                "100",
            )
    if gate_name == "prompt_source_position_binder":
        base = (
            "--train-jsonl",
            "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl",
            "--eval-jsonl",
            "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl",
            "--input-source",
            "token_embedding",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "16",
                "--eval-every",
                "5",
                "--log-every",
                "1",
                "--hidden-dim",
                "64",
                "--token-embedding-dim",
                "64",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "1000",
                "--batch-size",
                "64",
                "--eval-batch-size",
                "128",
                "--eval-every",
                "100",
                "--log-every",
                "100",
                "--hidden-dim",
                "512",
                "--token-embedding-dim",
                "256",
                "--transformer-layers",
                "2",
                "--transformer-heads",
                "8",
            )
    if gate_name == "prompt_source_position_binder_numeric":
        base = (
            "--train-jsonl",
            "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl",
            "--eval-jsonl",
            "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl",
            "--input-source",
            "numeric_value_embedding",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "16",
                "--eval-every",
                "5",
                "--log-every",
                "1",
                "--hidden-dim",
                "64",
                "--token-embedding-dim",
                "64",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "300",
                "--batch-size",
                "64",
                "--eval-batch-size",
                "128",
                "--eval-every",
                "50",
                "--log-every",
                "25",
                "--hidden-dim",
                "256",
                "--token-embedding-dim",
                "128",
                "--transformer-layers",
                "1",
                "--transformer-heads",
                "4",
            )
    if gate_name == "prompt_source_position_binder_token_plus_numeric":
        base = (
            "--train-jsonl",
            "data/filtered/qtrm_absolute_ordered_state_train512_v0to31.jsonl",
            "--eval-jsonl",
            "data/eval/qtrm_absolute_ordered_state_eval128_v0to31.jsonl",
            "--input-source",
            "token_plus_numeric_value",
        )
        if profile == "smoke":
            return (
                *base,
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-batch-size",
                "16",
                "--eval-every",
                "5",
                "--log-every",
                "1",
                "--hidden-dim",
                "64",
                "--token-embedding-dim",
                "64",
            )
        if profile == "standard":
            return (
                *base,
                "--steps",
                "300",
                "--batch-size",
                "64",
                "--eval-batch-size",
                "128",
                "--eval-every",
                "50",
                "--log-every",
                "25",
                "--hidden-dim",
                "256",
                "--token-embedding-dim",
                "128",
                "--transformer-layers",
                "1",
                "--transformer-heads",
                "4",
            )
    if gate_name == "qtrm_absolute_ordered_state":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "18",
                "--log-every",
                "25",
            )
    if gate_name == "qtrm_source_pointer_state":
        if profile == "smoke":
            return (
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "128",
                "--log-every",
                "25",
            )
    if gate_name == "qtrm_numeric_source_pointer_state":
        if profile == "smoke":
            return (
                "--numeric-source-features",
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
                "--min-numeric-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--numeric-source-features",
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "128",
                "--log-every",
                "25",
                "--min-numeric-value-drop",
                "0.25",
            )
    if gate_name == "qtrm_token_numeric_source_pointer_state":
        if profile == "smoke":
            return (
                "--token-numeric-value-features",
                "--steps",
                "5",
                "--save-every",
                "5",
                "--max-eval-cases",
                "4",
                "--log-every",
                "1",
                "--min-trace-exact",
                "0.01",
                "--min-value-accuracy",
                "0.01",
                "--min-value-drop",
                "0.01",
                "--min-token-numeric-value-drop",
                "0.01",
            )
        if profile == "standard":
            return (
                "--token-numeric-value-features",
                "--steps",
                "300",
                "--save-every",
                "100",
                "--max-eval-cases",
                "128",
                "--log-every",
                "25",
                "--min-token-numeric-value-drop",
                "0.25",
            )
    if gate_name == "qtrm_minimal_depth":
        if profile in {"smoke", "standard"}:
            return ()
    if gate_name == "renderer_canonical_lm":
        if profile in {"smoke", "standard"}:
            return ()
    if gate_name == "small_general_reasoning":
        if profile == "smoke":
            return (
                "--max-train-per-source",
                "1",
                "--max-eval-per-source",
                "1",
                "--max-train-cases",
                "2",
                "--max-eval-cases",
                "2",
                "--soft-prefix-steps",
                "2",
                "--max-new-tokens",
                "4",
                "--log-every",
                "1",
                "--no-require-family-full-hit",
                "--min-full-accuracy",
                "0.0",
            )
        if profile == "standard":
            return ()
    raise ValueError(f"unsupported profile for {gate_name}: {profile}")


def gate_specs(profile: str) -> dict[str, GateSpec]:
    return {
        "donorless_recurrent_depth": GateSpec(
            name="donorless_recurrent_depth",
            target_level="L1 scaffold",
            major_bottleneck="reset prerequisite for bottleneck 2 recursive depth scaling",
            script="scripts/260_train_donorless_recurrent_depth_probe.py",
            default_args=profile_args("donorless_recurrent_depth", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/donorless-recurrent-depth-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "open qtrm_minimal_depth gate: port the same recurrence pressure "
                "into QTRM and require donor-only < QTRM plus core_off < QTRM"
            ),
            on_reject=(
                "stop integrated donor-QTRM tuning; redesign the donorless "
                "recurrence/task until an isolated depth gain is accepted"
            ),
        ),
        "ordered_list_state": GateSpec(
            name="ordered_list_state",
            target_level="L1 scaffold",
            major_bottleneck=(
                "ordered select/map/copy recurrent state before canonical LM "
                "renderer integration"
            ),
            script="scripts/315_train_ordered_list_state_probe.py",
            default_args=profile_args("ordered_list_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/ordered-list-state-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "port the ordered-slot transition into QTRM so final LM logits "
                "depend on the ordered recurrent state; require source/state-off "
                "ablation drop before L3"
            ),
            on_reject=(
                "do not tune answer bridges; redesign the ordered recurrent "
                "state until filter->double composition is accepted in isolation"
            ),
        ),
        "prompt_source_position_binder": GateSpec(
            name="prompt_source_position_binder",
            target_level="L1 scaffold",
            major_bottleneck=(
                "prompt-token numeric source-position binding before QTRM "
                "recurrent pointer-state integration"
            ),
            script="scripts/320_train_prompt_source_position_binder_probe.py",
            default_args=profile_args("prompt_source_position_binder", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/prompt-source-position-binder-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "port the source-position binder into QTRM and require "
                "numeric-feature/binder-off ablation drop"
            ),
            on_reject=(
                "add numeric-aware input representation or digit/value features "
                "before retrying recurrent pointer-state QTRM L2"
            ),
        ),
        "prompt_source_position_binder_numeric": GateSpec(
            name="prompt_source_position_binder_numeric",
            target_level="L1 scaffold",
            major_bottleneck=(
                "numeric-aware source-position binding before QTRM recurrent "
                "pointer-state integration"
            ),
            script="scripts/320_train_prompt_source_position_binder_probe.py",
            default_args=profile_args("prompt_source_position_binder_numeric", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/prompt-source-position-binder-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "port numeric-aware source-slot embeddings into QTRM and require "
                "numeric-feature-off plus core-off ablation drops"
            ),
            on_reject=(
                "redesign numeric-aware input representation before retrying "
                "QTRM source-pointer L2"
            ),
        ),
        "prompt_source_position_binder_token_plus_numeric": GateSpec(
            name="prompt_source_position_binder_token_plus_numeric",
            target_level="L1 scaffold",
            major_bottleneck=(
                "canonical token-path value-aware source-position binding before "
                "QTRM recurrent pointer-state integration"
            ),
            script="scripts/320_train_prompt_source_position_binder_probe.py",
            default_args=profile_args(
                "prompt_source_position_binder_token_plus_numeric",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/prompt-source-position-binder-probe.md",
            accepted_decisions=("accepted_l1", "accepted"),
            on_accept=(
                "replace side-channel numeric source features with token-path "
                "value-aware embeddings in QTRM source-pointer L2"
            ),
            on_reject=(
                "canonical token-path numeric binding is still insufficient; "
                "improve token-aligned value representation before QTRM L2"
            ),
        ),
        "qtrm_absolute_ordered_state": GateSpec(
            name="qtrm_absolute_ordered_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "port accepted ordered-list recurrent state into QTRM primitive "
                "role/value state with absolute value targets"
            ),
            script="scripts/316_run_qtrm_absolute_ordered_state_gate.py",
            default_args=profile_args("qtrm_absolute_ordered_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-absolute-ordered-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open canonical LM renderer gate: require final LM logits or "
                "generation to improve and drop under ordered-state-off ablation"
            ),
            on_reject=(
                "do not add answer bridges; fix QTRM ordered state learning or "
                "port the donorless ordered-slot transition more directly"
            ),
        ),
        "qtrm_source_pointer_state": GateSpec(
            name="qtrm_source_pointer_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "replace brittle absolute value classes with source-position "
                "pointer state on the corrected list combination split"
            ),
            script="scripts/319_run_qtrm_source_pointer_state_gate.py",
            default_args=profile_args("qtrm_source_pointer_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-source-pointer-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open copy/edit LM renderer gate: require final autoregressive "
                "text to depend causally on source-pointer state"
            ),
            on_reject=(
                "do not add renderer complexity; fix prompt-position binding "
                "or recurrent pointer updates before claiming L2 state progress"
            ),
        ),
        "qtrm_numeric_source_pointer_state": GateSpec(
            name="qtrm_numeric_source_pointer_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "numeric-aware prompt source binding must become causal inside "
                "QTRM source-position recurrent state"
            ),
            script="scripts/319_run_qtrm_source_pointer_state_gate.py",
            default_args=profile_args("qtrm_numeric_source_pointer_state", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-source-pointer-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open copy/edit LM renderer gate: require autoregressive text "
                "to depend on numeric-aware source-pointer state, primitive-off, "
                "and numeric-feature-off ablations"
            ),
            on_reject=(
                "numeric-aware L1 does not yet route causally through QTRM; "
                "inspect projector/core binding and recurrent pointer update"
            ),
        ),
        "qtrm_token_numeric_source_pointer_state": GateSpec(
            name="qtrm_token_numeric_source_pointer_state",
            target_level="L2 local gate",
            major_bottleneck=(
                "token-path value-aware numeric binding must become causal "
                "inside QTRM source-position recurrent state"
            ),
            script="scripts/319_run_qtrm_source_pointer_state_gate.py",
            default_args=profile_args(
                "qtrm_token_numeric_source_pointer_state",
                profile,
            ),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-source-pointer-state-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open copy/edit LM renderer gate using canonical token-path "
                "numeric source-pointer state"
            ),
            on_reject=(
                "token-path L1 binding has not yet become QTRM recurrent L2; "
                "inspect token numeric embedding load/training and pointer update"
            ),
        ),
        "qtrm_minimal_depth": GateSpec(
            name="qtrm_minimal_depth",
            target_level="L2 local gate",
            major_bottleneck="minimal QTRM depth scaffold after donorless recurrence L1",
            script="scripts/301_build_qtrm_minimal_depth_gate.py",
            default_args=profile_args("qtrm_minimal_depth", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/qtrm-minimal-depth-gate.md",
            accepted_decisions=("accepted_l2", "accepted"),
            on_accept=(
                "open renderer/canonical-LLM-path gate; primitive executor success "
                "is not yet normal autoregressive text generation"
            ),
            on_reject=(
                "redesign QTRM minimal depth path before renderer, memory, or "
                "metacognition work"
            ),
        ),
        "renderer_canonical_lm": GateSpec(
            name="renderer_canonical_lm",
            target_level="L3 candidate",
            major_bottleneck="bottleneck 4 latent-state to autoregressive text renderer",
            script="scripts/302_build_renderer_canonical_lm_gate.py",
            default_args=profile_args("renderer_canonical_lm", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/renderer-canonical-lm-gate.md",
            accepted_decisions=("accepted_l3_candidate", "accepted"),
            on_accept="promote renderer candidate to broader held-out generation gate",
            on_reject=(
                "renderer remains bottleneck; design a donor-compatible text "
                "renderer before memory/metacognition expansion"
            ),
        ),
        "small_general_reasoning": GateSpec(
            name="small_general_reasoning",
            target_level="L2 local gate / L3 candidate",
            major_bottleneck=(
                "recursive core + state codec + autoregressive final answer path "
                "must beat donor-only on a mixed small reasoning gate"
            ),
            script="scripts/308_run_small_general_reasoning_gate.py",
            default_args=profile_args("small_general_reasoning", profile),
            report_name="report.json",
            wiki_path="docs/wiki/decisions/small-general-reasoning-gate.md",
            accepted_decisions=(
                "accepted_l3_candidate_small_general_reasoning",
                "accepted",
            ),
            on_accept=(
                "promote to broader universal-LLM causal-path gate with more "
                "families, donor-preservation checks, and harder ablations"
            ),
            on_reject=(
                "inspect whether failure is donor-only tie, core_off tie, "
                "state_off tie, or family coverage; fix that axis before "
                "claiming general LLM progress"
            ),
        ),
    }


def default_out_dir(gate: GateSpec, profile: str, out_root: str | Path) -> Path:
    return Path(out_root) / f"{gate.name}_{profile}"


def gate_command(gate: GateSpec, out_dir: str | Path) -> list[str]:
    return [
        sys.executable,
        gate.script,
        "--out-dir",
        str(out_dir),
        *gate.default_args,
    ]


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"missing gate report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def normalize_decision(report: dict[str, Any]) -> str:
    decision = str(report.get("decision") or report.get("status") or "").strip().lower()
    return decision or "unknown"


def is_accepted(report: dict[str, Any], gate: GateSpec) -> bool:
    decision = normalize_decision(report)
    return decision in {item.lower() for item in gate.accepted_decisions}


def _get_nested(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def decisive_metrics(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "eval_metrics.depth8_final_exact",
        "eval_metrics.depth4_final_exact",
        "eval_metrics.depth1_final_exact",
        "eval_metrics.depth2_final_exact",
        "eval_metrics.depth2_state_exact",
        "ablations.state_reset.depth8_final_exact",
        "ablations.state_reset.depth4_final_exact",
        "ablations.state_reset.depth2_final_exact",
        "ablations.op_zero.depth8_final_exact",
        "ablations.op_zero.depth4_final_exact",
        "ablations.op_zero.depth2_final_exact",
        "ablations.op_shuffle.depth8_final_exact",
        "ablations.order_shuffle.depth4_final_exact",
        "ablations.order_shuffle.depth2_final_exact",
        "last_loss",
        "metrics.full_answer_accuracy",
        "metrics.core_off_answer_accuracy",
        "metrics.full_minus_core_off",
        "metrics.donor_forced_choice_accuracy",
        "metrics.donor_greedy_accuracy",
        "metrics.full_minus_donor",
        "metrics.full_generation_accuracy",
        "metrics.core_off_generation_accuracy",
        "metrics.donor_generation_accuracy",
        "metrics.state_off_generation_accuracy",
        "metrics.ablation_generation_accuracy",
        "metrics.full_minus_ablation",
        "metrics.full_minus_state_off",
        "metrics.eval_family_count",
        "full_trace_exact_accuracy",
        "full_value_accuracy",
        "full_step_exact_accuracy",
        "ablation_trace_exact_accuracy",
        "ablation_value_accuracy",
        "ablation_step_exact_accuracy",
        "trace_drop",
        "value_drop",
        "numeric_ablation_value_accuracy",
        "numeric_value_drop",
        "token_numeric_ablation_value_accuracy",
        "token_numeric_value_drop",
        "best_exact_acc",
    )
    metrics: dict[str, Any] = {}
    for key in keys:
        value = _get_nested(report, key) if "." in key else report.get(key)
        if value is not None:
            metrics[key] = value
    return metrics


def append_wiki_result(wiki_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(wiki_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = str(summary["timestamp"])
    metrics = summary.get("decisive_metrics") or {}
    lines = [
        "",
        f"## Runner Result {timestamp}",
        "",
        "```text",
        f"gate: {summary['gate']}",
        f"target_level: {summary['target_level']}",
        f"profile: {summary['profile']}",
        f"decision: {summary['decision']}",
        f"accepted: {summary['accepted']}",
        f"next_action: {summary['next_action']}",
        "```",
        "",
        "Decisive metrics:",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        f"Report: `{summary['report_path']}`",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_gate(
    *,
    gate_name: str,
    profile: str,
    out_root: str | Path,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    write_wiki: bool = False,
) -> dict[str, Any]:
    specs = gate_specs(profile)
    if gate_name not in specs:
        raise ValueError(f"unknown gate: {gate_name}")
    gate = specs[gate_name]
    run_dir = Path(out_dir) if out_dir is not None else default_out_dir(gate, profile, out_root)
    report_path = run_dir / gate.report_name
    command = gate_command(gate, run_dir)
    root = repo_root()
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    exit_code: int | None = None

    if dry_run:
        report = {
            "decision": "dry_run",
            "status": "dry_run",
            "target_level": gate.target_level,
        }
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if skip_existing and report_path.exists():
            report = load_report(report_path)
            exit_code = 0
        else:
            env = dict(os.environ)
            env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            exit_code = int(completed.returncode)
            (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            if exit_code != 0:
                report = {
                    "decision": "command_failed",
                    "status": "failed",
                    "returncode": exit_code,
                }
            else:
                report = load_report(report_path)

    accepted = is_accepted(report, gate)
    decision = normalize_decision(report)
    next_action = gate.on_accept if accepted else gate.on_reject
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "gate": gate.name,
        "target_level": gate.target_level,
        "major_bottleneck": gate.major_bottleneck,
        "profile": profile,
        "command": command,
        "out_dir": str(run_dir),
        "report_path": str(report_path),
        "exit_code": exit_code,
        "decision": decision,
        "accepted": accepted,
        "next_action": next_action,
        "decisive_metrics": decisive_metrics(report),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if write_wiki and not dry_run:
        append_wiki_result(root / gate.wiki_path, summary)
    return summary


def list_gates(profile: str) -> list[dict[str, str]]:
    return [
        {
            "name": gate.name,
            "target_level": gate.target_level,
            "major_bottleneck": gate.major_bottleneck,
            "on_accept": gate.on_accept,
            "on_reject": gate.on_reject,
        }
        for gate in gate_specs(profile).values()
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-click research gate runner. Executes a falsifiable gate, parses "
            "report.json, writes gate_summary.json, and emits the next branch."
        )
    )
    parser.add_argument("--gate", default="donorless_recurrent_depth")
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--out-root", default="local_eval/research_gate_runner")
    parser.add_argument("--out-dir", default=None, help="Override the run directory for one gate.")
    parser.add_argument("--list-gates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-wiki", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.list_gates:
        print(json.dumps(list_gates(args.profile), ensure_ascii=False, indent=2))
        return 0
    summary = run_gate(
        gate_name=args.gate,
        profile=args.profile,
        out_root=args.out_root,
        out_dir=args.out_dir,
        dry_run=bool(args.dry_run),
        skip_existing=bool(args.skip_existing),
        write_wiki=bool(args.write_wiki),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision"] != "command_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
