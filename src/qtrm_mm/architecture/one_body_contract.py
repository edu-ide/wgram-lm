from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BridgeContractFields:
    answer_readback_mode: str = "none"
    cot_anchor_loss_weight: float = 0.0
    workspace_selector_critic_weight: float = 0.0
    workspace_selector_final_ce_critic_weight: float = 0.0

    def enabled_field_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.answer_readback_mode != "none":
            names.append("answer_readback_mode")
        if self.cot_anchor_loss_weight > 0.0:
            names.append("cot_anchor_loss_weight")
        if self.workspace_selector_critic_weight > 0.0:
            names.append("workspace_selector_critic_weight")
        if self.workspace_selector_final_ce_critic_weight > 0.0:
            names.append("workspace_selector_final_ce_critic_weight")
        return tuple(names)

    def has_diagnostic_bridge(self) -> bool:
        return bool(self.enabled_field_names())


@dataclass(frozen=True)
class PastSuccessPreflightFields:
    decoder_latent_mode: str = ""
    steps: int = 0
    past_success_preflight_min_steps: int = 1000
    past_success_report_json: str = ""
    past_success_restoration_gate_json: str = ""
    allow_missing_past_success_preflight: bool = False
    acknowledge_past_success_restoration_gap: bool = False

    def is_long_one_body_run(self) -> bool:
        return (
            self.decoder_latent_mode == "one_body"
            and self.steps >= max(1, self.past_success_preflight_min_steps)
        )


def collect_bridge_contract_fields(args: Any) -> BridgeContractFields:
    return BridgeContractFields(
        answer_readback_mode=str(getattr(args, "answer_readback_mode", "none")),
        cot_anchor_loss_weight=float(getattr(args, "cot_anchor_loss_weight", 0.0)),
        workspace_selector_critic_weight=float(getattr(args, "workspace_selector_critic_weight", 0.0)),
        workspace_selector_final_ce_critic_weight=float(
            getattr(args, "workspace_selector_final_ce_critic_weight", 0.0)
        ),
    )


def collect_past_success_preflight_fields(args: Any) -> PastSuccessPreflightFields:
    return PastSuccessPreflightFields(
        decoder_latent_mode=str(getattr(args, "decoder_latent_mode", "")),
        steps=int(getattr(args, "steps", 0) or 0),
        past_success_preflight_min_steps=int(getattr(args, "past_success_preflight_min_steps", 1000) or 1000),
        past_success_report_json=str(getattr(args, "past_success_report_json", "")),
        past_success_restoration_gate_json=str(getattr(args, "past_success_restoration_gate_json", "")),
        allow_missing_past_success_preflight=bool(getattr(args, "allow_missing_past_success_preflight", False)),
        acknowledge_past_success_restoration_gap=bool(
            getattr(args, "acknowledge_past_success_restoration_gap", False)
        ),
    )


def _load_past_success_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        raise ValueError(f"past-success preflight report does not exist: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("report_type") != "past_success_doubt_loop":
        raise ValueError(f"invalid past-success preflight report: {report_path}")
    recommended = payload.get("recommended_comparison_row")
    if not isinstance(recommended, dict) or not all(
        str(recommended.get(key, "")).strip()
        for key in (
            "old_success",
            "exact_metric",
            "causal_ingredient",
            "missing_in_current_run",
            "smallest_restoration_test",
        )
    ):
        raise ValueError(f"past-success preflight report has no complete recommended comparison row: {report_path}")
    return payload


def _restoration_gate_satisfies_gap(path: str) -> bool:
    if not path:
        return False
    gate_path = Path(path)
    if not gate_path.exists():
        raise ValueError(f"past-success restoration gate does not exist: {gate_path}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("gate_type") != "past_success_restoration_gate":
        raise ValueError(f"invalid past-success restoration gate: {gate_path}")
    if not bool(payload.get("all_required_signals_present", False)):
        return False
    recommendation = str(payload.get("current_checkpoint_recommendation", ""))
    if recommendation in {"not_interpretable", "do_not_promote_current_checkpoint"}:
        return False
    return True


def validate_past_success_preflight_contract(args: Any) -> None:
    fields = collect_past_success_preflight_fields(args)
    if not fields.is_long_one_body_run():
        return
    if fields.allow_missing_past_success_preflight:
        return
    if not fields.past_success_report_json:
        raise ValueError(
            "Long one-body language runs require a past-success preflight report. "
            "Build one with scripts/562_build_past_success_doubt_report.py and pass "
            "--past-success-report-json, or explicitly pass "
            "--allow-missing-past-success-preflight for a diagnostic-only run."
        )
    report = _load_past_success_report(fields.past_success_report_json)
    if (
        str(report.get("launch_recommendation", ""))
        == "do_not_launch_long_run_until_restoration_gate_exists"
        and not fields.acknowledge_past_success_restoration_gap
        and not _restoration_gate_satisfies_gap(fields.past_success_restoration_gate_json)
    ):
        raise ValueError(
            "The past-success preflight report says a restoration gate gap remains. "
            "Run the recommended small restoration gate first, or pass "
            "--acknowledge-past-success-restoration-gap only if this is an explicit "
            "diagnostic override."
        )


def validate_one_body_architecture_contract(args: Any) -> None:
    fields = collect_bridge_contract_fields(args)
    if fields.has_diagnostic_bridge() and not bool(getattr(args, "allow_diagnostic_bridge_experiment", False)):
        enabled = ", ".join(fields.enabled_field_names())
        raise ValueError(
            "Stage99-style diagnostic bridge/readback experiments are blocked by default. "
            "The main architecture path must be HRM-Text-style one-body: reader -> "
            "recurrent thought -> same decoder/LM head. If you are intentionally "
            "reproducing a diagnostic bridge ablation, pass --allow-diagnostic-bridge-experiment. "
            f"Enabled diagnostic fields: {enabled}."
        )
    validate_past_success_preflight_contract(args)
