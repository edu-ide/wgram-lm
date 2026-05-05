from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("scripts/159_eval_learned_state_answer_loop.py")
    spec = importlib.util.spec_from_file_location("learned_state_answer_loop", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_trace_row_hides_oracle_phase_when_strict() -> None:
    module = _load_script()
    row = module.runtime_trace_row(
        case={"id": "case-1", "question": "Q?"},
        prompt="Prompt",
        workspace_context="Evidence",
        step=2,
        previous_observation="verified_candidate_answer=A",
        strict_runtime_state=True,
    )

    assert row["hide_trace_step_from_input"] is True
    assert row["state_summary"] == module.RUNTIME_STATE_SUMMARY
    assert "verified_candidate_answer=A" == row["previous_observation"]


def test_answer_loop_gate_rejects_without_baseline_gain() -> None:
    module = _load_script()
    records = [
        {"mode": "learned_state_qtrm", "action_success": True},
        {"mode": "learned_state_qtrm_state_off", "action_success": False},
    ]
    summary = {
        "by_mode": {
            "learned_state_qtrm": {"accuracy": 0.5},
            "learned_state_qtrm_state_off": {"accuracy": 0.0},
            "scripted_qtrm_answer_channel": {"accuracy": 0.5},
            "scripted_donor_answer_channel": {"accuracy": 0.5},
        }
    }

    gate = module.build_answer_loop_gate(
        records,
        summary,
        min_gain=0.02,
        min_drop=0.03,
        min_action_success=0.9,
    )

    assert gate["status"] == "rejected"
    assert "learned_state_does_not_beat_scripted_qtrm" in gate["failed_checks"]
    assert "learned_state_does_not_beat_scripted_donor" in gate["failed_checks"]
    assert gate["transition_state_drop"] == 0.5


def test_answer_loop_gate_accepts_gain_drop_and_action_success() -> None:
    module = _load_script()
    records = [
        {"mode": "learned_state_qtrm", "action_success": True},
        {"mode": "learned_state_qtrm", "action_success": True},
        {"mode": "learned_state_qtrm_state_off", "action_success": False},
    ]
    summary = {
        "by_mode": {
            "learned_state_qtrm": {"accuracy": 0.75},
            "learned_state_qtrm_state_off": {"accuracy": 0.25},
            "scripted_qtrm_answer_channel": {"accuracy": 0.5},
            "scripted_donor_answer_channel": {"accuracy": 0.5},
        }
    }

    gate = module.build_answer_loop_gate(
        records,
        summary,
        min_gain=0.02,
        min_drop=0.03,
        min_action_success=0.9,
    )

    assert gate["status"] == "accepted"
    assert gate["failed_checks"] == []


def test_runtime_controller_prompt_keeps_evidence_out_of_visible_prompt() -> None:
    module = _load_script()
    # This test locks the contract behind the answer-loop harness: the
    # controller must ask for RETRIEVE before evidence is visible in the prompt.
    row = module.runtime_trace_row(
        case={"id": "case-1", "question": "What is the code?"},
        prompt="Question: What is the code?",
        workspace_context="MemoryOS evidence\nSOURCE=a.md\nThe code is VX-913.",
        step=0,
        previous_observation="",
        strict_runtime_state=True,
    )

    assert "VX-913" not in row["chat_prompt"]
    assert "VX-913" in row["workspace_context"]
