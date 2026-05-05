from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def test_transition_state_controller_requires_recurrence_when_inputs_repeat() -> None:
    from qtrm_mm.agentic.transition_controller import (
        TransitionStateController,
        transition_action_loss,
    )

    torch.manual_seed(0)
    controller = TransitionStateController(
        d_model=8,
        num_actions=4,
        hidden_dim=16,
        use_prev_action=False,
    )
    features = torch.zeros(12, 3, 8)
    targets = torch.tensor([[1, 2, 3]] * 12)
    opt = torch.optim.AdamW(controller.parameters(), lr=0.02)

    for _ in range(120):
        out = controller(features)
        loss, _ = transition_action_loss(out["action_logits"], targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    full = controller.predict_autoregressive(features, reset_each_step=False)
    reset = controller.predict_autoregressive(features, reset_each_step=True)
    full_acc = (full["action_logits"].argmax(dim=-1) == targets).float().mean()
    reset_acc = (reset["action_logits"].argmax(dim=-1) == targets).float().mean()

    assert float(full_acc) > 0.95
    assert float(reset_acc) < 0.50


def test_transition_state_controller_accepts_explicit_transition_state() -> None:
    from qtrm_mm.agentic.transition_controller import TransitionStateController

    controller = TransitionStateController(
        d_model=8,
        num_actions=4,
        hidden_dim=16,
        transition_state_dim=3,
        use_prev_action=False,
    )
    features = torch.zeros(2, 3, 8)
    transition_state_features = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    out = controller(
        features,
        transition_state_features=transition_state_features,
        reset_each_step=True,
    )
    pred = controller.predict_autoregressive(
        features,
        transition_state_features=transition_state_features,
        reset_each_step=True,
    )

    assert out["action_logits"].shape == (2, 3, 4)
    assert pred["action_logits"].shape == (2, 3, 4)


def test_transition_state_predictor_trains_against_state_targets() -> None:
    from qtrm_mm.agentic.transition_controller import (
        TransitionStatePredictor,
        transition_state_prediction_loss,
    )

    torch.manual_seed(0)
    predictor = TransitionStatePredictor(d_model=6, state_dim=3, hidden_dim=12)
    features = torch.randn(4, 2, 6)
    targets = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    opt = torch.optim.AdamW(predictor.parameters(), lr=0.05)

    first_loss = None
    last_loss = None
    for _ in range(80):
        out = predictor(features)
        loss, metrics = transition_state_prediction_loss(
            out["transition_state_logits"],
            targets,
        )
        if first_loss is None:
            first_loss = float(loss.detach().cpu().item())
        last_loss = float(loss.detach().cpu().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert first_loss is not None
    assert last_loss is not None
    assert last_loss < first_loss
    assert float(metrics.binary_accuracy) > 0.80


def _load_train_script():
    path = Path("scripts/158_train_transition_state_controller.py")
    spec = importlib.util.spec_from_file_location("transition_state_train_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trace_sequence_collate_groups_rows_by_task(tmp_path: Path) -> None:
    module = _load_train_script()
    from qtrm_mm.data.jsonl_dataset import HashTokenizer

    rows = [
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 1,
            "chat_prompt": "Question?",
            "workspace_context": "Evidence.",
            "state_summary": "Same state.",
            "hide_trace_step_from_input": True,
            "hide_previous_observation_from_input": True,
            "previous_observation": "",
            "action_target": "VERIFY_EVIDENCE",
        },
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 0,
            "chat_prompt": "Question?",
            "workspace_context": "Evidence.",
            "state_summary": "Same state.",
            "hide_trace_step_from_input": True,
            "action_target": "RETRIEVE_MEMORY",
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    sequences = module.read_trace_sequences([path])
    batch = module.collate_trace_sequences(
        sequences,
        tokenizer=HashTokenizer(vocab_size=128),
        seq_len=32,
    )

    assert len(sequences) == 1
    assert [row["step"] for row in sequences[0]] == [0, 1]
    assert batch["input_ids"].shape == (1, 2, 32)
    assert batch["action_targets"].tolist()[0][:2] == [1, 2]
    assert batch["sequence_mask"].tolist() == [[True, True]]
    assert batch["transition_state_features"].shape == (1, 2, module.TRANSITION_STATE_DIM)


def test_trace_sequence_collate_builds_previous_observation_state(tmp_path: Path) -> None:
    module = _load_train_script()
    from qtrm_mm.data.jsonl_dataset import HashTokenizer

    rows = [
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 0,
            "chat_prompt": "Question?",
            "workspace_context": "Evidence.",
            "controller_signal": [0.0, 0.0],
            "controller_world_model_signal": 0.0,
            "controller_verifier_signal": 0.0,
            "action_target": "RETRIEVE_MEMORY",
            "observation": "MemoryOS evidence\nSOURCE=a.md\nfact",
            "reward": 0.0,
        },
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 1,
            "chat_prompt": "Question?",
            "workspace_context": "Evidence.",
            "previous_observation": "MemoryOS evidence\nSOURCE=a.md\nfact",
            "controller_signal": [1.0, 0.0],
            "controller_world_model_signal": 1.0,
            "controller_verifier_signal": 0.0,
            "action_target": "VERIFY_EVIDENCE",
            "observation": "candidate_answer=alpha",
            "reward": 1.0,
        },
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 2,
            "chat_prompt": "Question?",
            "workspace_context": "Evidence.",
            "previous_observation": "verified_candidate_answer=alpha",
            "controller_signal": [1.0, 1.0],
            "controller_world_model_signal": 1.0,
            "controller_verifier_signal": 1.0,
            "action_target": "ANSWER",
            "observation": "alpha",
            "reward": 1.0,
        },
    ]
    batch = module.collate_trace_sequences(
        [rows],
        tokenizer=HashTokenizer(vocab_size=128),
        seq_len=32,
    )
    state = batch["transition_state_features"][0]

    assert state.shape == (3, module.TRANSITION_STATE_DIM)
    assert state[0].tolist() == [0.0] * module.TRANSITION_STATE_DIM
    assert float(state[1, 0]) == 1.0
    assert float(state[1, 1]) == 1.0
    assert float(state[1, 5]) > 0.0
    assert float(state[2, 3]) == 1.0
    assert float(state[2, 6]) == 1.0
    assert float(state[2, 7]) == 1.0
    assert float(state[2, 8]) == 0.0


def test_runtime_state_training_row_removes_phase_oracle() -> None:
    module = _load_train_script()
    row = module.runtime_state_training_row(
        {
            "step": 2,
            "state_summary": "Candidate answer has been verified; emit the final answer.",
            "previous_observation": "verified_candidate_answer=alpha",
        }
    )

    assert row["hide_trace_step_from_input"] is True
    assert row["state_summary"] == module.RUNTIME_STATE_SUMMARY
    assert row["previous_observation"] == "verified_candidate_answer=alpha"


def test_trace_sequence_reader_keeps_augmented_task_variants(tmp_path: Path) -> None:
    module = _load_train_script()
    rows = []
    for workspace in ["Evidence A.", "Evidence B."]:
        rows.extend(
            [
                {
                    "type": "trace_replay",
                    "task_id": "case-1",
                    "step": 0,
                    "chat_prompt": "Question?",
                    "workspace_context": workspace,
                    "action_target": "RETRIEVE_MEMORY",
                },
                {
                    "type": "trace_replay",
                    "task_id": "case-1",
                    "step": 1,
                    "chat_prompt": "Question?",
                    "workspace_context": workspace,
                    "action_target": "VERIFY_EVIDENCE",
                },
            ]
        )
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    sequences = module.read_trace_sequences([path])

    assert len(sequences) == 2
    assert [[row["step"] for row in sequence] for sequence in sequences] == [[0, 1], [0, 1]]


def test_trace_sequence_reader_dedupes_duplicate_steps(tmp_path: Path) -> None:
    module = _load_train_script()
    rows = [
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 0,
            "action_target": "RETRIEVE_MEMORY",
        },
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 0,
            "action_target": "RETRIEVE_MEMORY",
        },
        {
            "type": "trace_replay",
            "task_id": "case-1",
            "step": 1,
            "action_target": "VERIFY_EVIDENCE",
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    sequences = module.read_trace_sequences([path])

    assert len(sequences) == 1
    assert [row["step"] for row in sequences[0]] == [0, 1]
