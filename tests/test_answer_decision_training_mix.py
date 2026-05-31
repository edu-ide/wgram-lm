from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    path = Path("scripts/162_build_answer_decision_training_mix.py")
    spec = importlib.util.spec_from_file_location("answer_decision_mix", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_labels_only_unknown_improving_blocks(tmp_path: Path) -> None:
    module = _load_script()
    cases = [
        {
            "id": "missing",
            "question": "What is the redacted code?",
            "category": "negative_missing_synth",
            "answer_aliases": ["UNKNOWN", "unknown"],
            "evidence": [{"source": "redacted.md", "text": "The code is redacted."}],
            "distractors": [{"source": "old.md", "text": "The old code was fake."}],
        },
        {
            "id": "positive-wrong",
            "question": "What is the code?",
            "category": "multi_hop_synth",
            "answer_aliases": ["gold"],
            "evidence": [{"source": "gold.md", "text": "The code is gold."}],
            "distractors": [{"source": "fake.md", "text": "The code is fake."}],
        },
    ]
    records = [
        {
            "id": "missing",
            "mode": "qtrm_residual_with_evidence",
            "question": "What is the redacted code?",
            "completion": "Answer: fake",
            "raw_completion": "Answer: fake",
            "answer_aliases": ["UNKNOWN", "unknown"],
            "expected_unknown": True,
            "hit": False,
        },
        {
            "id": "missing",
            "mode": "qtrm_evidence_span_reader_off_with_evidence",
            "question": "What is the redacted code?",
            "completion": "Answer: UNKNOWN",
            "raw_completion": "Answer: UNKNOWN",
            "answer_aliases": ["UNKNOWN", "unknown"],
            "expected_unknown": True,
            "hit": True,
        },
        {
            "id": "positive-wrong",
            "mode": "qtrm_residual_with_evidence",
            "question": "What is the code?",
            "completion": "Answer: fake",
            "raw_completion": "Answer: fake",
            "answer_aliases": ["gold"],
            "expected_unknown": False,
            "hit": False,
        },
    ]
    cases_path = tmp_path / "cases.jsonl"
    records_path = tmp_path / "records.jsonl"
    out_path = tmp_path / "mix.jsonl"
    cases_path.write_text(
        "\n".join(json.dumps(row) for row in cases) + "\n",
        encoding="utf-8",
    )
    records_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )

    module.main(
        [
            "--cases-jsonl",
            str(cases_path),
            "--records-jsonl",
            str(records_path),
            "--out-jsonl",
            str(out_path),
            "--evidence-mode",
            "all",
            "--retrieval-top-k",
            "2",
            "--record-mode",
            "qtrm_residual_with_evidence",
        ]
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert [row["answer_decision_target"] for row in rows] == [1.0, 0.0]
    assert all("answer_decision_features" in row for row in rows)
    assert all(len(row["answer_decision_features"]) == 23 for row in rows)
    assert rows[0]["answer_decision_feature_names"][0] == "support_prob"
    assert all("Candidate answer:" in row["answer"] for row in rows)
    assert all("MemoryOS evidence" in row["prompt"] for row in rows)
    assert rows[0]["metadata"]["label_reason"] == "unknown_block_improves"


def test_builder_balances_positive_answer_decision_weights() -> None:
    module = _load_script()
    rows = [
        {"answer_decision_target": 1.0, "answer_decision_sample_weight": 1.0},
        {"answer_decision_target": 0.0, "answer_decision_sample_weight": 1.0},
        {"answer_decision_target": 0.0, "answer_decision_sample_weight": 1.0},
    ]

    module.apply_answer_decision_class_balance(rows)

    assert rows[0]["answer_decision_sample_weight"] == 2.0
    assert rows[1]["answer_decision_sample_weight"] == 1.0
    assert rows[2]["answer_decision_sample_weight"] == 1.0


def test_answer_decision_head_config_and_runner_are_wired() -> None:
    from wgram_lm.config import load_config

    cfg = load_config("configs/qwen35_2b_4090_answer_decision_head_truthcal_s200.yaml")
    runner = Path("scripts/163_run_answer_decision_head_train.sh").read_text(
        encoding="utf-8"
    )

    assert cfg.model.answer_decision_head_enabled
    assert cfg.model.answer_decision_feature_dim == 23
    assert cfg.train.loss_answer_decision_weight == 1.0
    assert cfg.train.trainable_param_policy == "answer_decision_head_only"
    assert cfg.train.workspace_evidence_injection
    assert "162_build_answer_decision_training_mix.py" in runner
    assert "--init-checkpoint \"$INIT_CHECKPOINT\"" in runner
    assert "--model-answer-decision" in runner
