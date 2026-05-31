from pathlib import Path


def test_truth_gate_calibration_config_trains_span_reader_and_truth_heads() -> None:
    from wgram_lm.config import load_config

    cfg = load_config("configs/qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml")

    assert cfg.model.evidence_span_reader_enabled
    assert cfg.model.evidence_bottleneck_enabled
    assert cfg.train.trainable_param_policy == "evidence_span_reader_only"
    assert cfg.train.loss_evidence_span_reader_weight > 0.0
    assert cfg.train.loss_logical_evidence_weight >= 1.0
    assert cfg.train.loss_causal_evidence_gate_weight >= 1.0
    assert "truthcal" in cfg.train.out_dir


def test_truth_gate_calibration_runner_continues_from_span_reader_checkpoint() -> None:
    script = Path("scripts/154_run_truth_gate_calibration_train.sh").read_text(encoding="utf-8")

    assert "qwen35_2b_4090_evidence_span_reader_truthcal_s300.yaml" in script
    assert "qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500/last.pt" in script
    assert "build_evidence_span_reader_training_mix.py" in script
    assert "--init-checkpoint \"$INIT_CHECKPOINT\"" in script
    assert "TRUTH_GATE=1" in script
    assert "scripts/153_run_reasoning_safe_span_copy_gate.sh" in script
