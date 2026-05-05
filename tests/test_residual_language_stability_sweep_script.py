from pathlib import Path


def test_residual_language_stability_sweep_runs_metric_gate_per_scale() -> None:
    script = Path("scripts/152_run_residual_language_stability_sweep.sh").read_text(encoding="utf-8")

    assert "SCALES=${SCALES:-0.0 0.05 0.10}" in script
    assert "scripts/92_eval_qtrm_logits.py" in script
    assert "--donor-logits-scale 1.0" in script
    assert "--qtrm-logits-scale \"$scale\"" in script
    assert "--suppress-visible-reasoning-tokens" in script
    assert "--no-repeat-ngram-size \"$NO_REPEAT_NGRAM_SIZE\"" in script
    assert "--stop-after-sentence" in script
    assert "scripts/147_summarize_generation_format.py" in script
    assert "summary.jsonl" in script
