from pathlib import Path


def test_language_gates_write_all_eval_reports_to_tensorboard():
    script = Path("scripts/545_run_prefixlm_language_gates_dgx.sh").read_text(encoding="utf-8")

    assert '--raw-json "${OUT_DIR}/language_heldout_loss.json"' in script
    assert "--prefix eval/language_heldout" in script
    assert '--raw-json "${OUT_DIR}/multilingual_generation_probe.json"' in script
    assert "--prefix eval/multilingual_generation" in script
    assert '--raw-json "${OUT_DIR}/raw_intelligence_suite.json"' in script
    assert "--prefix eval/raw_intelligence" in script
