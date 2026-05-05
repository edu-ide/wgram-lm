from pathlib import Path
import unittest


class CanonicalPlainAnswerTrainScriptTests(unittest.TestCase):
    def test_config_and_runner_use_plain_answer_contract_and_kiss_core_path(self) -> None:
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_canonical_plain_answer_kiss_s120.yaml")
        script = Path("scripts/174_run_canonical_plain_answer_kiss_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertFalse(cfg.model.core_to_text_enabled)
        self.assertFalse(cfg.model.core_context_enabled)
        self.assertGreater(cfg.train.loss_greedy_token_margin_weight, 0.0)
        self.assertTrue(cfg.train.greedy_token_margin_only_donor_errors)
        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertIn("173_build_canonical_plain_answer_data.py", script)
        self.assertIn("memory_reasoning_canonical_plain_answer.jsonl", script)
        self.assertIn("qwen35_2b_4090_donor_residual_s010_1000/last.pt", script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)
        self.assertIn('QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.50}"', script)


if __name__ == "__main__":
    unittest.main()
