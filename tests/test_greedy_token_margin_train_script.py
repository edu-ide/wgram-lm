from pathlib import Path
import unittest


class GreedyTokenMarginTrainScriptTests(unittest.TestCase):
    def test_config_and_runner_wire_greedy_margin_to_canonical_gate(self) -> None:
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_canonical_greedy_margin_s120.yaml")
        script = Path("scripts/172_run_canonical_greedy_margin_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertGreater(cfg.train.loss_greedy_token_margin_weight, 0.0)
        self.assertGreater(cfg.train.greedy_token_margin, 0.0)
        self.assertTrue(cfg.train.greedy_token_margin_only_donor_errors)
        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertIn("memory_reasoning_canonical_decision_tokens.jsonl", script)
        self.assertIn("qwen35_2b_4090_canonical_decision_tokens_s120/last.pt", script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)
        self.assertIn('QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.50}"', script)


if __name__ == "__main__":
    unittest.main()
