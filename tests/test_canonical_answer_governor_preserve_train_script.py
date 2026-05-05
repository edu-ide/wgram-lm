from pathlib import Path
import unittest

from qtrm_mm.config import load_config


class CanonicalAnswerGovernorPreserveTrainScriptTests(unittest.TestCase):
    def test_config_adds_donor_correct_preservation_loss(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_canonical_answer_governor_preserve_s120.yaml"
        )

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_residual_governor_enabled)
        self.assertGreater(cfg.train.loss_donor_correct_margin_weight, 0.0)
        self.assertGreater(cfg.train.donor_correct_margin, 0.0)
        self.assertTrue(cfg.train.greedy_token_margin_only_donor_errors)
        self.assertIn("governor_preserve", cfg.train.out_dir)

    def test_runner_finetunes_from_answer_governor_and_runs_strict_gate(self):
        script = Path(
            "scripts/176_run_canonical_answer_governor_preserve_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("qwen35_2b_4090_canonical_answer_governor_preserve_s120.yaml", script)
        self.assertIn("qwen35_2b_4090_canonical_answer_governor_s120/last.pt", script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)
        self.assertIn("canonical_answer_governor_preserve_s120_answer_gate", script)


if __name__ == "__main__":
    unittest.main()
