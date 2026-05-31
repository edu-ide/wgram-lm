from pathlib import Path
import unittest

from wgram_lm.config import load_config


class CanonicalAnswerGovernorTrainScriptTests(unittest.TestCase):
    def test_config_enables_latent_renderer_with_answer_residual_governor(self):
        cfg = load_config("configs/qwen35_2b_4090_canonical_answer_governor_s120.yaml")

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertFalse(cfg.model.answer_bottleneck_requires_workspace_memory)
        self.assertTrue(cfg.model.answer_residual_governor_enabled)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertGreater(cfg.train.loss_answer_residual_governor_weight, 0.0)
        self.assertTrue(cfg.train.greedy_token_margin_only_donor_errors)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertIn("canonical_answer_governor", cfg.train.out_dir)

    def test_runner_builds_plain_answer_data_and_runs_canonical_gate(self):
        script = Path("scripts/175_run_canonical_answer_governor_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("173_build_canonical_plain_answer_data.py", script)
        self.assertIn("memory_reasoning_canonical_plain_answer.jsonl", script)
        self.assertIn("qwen35_2b_4090_canonical_answer_governor_s120.yaml", script)
        self.assertIn("canonical_ssot_core_answer_bottleneck_s150/last.pt", script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)
        self.assertIn("qtrm_answer_residual_governor_off_with_evidence", script)


if __name__ == "__main__":
    unittest.main()
