from pathlib import Path
import unittest

from wgram_lm.config import load_config


class CanonicalAnswerPreferenceTrainScriptTests(unittest.TestCase):
    def test_config_enables_preference_training_on_ssot_answer_path(self):
        cfg = load_config("configs/qwen35_2b_4090_canonical_answer_preference_s160.yaml")

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_residual_governor_enabled)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertGreater(cfg.train.preference_margin, 0.0)
        self.assertEqual(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertEqual(cfg.train.canonical_causal_ablation_modes, [])
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertIn("answer_preference", cfg.train.out_dir)

    def test_runner_builds_preferences_and_runs_strict_gate(self):
        script = Path("scripts/178_run_canonical_answer_preference_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("177_build_canonical_plain_answer_preferences.py", script)
        self.assertIn("memory_reasoning_canonical_plain_answer_preferences.jsonl", script)
        self.assertIn("qwen35_2b_4090_canonical_answer_preference_s160.yaml", script)
        self.assertIn("canonical_answer_governor_preserve_s120/last.pt", script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
