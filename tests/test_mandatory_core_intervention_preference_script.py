from pathlib import Path
import unittest

from wgram_lm.config import load_config


class MandatoryCoreInterventionPreferenceScriptTests(unittest.TestCase):
    def test_config_keeps_strict_core_and_enables_preference(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_mandatory_core_intervention_preference_s080.yaml"
        )

        self.assertTrue(cfg.model.core_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_requires_core)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertEqual(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertIn("mandatory_core_intervention_preference", cfg.train.out_dir)

    def test_runner_builds_clean_preferences_before_training(self):
        script = Path(
            "scripts/187_run_mandatory_core_intervention_preference_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("186_build_clean_intervention_preferences.py", script)
        self.assertIn("memory_reasoning_intervention_preferences_clean_train24.jsonl", script)
        self.assertIn("mandatory_core_answer_bottleneck_causal_s120/last.pt", script)
        self.assertIn("183_run_mandatory_identity_core_candidate_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
