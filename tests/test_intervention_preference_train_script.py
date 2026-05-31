from pathlib import Path
import unittest

from wgram_lm.config import load_config


class InterventionPreferenceTrainScriptTests(unittest.TestCase):
    def test_config_trains_on_policy_intervention_preferences(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_intervention_preference_train24_s080.yaml"
        )

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_residual_governor_enabled)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertGreater(cfg.train.loss_donor_correct_margin_weight, 0.0)
        self.assertEqual(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertEqual(cfg.train.canonical_causal_ablation_modes, [])
        self.assertIn("intervention_preference_train24", cfg.train.out_dir)

    def test_runner_uses_train_split_intervention_data_and_heldout_gate(self):
        script = Path("scripts/181_run_intervention_preference_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("memory_reasoning_intervention_preferences_train24.jsonl", script)
        self.assertIn("qwen35_2b_4090_intervention_preference_train24_s080.yaml", script)
        self.assertIn("canonical_answer_preference_s160/last.pt", script)
        self.assertIn("data/eval/memory_reasoning_heldout_expanded_72.jsonl", script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
