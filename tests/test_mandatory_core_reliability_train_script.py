from pathlib import Path
import unittest

from wgram_lm.config import load_config


class MandatoryCoreReliabilityTrainScriptTests(unittest.TestCase):
    def test_config_keeps_strict_core_and_targets_reliability_preference(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_mandatory_core_reliability_s120.yaml"
        )

        self.assertTrue(cfg.model.core_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_requires_core)
        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertGreaterEqual(cfg.train.loss_preference_weight, 1.0)
        self.assertGreaterEqual(cfg.train.loss_donor_correct_margin_weight, 0.7)
        self.assertIn("mandatory_core_reliability", cfg.train.out_dir)

    def test_runner_builds_train_split_hard_negatives_then_evaluates_heldout(self):
        script = Path(
            "scripts/189_run_mandatory_core_reliability_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("memory_reasoning_synth_train_cases.jsonl", script)
        self.assertIn("188_build_reliability_hard_negative_preferences.py", script)
        self.assertIn("mandatory_core_intervention_preference_s080/last.pt", script)
        self.assertIn("memory_reasoning_heldout_expanded_72.jsonl", script)
        self.assertIn("183_run_mandatory_identity_core_candidate_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
