from pathlib import Path
import unittest

from qtrm_mm.config import load_config


class MandatoryIdentityCoreCausalTrainScriptTests(unittest.TestCase):
    def test_config_trains_mandatory_core_with_causal_ablations(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_mandatory_identity_core_causal_s080.yaml"
        )

        self.assertTrue(cfg.model.core_enabled)
        self.assertTrue(cfg.model.core_output_blend_enabled)
        self.assertGreaterEqual(cfg.model.core_output_blend_init_bias, -12.0)
        self.assertLess(cfg.model.core_output_blend_init_bias, 0.0)
        self.assertGreater(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("workspace_off", cfg.train.canonical_causal_ablation_modes)
        self.assertEqual(cfg.train.loss_preference_weight, 0.0)
        self.assertIn("mandatory_identity_core_causal", cfg.train.out_dir)

    def test_runner_trains_from_intervention_checkpoint_and_reuses_gate(self):
        script = Path("scripts/184_run_mandatory_identity_core_causal_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "qwen35_2b_4090_mandatory_identity_core_causal_s080.yaml", script
        )
        self.assertIn("memory_reasoning_canonical_plain_answer.jsonl", script)
        self.assertIn("intervention_preference_train24_s080/last.pt", script)
        self.assertIn("qtrm_mm.training.train", script)
        self.assertIn("183_run_mandatory_identity_core_candidate_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
