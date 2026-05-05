from pathlib import Path
import unittest

from qtrm_mm.config import load_config


class MandatoryIdentityCoreCandidateScriptTests(unittest.TestCase):
    def test_config_keeps_core_mandatory_with_identity_safe_blend(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_mandatory_identity_core_candidate.yaml"
        )

        self.assertTrue(cfg.model.core_enabled)
        self.assertTrue(cfg.model.core_output_blend_enabled)
        self.assertLess(cfg.model.core_output_blend_init_bias, -10.0)
        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_residual_governor_enabled)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")

    def test_runner_checks_core_and_workspace_ablations(self):
        script = Path("scripts/183_run_mandatory_identity_core_candidate_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_mandatory_identity_core_candidate.yaml", script)
        self.assertIn("intervention_preference_train24_s080/last.pt", script)
        self.assertIn("--mode qtrm_core_off_with_evidence", script)
        self.assertIn("--mode qtrm_workspace_off_with_evidence", script)
        self.assertIn("--baseline-mode qtrm_residual_with_evidence", script)


if __name__ == "__main__":
    unittest.main()
