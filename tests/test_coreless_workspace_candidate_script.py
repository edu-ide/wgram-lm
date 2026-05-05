from pathlib import Path
import unittest

from qtrm_mm.config import load_config


class CorelessWorkspaceCandidateScriptTests(unittest.TestCase):
    def test_config_disables_recursive_core_for_canonical_path(self):
        cfg = load_config("configs/qwen35_2b_4090_coreless_workspace_answer_candidate.yaml")

        self.assertFalse(cfg.model.core_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_residual_governor_enabled)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")

    def test_runner_uses_active_workspace_only_gate(self):
        script = Path("scripts/182_run_coreless_workspace_answer_candidate_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qwen35_2b_4090_coreless_workspace_answer_candidate.yaml", script)
        self.assertIn("intervention_preference_train24_s080/last.pt", script)
        self.assertIn("--baseline-mode qtrm_residual_with_evidence", script)
        self.assertIn("--critical-mode qtrm_workspace_off_with_evidence", script)
        self.assertIn("--comparison-mode donor_only_with_evidence", script)


if __name__ == "__main__":
    unittest.main()
