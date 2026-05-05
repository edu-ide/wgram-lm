from pathlib import Path
import unittest

from qtrm_mm.config import load_config


class MandatoryCoreAnswerBottleneckScriptTests(unittest.TestCase):
    def test_config_forces_answer_residual_through_running_core(self):
        cfg = load_config(
            "configs/qwen35_2b_4090_mandatory_core_answer_bottleneck_causal_s120.yaml"
        )

        self.assertTrue(cfg.model.core_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_requires_core)
        self.assertFalse(cfg.model.answer_bottleneck_requires_workspace_memory)
        self.assertGreater(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("workspace_off", cfg.train.canonical_causal_ablation_modes)
        self.assertEqual(cfg.train.loss_preference_weight, 0.0)
        self.assertIn("mandatory_core_answer_bottleneck", cfg.train.out_dir)

    def test_runner_continues_from_identity_core_checkpoint_and_reuses_gate(self):
        script = Path(
            "scripts/185_run_mandatory_core_answer_bottleneck_causal_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "qwen35_2b_4090_mandatory_core_answer_bottleneck_causal_s120.yaml",
            script,
        )
        self.assertIn("mandatory_identity_core_causal_s080/last.pt", script)
        self.assertIn("memory_reasoning_canonical_plain_answer.jsonl", script)
        self.assertIn("qtrm_mm.training.train", script)
        self.assertIn("183_run_mandatory_identity_core_candidate_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
