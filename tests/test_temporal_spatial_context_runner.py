import unittest
from pathlib import Path


class TemporalSpatialContextRunnerTests(unittest.TestCase):
    def test_runner_builds_train_eval_and_context_ablation_modes(self):
        text = Path("scripts/208_run_temporal_spatial_context_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("scripts/207_build_temporal_spatial_context_cases.py", text)
        self.assertIn("scripts/196_train_pure_recursive_depth_supervised.py", text)
        self.assertIn("scripts/192_eval_raw_intelligence.py", text)
        self.assertIn("scripts/191_build_raw_intelligence_gate.py", text)
        self.assertIn("configs/qwen35_2b_4090_temporal_spatial_context_probe.yaml", text)
        self.assertIn("PYTHON_BIN", text)
        self.assertIn("temporal_spatial_context", text)
        self.assertIn("TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT", text)
        self.assertIn("qtrm_core_steps_8_no_evidence", text)
        self.assertIn("qtrm_core_steps_8_temporal_spatial_off_no_evidence", text)
        self.assertIn("temporal_spatial_context_gate", text)
        self.assertNotIn("scripts/95_eval_memory_retrieval.py", text)


if __name__ == "__main__":
    unittest.main()
