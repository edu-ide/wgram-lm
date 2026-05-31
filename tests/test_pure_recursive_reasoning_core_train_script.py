from pathlib import Path
import unittest


class PureRecursiveReasoningCoreTrainScriptTests(unittest.TestCase):
    def test_runner_builds_preferences_trains_and_runs_raw_gate(self):
        script = Path("scripts/195_run_pure_recursive_reasoning_core_train.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("scripts/190_build_pure_recursive_reasoning_cases.py", text)
        self.assertIn("scripts/194_build_pure_recursive_reasoning_preferences.py", text)
        self.assertIn("wgram_lm.training.train", text)
        self.assertIn("scripts/193_run_pure_recursive_reasoning_depth_gate.sh", text)
        self.assertIn("loss_canonical_causal_weight", Path("configs/qwen35_2b_4090_pure_recursive_reasoning_core_s160.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("scripts/95_eval_memory_retrieval.py", text)


if __name__ == "__main__":
    unittest.main()
