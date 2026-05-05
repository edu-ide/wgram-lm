from pathlib import Path
import unittest


class PureRecursiveReasoningGateRunnerTests(unittest.TestCase):
    def test_runner_uses_raw_eval_and_gate_without_memoryos_eval(self):
        script = Path("scripts/193_run_pure_recursive_reasoning_depth_gate.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("scripts/190_build_pure_recursive_reasoning_cases.py", text)
        self.assertIn("scripts/192_eval_raw_intelligence.py", text)
        self.assertIn("scripts/191_build_raw_intelligence_gate.py", text)
        self.assertIn("--gate-type pure_recursive_reasoning", text)
        self.assertIn("SCORING", text)
        self.assertIn("causal_forced_choice", text)
        self.assertNotIn("scripts/95_eval_memory_retrieval.py", text)


if __name__ == "__main__":
    unittest.main()
