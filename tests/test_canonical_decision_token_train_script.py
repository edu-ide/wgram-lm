from pathlib import Path
import unittest


class CanonicalDecisionTokenTrainScriptTests(unittest.TestCase):
    def test_runner_builds_decision_token_data_and_uses_canonical_config(self) -> None:
        script = Path("scripts/171_run_canonical_decision_token_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("170_build_canonical_decision_token_data.py", script)
        self.assertIn("memory_reasoning_canonical_decision_tokens.jsonl", script)
        self.assertIn("qwen35_2b_4090_canonical_decision_tokens_s120.yaml", script)
        self.assertIn("qwen35_2b_4090_donor_residual_s010_1000/last.pt", script)
        self.assertIn('QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.30}"', script)
        self.assertIn("166_run_canonical_ssot_answer_gate.sh", script)


if __name__ == "__main__":
    unittest.main()
