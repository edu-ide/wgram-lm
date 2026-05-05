import importlib.util
from pathlib import Path
import unittest


def load_eval_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "231_eval_qtrm_transition_state_text.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_transition_state_text_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMTransitionStateTextEvalTests(unittest.TestCase):
    def test_first_token_targets_from_row_uses_depth_targets_and_masks_missing(self):
        module = load_eval_script()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {" alpha": [11], " beta": [22]}[text]

        row = {"depth_targets": {"1": "alpha", "4": "beta"}}

        self.assertEqual(
            module.first_token_targets_from_row(FakeTokenizer(), row, num_steps=4),
            [11, -100, -100, 22],
        )

    def test_score_token_predictions_counts_masked_step_accuracy(self):
        module = load_eval_script()

        report = module.score_token_predictions(
            predicted_tokens=[11, 12, 22, 0],
            target_tokens=[11, -100, 23, 0],
        )

        self.assertEqual(report["correct_steps"], 2)
        self.assertEqual(report["total_steps"], 3)
        self.assertAlmostEqual(report["step_accuracy"], 2 / 3)
        self.assertFalse(report["trace_exact"])

    def test_predicted_tokens_from_logits_allows_disabled_text_head(self):
        import torch

        module = load_eval_script()

        self.assertEqual(
            module.predicted_tokens_from_logits(torch.empty(1, 0, 128)),
            [],
        )


if __name__ == "__main__":
    unittest.main()
