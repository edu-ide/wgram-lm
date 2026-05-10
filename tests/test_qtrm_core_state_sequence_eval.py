import importlib.util
from pathlib import Path
import unittest


def load_eval_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "236_eval_qtrm_core_state_sequence.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_core_state_sequence_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMCoreStateSequenceEvalTests(unittest.TestCase):
    def test_sequence_targets_from_row_pads_and_masks_missing_depths(self):
        module = load_eval_script()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"alpha": [11, 12], "beta": [21]}[text]

        row = {"depth_targets": {"1": "alpha", "3": "beta"}}

        self.assertEqual(
            module.sequence_targets_from_row(
                FakeTokenizer(),
                row,
                num_steps=3,
                max_target_tokens=3,
            ),
            [[11, 12, -100], [-100, -100, -100], [21, -100, -100]],
        )

    def test_score_sequence_predictions_tracks_token_step_and_trace_exact(self):
        module = load_eval_script()

        report = module.score_sequence_predictions(
            predicted_sequences=[[11, 12, 0], [99, 0, 0], [21, 22, 0]],
            target_sequences=[[11, 12, -100], [-100, -100, -100], [21, 23, -100]],
        )

        self.assertEqual(report["correct_tokens"], 3)
        self.assertEqual(report["total_tokens"], 4)
        self.assertEqual(report["exact_steps"], 1)
        self.assertEqual(report["total_steps"], 2)
        self.assertFalse(report["trace_exact"])

    def test_predicted_token_sequences_from_logits_allows_disabled_head(self):
        import torch

        module = load_eval_script()

        self.assertEqual(
            module.predicted_token_sequences_from_logits(torch.empty(1, 0, 0, 128)),
            [],
        )

    def test_select_sequence_logits_uses_requested_output_key(self):
        import torch

        module = load_eval_script()
        outputs = {
            "core_depth_text_logits": torch.zeros(1, 2, 3, 5),
            "transition_state_sequence_logits": torch.ones(1, 2, 4, 5),
        }

        selected = module.select_sequence_logits(
            outputs,
            logits_key="transition_state_sequence_logits",
        )

        self.assertEqual(tuple(selected.shape), (1, 2, 4, 5))
        self.assertEqual(float(selected.sum()), 40.0)


if __name__ == "__main__":
    unittest.main()
