import importlib.util
from pathlib import Path
import unittest


def load_eval_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "237_eval_qtrm_value_state.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_value_state_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMValueStateEvalTests(unittest.TestCase):
    def test_value_state_targets_mask_unsupported_text(self):
        module = load_eval_script()
        row = {
            "depth_targets": {
                "1": "50002,50004",
                "2": "EMPTY",
                "3": "-12",
            }
        }

        self.assertEqual(
            module.value_state_targets_from_row(row, num_steps=3, max_target_tokens=12),
            [
                [5, 0, 0, 0, 2, 10, 5, 0, 0, 0, 4, -100],
                [-100] * 12,
                [11, 1, 2] + [-100] * 9,
            ],
        )

    def test_score_value_sequences_tracks_token_step_and_trace_exact(self):
        module = load_eval_script()

        report = module.score_value_sequences(
            predicted_sequences=[[5, 0, 0], [1, 2, 3]],
            target_sequences=[[5, 0, -100], [1, 9, -100]],
        )

        self.assertEqual(report["correct_tokens"], 3)
        self.assertEqual(report["total_tokens"], 4)
        self.assertEqual(report["exact_steps"], 1)
        self.assertEqual(report["total_steps"], 2)
        self.assertFalse(report["trace_exact"])

    def test_predicted_value_sequences_handles_disabled_head(self):
        import torch

        module = load_eval_script()

        self.assertEqual(
            module.predicted_value_sequences_from_logits(torch.empty(1, 0, 0, 12)),
            [],
        )


if __name__ == "__main__":
    unittest.main()
