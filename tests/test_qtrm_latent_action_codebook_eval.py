import importlib.util
from pathlib import Path
import unittest


def load_eval_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "230_eval_qtrm_latent_action_codebook.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_latent_action_codebook_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMLatentActionCodebookEvalTests(unittest.TestCase):
    def test_code_targets_from_row_uses_depth_order_and_masks_missing(self):
        module = load_eval_script()

        row = {"transition_state_codes": {"1": 0, "4": 3}}

        self.assertEqual(module.code_targets_from_row(row, num_steps=4), [0, -100, -100, 3])

    def test_score_code_predictions_counts_masked_step_accuracy(self):
        module = load_eval_script()

        report = module.score_code_predictions(
            predicted_codes=[0, 1, 3, 3],
            target_codes=[0, 2, 3, -100],
        )

        self.assertEqual(report["correct_steps"], 2)
        self.assertEqual(report["total_steps"], 3)
        self.assertAlmostEqual(report["step_accuracy"], 2 / 3)
        self.assertFalse(report["trace_exact"])

    def test_score_finality_predictions_counts_binary_targets(self):
        module = load_eval_script()

        report = module.score_finality_predictions(
            finality_logits=[-2.0, 3.0, 1.0, -1.0],
            target_values=[0.0, 1.0, -100.0, 1.0],
        )

        self.assertEqual(report["finality_correct_steps"], 2)
        self.assertEqual(report["finality_total_steps"], 3)
        self.assertAlmostEqual(report["finality_step_accuracy"], 2 / 3)

    def test_score_finality_predictions_treats_zero_logit_as_nonfinal_tie(self):
        module = load_eval_script()

        report = module.score_finality_predictions(
            finality_logits=[0.0, 0.0],
            target_values=[0.0, 0.0],
        )

        self.assertEqual(report["finality_correct_steps"], 2)
        self.assertEqual(report["finality_total_steps"], 2)
        self.assertAlmostEqual(report["finality_step_accuracy"], 1.0)

    def test_predicted_codes_from_logits_allows_disabled_code_head(self):
        import torch

        module = load_eval_script()

        self.assertEqual(
            module.predicted_codes_from_logits(torch.empty(1, 4, 0)),
            [],
        )

    def test_arg_parser_accepts_transition_state_off(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--disable-transition-state",
            ]
        )

        self.assertTrue(args.disable_transition_state)


if __name__ == "__main__":
    unittest.main()
