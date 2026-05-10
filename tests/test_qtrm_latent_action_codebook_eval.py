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

    def test_score_halted_transition_predictions_accepts_early_final_prefix(self):
        module = load_eval_script()

        report = module.score_halted_transition_predictions(
            predicted_codes=[0, 1, 2, 2],
            target_codes=[0, 1, 3, 3],
            finality_logits=[-2.0, 3.0, -1.0, -1.0],
            target_finality=[0.0, 1.0, 1.0, 1.0],
        )

        self.assertTrue(report["halted_trace_exact"])
        self.assertEqual(report["halted_depth"], 2)
        self.assertEqual(report["halted_prefix_steps"], 2)

    def test_score_halted_transition_predictions_rejects_premature_halt(self):
        module = load_eval_script()

        report = module.score_halted_transition_predictions(
            predicted_codes=[0, 1, 3, 3],
            target_codes=[0, 1, 3, 3],
            finality_logits=[2.0, -1.0, -1.0, -1.0],
            target_finality=[0.0, 1.0, 1.0, 1.0],
        )

        self.assertFalse(report["halted_trace_exact"])
        self.assertEqual(report["halted_depth"], 1)

    def test_predicted_codes_from_logits_allows_disabled_code_head(self):
        import torch

        module = load_eval_script()

        self.assertEqual(
            module.predicted_codes_from_logits(torch.empty(1, 4, 0)),
            [],
        )

    def test_predicted_joint_states_from_logits_returns_argmax_states(self):
        import torch

        module = load_eval_script()
        logits = torch.zeros(1, 3, 8)
        logits[:, 0, 0] = 5.0
        logits[:, 1, 3] = 5.0
        logits[:, 2, 7] = 5.0

        self.assertEqual(module.predicted_joint_states_from_logits(logits), [0, 3, 7])

    def test_primitive_operations_map_to_dynamic_halt_codes(self):
        module = load_eval_script()

        operation_ids = [
            module.PRIMITIVE_TRANSITION_OPERATION_ORDER.index("filter_even"),
            module.PRIMITIVE_TRANSITION_OPERATION_ORDER.index("double_filtered"),
            module.PRIMITIVE_TRANSITION_OPERATION_ORDER.index("multiply_sum"),
            module.PRIMITIVE_TRANSITION_OPERATION_ORDER.index("subtract_offset"),
            module.PRIMITIVE_TRANSITION_OPERATION_ORDER.index("hold_final"),
            module.PRIMITIVE_TRANSITION_OPERATION_ORDER.index("filter_above_threshold"),
        ]

        self.assertEqual(
            module.transition_codes_from_primitive_operations(operation_ids),
            [0, 1, 2, 3, 4, 1],
        )

    def test_predicted_source_ids_from_logits_returns_argmax_sources(self):
        import torch

        module = load_eval_script()
        logits = torch.zeros(1, 3, 2)
        logits[0, 0, 0] = 1.0
        logits[0, 1, 1] = 1.0
        logits[0, 2, 1] = 2.0

        self.assertEqual(module.predicted_source_ids_from_logits(logits), [0, 1, 1])

    def test_parse_code_permutation_requires_complete_mapping(self):
        module = load_eval_script()

        self.assertEqual(
            module.parse_code_permutation("0:0,1:2,2:1,3:3,4:4"),
            {0: 0, 1: 2, 2: 1, 3: 3, 4: 4},
        )
        with self.assertRaises(ValueError):
            module.parse_code_permutation("0:1,2:3")

    def test_apply_predicted_code_ablation_can_shuffle_or_drop(self):
        module = load_eval_script()

        self.assertEqual(
            module.apply_predicted_code_ablation(
                [0, 1, 4],
                code_permutation={0: 0, 1: 2, 2: 1, 3: 3, 4: 4},
                drop_codes_to=None,
            ),
            [0, 2, 4],
        )
        self.assertEqual(
            module.apply_predicted_code_ablation(
                [0, 1, 4],
                code_permutation=None,
                drop_codes_to=4,
            ),
            [4, 4, 4],
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
                "--prediction-source",
                "routed",
                "--predicted-code-permutation",
                "0:0,1:2,2:1,3:3,4:4",
            ]
        )

        self.assertTrue(args.disable_transition_state)
        self.assertEqual(args.prediction_source, "routed")
        self.assertEqual(args.predicted_code_permutation, "0:0,1:2,2:1,3:3,4:4")


if __name__ == "__main__":
    unittest.main()
