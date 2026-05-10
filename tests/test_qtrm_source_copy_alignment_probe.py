from pathlib import Path
import importlib.util
import unittest

import torch


def _load_module():
    path = Path("scripts/328_probe_qtrm_source_copy_alignment.py")
    spec = importlib.util.spec_from_file_location("qtrm_source_copy_alignment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMSourceCopyAlignmentProbeTests(unittest.TestCase):
    def test_target_source_position_classes_are_one_based_and_null_padded(self):
        module = _load_module()
        row = {"input_list": [44, 39, 55, 40, 32]}

        classes = module.target_source_position_classes(row, num_roles=6)

        self.assertEqual(classes, [1, 4, 5, 0, 0, 0])

    def test_score_source_position_logits_requires_content_order(self):
        module = _load_module()
        logits = torch.full((1, 1, 6, 8), -20.0)
        for role, target in enumerate([1, 4, 5, 0, 0, 0]):
            logits[0, 0, role, target] = 20.0

        score = module.score_source_position_logits(
            logits,
            target_classes=[1, 4, 5, 0, 0, 0],
        )

        self.assertTrue(score["row_content_exact"])
        self.assertTrue(score["row_full_exact"])
        self.assertEqual(score["predicted_classes"], [1, 4, 5, 0, 0, 0])
        self.assertEqual(score["correct_content_positions"], 3)
        self.assertEqual(score["correct_null_positions"], 3)

    def test_score_source_position_logits_rejects_shifted_copy_classes(self):
        module = _load_module()
        logits = torch.full((1, 1, 6, 8), -20.0)
        for role, predicted in enumerate([2, 5, 6, 0, 0, 0]):
            logits[0, 0, role, predicted] = 20.0

        score = module.score_source_position_logits(
            logits,
            target_classes=[1, 4, 5, 0, 0, 0],
        )

        self.assertFalse(score["row_content_exact"])
        self.assertFalse(score["row_full_exact"])
        self.assertEqual(score["correct_content_positions"], 0)
        self.assertEqual(score["correct_null_positions"], 3)

    def test_select_alignment_logits_prefers_final_recurrent_state(self):
        module = _load_module()
        prompt_logits = torch.full((1, 1, 2, 8), -20.0)
        prompt_logits[0, 0, 0, 1] = 20.0
        recurrent_logits = torch.full((1, 3, 2, 8), -20.0)
        recurrent_logits[0, -1, 0, 4] = 20.0

        selected = module.select_alignment_logits(
            {
                "core_source_position_prompt_logits": prompt_logits,
                "core_role_value_state_logits": recurrent_logits,
            }
        )

        self.assertEqual(tuple(selected.shape), (1, 1, 2, 8))
        self.assertEqual(int(selected[0, 0, 0].argmax().item()), 4)

    def test_select_alignment_logits_prefers_primitive_when_renderer_does(self):
        module = _load_module()
        recurrent_logits = torch.full((1, 3, 2, 8), -20.0)
        recurrent_logits[0, -1, 0, 4] = 20.0
        primitive_logits = torch.full((1, 3, 2, 8), -20.0)
        primitive_logits[0, -1, 0, 6] = 20.0

        selected = module.select_alignment_logits(
            {
                "core_role_value_state_logits": recurrent_logits,
                "core_primitive_role_value_state_logits": primitive_logits,
            },
            prefer_primitive=True,
        )

        self.assertEqual(int(selected[0, 0, 0].argmax().item()), 6)

    def test_mask_alignment_logits_to_answer_roles_sets_non_answer_roles_null(self):
        module = _load_module()
        logits = torch.full((1, 1, 10, 8), -20.0)
        for role in range(10):
            logits[0, 0, role, min(role + 1, 7)] = 20.0

        masked = module.mask_alignment_logits_to_answer_roles(
            logits,
            num_roles=10,
        )

        self.assertEqual(int(masked[0, 0, 0].argmax().item()), 1)
        self.assertEqual(int(masked[0, 0, 3].argmax().item()), 4)
        for role in range(4, 10):
            self.assertEqual(int(masked[0, 0, role].argmax().item()), 0)

    def test_default_parser_targets_current_source_copy_split(self):
        module = _load_module()
        args = module.build_arg_parser().parse_args([])

        self.assertEqual(
            args.cases,
            "data/eval/qtrm_source_copy_lexicalization_eval128.jsonl",
        )
        self.assertIn("accepted_l3_last.pt", args.checkpoint)


if __name__ == "__main__":
    unittest.main()
