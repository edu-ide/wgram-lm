from pathlib import Path
import importlib.util
from types import SimpleNamespace
import unittest

import torch


def _load_module():
    path = Path("scripts/328_probe_qtrm_source_position_logits.py")
    spec = importlib.util.spec_from_file_location("source_position_logits_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SourcePositionLogitsProbeTests(unittest.TestCase):
    def test_oracle_source_positions_are_zero_based_prompt_source_slots(self):
        module = _load_module()

        positions = module.oracle_source_positions({"input_list": [44, 39, 55, 40, 32]})

        self.assertEqual(positions, [0, 3, 4])

    def test_oracle_source_classes_are_one_based_with_zero_reserved_for_null(self):
        module = _load_module()

        classes = module.oracle_source_classes({"input_list": [44, 39, 55, 40, 32]})

        self.assertEqual(classes, [1, 4, 5])

    def test_select_depth_logits_uses_last_depth_slice(self):
        module = _load_module()
        logits = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)

        selected = module.select_depth_logits(logits, batch_index=1, depth_index=-1)

        self.assertTrue(torch.equal(selected, logits[1, 2]))

    def test_select_renderer_copy_logits_prefers_final_core_state_when_compatible(self):
        module = _load_module()
        prompt_logits = torch.zeros(1, 1, 3, 8)
        core_logits = torch.ones(1, 4, 3, 8)

        selected, source = module.select_renderer_copy_logits(
            source_position_prompt_logits=prompt_logits,
            core_role_value_state_logits=core_logits,
        )

        self.assertEqual(source, "core_role_value_state_logits")
        self.assertTrue(torch.equal(selected, core_logits[:, -1:, :, :]))

    def test_select_renderer_copy_logits_can_prefer_primitive_state(self):
        module = _load_module()
        prompt_logits = torch.zeros(1, 1, 3, 8)
        core_logits = torch.ones(1, 4, 3, 8)
        primitive_logits = torch.full((1, 4, 3, 8), 2.0)

        selected, source = module.select_renderer_copy_logits(
            source_position_prompt_logits=prompt_logits,
            core_role_value_state_logits=core_logits,
            core_primitive_role_value_state_logits=primitive_logits,
            prefer_primitive=True,
        )

        self.assertEqual(source, "core_primitive_role_value_state_logits")
        self.assertTrue(torch.equal(selected, primitive_logits[:, -1:, :, :]))

    def test_select_renderer_copy_logits_falls_back_to_prompt_when_core_roles_mismatch(self):
        module = _load_module()
        prompt_logits = torch.zeros(1, 1, 3, 8)
        core_logits = torch.ones(1, 4, 2, 8)

        selected, source = module.select_renderer_copy_logits(
            source_position_prompt_logits=prompt_logits,
            core_role_value_state_logits=core_logits,
        )

        self.assertEqual(source, "source_position_prompt_logits")
        self.assertTrue(torch.equal(selected, prompt_logits))

    def test_summarize_pointer_logits_measures_role_ordered_exactness(self):
        module = _load_module()
        logits = torch.zeros(4, 5)
        logits[0, 0] = 5.0
        logits[1, 3] = 4.0
        logits[2, 4] = 3.0

        summary = module.summarize_pointer_logits(logits, oracle_positions=[0, 3, 4])

        self.assertTrue(summary["selected_role_top1_exact"])
        self.assertEqual(summary["selected_role_correct"], 3)
        self.assertEqual(summary["selected_role_count"], 3)
        self.assertEqual(summary["predicted_positions"], [0, 3, 4])
        self.assertEqual([row["rank"] for row in summary["roles"]], [1, 1, 1])

    def test_summarize_pointer_logits_can_read_final_answer_role_block(self):
        module = _load_module()
        logits = torch.zeros(10, 8)
        logits[4, 3] = 5.0
        logits[5, 5] = 4.0
        logits[6, 0] = 3.0
        logits[0, 1] = 9.0
        logits[1, 2] = 9.0
        logits[2, 4] = 9.0

        summary = module.summarize_pointer_logits(
            logits,
            oracle_positions=[3, 5, 0],
            role_offset=4,
        )

        self.assertTrue(summary["selected_role_top1_exact"])
        self.assertEqual(summary["predicted_positions"], [3, 5, 0])
        self.assertEqual([row["role"] for row in summary["roles"]], [4, 5, 6])

    def test_summarize_pointer_logits_rejects_permutation_even_if_set_matches(self):
        module = _load_module()
        logits = torch.zeros(3, 5)
        logits[0, 4] = 5.0
        logits[1, 3] = 4.0
        logits[2, 0] = 3.0

        summary = module.summarize_pointer_logits(logits, oracle_positions=[0, 3, 4])

        self.assertFalse(summary["selected_role_top1_exact"])
        self.assertEqual(summary["selected_role_accuracy"], 1.0 / 3.0)
        self.assertEqual(summary["predicted_positions"], [4, 3, 0])

    def test_summarize_pointer_logits_reports_valid_slot_masked_exactness(self):
        module = _load_module()
        logits = torch.zeros(3, 8)
        logits[0, 6] = 9.0
        logits[1, 7] = 9.0
        logits[2, 5] = 9.0
        logits[0, 0] = 5.0
        logits[1, 3] = 4.0
        logits[2, 4] = 3.0

        summary = module.summarize_pointer_logits(
            logits,
            oracle_positions=[0, 3, 4],
            valid_position_count=5,
        )

        self.assertFalse(summary["selected_role_top1_exact"])
        self.assertEqual(summary["invalid_top_position_count"], 3)
        self.assertTrue(summary["valid_selected_role_top1_exact"])
        self.assertEqual(summary["valid_predicted_positions"], [0, 3, 4])

    def test_copy_answer_from_positions_renders_prompt_values_in_predicted_order(self):
        module = _load_module()
        row = {"input_list": [44, 39, 55, 40, 32]}

        answer = module.copy_answer_from_positions(row, [4, 3, 0])

        self.assertEqual(answer, "32,40,44")

    def test_copy_answer_from_source_classes_keeps_zero_as_null(self):
        module = _load_module()
        row = {"input_list": [44, 39, 55, 40, 32]}

        answer = module.copy_answer_from_source_classes(row, [1, 4, 5, 0])

        self.assertEqual(answer, "44,40,32")

    def test_source_pointer_defaults_match_l4_runner_source_slot_path(self):
        module = _load_module()
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                token_numeric_source_slot_embedding_enabled=False,
                token_numeric_source_slot_vocab_size=0,
                token_numeric_source_slot_max_slots=0,
                token_numeric_source_slot_gate_min=0.0,
                token_numeric_source_slot_predicate_feedback_enabled=False,
                token_numeric_source_slot_predicate_gate_min=0.0,
                core_source_position_binder_enabled=False,
                core_source_position_binder_gate_min=0.0,
                core_source_position_binder_state_gate_min=0.0,
                core_source_position_binder_state_straight_through=False,
                core_source_position_binder_source_slots_only=False,
                core_source_position_binder_raw_source_slots_enabled=False,
            )
        )
        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--checkpoint",
                "model.pt",
                "--cases",
                "cases.jsonl",
                "--out",
                "report.json",
            ]
        )

        module.apply_source_pointer_defaults(cfg, args)

        self.assertTrue(cfg.model.token_numeric_source_slot_embedding_enabled)
        self.assertEqual(128, cfg.model.token_numeric_source_slot_vocab_size)
        self.assertEqual(5, cfg.model.token_numeric_source_slot_max_slots)
        self.assertEqual(1.0, cfg.model.token_numeric_source_slot_gate_min)
        self.assertTrue(cfg.model.token_numeric_source_slot_predicate_feedback_enabled)
        self.assertEqual(1.0, cfg.model.token_numeric_source_slot_predicate_gate_min)
        self.assertTrue(cfg.model.core_source_position_binder_enabled)
        self.assertEqual(1.0, cfg.model.core_source_position_binder_gate_min)
        self.assertEqual(0.25, cfg.model.core_source_position_binder_state_gate_min)
        self.assertTrue(cfg.model.core_source_position_binder_state_straight_through)
        self.assertTrue(cfg.model.core_source_position_binder_source_slots_only)
        self.assertTrue(cfg.model.core_source_position_binder_raw_source_slots_enabled)


if __name__ == "__main__":
    unittest.main()
