from pathlib import Path
import importlib.util
import unittest


def load_module():
    path = Path("scripts/324_train_qtrm_source_pointer_batch.py")
    spec = importlib.util.spec_from_file_location("qtrm_source_pointer_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMSourcePointerBatchTrainerTests(unittest.TestCase):
    def test_parser_accepts_batch_training_controls(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--data-jsonl",
                "train.jsonl",
                "--out-dir",
                "out",
                "--init-checkpoint",
                "init.pt",
                "--steps",
                "7",
                "--row-batch-size",
                "16",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-predicate-feedback",
                "--core-source-position-binder",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-raw-source-slots",
                "--core-primitive-role-value-order-contrast-weight",
                "0.75",
                "--core-primitive-role-value-order-contrast-margin",
                "0.2",
                "--core-primitive-role-value-source-margin-weight",
                "0.6",
                "--core-primitive-role-value-source-margin",
                "0.4",
                "--core-role-value-source-copy-answer-role-targets",
            ]
        )

        self.assertEqual(args.row_batch_size, 16)
        self.assertTrue(args.token_numeric_source_slots)
        self.assertTrue(args.token_numeric_source_slot_predicate_feedback)
        self.assertTrue(args.core_source_position_binder)
        self.assertTrue(args.core_source_position_binder_source_slots_only)
        self.assertTrue(args.core_source_position_binder_raw_source_slots)
        self.assertEqual(args.core_primitive_role_value_order_contrast_weight, 0.75)
        self.assertEqual(args.core_primitive_role_value_order_contrast_margin, 0.2)
        self.assertEqual(args.core_primitive_role_value_source_margin_weight, 0.6)
        self.assertEqual(args.core_primitive_role_value_source_margin, 0.4)
        self.assertTrue(args.core_role_value_source_copy_answer_role_targets)

    def test_source_slot_tensors_batch_rows_from_offsets(self):
        import torch

        module = load_module()
        rows = [
            {
                "prompt": "Question: From the list [4, 19], return evens.",
                "input_list": [4, 19],
            },
            {
                "prompt": "Question: From the list [1, 6], return evens.",
                "input_list": [1, 6],
            },
        ]
        offsets = [
            [(0, 9), (10, 14), (15, 18), (19, 23), (24, 25), (25, 26), (26, 27), (27, 29), (29, 31), (31, 32)],
            [(0, 9), (10, 14), (15, 18), (19, 23), (24, 25), (25, 26), (26, 27), (27, 28), (28, 30), (30, 31)],
        ]

        ids, mask = module.source_slot_tensors_from_offsets(
            rows,
            offsets=offsets,
            max_slots=4,
            value_vocab_size=64,
            device="cpu",
        )

        self.assertTrue(torch.equal(ids, torch.tensor([[5, 20, 0, 0], [2, 7, 0, 0]])))
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])))

    def test_role_value_targets_are_batched(self):
        import torch

        module = load_module()
        rows = [
            {
                "input_list": [4, 6, 1],
                "depth_targets": {"1": "4,6", "2": "8,12"},
                "role_value_list_class_mode": "source_position",
                "role_value_supervise_null_slots": True,
            },
            {
                "input_list": [1, 4, 6],
                "depth_targets": {"1": "4,6", "2": "8,12"},
                "role_value_list_class_mode": "source_position",
                "role_value_supervise_null_slots": True,
            },
        ]

        targets = module.batch_role_value_targets(
            rows,
            num_depths=2,
            num_roles=6,
            value_vocab_size=64,
            device="cpu",
        )

        self.assertEqual(tuple(targets.shape), (2, 2, 6))
        self.assertTrue(
            torch.equal(targets[0, 0, :4], torch.tensor([1, 2, -100, -100]))
        )
        self.assertTrue(
            torch.equal(targets[1, 0, :4], torch.tensor([2, 3, -100, -100]))
        )

    def test_source_copy_targets_can_be_shifted_to_answer_role_block(self):
        import torch

        module = load_module()
        rows = [
            {
                "input_list": [4, 5, 6],
                "depth_targets": {"1": "4,6", "2": "4,6"},
                "role_value_list_class_mode": "source_position",
                "role_value_supervise_null_slots": True,
                "role_value_source_copy_no_doubled": True,
            }
        ]

        targets = module.batch_role_value_targets(
            rows,
            num_depths=2,
            num_roles=10,
            value_vocab_size=64,
            device="cpu",
            source_copy_answer_role_targets=True,
        )

        self.assertTrue(torch.equal(targets[0, 0, :4], torch.tensor([-100, -100, -100, -100])))
        self.assertTrue(torch.equal(targets[0, 0, 4:8], torch.tensor([1, 3, 0, 0])))
        self.assertTrue(torch.equal(targets[0, 1, :4], torch.tensor([-100, -100, -100, -100])))
        self.assertTrue(torch.equal(targets[0, 1, 4:8], torch.tensor([1, 3, 0, 0])))

    def test_previous_target_order_contrast_penalizes_copying_prior_source_role(self):
        import torch

        module = load_module()
        logits = torch.zeros(1, 1, 3, 8)
        targets = torch.tensor([[[1, 3, 5]]])
        logits[0, 0, 0, 1] = 4.0
        logits[0, 0, 1, 1] = 4.0
        logits[0, 0, 1, 3] = 3.0
        logits[0, 0, 2, 3] = 4.0
        logits[0, 0, 2, 5] = 3.0

        loss, metrics = module.role_value_previous_target_contrast_loss(
            logits,
            targets,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(metrics["samples"]), 3.0)
        self.assertLess(float(metrics["win_rate"]), 1.0)

    def test_previous_target_order_contrast_is_zero_when_ordered_roles_win(self):
        import torch

        module = load_module()
        logits = torch.zeros(1, 1, 3, 8)
        targets = torch.tensor([[[1, 3, 5]]])
        logits[0, 0, 0, 1] = 4.0
        logits[0, 0, 1, 1] = 1.0
        logits[0, 0, 1, 3] = 4.0
        logits[0, 0, 2, 1] = 1.0
        logits[0, 0, 2, 3] = 1.0
        logits[0, 0, 2, 5] = 4.0

        loss, metrics = module.role_value_previous_target_contrast_loss(
            logits,
            targets,
            margin=0.5,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["samples"]), 3.0)
        self.assertEqual(float(metrics["win_rate"]), 1.0)

    def test_source_slot_margin_penalizes_distractor_source_slot(self):
        import torch

        module = load_module()
        logits = torch.zeros(1, 1, 3, 8)
        targets = torch.tensor([[[1, 3, 5]]])
        source_slot_mask = torch.tensor([[1, 1, 1, 1, 1]])
        logits[0, 0, 1, 2] = 5.0
        logits[0, 0, 1, 3] = 4.0

        loss, metrics = module.role_value_source_slot_margin_loss(
            logits,
            targets,
            source_slot_mask,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(metrics["samples"]), 12.0)
        self.assertLess(float(metrics["win_rate"]), 1.0)

    def test_source_slot_margin_is_zero_when_targets_beat_valid_source_slots(self):
        import torch

        module = load_module()
        logits = torch.zeros(1, 1, 3, 8)
        targets = torch.tensor([[[1, 3, 5]]])
        source_slot_mask = torch.tensor([[1, 1, 1, 1, 1]])
        logits[0, 0, 0, 1] = 4.0
        logits[0, 0, 1, 3] = 4.0
        logits[0, 0, 2, 5] = 4.0

        loss, metrics = module.role_value_source_slot_margin_loss(
            logits,
            targets,
            source_slot_mask,
            margin=0.5,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["samples"]), 12.0)
        self.assertEqual(float(metrics["win_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()
