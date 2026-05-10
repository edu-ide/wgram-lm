import importlib.util
from pathlib import Path
import unittest


def load_eval_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "238_eval_qtrm_algorithmic_value_state.py"
    )
    spec = importlib.util.spec_from_file_location("qtrm_algorithmic_value_state_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QTRMAlgorithmicValueStateEvalTests(unittest.TestCase):
    def test_filter_rows_by_family_runs_before_max_cases(self):
        module = load_eval_script()
        rows = [
            {"task_family": "arithmetic_chain", "id": "a0"},
            {"task_family": "list_transform", "id": "l0"},
            {"task_family": "boolean_logic", "id": "b0"},
            {"task_family": "list_transform", "id": "l1"},
        ]

        filtered = module.filter_rows_by_family(
            rows,
            include_family="list_transform",
            max_cases=1,
        )

        self.assertEqual([row["id"] for row in filtered], ["l0"])

    def test_algorithmic_targets_encode_relative_list_and_scalar_slots(self):
        module = load_eval_script()
        row = {
            "task_family": "mixed_list_arithmetic",
            "list_value_start": 50001,
            "mixed_offset": 9,
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
            },
        }

        kind_targets, slot_targets = module.algorithmic_targets_from_row(
            row,
            num_steps=5,
            max_slots=5,
            slot_vocab_size=128,
        )

        self.assertEqual(kind_targets, [1, 1, 2, 2, -100])
        self.assertEqual(
            slot_targets,
            [
                [2, 4, 6, 0, 0],
                [3, 7, 11, 0, 0],
                [7, 19, 0, 0, 0],
                [7, 10, 0, 0, 0],
                [-100, -100, -100, -100, -100],
            ],
        )

    def test_score_algorithmic_predictions_requires_kind_and_slots(self):
        module = load_eval_script()

        report = module.score_algorithmic_sequences(
            predicted_kinds=[1, 2, 2],
            predicted_slots=[[2, 4, 0], [7, 19, 0], [7, 10, 0]],
            target_kinds=[1, 2, 2],
            target_slots=[[2, 4, 0], [7, 18, 0], [-100, -100, -100]],
        )

        self.assertEqual(report["correct_kinds"], 3)
        self.assertEqual(report["total_kinds"], 3)
        self.assertEqual(report["correct_slots"], 5)
        self.assertEqual(report["total_slots"], 6)
        self.assertEqual(report["correct_content_slots"], 3)
        self.assertEqual(report["total_content_slots"], 4)
        self.assertEqual(report["exact_steps"], 1)
        self.assertEqual(report["total_steps"], 2)
        self.assertFalse(report["trace_exact"])

    def test_role_value_targets_bind_fields_to_stable_roles(self):
        module = load_eval_script()
        row = {
            "task_family": "mixed_list_arithmetic",
            "list_value_start": 50001,
            "mixed_offset": 9,
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
            },
        }

        targets = module.role_value_targets_from_row(
            row,
            num_steps=4,
            num_roles=10,
            value_vocab_size=128,
        )

        self.assertEqual(
            targets,
            [
                [2, 4, 6, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, 3, 7, 11, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, 7, 19],
                [-100, -100, -100, -100, -100, -100, -100, -100, 7, 10],
            ],
        )

    def test_plain_list_transform_targets_use_source_position_slots(self):
        module = load_eval_script()
        row = {
            "task_family": "list_transform",
            "question": (
                "From the list [1, 4, 2, 7, 3], keep only even numbers, "
                "double each kept number."
            ),
            "depth_targets": {
                "1": "4,2",
                "2": "8,4",
                "4": "8,4",
            },
        }

        kind_targets, slot_targets = module.algorithmic_targets_from_row(
            row,
            num_steps=4,
            max_slots=4,
            slot_vocab_size=128,
        )
        role_targets = module.role_value_targets_from_row(
            row,
            num_steps=4,
            num_roles=10,
            value_vocab_size=128,
        )

        self.assertEqual(kind_targets, [1, 1, -100, 1])
        self.assertEqual(
            slot_targets,
            [
                [2, 3, 0, 0],
                [2, 3, 0, 0],
                [-100, -100, -100, -100],
                [2, 3, 0, 0],
            ],
        )
        self.assertEqual(
            role_targets,
            [
                [2, 3, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, 2, 3, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, 2, 3, -100, -100, -100, -100],
            ],
        )

    def test_plain_list_transform_initial_role_targets_bind_prompt_order(self):
        module = load_eval_script()
        row = {
            "task_family": "list_transform",
            "role_value_list_class_mode": "absolute",
            "question": (
                "From the list [1, 4, 2, 7, 3], keep only even numbers, "
                "double each kept number."
            ),
            "depth_targets": {
                "1": "4,2",
                "2": "8,4",
                "4": "8,4",
            },
        }

        targets = module.role_value_initial_targets_from_row(
            row,
            num_steps=1,
            num_roles=10,
            value_vocab_size=128,
            list_class_mode="source_position",
        )

        self.assertEqual(
            targets,
            [[1, 2, 3, 4, -100, -100, -100, -100, -100, -100]],
        )

    def test_plain_list_transform_role_targets_cover_fifth_source_position(self):
        module = load_eval_script()
        row = {
            "task_family": "list_transform",
            "question": (
                "From the list [1, 3, 5, 7, 8], keep only even numbers, "
                "double each kept number."
            ),
            "depth_targets": {
                "1": "8",
                "2": "16",
                "4": "16",
            },
        }

        initial = module.role_value_initial_targets_from_row(
            row,
            num_steps=1,
            num_roles=12,
            value_vocab_size=128,
            list_class_mode="source_position",
        )
        targets = module.role_value_targets_from_row(
            row,
            num_steps=4,
            num_roles=12,
            value_vocab_size=128,
            list_class_mode="source_position",
        )

        self.assertEqual(
            initial,
            [[1, 2, 3, 4, 5, -100, -100, -100, -100, -100, -100, -100]],
        )
        self.assertEqual(targets[0][0], 5)
        self.assertEqual(targets[1][5], 5)

    def test_plain_list_transform_role_targets_can_supervise_null_slots(self):
        module = load_eval_script()
        row = {
            "task_family": "list_transform",
            "role_value_supervise_null_slots": True,
            "question": (
                "From the list [1, 3, 5, 7, 8], keep only even numbers, "
                "double each kept number."
            ),
            "depth_targets": {
                "1": "8",
                "2": "16",
            },
        }

        targets = module.role_value_targets_from_row(
            row,
            num_steps=2,
            num_roles=12,
            value_vocab_size=128,
            list_class_mode="source_position",
        )

        self.assertEqual(targets[0][:5], [5, 0, 0, 0, 0])
        self.assertEqual(targets[1][5:10], [5, 0, 0, 0, 0])

    def test_apply_role_value_list_class_mode_overrides_row_metadata(self):
        module = load_eval_script()
        rows = [{"role_value_list_class_mode": "absolute"}, {}]

        selected = module.apply_role_value_list_class_mode(
            rows,
            "source_position",
        )

        self.assertEqual(selected, "source_position")
        self.assertEqual(
            [row["role_value_list_class_mode"] for row in rows],
            ["source_position", "source_position"],
        )

    def test_plain_list_transform_role_targets_can_use_absolute_value_slots(self):
        module = load_eval_script()
        row = {
            "task_family": "list_transform",
            "question": (
                "From the list [1, 4, 2, 7, 3], keep only even numbers, "
                "double each kept number."
            ),
            "depth_targets": {
                "1": "4,2",
                "2": "8,4",
                "4": "8,4",
            },
        }

        role_targets = module.role_value_targets_from_row(
            row,
            num_steps=4,
            num_roles=10,
            value_vocab_size=128,
            list_class_mode="absolute",
        )

        self.assertEqual(
            role_targets,
            [
                [5, 3, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, 9, 5, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                [-100, -100, -100, -100, 9, 5, -100, -100, -100, -100],
            ],
        )

    def test_numeric_source_feature_matrix_encodes_value_position_and_mask(self):
        module = load_eval_script()

        features, mask = module.numeric_source_feature_matrix(
            {"input_list": [2, 5]},
            visual_dim=32,
            max_list_len=4,
            value_vocab_size=8,
        )

        self.assertEqual(mask, [1, 1, 0, 0])
        self.assertEqual(len(features), 4)
        self.assertEqual(len(features[0]), 32)
        self.assertEqual(features[0][1], 1.0)
        self.assertEqual(features[0][2], 0.0)
        self.assertEqual(features[0][3], 1.0)
        self.assertEqual(features[0][4 + 2], 1.0)
        self.assertEqual(features[1][1], 0.0)
        self.assertEqual(features[1][2], 1.0 / 3.0)
        self.assertEqual(features[1][3], 1.0)
        self.assertEqual(features[1][4 + 5], 1.0)

    def test_parser_accepts_numeric_source_feature_ablation_options(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "model.pt",
                "--data-jsonl",
                "rows.jsonl",
                "--numeric-source-features",
                "--numeric-source-max-list-len",
                "7",
                "--numeric-source-value-vocab-size",
                "64",
                "--disable-numeric-source-features",
            ]
        )

        self.assertTrue(args.numeric_source_features)
        self.assertTrue(args.disable_numeric_source_features)
        self.assertEqual(args.numeric_source_max_list_len, 7)
        self.assertEqual(args.numeric_source_value_vocab_size, 64)

    def test_parser_accepts_token_numeric_value_feature_ablation_options(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "model.pt",
                "--data-jsonl",
                "rows.jsonl",
                "--token-numeric-value-features",
                "--token-numeric-value-vocab-size",
                "64",
                "--disable-token-numeric-value-features",
            ]
        )

        self.assertTrue(args.token_numeric_value_features)
        self.assertTrue(args.disable_token_numeric_value_features)
        self.assertEqual(args.token_numeric_value_vocab_size, 64)

    def test_parser_accepts_token_numeric_source_slot_ablation_options(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "model.pt",
                "--data-jsonl",
                "rows.jsonl",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                "64",
                "--token-numeric-source-slot-max-slots",
                "7",
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-gate-min",
                "1.0",
                "--token-numeric-source-slot-predicate-feedback",
                "--disable-token-numeric-source-slots",
            ]
        )

        self.assertTrue(args.token_numeric_source_slots)
        self.assertTrue(args.disable_token_numeric_source_slots)
        self.assertEqual(args.token_numeric_source_slot_vocab_size, 64)
        self.assertEqual(args.token_numeric_source_slot_max_slots, 7)
        self.assertEqual(args.token_numeric_source_slot_id_mode, "relative_parity")
        self.assertEqual(args.token_numeric_source_slot_gate_min, 1.0)
        self.assertTrue(args.token_numeric_source_slot_predicate_feedback)

    def test_prepare_prompt_with_token_numeric_value_ids_uses_offsets(self):
        import torch

        module = load_eval_script()

        class FakeTokenizer:
            def __call__(self, prompt, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                    "offset_mapping": torch.tensor([[[0, 1], [1, 3], [3, 5], [5, 6]]]),
                }

        row = {"prompt": "[14]", "input_list": [14]}
        tokenizer = FakeTokenizer()

        (
            input_ids,
            attention_mask,
            token_numeric_ids,
            source_slot_ids,
            source_slot_mask,
        ) = (
            module._prepare_prompt_with_token_numeric(
                tokenizer,
                row,
                max_length=16,
                device="cpu",
                token_numeric_value_features=True,
                disable_token_numeric_value_features=False,
                token_numeric_value_vocab_size=64,
            )
        )

        self.assertEqual(input_ids.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1, 1]])
        self.assertEqual(token_numeric_ids.tolist(), [[0, 15, 0, 0]])
        self.assertIsNone(source_slot_ids)
        self.assertIsNone(source_slot_mask)
        self.assertTrue(tokenizer.kwargs["return_offsets_mapping"])

    def test_prepare_prompt_with_token_numeric_source_slots_uses_offsets(self):
        import torch

        module = load_eval_script()

        class FakeTokenizer:
            def __call__(self, prompt, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                    "offset_mapping": torch.tensor(
                        [[[0, 1], [1, 3], [3, 5], [5, 7], [7, 8]]]
                    ),
                }

        row = {"prompt": "[14,31]", "input_list": [14, 31]}
        tokenizer = FakeTokenizer()

        (
            _input_ids,
            _attention_mask,
            token_numeric_ids,
            source_slot_ids,
            source_slot_mask,
        ) = module._prepare_prompt_with_token_numeric(
            tokenizer,
            row,
            max_length=16,
            device="cpu",
            token_numeric_source_slots=True,
            disable_token_numeric_source_slots=False,
            token_numeric_source_slot_vocab_size=64,
            token_numeric_source_slot_max_slots=4,
        )

        self.assertIsNone(token_numeric_ids)
        self.assertEqual(source_slot_ids.tolist(), [[15, 32, 0, 0]])
        self.assertEqual(source_slot_mask.tolist(), [[1, 1, 0, 0]])
        self.assertTrue(tokenizer.kwargs["return_offsets_mapping"])

    def test_prepare_prompt_with_relative_parity_source_slots(self):
        import torch

        module = load_eval_script()

        class FakeTokenizer:
            def __call__(self, prompt, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                    "offset_mapping": torch.tensor(
                        [[[0, 1], [1, 6], [6, 7], [7, 12], [12, 13]]]
                    ),
                }

        row = {
            "prompt": "[60001,60002]",
            "list_value_start": 60001,
            "list_length": 2,
        }
        tokenizer = FakeTokenizer()

        (
            _input_ids,
            _attention_mask,
            token_numeric_ids,
            source_slot_ids,
            source_slot_mask,
        ) = module._prepare_prompt_with_token_numeric(
            tokenizer,
            row,
            max_length=16,
            device="cpu",
            token_numeric_source_slots=True,
            disable_token_numeric_source_slots=False,
            token_numeric_source_slot_vocab_size=3,
            token_numeric_source_slot_max_slots=4,
            token_numeric_source_slot_id_mode="relative_parity",
        )

        self.assertIsNone(token_numeric_ids)
        self.assertEqual(source_slot_ids.tolist(), [[1, 2, 0, 0]])
        self.assertEqual(source_slot_mask.tolist(), [[1, 1, 0, 0]])
        self.assertTrue(tokenizer.kwargs["return_offsets_mapping"])

    def test_typed_algorithmic_targets_separate_field_vocabularies(self):
        module = load_eval_script()
        row = {
            "task_family": "mixed_list_arithmetic",
            "list_value_start": 50001,
            "mixed_offset": 9,
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
            },
        }

        targets = module.typed_algorithmic_field_targets_from_row(
            row,
            num_steps=5,
            max_list_slots=4,
            offset_vocab_size=128,
            scalar_vocab_size=128,
        )

        self.assertEqual(targets["kind"], [0, 1, 2, 2, -100])
        self.assertEqual(targets["raw_list_offsets"][0], [2, 4, 6, 0])
        self.assertEqual(targets["doubled_list_offsets"][1], [3, 7, 11, 0])
        self.assertEqual(targets["scalar_coeff"], [-100, -100, 7, 7, -100])
        self.assertEqual(targets["scalar_offset"], [-100, -100, 10, 10, -100])
        self.assertEqual(targets["scalar_residual"], [-100, -100, 19, 10, -100])
        self.assertEqual(targets["scalar_residual_delta"], [-100, -100, -100, 55, -100])
        self.assertEqual(targets["final_residual"], [-100, -100, -100, 10, -100])

    def test_plain_list_transform_typed_targets_use_raw_then_doubled_roles(self):
        module = load_eval_script()
        row = {
            "task_family": "list_transform",
            "question": (
                "From the list [1, 4, 2, 7, 3], keep only even numbers, "
                "double each kept number."
            ),
            "depth_targets": {
                "1": "4,2",
                "2": "8,4",
                "4": "8,4",
            },
        }

        targets = module.typed_algorithmic_field_targets_from_row(
            row,
            num_steps=4,
            max_list_slots=4,
            offset_vocab_size=128,
            scalar_vocab_size=128,
        )

        self.assertEqual(targets["kind"], [0, 1, -100, 1])
        self.assertEqual(targets["raw_list_offsets"][0], [2, 3, 0, 0])
        self.assertEqual(targets["doubled_list_offsets"][1], [2, 3, 0, 0])
        self.assertEqual(targets["doubled_list_offsets"][3], [2, 3, 0, 0])
        self.assertEqual(targets["scalar_coeff"], [-100, -100, -100, -100])

    def test_typed_algorithmic_targets_label_all_finality_residuals(self):
        module = load_eval_script()
        row = {
            "task_family": "mixed_list_arithmetic",
            "list_value_start": 50001,
            "mixed_offset": 9,
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
                "5": "300015",
            },
            "transition_finality_targets": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 1,
                "5": 1,
            },
        }

        targets = module.typed_algorithmic_field_targets_from_row(
            row,
            num_steps=5,
            max_list_slots=4,
            offset_vocab_size=128,
            scalar_vocab_size=128,
        )

        self.assertEqual(targets["scalar_offset"], [-100, -100, 10, 10, 10])
        self.assertEqual(targets["scalar_residual_delta"], [-100, -100, -100, 55, 64])
        self.assertEqual(targets["final_residual"], [-100, -100, -100, 10, 10])

    def test_typed_algorithmic_targets_support_scalar_affine_gate(self):
        module = load_eval_script()
        row = {
            "task_family": "scalar_affine_arithmetic",
            "base_value": 50000,
            "scalar_coeff": 6,
            "subtract_offset": 9,
            "depth_targets": {
                "1": str(6 * 50000 + 18),
                "2": str(6 * 50000 + 9),
                "3": str(6 * 50000 + 9),
            },
            "transition_finality_targets": {"1": 0, "2": 1, "3": 1},
        }

        targets = module.typed_algorithmic_field_targets_from_row(
            row,
            num_steps=4,
            max_list_slots=4,
            offset_vocab_size=128,
            scalar_vocab_size=128,
        )

        self.assertEqual(targets["kind"], [2, 2, 2, -100])
        self.assertEqual(targets["scalar_coeff"], [7, 7, 7, -100])
        self.assertEqual(targets["scalar_offset"], [10, 10, 10, -100])
        self.assertEqual(targets["scalar_residual"], [19, 10, 10, -100])
        self.assertEqual(targets["scalar_residual_delta"], [-100, 55, 64, -100])
        self.assertEqual(targets["final_residual"], [-100, 10, 10, -100])

    def test_score_typed_algorithmic_predictions_requires_all_labelled_fields(self):
        module = load_eval_script()

        target = {
            "kind": [0, 2],
            "raw_list_offsets": [[2, 4, 0], [-100, -100, -100]],
            "doubled_list_offsets": [
                [-100, -100, -100],
                [-100, -100, -100],
            ],
            "scalar_coeff": [-100, 7],
            "scalar_offset": [-100, 9],
            "scalar_residual": [-100, 10],
            "final_residual": [-100, 10],
        }
        predicted = {
            "kind": [0, 2],
            "raw_list_offsets": [[2, 4, 0], [0, 0, 0]],
            "doubled_list_offsets": [[0, 0, 0], [0, 0, 0]],
            "scalar_coeff": [0, 7],
            "scalar_offset": [0, 9],
            "scalar_residual": [0, 9],
            "final_residual": [0, 10],
        }

        report = module.score_typed_algorithmic_field_predictions(
            predicted=predicted,
            target=target,
        )

        self.assertEqual(report["correct_fields"], 8)
        self.assertEqual(report["total_fields"], 9)
        self.assertEqual(report["exact_steps"], 1)
        self.assertEqual(report["total_steps"], 2)
        self.assertFalse(report["trace_exact"])

    def test_typed_scalar_regression_prediction_rounds_value_heads(self):
        import torch

        module = load_eval_script()

        outputs = {
            "typed_algorithmic_kind_logits": torch.zeros(1, 2, 3),
            "typed_algorithmic_raw_list_offset_logits": torch.zeros(1, 2, 1, 20),
            "typed_algorithmic_doubled_list_offset_logits": torch.zeros(
                1, 2, 1, 20
            ),
            "typed_algorithmic_scalar_coeff_logits": torch.zeros(1, 2, 20),
            "typed_algorithmic_scalar_coeff_value": torch.tensor([[0.26, 0.53]]),
            "typed_algorithmic_scalar_offset_logits": torch.zeros(1, 2, 20),
            "typed_algorithmic_scalar_offset_value": torch.tensor([[0.47, 0.58]]),
            "typed_algorithmic_scalar_residual_logits": torch.zeros(1, 2, 20),
            "typed_algorithmic_scalar_residual_value": torch.tensor([[0.1, 0.9]]),
            "typed_algorithmic_final_residual_logits": torch.zeros(1, 2, 20),
            "typed_algorithmic_final_residual_value": torch.tensor([[0.2, 0.8]]),
        }

        predicted = module.predicted_typed_algorithmic_fields_from_outputs(
            outputs,
            prefer_scalar_regression=True,
        )

        self.assertEqual(predicted["scalar_coeff"], [5, 10])
        self.assertEqual(predicted["scalar_offset"], [9, 11])
        self.assertEqual(predicted["scalar_residual"], [2, 17])
        self.assertEqual(predicted["final_residual"], [4, 15])

    def test_score_role_value_predictions_tracks_content_exact(self):
        module = load_eval_script()

        report = module.score_role_value_predictions(
            predicted_values=[[2, 4, 9], [7, 10, 0]],
            target_values=[[2, 4, -100], [7, 11, -100]],
        )

        self.assertEqual(report["correct_values"], 3)
        self.assertEqual(report["total_values"], 4)
        self.assertEqual(report["exact_steps"], 1)
        self.assertEqual(report["total_steps"], 2)
        self.assertFalse(report["trace_exact"])

    def test_select_role_logits_can_choose_core_role_path(self):
        import torch

        module = load_eval_script()
        outputs = {
            "role_value_state_logits": torch.zeros(1, 2, 3, 5),
            "core_role_value_state_logits": torch.ones(1, 2, 3, 7),
        }

        selected = module.select_role_value_logits(outputs, use_core_role_value_state=True)

        self.assertEqual(selected.shape, (1, 2, 3, 7))

    def test_select_role_logits_can_choose_core_value_delta_code_path(self):
        import torch

        module = load_eval_script()
        outputs = {
            "role_value_state_logits": torch.zeros(1, 2, 3, 5),
            "core_role_value_state_logits": torch.ones(1, 2, 3, 7),
            "core_value_delta_code_logits": torch.full((1, 2, 3, 11), 2.0),
        }

        selected = module.select_role_value_logits(
            outputs,
            use_core_role_value_state=True,
            use_core_value_delta_code=True,
        )

        self.assertEqual(selected.shape, (1, 2, 3, 11))

    def test_select_role_logits_can_choose_prompt_role_path(self):
        import torch

        module = load_eval_script()
        outputs = {
            "role_value_state_logits": torch.zeros(1, 2, 3, 5),
            "core_role_value_state_logits": torch.ones(1, 2, 3, 7),
            "core_role_value_state_prompt_logits": torch.full((1, 1, 3, 13), 2.0),
        }

        selected = module.select_role_value_logits(
            outputs,
            use_core_role_value_state=True,
            use_core_role_value_prompt_state=True,
        )

        self.assertEqual(selected.shape, (1, 1, 3, 13))

    def test_arg_parser_exposes_core_state_carry_ablation(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--disable-core-state-carry",
            ]
        )

        self.assertTrue(args.disable_core_state_carry)

    def test_arg_parser_exposes_core_role_value_delta_ablation(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--disable-core-role-value-delta",
            ]
        )

        self.assertTrue(args.disable_core_role_value_delta)

    def test_arg_parser_exposes_core_value_delta_code_ablation(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--disable-core-value-delta-code",
            ]
        )

        self.assertTrue(args.disable_core_value_delta_code)

    def test_arg_parser_exposes_core_typed_register_ablation(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--disable-core-typed-register-executor",
            ]
        )

        self.assertTrue(args.disable_core_typed_register_executor)

    def test_arg_parser_exposes_core_value_delta_code_readout(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--use-core-value-delta-code",
                "--use-core-role-value-prompt-state",
            ]
        )

        self.assertTrue(args.use_core_value_delta_code)
        self.assertTrue(args.use_core_role_value_prompt_state)

    def test_arg_parser_exposes_typed_register_and_blend_readouts(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--use-core-typed-register-state",
                "--core-primitive-typed-register-blend",
                "step_confidence_switch",
            ]
        )

        self.assertTrue(args.use_core_typed_register_state)
        self.assertEqual(
            args.core_primitive_typed_register_blend,
            "step_confidence_switch",
        )

    def test_select_role_value_logits_can_blend_internal_readouts(self):
        import torch

        module = load_eval_script()
        primitive = torch.zeros(1, 1, 1, 3)
        typed = torch.zeros(1, 1, 1, 3)
        primitive[0, 0, 0, 1] = 4.0
        primitive[0, 0, 0, 2] = 3.9
        typed[0, 0, 0, 2] = 3.0
        primitive[0, 0, 0, 0] = 0.1

        summed = module.select_role_value_logits(
            {
                "core_primitive_role_value_state_logits": primitive,
                "core_typed_register_value_logits": typed,
            },
            core_primitive_typed_register_blend="sum",
        )
        switched = module.select_role_value_logits(
            {
                "core_primitive_role_value_state_logits": primitive,
                "core_typed_register_value_logits": typed,
            },
            core_primitive_typed_register_blend="confidence_switch",
        )
        step_switched = module.select_role_value_logits(
            {
                "core_primitive_role_value_state_logits": primitive,
                "core_typed_register_value_logits": typed,
            },
            core_primitive_typed_register_blend="step_confidence_switch",
        )

        self.assertEqual(int(summed.argmax(dim=-1)[0, 0, 0]), 2)
        self.assertTrue(torch.allclose(switched, typed))
        self.assertTrue(torch.allclose(step_switched, typed))

    def test_arg_parser_exposes_typed_algorithmic_value_state(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--use-typed-algorithmic-value-state",
                "--use-typed-scalar-regression-values",
                "--disable-typed-algorithmic-value-state",
                "--disable-typed-algorithmic-value-state-recurrent",
            ]
        )

        self.assertTrue(args.use_typed_algorithmic_value_state)
        self.assertTrue(args.use_typed_scalar_regression_values)
        self.assertTrue(args.disable_typed_algorithmic_value_state)
        self.assertTrue(args.disable_typed_algorithmic_value_state_recurrent)

    def test_arg_parser_exposes_core_source_position_binder_ablation(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--core-source-position-binder",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-raw-source-slots",
                "--disable-core-source-position-binder",
                "--disable-core-role-value-prompt-extract",
                "--disable-core-primitive-prompt-context",
                "--core-source-position-binder-gate-min",
                "1.0",
                "--core-source-position-binder-state-gate-min",
                "0.25",
                "--core-source-position-binder-state-st",
                "--core-source-position-binder-query-state",
                "--disable-core-source-position-binder-query-state",
                "--core-source-position-binder-query-state-gate-min",
                "0.5",
                "--core-source-value-binder",
                "--disable-core-source-value-binder",
                "--core-source-value-binder-state-gate-min",
                "0.75",
                "--core-source-value-binder-state-st",
                "--core-primitive-role-value-source-value-conditioning",
                "--core-primitive-role-value-source-value-gate-min",
                "0.6",
            ]
        )

        self.assertTrue(args.core_source_position_binder)
        self.assertTrue(args.core_source_position_binder_source_slots_only)
        self.assertTrue(args.core_source_position_binder_raw_source_slots)
        self.assertTrue(args.disable_core_source_position_binder)
        self.assertTrue(args.disable_core_role_value_prompt_extract)
        self.assertTrue(args.disable_core_primitive_prompt_context)
        self.assertEqual(args.core_source_position_binder_gate_min, 1.0)
        self.assertEqual(args.core_source_position_binder_state_gate_min, 0.25)
        self.assertTrue(args.core_source_position_binder_state_st)
        self.assertTrue(args.core_source_position_binder_query_state)
        self.assertTrue(args.disable_core_source_position_binder_query_state)
        self.assertEqual(args.core_source_position_binder_query_state_gate_min, 0.5)
        self.assertTrue(args.core_source_value_binder)
        self.assertTrue(args.disable_core_source_value_binder)
        self.assertEqual(args.core_source_value_binder_state_gate_min, 0.75)
        self.assertTrue(args.core_source_value_binder_state_st)
        self.assertTrue(args.core_primitive_role_value_source_value_conditioning)
        self.assertEqual(args.core_primitive_role_value_source_value_gate_min, 0.6)

    def test_arg_parser_exposes_role_value_target_mode(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "last.pt",
                "--data-jsonl",
                "eval.jsonl",
                "--role-value-target-mode",
                "initial",
            ]
        )

        self.assertEqual(args.role_value_target_mode, "initial")


if __name__ == "__main__":
    unittest.main()
