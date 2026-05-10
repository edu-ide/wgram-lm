from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


def _load_module():
    path = Path("scripts/196_train_pure_recursive_depth_supervised.py")
    spec = importlib.util.spec_from_file_location("depth_supervised_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveDepthSupervisedTrainScriptTests(unittest.TestCase):
    def test_load_rows_rejects_evidence_shortcuts(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Question?\nAnswer:",
                        "chosen": "A",
                        "evidence": ["shortcut"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                module.load_rows(path)

    def test_load_rows_accepts_answer_alias_as_canonical_answer(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Question?\nAnswer:",
                        "answer_aliases": ["17"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = module.load_rows(path)

        self.assertEqual(rows[0]["answer"], "17")

    def test_answer_first_token_uses_space_prefixed_answer(self):
        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last_text = text
                return [123] if text.startswith(" ") else [456]

        tokenizer = FakeTokenizer()

        self.assertEqual(module.answer_first_token_id(tokenizer, "violet"), 123)
        self.assertEqual(tokenizer.last_text, " violet")

    def test_target_for_core_steps_uses_depth_targets_before_final_answer(self):
        module = _load_module()
        row = {
            "chosen": "final",
            "depth_targets": {
                "1": "stage-one",
                "2": "stage-two",
                "4": "final",
                "8": "final",
            },
        }

        self.assertEqual(module.target_for_core_steps(row, 1), "stage-one")
        self.assertEqual(module.target_for_core_steps(row, 2), "stage-two")
        self.assertEqual(module.target_for_core_steps(row, 4), "final")
        self.assertEqual(module.target_for_core_steps(row, 8), "final")
        self.assertEqual(module.target_for_core_steps({"chosen": "fallback"}, 2), "fallback")
        self.assertEqual(
            module.target_for_core_steps(row, 1, target_mode="final"),
            "final",
        )

    def test_algorithmic_role_value_state_targets_final_mode_keeps_terminal_step(self):
        module = _load_module()
        row = {
            "chosen": "44,40,32",
            "input_list": [44, 39, 55, 40, 32],
            "role_value_list_class_mode": "source_position",
            "role_value_source_copy_no_doubled": True,
            "role_value_supervise_null_slots": True,
            "depth_targets": {
                "1": "44,40,32",
                "2": "44,40,32",
                "4": "44,40,32",
                "8": "44,40,32",
            },
        }

        targets = module.algorithmic_role_value_state_targets(
            row,
            num_depths=3,
            num_roles=10,
            value_vocab_size=128,
            device="cpu",
            target_mode="final",
        )

        self.assertEqual(targets[0, 0].tolist(), [-100] * 10)
        self.assertEqual(targets[0, 1].tolist(), [-100] * 10)
        self.assertEqual(targets[0, 2, :4].tolist(), [1, 4, 5, 0])

    def test_paired_hard_negative_index_chooses_same_group_different_trace(self):
        module = _load_module()
        rows = [
            {
                "id": "a",
                "pair_group_id": "g1",
                "source_even_position_signature": [1, 2, 3],
            },
            {
                "id": "b",
                "pair_group_id": "g1",
                "source_even_position_signature": [2, 3, 4],
            },
            {
                "id": "c",
                "pair_group_id": "g2",
                "source_even_position_signature": [1, 2, 3],
            },
        ]

        lookup = module.build_paired_hard_negative_lookup(rows)

        self.assertEqual(lookup[0], 1)
        self.assertEqual(lookup[1], 0)
        self.assertNotIn(2, lookup)

    def test_core_primitive_role_value_pair_trace_contrastive_loss_penalizes_template_tie(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 1, 4, 6)
        positive_targets = torch.tensor([[[1, 3, 4, 0]]])
        negative_targets = torch.tensor([[[2, 3, 4, 0]]])

        loss, metrics = module.core_primitive_role_value_pair_trace_contrastive_loss(
            logits,
            positive_targets,
            negative_targets,
            margin=0.25,
        )

        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(float(metrics["core_primitive_role_value_pair_trace_contrast_samples"]), 1.0)
        self.assertEqual(float(metrics["core_primitive_role_value_pair_trace_contrast_win_rate"]), 0.0)

    def test_core_primitive_role_value_pair_trace_contrastive_loss_passes_when_positive_trace_wins(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 1, 4, 6)
        positive_targets = torch.tensor([[[1, 3, 4, 0]]])
        negative_targets = torch.tensor([[[2, 3, 4, 0]]])
        logits[0, 0, 0, 1] = 2.0
        logits[0, 0, 0, 2] = -2.0

        loss, metrics = module.core_primitive_role_value_pair_trace_contrastive_loss(
            logits,
            positive_targets,
            negative_targets,
            margin=0.25,
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(float(metrics["core_primitive_role_value_pair_trace_contrast_win_rate"]), 1.0)

    def test_token_numeric_source_slot_predicate_ce_loss_uses_predicate_metrics(self):
        import torch

        module = _load_module()
        logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0], [0.0, 0.0]]])
        source_slot_ids = torch.tensor([[1, 2, 0]])

        loss, metrics = module.token_numeric_source_slot_predicate_ce_loss(
            logits,
            source_slot_ids,
        )

        self.assertLess(float(loss), 0.10)
        self.assertIn("token_numeric_source_slot_predicate_ce", metrics)
        self.assertIn("token_numeric_source_slot_predicate_acc", metrics)
        self.assertEqual(
            float(metrics["token_numeric_source_slot_predicate_acc"]),
            1.0,
        )

    def test_token_numeric_source_slot_parity_ce_loss_supports_relative_parity_ids(self):
        import torch

        module = _load_module()
        logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]]])
        source_slot_ids = torch.tensor([[1, 2, 0]])

        loss, metrics = module.token_numeric_source_slot_parity_ce_loss(
            logits,
            source_slot_ids,
            id_mode="relative_parity",
        )

        self.assertLess(float(loss), 0.10)
        self.assertEqual(
            float(metrics["token_numeric_source_slot_parity_acc"]),
            1.0,
        )

    def test_row_temporal_spatial_context_converts_json_vectors(self):
        module = _load_module()

        one_token = module._row_temporal_spatial_context(
            {"temporal_spatial_context": [0.1, 0.2, 0.3]},
            device="cpu",
        )
        two_tokens = module._row_temporal_spatial_context(
            {"temporal_spatial_context": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            device="cpu",
        )

        self.assertEqual(tuple(one_token.shape), (1, 3))
        self.assertEqual(tuple(two_tokens.shape), (1, 2, 3))
        self.assertIsNone(module._row_temporal_spatial_context({}, device="cpu"))

    def test_row_temporal_spatial_context_rejects_bad_shape(self):
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "temporal_spatial_context"):
            module._row_temporal_spatial_context(
                {"temporal_spatial_context": [[[0.1]], [[0.2]]]},
                device="cpu",
            )

    def test_prepare_prompt_answer_tracks_full_answer_token_span(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def __call__(
                self,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=None,
                padding=False,
                add_special_tokens=True,
            ):
                ids_by_text = {
                    "Question?": [11, 12],
                    "Question? stage one": [11, 12, 21, 22],
                }
                ids = ids_by_text[text]
                return {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                }

        input_ids, attention_mask, target_ids, target_start, target_end = module._prepare_prompt_answer(
            FakeTokenizer(),
            "Question?",
            "stage one",
            max_length=16,
            device="cpu",
        )

        self.assertEqual(input_ids.tolist(), [[11, 12, 21, 22]])
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1, 1]])
        self.assertEqual(target_ids.tolist(), [[21, 22]])
        self.assertEqual(target_start, 2)
        self.assertEqual(target_end, 4)

    def test_prepare_causal_prefix_answer_uses_prompt_only_and_first_answer_token(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def __call__(
                self,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=None,
                padding=False,
                add_special_tokens=True,
            ):
                self.last_call_text = text
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def encode(self, text, add_special_tokens=False):
                self.last_encode_text = text
                return [21, 22]

        input_ids, attention_mask, target_ids, target_start, target_end = (
            module._prepare_causal_prefix_answer(
                FakeTokenizer(),
                "Question?",
                "stage one",
                max_length=16,
                device="cpu",
            )
        )

        self.assertEqual(input_ids.tolist(), [[11, 12]])
        self.assertEqual(attention_mask.tolist(), [[1, 1]])
        self.assertEqual(target_ids.tolist(), [[21]])
        self.assertEqual(target_start, 2)
        self.assertEqual(target_end, 3)

    def test_prepare_causal_prefix_answer_examples_extend_prefix_one_token_at_a_time(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def __call__(
                self,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=None,
                padding=False,
                add_special_tokens=True,
            ):
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def encode(self, text, add_special_tokens=False):
                self.last_encode_text = text
                return [21, 22, 23]

        examples = module._prepare_causal_prefix_answer_examples(
            FakeTokenizer(),
            "Question?",
            "stage one now",
            max_length=16,
            device="cpu",
            max_target_tokens=2,
        )

        self.assertEqual(len(examples), 2)
        first_input, first_mask, first_target, first_start, first_end = examples[0]
        second_input, second_mask, second_target, second_start, second_end = examples[1]
        self.assertEqual(first_input.tolist(), [[11, 12]])
        self.assertEqual(first_mask.tolist(), [[1, 1]])
        self.assertEqual(first_target.tolist(), [[21]])
        self.assertEqual((first_start, first_end), (2, 3))
        self.assertEqual(second_input.tolist(), [[11, 12, 21]])
        self.assertEqual(second_mask.tolist(), [[1, 1, 1]])
        self.assertEqual(second_target.tolist(), [[22]])
        self.assertEqual((second_start, second_end), (3, 4))

    def test_prepare_causal_prefix_answer_examples_can_skip_leading_whitespace_target(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def __call__(
                self,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=None,
                padding=False,
                add_special_tokens=True,
            ):
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def encode(self, text, add_special_tokens=False):
                return [20, 21, 22] if text.startswith(" ") else [21, 22]

        examples = module._prepare_causal_prefix_answer_examples(
            FakeTokenizer(),
            "Question?",
            "42",
            max_length=16,
            device="cpu",
            max_target_tokens=2,
            skip_leading_whitespace_targets=True,
        )

        first_input, _, first_target, first_start, first_end = examples[0]
        second_input, _, second_target, second_start, second_end = examples[1]
        self.assertEqual(first_input.tolist(), [[11, 12]])
        self.assertEqual(first_target.tolist(), [[21]])
        self.assertEqual((first_start, first_end), (2, 3))
        self.assertEqual(second_input.tolist(), [[11, 12, 21]])
        self.assertEqual(second_target.tolist(), [[22]])
        self.assertEqual((second_start, second_end), (3, 4))

    def test_prepare_causal_prefix_rollout_examples_use_generated_prefix(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def __call__(
                self,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=None,
                padding=False,
                add_special_tokens=True,
            ):
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def encode(self, text, add_special_tokens=False):
                return [21, 22, 23]

        examples = module._prepare_causal_prefix_rollout_answer_examples(
            FakeTokenizer(),
            "Question?",
            "stage one now",
            rollout_prefix_ids=[99, 98],
            max_length=16,
            device="cpu",
            max_target_tokens=3,
        )

        self.assertEqual(len(examples), 3)
        first_input, _, first_target, first_start, first_end = examples[0]
        second_input, _, second_target, second_start, second_end = examples[1]
        third_input, _, third_target, third_start, third_end = examples[2]
        self.assertEqual(first_input.tolist(), [[11, 12]])
        self.assertEqual(first_target.tolist(), [[21]])
        self.assertEqual((first_start, first_end), (2, 3))
        self.assertEqual(second_input.tolist(), [[11, 12, 99]])
        self.assertEqual(second_target.tolist(), [[22]])
        self.assertEqual((second_start, second_end), (3, 4))
        self.assertEqual(third_input.tolist(), [[11, 12, 99, 98]])
        self.assertEqual(third_target.tolist(), [[23]])
        self.assertEqual((third_start, third_end), (4, 5))

    def test_training_forward_passes_donor_logits_when_donor_scale_is_enabled(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("return_logits=needs_donor_logits_for(model, teacher_model)", text)
        self.assertIn('"donor_logits"', text)
        self.assertIn("**donor_forward_kwargs(donor_out)", text)

    def test_answer_state_loop_future_token_targets_and_ce(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last_encode_text = text
                return [21, 22]

        targets = module.answer_state_loop_future_token_targets(
            FakeTokenizer(),
            "answer",
            max_target_tokens=4,
            device="cpu",
        )
        self.assertEqual(targets.tolist(), [[21, 22, -100, -100]])

        logits = torch.zeros(1, 4, 32)
        logits[0, 0, 21] = 5.0
        logits[0, 1, 22] = 5.0
        loss, metrics = module.answer_state_loop_future_token_ce_loss(logits, targets)

        self.assertLess(float(loss), 0.3)
        self.assertEqual(float(metrics["answer_state_future_token_acc"]), 1.0)
        self.assertEqual(float(metrics["answer_state_future_token_samples"]), 2.0)

    def test_answer_state_loop_logit_ce_targets_loop_logits_directly(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[4, 7]])
        logits = torch.zeros(1, 2, 16)
        logits[0, 0, 4] = 5.0
        logits[0, 1, 7] = 5.0

        loss, metrics = module.answer_state_loop_logit_ce_loss(logits, target_ids)

        self.assertLess(float(loss), 0.3)
        self.assertEqual(float(metrics["answer_state_loop_logit_acc"]), 1.0)
        self.assertEqual(float(metrics["answer_state_loop_logit_samples"]), 2.0)

    def test_parser_accepts_direct_answer_state_loop_logit_ce_weight(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--answer-state-loop-logit-ce-weight",
                "0.42",
            ]
        )

        self.assertEqual(args.answer_state_loop_logit_ce_weight, 0.42)

    def test_parser_accepts_causal_prefix_max_target_tokens(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--init-checkpoint",
                "last.pt",
                "--trainable-param-policy",
                "core_and_answer_state_loop",
                "--depth-final-ce-weight",
                "0.0",
                "--depth-trajectory-monotonic-weight",
                "0.31",
                "--depth-trajectory-monotonic-margin",
                "0.04",
                "--terminal-depth-ce-weight",
                "0.21",
                "--answer-state-loop-halt-ce-weight",
                "0.27",
                "--answer-state-loop-future-token-ce-weight",
                "0.36",
                "--answer-state-loop-future-token-max-target-tokens",
                "6",
                "--causal-prefix-max-target-tokens",
                "4",
                "--causal-prefix-later-token-weight",
                "0.1",
                "--causal-prefix-skip-leading-whitespace-targets",
                "--causal-prefix-self-rollout-weight",
                "0.2",
                "--causal-prefix-self-rollout-max-target-tokens",
                "3",
                "--teacher-checkpoint",
                "teacher.pt",
                "--teacher-first-token-depth-kl-weight",
                "0.5",
                "--teacher-final-logit-kl-weight",
                "0.6",
                "--teacher-depth-kl-temperature",
                "2.0",
                "--core-world-model-weight",
                "0.02",
                "--staged-internal-first-token-ce-weight",
                "0.4",
                "--staged-internal-sequence-ce-weight",
                "0.45",
                "--staged-internal-sequence-max-target-tokens",
                "5",
                "--transition-state-sequence-ce-weight",
                "0.55",
                "--transition-value-state-ce-weight",
                "0.6",
                "--transition-value-state-max-target-tokens",
                "9",
                "--algorithmic-value-state-ce-weight",
                "0.7",
                "--algorithmic-value-state-pad-ce-weight",
                "0.05",
                "--algorithmic-role-value-state-ce-weight",
                "0.8",
                "--algorithmic-role-value-step-margin-weight",
                "0.82",
                "--algorithmic-role-value-step-margin",
                "0.13",
                "--algorithmic-role-value-transition-ce-weight",
                "0.85",
                "--typed-algorithmic-kind-ce-multiplier",
                "0.25",
                "--typed-algorithmic-list-ce-multiplier",
                "0.5",
                "--typed-algorithmic-scalar-ce-multiplier",
                "3.0",
                "--typed-algorithmic-scalar-ordinal-weight",
                "0.31",
                "--typed-algorithmic-scalar-regression-weight",
                "0.32",
                "--core-role-value-prompt-ce-weight",
                "0.88",
                "--core-role-value-prompt-initial-metadata-targets",
                "--core-role-value-prompt-parity-ce-weight",
                "0.77",
                "--core-role-value-template-ce-weight",
                "0.79",
                "--core-role-value-template-table-ce-weight",
                "0.81",
                "--core-value-delta-code-ce-weight",
                "0.9",
                "--core-typed-register-ce-weight",
                "0.95",
                "--core-typed-register-operation-ce-weight",
                "0.33",
                "--core-typed-register-operation-target-shift",
                "1",
                "--core-typed-register-transition-ce-weight",
                "0.44",
                "--core-typed-register-step-margin-weight",
                "0.46",
                "--core-typed-register-step-margin",
                "0.09",
                "--core-typed-register-trace-margin-weight",
                "0.47",
                "--core-typed-register-trace-margin",
                "0.08",
                "--core-typed-register-scalar-role-ce-multiplier",
                "3.5",
                "--noise-warmup-steps",
                "7",
                "--noise-warmup-seq-len",
                "5",
                "--noise-warmup-batch-size",
                "3",
                "--noise-warmup-core-steps",
                "2",
                "--noise-warmup-target-vocab-size",
                "11",
                "--noise-warmup-final-ce-weight",
                "0.25",
                "--noise-warmup-depth-ce-weight",
                "0.75",
                "--noise-warmup-uniform-weight",
                "0.5",
                "--temporal-spatial-context-contrast-weight",
                "0.6",
                "--temporal-spatial-context-contrast-margin",
                "0.2",
                "--transition-state-contrast-weight",
                "0.7",
                "--transition-state-contrast-margin",
                "0.3",
                "--transition-state-ce-weight",
                "0.9",
                "--transition-state-depth-contrast-weight",
                "0.35",
                "--transition-state-depth-contrast-margin",
                "0.12",
                "--transition-state-code-ce-weight",
                "0.8",
                "--transition-state-finality-ce-weight",
                "0.65",
                "--transition-state-joint-ce-weight",
                "0.45",
                "--transition-joint-answer-bridge-contrast-weight",
                "0.75",
                "--transition-joint-answer-bridge-contrast-margin",
                "0.11",
                "--transition-joint-answer-bridge-contrast-all-prefix-tokens",
                "--primitive-transition-operation-ce-weight",
                "0.55",
                "--core-transition-feedback-operation-ce-weight",
                "0.56",
                "--core-transition-feedback-finality-ce-weight",
                "0.57",
                "--core-transition-feedback-teacher-forcing",
                "--core-transition-order-bottleneck-ce-weight",
                "0.58",
                "--transition-phase-ce-weight",
                "0.62",
                "--transition-source-router-ce-weight",
                "0.66",
                "--choice-margin-mode",
                "sequence",
                "--tail-negative-margin-weight",
                "0.25",
                "--tail-negative-margin",
                "0.07",
                "--tail-negative-family-filter",
                "",
                "--subtract-tail-counterfactual-margin-weight",
                "0.41",
                "--subtract-tail-counterfactual-margin",
                "0.03",
                "--subtract-tail-counterfactual-family-filter",
                "",
            ]
        )

        self.assertEqual(args.trainable_param_policy, "core_and_answer_state_loop")
        self.assertEqual(args.causal_prefix_max_target_tokens, 4)
        self.assertEqual(args.depth_final_ce_weight, 0.0)
        self.assertEqual(args.depth_trajectory_monotonic_weight, 0.31)
        self.assertEqual(args.depth_trajectory_monotonic_margin, 0.04)
        self.assertEqual(args.terminal_depth_ce_weight, 0.21)
        self.assertEqual(args.answer_state_loop_halt_ce_weight, 0.27)
        self.assertEqual(args.answer_state_loop_future_token_ce_weight, 0.36)
        self.assertEqual(args.answer_state_loop_future_token_max_target_tokens, 6)
        self.assertEqual(args.causal_prefix_later_token_weight, 0.1)
        self.assertTrue(args.causal_prefix_skip_leading_whitespace_targets)
        self.assertEqual(args.causal_prefix_self_rollout_weight, 0.2)
        self.assertEqual(args.causal_prefix_self_rollout_max_target_tokens, 3)
        self.assertEqual(args.teacher_checkpoint, "teacher.pt")
        self.assertEqual(args.teacher_first_token_depth_kl_weight, 0.5)
        self.assertEqual(args.teacher_final_logit_kl_weight, 0.6)
        self.assertEqual(args.teacher_depth_kl_temperature, 2.0)
        self.assertEqual(args.core_world_model_weight, 0.02)
        self.assertEqual(args.staged_internal_first_token_ce_weight, 0.4)
        self.assertEqual(args.staged_internal_sequence_ce_weight, 0.45)
        self.assertEqual(args.staged_internal_sequence_max_target_tokens, 5)
        self.assertEqual(args.transition_state_sequence_ce_weight, 0.55)
        self.assertEqual(args.transition_value_state_ce_weight, 0.6)
        self.assertEqual(args.transition_value_state_max_target_tokens, 9)
        self.assertEqual(args.algorithmic_value_state_ce_weight, 0.7)
        self.assertEqual(args.algorithmic_value_state_pad_ce_weight, 0.05)
        self.assertEqual(args.algorithmic_role_value_state_ce_weight, 0.8)
        self.assertEqual(args.algorithmic_role_value_step_margin_weight, 0.82)
        self.assertEqual(args.algorithmic_role_value_step_margin, 0.13)
        self.assertEqual(args.algorithmic_role_value_transition_ce_weight, 0.85)
        self.assertEqual(args.typed_algorithmic_kind_ce_multiplier, 0.25)
        self.assertEqual(args.typed_algorithmic_list_ce_multiplier, 0.5)
        self.assertEqual(args.typed_algorithmic_scalar_ce_multiplier, 3.0)
        self.assertEqual(args.typed_algorithmic_scalar_ordinal_weight, 0.31)
        self.assertEqual(args.typed_algorithmic_scalar_regression_weight, 0.32)
        self.assertEqual(args.core_role_value_prompt_ce_weight, 0.88)
        self.assertTrue(args.core_role_value_prompt_initial_metadata_targets)
        self.assertEqual(args.core_role_value_prompt_parity_ce_weight, 0.77)
        self.assertEqual(args.core_role_value_template_ce_weight, 0.79)
        self.assertEqual(args.core_role_value_template_table_ce_weight, 0.81)
        self.assertEqual(args.core_value_delta_code_ce_weight, 0.9)
        self.assertEqual(args.core_typed_register_ce_weight, 0.95)
        self.assertEqual(args.core_typed_register_operation_ce_weight, 0.33)
        self.assertEqual(args.core_typed_register_operation_target_shift, 1)
        self.assertEqual(args.core_typed_register_transition_ce_weight, 0.44)
        self.assertEqual(args.core_typed_register_step_margin_weight, 0.46)
        self.assertEqual(args.core_typed_register_step_margin, 0.09)
        self.assertEqual(args.core_typed_register_trace_margin_weight, 0.47)
        self.assertEqual(args.core_typed_register_trace_margin, 0.08)
        self.assertEqual(args.core_typed_register_scalar_role_ce_multiplier, 3.5)
        self.assertEqual(args.noise_warmup_steps, 7)
        self.assertEqual(args.noise_warmup_seq_len, 5)
        self.assertEqual(args.noise_warmup_batch_size, 3)
        self.assertEqual(args.noise_warmup_core_steps, 2)
        self.assertEqual(args.noise_warmup_target_vocab_size, 11)
        self.assertEqual(args.noise_warmup_final_ce_weight, 0.25)
        self.assertEqual(args.noise_warmup_depth_ce_weight, 0.75)
        self.assertEqual(args.noise_warmup_uniform_weight, 0.5)
        self.assertEqual(args.temporal_spatial_context_contrast_weight, 0.6)
        self.assertEqual(args.temporal_spatial_context_contrast_margin, 0.2)
        self.assertEqual(args.transition_state_contrast_weight, 0.7)
        self.assertEqual(args.transition_state_contrast_margin, 0.3)
        self.assertEqual(args.transition_state_ce_weight, 0.9)
        self.assertEqual(args.transition_state_depth_contrast_weight, 0.35)
        self.assertEqual(args.transition_state_depth_contrast_margin, 0.12)
        self.assertEqual(args.transition_state_code_ce_weight, 0.8)
        self.assertEqual(args.transition_state_finality_ce_weight, 0.65)
        self.assertEqual(args.transition_state_joint_ce_weight, 0.45)
        self.assertEqual(args.transition_joint_answer_bridge_contrast_weight, 0.75)
        self.assertEqual(args.transition_joint_answer_bridge_contrast_margin, 0.11)
        self.assertTrue(args.transition_joint_answer_bridge_contrast_all_prefix_tokens)
        self.assertEqual(args.primitive_transition_operation_ce_weight, 0.55)
        self.assertEqual(args.core_transition_feedback_operation_ce_weight, 0.56)
        self.assertEqual(args.core_transition_feedback_finality_ce_weight, 0.57)
        self.assertTrue(args.core_transition_feedback_teacher_forcing)
        self.assertEqual(args.core_transition_order_bottleneck_ce_weight, 0.58)
        self.assertEqual(args.transition_phase_ce_weight, 0.62)
        self.assertEqual(args.transition_source_router_ce_weight, 0.66)
        self.assertEqual(args.choice_margin_mode, "sequence")
        self.assertEqual(args.tail_negative_margin_weight, 0.25)
        self.assertEqual(args.tail_negative_margin, 0.07)
        self.assertEqual(args.tail_negative_family_filter, "")
        self.assertEqual(args.subtract_tail_counterfactual_margin_weight, 0.41)
        self.assertEqual(args.subtract_tail_counterfactual_margin, 0.03)
        self.assertEqual(args.subtract_tail_counterfactual_family_filter, "")

    def test_bridge_contrast_scope_defaults_to_first_prefix_token(self):
        module = _load_module()

        should_apply = module._should_apply_transition_joint_answer_bridge_contrast

        self.assertFalse(should_apply(0, 0.0))
        self.assertTrue(should_apply(0, 0.5))
        self.assertFalse(should_apply(1, 0.5))
        self.assertTrue(should_apply(1, 0.5, all_prefix_tokens=True))

    def test_typed_algorithmic_scalar_ordinal_loss_prefers_nearer_class(self):
        import torch

        module = _load_module()

        def make_logits(scalar_class):
            logits = {
                "kind_logits": torch.zeros(1, 1, 3),
                "raw_list_offset_logits": torch.zeros(1, 1, 1, 20),
                "doubled_list_offset_logits": torch.zeros(1, 1, 1, 20),
                "scalar_coeff_logits": torch.zeros(1, 1, 20),
                "scalar_residual_logits": torch.zeros(1, 1, 20),
                "final_residual_logits": torch.zeros(1, 1, 20),
            }
            logits["scalar_residual_logits"][0, 0, int(scalar_class)] = 10.0
            return logits

        targets = {
            "kind": torch.tensor([[-100]]),
            "raw_list_offsets": torch.tensor([[[-100]]]),
            "doubled_list_offsets": torch.tensor([[[-100]]]),
            "scalar_coeff": torch.tensor([[-100]]),
            "scalar_residual": torch.tensor([[5]]),
            "final_residual": torch.tensor([[-100]]),
        }

        near_loss, _ = module.typed_algorithmic_value_state_ce_loss(
            make_logits(5),
            targets,
            scalar_ce_multiplier=0.0,
            scalar_ordinal_weight=1.0,
        )
        far_loss, _ = module.typed_algorithmic_value_state_ce_loss(
            make_logits(15),
            targets,
            scalar_ce_multiplier=0.0,
            scalar_ordinal_weight=1.0,
        )

        self.assertLess(float(near_loss), float(far_loss))

    def test_typed_algorithmic_scalar_regression_loss_prefers_nearer_value(self):
        import torch

        module = _load_module()

        def make_logits(normalized_value):
            return {
                "kind_logits": torch.zeros(1, 1, 3),
                "raw_list_offset_logits": torch.zeros(1, 1, 1, 20),
                "doubled_list_offset_logits": torch.zeros(1, 1, 1, 20),
                "scalar_coeff_logits": torch.zeros(1, 1, 20),
                "scalar_coeff_value": torch.zeros(1, 1),
                "scalar_residual_logits": torch.zeros(1, 1, 20),
                "scalar_residual_value": torch.tensor([[float(normalized_value)]]),
                "final_residual_logits": torch.zeros(1, 1, 20),
                "final_residual_value": torch.zeros(1, 1),
            }

        targets = {
            "kind": torch.tensor([[-100]]),
            "raw_list_offsets": torch.tensor([[[-100]]]),
            "doubled_list_offsets": torch.tensor([[[-100]]]),
            "scalar_coeff": torch.tensor([[-100]]),
            "scalar_residual": torch.tensor([[5]]),
            "final_residual": torch.tensor([[-100]]),
        }

        near_loss, _ = module.typed_algorithmic_value_state_ce_loss(
            make_logits(5.0 / 19.0),
            targets,
            scalar_ce_multiplier=0.0,
            scalar_regression_weight=1.0,
        )
        far_loss, _ = module.typed_algorithmic_value_state_ce_loss(
            make_logits(15.0 / 19.0),
            targets,
            scalar_ce_multiplier=0.0,
            scalar_regression_weight=1.0,
        )

        self.assertLess(float(near_loss), float(far_loss))

    def test_tail_negative_rejected_texts_use_preterminal_state_for_final_answer(self):
        module = _load_module()

        row = {
            "task_family": "mixed_list_arithmetic",
            "answer_aliases": ["300015"],
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
                "8": "300015",
            },
            "transition_finality_targets": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 1,
                "8": 1,
            },
        }

        self.assertEqual(
            module.tail_negative_rejected_texts(row, current_answer="300015"),
            ["300024"],
        )
        self.assertEqual(
            module.tail_negative_rejected_texts(row, current_answer="100004,100008,100012"),
            [],
        )
        self.assertEqual(
            module.tail_negative_rejected_texts(
                {**row, "task_family": "arithmetic_chain"},
                current_answer="300015",
            ),
            [],
        )

    def test_subtract_tail_counterfactual_rejected_texts_include_off_by_one(self):
        module = _load_module()

        row = {
            "task_family": "mixed_list_arithmetic",
            "answer": "300015",
            "answer_aliases": ["300015"],
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
            },
            "transition_finality_targets": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 1,
            },
            "mixed_offset": 9,
        }

        rejected = module.subtract_tail_counterfactual_rejected_texts(
            row,
            current_answer="300015",
        )

        self.assertEqual(rejected, ["300024", "300014", "300016"])
        self.assertEqual(
            module.subtract_tail_counterfactual_rejected_texts(
                row,
                current_answer="300024",
            ),
            [],
        )

    def test_tail_negative_sequence_margin_loss_renames_choice_metrics(self):
        import torch

        module = _load_module()

        depth_logits = torch.zeros(1, 2, 1, 4)
        final_logits = torch.zeros(1, 1, 4)
        depth_logits[:, :, 0, 1] = 4.0
        final_logits[:, 0, 1] = 4.0
        chosen = torch.tensor([[1]])
        rejected = torch.tensor([[2]])

        loss, metrics = module.tail_negative_sequence_margin_loss(
            depth_logits,
            final_logits,
            chosen,
            rejected,
            margin=0.1,
        )

        self.assertLess(float(loss), 0.1)
        self.assertIn("tail_negative_margin_all_depth", metrics)
        self.assertIn("tail_negative_margin_final_path", metrics)

    def test_subtract_tail_counterfactual_sequence_margin_loss_renames_metrics(self):
        import torch

        module = _load_module()

        depth_logits = torch.zeros(1, 2, 1, 4)
        final_logits = torch.zeros(1, 1, 4)
        chosen = torch.tensor([[2]])
        rejected = torch.tensor([[1]])

        loss, metrics = module.subtract_tail_counterfactual_sequence_margin_loss(
            depth_logits,
            final_logits,
            chosen,
            rejected,
            margin=0.1,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("subtract_tail_counterfactual_margin_all_depth", metrics)
        self.assertIn("subtract_tail_counterfactual_margin_final_path", metrics)

    def test_final_subtract_tail_counterfactual_margin_uses_final_lm_path_only(self):
        import torch

        module = _load_module()

        final_logits = torch.zeros(1, 1, 4)
        chosen = torch.tensor([[2]])
        rejected = torch.tensor([[1]])

        loss, metrics = module.final_subtract_tail_counterfactual_sequence_margin_loss(
            final_logits,
            chosen,
            rejected,
            margin=0.1,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("final_subtract_tail_counterfactual_margin_final_path", metrics)

    def test_parser_accepts_final_subtract_tail_counterfactual_margin(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--causal-prefix-supervision",
                "--target-logit-positions-only",
                "--final-path-only-supervision",
                "--final-subtract-tail-counterfactual-margin-weight",
                "0.9",
                "--final-subtract-tail-counterfactual-margin",
                "0.07",
                "--final-subtract-tail-counterfactual-family-filter",
                "",
            ]
        )

        self.assertEqual(args.final_subtract_tail_counterfactual_margin_weight, 0.9)
        self.assertEqual(args.final_subtract_tail_counterfactual_margin, 0.07)
        self.assertEqual(args.final_subtract_tail_counterfactual_family_filter, "")

    def test_parser_accepts_save_every_for_validation_checkpoint_selection(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--init-checkpoint",
                "last.pt",
                "--save-every",
                "40",
            ]
        )

        self.assertEqual(args.save_every, 40)

    def test_parser_accepts_answer_selective_context_alignment_args(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--init-checkpoint",
                "last.pt",
                "--answer-selective-context-alignment-weight",
                "0.25",
                "--answer-selective-context-alignment-temperature",
                "2.0",
            ]
        )

        self.assertEqual(args.answer_selective_context_alignment_weight, 0.25)
        self.assertEqual(args.answer_selective_context_alignment_temperature, 2.0)

    def test_answer_selective_context_alignment_loss_matches_dense_teacher(self):
        import torch

        module = _load_module()

        teacher_logits = torch.tensor([[[0.0, 4.0, -1.0]]])
        same_loss, same_metrics = module.answer_selective_context_alignment_loss(
            teacher_logits.clone(),
            teacher_logits,
            temperature=1.0,
        )
        bad_student = torch.tensor([[[4.0, 0.0, -1.0]]])
        bad_loss, bad_metrics = module.answer_selective_context_alignment_loss(
            bad_student,
            teacher_logits,
            temperature=1.0,
        )

        self.assertLess(float(same_loss), float(bad_loss))
        self.assertAlmostEqual(
            float(same_metrics["answer_selective_context_alignment_kl"]),
            0.0,
            places=5,
        )
        self.assertGreater(float(bad_metrics["answer_selective_context_alignment_kl"]), 1.0)

    def test_algorithmic_value_state_targets_use_relative_slots(self):
        module = _load_module()

        row = {
            "family": "mixed_list_arithmetic",
            "base_value": 50001,
            "depth_targets": {
                "1": "[50002, 50004, 50006]",
                "2": "[100004, 100008, 100012]",
                "3": "300024",
                "4": "300015",
            },
        }

        kind_targets, slot_targets = module.algorithmic_value_state_targets(
            row,
            num_depths=5,
            max_slots=5,
            slot_vocab_size=128,
            device="cpu",
        )

        self.assertEqual(kind_targets.tolist(), [[1, 1, 2, 2, -100]])
        self.assertEqual(
            slot_targets.tolist(),
            [
                [
                    [2, 4, 6, 0, 0],
                    [3, 7, 11, 0, 0],
                    [7, 19, 0, 0, 0],
                    [7, 10, 0, 0, 0],
                    [-100, -100, -100, -100, -100],
                ]
            ],
        )

    def test_primitive_transition_operation_targets_follow_solver_trace_order(self):
        module = _load_module()

        row = {
            "transition_state_codes": {"1": 0, "2": 2, "3": 3, "4": 4, "5": 4},
            "solver_trace": [
                {"operation": "add_operands", "state_text": "10"},
                {"operation": "multiply_sum", "state_text": "20"},
                {"operation": "subtract_offset", "state_text": "17"},
                {"operation": "hold_final", "state_text": "17"},
            ]
        }
        operation_to_id = {
            "add_operands": 0,
            "multiply_sum": 1,
            "subtract_offset": 2,
            "hold_final": 3,
        }

        targets = module.primitive_transition_operation_targets(
            row,
            num_steps=5,
            operation_to_id=operation_to_id,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[0, 1, 2, 3, 3]])

    def test_primitive_transition_operation_targets_fill_hold_final_from_codes(self):
        module = _load_module()

        row = {
            "transition_state_codes": {"1": 0, "2": 2, "3": 3, "4": 1, "5": 1, "6": 4, "7": 4},
            "solver_trace": [
                {"operation": "add_operands", "state_text": "10"},
                {"operation": "multiply_sum", "state_text": "20"},
                {"operation": "subtract_offset", "state_text": "17"},
                {"operation": "filter_above_threshold", "state_text": "18"},
                {"operation": "double_filtered", "state_text": "36"},
            ],
        }
        operation_to_id = {
            "add_operands": 0,
            "multiply_sum": 1,
            "subtract_offset": 2,
            "filter_above_threshold": 3,
            "double_filtered": 4,
            "hold_final": 5,
        }

        targets = module.primitive_transition_operation_targets(
            row,
            num_steps=8,
            operation_to_id=operation_to_id,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[0, 1, 2, 3, 4, 5, 5, -100]])

    def test_transition_source_router_targets_follow_composition_order(self):
        module = _load_module()

        list_first = module.transition_source_router_targets(
            {"composition_order": "list_to_arithmetic"},
            num_steps=3,
            device="cpu",
        )
        arithmetic_first = module.transition_source_router_targets(
            {"composition_order": "arithmetic_to_list"},
            num_steps=3,
            device="cpu",
        )
        list_first_family = module.transition_source_router_targets(
            {"task_family": "mixed_list_arithmetic"},
            num_steps=2,
            device="cpu",
        )
        arithmetic_first_family = module.transition_source_router_targets(
            {"task_family": "mixed_arithmetic_list"},
            num_steps=2,
            device="cpu",
        )
        general_family = module.transition_source_router_targets(
            {"task_family": "arithmetic_chain"},
            num_steps=2,
            device="cpu",
        )
        unknown = module.transition_source_router_targets({}, num_steps=2, device="cpu")

        self.assertEqual(list_first.tolist(), [[0, 0, 0]])
        self.assertEqual(arithmetic_first.tolist(), [[1, 1, 1]])
        self.assertEqual(list_first_family.tolist(), [[0, 0]])
        self.assertEqual(arithmetic_first_family.tolist(), [[1, 1]])
        self.assertEqual(general_family.tolist(), [[0, 0]])
        self.assertEqual(unknown.tolist(), [[-100, -100]])

    def test_transition_phase_targets_follow_composition_order(self):
        module = _load_module()

        list_first = module.transition_phase_targets(
            {"composition_order": "list_to_arithmetic"},
            num_steps=3,
            device="cpu",
        )
        arithmetic_first = module.transition_phase_targets(
            {"composition_order": "arithmetic_to_list"},
            num_steps=3,
            device="cpu",
        )

        self.assertEqual(list_first.tolist(), [[0, 0, 0]])
        self.assertEqual(arithmetic_first.tolist(), [[1, 1, 1]])

    def test_transition_state_joint_order_contrast_scores_opposite_order(self):
        import torch

        module = _load_module()

        row = {
            "composition_order": "list_to_arithmetic",
            "transition_state_codes": {"1": 0, "2": 1, "3": 2, "4": 3},
            "transition_finality_targets": {"1": 0, "2": 0, "3": 0, "4": 0},
        }
        logits = torch.zeros(1, 4, 10)
        logits[0, 1, 2] = 2.0  # target code 1
        logits[0, 1, 4] = 0.0  # opposite code 2
        logits[0, 2, 4] = 2.0  # target code 2
        logits[0, 2, 6] = 0.0  # opposite code 3
        logits[0, 3, 6] = 2.0  # target code 3
        logits[0, 3, 2] = 0.0  # opposite code 1

        loss, metrics = module.transition_state_joint_order_contrast_loss(
            logits,
            row,
            margin=0.5,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["transition_state_joint_order_contrast_pairs"]), 3.0)
        self.assertEqual(float(metrics["transition_state_joint_order_contrast_win_rate"]), 1.0)

        logits[0, 1, 4] = 3.0
        loss, metrics = module.transition_state_joint_order_contrast_loss(
            logits,
            row,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(metrics["transition_state_joint_order_contrast_win_rate"]), 1.0)

    def test_transition_source_router_ce_loss_scores_logits(self):
        import torch

        module = _load_module()

        logits = torch.zeros(1, 2, 2)
        logits[0, 0, 0] = 2.0
        logits[0, 1, 1] = 2.0
        targets = torch.tensor([[0, 1]])

        loss, metrics = module.transition_source_router_ce_loss(logits, targets)

        self.assertLess(float(loss), 0.2)
        self.assertEqual(float(metrics["transition_source_router_acc"]), 1.0)

    def test_core_primitive_update_gate_bce_scores_copy_and_write(self):
        import torch

        module = _load_module()

        gate = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
        role_targets = torch.tensor([[[1, 2], [3, 2]]])
        initial_targets = torch.tensor([[[1, 0]]])

        loss, metrics = module.core_primitive_role_value_update_gate_bce_loss(
            gate,
            role_targets,
            initial_targets,
        )

        self.assertLess(float(loss), 0.2)
        self.assertEqual(float(metrics["core_primitive_role_value_update_gate_acc"]), 1.0)
        self.assertEqual(
            float(metrics["core_primitive_role_value_update_gate_changed_rate"]),
            0.5,
        )

    def test_transition_phase_ce_loss_scores_logits(self):
        import torch

        module = _load_module()

        logits = torch.zeros(1, 2, 2)
        logits[0, 0, 0] = 2.0
        logits[0, 1, 1] = 2.0
        targets = torch.tensor([[0, 1]])

        loss, metrics = module.transition_phase_ce_loss(logits, targets)

        self.assertLess(float(loss), 0.2)
        self.assertEqual(float(metrics["transition_phase_acc"]), 1.0)

    def test_algorithmic_role_value_transition_ce_loss_shifts_targets_forward(self):
        import torch

        module = _load_module()

        logits = torch.zeros(1, 2, 3, 8)
        targets = torch.tensor(
            [
                [
                    [1, 2, 3],
                    [4, 5, -100],
                    [6, -100, 7],
                ]
            ]
        )
        logits[:, 0, 0, 4] = 8.0
        logits[:, 0, 1, 5] = 8.0
        logits[:, 1, 0, 6] = 8.0
        logits[:, 1, 2, 7] = 8.0

        loss, metrics = module.algorithmic_role_value_transition_ce_loss(
            logits,
            targets,
        )

        self.assertLess(float(loss), 0.01)
        self.assertEqual(
            float(metrics["algorithmic_role_value_transition_acc"]),
            1.0,
        )
        self.assertEqual(
            float(metrics["algorithmic_role_value_transition_step_exact"]),
            1.0,
        )

    def test_algorithmic_role_value_step_margin_focuses_weakest_role(self):
        import torch

        module = _load_module()

        targets = torch.tensor([[[1, 2, -100], [3, 4, 5]]])
        good_logits = torch.zeros(1, 2, 3, 8)
        good_logits[0, 0, 0, 1] = 3.0
        good_logits[0, 0, 1, 2] = 3.0
        good_logits[0, 1, 0, 3] = 3.0
        good_logits[0, 1, 1, 4] = 3.0
        good_logits[0, 1, 2, 5] = 3.0

        good_loss, good_metrics = module.algorithmic_role_value_step_margin_loss(
            good_logits,
            targets,
            margin=0.5,
        )

        self.assertEqual(float(good_loss), 0.0)
        self.assertEqual(
            float(good_metrics["algorithmic_role_value_step_margin_pass_rate"]),
            1.0,
        )

        weak_logits = good_logits.clone()
        weak_logits[0, 1, 2, 6] = 3.2
        weak_loss, weak_metrics = module.algorithmic_role_value_step_margin_loss(
            weak_logits,
            targets,
            margin=0.5,
        )

        self.assertGreater(float(weak_loss), 0.0)
        self.assertLess(
            float(weak_metrics["algorithmic_role_value_step_margin_pass_rate"]),
            1.0,
        )

    def test_algorithmic_role_value_trace_margin_requires_whole_trace(self):
        import torch

        module = _load_module()

        targets = torch.tensor(
            [
                [[1, 2, -100], [3, 4, 5]],
                [[1, -100, -100], [3, -100, -100]],
            ]
        )
        logits = torch.zeros(2, 2, 3, 8)
        logits[0, 0, 0, 1] = 3.0
        logits[0, 0, 1, 2] = 3.0
        logits[0, 1, 0, 3] = 3.0
        logits[0, 1, 1, 4] = 3.0
        logits[0, 1, 2, 5] = 3.0
        logits[1, 0, 0, 1] = 3.0
        logits[1, 1, 0, 3] = 3.0

        good_loss, good_metrics = module.algorithmic_role_value_trace_margin_loss(
            logits,
            targets,
            margin=0.5,
        )

        self.assertEqual(float(good_loss), 0.0)
        self.assertEqual(
            float(good_metrics["algorithmic_role_value_trace_margin_trace_pass_rate"]),
            1.0,
        )

        weak_logits = logits.clone()
        weak_logits[0, 1, 2, 6] = 3.2
        weak_loss, weak_metrics = module.algorithmic_role_value_trace_margin_loss(
            weak_logits,
            targets,
            margin=0.5,
        )

        self.assertGreater(float(weak_loss), 0.0)
        self.assertLess(
            float(weak_metrics["algorithmic_role_value_trace_margin_trace_pass_rate"]),
            1.0,
        )

    def test_algorithmic_role_value_state_ce_can_upweight_scalar_roles(self):
        import torch

        module = _load_module()

        logits = torch.zeros(1, 1, 4, 8)
        targets = torch.tensor([[[1, 2, 3, 4]]])
        logits[0, 0, 0, 1] = 5.0
        logits[0, 0, 1, 2] = 5.0
        logits[0, 0, 2, 0] = 5.0
        logits[0, 0, 3, 0] = 5.0

        plain_loss, _ = module.algorithmic_role_value_state_ce_loss(logits, targets)
        weights = module.algorithmic_role_value_scalar_role_weights(
            targets,
            multiplier=5.0,
        )
        weighted_loss, _ = module.algorithmic_role_value_state_ce_loss(
            logits,
            targets,
            role_weights=weights,
        )

        self.assertGreater(float(weighted_loss), float(plain_loss))

    def test_core_role_value_prompt_parity_loss_uses_base_parity(self):
        import torch

        module = _load_module()

        odd_target = module.core_role_value_prompt_parity_target(
            {"list_value_start": 40001},
            device="cpu",
        )
        even_target = module.core_role_value_prompt_parity_target(
            {"list_value_start": 40002},
            device="cpu",
        )

        self.assertEqual(int(odd_target.item()), 1)
        self.assertEqual(int(even_target.item()), 0)

        logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
        targets = torch.tensor([1, 0])
        loss, metrics = module.core_role_value_prompt_parity_ce_loss(
            logits,
            targets,
        )

        self.assertLess(float(loss), 0.01)
        self.assertEqual(float(metrics["core_role_value_prompt_parity_acc"]), 1.0)

    def test_initial_role_value_targets_can_encode_input_metadata(self):
        module = _load_module()

        row = {
            "list_value_start": 50001,
            "list_length": 7,
            "mixed_offset": 9,
            "depth_targets": {
                "1": "50002,50004,50006",
                "2": "100004,100008,100012",
                "3": "300024",
                "4": "300015",
            },
        }

        plain = module.algorithmic_role_value_initial_state_targets(
            row,
            num_steps=1,
            num_roles=10,
            value_vocab_size=128,
            device="cpu",
            include_metadata=False,
        )
        metadata = module.algorithmic_role_value_initial_state_targets(
            row,
            num_steps=1,
            num_roles=10,
            value_vocab_size=128,
            device="cpu",
            include_metadata=True,
        )

        self.assertEqual(int(plain[0, 0, 8]), 2)
        self.assertEqual(int(metadata[0, 0, 8]), 2)
        self.assertEqual(int(plain[0, 0, 4]), -100)
        self.assertEqual(int(metadata[0, 0, 4]), 7)
        self.assertEqual(int(plain[0, 0, 9]), -100)
        self.assertEqual(int(metadata[0, 0, 9]), 10)

    def test_initial_role_value_targets_can_encode_absolute_input_list_without_base(self):
        module = _load_module()

        row = {
            "task_family": "list_transform",
            "input_list": [14, 31, 10, 24, 27],
            "role_value_list_class_mode": "absolute",
        }

        targets = module.algorithmic_role_value_initial_state_targets(
            row,
            num_steps=1,
            num_roles=10,
            value_vocab_size=128,
            device="cpu",
            include_metadata=False,
        )

        self.assertEqual(targets[0, 0, :4].tolist(), [15, 32, 11, 25])
        self.assertEqual(int(targets[0, 0, 4]), -100)

    def test_initial_role_value_targets_can_encode_source_positions_without_base(self):
        module = _load_module()

        row = {
            "task_family": "list_transform",
            "input_list": [14, 31, 10, 24, 27],
            "role_value_list_class_mode": "source_position",
        }

        targets = module.algorithmic_role_value_initial_state_targets(
            row,
            num_steps=1,
            num_roles=10,
            value_vocab_size=128,
            device="cpu",
            include_metadata=False,
        )

        self.assertEqual(targets[0, 0, :4].tolist(), [1, 2, 3, 4])
        self.assertEqual(int(targets[0, 0, 4]), -100)

    def test_parser_accepts_numeric_source_feature_options(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--out-dir",
                "out",
                "--numeric-source-features",
                "--numeric-source-max-list-len",
                "7",
                "--numeric-source-value-vocab-size",
                "64",
            ]
        )

        self.assertTrue(args.numeric_source_features)
        self.assertEqual(args.numeric_source_max_list_len, 7)
        self.assertEqual(args.numeric_source_value_vocab_size, 64)

    def test_parser_accepts_token_numeric_value_feature_options(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--out-dir",
                "out",
                "--token-numeric-value-features",
                "--token-numeric-value-vocab-size",
                "64",
            ]
        )

        self.assertTrue(args.token_numeric_value_features)
        self.assertEqual(args.token_numeric_value_vocab_size, 64)

    def test_parser_accepts_token_numeric_source_slot_options(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--out-dir",
                "out",
                "--token-numeric-source-slots",
                "--token-numeric-source-slot-vocab-size",
                "64",
                "--token-numeric-source-slot-max-slots",
                "7",
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-gate-min",
                "1.0",
                "--token-numeric-source-slot-parity-ce-weight",
                "0.7",
                "--token-numeric-source-slot-predicate-feedback",
                "--token-numeric-source-slot-predicate-ce-weight",
                "0.9",
                "--core-source-position-binder-source-slots-only",
                "--core-source-position-binder-raw-source-slots",
                "--core-source-position-binder-gate-min",
                "1.0",
                "--core-source-position-binder-state-gate-min",
                "0.25",
                "--core-source-position-binder-state-st",
                "--core-source-position-binder-query-state",
                "--core-source-position-binder-query-state-gate-min",
                "0.5",
                "--core-source-value-binder",
                "--core-source-value-binder-state-gate-min",
                "0.75",
                "--core-source-value-binder-state-st",
                "--core-source-value-prompt-ce-weight",
                "0.8",
                "--core-primitive-role-value-source-value-conditioning",
                "--core-primitive-role-value-source-value-gate-min",
                "0.6",
                "--core-primitive-role-value-pair-trace-contrast-weight",
                "1.2",
                "--core-primitive-role-value-pair-trace-contrast-margin",
                "0.4",
            ]
        )

        self.assertTrue(args.token_numeric_source_slots)
        self.assertEqual(args.token_numeric_source_slot_vocab_size, 64)
        self.assertEqual(args.token_numeric_source_slot_max_slots, 7)
        self.assertEqual(args.token_numeric_source_slot_id_mode, "relative_parity")
        self.assertEqual(args.token_numeric_source_slot_gate_min, 1.0)
        self.assertEqual(args.token_numeric_source_slot_parity_ce_weight, 0.7)
        self.assertTrue(args.token_numeric_source_slot_predicate_feedback)
        self.assertEqual(args.token_numeric_source_slot_predicate_ce_weight, 0.9)
        self.assertTrue(args.core_source_position_binder_source_slots_only)
        self.assertTrue(args.core_source_position_binder_raw_source_slots)
        self.assertEqual(args.core_source_position_binder_gate_min, 1.0)
        self.assertEqual(args.core_source_position_binder_state_gate_min, 0.25)
        self.assertTrue(args.core_source_position_binder_state_st)
        self.assertTrue(args.core_source_position_binder_query_state)
        self.assertEqual(args.core_source_position_binder_query_state_gate_min, 0.5)
        self.assertTrue(args.core_source_value_binder)
        self.assertEqual(args.core_source_value_binder_state_gate_min, 0.75)
        self.assertTrue(args.core_source_value_binder_state_st)
        self.assertEqual(args.core_source_value_prompt_ce_weight, 0.8)
        self.assertTrue(args.core_primitive_role_value_source_value_conditioning)
        self.assertEqual(args.core_primitive_role_value_source_value_gate_min, 0.6)
        self.assertEqual(args.core_primitive_role_value_pair_trace_contrast_weight, 1.2)
        self.assertEqual(args.core_primitive_role_value_pair_trace_contrast_margin, 0.4)

    def test_numeric_source_visual_tensors_match_configured_visual_dim(self):
        module = _load_module()

        features, mask = module.row_numeric_source_visual_tensors(
            {"input_list": [2, 5]},
            visual_dim=16,
            max_list_len=4,
            value_vocab_size=8,
            device="cpu",
        )

        self.assertEqual(tuple(features.shape), (1, 4, 16))
        self.assertEqual(tuple(mask.shape), (1, 4))
        self.assertEqual(mask.tolist(), [[1, 1, 0, 0]])

    def test_core_role_value_template_targets_encode_length_parity_and_offset(self):
        import torch

        module = _load_module()

        target = module.core_role_value_template_targets(
            {"list_value_start": 50001, "list_length": 7, "mixed_offset": 9},
            num_templates=64,
            device="cpu",
        )
        self.assertEqual(target.tolist(), [27])

        logits = torch.zeros(1, 64)
        logits[0, 27] = 5.0
        loss, metrics = module.core_role_value_template_ce_loss(logits, target)

        self.assertLess(float(loss), 1.0)
        self.assertEqual(float(metrics["core_role_value_template_acc"]), 1.0)

    def test_core_role_value_template_table_ce_scores_gold_template_row(self):
        import torch

        module = _load_module()

        table = torch.zeros(4, 2, 3, 8, requires_grad=True)
        table.data[2, 0, 0, 5] = 5.0
        table.data[2, 1, 2, 6] = 5.0
        template_targets = torch.tensor([2])
        role_targets = torch.tensor(
            [
                [
                    [5, -100, -100],
                    [-100, -100, 6],
                ]
            ]
        )

        loss, metrics = module.core_role_value_template_table_ce_loss(
            table,
            template_targets,
            role_targets,
            num_steps=2,
        )

        self.assertLess(float(loss), 0.1)
        self.assertEqual(float(metrics["core_role_value_template_table_acc"]), 1.0)
        self.assertEqual(
            float(metrics["core_role_value_template_table_step_exact"]),
            1.0,
        )
        self.assertEqual(
            float(metrics["core_role_value_template_table_samples"]),
            2.0,
        )

    def test_primitive_transition_operation_ce_loss_scores_logits(self):
        import torch

        module = _load_module()

        logits = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            ]
        )
        targets = torch.tensor([[0, 1, -100]])

        loss, metrics = module.primitive_transition_operation_ce_loss(logits, targets)

        self.assertLess(float(loss), 0.1)
        self.assertEqual(float(metrics["primitive_transition_operation_acc"]), 1.0)

    def test_core_typed_register_operation_targets_follow_transition_codes(self):
        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [99]

        row = {"transition_state_codes": {"1": 0, "2": 3, "4": 4}}

        targets = module.core_typed_register_operation_targets(
            FakeTokenizer(),
            row,
            num_steps=4,
            num_operations=5,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[0, 3, -100, 4]])

    def test_core_typed_register_operation_targets_can_shift_for_transition_readout(self):
        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [99]

        row = {"transition_state_codes": {"1": 0, "2": 3, "4": 4}}

        targets = module.core_typed_register_operation_targets(
            FakeTokenizer(),
            row,
            num_steps=4,
            num_operations=5,
            device="cpu",
            target_shift=1,
        )

        self.assertEqual(targets.tolist(), [[3, -100, 4, -100]])

    def test_core_typed_register_operation_ce_loss_renames_metrics(self):
        import torch

        module = _load_module()

        logits = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            ]
        )
        targets = torch.tensor([[0, 1, -100]])

        loss, metrics = module.core_typed_register_operation_ce_loss(logits, targets)

        self.assertLess(float(loss), 0.1)
        self.assertEqual(float(metrics["core_typed_register_operation_acc"]), 1.0)
        self.assertIn("core_typed_register_operation_ce", metrics)

    def test_core_typed_register_transition_ce_loss_shifts_targets(self):
        import torch

        module = _load_module()

        logits = torch.zeros(1, 2, 3, 8)
        targets = torch.tensor(
            [
                [
                    [1, 2, 3],
                    [4, 5, -100],
                    [6, -100, 7],
                ]
            ]
        )
        logits[:, 0, 0, 4] = 8.0
        logits[:, 0, 1, 5] = 8.0
        logits[:, 1, 0, 6] = 8.0
        logits[:, 1, 2, 7] = 8.0

        loss, metrics = module.core_typed_register_transition_ce_loss(
            logits,
            targets,
        )

        self.assertLess(float(loss), 0.01)
        self.assertEqual(float(metrics["core_typed_register_transition_acc"]), 1.0)
        self.assertEqual(
            float(metrics["core_typed_register_transition_step_exact"]),
            1.0,
        )

    def test_parser_allows_explicit_random_init_without_checkpoint(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--allow-random-init",
            ]
        )

        self.assertEqual(args.init_checkpoint, "")
        self.assertTrue(args.allow_random_init)

    def test_validate_init_checkpoint_requires_opt_in_for_random_init(self):
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "--init-checkpoint"):
            module.validate_init_checkpoint_args("", allow_random_init=False)

        self.assertEqual(
            module.validate_init_checkpoint_args("", allow_random_init=True),
            "random_init",
        )
        self.assertEqual(
            module.validate_init_checkpoint_args("seed.pt", allow_random_init=False),
            "checkpoint",
        )

    def test_random_noise_warmup_batch_uses_random_tokens_and_targets(self):
        import torch

        module = _load_module()
        generator = torch.Generator().manual_seed(123)

        input_ids, attention_mask, target_ids = module.build_random_noise_warmup_batch(
            vocab_size=17,
            seq_len=5,
            batch_size=3,
            device="cpu",
            target_vocab_size=7,
            generator=generator,
        )

        self.assertEqual(tuple(input_ids.shape), (3, 5))
        self.assertEqual(tuple(attention_mask.shape), (3, 5))
        self.assertEqual(tuple(target_ids.shape), (3,))
        self.assertTrue(torch.equal(attention_mask, torch.ones_like(input_ids)))
        self.assertGreaterEqual(int(input_ids.min()), 0)
        self.assertLess(int(input_ids.max()), 17)
        self.assertGreaterEqual(int(target_ids.min()), 0)
        self.assertLess(int(target_ids.max()), 7)

    def test_random_noise_warmup_loss_prefers_matching_random_label(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([2])
        final_logits = torch.zeros(1, 6)
        depth_text_logits = torch.zeros(1, 3, 1, 6)
        final_logits[:, 2] = 8.0
        depth_text_logits[:, :, 0, 2] = 8.0

        matching, metrics = module.random_noise_warmup_loss(
            final_logits,
            depth_text_logits,
            target_ids,
            final_ce_weight=1.0,
            depth_ce_weight=1.0,
        )

        final_logits[:, 2] = 0.0
        final_logits[:, 1] = 8.0
        depth_text_logits[:, :, 0, 2] = 0.0
        depth_text_logits[:, :, 0, 1] = 8.0
        mismatched, _ = module.random_noise_warmup_loss(
            final_logits,
            depth_text_logits,
            target_ids,
            final_ce_weight=1.0,
            depth_ce_weight=1.0,
        )

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["noise_warmup_final_acc"]), 1.0)
        self.assertEqual(float(metrics["noise_warmup_depth_acc"]), 1.0)

    def test_random_noise_warmup_loss_can_penalize_overconfident_noise_logits(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([2])
        uniform_final_logits = torch.zeros(1, 6)
        uniform_depth_logits = torch.zeros(1, 3, 1, 6)
        sharp_final_logits = torch.zeros(1, 6)
        sharp_depth_logits = torch.zeros(1, 3, 1, 6)
        sharp_final_logits[:, 1] = 8.0
        sharp_depth_logits[:, :, 0, 1] = 8.0

        uniform_loss, uniform_metrics = module.random_noise_warmup_loss(
            uniform_final_logits,
            uniform_depth_logits,
            target_ids,
            final_ce_weight=0.0,
            depth_ce_weight=0.0,
            uniform_weight=1.0,
        )
        sharp_loss, sharp_metrics = module.random_noise_warmup_loss(
            sharp_final_logits,
            sharp_depth_logits,
            target_ids,
            final_ce_weight=0.0,
            depth_ce_weight=0.0,
            uniform_weight=1.0,
        )

        self.assertLess(float(uniform_loss), float(sharp_loss))
        self.assertLess(float(uniform_metrics["noise_warmup_final_uniform_ce"]), 2.0)
        self.assertGreater(float(sharp_metrics["noise_warmup_final_uniform_ce"]), 2.0)

    def test_staged_internal_first_token_targets_marks_only_labelled_depths(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {
                    " stage-one": [11],
                    " stage-two": [22],
                    " final": [44],
                }[text]

        row = {
            "chosen": "final",
            "depth_targets": {"1": "stage-one", "2": "stage-two", "4": "final"},
        }

        targets = module.staged_internal_first_token_targets(
            FakeTokenizer(),
            row,
            num_depths=5,
            device="cpu",
        )

        self.assertTrue(torch.equal(targets, torch.tensor([[11, 22, -100, 44, -100]])))

    def test_staged_internal_first_token_targets_can_use_content_token(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {
                    " 123": [220, 1001],
                    "123": [1001],
                }[text]

        row = {"depth_targets": {"1": "123"}}

        targets = module.staged_internal_first_token_targets(
            FakeTokenizer(),
            row,
            num_depths=1,
            device="cpu",
            content_token=True,
        )

        self.assertTrue(torch.equal(targets, torch.tensor([[1001]])))

    def test_staged_internal_first_token_ce_loss_masks_unlabelled_depths(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 4, 1, 8)
        targets = torch.tensor([[2, -100, 4, -100]])

        logits[:, 0, 0, 2] = 8.0
        logits[:, 2, 0, 4] = 8.0
        matching, metrics = module.staged_internal_first_token_ce_loss(logits, targets)

        logits[:, 0, 0, 2] = 0.0
        logits[:, 0, 0, 1] = 8.0
        mismatched, _ = module.staged_internal_first_token_ce_loss(logits, targets)

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["staged_internal_first_token_samples"]), 2.0)
        self.assertEqual(float(metrics["staged_internal_first_token_acc"]), 1.0)

    def test_staged_internal_sequence_targets_keep_full_depth_value_tokens(self):
        import torch

        module = _load_module()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {
                    " filtered": [11, 12],
                    " doubled": [21, 22, 23],
                    " final": [31],
                }[text]

        row = {
            "chosen": "final",
            "depth_targets": {"1": "filtered", "2": "doubled", "4": "final"},
        }

        targets = module.staged_internal_sequence_targets(
            FakeTokenizer(),
            row,
            num_depths=4,
            max_target_tokens=3,
            device="cpu",
        )

        self.assertTrue(
            torch.equal(
                targets,
                torch.tensor(
                    [[[11, 12, -100], [21, 22, 23], [-100, -100, -100], [31, -100, -100]]]
                ),
            )
        )

    def test_staged_internal_sequence_ce_loss_masks_unlabelled_tokens(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 3, 2, 8)
        targets = torch.tensor([[[2, 3], [-100, -100], [4, -100]]])

        logits[:, 0, 0, 2] = 8.0
        logits[:, 0, 1, 3] = 8.0
        logits[:, 2, 0, 4] = 8.0
        matching, metrics = module.staged_internal_sequence_ce_loss(logits, targets)

        logits[:, 0, 1, 3] = 0.0
        logits[:, 0, 1, 1] = 8.0
        mismatched, _ = module.staged_internal_sequence_ce_loss(logits, targets)

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["staged_internal_sequence_samples"]), 3.0)
        self.assertEqual(float(metrics["staged_internal_sequence_acc"]), 1.0)

    def test_algorithmic_value_state_ce_can_ignore_pad_slots(self):
        import torch

        module = _load_module()
        kind_logits = torch.zeros(1, 1, 3)
        kind_logits[:, :, 1] = 8.0
        slot_targets = torch.tensor([[[2, 4, 0, 0]]])
        kind_targets = torch.tensor([[1]])

        content_wrong = torch.zeros(1, 1, 4, 8)
        content_wrong[:, :, 0, 0] = 8.0
        content_wrong[:, :, 1, 0] = 8.0
        content_wrong[:, :, 2, 0] = 8.0
        content_wrong[:, :, 3, 0] = 8.0
        wrong_loss, wrong_metrics = module.algorithmic_value_state_ce_loss(
            kind_logits,
            content_wrong,
            kind_targets,
            slot_targets,
            pad_ce_weight=0.0,
        )

        content_right = content_wrong.clone()
        content_right[:, :, 0, 0] = 0.0
        content_right[:, :, 0, 2] = 8.0
        content_right[:, :, 1, 0] = 0.0
        content_right[:, :, 1, 4] = 8.0
        right_loss, right_metrics = module.algorithmic_value_state_ce_loss(
            kind_logits,
            content_right,
            kind_targets,
            slot_targets,
            pad_ce_weight=0.0,
        )

        self.assertGreater(float(wrong_loss), float(right_loss) + 1.0)
        self.assertEqual(float(wrong_metrics["algorithmic_value_state_content_slot_acc"]), 0.0)
        self.assertEqual(float(right_metrics["algorithmic_value_state_content_slot_acc"]), 1.0)

    def test_transition_state_first_token_ce_loss_masks_unlabelled_depths(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 4, 8)
        targets = torch.tensor([[2, -100, 4, -100]])

        logits[:, 0, 2] = 8.0
        logits[:, 2, 4] = 8.0
        matching, metrics = module.transition_state_first_token_ce_loss(logits, targets)

        logits[:, 0, 2] = 0.0
        logits[:, 0, 1] = 8.0
        mismatched, _ = module.transition_state_first_token_ce_loss(logits, targets)

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["transition_state_first_token_samples"]), 2.0)
        self.assertEqual(float(metrics["transition_state_first_token_acc"]), 1.0)

    def test_transition_state_depth_contrast_loss_separates_row_depth_targets(self):
        import torch

        module = _load_module()
        targets = torch.tensor([[1, 2, -100, 2]])
        good_logits = torch.zeros(1, 4, 4)
        good_logits[0, 0, 1] = 3.0
        good_logits[0, 0, 2] = 0.0
        good_logits[0, 1, 2] = 3.0
        good_logits[0, 1, 1] = 0.0
        good_logits[0, 3, 2] = 3.0
        good_logits[0, 3, 1] = 0.0
        collapsed_logits = torch.zeros(1, 4, 4)
        collapsed_logits[:, :, 1] = 3.0

        good_loss, good_metrics = module.transition_state_depth_contrast_loss(
            good_logits,
            targets,
            margin=0.5,
        )
        collapsed_loss, _ = module.transition_state_depth_contrast_loss(
            collapsed_logits,
            targets,
            margin=0.5,
        )

        self.assertLess(float(good_loss), float(collapsed_loss))
        self.assertEqual(float(good_metrics["transition_state_depth_contrast_pairs"]), 3.0)

    def test_transition_state_code_targets_use_explicit_semantic_codes(self):
        module = _load_module()

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [5] if "alpha" in text else [9]

        row = {
            "depth_targets": {"1": "alpha", "3": "beta"},
            "transition_state_codes": {"1": 7, "3": 11},
        }

        targets = module.transition_state_code_targets(
            Tokenizer(),
            row,
            num_depths=4,
            codebook_size=16,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[7, -100, 11, -100]])

    def test_transition_state_code_targets_reject_codes_outside_codebook(self):
        module = _load_module()

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [5]

        row = {"transition_state_codes": {"1": 17}}

        with self.assertRaisesRegex(ValueError, "transition_state_codes"):
            module.transition_state_code_targets(
                Tokenizer(),
                row,
                num_depths=1,
                codebook_size=16,
                device="cpu",
            )

    def test_transition_state_code_targets_fallback_hashes_first_tokens_into_codebook(self):
        module = _load_module()

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [5] if "alpha" in text else [9]

        row = {"depth_targets": {"1": "alpha", "3": "beta"}}

        targets = module.transition_state_code_targets(
            Tokenizer(),
            row,
            num_depths=4,
            codebook_size=4,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[1, -100, 1, -100]])

    def test_depth_choice_sequence_margin_matches_mean_forced_choice_scoring(self):
        import torch

        module = _load_module()
        depth_logits = torch.zeros(1, 2, 3, 8)
        final_logits = torch.zeros(1, 3, 8)
        chosen = torch.tensor([[1, 2, 3]])
        rejected = torch.tensor([[4, 5, 6]])

        depth_logits[:, :, 0, 1] = 6.0
        depth_logits[:, :, 1, 2] = 6.0
        depth_logits[:, :, 2, 3] = 6.0
        final_logits[:, 0, 1] = 6.0
        final_logits[:, 1, 2] = 6.0
        final_logits[:, 2, 3] = 6.0
        matching, metrics = module.depth_choice_sequence_margin_loss(
            depth_logits,
            final_logits,
            chosen,
            rejected,
            margin=0.2,
            all_depth_weight=1.0,
            final_weight=1.0,
        )

        depth_logits[:, :, 0, 1] = 0.0
        depth_logits[:, :, 1, 2] = 0.0
        depth_logits[:, :, 2, 3] = 0.0
        depth_logits[:, :, 0, 4] = 6.0
        depth_logits[:, :, 1, 5] = 6.0
        depth_logits[:, :, 2, 6] = 6.0
        final_logits[:, 0, 1] = 0.0
        final_logits[:, 1, 2] = 0.0
        final_logits[:, 2, 3] = 0.0
        final_logits[:, 0, 4] = 6.0
        final_logits[:, 1, 5] = 6.0
        final_logits[:, 2, 6] = 6.0
        mismatched, _ = module.depth_choice_sequence_margin_loss(
            depth_logits,
            final_logits,
            chosen,
            rejected,
            margin=0.2,
            all_depth_weight=1.0,
            final_weight=1.0,
        )

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertIn("choice_sequence_margin_all_depth", metrics)
        self.assertIn("choice_sequence_margin_final_path", metrics)

    def test_final_choice_sequence_margin_uses_final_lm_path_only(self):
        import torch

        module = _load_module()
        final_logits = torch.zeros(1, 3, 8)
        chosen = torch.tensor([[1, 2, 3]])
        rejected = torch.tensor([[4, 5, 6]])

        final_logits[:, 0, 1] = 6.0
        final_logits[:, 1, 2] = 6.0
        final_logits[:, 2, 3] = 6.0
        matching, metrics = module.final_choice_sequence_margin_loss(
            final_logits,
            chosen,
            rejected,
            margin=0.2,
        )

        final_logits[:, 0, 1] = 0.0
        final_logits[:, 1, 2] = 0.0
        final_logits[:, 2, 3] = 0.0
        final_logits[:, 0, 4] = 6.0
        final_logits[:, 1, 5] = 6.0
        final_logits[:, 2, 6] = 6.0
        mismatched, _ = module.final_choice_sequence_margin_loss(
            final_logits,
            chosen,
            rejected,
            margin=0.2,
        )

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertIn("final_choice_sequence_margin_final_path", metrics)

    def test_parser_accepts_final_choice_margin_for_final_path_training(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--final-path-only-supervision",
                "--target-logit-positions-only",
                "--final-choice-margin-weight",
                "0.7",
                "--final-choice-margin",
                "0.03",
            ]
        )

        self.assertTrue(args.final_path_only_supervision)
        self.assertEqual(args.final_choice_margin_weight, 0.7)
        self.assertEqual(args.final_choice_margin, 0.03)

    def test_parser_accepts_typed_value_bridge_final_contrast(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-jsonl",
                "rows.jsonl",
                "--target-logit-positions-only",
                "--final-path-only-supervision",
                "--typed-value-answer-bridge-final-contrast-weight",
                "0.8",
                "--typed-value-answer-bridge-final-contrast-margin",
                "0.04",
            ]
        )

        self.assertEqual(args.typed_value_answer_bridge_final_contrast_weight, 0.8)
        self.assertEqual(args.typed_value_answer_bridge_final_contrast_margin, 0.04)

    def test_transition_state_code_ce_loss_masks_unlabelled_depths(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 4, 6)
        targets = torch.tensor([[2, -100, 4, -100]])

        logits[:, 0, 2] = 8.0
        logits[:, 2, 4] = 8.0
        matching, metrics = module.transition_state_code_ce_loss(logits, targets)

        logits[:, 0, 2] = 0.0
        logits[:, 0, 1] = 8.0
        mismatched, _ = module.transition_state_code_ce_loss(logits, targets)

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["transition_state_code_samples"]), 2.0)
        self.assertEqual(float(metrics["transition_state_code_acc"]), 1.0)

    def test_transition_state_finality_targets_use_explicit_binary_targets(self):
        module = _load_module()

        row = {"transition_finality_targets": {"1": 0, "2": 1, "4": 1}}

        targets = module.transition_state_finality_targets(
            row,
            num_depths=4,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[0.0, 1.0, -100.0, 1.0]])

    def test_transition_state_finality_bce_loss_masks_unlabelled_depths(self):
        import torch

        module = _load_module()
        logits = torch.tensor([[[-8.0], [7.0], [3.0], [8.0]]]).squeeze(-1)
        targets = torch.tensor([[0.0, 1.0, -100.0, 1.0]])

        matching, metrics = module.transition_state_finality_bce_loss(logits, targets)

        wrong_logits = torch.tensor([[[8.0], [-7.0], [3.0], [8.0]]]).squeeze(-1)
        mismatched, _ = module.transition_state_finality_bce_loss(
            wrong_logits,
            targets,
        )

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["transition_state_finality_samples"]), 3.0)
        self.assertEqual(float(metrics["transition_state_finality_acc"]), 1.0)

    def test_transition_state_joint_targets_combine_code_and_finality(self):
        module = _load_module()

        row = {
            "transition_state_codes": {"1": 0, "2": 1, "4": 3},
            "transition_finality_targets": {"1": 0, "2": 1, "4": 1},
        }

        targets = module.transition_state_joint_targets(
            row,
            num_depths=4,
            joint_size=8,
            device="cpu",
        )

        self.assertEqual(targets.tolist(), [[0, 3, -100, 7]])

    def test_transition_state_joint_ce_loss_masks_unlabelled_depths(self):
        import torch

        module = _load_module()
        logits = torch.zeros(1, 4, 8)
        targets = torch.tensor([[0, 3, -100, 7]])

        logits[:, 0, 0] = 8.0
        logits[:, 1, 3] = 8.0
        logits[:, 3, 7] = 8.0
        matching, metrics = module.transition_state_joint_ce_loss(logits, targets)

        logits[:, 1, 3] = 0.0
        logits[:, 1, 2] = 8.0
        mismatched, _ = module.transition_state_joint_ce_loss(logits, targets)

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), float(matching) + 1.0)
        self.assertEqual(float(metrics["transition_state_joint_samples"]), 3.0)
        self.assertEqual(float(metrics["transition_state_joint_acc"]), 1.0)

    def test_causal_prefix_example_loss_weight_keeps_first_token_full(self):
        module = _load_module()

        self.assertEqual(module._causal_prefix_example_loss_weight(0, 0.1), 1.0)
        self.assertEqual(module._causal_prefix_example_loss_weight(1, 0.1), 0.1)
        self.assertEqual(module._causal_prefix_example_loss_weight(3, 0.25), 0.25)
        with self.assertRaises(ValueError):
            module._causal_prefix_example_loss_weight(1, -0.1)

    def test_teacher_depth_kl_applies_only_to_first_prefix_token(self):
        module = _load_module()

        self.assertTrue(module._should_apply_teacher_first_token_depth_kl(0, 0.5))
        self.assertFalse(module._should_apply_teacher_first_token_depth_kl(1, 0.5))
        self.assertFalse(module._should_apply_teacher_first_token_depth_kl(0, 0.0))

    def test_depth_text_logit_distillation_loss_prefers_matching_teacher_logits(self):
        import torch

        module = _load_module()
        teacher_logits = torch.zeros(1, 2, 1, 5)
        teacher_logits[:, :, :, 3] = 4.0
        matching_logits = teacher_logits.clone()
        mismatched_logits = torch.zeros(1, 2, 1, 5)
        mismatched_logits[:, :, :, 1] = 4.0

        matching = module.depth_text_logit_distillation_loss(
            matching_logits,
            teacher_logits,
            temperature=1.0,
        )
        mismatched = module.depth_text_logit_distillation_loss(
            mismatched_logits,
            teacher_logits,
            temperature=1.0,
        )

        self.assertLess(float(matching), 1e-6)
        self.assertGreater(float(mismatched), float(matching) + 1.0)

    def test_depth_sequence_supervision_loss_uses_all_answer_tokens(self):
        import torch

        module = _load_module()
        depth_logits = torch.zeros(1, 2, 2, 5)
        final_logits = torch.zeros(1, 2, 5)
        target_ids = torch.tensor([[3, 4]])
        depth_logits[:, -1, 0, 3] = 10.0
        depth_logits[:, -1, 1, 4] = 10.0
        final_logits[:, 0, 3] = 10.0
        final_logits[:, 1, 4] = 10.0

        loss, metrics = module.depth_sequence_supervision_loss(
            depth_logits,
            final_logits,
            target_ids,
            final_logit_ce_weight=1.0,
            all_depth_ce_weight=0.0,
            progress_margin_weight=0.0,
            progress_margin=0.10,
        )

        self.assertLess(float(loss), 0.01)
        self.assertEqual(float(metrics["depth_final_acc"]), 1.0)
        self.assertEqual(float(metrics["final_path_acc"]), 1.0)

    def test_depth_sequence_supervision_loss_can_disable_depth_final_ce(self):
        import torch

        module = _load_module()
        depth_logits = torch.zeros(1, 2, 2, 5)
        final_logits = torch.zeros(1, 2, 5)
        target_ids = torch.tensor([[3, 4]])

        loss, _ = module.depth_sequence_supervision_loss(
            depth_logits,
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            depth_final_ce_weight=0.0,
            all_depth_ce_weight=0.0,
            progress_margin_weight=0.0,
            progress_margin=0.10,
        )

        self.assertEqual(float(loss), 0.0)

    def test_depth_sequence_supervision_loss_can_penalize_greedy_competitor(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[3]])
        depth_logits = torch.zeros(1, 2, 1, 5)
        final_logits = torch.zeros(1, 1, 5)
        depth_logits[:, -1, 0, 2] = 2.0
        depth_logits[:, -1, 0, 3] = 1.0
        final_logits[:, 0, 2] = 2.0
        final_logits[:, 0, 3] = 1.0

        bad_loss, bad_metrics = module.depth_sequence_supervision_loss(
            depth_logits,
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            depth_final_ce_weight=0.0,
            all_depth_ce_weight=0.0,
            progress_margin_weight=0.0,
            progress_margin=0.10,
            final_greedy_token_margin_weight=1.0,
            depth_greedy_token_margin_weight=1.0,
            greedy_token_margin=0.5,
        )

        final_logits[:, 0, 3] = 4.0
        depth_logits[:, -1, 0, 3] = 4.0
        good_loss, good_metrics = module.depth_sequence_supervision_loss(
            depth_logits,
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            depth_final_ce_weight=0.0,
            all_depth_ce_weight=0.0,
            progress_margin_weight=0.0,
            progress_margin=0.10,
            final_greedy_token_margin_weight=1.0,
            depth_greedy_token_margin_weight=1.0,
            greedy_token_margin=0.5,
        )

        self.assertGreater(float(bad_loss), 2.0)
        self.assertEqual(float(bad_metrics["final_greedy_token_win_rate"]), 0.0)
        self.assertLess(float(good_loss), 1e-6)
        self.assertEqual(float(good_metrics["final_greedy_token_win_rate"]), 1.0)

    def test_final_path_only_supervision_loss_avoids_depth_logits(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[3]])
        final_logits = torch.zeros(1, 1, 5)
        final_logits[:, 0, 2] = 2.0
        final_logits[:, 0, 3] = 1.0

        bad_loss, bad_metrics = module.final_path_sequence_supervision_loss(
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            final_greedy_token_margin_weight=1.0,
            greedy_token_margin=0.5,
        )

        final_logits[:, 0, 3] = 4.0
        good_loss, good_metrics = module.final_path_sequence_supervision_loss(
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            final_greedy_token_margin_weight=1.0,
            greedy_token_margin=0.5,
        )

        self.assertGreater(float(bad_loss), 1.0)
        self.assertEqual(float(bad_metrics["final_greedy_token_win_rate"]), 0.0)
        self.assertEqual(float(bad_metrics["depth_final_ce"]), 0.0)
        self.assertLess(float(good_loss), 1e-6)
        self.assertEqual(float(good_metrics["final_greedy_token_win_rate"]), 1.0)

    def test_core_role_value_vocab_renderer_loss_targets_renderer_logits(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[3]])
        renderer_logits = torch.zeros(1, 1, 5)
        renderer_logits[:, 0, 2] = 2.0
        renderer_logits[:, 0, 3] = 1.0

        bad_loss, bad_metrics = (
            module.core_role_value_vocab_renderer_sequence_supervision_loss(
                renderer_logits,
                target_ids,
                renderer_ce_weight=0.0,
                renderer_greedy_token_margin_weight=1.0,
                greedy_token_margin=0.5,
            )
        )

        renderer_logits[:, 0, 3] = 4.0
        good_loss, good_metrics = (
            module.core_role_value_vocab_renderer_sequence_supervision_loss(
                renderer_logits,
                target_ids,
                renderer_ce_weight=0.0,
                renderer_greedy_token_margin_weight=1.0,
                greedy_token_margin=0.5,
            )
        )

        self.assertGreater(float(bad_loss), 1.0)
        self.assertEqual(
            float(bad_metrics["core_role_value_vocab_renderer_greedy_token_win_rate"]),
            0.0,
        )
        self.assertLess(float(good_loss), 1e-6)
        self.assertEqual(
            float(good_metrics["core_role_value_vocab_renderer_greedy_token_win_rate"]),
            1.0,
        )

    def test_depth_sequence_supervision_loss_penalizes_adjacent_depth_regression(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[1]])
        final_logits = torch.zeros(1, 1, 3)
        improving_logits = torch.zeros(1, 3, 1, 3)
        improving_logits[:, 0, 0, 1] = 0.0
        improving_logits[:, 1, 0, 1] = 2.0
        improving_logits[:, 2, 0, 1] = 4.0
        regressing_logits = torch.zeros(1, 3, 1, 3)
        regressing_logits[:, 0, 0, 1] = 4.0
        regressing_logits[:, 1, 0, 1] = 2.0
        regressing_logits[:, 2, 0, 1] = 0.0

        improving_loss, improving_metrics = module.depth_sequence_supervision_loss(
            improving_logits,
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            depth_final_ce_weight=0.0,
            all_depth_ce_weight=0.0,
            progress_margin_weight=0.0,
            progress_margin=0.10,
            depth_trajectory_monotonic_weight=1.0,
            depth_trajectory_monotonic_margin=0.02,
        )
        regressing_loss, regressing_metrics = module.depth_sequence_supervision_loss(
            regressing_logits,
            final_logits,
            target_ids,
            final_logit_ce_weight=0.0,
            depth_final_ce_weight=0.0,
            all_depth_ce_weight=0.0,
            progress_margin_weight=0.0,
            progress_margin=0.10,
            depth_trajectory_monotonic_weight=1.0,
            depth_trajectory_monotonic_margin=0.02,
        )

        self.assertLess(float(improving_loss), 1e-6)
        self.assertLess(float(improving_metrics["depth_trajectory_monotonic"]), 1e-6)
        self.assertGreater(float(regressing_loss), 0.1)
        self.assertLess(float(regressing_metrics["depth_trajectory_step_delta"]), 0.0)

    def test_terminal_depth_ce_loss_scores_only_finality_marked_depths(self):
        import torch

        module = _load_module()
        row = {
            "transition_finality_targets": {
                "1": 0,
                "2": 0,
                "3": 1,
            }
        }
        mask = module.terminal_depth_mask_from_row(row, num_depths=3, device="cpu")
        depth_logits = torch.zeros(1, 3, 1, 4)
        target_ids = torch.tensor([[2]])
        depth_logits[:, 0, 0, 1] = 8.0
        depth_logits[:, 1, 0, 1] = 8.0
        depth_logits[:, 2, 0, 2] = 8.0

        loss, metrics = module.terminal_depth_ce_loss(depth_logits, target_ids, mask)

        self.assertLess(float(loss), 0.01)
        self.assertEqual(float(metrics["terminal_depth_acc"]), 1.0)
        self.assertEqual(float(metrics["terminal_depth_count"]), 1.0)

    def test_terminal_depth_mask_falls_back_to_depth_targets_matching_answer(self):
        module = _load_module()
        row = {
            "answer_aliases": ["17"],
            "depth_targets": {
                "1": "10",
                "2": "20",
                "4": "17",
                "8": "17",
            },
        }

        mask = module.terminal_depth_mask_from_row(row, num_depths=4, device="cpu")

        self.assertEqual(mask.tolist(), [False, False, False, True])

    def test_terminal_depth_ce_loss_is_zero_without_terminal_depths(self):
        import torch

        module = _load_module()
        depth_logits = torch.zeros(1, 2, 1, 4)
        target_ids = torch.tensor([[2]])
        mask = torch.zeros(2, dtype=torch.bool)

        loss, metrics = module.terminal_depth_ce_loss(depth_logits, target_ids, mask)

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(metrics["terminal_depth_count"]), 0.0)

    def test_answer_state_loop_halt_ce_loss_selects_first_terminal_depth(self):
        import torch

        module = _load_module()
        halt_logits = torch.tensor([[0.0, -1.0, 3.0, 2.0]])
        mask = torch.tensor([False, False, True, True])

        loss, metrics = module.answer_state_loop_halt_ce_loss(halt_logits, mask)

        self.assertLess(float(loss), 0.5)
        self.assertEqual(float(metrics["answer_state_halt_acc"]), 1.0)
        self.assertEqual(float(metrics["answer_state_halt_count"]), 1.0)

    def test_schedule_cycles_each_row_through_every_depth(self):
        module = _load_module()

        schedule = [
            module.scheduled_row_and_core_steps(step, row_count=2, depth_steps=[1, 2, 4])
            for step in range(8)
        ]

        self.assertEqual(
            schedule,
            [
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 1),
                (1, 2),
                (1, 4),
                (0, 1),
                (0, 2),
            ],
        )

    def test_family_repeat_curriculum_oversamples_hard_families(self):
        module = _load_module()
        rows = [
            {"task_family": "arithmetic_chain"},
            {"task_family": "list_transform"},
            {"task_family": "boolean_logic"},
        ]

        repeats = module.parse_family_repeat_spec("list_transform=3,boolean_logic=2")
        indices = module.build_curriculum_indices(rows, repeats)

        self.assertEqual(repeats, {"list_transform": 3, "boolean_logic": 2})
        self.assertEqual(indices, [0, 1, 1, 1, 2, 2])

    def test_depth_choice_margin_loss_penalizes_rejected_first_token_at_all_depths(self):
        import torch

        module = _load_module()
        depth_logits = torch.zeros(1, 2, 1, 6)
        final_logits = torch.zeros(1, 1, 6)
        chosen = torch.tensor([3])
        rejected = torch.tensor([4])
        depth_logits[:, :, 0, 4] = 2.0
        final_logits[:, 0, 4] = 2.0

        loss, metrics = module.depth_choice_margin_loss(
            depth_logits,
            final_logits,
            chosen,
            rejected,
            margin=0.5,
            all_depth_weight=1.0,
            final_weight=1.0,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(metrics["choice_margin_all_depth"]), 0.0)
        self.assertGreater(float(metrics["choice_margin_final_path"]), 0.0)

        depth_logits[:, :, 0, 3] = 4.0
        final_logits[:, 0, 3] = 4.0
        loss, metrics = module.depth_choice_margin_loss(
            depth_logits,
            final_logits,
            chosen,
            rejected,
            margin=0.5,
            all_depth_weight=1.0,
            final_weight=1.0,
        )

        self.assertEqual(float(loss), 0.0)

    def test_choice_margin_rejected_texts_use_choices_when_rejected_missing(self):
        module = _load_module()

        row = {
            "answer": "300015",
            "chosen": "300015",
            "answer_aliases": ["300015", "300,015"],
            "choices": ["300015", "300024", "100004,100008,100012", "EMPTY"],
        }

        self.assertEqual(
            module.choice_margin_rejected_texts(row),
            ["300024", "100004,100008,100012", "EMPTY"],
        )

    def test_choice_margin_rejected_texts_exclude_current_staged_answer(self):
        module = _load_module()

        row = {
            "answer_aliases": ["217"],
            "choices": ["217", "220", "218", "216"],
        }

        self.assertEqual(
            module.choice_margin_rejected_texts(row, current_answer="220"),
            ["218", "216"],
        )

    def test_choice_margin_rejected_texts_prefer_explicit_rejected(self):
        module = _load_module()

        row = {
            "chosen": "300015",
            "choices": ["300015", "300024"],
            "rejected": "300024",
        }

        self.assertEqual(module.choice_margin_rejected_texts(row), ["300024"])

    def test_context_ablation_contrastive_loss_penalizes_context_off_better_than_on(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[2]])
        on_logits = torch.zeros(1, 2, 1, 5)
        off_logits = torch.zeros(1, 2, 1, 5)
        off_logits[:, -1, 0, 2] = 4.0

        loss, metrics = module.context_ablation_contrastive_loss(
            on_logits,
            off_logits,
            target_ids,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.5)
        self.assertLess(float(metrics["context_contrast_target_logp_delta"]), 0.0)

        on_logits[:, -1, 0, 2] = 8.0
        off_logits[:, -1, 0, 2] = 0.0
        loss, metrics = module.context_ablation_contrastive_loss(
            on_logits,
            off_logits,
            target_ids,
            margin=0.5,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertGreater(float(metrics["context_contrast_target_logp_delta"]), 0.0)

    def test_transition_state_ablation_contrastive_loss_renames_context_metrics(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[2]])
        on_logits = torch.zeros(1, 2, 1, 5)
        off_logits = torch.zeros(1, 2, 1, 5)
        off_logits[:, -1, 0, 2] = 4.0

        loss, metrics = module.transition_state_ablation_contrastive_loss(
            on_logits,
            off_logits,
            target_ids,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.5)
        self.assertIn("transition_state_contrast_target_logp_delta", metrics)
        self.assertNotIn("context_contrast_target_logp_delta", metrics)

    def test_transition_joint_answer_bridge_contrast_loss_renames_metrics(self):
        import torch

        module = _load_module()
        target_ids = torch.tensor([[2]])
        on_logits = torch.zeros(1, 2, 1, 5)
        off_logits = torch.zeros(1, 2, 1, 5)
        off_logits[:, -1, 0, 2] = 4.0

        loss, metrics = module.transition_joint_answer_bridge_contrastive_loss(
            on_logits,
            off_logits,
            target_ids,
            margin=0.5,
        )

        self.assertGreater(float(loss), 0.5)
        self.assertIn("transition_joint_answer_bridge_contrast_target_logp_delta", metrics)
        self.assertNotIn("transition_state_contrast_target_logp_delta", metrics)

    def test_runner_uses_prompt_only_depth_supervision_and_raw_gate(self):
        script = Path("scripts/197_run_pure_recursive_depth_supervised_train.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("scripts/196_train_pure_recursive_depth_supervised.py", text)
        self.assertIn("scripts/193_run_pure_recursive_reasoning_depth_gate.sh", text)
        self.assertIn("DEPTH_STEPS", text)
        self.assertIn("TARGET_MODE", text)
        self.assertIn("FINAL_LOGIT_CE_WEIGHT", text)
        self.assertIn("ALL_DEPTH_CE_WEIGHT", text)
        self.assertIn("--all-depth-ce-weight", text)
        self.assertIn("PROGRESS_MARGIN_WEIGHT", text)
        self.assertIn("FAMILY_REPEAT", text)
        self.assertIn("TRAIN_INCLUDE_FAMILIES", text)
        self.assertIn("HELDOUT_INCLUDE_FAMILIES", text)
        self.assertIn("HELDOUT_CASES_PER_FAMILY", text)
        self.assertIn("HELDOUT_START_INDEX", text)
        self.assertIn("--include-family", text)
        self.assertIn("CHOICE_MARGIN_WEIGHT", text)
        self.assertIn("CAUSAL_PREFIX_SUPERVISION", text)
        self.assertIn("CAUSAL_PREFIX_MAX_TARGET_TOKENS", text)
        self.assertIn("CAUSAL_PREFIX_LATER_TOKEN_WEIGHT", text)
        self.assertIn("TEACHER_CHECKPOINT", text)
        self.assertIn("TEACHER_FIRST_TOKEN_DEPTH_KL_WEIGHT", text)
        self.assertIn("CORE_WORLD_MODEL_WEIGHT", text)
        self.assertIn("--core-world-model-weight", text)
        self.assertIn("STAGED_INTERNAL_FIRST_TOKEN_CE_WEIGHT", text)
        self.assertIn("--staged-internal-first-token-ce-weight", text)
        self.assertIn("STAGED_INTERNAL_SEQUENCE_CE_WEIGHT", text)
        self.assertIn("--staged-internal-sequence-ce-weight", text)
        self.assertIn("STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS", text)
        self.assertIn("NOISE_WARMUP_STEPS", text)
        self.assertIn("--noise-warmup-steps", text)
        self.assertIn("NOISE_WARMUP_UNIFORM_WEIGHT", text)
        self.assertIn("--noise-warmup-uniform-weight", text)
        self.assertIn("TEMPORAL_SPATIAL_CONTEXT_CONTRAST_WEIGHT", text)
        self.assertIn("TRANSITION_STATE_CONTRAST_WEIGHT", text)
        self.assertIn("--transition-state-contrast-weight", text)
        self.assertIn("TRANSITION_STATE_CE_WEIGHT", text)
        self.assertIn("--transition-state-ce-weight", text)
        self.assertIn("TRANSITION_STATE_CODE_CE_WEIGHT", text)
        self.assertIn("--transition-state-code-ce-weight", text)
        self.assertIn("INCLUDE_TRANSITION_STATE_OFF", text)
        self.assertIn("EVAL_OUT", text)
        self.assertNotIn("scripts/95_eval_memory_retrieval.py", text)

    def test_train_loop_threads_temporal_spatial_context_to_student_and_teacher(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("temporal_spatial_context = _row_temporal_spatial_context", text)
        self.assertIn("temporal_spatial_context=temporal_spatial_context", text)
        self.assertIn("teacher_temporal_spatial_context", text)

    def test_train_loop_can_ablate_transition_state_path(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("transition_state_ablation_contrastive_loss", text)
        self.assertIn("transition_state_first_token_ce_loss", text)
        self.assertIn("transition_state_code_ce_loss", text)
        self.assertIn("staged_internal_sequence_ce_loss", text)
        self.assertIn("depth_choice_sequence_margin_loss", text)
        self.assertIn("--choice-margin-mode", text)
        self.assertIn("disable_transition_state=True", text)
        self.assertIn("transition_joint_answer_bridge_contrastive_loss", text)
        self.assertIn("disable_transition_state_joint_answer_bridge=True", text)

    def test_train_loop_exposes_direct_vocab_renderer_supervision(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )
        runner = Path("scripts/322_run_source_pointer_l4_lm_path_gate.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--core-role-value-vocab-renderer-ce-weight", text)
        self.assertIn(
            "core_role_value_vocab_renderer_sequence_supervision_loss",
            text,
        )
        self.assertIn("outputs[\"core_role_value_vocab_renderer_logits\"]", text)
        self.assertIn(
            "--core-role-value-vocab-renderer-primitive-contrast-weight",
            text,
        )
        self.assertIn(
            "core_role_value_vocab_renderer_primitive",
            text,
        )
        self.assertIn("--vocab-renderer-ce-weight", runner)
        self.assertIn("--core-role-value-vocab-renderer-ce-weight", runner)
        self.assertIn("--vocab-renderer-primitive-contrast-weight", runner)

    def test_train_loop_applies_answer_state_loop_logit_ce_to_all_prefixes(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--answer-state-loop-logit-ce-weight", text)
        self.assertIn("answer_state_loop_logit_ce_loss", text)
        self.assertIn('outputs["answer_state_loop_logits"]', text)
        loop_body_start = text.index(
            'answer_loop_logits = outputs["answer_state_loop_logits"]'
        )
        loop_start = max(0, loop_body_start - 300)
        loop_end = text.index(
            "float(args.answer_state_loop_future_token_ce_weight)",
            loop_body_start,
        )
        loop_block = text[loop_start:loop_end]
        self.assertNotIn("example_index == 0", loop_block)

    def test_train_script_exposes_core_source_position_binder_switch(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--core-source-position-binder", text)
        self.assertIn("--core-source-position-binder-gate-min", text)
        self.assertIn("--core-source-position-binder-state-gate-min", text)
        self.assertIn("--core-source-position-binder-state-st", text)
        self.assertIn("cfg.model.core_source_position_binder_enabled = True", text)
        self.assertIn(
            "cfg.model.core_source_position_binder_gate_min = float",
            text,
        )
        self.assertIn(
            "cfg.model.core_source_position_binder_state_gate_min = float",
            text,
        )
        self.assertIn(
            "cfg.model.core_source_position_binder_query_state_enabled = bool",
            text,
        )
        self.assertIn("cfg.model.core_source_value_binder_enabled = bool", text)

    def test_hard_family_overfit_runner_targets_list_and_arithmetic(self):
        text = Path("scripts/210_run_pure_recursive_hard_family_overfit8.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("arithmetic_chain,list_transform", text)
        self.assertIn("MAX_CASES=\"${MAX_CASES:-8}\"", text)
        self.assertIn("pure_recursive_hard_family_overfit8", text)
        self.assertIn("scripts/197_run_pure_recursive_depth_supervised_train.sh", text)

    def test_hard_family_generalization_runner_separates_train_and_heldout_ranges(self):
        text = Path(
            "scripts/211_run_pure_recursive_hard_family_generalization_s240.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("TRAIN_START_INDEX=\"${TRAIN_START_INDEX:-100}\"", text)
        self.assertIn("HELDOUT_START_INDEX=\"${HELDOUT_START_INDEX:-200}\"", text)
        self.assertIn("TRAIN_CASES_PER_FAMILY=\"${TRAIN_CASES_PER_FAMILY:-16}\"", text)
        self.assertIn("CHOICE_SCORE_NORMALIZATION=\"${CHOICE_SCORE_NORMALIZATION:-mean}\"", text)
        self.assertIn("pure_recursive_hard_family_generalization_s240", text)
        self.assertIn("scripts/197_run_pure_recursive_depth_supervised_train.sh", text)

    def test_semantic_state_sequence_margin_runner_uses_new_contract(self):
        text = Path(
            "scripts/212_run_pure_recursive_semantic_state_sequence_margin_s240.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("CHOICE_MARGIN_MODE=\"${CHOICE_MARGIN_MODE:-sequence}\"", text)
        self.assertIn("TRANSITION_STATE_CODE_CE_WEIGHT=\"${TRANSITION_STATE_CODE_CE_WEIGHT:-1.00}\"", text)
        self.assertIn("pure_recursive_semantic_state_sequence_margin_s240", text)
        self.assertIn("CHOICE_SCORE_NORMALIZATION=\"${CHOICE_SCORE_NORMALIZATION:-mean}\"", text)
        self.assertIn("scripts/197_run_pure_recursive_depth_supervised_train.sh", text)

    def test_value_state_sequence_margin_runner_uses_continuous_state_path(self):
        text = Path(
            "scripts/213_run_pure_recursive_value_state_sequence_margin_s240.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("qwen35_2b_4090_pure_recursive_transition_state_s080.yaml", text)
        self.assertIn("TRANSITION_STATE_CE_WEIGHT=\"${TRANSITION_STATE_CE_WEIGHT:-0.50}\"", text)
        self.assertIn("TRANSITION_STATE_CONTRAST_WEIGHT=\"${TRANSITION_STATE_CONTRAST_WEIGHT:-0.30}\"", text)
        self.assertIn("CHOICE_MARGIN_MODE=\"${CHOICE_MARGIN_MODE:-sequence}\"", text)
        self.assertIn("pure_recursive_value_state_sequence_margin_s240", text)
        self.assertIn("scripts/197_run_pure_recursive_depth_supervised_train.sh", text)

    def test_full_state_sequence_runner_uses_depth_sequence_state_supervision(self):
        text = Path(
            "scripts/214_run_pure_recursive_full_state_sequence_s240.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("STAGED_INTERNAL_SEQUENCE_CE_WEIGHT", text)
        self.assertIn("STAGED_INTERNAL_SEQUENCE_MAX_TARGET_TOKENS", text)
        self.assertIn("pure_recursive_full_state_sequence_s240", text)
        self.assertIn("CHOICE_MARGIN_MODE=\"${CHOICE_MARGIN_MODE:-sequence}\"", text)
        self.assertIn("scripts/197_run_pure_recursive_depth_supervised_train.sh", text)


if __name__ == "__main__":
    unittest.main()
