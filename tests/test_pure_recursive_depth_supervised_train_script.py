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
                "--depth-final-ce-weight",
                "0.0",
                "--causal-prefix-max-target-tokens",
                "4",
                "--causal-prefix-later-token-weight",
                "0.1",
                "--teacher-checkpoint",
                "teacher.pt",
                "--teacher-first-token-depth-kl-weight",
                "0.5",
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
                "--transition-state-code-ce-weight",
                "0.8",
                "--transition-state-finality-ce-weight",
                "0.65",
                "--primitive-transition-operation-ce-weight",
                "0.55",
                "--choice-margin-mode",
                "sequence",
            ]
        )

        self.assertEqual(args.causal_prefix_max_target_tokens, 4)
        self.assertEqual(args.depth_final_ce_weight, 0.0)
        self.assertEqual(args.causal_prefix_later_token_weight, 0.1)
        self.assertEqual(args.teacher_checkpoint, "teacher.pt")
        self.assertEqual(args.teacher_first_token_depth_kl_weight, 0.5)
        self.assertEqual(args.teacher_depth_kl_temperature, 2.0)
        self.assertEqual(args.core_world_model_weight, 0.02)
        self.assertEqual(args.staged_internal_first_token_ce_weight, 0.4)
        self.assertEqual(args.staged_internal_sequence_ce_weight, 0.45)
        self.assertEqual(args.staged_internal_sequence_max_target_tokens, 5)
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
        self.assertEqual(args.transition_state_code_ce_weight, 0.8)
        self.assertEqual(args.transition_state_finality_ce_weight, 0.65)
        self.assertEqual(args.primitive_transition_operation_ce_weight, 0.55)
        self.assertEqual(args.choice_margin_mode, "sequence")

    def test_primitive_transition_operation_targets_follow_solver_trace_order(self):
        module = _load_module()

        row = {
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

        self.assertEqual(targets.tolist(), [[0, 1, 2, 3, -100]])

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
