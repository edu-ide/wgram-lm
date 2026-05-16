from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch


def load_module():
    path = Path("scripts/394_train_qwen35_integrated_language_knowledge_healing.py")
    spec = importlib.util.spec_from_file_location(
        "qwen35_integrated_language_knowledge_healing",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Qwen35IntegratedLanguageKnowledgeHealingTests(unittest.TestCase):
    def test_text_loader_filters_think_blocks_and_short_rows(self):
        module = load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "short"}),
                        json.dumps({"text": "<think>hidden</think> visible text that is long enough"}),
                        json.dumps({"text": "User: Explain evidence.\nAssistant: Evidence should be checked against sources.", "source": "unit"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = module.load_text_rows([str(path)], max_rows=0, min_chars=20, seed=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "unit")

    def test_split_train_eval_keeps_nonempty_parts(self):
        module = load_module()
        rows = [{"text": f"row {index}"} for index in range(4)]

        train, eval_rows = module.split_train_eval(rows, eval_rows=8)

        self.assertTrue(train)
        self.assertTrue(eval_rows)

    def test_lm_loss_masks_padding_positions(self):
        module = load_module()
        logits = torch.zeros((1, 4, 5), dtype=torch.float32)
        input_ids = torch.tensor([[1, 2, 3, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        loss = module._lm_loss_from_logits(logits, input_ids, attention_mask)

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), torch.log(torch.tensor(5.0)).item(), places=5)

    def test_last_nonpad_logits_ignore_right_padding(self):
        module = load_module()
        logits = torch.tensor(
            [
                [[1.0, 0.0], [9.0, 0.0], [-5.0, 7.0]],
                [[0.0, 3.0], [0.0, 4.0], [0.0, 5.0]],
            ]
        )
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        selected = module.last_nonpad_logits(logits, attention_mask)

        self.assertTrue(torch.equal(selected, torch.tensor([[9.0, 0.0], [0.0, 5.0]])))

    def test_sequence_kl_zero_for_identical_logits(self):
        module = load_module()
        logits = torch.randn(1, 4, 7)
        attention_mask = torch.ones((1, 4), dtype=torch.long)

        loss = module.sequence_kl_loss(logits, logits.clone(), attention_mask)

        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_parser_and_runner_defaults_are_qwen_integrated_native(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(["--init-checkpoint", "last.pt"])

        self.assertEqual(args.core_impl, "qwen_layer_wrapped")
        self.assertIsNone(module.parse_int_list(""))
        self.assertTrue(args.text_jsonl == [])
        self.assertEqual(args.mcq_validation_jsonl, "")
        self.assertEqual(args.unfreeze_qwen_layer_indices, "23")
        self.assertEqual(args.core_insertion_mode, "mid_layer_suffix")
        self.assertEqual(args.core_insert_after_layer, 11)
        self.assertEqual(args.core_residual_gate_mode, "constant")
        self.assertEqual(args.core_residual_gate_dim, 128)
        self.assertEqual(args.core_residual_gate_init, -2.0)
        self.assertEqual(args.residual_gate_lr_multiplier, 1.0)
        self.assertEqual(args.n_core_layers, 1)
        self.assertEqual(args.h_cycles, 3)
        self.assertEqual(args.l_cycles, 6)
        self.assertEqual(args.outer_steps, 3)
        self.assertTrue(args.core_convergence_halt_enabled)
        self.assertEqual(args.core_convergence_halt_threshold, 0.2)
        self.assertEqual(args.core_convergence_halt_min_outer, 1)
        self.assertTrue(args.core_step_conditioning_enabled)
        self.assertEqual(args.core_step_conditioning_max_steps, 64)
        self.assertEqual(args.max_core_ce_regression, 0.01)
        self.assertEqual(args.min_base_wrong_core_correct, 0)
        self.assertEqual(args.max_base_correct_core_wrong, 1_000_000)
        self.assertEqual(args.language_anchor_weight, 0.0)
        self.assertEqual(args.language_anchor_batch_size, 4)
        self.assertEqual(args.eval_every_steps, 0)
        self.assertEqual(args.mcq_ce_focus, "all")
        self.assertEqual(args.mcq_loss_space, "full_vocab")
        self.assertEqual(args.mcq_margin_weight, 0.0)
        self.assertEqual(args.base_wrong_max_top_margin, -1.0)
        self.assertEqual(args.mcq_non_selected_option_kl_weight, 0.0)
        self.assertEqual(args.residual_gate_selected_open_weight, 0.0)
        self.assertEqual(args.residual_gate_non_selected_closed_weight, 0.0)
        self.assertEqual(args.base_wrong_mcq_retries, 1)
        self.assertEqual(args.base_correct_option_kl_weight, 0.0)
        self.assertEqual(args.base_correct_option_kl_focus, "base_correct")
        self.assertEqual(args.base_correct_kl_extra_batch_size, 0)
        self.assertFalse(args.train_only_core_delta_adapter)
        self.assertFalse(args.clone_qwen_core_layers)
        self.assertFalse(args.skip_save_checkpoint)
        self.assertFalse(args.restore_best_checkpoint)

        runner = Path("scripts/394_run_qwen35_integrated_language_knowledge_healing.sh")
        text = runner.read_text(encoding="utf-8")
        self.assertIn("--mandatory-core", text)
        self.assertIn("qtrm_native_external_bilingual_9000", text)
        self.assertIn("external_mcq_train_pool_2000", text)
        self.assertIn("external_mcq_validation_pool", text)
        self.assertIn("BASE_KL_WEIGHT", text)
        self.assertIn("LANGUAGE_ANCHOR_WEIGHT", text)
        self.assertIn("--language-anchor-weight", text)
        self.assertIn("LANGUAGE_ANCHOR_BATCH_SIZE", text)
        self.assertIn("--language-anchor-batch-size", text)
        self.assertIn("MCQ_CE_FOCUS", text)
        self.assertIn("MCQ_LOSS_SPACE", text)
        self.assertIn("--mcq-loss-space", text)
        self.assertIn("CORE_INSERTION_MODE", text)
        self.assertIn('CORE_INSERTION_MODE="${CORE_INSERTION_MODE:-mid_layer_suffix}"', text)
        self.assertIn("CORE_INSERT_AFTER_LAYER", text)
        self.assertIn('CORE_INSERT_AFTER_LAYER="${CORE_INSERT_AFTER_LAYER:-11}"', text)
        self.assertIn("--core-insertion-mode", text)
        self.assertIn("--core-insert-after-layer", text)
        self.assertIn("TRAIN_ONLY_CORE_DELTA_ADAPTER", text)
        self.assertIn("--train-only-core-delta-adapter", text)
        self.assertIn("CORE_RESIDUAL_GATE_MODE", text)
        self.assertIn("--core-residual-gate-mode", text)
        self.assertIn("CORE_RESIDUAL_GATE_DIM", text)
        self.assertIn("--core-residual-gate-dim", text)
        self.assertIn("CORE_RESIDUAL_GATE_INIT", text)
        self.assertIn("--core-residual-gate-init", text)
        self.assertIn("RESIDUAL_GATE_LR_MULTIPLIER", text)
        self.assertIn("--residual-gate-lr-multiplier", text)
        self.assertIn("MCQ_MARGIN_WEIGHT", text)
        self.assertIn("BASE_WRONG_MAX_TOP_MARGIN", text)
        self.assertIn("--base-wrong-max-top-margin", text)
        self.assertIn("MCQ_NON_SELECTED_OPTION_KL_WEIGHT", text)
        self.assertIn("--mcq-non-selected-option-kl-weight", text)
        self.assertIn("RESIDUAL_GATE_SELECTED_OPEN_WEIGHT", text)
        self.assertIn("--residual-gate-selected-open-weight", text)
        self.assertIn("RESIDUAL_GATE_NON_SELECTED_CLOSED_WEIGHT", text)
        self.assertIn("--residual-gate-non-selected-closed-weight", text)
        self.assertIn("BASE_WRONG_MCQ_RETRIES", text)
        self.assertIn("BASE_CORRECT_OPTION_KL_WEIGHT", text)
        self.assertIn("BASE_CORRECT_KL_EXTRA_BATCH_SIZE", text)
        self.assertIn("MCQ_BATCH_SIZE", text)
        self.assertIn('MCQ_BATCH_SIZE="${MCQ_BATCH_SIZE:-4}"', text)
        self.assertIn("--mcq-batch-size", text)
        self.assertIn("EVAL_BATCH_SIZE", text)
        self.assertIn("--eval-batch-size", text)
        self.assertIn("CLONE_QWEN_CORE_LAYERS", text)
        self.assertIn("H_CYCLES", text)
        self.assertIn("L_CYCLES", text)
        self.assertIn("OUTER_STEPS", text)
        self.assertIn('H_CYCLES="${H_CYCLES:-3}"', text)
        self.assertIn('L_CYCLES="${L_CYCLES:-6}"', text)
        self.assertIn('OUTER_STEPS="${OUTER_STEPS:-3}"', text)
        self.assertIn("CORE_CONVERGENCE_HALT_ENABLED", text)
        self.assertIn("--core-convergence-halt-enabled", text)
        self.assertIn("--no-core-convergence-halt", text)
        self.assertIn("CORE_STEP_CONDITIONING_ENABLED", text)
        self.assertIn("--core-step-conditioning-enabled", text)
        self.assertIn("--no-core-step-conditioning", text)
        self.assertIn("SKIP_SAVE_CHECKPOINT", text)
        self.assertIn('UNFREEZE_QWEN_LAYER_INDICES="${UNFREEZE_QWEN_LAYER_INDICES-23}"', text)
        self.assertIn("MAX_CORE_CE_REGRESSION", text)
        self.assertIn("MIN_BASE_WRONG_CORE_CORRECT", text)
        self.assertIn("--min-base-wrong-core-correct", text)
        self.assertIn("MAX_BASE_CORRECT_CORE_WRONG", text)
        self.assertIn("--max-base-correct-core-wrong", text)
        self.assertIn("--restore-best-checkpoint", text)

    def test_category_gain_summary_tracks_regressions(self):
        module = load_module()
        evaluation = {
            "by_category": {
                "science": {"total": 10, "base_hits": 5, "core_hits": 7},
                "commonsense": {"total": 10, "base_hits": 6, "core_hits": 4},
            }
        }

        summary = module.category_gain_summary(evaluation, min_cases=2)

        self.assertEqual(summary["min_hit_delta"], -2)
        self.assertAlmostEqual(summary["min_accuracy_delta"], -0.2)
        self.assertEqual(summary["negative_hit_delta_sum"], 2)

    def test_validation_selection_score_penalizes_text_and_category_regression(self):
        module = load_module()
        args = SimpleNamespace(
            max_core_ce_regression=0.01,
            text_ce_regression_penalty=2.0,
            category_guard_min_cases=1,
            category_regression_penalty=1.0,
        )
        before_text = {"core_ce": 1.0, "finite_logits": True}
        current_text = {"core_ce": 1.03, "finite_logits": True}
        current_mcq = {
            "gain": 0.05,
            "core_accuracy": 0.50,
            "finite_logits": True,
            "by_category": {
                "ok": {"total": 10, "base_hits": 3, "core_hits": 5},
                "bad": {"total": 10, "base_hits": 6, "core_hits": 5},
            },
        }

        score = module.validation_selection_score(
            before_text=before_text,
            current_text=current_text,
            current_mcq=current_mcq,
            args=args,
        )

        self.assertAlmostEqual(score, -0.04)

    def test_balanced_mcq_chunk_samples_across_category_groups(self):
        module = load_module()
        rows = [
            {"category": "a", "value": 1},
            {"category": "b", "value": 2},
        ]

        chunk = module.sample_mcq_chunk(
            __import__("random").Random(1),
            rows,
            batch_size=4,
            balanced_category_sampling=True,
        )

        self.assertEqual(len(chunk), 2)
        self.assertTrue(all("category" in row for row in chunk))

    def test_score_mcq_reports_base_core_flip_counts(self):
        module = load_module()

        class TinyTokenizer:
            def __call__(self, texts, **_kwargs):
                ids = [[int(str(text).replace("p", ""))] for text in texts]
                return {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.ones((len(ids), 1), dtype=torch.long),
                }

            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], " A": [0], " B": [1], "\nA": [0], "\nB": [1]}.get(text, [2])

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask=None, force_core_off=False):
                preds = {
                    0: "A",
                    1: "A" if force_core_off else "B",
                    2: "A" if force_core_off else "B",
                    3: "A",
                }
                logits = torch.full((input_ids.shape[0], 1, 3), -10.0)
                for row_index, case_id in enumerate(input_ids[:, 0].tolist()):
                    logits[row_index, 0, 0 if preds[int(case_id)] == "A" else 1] = 10.0
                if force_core_off:
                    return SimpleNamespace(logits=logits)
                return SimpleNamespace(
                    logits=logits,
                    qtrm_core_outer_iterations=torch.full((input_ids.shape[0],), 3),
                    qtrm_core_converged=torch.tensor([1, 0, 1, 0][: input_ids.shape[0]], dtype=torch.bool),
                    qtrm_core_convergence_delta=torch.full((input_ids.shape[0], 3), 0.125),
                )

        rows = [
            {"qtrm_prompt": "p0", "answer": "A", "options": ["a", "b"], "category": "x"},
            {"qtrm_prompt": "p1", "answer": "A", "options": ["a", "b"], "category": "x"},
            {"qtrm_prompt": "p2", "answer": "B", "options": ["a", "b"], "category": "y"},
            {"qtrm_prompt": "p3", "answer": "B", "options": ["a", "b"], "category": "y"},
        ]
        args = SimpleNamespace(eval_batch_size=4, max_seq_len=4)

        result = module.score_mcq(FakeModel(), TinyTokenizer(), rows, args)

        self.assertEqual(result["flip_counts"]["both_correct"], 1)
        self.assertEqual(result["flip_counts"]["base_correct_core_wrong"], 1)
        self.assertEqual(result["flip_counts"]["base_wrong_core_correct"], 1)
        self.assertEqual(result["flip_counts"]["both_wrong"], 1)
        self.assertEqual(result["by_category"]["x"]["base_correct_core_wrong"], 1)
        self.assertEqual(result["by_category"]["y"]["base_wrong_core_correct"], 1)
        self.assertEqual(result["mean_core_outer_iterations"], 3.0)
        self.assertEqual(result["core_converged_fraction"], 0.5)
        self.assertAlmostEqual(result["mean_core_convergence_delta"], 0.125)

    def test_base_wrong_indices_selects_only_base_mistakes(self):
        module = load_module()

        class TinyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], " A": [0], " B": [1], "\nA": [0], "\nB": [1]}.get(text, [2])

        rows = [
            {"answer": "A", "options": ["a", "b"]},
            {"answer": "B", "options": ["a", "b"]},
        ]
        logits = torch.tensor(
            [
                [5.0, 0.0, -1.0],
                [5.0, 0.0, -1.0],
            ]
        )

        selected = module.base_wrong_indices(TinyTokenizer(), logits, rows)

        self.assertEqual(selected, [1])

    def test_base_wrong_indices_can_filter_confident_base_mistakes(self):
        module = load_module()

        class TinyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], " A": [0], " B": [1], "\nA": [0], "\nB": [1]}.get(text, [2])

        rows = [
            {"answer": "B", "options": ["a", "b"]},
            {"answer": "B", "options": ["a", "b"]},
        ]
        logits = torch.tensor(
            [
                [5.0, 4.8, -1.0],
                [5.0, 0.0, -1.0],
            ]
        )

        selected = module.base_wrong_indices(
            TinyTokenizer(),
            logits,
            rows,
            max_top_margin=0.5,
        )

        self.assertEqual(selected, [0])

    def test_option_distribution_kl_for_indices_preserves_only_requested_rows(self):
        module = load_module()

        class TinyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], " A": [0], " B": [1], "\nA": [0], "\nB": [1]}.get(text, [2])

        rows = [
            {"answer": "A", "options": ["a", "b"]},
            {"answer": "B", "options": ["a", "b"]},
        ]
        base_logits = torch.tensor(
            [
                [5.0, 0.0, -1.0],
                [5.0, 0.0, -1.0],
            ]
        )
        shifted_core_logits = torch.tensor(
            [
                [0.0, 5.0, -1.0],
                [5.0, 0.0, -1.0],
            ]
        )

        first_loss = module.option_distribution_kl_loss_for_indices(
            TinyTokenizer(),
            shifted_core_logits,
            base_logits,
            rows,
            [0],
        )
        second_loss = module.option_distribution_kl_loss_for_indices(
            TinyTokenizer(),
            shifted_core_logits,
            base_logits,
            rows,
            [1],
        )

        self.assertGreater(float(first_loss), 1.0)
        self.assertAlmostEqual(float(second_loss), 0.0, places=5)

    def test_option_choice_ce_normalizes_over_choices_only(self):
        module = load_module()

        class TinyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], " A": [0], " B": [1], "\nA": [0], "\nB": [1]}.get(text, [2])

        rows = [
            {"answer": "A", "options": ["a", "b"]},
            {"answer": "B", "options": ["a", "b"]},
        ]
        logits = torch.tensor(
            [
                [4.0, 0.0, 20.0],
                [0.0, 4.0, 20.0],
            ]
        )

        loss = module.option_choice_ce_loss(TinyTokenizer(), logits, rows)

        self.assertLess(float(loss), 0.05)

    def test_residual_gate_target_loss_uses_last_nonpad_gate(self):
        module = load_module()
        gate = torch.tensor([[[0.1], [0.8], [0.2]], [[0.9], [0.2], [0.1]]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        reference = torch.zeros((2, 3))

        open_loss = module.residual_gate_target_loss(
            gate,
            attention_mask,
            [0],
            target=1.0,
            reference=reference,
        )
        closed_loss = module.residual_gate_target_loss(
            gate,
            attention_mask,
            [1],
            target=0.0,
            reference=reference,
        )

        self.assertLess(float(open_loss), 0.3)
        self.assertLess(float(closed_loss), 0.2)

    def test_option_distribution_kl_focuses_base_correct_rows(self):
        module = load_module()

        class TinyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"A": [0], "B": [1], " A": [0], " B": [1], "\nA": [0], "\nB": [1]}.get(text, [2])

        rows = [
            {"answer": "A", "options": ["a", "b"]},
            {"answer": "B", "options": ["a", "b"]},
        ]
        base_logits = torch.tensor(
            [
                [5.0, 0.0, -1.0],
                [5.0, 0.0, -1.0],
            ]
        )
        same_loss = module.option_distribution_kl_loss(
            TinyTokenizer(),
            base_logits.clone(),
            base_logits,
            rows,
            focus="base_correct",
        )
        shifted_core_logits = torch.tensor(
            [
                [0.0, 5.0, -1.0],
                [5.0, 0.0, -1.0],
            ]
        )
        shifted_loss = module.option_distribution_kl_loss(
            TinyTokenizer(),
            shifted_core_logits,
            base_logits,
            rows,
            focus="base_correct",
        )

        self.assertAlmostEqual(float(same_loss), 0.0, places=5)
        self.assertGreater(float(shifted_loss), 1.0)


if __name__ == "__main__":
    unittest.main()
