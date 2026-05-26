from __future__ import annotations

import importlib.util
import argparse
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "570_train_solution_aligned_answer_attractor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("solution_aligned_answer_attractor_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SolutionAlignedAnswerAttractorTrainTests(unittest.TestCase):
    def test_contrastive_terms_penalize_bad_deeper_margin(self) -> None:
        module = load_module()
        shallow = torch.tensor(0.2)
        deep = torch.tensor(-0.1)

        rank_loss, monotonic_loss, final_margin = module.contrastive_terms_from_margins(
            [shallow, deep],
            target_margin=0.1,
            monotonic_gain=0.02,
        )

        self.assertGreater(float(rank_loss.item()), 0.0)
        self.assertGreater(float(monotonic_loss.item()), 0.0)
        self.assertLess(float(final_margin.item()), float(shallow.item()))

    def test_contrastive_terms_accept_positive_monotonic_margin(self) -> None:
        module = load_module()
        shallow = torch.tensor(0.15)
        deep = torch.tensor(0.35)

        rank_loss, monotonic_loss, final_margin = module.contrastive_terms_from_margins(
            [shallow, deep],
            target_margin=0.1,
            monotonic_gain=0.02,
        )

        self.assertLess(float(rank_loss.item()), 0.7)
        self.assertLess(float(monotonic_loss.item()), 0.7)
        self.assertGreater(float(final_margin.item()), float(shallow.item()))

    def test_build_checkpoint_args_preserves_original_contract(self) -> None:
        module = load_module()

        class Args:
            checkpoint = "in.pt"
            probe_jsonl = "probe.jsonl"
            depths = [2, 4, 8]
            target_margin = 0.1
            monotonic_gain = 0.02
            intelligence_nll_weight = 0.03
            language_preserve_weight = 0.2
            language_sampled_data = "heldout/sampled"
            template_consistency_weight = 0.4
            template_consistency_depth = 16
            steps = 12
            lr = 1e-5

        original = argparse.Namespace(d_model=384, patch_size=4)
        payload = module.build_checkpoint_args(original, Args())

        self.assertEqual(payload["d_model"], 384)
        self.assertEqual(payload["patch_size"], 4)
        self.assertTrue(payload["stage101_solution_aligned_answer_attractor"])
        self.assertEqual(payload["stage101_depths"], [2, 4, 8])
        self.assertEqual(payload["stage101_language_preserve_weight"], 0.2)
        self.assertEqual(payload["stage101_template_consistency_weight"], 0.4)
        self.assertEqual(payload["stage101_template_consistency_depth"], 16)

    def test_source_template_grouping_uses_semantic_prefix_not_replay(self) -> None:
        module = load_module()
        rows = [
            {
                "id": "stage101g_source_para_train_00_context_first_replay00",
                "source_template": "context_first",
                "plain_language_axis": "same fact",
            },
            {
                "id": "stage101g_source_para_train_00_answer_from_note_replay01",
                "source_template": "answer_from_note",
                "plain_language_axis": "same fact",
            },
            {
                "id": "stage101g_source_para_train_01_context_first_replay00",
                "source_template": "context_first",
                "plain_language_axis": "different fact",
            },
        ]

        groups = module.build_source_template_groups(rows)

        self.assertIn("stage101g_source_para_train_00", groups)
        self.assertEqual(len(groups["stage101g_source_para_train_00"]), 2)
        self.assertNotIn("stage101g_source_para_train_01", groups)

    def test_template_consistency_loss_penalizes_opposite_template_margins(self) -> None:
        module = load_module()

        loss, rank_loss, variance_loss = module.template_consistency_terms_from_margins(
            [torch.tensor(0.4), torch.tensor(-0.4)],
            target_margin=0.1,
        )

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(rank_loss.item()), 0.0)
        self.assertGreater(float(variance_loss.item()), 0.0)

    def test_masked_teacher_kl_ignores_prompt_tokens(self) -> None:
        module = load_module()
        student_logits = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            ]
        )
        teacher_logits = student_logits.clone()
        labels = torch.tensor([[-100, 1, 2]])

        loss, metrics = module.masked_teacher_kl_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=1.0,
            max_targets=0,
        )

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertEqual(metrics["language_preserve_targets"], 2)


if __name__ == "__main__":
    unittest.main()
