from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "601_train_stage101x_counterfactual_answer_attractor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101x_counterfactual_answer_attractor_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101XCounterfactualAnswerAttractorTrainTests(unittest.TestCase):
    def test_row_to_choice_uses_same_lm_answer_contract(self) -> None:
        module = load_module()
        row = {
            "original_prompt": "Claim: x\nReal world: trusted says true.\nA:",
            "counterfactual_prompt": "Claim: x\nImagined change: rumor says true.\nA:",
            "original_answer": " yes",
            "counterfactual_answer": " no",
            "original_negative_answers": [" no"],
            "counterfactual_negative_answers": [" yes"],
        }

        original = module.row_to_choice(row, "original")
        counterfactual = module.row_to_choice(row, "counterfactual")

        self.assertEqual(row["original_prompt"], original["prompt"])
        self.assertEqual(" yes", original["intelligence_answer"])
        self.assertEqual([" no"], original["negative_answers"])
        self.assertEqual(row["counterfactual_prompt"], counterfactual["prompt"])
        self.assertEqual(" no", counterfactual["intelligence_answer"])
        self.assertEqual([" yes"], counterfactual["negative_answers"])

    def test_counterfactual_gap_direction_flips_with_original_answer(self) -> None:
        module = load_module()
        original_yes_minus_no = torch.tensor(2.5)
        counterfactual_yes_minus_no = torch.tensor(-1.0)

        yes_gap = module.counterfactual_gap(
            original_yes_minus_no,
            counterfactual_yes_minus_no,
            original_answer=" yes",
        )
        no_gap = module.counterfactual_gap(
            -original_yes_minus_no,
            -counterfactual_yes_minus_no,
            original_answer=" no",
        )

        self.assertAlmostEqual(3.5, float(yes_gap.item()), places=5)
        self.assertAlmostEqual(3.5, float(no_gap.item()), places=5)

    def test_build_pair_report_requires_both_worlds_and_positive_gap(self) -> None:
        module = load_module()
        rows = [
            {
                "pair_feature": "detail_sufficiency",
                "original_correct": True,
                "counterfactual_correct": True,
                "counterfactual_gap": 0.5,
                "original_margin": 0.4,
                "counterfactual_margin": 0.3,
            },
            {
                "pair_feature": "conflict_status",
                "original_correct": True,
                "counterfactual_correct": False,
                "counterfactual_gap": -0.2,
                "original_margin": 0.2,
                "counterfactual_margin": -0.1,
            },
        ]

        report = module.build_pair_report(rows, split="heldout", depth=16)

        self.assertFalse(report["accepted"])
        self.assertEqual(0.5, report["pair_accuracy"])
        self.assertEqual(-0.2, report["min_counterfactual_gap"])

    def test_build_checkpoint_args_preserves_base_model_shape_args(self) -> None:
        module = load_module()
        ckpt_args = Namespace(patch_size=2, dynamic_soft_patch_size=2, d_model=384, seq_len=128)
        run_args = Namespace(
            checkpoint="in.pt",
            train_jsonl="train.jsonl",
            eval_jsonl="eval.jsonl",
            depths=[2, 4],
            target_margin=0.25,
            target_gap=0.5,
            gap_weight=1.0,
            target_nll_weight=0.01,
            steps=10,
            lr=2e-5,
        )

        payload = module.build_checkpoint_args(ckpt_args, run_args)

        self.assertEqual(2, payload["patch_size"])
        self.assertEqual(2, payload["dynamic_soft_patch_size"])
        self.assertEqual(384, payload["d_model"])
        self.assertTrue(payload["stage101x_counterfactual_answer_attractor"])

    def test_batch_rows_for_step_wraps_and_preserves_order(self) -> None:
        module = load_module()
        rows = [{"id": str(index)} for index in range(4)]

        batch = module.batch_rows_for_step(rows, step=3, batch_size=3)

        self.assertEqual(["2", "3", "0"], [row["id"] for row in batch])


if __name__ == "__main__":
    unittest.main()
