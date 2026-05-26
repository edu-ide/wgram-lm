from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "599_train_stage101w9_paired_feature_difference_reader.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w9_paired_feature_difference_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W9PairedFeatureDifferenceTrainTests(unittest.TestCase):
    def test_world_target_row_selects_world_targets(self) -> None:
        module = load_module()
        pair = {
            "id": "pair0",
            "world_a_targets": {"answer_permission": "yes"},
            "world_b_targets": {"answer_permission": "no"},
        }

        self.assertEqual({"feature_targets": {"answer_permission": "yes"}}, module.world_target_row(pair, "A"))
        self.assertEqual({"feature_targets": {"answer_permission": "no"}}, module.world_target_row(pair, "B"))

    def test_pairwise_feature_difference_loss_rewards_correct_direction(self) -> None:
        module = load_module()
        reader = module.LatentFeatureReader(d_model=2)
        with torch.no_grad():
            for head in reader.heads.values():
                head.weight.zero_()
                head.bias.zero_()
            reader.heads["detail_sufficiency"].weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        pair = {
            "id": "pair0",
            "pair_feature": "detail_sufficiency",
            "positive_world": "A",
            "world_a_targets": {
                "source_reliability": "trusted",
                "evidence_relevance": "relevant",
                "detail_sufficiency": "enough",
                "conflict_status": "clear",
                "answer_permission": "yes",
            },
            "world_b_targets": {
                "source_reliability": "trusted",
                "evidence_relevance": "relevant",
                "detail_sufficiency": "missing",
                "conflict_status": "clear",
                "answer_permission": "no",
            },
        }

        loss, metrics = module.pairwise_feature_difference_loss(
            reader,
            hidden_a=torch.tensor([[2.0, 0.0]]),
            hidden_b=torch.tensor([[-2.0, 0.0]]),
            pair=pair,
            device=torch.device("cpu"),
            target_margin=1.0,
            world_ce_weight=0.0,
        )

        self.assertLess(float(loss.item()), 0.05)
        self.assertTrue(metrics["pair_correct"])
        self.assertGreater(metrics["positive_margin"], 1.0)
        self.assertGreater(metrics["negative_margin"], 1.0)

    def test_build_pair_report_requires_pair_and_world_correctness(self) -> None:
        module = load_module()
        rows = [
            {
                "pair_correct": True,
                "positive_margin": 0.3,
                "negative_margin": 0.4,
                "world_a_all_feature_correct": True,
                "world_b_all_feature_correct": True,
                "pair_feature": "detail_sufficiency",
            }
        ]

        report = module.build_pair_report(rows, split="heldout", depth=16)

        self.assertTrue(report["accepted"])
        self.assertEqual(1.0, report["pair_accuracy"])
        self.assertEqual(1.0, report["both_world_feature_accuracy"])
        self.assertEqual(0.3, report["min_pair_margin"])


if __name__ == "__main__":
    unittest.main()
