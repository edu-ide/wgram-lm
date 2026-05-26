import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/538_eval_prefixlm_token_verifier_selection.py")
    spec = importlib.util.spec_from_file_location("prefixlm_token_verifier_selection", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrefixLMTokenVerifierSelectionTests(unittest.TestCase):
    def test_summarize_reports_raw_oracle_and_verifier_accuracy(self):
        module = load_module()
        target_ids = torch.tensor([4, 7, 9], dtype=torch.long)
        topk_ids = torch.tensor(
            [
                [2, 4, 1],
                [7, 3, 5],
                [8, 6, 9],
            ],
            dtype=torch.long,
        )
        verifier_scores = torch.tensor(
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.1, 0.0],
                [0.1, 0.2, 0.7],
            ],
            dtype=torch.float32,
        )

        report = module.summarize_candidate_selection(
            target_ids,
            topk_ids,
            verifier_scores=verifier_scores,
        )

        self.assertEqual(report["targets"], 3)
        self.assertAlmostEqual(report["raw_lm_top1_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(report["oracle_topk_accuracy"], 1.0)
        self.assertAlmostEqual(report["verifier_selected_accuracy"], 1.0)
        self.assertAlmostEqual(report["verifier_gain"], 2.0 / 3.0)

    def test_merge_metric_sums_weights_by_targets(self):
        module = load_module()
        merged = module.merge_metric_sums(
            [
                {
                    "targets": 2,
                    "raw_lm_top1_accuracy": 0.5,
                    "oracle_topk_accuracy": 1.0,
                    "verifier_selected_accuracy": 1.0,
                    "verifier_gain": 0.5,
                },
                {
                    "targets": 6,
                    "raw_lm_top1_accuracy": 0.0,
                    "oracle_topk_accuracy": 0.5,
                    "verifier_selected_accuracy": 0.5,
                    "verifier_gain": 0.5,
                },
            ]
        )

        self.assertEqual(merged["targets"], 8)
        self.assertAlmostEqual(merged["raw_lm_top1_accuracy"], 0.125)
        self.assertAlmostEqual(merged["oracle_topk_accuracy"], 0.625)
        self.assertAlmostEqual(merged["verifier_selected_accuracy"], 0.625)
        self.assertAlmostEqual(merged["verifier_gain"], 0.5)

    def test_no_verifier_scores_keeps_selection_unproven(self):
        module = load_module()
        target_ids = torch.tensor([1, 2], dtype=torch.long)
        topk_ids = torch.tensor([[1, 3], [4, 2]], dtype=torch.long)

        report = module.summarize_candidate_selection(target_ids, topk_ids)

        self.assertIsNone(report["verifier_selected_accuracy"])
        self.assertIsNone(report["verifier_gain"])


if __name__ == "__main__":
    unittest.main()
