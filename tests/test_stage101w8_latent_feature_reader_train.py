from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "597_train_stage101w8_latent_feature_reader.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w8_latent_feature_reader_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeHiddenModel(torch.nn.Module):
    def forward_logits_and_decoder_hidden(self, input_ids, attention_mask, *, think_steps: int):
        hidden = torch.nn.functional.one_hot(input_ids.clamp(min=0, max=7), num_classes=8).float()
        logits = torch.zeros((*input_ids.shape, 16), dtype=hidden.dtype, device=input_ids.device)
        return logits, hidden


class Stage101W8LatentFeatureReaderTrainTests(unittest.TestCase):
    def test_feature_target_indices_follow_declared_choice_order(self) -> None:
        module = load_module()
        row = {
            "feature_targets": {
                "source_reliability": "untrusted",
                "evidence_relevance": "irrelevant",
                "detail_sufficiency": "missing",
                "conflict_status": "conflict",
                "answer_permission": "no",
            }
        }

        targets = module.feature_target_indices(row, torch.device("cpu"))

        self.assertEqual(
            {
                "source_reliability": 1,
                "evidence_relevance": 1,
                "detail_sufficiency": 1,
                "conflict_status": 1,
                "answer_permission": 1,
            },
            {key: int(value.item()) for key, value in targets.items()},
        )

    def test_pooled_prompt_hidden_uses_attention_mask_mean(self) -> None:
        module = load_module()
        input_ids = torch.tensor([[2, 4, 6, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        pooled = module.pooled_prompt_hidden(
            FakeHiddenModel(),
            input_ids,
            attention_mask,
            think_steps=3,
        )

        expected = (
            torch.nn.functional.one_hot(torch.tensor([2, 4, 6]), num_classes=8).float().mean(dim=0)
        )
        self.assertTrue(torch.allclose(expected, pooled[0]))

    def test_feature_loss_reports_per_feature_accuracy_and_margin(self) -> None:
        module = load_module()
        reader = module.LatentFeatureReader(d_model=4)
        hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        row = {
            "feature_targets": {
                "source_reliability": "trusted",
                "evidence_relevance": "relevant",
                "detail_sufficiency": "enough",
                "conflict_status": "clear",
                "answer_permission": "yes",
            }
        }

        loss, metrics = module.feature_reader_loss(reader, hidden, row, torch.device("cpu"))

        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(set(module.FEATURE_NAMES), set(metrics["feature_losses"]))
        self.assertEqual(set(module.FEATURE_NAMES), set(metrics["feature_correct"]))
        self.assertEqual(set(module.FEATURE_NAMES), set(metrics["feature_margins"]))

    def test_feature_class_weights_upweight_rare_labels(self) -> None:
        module = load_module()
        rows = [
            {
                "feature_targets": {
                    "source_reliability": "trusted",
                    "evidence_relevance": "relevant",
                    "detail_sufficiency": "enough",
                    "conflict_status": "clear",
                    "answer_permission": "yes",
                }
            },
            {
                "feature_targets": {
                    "source_reliability": "trusted",
                    "evidence_relevance": "relevant",
                    "detail_sufficiency": "enough",
                    "conflict_status": "clear",
                    "answer_permission": "yes",
                }
            },
            {
                "feature_targets": {
                    "source_reliability": "untrusted",
                    "evidence_relevance": "irrelevant",
                    "detail_sufficiency": "missing",
                    "conflict_status": "conflict",
                    "answer_permission": "no",
                }
            },
        ]

        weights = module.feature_class_weights(rows, torch.device("cpu"))

        self.assertGreater(weights["source_reliability"][1].item(), weights["source_reliability"][0].item())
        self.assertGreater(weights["answer_permission"][1].item(), weights["answer_permission"][0].item())

    def test_build_feature_report_requires_all_features_and_permission(self) -> None:
        module = load_module()
        rows = [
            {
                "id": "ok",
                "feature_correct": {
                    "source_reliability": True,
                    "evidence_relevance": True,
                    "detail_sufficiency": True,
                    "conflict_status": True,
                    "answer_permission": True,
                },
                "feature_margins": {
                    "source_reliability": 0.1,
                    "evidence_relevance": 0.2,
                    "detail_sufficiency": 0.3,
                    "conflict_status": 0.4,
                    "answer_permission": 0.5,
                },
            }
        ]

        report = module.build_feature_report(rows, split="heldout", depth=16)

        self.assertTrue(report["accepted"])
        self.assertEqual(1.0, report["all_feature_accuracy"])
        self.assertEqual(0.1, report["min_feature_margin"])

    def test_training_depth_contract_uses_all_depths_by_default(self) -> None:
        module = load_module()

        self.assertEqual([2, 4, 8, 16], module.training_depths_for_step([2, 4, 8, 16], step=7, single_depth=False))
        self.assertEqual([8], module.training_depths_for_step([2, 4, 8, 16], step=7, single_depth=True))


if __name__ == "__main__":
    unittest.main()
