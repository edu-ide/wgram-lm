from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "303_train_donor_hidden_reft_lite.py"
    spec = importlib.util.spec_from_file_location("donor_hidden_reft_lite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DonorHiddenReftLiteTests(unittest.TestCase):
    def test_adapter_maps_core_hidden_to_donor_delta(self) -> None:
        module = _load_module()
        adapter = module.DonorHiddenReftLite(core_dim=4, donor_dim=6, rank=2)
        delta = adapter(torch.randn(3, 4))
        self.assertEqual(tuple(delta.shape), (3, 6))

    def test_find_output_embeddings_prefers_getter(self) -> None:
        module = _load_module()

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = nn.Linear(4, 5)

            def get_output_embeddings(self):
                return self.head

        model = FakeModel()
        self.assertIs(module.find_output_embeddings(model), model.head)

    def test_feature_summary_reports_dimensions(self) -> None:
        module = _load_module()
        summary = module._feature_summary(
            [
                {
                    "case_id": "a",
                    "pos": 0,
                    "core_hidden": torch.zeros(4),
                    "donor_hidden": torch.zeros(6),
                },
                {
                    "case_id": "a",
                    "pos": 1,
                    "core_hidden": torch.zeros(4),
                    "donor_hidden": torch.zeros(6),
                },
            ]
        )
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["target_positions"], 2)
        self.assertEqual(summary["core_dim"], 4)
        self.assertEqual(summary["donor_dim"], 6)


if __name__ == "__main__":
    unittest.main()
