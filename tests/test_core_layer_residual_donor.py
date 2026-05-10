from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "306_train_core_layer_residual_donor.py"
    spec = importlib.util.spec_from_file_location("core_layer_residual", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreLayerResidualDonorTests(unittest.TestCase):
    def test_adapter_shape(self) -> None:
        module = _load_module()
        adapter = module.CoreLayerResidualAdapter(core_dim=4, donor_dim=6, rank=2)
        out = adapter(torch.randn(5, 4))
        self.assertEqual(tuple(out.shape), (5, 6))

    def test_target_positions_predict_answer_tokens(self) -> None:
        module = _load_module()
        self.assertEqual(module._target_positions(prompt_len=4, target_len=3), [3, 4, 5])

    def test_patch_layer_output_patches_tuple_hidden_only_at_positions(self) -> None:
        module = _load_module()
        hidden = torch.zeros(1, 4, 3)
        other = torch.ones(1)
        delta = torch.tensor([[1.0, 2.0, 3.0]])
        patched, same_other = module._patch_layer_output(
            (hidden, other),
            positions=[1, 3],
            delta=delta,
        )
        self.assertIs(same_other, other)
        self.assertTrue(torch.allclose(patched[:, 0, :], torch.zeros(1, 3)))
        self.assertTrue(torch.allclose(patched[:, 1, :], delta))
        self.assertTrue(torch.allclose(patched[:, 3, :], delta))

    def test_find_layers_uses_model_layers(self) -> None:
        module = _load_module()

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

        self.assertEqual(len(module.find_layers(FakeModel())), 2)


if __name__ == "__main__":
    unittest.main()
