from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "304_train_core_soft_prefix_donor.py"
    spec = importlib.util.spec_from_file_location("core_soft_prefix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreSoftPrefixDonorTests(unittest.TestCase):
    def test_adapter_shape(self) -> None:
        module = _load_module()
        adapter = module.CoreSoftPrefixAdapter(
            core_dim=4,
            donor_dim=6,
            prefix_tokens=3,
            rank=2,
        )
        out = adapter(torch.randn(5, 4))
        self.assertEqual(tuple(out.shape), (5, 3, 6))

    def test_state_conditioned_adapter_shape(self) -> None:
        module = _load_module()
        adapter = module.StateConditionedSoftPrefixAdapter(
            core_dim=4,
            state_dim=5,
            donor_dim=6,
            prefix_tokens=3,
            rank=2,
        )
        out = adapter(torch.randn(2, 4), torch.randn(2, 5))
        self.assertEqual(tuple(out.shape), (2, 3, 6))

    def test_zero_core_maps_to_zero_prefix_initially(self) -> None:
        module = _load_module()
        adapter = module.CoreSoftPrefixAdapter(
            core_dim=4,
            donor_dim=6,
            prefix_tokens=3,
            rank=2,
        )
        out = adapter(torch.zeros(1, 4))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))

    def test_state_features_use_final_step(self) -> None:
        module = _load_module()
        logits = torch.zeros(1, 2, 3, 4)
        logits[:, 1, :, 2] = 10.0
        features = module._state_features_from_outputs(
            {"core_role_value_state_logits": logits},
            key="core_role_value_state_logits",
            mode="argmax_onehot",
        )
        self.assertEqual(tuple(features.shape), (12,))
        self.assertEqual(float(features.sum().item()), 3.0)

    def test_state_features_can_concat_multiple_keys(self) -> None:
        module = _load_module()
        outputs = {
            "a": torch.zeros(1, 2, 3),
            "b": torch.zeros(1, 2, 5),
        }
        features = module._state_features_from_outputs(
            outputs,
            key="a,b",
            mode="softmax",
        )
        self.assertEqual(tuple(features.shape), (8,))

    def test_find_input_embeddings_prefers_getter(self) -> None:
        module = _load_module()

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(8, 4)

            def get_input_embeddings(self):
                return self.emb

        model = FakeModel()
        self.assertIs(module.find_input_embeddings(model), model.emb)

    def test_scheduled_sampling_respects_warmup_and_position(self) -> None:
        module = _load_module()
        self.assertFalse(
            module._use_scheduled_sampling(
                step=10,
                pos=1,
                scheduled_sampling_prob=0.3,
                warmup_steps=20,
            )
        )

    def test_answer_token_ids_can_append_eos_within_limit(self) -> None:
        module = _load_module()

        class FakeTokenizer:
            eos_token_id = 99

            def encode(self, text: str, add_special_tokens: bool = False):
                return [1, 2, 3]

        ids = module._answer_token_ids(
            FakeTokenizer(),
            "123",
            max_target_tokens=3,
            append_eos_target=True,
        )
        self.assertEqual(ids, [1, 2, 99])
        self.assertFalse(
            module._use_scheduled_sampling(
                step=30,
                pos=0,
                scheduled_sampling_prob=0.3,
                warmup_steps=20,
            )
        )
        self.assertTrue(
            module._use_scheduled_sampling(
                step=30,
                pos=1,
                scheduled_sampling_prob=0.3,
                warmup_steps=20,
            )
        )


if __name__ == "__main__":
    unittest.main()
