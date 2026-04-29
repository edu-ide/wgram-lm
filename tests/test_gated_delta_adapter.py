import sys
import types
import unittest
import importlib

import torch
from torch import nn


class _FakeGatedDeltaNet(nn.Module):
    def __init__(self, hidden_size, num_heads, mode="chunk", **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mode = mode
        self.kwargs = kwargs
        self.last_attention_mask = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.last_attention_mask = attention_mask
        return hidden_states + 1.0, None, kwargs.get("past_key_values")


class GatedDeltaAdapterTests(unittest.TestCase):
    def setUp(self):
        self._old_modules = {name: sys.modules.get(name) for name in ["fla", "fla.layers"]}
        fla = types.ModuleType("fla")
        layers = types.ModuleType("fla.layers")
        layers.GatedDeltaNet = _FakeGatedDeltaNet
        fla.layers = layers
        sys.modules["fla"] = fla
        sys.modules["fla.layers"] = layers

    def tearDown(self):
        for name, module in self._old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if "qtrm_mm.backends" in sys.modules:
            importlib.reload(sys.modules["qtrm_mm.backends"])

    def test_fla_gated_delta_uses_official_top_level_fla_layers_import(self):
        from qtrm_mm.mixers import FLADeltaMixer

        mixer = FLADeltaMixer(
            d_model=16,
            n_heads=4,
            backend="fla_gated_delta",
            strict=True,
            mode="chunk",
            expand_k=0.75,
        )

        self.assertTrue(mixer.is_official_backend)
        self.assertIsInstance(mixer.impl, _FakeGatedDeltaNet)
        self.assertEqual(mixer.impl.hidden_size, 16)
        self.assertEqual(mixer.impl.num_heads, 4)
        self.assertEqual(mixer.impl.mode, "chunk")
        self.assertEqual(mixer.impl.kwargs["expand_k"], 0.75)

    def test_fla_gated_delta_forward_preserves_qtrm_tensor_contract_and_mask(self):
        from qtrm_mm.mixers import FLADeltaMixer

        mixer = FLADeltaMixer(d_model=16, n_heads=4, backend="fla_gated_delta", strict=True)
        x = torch.randn(2, 5, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

        y = mixer(x, attention_mask=mask)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x + 1.0))
        self.assertIs(mixer.impl.last_attention_mask, mask)

    def test_strict_fla_gated_delta_raises_when_official_backend_missing(self):
        sys.modules.pop("fla", None)
        sys.modules.pop("fla.layers", None)

        from qtrm_mm.mixers import FLADeltaMixer

        with self.assertRaisesRegex(RuntimeError, "fla_gated_delta"):
            FLADeltaMixer(d_model=16, n_heads=4, backend="fla_gated_delta", strict=True)

    def test_non_strict_fla_gated_delta_marks_torch_fallback(self):
        sys.modules.pop("fla", None)
        sys.modules.pop("fla.layers", None)

        from qtrm_mm.mixers import FLADeltaMixer, TorchGatedDeltaMixer

        mixer = FLADeltaMixer(d_model=16, n_heads=4, backend="fla_gated_delta", strict=False)

        self.assertFalse(mixer.is_official_backend)
        self.assertIsInstance(mixer.impl, TorchGatedDeltaMixer)

    def test_backend_registry_detects_official_gated_delta_symbol(self):
        import qtrm_mm.backends as backends

        backends = importlib.reload(backends)

        self.assertTrue(backends.HAS_FLA_GATED_DELTA)
        self.assertEqual(backends.get_delta_backend("fla_gated_delta").__name__, "FLADeltaMixer")


if __name__ == "__main__":
    unittest.main()
