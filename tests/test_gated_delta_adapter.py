import sys
import types
import unittest
import importlib
import os

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
        self._old_disable_local_fla = os.environ.get("QTRM_DISABLE_LOCAL_FLA_REFERENCE")
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
        if self._old_disable_local_fla is None:
            os.environ.pop("QTRM_DISABLE_LOCAL_FLA_REFERENCE", None)
        else:
            os.environ["QTRM_DISABLE_LOCAL_FLA_REFERENCE"] = self._old_disable_local_fla
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

    def test_qtrm_block_passes_qwen35_gdn_parameters_to_official_backend(self):
        from qtrm_mm.blocks import QTRMBlockStack
        from qtrm_mm.config import QTRMConfig

        cfg = QTRMConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=8,
            delta_backend="fla_gated_delta",
            strict_backends=True,
            delta_head_dim=8,
            delta_num_v_heads=4,
            delta_expand_v=1.0,
            delta_mode="chunk",
            delta_use_short_conv=False,
            delta_conv_size=4,
            delta_norm_eps=1e-6,
        )

        stack = QTRMBlockStack(cfg, n_layers=1, causal=True, attn_every=2)
        mixer = stack.layers[0].mixer.impl

        self.assertIsInstance(mixer, _FakeGatedDeltaNet)
        self.assertEqual(mixer.hidden_size, 32)
        self.assertEqual(mixer.num_heads, 4)
        self.assertEqual(mixer.mode, "chunk")
        self.assertEqual(mixer.kwargs["head_dim"], 8)
        self.assertEqual(mixer.kwargs["num_v_heads"], 4)
        self.assertEqual(mixer.kwargs["expand_v"], 1.0)
        self.assertFalse(mixer.kwargs["use_short_conv"])
        self.assertEqual(mixer.kwargs["conv_size"], 4)
        self.assertEqual(mixer.kwargs["norm_eps"], 1e-6)

    def test_strict_fla_gated_delta_raises_when_official_backend_missing(self):
        sys.modules.pop("fla", None)
        sys.modules.pop("fla.layers", None)
        os.environ["QTRM_DISABLE_LOCAL_FLA_REFERENCE"] = "1"

        from qtrm_mm.mixers import FLADeltaMixer

        with self.assertRaisesRegex(RuntimeError, "fla_gated_delta"):
            FLADeltaMixer(d_model=16, n_heads=4, backend="fla_gated_delta", strict=True)

    def test_non_strict_fla_gated_delta_marks_torch_fallback(self):
        sys.modules.pop("fla", None)
        sys.modules.pop("fla.layers", None)
        os.environ["QTRM_DISABLE_LOCAL_FLA_REFERENCE"] = "1"

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
