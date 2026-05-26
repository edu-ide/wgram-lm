import sys
import types
import unittest
import importlib
import os
from unittest import mock

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
        for name in list(sys.modules):
            if name == "fla" or name.startswith("fla."):
                sys.modules.pop(name, None)
        sys.path[:] = [path for path in sys.path if "flash-linear-attention" not in path]
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

    def test_official_gated_delta2_exposes_separate_erase_and_write_gates(self):
        sys.modules.pop("fla", None)
        sys.modules.pop("fla.layers", None)

        from qtrm_mm.mixers import OfficialGatedDeltaNet2Mixer

        mixer = OfficialGatedDeltaNet2Mixer(
            d_model=64,
            n_heads=4,
            strict=True,
            head_dim=16,
            num_v_heads=4,
            use_short_conv=False,
        )

        self.assertTrue(mixer.is_official_backend)
        self.assertIsNot(mixer.impl.b_proj, mixer.impl.w_proj)
        self.assertEqual(mixer.impl.b_proj.out_features, 64)
        self.assertEqual(mixer.impl.w_proj.out_features, 64)
        if torch.cuda.is_available():
            x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
            y = mixer.cuda().eval()(x)
            self.assertEqual(y.shape, x.shape)
            mixer.train()
            loss = mixer(x).float().square().mean()
            loss.backward()
            self.assertIsNotNone(mixer.impl.b_proj.weight.grad)
            self.assertIsNotNone(mixer.impl.w_proj.weight.grad)

    def test_official_gated_delta2_never_constructs_torch_fallback_when_missing(self):
        from qtrm_mm.mixers import OfficialGatedDeltaNet2Mixer

        with mock.patch.object(OfficialGatedDeltaNet2Mixer, "_build_impl", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                OfficialGatedDeltaNet2Mixer(d_model=16, n_heads=4, strict=False)

    def test_official_gated_delta2_forward_does_not_runtime_fallback(self):
        class _FailingOfficial(nn.Module):
            def forward(self, hidden_states, attention_mask=None):
                raise RuntimeError("kernel compile failed")

        from qtrm_mm.mixers import OfficialGatedDeltaNet2Mixer

        with mock.patch.object(OfficialGatedDeltaNet2Mixer, "_build_impl", return_value=_FailingOfficial()):
            mixer = OfficialGatedDeltaNet2Mixer(d_model=16, n_heads=4, strict=False)

        self.assertTrue(mixer.is_official_backend)
        self.assertFalse(hasattr(mixer, "runtime_fallback"))
        with self.assertRaisesRegex(RuntimeError, "kernel compile failed"):
            mixer(torch.randn(1, 2, 16))

    def test_qtrm_block_accepts_official_gated_delta2_backend_for_3to1_schedule(self):
        sys.modules.pop("fla", None)
        sys.modules.pop("fla.layers", None)

        from qtrm_mm.blocks import CANONICAL_LT2_ATTN_EVERY, QTRMBlockStack
        from qtrm_mm.config import QTRMConfig
        from qtrm_mm.mixers import OfficialGatedDeltaNet2Mixer

        self.assertEqual(CANONICAL_LT2_ATTN_EVERY, 4)
        cfg = QTRMConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            max_seq_len=8,
            delta_backend="official_gated_delta2",
            strict_backends=True,
            delta_head_dim=16,
            delta_num_v_heads=4,
            delta_use_short_conv=False,
        )

        stack = QTRMBlockStack(cfg, n_layers=4, causal=True, attn_every=4)

        self.assertIsInstance(stack.layers[0].mixer, OfficialGatedDeltaNet2Mixer)
        self.assertIsInstance(stack.layers[1].mixer, OfficialGatedDeltaNet2Mixer)
        self.assertIsInstance(stack.layers[2].mixer, OfficialGatedDeltaNet2Mixer)
        self.assertTrue(stack.layers[3].use_attention)


if __name__ == "__main__":
    unittest.main()
