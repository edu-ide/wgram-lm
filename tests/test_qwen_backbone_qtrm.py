from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from qtrm_mm.config import QTRMConfig
from qtrm_mm.qwen_backbone_qtrm import (
    QwenBackboneQTRM,
    build_qtrm_core_config_from_qwen,
)


class _DummyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        text_config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=16,
            vocab_size=13,
            full_attention_interval=4,
            linear_key_head_dim=4,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_theta": 10000.0},
        )
        self.config = SimpleNamespace(text_config=text_config)
        self.embed = nn.Embedding(13, 8)
        self.lm_head = nn.Linear(8, 13, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        if not return_dict:
            return (logits,)
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            items=lambda: {"logits": logits, "hidden_states": hidden_states}.items(),
        )


class _DummyQwenLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layer_type = "full_attention"
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.calls = 0
        self.last_attention_mask_shape = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        del position_embeddings, position_ids, past_key_values, kwargs
        self.calls += 1
        self.last_attention_mask_shape = (
            tuple(attention_mask.shape) if attention_mask is not None else None
        )
        return hidden_states + self.proj(hidden_states)


class _DummyTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config.text_config
        self.layers = nn.ModuleList([_DummyQwenLayer(8), _DummyQwenLayer(8)])
        self.norm = nn.LayerNorm(8)

    def rotary_emb(self, hidden_states, position_ids):
        del position_ids
        return hidden_states, hidden_states

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        del past_key_values
        return attention_mask


class _DummyQwenWithLayers(_DummyQwen):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _DummyTextModel(self.config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        del use_cache, kwargs
        text_model = self.model.language_model
        hidden = self.embed(input_ids)
        hidden_states = [hidden] if output_hidden_states else None
        for layer in text_model.layers:
            position_embeddings = text_model.rotary_emb(hidden, None)
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
            )
            if output_hidden_states:
                hidden_states.append(hidden)
        hidden = text_model.norm(hidden)
        logits = self.lm_head(hidden)
        if not return_dict:
            return (logits,)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            items=lambda: {
                "logits": logits,
                "hidden_states": tuple(hidden_states) if output_hidden_states else None,
            }.items(),
        )


class _DummyOuroLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention_type = "full_attention"
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.calls = 0
        self.last_attention_mask_shape = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        del position_ids, past_key_value, use_cache, cache_position, position_embeddings, kwargs
        self.calls += 1
        self.last_attention_mask_shape = (
            tuple(attention_mask.shape) if attention_mask is not None else None
        )
        return hidden_states + self.proj(hidden_states)


class _DummyOuroModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            layer_types=["full_attention", "full_attention"],
            rope_theta=10000.0,
        )
        self.layers = nn.ModuleList([_DummyOuroLayer(8), _DummyOuroLayer(8)])

    def rotary_emb(self, hidden_states, position_ids):
        del position_ids
        return hidden_states, hidden_states


class _DummyOuroForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyOuroModel()


def _tiny_core_config() -> QTRMConfig:
    return QTRMConfig(
        vocab_size=13,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=16,
        max_seq_len=8,
        n_core_layers=1,
        h_cycles=1,
        l_cycles=1,
        outer_steps=1,
        delta_backend="torch_gated_delta",
        strict_backends=False,
        use_stable_inject=True,
        core_context_enabled=False,
    )


class QwenBackboneQTRMTests(unittest.TestCase):
    def test_gate_zero_matches_qwen_logits(self):
        torch.manual_seed(1)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=_tiny_core_config(),
            freeze_qwen=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        wrapped = model(input_ids, core_gate_override=0.0).logits
        self.assertTrue(torch.allclose(base, wrapped, atol=1e-6))

    def test_core_on_changes_logits_and_qwen_is_frozen(self):
        torch.manual_seed(2)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=_tiny_core_config(),
            freeze_qwen=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        core_on = model(input_ids, core_gate_override=0.25).logits
        self.assertGreater(float((base - core_on).abs().max()), 1e-6)
        report = model.report()
        self.assertEqual(report.qwen_trainable_parameters, 0)
        self.assertGreater(report.qtrm_trainable_parameters, 0)
        self.assertFalse(report.runtime_donor)
        self.assertTrue(report.integrated_qwen_backbone)
        self.assertTrue(report.standalone_graph)

    def test_mandatory_core_uses_full_gate_by_default_but_keeps_ablation(self):
        torch.manual_seed(20)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=_tiny_core_config(),
            freeze_qwen=True,
            core_gate_init=-12.0,
            mandatory_core=True,
            residual_scale=0.25,
        )
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        gate_zero = model(input_ids, core_gate_override=0.0).logits
        explicit_full = model(input_ids, core_gate_override=1.0).logits
        normal = model(input_ids)

        self.assertTrue(torch.allclose(base, gate_zero, atol=1e-6))
        self.assertTrue(torch.allclose(normal.logits, explicit_full, atol=1e-6))
        self.assertGreater(float((base - normal.logits).abs().max()), 1e-6)
        self.assertEqual(float(normal.qtrm_core_gate), 1.0)
        self.assertEqual(model.normal_core_gate_value(), 1.0)
        self.assertFalse(model.core_gate_logit.requires_grad)
        report = model.report()
        self.assertTrue(report.mandatory_core)
        self.assertEqual(report.normal_core_gate, 1.0)

    def test_qwen_backbone_core_defaults_to_causal_attention(self):
        cfg = build_qtrm_core_config_from_qwen(
            _DummyQwen().config,
            max_seq_len=8,
            n_core_layers=4,
        )
        self.assertTrue(cfg.core_causal)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=cfg,
            freeze_qwen=True,
        )
        fast_attention = model.core.fast_stack.layers[3].mixer
        slow_attention = model.core.slow_stack.layers[3].mixer
        self.assertTrue(fast_attention.causal)
        self.assertTrue(slow_attention.causal)

    def test_qwen_layer_wrapped_core_uses_qwen_layer_with_causal_mask(self):
        torch.manual_seed(3)
        qwen = _DummyQwenWithLayers()
        cfg = _tiny_core_config()
        cfg.core_causal = True
        model = QwenBackboneQTRM(
            qwen,
            core_config=cfg,
            core_impl="qwen_layer_wrapped",
            qwen_core_layer_indices=(1,),
            freeze_qwen=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        core_on = model(input_ids, core_gate_override=0.25).logits

        wrapped_layer = qwen.model.language_model.layers[1]
        self.assertGreater(wrapped_layer.calls, 0)
        self.assertEqual(wrapped_layer.last_attention_mask_shape, (1, 1, 4, 4))
        self.assertGreater(float((base - core_on).abs().max()), 1e-6)

    def test_mid_layer_suffix_insertion_reprocesses_remaining_qwen_layers(self):
        torch.manual_seed(30)
        qwen = _DummyQwenWithLayers()
        cfg = _tiny_core_config()
        cfg.core_causal = True
        model = QwenBackboneQTRM(
            qwen,
            core_config=cfg,
            core_impl="qwen_layer_wrapped",
            qwen_core_layer_indices=(1,),
            freeze_qwen=True,
            core_insertion_mode="mid_layer_suffix",
            core_insert_after_layer=0,
            core_adapter_dim=4,
            core_delta_adapter_mode="adapter_only",
            mandatory_core=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        core_on = model(input_ids).logits

        self.assertTrue(torch.allclose(base, core_on, atol=1e-5))
        self.assertEqual(model.core_suffix_stack.layer_indices, (1,))
        self.assertEqual(model.report().core_insertion_mode, "mid_layer_suffix")
        self.assertEqual(model.report().core_insert_after_layer, 0)

    def test_ouro_shared_qwen_layer_core_reuses_same_stack(self):
        qwen = _DummyQwenWithLayers()
        cfg = _tiny_core_config()
        cfg.core_causal = True
        model = QwenBackboneQTRM(
            qwen,
            core_config=cfg,
            core_impl="ouro_shared_qwen_layer",
            qwen_core_layer_indices=(1,),
            freeze_qwen=True,
        )
        self.assertIs(model.core.fast_stack, model.core.slow_stack)
        self.assertTrue(model.core.shared_stack)

    def test_optional_core_delta_adapter_is_trainable_without_breaking_gate_zero(self):
        torch.manual_seed(4)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=_tiny_core_config(),
            freeze_qwen=True,
            core_adapter_dim=4,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        gate_zero = model(input_ids, core_gate_override=0.0).logits

        self.assertTrue(torch.allclose(base, gate_zero, atol=1e-6))
        self.assertIsNotNone(model.core_delta_adapter)
        self.assertGreater(model.report().qtrm_trainable_parameters, 0)

    def test_adapter_only_delta_starts_as_noop_with_zero_init_adapter(self):
        torch.manual_seed(40)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=_tiny_core_config(),
            freeze_qwen=True,
            core_adapter_dim=4,
            core_delta_adapter_mode="adapter_only",
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        core_on = model(input_ids, core_gate_override=1.0).logits

        self.assertTrue(torch.allclose(base, core_on, atol=1e-6))
        self.assertEqual(model.core_delta_adapter_mode, "adapter_only")

    def test_token_mlp_residual_gate_reports_mean_and_is_trainable(self):
        torch.manual_seed(41)
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=_tiny_core_config(),
            freeze_qwen=True,
            core_residual_gate_mode="token_mlp",
            core_residual_gate_dim=4,
            core_residual_gate_init=-2.0,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])

        output = model(input_ids, core_gate_override=1.0)

        self.assertIsNotNone(model.core_residual_gate)
        self.assertEqual(model.report().core_residual_gate_mode, "token_mlp")
        self.assertTrue(hasattr(output, "qtrm_core_residual_gate_mean"))
        self.assertTrue(hasattr(output, "qtrm_core_residual_gate"))
        self.assertEqual(tuple(output.qtrm_core_residual_gate.shape), (1, 4, 1))
        self.assertTrue(output.qtrm_core_residual_gate.requires_grad)
        self.assertGreater(float(output.qtrm_core_residual_gate_mean), 0.0)
        self.assertLess(float(output.qtrm_core_residual_gate_mean), 0.5)
        self.assertTrue(any(parameter.requires_grad for parameter in model.core_residual_gate.parameters()))

    def test_ouro_weight_wrapped_core_uses_ouro_layer_with_qwen_backbone(self):
        torch.manual_seed(5)
        ouro = _DummyOuroForCausalLM()
        cfg = _tiny_core_config()
        cfg.core_causal = True
        model = QwenBackboneQTRM(
            _DummyQwen(),
            core_config=cfg,
            core_impl="ouro_weight_wrapped",
            ouro_model=ouro,
            ouro_core_layer_indices=(1,),
            freeze_qwen=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        base = model(input_ids, force_core_off=True).logits
        core_on = model(input_ids, core_gate_override=0.25).logits

        wrapped_layer = ouro.model.layers[1]
        self.assertGreater(wrapped_layer.calls, 0)
        self.assertEqual(wrapped_layer.last_attention_mask_shape, (1, 1, 4, 4))
        self.assertGreater(float((base - core_on).abs().max()), 1e-6)
        self.assertTrue(all(not parameter.requires_grad for parameter in ouro.parameters()))
        self.assertLess(
            model.report().qtrm_parameters,
            sum(int(parameter.numel()) for parameter in ouro.parameters()),
        )

    def test_qwen_layer_wrapped_core_convergence_halt_stops_outer_loop(self):
        torch.manual_seed(6)
        qwen = _DummyQwenWithLayers()
        cfg = _tiny_core_config()
        cfg.core_causal = True
        cfg.h_cycles = 1
        cfg.l_cycles = 1
        cfg.outer_steps = 3
        cfg.core_convergence_halt_enabled = True
        cfg.core_convergence_halt_threshold = 1.0e9
        cfg.core_convergence_halt_min_outer = 1
        model = QwenBackboneQTRM(
            qwen,
            core_config=cfg,
            core_impl="qwen_layer_wrapped",
            qwen_core_layer_indices=(1,),
            freeze_qwen=True,
        )

        hidden = torch.randn(2, 4, 8)
        _, _, trajectory, info = model.core(hidden, attention_mask=torch.ones(2, 4))

        self.assertEqual(len(trajectory), 1)
        self.assertTrue(torch.equal(info["outer_iterations"], torch.ones(2, dtype=torch.long)))
        self.assertTrue(torch.equal(info["converged"], torch.ones(2, dtype=torch.bool)))
        self.assertEqual(tuple(info["convergence_delta"].shape), (2, 1))

        output = model(torch.tensor([[1, 2, 3, 4]]), core_gate_override=0.25)
        self.assertTrue(hasattr(output, "qtrm_core_outer_iterations"))
        self.assertTrue(hasattr(output, "qtrm_core_converged"))
        self.assertTrue(hasattr(output, "qtrm_core_convergence_delta"))

    def test_qwen_layer_wrapped_core_uses_step_conditioning(self):
        torch.manual_seed(7)
        qwen = _DummyQwenWithLayers()
        cfg = _tiny_core_config()
        cfg.core_causal = True
        cfg.h_cycles = 1
        cfg.l_cycles = 2
        cfg.outer_steps = 1
        cfg.core_step_conditioning_enabled = True
        cfg.core_step_conditioning_max_steps = 8
        cfg.core_step_conditioning_scale = 1.0
        model = QwenBackboneQTRM(
            qwen,
            core_config=cfg,
            core_impl="qwen_layer_wrapped",
            qwen_core_layer_indices=(1,),
            freeze_qwen=True,
        )

        self.assertIsNotNone(model.core.step_conditioning)
        self.assertTrue(model.core.step_conditioning.weight.requires_grad)
        self.assertGreaterEqual(model.core.step_conditioning.num_embeddings, 8)
        report = model.report()
        self.assertGreater(report.qtrm_trainable_parameters, 8 * 8)

    def test_partial_qwen_unfreeze_opens_only_selected_original_backbone_parts(self):
        qwen = _DummyQwenWithLayers()
        model = QwenBackboneQTRM(
            qwen,
            core_config=_tiny_core_config(),
            core_impl="qwen_layer_wrapped",
            qwen_core_layer_indices=(1,),
            freeze_qwen=True,
            mandatory_core=True,
        )

        info = model.set_qwen_partial_trainable(
            layer_indices=(1,),
            train_lm_head=True,
        )

        layer0_trainable = any(
            parameter.requires_grad for parameter in qwen.model.language_model.layers[0].parameters()
        )
        layer1_trainable = any(
            parameter.requires_grad for parameter in qwen.model.language_model.layers[1].parameters()
        )
        embed_trainable = any(parameter.requires_grad for parameter in qwen.embed.parameters())
        lm_head_trainable = any(parameter.requires_grad for parameter in qwen.lm_head.parameters())

        self.assertFalse(layer0_trainable)
        self.assertTrue(layer1_trainable)
        self.assertFalse(embed_trainable)
        self.assertTrue(lm_head_trainable)
        self.assertEqual(info["layer_indices"], [1])
        self.assertGreater(info["qwen_trainable_parameters"], 0)
        self.assertGreater(model.report().qwen_trainable_parameters, 0)


if __name__ == "__main__":
    unittest.main()
