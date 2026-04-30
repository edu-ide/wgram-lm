import unittest


class ModelConfigTests(unittest.TestCase):
    def test_tie_embeddings_reuses_lm_head_weight(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            tie_embeddings=True,
        )

        model = QTRMMultimodalModel(cfg)

        self.assertIs(model.lm_head.weight, model.text_embed.weight)

    def test_untied_embeddings_keep_separate_weight(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            tie_embeddings=False,
        )

        model = QTRMMultimodalModel(cfg)

        self.assertIsNot(model.lm_head.weight, model.text_embed.weight)

    def test_fresh_tied_lm_logits_are_small_enough_for_stable_smoke_training(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            tie_embeddings=True,
        )
        model = QTRMMultimodalModel(cfg)
        out = model(torch.randint(0, cfg.vocab_size, (2, 8)))

        self.assertLess(float(out["logits"].detach().abs().max()), 5.0)

    def test_recursive_core_runs_on_fixed_workspace_prefix(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=5,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=3,
            tie_embeddings=True,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))
        visual = torch.randn(2, 3, cfg.visual_dim)

        out = model(input_ids, visual_features=visual)

        self.assertEqual(out["z_h"].shape[1], cfg.workspace_tokens)
        self.assertEqual(out["logits"].shape[1], cfg.workspace_tokens + 3 + input_ids.shape[1])

    def test_coda_can_use_a_separate_attention_schedule_from_core(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=4,
            n_coda_layers=2,
            attn_every=4,
            coda_attn_every=2,
            workspace_tokens=5,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=3,
            tie_embeddings=True,
        )

        model = QTRMMultimodalModel(cfg)

        self.assertEqual([layer.use_attention for layer in model.core.fast_stack.layers], [False, False, False, True])
        self.assertEqual([layer.use_attention for layer in model.coda.layers], [False, True])

    def test_donor_logits_can_be_used_as_final_token_distribution(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_logits_scale=0.0,
            donor_logits_scale=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        out = model(input_ids, text_states=text_states, donor_logits=donor_logits)
        offset = out["logits"].shape[1] - input_ids.shape[1]

        self.assertTrue(torch.allclose(out["logits"][:, offset:], donor_logits))

    def test_model_returns_student_logits_before_donor_fusion(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_logits_scale=1.0,
            donor_logits_scale=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        without_donor = model(input_ids, text_states=text_states)
        with_donor = model(input_ids, text_states=text_states, donor_logits=donor_logits)

        self.assertIn("qtrm_logits", with_donor)
        self.assertTrue(torch.allclose(with_donor["qtrm_logits"], without_donor["logits"], atol=1e-4))
        self.assertFalse(torch.allclose(with_donor["logits"], with_donor["qtrm_logits"]))

    def test_bounded_residual_clamps_qtrm_contribution_before_donor_fusion(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_logits_scale=10.0,
            donor_logits_scale=1.0,
            qtrm_residual_clamp=0.25,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        out = model(input_ids, text_states=text_states, donor_logits=donor_logits)
        offset = out["logits"].shape[1] - input_ids.shape[1]
        residual = out["logits"][:, offset:] - donor_logits

        self.assertIn("qtrm_residual_logits", out)
        self.assertLessEqual(float(out["qtrm_residual_logits"].abs().max()), 0.2501)
        self.assertLessEqual(float(residual.abs().max()), 0.2501)

    def test_bounded_residual_gate_uses_stable_initial_gate(self):
        import math
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_logits_scale=1.0,
            donor_logits_scale=1.0,
            qtrm_residual_gate_enabled=True,
            qtrm_residual_gate_init_bias=-2.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)
        donor_logits = torch.zeros(1, 4, cfg.vocab_size)

        out = model(input_ids, text_states=text_states, donor_logits=donor_logits)

        self.assertIn("qtrm_residual_gate", out)
        self.assertAlmostEqual(
            float(out["qtrm_residual_gate"].item()),
            1.0 / (1.0 + math.exp(2.0)),
            places=5,
        )

    def test_residual_gate_normalizes_large_latents_before_linear(self):
        import math
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_residual_gate_enabled=True,
            qtrm_residual_gate_init_bias=-2.0,
            qtrm_residual_gate_normalize=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.residual_gate.weight.data.fill_(0.001)
        model.residual_gate.bias.data.fill_(-2.0)
        z_h = torch.full((1, 3, cfg.d_model), 1000.0)

        gate = model._compute_residual_gate(z_h)

        expected = 1.0 / (1.0 + math.exp(-(-2.0 + cfg.d_model * 0.001)))
        self.assertAlmostEqual(float(gate.item()), expected, places=5)
        self.assertLess(float(gate.item()), 0.2)

    def test_residual_gate_minimum_floor_prevents_total_closure(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_residual_gate_enabled=True,
            qtrm_residual_gate_init_bias=-100.0,
            qtrm_residual_gate_min=0.05,
        )
        model = QTRMMultimodalModel(cfg)
        model.residual_gate.weight.data.zero_()
        z_h = torch.zeros((1, 3, cfg.d_model))

        gate = model._compute_residual_gate(z_h)

        self.assertAlmostEqual(float(gate.item()), 0.05, places=5)

    def test_workspace_can_use_perceiver_style_depth(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=5,
            workspace_layers=3,
            workspace_ff_mult=2,
            workspace_include_latents_in_kv=True,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        out = model(torch.randint(0, cfg.vocab_size, (2, 7)))

        self.assertEqual(len(model.workspace.layers), 3)
        self.assertEqual(out["z_h"].shape, (2, cfg.workspace_tokens, cfg.d_model))

    def test_workspace_ablation_removes_latent_prefix_from_logits(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=5,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        baseline = model(input_ids)
        ablated = model(input_ids, disable_workspace=True)

        self.assertEqual(baseline["logits"].shape[1], cfg.workspace_tokens + input_ids.shape[1])
        self.assertEqual(ablated["logits"].shape[1], input_ids.shape[1])
        self.assertEqual(int(ablated["trajectory_len"].item()), 0)

    def test_core_ablation_keeps_workspace_prefix_without_recursive_steps(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=5,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        out = model(input_ids, disable_core=True)

        self.assertEqual(out["logits"].shape[1], cfg.workspace_tokens + input_ids.shape[1])
        self.assertEqual(out["z_h"].shape, (2, cfg.workspace_tokens, cfg.d_model))
        self.assertEqual(int(out["trajectory_len"].item()), 0)

    def test_residual_head_ablation_zeros_qtrm_contribution_before_donor_fusion(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            qtrm_logits_scale=1.0,
            donor_logits_scale=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        out = model(input_ids, donor_logits=donor_logits, disable_qtrm_residual=True)
        offset = out["logits"].shape[1] - input_ids.shape[1]

        self.assertTrue(torch.allclose(out["qtrm_residual_logits"], torch.zeros_like(out["qtrm_residual_logits"])))
        self.assertTrue(torch.allclose(out["logits"][:, offset:], donor_logits))

    def test_coda_ablation_skips_coda_module(self):
        import torch
        from torch import nn
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        class FailingCoda(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError("coda should be skipped")

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        model.coda = FailingCoda()

        out = model(torch.randint(0, cfg.vocab_size, (1, 4)), disable_coda=True)

        self.assertEqual(out["logits"].shape[1], cfg.workspace_tokens + 4)

    def test_donor_context_ablation_removes_projected_donor_prefix(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)

        with_context = model(input_ids, text_states=text_states, disable_workspace=True)
        without_context = model(
            input_ids,
            text_states=text_states,
            disable_workspace=True,
            disable_donor_context=True,
        )

        self.assertEqual(with_context["logits"].shape[1], 8)
        self.assertEqual(without_context["logits"].shape[1], 4)

    def test_workspace_only_context_removes_direct_donor_prefix_from_coda(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)

        direct_context = model(input_ids, text_states=text_states)
        workspace_only = model(input_ids, text_states=text_states, workspace_only_context=True)

        self.assertEqual(direct_context["logits"].shape[1], cfg.workspace_tokens + 8)
        self.assertEqual(workspace_only["logits"].shape[1], cfg.workspace_tokens + 4)


if __name__ == "__main__":
    unittest.main()
