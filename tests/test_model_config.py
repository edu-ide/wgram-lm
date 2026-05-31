import unittest


class ModelConfigTests(unittest.TestCase):
    def test_tie_embeddings_reuses_lm_head_weight(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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

    def test_core_enabled_false_makes_core_off_the_canonical_forward(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            core_enabled=False,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        with torch.no_grad():
            canonical = model(input_ids)
            explicit_core_off = model(input_ids, disable_core=True)

        self.assertTrue(torch.allclose(canonical["logits"], explicit_core_off["logits"]))
        self.assertTrue(torch.equal(canonical["core_steps"], torch.zeros(2, dtype=torch.long)))
        self.assertEqual(int(canonical["trajectory_len"].item()), 0)

    def test_identity_safe_core_runs_but_starts_near_core_off_output(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            core_output_blend_enabled=True,
            core_output_blend_init_bias=-20.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        with torch.no_grad():
            mandatory_core = model(input_ids)
            explicit_core_off = model(input_ids, disable_core=True)

        self.assertTrue(torch.allclose(mandatory_core["logits"], explicit_core_off["logits"], atol=1e-5))
        self.assertTrue(torch.equal(mandatory_core["core_steps"], torch.ones(2, dtype=torch.long)))
        self.assertEqual(int(mandatory_core["trajectory_len"].item()), 1)
        self.assertLess(float(mandatory_core["core_output_blend_gate_mean"].max()), 1e-5)

    def test_answer_bottleneck_can_require_a_running_core(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            answer_bottleneck_enabled=True,
            answer_bottleneck_requires_core=True,
            answer_bottleneck_requires_workspace_memory=False,
            qtrm_logits_scale=1.0,
            donor_logits_scale=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        full = model(input_ids, donor_logits=donor_logits)
        core_off = model(input_ids, donor_logits=donor_logits, disable_core=True)
        offset = core_off["logits"].shape[1] - input_ids.shape[1]

        self.assertGreater(float(full["qtrm_residual_logits"].detach().abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(
                core_off["qtrm_residual_logits"],
                torch.zeros_like(core_off["qtrm_residual_logits"]),
            )
        )
        self.assertTrue(torch.allclose(core_off["logits"][:, offset:], donor_logits))

    def test_core_loop_readout_can_require_a_running_core_with_donor_fallback(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            core_loop_readout_enabled=True,
            core_loop_readout_requires_core=True,
            qtrm_logits_scale=1.0,
            donor_logits_scale=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        full = model(input_ids, donor_logits=donor_logits)
        core_off = model(input_ids, donor_logits=donor_logits, disable_core=True)
        offset = core_off["logits"].shape[1] - input_ids.shape[1]

        self.assertGreater(float(full["qtrm_residual_logits"].detach().abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(
                core_off["qtrm_residual_logits"],
                torch.zeros_like(core_off["qtrm_residual_logits"]),
            )
        )
        self.assertTrue(torch.allclose(core_off["logits"][:, offset:], donor_logits))

    def test_typed_algorithmic_value_state_bridge_exposes_answer_loop_tokens(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            typed_algorithmic_value_state_enabled=True,
            typed_algorithmic_value_state_recurrent_enabled=True,
            typed_algorithmic_value_state_scalar_vocab_size=17,
            typed_algorithmic_value_state_answer_bridge_enabled=True,
            typed_algorithmic_value_state_answer_bridge_gate_min=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])

        with torch.no_grad():
            full = model(input_ids)
            bridge_off = model(
                input_ids,
                disable_typed_algorithmic_value_state_answer_bridge=True,
            )

        tokens = full["typed_algorithmic_value_state_answer_bridge_tokens"]
        gate_mean = full["typed_algorithmic_value_state_answer_bridge_gate_mean"]
        off_tokens = bridge_off["typed_algorithmic_value_state_answer_bridge_tokens"]

        self.assertEqual(tokens.shape, (1, cfg.outer_steps, 1, cfg.d_model))
        self.assertEqual(gate_mean.shape, (1, cfg.outer_steps))
        self.assertGreater(float(tokens.detach().abs().max()), 0.0)
        self.assertTrue(torch.allclose(off_tokens, torch.zeros_like(off_tokens)))

    def test_coda_can_use_a_separate_attention_schedule_from_core(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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

    def test_generation_verifier_heads_return_prompt_level_logits(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
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
            generation_verifier_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        out = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertIn("generation_repeat_logits", out)
        self.assertIn("generation_stop_logits", out)
        self.assertIn("generation_quality_logits", out)
        self.assertEqual(out["generation_repeat_logits"].shape, (2,))
        self.assertEqual(out["generation_stop_logits"].shape, (2,))
        self.assertEqual(out["generation_quality_logits"].shape, (2,))
        self.assertEqual(out["generation_verifier_pooled"].shape, (2, cfg.d_model))

    def test_controller_pool_uses_final_text_state_not_workspace_slot(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
        )
        model = QTRMMultimodalModel(cfg)
        out = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(out["controller_pooled"].shape, (2, cfg.d_model))
        self.assertTrue(torch.equal(out["controller_pooled"], out["generation_verifier_pooled"]))
        self.assertFalse(torch.equal(out["controller_pooled"], out["pooled"]))

    def test_controller_signal_injects_into_controller_pool_only_when_enabled(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
            controller_signal_enabled=True,
            controller_signal_dim=2,
        )
        model = QTRMMultimodalModel(cfg)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        signal = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with_signal = model(ids, controller_signal=signal)
        without_signal = model(ids, controller_signal=signal, disable_controller_signal=True)

        self.assertEqual(with_signal["controller_pooled"].shape, (2, cfg.d_model))
        self.assertEqual(float(with_signal["controller_signal_used"].item()), 1.0)
        self.assertEqual(float(without_signal["controller_signal_used"].item()), 0.0)
        self.assertFalse(
            torch.equal(
                with_signal["controller_pooled"][1],
                without_signal["controller_pooled"][1],
            )
        )

    def test_temporal_spatial_context_is_causal_prefix_and_ablatable(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
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
            temporal_spatial_context_enabled=True,
            temporal_spatial_context_dim=5,
            temporal_spatial_context_max_tokens=2,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        context = torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0, 0.5]]])

        with_context = model(input_ids, temporal_spatial_context=context)
        without_context = model(
            input_ids,
            temporal_spatial_context=context,
            disable_temporal_spatial_context=True,
        )

        self.assertEqual(int(with_context["temporal_spatial_context_token_count"].item()), 2)
        self.assertEqual(int(without_context["temporal_spatial_context_token_count"].item()), 0)
        self.assertEqual(
            with_context["logits"].shape[1],
            without_context["logits"].shape[1] + 2,
        )
        self.assertEqual(with_context["logits"][:, -input_ids.shape[1] :, :].shape[1], 4)

    def test_temporal_spatial_context_accepts_single_vector_as_one_token(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
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
            temporal_spatial_context_enabled=True,
            temporal_spatial_context_dim=5,
            temporal_spatial_context_max_tokens=2,
        )
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.tensor([[1, 2, 3]]),
            temporal_spatial_context=torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]]),
        )

        self.assertEqual(int(out["temporal_spatial_context_token_count"].item()), 1)

    def test_temporal_spatial_context_rejects_wrong_feature_dim(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
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
            temporal_spatial_context_enabled=True,
            temporal_spatial_context_dim=5,
        )
        model = QTRMMultimodalModel(cfg)

        with self.assertRaisesRegex(ValueError, "temporal_spatial_context"):
            model(
                torch.tensor([[1, 2, 3]]),
                temporal_spatial_context=torch.zeros(1, 1, 4),
            )

    def test_learned_controller_signal_comes_from_core_latent_not_external_input(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
            controller_signal_enabled=True,
            controller_signal_dim=2,
            controller_signal_source="learned_core",
            controller_signal_base_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        external_signal = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        learned = model(ids, controller_signal=external_signal)
        core_off = model(ids, controller_signal=external_signal, disable_core=True)
        signal_off = model(ids, controller_signal=external_signal, disable_controller_signal=True)

        self.assertEqual(learned["controller_signal_logits"].shape, (2, 2))
        self.assertEqual(learned["controller_signal_pred"].shape, (2, 2))
        self.assertEqual(float(learned["controller_signal_used"].item()), 1.0)
        self.assertEqual(float(signal_off["controller_signal_used"].item()), 0.0)
        self.assertTrue(torch.allclose(signal_off["controller_pooled"], torch.zeros_like(signal_off["controller_pooled"])))
        self.assertFalse(torch.equal(learned["controller_pooled"], core_off["controller_pooled"]))

    def test_learned_readout_controller_signal_uses_generation_readout(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
            controller_signal_enabled=True,
            controller_signal_dim=2,
            controller_signal_source="learned_readout",
            controller_signal_base_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        external_signal = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        learned = model(ids, controller_signal=external_signal)
        signal_off = model(ids, controller_signal=external_signal, disable_controller_signal=True)

        self.assertEqual(learned["controller_signal_logits"].shape, (2, 2))
        self.assertEqual(learned["controller_signal_pred"].shape, (2, 2))
        self.assertEqual(float(learned["controller_signal_used"].item()), 1.0)
        self.assertEqual(float(signal_off["controller_signal_used"].item()), 0.0)
        self.assertTrue(torch.allclose(signal_off["controller_pooled"], torch.zeros_like(signal_off["controller_pooled"])))
        self.assertFalse(torch.equal(learned["controller_pooled"], signal_off["controller_pooled"]))

    def test_learned_controller_signal_can_use_hidden_mlp_head(self):
        import torch
        from torch import nn
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            controller_signal_enabled=True,
            controller_signal_dim=3,
            controller_signal_source="learned_core",
            controller_signal_hidden_dim=8,
        )
        model = QTRMMultimodalModel(cfg)

        self.assertIsInstance(model.controller_signal_head, nn.Sequential)
        out = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(out["controller_signal_logits"].shape, (2, 3))
        self.assertEqual(out["controller_signal_pred"].shape, (2, 3))

    def test_learned_controller_signal_can_use_core_trajectory_source(self):
        import torch
        from torch import nn
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=3,
            visual_dim=16,
            max_visual_tokens=4,
            controller_signal_enabled=True,
            controller_signal_dim=4,
            controller_signal_source="learned_core_trajectory",
            controller_signal_hidden_dim=8,
        )
        model = QTRMMultimodalModel(cfg)

        self.assertIsInstance(model.controller_signal_head, nn.Sequential)
        self.assertEqual(model.controller_signal_head[0].in_features, 16 * 3)
        out = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
        signal_off = model(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            disable_controller_signal=True,
        )

        self.assertEqual(out["core_depth_states"].shape, (2, 3, 16))
        self.assertEqual(out["controller_signal_logits"].shape, (2, 4))
        self.assertEqual(out["controller_signal_pred"].shape, (2, 4))
        self.assertEqual(float(out["controller_signal_used"].item()), 1.0)
        self.assertEqual(float(signal_off["controller_signal_used"].item()), 0.0)

    def test_return_features_only_skips_vocab_logits_but_keeps_controller_features(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
        )
        model = QTRMMultimodalModel(cfg)
        out = model(torch.tensor([[1, 2, 3]]), return_features_only=True)

        self.assertIn("generation_verifier_pooled", out)
        self.assertIn("action_logits", out)
        self.assertNotIn("logits", out)
        self.assertNotIn("qtrm_logits", out)

    def test_bounded_residual_clamps_qtrm_contribution_before_donor_fusion(self):
        import torch
        from torch import nn
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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

    def test_donor_qtrm_conflict_gate_suppresses_residual_before_fusion(self):
        import torch
        from torch import nn
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        class FixedHead(nn.Module):
            def __init__(self, vocab_size: int, winning_token: int):
                super().__init__()
                self.vocab_size = vocab_size
                self.winning_token = winning_token

            def forward(self, hidden):
                logits = hidden.new_zeros((*hidden.shape[:2], self.vocab_size))
                logits[..., self.winning_token] = 5.0
                return logits

        cfg = QTRMConfig(
            vocab_size=16,
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
            donor_qtrm_conflict_gate_enabled=True,
            donor_qtrm_conflict_qtrm_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.lm_head = FixedHead(cfg.vocab_size, winning_token=2)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.zeros(1, 4, cfg.vocab_size)
        donor_logits[..., 3] = 7.0

        out = model(input_ids, donor_logits=donor_logits)
        offset = out["logits"].shape[1] - input_ids.shape[1]

        self.assertIn("donor_qtrm_conflict_gate", out)
        self.assertTrue(torch.equal(out["donor_qtrm_conflict_gate"], torch.zeros(1, 4)))
        self.assertTrue(torch.allclose(out["logits"][:, offset:], donor_logits, atol=1e-6))

    def test_donor_qtrm_conflict_gate_keeps_residual_when_top_tokens_agree(self):
        import torch
        from torch import nn
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        class FixedHead(nn.Module):
            def __init__(self, vocab_size: int, winning_token: int):
                super().__init__()
                self.vocab_size = vocab_size
                self.winning_token = winning_token

            def forward(self, hidden):
                logits = hidden.new_zeros((*hidden.shape[:2], self.vocab_size))
                logits[..., self.winning_token] = 5.0
                return logits

        cfg = QTRMConfig(
            vocab_size=16,
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
            donor_qtrm_conflict_gate_enabled=True,
            donor_qtrm_conflict_qtrm_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.lm_head = FixedHead(cfg.vocab_size, winning_token=2)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.zeros(1, 4, cfg.vocab_size)
        donor_logits[..., 2] = 7.0

        out = model(input_ids, donor_logits=donor_logits)
        offset = out["logits"].shape[1] - input_ids.shape[1]

        self.assertTrue(torch.equal(out["donor_qtrm_conflict_gate"], torch.ones(1, 4)))
        self.assertTrue(
            torch.allclose(
                out["logits"][:, offset:],
                donor_logits + out["qtrm_residual_logits"][:, offset:],
                atol=1e-6,
            )
        )

    def test_bounded_residual_gate_uses_stable_initial_gate(self):
        import math
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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

    def test_residual_gate_can_be_disabled_for_causal_ablation(self):
        import torch
        from torch import nn
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        class FixedHead(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.vocab_size = vocab_size

            def forward(self, hidden):
                logits = hidden.new_zeros((*hidden.shape[:2], self.vocab_size))
                logits[..., 2] = 4.0
                return logits

        cfg = QTRMConfig(
            vocab_size=16,
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
            qtrm_residual_gate_init_bias=-100.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.lm_head = FixedHead(cfg.vocab_size)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.zeros(1, 4, cfg.vocab_size)

        gated = model(input_ids, donor_logits=donor_logits)
        gate_off = model(
            input_ids,
            donor_logits=donor_logits,
            disable_qtrm_residual_gate=True,
        )
        offset = gated["logits"].shape[1] - input_ids.shape[1]

        self.assertTrue(torch.allclose(gated["qtrm_residual_gate"], torch.zeros(1)))
        self.assertTrue(torch.allclose(gated["logits"][:, offset:], donor_logits))
        self.assertTrue(torch.allclose(gate_off["qtrm_residual_gate"], torch.ones(1)))
        self.assertTrue(
            torch.allclose(
                gate_off["logits"][:, offset:],
                donor_logits + gate_off["qtrm_residual_logits"][:, offset:],
            )
        )

    def test_residual_gate_normalizes_large_latents_before_linear(self):
        import math
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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

    def test_zero_core_trajectory_preserves_depth_count_but_removes_state_content(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        out = model(input_ids, zero_core_trajectory=True)

        self.assertEqual(int(out["trajectory_len"].item()), cfg.outer_steps)
        self.assertTrue(torch.allclose(out["core_depth_states"], torch.zeros_like(out["core_depth_states"])))

    def test_core_to_text_injection_directly_conditions_text_logits_without_coda_attention(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_to_text_enabled=True,
            core_to_text_gate_init_bias=8.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        with_core_injection = model(input_ids)
        without_core_injection = model(input_ids, disable_core_to_text=True)
        offset = with_core_injection["logits"].shape[1] - input_ids.shape[1]

        self.assertIn("core_to_text_gate_mean", with_core_injection)
        self.assertEqual(with_core_injection["core_to_text_gate_mean"].shape, (2,))
        self.assertFalse(
            torch.allclose(
                with_core_injection["logits"][:, offset:],
                without_core_injection["logits"][:, offset:],
            )
        )

    def test_residual_head_ablation_zeros_qtrm_contribution_before_donor_fusion(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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

    def test_workspace_text_states_are_not_exposed_as_direct_coda_tokens(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

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
            max_visual_tokens=6,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        workspace_text_states = torch.randn(1, 6, cfg.visual_dim)
        workspace_attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

        without_memory = model(input_ids)
        with_memory = model(
            input_ids,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
        )
        memory_off = model(
            input_ids,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
            disable_workspace_memory_context=True,
        )

        self.assertEqual(with_memory["logits"].shape[1], cfg.workspace_tokens + input_ids.shape[1])
        self.assertEqual(with_memory["workspace_memory_token_count"], 4)
        self.assertEqual(memory_off["workspace_memory_token_count"], 0)
        self.assertFalse(torch.allclose(with_memory["z_h"], without_memory["z_h"]))
        self.assertTrue(torch.allclose(memory_off["z_h"], without_memory["z_h"], atol=1e-5))

    def test_core_context_injection_is_gated_and_ablatable(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            max_visual_tokens=6,
            core_context_enabled=True,
            core_context_gate_init_bias=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        workspace_text_states = torch.randn(1, 6, cfg.visual_dim)
        workspace_attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

        with_context = model(
            input_ids,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
        )
        context_off = model(
            input_ids,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
            disable_core_context=True,
        )

        self.assertEqual(with_context["core_context_gate_mean"].shape, (1, 2))
        self.assertEqual(context_off["core_context_gate_mean"].shape, (1, 0))
        self.assertFalse(torch.allclose(with_context["z_h"], context_off["z_h"]))

    def test_evidence_bottleneck_suppresses_residual_without_workspace_memory(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            qtrm_residual_gate_enabled=False,
            evidence_bottleneck_enabled=True,
            evidence_bottleneck_gate_init_bias=0.0,
            evidence_bottleneck_suppress_without_workspace=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        no_memory = model(input_ids, donor_logits=donor_logits)
        no_memory_bypass = model(
            input_ids,
            donor_logits=donor_logits,
            disable_evidence_bottleneck=True,
        )
        workspace_memory = model(
            input_ids,
            donor_logits=donor_logits,
            workspace_text_states=torch.randn(1, 4, cfg.visual_dim),
            workspace_attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )

        offset = no_memory["logits"].shape[1] - input_ids.shape[1]
        self.assertTrue(torch.allclose(no_memory["evidence_bottleneck_gate"], torch.zeros(1)))
        self.assertTrue(torch.allclose(no_memory["logits"][:, offset:], donor_logits, atol=1e-5))
        self.assertGreater(float(no_memory_bypass["qtrm_residual_logits"].abs().max()), 0.0)
        self.assertTrue(torch.equal(workspace_memory["workspace_memory_present"], torch.ones(1)))
        self.assertAlmostEqual(float(workspace_memory["evidence_bottleneck_gate"].item()), 0.5, places=5)
        self.assertGreater(float(workspace_memory["qtrm_residual_logits"].abs().max()), 0.0)

    def test_evidence_bottleneck_can_be_verifier_only_without_suppressing_general_qtrm(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            qtrm_residual_gate_enabled=False,
            evidence_bottleneck_enabled=True,
            evidence_bottleneck_gate_init_bias=0.0,
            evidence_bottleneck_suppress_without_workspace=True,
            evidence_bottleneck_applies_to_residual=False,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        out = model(input_ids, donor_logits=donor_logits)

        offset = out["logits"].shape[1] - input_ids.shape[1]
        self.assertTrue(torch.allclose(out["evidence_bottleneck_gate"], torch.zeros(1)))
        self.assertGreater(float(out["qtrm_residual_logits"].abs().max()), 0.0)
        self.assertFalse(torch.allclose(out["logits"][:, offset:], donor_logits, atol=1e-5))

    def test_answer_bottleneck_replaces_residual_with_workspace_conditioned_logits(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            answer_bottleneck_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)
        workspace_text_states = torch.randn(1, 4, cfg.visual_dim)
        workspace_attention_mask = torch.tensor([[1, 1, 1, 0]])

        out = model(
            input_ids,
            donor_logits=donor_logits,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
        )
        workspace_off = model(
            input_ids,
            donor_logits=donor_logits,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
            disable_workspace=True,
        )

        offset = out["logits"].shape[1] - input_ids.shape[1]
        self.assertEqual(out["answer_bottleneck_logits"].shape, (1, 4, cfg.vocab_size))
        self.assertTrue(torch.allclose(out["qtrm_residual_logits"][:, :offset], torch.zeros_like(out["qtrm_residual_logits"][:, :offset])))
        self.assertTrue(torch.allclose(out["qtrm_residual_logits"][:, offset:], out["answer_bottleneck_logits"], atol=1e-6))
        self.assertGreater(float(out["answer_bottleneck_logits"].abs().max()), 0.0)
        self.assertTrue(torch.allclose(workspace_off["qtrm_residual_logits"], torch.zeros_like(workspace_off["qtrm_residual_logits"]), atol=1e-6))

    def test_answer_bottleneck_does_not_drive_residual_without_workspace_memory(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            answer_bottleneck_enabled=True,
            answer_bottleneck_requires_workspace_memory=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        out = model(input_ids, donor_logits=donor_logits)

        offset = out["logits"].shape[1] - input_ids.shape[1]
        self.assertTrue(torch.allclose(out["workspace_memory_present"], torch.zeros(1)))
        self.assertGreater(float(out["answer_bottleneck_logits"].abs().max()), 0.0)
        self.assertTrue(torch.allclose(out["qtrm_residual_logits"], torch.zeros_like(out["qtrm_residual_logits"]), atol=1e-6))
        self.assertTrue(torch.allclose(out["logits"][:, offset:], donor_logits, atol=1e-5))

    def test_answer_residual_governor_can_close_answer_bottleneck_residual(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
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
            answer_bottleneck_enabled=True,
            answer_bottleneck_requires_workspace_memory=False,
            answer_residual_governor_enabled=True,
            answer_residual_governor_init_bias=-100.0,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        donor_logits = torch.randn(1, 4, cfg.vocab_size)

        governed = model(input_ids, donor_logits=donor_logits)
        governor_off = model(
            input_ids,
            donor_logits=donor_logits,
            disable_answer_residual_governor=True,
        )
        offset = governed["logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(governed["answer_residual_governor_logits"].shape, (1, 4))
        self.assertTrue(torch.allclose(governed["answer_residual_governor_gate"], torch.zeros(1, 4), atol=1e-6))
        self.assertGreater(float(governed["answer_bottleneck_logits"].abs().max()), 0.0)
        self.assertTrue(torch.allclose(governed["qtrm_residual_logits"], torch.zeros_like(governed["qtrm_residual_logits"]), atol=1e-6))
        self.assertTrue(torch.allclose(governed["logits"][:, offset:], donor_logits, atol=1e-5))
        self.assertGreater(float(governor_off["qtrm_residual_logits"].abs().max()), 0.0)

    def test_evidence_span_reader_scores_workspace_tokens_with_prompt_query(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            evidence_span_reader_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)
        workspace_text_states = torch.randn(1, 4, cfg.visual_dim)
        workspace_attention_mask = torch.tensor([[1, 1, 0, 1]])

        out = model(
            input_ids,
            text_states=text_states,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
        )
        changed_prompt = model(
            torch.tensor([[1, 5, 6, 7]]),
            text_states=text_states,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
        )
        workspace_off = model(
            input_ids,
            text_states=text_states,
            workspace_text_states=workspace_text_states,
            workspace_attention_mask=workspace_attention_mask,
            disable_workspace=True,
        )

        self.assertEqual(out["evidence_span_start_logits"].shape, (1, 4))
        self.assertEqual(out["evidence_span_end_logits"].shape, (1, 4))
        self.assertLess(float(out["evidence_span_start_logits"][0, 2]), -999.0)
        self.assertFalse(torch.allclose(
            out["evidence_span_start_logits"],
            changed_prompt["evidence_span_start_logits"],
        ))
        self.assertEqual(workspace_off["evidence_span_start_logits"].shape, (1, 0))

    def test_evidence_span_reader_can_score_ssot_input_tokens_without_workspace_memory(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            evidence_span_reader_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])

        default_out = model(input_ids)
        ssot_out = model(input_ids, evidence_span_reader_context="input")

        self.assertEqual(default_out["evidence_span_start_logits"].shape, (1, 0))
        self.assertEqual(ssot_out["evidence_span_start_logits"].shape, (1, 4))
        self.assertEqual(ssot_out["evidence_span_end_logits"].shape, (1, 4))

    def test_evidence_span_no_answer_logit_is_conditioned_on_workspace_memory(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            evidence_span_reader_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        text_states = torch.randn(1, 4, cfg.visual_dim)
        workspace_a = torch.randn(1, 4, cfg.visual_dim)
        workspace_b = torch.randn(1, 4, cfg.visual_dim)
        workspace_attention_mask = torch.tensor([[1, 1, 1, 1]])

        out_a = model(
            input_ids,
            text_states=text_states,
            workspace_text_states=workspace_a,
            workspace_attention_mask=workspace_attention_mask,
        )
        out_b = model(
            input_ids,
            text_states=text_states,
            workspace_text_states=workspace_b,
            workspace_attention_mask=workspace_attention_mask,
        )

        self.assertFalse(torch.allclose(
            out_a["evidence_span_no_answer_logits"],
            out_b["evidence_span_no_answer_logits"],
        ))

    def test_answer_decision_head_is_ablatable_model_output(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            answer_decision_head_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4], [1, 4, 5, 6]])

        out = model(input_ids)
        ablated = model(input_ids, disable_answer_decision_head=True)

        self.assertEqual(out["answer_decision_logits"].shape, (2,))
        self.assertEqual(ablated["answer_decision_logits"].shape, (2, 0))
        self.assertFalse(torch.allclose(out["answer_decision_logits"], torch.zeros_like(out["answer_decision_logits"])))

    def test_answer_decision_head_can_condition_on_telemetry_features(self):
        import torch
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(0)
        cfg = QTRMConfig(
            vocab_size=64,
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
            answer_decision_head_enabled=True,
            answer_decision_feature_dim=3,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3, 4], [1, 4, 5, 6]])

        self.assertIsInstance(model.answer_decision_feature_head[0], torch.nn.Linear)
        low = model(
            input_ids,
            answer_decision_features=torch.zeros(2, 3),
        )
        high = model(
            input_ids,
            answer_decision_features=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
        )
        features_off = model(
            input_ids,
            answer_decision_features=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            disable_answer_decision_features=True,
        )

        self.assertEqual(low["answer_decision_logits"].shape, (2,))
        self.assertEqual(high["answer_decision_feature_logits"].shape, (2,))
        self.assertTrue(torch.allclose(
            high["answer_decision_logits"],
            high["answer_decision_feature_logits"],
        ))
        self.assertFalse(torch.allclose(
            low["answer_decision_logits"],
            high["answer_decision_logits"],
        ))
        self.assertFalse(torch.allclose(
            high["answer_decision_logits"],
            features_off["answer_decision_logits"],
        ))


if __name__ == "__main__":
    unittest.main()
