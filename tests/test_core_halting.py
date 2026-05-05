import unittest


class CoreHaltingTests(unittest.TestCase):
    def _cfg(self):
        from qtrm_mm import QTRMConfig

        return QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=4,
            visual_dim=16,
            max_visual_tokens=4,
            core_halt_enabled=True,
            core_halt_min_steps=1,
            core_halt_use_continue=False,
        )

    def test_core_halt_head_can_stop_latent_loop_early_when_enabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core.halt_head.weight.zero_()
            model.core.halt_head.bias.copy_(torch.tensor([2.0, -2.0]))

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=True,
        )

        self.assertEqual(int(out["trajectory_len"].item()), 1)
        self.assertEqual(out["core_q_halt_logits"].shape, (2, 1))
        self.assertEqual(out["core_q_continue_logits"].shape, (2, 1))
        self.assertTrue(torch.equal(out["core_halted"], torch.tensor([True, True])))
        self.assertTrue(torch.equal(out["core_steps"], torch.tensor([1, 1])))

    def test_core_halt_head_is_telemetry_only_when_halt_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core.halt_head.weight.zero_()
            model.core.halt_head.bias.copy_(torch.tensor([2.0, -2.0]))

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(int(out["trajectory_len"].item()), cfg.outer_steps)
        self.assertEqual(out["core_q_halt_logits"].shape, (2, cfg.outer_steps))
        self.assertEqual(out["core_q_continue_logits"].shape, (2, cfg.outer_steps))
        self.assertTrue(torch.equal(out["core_halted"], torch.tensor([False, False])))
        self.assertTrue(torch.equal(out["core_steps"], torch.tensor([cfg.outer_steps, cfg.outer_steps])))

    def test_model_exposes_per_outer_step_core_depth_states_for_teacher_targets(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(out["core_depth_states"].shape, (2, cfg.outer_steps, cfg.d_model))

    def test_model_can_expose_per_outer_step_last_token_logits_for_teacher_targets(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            return_core_depth_logits=True,
        )

        self.assertEqual(out["core_depth_last_logits"].shape, (2, cfg.outer_steps, cfg.vocab_size))

    def test_model_can_expose_per_outer_step_text_logits_for_full_answer_targets(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        out = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
        )

        self.assertEqual(
            out["core_depth_text_logits"].shape,
            (2, cfg.outer_steps, input_ids.shape[1], cfg.vocab_size),
        )

    def test_model_exposes_primitive_transition_operation_logits_from_core_depth_states(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 9
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["primitive_transition_operation_logits"].shape,
            (2, cfg.outer_steps, cfg.primitive_transition_num_operations),
        )

    def test_primitive_transition_operation_logits_are_zero_when_core_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 9
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            disable_core=True,
        )

        self.assertEqual(
            out["primitive_transition_operation_logits"].shape,
            (2, 0, cfg.primitive_transition_num_operations),
        )
        self.assertEqual(
            float(out["primitive_transition_operation_logits"].abs().sum()),
            0.0,
        )

    def test_prompt_context_can_condition_primitive_transition_operation_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 9
        cfg.primitive_transition_prompt_context_enabled = True
        model = QTRMMultimodalModel(cfg)

        core_depth_states = torch.zeros(2, cfg.outer_steps, cfg.d_model)
        core_depth_states[:, :, 0] = 1.0
        prompt_context_seq = torch.zeros(2, 3, cfg.d_model)
        prompt_context_seq[1] = torch.arange(cfg.d_model, dtype=torch.float32)
        prompt_context_mask = torch.ones(2, 3)

        out = model._compute_primitive_transition_outputs(
            core_depth_states,
            prompt_context_seq=prompt_context_seq,
            prompt_context_mask=prompt_context_mask,
        )

        self.assertFalse(
            torch.allclose(
                out["operation_logits"][0],
                out["operation_logits"][1],
            )
        )

    def test_prompt_token_attention_can_distinguish_same_mean_prompt_contexts(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 9
        cfg.primitive_transition_prompt_context_enabled = True
        cfg.primitive_transition_prompt_token_attention_enabled = True
        model = QTRMMultimodalModel(cfg)

        core_depth_states = torch.zeros(2, cfg.outer_steps, cfg.d_model)
        prompt_context_seq = torch.zeros(2, 2, cfg.d_model)
        prompt_context_seq[0, 0, 0] = 2.0
        prompt_context_seq[0, 1, 1] = 2.0
        prompt_context_seq[1, 0, 0] = 1.0
        prompt_context_seq[1, 0, 1] = 1.0
        prompt_context_seq[1, 1, 0] = 1.0
        prompt_context_seq[1, 1, 1] = 1.0
        prompt_context_mask = torch.ones(2, 2)

        self.assertTrue(
            torch.allclose(
                prompt_context_seq[0].mean(dim=0),
                prompt_context_seq[1].mean(dim=0),
            )
        )

        out = model._compute_primitive_transition_outputs(
            core_depth_states,
            prompt_context_seq=prompt_context_seq,
            prompt_context_mask=prompt_context_mask,
        )

        self.assertFalse(
            torch.allclose(
                out["operation_logits"][0],
                out["operation_logits"][1],
            )
        )

    def test_core_depth_last_token_logits_do_not_include_depth_invariant_donor_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.donor_logits_scale = 1.0
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        donor_logits = torch.randn(2, 6, cfg.vocab_size)

        without_donor = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_logits=True,
        )["core_depth_last_logits"]
        with_donor = model(
            input_ids,
            donor_logits=donor_logits,
            enable_core_halt=False,
            return_core_depth_logits=True,
        )["core_depth_last_logits"]

        self.assertTrue(torch.allclose(with_donor, without_donor))

    def test_core_step_conditioning_can_make_depth_states_step_specific(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.use_stable_inject = False
        cfg.core_step_conditioning_enabled = True
        cfg.core_step_conditioning_max_steps = 8
        cfg.core_step_conditioning_scale = 1.0
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core.step_conditioning.weight.zero_()
            model.core.step_conditioning.weight[0, 0] = 1.0
            model.core.step_conditioning.weight[1, 0] = -1.0

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(out["core_depth_states"].shape[1], cfg.outer_steps)
        self.assertFalse(
            torch.allclose(
                out["core_depth_states"][:, 0, :],
                out["core_depth_states"][:, 1, :],
            )
        )

    def test_answer_bottleneck_residual_is_final_logit_path_without_donor(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.answer_bottleneck_enabled = True
        cfg.answer_bottleneck_requires_core = True
        cfg.answer_bottleneck_requires_workspace_memory = False
        model = QTRMMultimodalModel(cfg)

        out = model(torch.randint(0, cfg.vocab_size, (2, 6)))

        self.assertEqual(out["answer_bottleneck_logits"].shape, (2, 6, cfg.vocab_size))
        self.assertTrue(torch.allclose(out["logits"], out["qtrm_residual_logits"]))
        self.assertFalse(torch.allclose(out["logits"], out["qtrm_logits"]))

    def test_core_loop_readout_is_final_logit_path_without_donor(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.core_loop_readout_enabled = True
        cfg.core_loop_readout_requires_core = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids)
        text_offset = out["logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(out["core_loop_readout_logits"].shape, (2, 6, cfg.vocab_size))
        self.assertEqual(out["core_loop_readout_hidden"].shape, (2, 6, cfg.d_model))
        self.assertTrue(
            torch.allclose(
                out["logits"][:, text_offset:, :],
                out["core_loop_readout_logits"],
            )
        )
        self.assertTrue(torch.allclose(out["logits"], out["qtrm_residual_logits"]))

    def test_core_loop_readout_requires_core_when_core_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.core_loop_readout_enabled = True
        cfg.core_loop_readout_requires_core = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, disable_core=True)

        self.assertTrue(torch.equal(out["core_loop_readout_logits"], torch.zeros_like(out["core_loop_readout_logits"])))
        self.assertTrue(torch.equal(out["qtrm_residual_logits"], torch.zeros_like(out["qtrm_residual_logits"])))
        self.assertTrue(torch.equal(out["logits"], torch.zeros_like(out["logits"])))

    def test_answer_state_loop_is_final_logit_path_without_donor(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids)
        text_offset = out["logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(out["answer_state_loop_logits"].shape, (2, 6, cfg.vocab_size))
        self.assertEqual(out["answer_state_loop_hidden"].shape, (2, 6, cfg.d_model))
        self.assertEqual(
            out["answer_state_loop_depth_hidden"].shape,
            (2, cfg.outer_steps, 6, cfg.d_model),
        )
        self.assertTrue(
            torch.allclose(
                out["logits"][:, text_offset:, :],
                out["answer_state_loop_logits"],
            )
        )
        self.assertTrue(torch.allclose(out["logits"], out["qtrm_residual_logits"]))

    def test_answer_state_loop_requires_core_when_core_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, disable_core=True)

        self.assertTrue(torch.equal(out["answer_state_loop_logits"], torch.zeros_like(out["answer_state_loop_logits"])))
        self.assertTrue(torch.equal(out["answer_state_loop_hidden"], torch.zeros_like(out["answer_state_loop_hidden"])))
        self.assertTrue(torch.equal(out["qtrm_residual_logits"], torch.zeros_like(out["qtrm_residual_logits"])))
        self.assertTrue(torch.equal(out["logits"], torch.zeros_like(out["logits"])))

    def test_transition_state_core_exposes_state_and_feeds_answer_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        torch.manual_seed(7)
        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.transition_state_enabled = True
        cfg.transition_state_dim = 3
        cfg.transition_state_hidden_dim = 16
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids)
        state_off = model(input_ids, disable_transition_state=True)
        text_offset = full["logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(
            full["transition_state_logits"].shape,
            (2, cfg.outer_steps, cfg.transition_state_dim),
        )
        self.assertEqual(
            full["transition_state_features"].shape,
            (2, cfg.outer_steps, cfg.transition_state_dim),
        )
        self.assertEqual(
            full["transition_state_text_logits"].shape,
            (2, cfg.outer_steps, cfg.vocab_size),
        )
        self.assertEqual(
            state_off["transition_state_features"].shape,
            full["transition_state_features"].shape,
        )
        self.assertTrue(
            torch.equal(
                state_off["transition_state_features"],
                torch.zeros_like(state_off["transition_state_features"]),
            )
        )
        self.assertTrue(
            torch.equal(
                state_off["transition_state_text_logits"],
                torch.zeros_like(state_off["transition_state_text_logits"]),
            )
        )
        self.assertFalse(
            torch.allclose(
                full["logits"][:, text_offset:, :],
                state_off["logits"][:, text_offset:, :],
            )
        )

    def test_transition_state_code_path_exposes_code_and_feeds_answer_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        torch.manual_seed(11)
        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.transition_state_code_enabled = True
        cfg.transition_state_codebook_size = 13
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids)
        state_off = model(input_ids, disable_transition_state=True)
        text_offset = full["logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(
            full["transition_state_code_logits"].shape,
            (2, cfg.outer_steps, cfg.transition_state_codebook_size),
        )
        self.assertEqual(
            full["transition_state_code_embeddings"].shape,
            (2, cfg.outer_steps, cfg.d_model),
        )
        self.assertTrue(
            torch.equal(
                state_off["transition_state_code_embeddings"],
                torch.zeros_like(state_off["transition_state_code_embeddings"]),
            )
        )
        self.assertFalse(
            torch.allclose(
                full["logits"][:, text_offset:, :],
                state_off["logits"][:, text_offset:, :],
            )
        )

    def test_transition_state_finality_head_exposes_ablatable_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        torch.manual_seed(17)
        cfg = self._cfg()
        cfg.transition_state_finality_enabled = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids)
        state_off = model(input_ids, disable_transition_state=True)

        self.assertEqual(
            full["transition_state_finality_logits"].shape,
            (2, cfg.outer_steps),
        )
        self.assertTrue(
            torch.equal(
                state_off["transition_state_finality_logits"],
                torch.zeros_like(state_off["transition_state_finality_logits"]),
            )
        )

    def test_transition_state_code_only_loop_uses_code_as_answer_state(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(13)
        self.assertFalse(QTRMConfig().transition_state_code_only_answer_loop)
        cfg = self._cfg()
        cfg.core_halt_enabled = False
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.transition_state_code_enabled = True
        cfg.transition_state_codebook_size = 7
        cfg.transition_state_code_only_answer_loop = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids)
        state_off = model(input_ids, disable_transition_state=True)
        normal_cfg = self._cfg()
        normal_cfg.core_halt_enabled = False
        normal_cfg.answer_state_loop_enabled = True
        normal_cfg.answer_state_loop_requires_core = True
        normal_cfg.transition_state_code_enabled = True
        normal_cfg.transition_state_codebook_size = 7
        normal_model = QTRMMultimodalModel(normal_cfg)
        normal_model.load_state_dict(model.state_dict(), strict=False)
        normal = normal_model(input_ids)
        text_offset = full["logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(
            full["transition_state_code_embeddings"].shape,
            (2, cfg.outer_steps, cfg.d_model),
        )
        self.assertFalse(
            torch.allclose(
                full["logits"][:, text_offset:, :],
                state_off["logits"][:, text_offset:, :],
            )
        )
        self.assertFalse(
            torch.allclose(
                full["logits"][:, text_offset:, :],
                normal["logits"][:, text_offset:, :],
            )
        )


if __name__ == "__main__":
    unittest.main()
