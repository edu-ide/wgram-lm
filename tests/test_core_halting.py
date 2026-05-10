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

    def test_model_accepts_token_numeric_value_ids_when_enabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.token_numeric_value_embedding_enabled = True
        cfg.token_numeric_value_vocab_size = 32
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        token_numeric_value_ids = torch.zeros_like(input_ids)
        token_numeric_value_ids[:, 2] = 7

        out = model(
            input_ids,
            token_numeric_value_ids=token_numeric_value_ids,
        )

        self.assertIsNotNone(model.token_numeric_value_embed)
        self.assertIsNotNone(model.token_numeric_value_gate)
        self.assertLess(float(torch.sigmoid(model.token_numeric_value_gate).item()), 0.05)
        self.assertEqual(out["logits"].shape[0], input_ids.shape[0])
        self.assertEqual(out["logits"].shape[-1], cfg.vocab_size)

    def test_model_accepts_token_numeric_source_slots_when_enabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.token_numeric_source_slot_embedding_enabled = True
        cfg.token_numeric_source_slot_vocab_size = 32
        cfg.token_numeric_source_slot_max_slots = 4
        cfg.token_numeric_source_slot_predicate_feedback_enabled = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        source_slot_ids = torch.tensor([[3, 7, 0, 0], [2, 4, 6, 0]])
        source_slot_mask = (source_slot_ids != 0).long()

        out = model(
            input_ids,
            token_numeric_source_slot_ids=source_slot_ids,
            token_numeric_source_slot_mask=source_slot_mask,
        )
        off = model(
            input_ids,
            token_numeric_source_slot_ids=source_slot_ids,
            token_numeric_source_slot_mask=source_slot_mask,
            disable_token_numeric_source_slots=True,
        )

        self.assertIsNotNone(model.token_numeric_source_slot_embed)
        self.assertIsNotNone(model.token_numeric_source_slot_gate)
        self.assertEqual(out["token_numeric_source_slot_token_count"], 4)
        self.assertEqual(out["token_numeric_source_slot_parity_logits"].shape, (2, 4, 2))
        self.assertEqual(out["token_numeric_source_slot_predicate_logits"].shape, (2, 4, 2))
        self.assertEqual(off["token_numeric_source_slot_token_count"], 0)
        self.assertEqual(off["token_numeric_source_slot_parity_logits"].shape, (2, 0, 2))
        self.assertEqual(off["token_numeric_source_slot_predicate_logits"].shape, (2, 0, 2))
        self.assertEqual(out["logits"].shape[0], input_ids.shape[0])
        self.assertEqual(out["logits"].shape[-1], cfg.vocab_size)

    def test_core_source_position_binder_initializes_prompt_state_from_tokens(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 5
        cfg.core_role_value_state_vocab_size = 9
        cfg.core_source_position_binder_enabled = True
        cfg.core_source_position_binder_layers = 1
        cfg.core_source_position_binder_heads = 4
        cfg.core_source_position_binder_gate_min = 1.0
        cfg.core_source_position_binder_state_gate_min = 1.0
        cfg.core_source_position_binder_state_straight_through = True
        cfg.core_source_position_binder_query_state_enabled = True
        cfg.core_source_position_binder_query_state_gate_min = 1.0
        cfg.core_source_value_binder_enabled = True
        cfg.core_source_value_binder_state_gate_min = 1.0
        cfg.core_source_value_binder_state_straight_through = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        full = model(input_ids)
        binder_off = model(input_ids, disable_core_source_position_binder=True)
        query_state_off = model(
            input_ids,
            disable_core_source_position_binder_query_state=True,
        )
        source_value_off = model(input_ids, disable_core_source_value_binder=True)

        self.assertIsNotNone(model.core_source_position_binder_slot_queries)
        self.assertIsNotNone(model.core_source_position_binder_logit_gate)
        self.assertIsNotNone(model.core_source_position_binder_value_embed)
        self.assertIsNotNone(model.core_source_position_binder_state_gate)
        self.assertIsNotNone(model.core_source_position_binder_query_state_proj)
        self.assertIsNotNone(model.core_source_position_binder_query_state_gate)
        self.assertIsNotNone(model.core_source_value_binder_head)
        self.assertIsNotNone(model.core_source_value_binder_state_gate)
        self.assertEqual(
            full["core_role_value_state_prompt_logits"].shape,
            (
                2,
                1,
                cfg.core_role_value_state_num_roles,
                cfg.core_role_value_state_vocab_size,
            ),
        )
        self.assertEqual(
            binder_off["core_role_value_state_prompt_logits"].shape,
            (2, 0, cfg.core_role_value_state_num_roles, cfg.core_role_value_state_vocab_size),
        )
        self.assertEqual(
            full["core_source_value_prompt_logits"].shape,
            (
                2,
                1,
                cfg.core_role_value_state_num_roles,
                cfg.core_role_value_state_vocab_size,
            ),
        )
        self.assertEqual(
            full["core_role_value_state_logits"].shape,
            binder_off["core_role_value_state_logits"].shape,
        )
        self.assertFalse(
            torch.allclose(
                full["core_role_value_state_logits"],
                binder_off["core_role_value_state_logits"],
            )
        )
        self.assertFalse(
            torch.allclose(
                full["core_role_value_state_logits"],
                query_state_off["core_role_value_state_logits"],
            )
        )
        self.assertFalse(
            torch.allclose(
                full["core_role_value_state_logits"],
                source_value_off["core_role_value_state_logits"],
            )
        )

    def test_core_source_position_binder_can_be_forced_to_source_slots(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.token_numeric_source_slot_embedding_enabled = True
        cfg.token_numeric_source_slot_vocab_size = 32
        cfg.token_numeric_source_slot_max_slots = 4
        cfg.token_numeric_source_slot_gate_min = 1.0
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 5
        cfg.core_role_value_state_vocab_size = 9
        cfg.core_source_position_binder_enabled = True
        cfg.core_source_position_binder_source_slots_only = True
        cfg.core_source_position_binder_gate_min = 1.0
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))
        source_slot_ids = torch.tensor([[3, 7, 0, 0], [2, 4, 6, 0]])
        source_slot_mask = (source_slot_ids != 0).long()

        full = model(
            input_ids,
            token_numeric_source_slot_ids=source_slot_ids,
            token_numeric_source_slot_mask=source_slot_mask,
        )
        source_slots_off = model(
            input_ids,
            token_numeric_source_slot_ids=source_slot_ids,
            token_numeric_source_slot_mask=source_slot_mask,
            disable_token_numeric_source_slots=True,
        )

        self.assertEqual(full["token_numeric_source_slot_token_count"], 4)
        self.assertEqual(source_slots_off["token_numeric_source_slot_token_count"], 0)
        self.assertEqual(
            full["core_role_value_state_prompt_logits"].shape,
            (
                2,
                1,
                cfg.core_role_value_state_num_roles,
                cfg.core_role_value_state_vocab_size,
            ),
        )
        self.assertEqual(
            source_slots_off["core_role_value_state_prompt_logits"].shape,
            (2, 0, cfg.core_role_value_state_num_roles, cfg.core_role_value_state_vocab_size),
        )

    def test_core_source_position_binder_can_read_raw_source_slots_before_prelude(self):
        import torch
        from torch import nn
        from qtrm_mm import QTRMMultimodalModel

        class AddLargeConstant(nn.Module):
            def forward(self, seq, attention_mask=None):
                return seq + 100.0

        cfg = self._cfg()
        cfg.token_numeric_source_slot_embedding_enabled = True
        cfg.token_numeric_source_slot_vocab_size = 32
        cfg.token_numeric_source_slot_max_slots = 4
        cfg.token_numeric_source_slot_gate_min = 1.0
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 5
        cfg.core_role_value_state_vocab_size = 9
        cfg.core_source_position_binder_enabled = True
        cfg.core_source_position_binder_source_slots_only = True
        cfg.core_source_position_binder_raw_source_slots_enabled = True
        cfg.core_source_position_binder_gate_min = 1.0
        model = QTRMMultimodalModel(cfg)
        model.prelude = AddLargeConstant()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))
        source_slot_ids = torch.tensor([[3, 7, 0, 0], [2, 4, 6, 0]])
        source_slot_mask = (source_slot_ids != 0).long()
        captured = []
        original = model._compute_core_source_position_binder_context

        def capture_context(prompt_context_seq, prompt_context_mask, **kwargs):
            if prompt_context_seq is not None:
                captured.append(prompt_context_seq.detach().clone())
            return original(prompt_context_seq, prompt_context_mask, **kwargs)

        model._compute_core_source_position_binder_context = capture_context

        model(
            input_ids,
            token_numeric_source_slot_ids=source_slot_ids,
            token_numeric_source_slot_mask=source_slot_mask,
        )

        self.assertTrue(captured)
        self.assertLess(float(captured[0].abs().max()), 10.0)

    def test_model_accepts_core_primitive_prompt_context_ablation(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 4
        cfg.primitive_transition_prompt_context_enabled = True
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 5
        cfg.core_role_value_state_vocab_size = 9
        cfg.core_primitive_role_value_executor_enabled = True
        cfg.core_primitive_role_value_mlp_enabled = True
        cfg.core_primitive_role_value_prompt_context_enabled = True
        cfg.core_primitive_role_value_prompt_token_attention_enabled = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        out = model(input_ids, disable_core_primitive_prompt_context=True)

        self.assertEqual(out["logits"].shape[0], input_ids.shape[0])
        self.assertEqual(out["logits"].shape[-1], cfg.vocab_size)

    def test_core_halt_head_is_telemetry_only_when_halt_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core.halt_head.weight.zero_()
            model.core.halt_head.bias.copy_(torch.tensor([2.0, -2.0]))

        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        out = model(input_ids, enable_core_halt=False)

        self.assertEqual(int(out["trajectory_len"].item()), cfg.outer_steps)
        self.assertEqual(out["core_q_halt_logits"].shape, (2, cfg.outer_steps))
        self.assertEqual(out["core_q_continue_logits"].shape, (2, cfg.outer_steps))
        self.assertTrue(torch.equal(out["core_halted"], torch.tensor([False, False])))
        self.assertTrue(torch.equal(out["core_steps"], torch.tensor([cfg.outer_steps, cfg.outer_steps])))

    def test_core_halt_head_uses_trm_conservative_initialization(self):
        import torch
        from qtrm_mm.core import QTRMRecursiveCore

        cfg = self._cfg()
        model = QTRMRecursiveCore(cfg)

        self.assertTrue(torch.allclose(model.halt_head.weight, torch.zeros_like(model.halt_head.weight)))
        self.assertTrue(
            torch.allclose(
                model.halt_head.bias,
                torch.full_like(model.halt_head.bias, float(cfg.core_halt_init_bias)),
            )
        )

    def test_core_transition_feedback_returns_step_logits(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
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
            outer_steps=3,
            visual_dim=16,
            max_visual_tokens=4,
            core_transition_feedback_enabled=True,
            core_transition_feedback_num_operations=7,
        )
        model = QTRMMultimodalModel(cfg)

        out = model(torch.randint(0, cfg.vocab_size, (2, 6)))

        self.assertEqual(
            out["core_transition_feedback_operation_logits"].shape,
            (2, cfg.outer_steps, 7),
        )
        self.assertEqual(
            out["core_transition_feedback_finality_logits"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["core_transition_feedback_gate_mean"].shape,
            (2, cfg.outer_steps),
        )

    def test_core_transition_order_bottleneck_adds_prompt_token_before_core(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
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
            outer_steps=3,
            visual_dim=16,
            max_visual_tokens=4,
            core_transition_order_bottleneck_enabled=True,
            core_transition_order_bottleneck_num_classes=2,
        )
        model = QTRMMultimodalModel(cfg)

        out = model(torch.randint(0, cfg.vocab_size, (2, 6)))

        self.assertEqual(
            out["core_transition_order_bottleneck_logits"].shape,
            (2, 1, 2),
        )
        self.assertEqual(
            out["core_transition_order_bottleneck_gate_mean"].shape,
            (2, 1),
        )
        self.assertEqual(out["z_h"].shape[1], cfg.workspace_tokens + 1)

    def test_core_transition_order_step_conditioning_changes_core_state(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMRecursiveCore

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
            core_transition_order_step_conditioning_enabled=True,
        )
        core = QTRMRecursiveCore(cfg)
        workspace = torch.randn(2, cfg.workspace_tokens, cfg.d_model)
        zeros = torch.zeros(2, 1, cfg.d_model)
        ones = torch.ones(2, 1, cfg.d_model)

        _, z_h_zeros, _, _ = core(
            workspace,
            transition_order_conditioning=zeros,
        )
        _, z_h_ones, _, _ = core(
            workspace,
            transition_order_conditioning=ones,
        )

        self.assertFalse(torch.allclose(z_h_zeros, z_h_ones))

    def test_core_trm_no_grad_inner_cycles_only_backpropagates_last_h_cycle(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMRecursiveCore

        class RecordingStack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled = []

            def forward(self, x, attention_mask=None):
                self.grad_enabled.append(torch.is_grad_enabled())
                return x

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=3,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
            core_trm_no_grad_inner_cycles_enabled=True,
        )
        core = QTRMRecursiveCore(cfg)
        core.fast_stack = RecordingStack()
        core.slow_stack = RecordingStack()

        workspace = torch.randn(2, cfg.workspace_tokens, cfg.d_model, requires_grad=True)
        _, z_h, _, _ = core(workspace)
        z_h.sum().backward()

        self.assertEqual(core.fast_stack.grad_enabled, [False, False, True])
        self.assertEqual(core.slow_stack.grad_enabled, [False, False, True])
        self.assertIsNotNone(workspace.grad)

    def test_core_respects_outer_torch_no_grad_context(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMRecursiveCore

        class RecordingStack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled = []

            def forward(self, x, attention_mask=None):
                self.grad_enabled.append(torch.is_grad_enabled())
                return x

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=2,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
        )
        core = QTRMRecursiveCore(cfg)
        core.fast_stack = RecordingStack()
        core.slow_stack = RecordingStack()

        with torch.no_grad():
            core(torch.randn(2, cfg.workspace_tokens, cfg.d_model))

        self.assertEqual(core.fast_stack.grad_enabled, [False, False])
        self.assertEqual(core.slow_stack.grad_enabled, [False, False])

    def test_core_trm_act_freezes_halted_samples_until_batch_finishes(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMRecursiveCore

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=1,
            l_cycles=0,
            outer_steps=3,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
            core_halt_enabled=True,
            core_halt_min_steps=1,
            core_halt_use_continue=False,
            core_halt_freeze_halted_state_enabled=True,
            core_step_conditioning_enabled=True,
            core_step_conditioning_max_steps=3,
            core_step_conditioning_scale=1.0,
        )
        core = QTRMRecursiveCore(cfg)
        with torch.no_grad():
            core.z_l_init.zero_()
            core.z_h_init.zero_()
            core.halt_head.weight.zero_()
            core.halt_head.bias.zero_()
            core.halt_head.weight[0, 0] = 10.0

        workspace = torch.zeros(2, cfg.workspace_tokens, cfg.d_model)
        workspace[0, 0, 0] = 2.0
        workspace[1, 0, 0] = -2.0

        _, _, trajectory, halt_info = core(workspace, enable_halt=True)

        self.assertEqual(len(trajectory), cfg.outer_steps)
        self.assertTrue(torch.equal(halt_info["halted"], torch.tensor([True, False])))
        self.assertTrue(torch.equal(halt_info["steps"], torch.tensor([1, 3])))
        self.assertTrue(torch.allclose(trajectory[0][0], trajectory[1][0]))
        self.assertTrue(torch.allclose(trajectory[1][0], trajectory[2][0]))
        self.assertFalse(torch.allclose(trajectory[0][1], trajectory[2][1]))

    def test_core_halt_exploration_delays_early_halt_during_training_only(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMRecursiveCore

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=1,
            l_cycles=0,
            outer_steps=4,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
            core_halt_enabled=True,
            core_halt_min_steps=1,
            core_halt_use_continue=False,
            core_halt_exploration_prob=1.0,
            core_halt_exploration_min_steps=3,
        )
        core = QTRMRecursiveCore(cfg)
        with torch.no_grad():
            core.halt_head.weight.zero_()
            core.halt_head.bias.copy_(torch.tensor([2.0, -2.0]))

        core.train()
        _, _, train_trajectory, train_halt = core(
            torch.zeros(2, cfg.workspace_tokens, cfg.d_model),
            enable_halt=True,
        )
        self.assertEqual(len(train_trajectory), 3)
        self.assertTrue(torch.equal(train_halt["steps"], torch.tensor([3, 3])))

        core.eval()
        _, _, eval_trajectory, eval_halt = core(
            torch.zeros(2, cfg.workspace_tokens, cfg.d_model),
            enable_halt=True,
        )
        self.assertEqual(len(eval_trajectory), 1)
        self.assertTrue(torch.equal(eval_halt["steps"], torch.tensor([1, 1])))

    def test_core_returns_detached_explicit_carry_for_continuation(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMRecursiveCore

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
        )
        core = QTRMRecursiveCore(cfg)
        workspace = torch.randn(2, cfg.workspace_tokens, cfg.d_model, requires_grad=True)

        _, z_h, _, halt_info = core(workspace, return_carry=True)
        z_h.sum().backward()
        carry = halt_info["carry"]

        self.assertEqual(carry.z_l.shape, workspace.shape)
        self.assertEqual(carry.z_h.shape, workspace.shape)
        self.assertEqual(carry.halted.shape, (2,))
        self.assertTrue(torch.equal(carry.steps, torch.tensor([cfg.outer_steps, cfg.outer_steps])))
        self.assertFalse(carry.z_l.requires_grad)
        self.assertFalse(carry.z_h.requires_grad)
        self.assertIsNotNone(workspace.grad)

    def test_core_carry_is_public_package_api(self):
        from qtrm_mm import QTRMCoreCarry

        self.assertEqual(QTRMCoreCarry.__name__, "QTRMCoreCarry")

    def test_core_resets_halted_carry_rows_on_next_call(self):
        import torch
        from qtrm_mm import QTRMConfig
        from qtrm_mm.core import QTRMCoreCarry, QTRMRecursiveCore

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=8,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=0,
            l_cycles=0,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=4,
            use_stable_inject=False,
        )
        core = QTRMRecursiveCore(cfg)
        with torch.no_grad():
            core.z_l_init.zero_()
            core.z_h_init.zero_()

        workspace = torch.zeros(2, cfg.workspace_tokens, cfg.d_model)
        previous_z_l = torch.full_like(workspace, 10.0)
        previous_z_h = torch.full_like(workspace, 20.0)
        previous_carry = QTRMCoreCarry(
            z_l=previous_z_l,
            z_h=previous_z_h,
            halted=torch.tensor([True, False]),
            steps=torch.tensor([5, 5]),
        )

        _, _, trajectory, halt_info = core(
            workspace,
            carry=previous_carry,
            return_carry=True,
        )

        self.assertTrue(torch.allclose(trajectory[0][0], torch.zeros_like(trajectory[0][0])))
        self.assertTrue(torch.allclose(trajectory[0][1], previous_z_h[1]))
        self.assertTrue(torch.equal(halt_info["steps"], torch.tensor([1, 6])))
        self.assertTrue(torch.equal(halt_info["carry"].steps, torch.tensor([1, 6])))

    def test_model_forward_can_return_and_reuse_core_carry(self):
        import torch
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=3,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            use_stable_inject=False,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        first = model(input_ids, return_core_carry=True)
        carry = first["core_carry"]
        second = model(input_ids, core_carry=carry, return_core_carry=True)

        self.assertEqual(carry.z_h.shape, first["z_h"].shape)
        self.assertFalse(carry.z_h.requires_grad)
        self.assertEqual(second["core_carry"].z_h.shape, second["z_h"].shape)

    def test_model_exposes_per_outer_step_core_depth_states_for_teacher_targets(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        out = model(input_ids, enable_core_halt=False)

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

    def test_model_exposes_transition_state_sequence_logits_from_core_depth_states(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.transition_state_sequence_enabled = True
        cfg.transition_state_sequence_max_tokens = 7
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["transition_state_sequence_logits"].shape,
            (2, cfg.outer_steps, 7, cfg.vocab_size),
        )

    def test_model_exposes_compact_transition_value_state_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.transition_value_state_enabled = True
        cfg.transition_value_state_max_tokens = 9
        cfg.transition_value_state_vocab_size = 12
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["transition_value_state_logits"].shape,
            (2, cfg.outer_steps, 9, 12),
        )

    def test_model_exposes_factorized_transition_value_state_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.factorized_value_state_enabled = True
        cfg.factorized_value_state_max_tokens = 9
        cfg.factorized_value_state_vocab_size = 12
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["transition_value_state_logits"].shape,
            (2, cfg.outer_steps, 9, 12),
        )
        self.assertEqual(
            out["factorized_value_state_logits"].shape,
            (2, cfg.outer_steps, 9, 12),
        )

    def test_model_exposes_factorized_value_state_kind_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.factorized_value_state_enabled = True
        cfg.factorized_value_state_max_tokens = 9
        cfg.factorized_value_state_vocab_size = 12
        cfg.factorized_value_state_kind_size = 3
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["factorized_value_state_kind_logits"].shape,
            (2, cfg.outer_steps, 3),
        )

    def test_model_exposes_role_value_state_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.factorized_value_state_enabled = True
        cfg.factorized_value_state_max_tokens = 9
        cfg.factorized_value_state_vocab_size = 12
        cfg.role_value_state_enabled = True
        cfg.role_value_state_num_roles = 10
        cfg.role_value_state_vocab_size = 128
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["role_value_state_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )

    def test_model_exposes_typed_algorithmic_value_state_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.typed_algorithmic_value_state_enabled = True
        cfg.typed_algorithmic_value_state_max_list_slots = 4
        cfg.typed_algorithmic_value_state_offset_vocab_size = 17
        cfg.typed_algorithmic_value_state_scalar_vocab_size = 19
        cfg.typed_algorithmic_value_state_kind_size = 3
        cfg.typed_algorithmic_value_state_recurrent_enabled = True
        cfg.typed_algorithmic_value_state_primitive_conditioning_enabled = True
        cfg.typed_algorithmic_value_state_subregisters_enabled = True
        cfg.typed_algorithmic_value_state_residual_feedback_enabled = True
        cfg.typed_algorithmic_value_state_residual_delta_enabled = True
        cfg.typed_algorithmic_value_state_scalar_offset_enabled = True
        cfg.typed_algorithmic_value_state_scalar_regression_enabled = True
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 5
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["typed_algorithmic_kind_logits"].shape,
            (2, cfg.outer_steps, 3),
        )
        self.assertEqual(
            out["typed_algorithmic_raw_list_offset_logits"].shape,
            (2, cfg.outer_steps, 4, 17),
        )
        self.assertEqual(
            out["typed_algorithmic_doubled_list_offset_logits"].shape,
            (2, cfg.outer_steps, 4, 17),
        )
        self.assertEqual(
            out["typed_algorithmic_scalar_coeff_logits"].shape,
            (2, cfg.outer_steps, 19),
        )
        self.assertEqual(
            out["typed_algorithmic_scalar_offset_logits"].shape,
            (2, cfg.outer_steps, 19),
        )
        self.assertEqual(
            out["typed_algorithmic_scalar_offset_value"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["typed_algorithmic_final_residual_value"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["typed_algorithmic_scalar_residual_delta_logits"].shape,
            (2, cfg.outer_steps, 19),
        )
        disabled = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            disable_typed_algorithmic_value_state=True,
        )
        self.assertEqual(
            disabled["typed_algorithmic_raw_list_offset_logits"].shape,
            (2, cfg.outer_steps, 4, 17),
        )
        recurrent_off = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            disable_typed_algorithmic_value_state_recurrent=True,
        )
        self.assertEqual(
            recurrent_off["typed_algorithmic_raw_list_offset_logits"].shape,
            (2, cfg.outer_steps, 4, 17),
        )
        self.assertIsNotNone(model.typed_algorithmic_recurrent_primitive_proj)
        self.assertEqual(
            model.typed_algorithmic_recurrent_primitive_proj.in_features,
            cfg.primitive_transition_num_operations,
        )
        self.assertIsNotNone(model.typed_algorithmic_list_subregister_update)
        self.assertIsNotNone(model.typed_algorithmic_scalar_subregister_update)
        self.assertIsNotNone(model.typed_algorithmic_final_subregister_update)
        self.assertIsNotNone(model.typed_algorithmic_scalar_residual_feedback_proj)
        self.assertIsNotNone(model.typed_algorithmic_final_residual_feedback_proj)
        self.assertIsNotNone(model.typed_algorithmic_scalar_residual_delta_head)
        self.assertIsNotNone(model.typed_algorithmic_scalar_offset_head)
        self.assertIsNotNone(model.typed_algorithmic_scalar_residual_value_head)
        self.assertIsNotNone(model.typed_algorithmic_final_residual_value_head)

    def test_model_exposes_core_role_value_state_logits_from_core_tokens(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_role_value_state_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )
        self.assertEqual(out["core_depth_states"].shape, (2, cfg.outer_steps, cfg.d_model))

    def test_model_exposes_core_role_value_transition_logits_between_core_steps(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_transition_enabled = True
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_role_value_transition_logits"].shape,
            (2, cfg.outer_steps - 1, 10, 128),
        )

    def test_core_state_carry_updates_role_tokens_and_exposes_gate_telemetry(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_state_carry_enabled = True
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_state_carry_gate_mean"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["core_role_value_state_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )

        disabled = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            disable_core_state_carry=True,
        )
        self.assertEqual(disabled["core_state_carry_gate_mean"].shape, (2, 0))

    def test_core_role_value_delta_adapter_exposes_gate_telemetry(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_delta_enabled = True
        cfg.core_role_value_delta_gate_init_bias = -4.0
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_role_value_delta_gate_mean"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["core_role_value_state_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )

        disabled = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            disable_core_role_value_delta=True,
        )
        self.assertEqual(disabled["core_role_value_delta_gate_mean"].shape, (2, 0))

    def test_core_value_delta_code_executor_exposes_discrete_code_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_value_delta_code_enabled = True
        cfg.core_value_delta_codebook_size = 128
        cfg.core_value_delta_code_gate_init_bias = -4.0
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_value_delta_code_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )
        self.assertEqual(
            out["core_value_delta_code_gate_mean"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["core_role_value_state_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )

        disabled = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            disable_core_value_delta_code=True,
        )
        self.assertEqual(disabled["core_value_delta_code_logits"].shape, (2, 0, 10, 128))
        self.assertEqual(disabled["core_value_delta_code_gate_mean"].shape, (2, 0))

    def test_core_typed_register_executor_exposes_persistent_register_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 6
        cfg.core_typed_register_gate_init_bias = -3.0
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_typed_register_operation_logits"].shape,
            (2, cfg.outer_steps, 6),
        )
        self.assertEqual(
            out["core_typed_register_value_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )
        self.assertEqual(
            out["core_typed_register_transition_logits"].shape,
            (2, cfg.outer_steps - 1, 10, 128),
        )
        self.assertEqual(
            out["core_typed_register_gate_mean"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            out["core_role_value_state_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )

        disabled = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
            disable_core_typed_register_executor=True,
        )
        self.assertEqual(
            disabled["core_typed_register_operation_logits"].shape,
            (2, 0, 6),
        )
        self.assertEqual(
            disabled["core_typed_register_value_logits"].shape,
            (2, 0, 10, 128),
        )
        self.assertEqual(
            disabled["core_typed_register_transition_logits"].shape,
            (2, 0, 10, 128),
        )
        self.assertEqual(disabled["core_typed_register_gate_mean"].shape, (2, 0))

    def test_core_typed_register_transition_readout_feeds_value_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 6
        cfg.core_typed_register_transition_readout_enabled = True
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertTrue(
            torch.allclose(
                out["core_typed_register_value_logits"][:, 1:],
                out["core_typed_register_transition_logits"],
            )
        )

    def test_core_typed_register_value_feedback_is_initialized_when_enabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 6
        cfg.core_typed_register_value_feedback_enabled = True
        cfg.core_typed_register_value_feedback_gate_init_bias = -1.5
        model = QTRMMultimodalModel(cfg)

        self.assertIsNotNone(model.core_typed_register_value_feedback_embed)
        self.assertAlmostEqual(
            float(model.core_typed_register_value_feedback_gate.bias.detach()[0]),
            cfg.core_typed_register_value_feedback_gate_init_bias,
        )

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_typed_register_value_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )

    def test_core_role_value_template_codec_can_override_value_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_prompt_extract_enabled = True
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 6
        cfg.core_role_value_template_codec_enabled = True
        cfg.core_role_value_template_num_templates = 16
        cfg.core_role_value_template_max_steps = 8
        model = QTRMMultimodalModel(cfg)

        self.assertIsNotNone(model.core_role_value_template_table)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(out["core_role_value_template_logits"].shape, (2, 16))
        self.assertEqual(
            out["core_typed_register_value_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )
        self.assertTrue(
            torch.allclose(
                out["core_role_value_state_logits"],
                out["core_typed_register_value_logits"],
            )
        )

    def test_core_role_value_template_factorized_codec_composes_template_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_prompt_extract_enabled = True
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 6
        cfg.core_role_value_template_codec_enabled = True
        cfg.core_role_value_template_factorized_enabled = True
        cfg.core_role_value_template_length_classes = 5
        cfg.core_role_value_template_parity_classes = 2
        cfg.core_role_value_template_offset_classes = 7
        cfg.core_role_value_template_num_templates = 70
        cfg.core_role_value_template_max_steps = 8
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(out["core_role_value_template_logits"].shape, (2, 70))
        probs = out["core_role_value_template_logits"].float().softmax(dim=-1)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4))

    def test_core_role_value_answer_bridge_feeds_answer_loop(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_answer_bridge_enabled = True
        cfg.core_role_value_state_answer_bridge_gate_init_bias = 0.0
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, enable_core_halt=False)
        disabled = model(
            input_ids,
            enable_core_halt=False,
            disable_core_role_value_answer_bridge=True,
        )

        self.assertEqual(
            out["core_role_value_state_answer_bridge_gate_mean"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            disabled["core_role_value_state_answer_bridge_gate_mean"].shape,
            (2, 0),
        )
        self.assertFalse(
            torch.allclose(
                out["answer_state_loop_logits"],
                disabled["answer_state_loop_logits"],
            )
        )

    def test_core_role_value_answer_bridge_feeds_depth_answer_loop(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_answer_bridge_enabled = True
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_logits=True,
            return_core_depth_text_logits=True,
        )
        disabled = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_logits=True,
            return_core_depth_text_logits=True,
            disable_core_role_value_answer_bridge=True,
        )

        self.assertEqual(out["core_depth_last_logits"].shape[:2], (2, cfg.outer_steps))
        self.assertEqual(
            out["core_depth_text_logits"].shape[:3],
            (2, cfg.outer_steps, input_ids.shape[1]),
        )
        self.assertFalse(
            torch.allclose(
                out["core_depth_last_logits"],
                disabled["core_depth_last_logits"],
            )
        )
        self.assertFalse(
            torch.allclose(
                out["core_depth_text_logits"],
                disabled["core_depth_text_logits"],
            )
        )

    def test_core_role_value_vocab_renderer_feeds_residual_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.qtrm_logits_scale = 0.0
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 6
        cfg.core_role_value_state_vocab_size = 32
        cfg.core_role_value_state_answer_bridge_enabled = True
        cfg.core_role_value_state_answer_bridge_gate_min = 1.0
        cfg.core_role_value_state_vocab_renderer_enabled = True
        cfg.core_role_value_state_vocab_renderer_gate_min = 1.0
        cfg.core_role_value_state_vocab_renderer_rank = 4
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.fill_(0.05)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, enable_core_halt=False)
        disabled = model(
            input_ids,
            enable_core_halt=False,
            disable_core_role_value_vocab_renderer=True,
        )
        offset = out["qtrm_residual_logits"].shape[1] - input_ids.shape[1]

        self.assertEqual(
            out["core_role_value_vocab_renderer_logits"].shape,
            (2, input_ids.shape[1], cfg.vocab_size),
        )
        self.assertGreater(
            float(out["core_role_value_vocab_renderer_logits"].abs().max()),
            0.0,
        )
        self.assertFalse(
            torch.allclose(
                out["qtrm_residual_logits"][:, offset:, :],
                disabled["qtrm_residual_logits"][:, offset:, :],
            )
        )

    def test_core_role_value_answer_prompt_context_changes_bridge_tokens(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_answer_bridge_enabled = True
        cfg.core_role_value_state_answer_prompt_context_enabled = True
        cfg.core_role_value_state_answer_prompt_gate_init_bias = 4.0
        cfg.core_role_value_state_answer_prompt_gate_min = 1.0
        model = QTRMMultimodalModel(cfg)
        role_value_logits = torch.randn(
            2,
            cfg.outer_steps,
            cfg.core_role_value_state_num_roles,
            cfg.core_role_value_state_vocab_size,
        )
        prompt_mask = torch.ones(2, 6, dtype=torch.bool)

        out_a = model._compute_core_role_value_state_answer_bridge(
            role_value_logits,
            prompt_context_seq=torch.randn(2, 6, cfg.d_model),
            prompt_context_mask=prompt_mask,
        )
        out_b = model._compute_core_role_value_state_answer_bridge(
            role_value_logits,
            prompt_context_seq=torch.randn(2, 6, cfg.d_model) * -1.0,
            prompt_context_mask=prompt_mask,
        )
        disabled = model._compute_core_role_value_state_answer_bridge(
            role_value_logits,
            prompt_context_seq=torch.randn(2, 6, cfg.d_model),
            prompt_context_mask=prompt_mask,
            disabled=True,
        )

        self.assertEqual(
            out_a["tokens"].shape,
            (
                2,
                cfg.outer_steps,
                cfg.core_role_value_state_num_roles,
                cfg.d_model,
            ),
        )
        self.assertFalse(torch.allclose(out_a["tokens"], out_b["tokens"]))
        self.assertEqual(disabled["tokens"].shape, (2, 0, 0, cfg.d_model))

    def test_core_role_value_answer_final_binder_is_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_answer_bridge_enabled = True
        cfg.core_role_value_state_answer_final_binder_enabled = True
        cfg.core_role_value_state_answer_final_gate_init_bias = 4.0
        cfg.core_role_value_state_answer_final_gate_min = 1.0
        torch.manual_seed(128)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids, enable_core_halt=False)
        binder_off = model(
            input_ids,
            enable_core_halt=False,
            disable_core_role_value_answer_final_binder=True,
        )

        self.assertEqual(
            full["core_role_value_state_answer_final_embedding"].shape,
            (2, 1, cfg.d_model),
        )
        self.assertFalse(
            torch.allclose(
                full["answer_state_loop_logits"],
                binder_off["answer_state_loop_logits"],
            )
        )

    def test_answer_state_loop_recurrent_block_feeds_answer_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_recurrent_block_enabled = True
        cfg.answer_state_loop_recurrent_layers = 1
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, enable_core_halt=False)
        disabled = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_recurrent=True,
        )

        self.assertEqual(
            out["answer_state_loop_recurrent_gate_mean"].shape,
            (2, cfg.outer_steps),
        )
        self.assertEqual(
            disabled["answer_state_loop_recurrent_gate_mean"].shape,
            (2, 0),
        )
        self.assertFalse(
            torch.allclose(
                out["answer_state_loop_logits"],
                disabled["answer_state_loop_logits"],
            )
        )

    def test_answer_state_loop_selective_context_router_is_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_selective_context_enabled = True
        cfg.answer_state_loop_selective_context_top_k = 1
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, enable_core_halt=False)
        disabled = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_selective_context=True,
        )

        self.assertFalse(
            torch.allclose(
                out["answer_state_loop_logits"],
                disabled["answer_state_loop_logits"],
            )
        )

    def test_answer_state_loop_selective_context_can_force_dense_teacher_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_selective_context_enabled = True
        cfg.answer_state_loop_selective_context_top_k = 1
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        sparse = model(input_ids, enable_core_halt=False)
        dense = model(
            input_ids,
            enable_core_halt=False,
            force_answer_state_loop_dense_context=True,
        )

        self.assertFalse(
            torch.allclose(
                sparse["answer_state_loop_logits"],
                dense["answer_state_loop_logits"],
            )
        )

    def test_answer_state_loop_finality_selector_uses_transition_joint_state(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_finality_selector_enabled = True
        cfg.transition_state_joint_enabled = True
        cfg.transition_state_joint_size = 10
        torch.manual_seed(123)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids, enable_core_halt=False)
        selector_off = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_finality_selector=True,
        )
        state_off = model(
            input_ids,
            enable_core_halt=False,
            disable_transition_state=True,
        )

        self.assertFalse(
            torch.allclose(
                full["answer_state_loop_logits"],
                selector_off["answer_state_loop_logits"],
            )
        )
        self.assertTrue(
            torch.allclose(
                state_off["answer_state_loop_logits"],
                selector_off["answer_state_loop_logits"],
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_answer_state_loop_finality_gate_is_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_finality_gate_enabled = True
        cfg.answer_state_loop_finality_gate_mode = "soft"
        cfg.transition_state_joint_enabled = True
        cfg.transition_state_joint_size = 10
        torch.manual_seed(125)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids, enable_core_halt=False)
        gate_off = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_finality_gate=True,
        )
        state_off = model(
            input_ids,
            enable_core_halt=False,
            disable_transition_state=True,
        )

        self.assertFalse(
            torch.allclose(
                full["answer_state_loop_logits"],
                gate_off["answer_state_loop_logits"],
            )
        )
        self.assertTrue(
            torch.allclose(
                state_off["answer_state_loop_logits"],
                gate_off["answer_state_loop_logits"],
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_answer_state_loop_halt_head_is_trainable_and_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_halt_enabled = True
        cfg.answer_state_loop_halt_gate_enabled = True
        cfg.answer_state_loop_halt_gate_mode = "soft"
        cfg.answer_state_loop_halt_init_bias = 0.0
        torch.manual_seed(126)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids, enable_core_halt=False)
        gate_off = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_halt_gate=True,
        )

        self.assertEqual(full["answer_state_loop_halt_logits"].shape, (2, cfg.outer_steps))
        self.assertFalse(
            torch.allclose(
                full["answer_state_loop_logits"],
                gate_off["answer_state_loop_logits"],
            )
        )

    def test_answer_state_loop_lm_adapter_is_zero_init_and_causal_logit_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_lm_adapter_enabled = True
        cfg.answer_state_loop_lm_adapter_rank = 4
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        self.assertTrue(
            torch.equal(
                model.answer_state_loop_lm_adapter_up.weight,
                torch.zeros_like(model.answer_state_loop_lm_adapter_up.weight),
            )
        )

        out = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
        )
        expected_logits = model.lm_head(out["answer_state_loop_hidden"]) * float(
            cfg.qtrm_logits_scale
        )
        expected_depth_logits = model.lm_head(
            out["answer_state_loop_depth_hidden"]
        ) * float(cfg.qtrm_logits_scale)

        self.assertTrue(
            torch.allclose(out["answer_state_loop_logits"], expected_logits)
        )
        self.assertTrue(
            torch.allclose(out["core_depth_text_logits"], expected_depth_logits)
        )

        with torch.no_grad():
            model.answer_state_loop_lm_adapter_up.weight[0, 0] = 1.0
        changed = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
        )

        self.assertFalse(
            torch.allclose(
                changed["answer_state_loop_logits"],
                out["answer_state_loop_logits"],
            )
        )
        self.assertFalse(
            torch.allclose(
                changed["core_depth_text_logits"],
                out["core_depth_text_logits"],
            )
        )

    def test_answer_state_loop_hidden_bridge_is_zero_init_and_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_hidden_bridge_enabled = True
        cfg.answer_state_loop_hidden_bridge_hidden_dim = 16
        model = QTRMMultimodalModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        self.assertTrue(
            torch.equal(
                model.answer_state_loop_hidden_bridge_up.weight,
                torch.zeros_like(model.answer_state_loop_hidden_bridge_up.weight),
            )
        )

        out = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
        )
        bridge_off = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
            disable_answer_state_loop_hidden_bridge=True,
        )

        self.assertTrue(
            torch.allclose(
                out["answer_state_loop_logits"],
                bridge_off["answer_state_loop_logits"],
            )
        )
        self.assertTrue(
            torch.allclose(
                out["core_depth_text_logits"],
                bridge_off["core_depth_text_logits"],
            )
        )

        with torch.no_grad():
            model.answer_state_loop_hidden_bridge_up.weight[0, 0] = 1.0
        changed = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
        )
        changed_bridge_off = model(
            input_ids,
            enable_core_halt=False,
            return_core_depth_text_logits=True,
            disable_answer_state_loop_hidden_bridge=True,
        )

        self.assertFalse(
            torch.allclose(
                changed["answer_state_loop_logits"],
                changed_bridge_off["answer_state_loop_logits"],
            )
        )
        self.assertFalse(
            torch.allclose(
                changed["core_depth_text_logits"],
                changed_bridge_off["core_depth_text_logits"],
            )
        )

    def test_answer_state_loop_future_token_decoder_is_auxiliary_only(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_future_token_decoder_enabled = True
        cfg.answer_state_loop_future_token_max_tokens = 5
        torch.manual_seed(129)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, enable_core_halt=False)

        self.assertEqual(
            tuple(out["answer_state_loop_future_token_logits"].shape),
            (2, 5, cfg.vocab_size),
        )
        self.assertEqual(
            tuple(out["answer_state_loop_logits"].shape),
            (2, 6, cfg.vocab_size),
        )

    def test_answer_state_loop_talker_is_zero_gated_and_causal_logit_path(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_talker_enabled = True
        cfg.answer_state_loop_talker_layers = 1
        cfg.answer_state_loop_talker_gate_init_bias = -30.0
        torch.manual_seed(130)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        out = model(input_ids, enable_core_halt=False)
        disabled_before = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_talker=True,
        )
        expected_logits = model.lm_head(out["answer_state_loop_hidden"]) * float(
            cfg.qtrm_logits_scale
        )

        self.assertTrue(
            torch.allclose(
                out["answer_state_loop_logits"],
                expected_logits,
                atol=1e-5,
                rtol=1e-5,
            )
        )

        with torch.no_grad():
            model.answer_state_loop_talker_gate.bias.fill_(30.0)
        changed = model(input_ids, enable_core_halt=False)

        self.assertFalse(
            torch.allclose(
                changed["answer_state_loop_logits"],
                out["answer_state_loop_logits"],
            )
        )
        disabled = model(
            input_ids,
            enable_core_halt=False,
            disable_answer_state_loop_talker=True,
        )
        self.assertTrue(
            torch.allclose(
                disabled["answer_state_loop_logits"],
                disabled_before["answer_state_loop_logits"],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_finality_selector_hard_first_uses_first_final_depth(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_finality_selector_enabled = True
        cfg.answer_state_loop_finality_selector_mode = "hard_first"
        cfg.transition_state_joint_enabled = True
        cfg.transition_state_joint_size = 4
        torch.manual_seed(124)
        model = QTRMMultimodalModel(cfg)
        depth_hidden = torch.zeros(1, 3, 1, cfg.d_model)
        depth_hidden[:, 0, :, 0] = 1.0
        depth_hidden[:, 1, :, 0] = 2.0
        depth_hidden[:, 2, :, 0] = 3.0
        joint_logits = torch.full((1, 3, 4), -8.0)
        joint_logits[:, 0, 0] = 8.0
        joint_logits[:, 1, 1] = 8.0
        joint_logits[:, 2, 1] = 9.0

        _, selected_hidden = model._select_answer_state_loop_by_finality(
            torch.zeros(1, 1, cfg.vocab_size),
            torch.zeros(1, 1, cfg.d_model),
            depth_hidden,
            joint_logits,
        )

        self.assertAlmostEqual(float(selected_hidden[0, 0, 0]), 2.0)

    def test_transition_joint_answer_bridge_is_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.answer_state_loop_recurrent_block_enabled = True
        cfg.transition_state_joint_enabled = True
        cfg.transition_state_joint_size = 6
        cfg.transition_state_joint_answer_bridge_enabled = True
        torch.manual_seed(125)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids, enable_core_halt=False)
        bridge_off = model(
            input_ids,
            enable_core_halt=False,
            disable_transition_state_joint_answer_bridge=True,
        )

        self.assertFalse(
            torch.allclose(
                full["answer_state_loop_logits"],
                bridge_off["answer_state_loop_logits"],
            )
        )

    def test_transition_final_answer_binder_is_ablatable(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.answer_state_loop_requires_core = True
        cfg.transition_state_joint_enabled = True
        cfg.transition_state_joint_size = 6
        cfg.transition_state_final_answer_binder_enabled = True
        torch.manual_seed(126)
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        full = model(input_ids, enable_core_halt=False)
        binder_off = model(
            input_ids,
            enable_core_halt=False,
            disable_transition_state_final_answer_binder=True,
        )

        self.assertEqual(
            full["transition_state_final_answer_embedding"].shape,
            (2, 1, cfg.d_model),
        )
        self.assertFalse(
            torch.allclose(
                full["answer_state_loop_logits"],
                binder_off["answer_state_loop_logits"],
            )
        )

    def test_transition_final_answer_binder_reads_selected_core_state(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.answer_state_loop_enabled = True
        cfg.transition_state_joint_enabled = True
        cfg.transition_state_joint_size = 4
        cfg.transition_state_final_answer_binder_enabled = True
        torch.manual_seed(127)
        model = QTRMMultimodalModel(cfg)
        core_depth_states = torch.zeros(1, 3, cfg.d_model)
        core_depth_states[:, 1, 0] = 1.0
        joint_logits = torch.full((1, 3, 4), -8.0)
        joint_logits[:, 0, 0] = 8.0
        joint_logits[:, 1, 1] = 8.0
        joint_logits[:, 2, 0] = 8.0

        selected = model._compute_transition_state_final_answer_embedding(
            core_depth_states,
            joint_logits,
        )
        changed_state = core_depth_states.clone()
        changed_state[:, 1, 0] = 2.0
        changed = model._compute_transition_state_final_answer_embedding(
            changed_state,
            joint_logits,
        )

        self.assertFalse(torch.allclose(selected, changed))

    def test_core_role_value_state_prompt_extract_path_is_initialized_when_enabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        cfg.core_role_value_state_prompt_extract_enabled = True
        cfg.core_role_value_state_prompt_self_condition_enabled = True
        cfg.core_role_value_state_prompt_parity_enabled = True
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 6
        cfg.core_typed_register_transition_readout_enabled = True
        cfg.core_typed_register_prompt_first_transition_readout_enabled = True
        model = QTRMMultimodalModel(cfg)

        self.assertIsNotNone(model.core_role_value_state_prompt_cross)
        self.assertIsNotNone(model.core_role_value_state_prompt_gate)
        self.assertIsNotNone(
            model.core_role_value_state_prompt_self_condition_value_embed
        )
        self.assertAlmostEqual(
            float(model.core_role_value_state_prompt_gate.bias.detach()[0]),
            cfg.core_role_value_state_prompt_extract_gate_init_bias,
        )
        self.assertAlmostEqual(
            float(
                model.core_role_value_state_prompt_self_condition_gate.bias.detach()[0]
            ),
            cfg.core_role_value_state_prompt_self_condition_gate_init_bias,
        )
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        out = model(input_ids, enable_core_halt=False)
        prompt_extract_off = model(
            input_ids,
            enable_core_halt=False,
            disable_core_role_value_prompt_extract=True,
        )
        self.assertEqual(
            out["core_role_value_state_prompt_logits"].shape,
            (2, 1, 10, 128),
        )
        self.assertEqual(
            out["core_role_value_state_prompt_parity_logits"].shape,
            (2, 2),
        )
        self.assertEqual(
            out["core_typed_register_value_logits"].shape,
            (2, cfg.outer_steps, 10, 128),
        )
        self.assertTrue(
            torch.allclose(
                out["core_typed_register_value_logits"][:, 0],
                out["core_role_value_state_prompt_logits"][:, 0],
            )
        )
        self.assertFalse(
            torch.allclose(
                out["core_role_value_state_prompt_logits"],
                prompt_extract_off["core_role_value_state_prompt_logits"],
            )
        )

    def test_core_role_value_state_logits_are_empty_when_core_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 10
        cfg.core_role_value_state_vocab_size = 128
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            disable_core=True,
        )

        self.assertEqual(out["core_role_value_state_logits"].shape, (2, 0, 10, 128))

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

        self.assertEqual(
            out["prompt_context"].shape,
            (2, cfg.outer_steps, cfg.d_model),
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

    def test_core_primitive_role_value_executor_exposes_recurrent_logits(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 4
        cfg.core_role_value_state_vocab_size = 8
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 12
        cfg.core_role_value_state_prompt_extract_enabled = True
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 12
        cfg.core_primitive_role_value_executor_enabled = True
        cfg.core_primitive_role_value_mlp_enabled = True
        cfg.core_primitive_role_value_prompt_context_enabled = True
        cfg.core_primitive_role_value_prompt_token_attention_enabled = True
        cfg.core_primitive_role_value_update_gate_enabled = True
        cfg.core_primitive_role_value_update_gate_init_bias = -3.0
        cfg.core_primitive_role_value_field_specific_heads_enabled = True
        cfg.core_primitive_role_value_operation_specific_heads_enabled = True
        cfg.core_primitive_typed_selector_enabled = True
        cfg.core_primitive_typed_selector_init_bias = -5.0
        model = QTRMMultimodalModel(cfg)

        self.assertIsNotNone(model.core_primitive_role_value_prompt_context_adapter)
        self.assertIsNotNone(model.core_primitive_role_value_prompt_query_norm)
        self.assertIsNotNone(model.core_primitive_role_value_prompt_token_context_norm)
        self.assertIsNotNone(model.core_primitive_role_value_prompt_cross)
        self.assertIsNotNone(model.core_primitive_role_value_prompt_token_output_norm)
        self.assertIsNotNone(model.core_primitive_role_value_update_gate)
        self.assertIsNotNone(model.core_primitive_role_value_operation_heads)
        self.assertEqual(len(model.core_primitive_role_value_operation_heads), 12)
        self.assertIsNotNone(model.core_primitive_role_value_list_head)
        self.assertIsNotNone(model.core_primitive_role_value_scalar_head)
        self.assertIsNotNone(model.core_primitive_typed_selector)
        self.assertTrue(
            torch.allclose(
                model.core_primitive_role_value_prompt_context_adapter.weight,
                torch.zeros_like(
                    model.core_primitive_role_value_prompt_context_adapter.weight
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.core_primitive_role_value_update_gate.weight,
                torch.zeros_like(model.core_primitive_role_value_update_gate.weight),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.core_primitive_role_value_update_gate.bias,
                torch.full_like(model.core_primitive_role_value_update_gate.bias, -3.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.core_primitive_typed_selector.bias,
                torch.full_like(model.core_primitive_typed_selector.bias, -5.0),
            )
        )

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(
            out["core_primitive_role_value_state_logits"].shape,
            (
                2,
                cfg.outer_steps,
                cfg.core_role_value_state_num_roles,
                cfg.core_role_value_state_vocab_size,
            ),
        )
        self.assertEqual(
            out["core_primitive_typed_selector_gate"].shape,
            (2, cfg.outer_steps, cfg.core_role_value_state_num_roles),
        )
        self.assertEqual(
            out["core_primitive_typed_selector_gate_mean"].shape,
            (2, cfg.outer_steps),
        )

    def test_core_primitive_role_value_executor_is_empty_when_core_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 4
        cfg.core_role_value_state_vocab_size = 8
        cfg.core_typed_register_executor_enabled = True
        cfg.core_typed_register_num_operations = 12
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 12
        cfg.core_primitive_role_value_executor_enabled = True
        model = QTRMMultimodalModel(cfg)

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            disable_core=True,
        )

        self.assertEqual(
            out["core_primitive_role_value_state_logits"].shape,
            (2, 0, cfg.core_role_value_state_num_roles, cfg.core_role_value_state_vocab_size),
        )

    def test_core_primitive_residual_delta_preserves_state_when_delta_is_zero(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        cfg.outer_steps = 3
        cfg.core_role_value_state_enabled = True
        cfg.core_role_value_state_num_roles = 3
        cfg.core_role_value_state_vocab_size = 5
        cfg.core_role_value_state_prompt_extract_enabled = False
        cfg.primitive_transition_enabled = True
        cfg.primitive_transition_num_operations = 4
        cfg.core_primitive_role_value_executor_enabled = True
        cfg.core_primitive_role_value_mlp_enabled = True
        cfg.core_primitive_role_value_update_gate_enabled = True
        cfg.core_primitive_role_value_residual_delta_enabled = True
        model = QTRMMultimodalModel(cfg)

        with torch.no_grad():
            model.core_primitive_role_value_head.weight.zero_()
            model.core_primitive_role_value_head.bias.zero_()

        prompt_logits = torch.randn(
            2,
            1,
            cfg.core_role_value_state_num_roles,
            cfg.core_role_value_state_vocab_size,
        )
        operation_logits = torch.zeros(2, cfg.outer_steps, 4)
        reference = torch.zeros(2, 1, cfg.d_model)

        logits, gate = model._compute_core_primitive_role_value_state_logits(
            {"operation_logits": operation_logits},
            prompt_logits=prompt_logits,
            fallback_logits=torch.empty(2, 0, 3, 5),
            reference=reference,
        )

        expected = prompt_logits[:, 0].unsqueeze(1).expand_as(logits)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))
        self.assertEqual(
            gate.shape,
            (2, cfg.outer_steps, cfg.core_role_value_state_num_roles),
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
