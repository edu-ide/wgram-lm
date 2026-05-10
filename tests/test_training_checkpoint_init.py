import tempfile
import unittest

import torch


class TrainingCheckpointInitTests(unittest.TestCase):
    def test_load_initial_checkpoint_restores_model_state(self):
        from qtrm_mm.training.train import load_initial_checkpoint

        model = torch.nn.Linear(2, 2)
        wanted = torch.nn.Linear(2, 2)
        with torch.no_grad():
            wanted.weight.fill_(3.0)
            wanted.bias.fill_(-2.0)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model": wanted.state_dict()}, f.name)
            missing, unexpected = load_initial_checkpoint(model, f.name, map_location="cpu")

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertTrue(torch.equal(model.weight, wanted.weight))
        self.assertTrue(torch.equal(model.bias, wanted.bias))

    def test_load_initial_checkpoint_skips_shape_mismatched_tensors(self):
        from qtrm_mm.training.train import load_initial_checkpoint

        model = torch.nn.Linear(3, 2)
        incompatible = torch.nn.Linear(2, 2)
        original = model.weight.detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model": incompatible.state_dict()}, f.name)
            missing, unexpected = load_initial_checkpoint(model, f.name, map_location="cpu")

        self.assertIn("weight", missing)
        self.assertEqual(unexpected, [])
        self.assertTrue(torch.equal(model.weight, original))

    def test_core_halt_only_policy_freezes_everything_except_halt_head(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_halt_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_halt_only")

        self.assertEqual(trainable, ["core.halt_head.weight", "core.halt_head.bias"])
        self.assertTrue(model.core.halt_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_core_halt_only_policy_requires_halt_head(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        model = QTRMMultimodalModel(
            QTRMConfig(
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
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                core_halt_enabled=False,
            )
        )

        with self.assertRaisesRegex(ValueError, "core_halt_only"):
            configure_trainable_parameters(model, "core_halt_only")

    def test_core_only_policy_freezes_everything_except_recursive_core(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_step_conditioning_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("core.") for name in trainable))
        self.assertTrue(any(name.startswith("core.fast_stack.") for name in trainable))
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_core_and_loop_readout_policy_freezes_everything_else(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_loop_readout_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_loop_readout")

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.") or name.startswith("core_loop_readout_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.core_loop_readout_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_core_and_answer_state_loop_policy_freezes_everything_else(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_answer_state_loop")

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.") or name.startswith("answer_state_loop_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_core_and_answer_state_loop_policy_includes_transition_state_when_enabled(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_enabled=True,
            transition_state_dim=3,
            transition_state_hidden_dim=16,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_answer_state_loop")

        self.assertTrue(any(name.startswith("transition_state_") for name in trainable))
        self.assertTrue(model.transition_state_predictor.net[1].weight.requires_grad)
        self.assertTrue(model.transition_state_to_answer.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_and_answer_state_loop_policy_includes_final_answer_binder(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_state_final_answer_binder_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_answer_state_loop")

        self.assertIn("transition_state_final_answer_proj.weight", trainable)
        self.assertTrue(model.transition_state_final_answer_proj.weight.requires_grad)
        self.assertTrue(model.transition_state_final_answer_gate.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_transition_order_bottleneck_policy_freezes_everything_else(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_transition_order_bottleneck_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=7,
            transition_state_finality_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_transition_order_bottleneck_and_readouts",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core_transition_order_bottleneck_")
                or name.startswith("primitive_transition_")
                or name.startswith("transition_state_finality_")
                for name in trainable
            )
        )
        self.assertTrue(model.core_transition_order_bottleneck_query.requires_grad)
        self.assertTrue(model.primitive_transition_operation_head[-1].weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.core.parameters()).requires_grad)

    def test_core_and_transition_order_bottleneck_policy_includes_core(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_transition_order_bottleneck_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=7,
            transition_state_finality_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_and_transition_order_bottleneck_and_readouts",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.")
                or name.startswith("core_transition_order_bottleneck_")
                or name.startswith("primitive_transition_")
                or name.startswith("transition_state_finality_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.core_transition_order_bottleneck_query.requires_grad)
        self.assertTrue(model.primitive_transition_operation_head[-1].weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_answer_state_loop_only_policy_freezes_core_and_transition_state(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_enabled=True,
            transition_state_dim=3,
            transition_state_hidden_dim=16,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "answer_state_loop_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("answer_state_loop_") for name in trainable))
        self.assertTrue(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_predictor.net[1].weight.requires_grad)
        self.assertFalse(model.transition_state_to_answer.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_role_value_answer_bridge_loop_only_trains_bridge_and_answer_loop(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=16,
            core_role_value_state_answer_bridge_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "role_value_answer_bridge_loop_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("answer_state_loop_")
                or name.startswith("core_role_value_state_embed.")
                or name.startswith("core_role_value_state_answer_")
                for name in trainable
            )
        )
        self.assertTrue(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_embed.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_answer_value_embed.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_answer_gate.weight.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_primitive_role_value_answer_bridge_loop_policy_is_scoped(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            core_role_value_state_enabled=True,
            core_role_value_state_prompt_extract_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=16,
            core_role_value_state_answer_bridge_enabled=True,
            core_typed_register_executor_enabled=True,
            core_typed_register_num_operations=3,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=3,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
            core_primitive_role_value_hidden_dim=32,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "primitive_role_value_answer_bridge_loop",
        )

        self.assertTrue(any(name.startswith("primitive_transition_") for name in trainable))
        self.assertTrue(any(name.startswith("core_primitive_role_value_") for name in trainable))
        self.assertTrue(any(name.startswith("answer_state_loop_") for name in trainable))
        self.assertTrue(any(name.startswith("core_role_value_state_embed.") for name in trainable))
        self.assertTrue(any(name.startswith("core_role_value_state_answer_") for name in trainable))
        self.assertTrue(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_embed.weight.requires_grad)
        self.assertTrue(model.core_primitive_role_value_head.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_answer_gate.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_role_value_answer_bridge_adapter_only_bottlenecks_shortcut(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_lm_adapter_enabled=True,
            answer_state_loop_lm_adapter_rank=4,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=16,
            core_role_value_state_answer_bridge_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "role_value_answer_bridge_adapter_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("answer_state_loop_lm_adapter_")
                or name.startswith("core_role_value_state_embed.")
                or name.startswith("core_role_value_state_answer_")
                for name in trainable
            )
        )
        self.assertTrue(model.answer_state_loop_lm_adapter_down.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_embed.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_answer_gate.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_role_value_vocab_renderer_only_policy_is_scoped(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=16,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_rank=4,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "role_value_vocab_renderer_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core_role_value_state_embed.")
                or name.startswith("core_role_value_state_answer_")
                or name.startswith("core_role_value_state_vocab_renderer_")
                for name in trainable
            )
        )
        self.assertTrue(model.core_role_value_state_embed.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_vocab_renderer_up.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_answer_gate.weight.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_answer_state_loop_only_policy_includes_talker_when_enabled(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_talker_enabled=True,
            answer_state_loop_talker_layers=1,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "answer_state_loop_only")

        self.assertIn("answer_state_loop_talker_gate.weight", trainable)
        self.assertTrue(model.answer_state_loop_talker_gate.weight.requires_grad)
        self.assertTrue(
            model.answer_state_loop_talker_stack.layers[0].norm1.weight.requires_grad
        )
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_answer_state_loop_only_policy_includes_mythos_update_when_enabled(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_recurrent_block_enabled=True,
            answer_state_loop_mythos_update_enabled=True,
            answer_state_loop_mythos_loop_index_enabled=True,
            answer_state_loop_mythos_lora_rank=4,
            answer_state_loop_halt_enabled=True,
            answer_state_loop_mythos_act_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "answer_state_loop_only")

        self.assertIn("answer_state_loop_mythos_log_A", trainable)
        self.assertIn("answer_state_loop_mythos_log_dt", trainable)
        self.assertIn("answer_state_loop_mythos_input_B", trainable)
        self.assertIn("answer_state_loop_mythos_loop_index.weight", trainable)
        self.assertIn("answer_state_loop_mythos_lora_down.weight", trainable)
        self.assertTrue(model.answer_state_loop_mythos_log_A.requires_grad)
        self.assertTrue(model.answer_state_loop_mythos_lora_up.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        A = torch.exp(
            -torch.exp(
                model.answer_state_loop_mythos_log_dt
                + model.answer_state_loop_mythos_log_A
            )
        )
        self.assertTrue(torch.all(A > 0.0))
        self.assertTrue(torch.all(A < 1.0))

    def test_answer_state_loop_mythos_update_runs_forward_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=3,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_recurrent_block_enabled=True,
            answer_state_loop_mythos_update_enabled=True,
            answer_state_loop_mythos_loop_index_enabled=True,
            answer_state_loop_mythos_lora_rank=4,
            answer_state_loop_halt_enabled=True,
            answer_state_loop_mythos_act_enabled=True,
            answer_state_loop_next_token_decoder_enabled=True,
            answer_state_loop_next_token_decoder_layers=1,
        )
        model = QTRMMultimodalModel(cfg)

        out = model(torch.randint(0, cfg.vocab_size, (2, 6)))

        self.assertEqual(out["answer_state_loop_logits"].shape, (2, 6, cfg.vocab_size))
        self.assertEqual(out["answer_state_loop_halt_logits"].shape, (2, 3))
        self.assertEqual(out["answer_state_loop_recurrent_gate_mean"].shape, (2, 3))

    def test_answer_state_loop_talker_only_policy_freezes_answer_loop(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_talker_enabled=True,
            answer_state_loop_talker_layers=1,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "answer_state_loop_talker_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("answer_state_loop_talker_") for name in trainable))
        self.assertTrue(model.answer_state_loop_talker_gate.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_answer_state_loop_lm_adapter_only_policy_freezes_answer_core(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_lm_adapter_enabled=True,
            answer_state_loop_lm_adapter_rank=4,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "answer_state_loop_lm_adapter_only",
        )

        self.assertEqual(
            set(trainable),
            {
                "answer_state_loop_lm_adapter_down.weight",
                "answer_state_loop_lm_adapter_up.weight",
            },
        )
        self.assertTrue(model.answer_state_loop_lm_adapter_down.weight.requires_grad)
        self.assertTrue(model.answer_state_loop_lm_adapter_up.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_answer_state_loop_next_token_decoder_only_policy_freezes_answer_core(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_next_token_decoder_enabled=True,
            answer_state_loop_next_token_decoder_layers=1,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "answer_state_loop_next_token_decoder_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("answer_state_loop_next_token_decoder_")
                for name in trainable
            )
        )
        self.assertTrue(
            model.answer_state_loop_next_token_decoder_gate.weight.requires_grad
        )
        self.assertFalse(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_answer_state_loop_next_token_decoder_only_policy_requires_decoder(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        with self.assertRaisesRegex(ValueError, "next_token_decoder"):
            configure_trainable_parameters(
                model,
                "answer_state_loop_next_token_decoder_only",
            )

    def test_lm_head_only_policy_requires_untied_lm_head(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            tie_embeddings=False,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "lm_head_only")

        self.assertEqual(trainable, ["lm_head.weight"])
        self.assertTrue(model.lm_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_lm_head_only_policy_rejects_tied_embedding_model(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            tie_embeddings=True,
        )
        model = QTRMMultimodalModel(cfg)

        with self.assertRaisesRegex(ValueError, "tie_embeddings=false"):
            configure_trainable_parameters(model, "lm_head_only")

    def test_answer_state_loop_hidden_bridge_only_policy_freezes_head_and_core(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            answer_state_loop_hidden_bridge_enabled=True,
            answer_state_loop_hidden_bridge_hidden_dim=16,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "answer_state_loop_hidden_bridge_only",
        )

        self.assertEqual(
            set(trainable),
            {
                "answer_state_loop_hidden_bridge_norm.weight",
                "answer_state_loop_hidden_bridge_down.weight",
                "answer_state_loop_hidden_bridge_down.bias",
                "answer_state_loop_hidden_bridge_up.weight",
                "answer_state_loop_hidden_bridge_up.bias",
            },
        )
        self.assertTrue(model.answer_state_loop_hidden_bridge_up.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_answer_state_loop_hidden_bridge_only_policy_requires_bridge(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        with self.assertRaisesRegex(ValueError, "hidden_bridge"):
            configure_trainable_parameters(
                model,
                "answer_state_loop_hidden_bridge_only",
            )

    def test_transition_state_sequence_only_policy_freezes_core_and_code_policy(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_state_sequence_enabled=True,
            transition_state_sequence_max_tokens=5,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "transition_state_sequence_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("transition_state_sequence_") for name in trainable)
        )
        self.assertTrue(model.transition_state_sequence_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_transition_state_joint_only_policy_freezes_core_and_answer_loop(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_state_joint_prompt_context_enabled=True,
            transition_state_joint_prompt_token_attention_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "transition_state_joint_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("transition_state_joint_") for name in trainable)
        )
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(
            model.transition_state_joint_prompt_context_proj.weight.requires_grad
        )
        self.assertTrue(model.transition_state_joint_prompt_cross.q_proj.weight.requires_grad)
        self.assertTrue(
            model.transition_state_joint_prompt_cross_proj.weight.requires_grad
        )
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_transition_state_joint_prompt_context_conditions_logits(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

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
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=8,
            max_visual_tokens=2,
            transition_state_joint_enabled=True,
            transition_state_joint_size=4,
            transition_state_joint_prompt_context_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.transition_state_joint_prompt_context_proj.weight.copy_(
                torch.eye(cfg.d_model)
            )
            model.transition_state_joint_head.weight.zero_()
            model.transition_state_joint_head.bias.zero_()
            model.transition_state_joint_head.weight[0, 0] = 1.0

        core_depth_states = torch.zeros(1, 2, cfg.d_model)
        empty_context = torch.zeros(1, 3, cfg.d_model)
        prompt_context = empty_context.clone()
        prompt_context[0, :, 0] = 1.0
        mask = torch.ones(1, 3)

        without_context = model._compute_transition_state_joint_logits(
            core_depth_states,
            prompt_context_seq=empty_context,
            prompt_context_mask=mask,
        )
        with_context = model._compute_transition_state_joint_logits(
            core_depth_states,
            prompt_context_seq=prompt_context,
            prompt_context_mask=mask,
        )

        self.assertTrue(torch.allclose(without_context, torch.zeros_like(without_context)))
        self.assertGreater(float(with_context[0, 0, 0]), 0.0)

    def test_transition_state_joint_prompt_token_attention_is_zero_init_residual(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=4,
            n_heads=2,
            n_kv_heads=1,
            d_ff=16,
            n_prelude_layers=0,
            n_core_layers=0,
            n_coda_layers=0,
            workspace_tokens=2,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=4,
            max_visual_tokens=2,
            transition_state_joint_enabled=True,
            transition_state_joint_size=4,
            transition_state_joint_prompt_context_enabled=True,
            transition_state_joint_prompt_token_attention_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.transition_state_joint_prompt_context_proj.weight.zero_()
            model.transition_state_joint_head.weight.zero_()
            model.transition_state_joint_head.bias.zero_()
            model.transition_state_joint_head.weight[0, 0] = 1.0

        core_depth_states = torch.zeros(1, 1, cfg.d_model)
        core_depth_states[0, 0, 0] = 2.0
        prompt_context = torch.zeros(1, 2, cfg.d_model)
        prompt_context[0, 0, 0] = 2.0
        prompt_context[0, 1, 1] = 2.0
        mask = torch.ones(1, 2)

        baseline_logits = model._compute_transition_state_joint_logits(
            core_depth_states,
            prompt_context_seq=None,
            prompt_context_mask=None,
        )
        zero_init_logits = model._compute_transition_state_joint_logits(
            core_depth_states,
            prompt_context_seq=prompt_context,
            prompt_context_mask=mask,
        )

        with torch.no_grad():
            eye = torch.eye(cfg.d_model)
            model.transition_state_joint_prompt_cross.q_proj.weight.copy_(eye)
            model.transition_state_joint_prompt_cross.k_proj.weight.copy_(eye)
            model.transition_state_joint_prompt_cross.v_proj.weight.copy_(eye)
            model.transition_state_joint_prompt_cross.o_proj.weight.copy_(eye)
            model.transition_state_joint_prompt_cross_proj.weight.copy_(eye)

        cross_logits = model._compute_transition_state_joint_logits(
            core_depth_states,
            prompt_context_seq=prompt_context,
            prompt_context_mask=mask,
        )

        self.assertTrue(torch.allclose(zero_init_logits, baseline_logits))
        self.assertFalse(torch.allclose(cross_logits, baseline_logits))

    def test_core_and_value_state_policy_preserves_joint_code_supervision_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_value_state_enabled=True,
            transition_value_state_max_tokens=6,
            transition_value_state_vocab_size=12,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_value_state")

        self.assertTrue(trainable)
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(model.transition_value_state_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.prelude.parameters()).requires_grad)

    def test_primitive_transition_only_policy_preserves_existing_joint_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=11,
            primitive_transition_prompt_context_enabled=True,
            primitive_transition_prompt_token_attention_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "primitive_transition_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("primitive_transition_") for name in trainable))
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_primitive_transition_and_finality_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_state_finality_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=11,
            primitive_transition_prompt_context_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "primitive_transition_and_finality"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("primitive_transition_")
                or name.startswith("transition_state_finality_")
                for name in trainable
            )
        )
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertTrue(model.transition_state_finality_head.weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_transition_source_router_only_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=12,
            transition_source_router_enabled=True,
            transition_source_router_prompt_context_enabled=True,
            transition_source_router_prompt_token_attention_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "transition_source_router_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("transition_source_router_") for name in trainable))
        self.assertTrue(model.transition_source_router_head[0].weight.requires_grad)
        self.assertTrue(model.transition_source_router_prompt_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_core_primitive_update_gate_only_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=8,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=4,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
            core_primitive_role_value_update_gate_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "core_primitive_role_value_update_gate_only"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("core_primitive_role_value_update_gate.") for name in trainable)
        )
        self.assertTrue(model.core_primitive_role_value_update_gate.weight.requires_grad)
        self.assertFalse(model.core_primitive_role_value_head.weight.requires_grad)
        self.assertFalse(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_transition_state_joint_operation_residual_only_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_state_joint_operation_residual_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=12,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "transition_state_joint_operation_residual_only"
        )

        self.assertEqual(trainable, ["transition_state_joint_operation_residual.weight"])
        self.assertTrue(model.transition_state_joint_operation_residual.weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_transition_phase_and_joint_phase_residual_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_phase_enabled=True,
            transition_phase_num_classes=2,
            transition_phase_prompt_context_enabled=True,
            transition_phase_prompt_token_attention_enabled=True,
            transition_state_joint_phase_residual_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "transition_phase_and_joint_phase_residual"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("transition_phase_")
                or name.startswith("transition_state_joint_phase_residual")
                for name in trainable
            )
        )
        self.assertTrue(model.transition_phase_head[0].weight.requires_grad)
        self.assertTrue(model.transition_phase_prompt_cross.q_proj.weight.requires_grad)
        self.assertTrue(model.transition_state_joint_phase_residual[0].weight.requires_grad)
        self.assertTrue(model.transition_state_joint_phase_residual[-1].weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_transition_phase_only_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_phase_enabled=True,
            transition_phase_num_classes=2,
            transition_phase_prompt_context_enabled=True,
            transition_phase_global_prompt_query_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "transition_phase_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("transition_phase_") for name in trainable))
        self.assertTrue(model.transition_phase_head[0].weight.requires_grad)
        self.assertTrue(model.transition_phase_global_query.requires_grad)
        self.assertTrue(model.transition_phase_global_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_transition_state_joint_phase_residual_only_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_phase_enabled=True,
            transition_phase_num_classes=2,
            transition_state_joint_phase_residual_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "transition_state_joint_phase_residual_only"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("transition_state_joint_phase_residual")
                for name in trainable
            )
        )
        self.assertTrue(model.transition_state_joint_phase_residual[0].weight.requires_grad)
        self.assertFalse(model.transition_phase_head[0].weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_transition_state_code_and_joint_code_residual_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_code_enabled=True,
            transition_state_codebook_size=5,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_state_joint_code_residual_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "transition_state_code_and_joint_code_residual"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("transition_state_code_")
                or name.startswith("transition_state_joint_code_residual")
                for name in trainable
            )
        )
        self.assertTrue(model.transition_state_code_head.weight.requires_grad)
        self.assertTrue(model.transition_state_joint_code_residual.weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)

    def test_token_numeric_binder_primitive_policy_trains_internal_binder(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

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
            token_numeric_value_embedding_enabled=True,
            token_numeric_value_vocab_size=32,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=5,
            core_role_value_state_vocab_size=16,
            core_source_position_binder_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=6,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "token_numeric_context_binder_primitive_role_value_state_machine"
        )

        self.assertTrue(any(name.startswith("token_numeric_value_embed.") for name in trainable))
        self.assertTrue(any(name.startswith("prelude.") for name in trainable))
        self.assertTrue(any(name.startswith("core_source_position_binder_") for name in trainable))
        self.assertTrue(any(name.startswith("core_primitive_role_value_") for name in trainable))
        self.assertTrue(model.core_source_position_binder_head[-1].weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_value_embed.weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_state_gate.requires_grad)
        self.assertTrue(model.token_numeric_value_embed.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_prompt_context_binder_primitive_policy_trains_internal_binder(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

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
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=5,
            core_role_value_state_vocab_size=16,
            core_source_position_binder_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=6,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "prompt_context_binder_primitive_role_value_state_machine",
        )

        self.assertTrue(any(name.startswith("prelude.") for name in trainable))
        self.assertTrue(any(name.startswith("core_source_position_binder_") for name in trainable))
        self.assertTrue(any(name.startswith("core_primitive_role_value_") for name in trainable))
        self.assertTrue(model.core_source_position_binder_head[-1].weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_value_embed.weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_state_gate.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_token_numeric_source_slot_policy_trains_source_slots(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

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
            token_numeric_source_slot_embedding_enabled=True,
            token_numeric_source_slot_vocab_size=32,
            token_numeric_source_slot_max_slots=5,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=5,
            core_role_value_state_vocab_size=16,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=6,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "token_numeric_source_slot_context_primitive_role_value_state_machine",
        )

        self.assertTrue(
            any(name.startswith("token_numeric_source_slot_embed.") for name in trainable)
        )
        self.assertTrue(any(name.startswith("prelude.") for name in trainable))
        self.assertTrue(any(name.startswith("core_primitive_role_value_") for name in trainable))
        self.assertTrue(model.token_numeric_source_slot_embed.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_token_numeric_source_slot_binder_policy_trains_internal_binder(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

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
            token_numeric_source_slot_embedding_enabled=True,
            token_numeric_source_slot_vocab_size=32,
            token_numeric_source_slot_max_slots=5,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=5,
            core_role_value_state_vocab_size=16,
            core_source_position_binder_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=6,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "token_numeric_source_slot_context_binder_primitive_role_value_state_machine",
        )

        self.assertTrue(
            any(name.startswith("token_numeric_source_slot_embed.") for name in trainable)
        )
        self.assertTrue(any(name.startswith("core_source_position_binder_") for name in trainable))
        self.assertTrue(model.token_numeric_source_slot_embed.weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_head[-1].weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_value_embed.weight.requires_grad)
        self.assertTrue(model.core_source_position_binder_state_gate.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_transition_value_state_only_policy_freezes_core_and_action_head(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            transition_value_state_enabled=True,
            transition_value_state_max_tokens=6,
            transition_value_state_vocab_size=12,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "transition_value_state_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("transition_value_state_") for name in trainable))
        self.assertTrue(model.transition_value_state_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_factorized_value_state_only_policy_freezes_core_and_action_head(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            factorized_value_state_enabled=True,
            factorized_value_state_max_tokens=6,
            factorized_value_state_vocab_size=12,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "factorized_value_state_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("factorized_value_state_") for name in trainable))
        self.assertTrue(model.factorized_value_state_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_role_value_state_only_policy_trains_role_and_value_slots(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            factorized_value_state_enabled=True,
            factorized_value_state_max_tokens=6,
            factorized_value_state_vocab_size=12,
            role_value_state_enabled=True,
            role_value_state_num_roles=10,
            role_value_state_vocab_size=128,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "role_value_state_only")

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("factorized_value_state_")
                or name.startswith("role_value_state_")
                for name in trainable
            )
        )
        self.assertTrue(model.role_value_state_head.weight.requires_grad)
        self.assertTrue(model.factorized_value_state_init.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_role_value_state_only_policy_freezes_core_and_action_head(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_role_value_transition_enabled=True,
            core_state_carry_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_role_value_state_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core_role_value_state_")
                or name.startswith("core_role_value_transition_")
                for name in trainable
            )
        )
        self.assertTrue(model.core_role_value_state_head.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_embed.weight.requires_grad)
        self.assertTrue(model.core_role_value_transition_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_typed_algorithmic_value_state_only_policy_trains_field_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            text_position_embed_enabled=True,
            core_depth_readout_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            typed_algorithmic_value_state_enabled=True,
            typed_algorithmic_value_state_max_list_slots=4,
            typed_algorithmic_value_state_offset_vocab_size=17,
            typed_algorithmic_value_state_scalar_vocab_size=19,
            typed_algorithmic_value_state_recurrent_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "typed_algorithmic_value_state_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("typed_algorithmic_") for name in trainable))
        self.assertTrue(model.typed_algorithmic_kind_head.weight.requires_grad)
        self.assertTrue(model.typed_algorithmic_recurrent_gate.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_and_typed_algorithmic_policy_trains_core_and_field_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            text_position_embed_enabled=True,
            core_depth_readout_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            typed_algorithmic_value_state_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_and_typed_algorithmic_value_state",
        )

        self.assertTrue(trainable)
        self.assertTrue(model.core_depth_readout_query.requires_grad)
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(model.typed_algorithmic_kind_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_answer_loop_and_typed_algorithmic_policy_trains_bridge_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

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
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            typed_algorithmic_value_state_enabled=True,
            typed_algorithmic_value_state_answer_bridge_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_answer_loop_and_typed_algorithmic_value_state",
        )

        self.assertTrue(trainable)
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.answer_state_loop_gate.weight.requires_grad)
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(model.typed_algorithmic_kind_head.weight.requires_grad)
        self.assertTrue(
            model.typed_algorithmic_value_state_answer_bridge_proj.weight.requires_grad
        )
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_token_embed_core_and_typed_policy_opens_token_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            text_position_embed_enabled=True,
            core_depth_readout_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            typed_algorithmic_value_state_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "token_embed_core_and_typed_algorithmic_value_state",
        )

        self.assertTrue(trainable)
        self.assertTrue(model.text_embed.weight.requires_grad)
        self.assertTrue(model.text_position_embed.weight.requires_grad)
        self.assertTrue(model.prelude.layers[0].norm1.weight.requires_grad)
        self.assertTrue(model.workspace.workspace.requires_grad)
        self.assertTrue(model.core_depth_readout_query.requires_grad)
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(model.typed_algorithmic_kind_head.weight.requires_grad)

    def test_token_core_answer_loop_and_typed_policy_opens_full_lm_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            tie_embeddings=False,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            text_position_embed_enabled=True,
            core_depth_readout_enabled=True,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            typed_algorithmic_value_state_enabled=True,
            typed_algorithmic_value_state_answer_bridge_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "token_core_answer_loop_and_typed_algorithmic_value_state",
        )

        self.assertTrue(trainable)
        self.assertTrue(model.text_embed.weight.requires_grad)
        self.assertTrue(model.text_position_embed.weight.requires_grad)
        self.assertTrue(model.prelude.layers[0].norm1.weight.requires_grad)
        self.assertTrue(model.workspace.workspace.requires_grad)
        self.assertTrue(model.core_depth_readout_query.requires_grad)
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.answer_state_loop_gate.weight.requires_grad)
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(model.typed_algorithmic_kind_head.weight.requires_grad)
        self.assertTrue(
            model.typed_algorithmic_value_state_answer_bridge_proj.weight.requires_grad
        )
        self.assertFalse(model.lm_head.weight.requires_grad)

    def test_primitive_and_typed_algorithmic_policy_trains_only_that_state_path(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=5,
            typed_algorithmic_value_state_enabled=True,
            typed_algorithmic_value_state_recurrent_enabled=True,
            typed_algorithmic_value_state_primitive_conditioning_enabled=True,
            typed_algorithmic_value_state_subregisters_enabled=True,
            typed_algorithmic_value_state_residual_feedback_enabled=True,
            typed_algorithmic_value_state_residual_delta_enabled=True,
            typed_algorithmic_value_state_scalar_offset_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "primitive_and_typed_algorithmic_state_machine",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("primitive_transition_")
                or name.startswith("typed_algorithmic_")
                for name in trainable
            )
        )
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertTrue(model.typed_algorithmic_kind_head.weight.requires_grad)
        self.assertTrue(
            model.typed_algorithmic_recurrent_primitive_proj.weight.requires_grad
        )
        self.assertTrue(
            model.typed_algorithmic_scalar_subregister_update[0].weight.requires_grad
        )
        self.assertTrue(
            model.typed_algorithmic_scalar_residual_feedback_proj.weight.requires_grad
        )
        self.assertTrue(
            model.typed_algorithmic_scalar_residual_delta_head.weight.requires_grad
        )
        self.assertTrue(model.typed_algorithmic_scalar_offset_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_and_role_value_state_policy_trains_core_action_and_role_tokens(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_role_value_transition_enabled=True,
            core_state_carry_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_and_role_value_state",
        )

        self.assertTrue(trainable)
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.answer_state_loop_gate.weight.requires_grad)
        self.assertTrue(model.transition_state_joint_head.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_head.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_embed.weight.requires_grad)
        self.assertTrue(model.core_role_value_transition_head.weight.requires_grad)
        self.assertTrue(model.core.state_carry_update[0].weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertTrue(
            all(
                name.startswith("core.")
                or name.startswith("answer_state_loop_")
                or name.startswith("transition_state_")
                or name.startswith("core_role_value_state_")
                or name.startswith("core_role_value_transition_")
                for name in trainable
            )
        )

    def test_core_state_carry_only_policy_freezes_existing_core_and_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_state_carry_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_state_carry_only")

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.state_carry_norm.")
                or name.startswith("core.state_carry_update.")
                or name.startswith("core.state_carry_gate.")
                for name in trainable
            )
        )
        self.assertTrue(model.core.state_carry_norm.weight.requires_grad)
        self.assertTrue(model.core.state_carry_update[0].weight.requires_grad)
        self.assertTrue(model.core.state_carry_gate.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_gate.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_role_value_delta_only_policy_freezes_existing_core_and_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_role_value_delta_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_role_value_delta_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("core_role_value_delta_") for name in trainable)
        )
        self.assertTrue(model.core_role_value_delta_update[0].weight.requires_grad)
        self.assertTrue(model.core_role_value_delta_gate.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_gate.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_value_delta_code_only_policy_freezes_existing_core_and_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_value_delta_code_enabled=True,
            core_value_delta_codebook_size=128,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_value_delta_code_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("core_value_delta_code_") for name in trainable)
        )
        self.assertTrue(model.core_value_delta_code_head.weight.requires_grad)
        self.assertTrue(model.core_value_delta_code_embed.weight.requires_grad)
        self.assertTrue(model.core_value_delta_code_gate.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_gate.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_typed_register_executor_only_policy_freezes_existing_core_and_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_typed_register_executor_enabled=True,
            core_typed_register_num_operations=6,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_typed_register_executor_only",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("core_typed_register_") for name in trainable)
        )
        self.assertTrue(model.core_typed_register_operation_head.weight.requires_grad)
        self.assertTrue(model.core_typed_register_update[0].weight.requires_grad)
        self.assertTrue(model.core_typed_register_value_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.answer_state_loop_gate.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_typed_register_executor_and_prompt_extract_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            transition_state_joint_enabled=True,
            transition_state_joint_size=10,
            core_role_value_state_enabled=True,
            core_role_value_state_prompt_extract_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=128,
            core_typed_register_executor_enabled=True,
            core_typed_register_num_operations=5,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_typed_register_executor_and_prompt_extract",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core_typed_register_")
                or name.startswith("core_role_value_state_prompt_")
                for name in trainable
            )
        )
        self.assertTrue(model.core_typed_register_operation_head.weight.requires_grad)
        self.assertTrue(model.core_role_value_state_prompt_cross.q_proj.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(model.core_role_value_state_head.weight.requires_grad)
        self.assertFalse(model.transition_state_joint_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_and_primitive_transition_policy_freezes_everything_else(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=11,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_primitive_transition")

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.")
                or name.startswith("primitive_transition_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_numeric_projector_primitive_role_value_policy_trains_projector(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=16,
            core_role_value_state_prompt_extract_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=12,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "numeric_projector_primitive_role_value_state_machine",
        )

        self.assertTrue(any(name.startswith("projector.visual_") for name in trainable))
        self.assertTrue(any(name.startswith("core_primitive_role_value_") for name in trainable))
        self.assertTrue(model.projector.visual_proj.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_token_numeric_primitive_role_value_policy_trains_token_numeric_embed(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            token_numeric_value_embedding_enabled=True,
            token_numeric_value_vocab_size=32,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=16,
            core_role_value_state_prompt_extract_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=12,
            core_primitive_role_value_executor_enabled=True,
            core_primitive_role_value_mlp_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "token_numeric_context_primitive_role_value_state_machine",
        )

        self.assertTrue(any(name.startswith("token_numeric_value_embed.") for name in trainable))
        self.assertTrue(any(name.startswith("prelude.") for name in trainable))
        self.assertTrue(any(name.startswith("core_primitive_role_value_") for name in trainable))
        self.assertTrue(model.token_numeric_value_embed.weight.requires_grad)
        self.assertTrue(next(model.prelude.parameters()).requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_primitive_transition_and_finality_policy_freezes_everything_else(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=11,
            transition_state_finality_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "core_primitive_transition_and_finality"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.")
                or name.startswith("primitive_transition_")
                or name.startswith("transition_state_finality_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertTrue(model.transition_state_finality_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_core_context_primitive_transition_and_finality_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_context_enabled=True,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=11,
            transition_state_finality_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "core_context_primitive_transition_and_finality"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.context_")
                or name.startswith("primitive_transition_")
                or name.startswith("transition_state_finality_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.context_cross_l.q_proj.weight.requires_grad)
        self.assertTrue(model.core.context_gate_h.weight.requires_grad)
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertTrue(model.transition_state_finality_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(next(model.core.fast_stack.parameters()).requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_transition_feedback_and_readouts_policy_is_narrow(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=2,
            visual_dim=16,
            max_visual_tokens=4,
            core_transition_feedback_enabled=True,
            core_transition_feedback_num_operations=11,
            primitive_transition_enabled=True,
            primitive_transition_num_operations=11,
            transition_state_finality_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model, "core_transition_feedback_and_readouts"
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.transition_feedback_")
                or name.startswith("primitive_transition_")
                or name.startswith("transition_state_finality_")
                for name in trainable
            )
        )
        self.assertTrue(model.core.transition_feedback_operation_head.weight.requires_grad)
        self.assertTrue(model.core.transition_feedback_gate.weight.requires_grad)
        self.assertTrue(model.primitive_transition_operation_head[0].weight.requires_grad)
        self.assertTrue(model.transition_state_finality_head.weight.requires_grad)
        self.assertFalse(model.core.z_l_init.requires_grad)
        self.assertFalse(next(model.core.fast_stack.parameters()).requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)

    def test_core_answer_state_loop_world_model_policy_freezes_everything_else(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=3,
            visual_dim=16,
            max_visual_tokens=4,
            answer_state_loop_enabled=True,
            core_world_model_enabled=True,
            core_world_model_predictor_dim=64,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_and_answer_state_loop_and_world_model",
        )

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("core.")
                or name.startswith("answer_state_loop_")
                or name.startswith("core_world_model.")
                for name in trainable
            )
        )
        self.assertTrue(model.core.z_l_init.requires_grad)
        self.assertTrue(model.answer_state_loop_cross.q_proj.weight.requires_grad)
        self.assertTrue(model.core_world_model.predictor.context_proj.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_workspace_gate_only_policy_freezes_everything_except_workspace_gate(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            workspace_layers=2,
            workspace_memory_gate_enabled=True,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "workspace_gate_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("workspace.layers.") for name in trainable))
        self.assertTrue(any(".update_gate." in name for name in trainable))
        self.assertTrue(any(".reset_gate." in name for name in trainable))
        self.assertTrue(any(".candidate." in name for name in trainable))
        self.assertTrue(model.workspace.layers[0].update_gate.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_workspace_gate_only_policy_requires_gated_workspace(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        model = QTRMMultimodalModel(
            QTRMConfig(
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
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                workspace_memory_gate_enabled=False,
            )
        )

        with self.assertRaisesRegex(ValueError, "workspace_gate_only"):
            configure_trainable_parameters(model, "workspace_gate_only")

    def test_generation_verifier_only_policy_freezes_everything_except_verifier_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            generation_verifier_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "generation_verifier_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("generation_") for name in trainable))
        self.assertIn("generation_repeat_head.weight", trainable)
        self.assertIn("generation_stop_head.weight", trainable)
        self.assertIn("generation_quality_head.weight", trainable)
        self.assertTrue(model.generation_repeat_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_controller_only_policy_freezes_everything_except_controller_heads(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "controller_only")

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("ctrl.") for name in trainable))
        self.assertTrue(model.ctrl.action.weight.requires_grad)
        self.assertTrue(model.ctrl.verify.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_controller_only_policy_includes_learned_signal_heads_when_enabled(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
            controller_signal_enabled=True,
            controller_signal_source="learned_core",
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "controller_only")

        self.assertIn("controller_signal_proj.weight", trainable)
        self.assertIn("controller_signal_head.weight", trainable)
        self.assertTrue(model.controller_signal_proj.weight.requires_grad)
        self.assertTrue(model.controller_signal_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.core.parameters()).requires_grad)

    def test_controller_signal_head_only_policy_freezes_action_mapping(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
            controller_signal_enabled=True,
            controller_signal_source="learned_core",
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "controller_signal_head_only")

        self.assertEqual(trainable, ["controller_signal_head.weight", "controller_signal_head.bias"])
        self.assertTrue(model.controller_signal_head.weight.requires_grad)
        self.assertFalse(model.controller_signal_proj.weight.requires_grad)
        self.assertFalse(model.ctrl.action.weight.requires_grad)

    def test_core_and_controller_signal_head_policy_trains_core_router_only(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            num_actions=10,
            controller_signal_enabled=True,
            controller_signal_source="learned_core",
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "core_and_controller_signal_head",
        )

        self.assertIn("controller_signal_head.weight", trainable)
        self.assertTrue(any(name.startswith("core.") for name in trainable))
        self.assertTrue(model.controller_signal_head.weight.requires_grad)
        self.assertTrue(next(model.core.parameters()).requires_grad)
        self.assertFalse(model.controller_signal_proj.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(model.ctrl.action.weight.requires_grad)

    def test_core_and_temporal_spatial_context_policy_trains_core_and_context_encoder(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            temporal_spatial_context_enabled=True,
            temporal_spatial_context_dim=6,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "core_and_temporal_spatial_context")

        self.assertIn("temporal_spatial_context_proj.weight", trainable)
        self.assertIn("temporal_spatial_context_pos", trainable)
        self.assertTrue(any(name.startswith("core.") for name in trainable))
        self.assertTrue(model.temporal_spatial_context_proj.weight.requires_grad)
        self.assertTrue(model.temporal_spatial_context_pos.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_answer_bottleneck_evidence_only_policy_freezes_base_model(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            answer_bottleneck_enabled=True,
            evidence_bottleneck_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(
            model,
            "answer_bottleneck_evidence_only",
        )

        self.assertTrue(trainable)
        self.assertIn("answer_bottleneck_cross.q_proj.weight", trainable)
        self.assertIn("evidence_support_head.weight", trainable)
        self.assertIn("evidence_causal_gate_head.bias", trainable)
        self.assertTrue(model.answer_bottleneck_cross.q_proj.weight.requires_grad)
        self.assertTrue(model.evidence_causal_gate_head.bias.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.prelude.parameters()).requires_grad)
        self.assertFalse(next(model.core.parameters()).requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_answer_bottleneck_evidence_only_policy_requires_answer_bottleneck(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        model = QTRMMultimodalModel(
            QTRMConfig(
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
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                answer_bottleneck_enabled=False,
                evidence_bottleneck_enabled=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "answer_bottleneck_evidence_only"):
            configure_trainable_parameters(model, "answer_bottleneck_evidence_only")

    def test_evidence_span_reader_only_policy_freezes_base_model(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.training.train import configure_trainable_parameters

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
            evidence_span_reader_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)

        trainable = configure_trainable_parameters(model, "evidence_span_reader_only")

        self.assertIn("evidence_span_query_proj.weight", trainable)
        self.assertIn("evidence_span_start_key.weight", trainable)
        self.assertIn("evidence_span_no_answer_head.bias", trainable)
        self.assertIn("evidence_support_head.weight", trainable)
        self.assertIn("evidence_refute_head.weight", trainable)
        self.assertIn("evidence_missing_head.weight", trainable)
        self.assertIn("evidence_causal_gate_head.weight", trainable)
        self.assertIn("projector.visual_proj.weight", trainable)
        self.assertTrue(model.evidence_span_start_key.weight.requires_grad)
        self.assertTrue(model.evidence_support_head.weight.requires_grad)
        self.assertFalse(model.text_embed.weight.requires_grad)
        self.assertFalse(next(model.prelude.parameters()).requires_grad)
        self.assertFalse(next(model.core.parameters()).requires_grad)
        self.assertFalse(next(model.coda.parameters()).requires_grad)

    def test_prepare_donor_batch_encodes_workspace_memory_inputs(self):
        from qtrm_mm.training.train import prepare_donor_batch, strip_training_only_batch_keys

        class FakeDonor:
            def encode_inputs(self, *, input_ids, attention_mask=None, return_logits=False):
                states = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
                out = {
                    "text_states": states,
                    "attention_mask": attention_mask,
                    "visual_features": None,
                }
                if return_logits:
                    out["logits"] = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
                return out

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "workspace_input_ids": torch.tensor([[4, 5, 0]]),
            "workspace_attention_mask": torch.tensor([[1, 1, 0]]),
        }

        model_batch = strip_training_only_batch_keys(batch)
        extra = prepare_donor_batch(FakeDonor(), batch, return_logits=True)

        self.assertNotIn("workspace_input_ids", model_batch)
        self.assertNotIn("workspace_attention_mask", model_batch)
        self.assertIn("workspace_text_states", extra)
        self.assertIn("workspace_attention_mask", extra)
        self.assertEqual(extra["workspace_text_states"].shape, (1, 3, 4))
        self.assertTrue(torch.equal(extra["workspace_attention_mask"], batch["workspace_attention_mask"]))

        zero_workspace = dict(batch)
        zero_workspace["workspace_attention_mask"] = torch.zeros_like(batch["workspace_attention_mask"])
        zero_extra = prepare_donor_batch(FakeDonor(), zero_workspace, return_logits=False)

        self.assertNotIn("workspace_text_states", zero_extra)

    def test_prepare_donor_batch_encodes_rejected_preference_inputs(self):
        from qtrm_mm.training.train import prepare_donor_batch

        class FakeDonor:
            def encode_inputs(self, *, input_ids, attention_mask=None, return_logits=False):
                out = {
                    "text_states": torch.ones(input_ids.shape[0], input_ids.shape[1], 4),
                    "attention_mask": attention_mask,
                    "visual_features": None,
                }
                if return_logits:
                    out["logits"] = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
                return out

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "preference_rejected_input_ids": torch.tensor([[1, 4, 4]]),
            "preference_rejected_attention_mask": torch.tensor([[1, 1, 1]]),
        }

        extra = prepare_donor_batch(FakeDonor(), batch, return_logits=True)

        self.assertIn("preference_rejected_text_states", extra)
        self.assertIn("preference_rejected_donor_logits", extra)
        self.assertEqual(extra["preference_rejected_text_states"].shape, (1, 3, 4))
        self.assertEqual(extra["preference_rejected_donor_logits"].shape, (1, 3, 8))

    def test_prepare_donor_batch_encodes_counterfactual_workspace_inputs(self):
        from qtrm_mm.training.train import prepare_donor_batch, strip_training_only_batch_keys

        class FakeDonor:
            def encode_inputs(self, *, input_ids, attention_mask=None, return_logits=False):
                out = {
                    "text_states": torch.ones(input_ids.shape[0], input_ids.shape[1], 4),
                    "attention_mask": attention_mask,
                    "visual_features": None,
                }
                if return_logits:
                    out["logits"] = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
                return out

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "workspace_input_ids": torch.tensor([[4, 5, 0]]),
            "workspace_attention_mask": torch.tensor([[1, 1, 0]]),
            "workspace_counterfactual_input_ids": torch.tensor([[6, 7, 0]]),
            "workspace_counterfactual_attention_mask": torch.tensor([[1, 1, 0]]),
        }

        model_batch = strip_training_only_batch_keys(batch)
        extra = prepare_donor_batch(FakeDonor(), batch, return_logits=False)

        self.assertNotIn("workspace_counterfactual_input_ids", model_batch)
        self.assertNotIn("workspace_counterfactual_attention_mask", model_batch)
        self.assertIn("workspace_counterfactual_text_states", extra)
        self.assertIn("workspace_counterfactual_attention_mask", extra)
        self.assertEqual(extra["workspace_counterfactual_text_states"].shape, (1, 3, 4))

    def test_memory_halt_preserve_config_uses_halt_only_training(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_memory_halt_preserve_s050.yaml")

        self.assertTrue(cfg.model.core_halt_enabled)
        self.assertEqual(cfg.model.outer_steps, 2)
        self.assertEqual(cfg.model.qtrm_logits_scale, 0.5)
        self.assertEqual(cfg.train.trainable_param_policy, "core_halt_only")
        self.assertEqual(cfg.train.core_halt_target_mode, "teacher_depth")
        self.assertGreater(cfg.train.loss_core_halt_weight, 0.0)
        self.assertEqual(cfg.train.loss_jepa_weight, 0.0)
        self.assertEqual(cfg.train.loss_aux_weight, 0.0)
        self.assertIn("memory_halt_preserve", cfg.train.out_dir)

    def test_memory_gated_workspace_config_trains_only_workspace_gate(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_memory_gated_workspace_s050.yaml")

        self.assertTrue(cfg.model.workspace_memory_gate_enabled)
        self.assertEqual(cfg.model.workspace_memory_gate_init_bias, -2.0)
        self.assertEqual(cfg.model.qtrm_logits_scale, 0.5)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.train.trainable_param_policy, "workspace_gate_only")
        self.assertEqual(cfg.train.loss_jepa_weight, 0.0)
        self.assertEqual(cfg.train.loss_aux_weight, 0.0)
        self.assertIn("memory_gated_workspace", cfg.train.out_dir)

    def test_generation_verifier_config_trains_only_verifier_heads(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_generation_verifier_s020.yaml")

        self.assertTrue(cfg.model.generation_verifier_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "generation_verifier_only")
        self.assertEqual(cfg.train.loss_generation_verifier_weight, 1.0)
        self.assertEqual(cfg.train.loss_jepa_weight, 0.0)
        self.assertEqual(cfg.train.loss_aux_weight, 0.0)
        self.assertIn("generation_verifier", cfg.train.out_dir)

    def test_controller_trace_config_trains_only_controller_heads(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_controller_trace_s050.yaml")

        self.assertEqual(cfg.model.num_actions, 10)
        self.assertTrue(cfg.model.core_context_enabled)
        self.assertEqual(cfg.model.outer_steps, 3)
        self.assertEqual(cfg.train.trainable_param_policy, "controller_only")
        self.assertEqual(cfg.train.loss_lm_weight, 0.0)
        self.assertEqual(cfg.train.loss_jepa_weight, 0.0)
        self.assertEqual(cfg.train.loss_aux_weight, 0.0)
        self.assertEqual(cfg.train.loss_action_policy_weight, 1.0)
        self.assertIn("controller_trace", cfg.train.out_dir)

    def test_controller_signal_config_enables_signal_adapter(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_controller_signal_s300.yaml")

        self.assertTrue(cfg.model.controller_signal_enabled)
        self.assertEqual(cfg.model.controller_signal_dim, 2)
        self.assertEqual(cfg.train.trainable_param_policy, "controller_only")
        self.assertEqual(cfg.train.loss_action_policy_weight, 1.0)
        self.assertIn("controller_signal", cfg.train.out_dir)

    def test_controller_learned_signal_config_uses_core_signal_targets(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_controller_learned_signal_s300.yaml")

        self.assertTrue(cfg.model.controller_signal_enabled)
        self.assertEqual(cfg.model.controller_signal_source, "learned_core")
        self.assertEqual(cfg.model.controller_signal_base_scale, 0.0)
        self.assertEqual(cfg.train.trainable_param_policy, "controller_only")
        self.assertEqual(cfg.train.loss_controller_signal_weight, 1.0)
        self.assertEqual(cfg.train.loss_action_policy_weight, 1.0)
        self.assertIn("controller_learned_signal", cfg.train.out_dir)

    def test_controller_learned_signal_head_config_preserves_action_mapping(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_controller_learned_signal_head_s300.yaml")

        self.assertEqual(cfg.model.controller_signal_source, "learned_core")
        self.assertEqual(cfg.train.trainable_param_policy, "controller_signal_head_only")
        self.assertEqual(cfg.train.loss_controller_signal_weight, 2.0)
        self.assertEqual(cfg.train.loss_action_policy_weight, 1.0)
        self.assertIn("controller_learned_signal_head", cfg.train.out_dir)

    def test_controller_learned_signal_readout_config_is_diagnostic(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_controller_learned_signal_readout_s300.yaml")

        self.assertEqual(cfg.model.controller_signal_source, "learned_readout")
        self.assertEqual(cfg.model.controller_signal_base_scale, 0.0)
        self.assertEqual(cfg.train.trainable_param_policy, "controller_signal_head_only")
        self.assertEqual(cfg.train.loss_controller_signal_weight, 2.0)
        self.assertEqual(cfg.train.loss_action_policy_weight, 1.0)
        self.assertIn("controller_learned_signal_readout", cfg.train.out_dir)

    def test_workspace_answer_bottleneck_causal_config_trains_only_causal_heads(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_workspace_answer_bottleneck_causal_s050.yaml")

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_requires_workspace_memory)
        self.assertTrue(cfg.model.evidence_bottleneck_enabled)
        self.assertTrue(cfg.model.evidence_bottleneck_applies_to_residual)
        self.assertEqual(cfg.train.trainable_param_policy, "answer_bottleneck_evidence_only")
        self.assertGreater(cfg.train.loss_workspace_contrastive_weight, 0.0)
        self.assertGreater(cfg.train.loss_logical_evidence_weight, 0.0)
        self.assertGreater(cfg.train.loss_causal_evidence_gate_weight, 0.0)
        self.assertGreater(cfg.train.loss_repeat_unlikelihood_weight, 0.0)
        self.assertIn("workspace_answer_bottleneck_causal", cfg.train.out_dir)

    def test_canonical_ssot_greedy_causal_config_trains_main_answer_path(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml")

        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertGreater(cfg.train.loss_lm_weight, 0.0)
        self.assertGreater(cfg.train.loss_student_lm_weight, 0.0)
        self.assertGreater(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("workspace_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("canonical_ssot_greedy_causal", cfg.train.out_dir)

    def test_canonical_ssot_core_to_text_config_forces_latent_bridge(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_canonical_ssot_coretotext_causal_s050.yaml")

        self.assertTrue(cfg.model.core_context_enabled)
        self.assertTrue(cfg.model.core_to_text_enabled)
        self.assertGreater(cfg.model.core_to_text_gate_min, 0.0)
        self.assertEqual(cfg.model.coda_attn_every, 1)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertIn("core_to_text_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("canonical_ssot_coretotext_causal", cfg.train.out_dir)

    def test_canonical_ssot_core_to_text_forced_config_strengthens_causal_bridge(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_canonical_ssot_coretotext_forced_s150.yaml")

        self.assertTrue(cfg.model.core_context_enabled)
        self.assertTrue(cfg.model.core_to_text_enabled)
        self.assertGreaterEqual(cfg.model.core_to_text_gate_min, 0.5)
        self.assertGreaterEqual(cfg.model.qtrm_logits_scale, 0.5)
        self.assertGreaterEqual(cfg.train.loss_canonical_causal_weight, 0.8)
        self.assertGreaterEqual(cfg.train.canonical_causal_margin, 0.05)
        self.assertGreaterEqual(cfg.train.steps, 100)
        self.assertIn("core_to_text_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("canonical_ssot_coretotext_forced", cfg.train.out_dir)

    def test_canonical_ssot_core_answer_bottleneck_config_uses_prompt_ssot(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_s150.yaml"
        )

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertFalse(cfg.model.answer_bottleneck_requires_workspace_memory)
        self.assertTrue(cfg.model.core_context_enabled)
        self.assertTrue(cfg.model.core_to_text_enabled)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertGreaterEqual(cfg.train.loss_canonical_causal_weight, 1.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("workspace_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("canonical_ssot_core_answer_bottleneck", cfg.train.out_dir)

    def test_canonical_ssot_core_answer_bottleneck_safe_gate_eval_preserves_donor_default(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_safe_gate_eval.yaml"
        )

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertFalse(cfg.model.answer_bottleneck_requires_workspace_memory)
        self.assertTrue(cfg.model.qtrm_residual_gate_enabled)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertLessEqual(cfg.model.qtrm_residual_gate_init_bias, -2.0)
        self.assertEqual(cfg.train.workspace_evidence_injection_mode, "ssot")
        self.assertEqual(cfg.train.steps, 0)

    def test_canonical_ssot_core_answer_bottleneck_selective_gate_keeps_donor_default(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_canonical_ssot_core_answer_bottleneck_selective_gate_s150.yaml"
        )

        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertTrue(cfg.model.qtrm_residual_gate_enabled)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.train.donor_logits_scale_start, 1.0)
        self.assertEqual(cfg.train.donor_logits_scale_end, 1.0)
        self.assertLessEqual(cfg.train.loss_canonical_causal_weight, 0.3)
        self.assertIn("workspace_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("selective_gate", cfg.train.out_dir)

    def test_donor_preserving_logit_guider_config_keeps_donor_as_base_policy(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_donor_preserving_logit_guider_s120.yaml")

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertLessEqual(cfg.model.qtrm_logits_scale, 0.35)
        self.assertLessEqual(cfg.model.qtrm_residual_clamp, 1.0)
        self.assertTrue(cfg.model.qtrm_residual_gate_enabled)
        self.assertLessEqual(cfg.model.qtrm_residual_gate_init_bias, -3.0)
        self.assertFalse(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.train.donor_logits_scale_start, 1.0)
        self.assertEqual(cfg.train.donor_logits_scale_end, 1.0)
        self.assertGreater(cfg.train.loss_donor_kl_weight, 0.0)
        self.assertGreater(cfg.train.loss_donor_correct_margin_weight, 0.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("donor_preserving_logit_guider", cfg.train.out_dir)

    def test_donor_preserving_bounded_delta_nogate_config_removes_undertrained_gate(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_bounded_delta_nogate_s120.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertLessEqual(cfg.model.qtrm_logits_scale, 0.35)
        self.assertLessEqual(cfg.model.qtrm_residual_clamp, 1.0)
        self.assertFalse(cfg.model.qtrm_residual_gate_enabled)
        self.assertFalse(cfg.train.workspace_evidence_injection)
        self.assertEqual(cfg.train.donor_logits_scale_start, 1.0)
        self.assertEqual(cfg.train.donor_logits_scale_end, 1.0)
        self.assertGreater(cfg.train.loss_donor_kl_weight, 0.0)
        self.assertGreater(cfg.train.loss_donor_correct_margin_weight, 0.0)
        self.assertIn("bounded_delta_nogate", cfg.train.out_dir)

    def test_donor_preserving_pure_recursive_preference_config_targets_raw_reasoning(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_pure_recursive_pref_s160.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertLessEqual(cfg.model.qtrm_logits_scale, 0.35)
        self.assertLessEqual(cfg.model.qtrm_residual_clamp, 1.0)
        self.assertFalse(cfg.model.qtrm_residual_gate_enabled)
        self.assertFalse(cfg.train.workspace_evidence_injection)
        self.assertGreater(cfg.train.loss_preference_weight, 0.0)
        self.assertGreater(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("pure_recursive_pref", cfg.train.out_dir)

    def test_donor_preserving_core_forced_readout_config_keeps_core_causal(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_s160.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.n_coda_layers, 0)
        self.assertTrue(cfg.model.core_loop_readout_enabled)
        self.assertTrue(cfg.model.core_loop_readout_requires_core)
        self.assertFalse(cfg.model.answer_bottleneck_enabled)
        self.assertFalse(cfg.model.qtrm_residual_gate_enabled)
        self.assertLessEqual(cfg.model.qtrm_residual_clamp, 1.5)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_loop_readout")
        self.assertFalse(cfg.train.workspace_evidence_injection)
        self.assertGreaterEqual(cfg.train.loss_canonical_causal_weight, 1.0)
        self.assertIn("core_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("core_forced_readout", cfg.train.out_dir)

    def test_donor_preserving_core_forced_readout_outer4_config_trains_deeper_loop(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_core_forced_readout_pref_outer4_s120.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.outer_steps, 4)
        self.assertTrue(cfg.model.core_step_conditioning_enabled)
        self.assertTrue(cfg.model.core_loop_readout_enabled)
        self.assertTrue(cfg.model.core_loop_readout_requires_core)
        self.assertEqual(cfg.model.n_coda_layers, 0)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_loop_readout")
        self.assertLessEqual(cfg.train.lr, 1.0e-5)
        self.assertGreaterEqual(cfg.train.loss_canonical_causal_weight, 1.0)
        self.assertIn("outer4", cfg.train.out_dir)

    def test_donor_preserving_core_forced_causal_prefix_sharpener_config(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_core_forced_causal_prefix_sharpener_s080.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertTrue(cfg.model.core_loop_readout_enabled)
        self.assertTrue(cfg.model.core_loop_readout_requires_core)
        self.assertFalse(cfg.model.answer_bottleneck_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_loop_readout")
        self.assertFalse(cfg.train.workspace_evidence_injection)
        self.assertGreater(cfg.train.loss_greedy_token_margin_weight, 0.0)
        self.assertFalse(cfg.train.greedy_token_margin_only_donor_errors)
        self.assertIn("causal_prefix_sharpener", cfg.train.out_dir)

    def test_donor_preserving_core_forced_variable_trajectory_config_uses_short_long_loss(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_core_forced_variable_traj_s080.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.outer_steps, 4)
        self.assertTrue(cfg.model.core_step_conditioning_enabled)
        self.assertTrue(cfg.model.core_loop_readout_enabled)
        self.assertTrue(cfg.model.core_loop_readout_requires_core)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_loop_readout")
        self.assertGreater(cfg.train.loss_core_variable_trajectory_weight, 0.0)
        self.assertEqual(cfg.train.core_variable_trajectory_short_steps, 1)
        self.assertGreater(cfg.train.core_variable_trajectory_short_lm_weight, 0.0)
        self.assertIn("variable_traj", cfg.train.out_dir)

    def test_donor_preserving_core_forced_depth_text_ce_config_uses_process_credit(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_donor_preserving_core_forced_depth_text_ce_s080.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.outer_steps, 4)
        self.assertTrue(cfg.model.core_loop_readout_enabled)
        self.assertTrue(cfg.model.core_loop_readout_requires_core)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_loop_readout")
        self.assertGreater(cfg.train.loss_core_depth_text_ce_weight, 0.0)
        self.assertEqual(cfg.train.core_depth_text_ce_min_step, 1)
        self.assertIn("depth_text_ce", cfg.train.out_dir)

    def test_ouro_donor_guided_adapter_config_uses_donor_as_renderer(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_transition_joint_dynamic_halt_v3_ouro_donor_guided_adapter_s060.yaml"
        )

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.qtrm_logits_scale, 0.0)
        self.assertTrue(cfg.model.answer_state_loop_halt_gate_enabled)
        self.assertTrue(cfg.model.answer_state_loop_lm_adapter_enabled)
        self.assertLessEqual(cfg.model.qtrm_residual_clamp, 2.0)
        self.assertEqual(cfg.train.trainable_param_policy, "answer_state_loop_lm_adapter_only")
        self.assertFalse(cfg.train.workspace_evidence_injection)
        self.assertIn("donor_guided_adapter", cfg.train.out_dir)

    def test_pure_recursive_transition_state_config_has_causal_state_ablation(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_pure_recursive_transition_state_s080.yaml")

        self.assertTrue(cfg.model.answer_state_loop_enabled)
        self.assertTrue(cfg.model.transition_state_enabled)
        self.assertGreaterEqual(cfg.model.transition_state_dim, 4)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertGreater(cfg.train.loss_canonical_causal_weight, 0.0)
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("pure_recursive_transition_state", cfg.train.out_dir)

    def test_pure_recursive_latent_action_codebook_config_uses_four_codes(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_s120.yaml"
        )

        self.assertTrue(cfg.model.answer_state_loop_enabled)
        self.assertTrue(cfg.model.transition_state_code_enabled)
        self.assertEqual(cfg.model.transition_state_codebook_size, 4)
        self.assertTrue(cfg.model.transition_state_code_only_answer_loop)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("latent_action_codebook", cfg.train.out_dir)

    def test_pure_recursive_latent_action_codebook_v2_config_uses_five_codes(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_v2_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_code_enabled)
        self.assertEqual(cfg.model.transition_state_codebook_size, 5)
        self.assertTrue(cfg.model.transition_state_code_only_answer_loop)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("latent_action_codebook_v2", cfg.train.out_dir)

    def test_pure_recursive_transition_finality_config_enables_finality_head(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_transition_finality_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_finality_enabled)
        self.assertFalse(cfg.model.transition_state_code_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("transition_finality", cfg.train.out_dir)

    def test_pure_recursive_latent_action_codebook_finality_config_combines_code_and_halt(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_latent_action_codebook_finality_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_code_enabled)
        self.assertEqual(cfg.model.transition_state_codebook_size, 4)
        self.assertTrue(cfg.model.transition_state_finality_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("latent_action_codebook_finality", cfg.train.out_dir)

    def test_pure_recursive_transition_joint_config_uses_single_code_halt_head(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_transition_joint_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_joint_enabled)
        self.assertEqual(cfg.model.transition_state_joint_size, 8)
        self.assertFalse(cfg.model.transition_state_code_enabled)
        self.assertFalse(cfg.model.transition_state_finality_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("transition_joint", cfg.train.out_dir)

    def test_pure_recursive_transition_joint_dense_config_uses_single_code_halt_head(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_joint_enabled)
        self.assertEqual(cfg.model.transition_state_joint_size, 8)
        self.assertFalse(cfg.model.transition_state_code_enabled)
        self.assertFalse(cfg.model.transition_state_finality_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("transition_joint_dense", cfg.train.out_dir)

    def test_pure_recursive_transition_joint_dense_terminal_v2_config_uses_ten_joint_states(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_joint_enabled)
        self.assertEqual(cfg.model.transition_state_codebook_size, 5)
        self.assertEqual(cfg.model.transition_state_joint_size, 10)
        self.assertFalse(cfg.model.transition_state_code_enabled)
        self.assertFalse(cfg.model.transition_state_finality_enabled)
        self.assertIn("transition_joint_dense_terminal_v2", cfg.train.out_dir)

    def test_pure_recursive_transition_text_config_enables_text_state_path(self):
        from qtrm_mm.config import load_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_transition_text_s120.yaml"
        )

        self.assertTrue(cfg.model.transition_state_enabled)
        self.assertFalse(cfg.model.transition_state_code_enabled)
        self.assertFalse(cfg.model.transition_state_finality_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "core_and_answer_state_loop")
        self.assertIn("transition_state_off", cfg.train.canonical_causal_ablation_modes)
        self.assertIn("transition_text", cfg.train.out_dir)

    def test_evidence_span_reader_config_trains_span_reader_only(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_evidence_span_reader_s050.yaml")

        self.assertTrue(cfg.model.evidence_span_reader_enabled)
        self.assertTrue(cfg.model.answer_bottleneck_enabled)
        self.assertEqual(cfg.train.trainable_param_policy, "evidence_span_reader_only")
        self.assertEqual(cfg.train.loss_lm_weight, 0.0)
        self.assertGreater(cfg.train.loss_evidence_span_reader_weight, 0.0)
        self.assertTrue(cfg.train.workspace_evidence_injection)
        self.assertIn("evidence_span_reader", cfg.train.out_dir)

    def test_scheduled_donor_logits_scale_linearly_anneals_to_student(self):
        from qtrm_mm.training.train import scheduled_donor_logits_scale

        values = [
            scheduled_donor_logits_scale(
                config_scale=1.0,
                start=1.0,
                end=0.0,
                step=step,
                total_steps=5,
            )
            for step in range(5)
        ]

        self.assertEqual(values, [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_scheduled_donor_logits_scale_defaults_to_config_scale(self):
        from qtrm_mm.training.train import scheduled_donor_logits_scale

        self.assertEqual(
            scheduled_donor_logits_scale(
                config_scale=0.7,
                start=None,
                end=None,
                step=3,
                total_steps=10,
            ),
            0.7,
        )

    def test_build_core_world_model_actions_routes_memory_rows_through_retrieve_verify_answer(self):
        import torch
        from qtrm_mm.training.train import build_core_world_model_actions

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            "workspace_attention_mask": torch.tensor([[1, 1, 0], [0, 0, 0]]),
        }

        actions = build_core_world_model_actions(
            batch,
            num_steps=3,
            num_actions=10,
            device="cpu",
        )

        self.assertEqual(actions.shape, (2, 3, 10))
        self.assertTrue(torch.equal(actions[0].argmax(dim=-1), torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(actions[1].argmax(dim=-1), torch.tensor([0, 2, 3])))


if __name__ == "__main__":
    unittest.main()
