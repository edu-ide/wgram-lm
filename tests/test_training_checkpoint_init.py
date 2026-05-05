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
