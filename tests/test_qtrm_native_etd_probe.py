import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path("scripts/335_train_qtrm_native_etd_probe.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_etd_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeETDProbeTests(unittest.TestCase):
    def test_cases_encode_prompt_and_answer_tokens(self):
        module = load_module()
        case = module.NativeCase(
            case_id="x",
            start=3,
            op_ids=(1, 2),
            answer=7,
        )

        prompt = module.case_prompt_tokens(case)
        full = module.case_full_tokens(case)

        self.assertEqual(prompt[:3], [module.BOS, module.START, module.value_token(3)])
        self.assertEqual(prompt[-1], module.ANS)
        self.assertEqual(full[-2:], [module.value_token(7), module.EOS])

    def test_depth_target_tokens_follow_stepwise_program_state(self):
        module = load_module()
        case = module.NativeCase(
            case_id="x",
            start=3,
            op_ids=(1, 4, 2),
            answer=0,
        )

        targets = module.depth_target_tokens([case], max_depth=4, modulus=8, device=torch.device("cpu"))

        value = 3
        expected_values = []
        for op_id in case.op_ids:
            value = module.apply_op(value, op_id, 8)
            expected_values.append(value)
        expected_values.append(value)
        self.assertEqual(
            targets.tolist(),
            [[module.value_token(value) for value in expected_values]],
        )

    def test_case_with_active_program_len_replaces_tail_with_noop_and_recomputes_answer(self):
        module = load_module()
        case = module.NativeCase(
            case_id="x",
            start=3,
            op_ids=(1, 4, 2, 5),
            answer=999,
        )

        shortened = module.case_with_active_program_len(
            case,
            active_len=2,
            modulus=8,
        )

        self.assertEqual(shortened.op_ids, (1, 4, module.NOOP_OP_ID, module.NOOP_OP_ID))
        value = 3
        for op_id in shortened.op_ids:
            value = module.apply_op(value, op_id, 8)
        self.assertEqual(shortened.answer, value)

    def test_active_program_len_schedule_warms_up_then_uses_full_program(self):
        module = load_module()

        early = module.active_program_len_for_step(
            step=1,
            total_steps=100,
            program_len=4,
            min_active_len=1,
            warmup_fraction=0.5,
        )
        middle = module.active_program_len_for_step(
            step=25,
            total_steps=100,
            program_len=4,
            min_active_len=1,
            warmup_fraction=0.5,
        )
        late = module.active_program_len_for_step(
            step=75,
            total_steps=100,
            program_len=4,
            min_active_len=1,
            warmup_fraction=0.5,
        )

        self.assertEqual(early, 1)
        self.assertGreaterEqual(middle, early)
        self.assertEqual(late, 4)

    def test_model_outputs_lm_logits_for_each_token_position(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))

    def test_single_core_carrier_keeps_lm_logit_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            think_structure="single_core_carrier",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        result = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )

        self.assertEqual(tuple(result["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", result)
        self.assertEqual(tuple(result["core_state_trace_h"].shape[:3]), (1, 2, 3))

    def test_carrier_gate_init_controls_carrier_residual_gate(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            think_structure="single_core_carrier",
            carrier_gate_init=-4.0,
        )

        self.assertAlmostEqual(float(model.single_carrier_gate_logit.item()), -4.0)

    def test_position_embedding_none_ignores_absolute_position_table(self):
        module = load_module()
        torch.manual_seed(123)
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official_prenorm",
            position_embedding_mode="none",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        before = model(input_ids, think_steps=2)
        with torch.no_grad():
            model.pos_embed.weight.add_(1000.0)
        after = model(input_ids, think_steps=2)

        self.assertTrue(torch.allclose(before, after))

    def test_circular_value_codec_replaces_value_token_embeddings_and_keeps_lm_logits(self):
        module = load_module()
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1), module.ANS]])
        learned = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            value_codec="learned",
        )
        circular = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            value_codec="circular",
        )

        learned_embeddings = learned._token_embeddings(input_ids)
        circular_embeddings = circular._token_embeddings(input_ids)
        logits = circular(input_ids, think_steps=1)

        self.assertEqual(circular.value_codec, "circular")
        self.assertFalse(torch.allclose(learned_embeddings, circular_embeddings))
        self.assertEqual(tuple(logits.shape), (1, 4, module.vocab_size(8)))
        custom = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            value_codec="circular",
            value_token_ids=(module.value_token(3), module.value_token(5)),
        )
        self.assertEqual(
            custom._value_id_lookup[
                torch.tensor([module.value_token(3), module.value_token(5), module.value_token(7)])
            ].tolist(),
            [0, 1, -1],
        )

    def test_qwen35_style_hybrid_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="qtrm_hybrid_3to1",
            n_kv_heads=2,
            hybrid_layers=4,
            attn_every=4,
            delta_backend="torch_gated_delta",
            attention_backend="sdpa",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))

    def test_official_trm_style_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMOfficialStack)

    def test_trm_shell_gated_delta_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_gated_delta",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMGatedDeltaBlock)

    def test_trm_shell_qwen35_3to1_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_qwen35_3to1",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMQwen35HybridBlock)
        self.assertEqual(len(model.think.layers), 4)

    def test_stage_backbone_overrides_allow_mixed_fla_and_mha_paths(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            encode_backbone="qtrm_hybrid_3to1",
            think_backbone="mha_etd",
            decode_backbone="qtrm_hybrid_3to1",
            n_kv_heads=2,
            hybrid_layers=1,
            attn_every=4,
            delta_backend="torch_gated_delta",
            attention_backend="sdpa",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(model.encode_backbone, "qtrm_hybrid_3to1")
        self.assertEqual(model.think_backbone, "mha_etd")
        self.assertEqual(model.decode_backbone, "qtrm_hybrid_3to1")
        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))

    def test_parser_accepts_mamba3_as_stage_backbone_candidate(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--backbone",
                "mamba3",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "mamba3",
                "--decode-backbone",
                "mha_etd",
            ]
        )

        self.assertEqual(args.backbone, "mamba3")
        self.assertEqual(args.think_backbone, "mamba3")

    def test_parser_accepts_official_trm_style_backbone_candidate(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--backbone",
                "trm_official",
                "--encode-backbone",
                "mha_etd",
                "--think-backbone",
                "trm_official",
                "--decode-backbone",
                "mha_etd",
            ]
        )

        self.assertEqual(args.backbone, "trm_official")
        self.assertEqual(args.think_backbone, "trm_official")

    def test_parser_accepts_trm_shell_mixer_backbone_candidates(self):
        module = load_module()

        for backbone in (
            "trm_mamba3",
            "trm_gated_delta",
            "trm_qwen35_3to1",
            "trm_tri_mixer",
            "trm_gated_attention",
            "trm_qwen_attention",
        ):
            with self.subTest(backbone=backbone):
                args = module.build_arg_parser().parse_args(
                    [
                        "--backbone",
                        backbone,
                        "--encode-backbone",
                        "mha_etd",
                        "--think-backbone",
                        backbone,
                        "--decode-backbone",
                        "mha_etd",
                    ]
                )

                self.assertEqual(args.backbone, backbone)
                self.assertEqual(args.think_backbone, backbone)

    def test_trm_gated_attention_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_gated_attention",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMGatedAttentionBlock)
        self.assertTrue(hasattr(model.think, "attn_gate_logit"))
        self.assertGreater(float(torch.sigmoid(model.think.attn_gate_logit.detach())), 0.5)

    def test_trm_qwen_attention_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_qwen_attention",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMQwenAttentionBlock)
        self.assertTrue(hasattr(model.think, "q_norm"))
        self.assertIsNone(model.think.q_proj.bias)

    def test_trm_shell_tri_mixer_backbone_outputs_lm_logits(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_tri_mixer",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMTriMixerBlock)
        self.assertEqual(model.think.mixer_kinds, ("gated_delta", "mamba3", "gated_delta", "attention"))

    def test_parser_accepts_trm_dual_z_thinking_structure(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z"])

        self.assertEqual(args.think_structure, "trm_dual_z")

    def test_parser_accepts_trm_dual_z_gated_thinking_structure(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z_gated"])

        self.assertEqual(args.think_structure, "trm_dual_z_gated")

    def test_parser_accepts_trm_dual_z_residual_thinking_structure(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z_residual"])

        self.assertEqual(args.think_structure, "trm_dual_z_residual")

    def test_parser_accepts_trm_dual_z_coupled_residual_thinking_structure(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z_coupled_residual"])

        self.assertEqual(args.think_structure, "trm_dual_z_coupled_residual")

    def test_parser_accepts_trm_dual_z_coupled_thinking_structure(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--think-structure", "trm_dual_z_coupled"])

        self.assertEqual(args.think_structure, "trm_dual_z_coupled")

    def test_parser_accepts_coupled_dual_trm_proposal_variants(self):
        module = load_module()

        for structure in (
            "trm_dual_z_coupled_delta_l_only",
            "trm_dual_z_coupled_mamba_h_only",
            "trm_dual_z_coupled_gated_proposal",
            "trm_dual_z_coupled_hybrid_router",
            "trm_dual_z_coupled_cross_attention",
            "trm_dual_z_coupled_step_conditioned_attention",
            "trm_dual_z_interactive_transition_gate",
            "trm_dual_z_interactive_residual_readout",
            "trm_dual_z_interactive_prefix_scratch",
            "trm_dual_z_interactive_core_carrier",
        ):
            with self.subTest(structure=structure):
                args = module.build_arg_parser().parse_args(["--think-structure", structure])

                self.assertEqual(args.think_structure, structure)

    def test_trm_dual_z_structure_uses_low_high_recurrent_states(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            think_structure="trm_dual_z",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)
        off_logits = model(input_ids, think_steps=2, thinking_block_off=True)

        self.assertEqual(model.think_structure, "trm_dual_z")
        self.assertIsNotNone(model.think)
        self.assertTrue(hasattr(model, "z_l_init"))
        self.assertTrue(hasattr(model, "z_h_init"))
        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertEqual(tuple(off_logits.shape), (1, 3, module.vocab_size(8)))

    def test_dual_z_state_ablation_changes_recurrent_runtime_path(self):
        module = load_module()
        torch.manual_seed(7)
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)
        z_h_zero_logits = model(input_ids, think_steps=2, z_h_zero=True)

        self.assertEqual(tuple(z_h_zero_logits.shape), tuple(logits.shape))
        self.assertFalse(torch.allclose(logits, z_h_zero_logits))

    def test_native_adaptive_halt_can_stop_recurrent_core_early(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z",
        )
        with torch.no_grad():
            model.core_halt_head.weight.zero_()
            model.core_halt_head.bias.fill_(10.0)
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        out = model.forward_with_runtime(
            input_ids,
            think_steps=4,
            adaptive_halt=True,
            halt_threshold=0.5,
            halt_min_steps=1,
        )

        self.assertEqual(tuple(out["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertEqual(int(out["halt_steps"].item()), 1)
        self.assertEqual(tuple(out["core_q_halt_logits"].shape), (1, 1))
        self.assertTrue(torch.equal(out["core_halted"], torch.tensor([True])))

    def test_native_adaptive_halt_runs_full_depth_when_halt_head_says_continue(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z",
        )
        with torch.no_grad():
            model.core_halt_head.weight.zero_()
            model.core_halt_head.bias.fill_(-10.0)
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        out = model.forward_with_runtime(
            input_ids,
            think_steps=4,
            adaptive_halt=True,
            halt_threshold=0.5,
            halt_min_steps=1,
        )

        self.assertEqual(int(out["halt_steps"].item()), 4)
        self.assertEqual(tuple(out["core_q_halt_logits"].shape), (1, 4))
        self.assertTrue(torch.equal(out["core_halted"], torch.tensor([False])))

    def test_forward_with_runtime_can_return_state_trace(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        out = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )

        self.assertEqual(tuple(out["core_state_trace_h"].shape), (1, 3, 3, 16))
        self.assertEqual(tuple(out["core_state_trace_l"].shape), (1, 3, 3, 16))

    def test_core_halt_logits_can_use_mean_pooling_instead_of_last_token(self):
        module = load_module()
        last_model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=4,
            n_heads=4,
            d_ff=8,
            dropout=0.0,
            halt_pooling="last",
        )
        mean_model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=4,
            n_heads=4,
            d_ff=8,
            dropout=0.0,
            halt_pooling="mean",
        )
        with torch.no_grad():
            last_model.core_halt_head.weight.fill_(1.0)
            last_model.core_halt_head.bias.zero_()
            mean_model.core_halt_head.weight.fill_(1.0)
            mean_model.core_halt_head.bias.zero_()
        state = torch.tensor([[[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]])

        self.assertEqual(float(last_model._core_halt_logits(state).item()), 0.0)
        self.assertGreater(float(mean_model._core_halt_logits(state).item()), 0.0)

    def test_core_halt_logits_can_use_dedicated_halt_state(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=4,
            n_heads=4,
            d_ff=8,
            dropout=0.0,
            halt_pooling="dedicated",
        )
        with torch.no_grad():
            model.core_halt_head.weight.fill_(1.0)
            model.core_halt_head.bias.zero_()
        state = torch.zeros(1, 2, 4)
        halt_state = torch.ones(1, 4)

        self.assertEqual(
            float(model._core_halt_logits(state, halt_state=halt_state).item()),
            4.0,
        )

    def test_dedicated_halt_pooling_is_available_from_parser(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(["--halt-pooling", "dedicated"])

        self.assertEqual(args.halt_pooling, "dedicated")

    def test_applicable_ablations_exclude_nonexistent_coupling_for_single_core(self):
        module = load_module()

        self.assertEqual(
            module.applicable_ablation_names("single"),
            ("state_reset", "op_zero"),
        )
        self.assertEqual(
            module.applicable_ablation_names("trm_dual_z"),
            ("state_reset", "op_zero", "z_l_zero", "z_h_zero"),
        )
        self.assertEqual(
            module.applicable_ablation_names("trm_dual_z_interactive"),
            ("state_reset", "op_zero", "z_l_zero", "z_h_zero"),
        )
        self.assertEqual(
            module.applicable_ablation_names("trm_dual_z_coupled"),
            ("state_reset", "op_zero", "coupling_off", "z_l_zero", "z_h_zero"),
        )
        self.assertEqual(
            module.applicable_ablation_names(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange"
            ),
            (
                "state_reset",
                "op_zero",
                "coupling_off",
                "z_l_zero",
                "z_h_zero",
                "carrier_off",
            ),
        )
        self.assertEqual(
            module.applicable_ablation_names(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned"
            ),
            (
                "state_reset",
                "op_zero",
                "coupling_off",
                "z_l_zero",
                "z_h_zero",
                "carrier_off",
            ),
        )
        self.assertEqual(
            module.applicable_ablation_names(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_order_router"
            ),
            (
                "state_reset",
                "op_zero",
                "coupling_off",
                "z_l_zero",
                "z_h_zero",
                "carrier_off",
            ),
        )

    def test_trm_dual_z_interactive_uses_two_state_updates_without_direct_coupling_knob(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_interactive",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)
        z_l_zero_logits = model(input_ids, think_steps=2, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=2, z_h_zero=True)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMOfficialStack)
        self.assertFalse(hasattr(model, "trm_l_to_h"))
        self.assertFalse(hasattr(model, "trm_h_to_l"))
        self.assertFalse(hasattr(model, "trm_coupling_alpha"))
        self.assertFalse(torch.allclose(logits, z_l_zero_logits))
        self.assertFalse(torch.allclose(logits, z_h_zero_logits))

    def test_trm_dual_z_interactive_transition_gate_keeps_recurrent_lm_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_interactive_transition_gate",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=2, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=2, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_l_transition_gate_logit"))
        self.assertTrue(hasattr(model, "trm_h_transition_gate_logit"))
        self.assertFalse(hasattr(model, "trm_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_diffusive_conditions_core_on_denoise_time(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_diffusive",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertEqual(tuple(runtime["core_state_trace_h"].shape[:2]), (1, 3))
        self.assertTrue(hasattr(model, "trm_diffusion_time_mlp"))
        self.assertTrue(hasattr(model, "trm_diffusion_input_norm"))
        self.assertTrue(hasattr(model, "trm_init_l_proj"))
        self.assertTrue(hasattr(model, "trm_init_h_proj"))
        self.assertFalse(hasattr(model, "trm_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_diffusive_reversed_hybrid_uses_mamba_l_delta_h_attention_sync(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_diffusive_reversed_hybrid_3to1",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertEqual(tuple(runtime["core_state_trace_h"].shape[:2]), (1, 3))
        self.assertTrue(hasattr(model, "trm_l_mamba3_attention_hybrid"))
        self.assertTrue(hasattr(model, "trm_h_gated_delta_attention_hybrid"))
        self.assertEqual(
            model.trm_l_mamba3_attention_hybrid.mixer_kinds,
            ("mamba3", "mamba3", "mamba3", "attention"),
        )
        self.assertEqual(
            model.trm_h_gated_delta_attention_hybrid.mixer_kinds,
            ("gated_delta", "gated_delta", "gated_delta", "attention"),
        )
        self.assertTrue(hasattr(model, "trm_diffusion_time_mlp"))
        self.assertTrue(hasattr(model, "trm_init_l_proj"))
        self.assertTrue(hasattr(model, "trm_init_h_proj"))
        self.assertFalse(hasattr(model, "trm_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_diffusive_reversed_hybrid_joint_readout_uses_both_states(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_diffusive_reversed_hybrid_3to1_joint_readout",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_l_mamba3_attention_hybrid"))
        self.assertTrue(hasattr(model, "trm_h_gated_delta_attention_hybrid"))
        self.assertTrue(hasattr(model, "trm_joint_readout_norm"))
        self.assertTrue(hasattr(model, "trm_joint_readout_proj"))
        self.assertFalse(hasattr(model, "trm_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_reversed_hybrid_uses_mamba_l_delta_h_without_diffusion_time(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_reversed_hybrid_3to1",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_l_mamba3_attention_hybrid"))
        self.assertTrue(hasattr(model, "trm_h_gated_delta_attention_hybrid"))
        self.assertEqual(
            model.trm_l_mamba3_attention_hybrid.mixer_kinds,
            ("mamba3", "mamba3", "mamba3", "attention"),
        )
        self.assertEqual(
            model.trm_h_gated_delta_attention_hybrid.mixer_kinds,
            ("gated_delta", "gated_delta", "gated_delta", "attention"),
        )
        self.assertTrue(hasattr(model, "trm_reversed_hybrid_input_norm"))
        self.assertFalse(hasattr(model, "trm_reversed_mha_readout_alpha"))
        self.assertTrue(hasattr(model, "trm_init_l_proj"))
        self.assertTrue(hasattr(model, "trm_init_h_proj"))
        self.assertFalse(hasattr(model, "trm_diffusion_time_mlp"))
        self.assertFalse(hasattr(model, "trm_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_reversed_mha_etd_reuses_warmstartable_think_block(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure="trm_dual_z_reversed_mha_etd",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_reversed_hybrid_input_norm"))
        self.assertTrue(hasattr(model, "trm_init_l_proj"))
        self.assertTrue(hasattr(model, "trm_init_h_proj"))
        self.assertFalse(hasattr(model, "trm_l_mamba3_attention_hybrid"))
        self.assertFalse(hasattr(model, "trm_h_gated_delta_attention_hybrid"))
        self.assertFalse(hasattr(model, "trm_diffusion_time_mlp"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_nested_reversed_mha_etd_adds_learned_update_levels(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure="trm_dual_z_nested_reversed_mha_etd",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_reversed_hybrid_input_norm"))
        self.assertTrue(hasattr(model, "trm_reversed_mha_readout_alpha"))
        self.assertTrue(hasattr(model, "trm_nested_l_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_h_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_l_update_gate_logit"))
        self.assertTrue(hasattr(model, "trm_nested_h_update_gate_logit"))
        self.assertFalse(hasattr(model, "trm_l_mamba3_attention_hybrid"))
        self.assertFalse(hasattr(model, "trm_h_gated_delta_attention_hybrid"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_nested_reversed_mha_etd_joint_readout_uses_both_states(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure="trm_dual_z_nested_reversed_mha_etd_joint_readout",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_reversed_hybrid_input_norm"))
        self.assertTrue(hasattr(model, "trm_nested_l_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_h_optimizer"))
        self.assertTrue(hasattr(model, "trm_joint_readout_norm"))
        self.assertTrue(hasattr(model, "trm_joint_readout_proj"))
        self.assertFalse(hasattr(model, "trm_reversed_mha_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_preserves_base_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure="trm_dual_z_nested_reversed_mha_etd_residual_joint_readout",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_reversed_mha_readout_alpha"))
        self.assertTrue(hasattr(model, "trm_nested_mha_joint_readout_alpha"))
        self.assertTrue(hasattr(model, "trm_joint_readout_norm"))
        self.assertTrue(hasattr(model, "trm_joint_readout_proj"))
        self.assertTrue(hasattr(model, "trm_nested_l_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_h_optimizer"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_nested_residual_joint_readout_core_carrier_keeps_nested_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier"
            ),
            carrier_gate_init=-4.0,
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        carrier_off_logits = model(input_ids, think_steps=3, carrier_off=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertEqual(tuple(carrier_off_logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_l", runtime)
        self.assertIn(
            "carrier_off",
            module.applicable_ablation_names(model.think_structure),
        )
        self.assertTrue(hasattr(model, "trm_reversed_mha_readout_alpha"))
        self.assertTrue(hasattr(model, "trm_nested_mha_joint_readout_alpha"))
        self.assertTrue(hasattr(model, "trm_nested_l_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_h_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_core_carrier_in"))
        self.assertTrue(hasattr(model, "trm_nested_core_carrier_rnn"))
        self.assertAlmostEqual(
            float(model.trm_nested_core_carrier_gate_logit.item()),
            -4.0,
        )

    def test_trm_dual_z_nested_core_carrier_cross_exchange_is_inside_core(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_cross_exchange"
            ),
            carrier_gate_init=-8.0,
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])
        with torch.no_grad():
            model.trm_nested_cross_exchange_gate_logit.fill_(3.0)
            for optimizer in (
                model.trm_nested_l_optimizer,
                model.trm_nested_h_optimizer,
            ):
                optimizer[-1].weight.normal_()

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        cross_logits = model(input_ids, think_steps=3, carrier_off=True)
        coupling_off_logits = model(
            input_ids,
            think_steps=3,
            carrier_off=True,
            coupling_off=True,
        )

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_l", runtime)
        self.assertIn("core_state_trace_h", runtime)
        self.assertIn(
            "coupling_off",
            module.applicable_ablation_names(model.think_structure),
        )
        self.assertIn(
            "carrier_off",
            module.applicable_ablation_names(model.think_structure),
        )
        self.assertTrue(hasattr(model, "trm_nested_cross_to_l_norm"))
        self.assertTrue(hasattr(model, "trm_nested_cross_to_h_norm"))
        self.assertTrue(hasattr(model, "trm_nested_cross_exchange_gate_logit"))
        self.assertFalse(torch.allclose(cross_logits, coupling_off_logits))

    def test_trm_dual_z_nested_core_carrier_step_conditioned_is_inside_core(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_step_conditioned"
            ),
            carrier_gate_init=-8.0,
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])
        with torch.no_grad():
            model.trm_nested_step_update_gate_logit.fill_(3.0)
            for optimizer in (
                model.trm_nested_step_l_optimizer,
                model.trm_nested_step_h_optimizer,
            ):
                optimizer[-1].weight.normal_()

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        step_logits = model(input_ids, think_steps=3, carrier_off=True)
        coupling_off_logits = model(
            input_ids,
            think_steps=3,
            carrier_off=True,
            coupling_off=True,
        )

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_l", runtime)
        self.assertIn("core_state_trace_h", runtime)
        self.assertIn(
            "coupling_off",
            module.applicable_ablation_names(model.think_structure),
        )
        self.assertIn(
            "carrier_off",
            module.applicable_ablation_names(model.think_structure),
        )
        self.assertTrue(hasattr(model, "trm_nested_step_embed"))
        self.assertTrue(hasattr(model, "trm_nested_step_l_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_step_h_optimizer"))
        self.assertFalse(torch.allclose(step_logits, coupling_off_logits))

    def test_trm_dual_z_nested_order_bound_router_replaces_route1_transition(self):
        module = load_module()
        structure = (
            "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_"
            "order_bound_router"
        )
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure=structure,
            carrier_gate_init=-8.0,
        )
        input_ids = torch.tensor(
            [[module.BOS, module.START, module.value_token(1), module.op_token(3)]]
        )

        self.assertIn("coupling_off", module.applicable_ablation_names(structure))
        self.assertIn("carrier_off", module.applicable_ablation_names(structure))
        self.assertTrue(hasattr(model, "trm_nested_order_router"))
        self.assertTrue(hasattr(model, "trm_nested_route1_order_attn"))
        self.assertTrue(hasattr(model, "trm_nested_route1_order_gate_logit"))

        model.trm_nested_order_router_force_route = 1
        route1_logits = model(input_ids, think_steps=3, carrier_off=True)
        route0_logits = model(
            input_ids,
            think_steps=3,
            carrier_off=True,
            coupling_off=True,
        )

        self.assertEqual(tuple(route1_logits.shape), (1, 4, module.vocab_size(8)))
        self.assertFalse(torch.allclose(route1_logits, route0_logits))

    def test_nested_core_carrier_deterministic_mode_ignores_gru_random_state(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier"
            ),
            carrier_gate_init=-4.0,
            carrier_state_mode="state_mean",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        before = model(input_ids, think_steps=3).detach()
        with torch.no_grad():
            model.trm_nested_core_carrier_in.weight.normal_()
            for parameter in model.trm_nested_core_carrier_rnn.parameters():
                parameter.normal_()
        after = model(input_ids, think_steps=3).detach()

        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_trm_dual_z_nested_official_schedule_split_mixer_uses_h3_l6_mamba_l_delta_h(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_nested_official_schedule_split_mixer_3to1",
            trm_l_cycles=6,
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertEqual(model.trm_l_cycles, 6)
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_l_mamba3_attention_hybrid"))
        self.assertTrue(hasattr(model, "trm_h_gated_delta_attention_hybrid"))
        self.assertEqual(
            model.trm_l_mamba3_attention_hybrid.mixer_kinds,
            ("mamba3", "mamba3", "mamba3", "attention"),
        )
        self.assertEqual(
            model.trm_h_gated_delta_attention_hybrid.mixer_kinds,
            ("gated_delta", "gated_delta", "gated_delta", "attention"),
        )
        self.assertTrue(hasattr(model, "trm_nested_l_optimizer"))
        self.assertTrue(hasattr(model, "trm_nested_h_optimizer"))
        self.assertFalse(hasattr(model, "trm_diffusion_time_mlp"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_reversed_hybrid_semantic_carry_anchors_depth_state(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_reversed_hybrid_3to1_semantic_carry",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_semantic_l_carry_gate"))
        self.assertTrue(hasattr(model, "trm_semantic_h_carry_gate"))
        self.assertTrue(hasattr(model, "trm_init_l_proj"))
        self.assertTrue(hasattr(model, "trm_init_h_proj"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_reversed_hybrid_order_router_blends_update_orders(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_reversed_hybrid_3to1_order_router",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=3,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=3, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=3, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_order_router"))
        self.assertEqual(tuple(model.trm_order_router.bias.shape), (2,))
        self.assertGreater(float(model.trm_order_router.bias[0]), float(model.trm_order_router.bias[1]))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_recent_order_router_keeps_causal_lm_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_reversed_hybrid_3to1_recent_order_router",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )
        model.trm_order_router_force_route = 1
        forced_route_logits = model(input_ids, think_steps=2)
        delattr(model, "trm_order_router_force_route")

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_order_router"))
        self.assertEqual(tuple(forced_route_logits.shape), (1, 3, module.vocab_size(8)))

    def test_trm_dual_z_state_gru_order_router_adds_trainable_order_state(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_reversed_hybrid_3to1_state_gru_order_router",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        model.trm_order_router_force_route = 1
        logits = model(input_ids, think_steps=2)
        loss = logits.sum()
        loss.backward()
        delattr(model, "trm_order_router_force_route")

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_order_state_gru"))
        self.assertIsNotNone(model.trm_order_state_gru.weight_ih_l0.grad)

    def test_trm_dual_z_transition_state_order_router_updates_inside_route(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_reversed_hybrid_3to1_transition_state_order_router",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )
        model.trm_order_router_force_route = 1
        forced_route_logits = model(input_ids, think_steps=2)
        loss = forced_route_logits.sum()
        loss.backward()
        delattr(model, "trm_order_router_force_route")

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_order_transition_cell"))
        self.assertIsNotNone(model.trm_order_transition_cell.weight_ih.grad)

    def test_trm_dual_z_nested_core_carrier_order_router_stays_inside_native_core(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="mha_etd",
            think_structure=(
                "trm_dual_z_nested_reversed_mha_etd_residual_joint_readout_core_carrier_"
                "order_router"
            ),
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )
        model.trm_nested_order_router_force_route = 0
        route0_logits = model(input_ids, think_steps=2)
        model.trm_nested_order_router_force_route = 1
        route1_logits = model(input_ids, think_steps=2)
        delattr(model, "trm_nested_order_router_force_route")

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_nested_order_router"))
        self.assertTrue(hasattr(model, "trm_nested_core_carrier_in"))
        self.assertGreater(
            float(model.trm_nested_order_router.bias[0]),
            float(model.trm_nested_order_router.bias[1]),
        )
        self.assertFalse(torch.allclose(route0_logits, route1_logits))

    def test_trm_dual_z_interactive_core_carrier_keeps_recurrent_lm_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_interactive_core_carrier",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )
        z_l_zero_logits = model(input_ids, think_steps=2, z_l_zero=True)
        z_h_zero_logits = model(input_ids, think_steps=2, z_h_zero=True)

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_carrier_in"))
        self.assertTrue(hasattr(model, "trm_carrier_rnn"))
        self.assertTrue(hasattr(model, "trm_carrier_gate_logit"))
        self.assertFalse(hasattr(model, "trm_readout_alpha"))
        self.assertFalse(torch.allclose(runtime["logits"], z_l_zero_logits))
        self.assertFalse(torch.allclose(runtime["logits"], z_h_zero_logits))

    def test_trm_dual_z_interactive_prefix_scratch_is_causal_recurrent_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_interactive_prefix_scratch",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        runtime = model.forward_with_runtime(
            input_ids,
            think_steps=2,
            return_state_trace=True,
        )

        self.assertEqual(tuple(runtime["logits"].shape), (1, 3, module.vocab_size(8)))
        self.assertIn("core_state_trace_h", runtime)
        self.assertTrue(hasattr(model, "trm_scratch_proj"))
        self.assertTrue(hasattr(model, "trm_scratch_gate_logit"))

    def test_trm_dual_z_interactive_residual_readout_keeps_lm_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_interactive_residual_readout",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_readout_norm"))
        self.assertTrue(hasattr(model, "trm_readout_alpha"))

    def test_trm_dual_z_gated_structure_uses_identity_biased_state_updates(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_gated",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)
        off_logits = model(input_ids, think_steps=2, thinking_block_off=True)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertEqual(tuple(off_logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_step_embed"))
        self.assertTrue(hasattr(model, "trm_l_update_gate"))
        self.assertTrue(hasattr(model, "trm_h_update_gate"))
        self.assertTrue(torch.all(model.trm_l_update_gate.bias.detach() < 0.0))
        self.assertTrue(torch.all(model.trm_h_update_gate.bias.detach() < 0.0))
        self.assertLess(abs(float(model.trm_readout_alpha.detach())), 0.2)

    def test_trm_dual_z_residual_structure_uses_prompt_conditioned_state_readout(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_residual",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_init_l_proj"))
        self.assertTrue(hasattr(model, "trm_init_h_proj"))
        self.assertTrue(hasattr(model, "trm_readout_norm"))
        self.assertGreater(float(model.trm_readout_alpha.detach()), 0.0)

    def test_trm_dual_z_coupled_residual_keeps_attention_path_with_delta_and_mamba_proposals(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_residual",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMOfficialStack)
        self.assertTrue(hasattr(model, "trm_l_delta_proposal"))
        self.assertTrue(hasattr(model, "trm_h_mamba_proposal"))
        self.assertTrue(hasattr(model, "trm_l_to_h"))
        self.assertTrue(hasattr(model, "trm_h_to_l"))
        self.assertGreater(float(model.trm_delta_alpha.detach()), 0.0)
        self.assertGreater(float(model.trm_mamba_alpha.detach()), 0.0)

    def test_trm_dual_z_coupled_structure_adds_cross_state_coupling_without_proposal_mixers(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMOfficialStack)
        self.assertTrue(hasattr(model, "trm_l_to_h"))
        self.assertTrue(hasattr(model, "trm_h_to_l"))
        self.assertFalse(hasattr(model, "trm_l_delta_proposal"))
        self.assertFalse(hasattr(model, "trm_h_mamba_proposal"))
        self.assertGreater(float(model.trm_coupling_alpha.detach()), 0.0)

    def test_coupled_core_forward_accepts_state_and_coupling_ablations(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        for kwargs in (
            {"coupling_off": True},
            {"z_l_zero": True},
            {"z_h_zero": True},
        ):
            with self.subTest(kwargs=kwargs):
                logits = model(input_ids, think_steps=2, **kwargs)

                self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))

    def test_coupled_delta_l_only_adds_local_delta_proposal_without_mamba(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_delta_l_only",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_l_delta_proposal"))
        self.assertFalse(hasattr(model, "trm_h_mamba_proposal"))
        self.assertGreater(float(model.trm_delta_alpha.detach()), 0.0)

    def test_coupled_mamba_h_only_adds_slow_mamba_proposal_without_delta(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_mamba_h_only",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertFalse(hasattr(model, "trm_l_delta_proposal"))
        self.assertTrue(hasattr(model, "trm_h_mamba_proposal"))
        self.assertGreater(float(model.trm_mamba_alpha.detach()), 0.0)

    def test_coupled_gated_proposal_adds_small_gates_for_delta_and_mamba(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_gated_proposal",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_l_delta_proposal"))
        self.assertTrue(hasattr(model, "trm_h_mamba_proposal"))
        self.assertTrue(torch.sigmoid(model.trm_delta_gate_logit.detach()).item() < 0.05)
        self.assertTrue(torch.sigmoid(model.trm_mamba_gate_logit.detach()).item() < 0.05)

    def test_coupled_hybrid_router_keeps_dual_z_with_delta_mamba_and_official_path(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_hybrid_router",
            delta_backend="torch_gated_delta",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)
        no_coupling_logits = model(input_ids, think_steps=2, coupling_off=True)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertEqual(tuple(no_coupling_logits.shape), (1, 3, module.vocab_size(8)))
        self.assertIsInstance(model.think, module.NativeTRMOfficialStack)
        self.assertTrue(hasattr(model, "trm_l_delta_proposal"))
        self.assertTrue(hasattr(model, "trm_h_mamba_proposal"))
        self.assertTrue(hasattr(model, "trm_l_hybrid_router"))
        self.assertTrue(hasattr(model, "trm_h_hybrid_router"))
        l_prior = torch.softmax(model.trm_l_hybrid_router.bias.detach(), dim=-1)
        h_prior = torch.softmax(model.trm_h_hybrid_router.bias.detach(), dim=-1)
        self.assertGreater(float(l_prior[0]), float(l_prior[1]))
        self.assertGreater(float(h_prior[0]), float(h_prior[1]))

    def test_coupled_cross_attention_adds_explicit_low_high_attention_links(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_cross_attention",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_l_cross_attn"))
        self.assertTrue(hasattr(model, "trm_h_cross_attn"))
        self.assertGreater(float(model.trm_cross_alpha.detach()), 0.0)

    def test_coupled_step_conditioned_attention_adds_step_state_to_coupled_core(self):
        module = load_module()
        model = module.NativeQTRMETDLM(
            vocab=module.vocab_size(8),
            max_seq_len=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            backbone="trm_official",
            think_structure="trm_dual_z_coupled_step_conditioned_attention",
        )
        input_ids = torch.tensor([[module.BOS, module.START, module.value_token(1)]])

        logits = model(input_ids, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 3, module.vocab_size(8)))
        self.assertTrue(hasattr(model, "trm_step_embed"))
        self.assertTrue(hasattr(model, "trm_l_to_h"))
        self.assertTrue(hasattr(model, "trm_h_to_l"))

    def test_decision_requires_depth_and_ablation_gain(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.8",
                "--accept-min-depth-gain",
                "0.1",
                "--accept-min-ablation-drop",
                "0.1",
                "--think-structure",
                "trm_dual_z_coupled",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.5},
            "think4": {"generation_exact": 0.85},
            "state_reset": {"generation_exact": 0.6},
            "op_zero": {"generation_exact": 0.1},
            "coupling_off": {"generation_exact": 0.2},
            "z_l_zero": {"generation_exact": 0.3},
            "z_h_zero": {"generation_exact": 0.4},
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["decision"], "accepted_l1_native_etd")

    def test_decision_rejects_when_coupling_ablation_matches_full_score(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.8",
                "--accept-min-depth-gain",
                "0.1",
                "--accept-min-ablation-drop",
                "0.1",
                "--think-structure",
                "trm_dual_z_coupled",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.5},
            "think4": {"generation_exact": 0.85},
            "state_reset": {"generation_exact": 0.1},
            "op_zero": {"generation_exact": 0.1},
            "coupling_off": {"generation_exact": 0.84},
            "z_l_zero": {"generation_exact": 0.1},
            "z_h_zero": {"generation_exact": 0.1},
        }

        decision = module.make_decision(metrics, args)

        self.assertFalse(decision["accepted"])
        self.assertIn("ablation_drop_below_threshold", decision["reject_reasons"])
        self.assertEqual(decision["decisive_metrics"]["coupling_off_generation_exact"], 0.84)

    def test_accept_decision_label_can_be_overridden_for_harder_gates(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--accept-min-exact",
                "0.8",
                "--accept-min-depth-gain",
                "0.1",
                "--accept-min-ablation-drop",
                "0.1",
                "--accepted-decision",
                "accepted_l2_native_recursive_gain",
            ]
        )
        metrics = {
            "think0": {"generation_exact": 0.5},
            "think4": {"generation_exact": 0.85},
            "state_reset": {"generation_exact": 0.6},
            "op_zero": {"generation_exact": 0.1},
        }

        decision = module.make_decision(metrics, args)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["decision"], "accepted_l2_native_recursive_gain")

    def test_answer_loss_can_downweight_eos_and_penalize_first_eos(self):
        module = load_module()
        vocab = module.vocab_size(8)
        prompt_len = 4
        full_tokens = torch.tensor(
            [[module.BOS, module.START, module.value_token(1), module.ANS, module.value_token(2), module.EOS]]
        )
        logits = torch.zeros((1, 5, vocab))
        logits[:, prompt_len - 1, module.value_token(2)] = 5.0
        logits[:, prompt_len, module.EOS] = -5.0

        full_weight = module.answer_loss(
            logits,
            full_tokens,
            prompt_len=prompt_len,
            answer_loss_weight=1.0,
            eos_loss_weight=1.0,
        )
        downweighted = module.answer_loss(
            logits,
            full_tokens,
            prompt_len=prompt_len,
            answer_loss_weight=1.0,
            eos_loss_weight=0.0,
        )

        self.assertLess(float(downweighted), float(full_weight))

        logits[:, prompt_len - 1, module.EOS] = 8.0
        no_margin = module.answer_loss(
            logits,
            full_tokens,
            prompt_len=prompt_len,
            answer_loss_weight=1.0,
            eos_loss_weight=0.0,
            answer_eos_margin_weight=0.0,
        )
        with_margin = module.answer_loss(
            logits,
            full_tokens,
            prompt_len=prompt_len,
            answer_loss_weight=1.0,
            eos_loss_weight=0.0,
            answer_eos_margin_weight=1.0,
            answer_eos_margin=1.0,
        )

        self.assertGreater(float(with_margin), float(no_margin))

    def test_tiny_training_run_writes_report(self):
        module = load_module()
        out_dir = Path("local_eval/test_qtrm_native_etd_probe")
        args = module.build_arg_parser().parse_args(
            [
                "--out-dir",
                str(out_dir),
                "--steps",
                "2",
                "--train-cases",
                "16",
                "--eval-cases",
                "4",
                "--program-len",
                "2",
                "--modulus",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--d-ff",
                "32",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--log-every",
                "0",
            ]
        )

        report = module.train_probe(args)

        self.assertEqual(report["target_level"], "L1 QTRM-native ETD/TRM scaffold")
        self.assertTrue((out_dir / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
