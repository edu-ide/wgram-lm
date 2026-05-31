from __future__ import annotations

import inspect
import unittest

import torch


class WGRAMV2CanonicalTests(unittest.TestCase):
    def test_v2_public_wgram_names_keep_legacy_aliases(self) -> None:
        from wgram_lm.v2 import QTRMReasoningLMV2, QTRMV2Config, WGRAMReasoningLMV2, WGRAMV2Config

        self.assertIs(WGRAMV2Config, QTRMV2Config)
        self.assertIs(WGRAMReasoningLMV2, QTRMReasoningLMV2)

    def test_v2_contract_rejects_noncanonical_answer_paths(self) -> None:
        from wgram_lm.v2 import WGRAMV2Config
        from wgram_lm.v2.contracts import build_v2_contract, validate_v2_contract

        cfg = WGRAMV2Config(
            vocab_size=64,
            d_model=16,
            patch_size=3,
            imta_trajectories=3,
            runtime_profile="promotion",
            allow_torch_smoke_core=False,
            core_implementation="official_gated_delta2",
        )
        contract = build_v2_contract(cfg)

        self.assertEqual(contract["answer_path"], "hnet_causal_speaker_same_lm_head")
        self.assertEqual(contract["model_class"], "wgram_lm.v2.model.WGRAMReasoningLMV2")
        self.assertEqual(contract["legacy_model_class"], "wgram_lm.v2.model.QTRMReasoningLMV2")
        self.assertEqual(
            contract["answer_transition_path"],
            "prompt_context_answer_memory_prefix_plan_commitment_then_causal_token_maturation_before_same_lm_head",
        )
        self.assertEqual(contract["evaluation_policy"], "free_generation_only")
        self.assertIn("RI-7", contract["ri_requirements"])
        self.assertTrue(validate_v2_contract(cfg, require_promotion_ready=True))

        for bad_cfg, message in [
            (WGRAMV2Config(forced_choice_promotion_enabled=True), "forced-choice"),
            (WGRAMV2Config(candidate_rerank_promotion_enabled=True), "candidate rerank"),
            (WGRAMV2Config(external_gram_ptrm_answer_selection=True), "external GRAM/PTRM"),
            (WGRAMV2Config(lewm_answer_path_enabled=True), "LeWM"),
            (WGRAMV2Config(boundary_state_source="boundary_byte"), "causal chunk summary"),
            (WGRAMV2Config(answer_head_count=2), "single same LM head"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                validate_v2_contract(bad_cfg)

        smoke_cfg = WGRAMV2Config(runtime_profile="smoke", allow_torch_smoke_core=True)
        with self.assertRaisesRegex(ValueError, "promotion-ready"):
            validate_v2_contract(smoke_cfg, require_promotion_ready=True)

        mislabeled_smoke_cfg = WGRAMV2Config(
            runtime_profile="promotion",
            allow_torch_smoke_core=False,
            core_implementation="torch_smoke",
        )
        with self.assertRaisesRegex(ValueError, "promotion-ready"):
            validate_v2_contract(mislabeled_smoke_cfg, require_promotion_ready=True)

    def test_v2_official_core_factory_is_promotion_ready_and_not_smoke(self) -> None:
        from wgram_lm.v2 import WGRAMReasoningLMV2, WGRAMV2Config
        from wgram_lm.v2.recurrent_core import OfficialGatedDeltaNet2Core, build_v2_recurrent_core

        cfg = WGRAMV2Config(
            vocab_size=64,
            d_model=16,
            local_heads=4,
            core_layers=1,
            runtime_profile="promotion",
            allow_torch_smoke_core=False,
            core_implementation="official_gated_delta2",
            official_gdn2_use_short_conv=False,
        )

        core = build_v2_recurrent_core(cfg)
        model = WGRAMReasoningLMV2(cfg)

        self.assertIsInstance(core, OfficialGatedDeltaNet2Core)
        self.assertIsInstance(model.core, OfficialGatedDeltaNet2Core)
        self.assertFalse(getattr(model.core, "is_torch_smoke_core", True))

    @unittest.skipUnless(torch.cuda.is_available(), "official GDN2 smoke requires CUDA/Triton")
    def test_v2_official_core_forward_smoke_on_cuda(self) -> None:
        from wgram_lm.v2 import WGRAMReasoningLMV2, WGRAMV2Config

        cfg = WGRAMV2Config(
            vocab_size=64,
            d_model=16,
            patch_size=4,
            local_layers=1,
            local_heads=4,
            core_layers=1,
            imta_trajectories=1,
            runtime_profile="promotion",
            allow_torch_smoke_core=False,
            core_implementation="official_gated_delta2",
            official_gdn2_use_short_conv=False,
            force_fixed_boundaries=True,
        )
        model = WGRAMReasoningLMV2(cfg).cuda().eval()
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits, hidden, metrics = model.forward_logits_and_hidden(input_ids, attention_mask, think_steps=1)

        self.assertEqual(tuple(logits.shape), (1, 8, cfg.vocab_size))
        self.assertEqual(tuple(hidden.shape), (1, 8, cfg.d_model))
        self.assertEqual(metrics["core_implementation"], "official_gated_delta2")
        self.assertTrue(metrics["core_attention_causal"])
        self.assertEqual(metrics["answer_path"], "hnet_causal_speaker_same_lm_head")
        self.assertEqual(
            metrics["answer_transition_path"],
            "prompt_context_answer_memory_prefix_plan_commitment_then_causal_token_maturation_before_same_lm_head",
        )
        self.assertEqual(metrics["answer_memory_mode"], "disabled")
        self.assertEqual(metrics["token_maturation_mode"], "causal_latent_refinement_same_lm_head")

    def test_chunk_encoder_is_causal_and_uses_nonboundary_bytes(self) -> None:
        from wgram_lm.v2 import WGRAMV2Config
        from wgram_lm.v2.chunk_encoder import CausalByteChunkEncoder

        torch.manual_seed(101)
        cfg = WGRAMV2Config(vocab_size=80, d_model=12, patch_size=3, force_fixed_boundaries=True)
        encoder = CausalByteChunkEncoder(cfg)
        byte_embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        attention_mask = torch.ones((1, 6), dtype=torch.long)
        ids_a = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        ids_b = torch.tensor([[2, 13, 4, 5, 6, 7]], dtype=torch.long)
        ids_future_changed = torch.tensor([[2, 3, 4, 5, 16, 7]], dtype=torch.long)

        enc_a = encoder(byte_embed(ids_a), ids_a, attention_mask)
        enc_b = encoder(byte_embed(ids_b), ids_b, attention_mask)
        enc_future = encoder(byte_embed(ids_future_changed), ids_future_changed, attention_mask)

        self.assertEqual(tuple(enc_a.chunk_states.shape), (1, 2, cfg.d_model))
        self.assertEqual(enc_a.metrics["boundary_state_source"], "causal_chunk_summary")
        self.assertEqual(enc_a.metrics["boundary_selection_mode"], "fixed_cap")
        self.assertEqual(enc_a.metrics["dechunk_context_mode"], "completed_chunk_or_bos")
        self.assertGreater(enc_a.metrics["causal_chunk_summary_nonboundary_tokens"], 0)
        self.assertEqual(enc_a.dechunk_indices.tolist(), [[0, 0, 1, 1, 1, 2]])
        self.assertEqual(
            enc_a.dechunk_has_completed_chunk.tolist(),
            [[False, False, True, True, True, True]],
        )
        self.assertGreater((enc_a.chunk_states[:, 0] - enc_b.chunk_states[:, 0]).abs().max().item(), 1.0e-6)
        self.assertLess((enc_a.chunk_states[:, 0] - enc_future.chunk_states[:, 0]).abs().max().item(), 1.0e-7)
        self.assertGreater((enc_a.chunk_states[:, 1] - enc_future.chunk_states[:, 1]).abs().max().item(), 1.0e-6)

    def test_v2_token_logits_do_not_depend_on_future_input_tokens(self) -> None:
        from wgram_lm.v2 import WGRAMReasoningLMV2, WGRAMV2Config

        torch.manual_seed(303)
        cfg = WGRAMV2Config(
            vocab_size=96,
            d_model=16,
            patch_size=3,
            local_layers=1,
            local_heads=4,
            core_layers=1,
            imta_trajectories=1,
            runtime_profile="smoke",
            allow_torch_smoke_core=True,
            force_fixed_boundaries=True,
        )
        model = WGRAMReasoningLMV2(cfg).eval()
        input_a = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        input_b = torch.tensor([[2, 3, 4, 55, 66, 77]], dtype=torch.long)
        attention_mask = torch.ones_like(input_a)

        with torch.no_grad():
            logits_a, _, _ = model.forward_logits_and_hidden(input_a, attention_mask, think_steps=1)
            logits_b, _, _ = model.forward_logits_and_hidden(input_b, attention_mask, think_steps=1)

        self.assertLess((logits_a[:, :3] - logits_b[:, :3]).abs().max().item(), 1.0e-6)
        self.assertGreater((logits_a[:, 3:] - logits_b[:, 3:]).abs().max().item(), 1.0e-6)

    def test_v2_model_forward_uses_imta_same_speaker_and_own_latent_auxiliary(self) -> None:
        from wgram_lm.v2 import WGRAMReasoningLMV2, WGRAMV2Config

        torch.manual_seed(202)
        cfg = WGRAMV2Config(
            vocab_size=96,
            d_model=16,
            patch_size=3,
            local_layers=1,
            local_heads=4,
            core_layers=1,
            imta_trajectories=3,
            runtime_profile="smoke",
            allow_torch_smoke_core=True,
            own_latent_prediction_weight=0.05,
            imta_diversity_weight=0.03,
            imta_route_min_probability=0.05,
            imta_route_entropy_floor=0.35,
            imta_route_entropy_weight=0.02,
            imta_route_balance_weight=0.005,
            answer_memory_stop_margin_loss_weight=0.1,
            answer_memory_prompt_context_enabled=True,
            response_continue_stop_margin_weight=0.1,
            force_fixed_boundaries=True,
        )
        model = WGRAMReasoningLMV2(cfg)
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        labels = torch.tensor([[-100, 3, 4, 5, 6, 7]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        response_prediction_mask = (labels != -100).to(torch.long)
        response_start_mask = torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.long)

        logits, hidden, metrics = model.forward_logits_and_hidden(
            input_ids,
            attention_mask,
            think_steps=2,
            response_prediction_mask=response_prediction_mask,
        )
        _, _, scheduled_off_metrics = model.forward_logits_and_hidden(
            input_ids,
            attention_mask,
            think_steps=2,
            response_prediction_mask=response_prediction_mask,
            answer_memory_prompt_context_scale=0.0,
        )
        loss, loss_metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=2,
            response_start_mask=response_start_mask,
        )

        self.assertEqual(tuple(logits.shape), (1, 6, cfg.vocab_size))
        self.assertEqual(tuple(hidden.shape), (1, 6, cfg.d_model))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["imta_trajectory_count"], 3)
        self.assertEqual(metrics["answer_path"], "hnet_causal_speaker_same_lm_head")
        self.assertEqual(
            metrics["answer_transition_path"],
            "prompt_context_answer_memory_prefix_plan_commitment_then_causal_token_maturation_before_same_lm_head",
        )
        self.assertIn("answer_memory_mode", metrics)
        self.assertEqual(metrics["answer_memory_plan_tokens"], 4)
        self.assertEqual(metrics["answer_memory_plan_layers"], 1)
        self.assertEqual(
            metrics["answer_memory_prompt_context_mode"],
            "same_body_causal_prompt_context_read",
        )
        self.assertGreater(metrics["answer_memory_prompt_context_tokens_mean"], 0.0)
        self.assertGreater(metrics["answer_memory_prompt_context_gate_mean"], 0.0)
        self.assertEqual(metrics["answer_memory_prompt_context_scale"], 1.0)
        self.assertEqual(scheduled_off_metrics["answer_memory_prompt_context_mode"], "disabled")
        self.assertEqual(scheduled_off_metrics["answer_memory_prompt_context_scale"], 0.0)
        self.assertEqual(
            metrics["answer_memory_commitment_mode"],
            "same_lm_head_answer_prefix_state_commitment",
        )
        self.assertGreater(metrics["answer_memory_commitment_positions"], 0)
        self.assertGreater(metrics["answer_memory_commitment_gate_mean"], 0.0)
        self.assertEqual(metrics["answer_memory_commitment_scale"], 1.0)
        self.assertEqual(metrics["answer_memory_commitment_confidence_gate"], "disabled")
        self.assertEqual(metrics["answer_memory_commitment_confidence_scale_mean"], 1.0)
        self.assertEqual(metrics["answer_memory_confidence_gate"], "same_lm_head_plan_confidence_topk_mass")
        self.assertEqual(metrics["answer_memory_confidence_mode"], "topk_mass")
        self.assertEqual(metrics["answer_memory_confidence_topk"], 5)
        self.assertEqual(metrics["answer_memory_injection_context"], "answer_prefix_only_no_tail_clamp")
        self.assertGreater(metrics["answer_memory_injection_positions"], 0)
        self.assertGreaterEqual(metrics["answer_memory_plan_confidence_mean"], 0.0)
        self.assertGreaterEqual(metrics["answer_memory_plan_top1_confidence_mean"], 0.0)
        self.assertGreaterEqual(metrics["answer_memory_plan_topk_mass_mean"], 0.0)
        self.assertGreaterEqual(metrics["answer_memory_plan_entropy_complement_mean"], 0.0)
        self.assertEqual(metrics["token_maturation_steps"], 2)
        self.assertGreater(metrics["token_maturation_delta_norm"], 0.0)
        self.assertTrue(metrics["core_attention_causal"])
        self.assertEqual(metrics["speaker_position_encoding"], "learned_absolute")
        self.assertEqual(metrics["speaker_response_phase_encoding"], "learned_segment_relative")
        self.assertFalse(metrics["speaker_input_output_embeddings_tied"])
        self.assertTrue(metrics["speaker_adaptive_latent_bridge"])
        self.assertGreater(metrics["speaker_adaptive_latent_bridge_gate_mean"], 0.0)
        self.assertEqual(metrics["boundary_state_source"], "causal_chunk_summary")
        self.assertEqual(metrics["dechunk_context_mode"], "completed_chunk_or_bos")
        self.assertIn("dynamic_boundary_mean_probability", metrics)
        self.assertGreater(metrics["imta_adapter_delta_norm"], 0.0)
        self.assertGreaterEqual(metrics["imta_post_adapter_delta_norm"], 0.0)
        self.assertGreater(metrics["imta_raw_selector_effective_routes"], 1.0)
        self.assertGreater(metrics["imta_selector_effective_routes"], 1.0)
        self.assertEqual(metrics["imta_route_min_probability"], 0.05)
        self.assertEqual(metrics["imta_route_entropy_floor"], 0.35)
        self.assertGreaterEqual(metrics["imta_route_entropy_loss"], 0.0)
        self.assertGreaterEqual(metrics["imta_route_balance_loss"], 0.0)
        self.assertGreaterEqual(loss_metrics["own_latent_prediction_loss"], 0.0)
        self.assertIn("token_maturation_aux_loss", loss_metrics)
        self.assertIn("token_maturation_aux_loss_weight", loss_metrics)
        self.assertIn("answer_prefix_commitment_loss", loss_metrics)
        self.assertIn("answer_prefix_commitment_loss_weight", loss_metrics)
        self.assertIn("answer_memory_aux_loss", loss_metrics)
        self.assertIn("answer_memory_aux_loss_weight", loss_metrics)
        self.assertIn("answer_memory_aux_tokens", loss_metrics)
        self.assertIn("answer_memory_stop_margin_loss", loss_metrics)
        self.assertIn("answer_memory_stop_margin_positions", loss_metrics)
        self.assertGreater(loss_metrics["own_latent_prediction_targets"], 0)
        self.assertEqual(loss_metrics["own_latent_prediction_target_source"], "next_causal_chunk_state")
        self.assertGreaterEqual(loss_metrics["imta_diversity_loss"], 0.0)
        self.assertEqual(loss_metrics["imta_route_entropy_weight"], 0.02)
        self.assertEqual(loss_metrics["imta_route_balance_weight"], 0.005)
        self.assertGreaterEqual(loss_metrics["imta_route_entropy_loss"], 0.0)
        self.assertGreaterEqual(loss_metrics["imta_route_balance_loss"], 0.0)
        self.assertEqual(loss_metrics["response_continue_stop_margin_weight"], 0.1)
        self.assertIn("response_continue_stop_margin_loss", loss_metrics)
        self.assertIn("response_continue_stop_margin_positions", loss_metrics)
        self.assertIs(model.speaker.head.weight, model.speaker.decoder.head.weight)
        self.assertIsNot(model.speaker.head.weight, model.byte_embed.weight)

    def test_v2_generation_api_is_free_autoregressive_only(self) -> None:
        from wgram_lm.v2 import WGRAMReasoningLMV2, WGRAMV2Config
        from wgram_lm.v2.generation import build_v2_generation_policy, generate_free, generation_repetition_stats

        cfg = WGRAMV2Config(
            vocab_size=48,
            d_model=12,
            patch_size=3,
            local_layers=1,
            local_heads=3,
            core_layers=1,
            runtime_profile="smoke",
            allow_torch_smoke_core=True,
            force_fixed_boundaries=True,
        )
        model = WGRAMReasoningLMV2(cfg)
        generated = generate_free(model, [2, 3, 4], max_new_tokens=4, eos_id=1)
        default_policy = build_v2_generation_policy()
        policy = build_v2_generation_policy(repetition_penalty=1.2)
        stats = generation_repetition_stats([11, 11, 11, 11])
        alternating_stats = generation_repetition_stats([48, 121, 48, 121, 48, 121, 48, 121])
        stop_stats = generation_repetition_stats([11])
        signature = inspect.signature(generate_free)

        self.assertLessEqual(len(generated), 4)
        self.assertEqual(default_policy["repetition_penalty"], 1.0)
        self.assertEqual(default_policy["repetition_penalty_mode"], "disabled")
        self.assertEqual(default_policy["promotion_evidence_eligible"], "true")
        self.assertEqual(policy["promotion_policy"], "free_generation_only")
        self.assertEqual(policy["repetition_penalty"], 1.2)
        self.assertEqual(policy["repetition_penalty_mode"], "diagnostic_only")
        self.assertEqual(policy["promotion_evidence_eligible"], "false")
        self.assertTrue(stats["loop_like"])
        self.assertEqual(stats["max_consecutive_run"], 4)
        self.assertTrue(alternating_stats["loop_like"])
        self.assertEqual(alternating_stats["best_periodic_repeat_period"], 2)
        self.assertGreaterEqual(alternating_stats["best_periodic_repeat_fraction"], 0.80)
        self.assertFalse(stop_stats["loop_like"])
        self.assertNotIn("choices", signature.parameters)
        self.assertNotIn("candidate_answers", signature.parameters)


if __name__ == "__main__":
    unittest.main()
