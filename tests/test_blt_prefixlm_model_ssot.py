from __future__ import annotations

import unittest

import torch
from torch import nn


class IdentityGlobalCore(nn.Module):
    position_embedding_mode = "none"

    def _forward_embedded_impl(
        self,
        x: torch.Tensor,
        *,
        think_steps: int,
        return_hidden: bool,
    ) -> torch.Tensor:
        return x + (0.0 * float(think_steps))


class StepSensitiveGlobalCore(nn.Module):
    position_embedding_mode = "none"

    def _forward_embedded_impl(
        self,
        x: torch.Tensor,
        *,
        think_steps: int,
        return_hidden: bool,
    ) -> torch.Tensor:
        direction = torch.linspace(-1.0, 1.0, steps=x.shape[-1], dtype=x.dtype, device=x.device)
        return x + float(think_steps) * 0.125 * direction.view(1, 1, -1)


class BLTPrefixLMModelSSOTTests(unittest.TestCase):
    def test_blt_prefixlm_model_is_importable_from_src(self) -> None:
        from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

        model = BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=16,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = model.forward_logits(input_ids, attention_mask, think_steps=2)

        self.assertEqual(tuple(logits.shape), (1, 4, 16))

    def test_external_register_changes_same_lm_head_logits(self) -> None:
        from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

        torch.manual_seed(7)
        model = BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=16,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        base_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)
        register_logits = model.forward_logits(
            input_ids,
            attention_mask,
            think_steps=2,
            external_register=torch.tensor([[1.0, 0.0, 0.5, -1.0]]),
        )

        self.assertEqual(tuple(register_logits.shape), tuple(base_logits.shape))
        self.assertFalse(torch.allclose(base_logits, register_logits))

    def test_hnet_one_body_uses_gated_latent_causal_speaker(self) -> None:
        from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

        torch.manual_seed(11)
        model = BLTDByteLatentPrefixLM(
            global_core=StepSensitiveGlobalCore(),
            vocab_size=32,
            d_model=8,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
            patch_boundary_mode="hnet_dechunk",
            hnet_one_body_byte_gate_init=-4.0,
            hnet_one_body_latent_gate_init=4.0,
        )
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        logits_1, hidden_1 = model.forward_logits_and_decoder_hidden(input_ids, attention_mask, think_steps=1)
        metrics = dict(model.last_pack_metrics)
        logits_4, hidden_4 = model.forward_logits_and_decoder_hidden(input_ids, attention_mask, think_steps=4)

        self.assertEqual(tuple(logits_1.shape), (1, 6, 32))
        self.assertEqual(tuple(hidden_1.shape), (1, 6, 8))
        self.assertFalse(torch.allclose(logits_1, logits_4))
        self.assertFalse(torch.allclose(hidden_1, hidden_4))
        self.assertLess(metrics["hnet_byte_residual_gate"], 0.05)
        self.assertGreater(metrics["hnet_latent_residual_gate"], 0.95)
        self.assertIs(model.hnet_causal_speaker.head.weight, model.hnet_byte_speaker[-1].weight)

    def test_hnet_dechunk_causal_summary_uses_nonboundary_bytes(self) -> None:
        from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

        torch.manual_seed(17)
        model = BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=32,
            d_model=8,
            patch_size=4,
            dynamic_min_patch_size=4,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
            patch_boundary_mode="hnet_dechunk",
            hbf_boundary_threshold=2.0,
            hnet_one_body_byte_gate_init=-20.0,
            hnet_one_body_latent_gate_init=4.0,
        )
        model.eval()
        input_a = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        input_b = torch.tensor([[2, 3, 12, 5, 6, 7, 8, 9]], dtype=torch.long)
        attention_mask = torch.ones_like(input_a)

        hidden_a, *_ = model._hnet_boundary_states(input_a, attention_mask, think_steps=1)
        metrics_a = dict(model.last_pack_metrics)
        hidden_b, *_ = model._hnet_boundary_states(input_b, attention_mask, think_steps=1)

        self.assertEqual(metrics_a["hnet_selected_len"], 2)
        self.assertGreater(metrics_a["hnet_causal_chunk_summary_nonboundary_tokens"], 0)
        later_delta = (hidden_a[:, 4:] - hidden_b[:, 4:]).abs().max().item()
        self.assertGreater(later_delta, 1.0e-5)

    def test_hnet_imta_uses_internal_trajectory_breadth_before_same_speaker(self) -> None:
        from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

        torch.manual_seed(19)
        model = BLTDByteLatentPrefixLM(
            global_core=StepSensitiveGlobalCore(),
            vocab_size=32,
            d_model=8,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
            patch_boundary_mode="hnet_dechunk",
            hnet_one_body_byte_gate_init=-4.0,
            hnet_one_body_latent_gate_init=4.0,
            imta_trajectories=3,
            imta_noise_std=0.0,
            imta_selector_temperature=0.7,
        )
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        logits, hidden = model.forward_logits_and_decoder_hidden(input_ids, attention_mask, think_steps=3)
        metrics = dict(model.last_pack_metrics)

        self.assertEqual(tuple(logits.shape), (1, 6, 32))
        self.assertEqual(tuple(hidden.shape), (1, 6, 8))
        self.assertEqual(metrics["imta_trajectory_count"], 3)
        self.assertEqual(metrics["imta_active"], 1)
        self.assertGreater(metrics["imta_selector_entropy"], 0.0)
        self.assertGreater(metrics["imta_selector_confidence"], 0.0)
        self.assertGreater(metrics["imta_trajectory_state_std"], 0.0)
        self.assertGreater(metrics["imta_adapter_gate_mean"], 0.0)
        self.assertGreater(metrics["imta_adapter_delta_norm"], 0.0)
        self.assertGreaterEqual(metrics["imta_diversity_loss"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["imta_diversity_mean_cosine"])))
        self.assertIs(model.hnet_causal_speaker.head.weight, model.hnet_byte_speaker[-1].weight)

        labels = torch.tensor([[-100, -100, 4, 5, 6, 7]], dtype=torch.long)
        loss, loss_metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=3,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
            imta_diversity_weight=0.07,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss_metrics["imta_diversity_weight"], 0.07)
        self.assertGreaterEqual(loss_metrics["imta_diversity_loss"], 0.0)

    def test_hnet_own_latent_prediction_is_auxiliary_not_answer_head(self) -> None:
        from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM

        torch.manual_seed(23)
        model = BLTDByteLatentPrefixLM(
            global_core=StepSensitiveGlobalCore(),
            vocab_size=32,
            d_model=8,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
            patch_boundary_mode="hnet_dechunk",
            hnet_one_body_byte_gate_init=-4.0,
            hnet_one_body_latent_gate_init=4.0,
            imta_trajectories=2,
        )
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 4, 5, 6, 7]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=3,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
            own_latent_prediction_weight=0.05,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["own_latent_prediction_enabled"], 1)
        self.assertEqual(metrics["own_latent_prediction_weight"], 0.05)
        self.assertGreater(metrics["own_latent_prediction_targets"], 0)
        self.assertGreaterEqual(metrics["own_latent_prediction_loss"], 0.0)
        self.assertIs(model.hnet_causal_speaker.head.weight, model.hnet_byte_speaker[-1].weight)


if __name__ == "__main__":
    unittest.main()
