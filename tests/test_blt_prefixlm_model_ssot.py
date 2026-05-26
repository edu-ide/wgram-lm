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


class BLTPrefixLMModelSSOTTests(unittest.TestCase):
    def test_blt_prefixlm_model_is_importable_from_src(self) -> None:
        from qtrm_mm.models.blt_prefixlm import BLTDByteLatentPrefixLM

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
        from qtrm_mm.models.blt_prefixlm import BLTDByteLatentPrefixLM

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


if __name__ == "__main__":
    unittest.main()
