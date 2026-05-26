from __future__ import annotations

import unittest

import torch


class BLTComponentsSSOTTests(unittest.TestCase):
    def test_blt_local_decoder_components_are_importable_from_src(self) -> None:
        from qtrm_mm.models.blt_components import BLTDLocalDecoder, NextImplicitByteProjector

        projector = NextImplicitByteProjector(d_model=4, hidden_dim=8)
        hidden = torch.randn(2, 3, 4)
        projected = projector(hidden)

        decoder = BLTDLocalDecoder(
            d_model=4,
            vocab_size=16,
            patch_size=3,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            causal=True,
        )
        logits = decoder(torch.randn(2, 3, 4))

        self.assertEqual(tuple(projected.shape), (2, 3, 4))
        self.assertEqual(tuple(logits.shape), (2, 3, 16))


if __name__ == "__main__":
    unittest.main()
