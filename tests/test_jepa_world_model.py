import unittest

import torch


class JepaWorldModelTests(unittest.TestCase):
    def test_world_model_builds_next_latent_targets_and_transition_mask(self):
        from qtrm_mm.world_model import JepaWorldModelHead

        head = JepaWorldModelHead(
            d_model=16,
            n_heads=4,
            num_actions=3,
            predictor_layers=1,
            predictor_dim=32,
        )
        latents = torch.randn(2, 5, 16, requires_grad=True)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        out = head(latents, attention_mask=attention_mask)

        self.assertEqual(out["pred"].shape, (2, 4, 16))
        self.assertEqual(out["target"].shape, (2, 4, 16))
        self.assertTrue(out["target"].requires_grad)
        self.assertTrue(out["pred"].requires_grad)
        self.assertTrue(
            torch.equal(
                out["mask"],
                torch.tensor(
                    [
                        [True, True, False, False],
                        [True, True, True, True],
                    ]
                ),
            )
        )

    def test_world_model_loss_backprops_through_future_target_for_end_to_end_lewm(self):
        from qtrm_mm.losses import jepa_world_model_loss
        from qtrm_mm.world_model import JepaWorldModelHead

        torch.manual_seed(0)
        head = JepaWorldModelHead(
            d_model=8,
            n_heads=2,
            num_actions=2,
            predictor_layers=1,
            predictor_dim=16,
        )
        latents = torch.randn(1, 4, 8, requires_grad=True)
        out = head(latents)

        loss = jepa_world_model_loss(out["pred"], out["target"], out["mask"])
        loss.backward()

        self.assertIsNotNone(latents.grad)
        self.assertGreater(float(latents.grad[:, :-1].abs().sum()), 0.0)
        self.assertGreater(float(latents.grad[:, -1].abs().sum()), 0.0)

    def test_world_model_loss_accepts_sigreg_regularizer(self):
        from qtrm_mm.losses import jepa_world_model_loss
        from qtrm_mm.world_model import SIGReg

        pred = torch.randn(2, 3, 8, requires_grad=True)
        target = torch.randn(2, 3, 8, requires_grad=True)
        latents = torch.randn(2, 4, 8, requires_grad=True)
        mask = torch.ones(2, 3, dtype=torch.bool)
        latent_mask = torch.ones(2, 4, dtype=torch.bool)
        sigreg = SIGReg(knots=5, num_proj=16)

        loss = jepa_world_model_loss(
            pred,
            target,
            mask,
            latents=latents,
            latent_mask=latent_mask,
            sigreg=sigreg,
            sigreg_weight=0.09,
        )
        loss.backward()

        self.assertGreater(float(pred.grad.abs().sum()), 0.0)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)
        self.assertGreater(float(latents.grad.abs().sum()), 0.0)

    def test_qtrm_forward_exposes_sequence_jepa_world_model_outputs(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=32,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=1,
            workspace_tokens=4,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            attn_every=1,
            visual_dim=16,
            max_visual_tokens=4,
            jepa_predictor_layers=1,
            jepa_predictor_dim=64,
        )
        model = QTRMMultimodalModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 6))
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )

        out = model(ids, attention_mask=attention_mask)

        self.assertEqual(out["jepa_pred"].shape, (2, 5, cfg.d_model))
        self.assertEqual(out["jepa_target"].shape, (2, 5, cfg.d_model))
        self.assertTrue(
            torch.equal(
                out["jepa_mask"],
                torch.tensor(
                    [
                        [True, True, True, False, False],
                        [True, True, True, True, True],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
