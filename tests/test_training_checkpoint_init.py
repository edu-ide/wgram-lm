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


if __name__ == "__main__":
    unittest.main()
