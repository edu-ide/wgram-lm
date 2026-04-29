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

    def test_core_halt_head_is_telemetry_only_when_halt_is_disabled(self):
        import torch
        from qtrm_mm import QTRMMultimodalModel

        cfg = self._cfg()
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core.halt_head.weight.zero_()
            model.core.halt_head.bias.copy_(torch.tensor([2.0, -2.0]))

        out = model(
            torch.randint(0, cfg.vocab_size, (2, 6)),
            enable_core_halt=False,
        )

        self.assertEqual(int(out["trajectory_len"].item()), cfg.outer_steps)
        self.assertEqual(out["core_q_halt_logits"].shape, (2, cfg.outer_steps))
        self.assertEqual(out["core_q_continue_logits"].shape, (2, cfg.outer_steps))
        self.assertTrue(torch.equal(out["core_halted"], torch.tensor([False, False])))
        self.assertTrue(torch.equal(out["core_steps"], torch.tensor([cfg.outer_steps, cfg.outer_steps])))


if __name__ == "__main__":
    unittest.main()
