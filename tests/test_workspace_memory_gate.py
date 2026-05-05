import unittest

import torch


class WorkspaceMemoryGateTests(unittest.TestCase):
    def test_gated_workspace_reports_per_layer_update_gate(self):
        from qtrm_mm.workspace import LatentWorkspace

        workspace = LatentWorkspace(
            d_model=16,
            workspace_tokens=3,
            n_heads=4,
            layers=2,
            ff_mult=1,
            memory_gate_enabled=True,
            memory_gate_init_bias=-1.0,
        )
        context = torch.randn(2, 5, 16)
        context_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        )

        latents, info = workspace(context, context_mask=context_mask, return_info=True)

        self.assertEqual(latents.shape, (2, 3, 16))
        self.assertEqual(info["update_gate_mean"].shape, (2, 2))
        self.assertTrue(torch.all(info["update_gate_mean"] >= 0.0))
        self.assertTrue(torch.all(info["update_gate_mean"] <= 1.0))

    def test_qtrm_config_wires_gated_workspace_to_forward_outputs(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=5,
            workspace_layers=3,
            workspace_ff_mult=1,
            workspace_memory_gate_enabled=True,
            workspace_memory_gate_init_bias=-1.0,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        out = model(input_ids)

        self.assertEqual(out["z_h"].shape, (2, cfg.workspace_tokens, cfg.d_model))
        self.assertEqual(out["workspace_update_gate_mean"].shape, (2, cfg.workspace_layers))

    def test_qtrm_forward_can_disable_workspace_memory_gate(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=5,
            workspace_layers=2,
            workspace_memory_gate_enabled=True,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=16,
            max_visual_tokens=4,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 7))

        full = model(input_ids)
        gate_off = model(input_ids, disable_workspace_memory_gate=True)

        self.assertEqual(full["workspace_update_gate_mean"].shape, (2, cfg.workspace_layers))
        self.assertEqual(gate_off["workspace_update_gate_mean"].shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
