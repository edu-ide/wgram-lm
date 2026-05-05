from pathlib import Path
import unittest


class CurrentArchPretrainProbeTests(unittest.TestCase):
    def test_probe_config_uses_current_residual_architecture(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml")

        self.assertEqual(cfg.donor.model_id, "Qwen/Qwen3.5-2B-Base")
        self.assertTrue(cfg.donor.load_in_4bit)
        self.assertTrue(cfg.donor.freeze_donor)
        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.model.qtrm_logits_scale, 0.1)
        self.assertGreaterEqual(cfg.model.workspace_tokens, 64)
        self.assertGreaterEqual(cfg.model.workspace_layers, 3)
        self.assertTrue(cfg.model.workspace_include_latents_in_kv)
        self.assertTrue(cfg.model.workspace_memory_gate_enabled)
        self.assertLessEqual(cfg.model.workspace_memory_gate_init_bias, -1.0)
        self.assertEqual(cfg.train.loss_jepa_weight, 0.0)
        self.assertEqual(cfg.train.loss_aux_weight, 0.0)
        self.assertGreaterEqual(cfg.train.steps, 2000)
        self.assertLessEqual(cfg.train.lr, 5.0e-5)
        self.assertIn("current_arch_pretrain_probe", cfg.train.out_dir)

    def test_probe_script_runs_training_with_diagnostics_and_post_eval(self):
        script = Path("scripts/105_run_current_arch_pretrain_probe.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml", text)
        self.assertIn("data/filtered/qtrm_clean_pilot.jsonl", text)
        self.assertIn("MULTIMODAL=0", text)
        self.assertIn("SAVE_EVERY", text)
        self.assertIn("--save-every", text)
        self.assertIn("--diag-every", text)
        self.assertIn("--diag-prompt", text)
        self.assertIn("scripts/92_eval_qtrm_logits.py", text)
        self.assertIn("post_eval.jsonl", text)

    def test_training_parser_accepts_save_every_flag(self):
        from qtrm_mm.training.train import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--config",
                "configs/qwen35_2b_4090_current_arch_pretrain_probe.yaml",
                "--save-every",
                "500",
            ]
        )

        self.assertEqual(args.save_every, 500)


if __name__ == "__main__":
    unittest.main()
