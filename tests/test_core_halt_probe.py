from pathlib import Path
import unittest


class CoreHaltProbeTests(unittest.TestCase):
    def test_probe_config_trains_auto_halt_targets(self):
        from qtrm_mm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_core_halt_probe.yaml")

        self.assertTrue(cfg.model.core_halt_enabled)
        self.assertGreaterEqual(cfg.model.outer_steps, 3)
        self.assertGreater(cfg.train.loss_core_halt_weight, 0.0)
        self.assertTrue(cfg.train.core_halt_auto_targets)
        self.assertIsNotNone(cfg.train.core_halt_donor_kl_threshold)
        self.assertEqual(cfg.train.loss_jepa_weight, 0.0)
        self.assertEqual(cfg.train.loss_aux_weight, 0.0)
        self.assertIn("core_halt_probe", cfg.train.out_dir)

    def test_probe_script_warms_from_existing_checkpoint_and_records_halt_eval(self):
        script = Path("scripts/107_run_core_halt_probe.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("configs/qwen35_2b_4090_core_halt_probe.yaml", text)
        self.assertIn("INIT_CHECKPOINT", text)
        self.assertIn("--init-checkpoint", text)
        self.assertIn("--enable-core-halt", text)
        self.assertIn("post_eval_core_halt.jsonl", text)
        self.assertIn("scripts/92_eval_qtrm_logits.py", text)


if __name__ == "__main__":
    unittest.main()
