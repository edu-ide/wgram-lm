from pathlib import Path
import unittest


class CoreHaltProbeTests(unittest.TestCase):
    def test_probe_config_trains_auto_halt_targets(self):
        from wgram_lm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_core_halt_probe.yaml")

        self.assertTrue(cfg.model.core_halt_enabled)
        self.assertGreaterEqual(cfg.model.outer_steps, 3)
        self.assertEqual(cfg.model.coda_attn_every, 2)
        self.assertGreater(cfg.train.loss_core_halt_weight, 0.0)
        self.assertTrue(cfg.train.core_halt_auto_targets)
        self.assertEqual(cfg.train.core_halt_target_mode, "teacher_depth")
        self.assertGreater(cfg.train.core_halt_teacher_depth_threshold, 0.999)
        self.assertLess(cfg.train.core_halt_teacher_depth_threshold, 1.0)
        self.assertGreater(cfg.train.core_halt_teacher_depth_logit_kl_threshold, 0.0)
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

    def test_memory_halt_preserve_script_trains_halt_only_and_runs_memoryos_gates(self):
        script = Path("scripts/108_run_memory_halt_preserve.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("configs/qwen35_2b_4090_memory_halt_preserve_s050.yaml", text)
        self.assertIn("memory_reasoning_synth_traces.jsonl", text)
        self.assertIn("qwen35_2b_4090_memory_synth_generalization_s050/last.pt", text)
        self.assertIn("--init-checkpoint", text)
        self.assertIn('--core-halt-mode "$mode"', text)
        self.assertIn('eval_gate "$HARD_CASES" "$HARD_INDEX" disabled', text)
        self.assertIn('eval_gate "$HARD_CASES" "$HARD_INDEX" enabled', text)
        self.assertIn('eval_gate "$HELDOUT_CASES" "$HELDOUT_INDEX" disabled', text)
        self.assertIn('eval_gate "$HELDOUT_CASES" "$HELDOUT_INDEX" enabled', text)
        self.assertIn("--memory-link-expansion", text)
        self.assertIn("memory_reasoning_probe.jsonl", text)
        self.assertIn("memory_reasoning_heldout_probe.jsonl", text)

    def test_donor_anneal_config_reduces_donor_logits_dependency(self):
        from wgram_lm.config import load_config

        cfg = load_config("configs/qwen35_2b_4090_donor_anneal_probe.yaml")

        self.assertEqual(cfg.model.donor_logits_scale, 1.0)
        self.assertEqual(cfg.train.donor_logits_scale_start, 1.0)
        self.assertEqual(cfg.train.donor_logits_scale_end, 0.0)
        self.assertGreater(cfg.train.loss_student_lm_weight, 0.0)
        self.assertGreater(cfg.train.loss_donor_kl_weight, 0.0)
        self.assertEqual(cfg.train.donor_kl_beta, 1.0)
        self.assertIn("donor_anneal", cfg.train.out_dir)


if __name__ == "__main__":
    unittest.main()
