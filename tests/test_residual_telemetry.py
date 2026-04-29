import unittest

import torch


class ResidualTelemetryTests(unittest.TestCase):
    def test_reports_argmax_shift_and_residual_norms(self):
        from qtrm_mm.diagnostics import residual_logit_telemetry

        donor = torch.tensor([4.0, 1.0, 0.0])
        fused = torch.tensor([3.0, 5.0, 0.0])

        report = residual_logit_telemetry(donor, fused)

        self.assertEqual(report["donor_top_id"], 0)
        self.assertEqual(report["fused_top_id"], 1)
        self.assertTrue(report["argmax_changed"])
        self.assertGreater(report["kl_fused_to_donor"], 0.0)
        self.assertGreater(report["residual_l2_norm"], 0.0)
        self.assertEqual(report["residual_top_id"], 1)

    def test_applies_donor_scale_before_residual_comparison(self):
        from qtrm_mm.diagnostics import residual_logit_telemetry

        donor = torch.tensor([2.0, 0.0])
        fused = torch.tensor([2.5, 1.0])

        report = residual_logit_telemetry(donor, fused, donor_logits_scale=0.5)

        self.assertAlmostEqual(report["residual_linf_norm"], 1.5, places=5)
        self.assertEqual(report["donor_logits_scale"], 0.5)

    def test_eval_script_emits_residual_telemetry_records(self):
        from pathlib import Path

        text = Path("scripts/92_eval_qtrm_logits.py").read_text(encoding="utf-8")

        self.assertIn("residual_logit_telemetry", text)
        self.assertIn('"residual_telemetry"', text)

    def test_eval_script_emits_residual_gate_telemetry_records(self):
        from pathlib import Path

        text = Path("scripts/92_eval_qtrm_logits.py").read_text(encoding="utf-8")

        self.assertIn("residual_gate_telemetry", text)
        self.assertIn('"residual_gate"', text)
        self.assertIn("qtrm_residual_gate", text)


if __name__ == "__main__":
    unittest.main()
