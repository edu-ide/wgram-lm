from pathlib import Path
import unittest


class InferWithDonorScriptTest(unittest.TestCase):
    def test_infer_with_donor_script_exposes_logit_scale_overrides(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("DONOR_LOGITS_SCALE=${DONOR_LOGITS_SCALE:-}", script)
        self.assertIn("QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-}", script)
        self.assertIn("model.cfg.donor_logits_scale = float(donor_scale_override)", script)
        self.assertIn("model.cfg.qtrm_logits_scale = float(qtrm_scale_override)", script)
        self.assertIn("Donor logits scale:", script)
        self.assertIn("QTRM logits scale:", script)

    def test_infer_with_donor_script_exposes_bounded_residual_overrides(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("QTRM_RESIDUAL_CLAMP=${QTRM_RESIDUAL_CLAMP:-}", script)
        self.assertIn("QTRM_RESIDUAL_GATE=${QTRM_RESIDUAL_GATE:-}", script)
        self.assertIn("cfg.model.qtrm_residual_clamp = float(residual_clamp_override)", script)
        self.assertIn("cfg.model.qtrm_residual_gate_enabled = parse_bool(residual_gate_override)", script)
        self.assertIn("model.residual_gate.bias.data.fill_(float(residual_gate_bias_override))", script)
        self.assertIn("QTRM residual clamp:", script)
        self.assertIn("QTRM residual gate:", script)


if __name__ == "__main__":
    unittest.main()
