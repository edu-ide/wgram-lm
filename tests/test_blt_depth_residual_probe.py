from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "560_eval_blt_depth_residual_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_depth_residual_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BLTDepthResidualProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_probe_imports_blt_model_from_src_ssot(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("from wgram_lm.models.blt_prefixlm import BLTDByteLatentPrefixLM", source)
        self.assertNotIn("trainer.BLTDByteLatentPrefixLM", source)

    def test_summarize_depth_rows_uses_token_weighted_loss_and_residual(self) -> None:
        rows = [
            {"think_steps": 1, "loss": 4.0, "target_tokens": 2, "fixed_point_residual": 0.8},
            {"think_steps": 1, "loss": 2.0, "target_tokens": 6, "fixed_point_residual": 0.4},
            {"think_steps": 4, "loss": 1.0, "target_tokens": 3, "fixed_point_residual": 0.2},
        ]

        report = self.module.build_depth_residual_report(rows, checkpoint="toy.pt")

        self.assertEqual(report["checkpoint"], "toy.pt")
        depth1 = report["depth_summaries"][0]
        depth4 = report["depth_summaries"][1]
        self.assertEqual(depth1["think_steps"], 1)
        self.assertAlmostEqual(depth1["loss"], 2.5)
        self.assertAlmostEqual(depth1["mean_fixed_point_residual"], 0.5)
        self.assertEqual(depth1["target_tokens"], 8)
        self.assertEqual(depth4["think_steps"], 4)
        self.assertAlmostEqual(report["best_loss"], 1.0)
        self.assertEqual(report["best_loss_depth"], 4)

    def test_report_accepts_only_when_deeper_loss_and_residual_improve(self) -> None:
        accepted = self.module.build_depth_residual_report(
            [
                {"think_steps": 1, "loss": 3.0, "target_tokens": 2, "fixed_point_residual": 0.7},
                {"think_steps": 4, "loss": 2.0, "target_tokens": 2, "fixed_point_residual": 0.2},
            ],
            checkpoint="toy.pt",
        )
        rejected = self.module.build_depth_residual_report(
            [
                {"think_steps": 1, "loss": 2.0, "target_tokens": 2, "fixed_point_residual": 0.2},
                {"think_steps": 4, "loss": 3.0, "target_tokens": 2, "fixed_point_residual": 0.7},
            ],
            checkpoint="toy.pt",
        )

        self.assertTrue(accepted["accepted"])
        self.assertIn("deepest_loss_beats_shallowest", accepted["passed_checks"])
        self.assertIn("deepest_residual_below_shallowest", accepted["passed_checks"])
        self.assertFalse(rejected["accepted"])
        self.assertIn("no_depth_loss_gain", rejected["failed_checks"])


if __name__ == "__main__":
    unittest.main()
