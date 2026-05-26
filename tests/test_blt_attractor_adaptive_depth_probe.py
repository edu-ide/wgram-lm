from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "561_eval_blt_attractor_adaptive_depth_from_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_attractor_adaptive_depth_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BLTAttractorAdaptiveDepthProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_selects_first_depth_whose_residual_converges(self) -> None:
        rows = [
            {"case_id": "a", "think_steps": 1, "loss": 4.0, "target_tokens": 2, "fixed_point_residual": 0.9},
            {"case_id": "a", "think_steps": 2, "loss": 2.0, "target_tokens": 2, "fixed_point_residual": 0.2},
            {"case_id": "a", "think_steps": 4, "loss": 1.0, "target_tokens": 2, "fixed_point_residual": 0.05},
            {"case_id": "b", "think_steps": 1, "loss": 3.0, "target_tokens": 4, "fixed_point_residual": 0.7},
            {"case_id": "b", "think_steps": 2, "loss": 2.5, "target_tokens": 4, "fixed_point_residual": 0.4},
            {"case_id": "b", "think_steps": 4, "loss": 2.0, "target_tokens": 4, "fixed_point_residual": 0.09},
        ]

        report = self.module.build_adaptive_depth_report(
            rows,
            residual_thresholds=[0.1],
            min_depth=1,
            max_depth=4,
        )

        summary = report["threshold_summaries"][0]
        self.assertEqual(summary["residual_threshold"], 0.1)
        self.assertEqual(summary["selected_count"], 2)
        self.assertEqual(summary["mean_selected_depth"], 4.0)
        self.assertAlmostEqual(summary["adaptive_loss"], (1.0 * 2 + 2.0 * 4) / 6)
        self.assertEqual(summary["selected_depth_counts"], {"4": 2})

    def test_falls_back_to_max_depth_when_no_depth_converges(self) -> None:
        rows = [
            {"case_id": "a", "think_steps": 1, "loss": 3.0, "target_tokens": 5, "fixed_point_residual": 0.8},
            {"case_id": "a", "think_steps": 2, "loss": 2.0, "target_tokens": 5, "fixed_point_residual": 0.5},
        ]

        report = self.module.build_adaptive_depth_report(
            rows,
            residual_thresholds=[0.1],
            min_depth=1,
            max_depth=2,
        )

        summary = report["threshold_summaries"][0]
        self.assertAlmostEqual(summary["adaptive_loss"], 2.0)
        self.assertEqual(summary["mean_selected_depth"], 2.0)
        self.assertEqual(summary["fallback_count"], 1)

    def test_reports_oracle_best_depth_headroom(self) -> None:
        rows = [
            {"case_id": "a", "think_steps": 1, "loss": 4.0, "target_tokens": 2, "fixed_point_residual": 0.9},
            {"case_id": "a", "think_steps": 2, "loss": 3.0, "target_tokens": 2, "fixed_point_residual": 0.2},
            {"case_id": "a", "think_steps": 4, "loss": 1.0, "target_tokens": 2, "fixed_point_residual": 0.05},
            {"case_id": "b", "think_steps": 1, "loss": 5.0, "target_tokens": 4, "fixed_point_residual": 0.7},
            {"case_id": "b", "think_steps": 2, "loss": 2.0, "target_tokens": 4, "fixed_point_residual": 0.4},
            {"case_id": "b", "think_steps": 4, "loss": 3.0, "target_tokens": 4, "fixed_point_residual": 0.09},
        ]

        report = self.module.build_adaptive_depth_report(
            rows,
            residual_thresholds=[0.1],
            min_depth=1,
            max_depth=4,
        )

        oracle = report["oracle_best_depth"]
        self.assertAlmostEqual(oracle["oracle_loss"], (1.0 * 2 + 2.0 * 4) / 6)
        self.assertAlmostEqual(oracle["mean_oracle_depth"], (4.0 + 2.0) / 2)
        self.assertEqual(oracle["oracle_depth_counts"], {"2": 1, "4": 1})


if __name__ == "__main__":
    unittest.main()
