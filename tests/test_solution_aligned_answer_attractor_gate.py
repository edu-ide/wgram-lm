from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "569_eval_solution_aligned_answer_attractor_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("solution_aligned_answer_attractor_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SolutionAlignedAnswerAttractorGateTests(unittest.TestCase):
    def test_rejects_residual_only_progress_when_answer_margin_regresses(self) -> None:
        module = load_module()
        rows = [
            {
                "depth": 2,
                "gd_accuracy": 0.33,
                "gd_mean_margin": 0.02,
                "gd_failed_tasks": ["flipped_answer_icl", "successive_answer_icl", "truthy_answer_icl"],
                "gd_passed_tasks": ["repetitive_answer_icl"],
                "heldout_loss": 2.5,
                "mean_fixed_point_residual": 0.2,
                "elapsed_sec": 10.0,
            },
            {
                "depth": 8,
                "gd_accuracy": 0.5,
                "gd_mean_margin": -0.04,
                "gd_failed_tasks": ["flipped_answer_icl", "successive_answer_icl", "truthy_answer_icl"],
                "gd_passed_tasks": ["repetitive_answer_icl", "persona_multihop_icl"],
                "heldout_loss": 2.7,
                "mean_fixed_point_residual": 0.05,
                "elapsed_sec": 12.0,
            },
        ]
        baseline = rows[0]
        candidate = rows[1]
        checks = module.build_checks(
            baseline=baseline,
            candidate=candidate,
            critical_tasks=("flipped_answer_icl", "successive_answer_icl", "truthy_answer_icl"),
            min_margin_gain=0.02,
            min_accuracy_gain=0.0,
            max_heldout_loss_regression=0.01,
            max_elapsed_ratio=1.5,
        )

        self.assertTrue(checks["residual_decreases"]["passed"])
        self.assertFalse(checks["gd_mean_margin_improves"]["passed"])
        self.assertFalse(checks["critical_tasks_pass"]["passed"])
        self.assertFalse(checks["heldout_loss_not_regressed"]["passed"])

    def test_accepts_deeper_depth_only_when_answer_facing_signals_improve(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            out_path = Path(tmp) / "gate.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "checkpoint": "fake.pt",
                        "rows": [
                            {
                                "depth": 2,
                                "gd_accuracy": 0.33,
                                "gd_mean_margin": 0.0,
                                "gd_failed_tasks": [
                                    "flipped_answer_icl",
                                    "successive_answer_icl",
                                    "truthy_answer_icl",
                                ],
                                "gd_passed_tasks": ["repetitive_answer_icl"],
                                "heldout_loss": 2.5,
                                "mean_fixed_point_residual": 0.2,
                                "elapsed_sec": 10.0,
                            },
                            {
                                "depth": 8,
                                "gd_accuracy": 0.83,
                                "gd_mean_margin": 0.15,
                                "gd_failed_tasks": [],
                                "gd_passed_tasks": [
                                    "flipped_answer_icl",
                                    "successive_answer_icl",
                                    "truthy_answer_icl",
                                    "repetitive_answer_icl",
                                ],
                                "heldout_loss": 2.49,
                                "mean_fixed_point_residual": 0.05,
                                "elapsed_sec": 12.0,
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--depth-sweep-summary",
                    str(summary_path),
                    "--out",
                    str(out_path),
                    "--baseline-depth",
                    "2",
                    "--min-candidate-depth",
                    "4",
                ]
            )
            report = module.run_gate(args)
            self.assertTrue(out_path.exists())

        self.assertTrue(report["accepted"])
        self.assertEqual(report["candidate_depth"], 8)


if __name__ == "__main__":
    unittest.main()
