from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "576_eval_overthinking_noise_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("overthinking_noise_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OverthinkingNoiseProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_report_flags_shallow_correct_answer_lost_at_deeper_depth(self) -> None:
        rows = [
            {
                "id": "case-0",
                "task": "toy",
                "think_steps": 2,
                "normalized_margin": 0.2,
                "correct": True,
                "skipped_reason": None,
            },
            {
                "id": "case-0",
                "task": "toy",
                "think_steps": 8,
                "normalized_margin": -0.1,
                "correct": False,
                "skipped_reason": None,
            },
        ]

        report = self.module.build_overthinking_noise_report(
            rows=rows,
            depths=[2, 8],
            checkpoint="toy.pt",
            probe_jsonl="toy.jsonl",
        )

        self.assertFalse(report["stability_accepted"])
        self.assertEqual(report["flip_to_wrong_count"], 1)
        self.assertIn("shallow_correct_answers_lost_at_deeper_depth", report["failed_checks"])

    def test_report_separates_stable_wrong_from_overthinking_noise(self) -> None:
        rows = [
            {
                "id": "case-0",
                "task": "toy",
                "think_steps": 2,
                "normalized_margin": -0.2,
                "correct": False,
                "skipped_reason": None,
            },
            {
                "id": "case-0",
                "task": "toy",
                "think_steps": 8,
                "normalized_margin": -0.19,
                "correct": False,
                "skipped_reason": None,
            },
        ]

        report = self.module.build_overthinking_noise_report(
            rows=rows,
            depths=[2, 8],
            checkpoint="toy.pt",
            probe_jsonl="toy.jsonl",
        )

        self.assertTrue(report["stability_accepted"])
        self.assertFalse(report["quality_accepted"])
        self.assertFalse(report["accepted"])
        self.assertEqual(report["wrong_at_all_depths_count"], 1)
        self.assertEqual(report["flip_to_wrong_count"], 0)

    def test_report_accepts_only_when_stable_and_deep_quality_passes(self) -> None:
        rows = [
            {
                "id": "case-0",
                "task": "toy",
                "think_steps": 2,
                "normalized_margin": 0.12,
                "correct": True,
                "skipped_reason": None,
            },
            {
                "id": "case-0",
                "task": "toy",
                "think_steps": 8,
                "normalized_margin": 0.13,
                "correct": True,
                "skipped_reason": None,
            },
        ]

        report = self.module.build_overthinking_noise_report(
            rows=rows,
            depths=[2, 8],
            checkpoint="toy.pt",
            probe_jsonl="toy.jsonl",
        )

        self.assertTrue(report["stability_accepted"])
        self.assertTrue(report["quality_accepted"])
        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
