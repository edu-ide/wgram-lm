from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_script():
    path = Path("scripts/373_eval_qwen_backbone_wgram_stability.py")
    spec = importlib.util.spec_from_file_location("qwen_backbone_wgram_stability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenBackboneQTRMStabilityEvalTests(unittest.TestCase):
    def test_parse_seed_list_accepts_spaces_and_commas(self):
        module = _load_script()

        self.assertEqual(module.parse_seed_list("1 2, 3"), [1, 2, 3])

    def test_aggregate_seed_reports_requires_all_seeds_accepted(self):
        module = _load_script()
        rows = [
            {
                "accepted": True,
                "after_eval": {"gain": 0.06},
                "accepted_family_summary": {
                    "min_gain": 0.02,
                    "min_core_accuracy": 0.12,
                },
                "after_language": {"top1_agreement": 1.0},
            },
            {
                "accepted": False,
                "after_eval": {"gain": 0.04},
                "accepted_family_summary": {
                    "min_gain": -0.01,
                    "min_core_accuracy": 0.09,
                },
                "after_language": {"top1_agreement": 1.0},
            },
        ]

        summary = module.aggregate_seed_reports(rows)

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["num_seeds"], 2)
        self.assertEqual(summary["num_accepted"], 1)
        self.assertAlmostEqual(summary["min_gain"], 0.04)
        self.assertAlmostEqual(summary["min_family_gain"], -0.01)


if __name__ == "__main__":
    unittest.main()
