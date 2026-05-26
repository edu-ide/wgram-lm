from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "546_eval_prefixlm_raw_intelligence_suite.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prefixlm_raw_intelligence_suite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrefixLMRawIntelligenceSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_load_raw_cases_defaults_expected_to_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "case_id": "math",
                        "primitive": "reasoning_arithmetic",
                        "family": "math",
                        "language": "en",
                        "instruction": "2+2?",
                        "response": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cases = self.module.load_raw_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expected_contains, ("4",))
        self.assertEqual(cases[0].match_mode, "all_contains")

    def test_score_generation_all_contains_requires_every_expected_part(self) -> None:
        class Helper:
            @staticmethod
            def strip_response_text(text):
                return str(text)

            @staticmethod
            def normalize_for_match(text):
                return str(text).casefold()

        scored = self.module.score_generation(
            helper=Helper,
            response='{"tool":"weather.get_current","location":"Seoul"}',
            expected_contains=("weather.get_current", "location", "Seoul"),
            match_mode="all_contains",
        )

        self.assertTrue(scored["generation_hit"])
        self.assertEqual(len(scored["generation_matched"]), 3)

    def test_score_generation_exact_rejects_extra_text(self) -> None:
        class Helper:
            @staticmethod
            def strip_response_text(text):
                return str(text).strip()

            @staticmethod
            def normalize_for_match(text):
                return str(text).casefold().strip()

        scored = self.module.score_generation(
            helper=Helper,
            response="READY now",
            expected_contains=("READY",),
            match_mode="exact",
        )

        self.assertFalse(scored["generation_hit"])

    def test_summarize_reports_primitive_loss_and_generation_accuracy(self) -> None:
        summary = self.module.summarize(
            [
                {
                    "primitive": "language",
                    "loss": 2.0,
                    "target_tokens": 2,
                    "correct_tokens": 1,
                    "generation_hit": True,
                },
                {
                    "primitive": "language",
                    "loss": 1.0,
                    "target_tokens": 6,
                    "correct_tokens": 3,
                    "generation_hit": False,
                },
            ],
            "primitive",
        )

        self.assertAlmostEqual(summary["language"]["loss"], 1.25)
        self.assertEqual(summary["language"]["target_tokens"], 8)
        self.assertAlmostEqual(summary["language"]["token_accuracy"], 0.5)
        self.assertAlmostEqual(summary["language"]["generation_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
