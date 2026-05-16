from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_script():
    path = Path("scripts/378_eval_qwen36_scoped_raw_reasoning_baseline.py")
    spec = importlib.util.spec_from_file_location("qwen36_scoped_raw_reasoning_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Qwen36ScopedRawReasoningBaselineEvalTests(unittest.TestCase):
    def test_normalize_two_digit_answer(self):
        module = _load_script()

        self.assertEqual(module.normalize_two_digit_answer("7"), "07")
        self.assertEqual(module.normalize_two_digit_answer("Answer: 12."), "12")
        self.assertEqual(module.normalize_two_digit_answer("no answer"), "")

    def test_score_rows_reports_family_breakdown(self):
        module = _load_script()
        rows = [
            {"family": "modchain", "exact": True},
            {"family": "modchain", "exact": False},
            {"family": "checksum", "exact": True},
        ]

        metrics = module.score_rows(rows)

        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["cases"], 3)
        self.assertAlmostEqual(metrics["generation_exact"], 2 / 3)
        self.assertEqual(metrics["by_family"]["checksum"]["generation_exact"], 1.0)

    def test_load_suite_requires_core_fields(self):
        module = _load_script()
        with TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite.jsonl"
            suite.write_text(
                json.dumps(
                    {
                        "suite_id": "suite",
                        "prompt_protocol": "operation_definitions_v1",
                        "case_id": "case-1",
                        "qwen_prompt": "Prompt",
                        "answer_text": "01",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = module.load_suite(suite)

        self.assertEqual(rows[0]["case_id"], "case-1")


if __name__ == "__main__":
    unittest.main()
