from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def _load_script():
    path = Path("scripts/383_materialize_m7_public_reasoning_suite.py")
    spec = importlib.util.spec_from_file_location("m7_public_reasoning_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_row():
    return {
        "question_id": 7,
        "question": "Which value is even?",
        "options": ["3", "5", "8", "9"],
        "answer": "C",
        "answer_index": 2,
        "category": "math",
        "src": "unit",
    }


class M7PublicReasoningSuiteTests(unittest.TestCase):
    def test_format_mmlu_pro_prompt_uses_option_letters_and_answer_only_instruction(self):
        module = _load_script()

        prompt = module.format_mmlu_pro_prompt(_sample_row())

        self.assertTrue(prompt.startswith("User:"))
        self.assertIn("Return only one option letter", prompt)
        self.assertIn("A. 3", prompt)
        self.assertIn("C. 8", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_row_to_case_preserves_public_scoring_fields(self):
        module = _load_script()

        case = module.row_to_case(4, _sample_row(), split="test")

        self.assertEqual(case["benchmark_id"], "mmlu_pro")
        self.assertEqual(case["case_id"], "mmlu-pro-test-000007")
        self.assertEqual(case["answer"], "C")
        self.assertEqual(case["answer_index"], 2)
        self.assertEqual(case["category"], "math")
        self.assertEqual(case["scorer"], "exact option-letter match")


if __name__ == "__main__":
    unittest.main()
