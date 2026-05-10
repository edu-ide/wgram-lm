import unittest
from pathlib import Path
import importlib.util


def _load_module():
    path = Path("scripts/241_summarize_mixed_tail_errors.py")
    spec = importlib.util.spec_from_file_location("mixed_tail_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MixedTailErrorSummaryTests(unittest.TestCase):
    def test_parse_mixed_question_extracts_sum_and_final_answer(self):
        tail_summary = _load_module()
        parsed = tail_summary.parse_mixed_question(
            "Keep evens in [50001, 50002, 50003, 50004, 50005, 50006, 50007], "
            "double those values, sum them, then subtract 9."
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["doubled_text"], "100004,100008,100012")
        self.assertEqual(parsed["pre_subtract_sum"], "300024")
        self.assertEqual(parsed["final_answer"], "300015")

    def test_classifies_pre_subtract_sum_vs_doubled_list(self):
        tail_summary = _load_module()
        base = {
            "question": (
                "Apply an even-filter, double transform, sum, and subtract-5 "
                "step to [50004, 50005, 50006, 50007, 50008, 50009, 50010]."
            ),
            "answer_aliases": ["400051"],
            "hit": False,
        }

        self.assertEqual(
            tail_summary.classify_record({**base, "completion": "400056"}),
            "pre_subtract_sum",
        )
        self.assertEqual(
            tail_summary.classify_record(
                {**base, "completion": "100008,100012,100016,100020"}
            ),
            "doubled_list",
        )
        self.assertEqual(
            tail_summary.classify_record({**base, "completion": "400051", "hit": True}),
            "correct_final",
        )

    def test_summarize_groups_by_mode(self):
        tail_summary = _load_module()
        records = [
            {
                "mode": "full",
                "task_family": "mixed_list_arithmetic",
                "question": "From [2, 3, 4], keep evens, double, sum, subtract 1.",
                "answer_aliases": ["11"],
                "completion": "12",
                "hit": False,
            },
            {
                "mode": "full",
                "task_family": "mixed_list_arithmetic",
                "question": "From [2, 3, 4], keep evens, double, sum, subtract 1.",
                "answer_aliases": ["11"],
                "completion": "11",
                "hit": True,
            },
        ]

        summary = tail_summary.summarize(records)

        self.assertEqual(summary["by_mode"]["full"]["pre_subtract_sum"], 1)
        self.assertEqual(summary["by_mode"]["full"]["correct_final"], 1)


if __name__ == "__main__":
    unittest.main()
