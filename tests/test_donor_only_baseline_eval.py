from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/224_eval_donor_only_baseline.py")
    spec = importlib.util.spec_from_file_location("donor_only_baseline_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DonorOnlyBaselineEvalTests(unittest.TestCase):
    def test_choices_for_row_prefers_explicit_choices_and_includes_target(self):
        module = _load_module()

        choices = module.choices_for_row(
            {"chosen": "A", "choices": ["B", "A", "A"]}
        )

        self.assertEqual(choices, ["B", "A"])

    def test_choices_for_row_falls_back_to_chosen_and_rejected(self):
        module = _load_module()

        choices = module.choices_for_row({"chosen": "A", "rejected": "B"})

        self.assertEqual(choices, ["A", "B"])

    def test_score_record_marks_exact_match_after_stripping(self):
        module = _load_module()

        record = module.score_record(
            {"id": "case-1", "chosen": "417", "task_family": "arithmetic_chain"},
            predicted_answer=" 417\n",
            mode="greedy",
            choice_scores=[],
        )

        self.assertTrue(record["hit"])
        self.assertEqual(record["target_answer"], "417")
        self.assertEqual(record["completion"], "417")

    def test_summarize_counts_modes_and_families(self):
        module = _load_module()

        summary = module.summarize_records(
            [
                {"mode": "forced_choice", "task_family": "a", "hit": True},
                {"mode": "forced_choice", "task_family": "a", "hit": False},
                {"mode": "greedy", "task_family": "b", "hit": True},
            ]
        )

        self.assertEqual(summary["by_mode"]["forced_choice"]["exact"], "1/2")
        self.assertEqual(summary["by_mode"]["greedy"]["accuracy"], 1.0)
        self.assertEqual(summary["by_mode_family"]["forced_choice"]["a"]["exact"], "1/2")


if __name__ == "__main__":
    unittest.main()
