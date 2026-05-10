import importlib.util
from pathlib import Path
import unittest


def _load_module():
    path = Path("scripts/259_summarize_list_transform_failures.py")
    spec = importlib.util.spec_from_file_location("list_failure_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ListTransformFailureSummaryTests(unittest.TestCase):
    def test_classifies_filtered_and_reversed_failures_with_score_gap(self):
        module = _load_module()
        case = {
            "id": "list-transform-000",
            "task_family": "list_transform",
            "answer_aliases": ["8,4"],
            "depth_targets": {"1": "4,2", "2": "8,4"},
            "choices": ["8,4", "4,8", "4,2", "EMPTY"],
        }
        eval_rows = [
            {
                "id": "list-transform-000",
                "mode": "qtrm_core_steps_8_no_evidence",
                "task_family": "list_transform",
                "hit": False,
                "completion": "4,2",
                "answer_aliases": ["8,4"],
                "choice_scores": [
                    {"choice": "4,2", "logprob": -4.0},
                    {"choice": "8,4", "logprob": -5.0},
                ],
            }
        ]

        summary = module.summarize_failures(
            eval_rows,
            {"list-transform-000": case},
            mode="qtrm_core_steps_8_no_evidence",
        )

        self.assertEqual(summary["hits"], 0)
        self.assertEqual(summary["by_error"], {"filtered_state_selected": 1})
        self.assertEqual(summary["records"][0]["correct_rank"], 2)
        self.assertEqual(summary["records"][0]["correct_minus_selected_score"], -1.0)

        reversed_row = dict(eval_rows[0], completion="4,8")
        self.assertEqual(
            module.classify_list_failure(reversed_row, case),
            "reversed_final_selected",
        )


if __name__ == "__main__":
    unittest.main()
