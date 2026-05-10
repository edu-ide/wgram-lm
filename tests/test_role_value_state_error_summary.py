from __future__ import annotations

from importlib import util
from pathlib import Path
import unittest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "311_summarize_role_value_state_errors.py"
)
_SPEC = util.spec_from_file_location("role_value_state_error_summary", _SCRIPT_PATH)
summary = util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(summary)


class RoleValueStateErrorSummaryTests(unittest.TestCase):
    def test_role_name_maps_standard_role_value_layout(self) -> None:
        self.assertEqual(summary.role_name(0, num_roles=10), "raw_list_0")
        self.assertEqual(summary.role_name(4, num_roles=10), "doubled_list_0")
        self.assertEqual(summary.role_name(8, num_roles=10), "scalar_coeff")
        self.assertEqual(summary.role_name(9, num_roles=10), "scalar_residual")

    def test_summarize_counts_role_step_and_action_errors(self) -> None:
        data = {
            "summary": {"exact_rows": 0},
            "records": [
                {
                    "id": "row-1",
                    "target_codes": [0, 1],
                    "predicted_values": [
                        [1, 2, 3, 4, -100, -100, -100, -100, -100, -100],
                        [5, 6, 7, 8, -100, -100, -100, -100, -100, -100],
                    ],
                    "target_values": [
                        [1, 9, -100, -100, -100, -100, -100, -100, -100, -100],
                        [5, 0, 7, -100, -100, -100, -100, -100, -100, -100],
                    ],
                }
            ],
        }

        report = summary.summarize(data)

        by_role = {row["key"]: row for row in report["by_role"]}
        by_step = {row["key"]: row for row in report["by_step"]}
        by_action = {row["key"]: row for row in report["by_action"]}

        self.assertEqual(by_role["raw_list_1"]["errors"], 2)
        self.assertEqual(by_step["1"]["errors"], 1)
        self.assertEqual(by_step["2"]["errors"], 1)
        self.assertEqual(by_action["0"]["errors"], 1)
        self.assertEqual(by_action["1"]["errors"], 1)
        self.assertEqual(report["examples"][0]["role"], "raw_list_1")

    def test_summarize_can_join_actions_from_source_rows(self) -> None:
        data = {
            "records": [
                {
                    "id": "row-1",
                    "predicted_values": [[1, 2, -100, -100]],
                    "target_values": [[1, 3, -100, -100]],
                }
            ],
        }
        source_rows = {
            "row-1": {
                "transition_state_codes": {"1": 4},
            }
        }

        report = summary.summarize(data, source_rows=source_rows)
        by_action = {row["key"]: row for row in report["by_action"]}

        self.assertEqual(by_action["4"]["errors"], 1)
        self.assertEqual(report["examples"][0]["action"], 4)
