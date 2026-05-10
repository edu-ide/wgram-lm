import unittest
from importlib import util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "312_analyze_value_trace_oracle_roles.py"
)
_SPEC = util.spec_from_file_location("value_trace_oracle_roles", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
oracle_roles = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle_roles)


class ValueTraceOracleRolesTests(unittest.TestCase):
    def test_parse_role_spec_supports_groups_ranges_and_indices(self):
        self.assertEqual(
            oracle_roles.parse_role_spec("scalar", num_roles=10),
            {8, 9},
        )
        self.assertEqual(
            oracle_roles.parse_role_spec("0-2,scalar_residual", num_roles=10),
            {0, 1, 2, 9},
        )

    def test_trace_exact_with_oracle_roles_ignores_selected_roles(self):
        record = {
            "predicted_values": [
                [1, 9, -100, -100, -100, -100, -100, -100, 3, 99],
                [2, 8, -100, -100, -100, -100, -100, -100, 4, 88],
            ],
            "target_values": [
                [1, 9, -100, -100, -100, -100, -100, -100, 3, 7],
                [2, 8, -100, -100, -100, -100, -100, -100, 4, 6],
            ],
        }

        self.assertFalse(
            oracle_roles.trace_exact_with_oracle_roles(
                record,
                role_spec="scalar_coeff",
            )
        )
        self.assertTrue(
            oracle_roles.trace_exact_with_oracle_roles(
                record,
                role_spec="scalar_residual",
            )
        )

    def test_oracle_role_report_counts_exact_rows_per_spec(self):
        data = {
            "summary": {"exact_rows": 0},
            "records": [
                {
                    "predicted_values": [[1, 2, 3, 4]],
                    "target_values": [[1, 2, 3, 4]],
                },
                {
                    "predicted_values": [[1, 9, -100, -100]],
                    "target_values": [[1, 2, -100, -100]],
                },
            ],
        }

        report = oracle_roles.oracle_role_report(
            data,
            role_specs=["0-1", "all"],
        )

        self.assertEqual(report["oracle_roles"][0]["exact_rows"], 2)
        self.assertEqual(report["oracle_roles"][1]["exact_rows"], 2)


if __name__ == "__main__":
    unittest.main()
