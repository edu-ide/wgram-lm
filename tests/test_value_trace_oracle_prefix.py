import unittest
from importlib import util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "310_analyze_value_trace_oracle_prefix.py"
)
_SPEC = util.spec_from_file_location("value_trace_oracle_prefix", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
oracle_prefix = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle_prefix)


class ValueTraceOraclePrefixTests(unittest.TestCase):
    def test_trace_exact_with_oracle_prefix_ignores_leading_steps(self):
        record = {
            "predicted_values": [[9, 9], [2, 4], [5, 6]],
            "target_values": [[1, 3], [2, 4], [5, 6]],
        }

        self.assertFalse(
            oracle_prefix.trace_exact_with_oracle_prefix(record, prefix_steps=0)
        )
        self.assertTrue(
            oracle_prefix.trace_exact_with_oracle_prefix(record, prefix_steps=1)
        )

    def test_oracle_prefix_report_tracks_unrecovered_rows(self):
        data = {
            "summary": {"exact_rows": 0},
            "records": [
                {
                    "predicted_values": [[1], [2]],
                    "target_values": [[1], [2]],
                },
                {
                    "predicted_values": [[9], [9]],
                    "target_values": [[1], [2]],
                },
            ],
        }

        report = oracle_prefix.oracle_prefix_report(data, max_prefix_steps=1)

        self.assertEqual(report["oracle_prefix"][0]["exact_rows"], 1)
        self.assertEqual(report["oracle_prefix"][1]["exact_rows"], 1)
        self.assertEqual(report["min_prefix_histogram"], {"0": 1, "unrecovered": 1})


if __name__ == "__main__":
    unittest.main()
