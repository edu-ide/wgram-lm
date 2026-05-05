import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_pref_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "194_build_pure_recursive_reasoning_preferences.py"
    spec = importlib.util.spec_from_file_location("pure_reasoning_pref_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveReasoningPreferenceTests(unittest.TestCase):
    def test_builder_converts_choices_to_chosen_rejected_rows(self):
        module = load_pref_script()
        case = {
            "id": "arith-chain-100",
            "raw_intelligence_axis": "pure_recursive_reasoning",
            "category": "arithmetic_chain",
            "task_family": "arithmetic_chain",
            "reasoning_family": "sequential_arithmetic",
            "expected_paradigm": "hybrid_or_cot",
            "requires_stochasticity": False,
            "parallel_depth_estimate": 0,
            "serial_trace_length_estimate": 3,
            "prompt": "Question: Compute 1+1.\nAnswer:",
            "answer_aliases": ["2"],
            "choices": ["2", "3", "4"],
            "depth_targets": {"1": "1", "2": "2", "4": "2", "8": "2"},
            "transition_state_codes": {"1": 10, "2": 11, "4": 12, "8": 12},
            "solver_trace": [
                {"depth": 1, "operation": "add", "state_text": "1"},
                {"depth": 2, "operation": "final", "state_text": "2"},
                {"depth": 4, "operation": "hold", "state_text": "2"},
                {"depth": 8, "operation": "hold", "state_text": "2"},
            ],
            "evidence": [],
        }

        rows = module.preference_rows_for_case(case, max_rejected_per_case=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["chosen"] for row in rows}, {"2"})
        self.assertEqual({row["rejected"] for row in rows}, {"3", "4"})
        self.assertEqual({row["raw_intelligence_axis"] for row in rows}, {"pure_recursive_reasoning"})
        self.assertTrue(all(row["prompt"].endswith("Answer:") for row in rows))
        self.assertTrue(all(row["memoryos_allowed"] is False for row in rows))
        self.assertTrue(all(row["depth_targets"]["1"] == "1" for row in rows))
        self.assertTrue(all(row["transition_state_codes"]["2"] == 11 for row in rows))
        self.assertTrue(all(row["solver_trace"][0]["operation"] == "add" for row in rows))
        self.assertEqual({row["reasoning_family"] for row in rows}, {"sequential_arithmetic"})
        self.assertEqual({row["expected_paradigm"] for row in rows}, {"hybrid_or_cot"})
        self.assertEqual({row["serial_trace_length_estimate"] for row in rows}, {3})

    def test_write_preferences_writes_jsonl(self):
        module = load_pref_script()
        cases = [
            {
                "id": "bool-100",
                "raw_intelligence_axis": "pure_recursive_reasoning",
                "category": "boolean_logic",
                "prompt": "Question: TRUE OR FALSE?\nAnswer:",
                "answer_aliases": ["TRUE"],
                "choices": ["TRUE", "FALSE"],
                "evidence": [],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "cases.jsonl"
            out_path = Path(tmp) / "prefs.jsonl"
            case_path.write_text(
                "\n".join(json.dumps(case) for case in cases) + "\n",
                encoding="utf-8",
            )

            rows = module.write_preferences(
                case_path,
                out_path,
                max_rejected_per_case=1,
            )
            written = [json.loads(line) for line in out_path.read_text().splitlines()]

        self.assertEqual(rows, written)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["chosen"], "TRUE")
        self.assertEqual(written[0]["rejected"], "FALSE")

    def test_write_preferences_can_filter_expected_unknown_rows(self):
        module = load_pref_script()
        cases = [
            {
                "id": "known-100",
                "raw_intelligence_axis": "metacognitive_calibration",
                "category": "answerable_boolean",
                "prompt": "Question: TRUE OR FALSE?\nAnswer:",
                "answer_aliases": ["TRUE"],
                "choices": ["TRUE", "FALSE", "UNKNOWN"],
                "expected_unknown": False,
                "uncertainty_type": "answerable",
                "evidence": [],
            },
            {
                "id": "unknown-100",
                "raw_intelligence_axis": "metacognitive_calibration",
                "category": "unknown_missing",
                "prompt": "Question: Missing private value?\nAnswer:",
                "answer_aliases": ["UNKNOWN"],
                "choices": ["UNKNOWN", "A", "B"],
                "expected_unknown": True,
                "uncertainty_type": "missing_information",
                "evidence": [],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "cases.jsonl"
            out_path = Path(tmp) / "prefs.jsonl"
            case_path.write_text(
                "\n".join(json.dumps(case) for case in cases) + "\n",
                encoding="utf-8",
            )

            rows = module.write_preferences(
                case_path,
                out_path,
                max_rejected_per_case=2,
                only_expected_unknown=True,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_id"] for row in rows}, {"unknown-100"})
        self.assertEqual({row["chosen"] for row in rows}, {"UNKNOWN"})
        self.assertEqual({row["expected_unknown"] for row in rows}, {True})
        self.assertEqual({row["raw_intelligence_axis"] for row in rows}, {"metacognitive_calibration"})
        self.assertEqual({row["uncertainty_type"] for row in rows}, {"missing_information"})


if __name__ == "__main__":
    unittest.main()
