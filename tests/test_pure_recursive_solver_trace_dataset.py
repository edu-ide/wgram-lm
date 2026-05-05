import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_trace_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "215_build_pure_recursive_solver_trace_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("pure_recursive_solver_trace_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveSolverTraceDatasetTests(unittest.TestCase):
    def test_rows_for_case_flatten_solver_trace_with_previous_state(self):
        module = load_trace_script()
        case = {
            "id": "list-transform-200",
            "raw_intelligence_axis": "pure_recursive_reasoning",
            "category": "list_transform",
            "task_family": "list_transform",
            "prompt": "Question: transform list\nAnswer:",
            "answer_aliases": ["408,404"],
            "solver_trace": [
                {"depth": 1, "operation": "filter_even", "state_text": "204,202"},
                {"depth": 2, "operation": "double_filtered", "state_text": "408,404"},
                {"depth": 4, "operation": "hold_final", "state_text": "408,404"},
                {"depth": 8, "operation": "hold_final", "state_text": "408,404"},
            ],
            "evidence": [],
        }

        rows = module.rows_for_case(case)

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["type"], "pure_recursive_solver_trace")
        self.assertEqual(rows[0]["previous_state_text"], "")
        self.assertEqual(rows[0]["operation"], "filter_even")
        self.assertEqual(rows[0]["target_state_text"], "204,202")
        self.assertEqual(rows[1]["previous_state_text"], "204,202")
        self.assertEqual(rows[1]["target_state_text"], "408,404")
        self.assertFalse(rows[0]["retrieval_allowed"])
        self.assertFalse(rows[0]["memoryos_allowed"])

    def test_write_trace_dataset_rejects_evidence_shortcuts(self):
        module = load_trace_script()
        case = {
            "id": "bad",
            "prompt": "Question?\nAnswer:",
            "answer_aliases": ["A"],
            "solver_trace": [{"depth": 1, "operation": "x", "state_text": "A"}],
            "evidence": ["shortcut"],
        }

        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            out_path = Path(tmp) / "trace.jsonl"
            cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not include evidence"):
                module.write_trace_dataset(cases_path, out_path)

    def test_write_trace_dataset_writes_jsonl(self):
        module = load_trace_script()
        case = {
            "id": "arith-chain-100",
            "raw_intelligence_axis": "pure_recursive_reasoning",
            "category": "arithmetic_chain",
            "task_family": "arithmetic_chain",
            "prompt": "Question: compute\nAnswer:",
            "answer_aliases": ["417"],
            "solver_trace": [
                {"depth": 1, "operation": "add_operands", "state_text": "210"},
                {"depth": 2, "operation": "multiply_sum", "state_text": "420"},
            ],
            "evidence": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            out_path = Path(tmp) / "trace.jsonl"
            cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

            rows = module.write_trace_dataset(cases_path, out_path)
            written = [json.loads(line) for line in out_path.read_text().splitlines()]

        self.assertEqual(rows, written)
        self.assertEqual(len(written), 2)
        self.assertEqual(written[-1]["final_answer"], "417")


if __name__ == "__main__":
    unittest.main()
