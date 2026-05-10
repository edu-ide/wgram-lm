from pathlib import Path
import importlib.util
import tempfile
import unittest


def _load_module():
    path = Path("scripts/327_eval_pointer_copy_oracle_renderer.py")
    spec = importlib.util.spec_from_file_location("pointer_copy_oracle_renderer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PointerCopyOracleRendererTests(unittest.TestCase):
    def test_selected_copy_positions_keep_even_values_in_order(self):
        module = _load_module()

        self.assertEqual(
            module.selected_copy_positions({"input_list": [44, 39, 55, 40, 32]}),
            [0, 3, 4],
        )

    def test_oracle_pointer_copy_answer_matches_source_copy_target(self):
        module = _load_module()
        row = {"input_list": [44, 39, 55, 40, 32], "answer": "44,40,32"}

        rendered = module.oracle_pointer_copy_answer(row)

        self.assertEqual(rendered, "44,40,32")

    def test_oracle_gate_requires_renderer_off_drop(self):
        module = _load_module()
        rows = [
            {"id": "a", "input_list": [1, 2, 3, 4], "answer": "2,4"},
            {"id": "b", "input_list": [5, 7, 9, 11], "answer": "EMPTY"},
        ]

        report = module.evaluate_rows(rows)

        self.assertEqual(report["decision"], "accepted_l1_pointer_copy_oracle")
        self.assertEqual(report["full_exact_rows"], 2)
        self.assertEqual(report["renderer_off_exact_rows"], 1)
        self.assertEqual(report["non_empty_full_exact_rows"], 1)
        self.assertEqual(report["non_empty_renderer_off_exact_rows"], 0)

    def test_script_writes_report(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            report_path = Path(tmp) / "report.json"
            cases.write_text(
                '{"id":"a","input_list":[1,2,3,4],"answer":"2,4"}\n',
                encoding="utf-8",
            )

            report = module.run_gate(
                cases_path=cases,
                report_path=report_path,
                max_cases=0,
            )

            self.assertTrue(report_path.exists())
            self.assertEqual(report["decision"], "accepted_l1_pointer_copy_oracle")


if __name__ == "__main__":
    unittest.main()
