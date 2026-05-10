from pathlib import Path
import importlib.util
import json
import unittest


def _load_module():
    path = Path("scripts/327_eval_oracle_pointer_copy_lexicalizer.py")
    spec = importlib.util.spec_from_file_location("oracle_pointer_copy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OraclePointerCopyLexicalizerTests(unittest.TestCase):
    def test_even_source_positions_are_derived_from_prompt_source_values(self):
        module = _load_module()

        self.assertEqual(
            module.even_source_positions({"input_list": [44, 39, 55, 40, 32]}),
            [0, 3, 4],
        )

    def test_oracle_pointer_copy_answer_uses_positions_not_value_classes(self):
        module = _load_module()
        row = {"input_list": [44, 39, 55, 40, 32]}

        answer = module.oracle_pointer_copy_answer(
            row,
            source_positions=[0, 3, 4],
        )

        self.assertEqual(answer, "44,40,32")

    def test_oracle_pointer_copy_empty_answer(self):
        module = _load_module()

        answer = module.oracle_pointer_copy_answer(
            {"input_list": [41, 39, 55]},
            source_positions=[],
        )

        self.assertEqual(answer, "EMPTY")

    def test_report_accepts_only_when_pointer_copy_beats_renderer_off(self):
        module = _load_module()
        rows = [
            {"id": "a", "input_list": [44, 39, 40], "answer": "44,40"},
            {"id": "b", "input_list": [41, 39, 55], "answer": "EMPTY"},
        ]

        report = module.evaluate_rows(rows)

        self.assertEqual(report["decision"], "accepted_l1_oracle_pointer_copy")
        self.assertEqual(report["full_exact"], 2)
        self.assertEqual(report["renderer_off_exact"], 1)
        self.assertEqual(report["nonempty_pointer_drop"], 1.0)

    def test_cli_writes_report(self):
        module = _load_module()
        tmp = Path("local_eval/test_oracle_pointer_copy")
        cases = tmp / "cases.jsonl"
        out = tmp / "report.json"
        tmp.mkdir(parents=True, exist_ok=True)
        cases.write_text(
            json.dumps({"id": "a", "input_list": [44, 39, 40], "answer": "44,40"})
            + "\n",
            encoding="utf-8",
        )

        args = module.build_arg_parser().parse_args(
            ["--cases", str(cases), "--out", str(out)]
        )
        report = module.run(args)

        self.assertTrue(out.exists())
        self.assertEqual(report["full_exact"], 1)


if __name__ == "__main__":
    unittest.main()
