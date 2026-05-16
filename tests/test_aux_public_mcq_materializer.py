from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def load_module():
    path = Path("scripts/392_materialize_aux_public_mcq.py")
    spec = importlib.util.spec_from_file_location("aux_public_mcq_materializer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AuxPublicMCQMaterializerTests(unittest.TestCase):
    def test_row_to_case_matches_public_mcq_schema(self):
        module = load_module()

        case = module.row_to_case(
            dataset="cais/mmlu",
            config="professional_law",
            split="validation",
            row_idx=3,
            row={
                "question": "Which rule applies?",
                "subject": "professional_law",
                "choices": ["Rule A", "Rule B", "Rule C", "Rule D"],
                "answer": 1,
            },
        )

        self.assertEqual(case["benchmark_id"], "aux_public_mcq")
        self.assertEqual(case["answer"], "B")
        self.assertEqual(case["answer_index"], 1)
        self.assertEqual(case["category"], "law")
        self.assertIn("qtrm_prompt", case)
        self.assertIn("Return only one option letter", case["qtrm_prompt"])

    def test_write_outputs_records_leakage_policy(self):
        module = load_module()
        with TemporaryDirectory() as tmp:
            out_jsonl = Path(tmp) / "cases.jsonl"
            out_report = Path(tmp) / "report.json"
            args = module.build_arg_parser().parse_args(
                [
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-report",
                    str(out_report),
                ]
            )
            report = module.write_outputs(
                args,
                [
                    {
                        "category": "health",
                        "subject": "clinical_knowledge",
                        "answer": "A",
                        "qtrm_prompt": "User: Q\nAssistant:",
                    }
                ],
            )

            saved = json.loads(out_report.read_text(encoding="utf-8"))
            rows = out_jsonl.read_text(encoding="utf-8").splitlines()

        self.assertTrue(report["accepted"])
        self.assertEqual(saved["by_category"], {"health": 1})
        self.assertEqual(len(rows), 1)
        self.assertIn("MMLU-Pro test", " ".join(saved["leakage_policy"]))

    def test_default_configs_focus_current_1024_regression_categories(self):
        module = load_module()
        args = module.build_arg_parser().parse_args([])

        self.assertIn("clinical_knowledge", args.configs)
        self.assertIn("college_chemistry", args.configs)
        self.assertIn("econometrics", args.configs)
        self.assertIn("professional_law", args.configs)
        self.assertEqual(args.splits, "dev,validation")


if __name__ == "__main__":
    unittest.main()
