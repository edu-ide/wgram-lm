import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_builder_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "203_build_metacognitive_calibration_cases.py"
    spec = importlib.util.spec_from_file_location("metacognitive_calibration_case_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MetacognitiveCalibrationCasesTests(unittest.TestCase):
    def test_builder_writes_answerable_unknown_contradiction_and_ood_cases(self):
        module = load_builder_script()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cases.jsonl"
            cases = module.write_cases(out, cases_per_family=2)
            rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]

        self.assertEqual(len(cases), 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["category"] for row in rows},
            {
                "answerable_arithmetic",
                "answerable_boolean",
                "unknown_missing",
                "contradiction",
                "ood_random_token",
            },
        )
        self.assertEqual({row["raw_intelligence_axis"] for row in rows}, {"metacognitive_calibration"})
        self.assertEqual({row["retrieval_allowed"] for row in rows}, {False})
        self.assertEqual({row["memoryos_allowed"] for row in rows}, {False})
        self.assertEqual({tuple(row.get("evidence", [])) for row in rows}, {()})

        unknown_rows = [row for row in rows if row["expected_unknown"]]
        answerable_rows = [row for row in rows if not row["expected_unknown"]]
        self.assertTrue(unknown_rows)
        self.assertTrue(answerable_rows)
        for row in unknown_rows:
            self.assertEqual(row["answer_aliases"], ["UNKNOWN", "unknown"])
            self.assertIn("UNKNOWN", row["choices"])
            self.assertIn(row["category"], {"unknown_missing", "contradiction", "ood_random_token"})
        for row in answerable_rows:
            self.assertNotIn("UNKNOWN", row["answer_aliases"])
            self.assertIn(row["answer_aliases"][0], row["choices"])

    def test_cli_defaults_to_40_cases(self):
        module = load_builder_script()

        args = module.build_arg_parser().parse_args([])

        self.assertEqual(args.cases_per_family, 8)


if __name__ == "__main__":
    unittest.main()
