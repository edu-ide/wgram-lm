import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_builder_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "207_build_temporal_spatial_context_cases.py"
    )
    spec = importlib.util.spec_from_file_location("temporal_spatial_context_cases", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TemporalSpatialContextCasesTests(unittest.TestCase):
    def test_build_cases_emit_temporal_and_spatial_context_vectors(self):
        module = load_builder_script()

        cases = module.build_cases(cases_per_family=2, start_index=3)

        self.assertEqual(len(cases), 4)
        self.assertEqual(
            {case["category"] for case in cases},
            {"temporal_freshness", "spatial_relation"},
        )
        self.assertEqual(
            {case["raw_intelligence_axis"] for case in cases},
            {"temporal_spatial_context"},
        )
        for case in cases:
            self.assertFalse(case["retrieval_allowed"])
            self.assertFalse(case["memoryos_allowed"])
            self.assertEqual(case["evidence"], [])
            self.assertIn("temporal_spatial_context", case)
            self.assertEqual(len(case["temporal_spatial_context"]), 2)
            self.assertTrue(
                all(len(token) == 8 for token in case["temporal_spatial_context"])
            )
            self.assertEqual(case["answer"], case["answer_aliases"][0])
            self.assertEqual(case["chosen"], case["answer_aliases"][0])
            self.assertIn(case["answer_aliases"][0], case["choices"])

    def test_write_cases_writes_jsonl(self):
        module = load_builder_script()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "temporal_spatial_context.jsonl"
            cases = module.write_cases(out, cases_per_family=1, start_index=0)
            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(cases), 2)
        self.assertEqual(rows, cases)


if __name__ == "__main__":
    unittest.main()
