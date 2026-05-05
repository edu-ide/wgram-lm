import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_builder_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "190_build_pure_recursive_reasoning_cases.py"
    spec = importlib.util.spec_from_file_location("pure_reasoning_case_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveReasoningCasesTests(unittest.TestCase):
    def test_builder_writes_no_retrieval_no_evidence_cases(self):
        module = load_builder_script()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cases.jsonl"
            cases = module.write_cases(out, cases_per_family=2)

            rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]

        self.assertEqual(len(cases), 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [row["category"] for row in rows[:4]],
            ["arithmetic_chain", "symbolic_binding", "boolean_logic", "list_transform"],
        )
        self.assertEqual(
            {row["raw_intelligence_axis"] for row in rows},
            {"pure_recursive_reasoning"},
        )
        self.assertEqual(
            {row["retrieval_allowed"] for row in rows},
            {False},
        )
        self.assertEqual(
            {row["memoryos_allowed"] for row in rows},
            {False},
        )
        for row in rows:
            self.assertNotIn("MemoryOS evidence", row["prompt"])
            self.assertNotIn("Evidence:", row["prompt"])
            self.assertTrue(row["prompt"].rstrip().endswith("Answer:"))
            self.assertEqual(row.get("evidence", []), [])
            self.assertIn("reasoning_family", row)
            self.assertIn("expected_paradigm", row)
            self.assertIn("requires_stochasticity", row)
            self.assertIn("parallel_depth_estimate", row)
            self.assertIn("serial_trace_length_estimate", row)
            self.assertTrue(row["answer_aliases"])
            self.assertIn(row["answer_aliases"][0], row["choices"])
            self.assertGreaterEqual(len(row["choices"]), 2)
            self.assertIn("depth_targets", row)
            self.assertEqual(row["depth_targets"]["8"], row["answer_aliases"][0])
            self.assertIn("transition_state_codes", row)
            self.assertEqual(set(row["transition_state_codes"]), {"1", "2", "4", "8"})
            self.assertIn("solver_trace", row)
            self.assertEqual([step["depth"] for step in row["solver_trace"]], [1, 2, 4, 8])

    def test_builder_emits_intermediate_depth_targets(self):
        module = load_builder_script()

        cases = module.build_cases(cases_per_family=1, start_index=0)
        by_category = {case["category"]: case for case in cases}

        arithmetic = by_category["arithmetic_chain"]
        self.assertEqual(arithmetic["depth_targets"]["1"], "10")
        self.assertEqual(arithmetic["depth_targets"]["2"], "20")
        self.assertEqual(arithmetic["depth_targets"]["4"], arithmetic["answer_aliases"][0])

        symbolic = by_category["symbolic_binding"]
        self.assertEqual(symbolic["depth_targets"]["1"], "green")
        self.assertEqual(symbolic["depth_targets"]["2"], symbolic["answer_aliases"][0])

        boolean = by_category["boolean_logic"]
        self.assertIn(boolean["depth_targets"]["1"], {"TRUE", "FALSE"})
        self.assertEqual(boolean["depth_targets"]["4"], boolean["answer_aliases"][0])

        list_case = by_category["list_transform"]
        self.assertIn("depth_targets", list_case)
        self.assertEqual(list_case["depth_targets"]["4"], list_case["answer_aliases"][0])
        self.assertEqual(list_case["solver_trace"][0]["operation"], "filter_even")
        self.assertEqual(list_case["solver_trace"][1]["operation"], "double_filtered")
        self.assertEqual(list_case["solver_trace"][0]["state_text"], list_case["depth_targets"]["1"])
        self.assertEqual(list_case["solver_trace"][1]["state_text"], list_case["depth_targets"]["2"])

    def test_builder_emits_semantic_transition_state_codes(self):
        module = load_builder_script()

        cases_0 = module.build_cases(cases_per_family=1, start_index=0)
        cases_200 = module.build_cases(cases_per_family=1, start_index=200)
        by_category_0 = {case["category"]: case for case in cases_0}
        by_category_200 = {case["category"]: case for case in cases_200}

        list_codes = by_category_0["list_transform"]["transition_state_codes"]
        self.assertNotEqual(list_codes["1"], list_codes["2"])
        self.assertEqual(list_codes["2"], list_codes["4"])
        self.assertEqual(list_codes, by_category_200["list_transform"]["transition_state_codes"])

        arithmetic_codes = by_category_0["arithmetic_chain"]["transition_state_codes"]
        self.assertNotEqual(arithmetic_codes["1"], arithmetic_codes["2"])
        self.assertNotEqual(arithmetic_codes["2"], arithmetic_codes["4"])
        self.assertEqual(arithmetic_codes["4"], arithmetic_codes["8"])

    def test_builder_labels_cot_vs_latent_task_family_metadata(self):
        module = load_builder_script()

        cases = module.build_cases(cases_per_family=1, start_index=0)
        by_category = {case["category"]: case for case in cases}

        self.assertEqual(by_category["arithmetic_chain"]["expected_paradigm"], "hybrid_or_cot")
        self.assertEqual(by_category["arithmetic_chain"]["reasoning_family"], "sequential_arithmetic")
        self.assertEqual(by_category["boolean_logic"]["expected_paradigm"], "latent_parallel")
        self.assertEqual(by_category["boolean_logic"]["reasoning_family"], "parallel_boolean")
        self.assertEqual(by_category["symbolic_binding"]["expected_paradigm"], "latent_recurrent")
        self.assertFalse(any(case["requires_stochasticity"] for case in cases))

    def test_builder_can_filter_to_hard_families(self):
        module = load_builder_script()

        with tempfile.TemporaryDirectory() as tmp:
            cases = module.write_cases(
                Path(tmp) / "filtered.jsonl",
                cases_per_family=2,
                include_families=module.parse_family_filter(
                    ["arithmetic_chain,list_transform"]
                ),
            )

        self.assertEqual(len(cases), 4)
        self.assertEqual(
            {case["task_family"] for case in cases},
            {"arithmetic_chain", "list_transform"},
        )

    def test_cli_accepts_family_filter(self):
        module = load_builder_script()

        args = module.build_arg_parser().parse_args(
            ["--include-family", "arithmetic_chain,list_transform"]
        )

        self.assertEqual(
            module.parse_family_filter(args.include_family),
            {"arithmetic_chain", "list_transform"},
        )

    def test_cli_defaults_to_72_cases(self):
        module = load_builder_script()

        args = module.build_arg_parser().parse_args([])

        self.assertEqual(args.cases_per_family, 18)


if __name__ == "__main__":
    unittest.main()
