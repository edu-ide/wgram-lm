from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/227_build_pure_recursive_ood_paraphrase_stress_cases.py")
    spec = importlib.util.spec_from_file_location(
        "pure_recursive_ood_paraphrase_stress_cases",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveOodParaphraseStressCasesTests(unittest.TestCase):
    def test_build_stress_cases_emits_all_variants_per_base_case(self):
        module = _load_module()

        cases = module.build_ood_paraphrase_stress_cases(
            cases_per_family=1,
            start_index=10000,
            variants_per_case=4,
        )

        self.assertEqual(len(cases), 16)
        self.assertEqual(
            {case["task_family"] for case in cases},
            {"arithmetic_chain", "symbolic_binding", "boolean_logic", "list_transform"},
        )
        self.assertEqual(
            {case["surface_variant_index"] for case in cases if case["task_family"] == "arithmetic_chain"},
            {0, 1, 2, 3},
        )

    def test_rewrite_preserves_runtime_parse_markers_and_targets(self):
        module = _load_module()
        base_cases = module._load_case_builder_module().build_cases(
            cases_per_family=1,
            start_index=10000,
        )

        for case in base_cases:
            rewritten = module.rewrite_case_surface_stress(case, variant_index=6)

            self.assertEqual(rewritten["answer_aliases"], case["answer_aliases"])
            self.assertEqual(rewritten["answer"], case["answer_aliases"][0])
            self.assertEqual(rewritten["chosen"], case["answer_aliases"][0])
            self.assertEqual(rewritten["solver_trace"], case["solver_trace"])
            self.assertNotEqual(rewritten["question"], case["question"])
            self.assertEqual(rewritten["surface_distribution"], "ood_surface_paraphrase_stress_v1")
            self.assertIn("Answer with only the final answer", rewritten["prompt"])

            family = rewritten["task_family"]
            question = rewritten["question"]
            if family == "arithmetic_chain":
                self.assertIn("((", question)
                self.assertIn(")", question)
            elif family == "list_transform":
                self.assertIn("[", question)
                self.assertIn("]", question)
            elif family == "symbolic_binding":
                self.assertIn("maps to", question)
            elif family == "boolean_logic":
                self.assertIn("P=", question)
                self.assertIn("Q=", question)
                self.assertIn("R=", question)


if __name__ == "__main__":
    unittest.main()
