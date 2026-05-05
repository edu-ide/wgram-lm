from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/225_build_pure_recursive_ood_surface_cases.py")
    spec = importlib.util.spec_from_file_location("pure_recursive_ood_surface_cases", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveOodSurfaceCasesTests(unittest.TestCase):
    def test_rewrite_case_preserves_answer_and_solver_trace(self):
        module = _load_module()

        case = {
            "id": "arith-chain-8000",
            "task_family": "arithmetic_chain",
            "question": "Compute ((8007 + 3) * 2) - 3.",
            "prompt": "Question: Compute ((8007 + 3) * 2) - 3.\nAnswer:",
            "answer_aliases": ["16017"],
            "solver_trace": [{"operation": "add_operands", "state_text": "8010"}],
        }

        rewritten = module.rewrite_case_surface(case, variant_index=0)

        self.assertEqual(rewritten["answer_aliases"], ["16017"])
        self.assertEqual(rewritten["answer"], "16017")
        self.assertEqual(rewritten["chosen"], "16017")
        self.assertEqual(rewritten["solver_trace"], case["solver_trace"])
        self.assertNotEqual(rewritten["question"], case["question"])
        self.assertIn("((8007 + 3) * 2) - 3", rewritten["question"])
        self.assertIn("Answer with only the final answer", rewritten["prompt"])

    def test_build_ood_cases_rewrites_all_supported_families(self):
        module = _load_module()

        cases = module.build_ood_surface_cases(cases_per_family=1, start_index=8000)
        families = {case["task_family"] for case in cases}

        self.assertEqual(
            families,
            {"arithmetic_chain", "symbolic_binding", "boolean_logic", "list_transform"},
        )
        for case in cases:
            self.assertEqual(case["surface_distribution"], "ood_surface_paraphrase_v1")
            self.assertIn("prompt", case)


if __name__ == "__main__":
    unittest.main()
