import unittest
from collections import Counter


class SyntheticMemoryCasesTests(unittest.TestCase):
    def test_build_synthetic_memory_reasoning_cases_is_balanced_and_deterministic(self):
        from qtrm_mm.eval.memory_retrieval import case_task_family, expected_unknown_case
        from qtrm_mm.training.synthetic_memory_cases import build_synthetic_memory_reasoning_cases

        first = build_synthetic_memory_reasoning_cases(num_sets=2, seed=7)
        second = build_synthetic_memory_reasoning_cases(num_sets=2, seed=7)
        families = Counter(case_task_family(case) for case in first)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(families["conflict"], 8)
        self.assertEqual(families["multi_hop"], 8)
        self.assertEqual(families["abstention"], 8)
        self.assertEqual(sum(1 for case in first if expected_unknown_case(case)), 8)

    def test_build_synthetic_memory_reasoning_cases_can_avoid_ids(self):
        from qtrm_mm.training.synthetic_memory_cases import build_synthetic_memory_reasoning_cases

        blocked = {"synthetic-temporal-code-0000", "synthetic-negative-vault-0000"}
        cases = build_synthetic_memory_reasoning_cases(num_sets=1, seed=3, avoid_ids=blocked)

        self.assertFalse(blocked & {case["id"] for case in cases})


if __name__ == "__main__":
    unittest.main()
