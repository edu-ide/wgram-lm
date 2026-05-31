import unittest
from collections import Counter


class SyntheticMemoryCasesTests(unittest.TestCase):
    def test_build_synthetic_memory_reasoning_cases_is_balanced_and_deterministic(self):
        from wgram_lm.eval.memory_retrieval import case_task_family, expected_unknown_case
        from wgram_lm.training.synthetic_memory_cases import build_synthetic_memory_reasoning_cases

        first = build_synthetic_memory_reasoning_cases(num_sets=2, seed=7)
        second = build_synthetic_memory_reasoning_cases(num_sets=2, seed=7)
        families = Counter(case_task_family(case) for case in first)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        self.assertEqual(families["conflict"], 12)
        self.assertEqual(families["multi_hop"], 12)
        self.assertEqual(families["abstention"], 12)
        self.assertEqual(sum(1 for case in first if expected_unknown_case(case)), 12)

    def test_build_synthetic_memory_reasoning_cases_covers_heldout_failure_patterns(self):
        from wgram_lm.training.synthetic_memory_cases import build_synthetic_memory_reasoning_cases

        cases = build_synthetic_memory_reasoning_cases(num_sets=1, seed=7)
        by_category = {case["category"]: case for case in cases}

        self.assertIn("temporal_location_ko_synth", by_category)
        self.assertIn("authority_location_ko_synth", by_category)
        self.assertIn("multi_hop_maintainer_3hop_synth", by_category)
        self.assertIn("negative_authority_location_ko_synth", by_category)
        self.assertIn("격납고", by_category["temporal_location_ko_synth"]["question"])
        self.assertIn("통신실", by_category["authority_location_ko_synth"]["question"])
        self.assertEqual(len(by_category["multi_hop_maintainer_3hop_synth"]["evidence"]), 3)

    def test_build_synthetic_memory_reasoning_cases_can_avoid_ids(self):
        from wgram_lm.training.synthetic_memory_cases import build_synthetic_memory_reasoning_cases

        blocked = {"synthetic-temporal-code-0000", "synthetic-negative-vault-0000"}
        cases = build_synthetic_memory_reasoning_cases(num_sets=1, seed=3, avoid_ids=blocked)

        self.assertFalse(blocked & {case["id"] for case in cases})

    def test_build_synthetic_memory_reasoning_cases_supports_start_index_for_expanded_heldout(self):
        from wgram_lm.training.synthetic_memory_cases import build_synthetic_memory_reasoning_cases

        baseline = build_synthetic_memory_reasoning_cases(num_sets=1, seed=7)
        expanded = build_synthetic_memory_reasoning_cases(num_sets=1, seed=7, start_index=100)

        self.assertEqual(len(expanded), 18)
        self.assertFalse({case["id"] for case in baseline} & {case["id"] for case in expanded})
        self.assertTrue(any("-0100" in case["id"] for case in expanded))
        self.assertNotEqual(
            {case["question"] for case in baseline},
            {case["question"] for case in expanded},
        )


if __name__ == "__main__":
    unittest.main()
