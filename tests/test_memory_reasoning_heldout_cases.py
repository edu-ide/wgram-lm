import unittest
from collections import Counter


class MemoryReasoningHeldoutCasesTests(unittest.TestCase):
    def test_heldout_cases_are_valid_balanced_and_disjoint_from_training_probe(self):
        from qtrm_mm.eval.memory_retrieval import case_task_family, expected_unknown_case, load_cases

        train_cases = load_cases("data/eval/memory_reasoning_probe.jsonl")
        heldout_cases = load_cases("data/eval/memory_reasoning_heldout_probe.jsonl")

        train_ids = {case["id"] for case in train_cases}
        heldout_ids = {case["id"] for case in heldout_cases}
        families = Counter(case_task_family(case) for case in heldout_cases)

        self.assertEqual(len(heldout_cases), 12)
        self.assertFalse(train_ids & heldout_ids)
        self.assertEqual(families["conflict"], 4)
        self.assertEqual(families["multi_hop"], 4)
        self.assertEqual(families["abstention"], 4)
        self.assertEqual(sum(1 for case in heldout_cases if expected_unknown_case(case)), 4)


if __name__ == "__main__":
    unittest.main()
