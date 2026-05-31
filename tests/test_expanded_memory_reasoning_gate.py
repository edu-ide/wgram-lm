import unittest
from collections import Counter
from pathlib import Path


class ExpandedMemoryReasoningGateTests(unittest.TestCase):
    def test_builder_script_defaults_to_72_case_expanded_heldout(self):
        script = Path("scripts/110_build_expanded_memory_reasoning_heldout.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("memory_reasoning_heldout_expanded_72.jsonl", script)
        self.assertIn("default=4", script)
        self.assertIn("default=100", script)
        self.assertIn("memory_reasoning_synth_train_cases.jsonl", script)

    def test_run_script_pairs_donor_and_residual_modes(self):
        script = Path("scripts/111_run_residual_adapter_expanded_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("110_build_expanded_memory_reasoning_heldout.py", script)
        self.assertIn("memory_reasoning_heldout_expanded_qwen3_rerank_32tok", script)
        self.assertIn("--mode donor_only_with_evidence", script)
        self.assertIn("--mode qtrm_residual_with_evidence", script)
        self.assertIn("--evidence-mode memoryos", script)
        self.assertIn("--rerank-backend \"$RERANK_BACKEND\"", script)

    def test_expanded_heldout_file_is_balanced_and_disjoint(self):
        from wgram_lm.eval.memory_retrieval import case_task_family, load_cases

        expanded = load_cases("data/eval/memory_reasoning_heldout_expanded_72.jsonl")
        hard = load_cases("data/eval/memory_reasoning_probe.jsonl")
        heldout = load_cases("data/eval/memory_reasoning_heldout_probe.jsonl")
        train = load_cases("data/filtered/memory_reasoning_synth_train_cases.jsonl")
        families = Counter(case_task_family(case) for case in expanded)
        expanded_ids = {case["id"] for case in expanded}

        self.assertEqual(len(expanded), 72)
        self.assertEqual(families["abstention"], 24)
        self.assertEqual(families["conflict"], 24)
        self.assertEqual(families["multi_hop"], 24)
        self.assertFalse(expanded_ids & {case["id"] for case in hard})
        self.assertFalse(expanded_ids & {case["id"] for case in heldout})
        self.assertFalse(expanded_ids & {case["id"] for case in train})


if __name__ == "__main__":
    unittest.main()
