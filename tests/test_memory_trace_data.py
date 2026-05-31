import unittest


class MemoryTraceDataTests(unittest.TestCase):
    def test_build_memory_trace_rows_turns_missing_answer_cases_into_unknown_traces(self):
        from wgram_lm.training.memory_trace_data import build_memory_trace_rows

        case = {
            "id": "missing-north-vault",
            "category": "negative_missing",
            "instruction": "If the requested answer is not present in the evidence, answer UNKNOWN.",
            "question": "Which passphrase opens the north vault?",
            "answer_aliases": ["UNKNOWN", "unknown"],
            "evidence": [
                {"source": "west.md", "chunk_id": 0, "text": "The west vault passphrase is jade-circuit."},
                {"source": "east.md", "chunk_id": 1, "text": "The east vault passphrase is amber-harbor."},
            ],
            "distractors": [
                {"source": "north-storage.md", "chunk_id": 2, "text": "The north storage marker is Polaris-42."}
            ],
        }

        rows = build_memory_trace_rows([case], variants=["target", "all"], max_evidence_chars=1200)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["answer"] for row in rows}, {"Answer: UNKNOWN"})
        self.assertEqual({row["task_family"] for row in rows}, {"abstention"})
        self.assertTrue(all("If the evidence does not explicitly contain" in row["prompt"] for row in rows))
        self.assertTrue(any("Polaris-42" in row["prompt"] for row in rows))

    def test_build_memory_trace_rows_keeps_short_gold_answer_for_conflict_cases(self):
        from wgram_lm.training.memory_trace_data import build_memory_trace_rows

        case = {
            "id": "signed-vault",
            "category": "authority_conflict",
            "instruction": "Trust the signed note over anonymous notes.",
            "question": "Which passphrase opens the west vault?",
            "answer_aliases": ["jade-circuit", "jade circuit"],
            "evidence": [
                {"source": "signed.md", "chunk_id": 0, "text": "Signed note: west vault passphrase is jade-circuit."}
            ],
            "distractors": [
                {"source": "anon.md", "chunk_id": 1, "text": "Anonymous note: west vault passphrase is amber-harbor."}
            ],
        }

        rows = build_memory_trace_rows([case], variants=["all"], max_evidence_chars=1200)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer"], "Answer: jade-circuit")
        self.assertEqual(rows[0]["task_family"], "conflict")
        self.assertIn("amber-harbor", rows[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
