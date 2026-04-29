import unittest


class FactVerificationDataTests(unittest.TestCase):
    def test_build_fact_trace_rows_uses_structured_verdict_and_action(self):
        from qtrm_mm.training.fact_verification_data import build_fact_trace_rows

        case = {
            "id": "unsupported-answer",
            "claim": "The south vault passphrase is jade.",
            "question": "Is the south vault passphrase jade?",
            "expected_verdict": "NOT_ENOUGH_INFO",
            "expected_action": "NEEDS_SEARCH",
            "expected_answer": "Insufficient evidence; search is required.",
            "evidence": [
                {
                    "source": "north_vault.md",
                    "text": "The north vault passphrase is jade.",
                    "verdict": "NOT_ENOUGH_INFO",
                }
            ],
        }

        rows = build_fact_trace_rows([case], variants=("target",), top_k=3)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["type"], "fact_verification_trace")
        self.assertEqual(row["case_id"], "unsupported-answer")
        self.assertEqual(row["expected_verdict"], "NOT_ENOUGH_INFO")
        self.assertEqual(row["expected_action"], "NEEDS_SEARCH")
        self.assertIn("Verdict: NOT_ENOUGH_INFO", row["answer"])
        self.assertIn("Action: NEEDS_SEARCH", row["answer"])
        self.assertIn("Insufficient evidence", row["answer"])
        self.assertIn("north_vault.md", row["prompt"])


if __name__ == "__main__":
    unittest.main()
