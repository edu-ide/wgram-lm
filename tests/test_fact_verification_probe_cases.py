import unittest


class FactVerificationProbeCaseTests(unittest.TestCase):
    def test_probe_cases_cover_all_core_verdicts_and_metadata(self):
        from qtrm_mm.eval.fact_verification import VALID_VERDICTS, load_fact_cases

        cases = load_fact_cases("data/eval/fact_verification_probe.jsonl")
        verdicts = {case["expected_verdict"] for case in cases}

        self.assertTrue({"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO", "CONFLICT", "STALE_OR_TIME_DEPENDENT"}.issubset(verdicts))
        for case in cases:
            self.assertIn(case["expected_verdict"], VALID_VERDICTS)
            self.assertIn(case["expected_action"], {"ANSWER", "NEEDS_SEARCH"})
            self.assertTrue(case.get("claim"))
            self.assertTrue(case.get("question"))
            self.assertTrue(case.get("evidence"))
            for rec in case["evidence"]:
                self.assertTrue(rec.get("source"))
                self.assertTrue(rec.get("text"))
                self.assertTrue(rec.get("verdict"))
                self.assertTrue(rec.get("source_type"))
                self.assertTrue(rec.get("credibility_tier"))


if __name__ == "__main__":
    unittest.main()
