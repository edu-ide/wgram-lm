import unittest


class FactVerificationEvalTests(unittest.TestCase):
    def test_build_fact_prompt_exposes_verdict_action_and_metadata(self):
        from wgram_lm.eval.fact_verification import build_fact_prompt

        case = {
            "id": "signed-current-code",
            "claim": "The current archive code is AX-42.",
            "question": "Is the current archive code AX-42?",
            "expected_verdict": "SUPPORTED",
            "expected_action": "ANSWER",
            "evidence": [
                {
                    "source": "signed_ops.md",
                    "text": "Signed operations memo: the current archive code is AX-42.",
                    "published_at": "2026-04-29",
                    "source_type": "signed_notice",
                    "credibility_tier": "signed",
                    "verdict": "SUPPORTED",
                }
            ],
        }

        prompt = build_fact_prompt(case, include_evidence=True)

        self.assertIn("Allowed verdicts", prompt)
        self.assertIn("Verdict:", prompt)
        self.assertIn("Action:", prompt)
        self.assertIn("signed_ops.md", prompt)
        self.assertIn("2026-04-29", prompt)
        self.assertIn("signed_notice", prompt)
        self.assertIn("signed", prompt)
        self.assertIn("AX-42", prompt)

    def test_infer_fact_verdict_handles_support_refute_conflict_temporal_and_authority(self):
        from wgram_lm.eval.fact_verification import infer_fact_verdict

        supported = {
            "id": "supported",
            "expected_verdict": "SUPPORTED",
            "evidence": [{"source": "a.md", "text": "A.", "verdict": "SUPPORTED"}],
        }
        refuted = {
            "id": "refuted",
            "expected_verdict": "REFUTED",
            "evidence": [{"source": "a.md", "text": "A.", "verdict": "REFUTED"}],
        }
        conflict = {
            "id": "conflict",
            "expected_verdict": "CONFLICT",
            "evidence": [
                {"source": "a.md", "text": "A.", "verdict": "SUPPORTED", "credibility_tier": "official"},
                {"source": "b.md", "text": "B.", "verdict": "REFUTED", "credibility_tier": "official"},
            ],
        }
        temporal = {
            "id": "temporal",
            "verification_strategy": "temporal",
            "expected_verdict": "NOT_ENOUGH_INFO",
            "evidence": [
                {
                    "source": "old.md",
                    "text": "Old memo names Mira.",
                    "published_at": "2025-01-01",
                    "verdict": "SUPPORTED",
                },
                {
                    "source": "new.md",
                    "text": "New memo says the current lead is not named.",
                    "published_at": "2026-04-29",
                    "verdict": "NOT_ENOUGH_INFO",
                },
            ],
        }
        authority = {
            "id": "authority",
            "verification_strategy": "authority",
            "expected_verdict": "SUPPORTED",
            "evidence": [
                {
                    "source": "signed.md",
                    "text": "Signed note supports the claim.",
                    "credibility_tier": "signed",
                    "verdict": "SUPPORTED",
                },
                {
                    "source": "anon.md",
                    "text": "Anonymous note refutes it.",
                    "credibility_tier": "anonymous",
                    "verdict": "REFUTED",
                },
            ],
        }

        self.assertEqual(infer_fact_verdict(supported), "SUPPORTED")
        self.assertEqual(infer_fact_verdict(refuted), "REFUTED")
        self.assertEqual(infer_fact_verdict(conflict), "CONFLICT")
        self.assertEqual(infer_fact_verdict(temporal), "NOT_ENOUGH_INFO")
        self.assertEqual(infer_fact_verdict(authority), "SUPPORTED")

    def test_evaluate_fact_case_scores_retrieval_verdict_and_action(self):
        from wgram_lm.eval.fact_verification import evaluate_fact_case

        case = {
            "id": "missing-current",
            "claim": "The current team lead is Mira.",
            "question": "Is Mira the current team lead?",
            "expected_verdict": "NOT_ENOUGH_INFO",
            "expected_action": "NEEDS_SEARCH",
            "verification_strategy": "temporal",
            "evidence": [
                {
                    "source": "old.md",
                    "text": "2025 memo: Mira was team lead.",
                    "published_at": "2025-01-01",
                    "verdict": "SUPPORTED",
                },
                {
                    "source": "new.md",
                    "text": "2026 memo: the new lead is not named.",
                    "published_at": "2026-04-29",
                    "verdict": "NOT_ENOUGH_INFO",
                },
            ],
        }

        record = evaluate_fact_case(case, evidence_mode="target", retrieval_top_k=5)

        self.assertEqual(record["predicted_verdict"], "NOT_ENOUGH_INFO")
        self.assertEqual(record["predicted_action"], "NEEDS_SEARCH")
        self.assertTrue(record["verdict_hit"])
        self.assertTrue(record["action_hit"])
        self.assertTrue(record["all_targets_retrieved"])

    def test_summarize_fact_records_counts_accuracy_by_label(self):
        from wgram_lm.eval.fact_verification import summarize_fact_records

        summary = summarize_fact_records(
            [
                {
                    "expected_verdict": "SUPPORTED",
                    "predicted_verdict": "SUPPORTED",
                    "expected_action": "ANSWER",
                    "predicted_action": "ANSWER",
                    "verdict_hit": True,
                    "action_hit": True,
                    "retrieved_target": True,
                    "all_targets_retrieved": True,
                    "target_recall": 1.0,
                },
                {
                    "expected_verdict": "REFUTED",
                    "predicted_verdict": "SUPPORTED",
                    "expected_action": "ANSWER",
                    "predicted_action": "ANSWER",
                    "verdict_hit": False,
                    "action_hit": True,
                    "retrieved_target": True,
                    "all_targets_retrieved": False,
                    "target_recall": 0.5,
                },
            ]
        )

        self.assertEqual(summary["overall"]["count"], 2)
        self.assertAlmostEqual(summary["overall"]["verdict_accuracy"], 0.5)
        self.assertAlmostEqual(summary["overall"]["action_accuracy"], 1.0)
        self.assertEqual(summary["by_expected_verdict"]["SUPPORTED"]["count"], 1)
        self.assertEqual(summary["by_expected_verdict"]["REFUTED"]["verdict_hits"], 0)


if __name__ == "__main__":
    unittest.main()
