import unittest


class MemoryRetrievalEvalTests(unittest.TestCase):
    def test_answer_hit_normalizes_case_punctuation_and_korean_spacing(self):
        from qtrm_mm.eval.memory_retrieval import answer_hit

        self.assertTrue(answer_hit("정답은 루미나 17입니다.", ["루미나-17"]))
        self.assertTrue(answer_hit("The code is vx 913.", ["VX-913"]))
        self.assertFalse(answer_hit("The code is VX-914.", ["VX-913"]))

    def test_build_case_prompt_can_include_or_omit_evidence(self):
        from qtrm_mm.eval.memory_retrieval import build_case_prompt

        case = {
            "id": "synthetic-code",
            "question": "What is the access code?",
            "evidence": [
                {
                    "source": "synthetic.md",
                    "chunk_id": 0,
                    "text": "The access code is VX-913.",
                }
            ],
        }

        with_evidence = build_case_prompt(case, include_evidence=True, max_evidence_chars=200)
        without_evidence = build_case_prompt(case, include_evidence=False, max_evidence_chars=200)

        self.assertIn("MemoryOS evidence", with_evidence)
        self.assertIn("VX-913", with_evidence)
        self.assertIn("Answer using only the evidence", with_evidence)
        self.assertNotIn("MemoryOS evidence", without_evidence)
        self.assertIn("What is the access code?", without_evidence)

    def test_build_case_prompt_includes_case_instruction(self):
        from qtrm_mm.eval.memory_retrieval import build_case_prompt

        case = {
            "id": "temporal-code",
            "instruction": "Prefer the newest dated evidence when records conflict.",
            "question": "What is the current archive code?",
            "evidence": [{"source": "latest.md", "text": "2026-04-29: current code is VX-913."}],
        }

        prompt = build_case_prompt(case, include_evidence=True)

        self.assertIn("Prefer the newest dated evidence", prompt)
        self.assertIn("What is the current archive code?", prompt)

    def test_distractor_retrieval_ranks_target_and_preserves_roles(self):
        from qtrm_mm.eval.memory_retrieval import (
            build_case_prompt,
            evidence_records,
            lexical_retrieve_case,
            target_retrieved,
        )

        case = {
            "id": "archive-code",
            "question": "What is the archive access code?",
            "evidence": [
                {
                    "source": "archive.md",
                    "chunk_id": 0,
                    "text": "The archive access code is VX-913.",
                }
            ],
            "distractors": [
                {
                    "source": "vault.md",
                    "chunk_id": 1,
                    "text": "The west vault passphrase is jade-circuit.",
                },
                {
                    "source": "relay.md",
                    "chunk_id": 2,
                    "text": "The Marigold relay launch date is 2047-11-03.",
                },
            ],
        }

        records = evidence_records(case, include_distractors=True)
        self.assertEqual([r["evidence_role"] for r in records], ["target", "distractor", "distractor"])

        retrieved = lexical_retrieve_case(case, top_k=2, include_distractors=True)
        self.assertTrue(target_retrieved(retrieved))
        self.assertEqual(retrieved[0][1]["evidence_role"], "target")

        prompt = build_case_prompt(
            case,
            include_evidence=True,
            evidence_results=retrieved,
            max_evidence_chars=400,
        )
        self.assertIn("VX-913", prompt)
        self.assertIn("jade-circuit", prompt)

    def test_select_evidence_results_supports_target_all_and_lexical_modes(self):
        from qtrm_mm.eval.memory_retrieval import select_evidence_results

        case = {
            "id": "archive-code",
            "question": "What is the archive access code?",
            "evidence": [{"source": "archive.md", "text": "The archive access code is VX-913."}],
            "distractors": [
                {"source": "vault.md", "text": "The west vault passphrase is jade-circuit."},
                {"source": "relay.md", "text": "The Marigold relay launch date is 2047-11-03."},
            ],
        }

        target = select_evidence_results(case, evidence_mode="target", top_k=3)
        all_records = select_evidence_results(case, evidence_mode="all", top_k=3)
        lexical = select_evidence_results(case, evidence_mode="lexical", top_k=2)

        self.assertEqual(len(target), 1)
        self.assertEqual(len(all_records), 3)
        self.assertEqual(len(lexical), 2)
        self.assertEqual(lexical[0][1]["evidence_role"], "target")

    def test_case_index_records_preserve_target_metadata_for_memoryos_index(self):
        from qtrm_mm.eval.memory_retrieval import case_index_records

        cases = [
            {
                "id": "archive-code",
                "question": "What is the archive access code?",
                "evidence": [{"source": "archive.md", "text": "The archive access code is VX-913."}],
                "distractors": [{"source": "vault.md", "text": "The west vault passphrase is jade-circuit."}],
            }
        ]

        records = case_index_records(cases, include_distractors=True)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["case_id"], "archive-code")
        self.assertTrue(records[0]["is_target"])
        self.assertFalse(records[1]["is_target"])
        self.assertEqual(records[1]["evidence_role"], "distractor")

    def test_filter_results_for_case_keeps_case_scoped_memoryos_hits(self):
        from qtrm_mm.eval.memory_retrieval import filter_results_for_case

        results = [
            (0.99, {"case_id": "other", "source": "other.md"}),
            (0.90, {"case_id": "archive-code", "source": "target.md"}),
            (0.80, {"case_id": "archive-code", "source": "distractor.md"}),
        ]

        filtered = filter_results_for_case(results, case_id="archive-code", top_k=1)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][1]["source"], "target.md")

    def test_target_retrieval_stats_counts_multihop_targets(self):
        from qtrm_mm.eval.memory_retrieval import evidence_records, target_retrieval_stats

        case = {
            "id": "project-label",
            "question": "What label belongs to Project Lumen?",
            "evidence": [
                {"source": "project.md", "text": "Project Lumen uses container C-17."},
                {"source": "container.md", "text": "Container C-17 has label Quartz-58."},
            ],
            "distractors": [{"source": "wrong.md", "text": "Container C-19 has label Amber-02."}],
        }
        records = evidence_records(case, include_distractors=True)
        results = [(1.0, records[0]), (0.8, records[2])]

        stats = target_retrieval_stats(case, results)

        self.assertEqual(stats["target_count"], 2)
        self.assertEqual(stats["retrieved_target_count"], 1)
        self.assertFalse(stats["all_targets_retrieved"])
        self.assertAlmostEqual(stats["target_recall"], 0.5)

    def test_summarize_records_counts_accuracy_by_mode(self):
        from qtrm_mm.eval.memory_retrieval import summarize_records

        records = [
            {
                "category": "negative_missing",
                "expected_unknown": True,
                "mode": "donor_only_with_evidence",
                "hit": True,
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "category": "negative_missing",
                "expected_unknown": True,
                "mode": "donor_only_with_evidence",
                "hit": False,
                "retrieved_target": False,
                "all_targets_retrieved": False,
                "target_recall": 0.0,
            },
            {
                "category": "temporal_conflict",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "retrieved_target": True,
                "all_targets_retrieved": False,
                "target_recall": 0.5,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["overall"]["count"], 3)
        self.assertAlmostEqual(summary["overall"]["accuracy"], 2 / 3)
        self.assertEqual(summary["overall"]["retrieved_target_count"], 2)
        self.assertAlmostEqual(summary["overall"]["retrieved_target_rate"], 2 / 3)
        self.assertEqual(summary["overall"]["all_targets_retrieved_count"], 1)
        self.assertAlmostEqual(summary["overall"]["target_recall_mean"], 0.5)
        self.assertEqual(summary["by_mode"]["donor_only_with_evidence"]["count"], 2)
        self.assertAlmostEqual(summary["by_mode"]["donor_only_with_evidence"]["accuracy"], 0.5)
        self.assertAlmostEqual(
            summary["by_mode"]["donor_only_with_evidence"]["retrieved_target_rate"],
            0.5,
        )
        self.assertEqual(summary["by_category"]["negative_missing"]["count"], 2)
        self.assertAlmostEqual(summary["by_category"]["negative_missing"]["accuracy"], 0.5)
        self.assertEqual(summary["by_task_family"]["abstention"]["count"], 2)
        self.assertAlmostEqual(summary["by_task_family"]["abstention"]["accuracy"], 0.5)
        self.assertEqual(summary["by_task_family"]["conflict"]["count"], 1)
        self.assertAlmostEqual(summary["by_task_family"]["conflict"]["accuracy"], 1.0)

    def test_case_task_family_detects_unknown_conflict_and_multihop(self):
        from qtrm_mm.eval.memory_retrieval import case_task_family, expected_unknown_case

        self.assertTrue(expected_unknown_case({"answer_aliases": ["UNKNOWN"]}))
        self.assertEqual(case_task_family({"category": "negative_missing"}), "abstention")
        self.assertEqual(case_task_family({"category": "temporal_conflict_ko"}), "conflict")
        self.assertEqual(case_task_family({"category": "multi_hop"}), "multi_hop")
        self.assertEqual(case_task_family({"category": "other"}), "other")


if __name__ == "__main__":
    unittest.main()
