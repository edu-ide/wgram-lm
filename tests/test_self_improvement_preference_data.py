import unittest


class SelfImprovementPreferenceDataTests(unittest.TestCase):
    def test_build_preference_rows_from_missed_eval_records_reconstructs_prompt(self):
        from qtrm_mm.training.self_improvement_data import build_preference_rows

        case = {
            "id": "temporal-ko-room",
            "category": "temporal_conflict_ko",
            "instruction": "날짜가 충돌하면 가장 최신 날짜의 증거를 우선하세요.",
            "question": "현재 동쪽 격납고의 확인 코드는 무엇인가요?",
            "answer_aliases": ["해돋이-31", "해돋이 31"],
            "evidence": [
                {
                    "source": "east_hangar_2026_ko.md",
                    "chunk_id": 0,
                    "text": "2026-04-29 공지: 현재 동쪽 격납고의 확인 코드는 해돋이-31이다.",
                }
            ],
            "distractors": [
                {
                    "source": "east_hangar_2025_ko.md",
                    "chunk_id": 1,
                    "text": "2025-04-02 공지: 동쪽 격납고의 확인 코드는 달빛-10이다.",
                }
            ],
        }
        records = [
            {
                "id": "temporal-ko-room",
                "category": "temporal_conflict_ko",
                "task_family": "conflict",
                "mode": "qtrm_residual_with_evidence",
                "hit": False,
                "completion": "Answer: 달빛-10.",
                "retrieved_sources": ["east_hangar_2026_ko.md", "east_hangar_2025_ko.md"],
                "retrieved_rerank_scores": [11.0, 10.0],
                "retrieved_retrieval_scores": [0.76, 0.73],
                "retrieved_rerank_backend": ["cross_encoder", "cross_encoder"],
                "retrieved_roles": ["target", "distractor"],
            }
        ]

        rows = build_preference_rows(
            [case],
            records,
            source_eval="runs/eval/heldout.jsonl",
            training_scope="analysis_only",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["type"], "memory_preference")
        self.assertEqual(row["case_id"], "temporal-ko-room")
        self.assertEqual(row["chosen"], "Answer: 해돋이-31")
        self.assertEqual(row["rejected"], "Answer: 달빛-10.")
        self.assertEqual(row["training_scope"], "analysis_only")
        self.assertIn("east_hangar_2026_ko.md", row["prompt"])
        self.assertIn("해돋이-31", row["prompt"])
        self.assertIn("달빛-10", row["prompt"])
        self.assertEqual(row["failure_tags"], ["wrong_answer"])

    def test_missing_answer_prefers_needs_search_not_final_unknown(self):
        from qtrm_mm.training.self_improvement_data import build_preference_rows

        case = {
            "id": "missing-answer",
            "category": "negative_missing",
            "instruction": "If the requested answer is not present in the evidence, answer UNKNOWN.",
            "question": "Which passphrase opens the south vault?",
            "answer_aliases": ["UNKNOWN", "unknown"],
            "evidence": [
                {"source": "north.md", "chunk_id": 0, "text": "The north vault passphrase is jade."}
            ],
            "distractors": [
                {"source": "south-storage.md", "chunk_id": 1, "text": "The south storage marker is Polaris."}
            ],
        }
        records = [
            {
                "id": "missing-answer",
                "category": "negative_missing",
                "task_family": "abstention",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "Answer: UNKNOWN. UNKNOWN UNKNOWN UNKNOWN UNKNOWN",
                "retrieved_sources": ["north.md", "south-storage.md"],
                "retrieved_rerank_scores": [5.0, 4.0],
                "retrieved_retrieval_scores": [0.6, 0.5],
                "retrieved_rerank_backend": ["cross_encoder", "cross_encoder"],
                "retrieved_roles": ["target", "distractor"],
            }
        ]

        rows = build_preference_rows(
            [case],
            records,
            include_hits_with_artifacts=True,
            missing_answer_policy="needs_search",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen"], "Action: NEEDS_SEARCH")
        self.assertEqual(rows[0]["rejected"], "Answer: UNKNOWN. UNKNOWN UNKNOWN UNKNOWN UNKNOWN")
        self.assertIn("unknown_repetition", rows[0]["failure_tags"])
        self.assertIn("abstention", rows[0]["failure_tags"])
        self.assertIn("needs_search", rows[0]["failure_tags"])
        self.assertEqual(rows[0]["resolution_state"], "needs_search")
        self.assertIn("Action: NEEDS_SEARCH", rows[0]["prompt"])
        self.assertNotIn("return UNKNOWN", rows[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
