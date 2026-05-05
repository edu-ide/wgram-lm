import unittest
import importlib.util
from pathlib import Path


def load_memory_eval_script():
    script_path = Path("scripts/95_eval_memory_retrieval.py")
    spec = importlib.util.spec_from_file_location("memory_eval_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryRetrievalEvalTests(unittest.TestCase):
    def test_answer_hit_normalizes_case_punctuation_and_korean_spacing(self):
        from qtrm_mm.eval.memory_retrieval import answer_hit

        self.assertTrue(answer_hit("정답은 루미나 17입니다.", ["루미나-17"]))
        self.assertTrue(answer_hit("The code is vx 913.", ["VX-913"]))
        self.assertFalse(answer_hit("The code is VX-914.", ["VX-913"]))

    def test_score_answer_separates_exact_normalized_loose_and_audit(self):
        from qtrm_mm.eval.memory_retrieval import score_answer

        exact = score_answer("VX-913", ["VX-913"])
        self.assertTrue(exact["hit"])
        self.assertTrue(exact["exact_match"])
        self.assertTrue(exact["normalized_exact"])
        self.assertTrue(exact["normalized_contains"])
        self.assertEqual(exact["match_type"], "exact")
        self.assertFalse(exact["needs_human_audit"])

        normalized = score_answer("Answer: vx 913.", ["VX-913"])
        self.assertTrue(normalized["hit"])
        self.assertFalse(normalized["exact_match"])
        self.assertTrue(normalized["normalized_exact"])
        self.assertEqual(normalized["match_type"], "normalized_exact")
        self.assertFalse(normalized["needs_human_audit"])

        loose = score_answer("Answer: VX-913. Older code VX-112 is deprecated.", ["VX-913"])
        self.assertTrue(loose["hit"])
        self.assertFalse(loose["normalized_exact"])
        self.assertTrue(loose["normalized_contains"])
        self.assertEqual(loose["match_type"], "normalized_contains")
        self.assertTrue(loose["needs_human_audit"])
        self.assertIn("loose_contains_match", loose["audit_reasons"])

    def test_score_answer_marks_unknown_repetition_for_audit(self):
        from qtrm_mm.eval.memory_retrieval import score_answer

        clean = score_answer("Answer: UNKNOWN", ["UNKNOWN"], expected_unknown=True)
        self.assertTrue(clean["hit"])
        self.assertTrue(clean["unknown_correct"])
        self.assertEqual(clean["match_type"], "unknown_exact")
        self.assertFalse(clean["needs_human_audit"])

        repeated = score_answer(
            "Answer: UNKNOWN. UNKNOWN UNKNOWN UNKNOWN",
            ["UNKNOWN"],
            expected_unknown=True,
        )
        self.assertTrue(repeated["hit"])
        self.assertTrue(repeated["unknown_correct"])
        self.assertTrue(repeated["needs_human_audit"])
        self.assertIn("unknown_with_extra_text", repeated["audit_reasons"])

    def test_eval_script_json_safe_value_summarizes_tensors(self):
        import json
        import torch

        module = load_memory_eval_script()
        record = {
            "forward_ablation": {
                "workspace_attention_mask": torch.ones(1, 3),
                "disable_workspace": False,
            }
        }

        safe = module.json_safe_value(record)

        json.dumps(safe)
        self.assertEqual(
            safe["forward_ablation"]["workspace_attention_mask"]["tensor_shape"],
            [1, 3],
        )
        self.assertEqual(safe["forward_ablation"]["disable_workspace"], False)

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

    def test_build_case_prompt_and_workspace_memory_splits_evidence_path(self):
        from qtrm_mm.eval.memory_retrieval import build_case_prompt_and_workspace_memory

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

        prompt, workspace_memory = build_case_prompt_and_workspace_memory(
            case,
            include_evidence=True,
            evidence_injection="workspace",
            max_evidence_chars=200,
        )

        self.assertIn("What is the access code?", prompt)
        self.assertNotIn("MemoryOS evidence", prompt)
        self.assertNotIn("VX-913", prompt)
        self.assertIsNotNone(workspace_memory)
        self.assertIn("MemoryOS evidence", workspace_memory)
        self.assertIn("VX-913", workspace_memory)

        prompt_path, prompt_workspace_memory = build_case_prompt_and_workspace_memory(
            case,
            include_evidence=True,
            evidence_injection="prompt",
            max_evidence_chars=200,
        )

        self.assertIn("MemoryOS evidence", prompt_path)
        self.assertIn("VX-913", prompt_path)
        self.assertIsNone(prompt_workspace_memory)

    def test_build_case_prompt_and_workspace_memory_dual_path_keeps_visible_and_workspace_evidence(self):
        from qtrm_mm.data.jsonl_dataset import split_memory_prompt_for_workspace
        from qtrm_mm.eval.memory_retrieval import (
            build_case_prompt_and_workspace_memory,
            build_shared_evidence_context,
        )

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
        _, shared_context = build_shared_evidence_context(case, max_evidence_chars=200)

        prompt, workspace_memory = build_case_prompt_and_workspace_memory(
            case,
            include_evidence=True,
            evidence_injection="dual",
            max_evidence_chars=200,
        )

        self.assertIn("MemoryOS evidence", prompt)
        self.assertIn("VX-913", prompt)
        self.assertIsNotNone(workspace_memory)
        self.assertEqual(workspace_memory, shared_context)
        _, visible_context = split_memory_prompt_for_workspace(prompt)
        self.assertEqual(visible_context, shared_context)
        self.assertIn("VX-913", workspace_memory)

    def test_build_case_prompt_and_workspace_memory_ssot_uses_one_canonical_prompt(self):
        from qtrm_mm.eval.memory_retrieval import build_case_prompt_and_workspace_memory

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

        prompt, workspace_memory = build_case_prompt_and_workspace_memory(
            case,
            include_evidence=True,
            evidence_injection="ssot",
            max_evidence_chars=200,
        )

        self.assertIn("MemoryOS evidence", prompt)
        self.assertIn("VX-913", prompt)
        self.assertIn("User prompt:", prompt)
        self.assertIsNone(workspace_memory)

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

    def test_evidence_source_governor_prefers_signed_over_anonymous(self):
        from qtrm_mm.eval.memory_retrieval import evidence_records, govern_evidence_sources

        case = {
            "id": "garnet-vault",
            "category": "authority_conflict_synth",
            "question": "Which passphrase opens the Garnet vault?",
            "evidence": [
                {
                    "source": "signed_garnet_vault.md",
                    "text": "Signed supervisor note: the Garnet vault passphrase is stone-arch.",
                }
            ],
            "distractors": [
                {
                    "source": "anonymous_garnet_vault.md",
                    "text": "Anonymous note: the Garnet vault passphrase is cedar-gate.",
                }
            ],
        }
        results = [(1.0, rec) for rec in evidence_records(case, include_distractors=True)]

        governed = govern_evidence_sources(case, results, governor="reliability")

        self.assertEqual([rec["source"] for _, rec in governed], ["signed_garnet_vault.md"])

    def test_evidence_source_governor_prefers_current_latest_temporal_record(self):
        from qtrm_mm.eval.memory_retrieval import evidence_records, govern_evidence_sources

        case = {
            "id": "garnet-observatory",
            "category": "temporal_conflict_ko_synth",
            "question": "현재 Garnet 관측실의 확인 코드는 무엇인가요?",
            "evidence": [
                {
                    "source": "garnet_observatory_2026_ko.md",
                    "text": "2026-04-29 공지: 현재 Garnet 관측실의 확인 코드는 새벽-14이다.",
                }
            ],
            "distractors": [
                {
                    "source": "garnet_observatory_2025_ko.md",
                    "text": "2025-02-01 공지: Garnet 관측실의 확인 코드는 구름-39이다.",
                }
            ],
        }
        results = [(1.0, rec) for rec in evidence_records(case, include_distractors=True)]

        governed = govern_evidence_sources(case, results, governor="reliability")

        self.assertEqual([rec["source"] for _, rec in governed], ["garnet_observatory_2026_ko.md"])

    def test_evidence_source_governor_prunes_decoy_multihop_records(self):
        from qtrm_mm.eval.memory_retrieval import evidence_records, govern_evidence_sources

        case = {
            "id": "project-frost",
            "category": "multi_hop_synth",
            "question": "Who maintains the crate assigned to Project Frost?",
            "evidence": [
                {"source": "project_frost.md", "text": "Project Frost is assigned to crate C-131."},
                {"source": "crate_c131.md", "text": "Crate C-131 is stored in Bay Frost."},
                {"source": "bay_frost.md", "text": "Bay Frost is maintained by Sena Cho."},
            ],
            "distractors": [
                {"source": "bay_decoy_frost.md", "text": "Bay Decoy-Frost is maintained by Ilya Moon."},
                {"source": "project_other_frost.md", "text": "Project Other-Frost is assigned to crate C-151."},
            ],
        }
        results = [(1.0, rec) for rec in evidence_records(case, include_distractors=True)]

        governed = govern_evidence_sources(case, results, governor="reliability")

        sources = [rec["source"] for _, rec in governed]
        self.assertIn("bay_frost.md", sources)
        self.assertNotIn("bay_decoy_frost.md", sources)
        self.assertNotIn("project_other_frost.md", sources)

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

    def test_link_expansion_adds_records_named_by_selected_multihop_evidence(self):
        from qtrm_mm.eval.memory_retrieval import expand_linked_evidence_results

        selected = [
            (
                10.0,
                {
                    "case_id": "project-ember",
                    "source": "project_ember.md",
                    "chunk_id": 0,
                    "text": "Project Ember is assigned to crate K-42.",
                    "is_target": True,
                },
            ),
            (
                8.0,
                {
                    "case_id": "project-ember",
                    "source": "crate_k42.md",
                    "chunk_id": 1,
                    "text": "Crate K-42 is stored in Bay Neon.",
                    "is_target": True,
                },
            ),
            (
                7.0,
                {
                    "case_id": "project-ember",
                    "source": "bay_opal.md",
                    "chunk_id": 4,
                    "text": "Bay Opal is maintained by Mira Sol.",
                    "is_target": False,
                },
            ),
        ]
        candidates = selected + [
            (
                1.0,
                {
                    "case_id": "project-ember",
                    "source": "bay_neon.md",
                    "chunk_id": 2,
                    "text": "Bay Neon is maintained by Ilya Chen.",
                    "is_target": True,
                },
            ),
            (
                0.5,
                {
                    "case_id": "project-ember",
                    "source": "unrelated.md",
                    "chunk_id": 5,
                    "text": "Bay Silver is maintained by Nara Cho.",
                    "is_target": False,
                },
            ),
        ]

        expanded = expand_linked_evidence_results(selected, candidates, max_extra=1)

        self.assertEqual([rec["source"] for _, rec in expanded][-1], "bay_neon.md")
        self.assertEqual(sum(1 for _, rec in expanded if rec.get("is_target")), 3)

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
                "exact_match": True,
                "normalized_exact": True,
                "normalized_contains": True,
                "needs_human_audit": False,
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "category": "negative_missing",
                "expected_unknown": True,
                "mode": "donor_only_with_evidence",
                "hit": False,
                "exact_match": False,
                "normalized_exact": False,
                "normalized_contains": False,
                "needs_human_audit": True,
                "retrieved_target": False,
                "all_targets_retrieved": False,
                "target_recall": 0.0,
            },
            {
                "category": "temporal_conflict",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "exact_match": False,
                "normalized_exact": False,
                "normalized_contains": True,
                "needs_human_audit": True,
                "retrieved_target": True,
                "all_targets_retrieved": False,
                "target_recall": 0.5,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["overall"]["count"], 3)
        self.assertAlmostEqual(summary["overall"]["accuracy"], 2 / 3)
        self.assertEqual(summary["overall"]["exact_match_count"], 1)
        self.assertEqual(summary["overall"]["normalized_exact_count"], 1)
        self.assertEqual(summary["overall"]["normalized_contains_count"], 2)
        self.assertEqual(summary["overall"]["human_audit_count"], 2)
        self.assertAlmostEqual(summary["overall"]["human_audit_rate"], 2 / 3)
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
