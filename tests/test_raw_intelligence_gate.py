import unittest


class RawIntelligenceGateTests(unittest.TestCase):
    def test_pure_recursive_reasoning_accepts_only_depth_causal_gain(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-b", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": False},
            {"id": "case-b", "mode": "qtrm_core_off_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_steps_1_no_evidence", "hit": False},
            {"id": "case-b", "mode": "qtrm_core_steps_1_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_steps_2_no_evidence", "hit": True},
            {"id": "case-b", "mode": "qtrm_core_steps_2_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_steps_4_no_evidence", "hit": True},
            {"id": "case-b", "mode": "qtrm_core_steps_4_no_evidence", "hit": True},
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        self.assertEqual(gate["status"], "accepted")
        self.assertIn("deep_core_beats_core_off", gate["passed_checks"])
        self.assertIn("deep_core_beats_donor", gate["passed_checks"])
        self.assertIn("depth_scaling_gain_present", gate["passed_checks"])
        self.assertEqual(gate["deepest_core_mode"], "qtrm_core_steps_4_no_evidence")
        self.assertEqual(gate["core_off_comparison"]["hit_advantage"], 2)
        self.assertEqual(gate["depth_ladder"][-1]["hits"], 2)
        self.assertIn("mode_semantics", gate)
        self.assertIn("not equivalent to donor_only", gate["mode_semantics"]["core_off"])
        self.assertEqual(gate["eval_contract"]["scoring"], [])
        self.assertEqual(gate["eval_contract"]["choice_score_normalization"], [])

    def test_pure_recursive_reasoning_markdown_defines_ablation_modes(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
            render_markdown,
        )

        records = [
            {
                "id": "case-a",
                "mode": "donor_only_no_evidence",
                "hit": False,
                "scoring": "causal_forced_choice",
                "choice_score_normalization": "mean",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_off_no_evidence",
                "hit": False,
                "scoring": "causal_forced_choice",
                "choice_score_normalization": "mean",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "scoring": "causal_forced_choice",
                "choice_score_normalization": "mean",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": True,
                "scoring": "causal_forced_choice",
                "choice_score_normalization": "mean",
            },
        ]

        markdown = render_markdown(build_pure_recursive_reasoning_gate(records))

        self.assertIn("## Mode Semantics", markdown)
        self.assertIn("Donor baseline", markdown)
        self.assertIn("donor fallback is not forced", markdown)
        self.assertIn("## Eval Contract", markdown)
        self.assertIn("Scoring: `causal_forced_choice`", markdown)
        self.assertIn("Choice score normalization: `mean`", markdown)

    def test_pure_recursive_reasoning_rejects_when_core_off_ties_deep_core(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_core_steps_1_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_core_steps_4_no_evidence", "hit": True},
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("deep_core_does_not_beat_core_off", gate["failed_checks"])

    def test_pure_recursive_reasoning_rejects_identical_depth_outputs(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False, "completion": "red"},
            {"id": "case-b", "mode": "donor_only_no_evidence", "hit": False, "completion": "blue"},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": False, "completion": "red"},
            {"id": "case-b", "mode": "qtrm_core_off_no_evidence", "hit": False, "completion": "blue"},
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "completion": "violet",
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "completion": "orange",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_2_no_evidence",
                "hit": True,
                "completion": "violet",
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_2_no_evidence",
                "hit": False,
                "completion": "orange",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": True,
                "completion": "violet",
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": True,
                "completion": "orange",
            },
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("depth_outputs_identical_across_steps", gate["failed_checks"])
        self.assertEqual(gate["depth_output_diversity"]["identical_case_count"], 2)

    def test_pure_recursive_reasoning_reports_non_identical_depth_outputs(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False, "completion": "red"},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": False, "completion": "red"},
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "completion": "red",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_2_no_evidence",
                "hit": True,
                "completion": "violet",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": True,
                "completion": "violet",
            },
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        self.assertIn("depth_outputs_not_all_identical", gate["passed_checks"])
        self.assertEqual(gate["depth_output_diversity"]["changed_case_count"], 1)

    def test_pure_recursive_reasoning_checks_transition_state_off_when_present(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": False},
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "completion": "wrong",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_no_evidence",
                "hit": True,
                "completion": "right",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_transition_state_off_no_evidence",
                "hit": False,
                "completion": "wrong",
            },
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        self.assertEqual(gate["transition_state_off"]["hits"], 0)
        self.assertEqual(gate["transition_state_off_comparison"]["hit_advantage"], 1)
        self.assertIn("deep_core_beats_transition_state_off", gate["passed_checks"])

    def test_pure_recursive_reasoning_rejects_hidden_evidence_shortcuts(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_off_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_steps_1_no_evidence", "hit": False},
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": True,
                "memoryos_used": True,
            },
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("non_raw_shortcut_present", gate["failed_checks"])
        self.assertEqual(gate["shortcut_records"][0]["id"], "case-a")

    def test_pure_recursive_reasoning_reports_expected_paradigm_summaries(self):
        from qtrm_mm.eval.raw_intelligence_gate import (
            build_pure_recursive_reasoning_gate,
        )

        records = [
            {
                "id": "parallel-a",
                "mode": "donor_only_no_evidence",
                "hit": False,
                "expected_paradigm": "latent_parallel",
            },
            {
                "id": "serial-a",
                "mode": "donor_only_no_evidence",
                "hit": True,
                "expected_paradigm": "hybrid_or_cot",
            },
            {
                "id": "parallel-a",
                "mode": "qtrm_core_off_no_evidence",
                "hit": False,
                "expected_paradigm": "latent_parallel",
            },
            {
                "id": "serial-a",
                "mode": "qtrm_core_off_no_evidence",
                "hit": False,
                "expected_paradigm": "hybrid_or_cot",
            },
            {
                "id": "parallel-a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "expected_paradigm": "latent_parallel",
            },
            {
                "id": "serial-a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "hit": False,
                "expected_paradigm": "hybrid_or_cot",
            },
            {
                "id": "parallel-a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": True,
                "expected_paradigm": "latent_parallel",
            },
            {
                "id": "serial-a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "hit": False,
                "expected_paradigm": "hybrid_or_cot",
            },
        ]

        gate = build_pure_recursive_reasoning_gate(records)

        by_paradigm = gate["by_expected_paradigm"]
        self.assertEqual(
            by_paradigm["latent_parallel"]["qtrm_core_steps_4_no_evidence"]["hits"],
            1,
        )
        self.assertEqual(
            by_paradigm["hybrid_or_cot"]["donor_only_no_evidence"]["hits"],
            1,
        )

    def test_trainable_memory_gate_requires_memory_on_to_beat_memory_off(self):
        from qtrm_mm.eval.raw_intelligence_gate import build_trainable_memory_gate

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-b", "mode": "donor_only_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_memory_off_no_evidence", "hit": False},
            {"id": "case-b", "mode": "qtrm_memory_off_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_memory_on_no_evidence", "hit": True},
            {"id": "case-b", "mode": "qtrm_memory_on_no_evidence", "hit": True},
        ]

        gate = build_trainable_memory_gate(records)

        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["memory_off_comparison"]["hit_advantage"], 1)
        self.assertIn("memory_on_beats_memory_off", gate["passed_checks"])

    def test_temporal_spatial_context_gate_accepts_context_causal_gain(self):
        from qtrm_mm.eval.raw_intelligence_gate import build_temporal_spatial_context_gate

        records = [
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_no_evidence",
                "hit": True,
                "temporal_spatial_context_available": True,
                "temporal_spatial_context_token_count": 2,
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_8_no_evidence",
                "hit": True,
                "temporal_spatial_context_available": True,
                "temporal_spatial_context_token_count": 2,
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_temporal_spatial_off_no_evidence",
                "hit": False,
                "temporal_spatial_context_available": True,
                "disable_temporal_spatial_context": True,
                "temporal_spatial_context_token_count": 0,
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_steps_8_temporal_spatial_off_no_evidence",
                "hit": True,
                "temporal_spatial_context_available": True,
                "disable_temporal_spatial_context": True,
                "temporal_spatial_context_token_count": 0,
            },
        ]

        gate = build_temporal_spatial_context_gate(records)

        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["context_off_comparison"]["hit_advantage"], 1)
        self.assertIn("context_on_beats_context_off", gate["passed_checks"])
        self.assertIn("no_retrieval_or_memoryos_shortcut", gate["passed_checks"])

    def test_temporal_spatial_context_gate_rejects_when_context_off_ties(self):
        from qtrm_mm.eval.raw_intelligence_gate import build_temporal_spatial_context_gate

        records = [
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_no_evidence",
                "hit": True,
                "temporal_spatial_context_available": True,
                "temporal_spatial_context_token_count": 2,
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_steps_8_temporal_spatial_off_no_evidence",
                "hit": True,
                "temporal_spatial_context_available": True,
                "disable_temporal_spatial_context": True,
                "temporal_spatial_context_token_count": 0,
            },
        ]

        gate = build_temporal_spatial_context_gate(records)

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("context_on_does_not_beat_context_off", gate["failed_checks"])

    def test_composition_gate_requires_both_core_and_memory_to_be_causal(self):
        from qtrm_mm.eval.raw_intelligence_gate import build_composition_gate

        records = [
            {"id": "case-a", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-b", "mode": "donor_only_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_memory_on_no_evidence", "hit": True},
            {"id": "case-b", "mode": "qtrm_core_memory_on_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_core_off_memory_on_no_evidence", "hit": True},
            {"id": "case-b", "mode": "qtrm_core_off_memory_on_no_evidence", "hit": False},
            {"id": "case-a", "mode": "qtrm_core_on_memory_off_no_evidence", "hit": False},
            {"id": "case-b", "mode": "qtrm_core_on_memory_off_no_evidence", "hit": True},
        ]

        gate = build_composition_gate(records)

        self.assertEqual(gate["status"], "accepted")
        self.assertIn("full_beats_core_off", gate["passed_checks"])
        self.assertIn("full_beats_memory_off", gate["passed_checks"])

    def test_composition_gate_rejects_when_core_is_not_causal(self):
        from qtrm_mm.eval.raw_intelligence_gate import build_composition_gate

        records = [
            {"id": "case-a", "mode": "qtrm_core_memory_on_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_core_off_memory_on_no_evidence", "hit": True},
            {"id": "case-a", "mode": "qtrm_core_on_memory_off_no_evidence", "hit": False},
        ]

        gate = build_composition_gate(records)

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("full_does_not_beat_core_off", gate["failed_checks"])


if __name__ == "__main__":
    unittest.main()
