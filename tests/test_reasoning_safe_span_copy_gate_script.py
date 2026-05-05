from pathlib import Path
import unittest


class ReasoningSafeSpanCopyGateScriptTests(unittest.TestCase):
    def test_is_probe_only_and_defaults_to_ssot(self) -> None:
        script = Path("scripts/153_run_reasoning_safe_span_copy_gate.sh").read_text(encoding="utf-8")

        self.assertIn("PROBE-ONLY", script)
        self.assertIn("qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500.yaml", script)
        self.assertIn("qwen35_2b_4090_evidence_span_reader_trainhardnegx2_s500/last.pt", script)
        self.assertIn("EVIDENCE_INJECTION=\"${EVIDENCE_INJECTION:-ssot}\"", script)
        self.assertIn("--evidence-injection \"$EVIDENCE_INJECTION\"", script)
        self.assertIn("--answer-channel evidence_span_copy", script)
        self.assertIn("--evidence-span-no-answer-threshold \"$EVIDENCE_SPAN_NO_ANSWER_THRESHOLD\"", script)
        self.assertIn("EVIDENCE_SPAN_NO_ANSWER_THRESHOLD=\"${EVIDENCE_SPAN_NO_ANSWER_THRESHOLD:-0.1}\"", script)
        self.assertIn("--evidence-span-min-score \"$EVIDENCE_SPAN_MIN_SCORE\"", script)
        self.assertIn("EVIDENCE_SPAN_MIN_SCORE=\"${EVIDENCE_SPAN_MIN_SCORE:-12}\"", script)
        self.assertIn("TRUTH_GATE=\"${TRUTH_GATE:-0}\"", script)
        self.assertIn("truth_args=()", script)
        self.assertIn("--truth-gate", script)
        self.assertIn("--truth-support-threshold \"$TRUTH_SUPPORT_THRESHOLD\"", script)
        self.assertIn("--truth-causal-threshold \"$TRUTH_CAUSAL_THRESHOLD\"", script)
        self.assertIn("--truth-refute-threshold \"$TRUTH_REFUTE_THRESHOLD\"", script)
        self.assertIn("--truth-missing-threshold \"$TRUTH_MISSING_THRESHOLD\"", script)
        self.assertIn("--mode donor_only_with_evidence", script)
        self.assertIn("--mode qtrm_core_off_with_evidence", script)
        self.assertIn("--mode qtrm_core_context_off_with_evidence", script)
        self.assertIn("--mode qtrm_workspace_memory_off_with_evidence", script)
        self.assertIn("--mode qtrm_evidence_bottleneck_off_with_evidence", script)
        self.assertIn("scripts/148_build_root_architecture_gate.py", script)
