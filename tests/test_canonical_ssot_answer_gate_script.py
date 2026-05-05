from pathlib import Path
import unittest


class CanonicalSsoTAnswerGateScriptTests(unittest.TestCase):
    def test_uses_greedy_single_stream_contract(self) -> None:
        script = Path("scripts/166_run_canonical_ssot_answer_gate.sh").read_text(encoding="utf-8")

        self.assertIn("Canonical SSOT autoregressive answer gate", script)
        self.assertIn("--require-canonical-ssot", script)
        self.assertIn("--evidence-injection ssot", script)
        self.assertIn("--answer-channel greedy", script)
        self.assertIn("--mode donor_only_with_evidence", script)
        self.assertIn("--mode qtrm_residual_with_evidence", script)
        self.assertIn("--mode qtrm_core_off_with_evidence", script)
        self.assertIn("--mode qtrm_workspace_off_with_evidence", script)
        self.assertIn("--mode qtrm_workspace_memory_off_with_evidence", script)
        self.assertIn("--mode qtrm_core_context_off_with_evidence", script)
        self.assertIn("--mode qtrm_core_to_text_off_with_evidence", script)
        self.assertIn("--mode qtrm_evidence_bottleneck_off_with_evidence", script)
        self.assertIn("--mode qtrm_evidence_span_reader_off_with_evidence", script)
        self.assertIn("--mode qtrm_answer_residual_governor_off_with_evidence", script)
        self.assertIn("STRICT_PROMOTION_GATE", script)
        self.assertIn("--strict-promotion-gate", script)
        self.assertIn("QTRM_SCALE_ARGS=()", script)
        self.assertNotIn('QTRM_LOGITS_SCALE="${QTRM_LOGITS_SCALE:-0.10}"', script)
        self.assertNotIn("evidence_span_copy", script)
        self.assertNotIn("--evidence-injection workspace", script)
        self.assertNotIn("--evidence-injection dual", script)
