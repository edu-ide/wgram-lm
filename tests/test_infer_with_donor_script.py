from pathlib import Path
import unittest


class InferWithDonorScriptTest(unittest.TestCase):
    def test_infer_with_donor_script_exposes_logit_scale_overrides(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("DONOR_LOGITS_SCALE=${DONOR_LOGITS_SCALE:-}", script)
        self.assertIn("QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-}", script)
        self.assertIn("model.cfg.donor_logits_scale = float(donor_scale_override)", script)
        self.assertIn("model.cfg.qtrm_logits_scale = float(qtrm_scale_override)", script)
        self.assertIn("Donor logits scale:", script)
        self.assertIn("QTRM logits scale:", script)

    def test_infer_with_donor_script_exposes_bounded_residual_overrides(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("QTRM_RESIDUAL_CLAMP=${QTRM_RESIDUAL_CLAMP:-}", script)
        self.assertIn("QTRM_RESIDUAL_GATE=${QTRM_RESIDUAL_GATE:-}", script)
        self.assertIn("cfg.model.qtrm_residual_clamp = float(residual_clamp_override)", script)
        self.assertIn("cfg.model.qtrm_residual_gate_enabled = parse_bool(residual_gate_override)", script)
        self.assertIn("model.residual_gate.bias.data.fill_(float(residual_gate_bias_override))", script)
        self.assertIn("QTRM residual clamp:", script)
        self.assertIn("QTRM residual gate:", script)

    def test_infer_with_donor_script_can_suppress_visible_reasoning_tokens(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("SUPPRESS_VISIBLE_REASONING=${SUPPRESS_VISIBLE_REASONING:-0}", script)
        self.assertIn("NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-0}", script)
        self.assertIn("ANSWER_CONTRACT=${ANSWER_CONTRACT:-none}", script)
        self.assertIn("suppress_visible_reasoning", script)
        self.assertIn("apply_answer_contract", script)
        self.assertIn("visible_reasoning_token_ids", script)
        self.assertIn("no_repeat_ngram_banned_tokens", script)
        self.assertIn("last_logit[:, suppressed_token_ids] = -torch.inf", script)

    def test_infer_with_donor_script_appends_generation_history(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("HISTORY_JSONL=${HISTORY_JSONL:-auto}", script)
        self.assertIn("append_generation_history", script)
        self.assertIn("source=\"infer_with_donor\"", script)
        self.assertIn("completion=completion_text", script)

    def test_infer_with_donor_script_defaults_to_language_safe_mode(self):
        script = Path("scripts/90_infer_with_donor.sh").read_text(encoding="utf-8")

        self.assertIn("LANGUAGE_SAFE=${LANGUAGE_SAFE:-1}", script)
        self.assertIn('if [[ "$LANGUAGE_SAFE" == "1" ]]; then', script)
        self.assertIn("DONOR_LOGITS_SCALE=${DONOR_LOGITS_SCALE:-1.0}", script)
        self.assertIn("QTRM_LOGITS_SCALE=${QTRM_LOGITS_SCALE:-0.0}", script)
        self.assertIn("SUPPRESS_VISIBLE_REASONING=${SUPPRESS_VISIBLE_REASONING:-1}", script)
        self.assertIn("NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-2}", script)
        self.assertIn("STOP_AFTER_SENTENCE=${STOP_AFTER_SENTENCE:-1}", script)
        self.assertIn("MIN_NEW_TOKENS_BEFORE_STOP=${MIN_NEW_TOKENS_BEFORE_STOP:-16}", script)
        self.assertIn("should_stop_after_sentence", script)
        self.assertIn(r're.search(r"[.!?。！？]\s*$", completion)', script)
        self.assertIn("language_safe_donor", script)
        self.assertIn("mode=history_mode", script)
        self.assertIn('"language_safe": language_safe', script)
        self.assertIn('"stop_after_sentence": stop_after_sentence', script)


if __name__ == "__main__":
    unittest.main()
