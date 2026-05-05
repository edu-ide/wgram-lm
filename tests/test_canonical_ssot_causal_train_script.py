from pathlib import Path
import unittest


class CanonicalSsoTCausalTrainScriptTests(unittest.TestCase):
    def test_script_builds_ssot_traces_and_runs_canonical_training(self) -> None:
        script = Path("scripts/168_run_canonical_ssot_greedy_causal_train.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("configs/qwen35_2b_4090_canonical_ssot_greedy_causal_s050.yaml", script)
        self.assertIn("scripts/99_build_memory_trace_data.py", script)
        self.assertIn("data/filtered/memory_reasoning_synth_traces.jsonl", script)
        self.assertIn("scripts/166_run_canonical_ssot_answer_gate.sh", script)
        self.assertIn("--use-donor", script)
        self.assertIn("--data-jsonl", script)
        self.assertIn('EVAL_OUT="${OUT:-', script)
        self.assertIn('EVAL_ROOT_MD="${ROOT_MD:-', script)
        self.assertNotIn("evidence_span_copy", script)
        self.assertNotIn("EVIDENCE_INJECTION=workspace", script)
        self.assertNotIn("EVIDENCE_INJECTION=dual", script)
