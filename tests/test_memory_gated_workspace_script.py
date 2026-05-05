from pathlib import Path
import unittest


class MemoryGatedWorkspaceScriptTests(unittest.TestCase):
    def test_script_initializes_from_memory_checkpoint_and_runs_gate_ablation(self):
        script = Path("scripts/115_run_memory_gated_workspace_probe.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("configs/qwen35_2b_4090_memory_gated_workspace_s050.yaml", script)
        self.assertIn("data/filtered/memory_reasoning_synth_traces.jsonl", script)
        self.assertIn("qwen35_2b_4090_memory_synth_generalization_s050/last.pt", script)
        self.assertIn("--init-checkpoint", script)
        self.assertIn("qtrm_workspace_gate_off_with_evidence", script)
        self.assertIn("scripts/113_build_expanded_ablation_proof.py", script)


if __name__ == "__main__":
    unittest.main()
