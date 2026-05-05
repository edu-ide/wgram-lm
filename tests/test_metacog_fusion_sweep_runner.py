from pathlib import Path
import unittest


class MetacogFusionSweepRunnerTests(unittest.TestCase):
    def test_runner_writes_eval_outputs_to_local_eval_not_runs_eval(self) -> None:
        script = Path("scripts/204_run_metacog_fusion_conflict_gate_sweep.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('OUT_ROOT="${OUT_ROOT:-local_eval}"', script)
        self.assertIn('"$OUT_ROOT/metacognitive_fusion_scale_sweep_baseline_40.jsonl"', script)
        self.assertIn('"$OUT_ROOT/metacognitive_fusion_scale_sweep_candidate_conflict_gate_40.jsonl"', script)
        self.assertNotIn("runs/eval/metacognitive_fusion_scale_sweep", script)

    def test_runner_uses_full_scale_sweep_and_conflict_gate_probe(self) -> None:
        script = Path("scripts/204_run_metacog_fusion_conflict_gate_sweep.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("qtrm_core_steps_8_donor_scale_1p0_no_evidence", script)
        self.assertIn("qtrm_core_steps_8_donor_scale_0p75_no_evidence", script)
        self.assertIn("qtrm_core_steps_8_donor_scale_0p50_no_evidence", script)
        self.assertIn("qtrm_core_steps_8_donor_scale_0p25_no_evidence", script)
        self.assertIn("--donor-qtrm-conflict-gate", script)
        self.assertIn("--donor-qtrm-conflict-qtrm-scale", script)

    def test_runner_builds_plain_and_conflict_gate_reports(self) -> None:
        script = Path("scripts/204_run_metacog_fusion_conflict_gate_sweep.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("metacog-fusion-scale-sweep-conservative-s040-full40.md", script)
        self.assertIn("metacog-fusion-conflict-gate-conservative-s040-full40.md", script)
        self.assertIn("scripts/202_build_metacognitive_calibration_gate.py", script)

    def test_runner_preflights_real_checkpoint_reads_and_output_writes(self) -> None:
        script = Path("scripts/204_run_metacog_fusion_conflict_gate_sweep.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("check_readable_file()", script)
        self.assertIn("check_writable_dir()", script)
        self.assertIn('check_readable_file "baseline checkpoint" "$BASELINE_CHECKPOINT" "BASELINE_CHECKPOINT"', script)
        self.assertIn('check_readable_file "candidate checkpoint" "$CANDIDATE_CHECKPOINT" "CANDIDATE_CHECKPOINT"', script)
        self.assertIn("open(path, 'rb').read(1)", script)
        self.assertIn("preflight_write_test", script)
        self.assertNotIn("BASELINE_CHECKPOINT_CHECKPOINT", script)


if __name__ == "__main__":
    unittest.main()
