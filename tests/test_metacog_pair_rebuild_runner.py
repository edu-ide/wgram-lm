from pathlib import Path
import unittest


class MetacogPairRebuildRunnerTests(unittest.TestCase):
    def test_runner_can_create_random_init_baseline_on_healthy_disk(self) -> None:
        script = Path("scripts/206_run_metacog_pair_rebuild.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'LOCAL_CKPT_ROOT="${LOCAL_CKPT_ROOT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_pair_rebuild}"',
            script,
        )
        self.assertIn('ALLOW_RANDOM_INIT="${ALLOW_RANDOM_INIT:-0}"', script)
        self.assertIn("--allow-random-init", script)
        self.assertIn('BASELINE_STEPS="${BASELINE_STEPS:-0}"', script)
        self.assertIn("preflight_write_test", script)

    def test_runner_uses_rebuilt_baseline_as_candidate_teacher(self) -> None:
        script = Path("scripts/206_run_metacog_pair_rebuild.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('BASELINE_CHECKPOINT="$BASELINE_DIR/last.pt"', script)
        self.assertIn('TEACHER_CHECKPOINT="$BASELINE_CHECKPOINT"', script)
        self.assertIn('TEACHER_FIRST_TOKEN_DEPTH_KL_WEIGHT=5.0', script)
        self.assertIn('ALL_DEPTH_CE_WEIGHT=0.10', script)
        self.assertIn('CHOICE_MARGIN_WEIGHT=0.25', script)
        self.assertIn('STEPS="${CANDIDATE_STEPS:-40}"', script)

    def test_runner_prints_full_sweep_command_with_rebuilt_pair(self) -> None:
        script = Path("scripts/206_run_metacog_pair_rebuild.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("scripts/204_run_metacog_fusion_conflict_gate_sweep.sh", script)
        self.assertIn("BASELINE_CHECKPOINT=", script)
        self.assertIn("CANDIDATE_CHECKPOINT=", script)
        self.assertIn("CONFIG=$CONFIG", script)


if __name__ == "__main__":
    unittest.main()
