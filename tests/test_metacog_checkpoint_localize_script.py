from pathlib import Path
import unittest


class MetacogCheckpointLocalizeScriptTests(unittest.TestCase):
    def test_localize_script_copies_checkpoints_to_healthy_disk(self) -> None:
        script = Path("scripts/205_localize_metacog_checkpoints.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('PYTHON="${PYTHON:-$PWD/.venv/bin/python}"', script)
        self.assertIn('"$PYTHON" - "$path"', script)
        self.assertIn('LOCAL_CKPT_ROOT="${LOCAL_CKPT_ROOT:-/mnt/nvme1n1p2/qtrm-local-checkpoints/metacog_fusion_sweep}"', script)
        self.assertIn("shutil.copy2", script)
        self.assertIn("hashlib.sha256", script)
        self.assertIn("open(src, 'rb').read(1)", script)
        self.assertIn("preflight_write_test", script)

    def test_localize_script_prints_exports_for_sweep_runner(self) -> None:
        script = Path("scripts/205_localize_metacog_checkpoints.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("BASELINE_CHECKPOINT=", script)
        self.assertIn("CANDIDATE_CHECKPOINT=", script)
        self.assertIn("scripts/204_run_metacog_fusion_conflict_gate_sweep.sh", script)


if __name__ == "__main__":
    unittest.main()
