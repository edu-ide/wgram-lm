import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest


def load_module():
    path = Path("scripts/330_run_mixed_noncopy_lm_gate.py")
    spec = importlib.util.spec_from_file_location("mixed_noncopy_lm_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MixedNoncopyLmGateTests(unittest.TestCase):
    def test_summarize_generation_counts_hits_by_mode(self):
        module = load_module()
        rows = [
            {"mode": "donor_only_no_evidence", "hit": False},
            {"mode": "qtrm_core_off_no_evidence", "hit": False},
            {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
            {"mode": "qtrm_core_steps_8_no_evidence", "normalized_exact": True},
        ]

        summary = module.summarize_generation(rows)

        self.assertEqual(summary["qtrm_core_steps_8_no_evidence"]["hits"], 2)
        self.assertEqual(summary["qtrm_core_steps_8_no_evidence"]["total"], 2)
        self.assertEqual(
            summary["qtrm_core_steps_8_no_evidence"]["accuracy"],
            1.0,
        )

    def test_build_report_rejects_zero_hit_full_model(self):
        module = load_module()
        rows = []
        for mode in module.DEFAULT_MODES:
            for index in range(2):
                rows.append({"id": str(index), "mode": mode, "hit": False})

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.10,
            min_donor_margin=0.01,
            min_core_off_margin=0.01,
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(report["decision"], "rejected_noncopy_lm_gate")
        self.assertIn("full_generation_accuracy_below_min", report["reject_reasons"])

    def test_output_is_complete_requires_expected_row_count(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                "\n".join(json.dumps({"mode": "m", "hit": False}) for _ in range(3))
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(module.output_is_complete(path, expected_rows=3))
            self.assertFalse(module.output_is_complete(path, expected_rows=4))

    def test_eval_command_keeps_noncopy_contract_minimal(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "ckpt.pt",
                "--cases",
                "cases.jsonl",
                "--out-dir",
                "out",
            ]
        )

        command = module.eval_command(
            args,
            mode="qtrm_core_steps_8_no_evidence",
            cases_path=Path("chunk.jsonl"),
            out_path=Path("out.jsonl"),
        )

        self.assertIn("--scoring", command)
        self.assertIn("generation", command)
        self.assertNotIn("--token-numeric-source-slots", command)
        self.assertEqual(
            command[command.index("--mode") + 1],
            "qtrm_core_steps_8_no_evidence",
        )

    def test_run_command_creates_log_parent_directories(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exit_code = module.run_command(
                [sys.executable, "-c", "print('ok')"],
                cwd=Path.cwd(),
                env={},
                stdout_path=tmp_path / "logs" / "out.log",
                stderr_path=tmp_path / "logs" / "err.log",
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual((tmp_path / "logs" / "out.log").read_text().strip(), "ok")


if __name__ == "__main__":
    unittest.main()
