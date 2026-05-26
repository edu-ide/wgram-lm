from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qtrm_mm.eval.depth_breadth_probe import (
    build_depth_breadth_report,
    extract_depth,
)


ROOT = Path(__file__).resolve().parents[1]


class DepthBreadthProbeTests(unittest.TestCase):
    def test_extract_depth_accepts_explicit_depth_and_core_steps_mode(self) -> None:
        self.assertEqual(extract_depth({"depth": 6}), 6)
        self.assertEqual(extract_depth({"mode": "qtrm_core_steps_8_no_evidence"}), 8)

    def test_report_selects_lowest_residual_restart_per_case(self) -> None:
        rows = [
            {
                "case_id": "a",
                "mode": "qtrm_core_steps_1_no_evidence",
                "restart_id": 0,
                "completion": "wrong",
                "hit": False,
                "fixed_point_residual": 0.90,
            },
            {
                "case_id": "b",
                "mode": "qtrm_core_steps_1_no_evidence",
                "restart_id": 0,
                "completion": "wrong",
                "hit": False,
                "fixed_point_residual": 0.95,
            },
            {
                "case_id": "a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "restart_id": 0,
                "completion": "wrong",
                "hit": False,
                "fixed_point_residual": 0.72,
            },
            {
                "case_id": "a",
                "mode": "qtrm_core_steps_4_no_evidence",
                "restart_id": 1,
                "completion": "A",
                "hit": True,
                "fixed_point_residual": 0.20,
            },
            {
                "case_id": "b",
                "mode": "qtrm_core_steps_4_no_evidence",
                "restart_id": 0,
                "completion": "B",
                "hit": True,
                "fixed_point_residual": 0.10,
            },
            {
                "case_id": "b",
                "mode": "qtrm_core_steps_4_no_evidence",
                "restart_id": 1,
                "completion": "wrong",
                "hit": False,
                "fixed_point_residual": 0.80,
            },
        ]

        report = build_depth_breadth_report(rows)

        self.assertEqual(report["depth_ladder"][0]["depth"], 1)
        self.assertEqual(report["depth_ladder"][1]["depth"], 4)
        depth4 = report["breadth_by_depth"][4]
        self.assertAlmostEqual(depth4["trajectory_accuracy"], 0.5)
        self.assertAlmostEqual(depth4["top1_convergence_accuracy"], 1.0)
        self.assertEqual(depth4["top1_case_ids"], ["a", "b"])
        self.assertEqual(report["best_top1_depth"], 4)
        self.assertIn("top1_convergence_beats_trajectory_average", report["passed_checks"])

    def test_cli_reads_jsonl_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_path = Path(tmpdir) / "rows.jsonl"
            out_path = Path(tmpdir) / "report.json"
            rows_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "case-a",
                                "depth": 2,
                                "restart_id": 0,
                                "completion": "yes",
                                "hit": True,
                                "fixed_point_residual": 0.1,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "case-a",
                                "depth": 2,
                                "restart_id": 1,
                                "completion": "no",
                                "hit": False,
                                "fixed_point_residual": 0.5,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "548_build_depth_breadth_probe_report.py"),
                    "--rows",
                    str(rows_path),
                    "--out",
                    str(out_path),
                ],
                check=True,
                cwd=ROOT,
            )

            report = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(report["trajectory_count"], 2)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["best_top1_depth"], 2)


if __name__ == "__main__":
    unittest.main()
