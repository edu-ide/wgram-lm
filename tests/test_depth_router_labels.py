import json
import tempfile
import unittest
from pathlib import Path

from scripts import depth_router_labels


class DepthRouterLabelTests(unittest.TestCase):
    def test_builds_shallowest_correct_depth_label(self):
        records = [
            {
                "id": "case-1",
                "mode": "donor_only_no_evidence",
                "hit": False,
                "prompt": "Question?",
                "answer_aliases": ["A"],
                "choices": ["A", "B"],
            },
            {"id": "case-1", "mode": "qtrm_core_off_no_evidence", "hit": False},
            {"id": "case-1", "mode": "qtrm_core_steps_1_no_evidence", "hit": False},
            {"id": "case-1", "mode": "qtrm_core_steps_4_no_evidence", "hit": True},
            {"id": "case-1", "mode": "qtrm_core_steps_8_no_evidence", "hit": True},
            {"id": "case-1", "mode": "qtrm_core_steps_8_delta_off_no_evidence", "hit": False},
        ]

        labels, summary = depth_router_labels.build_depth_labels(records)

        self.assertEqual(summary["cases"], 1)
        self.assertEqual(summary["oracle_hits"], 1)
        self.assertEqual(labels[0]["target_route"], "core_steps_4")
        self.assertEqual(labels[0]["best_depth"], 4)
        self.assertTrue(labels[0]["causal_core_gain"])
        self.assertEqual(labels[0]["depth_hits"], {"1": False, "4": True, "8": True})

    def test_prefers_donor_when_donor_is_already_correct(self):
        labels, summary = depth_router_labels.build_depth_labels(
            [
                {"id": "case-1", "mode": "donor_only_no_evidence", "hit": True},
                {"id": "case-1", "mode": "qtrm_core_steps_1_no_evidence", "hit": True},
            ]
        )

        self.assertEqual(summary["donor_hits"], 1)
        self.assertEqual(labels[0]["target_route"], "donor")
        self.assertEqual(labels[0]["best_depth"], 1)
        self.assertFalse(labels[0]["causal_core_gain"])

    def test_marks_unknown_when_no_route_is_correct(self):
        labels, summary = depth_router_labels.build_depth_labels(
            [
                {"id": "case-1", "mode": "donor_only_no_evidence", "hit": False},
                {"id": "case-1", "mode": "qtrm_core_off_no_evidence", "hit": False},
                {"id": "case-1", "mode": "qtrm_core_steps_1_no_evidence", "hit": False},
            ]
        )

        self.assertEqual(summary["unknown_routes"], 1)
        self.assertEqual(labels[0]["target_route"], "unknown")
        self.assertIsNone(labels[0]["best_depth"])
        self.assertFalse(labels[0]["oracle_hit"])

    def test_builds_controller_signal_training_rows(self):
        rows = depth_router_labels.build_controller_signal_rows(
            [
                {
                    "id": "case-1",
                    "prompt": "Question?",
                    "target_route": "core_steps_4",
                    "answer_aliases": ["A"],
                }
            ]
        )

        self.assertEqual(rows[0]["prompt"], "Question?")
        self.assertEqual(rows[0]["controller_signal_route"], "core_steps_4")
        self.assertEqual(rows[0]["controller_signal"], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(rows[0]["controller_signal_route_order"], list(depth_router_labels.ROUTE_ORDER))

    def test_cli_writes_jsonl_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "eval.jsonl"
            out_path = Path(tmp) / "labels.jsonl"
            signal_path = Path(tmp) / "signals.jsonl"
            in_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"id": "case-1", "mode": "donor_only_no_evidence", "hit": False},
                        {
                            "id": "case-1",
                            "mode": "qtrm_core_steps_2_no_evidence",
                            "hit": True,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            depth_router_labels.main(
                [
                    "--eval-jsonl",
                    str(in_path),
                    "--out",
                    str(out_path),
                    "--controller-signal-out",
                    str(signal_path),
                ]
            )

            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["target_route"], "core_steps_2")
            signal_rows = [
                json.loads(line)
                for line in signal_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(signal_rows[0]["controller_signal"], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
