import json
import tempfile
import unittest
from pathlib import Path


class RootArchitectureGateTests(unittest.TestCase):
    def test_rejects_when_ablations_match_successful_baseline(self):
        from qtrm_mm.eval.root_architecture_gate import build_root_architecture_gate

        records = [
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_workspace_off_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_off_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_workspace_memory_off_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_context_off_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
        ]

        gate = build_root_architecture_gate(records)

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("no_critical_causal_drop", gate["failed_checks"])
        self.assertIn(
            "qtrm_workspace_memory_off_with_evidence",
            gate["mode_checks"],
        )
        self.assertEqual(
            gate["mode_checks"]["qtrm_workspace_memory_off_with_evidence"][
                "same_completion_rate"
            ],
            1.0,
        )

    def test_accepts_when_workspace_memory_ablation_drops(self):
        from qtrm_mm.eval.root_architecture_gate import build_root_architecture_gate

        records = [
            {
                "id": "case-a",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_workspace_memory_off_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_context_off_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
        ]

        gate = build_root_architecture_gate(records)

        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["causal_gate_status"], "accepted")
        self.assertIn("critical_causal_drop_present", gate["passed_checks"])
        self.assertEqual(
            gate["mode_checks"]["qtrm_workspace_memory_off_with_evidence"]["hit_drop"],
            1,
        )

    def test_strict_promotion_rejects_when_donor_ties_baseline(self):
        from qtrm_mm.eval.root_architecture_gate import build_root_architecture_gate

        records = [
            {
                "id": "case-a",
                "mode": "donor_only_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-a",
                "mode": "qtrm_answer_residual_governor_off_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
        ]

        gate = build_root_architecture_gate(
            records,
            require_donor_advantage=True,
            require_no_critical_ablation_improvement=True,
        )

        self.assertEqual(gate["causal_gate_status"], "accepted")
        self.assertEqual(gate["status"], "rejected")
        self.assertIn("baseline_does_not_beat_comparison", gate["failed_checks"])
        self.assertIn("donor_only_with_evidence", gate["weak_comparison_modes"])

    def test_strict_promotion_rejects_when_critical_ablation_improves(self):
        from qtrm_mm.eval.root_architecture_gate import build_root_architecture_gate

        records = [
            {
                "id": "case-a",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-b",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-b",
                "mode": "qtrm_residual_with_evidence",
                "hit": False,
                "completion": "wrong",
            },
            {
                "id": "case-a",
                "mode": "qtrm_workspace_memory_off_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-b",
                "mode": "qtrm_workspace_memory_off_with_evidence",
                "hit": False,
                "completion": "wrong",
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_off_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_off_with_evidence",
                "hit": True,
                "completion": "VX-914",
            },
        ]

        gate = build_root_architecture_gate(
            records,
            require_donor_advantage=True,
            require_no_critical_ablation_improvement=True,
        )

        self.assertEqual(gate["causal_gate_status"], "accepted")
        self.assertEqual(gate["status"], "rejected")
        self.assertIn("critical_ablation_beats_baseline", gate["failed_checks"])
        self.assertIn("qtrm_core_off_with_evidence", gate["improving_critical_modes"])

    def test_strict_promotion_accepts_when_qtrm_beats_donor_and_ablations_drop(self):
        from qtrm_mm.eval.root_architecture_gate import build_root_architecture_gate

        records = [
            {
                "id": "case-a",
                "mode": "donor_only_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-b",
                "mode": "donor_only_with_evidence",
                "hit": True,
                "completion": "opal-river",
            },
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "VX-913",
            },
            {
                "id": "case-b",
                "mode": "qtrm_residual_with_evidence",
                "hit": True,
                "completion": "opal-river",
            },
            {
                "id": "case-a",
                "mode": "qtrm_workspace_memory_off_with_evidence",
                "hit": False,
                "completion": "UNKNOWN",
            },
            {
                "id": "case-b",
                "mode": "qtrm_workspace_memory_off_with_evidence",
                "hit": True,
                "completion": "opal-river",
            },
        ]

        gate = build_root_architecture_gate(
            records,
            require_donor_advantage=True,
            require_no_critical_ablation_improvement=True,
        )

        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["causal_gate_status"], "accepted")
        self.assertEqual(
            gate["comparison_checks"]["donor_only_with_evidence"]["hit_advantage"],
            1,
        )

    def test_default_critical_modes_include_answer_residual_governor(self):
        from qtrm_mm.eval.root_architecture_gate import DEFAULT_CRITICAL_MODES

        self.assertIn(
            "qtrm_answer_residual_governor_off_with_evidence",
            DEFAULT_CRITICAL_MODES,
        )

    def test_rejects_when_baseline_has_no_successes(self):
        from qtrm_mm.eval.root_architecture_gate import build_root_architecture_gate

        gate = build_root_architecture_gate(
            [
                {
                    "id": "case-a",
                    "mode": "qtrm_residual_with_evidence",
                    "hit": False,
                    "completion": "wrong",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_workspace_memory_off_with_evidence",
                    "hit": False,
                    "completion": "wrong",
                },
            ]
        )

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("baseline_has_no_successes", gate["failed_checks"])

    def test_renders_markdown_and_script_writes_outputs(self):
        from qtrm_mm.eval.root_architecture_gate import render_markdown

        script = Path(__file__).resolve().parents[1] / "scripts" / "148_build_root_architecture_gate.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("root_arch_gate_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "eval.jsonl"
            md_path = Path(tmp) / "gate.md"
            json_path = Path(tmp) / "gate.json"
            rows = [
                {
                    "id": "case-a",
                    "mode": "qtrm_residual_with_evidence",
                    "hit": True,
                    "completion": "A",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_workspace_memory_off_with_evidence",
                    "hit": False,
                    "completion": "B",
                },
            ]
            in_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            module.write_gate_report(
                [str(in_path)],
                markdown_out=str(md_path),
                json_out=str(json_path),
                strict_promotion_gate=False,
            )

            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("# Root Architecture Causality Gate", markdown)
            self.assertIn("Strict promotion required: `False`", markdown)
            self.assertIn("qtrm_workspace_memory_off_with_evidence", markdown)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["status"], "accepted")

        self.assertIn("## Verdict", render_markdown({"status": "rejected"}))

    def test_script_can_limit_critical_modes_for_active_path_gate(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "148_build_root_architecture_gate.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("root_arch_gate_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "eval.jsonl"
            md_path = Path(tmp) / "gate.md"
            json_path = Path(tmp) / "gate.json"
            rows = [
                {
                    "id": "case-a",
                    "mode": "qtrm_residual_with_evidence",
                    "hit": True,
                    "completion": "A",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_workspace_off_with_evidence",
                    "hit": False,
                    "completion": "B",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_evidence_bottleneck_off_with_evidence",
                    "hit": True,
                    "completion": "A",
                },
            ]
            in_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            gate = module.write_gate_report(
                [str(in_path)],
                markdown_out=str(md_path),
                json_out=str(json_path),
                strict_promotion_gate=False,
                critical_modes=["qtrm_workspace_off_with_evidence"],
            )

            self.assertEqual(gate["status"], "accepted")
            self.assertIn("qtrm_workspace_off_with_evidence", gate["mode_checks"])
            self.assertNotIn(
                "qtrm_evidence_bottleneck_off_with_evidence",
                gate["mode_checks"],
            )

    def test_script_can_override_baseline_mode_for_pruned_candidate_gate(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "148_build_root_architecture_gate.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("root_arch_gate_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "eval.jsonl"
            md_path = Path(tmp) / "gate.md"
            json_path = Path(tmp) / "gate.json"
            rows = [
                {
                    "id": "case-a",
                    "mode": "donor_only_with_evidence",
                    "hit": False,
                    "completion": "wrong",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_residual_with_evidence",
                    "hit": False,
                    "completion": "wrong",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_core_off_with_evidence",
                    "hit": True,
                    "completion": "A",
                },
                {
                    "id": "case-a",
                    "mode": "qtrm_workspace_off_with_evidence",
                    "hit": False,
                    "completion": "wrong",
                },
            ]
            in_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            gate = module.write_gate_report(
                [str(in_path)],
                markdown_out=str(md_path),
                json_out=str(json_path),
                strict_promotion_gate=True,
                baseline_mode="qtrm_core_off_with_evidence",
                critical_modes=["qtrm_workspace_off_with_evidence"],
            )

            self.assertEqual(gate["status"], "accepted")
            self.assertEqual(gate["baseline_mode"], "qtrm_core_off_with_evidence")
            self.assertEqual(
                gate["comparison_checks"]["donor_only_with_evidence"]["hit_advantage"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
