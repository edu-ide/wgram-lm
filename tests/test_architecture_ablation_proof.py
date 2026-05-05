import tempfile
import unittest
from pathlib import Path


class ArchitectureAblationProofTests(unittest.TestCase):
    def test_build_ablation_summary_measures_drop_from_residual(self):
        from qtrm_mm.eval.architecture_ablation_proof import build_ablation_summary

        records = [
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "task_family": "abstention",
                "hit": True,
                "completion": "UNKNOWN",
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "id": "case-b",
                "mode": "qtrm_residual_with_evidence",
                "task_family": "multi_hop",
                "hit": True,
                "completion": "Rover badge A",
                "retrieved_target": True,
                "all_targets_retrieved": False,
                "target_recall": 0.5,
            },
            {
                "id": "case-a",
                "mode": "qtrm_workspace_off_with_evidence",
                "task_family": "abstention",
                "hit": False,
                "completion": "redacted",
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "id": "case-b",
                "mode": "qtrm_workspace_off_with_evidence",
                "task_family": "multi_hop",
                "hit": True,
                "completion": "Rover badge A",
                "retrieved_target": True,
                "all_targets_retrieved": False,
                "target_recall": 0.5,
            },
            {
                "id": "case-a",
                "mode": "qtrm_core_off_with_evidence",
                "task_family": "abstention",
                "hit": True,
                "completion": "UNKNOWN",
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "id": "case-b",
                "mode": "qtrm_core_off_with_evidence",
                "task_family": "multi_hop",
                "hit": False,
                "completion": "wrong",
                "retrieved_target": True,
                "all_targets_retrieved": False,
                "target_recall": 0.5,
            },
        ]

        proof = build_ablation_summary(
            [{"name": "toy ablation", "path": "runs/eval/toy.jsonl", "records": records}]
        )

        self.assertEqual(proof["modes"]["qtrm_residual_with_evidence"]["hits"], 2)
        self.assertEqual(
            proof["drop_from_residual"]["qtrm_workspace_off_with_evidence"]["hit_drop"],
            1,
        )
        self.assertEqual(
            proof["drop_from_residual"]["qtrm_core_off_with_evidence"]["hit_drop"],
            1,
        )
        self.assertEqual(
            proof["drop_from_residual"]["qtrm_workspace_off_with_evidence"]["same_completion_count"],
            1,
        )
        self.assertEqual(
            proof["drop_from_residual"]["qtrm_core_off_with_evidence"]["same_completion_count"],
            1,
        )
        self.assertEqual(
            proof["by_task_family"]["abstention"]["qtrm_workspace_off_with_evidence"]["hit_drop"],
            1,
        )
        self.assertEqual(
            proof["by_task_family"]["multi_hop"]["qtrm_core_off_with_evidence"]["hit_drop"],
            1,
        )

    def test_build_ablation_summary_loads_multiple_eval_files(self):
        from qtrm_mm.eval.architecture_ablation_proof import build_ablation_summary

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "residual.jsonl"
            second = Path(tmp) / "ablations.jsonl"
            first.write_text(
                '{"id":"case-a","mode":"qtrm_residual_with_evidence","task_family":"conflict","hit":true}\n'
                '{"summary":{"overall":{"count":1}}}\n',
                encoding="utf-8",
            )
            second.write_text(
                '{"id":"case-a","mode":"qtrm_workspace_off_with_evidence","task_family":"conflict","hit":false}\n',
                encoding="utf-8",
            )

            proof = build_ablation_summary(
                [
                    {"name": "residual", "path": str(first)},
                    {"name": "ablations", "path": str(second)},
                ]
            )

        self.assertEqual(proof["modes"]["qtrm_residual_with_evidence"]["hits"], 1)
        self.assertEqual(proof["modes"]["qtrm_workspace_off_with_evidence"]["hits"], 0)

    def test_render_markdown_includes_ablation_tables(self):
        from qtrm_mm.eval.architecture_ablation_proof import (
            DEFAULT_MODES,
            build_ablation_summary,
            render_markdown,
        )

        self.assertIn("qtrm_coda_off_with_evidence", DEFAULT_MODES)
        self.assertIn("qtrm_residual_head_off_with_evidence", DEFAULT_MODES)
        self.assertIn("qtrm_donor_hidden_off_with_evidence", DEFAULT_MODES)
        self.assertIn("qtrm_workspace_only_with_evidence", DEFAULT_MODES)
        self.assertIn("qtrm_workspace_gate_off_with_evidence", DEFAULT_MODES)
        self.assertIn("qtrm_workspace_memory_off_with_evidence", DEFAULT_MODES)
        self.assertIn("qtrm_core_context_off_with_evidence", DEFAULT_MODES)

        proof = build_ablation_summary(
            [
                {
                    "name": "toy",
                    "path": "runs/eval/toy.jsonl",
                    "records": [
                        {
                            "id": "case-a",
                            "mode": "qtrm_residual_with_evidence",
                            "task_family": "abstention",
                            "hit": True,
                        },
                        {
                            "id": "case-a",
                            "mode": "qtrm_workspace_off_with_evidence",
                            "task_family": "abstention",
                            "hit": False,
                        },
                        {
                            "id": "case-a",
                            "mode": "qtrm_coda_off_with_evidence",
                            "task_family": "abstention",
                            "hit": False,
                        },
                    ],
                }
            ]
        )

        markdown = render_markdown(proof)

        self.assertIn("# Expanded Workspace/Core Ablation Proof", markdown)
        self.assertIn("measures whether residual behavior is localized", markdown)
        self.assertIn("qtrm_workspace_off_with_evidence", markdown)
        self.assertIn("| qtrm_workspace_off_with_evidence | 0/1 | +1 |", markdown)
        self.assertIn("## Completion Identity", markdown)

    def test_scripts_define_expanded_ablation_defaults(self):
        runner = Path("scripts/112_run_expanded_workspace_core_ablation.sh").read_text(
            encoding="utf-8"
        )
        strict_runner = Path("scripts/114_run_expanded_strict_causality_ablation.sh").read_text(
            encoding="utf-8"
        )
        builder = Path("scripts/113_build_expanded_ablation_proof.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--mode qtrm_workspace_off_with_evidence", runner)
        self.assertIn("--mode qtrm_core_off_with_evidence", runner)
        self.assertIn("memory_reasoning_heldout_expanded_workspace_core_ablation", runner)
        self.assertIn("--mode qtrm_coda_off_with_evidence", strict_runner)
        self.assertIn("--mode qtrm_residual_head_off_with_evidence", strict_runner)
        self.assertIn("--mode qtrm_donor_hidden_off_with_evidence", strict_runner)
        self.assertIn("--mode qtrm_workspace_only_with_evidence", strict_runner)
        self.assertIn("--mode qtrm_workspace_gate_off_with_evidence", strict_runner)
        self.assertIn("--mode qtrm_core_context_off_with_evidence", strict_runner)
        self.assertIn("memory_reasoning_heldout_expanded_strict_causality_ablation", strict_runner)
        self.assertIn("expanded-workspace-core-ablation.md", builder)
        self.assertIn("memory_reasoning_heldout_expanded_qwen3_rerank_32tok", builder)
        self.assertIn("memory_reasoning_heldout_expanded_strict_causality_ablation", builder)


if __name__ == "__main__":
    unittest.main()
