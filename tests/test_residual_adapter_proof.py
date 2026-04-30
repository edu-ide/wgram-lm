import tempfile
import unittest
from pathlib import Path


class ResidualAdapterProofTests(unittest.TestCase):
    def test_build_proof_summary_compares_donor_and_residual_modes(self):
        from qtrm_mm.eval.residual_adapter_proof import build_proof_summary

        records = [
            {
                "id": "case-a",
                "mode": "donor_only_with_evidence",
                "task_family": "abstention",
                "category": "negative_missing",
                "hit": False,
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "task_family": "abstention",
                "category": "negative_missing",
                "hit": True,
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "id": "case-b",
                "mode": "donor_only_with_evidence",
                "task_family": "conflict",
                "category": "temporal_conflict",
                "hit": True,
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {
                "id": "case-b",
                "mode": "qtrm_residual_with_evidence",
                "task_family": "conflict",
                "category": "temporal_conflict",
                "hit": True,
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
        ]

        proof = build_proof_summary(
            [
                {
                    "name": "synthetic gate",
                    "path": "runs/eval/synthetic.jsonl",
                    "records": records,
                }
            ]
        )

        gate = proof["evals"][0]
        self.assertEqual(gate["donor"]["hits"], 1)
        self.assertEqual(gate["residual"]["hits"], 2)
        self.assertEqual(gate["delta_hits"], 1)
        self.assertAlmostEqual(gate["delta_accuracy"], 0.5)
        self.assertEqual(gate["by_task_family"]["abstention"]["delta_hits"], 1)
        self.assertEqual(proof["aggregate"]["donor"]["hits"], 1)
        self.assertEqual(proof["aggregate"]["residual"]["hits"], 2)

    def test_load_eval_records_ignores_summary_rows(self):
        from qtrm_mm.eval.residual_adapter_proof import load_eval_records

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"id":"case-a","mode":"donor_only_with_evidence","hit":false}',
                        '{"summary":{"overall":{"count":1}}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_eval_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "case-a")

    def test_render_markdown_includes_delta_table_and_limitations(self):
        from qtrm_mm.eval.residual_adapter_proof import build_proof_summary, render_markdown

        proof = build_proof_summary(
            [
                {
                    "name": "hard probe",
                    "path": "runs/eval/hard.jsonl",
                    "records": [
                        {
                            "id": "case-a",
                            "mode": "donor_only_with_evidence",
                            "task_family": "abstention",
                            "hit": False,
                            "retrieved_target": True,
                            "all_targets_retrieved": True,
                            "target_recall": 1.0,
                        },
                        {
                            "id": "case-a",
                            "mode": "qtrm_residual_with_evidence",
                            "task_family": "abstention",
                            "hit": True,
                            "retrieved_target": True,
                            "all_targets_retrieved": True,
                            "target_recall": 1.0,
                        },
                    ],
                }
            ]
        )

        markdown = render_markdown(proof)

        self.assertIn("# Residual Adapter Proof", markdown)
        self.assertIn("| hard probe | runs/eval/hard.jsonl | 0/1 | 1/1 | +1 | +1.000 |", markdown)
        self.assertIn("This is not a donor-free standalone-LM claim.", markdown)

    def test_proof_script_defaults_include_expanded_heldout_gate(self):
        script = Path("scripts/109_build_residual_adapter_proof.py").read_text(encoding="utf-8")

        self.assertIn("expanded held-out memory probe", script)
        self.assertIn(
            "memory_reasoning_heldout_expanded_qwen3_rerank_32tok_synth_generalization_s050.jsonl",
            script,
        )


if __name__ == "__main__":
    unittest.main()
