import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "308_run_small_general_reasoning_gate.py"
    )
    spec = importlib.util.spec_from_file_location("small_general_reasoning_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmallGeneralReasoningGateTests(unittest.TestCase):
    def test_interleave_groups_preserves_source_mix(self):
        module = _load_module()

        rows = module.interleave_groups(
            [
                [{"id": "a1"}, {"id": "a2"}],
                [{"id": "b1"}],
            ]
        )

        self.assertEqual([row["id"] for row in rows], ["a1", "b1", "a2"])

    def test_build_mixed_gate_cases_summarizes_families(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.jsonl"
            p2 = Path(tmp) / "b.jsonl"
            p1.write_text(
                json.dumps(
                    {
                        "id": "a",
                        "prompt": "p",
                        "answer_aliases": ["1"],
                        "task_family": "arith",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            p2.write_text(
                json.dumps(
                    {
                        "id": "b",
                        "prompt": "p",
                        "answer": "2",
                        "task_family": "list",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows, summary = module.build_mixed_gate_cases(
                sources=[p1, p2],
                max_per_source=1,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["families"], ["arith", "list"])
        self.assertEqual(rows[1]["answer_aliases"], ["2"])

    def test_build_gate_report_accepts_full_beating_donor_and_ablations(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            report = module.build_gate_report(
                soft_prefix_report={
                    "adapter": {"state_dim": 3},
                    "generation": {
                        "soft_full_no_evidence": {"accuracy": 0.75},
                        "donor_only_no_evidence": {"accuracy": 0.25},
                        "soft_core_off_no_evidence": {"accuracy": 0.25},
                        "soft_state_off_no_evidence": {"accuracy": 0.5},
                    },
                },
                generation_rows=[
                    {"mode": "soft_full_no_evidence", "task_family": "a", "hit": True},
                    {"mode": "soft_full_no_evidence", "task_family": "b", "hit": True},
                ],
                train_summary={"families": ["a", "b"]},
                eval_summary={"families": ["a", "b"]},
                out_dir=tmp,
                min_full_accuracy=0.5,
                min_donor_margin=0.0,
                min_core_off_margin=0.0,
                min_state_off_margin=0.0,
                min_eval_families=2,
                require_family_full_hit=True,
            )

        self.assertTrue(report["accepted"])
        self.assertEqual(
            report["decision"],
            "accepted_l3_candidate_small_general_reasoning",
        )

    def test_build_soft_prefix_command_contains_state_key(self):
        module = _load_module()

        args = argparse.Namespace(
            config="cfg.yaml",
            checkpoint="ckpt.pt",
            out_dir="out",
            max_train_cases=4,
            max_eval_cases=2,
            max_length=128,
            max_target_tokens=4,
            max_new_tokens=4,
            core_steps=8,
            prefix_tokens=2,
            rank=8,
            scale=1.0,
            soft_prefix_steps=3,
            lr=1e-3,
            scheduled_sampling_prob=0.0,
            scheduled_sampling_warmup_steps=0,
            state_logits_key="typed_logits",
            state_feature_mode="softmax",
            device="cpu",
            log_every=1,
            append_eos_target=True,
            suppress_visible_reasoning_tokens=False,
        )

        command = module.build_soft_prefix_command(
            args,
            Path("train.jsonl"),
            Path("eval.jsonl"),
        )

        self.assertIn("scripts/304_train_core_soft_prefix_donor.py", command)
        self.assertIn("--state-logits-key", command)
        self.assertIn("typed_logits", command)
        self.assertIn("--append-eos-target", command)


if __name__ == "__main__":
    unittest.main()
