from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "580_build_stage101n_nested_source_anchor_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101n_nested_source_anchor_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101NNestedSourceAnchorProbeTests(unittest.TestCase):
    def test_microburst_groups_source_rows_then_anchor_lock_rows(self) -> None:
        module = load_module()
        source_rows = [
            {"id": "s_a_0", "source_concept": "a", "source_template": "context_first"},
            {"id": "s_a_1", "source_concept": "a", "source_template": "claim_first"},
            {"id": "s_b_0", "source_concept": "b", "source_template": "context_first"},
        ]
        anchors = [{"id": "gd_lite_0"}, {"id": "stage101b_0"}]

        rows = module.build_microburst_curriculum(
            source_rows=source_rows,
            anchors=anchors,
            source_replay_factor=1,
            anchor_lock_replay_factor=1,
        )

        self.assertEqual([row["nested_phase"] for row in rows[:4]], ["source_microburst"] * 2 + ["anchor_lock"] * 2)
        self.assertEqual({row["source_concept"] for row in rows[:2]}, {"a"})
        self.assertTrue(all(row["id"].startswith(("s_a_", "gd_lite_", "stage101b_")) for row in rows[:4]))
        self.assertEqual(rows[4]["source_concept"], "b")

    def test_build_writes_nested_curriculum_and_eval(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.jsonl"
            extra_anchor = root / "extra_anchor.jsonl"
            base_eval = root / "base_eval.jsonl"
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            anchor.write_text(
                json.dumps(
                    {
                        "id": "gd_lite_anchor",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " True",
                        "parrot_answer": " False",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            extra_anchor.write_text(
                json.dumps(
                    {
                        "id": "stage101b_anchor",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " False",
                        "parrot_answer": " True",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            base_eval.write_text(
                json.dumps(
                    {
                        "id": "base_eval",
                        "task": "source_grounded_truthy_answer_icl",
                        "prompt": "Context: x\nQ: y\nA:",
                        "intelligence_answer": " True",
                        "parrot_answer": " False",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--anchor-jsonl",
                    str(anchor),
                    "--extra-anchor-jsonl",
                    str(extra_anchor),
                    "--base-eval-jsonl",
                    str(base_eval),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--source-replay-factor",
                    "1",
                    "--anchor-lock-replay-factor",
                    "1",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        self.assertGreater(report["source_microburst_rows"], 0)
        self.assertGreater(report["anchor_lock_rows"], 0)
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertIn("source_microburst", {row.get("nested_phase") for row in train_rows})
        self.assertIn("anchor_lock", {row.get("nested_phase") for row in train_rows})


if __name__ == "__main__":
    unittest.main()
