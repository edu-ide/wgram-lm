from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "573_build_stage101e_world_truth_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101e_world_truth_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101EWorldTruthProbeTests(unittest.TestCase):
    def test_build_adds_physics_and_equal_weight_traps(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_train = root / "base_train.jsonl"
            base_eval = root / "base_eval.jsonl"
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            base_train.write_text(
                json.dumps(
                    {
                        "id": "base_train",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " True",
                        "parrot_answer": " False",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            base_eval.write_text(
                json.dumps(
                    {
                        "id": "base_eval",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " False",
                        "parrot_answer": " True",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--base-train-jsonl",
                    str(base_train),
                    "--base-eval-jsonl",
                    str(base_eval),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["base_train_rows"], 1)
        self.assertEqual(report["base_eval_rows"], 1)
        self.assertGreaterEqual(report["added_train_rows"], 10)
        self.assertGreaterEqual(report["added_eval_rows"], 4)
        stage101e_rows = [row for row in train_rows + heldout_rows if row["id"].startswith("stage101e_")]
        text = "\n".join(row["prompt"] + " " + row["plain_language_axis"] for row in stage101e_rows)
        self.assertIn("Sound", text)
        self.assertIn("kilogram", text)
        self.assertIn("pound", text)
        self.assertTrue(all(row["task"] == "truthy_answer_icl" for row in train_rows if row["id"].startswith("stage101e_")))

    def test_hard_plus_anchors_replays_hard_rows(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_train = root / "base_train.jsonl"
            base_eval = root / "base_eval.jsonl"
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            base_train.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "gd_lite_anchor",
                                "task": "truthy_answer_icl",
                                "prompt": "Q:",
                                "intelligence_answer": " True",
                                "parrot_answer": " False",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "unrelated_base",
                                "task": "truthy_answer_icl",
                                "prompt": "Q:",
                                "intelligence_answer": " True",
                                "parrot_answer": " False",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            base_eval.write_text(
                json.dumps(
                    {
                        "id": "base_eval",
                        "task": "truthy_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " False",
                        "parrot_answer": " True",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--base-train-jsonl",
                    str(base_train),
                    "--base-eval-jsonl",
                    str(base_eval),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--train-mode",
                    "hard-plus-anchors",
                    "--hard-replay-factor",
                    "2",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]

        row_ids = {row["id"] for row in train_rows}
        self.assertEqual(report["train_mode"], "hard-plus-anchors")
        self.assertIn("gd_lite_anchor", row_ids)
        self.assertNotIn("unrelated_base", row_ids)
        self.assertTrue(any(row_id.endswith("_replay01") for row_id in row_ids))


if __name__ == "__main__":
    unittest.main()
