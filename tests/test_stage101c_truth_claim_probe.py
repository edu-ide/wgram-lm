from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "572_build_stage101c_truth_claim_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101c_truth_claim_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101CTruthClaimProbeTests(unittest.TestCase):
    def test_build_adds_truth_claim_repair_rows_without_losing_base(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_train = root / "base_train.jsonl"
            base_eval = root / "base_eval.jsonl"
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            base_row = {
                "id": "base",
                "task": "successive_answer_icl",
                "prompt": "Q:",
                "intelligence_answer": " A",
                "parrot_answer": " B",
            }
            base_train.write_text(json.dumps(base_row) + "\n", encoding="utf-8")
            base_eval.write_text(json.dumps(base_row | {"id": "base_eval"}) + "\n", encoding="utf-8")
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
        self.assertGreater(report["added_train_rows"], 0)
        self.assertGreater(report["added_eval_rows"], 0)
        self.assertIn("base", {row["id"] for row in train_rows})
        self.assertIn("base_eval", {row["id"] for row in heldout_rows})
        self.assertTrue(all(row["task"] == "truthy_answer_icl" for row in train_rows if row["id"].startswith("stage101c_")))
        axes = " ".join(row["plain_language_axis"] for row in train_rows if row["id"].startswith("stage101c_"))
        self.assertIn("Counterintuitive true", axes)
        self.assertIn("Popular myth false", axes)


if __name__ == "__main__":
    unittest.main()
