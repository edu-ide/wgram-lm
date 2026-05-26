from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "571_build_stage101b_solution_attractor_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101b_solution_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101BSolutionAttractorProbeTests(unittest.TestCase):
    def test_build_adds_successive_and_truthy_counterexamples(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            base.write_text(
                json.dumps(
                    {
                        "id": "base",
                        "task": "repetitive_answer_icl",
                        "prompt": "Q:",
                        "intelligence_answer": " A",
                        "parrot_answer": " B",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = module.build_arg_parser().parse_args(
                [
                    "--base-probe-jsonl",
                    str(base),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        train_tasks = [row["task"] for row in train_rows]
        heldout_tasks = [row["task"] for row in heldout_rows]
        self.assertEqual(report["base_rows"], 1)
        self.assertIn("successive_answer_icl", train_tasks)
        self.assertIn("truthy_answer_icl", train_tasks)
        self.assertIn("successive_answer_icl", heldout_tasks)
        self.assertIn("truthy_answer_icl", heldout_tasks)
        self.assertGreater(report["train_rows"], report["eval_rows"])


if __name__ == "__main__":
    unittest.main()
