from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "574_build_stage101f_source_grounded_truth_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101f_source_grounded_truth_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101FSourceGroundedTruthProbeTests(unittest.TestCase):
    def test_build_creates_context_grounded_truth_rows(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            args = module.build_arg_parser().parse_args(
                ["--train-out", str(train), "--eval-out", str(heldout)]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["train_rows"], 6)
        self.assertEqual(report["eval_rows"], 4)
        self.assertTrue(all(row["prompt"].startswith("Context:") for row in train_rows + heldout_rows))
        joined = "\n".join(row["prompt"] for row in train_rows + heldout_rows)
        self.assertIn("Sound", joined)
        self.assertIn("kilogram", joined)
        self.assertIn("pound", joined)


if __name__ == "__main__":
    unittest.main()
