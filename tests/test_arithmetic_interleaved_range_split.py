from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "307_build_arithmetic_interleaved_range_split.py"
    )
    spec = importlib.util.spec_from_file_location("arith_interleaved_split", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArithmeticInterleavedRangeSplitTests(unittest.TestCase):
    def test_builds_disjoint_interleaved_arithmetic_split(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.jsonl"
            eval_path = Path(tmp) / "eval.jsonl"
            summary = module.build_interleaved_split(
                train_out=train,
                eval_out=eval_path,
                start_index=18,
                train_cases=4,
                eval_cases=4,
            )
            train_rows = [json.loads(line) for line in train.read_text().splitlines()]
            eval_rows = [json.loads(line) for line in eval_path.read_text().splitlines()]
            self.assertEqual(summary["train_cases"], 4)
            self.assertEqual(summary["eval_cases"], 4)
            self.assertEqual({row["task_family"] for row in train_rows}, {"arithmetic_chain"})
            self.assertEqual({row["task_family"] for row in eval_rows}, {"arithmetic_chain"})
            self.assertFalse({row["id"] for row in train_rows} & {row["id"] for row in eval_rows})


if __name__ == "__main__":
    unittest.main()
