from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "578_build_stage101l_balanced_replay_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101l_balanced_replay_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101LBalancedReplayProbeTests(unittest.TestCase):
    def test_interleave_puts_anchors_before_each_source_row(self) -> None:
        module = load_module()
        source = [{"id": "source0"}, {"id": "source1"}]
        anchors = [{"id": "anchor0"}, {"id": "anchor1"}, {"id": "anchor2"}, {"id": "anchor3"}]

        rows = module.interleave_source_and_anchors(
            source_rows=source,
            anchor_rows_replayed=anchors,
            anchors_per_source=2,
        )

        self.assertEqual(
            [row["id"] for row in rows[:6]],
            ["anchor0", "anchor1", "source0", "anchor2", "anchor3", "source1"],
        )

    def test_build_writes_anchor_heavy_balanced_curriculum(self) -> None:
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
                    "--anchor-replay-factor",
                    "3",
                    "--source-replay-factor",
                    "1",
                    "--anchors-per-source",
                    "2",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]

        anchor_count = sum(1 for row in train_rows if "anchor_replay" in row["id"])
        source_count = sum(1 for row in train_rows if "source_replay" in row["id"])
        self.assertGreater(anchor_count, 0)
        self.assertGreater(source_count, 0)
        self.assertGreaterEqual(anchor_count, source_count // 4)
        self.assertEqual(report["train_rows"], len(train_rows))


if __name__ == "__main__":
    unittest.main()
