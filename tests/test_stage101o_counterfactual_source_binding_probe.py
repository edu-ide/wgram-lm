from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "581_build_stage101o_counterfactual_source_binding_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101o_counterfactual_source_binding_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101OCounterfactualSourceBindingProbeTests(unittest.TestCase):
    def test_counterfactual_rows_keep_claim_fixed_and_flip_source_answer(self) -> None:
        module = load_module()
        rows = module.counterfactual_source_rows()

        by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_pair[str(row["counterfactual_pair_id"])].append(row)

        self.assertGreaterEqual(len(by_pair), 4)
        for pair_rows in by_pair.values():
            claims = {row["source_claim"] for row in pair_rows}
            answers = {row["intelligence_answer"] for row in pair_rows}
            source_truths = {row["source_truth_value"] for row in pair_rows}
            self.assertEqual(len(claims), 1)
            self.assertEqual(answers, {" True", " False"})
            self.assertEqual(source_truths, {"True", "False"})
            self.assertTrue(all(row["source_binding_required"] for row in pair_rows))

    def test_build_writes_anchors_replayed_binding_rows_and_heldout(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.jsonl"
            extra_anchor = root / "extra_anchor.jsonl"
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
            args = module.build_arg_parser().parse_args(
                [
                    "--anchor-jsonl",
                    str(anchor),
                    "--extra-anchor-jsonl",
                    str(extra_anchor),
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--source-replay-factor",
                    "2",
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]

        row_ids = {row["id"] for row in train_rows}
        self.assertIn("gd_lite_anchor", row_ids)
        self.assertIn("stage101b_anchor", row_ids)
        self.assertTrue(any(row["id"].endswith("_replay01") for row in train_rows))
        self.assertTrue(any(row.get("source_binding_required") for row in train_rows))
        self.assertTrue(all(row.get("source_binding_required") for row in heldout_rows))
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertGreater(report["source_binding_rows"], 20)


if __name__ == "__main__":
    unittest.main()
