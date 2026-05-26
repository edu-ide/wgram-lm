from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "577_build_stage101k_polarity_balanced_source_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101k_polarity_balanced_source_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101KPolarityBalancedSourceProbeTests(unittest.TestCase):
    def test_polarity_balanced_rows_cover_both_answers_for_each_concept_and_template(self) -> None:
        module = load_module()
        rows = module.polarity_balanced_rows()

        by_concept: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_concept[str(row["source_concept"])].append(row)

        self.assertGreaterEqual(len(by_concept), 4)
        for concept_rows in by_concept.values():
            answers = {row["intelligence_answer"] for row in concept_rows}
            templates = {row["source_template"] for row in concept_rows}
            self.assertEqual(answers, {" True", " False"})
            self.assertEqual(templates, set(module.TEMPLATES))

    def test_build_writes_anchors_replayed_source_rows_and_heldout(self) -> None:
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
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertGreater(report["source_rows"], 40)


if __name__ == "__main__":
    unittest.main()
