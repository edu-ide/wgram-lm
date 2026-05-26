from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "598_build_stage101w9_paired_feature_difference_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w9_paired_feature_difference_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W9PairedFeatureDifferenceProbeTests(unittest.TestCase):
    def test_pairs_change_exactly_one_non_permission_feature(self) -> None:
        module = load_module()
        for pair in module.paired_feature_difference_rows():
            self.assertTrue(pair["stage101w9_paired_feature_difference_required"])
            changed = [
                name
                for name in module.CAUSAL_FEATURE_NAMES
                if pair["world_a_targets"][name] != pair["world_b_targets"][name]
            ]
            self.assertEqual([pair["pair_feature"]], changed, pair["id"])
            self.assertNotEqual(
                pair["world_a_targets"]["answer_permission"],
                pair["world_b_targets"]["answer_permission"],
                pair["id"],
            )

    def test_each_split_covers_axes_and_balances_positive_position(self) -> None:
        module = load_module()
        expected = set(module.CAUSAL_FEATURE_NAMES)
        for rows in [module.paired_feature_difference_rows(), module.paired_feature_difference_heldout_rows()]:
            self.assertGreaterEqual(len(rows), 32)
            axes = Counter(row["pair_feature"] for row in rows)
            self.assertEqual(expected, set(axes))
            self.assertGreaterEqual(min(axes.values()), 8)
            positions = Counter(row["positive_world"] for row in rows)
            self.assertGreaterEqual(positions["A"], 12)
            self.assertGreaterEqual(positions["B"], 12)

    def test_prompts_hide_feature_labels_and_keep_same_claim(self) -> None:
        module = load_module()
        forbidden = [
            "source_reliability=",
            "evidence_relevance=",
            "detail_sufficiency=",
            "conflict_status=",
        ]
        for row in module.paired_feature_difference_rows() + module.paired_feature_difference_heldout_rows():
            self.assertEqual(row["claim_a"], row["claim_b"])
            for key in ["world_a_prompt", "world_b_prompt"]:
                prompt = str(row[key])
                self.assertIn("Claim:", prompt)
                self.assertIn("World:", prompt)
                self.assertIn("Can answer now?", prompt)
                for token in forbidden:
                    self.assertNotIn(token, prompt)

    def test_build_writes_train_heldout_and_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            report_out = root / "report.json"
            args = module.build_arg_parser().parse_args(
                [
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(heldout),
                    "--report-out",
                    str(report_out),
                ]
            )
            report = module.build(args)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            heldout_rows = [json.loads(line) for line in heldout.read_text(encoding="utf-8").splitlines()]
            saved_report = json.loads(report_out.read_text(encoding="utf-8"))

        self.assertEqual(report, saved_report)
        self.assertEqual(report["train_rows"], len(train_rows))
        self.assertEqual(report["eval_rows"], len(heldout_rows))
        self.assertEqual(module.CAUSAL_FEATURE_NAMES, report["paired_feature_difference_contract"]["causal_feature_names"])

    def test_pair_prompts_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.paired_feature_difference_rows() + module.paired_feature_difference_heldout_rows()
        for row in rows:
            for key in ["world_a_prompt", "world_b_prompt"]:
                self.assertLessEqual(len(str(row[key]).encode("utf-8")), 384, row["id"])


if __name__ == "__main__":
    unittest.main()
