from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "589_build_stage101w_curiosity_brake_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w_curiosity_brake_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101WCuriosityBrakeProbeTests(unittest.TestCase):
    def test_rows_split_answer_permission_from_missing_material(self) -> None:
        module = load_module()
        rows = module.curiosity_brake_rows()

        self.assertTrue(rows)
        self.assertTrue(all(row["stage101w_curiosity_brake_required"] for row in rows))

        by_case: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by_case[str(row["curiosity_brake_case_id"])].add(str(row["stage101w_chain_step"]))

        self.assertTrue(by_case)
        self.assertTrue(all({"answer_permission", "missing_material"}.issubset(steps) for steps in by_case.values()))

    def test_balanced_answer_permission_has_yes_and_no_counterfactuals(self) -> None:
        module = load_module()
        permission_rows = [
            row
            for row in module.curiosity_brake_rows()
            if row["stage101w_chain_step"] == "answer_permission"
        ]

        answers = {row["intelligence_answer"] for row in permission_rows}
        self.assertEqual({" yes", " no"}, answers)
        yes_count = sum(1 for row in permission_rows if row["intelligence_answer"] == " yes")
        no_count = sum(1 for row in permission_rows if row["intelligence_answer"] == " no")
        self.assertEqual(yes_count, no_count)
        self.assertGreaterEqual(yes_count, 4)

        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in permission_rows:
            pair_id = str(row.get("curiosity_brake_pair_id", ""))
            if pair_id.startswith("source_quality"):
                by_pair[pair_id].append(row)
        self.assertTrue(by_pair)
        pair_rows = next(iter(by_pair.values()))
        self.assertEqual({" yes", " no"}, {row["intelligence_answer"] for row in pair_rows})
        self.assertEqual({row["source_claim"] for row in pair_rows}, {pair_rows[0]["source_claim"]})
        self.assertEqual({row["evidence_payload"] for row in pair_rows}, {pair_rows[0]["evidence_payload"]})

    def test_missing_material_uses_short_atomic_labels(self) -> None:
        module = load_module()
        rows = [
            row
            for row in module.curiosity_brake_rows()
            if row["stage101w_chain_step"] == "missing_material"
        ]

        answers = {row["intelligence_answer"] for row in rows}
        self.assertIn(" none", answers)
        self.assertIn(" source", answers)
        self.assertIn(" relevance", answers)
        self.assertIn(" detail", answers)
        self.assertIn(" conflict", answers)
        self.assertTrue(all(len(row["intelligence_answer"].strip().split()) == 1 for row in rows))

    def test_train_and_heldout_keep_permission_and_missing_material_balanced(self) -> None:
        module = load_module()
        for rows in [module.curiosity_brake_rows(), module.curiosity_brake_heldout_rows()]:
            permission_rows = [
                row for row in rows if row["stage101w_chain_step"] == "answer_permission"
            ]
            permission_counts = defaultdict(int)
            for row in permission_rows:
                permission_counts[str(row["intelligence_answer"])] += 1
            self.assertEqual(permission_counts[" yes"], permission_counts[" no"])

            missing_rows = [
                row for row in rows if row["stage101w_chain_step"] == "missing_material"
            ]
            missing_counts = defaultdict(int)
            for row in missing_rows:
                missing_counts[str(row["intelligence_answer"])] += 1
            self.assertEqual(
                {
                    " none",
                    " source",
                    " relevance",
                    " detail",
                    " conflict",
                },
                set(missing_counts),
            )
            self.assertEqual(max(missing_counts.values()), min(missing_counts.values()))

    def test_build_writes_balanced_train_heldout_and_report(self) -> None:
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
        self.assertGreaterEqual(report["curiosity_brake_contract"]["permission_yes_rows"], 2)
        self.assertGreaterEqual(report["curiosity_brake_contract"]["permission_no_rows"], 2)

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.curiosity_brake_rows() + module.curiosity_brake_heldout_rows()

        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
