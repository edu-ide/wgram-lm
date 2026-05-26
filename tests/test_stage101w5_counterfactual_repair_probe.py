from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "592_build_stage101w5_counterfactual_repair_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w5_counterfactual_repair_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W5CounterfactualRepairProbeTests(unittest.TestCase):
    def test_every_case_teaches_answerability_then_minimal_repair(self) -> None:
        module = load_module()
        rows = module.counterfactual_repair_rows()

        by_case: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            self.assertTrue(row["stage101w5_counterfactual_repair_required"])
            by_case[str(row["repair_case_id"])].add(str(row["stage101w5_chain_step"]))

        required = {"answerable_now", "minimal_repair", "repaired_answer_permission"}
        self.assertTrue(by_case)
        self.assertTrue(all(required.issubset(steps) for steps in by_case.values()))

    def test_minimal_repair_rows_cover_all_interventions(self) -> None:
        module = load_module()
        expected = {
            " answer_now",
            " verify_source",
            " add_relevant_evidence",
            " add_missing_detail",
            " resolve_conflict",
        }
        for rows in [module.counterfactual_repair_rows(), module.counterfactual_repair_heldout_rows()]:
            repair_rows = [row for row in rows if row["stage101w5_chain_step"] == "minimal_repair"]
            counts = Counter(row["intelligence_answer"] for row in repair_rows)
            self.assertEqual(expected, set(counts))
            self.assertGreaterEqual(min(counts.values()), 1)
            for row in repair_rows:
                prompt = str(row["prompt"])
                self.assertIn("Claim:", prompt)
                self.assertIn("Source:", prompt)
                self.assertIn("Evidence:", prompt)
                self.assertIn("What minimal repair makes the answer valid?", prompt)

    def test_answerable_cases_use_answer_now_and_blocked_cases_choose_real_repair(self) -> None:
        module = load_module()
        for rows in [module.counterfactual_repair_rows(), module.counterfactual_repair_heldout_rows()]:
            answerable_rows = {
                row["repair_case_id"]: row
                for row in rows
                if row["stage101w5_chain_step"] == "answerable_now"
            }
            repair_rows = [
                row for row in rows if row["stage101w5_chain_step"] == "minimal_repair"
            ]
            for row in repair_rows:
                answerable = answerable_rows[row["repair_case_id"]]["intelligence_answer"]
                if answerable == " yes":
                    self.assertEqual(row["intelligence_answer"], " answer_now", row["id"])
                else:
                    self.assertEqual(answerable, " no", row["id"])
                    self.assertNotEqual(row["intelligence_answer"], " answer_now", row["id"])

    def test_repaired_permission_is_yes_after_minimal_intervention(self) -> None:
        module = load_module()
        for rows in [module.counterfactual_repair_rows(), module.counterfactual_repair_heldout_rows()]:
            permission_rows = [
                row for row in rows if row["stage101w5_chain_step"] == "repaired_answer_permission"
            ]
            self.assertTrue(permission_rows)
            for row in permission_rows:
                self.assertEqual(row["intelligence_answer"], " yes", row["id"])
                prompt = str(row["prompt"])
                self.assertIn("minimal_repair=", prompt)
                self.assertIn("After that repair, can the model answer?", prompt)

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
        self.assertEqual(
            ["answerable_now", "minimal_repair", "repaired_answer_permission"],
            report["counterfactual_repair_contract"]["chain_steps"],
        )

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.counterfactual_repair_rows() + module.counterfactual_repair_heldout_rows()
        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
