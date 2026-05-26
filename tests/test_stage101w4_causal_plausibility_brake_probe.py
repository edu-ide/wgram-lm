from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "591_build_stage101w4_causal_plausibility_brake_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w4_causal_plausibility_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W4CausalPlausibilityBrakeProbeTests(unittest.TestCase):
    def test_every_case_teaches_real_cards_then_plausibility_then_permission(self) -> None:
        module = load_module()
        rows = module.causal_plausibility_rows()

        by_case: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            self.assertTrue(row["stage101w4_causal_plausibility_required"])
            by_case[str(row["cause_card_case_id"])].add(str(row["stage101w4_chain_step"]))

        required = {
            "source_trust",
            "evidence_relevance",
            "detail_sufficiency",
            "conflict_status",
            "card_plausibility",
            "impossible_card",
            "answer_permission",
        }
        self.assertTrue(by_case)
        self.assertTrue(all(required.issubset(steps) for steps in by_case.values()))

    def test_plausibility_rows_read_evidence_and_proposed_cards(self) -> None:
        module = load_module()
        for rows in [module.causal_plausibility_rows(), module.causal_plausibility_heldout_rows()]:
            plausibility_rows = [
                row for row in rows if row["stage101w4_chain_step"] == "card_plausibility"
            ]
            counts = Counter(row["intelligence_answer"] for row in plausibility_rows)
            self.assertGreaterEqual(counts[" plausible"], 4)
            self.assertGreaterEqual(counts[" impossible"], 4)
            for row in plausibility_rows:
                prompt = str(row["prompt"])
                self.assertIn("Claim:", prompt)
                self.assertIn("Source:", prompt)
                self.assertIn("Evidence:", prompt)
                self.assertIn("Proposed cause cards:", prompt)
                self.assertIn("Are these cause cards plausible?", prompt)

    def test_impossible_card_rows_cover_all_failure_causes(self) -> None:
        module = load_module()
        expected = {" none", " source", " relevance", " detail", " conflict"}
        for rows in [module.causal_plausibility_rows(), module.causal_plausibility_heldout_rows()]:
            impossible_rows = [
                row for row in rows if row["stage101w4_chain_step"] == "impossible_card"
            ]
            counts = Counter(row["intelligence_answer"] for row in impossible_rows)
            self.assertEqual(expected, set(counts))
            self.assertGreaterEqual(min(counts.values()), 1)
            for row in impossible_rows:
                if row["intelligence_answer"] == " none":
                    self.assertEqual(row["card_plausibility"], " plausible")
                else:
                    self.assertEqual(row["card_plausibility"], " impossible")

    def test_answer_permission_blocks_impossible_card_chains(self) -> None:
        module = load_module()
        for rows in [module.causal_plausibility_rows(), module.causal_plausibility_heldout_rows()]:
            permission_rows = [
                row for row in rows if row["stage101w4_chain_step"] == "answer_permission"
            ]
            self.assertTrue(permission_rows)
            for row in permission_rows:
                prompt = str(row["prompt"])
                self.assertIn("card_plausibility=", prompt)
                self.assertIn("impossible_card=", prompt)
                if row["card_plausibility"] == " impossible":
                    self.assertEqual(row["intelligence_answer"], " no", row["id"])
                    self.assertNotEqual(row["impossible_card"], " none", row["id"])

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
            [
                "source_trust",
                "evidence_relevance",
                "detail_sufficiency",
                "conflict_status",
                "card_plausibility",
                "impossible_card",
                "answer_permission",
                "missing_material",
            ],
            report["causal_plausibility_contract"]["chain_steps"],
        )

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.causal_plausibility_rows() + module.causal_plausibility_heldout_rows()
        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
