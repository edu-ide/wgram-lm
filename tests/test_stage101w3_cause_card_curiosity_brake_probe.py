from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "590_build_stage101w3_cause_card_curiosity_brake_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w3_cause_card_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W3CauseCardCuriosityBrakeProbeTests(unittest.TestCase):
    def test_every_case_teaches_cause_cards_before_answer_permission(self) -> None:
        module = load_module()
        rows = module.cause_card_rows()

        by_case: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            self.assertTrue(row["stage101w3_cause_card_required"])
            by_case[str(row["cause_card_case_id"])].add(str(row["stage101w3_chain_step"]))

        required = {
            "source_trust",
            "evidence_relevance",
            "detail_sufficiency",
            "conflict_status",
            "answer_permission",
        }
        self.assertTrue(by_case)
        self.assertTrue(all(required.issubset(steps) for steps in by_case.values()))

    def test_answer_permission_uses_parent_cause_cards_and_balances_yes_no(self) -> None:
        module = load_module()
        for rows in [module.cause_card_rows(), module.cause_card_heldout_rows()]:
            permission_rows = [
                row for row in rows if row["stage101w3_chain_step"] == "answer_permission"
            ]
            counts = Counter(row["intelligence_answer"] for row in permission_rows)
            self.assertEqual(counts[" yes"], counts[" no"])
            self.assertGreaterEqual(counts[" yes"], 4)
            for row in permission_rows:
                prompt = str(row["prompt"])
                self.assertIn("Cause cards:", prompt)
                self.assertIn("source_trust=", prompt)
                self.assertIn("evidence_relevance=", prompt)
                self.assertIn("detail_sufficiency=", prompt)
                self.assertIn("conflict_status=", prompt)

    def test_source_quality_pair_flips_permission_with_same_claim_and_evidence(self) -> None:
        module = load_module()
        permission_rows = [
            row for row in module.cause_card_rows() if row["stage101w3_chain_step"] == "answer_permission"
        ]
        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in permission_rows:
            pair_id = str(row.get("cause_card_pair_id", ""))
            if pair_id.startswith("source_quality"):
                by_pair[pair_id].append(row)

        self.assertTrue(by_pair)
        pair_rows = next(iter(by_pair.values()))
        self.assertEqual({" yes", " no"}, {row["intelligence_answer"] for row in pair_rows})
        self.assertEqual({row["source_claim"] for row in pair_rows}, {pair_rows[0]["source_claim"]})
        self.assertEqual({row["evidence_payload"] for row in pair_rows}, {pair_rows[0]["evidence_payload"]})
        self.assertEqual({" trusted", " untrusted"}, {row["source_trust_answer"] for row in pair_rows})

    def test_missing_material_rows_are_short_and_balanced(self) -> None:
        module = load_module()
        for rows in [module.cause_card_rows(), module.cause_card_heldout_rows()]:
            missing_rows = [
                row for row in rows if row["stage101w3_chain_step"] == "missing_material"
            ]
            counts = Counter(row["intelligence_answer"] for row in missing_rows)
            self.assertEqual({" none", " source", " relevance", " detail", " conflict"}, set(counts))
            self.assertEqual(max(counts.values()), min(counts.values()))
            self.assertTrue(
                all(len(str(row["intelligence_answer"]).strip().split()) == 1 for row in missing_rows)
            )

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
                "answer_permission",
                "missing_material",
            ],
            report["cause_card_contract"]["chain_steps"],
        )

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.cause_card_rows() + module.cause_card_heldout_rows()
        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
