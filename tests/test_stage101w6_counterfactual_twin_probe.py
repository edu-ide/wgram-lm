from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "593_build_stage101w6_counterfactual_twin_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w6_counterfactual_twin_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W6CounterfactualTwinProbeTests(unittest.TestCase):
    def test_every_pair_teaches_answerable_and_blocked_worlds(self) -> None:
        module = load_module()
        rows = module.counterfactual_twin_rows()

        by_pair: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            self.assertTrue(row["stage101w6_counterfactual_twin_required"])
            by_pair[str(row["twin_pair_id"])].add(str(row["stage101w6_chain_step"]))

        self.assertTrue(by_pair)
        self.assertTrue(
            all({"answerable_world", "blocked_world"}.issubset(steps) for steps in by_pair.values())
        )

    def test_twin_rows_cover_all_causal_axes_and_both_orders(self) -> None:
        module = load_module()
        expected_axes = {"source", "relevance", "detail", "conflict"}
        for rows in [module.counterfactual_twin_rows(), module.counterfactual_twin_heldout_rows()]:
            answerable_rows = [
                row for row in rows if row["stage101w6_chain_step"] == "answerable_world"
            ]
            axes = {str(row["repair_axis"]) for row in answerable_rows}
            positions = Counter(row["intelligence_answer"] for row in answerable_rows)
            self.assertEqual(expected_axes, axes)
            self.assertGreaterEqual(positions[" A"], 4)
            self.assertGreaterEqual(positions[" B"], 4)

    def test_prompts_use_world_contrast_not_cause_card_labels(self) -> None:
        module = load_module()
        forbidden = ["source_trust", "evidence_relevance", "detail_sufficiency", "conflict_status"]
        for rows in [module.counterfactual_twin_rows(), module.counterfactual_twin_heldout_rows()]:
            for row in rows:
                prompt = str(row["prompt"])
                self.assertIn("Claim:", prompt)
                self.assertIn("World A:", prompt)
                self.assertIn("World B:", prompt)
                for token in forbidden:
                    self.assertNotIn(token, prompt)

    def test_answerable_and_blocked_worlds_are_opposites(self) -> None:
        module = load_module()
        for rows in [module.counterfactual_twin_rows(), module.counterfactual_twin_heldout_rows()]:
            grouped: dict[str, dict[str, str]] = defaultdict(dict)
            for row in rows:
                grouped[str(row["twin_pair_id"])][str(row["stage101w6_chain_step"])] = str(
                    row["intelligence_answer"]
                )
            for pair_id, answers in grouped.items():
                self.assertEqual({"answerable_world", "blocked_world"}, set(answers), pair_id)
                self.assertEqual({" A", " B"}, set(answers.values()), pair_id)

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
            ["answerable_world", "blocked_world"],
            report["counterfactual_twin_contract"]["chain_steps"],
        )

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.counterfactual_twin_rows() + module.counterfactual_twin_heldout_rows()
        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
