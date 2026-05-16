from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def load_module():
    path = Path("scripts/395_autoresearch_arbitration_probe.py")
    spec = importlib.util.spec_from_file_location("autoresearch_arbitration_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AutoresearchArbitrationProbeTests(unittest.TestCase):
    def test_fit_best_rule_keeps_useful_switch_and_rejects_harmful_switch(self):
        module = load_module()
        cases = [
            {
                "category": "physics",
                "gold": "A",
                "base_pred": "B",
                "core_pred": "A",
                "base_margin": 0.8,
                "core_margin": 0.4,
                "switch_adv": 1.2,
            },
            {
                "category": "health",
                "gold": "C",
                "base_pred": "C",
                "core_pred": "D",
                "base_margin": 0.1,
                "core_margin": 0.8,
                "switch_adv": 0.5,
            },
            {
                "category": "math",
                "gold": "E",
                "base_pred": "E",
                "core_pred": "E",
                "base_margin": 1.5,
                "core_margin": 1.5,
                "switch_adv": 0.0,
            },
        ]

        rule, summary = module.fit_best_rule(
            cases,
            base_margin_grid=(0.5, 1.0),
            core_margin_grid=(0.25,),
            switch_adv_grid=(0.0, 1.0),
        )

        self.assertEqual(summary["base_hits"], 2)
        self.assertEqual(summary["arb_hits"], 3)
        self.assertEqual(summary["corrections"], 1)
        self.assertEqual(summary["regressions"], 0)
        self.assertEqual(rule.switch_adv_min, 1.0)

    def test_append_ledger_writes_keep_discard_row(self):
        module = load_module()
        report = {
            "timestamp": "2026-05-16T20:00:00",
            "probe": "autoresearch_qtrm_score_geometry_arbitration",
            "policy": "threshold",
            "decision": "accepted_arbitration_probe",
            "accepted": True,
            "fit_summary": {"base_hits": 10, "core_hits": 9, "arb_hits": 11},
            "eval_summary": {"base_hits": 8, "core_hits": 7, "arb_hits": 9},
            "policy_detail": "bm<=5.0,cm>=0.25,adv>=1.0",
            "report_path": "local_eval/example/report.json",
            "next_action": "train arbitration head",
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            module.append_ledger(path, report)
            rows = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            rows[0].split("\t")[:4],
            ["timestamp", "probe", "policy", "decision"],
        )
        self.assertIn("\taccepted_arbitration_probe\tkeep\t", rows[1])
        self.assertIn("bm<=5.0,cm>=0.25,adv>=1.0", rows[1])

    def test_linear_policy_can_fit_useful_switch_feature(self):
        module = load_module()
        cases = [
            {
                "gold": "A",
                "choices": "ABCD",
                "base_pred": "B",
                "core_pred": "A",
                "base_margin": 0.8,
                "core_margin": 0.5,
                "switch_adv": 1.5,
                "base_confidence": 0.3,
                "core_confidence": 0.7,
                "base_entropy": 1.2,
                "core_entropy": 0.7,
            },
            {
                "gold": "C",
                "choices": "ABCD",
                "base_pred": "C",
                "core_pred": "D",
                "base_margin": 0.2,
                "core_margin": 0.7,
                "switch_adv": 0.1,
                "base_confidence": 0.4,
                "core_confidence": 0.45,
                "base_entropy": 1.1,
                "core_entropy": 1.0,
            },
            {
                "gold": "E",
                "choices": "ABCDE",
                "base_pred": "E",
                "core_pred": "E",
                "base_margin": 1.0,
                "core_margin": 1.0,
                "switch_adv": 0.0,
                "base_confidence": 0.8,
                "core_confidence": 0.8,
                "base_entropy": 0.4,
                "core_entropy": 0.4,
            },
        ]

        policy, summary = module.fit_linear_policy(
            cases,
            steps=80,
            lr=0.1,
            weight_decay=0.0,
            threshold_grid=(0.3, 0.5, 0.7),
        )

        self.assertEqual(summary["arb_hits"], 3)
        self.assertEqual(summary["corrections"], 1)
        self.assertEqual(summary["regressions"], 0)
        self.assertEqual(policy.feature_names, module.linear_feature_names())

    def test_runner_passes_autoresearch_reference_and_checkpoint(self):
        runner = Path("scripts/395_run_autoresearch_arbitration_probe.sh").read_text(encoding="utf-8")

        self.assertIn("references/official/autoresearch", runner)
        self.assertIn("AUTORESEARCH_COMMIT", runner)
        self.assertIn("--autoresearch-commit", runner)
        self.assertIn("POLICY", runner)
        self.assertIn("--policy", runner)
        self.assertIn("qwen35_integrated_midlayer_suffix_adapteronly_coretrain_langanchor", runner)

    def test_report_json_shape_is_serializable(self):
        module = load_module()
        rule = module.ArbitrationRule(5.0, 0.25, 1.0)
        summary = module.summarize_cases(
            [
                {
                    "category": "physics",
                    "gold": "A",
                    "base_pred": "B",
                    "core_pred": "A",
                    "base_margin": 0.8,
                    "core_margin": 0.4,
                    "switch_adv": 1.2,
                }
            ],
            rule,
        )

        encoded = json.dumps({"best_rule": rule.__dict__, "summary": summary})

        self.assertIn('"arb_hits": 1', encoded)


if __name__ == "__main__":
    unittest.main()
