from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "596_build_stage101w8_latent_feature_reader_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage101w8_latent_feature_reader_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage101W8LatentFeatureReaderProbeTests(unittest.TestCase):
    def test_rows_supervise_all_latent_features_without_feature_label_prompting(self) -> None:
        module = load_module()
        rows = module.latent_feature_reader_rows()
        self.assertTrue(rows)
        forbidden = [
            "source_reliability=",
            "evidence_relevance=",
            "detail_sufficiency=",
            "conflict_status=",
        ]
        for row in rows:
            self.assertTrue(row["stage101w8_latent_feature_reader_required"])
            self.assertEqual(
                set(module.FEATURE_NAMES),
                set(row["feature_targets"]),
            )
            prompt = str(row["prompt"])
            self.assertIn("Claim:", prompt)
            self.assertIn("World:", prompt)
            self.assertIn("Can answer now?", prompt)
            for token in forbidden:
                self.assertNotIn(token, prompt)

    def test_each_split_balances_feature_labels_and_permission(self) -> None:
        module = load_module()
        for rows in [module.latent_feature_reader_rows(), module.latent_feature_reader_heldout_rows()]:
            self.assertGreaterEqual(len(rows), 48)
            permission = Counter(row["feature_targets"]["answer_permission"] for row in rows)
            self.assertGreaterEqual(permission["yes"], 8)
            self.assertGreaterEqual(permission["no"], 24)
            self.assertGreaterEqual(Counter(row["feature_targets"]["source_reliability"] for row in rows)["untrusted"], 16)
            self.assertGreaterEqual(Counter(row["feature_targets"]["evidence_relevance"] for row in rows)["irrelevant"], 16)
            self.assertGreaterEqual(Counter(row["feature_targets"]["detail_sufficiency"] for row in rows)["missing"], 24)
            self.assertGreaterEqual(Counter(row["feature_targets"]["conflict_status"] for row in rows)["conflict"], 16)

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
        self.assertEqual(module.FEATURE_NAMES, report["latent_feature_reader_contract"]["feature_names"])

    def test_probe_rows_fit_stage101_seq_len_384(self) -> None:
        module = load_module()
        rows = module.latent_feature_reader_rows() + module.latent_feature_reader_heldout_rows()
        for row in rows:
            shifted_len = len(str(row["prompt"]).encode("utf-8")) + len(
                str(row["intelligence_answer"]).encode("utf-8")
            ) - 1
            self.assertLessEqual(shifted_len, 384, row["id"])


if __name__ == "__main__":
    unittest.main()
