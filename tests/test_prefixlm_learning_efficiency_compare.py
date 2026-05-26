import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/535_compare_prefixlm_learning_efficiency.py")
    spec = importlib.util.spec_from_file_location("prefixlm_learning_efficiency", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_report(root: Path, name: str, *, seq_len: int, losses: list[tuple[int, float, int]]):
    path = root / f"{name}.json"
    history = [
        {"step": step, "loss": loss, "tokens_seen": tokens, "target_tokens_seen": tokens // 2}
        for step, loss, tokens in losses
    ]
    path.write_text(
        json.dumps(
            {
                "target_level": "HRM-Text Data-IO PrefixLM native one-body learning-efficiency gate",
                "dataset": {
                    "contract": "hrm_text_data_io_prefixlm",
                    "vocab_size": 65536,
                    "seq_len": seq_len,
                    "target_only": True,
                    "max_seq_len": 1025,
                    "total_length": 3440870,
                },
                "train": {"tokens_seen": losses[-1][2], "steps": losses[-1][0]},
                "loss_history": history,
                "initial_logged_loss": losses[0][1],
                "final_logged_loss": losses[-1][1],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_eval_report(
    root: Path,
    name: str,
    *,
    seq_len: int,
    train_final_loss: float,
    eval_losses: list[tuple[int, float, int]],
):
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "dataset": {
                    "contract": "hrm_text_data_io_prefixlm",
                    "vocab_size": 65536,
                    "seq_len": seq_len,
                    "target_only": True,
                    "max_seq_len": 1025,
                    "total_length": 3440870,
                },
                "eval_dataset": {
                    "contract": "hrm_text_data_io_prefixlm",
                    "vocab_size": 65536,
                    "seq_len": seq_len,
                    "target_only": True,
                    "max_seq_len": 1025,
                    "total_length": 3440870,
                    "epoch": 1,
                    "rows": 128,
                    "drop_overlength": True,
                    "eval_protocol": "unit_test_fixed",
                    "eval_fingerprint": "unit-test",
                    "eval_batch_size": 4,
                    "eval_max_batches": 0,
                },
                "train": {"tokens_seen": eval_losses[-1][2], "steps": eval_losses[-1][0]},
                "loss_history": [
                    {"step": eval_losses[-1][0], "loss": train_final_loss, "tokens_seen": eval_losses[-1][2]}
                ],
                "eval_loss_history": [
                    {
                        "step": step,
                        "eval_loss": loss,
                        "tokens_seen": tokens,
                        "eval_tokens": 512,
                        "eval_target_tokens": 256,
                    }
                    for step, loss, tokens in eval_losses
                ],
                "final_logged_loss": train_final_loss,
                "final_eval_loss": eval_losses[-1][1],
            }
        ),
        encoding="utf-8",
    )
    return path


class PrefixLMLearningEfficiencyCompareTests(unittest.TestCase):
    def test_supports_10x_when_candidate_reaches_baseline_loss_within_token_budget(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_report(
                root,
                "baseline",
                seq_len=128,
                losses=[(10, 3.0, 1000), (20, 2.0, 2000)],
            )
            candidate = write_report(
                root,
                "candidate",
                seq_len=128,
                losses=[(1, 2.4, 100), (2, 1.9, 150)],
            )

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "supports_10x_on_prefixlm_loss")
        self.assertTrue(report["comparable"])
        self.assertEqual(report["metrics"]["baseline_final_loss"], 2.0)
        self.assertEqual(report["metrics"]["candidate_tokens_to_baseline_loss"], 150)
        self.assertEqual(report["metrics"]["max_tokens_for_factor_claim"], 200.0)
        self.assertIsNone(report["metrics"]["baseline_tokens_to_candidate_final_loss"])
        self.assertIsNone(report["metrics"]["observed_candidate_speedup_at_candidate_final_loss"])

    def test_rejects_mismatched_prefixlm_contract(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_report(
                root,
                "baseline",
                seq_len=128,
                losses=[(10, 3.0, 1000), (20, 2.0, 2000)],
            )
            candidate = write_report(
                root,
                "candidate",
                seq_len=256,
                losses=[(1, 2.4, 100), (2, 1.9, 150)],
            )

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "invalid_comparison")
        self.assertFalse(report["comparable"])
        self.assertIn("seq_len", report["contract_mismatches"])

    def test_unproven_when_candidate_reaches_target_too_late(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_report(
                root,
                "baseline",
                seq_len=128,
                losses=[(10, 3.0, 1000), (20, 2.0, 2000)],
            )
            candidate = write_report(
                root,
                "candidate",
                seq_len=128,
                losses=[(1, 2.4, 100), (4, 1.9, 500)],
            )

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "unproven")
        self.assertIn("later than the 10x cutoff", " ".join(report["reasons"]))

    def test_eval_loss_history_is_preferred_over_train_loss(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_eval_report(
                root,
                "baseline",
                seq_len=128,
                train_final_loss=1.0,
                eval_losses=[(10, 3.0, 1000), (20, 2.0, 2000)],
            )
            candidate = write_eval_report(
                root,
                "candidate",
                seq_len=128,
                train_final_loss=1.5,
                eval_losses=[(1, 2.4, 100), (2, 1.9, 150)],
            )

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "supports_10x_on_prefixlm_loss")
        self.assertEqual(report["metrics"]["metric_source"], "eval_loss_history")
        self.assertEqual(report["metrics"]["baseline_final_loss"], 2.0)

    def test_reports_baseline_tokens_to_candidate_final_loss_when_baseline_is_stronger(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_eval_report(
                root,
                "baseline",
                seq_len=128,
                train_final_loss=1.0,
                eval_losses=[(10, 3.0, 1000), (20, 2.0, 2000), (30, 1.0, 3000)],
            )
            candidate = write_eval_report(
                root,
                "candidate",
                seq_len=128,
                train_final_loss=1.5,
                eval_losses=[(1, 2.4, 100), (2, 1.9, 150)],
            )

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "unproven")
        self.assertEqual(report["metrics"]["baseline_tokens_to_candidate_final_loss"], 3000)
        self.assertEqual(report["metrics"]["observed_candidate_speedup_at_candidate_final_loss"], 3000 / 150)

    def test_rejects_metric_source_mismatch(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_eval_report(
                root,
                "baseline",
                seq_len=128,
                train_final_loss=1.0,
                eval_losses=[(10, 3.0, 1000), (20, 2.0, 2000)],
            )
            candidate = write_report(
                root,
                "candidate",
                seq_len=128,
                losses=[(1, 2.4, 100), (2, 1.9, 150)],
            )

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "invalid_comparison")
        self.assertIn("metric_source", report["contract_mismatches"])

    def test_rejects_eval_coverage_mismatch(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = write_eval_report(
                root,
                "baseline",
                seq_len=128,
                train_final_loss=1.0,
                eval_losses=[(10, 3.0, 1000), (20, 2.0, 2000)],
            )
            candidate = write_eval_report(
                root,
                "candidate",
                seq_len=128,
                train_final_loss=1.5,
                eval_losses=[(1, 2.4, 100), (2, 1.9, 150)],
            )
            data = json.loads(candidate.read_text(encoding="utf-8"))
            data["eval_loss_history"][-1]["eval_target_tokens"] = 128
            candidate.write_text(json.dumps(data), encoding="utf-8")

            report = module.compare_reports(baseline, candidate, factor=10.0)

        self.assertEqual(report["verdict"], "invalid_comparison")
        self.assertIn("eval_target_tokens", report["contract_mismatches"])


if __name__ == "__main__":
    unittest.main()
