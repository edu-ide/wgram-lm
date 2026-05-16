from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


def load_module():
    path = Path("scripts/391_train_qwen35_integrated_public_mcq_healing.py")
    spec = importlib.util.spec_from_file_location("qwen35_integrated_public_mcq_healing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Qwen35IntegratedPublicMCQHealingTests(unittest.TestCase):
    def test_parser_defaults_to_disjoint_mmlu_pro_split(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(["--init-checkpoint", "last.pt"])

        self.assertIn("validation_64", args.train_jsonl)
        self.assertIn("test_balanced_256", args.eval_jsonl)
        self.assertEqual(args.core_impl, "qwen_layer_wrapped")
        self.assertEqual(args.qwen_core_layer_indices, "3")
        self.assertEqual(args.core_adapter_dim, 128)

    def test_load_public_eval_module_reuses_mcq_helpers(self):
        module = load_module()
        public_eval = module.load_public_eval_module()

        self.assertEqual(public_eval.normalize_mcq_answer("Answer: c"), "C")
        self.assertEqual(public_eval.option_count({"options": ["a", "b", "c"]}), 3)
        args = module.build_arg_parser().parse_args(["--init-checkpoint", "last.pt"])
        self.assertEqual(args.margin_focus, "base_wrong")
        self.assertEqual(args.margin_weight, 0.0)
        self.assertEqual(args.ce_focus, "all")
        self.assertFalse(args.skip_train_eval)

    def test_batch_rows_chunks_without_dropping_tail(self):
        module = load_module()

        chunks = list(module.batch_rows([{"i": i} for i in range(5)], 2))

        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])

    def test_last_logits_ignore_right_padding(self):
        module = load_module()

        class TinyModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, force_core_off=False):
                del input_ids, attention_mask, force_core_off
                return type(
                    "Output",
                    (),
                    {
                        "logits": torch.tensor(
                            [
                                [[1.0, 0.0], [9.0, 0.0], [-5.0, 7.0]],
                                [[0.0, 3.0], [0.0, 4.0], [0.0, 5.0]],
                            ]
                        )
                    },
                )()

        selected = module._last_logits(
            TinyModel(),
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )

        self.assertTrue(torch.equal(selected, torch.tensor([[9.0, 0.0], [0.0, 5.0]])))

    def test_category_gain_summary_reports_regressions(self):
        module = load_module()

        summary = module.category_gain_summary(
            {
                "base_metrics": {
                    "by_category": {
                        "health": {"hits": 3, "total": 4, "accuracy": 0.75},
                        "law": {"hits": 1, "total": 2, "accuracy": 0.5},
                    }
                },
                "core_metrics": {
                    "by_category": {
                        "health": {"hits": 2, "total": 4, "accuracy": 0.5},
                        "law": {"hits": 2, "total": 2, "accuracy": 1.0},
                    }
                },
            },
            min_cases=1,
        )

        self.assertEqual(summary["min_hit_delta"], -1)
        self.assertAlmostEqual(summary["min_accuracy_delta"], -0.25)
        self.assertEqual(summary["negative_hit_delta_sum"], 1)
        self.assertAlmostEqual(summary["negative_accuracy_delta_sum"], 0.25)

    def test_balanced_category_sampling_uses_category_groups(self):
        module = load_module()
        rng = module.random.Random(7)
        rows = [
            {"category": "a", "id": "a0"},
            {"category": "b", "id": "b0"},
        ]

        chunk = module.sample_training_chunk(
            rng,
            rows,
            batch_size=6,
            balanced_category_sampling=True,
        )

        self.assertEqual(len(chunk), 2)
        self.assertTrue(all(row["category"] in {"a", "b"} for row in chunk))

    def test_train_and_eval_files_remain_distinct_in_runner_defaults(self):
        runner = Path("scripts/391_run_qwen35_integrated_public_mcq_healing.sh")
        text = runner.read_text(encoding="utf-8")

        self.assertIn("TRAIN_JSONL", text)
        self.assertIn("EVAL_JSONL", text)
        self.assertIn("SEED", text)
        self.assertIn("MARGIN_WEIGHT", text)
        self.assertIn("MARGIN_FOCUS", text)
        self.assertIn("CE_FOCUS", text)
        self.assertIn("BALANCED_CATEGORY_SAMPLING", text)
        self.assertIn("CATEGORY_REGRESSION_PENALTY", text)
        self.assertIn("MIN_EVAL_CATEGORY_GAIN", text)
        self.assertIn("QWEN_CORE_LAYER_INDICES", text)
        self.assertIn("CORE_ADAPTER_DIM", text)
        self.assertIn("CHECKPOINT_LOAD_MODE", text)
        self.assertIn("SKIP_TRAIN_EVAL", text)
        self.assertIn("MAX_EVAL_CASES", text)
        self.assertIn("mmlu_pro_validation_64", text)
        self.assertIn("mmlu_pro_test_balanced_256", text)

    def test_checkpoint_loader_can_skip_shape_mismatch_for_core_warm_start(self):
        module = load_module()

        class TinyModel(torch.nn.Module):
            def __init__(self, out_features: int) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, out_features, bias=False)

        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mismatch.pt"
            source = TinyModel(3)
            target = TinyModel(5)
            torch.save({"model": source.state_dict(), "report": {"source": "unit"}}, checkpoint)

            info = module.load_checkpoint(
                target,
                str(checkpoint),
                load_mode="skip_mismatch",
            )

        self.assertEqual(info["load_mode"], "skip_mismatch")
        self.assertEqual(info["shape_mismatch_key_count"], 1)
        self.assertEqual(info["missing_key_count"], 1)
        self.assertEqual(info["checkpoint_report"], {"source": "unit"})


if __name__ == "__main__":
    unittest.main()
