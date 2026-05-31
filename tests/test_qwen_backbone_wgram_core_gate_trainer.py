from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


def load_module():
    path = Path("scripts/362_train_qwen_backbone_wgram_core_gate.py")
    spec = importlib.util.spec_from_file_location("qwen_backbone_wgram_core_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QwenBackboneQTRMCoreGateTrainerTests(unittest.TestCase):
    def test_parser_accepts_wrapped_and_shared_core_options(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--core-impl",
                "ouro_shared_qwen_layer",
                "--qwen-core-layer-indices",
                "3,7",
                "--core-adapter-dim",
                "16",
                "--core-delta-adapter-mode",
                "adapter_only",
                "--steps",
                "2",
                "--h-cycles",
                "3",
                "--l-cycles",
                "6",
                "--outer-steps",
                "3",
                "--core-convergence-halt-enabled",
                "--core-convergence-halt-threshold",
                "0.25",
                "--core-step-conditioning-enabled",
                "--core-step-conditioning-max-steps",
                "128",
                "--core-step-conditioning-scale",
                "0.5",
                "--case-mode",
                "mixed_v1",
                "--min-family-gain",
                "0.01",
                "--train-case-mode",
                "hard_repair_v1",
                "--eval-case-mode",
                "hard_v1",
                "--family-loss-weights",
                "select_pair=2,checksum4=1.5",
                "--acceptance-metric",
                "label_choice",
                "--eval-every-steps",
                "50",
                "--restore-best-checkpoint",
                "--init-checkpoint",
                "/tmp/init.pt",
                "--train-qwen",
                "--mandatory-core",
                "--unfreeze-qwen-layer-indices",
                "3",
                "--unfreeze-qwen-final-norm",
                "--qwen-lr",
                "2e-5",
                "--qwen-weight-decay",
                "0.01",
                "--language-kl-weight",
                "0.2",
                "--language-kl-batch-size",
                "3",
            ]
        )

        self.assertEqual(args.core_impl, "ouro_shared_qwen_layer")
        self.assertEqual(module.parse_int_list(args.qwen_core_layer_indices), (3, 7))
        self.assertEqual(args.core_adapter_dim, 16)
        self.assertEqual(args.core_delta_adapter_mode, "adapter_only")
        self.assertEqual(args.steps, 2)
        self.assertEqual(args.h_cycles, 3)
        self.assertEqual(args.l_cycles, 6)
        self.assertEqual(args.outer_steps, 3)
        self.assertTrue(args.core_convergence_halt_enabled)
        self.assertEqual(args.core_convergence_halt_threshold, 0.25)
        self.assertTrue(args.core_step_conditioning_enabled)
        self.assertEqual(args.core_step_conditioning_max_steps, 128)
        self.assertEqual(args.core_step_conditioning_scale, 0.5)
        self.assertEqual(args.case_mode, "mixed_v1")
        self.assertEqual(args.train_case_mode, "hard_repair_v1")
        self.assertEqual(args.eval_case_mode, "hard_v1")
        self.assertEqual(args.min_family_gain, 0.01)
        self.assertEqual(
            module.parse_float_map(args.family_loss_weights),
            {"select_pair": 2.0, "checksum4": 1.5},
        )
        self.assertEqual(args.acceptance_metric, "label_choice")
        self.assertEqual(args.eval_every_steps, 50)
        self.assertTrue(args.restore_best_checkpoint)
        self.assertEqual(args.init_checkpoint, "/tmp/init.pt")
        self.assertTrue(args.train_qwen)
        self.assertTrue(args.mandatory_core)
        self.assertEqual(module.parse_int_list(args.unfreeze_qwen_layer_indices), (3,))
        self.assertTrue(args.unfreeze_qwen_final_norm)
        self.assertEqual(args.qwen_lr, 2e-5)
        self.assertEqual(args.qwen_weight_decay, 0.01)
        self.assertEqual(args.language_kl_weight, 0.2)
        self.assertEqual(args.language_kl_batch_size, 3)

        ouro_args = module.build_arg_parser().parse_args(
            [
                "--core-impl",
                "ouro_weight_wrapped",
                "--ouro-model-id",
                "/tmp/ouro",
                "--ouro-core-layer-indices",
                "24",
                "--ouro-partial-safetensors",
            ]
        )

        self.assertEqual(ouro_args.core_impl, "ouro_weight_wrapped")
        self.assertEqual(ouro_args.ouro_model_id, "/tmp/ouro")
        self.assertEqual(module.parse_int_list(ouro_args.ouro_core_layer_indices), (24,))
        self.assertTrue(ouro_args.ouro_partial_safetensors)

    def test_synthetic_cases_are_deterministic_and_have_small_digit_labels(self):
        module = load_module()

        first = module.build_synthetic_cases(count=8, seed=123)
        second = module.build_synthetic_cases(count=8, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        labels = {case.label for case in first}
        self.assertTrue(labels <= set("0123456789"))
        self.assertTrue(any("mod 10" in case.prompt for case in first))
        self.assertTrue(any("Answer:" in case.prompt for case in first))

    def test_hard_and_mixed_case_modes_add_harder_families(self):
        module = load_module()

        hard = module.build_synthetic_cases(count=9, seed=123, case_mode="hard_v1")
        repair = module.build_synthetic_cases(count=8, seed=123, case_mode="hard_repair_v1")
        mixed = module.build_synthetic_cases(count=12, seed=123, case_mode="mixed_v1")

        self.assertEqual({case.family for case in hard}, {"checksum4", "chain5", "select_pair"})
        self.assertGreater(
            sum(1 for case in repair if case.family == "select_pair"),
            sum(1 for case in repair if case.family == "chain5"),
        )
        self.assertIn("checksum", {case.family for case in mixed})
        self.assertIn("checksum4", {case.family for case in mixed})
        self.assertTrue({case.label for case in hard} <= set("0123456789"))

    def test_family_gain_summary_reports_minima(self):
        module = load_module()

        evaluation = {
            "by_family": {
                "a": {
                    "base_accuracy": 0.1,
                    "core_accuracy": 0.3,
                    "base_choice_accuracy": 0.2,
                    "core_choice_accuracy": 0.5,
                },
                "b": {
                    "base_accuracy": 0.4,
                    "core_accuracy": 0.35,
                    "base_choice_accuracy": 0.1,
                    "core_choice_accuracy": 0.2,
                },
            }
        }

        summary = module.family_gain_summary(evaluation)
        choice = module.family_gain_summary(evaluation, metric="label_choice")

        self.assertAlmostEqual(summary["gains"]["a"], 0.2)
        self.assertAlmostEqual(summary["gains"]["b"], -0.05)
        self.assertAlmostEqual(summary["min_gain"], -0.05)
        self.assertAlmostEqual(summary["min_core_accuracy"], 0.3)
        self.assertAlmostEqual(choice["gains"]["a"], 0.3)
        self.assertAlmostEqual(choice["min_gain"], 0.1)
        self.assertEqual(choice["metric"], "label_choice")

    def test_parse_float_map_rejects_malformed_items(self):
        module = load_module()

        self.assertEqual(module.parse_float_map("a=1.5,b=2"), {"a": 1.5, "b": 2.0})
        with self.assertRaises(ValueError):
            module.parse_float_map("bad")

    def test_evaluation_acceptance_summary_respects_choice_metric(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--acceptance-metric",
                "label_choice",
                "--min-reasoning-gain",
                "0.05",
                "--min-family-gain",
                "0.01",
                "--min-family-core-accuracy",
                "0.1",
            ]
        )

        summary = module.evaluation_acceptance_summary(
            {
                "gain": 0.0,
                "choice_gain": 0.06,
                "by_family": {
                    "a": {
                        "base_accuracy": 0.0,
                        "core_accuracy": 0.0,
                        "base_choice_accuracy": 0.1,
                        "core_choice_accuracy": 0.2,
                    }
                },
            },
            args,
        )

        self.assertTrue(summary["accepted_reasoning_gain"])
        self.assertTrue(summary["accepted_family_gain"])
        self.assertTrue(summary["accepted_family_core_accuracy"])
        self.assertEqual(summary["metric"], "label_choice")

    def test_empty_init_checkpoint_is_noop(self):
        module = load_module()

        self.assertEqual(module._load_trainable_checkpoint(object(), ""), {"path": "", "loaded": False})

    def test_init_checkpoint_report_separates_frozen_and_trainable_missing(self):
        module = load_module()

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.trainable = nn.Parameter(torch.ones(2))
                self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)

        model = TinyModel()
        checkpoint_path = Path("local_eval/test_trainable_checkpoint_report.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {"trainable": torch.zeros(2)}, "report": {"accepted": True}}, checkpoint_path)

        report = module._load_trainable_checkpoint(model, str(checkpoint_path))

        self.assertTrue(report["loaded"])
        self.assertEqual(report["trainable_missing_key_count"], 0)
        self.assertEqual(report["trainable_loaded_key_count"], 1)
        self.assertGreaterEqual(report["missing_key_count"], 1)


if __name__ == "__main__":
    unittest.main()
