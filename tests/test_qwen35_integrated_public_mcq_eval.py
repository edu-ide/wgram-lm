from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


def load_module():
    path = Path("scripts/390_eval_qwen35_integrated_public_mcq.py")
    spec = importlib.util.spec_from_file_location("qwen35_integrated_public_mcq_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Qwen35IntegratedPublicMCQEvalTests(unittest.TestCase):
    def test_normalize_mcq_answer_extracts_option_letters(self):
        module = load_module()

        self.assertEqual(module.normalize_mcq_answer("A"), "A")
        self.assertEqual(module.normalize_mcq_answer("(D)"), "D")
        self.assertEqual(module.normalize_mcq_answer("Answer: b"), "B")
        self.assertEqual(module.normalize_mcq_answer("unknown"), "")

    def test_load_suite_validates_required_schema(self):
        module = load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "mmlu_pro",
                        "case_id": "case-1",
                        "qtrm_prompt": "User: Q\nAssistant:",
                        "answer": "C",
                        "options": ["one", "two", "three"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = module.load_suite(path)

        self.assertEqual(rows[0]["answer"], "C")
        self.assertEqual(module.option_count(rows[0]), 3)

    def test_score_rows_compares_named_prediction_key(self):
        module = load_module()

        metrics = module.score_rows(
            [
                {"category": "math", "gold_answer": "A", "core_pred_answer": "A"},
                {"category": "math", "gold_answer": "B", "core_pred_answer": "C"},
                {"category": "physics", "gold_answer": "D", "core_pred_answer": "D"},
            ],
            pred_key="core_pred_answer",
        )

        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["cases"], 3)
        self.assertEqual(metrics["by_category"]["math"]["accuracy"], 0.5)

    def test_option_score_aggregates_acceptable_token_variants(self):
        module = load_module()
        log_probs = torch.log_softmax(torch.tensor([0.0, 0.0, -10.0]), dim=-1)

        score = module.option_score_from_log_probs(log_probs, [0, 1])

        self.assertGreater(float(score), float(log_probs[0]))
        self.assertAlmostEqual(float(score), float(torch.logsumexp(log_probs[:2], dim=0)), places=6)

    def test_parser_defaults_to_m3_checkpoint_and_mandatory_flag_available(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            ["--checkpoint", "local_eval/checkpoint.pt", "--mandatory-core"]
        )

        self.assertEqual(args.core_impl, "qwen_layer_wrapped")
        self.assertEqual(args.core_insertion_mode, "final_residual")
        self.assertEqual(args.core_insert_after_layer, -1)
        self.assertEqual(args.qwen_core_layer_indices, "3")
        self.assertEqual(args.core_adapter_dim, 128)
        self.assertEqual(args.core_residual_gate_mode, "constant")
        self.assertEqual(args.core_residual_gate_dim, 128)
        self.assertEqual(args.core_residual_gate_init, -2.0)
        self.assertEqual(args.h_cycles, 1)
        self.assertEqual(args.l_cycles, 1)
        self.assertEqual(args.outer_steps, 1)
        self.assertFalse(args.core_convergence_halt_enabled)
        self.assertFalse(args.core_step_conditioning_enabled)
        self.assertTrue(args.mandatory_core)

    def test_runner_exposes_core_scale_knobs(self):
        text = Path("scripts/390_run_qwen35_integrated_m4_public_mcq.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("QWEN_CORE_LAYER_INDICES", text)
        self.assertIn("CORE_INSERTION_MODE", text)
        self.assertIn("--core-insertion-mode", text)
        self.assertIn("CORE_INSERT_AFTER_LAYER", text)
        self.assertIn("--core-insert-after-layer", text)
        self.assertIn("CORE_ADAPTER_DIM", text)
        self.assertIn("CORE_DELTA_ADAPTER_MODE", text)
        self.assertIn("CORE_RESIDUAL_GATE_MODE", text)
        self.assertIn("--core-residual-gate-mode", text)
        self.assertIn("H_CYCLES", text)
        self.assertIn("L_CYCLES", text)
        self.assertIn("OUTER_STEPS", text)
        self.assertIn("CORE_CONVERGENCE_HALT_ENABLED", text)
        self.assertIn("--core-convergence-halt-enabled", text)
        self.assertIn("CORE_STEP_CONDITIONING_ENABLED", text)
        self.assertIn("--core-step-conditioning-enabled", text)
        self.assertIn("RESIDUAL_SCALE", text)

    def test_ssot_revalidation_runner_lists_canonical_candidates(self):
        text = Path("scripts/396_run_qwen35_integrated_ssot_revalidation.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("midlayer_external64", text)
        self.assertIn("midlayer_mmlupro64", text)
        self.assertIn("optiononly_mmlupro64", text)
        self.assertIn("public_coreonly_mmlu256", text)
        self.assertIn("l23open_seed20260520_mmlu256", text)
        self.assertIn("public_coreonly_mmlu512_resid0p06", text)
        self.assertIn("summary.jsonl", text)
        self.assertIn("strict_accepted", text)
        self.assertIn("scripts/390_run_qwen35_integrated_m4_public_mcq.sh", text)


if __name__ == "__main__":
    unittest.main()
