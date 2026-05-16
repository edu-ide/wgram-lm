import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path("scripts/342_qtrm_native_l5d_backbone_compare.py")
    spec = importlib.util.spec_from_file_location("qtrm_native_l5d_backbone_compare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QTRMNativeL5DBackboneCompareTests(unittest.TestCase):
    def test_official_fla_command_uses_strict_backend(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/official"),
            candidate="official_fla",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--backbone", command)
        self.assertIn("qtrm_hybrid_3to1", command)
        self.assertIn("--delta-backend", command)
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)
        self.assertIn("--delta-head-dim", command)

    def test_standard_profile_uses_l5_scale_training_args(self):
        module = load_module()

        args = module.base_profile_args("standard")

        self.assertEqual(args[args.index("--steps") + 1], "12000")
        self.assertEqual(args[args.index("--train-cases") + 1], "24576")
        self.assertEqual(args[args.index("--eval-cases") + 1], "768")
        self.assertEqual(args[args.index("--d-model") + 1], "128")
        self.assertEqual(args[args.index("--accept-min-exact") + 1], "0.60")

    def test_official_fla_think_command_uses_mha_encode_decode(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/official_fla_think"),
            candidate="official_fla_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertIn("--backbone", command)
        self.assertIn("qtrm_hybrid_3to1", command)
        self.assertIn("--encode-backbone", command)
        self.assertIn("--think-backbone", command)
        self.assertIn("--decode-backbone", command)
        self.assertEqual(command[command.index("--encode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-backbone") + 1], "qtrm_hybrid_3to1")
        self.assertEqual(command[command.index("--decode-backbone") + 1], "mha_etd")
        self.assertIn("--strict-backends", command)

    def test_official_mamba3_think_command_uses_mha_encode_decode(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/official_mamba3_think"),
            candidate="official_mamba3_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertIn("--backbone", command)
        self.assertIn("mamba3", command)
        self.assertEqual(command[command.index("--encode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-backbone") + 1], "mamba3")
        self.assertEqual(command[command.index("--decode-backbone") + 1], "mha_etd")
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_mamba3_think_command_uses_dual_state_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_mamba3_think"),
            candidate="trm_dual_z_mamba3_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--encode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-backbone") + 1], "mamba3")
        self.assertEqual(command[command.index("--decode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_fla_think_command_uses_dual_state_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_fla_think"),
            candidate="trm_dual_z_fla_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--encode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-backbone") + 1], "qtrm_hybrid_3to1")
        self.assertEqual(command[command.index("--decode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_official_trm_think_command_uses_dual_state_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_official_trm_think"),
            candidate="trm_dual_z_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--encode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--decode-backbone") + 1], "mha_etd")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")

    def test_trm_dual_z_official_trm_l2_command_sets_l_cycles(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_official_trm_l2_think"),
            candidate="trm_dual_z_official_trm_l2_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--trm-l-cycles") + 1], "2")

    def test_trm_dual_z_official_trm_fullgrad_command_disables_no_grad_cycles(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_official_trm_fullgrad_think"),
            candidate="trm_dual_z_official_trm_fullgrad_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertIn("--trm-full-grad-cycles", command)

    def test_trm_dual_z_gated_official_trm_command_uses_stabilized_dual_state_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_gated_official_trm_think"),
            candidate="trm_dual_z_gated_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_gated")

    def test_trm_dual_z_residual_official_trm_command_uses_residual_dual_state_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_residual_official_trm_think"),
            candidate="trm_dual_z_residual_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_residual")

    def test_trm_dual_z_coupled_residual_command_keeps_official_trm_attention_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_residual_official_trm_think"),
            candidate="trm_dual_z_coupled_residual_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled_residual")

    def test_trm_dual_z_coupled_command_keeps_official_trm_attention_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_official_trm_think"),
            candidate="trm_dual_z_coupled_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled")
        self.assertNotIn("fla_gated_delta", command)

    def test_coupled_gated_attention_command_uses_gated_attention_think_block(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_gated_attention_think"),
            candidate="trm_dual_z_coupled_gated_attention_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_gated_attention")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled")
        self.assertNotIn("fla_gated_delta", command)

    def test_coupled_qwen_attention_command_uses_qwen_attention_think_block(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_qwen_attention_think"),
            candidate="trm_dual_z_coupled_qwen_attention_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_qwen_attention")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled")
        self.assertNotIn("fla_gated_delta", command)

    def test_coupled_cross_attention_command_uses_official_trm_cross_state_structure(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_cross_attention_think"),
            candidate="trm_dual_z_coupled_cross_attention_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled_cross_attention")
        self.assertNotIn("fla_gated_delta", command)

    def test_coupled_step_conditioned_attention_command_uses_official_trm_step_structure(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_step_conditioned_attention_think"),
            candidate="trm_dual_z_coupled_step_conditioned_attention_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(
            command[command.index("--think-structure") + 1],
            "trm_dual_z_coupled_step_conditioned_attention",
        )
        self.assertNotIn("fla_gated_delta", command)

    def test_coupled_delta_l_only_command_uses_official_trm_core_and_strict_delta(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_delta_l_only_official_trm_think"),
            candidate="trm_dual_z_coupled_delta_l_only_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled_delta_l_only")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_coupled_mamba_h_only_command_uses_official_trm_core_and_strict_mamba(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_mamba_h_only_official_trm_think"),
            candidate="trm_dual_z_coupled_mamba_h_only_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled_mamba_h_only")
        self.assertIn("--strict-backends", command)
        self.assertNotIn("fla_gated_delta", command)

    def test_coupled_gated_proposal_command_uses_official_trm_core_and_strict_proposals(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_coupled_gated_proposal_official_trm_think"),
            candidate="trm_dual_z_coupled_gated_proposal_official_trm_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_official")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_coupled_gated_proposal")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_gated_delta_command_uses_stabilized_state_and_strict_delta(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_gated_trm_gated_delta_think"),
            candidate="trm_dual_z_gated_trm_gated_delta_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_gated_delta")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z_gated")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_trm_mamba3_command_uses_trm_shell_mamba3_mixer(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_trm_mamba3_think"),
            candidate="trm_dual_z_trm_mamba3_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_mamba3")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_trm_gated_delta_command_uses_trm_shell_delta_mixer(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_trm_gated_delta_think"),
            candidate="trm_dual_z_trm_gated_delta_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_gated_delta")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_trm_qwen35_3to1_command_uses_hybrid_trm_shell(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_trm_qwen35_3to1_think"),
            candidate="trm_dual_z_trm_qwen35_3to1_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_qwen35_3to1")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_trm_dual_z_trm_tri_mixer_command_uses_all_three_mixers(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/trm_dual_z_trm_tri_mixer_think"),
            candidate="trm_dual_z_trm_tri_mixer_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertEqual(command[command.index("--think-backbone") + 1], "trm_tri_mixer")
        self.assertEqual(command[command.index("--think-structure") + 1], "trm_dual_z")
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)

    def test_official_fla_encode_decode_mamba3_think_command_keeps_mamba3_core(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/official_fla_encode_decode_mamba3_think"),
            candidate="official_fla_encode_decode_mamba3_think",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertIn("fla_gated_delta", command)
        self.assertEqual(command[command.index("--encode-backbone") + 1], "qtrm_hybrid_3to1")
        self.assertEqual(command[command.index("--think-backbone") + 1], "mamba3")
        self.assertEqual(command[command.index("--decode-backbone") + 1], "qtrm_hybrid_3to1")
        self.assertIn("--strict-backends", command)

    def test_official_mamba3_all_command_uses_mamba3_for_all_stages(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/official_mamba3"),
            candidate="official_mamba3",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertIn("--backbone", command)
        self.assertEqual(command[command.index("--backbone") + 1], "mamba3")
        self.assertNotIn("--encode-backbone", command)
        self.assertNotIn("--think-backbone", command)
        self.assertNotIn("--decode-backbone", command)
        self.assertIn("--strict-backends", command)

    def test_mha_command_keeps_plain_baseline(self):
        module = load_module()

        command = module.compare_command(
            python_bin=".venv/bin/python",
            out_dir=Path("local_eval/l5d/mha"),
            candidate="mha_etd",
            profile="smoke",
            seed=337,
            eval_seed=9337,
        )

        self.assertIn("--backbone", command)
        self.assertIn("mha_etd", command)
        self.assertNotIn("fla_gated_delta", command)
        self.assertNotIn("--strict-backends", command)

    def test_summarize_reports_computes_winner_and_fla_backend_guard(self):
        module = load_module()
        reports = {
            "mha_etd": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.60,
                    "full_minus_think0": 0.20,
                    "full_minus_worst_ablation": 0.20,
                },
            },
            "official_fla": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.55,
                    "full_minus_think0": 0.20,
                    "full_minus_worst_ablation": 0.20,
                },
                "backend_summary": {
                    "official_fla_delta_mixers": 9,
                    "torch_delta_mixers": 0,
                    "all_fla_mixers_official": True,
                },
            },
        }

        summary = module.summarize_reports(reports)

        self.assertEqual(summary["winner"], "mha_etd")
        self.assertEqual(summary["full_exact_delta_official_fla_minus_mha"], -0.05)
        self.assertTrue(summary["official_fla_backend_ok"])
        self.assertTrue(summary["official_fla_causal_ok"])
        self.assertFalse(summary["official_fla_promoted"])

    def test_summarize_reports_promotes_staged_official_fla_think_candidate(self):
        module = load_module()
        reports = {
            "mha_etd": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.50,
                    "full_minus_think0": 0.10,
                    "full_minus_worst_ablation": 0.10,
                },
            },
            "official_fla_think": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.58,
                    "full_minus_think0": 0.12,
                    "full_minus_worst_ablation": 0.11,
                },
                "backend_summary": {
                    "official_fla_delta_mixers": 3,
                    "torch_delta_mixers": 0,
                    "all_fla_mixers_official": True,
                },
            },
        }

        summary = module.summarize_reports(reports)

        self.assertEqual(summary["winner"], "official_fla_think")
        self.assertTrue(summary["candidate_promotions"]["official_fla_think"]["promoted"])
        self.assertEqual(
            summary["candidate_promotions"]["official_fla_think"]["full_exact_delta_vs_mha"],
            0.08,
        )

    def test_summarize_reports_promotes_staged_official_mamba3_think_candidate(self):
        module = load_module()
        reports = {
            "mha_etd": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.50,
                    "full_minus_think0": 0.10,
                    "full_minus_worst_ablation": 0.10,
                },
            },
            "official_mamba3_think": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.59,
                    "full_minus_think0": 0.12,
                    "full_minus_worst_ablation": 0.11,
                },
                "backend_summary": {
                    "official_mamba3_mixers": 1,
                    "all_mamba3_mixers_official": True,
                },
            },
        }

        summary = module.summarize_reports(reports)

        self.assertEqual(summary["winner"], "official_mamba3_think")
        self.assertTrue(summary["candidate_promotions"]["official_mamba3_think"]["promoted"])
        self.assertEqual(
            summary["candidate_promotions"]["official_mamba3_think"]["full_exact_delta_vs_mha"],
            0.09,
        )

    def test_summarize_reports_requires_both_backends_for_fla_mamba3_mixed_candidate(self):
        module = load_module()
        reports = {
            "mha_etd": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.50,
                    "full_minus_think0": 0.10,
                    "full_minus_worst_ablation": 0.10,
                },
            },
            "official_fla_encode_decode_mamba3_think": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.59,
                    "full_minus_think0": 0.12,
                    "full_minus_worst_ablation": 0.11,
                },
                "backend_summary": {
                    "official_fla_delta_mixers": 2,
                    "official_mamba3_mixers": 1,
                    "torch_delta_mixers": 0,
                    "all_fla_mixers_official": True,
                    "all_mamba3_mixers_official": True,
                },
            },
        }

        summary = module.summarize_reports(reports)

        self.assertTrue(
            summary["candidate_promotions"]["official_fla_encode_decode_mamba3_think"]["promoted"]
        )

    def test_summarize_reports_does_not_promote_exact_tie(self):
        module = load_module()
        reports = {
            "mha_etd": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.0,
                    "full_minus_think0": 0.0,
                    "full_minus_worst_ablation": 0.0,
                },
            },
            "official_fla": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.0,
                    "full_minus_think0": 0.0,
                    "full_minus_worst_ablation": 0.0,
                },
                "backend_summary": {
                    "official_fla_delta_mixers": 9,
                    "torch_delta_mixers": 0,
                    "all_fla_mixers_official": True,
                },
            },
        }

        summary = module.summarize_reports(reports)

        self.assertEqual(summary["full_exact_delta_official_fla_minus_mha"], 0.0)
        self.assertTrue(summary["official_fla_backend_ok"])
        self.assertFalse(summary["official_fla_causal_ok"])
        self.assertFalse(summary["official_fla_promoted"])

    def test_summarize_reports_does_not_promote_when_causal_ablation_fails(self):
        module = load_module()
        reports = {
            "mha_etd": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.50,
                    "full_minus_think0": 0.10,
                    "full_minus_worst_ablation": 0.10,
                },
            },
            "official_fla": {
                "accepted": True,
                "decisive_metrics": {
                    "full_generation_exact": 0.52,
                    "full_minus_think0": -0.01,
                    "full_minus_worst_ablation": -0.02,
                },
                "backend_summary": {
                    "official_fla_delta_mixers": 9,
                    "torch_delta_mixers": 0,
                    "all_fla_mixers_official": True,
                },
            },
        }

        summary = module.summarize_reports(reports)

        self.assertGreater(summary["full_exact_delta_official_fla_minus_mha"], 0.0)
        self.assertTrue(summary["official_fla_backend_ok"])
        self.assertFalse(summary["official_fla_causal_ok"])
        self.assertFalse(summary["official_fla_promoted"])

    def test_reuse_existing_reports_summarizes_without_running_training(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            for name, exact in [("mha_etd", 0.50), ("official_fla", 0.52)]:
                run_dir = out_root / name
                run_dir.mkdir()
                (run_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "accepted": True,
                            "decision": "accepted_l5d_compare_runtime",
                            "decisive_metrics": {
                                "full_generation_exact": exact,
                                "full_minus_think0": 0.10,
                                "full_minus_worst_ablation": 0.10,
                            },
                            "backend_summary": {
                                "official_fla_delta_mixers": 9 if name == "official_fla" else 0,
                                "torch_delta_mixers": 0,
                                "all_fla_mixers_official": name == "official_fla",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            args = module.build_arg_parser().parse_args(
                [
                    "--out-root",
                    tmp,
                    "--reuse-existing",
                    "--candidates",
                    "mha_etd,official_fla",
                ]
            )
            summary = module.run_compare(args)

        self.assertEqual(summary["decision"], "completed_l5d_backbone_compare")
        self.assertEqual(summary["winner"], "official_fla")
        self.assertEqual(summary["full_exact_delta_official_fla_minus_mha"], 0.02)

    def test_candidate_parser_limits_run_scope(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--dry-run",
                "--candidates",
                "mha_etd,official_fla_think",
            ]
        )
        summary = module.run_compare(args)

        self.assertEqual(list(summary["commands"].keys()), ["mha_etd", "official_fla_think"])


if __name__ == "__main__":
    unittest.main()
