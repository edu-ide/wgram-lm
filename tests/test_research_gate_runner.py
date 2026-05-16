import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "300_research_gate_runner.py"
    spec = importlib.util.spec_from_file_location("research_gate_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResearchGateRunnerTests(unittest.TestCase):
    def test_list_gates_includes_donorless_depth_gate(self):
        module = _load_module()

        gates = module.list_gates("smoke")

        by_name = {gate["name"]: gate for gate in gates}
        self.assertEqual(by_name["donorless_recurrent_depth"]["target_level"], "L1 scaffold")
        self.assertEqual(by_name["prompt_source_position_binder"]["target_level"], "L1 scaffold")
        self.assertEqual(by_name["prompt_source_position_binder_numeric"]["target_level"], "L1 scaffold")
        self.assertEqual(
            by_name["prompt_source_position_binder_token_plus_numeric"]["target_level"],
            "L1 scaffold",
        )
        self.assertEqual(by_name["qtrm_minimal_depth"]["target_level"], "L2 local gate")
        self.assertEqual(by_name["qtrm_source_pointer_state"]["target_level"], "L2 local gate")
        self.assertEqual(by_name["qtrm_numeric_source_pointer_state"]["target_level"], "L2 local gate")
        self.assertEqual(
            by_name["qtrm_token_numeric_source_pointer_state"]["target_level"],
            "L2 local gate",
        )
        self.assertEqual(by_name["renderer_canonical_lm"]["target_level"], "L3 candidate")
        self.assertEqual(
            by_name["small_general_reasoning"]["target_level"],
            "L2 local gate / L3 candidate",
        )
        self.assertEqual(by_name["qtrm_native_l1_mha"]["target_level"], "L1 native LM path")
        self.assertEqual(
            by_name["qtrm_native_l1_hybrid"]["target_level"],
            "L1 native LM path",
        )
        self.assertEqual(
            by_name["qtrm_native_l2_curriculum_depth"]["target_level"],
            "L2 native recursive gain",
        )
        self.assertEqual(
            by_name["qtrm_native_tiny_lm_first"]["target_level"],
            "L1 native tiny LM first",
        )
        self.assertEqual(
            by_name["qtrm_native_tiny_lm_depth_ablation"]["target_level"],
            "L2 native tiny LM depth ablation",
        )
        self.assertEqual(
            by_name["qtrm_native_l3_language_slice"]["target_level"],
            "L3 native language slice",
        )
        self.assertEqual(
            by_name["qtrm_native_l4_mixed_text_reasoning"]["target_level"],
            "L4 native reasoning + language",
        )
        self.assertEqual(
            by_name["qtrm_native_dual_reverse_l4_baseline_compare"]["target_level"],
            "L4 dual reverse versus single baseline",
        )
        self.assertEqual(
            by_name["qtrm_native_nested_dual_reverse_l4_baseline_compare"]["target_level"],
            "L4 nested dual reverse versus single baseline",
        )
        self.assertEqual(
            by_name[
                "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare"
            ]["target_level"],
            "L4 nested official-schedule split-mixer 3:1 versus single baseline",
        )
        self.assertEqual(
            by_name["qtrm_native_fast_slow_latent_update_l4_repair"]["target_level"],
            "L4 Fast-Slow latent update repair",
        )
        self.assertEqual(
            by_name["qtrm_native_l5_multifamily"]["target_level"],
            "L5 broader reasoning families",
        )
        self.assertEqual(
            by_name["qtrm_native_l5_language_nonregression"]["target_level"],
            "L5C native language non-regression",
        )
        self.assertEqual(
            by_name["qtrm_native_broad_wiki_text_nonregression"]["target_level"],
            "L5C broad native wiki text non-regression",
        )
        self.assertEqual(
            by_name["qtrm_native_broad_wiki_depth_ablation"]["target_level"],
            "L5C broad native wiki depth ablation",
        )
        self.assertEqual(
            by_name["qtrm_native_l5d_official_fla_runtime"]["target_level"],
            "L5D official FLA runtime",
        )
        self.assertEqual(
            by_name["qtrm_native_l5d_placement_seed_stability"]["target_level"],
            "L5D placement seed stability",
        )
        self.assertEqual(
            by_name["qtrm_native_l5d_placement_language_nonregression"]["target_level"],
            "L5D placement language non-regression",
        )
        self.assertEqual(
            by_name["qtrm_native_l5d_placement_scaled_reasoning"]["target_level"],
            "L5D placement scaled reasoning",
        )
        self.assertEqual(
            by_name["qtrm_native_l5d_mamba3_placement_language_nonregression"]["target_level"],
            "L5D Mamba3 placement language non-regression",
        )
        self.assertEqual(
            by_name["qtrm_native_l5d_mamba3_placement_scaled_reasoning"]["target_level"],
            "L5D Mamba3 placement scaled reasoning",
        )
        self.assertEqual(
            by_name["qtrm_native_dual_path_reverse_length_gate"]["target_level"],
            "L5R fixed dual-path reverse length gate",
        )

    def test_gate_command_uses_python_script_and_out_dir(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["donorless_recurrent_depth"]

        command = module.gate_command(gate, "local_eval/example")

        self.assertIn("scripts/260_train_donorless_recurrent_depth_probe.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/example", command)
        self.assertIn("--steps", command)

    def test_qtrm_native_l4_gate_uses_mixed_text_reasoning_probe(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_l4_mixed_text_reasoning"]

        command = module.gate_command(gate, "local_eval/native_l4")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/native_l4", command)
        self.assertIn("--d-model", command)
        self.assertIn("128", command)
        self.assertIn("--active-len-curriculum", command)
        self.assertIn("--depth-intermediate-loss-weight", command)

    def test_qtrm_native_dual_reverse_l4_compare_uses_dual_structure(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_dual_reverse_l4_baseline_compare"]

        command = module.gate_command(gate, "local_eval/native_dual_reverse_l4")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--think-structure", command)
        self.assertIn("trm_dual_z_reversed_mha_etd", command)
        self.assertIn("--trm-l-cycles", command)
        self.assertIn("2", command)
        self.assertIn("--resume-from", command)
        self.assertIn(
            "local_eval/research_gate_runner/qtrm_native_l4_mixed_text_reasoning_standard/last.pt",
            command,
        )
        self.assertIn("--resume-allow-missing", command)
        self.assertIn("--train-only-resume-missing-params", command)
        self.assertIn("--z-l-counterfactual-loss-weight", command)
        self.assertIn("0.05", command)
        self.assertIn("--accept-min-exact", command)
        self.assertIn("0.665", command)
        self.assertIn("accepted_dual_reverse_l4_baseline_compare", command)

    def test_qtrm_native_nested_dual_reverse_l4_compare_uses_nested_update(self):
        module = _load_module()
        gate = module.gate_specs("standard")[
            "qtrm_native_nested_dual_reverse_l4_baseline_compare"
        ]

        command = module.gate_command(gate, "local_eval/native_nested_dual_reverse_l4")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--think-structure", command)
        self.assertIn("trm_dual_z_nested_reversed_mha_etd", command)
        self.assertIn("--resume-allow-missing", command)
        self.assertIn("--train-only-resume-missing-params", command)
        self.assertIn("--z-l-counterfactual-loss-weight", command)
        self.assertIn("0.10", command)
        self.assertIn("--z-l-counterfactual-margin", command)
        self.assertIn("0.15", command)
        self.assertIn("accepted_nested_dual_reverse_l4_baseline_compare", command)

    def test_qtrm_native_nested_official_schedule_split_mixer_3to1_gate_uses_h3_l6_split_path(self):
        module = _load_module()
        gate = module.gate_specs("standard")[
            "qtrm_native_nested_official_schedule_split_mixer_3to1_l4_baseline_compare"
        ]

        command = module.gate_command(
            gate,
            "local_eval/native_nested_official_schedule_split_mixer_3to1_l4",
        )

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--think-structure", command)
        self.assertIn("trm_dual_z_nested_official_schedule_split_mixer_3to1", command)
        self.assertIn("--trm-l-cycles", command)
        self.assertIn("6", command)
        self.assertIn("--train-think-steps", command)
        self.assertIn("--eval-think-steps", command)
        self.assertIn("3", command)
        self.assertIn("--resume-allow-missing", command)
        self.assertIn("--train-only-resume-missing-params", command)
        self.assertIn("--z-l-counterfactual-loss-weight", command)
        self.assertIn("0.10", command)
        self.assertIn("--z-l-counterfactual-margin", command)
        self.assertIn("0.15", command)
        self.assertIn(
            "accepted_nested_official_schedule_split_mixer_3to1_l4_baseline_compare",
            command,
        )

    def test_qtrm_native_fast_slow_latent_update_gate_adds_fast_slow_loss(self):
        module = _load_module()
        gate = module.gate_specs("standard")[
            "qtrm_native_fast_slow_latent_update_l4_repair"
        ]

        command = module.gate_command(
            gate,
            "local_eval/native_fast_slow_latent_update_l4",
        )

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("trm_dual_z_nested_official_schedule_split_mixer_3to1", command)
        self.assertIn("--trm-l-cycles", command)
        self.assertIn("6", command)
        self.assertIn("--train-think-steps", command)
        self.assertIn("3", command)
        self.assertIn("--fast-slow-latent-loss-weight", command)
        self.assertIn("0.15", command)
        self.assertIn("--fast-slow-z-l-margin", command)
        self.assertIn("0.20", command)
        self.assertIn("--fast-slow-z-h-margin", command)
        self.assertIn("0.05", command)
        self.assertIn("--fast-slow-z-l-weight", command)
        self.assertIn("2.0", command)
        self.assertIn("--fast-slow-z-h-weight", command)
        self.assertIn("0.5", command)
        self.assertIn("accepted_fast_slow_latent_update_l4_repair", command)

    def test_qtrm_native_tiny_lm_first_gate_uses_text_probe(self):
        module = _load_module()
        gate = module.gate_specs("triage")["qtrm_native_tiny_lm_first"]

        command = module.gate_command(gate, "local_eval/native_tiny_lm_first")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/native_tiny_lm_first", command)
        self.assertIn("accepted_native_tiny_lm_first", command)
        self.assertIn("--max-full-vs-think0-loss-ratio", command)

    def test_qtrm_native_tiny_lm_depth_gate_uses_depth_sweep(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_tiny_lm_depth_ablation"]

        command = module.gate_command(gate, "local_eval/native_tiny_lm_depth")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--eval-depth-sweep", command)
        self.assertIn("0,1,2,4", command)
        self.assertIn("accepted_native_tiny_lm_depth_ablation", command)

    def test_qtrm_native_l5_gate_uses_multifamily_text_reasoning_probe(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_l5_multifamily"]

        command = module.gate_command(gate, "local_eval/native_l5")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--task-families", command)
        self.assertIn("modchain,revchain,modchain,revchain,checksum", command)
        self.assertIn("--eval-task-families", command)
        self.assertIn("modchain,revchain,checksum", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5_multifamily", command)
        self.assertIn("--accept-min-family-exact", command)

    def test_qtrm_native_l5_language_gate_uses_text_nonregression_probe(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_l5_language_nonregression"]

        command = module.gate_command(gate, "local_eval/native_l5c")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5_language_nonregression", command)
        self.assertIn("--max-full-vs-think0-loss-ratio", command)
        self.assertIn("--max-full-vs-off-loss-ratio", command)
        self.assertIn("--baseline-steps", command)
        self.assertIn("--max-full-vs-baseline-loss-ratio", command)

    def test_qtrm_native_l5_language_gate_has_triage_profile(self):
        module = _load_module()
        gate = module.gate_specs("triage")["qtrm_native_l5_language_nonregression"]

        command = module.gate_command(gate, "local_eval/native_l5_language_triage")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--text-file", command)
        self.assertIn("docs/wiki/decisions/qtrm-native-hard-lock.md", command)
        self.assertIn("--baseline-steps", command)

    def test_qtrm_native_broad_wiki_text_gate_uses_globs(self):
        module = _load_module()
        gate = module.gate_specs("triage")["qtrm_native_broad_wiki_text_nonregression"]

        command = module.gate_command(gate, "local_eval/native_broad_wiki_text")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--text-glob", command)
        self.assertIn("docs/wiki/decisions/*.md", command)
        self.assertIn("docs/wiki/architecture/*.md", command)
        self.assertIn("accepted_broad_wiki_text_nonregression", command)

    def test_qtrm_native_broad_wiki_depth_gate_uses_depth_sweep(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_broad_wiki_depth_ablation"]

        command = module.gate_command(gate, "local_eval/native_broad_wiki_depth")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--text-glob", command)
        self.assertIn("docs/wiki/**/*.md", command)
        self.assertIn("--eval-depth-sweep", command)
        self.assertIn("0,1,2,4", command)
        self.assertIn("--max-full-vs-best-shallow-loss-ratio", command)
        self.assertIn("accepted_broad_wiki_depth_ablation", command)

    def test_qtrm_native_l5d_official_fla_gate_uses_strict_backend(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_native_l5d_official_fla_runtime"]

        command = module.gate_command(gate, "local_eval/native_l5d_official")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--backbone", command)
        self.assertIn("qtrm_hybrid_3to1", command)
        self.assertIn("--delta-backend", command)
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)
        self.assertIn("--delta-head-dim", command)
        self.assertIn("--delta-num-v-heads", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5d_official_fla_runtime", command)

    def test_qtrm_native_l5d_placement_seed_gate_uses_seed_sweep(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_l5d_placement_seed_stability"]

        command = module.gate_command(gate, "local_eval/native_l5d_seed")

        self.assertIn("scripts/343_qtrm_native_l5d_placement_seed_sweep.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/native_l5d_seed", command)
        self.assertIn("--candidates", command)
        self.assertIn("mha_etd,official_fla_think", command)
        self.assertIn("--target-candidate", command)
        self.assertIn("official_fla_think", command)
        self.assertIn("--seeds", command)
        self.assertIn("337", command)
        self.assertIn("339", command)

    def test_qtrm_native_l5d_placement_language_gate_uses_staged_fla_think(self):
        module = _load_module()
        gate = module.gate_specs("standard")[
            "qtrm_native_l5d_placement_language_nonregression"
        ]

        command = module.gate_command(gate, "local_eval/native_l5d_language")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--backbone", command)
        self.assertIn("qtrm_hybrid_3to1", command)
        self.assertIn("--encode-backbone", command)
        self.assertIn("mha_etd", command)
        self.assertIn("--think-backbone", command)
        self.assertIn("--decode-backbone", command)
        self.assertIn("--delta-backend", command)
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5d_placement_language_nonregression", command)
        self.assertIn("--max-full-vs-baseline-loss-ratio", command)

    def test_qtrm_native_l5d_scaled_reasoning_gate_uses_staged_fla_think(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_l5d_placement_scaled_reasoning"]

        command = module.gate_command(gate, "local_eval/native_l5d_scaled")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--backbone", command)
        self.assertIn("qtrm_hybrid_3to1", command)
        self.assertIn("--encode-backbone", command)
        self.assertIn("mha_etd", command)
        self.assertIn("--think-backbone", command)
        self.assertIn("--decode-backbone", command)
        self.assertIn("--delta-backend", command)
        self.assertIn("fla_gated_delta", command)
        self.assertIn("--strict-backends", command)
        self.assertIn("--steps", command)
        self.assertIn("1200", command)
        self.assertIn("--eval-cases", command)
        self.assertIn("384", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5d_placement_scaled_reasoning", command)

    def test_qtrm_native_l5d_mamba3_language_gate_uses_staged_mamba3_think(self):
        module = _load_module()
        gate = module.gate_specs("standard")[
            "qtrm_native_l5d_mamba3_placement_language_nonregression"
        ]

        command = module.gate_command(gate, "local_eval/native_l5d_mamba3_language")

        self.assertIn("scripts/336_train_qtrm_native_text_probe.py", command)
        self.assertIn("--backbone", command)
        self.assertIn("mamba3", command)
        self.assertIn("--encode-backbone", command)
        self.assertIn("mha_etd", command)
        self.assertIn("--think-backbone", command)
        self.assertIn("--decode-backbone", command)
        self.assertIn("--strict-backends", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5d_mamba3_placement_language_nonregression", command)

    def test_qtrm_native_l5d_mamba3_scaled_gate_uses_staged_mamba3_think(self):
        module = _load_module()
        gate = module.gate_specs("standard")[
            "qtrm_native_l5d_mamba3_placement_scaled_reasoning"
        ]

        command = module.gate_command(gate, "local_eval/native_l5d_mamba3_scaled")

        self.assertIn("scripts/337_train_qtrm_native_mixed_text_reasoning_probe.py", command)
        self.assertIn("--backbone", command)
        self.assertIn("mamba3", command)
        self.assertIn("--encode-backbone", command)
        self.assertIn("mha_etd", command)
        self.assertIn("--think-backbone", command)
        self.assertIn("--decode-backbone", command)
        self.assertIn("--strict-backends", command)
        self.assertIn("--d-model", command)
        self.assertIn("64", command)
        self.assertIn("--d-ff", command)
        self.assertIn("128", command)
        self.assertIn("--accepted-decision", command)
        self.assertIn("accepted_l5d_mamba3_placement_scaled_reasoning", command)

    def test_qtrm_native_dual_path_reverse_gate_uses_fixed_candidate_wrapper(self):
        module = _load_module()
        gate = module.gate_specs("standard")["qtrm_native_dual_path_reverse_length_gate"]

        command = module.gate_command(gate, "local_eval/native_dual_path_reverse")

        self.assertIn("scripts/352_run_qtrm_native_dual_path_reverse_gate.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/native_dual_path_reverse", command)
        self.assertIn("--candidates", command)
        self.assertIn("official,dual_path_reverse", command)
        self.assertIn("--lengths", command)
        self.assertIn("4,6,8", command)

    def test_qtrm_native_decisive_metrics_include_generation_and_language_metrics(self):
        module = _load_module()

        metrics = module.decisive_metrics(
            {
                "eval_metrics": {
                    "think4": {"generation_exact": 0.75},
                    "think0": {"generation_exact": 0.05},
                    "state_reset": {"generation_exact": 0.04},
                    "op_zero": {"generation_exact": 0.03},
                    "think_eval_loss": 0.12,
                    "think0_loss": 3.4,
                    "thinking_block_off_loss": 3.5,
                    "sample_degeneracy": {
                        "unique_chars": 20,
                        "max_run_fraction": 0.02,
                    },
                },
                "full_minus_think0": 0.70,
                "full_minus_worst_ablation": 0.71,
                "decisive_metrics": {
                    "min_family_generation_exact": 0.51,
                    "active_rows": 1,
                    "min_active_full_generation_exact": 0.75,
                    "min_active_full_minus_think0": 0.70,
                    "min_active_full_minus_worst_ablation": 0.71,
                    "min_active_target_len_generation_exact": 0.25,
                    "min_active_minus_official": 0.01,
                    "min_active_target_len_minus_official": 0.02,
                },
                "promoted_count": 3,
                "promoted_rate": 1.0,
                "causal_ok_count": 3,
                "backend_ok_count": 3,
                "min_delta_vs_mha": 0.01,
                "min_full_generation_exact": 0.05,
            }
        )

        self.assertEqual(metrics["eval_metrics.think4.generation_exact"], 0.75)
        self.assertEqual(metrics["eval_metrics.think0.generation_exact"], 0.05)
        self.assertEqual(metrics["eval_metrics.state_reset.generation_exact"], 0.04)
        self.assertEqual(metrics["eval_metrics.op_zero.generation_exact"], 0.03)
        self.assertEqual(metrics["eval_metrics.think_eval_loss"], 0.12)
        self.assertEqual(metrics["eval_metrics.sample_degeneracy.unique_chars"], 20)
        self.assertEqual(metrics["full_minus_think0"], 0.70)
        self.assertEqual(metrics["decisive_metrics.min_family_generation_exact"], 0.51)
        self.assertEqual(metrics["decisive_metrics.active_rows"], 1)
        self.assertEqual(metrics["decisive_metrics.min_active_full_generation_exact"], 0.75)
        self.assertEqual(
            metrics["decisive_metrics.min_active_target_len_generation_exact"], 0.25
        )
        self.assertEqual(metrics["decisive_metrics.min_active_minus_official"], 0.01)
        self.assertEqual(metrics["promoted_count"], 3)
        self.assertEqual(metrics["min_delta_vs_mha"], 0.01)

    def test_source_pointer_state_gate_uses_refresh_script(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_source_pointer_state"]

        command = module.gate_command(gate, "local_eval/source_pointer")

        self.assertIn("scripts/319_run_qtrm_source_pointer_state_gate.py", command)
        self.assertIn("--out-dir", command)
        self.assertIn("local_eval/source_pointer", command)
        self.assertIn("--min-value-drop", command)

    def test_numeric_source_pointer_state_gate_enables_numeric_ablation(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_numeric_source_pointer_state"]

        command = module.gate_command(gate, "local_eval/numeric_source_pointer")

        self.assertIn("scripts/319_run_qtrm_source_pointer_state_gate.py", command)
        self.assertIn("--numeric-source-features", command)
        self.assertIn("--min-numeric-value-drop", command)
        self.assertIn("local_eval/numeric_source_pointer", command)

    def test_token_numeric_source_pointer_state_gate_enables_token_numeric_ablation(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["qtrm_token_numeric_source_pointer_state"]

        command = module.gate_command(gate, "local_eval/token_numeric_source_pointer")

        self.assertIn("scripts/319_run_qtrm_source_pointer_state_gate.py", command)
        self.assertIn("--token-numeric-value-features", command)
        self.assertIn("--min-token-numeric-value-drop", command)
        self.assertIn("local_eval/token_numeric_source_pointer", command)

    def test_prompt_source_position_binder_gate_uses_binder_script(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["prompt_source_position_binder"]

        command = module.gate_command(gate, "local_eval/prompt_binder")

        self.assertIn("scripts/320_train_prompt_source_position_binder_probe.py", command)
        self.assertIn("--train-jsonl", command)
        self.assertIn("--eval-jsonl", command)
        self.assertIn("local_eval/prompt_binder", command)

    def test_numeric_source_position_binder_gate_uses_numeric_input(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["prompt_source_position_binder_numeric"]

        command = module.gate_command(gate, "local_eval/numeric_binder")

        self.assertIn("scripts/320_train_prompt_source_position_binder_probe.py", command)
        self.assertIn("--input-source", command)
        self.assertIn("numeric_value_embedding", command)
        self.assertIn("local_eval/numeric_binder", command)

    def test_token_plus_numeric_source_position_binder_gate_uses_canonical_token_input(self):
        module = _load_module()
        gate = module.gate_specs("smoke")[
            "prompt_source_position_binder_token_plus_numeric"
        ]

        command = module.gate_command(gate, "local_eval/token_plus_numeric_binder")

        self.assertIn("scripts/320_train_prompt_source_position_binder_probe.py", command)
        self.assertIn("--input-source", command)
        self.assertIn("token_plus_numeric_value", command)
        self.assertIn("local_eval/token_plus_numeric_binder", command)

    def test_normalize_and_accept_decision(self):
        module = _load_module()
        gate = module.gate_specs("smoke")["donorless_recurrent_depth"]

        self.assertEqual(module.normalize_decision({"decision": "accepted_l1"}), "accepted_l1")
        self.assertTrue(module.is_accepted({"decision": "accepted_l1"}, gate))
        self.assertFalse(module.is_accepted({"decision": "rejected"}, gate))

    def test_dry_run_writes_gate_summary(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            summary = module.run_gate(
                gate_name="donorless_recurrent_depth",
                profile="smoke",
                out_root=tmp,
                dry_run=True,
            )
            summary_path = Path(summary["out_dir"]) / "gate_summary.json"
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["decision"], "dry_run")
        self.assertFalse(summary["accepted"])
        self.assertEqual(loaded["gate"], "donorless_recurrent_depth")

    def test_skip_existing_report_reuses_decision(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "existing"
            out_dir.mkdir()
            (out_dir / "report.json").write_text(
                json.dumps(
                    {
                        "decision": "accepted_l1",
                        "eval_metrics": {"depth8_final_exact": 1.0, "depth1_final_exact": 0.0},
                        "ablations": {"state_reset": {"depth8_final_exact": 0.0}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = module.run_gate(
                gate_name="donorless_recurrent_depth",
                profile="smoke",
                out_root=tmp,
                out_dir=out_dir,
                skip_existing=True,
            )

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["decision"], "accepted_l1")
        self.assertEqual(summary["decisive_metrics"]["eval_metrics.depth8_final_exact"], 1.0)

    def test_nonzero_gate_exit_still_uses_report_when_report_exists(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_rejected_gate.py"
            script.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "from pathlib import Path",
                        "out_dir = Path(sys.argv[sys.argv.index('--out-dir') + 1])",
                        "out_dir.mkdir(parents=True, exist_ok=True)",
                        "(out_dir / 'report.json').write_text(json.dumps({'decision': 'rejected'}) + '\\n')",
                        "raise SystemExit(1)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_gate = module.GateSpec(
                name="fake_rejected",
                target_level="fake",
                major_bottleneck="fake",
                script=str(script),
                default_args=(),
                report_name="report.json",
                wiki_path="docs/wiki/fake.md",
                accepted_decisions=("accepted",),
                on_accept="accepted",
                on_reject="rejected next",
            )
            original_gate_specs = module.gate_specs
            try:
                module.gate_specs = lambda profile: {"fake_rejected": fake_gate}
                summary = module.run_gate(
                    gate_name="fake_rejected",
                    profile="smoke",
                    out_root=tmp,
                )
            finally:
                module.gate_specs = original_gate_specs

        self.assertEqual(summary["exit_code"], 1)
        self.assertEqual(summary["decision"], "rejected")
        self.assertFalse(summary["accepted"])

    def test_operation_ledger_records_autoresearch_keep_discard_status(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "existing"
            out_dir.mkdir()
            (out_dir / "report.json").write_text(
                json.dumps(
                    {
                        "decision": "accepted_l1",
                        "eval_metrics": {"depth8_final_exact": 0.75},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ledger_path = Path(tmp) / "results.tsv"
            summary = module.run_gate(
                gate_name="donorless_recurrent_depth",
                profile="smoke",
                out_root=tmp,
                out_dir=out_dir,
                skip_existing=True,
                operation_ledger=ledger_path,
            )
            rows = ledger_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(summary["accepted"])
        self.assertEqual(rows[0].split("\t")[:5], ["timestamp", "gate", "profile", "decision", "status"])
        self.assertIn("\taccepted_l1\tkeep\teval_metrics.depth8_final_exact\t0.75\t", rows[1])

    def test_operation_ledger_marks_rejected_run_as_discard(self):
        module = _load_module()

        summary = {
            "timestamp": "2026-05-14T12:00:00",
            "gate": "qtrm_native_l5_multifamily",
            "profile": "standard",
            "decision": "rejected",
            "accepted": False,
            "out_dir": "local_eval/example",
            "report_path": "local_eval/example/report.json",
            "next_action": "fix recurrent drift",
            "decisive_metrics": {
                "decisive_metrics.full_generation_exact": 0.21,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "results.tsv"
            module.append_operation_ledger(ledger_path, summary)
            rows = ledger_path.read_text(encoding="utf-8").splitlines()

        self.assertIn(
            "\trejected\tdiscard\tdecisive_metrics.full_generation_exact\t0.21\t",
            rows[1],
        )


if __name__ == "__main__":
    unittest.main()
