from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path


class OneBodyArchitectureContractTests(unittest.TestCase):
    def test_default_contract_allows_clean_one_body_run(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        args = argparse.Namespace(
            answer_readback_mode="none",
            cot_anchor_loss_weight=0.0,
            workspace_selector_critic_weight=0.0,
            workspace_selector_final_ce_critic_weight=0.0,
            allow_diagnostic_bridge_experiment=False,
        )

        validate_one_body_architecture_contract(args)

    def test_bridge_experiment_requires_explicit_diagnostic_opt_in(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        args = argparse.Namespace(
            answer_readback_mode="anchor_embedding",
            cot_anchor_loss_weight=0.0,
            workspace_selector_critic_weight=0.0,
            workspace_selector_final_ce_critic_weight=0.0,
            allow_diagnostic_bridge_experiment=False,
        )

        with self.assertRaisesRegex(ValueError, "Stage99-style diagnostic bridge"):
            validate_one_body_architecture_contract(args)

    def test_contract_reports_enabled_bridge_fields(self) -> None:
        from qtrm_mm.architecture.one_body_contract import collect_bridge_contract_fields

        args = argparse.Namespace(
            answer_readback_mode="none",
            cot_anchor_loss_weight=0.25,
            workspace_selector_critic_weight=0.0,
            workspace_selector_final_ce_critic_weight=0.1,
        )

        fields = collect_bridge_contract_fields(args)

        self.assertEqual(fields.enabled_field_names(), ("cot_anchor_loss_weight", "workspace_selector_final_ce_critic_weight"))

    def test_short_one_body_gate_does_not_require_past_success_report(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        args = argparse.Namespace(
            answer_readback_mode="none",
            cot_anchor_loss_weight=0.0,
            workspace_selector_critic_weight=0.0,
            workspace_selector_final_ce_critic_weight=0.0,
            allow_diagnostic_bridge_experiment=False,
            decoder_latent_mode="one_body",
            steps=400,
            past_success_preflight_min_steps=1000,
            past_success_report_json="",
            allow_missing_past_success_preflight=False,
            acknowledge_past_success_restoration_gap=False,
        )

        validate_one_body_architecture_contract(args)

    def test_long_one_body_run_requires_past_success_report(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        args = argparse.Namespace(
            answer_readback_mode="none",
            cot_anchor_loss_weight=0.0,
            workspace_selector_critic_weight=0.0,
            workspace_selector_final_ce_critic_weight=0.0,
            allow_diagnostic_bridge_experiment=False,
            decoder_latent_mode="one_body",
            steps=1200,
            past_success_preflight_min_steps=1000,
            past_success_report_json="",
            allow_missing_past_success_preflight=False,
            acknowledge_past_success_restoration_gap=False,
        )

        with self.assertRaisesRegex(ValueError, "past-success preflight report"):
            validate_one_body_architecture_contract(args)

    def test_report_that_blocks_long_run_requires_explicit_acknowledgement(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "report_type": "past_success_doubt_loop",
                        "recommended_comparison_row": {
                            "old_success": "Stage58B",
                            "exact_metric": "selected=0.9336, oracle=0.9401",
                            "causal_ingredient": "candidate diversity plus verifier-selected compact answers",
                            "missing_in_current_run": "free generation samples",
                            "smallest_restoration_test": "small one-body restoration gate",
                        },
                        "launch_recommendation": "do_not_launch_long_run_until_restoration_gate_exists",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                answer_readback_mode="none",
                cot_anchor_loss_weight=0.0,
                workspace_selector_critic_weight=0.0,
                workspace_selector_final_ce_critic_weight=0.0,
                allow_diagnostic_bridge_experiment=False,
                decoder_latent_mode="one_body",
                steps=1200,
                past_success_preflight_min_steps=1000,
                past_success_report_json=str(report_path),
                allow_missing_past_success_preflight=False,
                acknowledge_past_success_restoration_gap=False,
            )

            with self.assertRaisesRegex(ValueError, "restoration gate gap"):
                validate_one_body_architecture_contract(args)

            args.acknowledge_past_success_restoration_gap = True
            validate_one_body_architecture_contract(args)

    def test_restoration_gate_satisfies_report_gap_without_override(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            gate_path = root / "restoration_gate.json"
            report_path.write_text(
                json.dumps(
                    {
                        "report_type": "past_success_doubt_loop",
                        "recommended_comparison_row": {
                            "old_success": "Stage58B",
                            "exact_metric": "selected=0.9336, oracle=0.9401",
                            "causal_ingredient": "candidate diversity plus verifier-selected compact answers",
                            "missing_in_current_run": "free generation samples",
                            "smallest_restoration_test": "small one-body restoration gate",
                        },
                        "launch_recommendation": "do_not_launch_long_run_until_restoration_gate_exists",
                    }
                ),
                encoding="utf-8",
            )
            gate_path.write_text(
                json.dumps(
                    {
                        "gate_type": "past_success_restoration_gate",
                        "all_required_signals_present": True,
                        "launch_recommendation": "restoration_gate_exists_review_metrics",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                answer_readback_mode="none",
                cot_anchor_loss_weight=0.0,
                workspace_selector_critic_weight=0.0,
                workspace_selector_final_ce_critic_weight=0.0,
                allow_diagnostic_bridge_experiment=False,
                decoder_latent_mode="one_body",
                steps=1200,
                past_success_preflight_min_steps=1000,
                past_success_report_json=str(report_path),
                past_success_restoration_gate_json=str(gate_path),
                allow_missing_past_success_preflight=False,
                acknowledge_past_success_restoration_gap=False,
            )

            validate_one_body_architecture_contract(args)

    def test_invalid_restoration_gate_does_not_satisfy_report_gap(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            gate_path = root / "restoration_gate.json"
            report_path.write_text(
                json.dumps(
                    {
                        "report_type": "past_success_doubt_loop",
                        "recommended_comparison_row": {
                            "old_success": "Stage58B",
                            "exact_metric": "selected=0.9336, oracle=0.9401",
                            "causal_ingredient": "candidate diversity plus verifier-selected compact answers",
                            "missing_in_current_run": "free generation samples",
                            "smallest_restoration_test": "small one-body restoration gate",
                        },
                        "launch_recommendation": "do_not_launch_long_run_until_restoration_gate_exists",
                    }
                ),
                encoding="utf-8",
            )
            gate_path.write_text(
                json.dumps(
                    {
                        "gate_type": "past_success_restoration_gate",
                        "all_required_signals_present": False,
                        "missing_required_signals": ["free_generation_samples"],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                answer_readback_mode="none",
                cot_anchor_loss_weight=0.0,
                workspace_selector_critic_weight=0.0,
                workspace_selector_final_ce_critic_weight=0.0,
                allow_diagnostic_bridge_experiment=False,
                decoder_latent_mode="one_body",
                steps=1200,
                past_success_preflight_min_steps=1000,
                past_success_report_json=str(report_path),
                past_success_restoration_gate_json=str(gate_path),
                allow_missing_past_success_preflight=False,
                acknowledge_past_success_restoration_gap=False,
            )

            with self.assertRaisesRegex(ValueError, "restoration gate gap"):
                validate_one_body_architecture_contract(args)

    def test_observable_but_rejected_restoration_gate_does_not_satisfy_report_gap(self) -> None:
        from qtrm_mm.architecture.one_body_contract import validate_one_body_architecture_contract

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            gate_path = root / "restoration_gate.json"
            report_path.write_text(
                json.dumps(
                    {
                        "report_type": "past_success_doubt_loop",
                        "recommended_comparison_row": {
                            "old_success": "Stage58B",
                            "exact_metric": "selected=0.9336, oracle=0.9401",
                            "causal_ingredient": "candidate diversity plus verifier-selected compact answers",
                            "missing_in_current_run": "free generation samples",
                            "smallest_restoration_test": "small one-body restoration gate",
                        },
                        "launch_recommendation": "do_not_launch_long_run_until_restoration_gate_exists",
                    }
                ),
                encoding="utf-8",
            )
            gate_path.write_text(
                json.dumps(
                    {
                        "gate_type": "past_success_restoration_gate",
                        "all_required_signals_present": True,
                        "current_checkpoint_recommendation": "do_not_promote_current_checkpoint",
                        "metric_warnings": ["free_generation_exact_zero"],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                answer_readback_mode="none",
                cot_anchor_loss_weight=0.0,
                workspace_selector_critic_weight=0.0,
                workspace_selector_final_ce_critic_weight=0.0,
                allow_diagnostic_bridge_experiment=False,
                decoder_latent_mode="one_body",
                steps=1200,
                past_success_preflight_min_steps=1000,
                past_success_report_json=str(report_path),
                past_success_restoration_gate_json=str(gate_path),
                allow_missing_past_success_preflight=False,
                acknowledge_past_success_restoration_gap=False,
            )

            with self.assertRaisesRegex(ValueError, "restoration gate gap"):
                validate_one_body_architecture_contract(args)


if __name__ == "__main__":
    unittest.main()
