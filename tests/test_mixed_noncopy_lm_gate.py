import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest


def load_module():
    path = Path("scripts/330_run_mixed_noncopy_lm_gate.py")
    spec = importlib.util.spec_from_file_location("mixed_noncopy_lm_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MixedNoncopyLmGateTests(unittest.TestCase):
    def test_summarize_generation_counts_hits_by_mode(self):
        module = load_module()
        rows = [
            {"mode": "donor_only_no_evidence", "hit": False},
            {"mode": "qtrm_core_off_no_evidence", "hit": False},
            {"mode": "qtrm_core_steps_8_no_evidence", "hit": True},
            {"mode": "qtrm_core_steps_8_no_evidence", "normalized_exact": True},
        ]

        summary = module.summarize_generation(rows)

        self.assertEqual(summary["qtrm_core_steps_8_no_evidence"]["hits"], 1)
        self.assertEqual(summary["qtrm_core_steps_8_no_evidence"]["loose_hits"], 2)
        self.assertEqual(summary["qtrm_core_steps_8_no_evidence"]["total"], 2)
        self.assertEqual(
            summary["qtrm_core_steps_8_no_evidence"]["accuracy"],
            0.5,
        )

    def test_build_report_rejects_zero_hit_full_model(self):
        module = load_module()
        rows = []
        for mode in module.DEFAULT_MODES:
            for index in range(2):
                rows.append({"id": str(index), "mode": mode, "hit": False})

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.10,
            min_donor_margin=0.01,
            min_core_off_margin=0.01,
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(report["decision"], "rejected_noncopy_lm_gate")
        self.assertIn("full_generation_accuracy_below_min", report["reject_reasons"])

    def test_build_report_accepts_only_with_baseline_gain_and_ablation_drops(self):
        module = load_module()
        rows = []
        mode_hits = {
            module.DONOR_MODE: [False, False, False, False],
            module.CORE_OFF_MODE: [False, False, False, False],
            module.FULL_MODE: [True, True, False, False],
            module.PRIMITIVE_OFF_MODE: [False, False, False, False],
            module.SOURCE_SLOT_OFF_MODE: [False, False, False, False],
            module.SOURCE_BINDER_OFF_MODE: [False, False, False, False],
            module.BRIDGE_OFF_MODE: [False, False, False, False],
            module.TYPED_VALUE_BRIDGE_OFF_MODE: [False, False, False, False],
            module.VOCAB_RENDERER_OFF_MODE: [False, False, False, False],
            module.CORE_STATE_ZERO_MODE: [False, False, False, False],
            module.ANSWER_RECURRENT_OFF_MODE: [False, False, False, False],
            module.ANSWER_NEXT_TOKEN_DECODER_OFF_MODE: [False, False, False, False],
        }
        for mode, hits in mode_hits.items():
            for index, hit in enumerate(hits):
                rows.append(
                    {
                        "id": f"{mode}-{index}",
                        "mode": mode,
                        "normalized_exact": hit,
                    }
                )

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.25,
            min_donor_margin=0.10,
            min_core_off_margin=0.10,
            min_primitive_drop=0.10,
            min_source_slot_drop=0.10,
            min_source_binder_drop=0.10,
            min_bridge_drop=0.10,
            min_typed_value_bridge_drop=0.10,
            min_vocab_renderer_drop=0.10,
            min_answer_recurrent_drop=0.10,
            min_answer_next_token_decoder_drop=0.10,
        )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["decision"], "accepted_l4_sufficient_noncopy_gate")
        self.assertEqual(report["reject_reasons"], [])

    def test_build_report_rejects_without_path_ablation_drop(self):
        module = load_module()
        rows = []
        mode_hits = {
            module.DONOR_MODE: [False, False, False, False],
            module.CORE_OFF_MODE: [False, False, False, False],
            module.FULL_MODE: [True, True, False, False],
            module.PRIMITIVE_OFF_MODE: [True, True, False, False],
            module.SOURCE_SLOT_OFF_MODE: [False, False, False, False],
            module.SOURCE_BINDER_OFF_MODE: [False, False, False, False],
            module.BRIDGE_OFF_MODE: [False, False, False, False],
            module.TYPED_VALUE_BRIDGE_OFF_MODE: [False, False, False, False],
            module.VOCAB_RENDERER_OFF_MODE: [False, False, False, False],
            module.CORE_STATE_ZERO_MODE: [False, False, False, False],
            module.ANSWER_RECURRENT_OFF_MODE: [False, False, False, False],
            module.ANSWER_NEXT_TOKEN_DECODER_OFF_MODE: [False, False, False, False],
        }
        for mode, hits in mode_hits.items():
            for index, hit in enumerate(hits):
                rows.append(
                    {
                        "id": f"{mode}-{index}",
                        "mode": mode,
                        "normalized_exact": hit,
                    }
                )

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.25,
            min_donor_margin=0.10,
            min_core_off_margin=0.10,
            min_primitive_drop=0.10,
            min_source_slot_drop=0.10,
            min_source_binder_drop=0.10,
            min_bridge_drop=0.10,
            min_typed_value_bridge_drop=0.10,
            min_vocab_renderer_drop=0.10,
            min_answer_recurrent_drop=0.10,
            min_answer_next_token_decoder_drop=0.10,
        )

        self.assertFalse(report["accepted"])
        self.assertIn("primitive_off_drop_below_min", report["reject_reasons"])

    def test_build_report_rejects_when_required_ablation_modes_are_missing(self):
        module = load_module()
        rows = []
        mode_hits = {
            module.DONOR_MODE: [False, False, False, False],
            module.CORE_OFF_MODE: [False, False, False, False],
            module.FULL_MODE: [True, True, False, False],
        }
        for mode, hits in mode_hits.items():
            for index, hit in enumerate(hits):
                rows.append(
                    {
                        "id": f"{mode}-{index}",
                        "mode": mode,
                        "normalized_exact": hit,
                    }
                )

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.25,
            min_donor_margin=0.10,
            min_core_off_margin=0.10,
            min_primitive_drop=0.10,
            min_source_slot_drop=0.10,
            min_source_binder_drop=0.10,
            min_bridge_drop=0.10,
            min_typed_value_bridge_drop=0.10,
            min_vocab_renderer_drop=0.10,
            min_answer_recurrent_drop=0.10,
        )

        self.assertFalse(report["accepted"])
        self.assertIn("missing_required_modes", report["reject_reasons"])
        self.assertIn(module.PRIMITIVE_OFF_MODE, report["missing_required_modes"])

    def test_output_is_complete_requires_expected_row_count(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                "\n".join(json.dumps({"mode": "m", "hit": False}) for _ in range(3))
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(module.output_is_complete(path, expected_rows=3))
            self.assertFalse(module.output_is_complete(path, expected_rows=4))

    def test_missing_checkpoint_base_chain_reports_relative_missing_base(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head = root / "head.pt"
            head.write_text("placeholder", encoding="utf-8")

            def load_state(path):
                self.assertEqual(path, head)
                return {"base_checkpoint": "local_eval/missing_base.pt"}

            missing = module.missing_checkpoint_base_chain(
                head,
                root=root,
                load_state=load_state,
            )

            self.assertEqual(missing, [str(root / "local_eval/missing_base.pt")])

    def test_missing_checkpoint_base_chain_accepts_self_contained_checkpoint(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head = root / "head.pt"
            head.write_text("placeholder", encoding="utf-8")

            missing = module.missing_checkpoint_base_chain(
                head,
                root=root,
                load_state=lambda _path: {"model": {}},
            )

            self.assertEqual(missing, [])

    def test_eval_command_keeps_noncopy_contract_minimal(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "ckpt.pt",
                "--cases",
                "cases.jsonl",
                "--out-dir",
                "out",
            ]
        )

        command = module.eval_command(
            args,
            mode="qtrm_core_steps_8_no_evidence",
            cases_path=Path("chunk.jsonl"),
            out_path=Path("out.jsonl"),
        )

        self.assertIn("--scoring", command)
        self.assertIn("generation", command)
        self.assertNotIn("--token-numeric-source-slots", command)
        self.assertEqual(
            command[command.index("--mode") + 1],
            "qtrm_core_steps_8_no_evidence",
        )

    def test_default_modes_include_sufficient_condition_ablations(self):
        module = load_module()

        self.assertIn(module.PRIMITIVE_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.SOURCE_SLOT_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.SOURCE_BINDER_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.BRIDGE_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.TYPED_VALUE_BRIDGE_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.VOCAB_RENDERER_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.CORE_STATE_ZERO_MODE, module.DEFAULT_MODES)
        self.assertIn(module.ANSWER_RECURRENT_OFF_MODE, module.DEFAULT_MODES)
        self.assertIn(module.ANSWER_NEXT_TOKEN_DECODER_OFF_MODE, module.DEFAULT_MODES)

    def test_mode_default_is_none(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "ckpt.pt",
                "--cases",
                "cases.jsonl",
                "--out-dir",
                "out",
            ]
        )

        self.assertIsNone(args.mode)

    def test_mode_override_replaces_default_modes(self):
        module = load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "cfg.yaml",
                "--checkpoint",
                "ckpt.pt",
                "--cases",
                "cases.jsonl",
                "--out-dir",
                "out",
                "--mode",
                module.FULL_MODE,
            ]
        )

        self.assertEqual(args.mode, [module.FULL_MODE])

    def test_build_report_rejects_without_next_token_decoder_drop(self):
        module = load_module()
        rows = []
        mode_hits = {
            module.DONOR_MODE: [False, False, False, False],
            module.CORE_OFF_MODE: [False, False, False, False],
            module.FULL_MODE: [True, True, False, False],
            module.PRIMITIVE_OFF_MODE: [False, False, False, False],
            module.SOURCE_SLOT_OFF_MODE: [False, False, False, False],
            module.SOURCE_BINDER_OFF_MODE: [False, False, False, False],
            module.BRIDGE_OFF_MODE: [False, False, False, False],
            module.TYPED_VALUE_BRIDGE_OFF_MODE: [False, False, False, False],
            module.VOCAB_RENDERER_OFF_MODE: [False, False, False, False],
            module.CORE_STATE_ZERO_MODE: [False, False, False, False],
            module.ANSWER_RECURRENT_OFF_MODE: [False, False, False, False],
            module.ANSWER_NEXT_TOKEN_DECODER_OFF_MODE: [True, True, False, False],
        }
        for mode, hits in mode_hits.items():
            for index, hit in enumerate(hits):
                rows.append(
                    {
                        "id": f"{mode}-{index}",
                        "mode": mode,
                        "normalized_exact": hit,
                    }
                )

        report = module.build_report(
            rows,
            out_dir=Path("out"),
            commands=[],
            exit_codes=[],
            min_full_accuracy=0.25,
            min_donor_margin=0.10,
            min_core_off_margin=0.10,
            min_primitive_drop=0.10,
            min_source_slot_drop=0.10,
            min_source_binder_drop=0.10,
            min_bridge_drop=0.10,
            min_typed_value_bridge_drop=0.10,
            min_vocab_renderer_drop=0.10,
            min_answer_recurrent_drop=0.10,
            min_answer_next_token_decoder_drop=0.10,
        )

        self.assertFalse(report["accepted"])
        self.assertIn(
            "answer_next_token_decoder_off_drop_below_min",
            report["reject_reasons"],
        )

    def test_default_config_uses_orthodox_core_state_only_answer_loop(self):
        module = load_module()
        config_text = Path(module.DEFAULT_CONFIG).read_text(encoding="utf-8")

        self.assertIn("core_state_only", module.DEFAULT_CONFIG)
        self.assertRegex(
            config_text,
            r"answer_state_loop_core_state_only_enabled:\s*true",
        )

    def test_run_command_creates_log_parent_directories(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exit_code = module.run_command(
                [sys.executable, "-c", "print('ok')"],
                cwd=Path.cwd(),
                env={},
                stdout_path=tmp_path / "logs" / "out.log",
                stderr_path=tmp_path / "logs" / "err.log",
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual((tmp_path / "logs" / "out.log").read_text().strip(), "ok")


if __name__ == "__main__":
    unittest.main()
