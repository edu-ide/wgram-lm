import importlib.util
import hashlib
import re
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

import torch


def load_script_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourcePointerL4LMPathGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script_module(
            "scripts/322_run_source_pointer_l4_lm_path_gate.py",
            "source_pointer_l4_lm_path_gate",
        )
        cls.raw_eval = load_script_module(
            "scripts/192_eval_raw_intelligence.py",
            "raw_intelligence_eval",
        )

    def test_default_checkpoint_uses_latest_source_slot_l3_acceptance(self):
        self.assertIn(
            "qtrm_source_position_l3_hard_batch_s240_b8_eval",
            self.runner.DEFAULT_INIT_CHECKPOINT,
        )
        self.assertTrue(
            self.runner.DEFAULT_INIT_CHECKPOINT.endswith("accepted_l3_last.pt")
        )

    def test_default_l4_config_matches_latest_l3_role_count(self):
        config_text = Path(self.runner.DEFAULT_CONFIG).read_text(encoding="utf-8")
        match = re.search(r"core_role_value_state_num_roles:\s*(\d+)", config_text)

        self.assertIsNotNone(match)
        self.assertEqual("10", match.group(1))

    def test_default_l4_config_enables_trainable_vocab_renderer_gate(self):
        config_text = Path(self.runner.DEFAULT_CONFIG).read_text(encoding="utf-8")
        train_text = Path("src/wgram_lm/training/train.py").read_text(encoding="utf-8")
        args = self.runner.build_arg_parser().parse_args([])
        command = self.runner.train_command(args, Path("out/train"))

        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_source_state_tokens_enabled:\s*true",
        )
        self.assertIn(
            "role_value_answer_bridge_loop_vocab_renderer_only",
            config_text,
        )
        self.assertIn(
            "role_value_answer_bridge_loop_vocab_renderer_only",
            train_text,
        )
        self.assertIn("core_role_value_state_vocab_renderer_", train_text)
        self.assertEqual(
            "role_value_answer_bridge_loop_vocab_renderer_only",
            command[command.index("--trainable-param-policy") + 1],
        )

    def test_default_l4_config_preserves_l3_core_without_typed_register_bypass(self):
        config_text = Path(self.runner.DEFAULT_CONFIG).read_text(encoding="utf-8")

        self.assertRegex(config_text, r"core_typed_register_executor_enabled:\s*true")
        self.assertRegex(config_text, r"core_primitive_typed_selector_enabled:\s*true")

    def test_hard_token_renderer_bottleneck_config_replaces_generic_residual(self):
        config_text = Path(
            "configs/qwen35_2b_4090_source_pointer_l4_hard_token_renderer_bottleneck.yaml"
        ).read_text(encoding="utf-8")

        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_replace_residual_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_candidate_token_ids:",
        )
        self.assertRegex(config_text, r"- 23")

    def test_source_copy_pointer_config_uses_copy_renderer_without_candidate_vocab(self):
        config_text = Path(
            "configs/qwen35_2b_4090_source_copy_pointer_renderer_scaffold.yaml"
        ).read_text(encoding="utf-8")

        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_source_copy_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_source_copy_span_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_replace_residual_enabled:\s*true",
        )
        self.assertRegex(config_text, r"core_typed_register_executor_enabled:\s*true")
        self.assertRegex(config_text, r"core_primitive_typed_selector_enabled:\s*true")
        self.assertNotIn(
            "core_role_value_state_vocab_renderer_candidate_token_ids",
            config_text,
        )

    def test_source_copy_answer_loop_config_enables_next_token_decoder_gate(self):
        config_text = Path(
            "configs/qwen35_2b_4090_source_copy_answer_loop_future_decoder_scaffold.yaml"
        ).read_text(encoding="utf-8")

        self.assertRegex(
            config_text,
            r"answer_state_loop_next_token_decoder_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"answer_state_loop_future_token_decoder_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"answer_state_loop_recurrent_gate_min:\s*1\.0",
        )
        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_source_copy_enabled:\s*true",
        )
        self.assertRegex(
            config_text,
            r"core_role_value_state_vocab_renderer_replace_residual_enabled:\s*false",
        )

    def test_train_command_uses_source_slot_binder_path(self):
        args = self.runner.build_arg_parser().parse_args([])
        command = self.runner.train_command(args, Path("out/train"))

        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--token-numeric-source-slot-gate-min", command)
        self.assertIn("--token-numeric-source-slot-predicate-feedback", command)
        self.assertIn("--core-source-position-binder-source-slots-only", command)
        self.assertIn("--core-source-position-binder-raw-source-slots", command)
        self.assertNotIn("--token-numeric-value-features", command)

    def test_train_command_can_use_relative_parity_source_slot_mode(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-vocab-size",
                "3",
            ]
        )
        command = self.runner.train_command(args, Path("out/train"))
        text = " ".join(command)

        self.assertIn("--token-numeric-source-slot-id-mode relative_parity", text)
        self.assertIn("--token-numeric-source-slot-vocab-size 3", text)

    def test_train_command_forwards_max_length_to_avoid_l4_oom(self):
        args = self.runner.build_arg_parser().parse_args(["--max-length", "256"])
        command = self.runner.train_command(args, Path("out/train"))

        self.assertIn("--max-length", command)
        self.assertEqual("256", command[command.index("--max-length") + 1])
        self.assertIn("--target-logit-positions-only", command)
        self.assertEqual(
            "0.0",
            command[command.index("--causal-prefix-self-rollout-weight") + 1],
        )
        self.assertIn(
            "--core-role-value-vocab-renderer-source-binder-contrast-weight",
            command,
        )

    def test_train_command_can_skip_leading_whitespace_targets(self):
        args = self.runner.build_arg_parser().parse_args(
            ["--skip-leading-whitespace-targets"]
        )
        command = self.runner.train_command(args, Path("out/train"))

        self.assertIn("--causal-prefix-skip-leading-whitespace-targets", command)

    def test_train_command_can_supervise_answer_loop_future_tokens(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--answer-state-loop-logit-ce-weight",
                "0.5",
                "--answer-state-loop-future-token-ce-weight",
                "0.75",
                "--answer-state-loop-future-token-max-target-tokens",
                "6",
            ]
        )
        command = self.runner.train_command(args, Path("out/train"))

        self.assertIn("--answer-state-loop-logit-ce-weight", command)
        self.assertEqual(
            "0.5",
            command[command.index("--answer-state-loop-logit-ce-weight") + 1],
        )
        self.assertIn("--answer-state-loop-future-token-ce-weight", command)
        self.assertEqual(
            "0.75",
            command[
                command.index("--answer-state-loop-future-token-ce-weight") + 1
            ],
        )
        self.assertIn("--answer-state-loop-future-token-max-target-tokens", command)
        self.assertEqual(
            "6",
            command[
                command.index("--answer-state-loop-future-token-max-target-tokens") + 1
            ],
        )

    def test_l4_runner_can_select_low_state_optimizer(self):
        args = self.runner.build_arg_parser().parse_args(["--optimizer", "sgd"])
        command = self.runner.train_command(args, Path("out/train"))
        train_text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--optimizer", command)
        self.assertEqual("sgd", command[command.index("--optimizer") + 1])
        self.assertIn('choices=["adamw", "sgd"]', train_text)
        self.assertIn("torch.optim.SGD", train_text)

    def test_train_process_plan_can_chunk_l4_steps_to_avoid_step2_oom(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--steps",
                "5",
                "--train-process-chunk-steps",
                "2",
                "--seed",
                "100",
            ]
        )
        plan = self.runner.train_process_plan(args, Path("out/train"))

        self.assertEqual([2, 2, 1], [chunk["steps"] for chunk in plan])
        self.assertEqual([100, 101, 102], [chunk["seed"] for chunk in plan])
        self.assertEqual(args.init_checkpoint, plan[0]["init_checkpoint"])
        self.assertEqual("out/train/chunk_0001/last.pt", plan[1]["init_checkpoint"])
        self.assertEqual("out/train/chunk_0002/last.pt", plan[2]["init_checkpoint"])
        self.assertIn("--steps", plan[0]["command"])
        self.assertEqual("2", plan[0]["command"][plan[0]["command"].index("--steps") + 1])

    def test_dry_run_reports_chunked_training_runtime(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--dry-run",
                "--steps",
                "3",
                "--train-process-chunk-steps",
                "1",
                "--out-dir",
                "local_eval/test_l4_dry_run",
            ]
        )
        report = self.runner.run_gate(args)

        self.assertEqual("dry_run", report["decision"])
        self.assertEqual("chunked_process", report["training_runtime"]["mode"])
        self.assertEqual(3, len(report["commands"]["train_chunks"]))
        self.assertIsNone(report["commands"]["train"])

    def test_missing_checkpoint_base_chain_reports_relative_base(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "child.pt"
            torch.save({"base_checkpoint": "missing_base.pt"}, checkpoint)

            missing = self.runner.missing_checkpoint_base_chain(checkpoint, root=root)

            self.assertEqual(missing, [str(root / "missing_base.pt")])

    def test_checkpoint_base_chain_issues_reports_known_sha_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path(
                "local_eval/research_gate_runner/"
                "primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt"
            )
            checkpoint = root / relative
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"wrong checkpoint bytes")
            load_calls = []

            def load_state(path):
                load_calls.append(path)
                return {}

            issues = self.runner.checkpoint_base_chain_issues(
                relative,
                root=root,
                load_state=load_state,
            )

            self.assertEqual("sha256_mismatch", issues[0]["issue"])
            self.assertEqual(str(checkpoint), issues[0]["path"])
            self.assertEqual([], load_calls)

    def test_checkpoint_base_chain_issues_continues_after_known_sha_match(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path(
                "local_eval/research_gate_runner/"
                "primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt"
            )
            checkpoint = root / relative
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"known checkpoint bytes")
            expected = hashlib.sha256(b"known checkpoint bytes").hexdigest()
            original = dict(self.runner.KNOWN_CHECKPOINT_SHA256)
            self.runner.KNOWN_CHECKPOINT_SHA256[str(relative)] = expected
            try:
                issues = self.runner.checkpoint_base_chain_issues(
                    relative,
                    root=root,
                    load_state=lambda path: {"base_checkpoint": "missing_base.pt"},
                )
            finally:
                self.runner.KNOWN_CHECKPOINT_SHA256.clear()
                self.runner.KNOWN_CHECKPOINT_SHA256.update(original)

            self.assertEqual(
                [{"issue": "missing", "path": str(root / "missing_base.pt")}],
                issues,
            )

    def test_run_gate_stops_before_training_when_init_chain_is_missing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "child.pt"
            out_dir = root / "out"
            torch.save({"base_checkpoint": str(root / "missing_base.pt")}, checkpoint)
            args = self.runner.build_arg_parser().parse_args(
                [
                    "--init-checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(out_dir),
                    "--steps",
                    "1",
                ]
            )

            report = self.runner.run_gate(args)

            self.assertEqual("checkpoint_chain_missing", report["decision"])
            self.assertFalse(report["accepted"])
            self.assertEqual(
                [str(root / "missing_base.pt")],
                report["missing_base_checkpoints"],
            )
            self.assertEqual(
                [{"issue": "missing", "path": str(root / "missing_base.pt")}],
                report["checkpoint_chain_issues"],
            )
            self.assertTrue((out_dir / "report.json").exists())

    def test_run_gate_stops_before_training_on_known_sha_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path(
                "local_eval/research_gate_runner/"
                "primitive_field_heads_delta_codec_s90_lr5e4_seed11/last.pt"
            )
            checkpoint = root / relative
            out_dir = root / "out"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"wrong checkpoint bytes")
            original_repo_root = self.runner.repo_root
            self.runner.repo_root = lambda: root
            try:
                args = self.runner.build_arg_parser().parse_args(
                    [
                        "--init-checkpoint",
                        str(relative),
                        "--out-dir",
                        str(out_dir),
                        "--steps",
                        "1",
                    ]
                )
                report = self.runner.run_gate(args)
            finally:
                self.runner.repo_root = original_repo_root

            self.assertEqual("checkpoint_chain_sha256_mismatch", report["decision"])
            self.assertFalse(report["accepted"])
            self.assertEqual([], report["missing_base_checkpoints"])
            self.assertEqual(
                "sha256_mismatch",
                report["checkpoint_chain_issues"][0]["issue"],
            )
            self.assertEqual(str(checkpoint), report["checkpoint_chain_issues"][0]["path"])
            self.assertTrue((out_dir / "report.json").exists())

    def test_eval_command_uses_source_slot_flags_and_ablation_modes(self):
        args = self.runner.build_arg_parser().parse_args([])
        command = self.runner.eval_command(args, Path("checkpoint.pt"), Path("eval.jsonl"))

        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--core-source-position-binder-source-slots-only", command)
        self.assertIn("--core-source-position-binder-raw-source-slots", command)
        self.assertNotIn("--token-numeric-value-features", command)
        self.assertIn(self.runner.SOURCE_SLOT_OFF_MODE, command)
        self.assertIn(self.runner.SOURCE_BINDER_OFF_MODE, command)
        self.assertIn(self.runner.ANSWER_NEXT_TOKEN_DECODER_OFF_MODE, command)

    def test_eval_command_can_use_relative_parity_source_slot_mode(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--token-numeric-source-slot-id-mode",
                "relative_parity",
                "--token-numeric-source-slot-vocab-size",
                "3",
            ]
        )
        command = self.runner.eval_command(
            args,
            Path("checkpoint.pt"),
            Path("eval.jsonl"),
        )
        text = " ".join(command)

        self.assertIn("--token-numeric-source-slot-id-mode relative_parity", text)
        self.assertIn("--token-numeric-source-slot-vocab-size 3", text)

    def test_runner_exposes_post_train_probe_without_generation_eval(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--post-train-source-copy-probe",
                "--skip-generation-eval",
                "--probe-max-cases",
                "4",
            ]
        )

        self.assertTrue(args.post_train_source_copy_probe)
        self.assertTrue(args.skip_generation_eval)
        self.assertEqual(4, args.probe_max_cases)

    def test_source_copy_probe_command_uses_l3_preserving_runtime_contract(self):
        args = self.runner.build_arg_parser().parse_args(
            [
                "--config",
                "configs/source_copy_train.yaml",
                "--eval-config",
                "configs/source_copy_eval.yaml",
                "--eval-jsonl",
                "data/eval/source_copy.jsonl",
                "--max-length",
                "256",
                "--probe-max-cases",
                "6",
            ]
        )
        command = self.runner.source_copy_probe_command(
            args,
            Path("out/train/last.pt"),
            Path("out/source_copy_probe.json"),
        )

        self.assertIn("scripts/328_probe_qtrm_source_position_logits.py", command)
        self.assertEqual(
            "configs/source_copy_eval.yaml",
            command[command.index("--config") + 1],
        )
        self.assertEqual(
            "out/train/last.pt",
            command[command.index("--checkpoint") + 1],
        )
        self.assertEqual(
            "data/eval/source_copy.jsonl",
            command[command.index("--cases") + 1],
        )
        self.assertEqual(
            "out/source_copy_probe.json",
            command[command.index("--out") + 1],
        )
        self.assertEqual("6", command[command.index("--max-cases") + 1])
        self.assertEqual("256", command[command.index("--max-length") + 1])
        self.assertIn("--token-numeric-source-slots", command)
        self.assertIn("--token-numeric-source-slot-predicate-feedback", command)
        self.assertIn("--core-source-position-binder", command)
        self.assertIn("--core-source-position-binder-state-st", command)
        self.assertIn("--core-source-position-binder-raw-source-slots", command)

    def test_l4_decision_requires_next_token_decoder_drop_when_present(self):
        summary = {
            self.runner.FULL_MODE: {"accuracy": 0.5},
            self.runner.DONOR_MODE: {"accuracy": 0.0},
            self.runner.CORE_OFF_MODE: {"accuracy": 0.0},
            self.runner.PRIMITIVE_OFF_MODE: {"accuracy": 0.0},
            self.runner.SOURCE_SLOT_OFF_MODE: {"accuracy": 0.0},
            self.runner.SOURCE_BINDER_OFF_MODE: {"accuracy": 0.0},
            self.runner.BRIDGE_OFF_MODE: {"accuracy": 0.0},
            self.runner.FINAL_BINDER_OFF_MODE: {"accuracy": 0.0},
            self.runner.VOCAB_RENDERER_OFF_MODE: {"accuracy": 0.0},
            self.runner.ANSWER_RECURRENT_OFF_MODE: {"accuracy": 0.0},
            self.runner.ANSWER_HALT_GATE_OFF_MODE: {"accuracy": 0.0},
            self.runner.ANSWER_NEXT_TOKEN_DECODER_OFF_MODE: {"accuracy": 0.5},
        }

        decision = self.runner.build_decision(
            summary=summary,
            min_full_accuracy=0.2,
            min_donor_margin=0.05,
            min_core_off_margin=0.05,
            min_primitive_drop=0.05,
            min_source_slot_drop=0.05,
            min_source_binder_drop=0.05,
            min_bridge_drop=0.05,
            min_vocab_renderer_drop=0.05,
            min_answer_recurrent_drop=0.05,
            min_answer_halt_gate_drop=0.05,
            min_answer_next_token_decoder_drop=0.05,
        )

        self.assertFalse(decision["accepted"])
        self.assertIn(
            "answer_next_token_decoder_drop_below_min",
            decision["reject_reasons"],
        )

    def test_raw_generation_eval_can_disable_source_slot_or_binder_path(self):
        source_slot_runtime = self.raw_eval.mode_runtime(
            "qtrm_core_steps_8_token_numeric_source_slots_off_no_evidence"
        )
        binder_runtime = self.raw_eval.mode_runtime(
            "qtrm_core_steps_8_core_source_position_binder_off_no_evidence"
        )

        self.assertTrue(source_slot_runtime["disable_token_numeric_source_slots"])
        self.assertTrue(binder_runtime["disable_core_source_position_binder"])

    def test_qtrm_forward_can_compute_l4_target_position_logits_only(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            answer_state_loop_enabled=True,
            answer_state_loop_requires_core=True,
            answer_state_loop_recurrent_block_enabled=True,
            answer_state_loop_lm_adapter_enabled=True,
            answer_state_loop_lm_adapter_rank=4,
            donor_logits_scale=1.0,
        )
        model = QTRMMultimodalModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        donor_logits = torch.randn(2, 6, cfg.vocab_size)
        indices = torch.tensor([1, 4])

        out = model(
            input_ids,
            donor_logits=donor_logits,
            logit_token_indices=indices,
        )

        self.assertEqual(out["logits"].shape, (2, 2, cfg.vocab_size))
        self.assertEqual(out["qtrm_logits"].shape, (2, 2, cfg.vocab_size))
        self.assertEqual(out["answer_state_loop_logits"].shape, (2, 2, cfg.vocab_size))
        self.assertEqual(out["donor_qtrm_conflict_gate"].shape, (2, 2))
        self.assertTrue(torch.equal(out["logit_token_indices"].cpu(), indices))

    def test_l4_vocab_renderer_can_read_direct_source_state_tokens(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(7)
        cfg = QTRMConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=3,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            token_numeric_source_slot_embedding_enabled=True,
            token_numeric_source_slot_vocab_size=32,
            token_numeric_source_slot_max_slots=3,
            token_numeric_source_slot_gate_min=1.0,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=3,
            core_role_value_state_vocab_size=32,
            core_source_position_binder_enabled=True,
            core_source_position_binder_gate_min=1.0,
            core_source_position_binder_state_gate_min=1.0,
            core_source_position_binder_state_straight_through=True,
            core_source_position_binder_source_slots_only=True,
            core_source_position_binder_raw_source_slots_enabled=True,
            core_primitive_role_value_executor_enabled=True,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_answer_bridge_gate_min=1.0,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            core_role_value_state_vocab_renderer_source_state_tokens_enabled=True,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.normal_(0.0, 0.2)

        input_ids = torch.randint(1, cfg.vocab_size, (1, 5))
        source_slots = torch.tensor([[3, 7, 11]])
        indices = torch.tensor([1, 3])

        full = model(
            input_ids,
            token_numeric_source_slot_ids=source_slots,
            logit_token_indices=indices,
        )
        binder_off = model(
            input_ids,
            token_numeric_source_slot_ids=source_slots,
            logit_token_indices=indices,
            disable_core_source_position_binder=True,
        )

        self.assertFalse(
            torch.allclose(
                full["core_role_value_vocab_renderer_logits"],
                binder_off["core_role_value_vocab_renderer_logits"],
            )
        )

    def test_l4_vocab_renderer_can_use_lm_head_for_lexicalization(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=24,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            tie_embeddings=False,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=2,
            core_role_value_state_vocab_size=8,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_use_lm_head=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.lm_head.weight.normal_(0.0, 0.3)

        out = model(
            torch.randint(1, cfg.vocab_size, (1, 4)),
            logit_token_indices=torch.tensor([1, 2]),
        )

        self.assertEqual(
            out["core_role_value_vocab_renderer_logits"].shape,
            (1, 2, cfg.vocab_size),
        )

    def test_l4_vocab_renderer_can_replace_generic_qtrm_residual(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=28,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=1,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=2,
            core_role_value_state_vocab_size=8,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            core_role_value_state_vocab_renderer_replace_residual_enabled=True,
            donor_logits_scale=0.0,
            qtrm_residual_clamp=None,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.normal_(0.0, 0.2)

        input_ids = torch.randint(1, cfg.vocab_size, (1, 5))
        indices = torch.tensor([1, 4])
        out = model(input_ids, logit_token_indices=indices)
        renderer_off = model(
            input_ids,
            logit_token_indices=indices,
            disable_core_role_value_vocab_renderer=True,
        )

        self.assertTrue(
            torch.allclose(
                out["qtrm_residual_logits"],
                out["core_role_value_vocab_renderer_logits"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                renderer_off["qtrm_residual_logits"],
                torch.zeros_like(renderer_off["qtrm_residual_logits"]),
            )
        )

    def test_l4_vocab_renderer_can_limit_residual_to_candidate_tokens(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=30,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=2,
            core_role_value_state_vocab_size=8,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            core_role_value_state_vocab_renderer_candidate_token_ids=[3, 7, 11],
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.normal_(0.0, 0.2)

        out = model(
            torch.randint(1, cfg.vocab_size, (1, 4)),
            logit_token_indices=torch.tensor([1, 2]),
        )
        logits = out["core_role_value_vocab_renderer_logits"]
        non_candidate = torch.ones(cfg.vocab_size, dtype=torch.bool)
        non_candidate[[3, 7, 11]] = False

        self.assertGreater(float(logits[..., [3, 7, 11]].detach().abs().max()), 0.0)
        self.assertEqual(float(logits[..., non_candidate].detach().abs().max()), 0.0)

    def test_l4_vocab_renderer_can_scatter_source_copy_logits_to_prompt_tokens(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=40,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=3,
            core_role_value_state_vocab_size=8,
            core_source_position_binder_enabled=True,
            core_source_position_binder_gate_min=1.0,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            core_role_value_state_vocab_renderer_source_copy_enabled=True,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.zero_()

        input_ids = torch.tensor([[5, 9, 13, 17, 21]])
        out = model(input_ids, logit_token_indices=torch.tensor([4]))
        binder_off = model(
            input_ids,
            logit_token_indices=torch.tensor([4]),
            disable_core_source_position_binder=True,
        )

        logits = out["core_role_value_vocab_renderer_logits"][0, 0]
        prompt_mask = torch.zeros(cfg.vocab_size, dtype=torch.bool)
        prompt_mask[input_ids[0]] = True

        self.assertGreater(float(logits[prompt_mask].detach().abs().max()), 0.0)
        self.assertEqual(float(logits[~prompt_mask].detach().abs().max()), 0.0)
        self.assertEqual(
            float(
                binder_off["core_role_value_vocab_renderer_logits"].detach().abs().max()
            ),
            0.0,
        )

    def test_l4_source_copy_uses_slot_token_ids_not_source_class_ids(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            token_numeric_source_slot_embedding_enabled=True,
            token_numeric_source_slot_vocab_size=64,
            token_numeric_source_slot_max_slots=2,
            token_numeric_source_slot_gate_min=1.0,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=3,
            core_role_value_state_vocab_size=8,
            core_source_position_binder_enabled=True,
            core_source_position_binder_gate_min=1.0,
            core_source_position_binder_source_slots_only=True,
            core_source_position_binder_raw_source_slots_enabled=True,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            core_role_value_state_vocab_renderer_source_copy_enabled=True,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.zero_()

        out = model(
            torch.tensor([[5, 9, 13]]),
            token_numeric_source_slot_ids=torch.tensor([[15, 32]]),
            token_numeric_source_slot_token_ids=torch.tensor([[206, 308]]),
            token_numeric_source_slot_mask=torch.tensor([[1, 1]]),
            logit_token_indices=torch.tensor([2]),
        )

        logits = out["core_role_value_vocab_renderer_logits"][0, 0]
        self.assertGreater(float(logits[[206, 308]].detach().abs().max()), 0.0)
        self.assertEqual(float(logits[[15, 32]].detach().abs().max()), 0.0)

    def test_l4_source_copy_position_classes_are_one_based_with_zero_null(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        torch.manual_seed(7)
        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=1,
            core_role_value_state_vocab_size=8,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_gate_min=1.0,
            core_role_value_state_vocab_renderer_rank=4,
            core_role_value_state_vocab_renderer_source_copy_enabled=True,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        with torch.no_grad():
            model.core_role_value_state_vocab_renderer_up.weight.zero_()

        source_position_logits = torch.full((1, 1, 1, 8), -30.0)
        source_position_logits[0, 0, 0, 1] = 30.0
        logits = model._compute_core_role_value_state_vocab_renderer_logits(
            torch.randn(1, 1, cfg.d_model),
            torch.randn(1, 1, 1, cfg.d_model),
            source_copy_position_logits=source_position_logits,
            source_copy_token_ids=torch.tensor([[101, 202, 303]]),
            input_seq_len=1,
        )[0, 0]

        self.assertGreater(float(logits[101].detach().abs()), 1e-6)
        self.assertLess(float(logits[202].detach().abs()), 1e-6)

    def test_l4_source_copy_renderer_uses_final_recurrent_position_state(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=2,
            core_role_value_state_vocab_size=8,
            core_source_position_binder_enabled=True,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_enabled=True,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        prompt_logits = torch.full((1, 1, 2, 8), -20.0)
        prompt_logits[0, 0, 0, 1] = 20.0
        prompt_logits[0, 0, 1, 2] = 20.0
        recurrent_logits = torch.full((1, 3, 2, 8), -20.0)
        recurrent_logits[0, -1, 0, 4] = 20.0
        recurrent_logits[0, -1, 1, 5] = 20.0
        bridge_tokens = torch.randn(1, 1, 2, cfg.d_model)

        selected = model._select_source_copy_position_logits_for_renderer(
            source_position_prompt_logits=prompt_logits,
            core_role_value_state_logits=recurrent_logits,
            bridge_tokens=bridge_tokens,
        )

        self.assertEqual(tuple(selected.shape), (1, 1, 2, 8))
        self.assertEqual(int(selected[0, 0, 0].argmax().item()), 4)
        self.assertEqual(int(selected[0, 0, 1].argmax().item()), 5)

    def test_l4_source_copy_selector_can_use_accepted_primitive_state(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_num_roles=2,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_from_primitive_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        prompt_logits = torch.zeros(1, 1, 2, 8)
        selected_logits = torch.ones(1, 4, 2, 8)
        primitive_logits = torch.full((1, 4, 2, 8), 2.0)

        selected = model._select_source_copy_position_logits_for_renderer(
            source_position_prompt_logits=prompt_logits,
            core_role_value_state_logits=selected_logits,
            core_primitive_role_value_state_logits=primitive_logits,
            bridge_tokens=torch.zeros(1, 1, 2, cfg.d_model),
        )

        self.assertTrue(torch.equal(selected, primitive_logits[:, -1:, :, :]))

    def test_l4_source_copy_masks_non_answer_roles_to_null(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=64,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=8,
            core_role_value_state_answer_bridge_enabled=True,
            core_role_value_state_vocab_renderer_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_enabled=True,
            donor_logits_scale=0.0,
        )
        model = QTRMMultimodalModel(cfg)
        logits = torch.full((1, 1, 10, 8), -20.0)
        for role in range(10):
            logits[0, 0, role, min(role + 1, 7)] = 20.0

        masked = model._mask_source_copy_position_logits_to_answer_roles(logits)

        for role in range(0, 4):
            self.assertEqual(
                int(masked[0, 0, role].argmax().item()),
                min(role + 1, 7),
            )
        for role in range(4, 10):
            self.assertEqual(int(masked[0, 0, role].argmax().item()), 0)

    def test_l4_source_copy_cursor_bias_tracks_visible_answer_prefix(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_cursor_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_cursor_bias=7.5,
        )
        model = QTRMMultimodalModel(cfg)
        source_ids = torch.tensor([[101, 202, 303]])
        source_mask = torch.tensor([[1, 1, 1]])
        # The visible prompt contains each source number once. At the prompt
        # boundary the next copy should read answer role 0.
        input_ids = torch.tensor([[101, 202, 303, 17]])

        bias = model._compute_source_copy_cursor_role_bias(
            input_ids=input_ids,
            source_copy_token_ids=source_ids,
            source_copy_token_mask=source_mask,
            query_token_indices=torch.tensor([3]),
            output_seq_len=1,
            role_count=10,
            device=input_ids.device,
            dtype=torch.float32,
        )

        self.assertEqual(tuple(bias.shape), (1, 1, 10))
        self.assertAlmostEqual(float(bias[0, 0, 0].item()), 7.5)
        self.assertEqual(float(bias[0, 0, 1:].abs().sum().item()), 0.0)

        # After the first copied value and a separator, the next copy should
        # read answer role 1.
        input_ids = torch.tensor([[101, 202, 303, 17, 101, 11]])
        bias = model._compute_source_copy_cursor_role_bias(
            input_ids=input_ids,
            source_copy_token_ids=source_ids,
            source_copy_token_mask=source_mask,
            query_token_indices=torch.tensor([5]),
            output_seq_len=1,
            role_count=10,
            device=input_ids.device,
            dtype=torch.float32,
        )

        self.assertAlmostEqual(float(bias[0, 0, 1].item()), 7.5)
        self.assertEqual(float(bias[0, 0, 0].abs().item()), 0.0)

    def test_l4_source_copy_cursor_does_not_bias_immediately_after_copy_token(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            core_role_value_state_num_roles=10,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_cursor_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_cursor_bias=7.5,
        )
        model = QTRMMultimodalModel(cfg)

        bias = model._compute_source_copy_cursor_role_bias(
            input_ids=torch.tensor([[101, 202, 303, 17, 101]]),
            source_copy_token_ids=torch.tensor([[101, 202, 303]]),
            source_copy_token_mask=torch.tensor([[1, 1, 1]]),
            query_token_indices=torch.tensor([4]),
            output_seq_len=1,
            role_count=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(float(bias.abs().sum().item()), 0.0)

    def test_l4_source_copy_span_cursor_continues_multi_token_source_slot(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            tie_embeddings=False,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_span_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        span_ids = torch.tensor([[[201, 202, 0], [301, 302, 0]]])
        span_mask = torch.tensor([[[1, 1, 0], [1, 1, 0]]])

        next_ids, valid = model._compute_source_copy_span_next_token_ids(
            input_ids=torch.tensor([[100, 201]]),
            source_copy_token_span_ids=span_ids,
            source_copy_token_span_mask=span_mask,
            query_token_indices=torch.tensor([1]),
            output_seq_len=1,
            position_count=3,
            device=span_ids.device,
        )

        self.assertEqual(int(next_ids[0, 0, 1]), 202)
        self.assertTrue(bool(valid[0, 0, 1]))

    def test_l4_source_copy_span_cursor_stops_after_complete_source_span(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            tie_embeddings=False,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=4,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_span_enabled=True,
        )
        model = QTRMMultimodalModel(cfg)
        span_ids = torch.tensor([[[201, 202, 0], [301, 302, 0]]])
        span_mask = torch.tensor([[[1, 1, 0], [1, 1, 0]]])

        next_ids, valid = model._compute_source_copy_span_next_token_ids(
            input_ids=torch.tensor([[100, 201, 202]]),
            source_copy_token_span_ids=span_ids,
            source_copy_token_span_mask=span_mask,
            query_token_indices=torch.tensor([2]),
            output_seq_len=1,
            position_count=3,
            device=span_ids.device,
        )

        self.assertEqual(int(next_ids[0, 0, 1]), 0)
        self.assertFalse(bool(valid[0, 0, 1]))

    def test_l4_source_copy_answer_role_cursor_starts_at_first_answer_role(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            tie_embeddings=False,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=6,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias=6.0,
        )
        model = QTRMMultimodalModel(cfg)
        span_ids = torch.tensor([[[201, 202, 0], [301, 302, 0]]])
        span_mask = torch.tensor([[[1, 1, 0], [1, 1, 0]]])

        bias = model._compute_source_copy_answer_role_cursor_bias(
            input_ids=torch.tensor([[201, 202, 301, 302, 17]]),
            source_copy_token_span_ids=span_ids,
            source_copy_token_span_mask=span_mask,
            query_token_indices=torch.tensor([4]),
            output_seq_len=1,
            role_count=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertAlmostEqual(float(bias[0, 0, 0].item()), 6.0)
        self.assertEqual(float(bias[0, 0, 1:].abs().sum().item()), 0.0)

    def test_l4_source_copy_answer_role_cursor_advances_after_separator(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            tie_embeddings=False,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=6,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias=6.0,
            core_role_value_state_vocab_renderer_source_copy_answer_role_separator_token_ids=[11],
        )
        model = QTRMMultimodalModel(cfg)
        span_ids = torch.tensor([[[201, 202, 0], [301, 302, 0]]])
        span_mask = torch.tensor([[[1, 1, 0], [1, 1, 0]]])

        bias = model._compute_source_copy_answer_role_cursor_bias(
            input_ids=torch.tensor([[201, 202, 301, 302, 17, 201, 202, 11]]),
            source_copy_token_span_ids=span_ids,
            source_copy_token_span_mask=span_mask,
            query_token_indices=torch.tensor([7]),
            output_seq_len=1,
            role_count=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertAlmostEqual(float(bias[0, 0, 1].item()), 6.0)
        self.assertEqual(float(bias[0, 0, 0].abs().item()), 0.0)

    def test_l4_source_copy_answer_role_cursor_does_not_force_separator_step(self):
        from wgram_lm import QTRMConfig, QTRMMultimodalModel

        cfg = QTRMConfig(
            vocab_size=512,
            d_model=12,
            n_heads=3,
            n_kv_heads=1,
            d_ff=24,
            n_prelude_layers=0,
            n_core_layers=1,
            n_coda_layers=0,
            workspace_tokens=2,
            workspace_layers=1,
            workspace_ff_mult=1,
            h_cycles=1,
            l_cycles=1,
            outer_steps=1,
            visual_dim=8,
            max_visual_tokens=2,
            tie_embeddings=False,
            core_role_value_state_enabled=True,
            core_role_value_state_num_roles=6,
            core_role_value_state_vocab_size=8,
            core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled=True,
            core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias=6.0,
        )
        model = QTRMMultimodalModel(cfg)
        span_ids = torch.tensor([[[201, 202, 0], [301, 302, 0]]])
        span_mask = torch.tensor([[[1, 1, 0], [1, 1, 0]]])

        bias = model._compute_source_copy_answer_role_cursor_bias(
            input_ids=torch.tensor([[201, 202, 301, 302, 17, 201, 202]]),
            source_copy_token_span_ids=span_ids,
            source_copy_token_span_mask=span_mask,
            query_token_indices=torch.tensor([6]),
            output_seq_len=1,
            role_count=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(float(bias.abs().sum().item()), 0.0)

    def test_train_script_releases_graph_heavy_tensors_between_l4_steps(self):
        text = Path("scripts/196_train_pure_recursive_depth_supervised.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("del loss, weighted_loss, losses, loss_weights", text)
        self.assertIn("torch.cuda.empty_cache()", text)


if __name__ == "__main__":
    unittest.main()
