import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class MemoryEvalScriptTests(unittest.TestCase):
    def test_resolve_qtrm_scale_uses_override_when_present(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(module.resolve_qtrm_scale(0.1, None), 0.1)
        self.assertEqual(module.resolve_qtrm_scale(0.1, 0.5), 0.5)

    def test_memory_eval_cli_exposes_answer_channel_guards(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--suppress-visible-reasoning-tokens",
                "--no-repeat-ngram-size",
                "2",
                "--short-answer-governor",
            ]
        )

        self.assertTrue(args.suppress_visible_reasoning_tokens)
        self.assertEqual(args.no_repeat_ngram_size, 2)
        self.assertTrue(args.short_answer_governor)

    def test_memory_eval_cli_exposes_evidence_span_copy_answer_channel(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--answer-channel",
                "evidence_span_copy",
                "--evidence-span-max-tokens",
                "8",
                "--evidence-span-no-answer-threshold",
                "0.7",
                "--evidence-span-min-score",
                "12.0",
            ]
        )

        self.assertEqual(args.answer_channel, "evidence_span_copy")
        self.assertEqual(args.evidence_span_max_tokens, 8)
        self.assertEqual(args.evidence_span_no_answer_threshold, 0.7)
        self.assertEqual(args.evidence_span_min_score, 12.0)

    def test_memory_eval_cli_exposes_answer_revision(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--answer-revision",
                "evidence_span_boundary",
                "--answer-revision-max-left-tokens",
                "2",
                "--answer-revision-max-right-tokens",
                "3",
            ]
        )

        self.assertEqual(args.answer_revision, "evidence_span_boundary")
        self.assertEqual(args.answer_revision_max_left_tokens, 2)
        self.assertEqual(args.answer_revision_max_right_tokens, 3)

    def test_memory_eval_cli_exposes_evidence_source_governor(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--evidence-source-governor",
                "reliability",
            ]
        )

        self.assertEqual(args.evidence_source_governor, "reliability")

    def test_memory_eval_cli_exposes_evidence_source_selector(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--evidence-source-selector-checkpoint",
                "runs/source_selector.pt",
                "--evidence-source-selector-threshold",
                "0.7",
                "--evidence-source-selector-mode",
                "span_mask",
            ]
        )

        self.assertEqual(args.evidence_source_selector_checkpoint, "runs/source_selector.pt")
        self.assertEqual(args.evidence_source_selector_threshold, 0.7)
        self.assertEqual(args.evidence_source_selector_mode, "span_mask")

    def test_memory_eval_cli_defaults_to_ssot_evidence_injection(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args([])

        self.assertEqual(args.evidence_injection, "ssot")

    def test_canonical_ssot_contract_requires_ssot_greedy(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        ok = module.build_arg_parser().parse_args(
            [
                "--require-canonical-ssot",
                "--evidence-injection",
                "ssot",
                "--answer-channel",
                "greedy",
            ]
        )
        module.validate_canonical_ssot_contract(ok)

        workspace = module.build_arg_parser().parse_args(
            [
                "--require-canonical-ssot",
                "--evidence-injection",
                "workspace",
                "--answer-channel",
                "greedy",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --evidence-injection ssot"):
            module.validate_canonical_ssot_contract(workspace)

        span_copy = module.build_arg_parser().parse_args(
            [
                "--require-canonical-ssot",
                "--evidence-injection",
                "ssot",
                "--answer-channel",
                "evidence_span_copy",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --answer-channel greedy"):
            module.validate_canonical_ssot_contract(span_copy)

    def test_canonical_model_contract_rejects_lewm_config_in_memory_eval(self):
        from qtrm_mm.config import load_config

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        cfg = load_config("configs/qwen35_2b_4090_pure_recursive_lewm_staged_s200.yaml")

        with self.assertRaisesRegex(ValueError, "core_world_model_enabled"):
            module.validate_canonical_model_contract(cfg)

    def test_memory_eval_cli_exposes_truth_gate_for_span_copy(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--truth-gate",
                "--truth-support-threshold",
                "0.6",
                "--truth-causal-threshold",
                "0.7",
                "--truth-refute-threshold",
                "0.3",
                "--truth-missing-threshold",
                "0.2",
            ]
        )

        self.assertTrue(args.truth_gate)
        self.assertEqual(args.truth_support_threshold, 0.6)
        self.assertEqual(args.truth_causal_threshold, 0.7)
        self.assertEqual(args.truth_refute_threshold, 0.3)
        self.assertEqual(args.truth_missing_threshold, 0.2)

    def test_memory_eval_cli_exposes_answer_decision_checkpoint(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--answer-decision-checkpoint",
                "runs/decision/last.pt",
                "--answer-decision-threshold",
                "0.7",
            ]
        )

        self.assertEqual(args.answer_decision_checkpoint, "runs/decision/last.pt")
        self.assertEqual(args.answer_decision_threshold, 0.7)

    def test_memory_eval_cli_exposes_model_answer_decision(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--model-answer-decision",
                "--model-answer-decision-threshold",
                "0.7",
                "--mode",
                "qtrm_answer_decision_off_with_evidence",
            ]
        )

        self.assertTrue(args.model_answer_decision)
        self.assertEqual(args.model_answer_decision_threshold, 0.7)
        self.assertEqual(args.mode, ["qtrm_answer_decision_off_with_evidence"])

    def test_apply_model_answer_decision_blocks_from_model_logit(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        completion, meta = module.apply_model_answer_decision(
            completion="Answer: guessed",
            answer_channel_meta={"status": "span", "model_answer_decision": {"block_probability": 0.9}},
            threshold=0.7,
        )

        self.assertEqual(completion, "Answer: UNKNOWN")
        self.assertTrue(meta["answer_decision"]["blocked"])
        self.assertEqual(meta["answer_decision"]["source"], "model")

    def test_build_model_answer_decision_features_matches_posthoc_feature_dim(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        features, names = module.build_model_answer_decision_features(
            case={
                "id": "case-a",
                "question": "What is the current code?",
                "task_family": "abstention",
            },
            completion="Answer: fake",
            answer_channel_meta={
                "status": "span",
                "selected_score": 12.0,
                "selected_token_ids": [1, 2, 3],
                "no_answer_prob": 0.6,
                "truth_gate": {
                    "allow": True,
                    "block_reasons": [],
                    "support_prob": 0.7,
                    "causal_prob": 0.7,
                    "refute_prob": 0.1,
                    "missing_prob": 0.6,
                },
            },
            completion_ids=[1, 2, 3],
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            logit_shift={"max_abs_delta": 0.5},
            prompt_telemetry={
                "latent_gates": {
                    "workspace_update_gate_mean": 0.1,
                    "workspace_update_gate_last_mean": 0.1,
                    "core_context_gate_mean": 0.2,
                    "core_context_gate_last_mean": 0.2,
                }
            },
        )

        self.assertEqual(len(features), 23)
        self.assertEqual(len(names), 23)
        self.assertEqual(names[0], "support_prob")
        self.assertAlmostEqual(features[6], 0.6, places=5)

    def test_answer_decision_features_treat_deferred_no_answer_as_span(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        features, names = module.build_model_answer_decision_features(
            case={"id": "case-a", "question": "What is the current code?"},
            completion="Answer: stone-arch",
            answer_channel_meta={
                "status": "span",
                "selected_score": 14.4,
                "selected_token_ids": [1, 2, 3],
                "no_answer_prob": 0.97,
                "no_answer_deferred_by_source_mask": True,
                "source_token_mask_count": 14,
                "truth_gate": {
                    "allow": True,
                    "block_reasons": [],
                    "support_prob": 0.67,
                    "causal_prob": 0.63,
                    "refute_prob": 0.07,
                    "missing_prob": 0.35,
                },
            },
            completion_ids=[1, 2, 3],
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            logit_shift={"max_abs_delta": 0.5},
            prompt_telemetry={"latent_gates": {}},
        )

        self.assertEqual(names[6], "no_answer_prob")
        self.assertAlmostEqual(features[6], 0.0, places=5)

    def test_answer_decision_features_off_mode_disables_feature_path(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        kwargs = module.mode_forward_kwargs("qtrm_answer_decision_features_off_with_evidence")

        self.assertTrue(kwargs["disable_answer_decision_features"])
        self.assertFalse(kwargs.get("disable_answer_decision_head", False))

    def test_memory_eval_cli_exposes_generation_history_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.build_arg_parser().parse_args(
            [
                "--history-jsonl-out",
                "auto",
            ]
        )

        self.assertEqual(args.history_jsonl_out, "auto")

    def test_replace_completion_suffix_updates_final_answer_text(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(
            module.replace_completion_suffix(
                "Question: xAnswer: fake",
                old_completion="Answer: fake",
                new_completion="Answer: UNKNOWN",
            ),
            "Question: xAnswer: UNKNOWN",
        )
        self.assertEqual(
            module.replace_completion_suffix(
                "Question: x",
                old_completion="Answer: fake",
                new_completion="Answer: UNKNOWN",
            ),
            "Question: x\nAnswer: UNKNOWN",
        )

    def test_truth_gate_is_disabled_by_evidence_bottleneck_off_ablation(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertTrue(
            module.truth_gate_enabled_for_mode(
                "qtrm_residual_with_evidence",
                requested=True,
            )
        )
        self.assertFalse(
            module.truth_gate_enabled_for_mode(
                "qtrm_evidence_bottleneck_off_with_evidence",
                requested=True,
            )
        )
        self.assertFalse(
            module.truth_gate_enabled_for_mode(
                "qtrm_residual_with_evidence",
                requested=False,
            )
        )

    def test_workspace_memory_text_uses_span_reader_training_format(self):
        from qtrm_mm.eval.memory_retrieval import build_workspace_memory_text

        text = build_workspace_memory_text(
            [
                (
                    1.0,
                    {
                        "source": "doc.md",
                        "chunk_id": 0,
                        "text": "현재 코드는 새벽-14이다.",
                    },
                )
            ]
        )

        self.assertTrue(text.startswith("MemoryOS evidence"))
        self.assertNotIn("Workspace-side MemoryOS evidence", text)

    def test_evidence_span_copy_selects_best_legal_span(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            eos_token_id = None

            def decode(self, ids, skip_special_tokens=True):
                vocab = {10: "현재", 11: " 코드는", 12: " 새벽-14", 13: " 이다"}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[0.0, 1.0, 5.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[0.0, 0.5, 1.0, 5.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
        }
        workspace_input_ids = torch.tensor([[10, 11, 12, 13]])

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            workspace_input_ids,
            max_span_tokens=2,
            no_answer_threshold=0.5,
        )

        self.assertEqual(completion, "Answer: 새벽-14 이다")
        self.assertEqual(meta["status"], "span")
        self.assertEqual(meta["selected_start"], 2)
        self.assertEqual(meta["selected_end"], 3)

    def test_evidence_span_boundary_revision_extends_truncated_identifier_right(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {
                    10: " badge",
                    11: " Frost",
                    12: "-B",
                    13: "adge",
                    14: "-",
                    15: "1",
                    16: "1",
                    17: "1",
                    18: ".",
                }
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
        }

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18]]),
            answer_revision="evidence_span_boundary",
            answer_revision_max_right_tokens=2,
        )

        self.assertEqual(completion, "Answer: Frost-Badge-111")
        self.assertEqual(meta["selected_start"], 1)
        self.assertEqual(meta["selected_end"], 7)
        self.assertEqual(meta["revision"]["status"], "revised")

    def test_evidence_span_boundary_revision_extends_truncated_identifier_left(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {20: " by", 21: " Sen", 22: "a", 23: " Cho", 24: "."}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
        }

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[20, 21, 22, 23, 24]]),
            answer_revision="evidence_span_boundary",
            answer_revision_max_left_tokens=2,
        )

        self.assertEqual(completion, "Answer: Sena Cho")
        self.assertEqual(meta["selected_start"], 1)
        self.assertEqual(meta["selected_end"], 3)
        self.assertEqual(meta["revision"]["status"], "revised")

    def test_evidence_span_boundary_revision_does_not_cross_whitespace_word(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {30: " code", 31: " is", 32: " active"}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[8.0, 0.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[8.0, 0.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
        }

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[30, 31, 32]]),
            answer_revision="evidence_span_boundary",
            answer_revision_max_right_tokens=2,
        )

        self.assertEqual(completion, "Answer: code")
        self.assertEqual(meta["selected_start"], 0)
        self.assertEqual(meta["selected_end"], 0)
        self.assertEqual(meta["revision"]["status"], "unchanged")

    def test_evidence_span_copy_respects_source_token_mask(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {10: " wrong", 11: " right"}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[9.0, 7.0]]),
            "evidence_span_end_logits": torch.tensor([[9.0, 7.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
        }

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[10, 11]]),
            source_token_mask=torch.tensor([[False, True]]),
        )

        self.assertEqual(completion, "Answer: right")
        self.assertEqual(meta["selected_start"], 1)
        self.assertTrue(meta["source_token_mask_active"])
        self.assertEqual(meta["source_token_mask_count"], 1)

    def test_evidence_span_copy_defers_no_answer_when_source_mask_has_tokens(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {10: " wrong", 11: " stone-arch"}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[9.0, 7.0]]),
            "evidence_span_end_logits": torch.tensor([[9.0, 7.0]]),
            "evidence_span_no_answer_logits": torch.tensor([4.0]),
        }

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[10, 11]]),
            no_answer_threshold=0.5,
            source_token_mask=torch.tensor([[False, True]]),
        )

        self.assertEqual(completion, "Answer: stone-arch")
        self.assertEqual(meta["status"], "span")
        self.assertEqual(meta["selected_start"], 1)
        self.assertTrue(meta["no_answer_deferred_by_source_mask"])

    def test_evidence_span_copy_empty_source_mask_still_returns_unknown(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "unused"

        outputs = {
            "evidence_span_start_logits": torch.tensor([[9.0, 7.0]]),
            "evidence_span_end_logits": torch.tensor([[9.0, 7.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
        }

        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[10, 11]]),
            no_answer_threshold=0.5,
            source_token_mask=torch.tensor([[False, False]]),
        )

        self.assertEqual(completion, "Answer: UNKNOWN")
        self.assertEqual(meta["status"], "source_mask_empty")
        self.assertFalse(meta["no_answer_deferred_by_source_mask"])

    def test_evidence_source_token_mask_uses_selected_source_text_only(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class OffsetTokenizer:
            def __call__(self, text, return_offsets_mapping=False, truncation=False, padding=False):
                offsets = []
                ids = []
                for idx, ch in enumerate(text):
                    if ch == "\n":
                        continue
                    ids.append(idx + 1)
                    offsets.append((idx, idx + 1))
                if return_offsets_mapping:
                    return {"input_ids": ids, "offset_mapping": offsets}
                return {"input_ids": ids}

        workspace = (
            "MemoryOS evidence\n"
            "SOURCE=a.md CHUNK=0 SCORE=1.0000\n"
            "wrong\n"
            "SOURCE=b.md CHUNK=1 SCORE=1.0000\n"
            "right"
        )
        token_count = len([ch for ch in workspace if ch != "\n"])

        mask = module.evidence_source_token_mask(
            OffsetTokenizer(),
            workspace,
            torch.arange(token_count).reshape(1, token_count),
            {("b.md", "1")},
        )

        selected_chars = [
            workspace[idx]
            for idx, ch in enumerate(workspace)
            if ch != "\n" and bool(mask[0, len([c for c in workspace[:idx] if c != "\n"])])
        ]
        self.assertEqual("".join(selected_chars), "right")

    def test_evidence_source_token_mask_does_not_include_prompt_after_last_source(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class OffsetTokenizer:
            def __call__(self, text, return_offsets_mapping=False, truncation=False, padding=False):
                offsets = []
                ids = []
                for idx, ch in enumerate(text):
                    if ch == "\n":
                        continue
                    ids.append(idx + 1)
                    offsets.append((idx, idx + 1))
                if return_offsets_mapping:
                    return {"input_ids": ids, "offset_mapping": offsets}
                return {"input_ids": ids}

        prompt = (
            "MemoryOS evidence\n"
            "SOURCE=b.md CHUNK=1 SCORE=1.0000\n"
            "right\n\n"
            "Use the evidence above when it is relevant.\n\n"
            "User prompt:\n"
            "What is the answer?"
        )
        token_count = len([ch for ch in prompt if ch != "\n"])

        mask = module.evidence_source_token_mask(
            OffsetTokenizer(),
            prompt,
            torch.arange(token_count).reshape(1, token_count),
            {("b.md", "1")},
        )

        selected_chars = [
            prompt[idx]
            for idx, ch in enumerate(prompt)
            if ch != "\n" and bool(mask[0, len([c for c in prompt[:idx] if c != "\n"])])
        ]
        self.assertEqual("".join(selected_chars), "right")

    def test_evidence_span_copy_returns_unknown_when_reader_off_or_no_answer(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "unused"

        disabled_completion, disabled_meta = module.evidence_span_copy_from_outputs(
            {
                "evidence_span_start_logits": torch.empty(1, 0),
                "evidence_span_end_logits": torch.empty(1, 0),
                "evidence_span_no_answer_logits": torch.tensor([0.0]),
            },
            TinyTokenizer(),
            torch.tensor([[1, 2, 3]]),
        )
        no_answer_completion, no_answer_meta = module.evidence_span_copy_from_outputs(
            {
                "evidence_span_start_logits": torch.tensor([[4.0, 0.0, 0.0]]),
                "evidence_span_end_logits": torch.tensor([[4.0, 0.0, 0.0]]),
                "evidence_span_no_answer_logits": torch.tensor([4.0]),
            },
            TinyTokenizer(),
            torch.tensor([[1, 2, 3]]),
            no_answer_threshold=0.5,
        )

        self.assertEqual(disabled_completion, "Answer: UNKNOWN")
        self.assertEqual(disabled_meta["status"], "reader_unavailable")
        self.assertEqual(no_answer_completion, "Answer: UNKNOWN")
        self.assertEqual(no_answer_meta["status"], "no_answer")

    def test_evidence_span_copy_returns_unknown_when_span_score_is_too_low(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "low confidence answer"

        completion, meta = module.evidence_span_copy_from_outputs(
            {
                "evidence_span_start_logits": torch.tensor([[2.0, 0.0]]),
                "evidence_span_end_logits": torch.tensor([[2.5, 0.0]]),
                "evidence_span_no_answer_logits": torch.tensor([-4.0]),
            },
            TinyTokenizer(),
            torch.tensor([[1, 2]]),
            min_span_score=12.0,
        )

        self.assertEqual(completion, "Answer: UNKNOWN")
        self.assertEqual(meta["status"], "low_span_score")
        self.assertLess(meta["selected_score"], 12.0)

    def test_truth_gate_blocks_span_copy_when_evidence_is_refuted(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {10: "정답", 11: " 후보"}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[8.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[8.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
            "evidence_support_logits": torch.tensor([3.0]),
            "evidence_refute_logits": torch.tensor([4.0]),
            "evidence_missing_logits": torch.tensor([-4.0]),
            "evidence_causal_gate_logits": torch.tensor([3.0]),
        }
        truth_gate = module.evidence_truth_gate_from_outputs(
            outputs,
            support_threshold=0.6,
            causal_threshold=0.6,
            refute_threshold=0.4,
            missing_threshold=0.4,
        )
        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[10, 11]]),
            truth_gate=truth_gate,
        )

        self.assertFalse(truth_gate["allow"])
        self.assertIn("refute_high", truth_gate["block_reasons"])
        self.assertEqual(completion, "Answer: UNKNOWN")
        self.assertEqual(meta["status"], "truth_gate_blocked")
        self.assertEqual(meta["truth_gate"]["block_reasons"], ["refute_high"])

    def test_truth_gate_allows_span_copy_when_evidence_supports_answer(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class TinyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                vocab = {10: "정답", 11: " 후보"}
                return "".join(vocab[int(i)] for i in ids)

        outputs = {
            "evidence_span_start_logits": torch.tensor([[8.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[8.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-4.0]),
            "evidence_support_logits": torch.tensor([3.0]),
            "evidence_refute_logits": torch.tensor([-4.0]),
            "evidence_missing_logits": torch.tensor([-4.0]),
            "evidence_causal_gate_logits": torch.tensor([3.0]),
        }
        truth_gate = module.evidence_truth_gate_from_outputs(
            outputs,
            support_threshold=0.6,
            causal_threshold=0.6,
            refute_threshold=0.4,
            missing_threshold=0.4,
        )
        completion, meta = module.evidence_span_copy_from_outputs(
            outputs,
            TinyTokenizer(),
            torch.tensor([[10, 11]]),
            truth_gate=truth_gate,
        )

        self.assertTrue(truth_gate["allow"])
        self.assertEqual(completion, "Answer: 정답")
        self.assertEqual(meta["status"], "span")
        self.assertEqual(meta["truth_gate"]["block_reasons"], [])

    def test_truth_gate_uses_trained_bottleneck_gate_over_raw_causal_logit(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        outputs = {
            "evidence_support_logits": torch.tensor([2.0]),
            "evidence_refute_logits": torch.tensor([-3.0]),
            "evidence_missing_logits": torch.tensor([-3.0]),
            "evidence_causal_gate_logits": torch.tensor([-5.0]),
            "evidence_bottleneck_gate": torch.tensor([0.9]),
        }

        truth_gate = module.evidence_truth_gate_from_outputs(
            outputs,
            support_threshold=0.6,
            causal_threshold=0.6,
            refute_threshold=0.4,
            missing_threshold=0.4,
        )

        self.assertTrue(truth_gate["allow"])
        self.assertEqual(truth_gate["causal_source"], "evidence_bottleneck_gate")
        self.assertAlmostEqual(truth_gate["causal_prob"], 0.9)

    def test_short_answer_governor_keeps_first_answer_line(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(
            module.apply_short_answer_governor(
                "Answer: UNKNOWN\n\nThe provided text is a transcript."
            ),
            "Answer: UNKNOWN",
        )
        self.assertEqual(
            module.apply_short_answer_governor(
                "Answer:\n\n**Answer:**\n\n- **이리스** (name: Iris)"
            ),
            "Answer: 이리스",
        )

    def test_memory_eval_token_selection_suppresses_and_bans_repeats(self):
        import torch

        script = Path(__file__).resolve().parents[1] / "scripts" / "95_eval_memory_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_memory_retrieval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        logits = torch.tensor([0.0, 4.0, 3.0])

        self.assertEqual(
            module.select_next_token(logits, suppressed_token_ids=[1]),
            2,
        )
        self.assertEqual(module.no_repeat_ngram_banned_tokens([7, 8, 7], 2), [8])

    def test_workspace_evidence_probe_can_pass_decode_guards(self):
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "117_run_workspace_evidence_path_probe.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("SUPPRESS_VISIBLE_REASONING", runner)
        self.assertIn("NO_REPEAT_NGRAM_SIZE", runner)
        self.assertIn("SHORT_ANSWER_GOVERNOR", runner)
        self.assertIn("--suppress-visible-reasoning-tokens", runner)
        self.assertIn("--no-repeat-ngram-size", runner)
        self.assertIn("--short-answer-governor", runner)

    def test_rescore_script_adds_strict_metrics_and_audit_items(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "116_rescore_memory_eval.py"
        spec = importlib.util.spec_from_file_location("rescore_memory_eval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        records = [
            {
                "id": "case-a",
                "mode": "qtrm_residual_with_evidence",
                "category": "temporal_conflict",
                "question": "What is the code?",
                "answer_aliases": ["VX-913"],
                "completion": "Answer: VX-913. Older code VX-112 is deprecated.",
                "retrieved_target": True,
                "all_targets_retrieved": True,
                "target_recall": 1.0,
            },
            {"summary": {"overall": {"count": 1}}},
        ]

        rescored, summary, audit_items = module.rescore_records(records)

        self.assertEqual(len(rescored), 1)
        self.assertTrue(rescored[0]["hit"])
        self.assertEqual(rescored[0]["match_type"], "normalized_contains")
        self.assertTrue(rescored[0]["needs_human_audit"])
        self.assertEqual(summary["overall"]["normalized_contains_count"], 1)
        self.assertEqual(len(audit_items), 1)
        self.assertIn("Judge whether", audit_items[0]["judge_prompt"])

    def test_rescore_script_roundtrips_jsonl_files(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "116_rescore_memory_eval.py"
        spec = importlib.util.spec_from_file_location("rescore_memory_eval_script", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "in.jsonl"
            out_path = Path(tmp) / "out.jsonl"
            audit_path = Path(tmp) / "audit.jsonl"
            in_path.write_text(
                json.dumps(
                    {
                        "id": "case-a",
                        "mode": "donor_only_with_evidence",
                        "category": "negative_missing",
                        "question": "What is missing?",
                        "answer_aliases": ["UNKNOWN"],
                        "expected_unknown": True,
                        "completion": "UNKNOWN UNKNOWN",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            module.rescore_file(str(in_path), str(out_path), audit_jsonl_out=str(audit_path))

            out_lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            audit_lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(out_lines[0]["match_type"], "unknown_contains")
            self.assertEqual(out_lines[-1]["summary"]["overall"]["human_audit_count"], 1)
            self.assertEqual(audit_lines[0]["id"], "case-a")


if __name__ == "__main__":
    unittest.main()
