import importlib.util
import unittest
from pathlib import Path


def load_eval_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "192_eval_raw_intelligence.py"
    spec = importlib.util.spec_from_file_location("raw_intelligence_eval_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RawIntelligenceEvalScriptTests(unittest.TestCase):
    def test_mode_runtime_maps_core_depth_without_hidden_evidence(self):
        module = load_eval_script()

        runtime = module.mode_runtime("qtrm_core_steps_4_no_evidence")

        self.assertEqual(runtime["mode"], "qtrm_core_steps_4_no_evidence")
        self.assertEqual(runtime["core_steps_override"], 4)
        self.assertFalse(runtime["disable_core"])
        self.assertFalse(runtime["memoryos_used"])
        self.assertFalse(runtime["retrieval_used"])

    def test_mode_runtime_maps_core_off_and_donor_only(self):
        module = load_eval_script()

        core_off = module.mode_runtime("qtrm_core_off_no_evidence")
        donor = module.mode_runtime("donor_only_no_evidence")

        self.assertTrue(core_off["disable_core"])
        self.assertEqual(core_off["qtrm_logits_scale"], None)
        self.assertEqual(core_off["donor_logits_scale"], None)
        self.assertFalse(donor["disable_core"])
        self.assertEqual(donor["qtrm_logits_scale"], 0.0)
        self.assertEqual(donor["donor_logits_scale"], 1.0)

    def test_mode_runtime_maps_low_donor_and_qtrm_only(self):
        module = load_eval_script()

        low_donor = module.mode_runtime("qtrm_core_steps_8_low_donor_no_evidence")
        qtrm_only = module.mode_runtime("qtrm_core_steps_8_qtrm_only_no_evidence")
        core_off_qtrm_only = module.mode_runtime("qtrm_core_off_qtrm_only_no_evidence")

        self.assertEqual(low_donor["core_steps_override"], 8)
        self.assertEqual(low_donor["qtrm_logits_scale"], 1.0)
        self.assertEqual(low_donor["donor_logits_scale"], 0.25)
        self.assertFalse(low_donor["disable_core"])
        self.assertEqual(qtrm_only["qtrm_logits_scale"], 1.0)
        self.assertEqual(qtrm_only["donor_logits_scale"], 0.0)
        self.assertFalse(qtrm_only["disable_core"])
        self.assertTrue(core_off_qtrm_only["disable_core"])
        self.assertEqual(core_off_qtrm_only["qtrm_logits_scale"], 1.0)
        self.assertEqual(core_off_qtrm_only["donor_logits_scale"], 0.0)

    def test_mode_runtime_maps_explicit_fusion_scales(self):
        module = load_eval_script()

        donor_scaled = module.mode_runtime("qtrm_core_steps_8_donor_scale_0p50_no_evidence")
        both_scaled = module.mode_runtime(
            "qtrm_core_steps_8_qtrm_scale_0p75_donor_scale_0p50_no_evidence"
        )

        self.assertEqual(donor_scaled["core_steps_override"], 8)
        self.assertEqual(donor_scaled["qtrm_logits_scale"], 1.0)
        self.assertEqual(donor_scaled["donor_logits_scale"], 0.5)
        self.assertFalse(donor_scaled["disable_core"])
        self.assertEqual(both_scaled["qtrm_logits_scale"], 0.75)
        self.assertEqual(both_scaled["donor_logits_scale"], 0.5)
        self.assertFalse(both_scaled["memoryos_used"])

    def test_mode_runtime_maps_temporal_spatial_context_off(self):
        module = load_eval_script()

        runtime = module.mode_runtime("qtrm_core_steps_8_temporal_spatial_off_no_evidence")

        self.assertEqual(runtime["core_steps_override"], 8)
        self.assertFalse(runtime["disable_core"])
        self.assertTrue(runtime["disable_temporal_spatial_context"])
        self.assertFalse(runtime["memoryos_used"])
        self.assertFalse(runtime["retrieval_used"])

    def test_mode_runtime_maps_transition_state_off(self):
        module = load_eval_script()

        runtime = module.mode_runtime("qtrm_core_steps_8_transition_state_off_no_evidence")

        self.assertEqual(runtime["core_steps_override"], 8)
        self.assertFalse(runtime["disable_core"])
        self.assertTrue(runtime["disable_transition_state"])
        self.assertFalse(runtime["memoryos_used"])
        self.assertFalse(runtime["retrieval_used"])

    def test_depth_gate_runner_can_include_transition_state_off_mode(self):
        text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "193_run_pure_recursive_reasoning_depth_gate.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("INCLUDE_TRANSITION_STATE_OFF", text)
        self.assertIn("INCLUDE_FAMILIES", text)
        self.assertIn("CHOICE_SCORE_NORMALIZATION", text)
        self.assertIn("--choice-score-normalization", text)
        self.assertIn("--include-family", text)
        self.assertIn("qtrm_core_steps_8_transition_state_off_no_evidence", text)

    def test_case_temporal_spatial_context_converts_json_vectors(self):
        module = load_eval_script()

        one_token = module._case_temporal_spatial_context(
            {"temporal_spatial_context": [0.1, 0.2, 0.3]},
            device="cpu",
        )
        two_tokens = module._case_temporal_spatial_context(
            {"temporal_spatial_context": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            device="cpu",
        )

        self.assertEqual(tuple(one_token.shape), (1, 3))
        self.assertEqual(tuple(two_tokens.shape), (1, 2, 3))
        self.assertIsNone(module._case_temporal_spatial_context({}, device="cpu"))

    def test_case_temporal_spatial_context_rejects_bad_shape(self):
        module = load_eval_script()

        with self.assertRaisesRegex(ValueError, "temporal_spatial_context"):
            module._case_temporal_spatial_context(
                {"temporal_spatial_context": [[[0.1]], [[0.2]]]},
                device="cpu",
            )

    def test_score_case_record_marks_no_shortcuts(self):
        module = load_eval_script()

        case = {
            "id": "arith-chain-000",
            "raw_intelligence_axis": "pure_recursive_reasoning",
            "category": "arithmetic_chain",
            "task_family": "arithmetic_chain",
            "reasoning_family": "sequential_arithmetic",
            "expected_paradigm": "hybrid_or_cot",
            "requires_stochasticity": False,
            "parallel_depth_estimate": 0,
            "serial_trace_length_estimate": 3,
            "question": "Compute 1+1.",
            "prompt": "Question: Compute 1+1.",
            "answer_aliases": ["2"],
            "choices": ["2", "3"],
        }

        record = module.score_case_record(
            case,
            mode="qtrm_core_steps_2_no_evidence",
            completion="Answer: 2",
            runtime={"core_steps_override": 2, "disable_core": False},
            generated_tokens=3,
        )

        self.assertTrue(record["hit"])
        self.assertEqual(record["mode"], "qtrm_core_steps_2_no_evidence")
        self.assertEqual(record["core_steps_requested"], 2)
        self.assertFalse(record["memoryos_used"])
        self.assertFalse(record["retrieval_used"])
        self.assertEqual(record["evidence_token_count"], 0)
        self.assertEqual(record["workspace_memory_token_count"], 0)
        self.assertEqual(record["reasoning_family"], "sequential_arithmetic")
        self.assertEqual(record["expected_paradigm"], "hybrid_or_cot")
        self.assertFalse(record["requires_stochasticity"])
        self.assertEqual(record["parallel_depth_estimate"], 0)
        self.assertEqual(record["serial_trace_length_estimate"], 3)

    def test_score_case_record_marks_temporal_spatial_context_status(self):
        module = load_eval_script()

        record = module.score_case_record(
            {
                "id": "temporal-000",
                "prompt": "Question?",
                "answer_aliases": ["green"],
                "temporal_spatial_context": [[0.0] * 8],
            },
            mode="qtrm_core_steps_8_temporal_spatial_off_no_evidence",
            completion="green",
            runtime={
                "core_steps_override": 8,
                "disable_core": False,
                "disable_temporal_spatial_context": True,
            },
            generated_tokens=0,
        )

        self.assertTrue(record["temporal_spatial_context_available"])
        self.assertTrue(record["disable_temporal_spatial_context"])
        self.assertEqual(record["temporal_spatial_context_token_count"], 0)

    def test_cli_defaults_to_forced_choice_scoring(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args([])

        self.assertEqual(args.scoring, "forced_choice")
        self.assertEqual(args.choice_score_normalization, "mean")

    def test_causal_choice_prefixes_do_not_include_future_answer_tokens(self):
        import torch

        module = load_eval_script()

        class FakeTokenizer:
            def __call__(
                self,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=None,
                padding=False,
                add_special_tokens=True,
            ):
                self.last_call_text = text
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

            def encode(self, text, add_special_tokens=False):
                self.last_encode_text = text
                return [21, 22]

        prefixes = module._causal_choice_prefixes(
            FakeTokenizer(),
            "Question?\nAnswer:",
            "right",
            max_length=16,
            device="cpu",
        )

        self.assertEqual(
            [(ids.tolist(), int(target)) for ids, _mask, target in prefixes],
            [
                ([[10, 11]], 21),
                ([[10, 11, 21]], 22),
            ],
        )

    def test_cli_defaults_to_raw_depth_modes(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args([])

        self.assertIn("donor_only_no_evidence", module.resolve_modes(args))
        self.assertIn("qtrm_core_steps_8_no_evidence", module.resolve_modes(args))

    def test_cli_custom_modes_replace_defaults(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(
            ["--mode", "qtrm_core_steps_4_no_evidence"]
        )

        self.assertEqual(module.resolve_modes(args), ["qtrm_core_steps_4_no_evidence"])

    def test_forced_choice_tie_returns_non_answer_sentinel(self):
        module = load_eval_script()

        original = module._answer_choice_logprob
        try:
            module._answer_choice_logprob = lambda *args, **kwargs: 0.0
            completion, scored = module._forced_choice_case(
                None,
                None,
                None,
                {
                    "prompt": "Question?",
                    "answer_aliases": ["right"],
                    "choices": ["right", "wrong"],
                },
                runtime={},
                max_length=16,
                device="cpu",
            )
        finally:
            module._answer_choice_logprob = original

        self.assertEqual(completion, module.FORCED_CHOICE_TIE_COMPLETION)
        self.assertTrue(all(row["tied_for_best"] for row in scored))
        self.assertFalse(module.score_answer(completion, ["right"])["hit"])

    def test_forced_choice_strict_winner_still_returns_choice(self):
        module = load_eval_script()

        def fake_logprob(*args, **kwargs):
            choice = args[4]
            return 1.0 if choice == "right" else 0.0

        original = module._answer_choice_logprob
        try:
            module._answer_choice_logprob = fake_logprob
            completion, scored = module._forced_choice_case(
                None,
                None,
                None,
                {
                    "prompt": "Question?",
                    "answer_aliases": ["right"],
                    "choices": ["right", "wrong"],
                },
                runtime={},
                max_length=16,
                device="cpu",
            )
        finally:
            module._answer_choice_logprob = original

        self.assertEqual(completion, "right")
        self.assertTrue(scored[0]["tied_for_best"])
        self.assertFalse(scored[1]["tied_for_best"])
        self.assertEqual(scored[0]["score_normalization"], "sum")
        self.assertIn("logprob_sum", scored[0])
        self.assertIn("token_count", scored[0])

    def test_forced_choice_mean_normalization_removes_short_answer_bias(self):
        module = load_eval_script()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                if "208,204" in text:
                    return [10, 11, 12, 13]
                if "EMPTY" in text:
                    return [20]
                return [99]

        def fake_logprob(*args, **kwargs):
            choice = args[4]
            return -4.0 if choice == "208,204" else -2.0

        original = module._answer_choice_logprob
        try:
            module._answer_choice_logprob = fake_logprob
            completion, scored = module._forced_choice_case(
                None,
                None,
                FakeTokenizer(),
                {
                    "prompt": "Question?",
                    "answer_aliases": ["208,204"],
                    "choices": ["208,204", "EMPTY"],
                },
                runtime={},
                max_length=16,
                device="cpu",
                choice_score_normalization="mean",
            )
        finally:
            module._answer_choice_logprob = original

        self.assertEqual(completion, "208,204")
        self.assertEqual(scored[0]["logprob_sum"], -4.0)
        self.assertEqual(scored[0]["token_count"], 4)
        self.assertEqual(scored[0]["logprob"], -1.0)
        self.assertEqual(scored[0]["score_normalization"], "mean")

    def test_forced_choice_records_conflict_gate_mean_telemetry(self):
        module = load_eval_script()

        def fake_logprob(*args, **kwargs):
            choice = args[4]
            telemetry = kwargs["telemetry"]
            telemetry["donor_qtrm_conflict_gate_mean_values"].append(
                0.25 if choice == "right" else 0.75
            )
            return 1.0 if choice == "right" else 0.0

        original = module._answer_choice_logprob
        try:
            module._answer_choice_logprob = fake_logprob
            completion, scored = module._forced_choice_case(
                None,
                None,
                None,
                {
                    "prompt": "Question?",
                    "answer_aliases": ["right"],
                    "choices": ["right", "wrong"],
                },
                runtime={},
                max_length=16,
                device="cpu",
            )
        finally:
            module._answer_choice_logprob = original

        self.assertEqual(completion, "right")
        self.assertEqual(scored[0]["donor_qtrm_conflict_gate_mean"], 0.25)
        self.assertEqual(scored[1]["donor_qtrm_conflict_gate_mean"], 0.75)

    def test_forced_choice_passes_temporal_spatial_context(self):
        module = load_eval_script()

        calls = []

        def fake_logprob(*args, **kwargs):
            calls.append(kwargs)
            choice = args[4]
            return 1.0 if choice == "right" else 0.0

        original = module._answer_choice_logprob
        try:
            module._answer_choice_logprob = fake_logprob
            completion, _scored = module._forced_choice_case(
                None,
                None,
                None,
                {
                    "prompt": "Question?",
                    "answer_aliases": ["right"],
                    "choices": ["right", "wrong"],
                    "temporal_spatial_context": [[0.0] * 8, [1.0] * 8],
                },
                runtime={
                    "core_steps_override": 8,
                    "disable_temporal_spatial_context": False,
                },
                max_length=16,
                device="cpu",
            )
        finally:
            module._answer_choice_logprob = original

        self.assertEqual(completion, "right")
        self.assertEqual(tuple(calls[0]["temporal_spatial_context"].shape), (1, 2, 8))
        self.assertFalse(calls[0]["runtime"]["disable_temporal_spatial_context"])

    def test_cli_accepts_causal_forced_choice_scoring(self):
        module = load_eval_script()

        args = module.build_arg_parser().parse_args(["--scoring", "causal_forced_choice"])

        self.assertEqual(args.scoring, "causal_forced_choice")

    def test_cli_can_enable_donor_qtrm_conflict_gate_probe(self):
        module = load_eval_script()
        from qtrm_mm import QTRMConfig

        args = module.build_arg_parser().parse_args(
            [
                "--donor-qtrm-conflict-gate",
                "--donor-qtrm-conflict-qtrm-scale",
                "0.25",
            ]
        )
        cfg = QTRMConfig()

        module.apply_eval_model_overrides(cfg, args)

        self.assertTrue(cfg.donor_qtrm_conflict_gate_enabled)
        self.assertEqual(cfg.donor_qtrm_conflict_qtrm_scale, 0.25)


if __name__ == "__main__":
    unittest.main()
