import argparse
import unittest


class SsoTContractTests(unittest.TestCase):
    def test_constants_define_canonical_answer_path(self):
        from qtrm_mm.eval.ssot_contract import (
            CANONICAL_ANSWER_CHANNEL,
            CANONICAL_EVIDENCE_INJECTION,
        )

        self.assertEqual(CANONICAL_EVIDENCE_INJECTION, "ssot")
        self.assertEqual(CANONICAL_ANSWER_CHANNEL, "greedy")

    def test_canonical_contract_accepts_single_stream_greedy_path(self):
        from qtrm_mm.eval.ssot_contract import validate_canonical_ssot_args

        args = argparse.Namespace(
            require_canonical_ssot=True,
            evidence_injection="ssot",
            answer_channel="greedy",
        )

        validate_canonical_ssot_args(args)

    def test_canonical_contract_rejects_probe_paths(self):
        from qtrm_mm.eval.ssot_contract import validate_canonical_ssot_args

        workspace = argparse.Namespace(
            require_canonical_ssot=True,
            evidence_injection="workspace",
            answer_channel="greedy",
        )
        span_copy = argparse.Namespace(
            require_canonical_ssot=True,
            evidence_injection="ssot",
            answer_channel="evidence_span_copy",
        )

        with self.assertRaisesRegex(ValueError, "requires --evidence-injection ssot"):
            validate_canonical_ssot_args(workspace)
        with self.assertRaisesRegex(ValueError, "requires --answer-channel greedy"):
            validate_canonical_ssot_args(span_copy)

    def test_canonical_model_contract_accepts_single_trace_trm_without_lewm(self):
        from qtrm_mm.config import load_config
        from qtrm_mm.eval.ssot_contract import validate_canonical_model_config

        cfg = load_config(
            "configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml"
        )

        validate_canonical_model_config(cfg)

    def test_canonical_model_contract_rejects_lewm_world_model_path(self):
        from qtrm_mm.config import load_config
        from qtrm_mm.eval.ssot_contract import validate_canonical_model_config

        cfg = load_config("configs/qwen35_2b_4090_pure_recursive_lewm_staged_s200.yaml")

        with self.assertRaisesRegex(ValueError, "core_world_model_enabled"):
            validate_canonical_model_config(cfg)
