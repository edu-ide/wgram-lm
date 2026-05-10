from pathlib import Path
import importlib.util
import unittest

import torch


def _load_module():
    path = Path("scripts/247_probe_qtrm_gold_token_ranks.py")
    spec = importlib.util.spec_from_file_location("gold_token_rank_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GoldTokenRankProbeTests(unittest.TestCase):
    def test_target_rank_stats_does_not_treat_all_zero_tie_as_unique_top1(self):
        module = _load_module()

        stats = module._target_rank_stats(torch.zeros(5), 3)

        self.assertEqual(stats["strict_rank"], 1)
        self.assertEqual(stats["tie_count"], 5)
        self.assertEqual(stats["top_tie_count"], 5)
        self.assertFalse(stats["unique_top1"])

    def test_target_rank_stats_accepts_unique_top1(self):
        module = _load_module()
        logits = torch.tensor([0.0, 3.0, 1.0])

        stats = module._target_rank_stats(logits, 1)

        self.assertEqual(stats["strict_rank"], 1)
        self.assertEqual(stats["tie_count"], 1)
        self.assertEqual(stats["top_tie_count"], 1)
        self.assertTrue(stats["unique_top1"])
        self.assertEqual(stats["target_logit"], 3.0)
        self.assertEqual(stats["max_logit"], 3.0)
        self.assertEqual(stats["target_minus_top_logit"], 0.0)

    def test_target_rank_stats_reports_margin_when_target_is_not_top(self):
        module = _load_module()
        logits = torch.tensor([0.0, 3.0, 1.0])

        stats = module._target_rank_stats(logits, 2)

        self.assertEqual(stats["strict_rank"], 2)
        self.assertEqual(stats["target_logit"], 1.0)
        self.assertEqual(stats["max_logit"], 3.0)
        self.assertEqual(stats["target_minus_top_logit"], -2.0)

    def test_first_content_token_index_skips_leading_whitespace(self):
        module = _load_module()

        self.assertEqual(module._first_content_token_index([" ", "\n", "8"]), 2)
        self.assertEqual(module._first_content_token_index(["52"]), 0)

    def test_select_position_top_tokens_keeps_first_and_content_positions(self):
        module = _load_module()
        first = [{"token": " ", "token_id": 220}]
        content = [{"token": "8", "token_id": 23}]
        later = [{"token": "0", "token_id": 15}]

        selected = module._select_position_top_tokens(
            [first, content, later],
            content_index=1,
        )

        self.assertEqual(selected["first_top_tokens"], first)
        self.assertEqual(selected["content_first_top_tokens"], content)

    def test_runtime_disable_kwargs_include_l4_ablation_flags(self):
        module = _load_module()

        kwargs = module._model_disable_kwargs_from_runtime(
            {
                "disable_core": True,
                "disable_core_source_position_binder": True,
                "disable_core_primitive_role_value_executor": True,
                "disable_core_role_value_vocab_renderer": True,
                "disable_answer_state_loop_halt_gate": True,
            }
        )

        self.assertTrue(kwargs["disable_core"])
        self.assertTrue(kwargs["disable_core_source_position_binder"])
        self.assertTrue(kwargs["disable_core_primitive_role_value_executor"])
        self.assertTrue(kwargs["disable_core_role_value_vocab_renderer"])
        self.assertTrue(kwargs["disable_answer_state_loop_halt_gate"])
        self.assertFalse(kwargs["disable_core_role_value_answer_bridge"])

    def test_parser_exposes_base_checkpoint_for_delta_probe(self):
        module = _load_module()

        args = module.build_arg_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--checkpoint",
                "delta.pt",
                "--base-checkpoint",
                "base.pt",
                "--cases",
                "cases.jsonl",
                "--out",
                "ranks.jsonl",
            ]
        )

        self.assertEqual(args.base_checkpoint, "base.pt")


if __name__ == "__main__":
    unittest.main()
