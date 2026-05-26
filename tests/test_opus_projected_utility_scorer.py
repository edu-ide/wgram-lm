from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "614_score_opus_projected_utility.py"


def load_module():
    spec = importlib.util.spec_from_file_location("opus_projected_utility_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OPUSProjectedUtilityScorerTests(unittest.TestCase):
    def test_countsketch_is_deterministic_and_seeded(self) -> None:
        module = load_module()
        values = torch.arange(12, dtype=torch.float32)

        first = module.sketch_tensor(values, projection_dim=8, seed=123)
        second = module.sketch_tensor(values, projection_dim=8, seed=123)
        different = module.sketch_tensor(values, projection_dim=8, seed=124)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(tuple(first.shape), (8,))

    def test_redundancy_adjusted_order_prefers_proxy_aligned_update(self) -> None:
        module = load_module()
        target = torch.tensor([1.0, 0.0])
        vectors = [
            torch.tensor([-1.0, 0.0]),
            torch.tensor([0.2, 1.0]),
            torch.tensor([1.0, 0.0]),
        ]

        order = module.redundancy_adjusted_order(
            vectors,
            target,
            lr=1.0,
            redundancy_weight=0.0,
        )

        self.assertEqual(order[0]["index"], 2)
        self.assertGreater(order[0]["utility"], order[-1]["utility"])

    def test_multi_proxy_minimax_prefers_candidate_that_does_not_hurt_weak_group(self) -> None:
        module = load_module()
        targets = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        vectors = [
            torch.tensor([10.0, -1.0]),
            torch.tensor([1.0, 1.0]),
        ]

        mean_order = module.multi_proxy_redundancy_adjusted_order(
            vectors,
            targets,
            ["language", "gd"],
            lr=1.0,
            redundancy_weight=0.0,
            score_mode="mean",
            mean_weight=0.25,
        )
        minimax_order = module.multi_proxy_redundancy_adjusted_order(
            vectors,
            targets,
            ["language", "gd"],
            lr=1.0,
            redundancy_weight=0.0,
            score_mode="minimax_mean",
            mean_weight=0.25,
        )

        self.assertEqual(mean_order[0]["index"], 0)
        self.assertEqual(minimax_order[0]["index"], 1)
        self.assertEqual(minimax_order[0]["min_proxy_alignment"], 1.0)
        self.assertEqual(minimax_order[0]["proxy_group_alignments"], {"language": 1.0, "gd": 1.0})

    def test_proxy_loader_accepts_language_heldout_shape(self) -> None:
        module = load_module()
        rows = module.load_proxy_rows(ROOT / "data" / "eval" / "prefixlm_language_heldout.jsonl", max_rows=2)

        self.assertEqual(len(rows), 2)
        self.assertIn("sky", rows[0].instruction.lower())
        self.assertTrue(rows[0].response)

    def test_proxy_loader_accepts_generalization_dynamics_intelligence_answer_shape(self) -> None:
        module = load_module()
        rows = module.load_proxy_rows(
            ROOT / "data" / "eval" / "generalization_dynamics_lite_probe.jsonl",
            max_rows=1,
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("Q:", rows[0].instruction)
        self.assertEqual(rows[0].response, " Negative")
        self.assertEqual(rows[0].bucket, "flipped_answer_icl")

    def test_proxy_loader_accepts_multiple_proxy_files(self) -> None:
        module = load_module()
        rows = module.load_proxy_rows(
            f"{ROOT / 'data' / 'eval' / 'prefixlm_language_heldout.jsonl'} "
            f"{ROOT / 'data' / 'eval' / 'generalization_dynamics_lite_probe.jsonl'}",
            max_rows=9,
        )

        self.assertGreaterEqual(len(rows), 9)
        self.assertTrue(any(row.bucket == "flipped_answer_icl" for row in rows))

    def test_proxy_grouping_exposes_language_and_gd_buckets(self) -> None:
        module = load_module()
        rows = module.load_proxy_rows(
            f"{ROOT / 'data' / 'eval' / 'prefixlm_language_heldout.jsonl'} "
            f"{ROOT / 'data' / 'eval' / 'generalization_dynamics_lite_probe.jsonl'}",
            max_rows=14,
        )

        grouped = module.group_proxy_rows(rows, grouping="source_file_bucket")

        self.assertEqual(len(rows), 14)
        self.assertTrue(any("prefixlm_language_heldout" in name for name in grouped))
        self.assertTrue(any("generalization_dynamics_lite_probe" in name for name in grouped))
        self.assertGreater(len(grouped), 2)

    def test_proxy_group_cap_is_deterministic_per_group(self) -> None:
        module = load_module()
        rows = [
            module.TextRow("proxy.jsonl", index, "Q", "A", bucket="a" if index < 5 else "b")
            for index in range(10)
        ]
        grouped = module.group_proxy_rows(rows, grouping="bucket")

        first = module.cap_proxy_groups(grouped, max_rows_per_group=2, seed=7)
        second = module.cap_proxy_groups(grouped, max_rows_per_group=2, seed=7)

        self.assertEqual({name: len(values) for name, values in first.items()}, {"a": 2, "b": 2})
        self.assertEqual(
            {name: [row.row_index for row in values] for name, values in first.items()},
            {name: [row.row_index for row in values] for name, values in second.items()},
        )

    def test_seq_len_default_uses_checkpoint_contract(self) -> None:
        module = load_module()

        self.assertEqual(module.resolve_checkpoint_seq_len(0, {"seq_len": 1024}), 1024)
        self.assertEqual(module.resolve_checkpoint_seq_len(768, {"seq_len": 1024}), 768)
        self.assertEqual(module.resolve_checkpoint_seq_len(0, {}, fallback=384), 384)

    def test_proxy_filter_drops_rows_whose_prompt_crowds_out_answer(self) -> None:
        module = load_module()

        class Prepare:
            EOS_TOKEN_ID = 1

            @staticmethod
            def render_instruction(text: str) -> str:
                return text

            @staticmethod
            def render_response(text: str) -> str:
                return text

            @staticmethod
            def byte_ids(text: str) -> list[int]:
                return list(text.encode("utf-8"))

        rows = [
            module.TextRow("proxy.jsonl", 0, "Q", "A", bucket="short"),
            module.TextRow("proxy.jsonl", 1, "Q" * 100, "A", bucket="long"),
        ]

        valid, skipped = module.filter_proxy_rows_with_supervised_targets(
            Prepare(),
            rows,
            seq_len=16,
            train_instruction_tokens=False,
        )

        self.assertEqual([row.bucket for row in valid], ["short"])
        self.assertEqual([row.bucket for row in skipped], ["long"])

    def test_adamw_state_falls_back_to_identity_for_missing_per_parameter_state(self) -> None:
        module = load_module()
        layer = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        layer.weight.grad = torch.ones_like(layer.weight)
        stats = module.PreconditionerFallbackStats()

        update = module.adamw_effective_grad(
            layer.weight,
            optimizer,
            parameter_name="weight",
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
            preconditioner="adamw_state",
            stats=stats,
        )
        report = stats.to_report()

        self.assertTrue(torch.equal(update, torch.ones_like(layer.weight)))
        self.assertEqual(report["missing_exp_avg_sq_parameter_tensors"], 1)
        self.assertEqual(report["identity_fallback_update_calls"], 1)
        self.assertEqual(report["adamw_preconditioned_update_calls"], 0)
        self.assertIn("weight", report["missing_exp_avg_sq_parameter_name_examples"])

    def test_no_target_token_error_is_classified_for_candidate_skip(self) -> None:
        module = load_module()

        self.assertTrue(
            module.is_no_supervised_target_tokens_error(
                ValueError("encoded batch has no supervised target tokens")
            )
        )
        self.assertFalse(module.is_no_supervised_target_tokens_error(ValueError("other error")))


if __name__ == "__main__":
    unittest.main()
