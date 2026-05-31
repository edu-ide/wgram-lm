from __future__ import annotations

import subprocess
import sys
import unittest
import importlib.util
from pathlib import Path
from unittest import mock

import torch


class LatestResearchArchitectureClosureTests(unittest.TestCase):
    def _tiny_cfg(self, **overrides):
        from wgram_lm.config import QTRMConfig

        values = dict(
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            max_seq_len=8,
            dropout=0.0,
            delta_backend="torch_gated_delta2_v2",
        )
        values.update(overrides)
        return QTRMConfig(**values)

    def test_backend_registry_exposes_torch_gated_delta2_v2_aliases(self) -> None:
        from wgram_lm import backends

        self.assertEqual(
            backends.get_delta_backend("torch_gated_delta2_v2").__name__,
            "TorchGatedDeltaNet2MixerV2",
        )
        self.assertEqual(
            backends.get_delta_backend("gdn2_v2").__name__,
            "TorchGatedDeltaNet2MixerV2",
        )

    def test_hybrid_block_strict_official_backend_does_not_silently_fallback(self) -> None:
        from wgram_lm import blocks
        from wgram_lm.blocks import OneBodyParallelHybridBlock

        cfg = self._tiny_cfg(
            delta_backend="official_gated_delta2",
            strict_backends=True,
        )

        with mock.patch.object(
            blocks,
            "OfficialGatedDeltaNet2Mixer",
            side_effect=RuntimeError("official kernel unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "official kernel unavailable"):
                OneBodyParallelHybridBlock(cfg, attention_type="gqa")

    def test_hybrid_block_posterior_guidance_forward_uses_instance_config(self) -> None:
        from wgram_lm.blocks import OneBodyParallelHybridBlock

        cfg = self._tiny_cfg(
            core_stochastic_breadth_enabled=True,
            core_stochastic_posterior_guidance=True,
        )
        block = OneBodyParallelHybridBlock(cfg, attention_type="gqa")
        block.train()
        x = torch.randn(2, 3, cfg.d_model)
        gold = torch.randn(2, cfg.d_model)

        out = block(x, rehearsal_gold_target=gold)

        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0].shape, x.shape)

    def test_sparse_slot_router_returns_2d_mask_for_slot_update_contract(self) -> None:
        from wgram_lm.memory.sparse_slot_router import SparseSlotRouter

        router = SparseSlotRouter(d_model=16, num_slots=5, top_k=2, dropout=0.0)
        x = torch.randn(3, 4, 16)

        read, mask, slots = router(x)

        self.assertEqual(read.shape, (3, 16))
        self.assertEqual(mask.shape, (3, 5))
        self.assertEqual(slots.shape, (3, 5, 16))
        updated = router.update_slots(slots, read, mask)
        self.assertEqual(updated.shape, slots.shape)

    def test_hybrid_block_sparse_slot_forward_preserves_batch_time_hidden_shape(self) -> None:
        from wgram_lm.blocks import OneBodyParallelHybridBlock

        cfg = self._tiny_cfg(
            core_sparse_slot_router_enabled=True,
            core_sparse_num_slots=5,
            core_sparse_slot_top_k=2,
        )
        block = OneBodyParallelHybridBlock(cfg, attention_type="gqa")
        x = torch.randn(2, 3, cfg.d_model)

        h, slots, _fast_state = block(x)

        self.assertEqual(h.shape, x.shape)
        self.assertEqual(slots.shape, (2, 5, cfg.d_model))

    def test_trainer_helper_unpacks_three_value_hybrid_outputs(self) -> None:
        from scripts.train_hybrid_ri4_real_continuation_minimal import _unpack_hybrid_output

        h = torch.randn(2, 3, 16)
        slots = torch.randn(2, 5, 16)
        fast_state = object()

        h2, slots2, fast2 = _unpack_hybrid_output((h, slots, fast_state))

        self.assertIs(h2, h)
        self.assertIs(slots2, slots)
        self.assertIs(fast2, fast_state)

    def test_strict_stochastic_breadth_gate_accepts_active_hybrid_replacement(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.gates.check_ssot_stochastic_breadth", "--strict"],
            cwd=".",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("OneBodyParallelHybridBlock", result.stdout)

    def test_raw_eval_runtime_exposes_hybrid_depth_and_stochastic_ablation_modes(self) -> None:
        path = Path("scripts/192_eval_raw_intelligence.py")
        spec = importlib.util.spec_from_file_location("raw_eval_192_latest_closure", path)
        raw_eval = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(raw_eval)

        depth = raw_eval.mode_runtime("hybrid_recurrence_depth_8_no_evidence")
        stoch_off = raw_eval.mode_runtime("hybrid_stochastic_breadth_off_no_evidence")
        gold_off = raw_eval.mode_runtime("hybrid_556_gold_off_no_evidence")
        protection_off = raw_eval.mode_runtime("hybrid_556_protection_off_no_evidence")
        decay_disabled = raw_eval.mode_runtime("hybrid_556_decay_disabled_no_evidence")

        self.assertTrue(depth["use_parallel_hybrid"])
        self.assertEqual(depth["core_steps_override"], 8)
        self.assertTrue(depth["sparse_slots_enabled"])
        self.assertTrue(depth["stochastic_breadth_enabled"])
        self.assertTrue(stoch_off["use_parallel_hybrid"])
        self.assertTrue(stoch_off["stochastic_breadth_ablation_zero"])
        self.assertTrue(gold_off["adaptive_rehearsal_enabled"])
        self.assertTrue(gold_off["gold_state_ablation_zero"])
        self.assertEqual(gold_off["gold_injection_alpha"], 0.0)
        self.assertFalse(protection_off["adaptive_rehearsal_protect_attractor"])
        self.assertTrue(decay_disabled["scheduled_binding_decay_disabled"])

    def test_hybrid_depth_gate_accepts_monotonic_depth_records(self) -> None:
        from wgram_lm.eval.raw_intelligence_gate import build_raw_intelligence_gate

        records = [
            {"id": "a", "mode": "hybrid_recurrence_off_no_evidence", "hit": False, "completion": "0"},
            {"id": "a", "mode": "hybrid_recurrence_depth_1_no_evidence", "hit": False, "completion": "1"},
            {"id": "a", "mode": "hybrid_recurrence_depth_4_no_evidence", "hit": False, "completion": "2"},
            {"id": "a", "mode": "hybrid_recurrence_depth_8_no_evidence", "hit": True, "completion": "3"},
            {"id": "a", "mode": "hybrid_recurrence_depth_12_no_evidence", "hit": True, "completion": "4"},
            {"id": "b", "mode": "hybrid_recurrence_off_no_evidence", "hit": False, "completion": "0"},
            {"id": "b", "mode": "hybrid_recurrence_depth_1_no_evidence", "hit": False, "completion": "1"},
            {"id": "b", "mode": "hybrid_recurrence_depth_4_no_evidence", "hit": True, "completion": "2"},
            {"id": "b", "mode": "hybrid_recurrence_depth_8_no_evidence", "hit": True, "completion": "3"},
            {"id": "b", "mode": "hybrid_recurrence_depth_12_no_evidence", "hit": True, "completion": "4"},
        ]

        gate = build_raw_intelligence_gate(records, gate_type="hybrid_recurrence_depth_scaling")

        self.assertEqual(gate["status"], "accepted")
        self.assertIn("depth_scaling_gain_present", gate["passed_checks"])
        self.assertIn("depth_scaling_monotonic", gate["passed_checks"])
        self.assertIn("deepest_hybrid_beats_recurrence_off", gate["passed_checks"])

    def test_hybrid_depth_gate_rejects_non_monotonic_depth_records(self) -> None:
        from wgram_lm.eval.raw_intelligence_gate import build_raw_intelligence_gate

        records = [
            {"id": "a", "mode": "hybrid_recurrence_off_no_evidence", "hit": False, "completion": "0"},
            {"id": "a", "mode": "hybrid_recurrence_depth_1_no_evidence", "hit": True, "completion": "1"},
            {"id": "a", "mode": "hybrid_recurrence_depth_4_no_evidence", "hit": False, "completion": "2"},
            {"id": "a", "mode": "hybrid_recurrence_depth_8_no_evidence", "hit": True, "completion": "3"},
            {"id": "a", "mode": "hybrid_recurrence_depth_12_no_evidence", "hit": True, "completion": "4"},
            {"id": "b", "mode": "hybrid_recurrence_off_no_evidence", "hit": False, "completion": "0"},
            {"id": "b", "mode": "hybrid_recurrence_depth_1_no_evidence", "hit": True, "completion": "1"},
            {"id": "b", "mode": "hybrid_recurrence_depth_4_no_evidence", "hit": True, "completion": "2"},
            {"id": "b", "mode": "hybrid_recurrence_depth_8_no_evidence", "hit": True, "completion": "3"},
            {"id": "b", "mode": "hybrid_recurrence_depth_12_no_evidence", "hit": True, "completion": "4"},
        ]

        gate = build_raw_intelligence_gate(records, gate_type="hybrid_recurrence_depth_scaling")

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("depth_scaling_not_monotonic", gate["failed_checks"])

    def test_hybrid_556_gate_accepts_full_matrix_with_clean_drops(self) -> None:
        from wgram_lm.eval.raw_intelligence_gate import build_raw_intelligence_gate

        records = [
            {"id": "a", "mode": "hybrid_556_full_no_evidence", "hit": True, "completion": "ok"},
            {"id": "a", "mode": "hybrid_556_stoch_zero_no_evidence", "hit": False, "completion": "bad"},
            {"id": "a", "mode": "hybrid_556_gold_off_no_evidence", "hit": False, "completion": "bad"},
            {"id": "a", "mode": "hybrid_556_protection_off_no_evidence", "hit": False, "completion": "bad"},
            {"id": "a", "mode": "hybrid_556_decay_disabled_no_evidence", "hit": False, "completion": "bad"},
            {"id": "b", "mode": "hybrid_556_full_no_evidence", "hit": True, "completion": "ok"},
            {"id": "b", "mode": "hybrid_556_stoch_zero_no_evidence", "hit": True, "completion": "ok"},
            {"id": "b", "mode": "hybrid_556_gold_off_no_evidence", "hit": False, "completion": "bad"},
            {"id": "b", "mode": "hybrid_556_protection_off_no_evidence", "hit": False, "completion": "bad"},
            {"id": "b", "mode": "hybrid_556_decay_disabled_no_evidence", "hit": False, "completion": "bad"},
        ]

        gate = build_raw_intelligence_gate(records, gate_type="hybrid_556_causal_matrix")

        self.assertEqual(gate["status"], "accepted")
        self.assertIn("full_beats_hybrid_556_stoch_zero_no_evidence", gate["passed_checks"])
        self.assertIn("full_beats_hybrid_556_gold_off_no_evidence", gate["passed_checks"])
        self.assertIn("full_beats_hybrid_556_protection_off_no_evidence", gate["passed_checks"])
        self.assertIn("full_beats_hybrid_556_decay_disabled_no_evidence", gate["passed_checks"])

    def test_hybrid_556_gate_rejects_non_causal_ablation(self) -> None:
        from wgram_lm.eval.raw_intelligence_gate import build_raw_intelligence_gate

        records = [
            {"id": "a", "mode": "hybrid_556_full_no_evidence", "hit": True, "completion": "ok"},
            {"id": "a", "mode": "hybrid_556_stoch_zero_no_evidence", "hit": True, "completion": "ok"},
            {"id": "a", "mode": "hybrid_556_gold_off_no_evidence", "hit": False, "completion": "bad"},
            {"id": "a", "mode": "hybrid_556_protection_off_no_evidence", "hit": False, "completion": "bad"},
            {"id": "a", "mode": "hybrid_556_decay_disabled_no_evidence", "hit": False, "completion": "bad"},
        ]

        gate = build_raw_intelligence_gate(records, gate_type="hybrid_556_causal_matrix")

        self.assertEqual(gate["status"], "rejected")
        self.assertIn("full_does_not_beat_hybrid_556_stoch_zero_no_evidence", gate["failed_checks"])


if __name__ == "__main__":
    unittest.main()
