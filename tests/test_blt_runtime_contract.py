from __future__ import annotations

import argparse
import unittest


def _args(**overrides):
    values = {
        "patch_boundary_mode": "hnet_dechunk",
        "decoder_latent_mode": "one_body",
        "backbone": "trm_qwen35_3to1",
        "think_structure": "trm_dual_z",
        "delta_backend": "official_gated_delta2",
        "train_think_steps": 4,
        "answer_attractor_ce_weight": 0.08,
        "answer_attractor_monotonic_weight": 0.02,
        "answer_attractor_residual_wrong_weight": 0.0,
        "core_world_model_enabled": False,
        "loss_core_world_model_weight": 0.0,
        "use_parallel_hybrid_block": False,
        "imta_trajectories": 3,
        "imta_noise_std": 0.02,
        "imta_selector_temperature": 0.7,
        "imta_adapter_gate_init": -1.0,
        "imta_diversity_weight": 0.03,
        "own_latent_prediction_enabled": True,
        "own_latent_prediction_weight": 0.05,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BLTRuntimeContractTests(unittest.TestCase):
    def test_contract_names_executable_blt_path(self) -> None:
        from wgram_lm.architecture.blt_runtime_contract import build_active_blt_runtime_contract

        contract = build_active_blt_runtime_contract(_args())

        self.assertEqual(contract["active_model_class"], "wgram_lm.models.blt_prefixlm.BLTDByteLatentPrefixLM")
        self.assertEqual(contract["global_core_class"], "scripts/335_train_qtrm_native_etd_probe.py::NativeQTRMETDLM")
        self.assertFalse(contract["uses_wgram_model_core_world_model"])
        self.assertFalse(contract["uses_one_body_parallel_hybrid_block"])
        self.assertIn("not in the active BLT PrefixLM runtime", contract["lewm_world_model_status"])
        self.assertIn("not wgram_lm.blocks.OneBodyParallelHybridBlock", contract["decoder_one_body_meaning"])
        self.assertIn("causal learned chunk summaries", contract["boundary_state_source"])
        self.assertIn("non-boundary bytes", contract["boundary_state_source"])
        self.assertIn("hnet_causal_speaker", contract["active_answer_path"])
        self.assertIn("gated byte residual", contract["decoder_one_body_meaning"])
        self.assertEqual(contract["imta_trajectories"], 3)
        self.assertIn("same-body IMTA", contract["gram_ptrm_status"])
        self.assertIn("per-trajectory adapters", contract["imta_answer_path"])
        self.assertEqual(contract["imta_diversity_status"], "active trajectory diversity auxiliary")
        self.assertEqual(contract["own_latent_prediction_status"].split(";")[0], "active auxiliary over recurrent BLT causal chunk/core states")
        self.assertEqual(contract["answer_attractor_status"], "training auxiliary over multiple think depths")

    def test_contract_names_disabled_imta_when_k_is_one(self) -> None:
        from wgram_lm.architecture.blt_runtime_contract import build_active_blt_runtime_contract

        contract = build_active_blt_runtime_contract(_args(imta_trajectories=1))

        self.assertEqual(contract["imta_trajectories"], 1)
        self.assertIn("disabled for this run", contract["gram_ptrm_status"])

    def test_contract_rejects_lewm_claim_on_blt_path(self) -> None:
        from wgram_lm.architecture.blt_runtime_contract import validate_active_blt_runtime_contract

        with self.assertRaisesRegex(ValueError, "does not instantiate QTRM core_world_model"):
            validate_active_blt_runtime_contract(_args(core_world_model_enabled=True))

        with self.assertRaisesRegex(ValueError, "keeps LeWM/core-world-model loss at 0.0"):
            validate_active_blt_runtime_contract(_args(loss_core_world_model_weight=0.1))

    def test_contract_rejects_parallel_hybrid_claim_on_blt_path(self) -> None:
        from wgram_lm.architecture.blt_runtime_contract import validate_active_blt_runtime_contract

        with self.assertRaisesRegex(ValueError, "does not wire OneBodyParallelHybridBlock"):
            validate_active_blt_runtime_contract(_args(use_parallel_hybrid_block=True))

    def test_contract_rejects_invalid_imta_settings(self) -> None:
        from wgram_lm.architecture.blt_runtime_contract import validate_active_blt_runtime_contract

        with self.assertRaisesRegex(ValueError, "imta-trajectories"):
            validate_active_blt_runtime_contract(_args(imta_trajectories=0))

        with self.assertRaisesRegex(ValueError, "imta-noise-std"):
            validate_active_blt_runtime_contract(_args(imta_noise_std=-0.1))

        with self.assertRaisesRegex(ValueError, "imta-selector-temperature"):
            validate_active_blt_runtime_contract(_args(imta_selector_temperature=0.0))

        with self.assertRaisesRegex(ValueError, "imta-diversity-weight"):
            validate_active_blt_runtime_contract(_args(imta_diversity_weight=-0.1))

        with self.assertRaisesRegex(ValueError, "own-latent-prediction-weight"):
            validate_active_blt_runtime_contract(_args(own_latent_prediction_weight=-0.1))


if __name__ == "__main__":
    unittest.main()
