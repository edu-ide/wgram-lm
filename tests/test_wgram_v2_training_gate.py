from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "590_train_wgram_v2_prefixlm.py"


def load_module():
    spec = importlib.util.spec_from_file_location("wgram_v2_prefixlm_trainer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_toy_sampled_data(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "epoch_0").mkdir(parents=True, exist_ok=True)
    tokens = np.array(
        [
            2 + ord("Q"),
            2 + ord("?"),
            2 + ord("A"),
            1,
            2 + ord("1"),
            2 + ord("+"),
            2 + ord("1"),
            2 + ord("="),
            2 + ord("2"),
            1,
        ],
        dtype=np.int64,
    )
    np.save(path / "tokens.npy", tokens)
    metadata = {
        "tokenizer_info": {
            "kind": "tokenizer_free_utf8_byte_shifted",
            "byte_offset": 2,
            "eos_token_id": 1,
            "vocab_size": 256,
        },
        "vocab_size": 256,
        "max_seq_len": 8,
        "total_length": int(tokens.shape[0]),
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(path / "epoch_0" / "inst_start.npy", np.array([0, 4], dtype=np.int64))
    np.save(path / "epoch_0" / "inst_len.npy", np.array([2, 4], dtype=np.int64))
    np.save(path / "epoch_0" / "resp_start.npy", np.array([2, 8], dtype=np.int64))
    np.save(path / "epoch_0" / "resp_len.npy", np.array([2, 2], dtype=np.int64))


class WGRAMV2TrainingGateTests(unittest.TestCase):
    def test_v2_trainer_runs_one_step_and_generation_gate_is_free_only(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sampled = root / "sampled"
            out_dir = root / "out"
            write_toy_sampled_data(sampled)
            args = module.build_arg_parser().parse_args(
                [
                    "--sampled-data",
                    str(sampled),
                    "--out-dir",
                    str(out_dir),
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "8",
                    "--max-rows",
                    "2",
                    "--d-model",
                    "16",
                    "--local-heads",
                    "4",
                    "--core-layers",
                    "1",
                    "--local-layers",
                    "1",
                    "--imta-trajectories",
                    "2",
                    "--runtime-profile",
                    "smoke",
                    "--core-implementation",
                    "torch_smoke",
                    "--allow-torch-smoke-core",
                    "--repeat-unlikelihood-weight",
                    "0.05",
                    "--premature-stop-loss-weight",
                    "0.05",
                    "--response-start-loss-weight",
                    "0.5",
                    "--response-start-stop-margin-weight",
                    "0.2",
                    "--response-continue-stop-margin-weight",
                    "0.1",
                    "--response-body-loss-weight",
                    "0.25",
                    "--response-stop-loss-weight",
                    "0.5",
                    "--answer-prefix-commitment-loss-weight",
                    "0.2",
                    "--answer-memory-commitment-start-after",
                    "1",
                    "--answer-memory-commitment-warmup-steps",
                    "0",
                    "--self-rollout-loss-weight",
                    "0.1",
                    "--self-rollout-max-tokens",
                    "2",
                    "--self-rollout-start-after",
                    "1",
                    "--balanced-response-sampler",
                    "--force-fixed-boundaries",
                    "--device",
                    "cpu",
                    "--eval-max-rows",
                    "1",
                    "--max-new-tokens",
                    "3",
                    "--log-every",
                    "0",
                ]
            )

            report = module.train(args)
            gate = module.run_generation_gate_from_checkpoint(
                out_dir / "last_model.pt",
                sampled_data=sampled,
                epoch=0,
                seq_len=8,
                max_rows=1,
                max_new_tokens=3,
                device="cpu",
            )

            self.assertEqual(report["steps"], 1)
            self.assertEqual(report["micro_steps"], 1)
            self.assertEqual(report["optimizer"]["grad_accum_steps"], 1)
            self.assertEqual(report["optimizer"]["lr_schedule"], "constant")
            self.assertIn("tensorboard", report)
            self.assertIn("aim", report)
            self.assertTrue(Path(report["checkpoint"]).exists())
            self.assertEqual(report["contract"]["evaluation_policy"], "free_generation_only")
            self.assertEqual(report["contract"]["promotion_ready"], False)
            self.assertIn("training_stop_token_ids", report)
            self.assertTrue(report["balanced_response_sampler"]["enabled"])
            self.assertIn("premature_stop_loss", report["loss_history"][0])
            self.assertIn("optimizer_step", report["loss_history"][0])
            self.assertIn("learning_rate", report["loss_history"][0])
            self.assertIn("grad_norm", report["loss_history"][0])
            self.assertIn("grad_accum_steps", report["loss_history"][0])
            self.assertIn("response_start_loss", report["loss_history"][0])
            self.assertIn("response_start_stop_margin_loss", report["loss_history"][0])
            self.assertIn("response_continue_stop_margin_loss", report["loss_history"][0])
            self.assertIn("response_continue_stop_margin_effective_weight", report["loss_history"][0])
            self.assertIn("response_body_loss", report["loss_history"][0])
            self.assertIn("response_stop_loss", report["loss_history"][0])
            self.assertIn("answer_memory_aux_tokens", report["loss_history"][0])
            self.assertIn("answer_memory_prompt_context_mode", report["loss_history"][0])
            self.assertIn("answer_memory_prompt_context_gate_mean", report["loss_history"][0])
            self.assertIn("answer_memory_stop_margin_loss", report["loss_history"][0])
            self.assertIn("answer_memory_stop_margin_positions", report["loss_history"][0])
            self.assertIn("answer_memory_commitment_effective_scale", report["loss_history"][0])
            self.assertIn("answer_memory_injection_context", report["loss_history"][0])
            self.assertIn("answer_memory_injection_positions", report["loss_history"][0])
            self.assertIn("answer_memory_commitment_positions", report["loss_history"][0])
            self.assertIn("answer_memory_commitment_confidence_gate", report["loss_history"][0])
            self.assertIn("answer_memory_commitment_confidence_scale_mean", report["loss_history"][0])
            self.assertIn("answer_prefix_commitment_loss", report["loss_history"][0])
            self.assertIn("answer_prefix_commitment_tokens", report["loss_history"][0])
            self.assertIn("imta_route_entropy_loss", report["loss_history"][0])
            self.assertIn("imta_route_balance_loss", report["loss_history"][0])
            self.assertIn("response_stop_probability", report["loss_history"][0])
            self.assertIn("response_stop_loss_effective_weight", report["loss_history"][0])
            self.assertIn("response_stop_loss_warmup_steps", report["loss_history"][0])
            self.assertIn("self_rollout_loss", report["loss_history"][0])
            self.assertIn("self_rollout_replaced_tokens", report["loss_history"][0])
            self.assertEqual(gate["evaluation_policy"], "free_generation_only")
            self.assertIn("loop_like_fraction", gate["generation"])
            self.assertIn("teacher_forced_first_token_mean_rank", gate["generation"])
            self.assertIn("teacher_forced_first_token_top5_fraction", gate["generation"])
            self.assertIn("teacher_forced_first_token_mean_gold_minus_best_stop_logit", gate["generation"])
            self.assertIn("teacher_forced_first_token_gold_beats_stop_fraction", gate["generation"])
            self.assertIn("answer_memory_plan_available_fraction", gate["generation"])
            self.assertIn("answer_memory_plan_target_tokens", gate["generation"])
            self.assertIn("answer_memory_plan_token_accuracy_fraction", gate["generation"])
            self.assertIn("answer_memory_plan_top5_fraction", gate["generation"])
            self.assertIn("answer_memory_plan_mean_rank", gate["generation"])
            self.assertIn("answer_memory_plan_mean_gold_probability", gate["generation"])
            self.assertIn("answer_memory_plan_mean_confidence", gate["generation"])
            self.assertIn("answer_memory_plan_mean_top5_probability_mass", gate["generation"])
            self.assertIn("answer_memory_plan_mean_entropy_complement", gate["generation"])
            self.assertIn("first_token_eos_fraction", gate["generation"])
            self.assertIn("stop_fraction", gate["generation"])
            self.assertIn("first_token_stop_fraction", gate["generation"])
            self.assertIn("mean_byte_decodable_fraction", gate["generation"])
            self.assertIn("stop_token_ids", gate)
            self.assertIn("repetition", gate["generation"]["samples"][0])
            self.assertIn("token_diagnostics", gate["generation"]["samples"][0])
            self.assertIn("first_response_token", gate["generation"]["samples"][0])
            self.assertIn("answer_memory_plan", gate["generation"]["samples"][0])
            self.assertIn("rank", gate["generation"]["samples"][0]["first_response_token"])
            self.assertIn("gold_minus_best_stop_logit", gate["generation"]["samples"][0]["first_response_token"])
            self.assertIn("gold_beats_best_stop", gate["generation"]["samples"][0]["first_response_token"])
            self.assertIn("token_accuracy_fraction", gate["generation"]["samples"][0]["answer_memory_plan"])
            self.assertIn("tokens", gate["generation"]["samples"][0]["answer_memory_plan"])
            self.assertIn("mean_topk_probability_mass", gate["generation"]["samples"][0]["answer_memory_plan"])
            self.assertIn("mean_entropy_complement", gate["generation"]["samples"][0]["answer_memory_plan"])
            self.assertIn("first_token_is_stop", gate["generation"]["samples"][0]["token_diagnostics"])
            self.assertIn("top1_is_stop", gate["generation"]["samples"][0]["first_response_token"])
            self.assertNotIn("forced_choice", json.dumps(gate))
            self.assertLessEqual(len(gate["generation"]["samples"][0]["generated_ids"]), 3)


if __name__ == "__main__":
    unittest.main()
