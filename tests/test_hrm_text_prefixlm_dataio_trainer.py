import importlib.util
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_module():
    path = Path("scripts/534_train_native_prefixlm_dataio.py")
    spec = importlib.util.spec_from_file_location("hrm_text_prefixlm_dataio_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HRMTextPrefixLMDataIOTrainerTests(unittest.TestCase):
    def make_sampled_dataset(self, root: Path) -> Path:
        sampled = root / "sampled"
        epoch = sampled / "epoch_0"
        epoch.mkdir(parents=True)
        tokens = np.array(
            [
                11,
                12,
                13,
                21,
                22,
                31,
                32,
                41,
            ],
            dtype=np.int32,
        )
        np.save(sampled / "tokens.npy", tokens)
        np.save(epoch / "inst_start.npy", np.array([0, 5], dtype=np.int64))
        np.save(epoch / "inst_len.npy", np.array([3, 2], dtype=np.int64))
        np.save(epoch / "resp_start.npy", np.array([3, 7], dtype=np.int64))
        np.save(epoch / "resp_len.npy", np.array([2, 1], dtype=np.int64))
        (sampled / "metadata.json").write_text(
            json.dumps(
                {
                    "tokenizer_info": {"vocab_size": 64},
                    "vocab_size": None,
                    "max_seq_len": 8,
                    "total_length": int(tokens.size),
                }
            ),
            encoding="utf-8",
        )
        return sampled

    def test_prefixlm_dataset_masks_instruction_and_keeps_response_targets(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = self.make_sampled_dataset(Path(tmp))
            dataset = module.DataIOSampledPrefixLMDataset(sampled, seq_len=6, epoch=0)

            row = dataset[0]

        self.assertEqual(row["input_ids"].tolist(), [11, 12, 13, 21, 0, 0])
        self.assertEqual(row["labels"].tolist(), [-100, -100, 21, 22, -100, -100])
        self.assertEqual(row["response_start_mask"].tolist(), [0, 0, 1, 0, 0, 0])
        self.assertEqual(row["attention_mask"].tolist(), [1, 1, 1, 1, 0, 0])

    def test_dataset_summary_exposes_efficiency_accounting_fields(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = self.make_sampled_dataset(Path(tmp))
            dataset = module.DataIOSampledPrefixLMDataset(sampled, seq_len=6, epoch=0)

            summary = dataset.summary()

        self.assertEqual(summary["contract"], "hrm_text_data_io_prefixlm")
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["vocab_size"], 64)
        self.assertEqual(summary["effective_max_seq_len"], 7)
        self.assertEqual(summary["seq_len"], 6)

    def test_collate_batches_prefixlm_rows(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = self.make_sampled_dataset(Path(tmp))
            dataset = module.DataIOSampledPrefixLMDataset(sampled, seq_len=6, epoch=0)
            batch = module.collate_prefixlm_rows([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 6))
        self.assertEqual(tuple(batch["labels"].shape), (2, 6))
        self.assertEqual(tuple(batch["response_start_mask"].shape), (2, 6))
        self.assertEqual(batch["labels"].dtype, torch.long)

    def test_trim_batch_to_max_valid_length_removes_trailing_padding_columns(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            sampled = self.make_sampled_dataset(Path(tmp))
            dataset = module.DataIOSampledPrefixLMDataset(sampled, seq_len=6, epoch=0)
            batch = module.collate_prefixlm_rows([dataset[0], dataset[1]])

        trimmed = module.trim_prefixlm_batch_to_max_valid_length(batch)

        self.assertEqual(tuple(trimmed["input_ids"].shape), (2, 4))
        self.assertEqual(trimmed["input_ids"].tolist(), [[11, 12, 13, 21], [31, 32, 0, 0]])
        self.assertEqual(trimmed["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])

    def test_length_bucketed_sampler_batches_similar_lengths(self):
        module = load_module()
        sampler = module.LengthBucketedBatchSampler(
            [2, 11, 3, 10, 4, 12],
            batch_size=3,
            generator=torch.Generator().manual_seed(0),
            bucket_size_multiplier=8,
        )

        batches = list(iter(sampler))

        self.assertEqual(len(batches), 2)
        for batch in batches:
            lengths = [sampler.lengths[index] for index in batch]
            self.assertLessEqual(max(lengths) - min(lengths), 2)

    def test_throughput_metrics_report_cumulative_and_interval_rates(self):
        module = load_module()

        metrics = module.throughput_metrics(
            step=20,
            start_step=10,
            current_time=130.0,
            train_start_time=100.0,
            previous_log_step=15,
            previous_log_time=120.0,
            tokens_seen=1_500,
            target_tokens_seen=300,
            compute_tokens_seen=2_000,
            previous_log_tokens=500,
            previous_log_target_tokens=100,
            previous_log_compute_tokens=1_000,
        )

        self.assertEqual(metrics["elapsed_sec"], 30.0)
        self.assertEqual(metrics["interval_sec"], 10.0)
        self.assertEqual(metrics["steps_per_sec"], 10 / 30)
        self.assertEqual(metrics["interval_steps_per_sec"], 5 / 10)
        self.assertEqual(metrics["tokens_per_sec"], 50.0)
        self.assertEqual(metrics["target_tokens_per_sec"], 10.0)
        self.assertEqual(metrics["compute_tokens_per_sec"], 2000 / 30)
        self.assertEqual(metrics["interval_tokens_per_sec"], 100.0)
        self.assertEqual(metrics["interval_target_tokens_per_sec"], 20.0)
        self.assertEqual(metrics["interval_compute_tokens_per_sec"], 100.0)

    def test_default_architecture_avoids_mamba3_paths(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(["--sampled-data", "/tmp/x", "--out-dir", "/tmp/y"])

        self.assertEqual(args.backbone, "trm_qwen35_3to1")
        self.assertEqual(args.delta_backend, "official_gated_delta2")
        self.assertEqual(args.think_structure, "trm_dual_z")
        self.assertFalse(args.activation_checkpointing)
        self.assertFalse(args.length_bucketed_batches)
        self.assertEqual(args.length_bucket_size_multiplier, 64)
        self.assertNotIn("mamba", args.backbone.lower())
        self.assertNotIn("mamba", args.think_structure.lower())

    def test_activation_checkpointing_small_model_backward(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--sampled-data",
                "/tmp/x",
                "--out-dir",
                "/tmp/y",
                "--model-vocab-size",
                "64",
                "--seq-len",
                "8",
                "--d-model",
                "16",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "2",
                "--d-ff",
                "32",
                "--train-think-steps",
                "1",
                "--activation-checkpointing",
            ]
        )
        model = module.build_model(args, vocab_size=64)
        model.train()
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        labels[:, 2:4] = torch.tensor([[3, 4]], dtype=torch.long)

        loss = module.prefixlm_loss_for_batch(
            model,
            input_ids,
            labels,
            think_steps=1,
            loss_chunk_size=2,
        )
        loss.backward()

        self.assertTrue(model.activation_checkpointing)
        self.assertIsNotNone(model.token_embed.weight.grad)

    def test_mamba_backed_think_structure_is_rejected(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--sampled-data",
                "/tmp/x",
                "--out-dir",
                "/tmp/y",
                "--think-structure",
                "trm_dual_z_reversed_hybrid_3to1",
            ]
        )

        with self.assertRaisesRegex(ValueError, "Mamba3"):
            module.assert_mamba3_free_args(args)

    def test_eval_prefixlm_loss_ignores_masked_instruction_labels(self):
        module = load_module()

        class ConstantLogitModel(nn.Module):
            def forward(self, input_ids, *, think_steps):
                del think_steps
                return torch.zeros((*input_ids.shape, 4), dtype=torch.float32)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long),
            "labels": torch.tensor([[1, -100, 2], [-100, 3, -100]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
        }

        metrics = module.evaluate_prefixlm_loss(
            ConstantLogitModel(),
            [batch],
            device=torch.device("cpu"),
            think_steps=1,
            max_batches=1,
        )

        self.assertAlmostEqual(metrics["loss"], float(torch.log(torch.tensor(4.0))), places=6)
        self.assertEqual(metrics["target_tokens"], 3)
        self.assertEqual(metrics["tokens"], 5)
        self.assertEqual(metrics["nonfinite_batches"], 0)
        self.assertEqual(metrics["fallback_batches"], 0)

    def test_eval_prefixlm_loss_falls_back_when_primary_loss_is_nonfinite(self):
        module = load_module()

        class NonFiniteForwardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embed = nn.Embedding(8, 5)
                self.lm_head = nn.Linear(5, 7, bias=False)
                self.value_codec = "learned"

            def forward_hidden(self, input_ids, *, think_steps):
                del think_steps
                return self.token_embed(input_ids)

            def forward(self, input_ids, *, think_steps):
                del think_steps
                return torch.full((*input_ids.shape, 7), float("nan"))

        torch.manual_seed(23)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long),
            "labels": torch.tensor([[1, -100, 2], [-100, 3, -100]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
        }

        metrics = module.evaluate_prefixlm_loss(
            NonFiniteForwardModel(),
            [batch],
            device=torch.device("cpu"),
            think_steps=1,
            max_batches=1,
            loss_chunk_size=0,
            loss_kernel="torch",
        )

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(metrics["nonfinite_batches"], 1)
        self.assertEqual(metrics["fallback_batches"], 1)
        self.assertEqual(metrics["unresolved_nonfinite_batches"], 0)
        self.assertEqual(metrics["attempted_target_tokens"], 3)
        self.assertEqual(metrics["nonfinite_batch_indices"], [0])
        self.assertEqual(metrics["unresolved_nonfinite_batch_indices"], [])
        self.assertEqual(metrics["nonfinite_target_tokens"], 3)
        self.assertEqual(metrics["unresolved_target_tokens"], 0)
        self.assertEqual(metrics["fallback_hidden_nonfinite_elements"], 0)
        self.assertEqual(metrics["fallback_hidden_nonfinite_batches"], 0)

    def test_eval_prefixlm_loss_records_unresolved_nonfinite_fallback_batch(self):
        module = load_module()

        class PersistentNonFiniteModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embed = nn.Embedding(8, 5)
                self.lm_head = nn.Linear(5, 7, bias=False)
                self.value_codec = "learned"

            def forward_hidden(self, input_ids, *, think_steps):
                del think_steps
                return torch.full((*input_ids.shape, 5), float("nan"))

            def forward(self, input_ids, *, think_steps):
                del think_steps
                return torch.full((*input_ids.shape, 7), float("nan"))

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long),
            "labels": torch.tensor([[1, -100, 2], [-100, 3, -100]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
        }

        metrics = module.evaluate_prefixlm_loss(
            PersistentNonFiniteModel(),
            [batch],
            device=torch.device("cpu"),
            think_steps=1,
            max_batches=1,
            loss_chunk_size=0,
            loss_kernel="torch",
        )

        self.assertTrue(torch.isinf(torch.tensor(metrics["loss"])))
        self.assertEqual(metrics["nonfinite_batches"], 1)
        self.assertEqual(metrics["fallback_batches"], 1)
        self.assertEqual(metrics["unresolved_nonfinite_batches"], 1)
        self.assertEqual(metrics["target_tokens"], 0)
        self.assertEqual(metrics["attempted_target_tokens"], 3)
        self.assertEqual(metrics["nonfinite_batch_indices"], [0])
        self.assertEqual(metrics["unresolved_nonfinite_batch_indices"], [0])
        self.assertEqual(metrics["nonfinite_target_tokens"], 3)
        self.assertEqual(metrics["unresolved_target_tokens"], 3)
        self.assertEqual(metrics["fallback_hidden_target_elements"], 15)
        self.assertEqual(metrics["fallback_hidden_nonfinite_elements"], 15)
        self.assertEqual(metrics["fallback_hidden_nonfinite_batches"], 1)
        self.assertEqual(metrics["unresolved_hidden_target_elements"], 15)
        self.assertEqual(metrics["unresolved_hidden_nonfinite_elements"], 15)
        self.assertEqual(metrics["unresolved_with_finite_hidden_batches"], 0)

    def test_chunked_prefixlm_loss_matches_full_logits_loss(self):
        module = load_module()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embed = nn.Embedding(8, 5)
                self.lm_head = nn.Linear(5, 7, bias=False)
                self.value_codec = "learned"

            def forward_hidden(self, input_ids, *, think_steps):
                del think_steps
                return self.token_embed(input_ids)

            def forward(self, input_ids, *, think_steps):
                return self.lm_head(self.forward_hidden(input_ids, think_steps=think_steps))

        torch.manual_seed(7)
        model = TinyLM()
        input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]], dtype=torch.long)
        labels = torch.tensor([[1, -100, 4, 5], [-100, 3, -100, -100]], dtype=torch.long)

        full_loss = module.prefixlm_loss_for_batch(
            model,
            input_ids,
            labels,
            think_steps=2,
            loss_chunk_size=0,
        )
        chunked_loss = module.prefixlm_loss_for_batch(
            model,
            input_ids,
            labels,
            think_steps=2,
            loss_chunk_size=2,
        )

        self.assertTrue(torch.allclose(full_loss, chunked_loss, atol=1e-6))

    def test_corrupted_token_ids_never_equal_targets(self):
        module = load_module()

        targets = torch.tensor([0, 1, 2, 6, 7], dtype=torch.long)
        corrupted = module.corrupted_token_ids(targets, vocab_size=8)

        self.assertEqual(tuple(corrupted.shape), tuple(targets.shape))
        self.assertTrue(torch.all(corrupted != targets))
        self.assertTrue(torch.all(corrupted >= 0))
        self.assertTrue(torch.all(corrupted < 8))

    def test_token_verifier_loss_uses_only_prefixlm_target_positions(self):
        module = load_module()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embed = nn.Embedding(11, 6)
                self.lm_head = nn.Linear(6, 11, bias=False)
                self.value_codec = "learned"

            def forward_hidden(self, input_ids, *, think_steps):
                del think_steps
                return self.token_embed(input_ids)

        torch.manual_seed(11)
        model = TinyLM()
        verifier = module.PrefixLMTokenVerifier(d_model=6, hidden_dim=4)
        input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]], dtype=torch.long)
        labels = torch.tensor([[1, -100, 4, 5], [-100, 3, -100, -100]], dtype=torch.long)
        hidden = model.forward_hidden(input_ids, think_steps=2)

        loss, metrics = module.prefixlm_token_verifier_loss_from_hidden(
            model,
            verifier,
            hidden,
            labels,
            vocab_size=11,
            max_targets=0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["verifier_targets"], 4)
        self.assertGreaterEqual(metrics["verifier_accuracy"], 0.0)
        self.assertLessEqual(metrics["verifier_accuracy"], 1.0)

    def test_nitp_loss_uses_only_prefixlm_target_positions(self):
        module = load_module()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embed = nn.Embedding(11, 6)
                self.lm_head = nn.Linear(6, 11, bias=False)
                self.value_codec = "learned"

            def forward_hidden(self, input_ids, *, think_steps):
                del think_steps
                return self.token_embed(input_ids)

        torch.manual_seed(13)
        model = TinyLM()
        projector = module.NextImplicitTokenProjector(d_model=6, hidden_dim=4)
        input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]], dtype=torch.long)
        labels = torch.tensor([[1, -100, 4, 5], [-100, 3, -100, -100]], dtype=torch.long)
        hidden = model.forward_hidden(input_ids, think_steps=2)

        loss, metrics = module.prefixlm_nitp_loss_from_hidden(
            model,
            projector,
            hidden,
            labels,
            max_targets=0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["nitp_targets"], 4)
        self.assertGreaterEqual(metrics["nitp_cosine_similarity"], -1.0)
        self.assertLessEqual(metrics["nitp_cosine_similarity"], 1.0)

    def test_token_verifier_cli_defaults_to_disabled(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(["--sampled-data", "/tmp/x", "--out-dir", "/tmp/y"])

        self.assertEqual(args.token_verifier_loss_weight, 0.0)
        self.assertEqual(args.token_verifier_max_targets, 256)
        self.assertFalse(args.token_verifier_freeze_model)
        self.assertEqual(args.nitp_loss_weight, 0.0)
        self.assertEqual(args.nitp_hidden_dim, 0)
        self.assertEqual(args.nitp_max_targets, 256)
        self.assertEqual(args.premature_stop_loss_weight, 0.0)
        self.assertEqual(args.premature_stop_token_ids, "")
        self.assertEqual(args.response_start_loss_weight, 0.0)
        self.assertEqual(args.row_balanced_response_loss_weight, 0.0)

    def test_memory_efficient_optimizer_cli_defaults_to_adamw(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(["--sampled-data", "/tmp/x", "--out-dir", "/tmp/y"])

        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.loss_kernel, "torch")
        self.assertEqual(args.model_checkpoint_every, 0)
        self.assertEqual(args.galore_rank, 128)
        self.assertEqual(args.galore_update_proj_gap, 200)
        self.assertEqual(args.galore_scale, 0.25)
        self.assertEqual(args.galore_proj_type, "std")
        self.assertEqual(args.galore_min_dim, 128)
        self.assertFalse(args.galore_include_embeddings)

    def test_memory_efficient_optimizer_cli_accepts_8bit_and_galore(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--sampled-data",
                "/tmp/x",
                "--out-dir",
                "/tmp/y",
                "--optimizer",
                "paged_ademamix8bit",
                "--loss-kernel",
                "liger_fused_linear_ce",
                "--galore-rank",
                "64",
                "--galore-update-proj-gap",
                "100",
                "--galore-scale",
                "0.5",
                "--galore-proj-type",
                "reverse_std",
                "--galore-min-dim",
                "256",
                "--galore-include-embeddings",
            ]
        )

        self.assertEqual(args.optimizer, "paged_ademamix8bit")
        self.assertEqual(args.loss_kernel, "liger_fused_linear_ce")
        self.assertEqual(args.galore_rank, 64)
        self.assertEqual(args.galore_update_proj_gap, 100)
        self.assertEqual(args.galore_scale, 0.5)
        self.assertEqual(args.galore_proj_type, "reverse_std")
        self.assertEqual(args.galore_min_dim, 256)
        self.assertTrue(args.galore_include_embeddings)

    def test_model_only_checkpoint_omits_optimizer_state(self):
        module = load_module()

        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = argparse.Namespace(sampled_data="/tmp/sample", out_dir="/tmp/out")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_model.pt"
            copy_path = Path(tmp) / "copy_last_model.pt"
            module.save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                verifier=None,
                step=7,
                tokens_seen=11,
                target_tokens_seen=5,
                compute_tokens_seen=13,
                losses=[],
                eval_losses=[],
                args=args,
                dataset_summary={"rows": 1},
                eval_dataset_summary=None,
                model_summary={"total_parameters": 8},
                include_optimizer=False,
                copy_safe_path=copy_path,
            )

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            copy_checkpoint = torch.load(copy_path, map_location="cpu", weights_only=False)
            self.assertEqual(list(path.parent.glob(".last_model.pt.tmp.*")), [])
            self.assertEqual(list(path.parent.glob(".copy_last_model.pt.tmp.*")), [])

        self.assertFalse(checkpoint["checkpoint_includes_optimizer"])
        self.assertFalse(copy_checkpoint["checkpoint_includes_optimizer"])
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("model_state_dict", copy_checkpoint)
        self.assertNotIn("optimizer_state_dict", checkpoint)
        self.assertNotIn("optimizer_state_dict", copy_checkpoint)
        self.assertEqual(checkpoint["step"], 7)
        self.assertEqual(copy_checkpoint["step"], 7)

    def test_premature_stop_loss_ignores_gold_stop_positions(self):
        module = load_module()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8, bias=False)

        torch.manual_seed(13)
        model = TinyLM()
        hidden = torch.randn(1, 4, 4)
        labels = torch.tensor([[3, 7, -100, 5]], dtype=torch.long)

        loss, metrics = module.premature_stop_unlikelihood_loss_from_hidden(
            model,
            hidden,
            labels,
            stop_token_ids=(7,),
            loss_chunk_size=2,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["premature_stop_positions"], 2)
        self.assertGreaterEqual(metrics["premature_stop_mean_probability"], 0.0)
        self.assertLessEqual(metrics["premature_stop_mean_probability"], 1.0)
        self.assertEqual(module.parse_token_id_list("7, 11 13"), (7, 11, 13))

    def test_response_start_loss_uses_only_first_response_positions(self):
        module = load_module()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8, bias=False)

        torch.manual_seed(17)
        model = TinyLM()
        hidden = torch.randn(2, 4, 4)
        labels = torch.tensor(
            [[-100, -100, 3, 7], [-100, 5, 6, -100]],
            dtype=torch.long,
        )
        response_start_mask = torch.tensor(
            [[0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=torch.long,
        )

        loss, metrics = module.response_start_loss_from_hidden(
            model,
            hidden,
            labels,
            response_start_mask,
            stop_token_ids=(7,),
            loss_chunk_size=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["response_start_positions"], 2)
        self.assertGreaterEqual(metrics["response_start_accuracy"], 0.0)
        self.assertLessEqual(metrics["response_start_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["response_start_gold_probability"], 0.0)
        self.assertLessEqual(metrics["response_start_gold_probability"], 1.0)
        self.assertIn("response_start_stop_probability", metrics)

    def test_row_balanced_response_loss_averages_each_row_once(self):
        module = load_module()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8, bias=False)

        torch.manual_seed(19)
        model = TinyLM()
        hidden = torch.randn(2, 4, 4)
        labels = torch.tensor(
            [[-100, 3, 4, 7], [-100, 5, -100, -100]],
            dtype=torch.long,
        )

        loss, metrics = module.row_balanced_response_loss_from_hidden(
            model,
            hidden,
            labels,
            loss_chunk_size=2,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["row_balanced_response_rows"], 2)
        self.assertEqual(metrics["row_balanced_response_targets"], 4)
        self.assertGreaterEqual(metrics["row_balanced_response_token_accuracy"], 0.0)
        self.assertLessEqual(metrics["row_balanced_response_token_accuracy"], 1.0)

    def test_amp_cli_defaults_to_fp32_safe_path(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(["--sampled-data", "/tmp/x", "--out-dir", "/tmp/y"])

        self.assertEqual(args.amp_dtype, "none")
        self.assertEqual(args.matmul_precision, "high")
        self.assertIsNone(module.resolve_amp_dtype(args.amp_dtype))
        self.assertIs(module.resolve_amp_dtype("bf16"), torch.bfloat16)

    def test_learning_rate_warmup_matches_hrm_text_style(self):
        module = load_module()

        parser = module.build_arg_parser()
        args = parser.parse_args(
            [
                "--sampled-data",
                "/tmp/x",
                "--out-dir",
                "/tmp/y",
                "--lr",
                "2.2e-4",
                "--lr-warmup-steps",
                "2000",
            ]
        )

        self.assertAlmostEqual(module.scheduled_learning_rate(args, 1), 2.2e-4 / 2000)
        self.assertAlmostEqual(module.scheduled_learning_rate(args, 1000), 1.1e-4)
        self.assertAlmostEqual(module.scheduled_learning_rate(args, 2000), 2.2e-4)
        self.assertAlmostEqual(module.scheduled_learning_rate(args, 3000), 2.2e-4)


if __name__ == "__main__":
    unittest.main()
