from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "557_train_blt_d_prefixlm_dataio.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_d_prefixlm_trainer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyGlobalCore(nn.Module):
    position_embedding_mode = "none"

    def _forward_embedded_impl(
        self,
        x: torch.Tensor,
        *,
        think_steps: int,
        return_hidden: bool,
    ) -> torch.Tensor:
        return x


class CountingGlobalCore(nn.Module):
    position_embedding_mode = "none"

    def _forward_embedded_impl(
        self,
        x: torch.Tensor,
        *,
        think_steps: int,
        return_hidden: bool,
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[..., 0] = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        return out


class FakeOffsetTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        if text == "ab cd":
            return {"offset_mapping": [(0, 2), (2, 3), (3, 5)]}
        return {"offset_mapping": [(0, len(text))]}


class BLTHBFBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def build_model(self):
        return self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hbf_byteflow",
            dynamic_min_patch_size=2,
            dynamic_soft_patch_size=2,
        )

    def test_hbf_byteflow_preserves_utf8_codepoint_inside_one_patch(self) -> None:
        model = self.build_model()
        # UTF-8 bytes for "한 " shifted by +2: ED 95 9C 20.
        input_ids = torch.tensor([[0xED + 2, 0x95 + 2, 0x9C + 2, 0x20 + 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        grouped_ids, grouped_mask, _, latent_len, valid_bytes, valid_patches = model.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )

        self.assertEqual(valid_bytes, 4)
        self.assertGreaterEqual(valid_patches, 1)
        self.assertLessEqual(latent_len, 2)
        first_patch_len = int(grouped_mask[0, 0].sum().item())
        self.assertGreaterEqual(first_patch_len, 3)
        self.assertEqual(grouped_ids[0, 0, :3].tolist(), input_ids[0, :3].tolist())

    def test_hbf_byteflow_keeps_patches_within_max_patch_size(self) -> None:
        model = self.build_model()
        # "abcd ef" shifted by +2.
        raw = [ord(ch) + 2 for ch in "abcd ef"]
        input_ids = torch.tensor([raw], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        _, grouped_mask, _, _, valid_bytes, valid_patches = model.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )

        self.assertEqual(valid_bytes, len(raw))
        self.assertGreater(valid_patches, 1)
        self.assertTrue(bool((grouped_mask.sum(dim=-1) <= 4).all().item()))

    def test_learned_primary_mode_has_trainable_chunker_and_soft_metrics(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="learned_primary",
            dynamic_min_patch_size=2,
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcd ef"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        grouped_ids, grouped_mask, _, _, valid_bytes, valid_patches = model.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )
        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
        )
        loss.backward()

        self.assertEqual(valid_bytes, int(attention_mask.sum().item()))
        self.assertGreater(valid_patches, 0)
        self.assertTrue(hasattr(model, "semantic_boundary_scorer"))
        self.assertGreater(sum(p.numel() for p in model.semantic_boundary_scorer.parameters()), 0)
        self.assertIn("learned_chunk_gate_mean", metrics)
        self.assertIn("learned_chunk_gate_entropy", metrics)
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.semantic_boundary_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)
        self.assertEqual(grouped_ids.shape[-1], model.patch_size)

    def test_learned_boundary_mode_changes_global_latent_sequence(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="learned_boundary",
            dynamic_min_patch_size=2,
            hbf_boundary_threshold=0.5,
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcdefgh"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        with torch.no_grad():
            model.semantic_boundary_scorer[-1].bias.fill_(2.0)

        grouped_ids, grouped_mask, _, latent_len, valid_bytes, valid_patches = model.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )
        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
        )
        loss.backward()

        self.assertEqual(valid_bytes, int(attention_mask.sum().item()))
        self.assertGreater(valid_patches, 2)
        self.assertGreater(latent_len, 2)
        self.assertEqual(grouped_ids.shape[1], latent_len)
        self.assertTrue(bool((grouped_mask.sum(dim=-1) <= model.patch_size).all().item()))
        self.assertIn("learned_boundary_prob_mean", metrics)
        self.assertIn("learned_boundary_valid_boundaries", metrics)
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.semantic_boundary_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)

    def test_hnet_dechunk_mode_maps_shortened_state_back_to_bytes(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hnet_dechunk",
            dynamic_min_patch_size=2,
            hbf_boundary_threshold=0.5,
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcdefgh"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        with torch.no_grad():
            model.semantic_boundary_scorer[-1].bias.fill_(2.0)

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
        )
        logits = model.forward_logits(input_ids, attention_mask, think_steps=1)
        loss.backward()

        self.assertEqual(logits.shape[:2], input_ids.shape)
        self.assertIn("hnet_selected_len", metrics)
        self.assertIn("hnet_dechunked_tokens", metrics)
        self.assertGreater(int(metrics["hnet_selected_len"]), 2)
        self.assertEqual(int(metrics["hnet_dechunked_tokens"]), int(attention_mask.sum().item()))
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.semantic_boundary_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)

    def test_hnet_dechunk_boundary_prior_adds_causal_boundary_loss(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hnet_dechunk",
            dynamic_min_patch_size=2,
            hbf_boundary_threshold=0.5,
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcdefgh"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
            boundary_prior_weight=0.2,
            boundary_target_ratio=0.5,
        )
        loss.backward()

        self.assertIn("boundary_prior_loss", metrics)
        self.assertGreater(float(metrics["boundary_prior_loss"]), 0.0)
        self.assertGreater(float(metrics["loss"]), float(metrics["clean_loss"]))
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.semantic_boundary_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)

    def test_hnetpp_flow_dechunk_diffusion_auxiliary_reconstructs_masked_bytes(self) -> None:
        torch.manual_seed(29)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hnetpp_flow_dechunk",
            dynamic_min_patch_size=2,
            hbf_boundary_threshold=0.5,
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcd efgh"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.25,
            diffusion_mask_prob=1.0,
        )
        loss.backward()

        self.assertGreater(int(metrics["diffusion_targets"]), 0)
        self.assertGreater(float(metrics["diffusion_loss"]), 0.0)
        self.assertGreater(float(metrics["loss"]), float(metrics["clean_loss"]))
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.semantic_boundary_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)

    def test_hnet_dechunk_qwen_boundary_prior_uses_external_boundary_labels(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hnet_dechunk",
            dynamic_min_patch_size=2,
            hbf_boundary_threshold=0.5,
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcd ef"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        qwen_targets = torch.tensor([[1, 0, 0, 0, 1, 0, 0]], dtype=torch.float32)

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
            qwen_boundary_targets=qwen_targets,
            qwen_boundary_prior_weight=0.05,
        )
        loss.backward()

        self.assertIn("qwen_boundary_prior_loss", metrics)
        self.assertGreater(float(metrics["qwen_boundary_prior_loss"]), 0.0)
        self.assertEqual(int(metrics["qwen_boundary_targets"]), int(attention_mask.sum().item()))
        self.assertGreater(float(metrics["loss"]), float(metrics["clean_loss"]))
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.semantic_boundary_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)

    def test_qwen_boundary_teacher_maps_tokenizer_offsets_to_byte_positions(self) -> None:
        teacher = self.module.QwenTokenizerBoundaryTeacher.__new__(
            self.module.QwenTokenizerBoundaryTeacher
        )
        teacher.model_id = "fake-qwen-tokenizer"
        teacher.tokenizer = FakeOffsetTokenizer()
        input_ids = torch.tensor([ord(ch) + 2 for ch in "ab cd"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        targets = teacher.row_targets(input_ids, attention_mask)

        self.assertEqual(targets.tolist(), [1.0, 0.0, 1.0, 1.0, 0.0])

    def test_hnet_dechunk_uses_boundary_probability_ema(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=CountingGlobalCore(),
            vocab_size=512,
            d_model=4,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hnet_dechunk",
            dynamic_min_patch_size=1,
            hbf_boundary_threshold=0.5,
        )
        with torch.no_grad():
            model.byte_embed.weight.zero_()
            for parameter in model.semantic_boundary_scorer.parameters():
                parameter.zero_()

        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcd"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        token_hidden, _, _, _, _, _ = model._hnet_boundary_states(
            input_ids,
            attention_mask,
            think_steps=1,
        )

        expected = torch.tensor([0.25, 0.625, 1.0625, 1.53125], dtype=token_hidden.dtype)
        self.assertTrue(
            torch.allclose(token_hidden[0, :, 0].detach().cpu(), expected, atol=1e-5),
            token_hidden[0, :, 0].detach().cpu().tolist(),
        )

    def test_hnetpp_flow_dechunk_uses_byteflow_change_to_keep_information_boundary(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=4,
            patch_size=8,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="hnetpp_flow_dechunk",
            dynamic_min_patch_size=2,
            hbf_boundary_threshold=0.9,
        )
        with torch.no_grad():
            model.byte_embed.weight.zero_()
            model.byte_embed.weight[ord("a") + 2, 0] = 1.0
            model.byte_embed.weight[ord("B") + 2, 0] = -1.0
            for parameter in model.semantic_boundary_scorer.parameters():
                parameter.zero_()

        input_ids = torch.tensor([[ord(ch) + 2 for ch in "aaaaBBBB"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        _, _, _, _, _, metrics = model._hnet_boundary_states(
            input_ids,
            attention_mask,
            think_steps=1,
        )

        self.assertGreaterEqual(int(metrics["hnet_selected_len"]), 2)
        self.assertLess(float(metrics["compression_ratio"]), float(input_ids.numel()))
        self.assertIn("hnetpp_flow_boundary_score_mean", metrics)
        self.assertGreater(float(metrics["hnetpp_flow_boundary_score_mean"]), 0.0)

    def test_hierarchical_add_mode_learns_upper_chunk_memory_on_answer_path(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=CountingGlobalCore(),
            vocab_size=512,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="hier_add",
            patch_boundary_mode="fixed",
        )
        input_ids = torch.tensor([[ord(ch) + 2 for ch in "abcdef"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=1,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
        )
        logits = model.forward_logits(input_ids, attention_mask, think_steps=1)
        loss.backward()

        self.assertEqual(logits.shape[:2], input_ids.shape)
        self.assertIn("hier_chunk_gate_mean", metrics)
        self.assertGreater(float(metrics["hier_chunk_gate_mean"]), 0.0)
        grad = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in list(model.hierarchical_chunk_gate.parameters())
            + list(model.hierarchical_chunk_proj.parameters())
            if parameter.grad is not None
        )
        self.assertGreater(grad, 0.0)

    def test_blt_ngram_entropy_uses_surprisal_boundaries_with_fixed_budget(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=DummyGlobalCore(),
            vocab_size=512,
            d_model=8,
            patch_size=4,
            local_layers=1,
            local_heads=2,
            patch_boundary_mode="blt_ngram_entropy",
            dynamic_min_patch_size=2,
        )
        unigram = torch.zeros(512)
        bigram = torch.zeros(512, 512)
        # Make starts before C and e the most informative boundaries.
        bigram[ord("b") + 2, ord("C") + 2] = 9.0
        bigram[ord("d") + 2, ord("e") + 2] = 8.0
        model.set_ngram_entropy_tables(unigram, bigram)

        raw = [ord(ch) + 2 for ch in "abCdef"]
        input_ids = torch.tensor([raw], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        grouped_ids, grouped_mask, _, latent_len, valid_bytes, valid_patches = model.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )

        self.assertEqual(valid_bytes, 6)
        self.assertEqual(valid_patches, 3)
        self.assertEqual(latent_len, 3)
        self.assertEqual(grouped_mask[0].sum(dim=-1).tolist(), [2, 2, 2])
        self.assertEqual(grouped_ids[0, 0, :2].tolist(), [ord("a") + 2, ord("b") + 2])
        self.assertEqual(grouped_ids[0, 1, :2].tolist(), [ord("C") + 2, ord("d") + 2])
        self.assertIn("ngram_entropy_selected_boundaries", model.last_boundary_metrics)

    def test_teacher_distillation_loss_matches_raw_byte_distribution(self) -> None:
        teacher_logits = torch.tensor(
            [
                [
                    [4.0, 0.0, -1.0],
                    [0.0, 5.0, -1.0],
                    [0.0, -1.0, 5.0],
                ]
            ]
        )
        labels = torch.tensor([[0, self.module.IGNORE_LABEL_ID, 2]])

        same_loss, same_metrics = self.module.teacher_distillation_loss(
            teacher_logits.clone(),
            teacher_logits,
            labels,
            temperature=1.0,
            max_targets=0,
        )
        bad_student = torch.flip(teacher_logits, dims=[-1])
        bad_loss, bad_metrics = self.module.teacher_distillation_loss(
            bad_student,
            teacher_logits,
            labels,
            temperature=1.0,
            max_targets=0,
        )

        self.assertLess(float(same_loss.item()), 1e-6)
        self.assertGreater(float(bad_loss.item()), 1.0)
        self.assertEqual(int(same_metrics["teacher_distill_targets"]), 2)
        self.assertEqual(int(bad_metrics["teacher_distill_targets"]), 2)

    def test_teacher_checkpoint_args_override_student_scale(self) -> None:
        teacher_args = argparse.Namespace(
            seq_len=1024,
            d_model=1792,
            n_heads=16,
            n_kv_heads=4,
            d_ff=4864,
            backbone="large_student",
            train_think_steps=2,
            delta_backend="official_gated_delta2",
        )
        checkpoint_args = {
            "seq_len": 384,
            "d_model": 384,
            "n_heads": 6,
            "n_kv_heads": 2,
            "d_ff": 1024,
            "backbone": "trm_qwen35_3to1",
            "train_think_steps": 3,
        }

        self.module.apply_teacher_checkpoint_args(teacher_args, checkpoint_args)

        self.assertEqual(teacher_args.seq_len, 384)
        self.assertEqual(teacher_args.d_model, 384)
        self.assertEqual(teacher_args.n_heads, 6)
        self.assertEqual(teacher_args.n_kv_heads, 2)
        self.assertEqual(teacher_args.d_ff, 1024)
        self.assertEqual(teacher_args.backbone, "trm_qwen35_3to1")
        self.assertEqual(teacher_args.train_think_steps, 3)
        self.assertEqual(teacher_args.delta_backend, "official_gated_delta2")

    def test_model_only_checkpoint_omits_optimizer_and_writes_copy(self) -> None:
        model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = argparse.Namespace(save_optimizer_checkpoint=False, resume="")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_model.pt"
            copy_path = Path(tmp) / "copy_last_model.pt"
            self.module.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=7,
                losses=[{"step": 7, "loss": 1.25}],
                eval_losses=[],
                args=args,
                dataset_summary={"contract": "test"},
                model_summary={"save_optimizer_checkpoint": False},
                include_optimizer=False,
                copy_safe_path=copy_path,
            )

            payload = torch.load(path, map_location="cpu")
            copy_payload = torch.load(copy_path, map_location="cpu")

        self.assertEqual(int(payload["step"]), 7)
        self.assertFalse(bool(payload["checkpoint_includes_optimizer"]))
        self.assertNotIn("optimizer_state_dict", payload)
        self.assertEqual(copy_payload["model"].get("save_optimizer_checkpoint"), False)

    def test_parser_accepts_model_only_checkpoint_and_resume_flags(self) -> None:
        parser = self.module.build_arg_parser()

        default_args = parser.parse_args(["--sampled-data", "sampled", "--out-dir", "out"])
        model_only_args = parser.parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--no-save-optimizer-checkpoint",
                "--resume",
                "partial/last_model.pt",
                "--no-resume-strict",
            ]
        )

        self.assertTrue(default_args.save_optimizer_checkpoint)
        self.assertEqual(default_args.optimizer_checkpoint_every, -1)
        self.assertEqual(default_args.resume, "")
        self.assertEqual(default_args.qwen_boundary_prior_weight, 0.0)
        self.assertEqual(default_args.qwen_boundary_tokenizer_model_id, "Qwen/Qwen3.5-0.8B-Base")
        self.assertFalse(model_only_args.save_optimizer_checkpoint)
        self.assertEqual(model_only_args.resume, "partial/last_model.pt")
        self.assertFalse(model_only_args.resume_strict)

        hnetpp_args = parser.parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--patch-boundary-mode",
                "hnetpp_flow_dechunk",
            ]
        )
        self.assertEqual(hnetpp_args.patch_boundary_mode, "hnetpp_flow_dechunk")

        hier_args = parser.parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--decoder-latent-mode",
                "hier_add",
            ]
        )
        self.assertEqual(hier_args.decoder_latent_mode, "hier_add")

    def test_training_loop_skips_periodic_checkpoint_at_final_step(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            "step % int(args.checkpoint_every) == 0 and step < int(args.steps)",
            source,
        )

    def test_resume_checkpoint_loads_model_only_weights(self) -> None:
        model = self.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.no_grad():
            model.byte_embed.weight.fill_(0.125)
        args = argparse.Namespace(save_optimizer_checkpoint=False, resume="")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_model.pt"
            self.module.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=11,
                losses=[],
                eval_losses=[],
                args=args,
                dataset_summary={"contract": "test"},
                model_summary={},
                include_optimizer=False,
            )
            resumed = self.build_model()
            summary = self.module.load_resume_checkpoint(
                path,
                model=resumed,
                optimizer=None,
                device=torch.device("cpu"),
                strict=True,
                load_optimizer=False,
            )

        self.assertEqual(summary["step"], 11)
        self.assertFalse(summary["checkpoint_includes_optimizer"])
        self.assertFalse(summary["optimizer_loaded"])
        self.assertEqual(summary["missing_keys"], [])
        self.assertEqual(summary["unexpected_keys"], [])
        self.assertTrue(torch.allclose(resumed.byte_embed.weight, model.byte_embed.weight))


if __name__ == "__main__":
    unittest.main()
