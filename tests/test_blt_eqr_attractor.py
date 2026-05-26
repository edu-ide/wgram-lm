from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "557_train_blt_d_prefixlm_dataio.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blt_d_prefixlm_trainer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DepthLogitModel(torch.nn.Module):
    def __init__(self, logits_by_depth: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.logits_by_depth = logits_by_depth
        self.calls: list[int] = []

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
    ) -> torch.Tensor:
        self.calls.append(int(think_steps))
        return self.logits_by_depth[int(think_steps)].clone().requires_grad_(True)


class DepthLogitStateModel(torch.nn.Module):
    def __init__(
        self,
        logits_by_depth: dict[int, torch.Tensor],
        states_by_depth: dict[int, torch.Tensor],
        speaker_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.logits_by_depth = logits_by_depth
        self.states_by_depth = states_by_depth
        self.speaker_weight = torch.nn.Parameter(speaker_weight.clone())
        self.calls: list[int] = []

    def forward_logits_and_decoder_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(int(think_steps))
        return (
            self.logits_by_depth[int(think_steps)].clone().requires_grad_(True),
            self.states_by_depth[int(think_steps)].clone().requires_grad_(True),
        )

    def answer_embedding_weight(self) -> torch.Tensor:
        return self.speaker_weight


class IdentityGlobalCore(torch.nn.Module):
    position_embedding_mode = "none"

    def _forward_embedded_impl(
        self,
        x: torch.Tensor,
        *,
        think_steps: int,
        return_hidden: bool,
    ) -> torch.Tensor:
        return x + (0.0 * float(think_steps))


class BLTEqRAttractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_eqr_loss_penalizes_deep_answer_that_is_worse_than_shallow(self) -> None:
        labels = torch.tensor([[1]], dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        shallow_good = torch.tensor([[[0.0, 3.0, -2.0]]])
        previous_bad = torch.tensor([[[3.0, 0.0, -2.0]]])
        deep_bad = torch.tensor([[[3.0, 0.0, -2.0]]])
        model = DepthLogitModel({1: shallow_good, 2: previous_bad, 3: deep_bad})

        loss, metrics = self.module.eqr_attractor_regularization_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            shallow_think_steps=1,
            deep_think_steps=3,
            deep_supervision_weight=0.0,
            consistency_weight=0.0,
            residual_weight=0.0,
            improvement_weight=1.0,
            improvement_margin=0.0,
            max_targets=0,
        )

        self.assertGreater(float(loss.detach().item()), 2.0)
        self.assertGreater(float(metrics["eqr_improvement_loss"]), 2.0)
        self.assertEqual(int(metrics["eqr_targets"]), 1)
        self.assertEqual(model.calls, [1, 2, 3])

    def test_eqr_fixed_point_residual_is_smaller_when_last_depths_match(self) -> None:
        labels = torch.tensor([[1]], dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        shallow = torch.tensor([[[0.0, 1.0, -1.0]]])
        previous = torch.tensor([[[0.0, 1.0, -1.0]]])
        deep_same = torch.tensor([[[0.0, 1.0, -1.0]]])
        deep_far = torch.tensor([[[2.0, -1.0, -1.0]]])

        same_loss, same_metrics = self.module.eqr_attractor_regularization_loss(
            DepthLogitModel({1: shallow, 2: previous, 3: deep_same}),
            input_ids,
            labels,
            attention_mask,
            shallow_think_steps=1,
            deep_think_steps=3,
            deep_supervision_weight=0.0,
            consistency_weight=0.0,
            residual_weight=1.0,
            improvement_weight=0.0,
            max_targets=0,
        )
        far_loss, far_metrics = self.module.eqr_attractor_regularization_loss(
            DepthLogitModel({1: shallow, 2: previous, 3: deep_far}),
            input_ids,
            labels,
            attention_mask,
            shallow_think_steps=1,
            deep_think_steps=3,
            deep_supervision_weight=0.0,
            consistency_weight=0.0,
            residual_weight=1.0,
            improvement_weight=0.0,
            max_targets=0,
        )

        self.assertLess(float(same_loss.detach().item()), 1e-6)
        self.assertLess(float(same_metrics["eqr_fixed_point_residual"]), 1e-6)
        self.assertGreater(float(far_loss.detach().item()), float(same_loss.detach().item()) + 0.1)
        self.assertGreater(float(far_metrics["eqr_fixed_point_residual"]), 0.1)

    def test_eqr_cli_defaults_keep_training_contract_disabled(self) -> None:
        args = self.module.build_arg_parser().parse_args(["--sampled-data", "sampled", "--out-dir", "out"])

        self.assertEqual(args.eqr_deep_supervision_weight, 0.0)
        self.assertEqual(args.eqr_consistency_weight, 0.0)
        self.assertEqual(args.eqr_residual_weight, 0.0)
        self.assertEqual(args.eqr_improvement_weight, 0.0)
        self.assertEqual(args.eqr_every, 1)

    def test_answer_attractor_loss_penalizes_deeper_ce_regression(self) -> None:
        labels = torch.tensor([[1]], dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        depth1_good = torch.tensor([[[0.0, 3.0, -2.0]]])
        depth2_bad = torch.tensor([[[3.0, 0.0, -2.0]]])
        model = DepthLogitModel({1: depth1_good, 2: depth2_bad})

        loss, metrics = self.module.answer_attractor_regularization_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            depths=[1, 2],
            ce_weight=0.0,
            monotonic_weight=1.0,
            residual_wrong_weight=0.0,
            improvement_margin=0.0,
            max_targets=0,
        )

        self.assertGreater(float(loss.detach().item()), 2.0)
        self.assertGreater(float(metrics["answer_attractor_monotonic_loss"]), 2.0)
        self.assertEqual(model.calls, [1, 2])

    def test_answer_attractor_loss_allows_deeper_ce_improvement(self) -> None:
        labels = torch.tensor([[1]], dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        depth1_bad = torch.tensor([[[3.0, 0.0, -2.0]]])
        depth2_good = torch.tensor([[[0.0, 3.0, -2.0]]])
        model = DepthLogitModel({1: depth1_bad, 2: depth2_good})

        loss, metrics = self.module.answer_attractor_regularization_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            depths=[1, 2],
            ce_weight=0.0,
            monotonic_weight=1.0,
            residual_wrong_weight=0.0,
            improvement_margin=0.0,
            max_targets=0,
        )

        self.assertLess(float(loss.detach().item()), 1e-6)
        self.assertLess(float(metrics["answer_attractor_monotonic_loss"]), 1e-6)

    def test_answer_attractor_cli_defaults_keep_training_contract_disabled(self) -> None:
        args = self.module.build_arg_parser().parse_args(["--sampled-data", "sampled", "--out-dir", "out"])

        self.assertEqual(args.answer_attractor_depths, [])
        self.assertEqual(args.answer_attractor_ce_weight, 0.0)
        self.assertEqual(args.answer_attractor_monotonic_weight, 0.0)
        self.assertEqual(args.answer_attractor_residual_wrong_weight, 0.0)

    def test_answer_state_attractor_pulls_hidden_toward_gold_speaker_embedding(self) -> None:
        labels = torch.tensor([[1]], dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = torch.tensor([[[0.0, 1.0, -1.0]]])
        speaker = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        depth1_far = torch.tensor([[[-1.0, 0.0]]])
        depth2_close = torch.tensor([[[1.0, 0.0]]])
        model = DepthLogitStateModel(
            {1: logits, 2: logits},
            {1: depth1_far, 2: depth2_close},
            speaker,
        )

        loss, metrics = self.module.answer_state_attractor_regularization_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            depths=[1, 2],
            state_weight=1.0,
            monotonic_weight=0.0,
            residual_wrong_weight=0.0,
            improvement_margin=0.0,
            max_targets=0,
        )

        self.assertGreater(float(loss.detach().item()), 0.9)
        self.assertEqual(int(metrics["answer_state_attractor_best_depth"]), 2)
        self.assertLess(float(metrics["answer_state_attractor_best_distance"]), 1e-6)
        self.assertEqual(model.calls, [1, 2])

    def test_answer_state_attractor_penalizes_deeper_state_regression(self) -> None:
        labels = torch.tensor([[1]], dtype=torch.long)
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = torch.tensor([[[0.0, 1.0, -1.0]]])
        speaker = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        depth1_close = torch.tensor([[[1.0, 0.0]]])
        depth2_far = torch.tensor([[[-1.0, 0.0]]])
        model = DepthLogitStateModel(
            {1: logits, 2: logits},
            {1: depth1_close, 2: depth2_far},
            speaker,
        )

        loss, metrics = self.module.answer_state_attractor_regularization_loss(
            model,
            input_ids,
            labels,
            attention_mask,
            depths=[1, 2],
            state_weight=0.0,
            monotonic_weight=1.0,
            residual_wrong_weight=0.0,
            improvement_margin=0.0,
            max_targets=0,
        )

        self.assertGreater(float(loss.detach().item()), 1.9)
        self.assertGreater(float(metrics["answer_state_attractor_monotonic_loss"]), 1.9)

    def test_answer_state_attractor_cli_defaults_keep_training_contract_disabled(self) -> None:
        args = self.module.build_arg_parser().parse_args(["--sampled-data", "sampled", "--out-dir", "out"])

        self.assertEqual(args.answer_state_attractor_depths, [])
        self.assertEqual(args.answer_state_attractor_weight, 0.0)
        self.assertEqual(args.answer_state_attractor_monotonic_weight, 0.0)
        self.assertEqual(args.answer_state_attractor_residual_wrong_weight, 0.0)

    def test_answer_readback_self_embedding_changes_normal_speaker_logits(self) -> None:
        torch.manual_seed(7)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="none",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        base_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)
        model.answer_readback_mode = "self_embedding"
        model.answer_readback_gate_logit.data.fill_(4.0)
        readback_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)

        self.assertEqual(tuple(base_logits.shape), tuple(readback_logits.shape))
        self.assertGreater(float((readback_logits - base_logits).abs().max().detach().item()), 1e-5)
        self.assertEqual(model.last_readback_metrics["answer_readback_mode"], "self_embedding")
        self.assertGreater(float(model.last_readback_metrics["answer_readback_gate_mean"]), 0.9)

    def test_answer_readback_anchor_embedding_uses_inner_speech_head(self) -> None:
        torch.manual_seed(11)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="anchor_embedding",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            model.answer_readback_gate_logit.fill_(4.0)
            for parameter in model.answer_anchor_head.parameters():
                parameter.zero_()
            model.answer_anchor_head[-1].bias[3] = 8.0
        anchor3_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)
        with torch.no_grad():
            model.answer_anchor_head[-1].bias.zero_()
            model.answer_anchor_head[-1].bias[6] = 8.0
        anchor6_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)

        self.assertGreater(float((anchor6_logits - anchor3_logits).abs().max().detach().item()), 1e-5)
        self.assertEqual(model.last_readback_metrics["answer_readback_mode"], "anchor_embedding")
        self.assertGreater(float(model.last_readback_metrics["cot_anchor_entropy"]), 0.0)

    def test_answer_readback_anchor_embedding_is_confidence_gated(self) -> None:
        torch.manual_seed(12)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="none",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        base_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)
        model.answer_readback_mode = "anchor_embedding"
        model.answer_readback_gate_logit.data.fill_(4.0)
        with torch.no_grad():
            for parameter in model.answer_anchor_head.parameters():
                parameter.zero_()
        uniform_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)
        uniform_delta = float((uniform_logits - base_logits).abs().max().detach().item())
        with torch.no_grad():
            model.answer_anchor_head[-1].bias[5] = 8.0
        confident_logits = model.forward_logits(input_ids, attention_mask, think_steps=2)
        anchor_change = float((confident_logits - uniform_logits).abs().max().detach().item())

        self.assertGreater(anchor_change, max(1e-5, uniform_delta * 0.1))
        self.assertGreater(float(model.last_readback_metrics["cot_anchor_confidence"]), 0.9)

    def test_answer_readback_selected_anchor_broadcasts_workspace_vector(self) -> None:
        torch.manual_seed(14)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="selected_anchor_embedding",
        )
        hidden = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        speaker_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10.0

        with torch.no_grad():
            model.answer_readback_gate_logit.fill_(4.0)
            for parameter in model.answer_anchor_head.parameters():
                parameter.zero_()
            model.answer_anchor_head[-1].bias[5] = 8.0
            model.answer_workspace_selector[-1].weight.zero_()
            model.answer_workspace_selector[-1].bias.zero_()
            model.answer_workspace_selector[-1].weight[0, 1] = 8.0

        refined = model.apply_answer_readback(hidden, speaker_weight)
        delta = refined - hidden

        self.assertEqual(model.last_readback_metrics["answer_readback_mode"], "selected_anchor_embedding")
        self.assertGreater(float(model.last_readback_metrics["answer_workspace_selection_confidence"]), 0.9)
        self.assertLess(float((delta[:, 0, :] - delta[:, 1, :]).abs().max().detach().item()), 1e-5)
        self.assertLess(float((delta[:, 1, :] - delta[:, 2, :]).abs().max().detach().item()), 1e-5)

    def test_workspace_selector_critic_targets_low_ce_anchor_position(self) -> None:
        torch.manual_seed(16)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="selected_anchor_embedding",
        )
        hidden = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 3, 3]], dtype=torch.long)
        with torch.no_grad():
            model.answer_anchor_head[-1].weight.zero_()
            model.answer_anchor_head[-1].bias.zero_()
            model.answer_anchor_head[-1].weight[3, 1] = 8.0
            model.answer_workspace_selector[-1].weight.zero_()
            model.answer_workspace_selector[-1].bias.zero_()

        loss, metrics = model.workspace_selector_critic_loss(hidden, labels, temperature=0.1)
        loss.backward()

        self.assertGreater(float(loss.detach().item()), 0.0)
        self.assertEqual(int(metrics["answer_workspace_selector_targets"]), 2)
        self.assertEqual(int(metrics["answer_workspace_selector_target_best_index"]), 1)
        self.assertGreater(float(metrics["answer_workspace_selector_target_confidence"]), 0.99)
        self.assertIsNotNone(model.answer_workspace_selector[-1].weight.grad)
        self.assertGreater(float(model.answer_workspace_selector[-1].weight.grad.abs().sum().item()), 0.0)

    def test_workspace_selector_final_ce_critic_targets_candidate_that_improves_speaker_ce(self) -> None:
        torch.manual_seed(17)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="selected_anchor_embedding",
        )
        hidden = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 3, 3]], dtype=torch.long)
        with torch.no_grad():
            model.answer_readback_gate_logit.fill_(4.0)
            for parameter in model.answer_anchor_head.parameters():
                parameter.zero_()
            model.answer_anchor_head[0].weight.fill_(1.0)
            model.answer_anchor_head[-1].weight[3, 1] = 8.0
            model.answer_anchor_head[-1].weight[4, 2] = 8.0
            model.clean_decoder.head.weight.zero_()
            model.clean_decoder.head.weight[3, 1] = 8.0
            model.clean_decoder.head.weight[4, 2] = 8.0
            model.answer_workspace_selector[-1].weight.zero_()
            model.answer_workspace_selector[-1].bias.zero_()

        loss, metrics = model.workspace_selector_final_ce_critic_loss(
            hidden,
            labels,
            temperature=0.1,
            max_candidates=3,
            max_targets=0,
        )
        loss.backward()

        self.assertGreater(float(loss.detach().item()), 0.0)
        self.assertEqual(int(metrics["answer_workspace_final_ce_selector_candidate_count"]), 2)
        self.assertEqual(int(metrics["answer_workspace_final_ce_selector_target_best_index"]), 1)
        self.assertGreater(float(metrics["answer_workspace_final_ce_selector_target_confidence"]), 0.99)
        self.assertIsNotNone(model.answer_workspace_selector[-1].weight.grad)
        self.assertGreater(float(model.answer_workspace_selector[-1].weight.grad.abs().sum().item()), 0.0)

    def test_cot_anchor_loss_adds_inner_speech_supervision(self) -> None:
        torch.manual_seed(13)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_mode="anchor_embedding",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        labels = torch.tensor([[-100, 3, -100, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=2,
            diffusion_weight=0.0,
            diffusion_mask_prob=0.0,
            cot_anchor_loss_weight=0.25,
        )

        self.assertGreater(float(metrics["cot_anchor_loss"]), 0.0)
        self.assertEqual(int(metrics["cot_anchor_targets"]), 2)
        self.assertGreater(float(loss.detach().item()), float(metrics["clean_loss"]))

    def test_diffusion_loss_path_uses_grouped_valid_mask(self) -> None:
        torch.manual_seed(15)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        labels = torch.tensor([[-100, 3, -100, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        loss, metrics = model.forward_losses(
            input_ids,
            labels,
            attention_mask,
            think_steps=2,
            diffusion_weight=0.25,
            diffusion_mask_prob=0.5,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(int(metrics["diffusion_targets"]), 0)
        self.assertGreater(float(metrics["diffusion_loss"]), 0.0)

    def test_one_body_decoder_mode_blocks_direct_byte_decoder_shortcut(self) -> None:
        torch.manual_seed(18)
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            decoder_latent_mode="one_body",
        )
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, int(self.module.IGNORE_LABEL_ID))

        captured_inputs: list[torch.Tensor] = []

        def capture_forward_hidden(x: torch.Tensor, **_: object) -> torch.Tensor:
            captured_inputs.append(x.detach().clone())
            return x

        model.clean_decoder.forward_hidden = capture_forward_hidden  # type: ignore[method-assign]
        _ = model.forward_logits_and_decoder_hidden(input_ids, attention_mask, think_steps=2)

        grouped_ids, grouped_mask, _, _, _, _ = model.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )
        _, grouped_byte_embeddings, _, patch_embeddings = model._grouped_patch_embeddings(
            grouped_ids,
            grouped_mask,
        )
        hidden = model._global_hidden(patch_embeddings, think_steps=2)
        pos_ids = torch.arange(model.patch_size, device=input_ids.device)
        pos = model.byte_pos_embed(pos_ids).view(1, 1, model.patch_size, model.d_model)
        expected = model.clean_patch_condition(hidden) + pos
        shortcut_input = grouped_byte_embeddings + expected

        self.assertEqual(len(captured_inputs), 1)
        self.assertLess(
            float((captured_inputs[0] - expected.reshape(-1, model.patch_size, model.d_model)).abs().max().item()),
            1e-6,
        )
        self.assertGreater(
            float((captured_inputs[0] - shortcut_input.reshape(-1, model.patch_size, model.d_model)).abs().max().item()),
            1e-3,
        )

    def test_answer_readback_cli_defaults_keep_training_contract_disabled(self) -> None:
        args = self.module.build_arg_parser().parse_args(["--sampled-data", "sampled", "--out-dir", "out"])

        self.assertEqual(args.answer_readback_mode, "none")
        self.assertEqual(args.answer_readback_gate_init, -4.0)
        self.assertEqual(args.answer_readback_temperature, 1.0)
        self.assertEqual(args.cot_anchor_loss_weight, 0.0)
        self.assertEqual(args.workspace_selector_critic_weight, 0.0)
        self.assertEqual(args.workspace_selector_critic_temperature, 0.25)
        self.assertEqual(args.workspace_selector_final_ce_critic_weight, 0.0)
        self.assertEqual(args.workspace_selector_final_ce_critic_temperature, 0.25)
        self.assertEqual(args.workspace_selector_final_ce_critic_max_candidates, 16)
        self.assertEqual(args.workspace_selector_final_ce_critic_max_targets, 512)
        self.assertFalse(args.allow_diagnostic_bridge_experiment)

    def test_bridge_experiments_require_explicit_diagnostic_opt_in(self) -> None:
        parser = self.module.build_arg_parser()
        args = parser.parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--answer-readback-mode",
                "anchor_embedding",
            ]
        )

        with self.assertRaisesRegex(ValueError, "diagnostic bridge"):
            self.module.validate_architecture_contract(args)

        allowed = parser.parse_args(
            [
                "--sampled-data",
                "sampled",
                "--out-dir",
                "out",
                "--answer-readback-mode",
                "anchor_embedding",
                "--allow-diagnostic-bridge-experiment",
            ]
        )
        self.module.validate_architecture_contract(allowed)

    def test_resume_adaptation_initializes_missing_answer_readback_gate(self) -> None:
        model = self.module.BLTDByteLatentPrefixLM(
            global_core=IdentityGlobalCore(),
            vocab_size=8,
            d_model=4,
            patch_size=2,
            local_layers=1,
            local_heads=2,
            answer_readback_gate_init=-2.0,
        )
        target_state = model.state_dict()
        source_state = {
            key: value
            for key, value in target_state.items()
            if key != "answer_readback_gate_logit"
            and not key.startswith("answer_anchor_head.")
            and not key.startswith("answer_workspace_selector.")
        }

        adapted_state, stats = self.module.adapt_resume_state_dict_for_current_model(source_state, target_state)

        self.assertIn("answer_readback_gate_logit", adapted_state)
        self.assertTrue(torch.equal(adapted_state["answer_readback_gate_logit"], target_state["answer_readback_gate_logit"]))
        self.assertIn("answer_anchor_head.1.weight", adapted_state)
        self.assertTrue(torch.equal(adapted_state["answer_anchor_head.1.weight"], target_state["answer_anchor_head.1.weight"]))
        self.assertIn("answer_workspace_selector.1.weight", adapted_state)
        self.assertTrue(
            torch.equal(
                adapted_state["answer_workspace_selector.1.weight"],
                target_state["answer_workspace_selector.1.weight"],
            )
        )
        self.assertEqual(stats["initialized_missing_answer_readback_gate"], 1)
        self.assertGreater(stats["initialized_missing_answer_anchor_head"], 0)
        self.assertGreater(stats["initialized_missing_answer_workspace_selector"], 0)


if __name__ == "__main__":
    unittest.main()
