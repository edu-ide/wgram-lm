import unittest

import torch
from torch import nn


class LossTests(unittest.TestCase):
    def test_next_token_loss_ignores_padding_targets(self):
        from qtrm_mm.losses import next_token_lm_loss

        logits = torch.zeros(1, 4, 5)
        logits[0, 0, 2] = 10.0
        logits[0, 1, 3] = 10.0
        logits[0, 2, 4] = -10.0
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        masked = next_token_lm_loss(logits, input_ids, attention_mask=attention_mask)
        unmasked = next_token_lm_loss(logits, input_ids)

        self.assertLess(float(masked), 0.01)
        self.assertGreater(float(unmasked), 3.0)

    def test_next_token_loss_can_train_only_unmasked_label_positions(self):
        from qtrm_mm.losses import next_token_lm_loss

        logits = torch.zeros(1, 4, 5)
        logits[0, 0, 2] = -10.0
        logits[0, 1, 3] = 10.0
        logits[0, 2, 4] = 10.0
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])

        masked = next_token_lm_loss(logits, input_ids, labels=labels)
        unmasked = next_token_lm_loss(logits, input_ids)

        self.assertLess(float(masked), 0.01)
        self.assertGreater(float(unmasked), 3.0)

    def test_repetition_unlikelihood_loss_penalizes_confident_wrong_adjacent_repeats(self):
        from qtrm_mm.losses import repetition_unlikelihood_loss

        logits = torch.zeros(1, 5, 6)
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        safe_logits = torch.zeros_like(logits)
        for index, candidate in enumerate([1, 2, 3, 4]):
            safe_logits[0, index, candidate] = -10.0
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        repeated_loss = repetition_unlikelihood_loss(logits, input_ids)
        safe_loss = repetition_unlikelihood_loss(safe_logits, input_ids)

        self.assertGreater(float(repeated_loss), 4.0)
        self.assertLess(float(safe_loss), 0.01)

    def test_repetition_unlikelihood_loss_does_not_penalize_gold_repeats(self):
        from qtrm_mm.losses import repetition_unlikelihood_loss

        logits = torch.zeros(1, 3, 5)
        logits[0, 1, 2] = 10.0
        input_ids = torch.tensor([[1, 2, 2]])
        labels = torch.tensor([[-100, 2, 2]])

        loss = repetition_unlikelihood_loss(logits, input_ids, labels=labels)

        self.assertEqual(float(loss), 0.0)

    def test_simpo_margin_loss_prefers_chosen_sequence(self):
        from qtrm_mm.losses import simpo_margin_loss

        good = simpo_margin_loss(
            chosen_logps=torch.tensor([-0.5]),
            rejected_logps=torch.tensor([-2.0]),
            beta=2.0,
            margin=0.0,
        )
        bad = simpo_margin_loss(
            chosen_logps=torch.tensor([-2.0]),
            rejected_logps=torch.tensor([-0.5]),
            beta=2.0,
            margin=0.0,
        )

        self.assertLess(float(good), float(bad))
        self.assertLess(float(good), 0.1)

    def test_sequence_average_logprob_masks_prompt_labels(self):
        from qtrm_mm.losses import sequence_average_logprob

        logits = torch.zeros(1, 4, 5)
        logits[0, 1, 3] = 8.0
        logits[0, 2, 4] = 8.0
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])

        logp = sequence_average_logprob(logits, input_ids, labels=labels)

        self.assertGreater(float(logp[0]), -0.01)

    def test_simpo_margin_loss_accepts_confidence_weights(self):
        from qtrm_mm.losses import simpo_margin_loss

        weighted = simpo_margin_loss(
            chosen_logps=torch.tensor([-3.0, -0.5]),
            rejected_logps=torch.tensor([-0.5, -3.0]),
            sample_weight=torch.tensor([0.0, 1.0]),
            beta=2.0,
            margin=0.0,
        )

        self.assertLess(float(weighted), 0.1)

    def test_action_policy_loss_prefers_target_controller_action(self):
        from qtrm_mm.losses import action_policy_loss

        good_outputs = {
            "action_logits": torch.tensor([[0.0, 0.0, 8.0, 0.0]], dtype=torch.float32),
            "logits": torch.zeros(1, 1, 4),
        }
        bad_outputs = {
            "action_logits": torch.tensor([[0.0, 0.0, -8.0, 0.0]], dtype=torch.float32),
            "logits": torch.zeros(1, 1, 4),
        }
        target = torch.tensor([2])

        good_loss, good_metrics = action_policy_loss(good_outputs, target=target)
        bad_loss, bad_metrics = action_policy_loss(bad_outputs, target=target)

        self.assertLess(float(good_loss), 0.01)
        self.assertGreater(float(bad_loss), 8.0)
        self.assertEqual(float(good_metrics["action_acc"]), 1.0)

    def test_controller_signal_prediction_loss_trains_binary_signal(self):
        from qtrm_mm.losses import controller_signal_prediction_loss

        good_outputs = {
            "controller_signal_logits": torch.tensor([[8.0, -8.0]], dtype=torch.float32),
            "logits": torch.zeros(1, 1, 4),
        }
        bad_outputs = {
            "controller_signal_logits": torch.tensor([[-8.0, 8.0]], dtype=torch.float32),
            "logits": torch.zeros(1, 1, 4),
        }
        target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        good_loss, good_metrics = controller_signal_prediction_loss(good_outputs, target=target)
        bad_loss, bad_metrics = controller_signal_prediction_loss(bad_outputs, target=target)

        self.assertLess(float(good_loss), 0.01)
        self.assertGreater(float(bad_loss), 7.0)
        self.assertEqual(float(good_metrics["controller_signal_acc"]), 1.0)
        self.assertEqual(float(bad_metrics["controller_signal_acc"]), 0.0)

    def test_answer_decision_loss_trains_block_signal(self):
        from qtrm_mm.losses import answer_decision_loss

        good_outputs = {
            "answer_decision_logits": torch.tensor([[8.0], [-8.0]], dtype=torch.float32),
            "logits": torch.zeros(2, 1, 4),
        }
        bad_outputs = {
            "answer_decision_logits": torch.tensor([[-8.0], [8.0]], dtype=torch.float32),
            "logits": torch.zeros(2, 1, 4),
        }
        target = torch.tensor([1.0, 0.0], dtype=torch.float32)

        good_loss, good_metrics = answer_decision_loss(good_outputs, target=target)
        bad_loss, bad_metrics = answer_decision_loss(bad_outputs, target=target)

        self.assertLess(float(good_loss), 0.01)
        self.assertGreater(float(bad_loss), 7.0)
        self.assertEqual(float(good_metrics["answer_decision_acc"]), 1.0)
        self.assertEqual(float(bad_metrics["answer_decision_acc"]), 0.0)

    def test_answer_residual_governor_loss_targets_donor_errors(self):
        from qtrm_mm.losses import answer_residual_governor_loss

        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, 2, 3, 4]])
        donor_logits = torch.zeros(1, 4, 6)
        donor_logits[0, 0, 2] = 8.0  # correct for label token 2
        donor_logits[0, 1, 5] = 8.0  # wrong for label token 3
        donor_logits[0, 2, 4] = 8.0  # correct for label token 4
        good_outputs = {
            "answer_residual_governor_logits": torch.tensor(
                [[-8.0, 8.0, -8.0, -8.0]],
                dtype=torch.float32,
            ),
            "logits": torch.zeros(1, 4, 6),
        }
        bad_outputs = {
            "answer_residual_governor_logits": torch.tensor(
                [[8.0, -8.0, 8.0, 8.0]],
                dtype=torch.float32,
            ),
            "logits": torch.zeros(1, 4, 6),
        }

        good_loss, good_metrics = answer_residual_governor_loss(
            good_outputs,
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
        )
        bad_loss, bad_metrics = answer_residual_governor_loss(
            bad_outputs,
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
        )

        self.assertLess(float(good_loss), 0.01)
        self.assertGreater(float(bad_loss), 7.0)
        self.assertEqual(float(good_metrics["answer_residual_governor_acc"]), 1.0)
        self.assertAlmostEqual(float(good_metrics["answer_residual_governor_target_open_rate"]), 1.0 / 3.0)

    def test_qtrm_smoke_loss_includes_answer_decision_component(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.losses import qtrm_smoke_loss

        torch.manual_seed(0)
        model = QTRMMultimodalModel(
            QTRMConfig(
                vocab_size=64,
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                d_ff=64,
                n_prelude_layers=1,
                n_core_layers=1,
                n_coda_layers=1,
                workspace_tokens=4,
                h_cycles=1,
                l_cycles=1,
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                answer_decision_head_enabled=True,
            )
        )
        ids = torch.randint(1, 32, (2, 8))
        labels = torch.full_like(ids, -100)
        target = torch.tensor([1.0, 0.0], dtype=torch.float32)

        base_loss, _, _ = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
        )
        decision_loss, metrics, outputs = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            answer_decision_weight=1.0,
            answer_decision_target=target,
        )

        self.assertEqual(float(base_loss), 0.0)
        self.assertGreater(float(decision_loss), 0.0)
        self.assertIn("answer_decision", metrics)
        self.assertIn("answer_decision_acc", metrics)
        self.assertEqual(outputs["answer_decision_logits"].shape, (2,))

    def test_qtrm_smoke_loss_includes_answer_residual_governor_component(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.losses import qtrm_smoke_loss

        torch.manual_seed(0)
        model = QTRMMultimodalModel(
            QTRMConfig(
                vocab_size=64,
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                d_ff=64,
                n_prelude_layers=1,
                n_core_layers=1,
                n_coda_layers=1,
                workspace_tokens=4,
                h_cycles=1,
                l_cycles=1,
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                answer_bottleneck_enabled=True,
                answer_bottleneck_requires_workspace_memory=False,
                answer_residual_governor_enabled=True,
            )
        )
        ids = torch.randint(1, 32, (2, 8))
        labels = ids.clone()
        labels[:, 0] = -100
        donor_logits = torch.zeros(2, 8, 64)

        base_loss, _, _ = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            donor_logits=donor_logits,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
        )
        governor_loss, metrics, outputs = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            donor_logits=donor_logits,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            answer_residual_governor_weight=1.0,
        )

        self.assertEqual(float(base_loss), 0.0)
        self.assertGreater(float(governor_loss), 0.0)
        self.assertIn("answer_residual_governor", metrics)
        self.assertIn("answer_residual_governor_acc", metrics)
        self.assertEqual(outputs["answer_residual_governor_logits"].shape, (2, 8))

    def test_answer_residual_governor_loss_backward_has_no_inplace_version_error(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.losses import qtrm_smoke_loss

        torch.manual_seed(0)
        model = QTRMMultimodalModel(
            QTRMConfig(
                vocab_size=64,
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                d_ff=64,
                n_prelude_layers=1,
                n_core_layers=1,
                n_coda_layers=1,
                workspace_tokens=4,
                h_cycles=1,
                l_cycles=1,
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                answer_bottleneck_enabled=True,
                answer_bottleneck_requires_workspace_memory=False,
                answer_residual_governor_enabled=True,
                donor_logits_scale=1.0,
            )
        )
        ids = torch.randint(1, 32, (1, 8))
        labels = ids.clone()
        labels[:, 0] = -100
        donor_logits = torch.zeros(1, 8, 64)

        loss, _, _ = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            donor_logits=donor_logits,
            lm_weight=1.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            answer_residual_governor_weight=1.0,
        )
        loss.backward()

        self.assertIsNotNone(model.answer_residual_governor.weight.grad)

    def test_qtrm_smoke_loss_includes_action_policy_component(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.losses import qtrm_smoke_loss

        torch.manual_seed(0)
        model = QTRMMultimodalModel(
            QTRMConfig(
                vocab_size=64,
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                d_ff=64,
                n_prelude_layers=1,
                n_core_layers=1,
                n_coda_layers=1,
                workspace_tokens=4,
                h_cycles=1,
                l_cycles=1,
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                num_actions=10,
            )
        )
        ids = torch.randint(1, 32, (2, 8))
        labels = torch.full_like(ids, -100)
        target = torch.tensor([1, 3], dtype=torch.long)

        base_loss, _, _ = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
        )
        action_loss, metrics, _ = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            action_policy_weight=1.0,
            action_targets=target,
        )

        self.assertEqual(float(base_loss), 0.0)
        self.assertGreater(float(action_loss), 0.0)
        self.assertIn("action_policy", metrics)
        self.assertIn("action_acc", metrics)

    def test_qtrm_smoke_loss_includes_learned_controller_signal_component(self):
        from qtrm_mm import QTRMConfig, QTRMMultimodalModel
        from qtrm_mm.losses import qtrm_smoke_loss

        torch.manual_seed(0)
        model = QTRMMultimodalModel(
            QTRMConfig(
                vocab_size=64,
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                d_ff=64,
                n_prelude_layers=1,
                n_core_layers=1,
                n_coda_layers=1,
                workspace_tokens=4,
                h_cycles=1,
                l_cycles=1,
                outer_steps=1,
                visual_dim=16,
                max_visual_tokens=4,
                num_actions=10,
                controller_signal_enabled=True,
                controller_signal_dim=2,
                controller_signal_source="learned_core",
                controller_signal_base_scale=0.0,
            )
        )
        ids = torch.randint(1, 32, (2, 8))
        labels = torch.full_like(ids, -100)
        target = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        loss, metrics, outputs = qtrm_smoke_loss(
            model,
            ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            controller_signal_weight=1.0,
            controller_signal_target=target,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertIn("controller_signal", metrics)
        self.assertIn("controller_signal_acc", metrics)
        self.assertEqual(outputs["controller_signal_logits"].shape, (2, 2))

    def test_donor_logit_distillation_loss_prefers_matching_teacher_distribution(self):
        from qtrm_mm.losses import donor_logit_distillation_loss

        donor_logits = torch.zeros(1, 3, 5)
        donor_logits[0, 0, 2] = 8.0
        donor_logits[0, 1, 3] = 8.0
        matching_logits = donor_logits.clone()
        mismatched_logits = torch.zeros(1, 3, 5)
        mismatched_logits[0, 0, 4] = 8.0
        mismatched_logits[0, 1, 4] = 8.0
        input_ids = torch.tensor([[1, 2, 3]])

        matching = donor_logit_distillation_loss(
            matching_logits,
            donor_logits,
            input_ids,
        )
        mismatched = donor_logit_distillation_loss(
            mismatched_logits,
            donor_logits,
            input_ids,
        )

        self.assertLess(float(matching), 0.01)
        self.assertGreater(float(mismatched), 7.0)

    def test_greedy_token_margin_loss_forces_target_above_top_competitor(self):
        from qtrm_mm.losses import greedy_token_margin_loss

        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, -100, 3]])
        good_logits = torch.zeros(1, 3, 5)
        good_logits[0, 1, 3] = 5.0
        bad_logits = torch.zeros(1, 3, 5)
        bad_logits[0, 1, 4] = 5.0

        good_loss, good_metrics = greedy_token_margin_loss(
            good_logits,
            input_ids,
            labels=labels,
            margin=1.0,
        )
        bad_loss, bad_metrics = greedy_token_margin_loss(
            bad_logits,
            input_ids,
            labels=labels,
            margin=1.0,
        )

        self.assertLess(float(good_loss), 0.01)
        self.assertGreater(float(bad_loss), 5.0)
        self.assertEqual(float(good_metrics["greedy_token_win_rate"]), 1.0)
        self.assertEqual(float(bad_metrics["greedy_token_win_rate"]), 0.0)

    def test_greedy_token_margin_loss_can_focus_on_donor_errors(self):
        from qtrm_mm.losses import greedy_token_margin_loss

        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, 2, 3, 4]])
        logits = torch.zeros(1, 4, 6)
        logits[0, 0, 5] = 8.0
        logits[0, 1, 5] = 8.0
        logits[0, 2, 5] = 8.0
        donor_logits = torch.zeros(1, 4, 6)
        donor_logits[0, 0, 2] = 8.0
        donor_logits[0, 1, 5] = 8.0
        donor_logits[0, 2, 4] = 8.0

        loss, metrics = greedy_token_margin_loss(
            logits,
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
            margin=1.0,
            only_donor_errors=True,
        )

        self.assertGreater(float(loss), 5.0)
        self.assertAlmostEqual(float(metrics["greedy_token_donor_error_rate"]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(metrics["greedy_token_active_rate"]), 1.0 / 3.0, places=5)

    def test_donor_correct_margin_loss_preserves_donor_correct_tokens(self):
        from qtrm_mm.losses import donor_correct_margin_loss

        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, 2, 3, 4]])
        donor_logits = torch.zeros(1, 4, 6)
        donor_logits[0, 0, 2] = 8.0  # donor correct, should be protected
        donor_logits[0, 1, 5] = 8.0  # donor wrong, should be ignored here
        donor_logits[0, 2, 4] = 8.0  # donor correct, should be protected
        good_logits = torch.zeros(1, 4, 6)
        good_logits[0, 0, 2] = 8.0
        good_logits[0, 1, 5] = 8.0
        good_logits[0, 2, 4] = 8.0
        bad_logits = torch.zeros(1, 4, 6)
        bad_logits[0, 0, 5] = 8.0
        bad_logits[0, 1, 5] = 8.0
        bad_logits[0, 2, 5] = 8.0

        good_loss, good_metrics = donor_correct_margin_loss(
            good_logits,
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
            margin=1.0,
        )
        bad_loss, bad_metrics = donor_correct_margin_loss(
            bad_logits,
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
            margin=1.0,
        )

        self.assertLess(float(good_loss), 0.01)
        self.assertGreater(float(bad_loss), 5.0)
        self.assertAlmostEqual(
            float(good_metrics["donor_correct_margin_active_rate"]),
            2.0 / 3.0,
            places=5,
        )
        self.assertEqual(float(good_metrics["donor_correct_margin_win_rate"]), 1.0)
        self.assertEqual(float(bad_metrics["donor_correct_margin_win_rate"]), 0.0)

    def test_qtrm_smoke_loss_adds_donor_correct_margin_component(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 4] = 8.0
                logits[0, 1, 4] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, 2, 3]])
        donor_logits = torch.zeros(1, 3, 5)
        donor_logits[0, 0, 2] = 8.0
        donor_logits[0, 1, 3] = 8.0

        disabled, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            donor_correct_margin_weight=0.0,
        )
        enabled, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            donor_logits=donor_logits,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            donor_correct_margin_weight=1.0,
            donor_correct_margin=1.0,
        )

        self.assertEqual(float(disabled), 0.0)
        self.assertGreater(float(enabled), 5.0)
        self.assertIn("donor_correct_margin", metrics)
        self.assertEqual(float(metrics["donor_correct_margin_win_rate"]), 0.0)

    def test_qtrm_smoke_loss_adds_donor_logit_distillation_metric(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                donor_logits = kwargs["donor_logits"]
                logits = torch.zeros_like(donor_logits)
                logits[0, 0, 4] = 8.0
                logits[0, 1, 4] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        donor_logits = torch.zeros(1, 3, 5)
        donor_logits[0, 0, 2] = 8.0
        donor_logits[0, 1, 3] = 8.0
        input_ids = torch.tensor([[1, 2, 3]])
        lm_only, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            donor_logits=donor_logits,
            jepa_weight=0.0,
            aux_weight=0.0,
            donor_kl_weight=0.0,
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            donor_logits=donor_logits,
            jepa_weight=0.0,
            aux_weight=0.0,
            donor_kl_weight=1.0,
        )

        self.assertIn("donor_kl", metrics)
        self.assertGreater(float(metrics["donor_kl"]), 7.0)
        self.assertGreater(float(weighted), float(lm_only))

    def test_qtrm_smoke_loss_adds_greedy_token_margin_component(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 4] = 8.0
                logits[0, 1, 4] = 8.0
                return {
                    "logits": logits,
                    "qtrm_residual_logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, 2, 3]])
        disabled, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            greedy_token_margin_weight=0.0,
        )
        enabled, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            greedy_token_margin_weight=1.0,
            greedy_token_margin=1.0,
        )

        self.assertEqual(float(disabled), 0.0)
        self.assertGreater(float(enabled), 5.0)
        self.assertIn("greedy_token_margin", metrics)
        self.assertEqual(float(metrics["greedy_token_win_rate"]), 0.0)

    def test_qtrm_smoke_loss_can_supervise_student_only_logits(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                fused_logits = torch.zeros(1, 3, 5)
                fused_logits[0, 0, 2] = 8.0
                fused_logits[0, 1, 3] = 8.0
                student_logits = torch.zeros(1, 3, 5)
                student_logits[0, 0, 4] = 8.0
                student_logits[0, 1, 4] = 8.0
                return {
                    "logits": fused_logits,
                    "qtrm_logits": student_logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        fused_only, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            student_lm_weight=0.0,
        )
        with_student, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            student_lm_weight=1.0,
        )

        self.assertIn("student_lm", metrics)
        self.assertLess(float(fused_only), 0.01)
        self.assertGreater(float(metrics["student_lm"]), 7.0)
        self.assertGreater(float(with_student), float(fused_only) + 7.0)

    def test_qtrm_smoke_loss_adds_canonical_causal_ablation_contrast(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()
                self.calls = []

            def forward(self, input_ids, **kwargs):
                self.calls.append(dict(kwargs))
                logits = torch.zeros(1, 3, 5)
                if kwargs.get("disable_core"):
                    logits[0, 0, 4] = 8.0
                    logits[0, 1, 4] = 8.0
                else:
                    logits[0, 0, 2] = 8.0
                    logits[0, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        loss, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            canonical_causal_weight=1.0,
            canonical_causal_beta=2.0,
            canonical_causal_margin=0.0,
            canonical_causal_ablation_modes=["core_off"],
        )

        self.assertIn("canonical_causal", metrics)
        self.assertIn("canonical_causal_margin", metrics)
        self.assertLess(float(loss), 0.1)
        self.assertGreater(float(metrics["canonical_causal_margin"]), 7.0)

    def test_qtrm_smoke_loss_can_contrast_transition_state_off_ablation(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()
                self.calls = []

            def forward(self, input_ids, **kwargs):
                self.calls.append(dict(kwargs))
                logits = torch.zeros(1, 3, 5)
                if kwargs.get("disable_transition_state"):
                    logits[0, 0, 4] = 8.0
                    logits[0, 1, 4] = 8.0
                else:
                    logits[0, 0, 2] = 8.0
                    logits[0, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        model = FakeModel()
        loss, metrics, _ = qtrm_smoke_loss(
            model,
            input_ids,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            canonical_causal_weight=1.0,
            canonical_causal_ablation_modes=["transition_state_off"],
        )

        self.assertTrue(any(call.get("disable_transition_state") for call in model.calls))
        self.assertLess(float(loss), 0.1)
        self.assertGreater(float(metrics["canonical_causal_margin"]), 7.0)

    def test_canonical_causal_loss_uses_qtrm_residual_not_donor_fused_logits(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                donor_dominated_logits = torch.zeros(1, 3, 5)
                residual_logits = torch.zeros(1, 3, 5)
                if kwargs.get("disable_core"):
                    residual_logits[0, 0, 4] = 8.0
                    residual_logits[0, 1, 4] = 8.0
                else:
                    residual_logits[0, 0, 2] = 8.0
                    residual_logits[0, 1, 3] = 8.0
                return {
                    "logits": donor_dominated_logits,
                    "qtrm_residual_logits": residual_logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        loss, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            lm_weight=0.0,
            jepa_weight=0.0,
            aux_weight=0.0,
            canonical_causal_weight=1.0,
            canonical_causal_ablation_modes=["core_off"],
        )

        self.assertLess(float(loss), 0.1)
        self.assertGreater(float(metrics["canonical_causal_margin"]), 7.0)

    def test_qtrm_smoke_loss_can_disable_base_lm_term_for_head_only_probes(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 4] = 8.0
                logits[0, 1, 4] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        weighted, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            lm_weight=1.0,
        )
        disabled, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            lm_weight=0.0,
        )

        self.assertGreater(float(metrics["lm"]), 7.0)
        self.assertGreater(float(weighted), 7.0)
        self.assertEqual(float(disabled), 0.0)

    def test_evidence_span_reader_loss_prefers_correct_span(self):
        from qtrm_mm.losses import evidence_span_reader_loss

        good_outputs = {
            "logits": torch.zeros(1, 1, 4),
            "evidence_span_start_logits": torch.tensor([[0.0, 8.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[0.0, 0.0, 8.0]]),
            "evidence_span_no_answer_logits": torch.tensor([-8.0]),
        }
        bad_outputs = {
            "logits": torch.zeros(1, 1, 4),
            "evidence_span_start_logits": torch.tensor([[8.0, 0.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[8.0, 0.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([8.0]),
        }
        kwargs = {
            "start_target": torch.tensor([1]),
            "end_target": torch.tensor([2]),
            "no_answer_target": torch.tensor([0.0]),
            "sample_weight": torch.tensor([1.0]),
        }

        good, metrics = evidence_span_reader_loss(good_outputs, **kwargs)
        bad, _ = evidence_span_reader_loss(bad_outputs, **kwargs)

        self.assertLess(float(good), 0.01)
        self.assertGreater(float(bad), 15.0)
        self.assertEqual(float(metrics["start_acc"]), 1.0)
        self.assertEqual(float(metrics["end_acc"]), 1.0)

    def test_evidence_span_reader_loss_suppresses_spans_on_no_answer(self):
        from qtrm_mm.losses import evidence_span_reader_loss

        confident_decoy_outputs = {
            "logits": torch.zeros(1, 1, 4),
            "evidence_span_start_logits": torch.tensor([[9.0, 0.0, 0.0]]),
            "evidence_span_end_logits": torch.tensor([[0.0, 9.0, 0.0]]),
            "evidence_span_no_answer_logits": torch.tensor([9.0]),
        }

        loss, metrics = evidence_span_reader_loss(
            confident_decoy_outputs,
            start_target=torch.tensor([-100]),
            end_target=torch.tensor([-100]),
            no_answer_target=torch.tensor([1.0]),
            sample_weight=torch.tensor([1.0]),
            no_answer_span_suppression_weight=1.0,
        )

        self.assertGreater(float(loss), 8.0)
        self.assertGreater(float(metrics["no_answer_span_score"]), 8.0)

    def test_qtrm_smoke_loss_adds_repetition_unlikelihood_metric(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 4, 5)
                logits[0, 0, 2] = 8.0
                logits[0, 1, 2] = 8.0
                logits[0, 2, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3, 4]])
        lm_only, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            repeat_unlikelihood_weight=0.0,
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            repeat_unlikelihood_weight=1.0,
        )

        self.assertIn("repeat_ul", metrics)
        self.assertGreater(float(metrics["repeat_ul"]), 3.0)
        self.assertGreater(float(weighted), float(lm_only))

    def test_generation_verifier_loss_trains_repeat_stop_and_quality_heads(self):
        from qtrm_mm.losses import generation_verifier_loss

        outputs = {
            "logits": torch.zeros(2, 3, 5),
            "generation_repeat_logits": torch.tensor([4.0, -4.0]),
            "generation_stop_logits": torch.tensor([3.0, -3.0]),
            "generation_quality_logits": torch.tensor([-3.0, 3.0]),
        }

        good_loss, probs = generation_verifier_loss(
            outputs,
            repeat_target=torch.tensor([1.0, 0.0]),
            stop_target=torch.tensor([1.0, 0.0]),
            quality_target=torch.tensor([0.0, 1.0]),
        )
        bad_loss, _ = generation_verifier_loss(
            outputs,
            repeat_target=torch.tensor([0.0, 1.0]),
            stop_target=torch.tensor([0.0, 1.0]),
            quality_target=torch.tensor([1.0, 0.0]),
        )

        self.assertLess(float(good_loss), 0.1)
        self.assertGreater(float(bad_loss), 3.0)
        self.assertIn("repeat_prob", probs)
        self.assertIn("stop_prob", probs)
        self.assertIn("quality_prob", probs)

    def test_qtrm_smoke_loss_adds_generation_verifier_metric(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 2] = 8.0
                logits[0, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                    "generation_repeat_logits": torch.tensor([-4.0]),
                    "generation_stop_logits": torch.tensor([-4.0]),
                    "generation_quality_logits": torch.tensor([4.0]),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        unweighted, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            generation_verifier_weight=0.0,
            generation_verifier_repeat_target=torch.tensor([1.0]),
            generation_verifier_stop_target=torch.tensor([1.0]),
            generation_verifier_quality_target=torch.tensor([0.0]),
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            generation_verifier_weight=1.0,
            generation_verifier_repeat_target=torch.tensor([1.0]),
            generation_verifier_stop_target=torch.tensor([1.0]),
            generation_verifier_quality_target=torch.tensor([0.0]),
        )

        self.assertIn("generation_verifier", metrics)
        self.assertGreater(float(metrics["generation_verifier"]), 3.0)
        self.assertGreater(float(weighted), float(unweighted) + 3.0)

    def test_qtrm_smoke_loss_adds_pairwise_preference_metric(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
                if int(input_ids[0, 1]) == 2:
                    logits[0, 0, 4] = 8.0
                    logits[0, 1, 4] = 8.0
                else:
                    logits[0, 0, 4] = 8.0
                    logits[0, 1, 4] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, 2, 3]])
        rejected_ids = torch.tensor([[1, 4, 4]])
        rejected_labels = torch.tensor([[-100, 4, 4]])
        base, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            jepa_weight=0.0,
            aux_weight=0.0,
            preference_weight=0.0,
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            preference_rejected_input_ids=rejected_ids,
            preference_rejected_labels=rejected_labels,
            jepa_weight=0.0,
            aux_weight=0.0,
            preference_weight=1.0,
            preference_beta=2.0,
            preference_margin=0.0,
        )

        self.assertIn("preference", metrics)
        self.assertGreater(float(metrics["preference"]), 4.0)
        self.assertLess(float(metrics["preference_margin_logp"]), 0.0)
        self.assertGreater(float(weighted), float(base))

    def test_qtrm_smoke_loss_adds_workspace_counterfactual_contrastive_metric(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
                workspace = kwargs.get("workspace_text_states")
                if workspace is not None and float(workspace.sum()) > 0.0:
                    logits[0, 0, 2] = 8.0
                    logits[0, 1, 3] = 8.0
                else:
                    logits[0, 0, 4] = 8.0
                    logits[0, 1, 4] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, 2, 3]])
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            workspace_text_states=torch.ones(1, 3, 4),
            workspace_counterfactual_text_states=torch.zeros(1, 3, 4),
            jepa_weight=0.0,
            aux_weight=0.0,
            workspace_contrastive_weight=1.0,
            workspace_contrastive_beta=2.0,
            workspace_contrastive_margin=0.0,
        )

        self.assertIn("workspace_contrastive", metrics)
        self.assertLess(float(metrics["workspace_contrastive"]), 0.1)
        self.assertGreater(float(metrics["workspace_margin_logp"]), 4.0)
        self.assertLess(float(weighted), 0.2)

        _, reversed_metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            workspace_text_states=torch.zeros(1, 3, 4),
            workspace_counterfactual_text_states=torch.ones(1, 3, 4),
            jepa_weight=0.0,
            aux_weight=0.0,
            workspace_contrastive_weight=1.0,
            workspace_contrastive_beta=2.0,
            workspace_contrastive_margin=0.0,
        )

        self.assertGreater(float(reversed_metrics["workspace_contrastive"]), 4.0)
        self.assertLess(float(reversed_metrics["workspace_margin_logp"]), -4.0)

    def test_qtrm_smoke_loss_respects_component_weights(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 2] = 8.0
                logits[0, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        lm_only, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
        )
        weighted, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.5,
            aux_weight=1.0,
        )

        self.assertAlmostEqual(float(lm_only), float(metrics["lm"]), places=5)
        self.assertGreater(float(weighted), float(lm_only))

    def test_core_halt_loss_trains_q_halt_logits_against_targets(self):
        from qtrm_mm.losses import core_halt_loss

        good = {
            "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
            "core_q_continue_logits": torch.empty(2, 0),
        }
        bad = {
            "core_q_halt_logits": torch.tensor([[-3.0, -3.0], [3.0, 3.0]]),
            "core_q_continue_logits": torch.empty(2, 0),
        }
        targets = torch.tensor([1.0, 0.0])

        self.assertLess(float(core_halt_loss(good, targets)), 0.1)
        self.assertGreater(float(core_halt_loss(bad, targets)), 2.0)

    def test_qtrm_smoke_loss_adds_core_halt_loss_without_forwarding_targets(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                if "core_halt_targets" in kwargs:
                    raise AssertionError("core_halt_targets must not be forwarded to the model")
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                }

        input_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        targets = torch.tensor([1.0, 0.0])
        lm_only, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=0.0,
            core_halt_targets=targets,
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_targets=targets,
        )

        self.assertIn("core_halt", metrics)
        self.assertGreater(float(weighted), float(lm_only))

    def test_infer_core_halt_targets_requires_exact_correctness_and_verifier_pass(self):
        from qtrm_mm.losses import infer_core_halt_targets

        logits = torch.zeros(2, 3, 5)
        logits[0, 0, 2] = 8.0
        logits[0, 1, 3] = 8.0
        logits[1, 0, 2] = 8.0
        logits[1, 1, 4] = 8.0
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
        outputs = {"logits": logits}

        targets = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=labels,
            verifier_passed=torch.tensor([True, True]),
        )

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0])))

        targets = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=labels,
            verifier_passed=torch.tensor([False, True]),
        )

        self.assertTrue(torch.equal(targets, torch.tensor([0.0, 0.0])))

    def test_infer_core_halt_targets_applies_donor_kl_stability_gate(self):
        from qtrm_mm.losses import infer_core_halt_targets

        logits = torch.zeros(2, 3, 5)
        logits[:, 0, 2] = 8.0
        logits[:, 1, 3] = 8.0
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
        donor_logits = logits.clone()
        donor_logits[1, 0] = 0.0
        donor_logits[1, 0, 4] = 8.0
        donor_logits[1, 1] = 0.0
        donor_logits[1, 1, 4] = 8.0
        outputs = {"logits": logits}

        targets = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=labels,
            donor_logits=donor_logits,
            donor_kl_threshold=0.1,
        )

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0])))

    def test_infer_core_halt_targets_can_report_availability_diagnostics(self):
        from qtrm_mm.losses import infer_core_halt_targets

        logits = torch.zeros(3, 3, 5)
        logits[:, 0, 2] = 8.0
        logits[:, 1, 3] = 8.0
        logits[1, 1] = 0.0
        logits[1, 1, 4] = 8.0
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3], [-100, 2, 3]])
        donor_logits = logits.clone()
        donor_logits[2, 0] = 0.0
        donor_logits[2, 0, 4] = 8.0
        donor_logits[2, 1] = 0.0
        donor_logits[2, 1, 4] = 8.0
        outputs = {"logits": logits}

        targets, diag = infer_core_halt_targets(
            outputs,
            input_ids=torch.tensor([[0, 2, 3], [0, 2, 3], [0, 2, 3]]),
            labels=labels,
            donor_logits=donor_logits,
            donor_kl_threshold=0.1,
            return_diagnostics=True,
        )

        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 0.0, 0.0])))
        self.assertAlmostEqual(float(diag["exact_next_token_pass_rate"]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["donor_kl_pass_rate"]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["halt_target_pos_rate"]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["halt_target_neg_rate"]), 2.0 / 3.0, places=5)

    def test_qtrm_smoke_loss_can_infer_core_halt_targets_from_batch(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                if "verifier_passed" in kwargs:
                    raise AssertionError("verifier_passed must not be forwarded to the model")
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                }

        input_ids = torch.tensor([[0, 2, 3], [0, 2, 3]])
        labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
        loss, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            labels=labels,
            verifier_passed=torch.tensor([True, False]),
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_auto_targets=True,
        )

        self.assertIn("core_halt", metrics)
        self.assertLess(float(metrics["core_halt"]), 0.1)
        self.assertGreater(float(loss), 0.0)

    def test_qtrm_smoke_loss_reports_auto_core_halt_target_rates(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                }

        _, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=torch.tensor([[-100, 2, 3], [-100, 2, 3]]),
            verifier_passed=torch.tensor([True, False]),
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_auto_targets=True,
        )

        self.assertIn("halt_target_pos_rate", metrics)
        self.assertIn("halt_target_neg_rate", metrics)
        self.assertIn("exact_next_token_pass_rate", metrics)
        self.assertIn("verifier_pass_rate", metrics)
        self.assertAlmostEqual(float(metrics["halt_target_pos_rate"]), 0.5, places=5)
        self.assertAlmostEqual(float(metrics["halt_target_neg_rate"]), 0.5, places=5)
        self.assertAlmostEqual(float(metrics["exact_next_token_pass_rate"]), 1.0, places=5)
        self.assertAlmostEqual(float(metrics["verifier_pass_rate"]), 0.5, places=5)

    def test_teacher_depth_halt_targets_use_earliest_stable_core_state(self):
        from qtrm_mm.losses import infer_core_halt_targets_from_teacher_depth

        states = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
            ]
        )
        outputs = {"core_depth_states": states}

        halt, cont, diag = infer_core_halt_targets_from_teacher_depth(
            outputs,
            similarity_threshold=0.99,
            min_step=1,
            return_diagnostics=True,
        )

        self.assertTrue(
            torch.equal(
                halt,
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(cont, 1.0 - halt))
        self.assertAlmostEqual(float(diag["teacher_depth_halt_pos_rate"]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(diag["teacher_depth_earliest_step_mean"]), 2.0, places=5)

    def test_teacher_depth_halt_targets_prefer_output_logit_stability(self):
        from qtrm_mm.losses import infer_core_halt_targets_from_teacher_depth

        logits = torch.zeros(2, 3, 5)
        logits[0, :, 2] = 8.0
        logits[1, 0, 3] = 8.0
        logits[1, 1:, 2] = 8.0
        outputs = {
            "core_depth_last_logits": logits,
            "core_depth_states": torch.zeros(2, 3, 4),
        }

        halt, cont, diag = infer_core_halt_targets_from_teacher_depth(
            outputs,
            logit_kl_threshold=0.01,
            return_diagnostics=True,
        )

        self.assertTrue(
            torch.equal(
                halt,
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(cont, 1.0 - halt))
        self.assertIn("teacher_depth_logit_kl_mean", diag)
        self.assertIn("teacher_depth_top1_match_rate", diag)
        self.assertAlmostEqual(float(diag["teacher_depth_earliest_step_mean"]), 1.5, places=5)

    def test_qtrm_smoke_loss_can_use_teacher_depth_core_halt_targets(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()
                self.saw_enable_core_halt = None
                self.saw_return_core_depth_logits = None

            def forward(self, input_ids, **kwargs):
                self.saw_enable_core_halt = kwargs.get("enable_core_halt")
                self.saw_return_core_depth_logits = kwargs.get("return_core_depth_logits")
                logits = torch.zeros(2, 3, 5)
                logits[:, 0, 2] = 8.0
                logits[:, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(2, 2, 4),
                    "jepa_target": torch.zeros(2, 2, 4),
                    "jepa_mask": torch.ones(2, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(2, 3, 4),
                    "jepa_latent_mask": torch.ones(2, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(2, 1),
                    "action_logits": torch.zeros(2, 3),
                    "core_q_halt_logits": torch.tensor([[3.0, 3.0, 3.0], [-3.0, 3.0, 3.0]]),
                    "core_q_continue_logits": torch.empty(2, 0),
                    "core_depth_last_logits": torch.tensor(
                        [
                            [[8.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                            [[0.0, 8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0, 0.0]],
                        ]
                    ),
                    "core_depth_states": torch.tensor(
                        [
                            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                        ]
                    ),
                }

        model = FakeModel()
        _, metrics, _ = qtrm_smoke_loss(
            model,
            torch.tensor([[0, 2, 3], [0, 2, 3]]),
            labels=torch.tensor([[-100, 2, 3], [-100, 2, 3]]),
            jepa_weight=0.0,
            aux_weight=0.0,
            core_halt_weight=1.0,
            core_halt_auto_targets=True,
            core_halt_target_mode="teacher_depth",
            core_halt_teacher_depth_threshold=0.99,
        )

        self.assertIn("teacher_depth_halt_pos_rate", metrics)
        self.assertIn("teacher_depth_earliest_step_mean", metrics)
        self.assertLess(float(metrics["core_halt"]), 0.1)
        self.assertIs(model.saw_enable_core_halt, False)
        self.assertIs(model.saw_return_core_depth_logits, True)

    def test_qtrm_smoke_loss_supervises_logical_evidence_and_causal_gate(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                for key in (
                    "logical_support_target",
                    "logical_refute_target",
                    "logical_missing_target",
                    "causal_evidence_target",
                ):
                    if key in kwargs:
                        raise AssertionError(f"{key} must not be forwarded to the model")
                workspace = kwargs.get("workspace_text_states")
                supported = workspace is not None and float(workspace.sum()) > 0.0
                evidence_value = 4.0 if supported else -4.0
                anti_evidence_value = -4.0 if supported else 4.0
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
                if supported:
                    logits[:, 0, 2] = 8.0
                    logits[:, 1, 3] = 8.0
                else:
                    logits[:, 0, 4] = 8.0
                    logits[:, 1, 4] = 8.0
                gate_logits = torch.full((input_ids.shape[0],), evidence_value)
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                    "evidence_support_logits": torch.full((input_ids.shape[0],), evidence_value),
                    "evidence_refute_logits": torch.full((input_ids.shape[0],), anti_evidence_value),
                    "evidence_missing_logits": torch.full((input_ids.shape[0],), anti_evidence_value),
                    "evidence_causal_gate_logits": gate_logits,
                    "evidence_bottleneck_gate_logits": gate_logits,
                    "evidence_bottleneck_gate": gate_logits.sigmoid(),
                }

        _, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            torch.tensor([[1, 2, 3]]),
            labels=torch.tensor([[-100, 2, 3]]),
            workspace_text_states=torch.ones(1, 3, 4),
            workspace_counterfactual_text_states=torch.zeros(1, 3, 4),
            logical_support_target=torch.tensor([1.0]),
            logical_refute_target=torch.tensor([0.0]),
            logical_missing_target=torch.tensor([0.0]),
            causal_evidence_target=torch.tensor([1.0]),
            jepa_weight=0.0,
            aux_weight=0.0,
            logical_evidence_weight=1.0,
            causal_evidence_gate_weight=1.0,
        )

        self.assertIn("logical_evidence", metrics)
        self.assertIn("causal_evidence_gate", metrics)
        self.assertGreater(float(metrics["logical_support_prob"]), 0.9)
        self.assertLess(float(metrics["counterfactual_support_prob"]), 0.1)
        self.assertGreater(float(metrics["evidence_gate_mean"]), 0.9)
        self.assertLess(float(metrics["counterfactual_gate_mean"]), 0.1)

    def test_qtrm_smoke_loss_adds_core_world_model_metric(self):
        from qtrm_mm.losses import qtrm_smoke_loss

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type("Cfg", (), {"jepa_sigreg_weight": 0.0})()

            def forward(self, input_ids, **kwargs):
                if "core_world_model_actions" not in kwargs:
                    raise AssertionError("core_world_model_actions should reach the model")
                logits = torch.zeros(1, 3, 5)
                logits[0, 0, 2] = 8.0
                logits[0, 1, 3] = 8.0
                return {
                    "logits": logits,
                    "jepa_pred": torch.ones(1, 2, 4),
                    "jepa_target": torch.zeros(1, 2, 4),
                    "jepa_mask": torch.ones(1, 2, dtype=torch.bool),
                    "jepa_latents": torch.ones(1, 3, 4),
                    "jepa_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "core_world_model_pred": torch.ones(1, 2, 4),
                    "core_world_model_target": torch.zeros(1, 2, 4),
                    "core_world_model_mask": torch.ones(1, 2, dtype=torch.bool),
                    "core_world_model_latents": torch.ones(1, 3, 4),
                    "core_world_model_latent_mask": torch.ones(1, 3, dtype=torch.bool),
                    "halt_logits": torch.ones(1, 1),
                    "action_logits": torch.zeros(1, 3),
                }

        input_ids = torch.tensor([[1, 2, 3]])
        lm_only, _, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            core_world_model_weight=0.0,
            core_world_model_actions=torch.zeros(1, 3, 10),
        )
        weighted, metrics, _ = qtrm_smoke_loss(
            FakeModel(),
            input_ids,
            jepa_weight=0.0,
            aux_weight=0.0,
            core_world_model_weight=1.0,
            core_world_model_actions=torch.zeros(1, 3, 10),
        )

        self.assertIn("core_world_model", metrics)
        self.assertGreater(float(metrics["core_world_model"]), 0.9)
        self.assertGreater(float(weighted), float(lm_only))


if __name__ == "__main__":
    unittest.main()
