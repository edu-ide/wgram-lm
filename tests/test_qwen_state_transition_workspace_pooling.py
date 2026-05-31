import torch
from torch import nn

from wgram_lm.qwen_backbone_state_transition import QwenBackboneStateTransition


class FakeConfig:
    hidden_size = 4
    vocab_size = 16


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.lm_head = nn.Linear(4, 16, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).to(torch.float32)

        class Output:
            pass

        output = Output()
        output.hidden_states = [hidden]
        output.logits = None
        return output


def _identity_compressor(model):
    model.compressor = nn.Identity()


def test_last_workspace_pooling_uses_last_attended_token():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="last",
    )
    _identity_compressor(model)
    hidden = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    workspace = model._compress_workspace(hidden, mask)

    assert workspace.shape == (2, 1, 4)
    assert torch.equal(workspace[:, 0, 0], torch.tensor([2.0, 5.0]))


def test_attention_workspace_pooling_ignores_masked_tokens():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
    )
    _identity_compressor(model)
    with torch.no_grad():
        model.workspace_attention.weight.zero_()
        model.workspace_attention.bias.zero_()

    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0]])

    workspace = model._compress_workspace(hidden, mask)

    assert workspace.shape == (1, 1, 4)
    assert torch.allclose(workspace[0, 0], torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_attention_workspace_pooling_accepts_bfloat16_hidden_states():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
    )
    _identity_compressor(model)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]], dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1]])

    workspace = model._compress_workspace(hidden, mask)

    assert workspace.dtype == torch.bfloat16
    assert workspace.shape == (1, 1, 4)


def test_sequence_workspace_pooling_preserves_unmasked_token_slots():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
    )
    _identity_compressor(model)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0]])

    workspace = model._compress_workspace(hidden, mask)

    assert workspace.shape == (1, 3, 4)
    assert torch.allclose(workspace[0, 0], hidden[0, 0])
    assert torch.allclose(workspace[0, 1], hidden[0, 1])
    assert torch.allclose(workspace[0, 2], torch.zeros(4))


def test_sequence_workspace_runs_cross_attention_core_path():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        n_steps=2,
    )
    _identity_compressor(model)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=2)

    assert out["qtrm_workspace"].shape == (1, 4, 4)
    assert out["qtrm_workspace"][0, 3].abs().sum().item() == 0.0
    assert out["qtrm_core_step_states"].shape == (1, 3, 4)
    assert out["answer_logits"].shape == (1, 10)


def test_source_numeric_feature_slots_are_appended_to_workspace():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        source_numeric_feature_dim=6,
        n_steps=2,
    )
    _identity_compressor(model)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])
    source_features = torch.tensor(
        [[[0.4, 1.0, 0.0, 1.0, 0.0, 0.0], [0.8, 0.0, 1.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    source_mask = torch.tensor([[1, 0]], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        n_steps=2,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )

    assert out["qtrm_workspace"].shape == (1, 6, 4)
    assert out["qtrm_workspace"][0, 4].abs().sum().item() > 0.0
    assert out["qtrm_workspace"][0, 5].abs().sum().item() == 0.0
    assert out["qtrm_workspace_attention_mask"].tolist() == [[1, 1, 1, 0, 1, 0]]


def test_typed_value_registers_join_working_register_trajectory():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        source_numeric_feature_dim=6,
        typed_value_registers=True,
        working_register_enabled=True,
        working_register_slots=2,
        n_steps=2,
    )
    _identity_compressor(model)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])
    source_features = torch.tensor(
        [[[0.4, 1.0, 0.0, 1.0, 0.0, 0.0], [0.8, 0.0, 1.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    source_mask = torch.tensor([[1, 0]], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        n_steps=2,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )

    assert out["qtrm_working_register_trajectory"].shape == (1, 3, 4, 4)
    assert out["qtrm_typed_value_register_trajectory"].shape == (1, 3, 2, 4)
    assert out["qtrm_typed_value_register_trajectory"][0, :, 0].abs().sum().item() > 0.0
    assert out["qtrm_typed_value_register_trajectory"][0, :, 1].abs().sum().item() == 0.0


def test_typed_value_registers_support_gated_delta_update_mode():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        source_numeric_feature_dim=6,
        typed_value_registers=True,
        typed_value_update_mode="gated_delta",
        working_register_enabled=True,
        working_register_slots=2,
        n_steps=2,
    )
    _identity_compressor(model)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])
    source_features = torch.tensor(
        [[[0.4, 1.0, 0.0, 1.0, 0.0, 0.0], [0.8, 0.0, 1.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    source_mask = torch.tensor([[1, 0]], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        n_steps=2,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )

    trajectory = out["qtrm_typed_value_register_trajectory"]
    assert trajectory.shape == (1, 3, 2, 4)
    assert trajectory[0, :, 0].abs().sum().item() > 0.0
    assert trajectory[0, :, 1].abs().sum().item() == 0.0
    assert out["qtrm_typed_value_register_gate_means"].shape == (2,)


def test_typed_digit_registers_join_answer_register_trajectory_with_carry_slot():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        source_numeric_feature_dim=12,
        typed_digit_registers=True,
        typed_digit_register_digits=4,
        working_register_enabled=True,
        working_register_slots=2,
        n_steps=2,
    )
    _identity_compressor(model)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])
    source_features = torch.tensor(
        [
            [
                [0.8017, 1.0, 0.0, 1.0, 8.0, 0.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0],
                [0.0000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    source_mask = torch.tensor([[1, 0]], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        n_steps=2,
        source_numeric_features=source_features,
        source_numeric_feature_mask=source_mask,
    )

    digit_trajectory = out["qtrm_typed_digit_register_trajectory"]
    # Two source slots, four digit columns plus one carry pocket per source slot.
    assert digit_trajectory.shape == (1, 3, 10, 4)
    assert out["qtrm_working_register_trajectory"].shape == (1, 3, 12, 4)
    assert digit_trajectory[0, :, :5].abs().sum().item() > 0.0
    assert digit_trajectory[0, :, 5:].abs().sum().item() == 0.0
    assert out["qtrm_typed_digit_register_gate_means"].shape == (2,)


def test_operation_arg_conditioning_accepts_step_operands():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        operation_arg_conditioning=True,
        n_steps=2,
    )
    _identity_compressor(model)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])
    operation_arg_ids = torch.tensor([[7, 3]])
    initial_labels = torch.tensor([5])

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        initial_labels=initial_labels,
        n_steps=2,
    )

    assert out["qtrm_core_step_states"].shape == (1, 3, 4)
    assert out["answer_logits"].shape == (1, 10)


def test_lm_head_answer_path_reads_qwen_vocab_logits_from_thought_state():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="sequence",
        answer_path="lm_head",
        n_steps=2,
    )
    _identity_compressor(model)
    model.set_label_token_ids(list(range(10)))
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    operation_ids = torch.tensor([[0, 1]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=2)

    assert out["qtrm_answer_path"] == "lm_head"
    assert out["qtrm_lm_answer_logits"].shape == (1, 16)
    assert out["answer_logits"].shape == (1, 10)
    assert torch.allclose(out["answer_logits"], out["qtrm_lm_answer_logits"][:, :10])


def test_recurrent_attention_readout_pools_transition_states_only():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        recurrent_readout_pooling="attention",
    )
    with torch.no_grad():
        model.recurrent_readout_attention.weight.zero_()
        model.recurrent_readout_attention.bias.zero_()

    trajectory = torch.tensor(
        [
            [
                [100.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    readout = model._pool_recurrent_readout(trajectory)

    assert readout.shape == (1, 4)
    assert torch.allclose(readout[0], torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_hybrid_recurrent_readout_starts_between_final_and_attention_states():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        recurrent_readout_pooling="hybrid_gate",
    )
    with torch.no_grad():
        model.recurrent_readout_attention.weight.zero_()
        model.recurrent_readout_attention.bias.zero_()
        model.recurrent_readout_gate.weight.zero_()
        model.recurrent_readout_gate.bias.zero_()

    trajectory = torch.tensor(
        [
            [
                [100.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    readout = model._pool_recurrent_readout(trajectory)
    telemetry = model._recurrent_readout_telemetry(trajectory)

    assert readout.shape == (1, 4)
    assert torch.allclose(readout[0], torch.tensor([2.5, 0.0, 0.0, 0.0]))
    assert torch.allclose(telemetry["qtrm_readout_gate"], torch.tensor([0.5]))


def test_latent_feedback_passes_rerun_core_and_preserve_output_contract():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
        recurrent_readout_pooling="sharp_attention",
        recurrent_readout_temperature=0.5,
        latent_feedback_passes=2,
        n_steps=2,
    )
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=2)

    assert out["qtrm_latent_feedback_passes"] == 2
    assert out["qtrm_core_step_states"].shape == (1, 3, 4)
    assert out["answer_logits"].shape == (1, 10)


def test_correction_feedback_runs_second_pass_and_exposes_error_logits():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
        correction_feedback=True,
        correction_feedback_scale=0.5,
        n_steps=2,
    )
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=2)

    assert out["qtrm_core_step_states"].shape == (1, 3, 4)
    assert out["answer_logits"].shape == (1, 10)
    assert out["qtrm_first_answer_logits"].shape == (1, 10)
    assert out["qtrm_correction_error_logits"].shape == (1, 10)
    assert out["qtrm_correction_gate"].shape == (1,)


def test_stochastic_high_level_guidance_exposes_gram_telemetry():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
        stochastic_high_level_guidance=True,
        stochastic_high_level_scale=0.2,
        n_steps=2,
    )
    model.train()
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=2)

    assert out["qtrm_core_step_states"].shape == (1, 3, 4)
    assert out["qtrm_stochastic_mu_norms"].shape == (2,)
    assert out["qtrm_stochastic_std_means"].shape == (2,)
    assert out["qtrm_stochastic_noise_norms"].shape == (2,)
    assert float(out["qtrm_stochastic_std_means"].mean()) > 0.0


def test_stochastic_posterior_guidance_exposes_kl_for_training():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
        stochastic_high_level_guidance=True,
        stochastic_posterior_guidance=True,
        stochastic_high_level_scale=0.2,
        n_steps=2,
    )
    model.train()
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1]])
    posterior_labels = torch.tensor([3])

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        posterior_labels=posterior_labels,
        n_steps=2,
    )

    assert out["qtrm_stochastic_posterior_kls"].shape == (2,)
    assert torch.isfinite(out["qtrm_stochastic_posterior_kls"]).all()
    assert float(out["qtrm_stochastic_posterior_kls"].mean()) >= 0.0


def test_true_gram_transition_mode_accepts_stepwise_posterior_labels():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
        stochastic_high_level_guidance=True,
        stochastic_posterior_guidance=True,
        stochastic_transition_mode="true_gram",
        stochastic_high_level_max_std=1.5,
        n_steps=2,
    )
    model.train()
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1]])
    posterior_labels = torch.tensor([[3, 4]])

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        operation_ids=operation_ids,
        posterior_labels=posterior_labels,
        n_steps=2,
    )

    assert out["qtrm_core_step_states"].shape == (1, 3, 4)
    assert out["qtrm_stochastic_posterior_kls"].shape == (2,)
    assert torch.isfinite(out["qtrm_stochastic_posterior_kls"]).all()


def test_gram_lprm_reward_head_scores_recurrent_trajectory():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        workspace_pooling="attention",
        stochastic_high_level_guidance=True,
        n_steps=2,
    )
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=2)

    assert out["qtrm_trajectory_reward_logits"].shape == (1,)
    assert torch.isfinite(out["qtrm_trajectory_reward_logits"]).all()


def test_hybrid_state_transition_core_impl():
    model = QwenBackboneStateTransition(
        FakeQwen(),
        d_state=4,
        freeze_qwen=True,
        core_impl="hybrid_state_transition",
        workspace_pooling="attention",
        recurrent_readout_pooling="sharp_attention",
        recurrent_readout_temperature=0.25,
        n_steps=4,
    )
    input_ids = torch.tensor([[0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1]])
    operation_ids = torch.tensor([[0, 1, 2, 3]])

    out = model(input_ids=input_ids, attention_mask=attention_mask, operation_ids=operation_ids, n_steps=4)

    assert out["qtrm_core_step_states"].shape == (1, 5, 4)
    assert out["answer_logits"].shape == (1, 10)
    assert out["logits"].shape == (1, 10)
