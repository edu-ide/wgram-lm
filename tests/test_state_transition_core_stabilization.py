import torch

from wgram_lm.config import QTRMConfig
from wgram_lm.state_transition_core import (
    HybridStateTransitionCore,
    MiniGatedDeltaReasoningCore,
    SharedReasoningCore,
    StateTransitionCore,
)


def test_shared_reasoning_core_layerscale_controls_transition_delta():
    core = SharedReasoningCore(d_state=4, n_operations=4, transition_scale_init=0.25)

    assert core.transition_scale.requires_grad
    assert torch.allclose(core.transition_scale.detach(), torch.tensor(0.25))


def test_mini_gated_delta_reasoning_core_preserves_shared_core_contract():
    core = MiniGatedDeltaReasoningCore(d_state=4, n_operations=4, transition_scale_init=0.25)
    z_main = torch.randn(2, 4)
    z_side = torch.randn(2, 4)
    op_vec = torch.randn(2, 4)

    out = core(z_main, z_side, op_vec)

    assert tuple(out.shape) == tuple(z_main.shape)
    assert core.transition_scale.requires_grad
    assert torch.allclose(core.transition_scale.detach(), torch.tensor(0.25))


def test_state_transition_core_initializes_small_step_embeddings():
    core = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        step_embedding_std=0.01,
    )

    assert core.step_embed.weight.detach().abs().max().item() < 0.05


def test_state_transition_core_forwards_stabilization_options_to_shared_core():
    core = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        transition_scale_init=0.125,
    )

    assert torch.allclose(core.shared_core.transition_scale.detach(), torch.tensor(0.125))


def test_state_transition_core_runs_with_mini_gated_delta_update():
    core = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        core_update="mini_gated_delta",
    )
    workspace = torch.randn(2, 1, 8)
    operation_ids = torch.randint(0, 4, (2, 4))

    out = core(workspace=workspace, operation_ids=operation_ids)

    assert isinstance(core.shared_core, MiniGatedDeltaReasoningCore)
    assert out.state_trajectory.shape == (2, 5, 8)
    assert out.answer_logits.shape == (2, 10)


def test_state_transition_core_runs_with_workspace_cross_attention_mask():
    core = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        workspace_cross_attention=True,
        workspace_cross_attention_heads=4,
    )
    workspace = torch.randn(2, 5, 8)
    workspace_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    operation_ids = torch.randint(0, 4, (2, 4))

    out = core(
        workspace=workspace,
        workspace_attention_mask=workspace_mask,
        operation_ids=operation_ids,
    )

    assert out.state_trajectory.shape == (2, 5, 8)
    assert out.answer_logits.shape == (2, 10)


def test_state_transition_core_runs_with_oracle_operation_args():
    core = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        operation_arg_conditioning=True,
    )
    workspace = torch.randn(2, 1, 8)
    operation_ids = torch.randint(0, 4, (2, 4))
    operation_arg_ids = torch.randint(0, 10, (2, 4))
    initial_labels = torch.randint(0, 10, (2,))

    out = core(
        workspace=workspace,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        initial_labels=initial_labels,
    )

    assert out.state_trajectory.shape == (2, 5, 8)
    assert out.answer_logits.shape == (2, 10)


def test_state_transition_core_runs_true_gram_transition_mode_with_state_labels():
    torch.manual_seed(0)
    core = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        stochastic_high_level_guidance=True,
        stochastic_high_level_scale=1.0,
        stochastic_high_level_max_std=1.5,
        stochastic_posterior_guidance=True,
        stochastic_transition_mode="true_gram",
        operation_arg_conditioning=True,
    )
    workspace = torch.randn(2, 1, 8)
    operation_ids = torch.randint(0, 4, (2, 4))
    operation_arg_ids = torch.randint(0, 10, (2, 4))
    initial_labels = torch.randint(0, 10, (2,))
    state_labels = torch.randint(0, 10, (2, 4))

    out = core(
        workspace=workspace,
        operation_ids=operation_ids,
        operation_arg_ids=operation_arg_ids,
        initial_labels=initial_labels,
        posterior_labels=state_labels,
    )

    assert out.state_trajectory.shape == (2, 5, 8)
    assert out.stochastic_posterior_kls is not None
    assert out.stochastic_posterior_kls.shape == (4,)
    assert out.stochastic_std_means is not None
    assert out.stochastic_std_means.mean().item() > 0.0


def test_state_transition_core_two_stream_schedule_runs_with_same_shapes():
    torch.manual_seed(0)
    nested = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        update_schedule="nested",
    )
    two_stream = StateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        update_schedule="two_stream",
    )
    two_stream.load_state_dict(nested.state_dict())
    workspace = torch.randn(2, 1, 8)
    operation_ids = torch.randint(0, 4, (2, 4))

    nested_out = nested(workspace=workspace, operation_ids=operation_ids)
    two_stream_out = two_stream(workspace=workspace, operation_ids=operation_ids)

    assert nested_out.state_trajectory.shape == two_stream_out.state_trajectory.shape == (2, 5, 8)
    assert nested_out.answer_logits.shape == two_stream_out.answer_logits.shape == (2, 10)
    assert not torch.allclose(nested_out.state_trajectory, two_stream_out.state_trajectory)


def test_hybrid_state_transition_core_runs_with_delta_attention_layout():
    core = HybridStateTransitionCore(
        QTRMConfig(d_model=8, num_actions=4, outer_steps=4),
        d_state=8,
        n_operations=4,
        n_steps=4,
        transition_scale_init=0.25,
    )
    workspace = torch.randn(2, 1, 8)
    operation_ids = torch.randint(0, 4, (2, 4))

    out = core(workspace=workspace, operation_ids=operation_ids)

    assert out.state_trajectory.shape == (2, 5, 8)
    assert out.answer_logits.shape == (2, 10)
    assert out.operation_logits.shape == (2, 4, 4)
