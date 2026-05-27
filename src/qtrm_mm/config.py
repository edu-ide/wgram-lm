from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class QTRMConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 768
    max_seq_len: int = 512
    n_prelude_layers: int = 2
    n_core_layers: int = 2
    n_coda_layers: int = 2
    core_enabled: bool = True
    core_causal: bool = False
    attn_every: int = 4
    coda_attn_every: Optional[int] = None
    workspace_tokens: int = 32
    workspace_layers: int = 1
    workspace_ff_mult: int = 0
    workspace_include_latents_in_kv: bool = False
    workspace_memory_gate_enabled: bool = False
    workspace_memory_gate_init_bias: float = -2.0
    h_cycles: int = 2
    l_cycles: int = 2
    outer_steps: int = 2
    dropout: float = 0.0
    rope_theta: float = 10000.0
    delta_backend: str = "torch_gated_delta"
    delta_head_dim: Optional[int] = None
    delta_num_v_heads: Optional[int] = None
    delta_expand_v: float = 1.0
    delta_mode: str = "chunk"
    delta_use_short_conv: bool = True
    delta_conv_size: int = 4
    delta_norm_eps: float = 1e-6
    attention_backend: str = "sdpa"
    strict_backends: bool = False
    visual_dim: int = 512
    max_visual_tokens: int = 64
    temporal_spatial_context_enabled: bool = False
    temporal_spatial_context_dim: int = 8
    temporal_spatial_context_max_tokens: int = 4
    num_actions: int = 10
    tie_embeddings: bool = True
    use_stable_inject: bool = True
    truncated_recurrence: bool = False
    text_position_embed_enabled: bool = False
    token_numeric_value_embedding_enabled: bool = False
    token_numeric_value_vocab_size: int = 128
    token_numeric_value_gate_init_bias: float = -4.0
    token_numeric_value_gate_min: float = 0.0
    token_numeric_source_slot_embedding_enabled: bool = False
    token_numeric_source_slot_vocab_size: int = 128
    token_numeric_source_slot_max_slots: int = 5
    token_numeric_source_slot_gate_init_bias: float = -4.0
    token_numeric_source_slot_gate_min: float = 0.0
    token_numeric_source_slot_predicate_feedback_enabled: bool = False
    token_numeric_source_slot_predicate_gate_init_bias: float = -4.0
    token_numeric_source_slot_predicate_gate_min: float = 0.0
    core_step_conditioning_enabled: bool = False
    core_step_conditioning_max_steps: int = 16
    core_step_conditioning_scale: float = 1.0
    core_context_enabled: bool = False
    core_context_gate_init_bias: float = -2.0
    core_transition_feedback_enabled: bool = False
    core_transition_feedback_num_operations: int = 12
    core_transition_feedback_scale: float = 1.0
    core_transition_feedback_gate_init_bias: float = 0.0
    core_transition_order_bottleneck_enabled: bool = False
    core_transition_order_bottleneck_num_classes: int = 2
    core_transition_order_bottleneck_hidden_dim: Optional[int] = None
    core_transition_order_bottleneck_gate_init_bias: float = -2.0
    core_transition_order_step_conditioning_enabled: bool = False
    core_transition_order_step_conditioning_scale: float = 1.0
    core_transition_order_step_conditioning_gate_init_bias: float = 0.0
    core_state_carry_enabled: bool = False
    core_state_carry_hidden_dim: Optional[int] = None
    core_state_carry_gate_init_bias: float = -4.0
    core_state_carry_gate_min: float = 0.0
    core_to_text_enabled: bool = False
    core_to_text_gate_init_bias: float = -2.0
    core_to_text_gate_min: float = 0.0
    core_output_blend_enabled: bool = False
    core_output_blend_init_bias: float = -4.0
    core_output_blend_min: float = 0.0
    core_halt_enabled: bool = False
    core_halt_min_steps: int = 1
    core_halt_use_continue: bool = False
    core_halt_init_bias: float = -5.0
    core_halt_freeze_halted_state_enabled: bool = False
    core_halt_exploration_prob: float = 0.0
    core_halt_exploration_min_steps: int = 2
    core_convergence_halt_enabled: bool = False
    core_convergence_halt_threshold: float = 1e-3
    core_convergence_halt_min_outer: int = 1
    core_trm_no_grad_inner_cycles_enabled: bool = False
    core_depth_readout_enabled: bool = False
    jepa_encoder_layers: int = 1
    jepa_predictor_layers: int = 2
    jepa_predictor_dim: Optional[int] = None
    jepa_horizon: int = 1
    jepa_sigreg_weight: float = 0.09
    jepa_sigreg_knots: int = 17
    jepa_sigreg_num_proj: int = 1024
    core_world_model_enabled: bool = False
    core_world_model_predictor_layers: int = 1
    core_world_model_predictor_dim: Optional[int] = None
    core_world_model_horizon: int = 1
    core_world_model_sigreg_weight: float = 0.09
    # Minimal isolated memory tiers scaffolding (Option 2 ablation track)
    core_memory_tiers_enabled: bool = False
    core_memory_manager_hidden_dim: Optional[int] = None
    core_memory_manager_num_actions: int = 8
    core_memory_tiers_ablation_zero: bool = False

    # Phase 1: Gated multi-domain Thought Workspaces + Broadcast (뇌량)
    core_thought_workspace_enabled: bool = False
    core_thought_workspace_domains: list[str] = field(default_factory=lambda: ["equation", "algorithm_step"])
    core_thought_workspace_hidden_dim: Optional[int] = None
    core_thought_workspace_injection_alpha: float = 0.35
    core_thought_workspace_ablation_zero: bool = False
    core_thought_workspace_selector_mode: str = "sum"  # "sum", "importance", "learned"

    # Next experimental track (I-stage start): Native equation binding + full thought workspaces + hierarchical memory tiers
    # (revival of stashed "new thought structure" from pre-new-thought-structure tag, +154 line diff)
    core_equation_binding_enabled: bool = False
    core_equation_binding_hidden_dim: Optional[int] = None
    core_equation_binding_num_fields: int = 8
    core_equation_binding_gate_init_bias: float = -4.0
    core_equation_binding_ablation_zero: bool = False

    # LeWM predictive memory tier (full port as native answer-causal predictive working memory)
    core_lewm_enabled: bool = False
    core_lewm_predictor_dim: Optional[int] = None
    core_lewm_horizon: int = 1
    core_lewm_ablation_zero: bool = False

    # Phase 2 groundwork: Answer Attractor pressure (depth-wise monotonic)
    core_answer_attractor_enabled: bool = False
    core_answer_attractor_weight: float = 0.02
    core_answer_attractor_monotonic_gain: float = 0.03
    core_answer_attractor_ablation_zero: bool = False

    # Phase 3 groundwork: Provenance / Graph reasoning register input
    core_provenance_register_enabled: bool = False
    core_provenance_register_dim: int = 64
    core_provenance_register_fusion_alpha: float = 0.25
    core_provenance_register_ablation_zero: bool = False

    qtrm_logits_scale: float = 1.0
    donor_logits_scale: float = 0.0
    qtrm_residual_clamp: Optional[float] = None
    qtrm_residual_gate_enabled: bool = False
    qtrm_residual_gate_init_bias: float = -2.0
    qtrm_residual_gate_normalize: bool = True
    qtrm_residual_gate_min: float = 0.0
    donor_qtrm_conflict_gate_enabled: bool = False
    donor_qtrm_conflict_qtrm_scale: float = 0.0
    evidence_bottleneck_enabled: bool = False
    evidence_bottleneck_applies_to_residual: bool = True
    evidence_bottleneck_gate_init_bias: float = -2.0
    evidence_bottleneck_gate_min: float = 0.0
    evidence_bottleneck_suppress_without_workspace: bool = True
    generation_verifier_enabled: bool = False
    answer_bottleneck_enabled: bool = False
    answer_bottleneck_requires_core: bool = False
    answer_bottleneck_requires_workspace_memory: bool = True
    core_loop_readout_enabled: bool = False
    core_loop_readout_requires_core: bool = True
    answer_state_loop_enabled: bool = False
    answer_state_loop_requires_core: bool = True
    answer_state_loop_core_state_only_enabled: bool = False
    answer_state_loop_gate_init_bias: float = 0.0
    answer_state_loop_gate_min: float = 0.0
    answer_state_loop_recurrent_block_enabled: bool = False
    answer_state_loop_recurrent_layers: int = 1
    answer_state_loop_recurrent_gate_init_bias: float = 0.0
    answer_state_loop_recurrent_gate_min: float = 0.0
    answer_state_loop_mythos_update_enabled: bool = False
    answer_state_loop_mythos_log_dt_init: float = -4.0
    answer_state_loop_mythos_input_injection_init: float = 0.02
    answer_state_loop_mythos_loop_index_enabled: bool = False
    answer_state_loop_mythos_loop_dim: Optional[int] = None
    answer_state_loop_mythos_lora_rank: int = 0
    answer_state_loop_mythos_act_enabled: bool = False
    answer_state_loop_mythos_act_threshold: float = 0.99
    answer_state_loop_selective_context_enabled: bool = False
    answer_state_loop_selective_context_top_k: int = 0
    answer_state_loop_halt_enabled: bool = False
    answer_state_loop_halt_init_bias: float = -4.0
    answer_state_loop_halt_gate_enabled: bool = False
    answer_state_loop_halt_gate_temperature: float = 1.0
    answer_state_loop_halt_gate_mode: str = "soft"
    answer_state_loop_lm_adapter_enabled: bool = False
    answer_state_loop_lm_adapter_rank: int = 16
    answer_state_loop_lm_adapter_scale: float = 1.0
    answer_state_loop_hidden_bridge_enabled: bool = False
    answer_state_loop_hidden_bridge_hidden_dim: Optional[int] = None
    answer_state_loop_hidden_bridge_scale: float = 1.0
    answer_state_loop_next_token_decoder_enabled: bool = False
    answer_state_loop_next_token_decoder_layers: int = 1
    answer_state_loop_next_token_decoder_gate_init_bias: float = 0.0
    answer_state_loop_next_token_decoder_gate_min: float = 0.0
    answer_state_loop_next_token_decoder_prev_token_enabled: bool = False
    answer_state_loop_next_token_decoder_prev_token_gate_init_bias: float = 0.0
    answer_state_loop_next_token_decoder_prev_token_gate_min: float = 0.0
    answer_state_loop_free_transformer_latent_enabled: bool = False
    answer_state_loop_free_transformer_latent_dim: Optional[int] = None
    answer_state_loop_free_transformer_gate_init_bias: float = 0.0
    answer_state_loop_free_transformer_gate_min: float = 1.0
    answer_state_loop_free_transformer_posterior_train_enabled: bool = True
    answer_state_loop_free_transformer_free_bits: float = 0.05
    answer_state_loop_future_token_decoder_enabled: bool = False
    answer_state_loop_future_token_max_tokens: int = 8
    answer_state_loop_future_token_position_scale: float = 1.0
    answer_state_loop_talker_enabled: bool = False
    answer_state_loop_talker_layers: int = 1
    answer_state_loop_talker_gate_init_bias: float = -4.0
    answer_state_loop_talker_gate_min: float = 0.0
    answer_state_loop_finality_selector_enabled: bool = False
    answer_state_loop_finality_selector_temperature: float = 1.0
    answer_state_loop_finality_selector_mode: str = "soft"
    answer_state_loop_finality_gate_enabled: bool = False
    answer_state_loop_finality_gate_temperature: float = 1.0
    answer_state_loop_finality_gate_mode: str = "soft"
    transition_state_enabled: bool = False
    transition_state_dim: int = 4
    transition_state_hidden_dim: Optional[int] = None
    transition_state_answer_gate_init_bias: float = 0.0
    transition_state_answer_gate_min: float = 0.0
    transition_state_code_enabled: bool = False
    transition_state_codebook_size: int = 128
    transition_state_code_only_answer_loop: bool = False
    transition_state_finality_enabled: bool = False
    transition_state_joint_enabled: bool = False
    transition_state_joint_size: int = 10
    transition_state_joint_prompt_context_enabled: bool = False
    transition_state_joint_prompt_token_attention_enabled: bool = False
    transition_state_joint_prompt_context_scale: float = 1.0
    transition_state_joint_operation_residual_enabled: bool = False
    transition_state_joint_operation_residual_scale: float = 1.0
    transition_state_joint_code_residual_enabled: bool = False
    transition_state_joint_code_residual_scale: float = 1.0
    transition_phase_enabled: bool = False
    transition_phase_num_classes: int = 2
    transition_phase_hidden_dim: Optional[int] = None
    transition_phase_prompt_context_enabled: bool = False
    transition_phase_prompt_token_attention_enabled: bool = False
    transition_phase_global_prompt_query_enabled: bool = False
    transition_state_joint_phase_residual_enabled: bool = False
    transition_state_joint_phase_residual_scale: float = 1.0
    transition_state_joint_phase_residual_centered: bool = False
    transition_state_joint_phase_reference_class: int = 0
    transition_state_joint_phase_residual_gated_by_nonreference: bool = False
    transition_state_joint_phase_residual_detach_gate: bool = True
    transition_state_joint_answer_bridge_enabled: bool = False
    transition_state_joint_answer_gate_init_bias: float = 0.0
    transition_state_joint_answer_gate_min: float = 0.0
    transition_state_final_answer_binder_enabled: bool = False
    transition_state_final_answer_temperature: float = 1.0
    transition_state_final_answer_gate_init_bias: float = 0.0
    transition_state_final_answer_gate_min: float = 0.0
    transition_state_sequence_enabled: bool = False
    transition_state_sequence_max_tokens: int = 8
    transition_value_state_enabled: bool = False
    transition_value_state_max_tokens: int = 16
    transition_value_state_vocab_size: int = 128
    factorized_value_state_enabled: bool = False
    factorized_value_state_max_tokens: int = 16
    factorized_value_state_vocab_size: int = 128
    factorized_value_state_kind_size: int = 0
    factorized_value_state_hidden_dim: Optional[int] = None
    role_value_state_enabled: bool = False
    role_value_state_num_roles: int = 0
    role_value_state_vocab_size: int = 0
    core_role_value_state_enabled: bool = False
    core_role_value_state_num_roles: int = 0
    core_role_value_state_vocab_size: int = 0
    core_role_value_state_answer_bridge_enabled: bool = False
    core_role_value_state_answer_bridge_gate_init_bias: float = 0.0
    core_role_value_state_answer_bridge_gate_min: float = 0.0
    core_role_value_state_answer_prompt_context_enabled: bool = False
    core_role_value_state_answer_prompt_gate_init_bias: float = 0.0
    core_role_value_state_answer_prompt_gate_min: float = 0.0
    core_role_value_state_answer_final_binder_enabled: bool = False
    core_role_value_state_answer_final_gate_init_bias: float = 0.0
    core_role_value_state_answer_final_gate_min: float = 0.0
    core_role_value_state_vocab_renderer_enabled: bool = False
    core_role_value_state_vocab_renderer_gate_init_bias: float = 0.0
    core_role_value_state_vocab_renderer_gate_min: float = 0.0
    core_role_value_state_vocab_renderer_rank: int = 16
    core_role_value_state_vocab_renderer_scale: float = 1.0
    core_role_value_state_vocab_renderer_transition_context_enabled: bool = False
    core_role_value_state_vocab_renderer_source_state_tokens_enabled: bool = False
    core_role_value_state_vocab_renderer_use_lm_head: bool = False
    core_role_value_state_vocab_renderer_replace_residual_enabled: bool = False
    core_role_value_state_vocab_renderer_candidate_token_ids: list[int] = field(
        default_factory=list
    )
    core_role_value_state_vocab_renderer_source_copy_enabled: bool = False
    core_role_value_state_vocab_renderer_source_copy_from_primitive_enabled: bool = False
    core_role_value_state_vocab_renderer_source_copy_span_enabled: bool = False
    core_role_value_state_vocab_renderer_source_copy_span_max_pieces: int = 8
    core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled: bool = False
    core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias: float = 0.0
    core_role_value_state_vocab_renderer_source_copy_answer_role_separator_token_ids: list[int] = field(
        default_factory=lambda: [11]
    )
    core_role_value_state_vocab_renderer_source_copy_cursor_enabled: bool = False
    core_role_value_state_vocab_renderer_source_copy_cursor_bias: float = 0.0
    core_role_value_state_prompt_extract_enabled: bool = False
    core_role_value_state_prompt_extract_gate_init_bias: float = -4.0
    core_source_position_binder_enabled: bool = False
    core_source_position_binder_hidden_dim: Optional[int] = None
    core_source_position_binder_layers: int = 1
    core_source_position_binder_heads: Optional[int] = None
    core_source_position_binder_max_positions: Optional[int] = None
    core_source_position_binder_gate_init_bias: float = -4.0
    core_source_position_binder_gate_min: float = 0.0
    core_source_position_binder_state_gate_init_bias: float = -4.0
    core_source_position_binder_state_gate_min: float = 0.0
    core_source_position_binder_state_straight_through: bool = False
    core_source_position_binder_source_slots_only: bool = False
    core_source_position_binder_raw_source_slots_enabled: bool = False
    core_source_position_binder_query_state_enabled: bool = False
    core_source_position_binder_query_state_gate_init_bias: float = -4.0
    core_source_position_binder_query_state_gate_min: float = 0.0
    core_source_value_binder_enabled: bool = False
    core_source_value_binder_gate_init_bias: float = -4.0
    core_source_value_binder_gate_min: float = 0.0
    core_source_value_binder_state_gate_init_bias: float = -4.0
    core_source_value_binder_state_gate_min: float = 0.0
    core_source_value_binder_state_straight_through: bool = False
    core_role_value_state_prompt_self_condition_enabled: bool = False
    core_role_value_state_prompt_self_condition_gate_init_bias: float = -4.0
    core_role_value_state_prompt_self_condition_gate_min: float = 0.0
    core_role_value_state_prompt_parity_enabled: bool = False
    core_role_value_state_prompt_parity_gate_init_bias: float = -2.0
    core_role_value_template_codec_enabled: bool = False
    core_role_value_template_num_templates: int = 0
    core_role_value_template_max_steps: int = 8
    core_role_value_template_hidden_dim: Optional[int] = None
    core_role_value_template_factorized_enabled: bool = False
    core_role_value_template_length_classes: int = 5
    core_role_value_template_parity_classes: int = 2
    core_role_value_template_offset_classes: int = 7
    core_role_value_transition_enabled: bool = False
    core_role_value_transition_hidden_dim: Optional[int] = None
    core_role_value_delta_enabled: bool = False
    core_role_value_delta_hidden_dim: Optional[int] = None
    core_role_value_delta_gate_init_bias: float = -4.0
    core_role_value_delta_gate_min: float = 0.0
    core_value_delta_code_enabled: bool = False
    core_value_delta_codebook_size: int = 0
    core_value_delta_code_gate_init_bias: float = -4.0
    core_value_delta_code_gate_min: float = 0.0
    typed_algorithmic_value_state_enabled: bool = False
    typed_algorithmic_value_state_recurrent_enabled: bool = False
    typed_algorithmic_value_state_recurrent_hidden_dim: Optional[int] = None
    typed_algorithmic_value_state_recurrent_gate_init_bias: float = -4.0
    typed_algorithmic_value_state_recurrent_gate_min: float = 0.0
    typed_algorithmic_value_state_primitive_conditioning_enabled: bool = False
    typed_algorithmic_value_state_subregisters_enabled: bool = False
    typed_algorithmic_value_state_residual_feedback_enabled: bool = False
    typed_algorithmic_value_state_residual_delta_enabled: bool = False
    typed_algorithmic_value_state_scalar_offset_enabled: bool = False
    typed_algorithmic_value_state_scalar_regression_enabled: bool = False
    typed_algorithmic_value_state_prompt_context_enabled: bool = False
    typed_algorithmic_value_state_prompt_gate_init_bias: float = -2.0
    typed_algorithmic_value_state_prompt_gate_min: float = 0.0
    typed_algorithmic_value_state_answer_bridge_enabled: bool = False
    typed_algorithmic_value_state_answer_bridge_gate_init_bias: float = 0.0
    typed_algorithmic_value_state_answer_bridge_gate_min: float = 0.0
    typed_algorithmic_value_state_max_list_slots: int = 4
    typed_algorithmic_value_state_scalar_vocab_size: int = 128
    typed_algorithmic_value_state_offset_vocab_size: int = 128
    typed_algorithmic_value_state_kind_size: int = 8
    core_typed_register_executor_enabled: bool = False
    core_typed_register_num_operations: int = 0
    core_typed_register_hidden_dim: Optional[int] = None
    core_typed_register_gate_init_bias: float = -4.0
    core_typed_register_gate_min: float = 0.0
    core_typed_register_transition_readout_enabled: bool = False
    core_typed_register_prompt_first_transition_readout_enabled: bool = False
    core_typed_register_value_feedback_enabled: bool = False
    core_typed_register_value_feedback_gate_init_bias: float = -2.0
    core_typed_register_value_feedback_gate_min: float = 0.0
    primitive_transition_enabled: bool = False
    primitive_transition_num_operations: int = 0
    primitive_transition_hidden_dim: Optional[int] = None
    primitive_transition_prompt_context_enabled: bool = False
    primitive_transition_prompt_token_attention_enabled: bool = False
    core_primitive_role_value_executor_enabled: bool = False
    core_primitive_role_value_mlp_enabled: bool = False
    core_primitive_role_value_hidden_dim: Optional[int] = None
    core_primitive_role_value_role_mixer_enabled: bool = False
    core_primitive_role_value_role_mixer_heads: Optional[int] = None
    core_primitive_role_value_prompt_context_enabled: bool = False
    core_primitive_role_value_prompt_token_attention_enabled: bool = False
    core_primitive_role_value_update_gate_enabled: bool = False
    core_primitive_role_value_update_gate_init_bias: float = -2.0
    core_primitive_role_value_update_gate_min: float = 0.0
    core_primitive_role_value_residual_delta_enabled: bool = False
    core_primitive_role_value_field_specific_heads_enabled: bool = False
    core_primitive_role_value_operation_specific_heads_enabled: bool = False
    core_primitive_role_value_source_value_conditioning_enabled: bool = False
    core_primitive_role_value_source_value_gate_init_bias: float = -4.0
    core_primitive_role_value_source_value_gate_min: float = 0.0
    core_primitive_typed_selector_enabled: bool = False
    core_primitive_typed_selector_init_bias: float = -4.0
    transition_source_router_enabled: bool = False
    transition_source_router_hidden_dim: Optional[int] = None
    transition_source_router_prompt_context_enabled: bool = False
    transition_source_router_prompt_token_attention_enabled: bool = False
    answer_residual_governor_enabled: bool = False
    answer_residual_governor_init_bias: float = -2.0
    answer_residual_governor_min: float = 0.0
    evidence_span_reader_enabled: bool = False
    answer_decision_head_enabled: bool = False
    answer_decision_feature_dim: int = 0
    answer_decision_feature_hidden_dim: int = 32
    controller_signal_enabled: bool = False
    controller_signal_dim: int = 2
    controller_signal_source: str = "external"
    controller_signal_hidden_dim: int = 0
    controller_signal_base_scale: float = 1.0
    qtrm_full_msa_fork: bool = False
    qtrm_original_layer_types: list[str] = field(default_factory=list)
    msa_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DonorConfig:
    model_id: Optional[str] = None
    load_in_4bit: bool = False
    freeze_donor: bool = True
    train_lora: bool = False
    train_last_n_layers: int = 0
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    trust_remote_code: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 4
    seq_len: int = 128
    steps: int = 50
    lr: float = 3e-4
    loss_lm_weight: float = 1.0
    loss_jepa_weight: float = 0.1
    loss_aux_weight: float = 1.0
    loss_core_halt_weight: float = 0.0
    loss_student_lm_weight: float = 0.0
    loss_donor_kl_weight: float = 0.0
    loss_repeat_unlikelihood_weight: float = 0.0
    loss_greedy_token_margin_weight: float = 0.0
    loss_donor_correct_margin_weight: float = 0.0
    loss_preference_weight: float = 0.0
    loss_workspace_contrastive_weight: float = 0.0
    loss_logical_evidence_weight: float = 0.0
    loss_causal_evidence_gate_weight: float = 0.0
    loss_core_world_model_weight: float = 0.0
    loss_generation_verifier_weight: float = 0.0
    loss_evidence_span_reader_weight: float = 0.0
    loss_evidence_span_no_answer_span_suppression_weight: float = 0.0
    loss_answer_decision_weight: float = 0.0
    loss_answer_residual_governor_weight: float = 0.0
    loss_canonical_causal_weight: float = 0.0
    loss_action_policy_weight: float = 0.0
    loss_controller_signal_weight: float = 0.0
    loss_core_trajectory_shortcut_weight: float = 0.0
    loss_core_variable_trajectory_weight: float = 0.0
    loss_core_depth_text_ce_weight: float = 0.0
    loss_donor_lm_weight: float = 0.0
    controller_signal_loss_mode: str = "bce"
    core_trajectory_shortcut_min_step: int = 1
    core_variable_trajectory_short_steps: int = 1
    core_variable_trajectory_short_lm_weight: float = 1.0
    core_variable_trajectory_preference_weight: float = 0.0
    core_variable_trajectory_preference_margin: float = 0.0
    core_depth_text_ce_min_step: int = 1
    preference_beta: float = 2.0
    preference_margin: float = 0.0
    greedy_token_margin: float = 0.0
    greedy_token_margin_only_donor_errors: bool = False
    donor_correct_margin: float = 0.0
    workspace_contrastive_beta: float = 2.0
    workspace_contrastive_margin: float = 0.0
    canonical_causal_beta: float = 2.0
    canonical_causal_margin: float = 0.0
    canonical_causal_ablation_modes: list[str] = field(default_factory=list)
    donor_kl_beta: float = 0.0
    donor_kl_temperature: float = 1.0
    core_halt_auto_targets: bool = False
    core_halt_target_mode: str = "exact"
    core_halt_loss_mode: str = "bce"
    core_halt_q_value_gamma: float = 0.99
    core_halt_donor_kl_threshold: Optional[float] = None
    core_halt_teacher_depth_threshold: float = 0.995
    core_halt_teacher_depth_logit_kl_threshold: float = 0.05
    core_halt_teacher_depth_min_step: int = 1
    trainable_param_policy: str = "all"
    workspace_evidence_injection: bool = False
    workspace_evidence_injection_mode: str = "workspace"
    donor_logits_scale_start: Optional[float] = None
    donor_logits_scale_end: Optional[float] = None
    device: str = "auto"
    use_amp: bool = True
    log_every: int = 10
    out_dir: str = "runs/smoke_multimodal"


@dataclass
class FullConfig:
    model: QTRMConfig = field(default_factory=QTRMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    donor: DonorConfig = field(default_factory=DonorConfig)


def _filter_dataclass(cls, data: Dict[str, Any]):
    names = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    return FullConfig(
        model=_filter_dataclass(QTRMConfig, raw.get("model", {})),
        train=_filter_dataclass(TrainConfig, raw.get("train", {})),
        donor=_filter_dataclass(DonorConfig, raw.get("donor", {})),
    )
