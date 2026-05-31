from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WGRAMV2Config:
    """Canonical W-GRAM V2 configuration.

    The defaults describe the research interface.  Tiny smoke tests may set
    ``runtime_profile="smoke"`` and ``allow_torch_smoke_core=True``; promotion
    gates must reject that profile.
    """

    vocab_size: int = 512
    d_model: int = 128
    max_position_embeddings: int = 4096
    max_response_position_embeddings: int = 1024
    tie_input_output_embeddings: bool = False
    patch_size: int = 4
    local_layers: int = 2
    local_heads: int = 4
    core_layers: int = 2
    dropout: float = 0.0
    runtime_profile: str = "promotion"
    delta_backend: str = "official_gated_delta2"
    core_implementation: str = "torch_smoke"
    allow_torch_smoke_core: bool = False
    official_gdn2_head_dim: int = 0
    official_gdn2_num_v_heads: int = 0
    official_gdn2_expand_v: float = 1.0
    official_gdn2_mode: str = "chunk"
    official_gdn2_use_short_conv: bool = True
    official_gdn2_force_chunk_eval: bool = True
    official_gdn2_conv_size: int = 4
    official_gdn2_norm_eps: float = 1.0e-5
    backbone_ratio: str = "gdn2_attention_3to1"
    core_attention_causal: bool = True
    boundary_state_source: str = "causal_chunk_summary"
    force_fixed_boundaries: bool = False
    dynamic_boundary_threshold: float = 0.6
    boundary_initial_logit: float = -2.0
    imta_trajectories: int = 3
    imta_noise_std: float = 0.0
    imta_selector_temperature: float = 0.8
    imta_adapter_gate_init: float = -1.0
    imta_post_adapter_gate_init: float = -1.0
    imta_selector_route_query_std: float = 0.02
    imta_diversity_weight: float = 0.0
    imta_route_min_probability: float = 0.0
    imta_route_entropy_floor: float = 0.0
    imta_route_entropy_weight: float = 0.0
    imta_route_balance_weight: float = 0.0
    own_latent_prediction_enabled: bool = True
    own_latent_prediction_weight: float = 0.0
    repeat_unlikelihood_weight: float = 0.0
    premature_stop_loss_weight: float = 0.0
    response_start_loss_weight: float = 0.0
    response_start_stop_margin_weight: float = 0.0
    response_start_stop_margin: float = 1.0
    response_continue_stop_margin_weight: float = 0.0
    response_continue_stop_margin: float = 1.0
    response_body_loss_weight: float = 0.0
    response_stop_loss_weight: float = 0.0
    use_response_phase_embeddings: bool = True
    token_maturation_steps: int = 2
    token_maturation_layers: int = 1
    token_maturation_aux_loss_weight: float = 0.0
    token_maturation_gate_init: float = -1.0
    token_maturation_confidence_threshold: float = 0.0
    answer_memory_enabled: bool = True
    answer_memory_steps: int = 2
    answer_memory_plan_tokens: int = 4
    answer_memory_plan_layers: int = 1
    answer_memory_prompt_context_enabled: bool = False
    answer_memory_prompt_context_gate_init: float = -1.0
    answer_memory_aux_loss_weight: float = 0.0
    answer_memory_confidence_gate_enabled: bool = True
    answer_memory_confidence_mode: str = "topk_mass"
    answer_memory_confidence_topk: int = 5
    answer_memory_confidence_floor: float = 0.20
    answer_memory_stop_margin_loss_weight: float = 0.0
    answer_memory_stop_margin: float = 1.0
    answer_memory_commitment_enabled: bool = True
    answer_memory_commitment_scale: float = 1.0
    answer_memory_commitment_confidence_gate_enabled: bool = False
    answer_memory_commitment_gate_init: float = -0.5
    answer_prefix_commitment_loss_weight: float = 0.0
    answer_memory_update_gate_init: float = -1.0
    answer_memory_injection_gate_init: float = -1.5
    answer_memory_default_injection_scale: float = 1.0
    adaptive_latent_bridge_enabled: bool = True
    adaptive_latent_bridge_gate_init: float = -2.0
    byte_residual_gate_init: float = -2.0
    latent_residual_gate_init: float = 2.0
    stability_activation_clip_value: float = 30.0
    answer_head_count: int = 1
    evaluation_policy: str = "free_generation_only"
    forced_choice_promotion_enabled: bool = False
    candidate_rerank_promotion_enabled: bool = False
    external_gram_ptrm_answer_selection: bool = False
    lewm_answer_path_enabled: bool = False

    def __post_init__(self) -> None:
        if int(self.vocab_size) <= 2:
            raise ValueError("vocab_size must be > 2")
        if int(self.d_model) <= 0:
            raise ValueError("d_model must be positive")
        if int(self.max_position_embeddings) <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if int(self.max_response_position_embeddings) <= 0:
            raise ValueError("max_response_position_embeddings must be positive")
        if int(self.patch_size) <= 0:
            raise ValueError("patch_size must be positive")
        if not 0.0 < float(self.dynamic_boundary_threshold) < 1.0:
            raise ValueError("dynamic_boundary_threshold must be between 0 and 1")
        if int(self.local_heads) <= 0 or int(self.d_model) % int(self.local_heads) != 0:
            raise ValueError("d_model must be divisible by local_heads")
        if int(self.imta_trajectories) < 1:
            raise ValueError("imta_trajectories must be >= 1")
        if float(self.imta_selector_temperature) <= 0.0:
            raise ValueError("imta_selector_temperature must be > 0")
        if float(self.imta_noise_std) < 0.0:
            raise ValueError("imta_noise_std must be >= 0")
        if float(self.imta_diversity_weight) < 0.0:
            raise ValueError("imta_diversity_weight must be >= 0")
        if float(self.imta_selector_route_query_std) < 0.0:
            raise ValueError("imta_selector_route_query_std must be >= 0")
        if float(self.imta_route_min_probability) < 0.0:
            raise ValueError("imta_route_min_probability must be >= 0")
        if (
            int(self.imta_trajectories) > 0
            and float(self.imta_route_min_probability) >= 1.0 / float(self.imta_trajectories)
        ):
            raise ValueError("imta_route_min_probability must be < 1 / imta_trajectories")
        if float(self.imta_route_entropy_floor) < 0.0:
            raise ValueError("imta_route_entropy_floor must be >= 0")
        if float(self.imta_route_entropy_weight) < 0.0:
            raise ValueError("imta_route_entropy_weight must be >= 0")
        if float(self.imta_route_balance_weight) < 0.0:
            raise ValueError("imta_route_balance_weight must be >= 0")
        if float(self.own_latent_prediction_weight) < 0.0:
            raise ValueError("own_latent_prediction_weight must be >= 0")
        if float(self.repeat_unlikelihood_weight) < 0.0:
            raise ValueError("repeat_unlikelihood_weight must be >= 0")
        if float(self.premature_stop_loss_weight) < 0.0:
            raise ValueError("premature_stop_loss_weight must be >= 0")
        if float(self.response_start_loss_weight) < 0.0:
            raise ValueError("response_start_loss_weight must be >= 0")
        if float(self.response_start_stop_margin_weight) < 0.0:
            raise ValueError("response_start_stop_margin_weight must be >= 0")
        if float(self.response_start_stop_margin) < 0.0:
            raise ValueError("response_start_stop_margin must be >= 0")
        if float(self.response_continue_stop_margin_weight) < 0.0:
            raise ValueError("response_continue_stop_margin_weight must be >= 0")
        if float(self.response_continue_stop_margin) < 0.0:
            raise ValueError("response_continue_stop_margin must be >= 0")
        if float(self.response_body_loss_weight) < 0.0:
            raise ValueError("response_body_loss_weight must be >= 0")
        if float(self.response_stop_loss_weight) < 0.0:
            raise ValueError("response_stop_loss_weight must be >= 0")
        if int(self.token_maturation_steps) < 0:
            raise ValueError("token_maturation_steps must be >= 0")
        if int(self.token_maturation_layers) < 1:
            raise ValueError("token_maturation_layers must be >= 1")
        if float(self.token_maturation_aux_loss_weight) < 0.0:
            raise ValueError("token_maturation_aux_loss_weight must be >= 0")
        if float(self.token_maturation_confidence_threshold) < 0.0:
            raise ValueError("token_maturation_confidence_threshold must be >= 0")
        if int(self.answer_memory_steps) < 0:
            raise ValueError("answer_memory_steps must be >= 0")
        if int(self.answer_memory_plan_tokens) < 1:
            raise ValueError("answer_memory_plan_tokens must be >= 1")
        if int(self.answer_memory_plan_layers) < 0:
            raise ValueError("answer_memory_plan_layers must be >= 0")
        if float(self.answer_memory_prompt_context_gate_init) > 10.0:
            raise ValueError("answer_memory_prompt_context_gate_init must be <= 10")
        if float(self.answer_memory_aux_loss_weight) < 0.0:
            raise ValueError("answer_memory_aux_loss_weight must be >= 0")
        if str(self.answer_memory_confidence_mode) not in {
            "top1_probability",
            "topk_mass",
            "entropy_complement",
            "hybrid_topk_entropy",
        }:
            raise ValueError(
                "answer_memory_confidence_mode must be top1_probability, topk_mass, "
                "entropy_complement, or hybrid_topk_entropy"
            )
        if int(self.answer_memory_confidence_topk) < 1:
            raise ValueError("answer_memory_confidence_topk must be >= 1")
        if float(self.answer_memory_confidence_floor) < 0.0:
            raise ValueError("answer_memory_confidence_floor must be >= 0")
        if float(self.answer_memory_stop_margin_loss_weight) < 0.0:
            raise ValueError("answer_memory_stop_margin_loss_weight must be >= 0")
        if float(self.answer_memory_stop_margin) < 0.0:
            raise ValueError("answer_memory_stop_margin must be >= 0")
        if float(self.answer_memory_commitment_scale) < 0.0:
            raise ValueError("answer_memory_commitment_scale must be >= 0")
        if float(self.answer_prefix_commitment_loss_weight) < 0.0:
            raise ValueError("answer_prefix_commitment_loss_weight must be >= 0")
        if float(self.answer_memory_default_injection_scale) < 0.0:
            raise ValueError("answer_memory_default_injection_scale must be >= 0")
        if float(self.stability_activation_clip_value) < 0.0:
            raise ValueError("stability_activation_clip_value must be >= 0")
        if str(self.core_implementation) not in {"torch_smoke", "official_gated_delta2"}:
            raise ValueError("core_implementation must be torch_smoke or official_gated_delta2")
        if int(self.official_gdn2_head_dim) < 0:
            raise ValueError("official_gdn2_head_dim must be >= 0")
        if int(self.official_gdn2_num_v_heads) < 0:
            raise ValueError("official_gdn2_num_v_heads must be >= 0")
        if float(self.official_gdn2_expand_v) <= 0.0:
            raise ValueError("official_gdn2_expand_v must be > 0")
        if int(self.official_gdn2_conv_size) <= 0:
            raise ValueError("official_gdn2_conv_size must be positive")
        if float(self.official_gdn2_norm_eps) <= 0.0:
            raise ValueError("official_gdn2_norm_eps must be > 0")


# Backward-compatible implementation alias.  Older checkpoints, tests, and
# historical scripts may still import QTRMV2Config during the W-GRAM rebrand.
QTRMV2Config = WGRAMV2Config
