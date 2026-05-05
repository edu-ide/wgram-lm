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
    core_step_conditioning_enabled: bool = False
    core_step_conditioning_max_steps: int = 16
    core_step_conditioning_scale: float = 1.0
    core_context_enabled: bool = False
    core_context_gate_init_bias: float = -2.0
    core_to_text_enabled: bool = False
    core_to_text_gate_init_bias: float = -2.0
    core_to_text_gate_min: float = 0.0
    core_output_blend_enabled: bool = False
    core_output_blend_init_bias: float = -4.0
    core_output_blend_min: float = 0.0
    core_halt_enabled: bool = False
    core_halt_min_steps: int = 1
    core_halt_use_continue: bool = False
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
    answer_state_loop_gate_init_bias: float = 0.0
    answer_state_loop_gate_min: float = 0.0
    transition_state_enabled: bool = False
    transition_state_dim: int = 4
    transition_state_hidden_dim: Optional[int] = None
    transition_state_answer_gate_init_bias: float = 0.0
    transition_state_answer_gate_min: float = 0.0
    transition_state_code_enabled: bool = False
    transition_state_codebook_size: int = 128
    transition_state_code_only_answer_loop: bool = False
    transition_state_finality_enabled: bool = False
    primitive_transition_enabled: bool = False
    primitive_transition_num_operations: int = 0
    primitive_transition_hidden_dim: Optional[int] = None
    primitive_transition_prompt_context_enabled: bool = False
    primitive_transition_prompt_token_attention_enabled: bool = False
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
    controller_signal_base_scale: float = 1.0


@dataclass
class DonorConfig:
    model_id: Optional[str] = None
    load_in_4bit: bool = False
    freeze_donor: bool = True
    train_lora: bool = False
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
