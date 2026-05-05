from __future__ import annotations
from typing import Optional
import torch
from torch import nn

from .config import QTRMConfig
from .blocks import QTRMBlockStack
from .workspace import LatentWorkspace
from .multimodal_projector import MultimodalProjector
from .core import QTRMRecursiveCore
from .heads import ControllerHeads
from .norm import RMSNorm
from .attention import CrossAttention
from .world_model import JepaWorldModelHead, SIGReg
from .agentic.transition_controller import TransitionStatePredictor


class QTRMMultimodalModel(nn.Module):
    """Standalone multimodal QTRM model."""

    def __init__(self, cfg: QTRMConfig):
        super().__init__()
        self.cfg = cfg
        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)
        self.prelude = QTRMBlockStack(cfg, cfg.n_prelude_layers, causal=True, attn_every=cfg.attn_every)
        self.jepa_encoder = QTRMBlockStack(cfg, cfg.jepa_encoder_layers, causal=True, attn_every=cfg.attn_every)
        self.jepa_encoder_norm = RMSNorm(cfg.d_model)
        self.core = QTRMRecursiveCore(cfg)
        self.core_to_text_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.core_to_text_enabled
            else None
        )
        self.core_to_text_norm_text = RMSNorm(cfg.d_model) if cfg.core_to_text_enabled else None
        self.core_to_text_norm_core = RMSNorm(cfg.d_model) if cfg.core_to_text_enabled else None
        self.core_to_text_gate = nn.Linear(cfg.d_model, 1) if cfg.core_to_text_enabled else None
        if self.core_to_text_gate is not None:
            nn.init.zeros_(self.core_to_text_gate.weight)
            nn.init.constant_(self.core_to_text_gate.bias, float(cfg.core_to_text_gate_init_bias))
        self.core_output_blend_gate = (
            nn.Linear(cfg.d_model, 1) if cfg.core_output_blend_enabled else None
        )
        if self.core_output_blend_gate is not None:
            nn.init.zeros_(self.core_output_blend_gate.weight)
            nn.init.constant_(
                self.core_output_blend_gate.bias,
                float(cfg.core_output_blend_init_bias),
            )
        self.answer_bottleneck_query_norm = (
            RMSNorm(cfg.d_model) if cfg.answer_bottleneck_enabled else None
        )
        self.answer_bottleneck_workspace_norm = (
            RMSNorm(cfg.d_model) if cfg.answer_bottleneck_enabled else None
        )
        self.answer_bottleneck_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.answer_bottleneck_enabled
            else None
        )
        self.answer_bottleneck_output_norm = (
            RMSNorm(cfg.d_model) if cfg.answer_bottleneck_enabled else None
        )
        self.core_loop_readout_query_norm = (
            RMSNorm(cfg.d_model) if cfg.core_loop_readout_enabled else None
        )
        self.core_loop_readout_state_norm = (
            RMSNorm(cfg.d_model) if cfg.core_loop_readout_enabled else None
        )
        self.core_loop_readout_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.core_loop_readout_enabled
            else None
        )
        self.core_loop_readout_output_norm = (
            RMSNorm(cfg.d_model) if cfg.core_loop_readout_enabled else None
        )
        self.answer_state_loop_query_norm = (
            RMSNorm(cfg.d_model) if cfg.answer_state_loop_enabled else None
        )
        self.answer_state_loop_state_norm = (
            RMSNorm(cfg.d_model) if cfg.answer_state_loop_enabled else None
        )
        self.answer_state_loop_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.answer_state_loop_enabled
            else None
        )
        self.answer_state_loop_gate = (
            nn.Linear(cfg.d_model, 1) if cfg.answer_state_loop_enabled else None
        )
        self.answer_state_loop_output_norm = (
            RMSNorm(cfg.d_model) if cfg.answer_state_loop_enabled else None
        )
        if self.answer_state_loop_gate is not None:
            nn.init.zeros_(self.answer_state_loop_gate.weight)
            nn.init.constant_(
                self.answer_state_loop_gate.bias,
                float(cfg.answer_state_loop_gate_init_bias),
            )
        transition_state_dim = max(1, int(cfg.transition_state_dim))
        self.transition_state_predictor = (
            TransitionStatePredictor(
                d_model=cfg.d_model,
                state_dim=transition_state_dim,
                hidden_dim=cfg.transition_state_hidden_dim,
                dropout=cfg.dropout,
            )
            if cfg.transition_state_enabled
            else None
        )
        self.transition_state_to_answer = (
            nn.Linear(transition_state_dim, cfg.d_model, bias=False)
            if cfg.transition_state_enabled
            else None
        )
        self.transition_state_answer_gate = (
            nn.Linear(cfg.d_model, 1) if cfg.transition_state_enabled else None
        )
        if self.transition_state_answer_gate is not None:
            nn.init.zeros_(self.transition_state_answer_gate.weight)
            nn.init.constant_(
                self.transition_state_answer_gate.bias,
                float(cfg.transition_state_answer_gate_init_bias),
            )
        transition_state_codebook_size = max(1, int(cfg.transition_state_codebook_size))
        self.transition_state_code_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.transition_state_code_enabled else None
        )
        self.transition_state_code_head = (
            nn.Linear(cfg.d_model, transition_state_codebook_size)
            if cfg.transition_state_code_enabled
            else None
        )
        self.transition_state_code_embed = (
            nn.Embedding(transition_state_codebook_size, cfg.d_model)
            if cfg.transition_state_code_enabled
            else None
        )
        if self.transition_state_code_head is not None:
            nn.init.xavier_uniform_(self.transition_state_code_head.weight)
            nn.init.zeros_(self.transition_state_code_head.bias)
        if self.transition_state_code_embed is not None:
            nn.init.normal_(self.transition_state_code_embed.weight, mean=0.0, std=0.02)
        self.transition_state_finality_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.transition_state_finality_enabled else None
        )
        self.transition_state_finality_head = (
            nn.Linear(cfg.d_model, 1) if cfg.transition_state_finality_enabled else None
        )
        if self.transition_state_finality_head is not None:
            nn.init.xavier_uniform_(self.transition_state_finality_head.weight)
            nn.init.zeros_(self.transition_state_finality_head.bias)
        primitive_transition_num_operations = max(
            0,
            int(cfg.primitive_transition_num_operations),
        )
        primitive_transition_hidden_dim = int(
            cfg.primitive_transition_hidden_dim or cfg.d_model
        )
        self.primitive_transition_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.primitive_transition_enabled
            and primitive_transition_num_operations > 0
            else None
        )
        self.primitive_transition_prompt_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.primitive_transition_enabled
            and primitive_transition_num_operations > 0
            and cfg.primitive_transition_prompt_context_enabled
            else None
        )
        self.primitive_transition_prompt_query_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.primitive_transition_enabled
            and primitive_transition_num_operations > 0
            and cfg.primitive_transition_prompt_context_enabled
            and cfg.primitive_transition_prompt_token_attention_enabled
            else None
        )
        self.primitive_transition_prompt_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.primitive_transition_enabled
            and primitive_transition_num_operations > 0
            and cfg.primitive_transition_prompt_context_enabled
            and cfg.primitive_transition_prompt_token_attention_enabled
            else None
        )
        self.primitive_transition_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.primitive_transition_enabled
            and primitive_transition_num_operations > 0
            and cfg.primitive_transition_prompt_context_enabled
            and cfg.primitive_transition_prompt_token_attention_enabled
            else None
        )
        primitive_transition_input_dim = (
            cfg.d_model * 2
            if cfg.primitive_transition_prompt_context_enabled
            else cfg.d_model
        )
        self.primitive_transition_operation_head = (
            nn.Sequential(
                nn.Linear(primitive_transition_input_dim, primitive_transition_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    primitive_transition_hidden_dim,
                    primitive_transition_num_operations,
                ),
            )
            if cfg.primitive_transition_enabled
            and primitive_transition_num_operations > 0
            else None
        )
        if self.primitive_transition_operation_head is not None:
            for module in self.primitive_transition_operation_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        self.answer_residual_governor = (
            nn.Linear(cfg.d_model, 1) if cfg.answer_residual_governor_enabled else None
        )
        if self.answer_residual_governor is not None:
            nn.init.zeros_(self.answer_residual_governor.weight)
            nn.init.constant_(
                self.answer_residual_governor.bias,
                float(cfg.answer_residual_governor_init_bias),
            )
        self.evidence_span_query_norm = (
            RMSNorm(cfg.d_model) if cfg.evidence_span_reader_enabled else None
        )
        self.evidence_span_workspace_norm = (
            RMSNorm(cfg.d_model) if cfg.evidence_span_reader_enabled else None
        )
        self.evidence_span_query_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.evidence_span_reader_enabled
            else None
        )
        self.evidence_span_start_key = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.evidence_span_reader_enabled
            else None
        )
        self.evidence_span_end_key = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.evidence_span_reader_enabled
            else None
        )
        self.evidence_span_no_answer_head = (
            nn.Linear(cfg.d_model, 1) if cfg.evidence_span_reader_enabled else None
        )
        self.coda = QTRMBlockStack(
            cfg,
            cfg.n_coda_layers,
            causal=True,
            attn_every=cfg.coda_attn_every or cfg.attn_every,
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.text_embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.workspace = LatentWorkspace(
            cfg.d_model,
            cfg.workspace_tokens,
            cfg.n_heads,
            layers=cfg.workspace_layers,
            ff_mult=cfg.workspace_ff_mult,
            include_latents_in_kv=cfg.workspace_include_latents_in_kv,
            memory_gate_enabled=cfg.workspace_memory_gate_enabled,
            memory_gate_init_bias=cfg.workspace_memory_gate_init_bias,
        )
        self.projector = MultimodalProjector(
            cfg.d_model, cfg.visual_dim, cfg.max_visual_tokens, cfg.n_heads,
        )
        if int(cfg.temporal_spatial_context_max_tokens) <= 0:
            raise ValueError("temporal_spatial_context_max_tokens must be positive")
        if int(cfg.temporal_spatial_context_dim) <= 0:
            raise ValueError("temporal_spatial_context_dim must be positive")
        self.temporal_spatial_context_proj = (
            nn.Linear(int(cfg.temporal_spatial_context_dim), cfg.d_model, bias=False)
            if cfg.temporal_spatial_context_enabled
            else None
        )
        self.temporal_spatial_context_norm = (
            RMSNorm(cfg.d_model) if cfg.temporal_spatial_context_enabled else None
        )
        self.temporal_spatial_context_pos = (
            nn.Parameter(
                torch.randn(
                    int(cfg.temporal_spatial_context_max_tokens),
                    cfg.d_model,
                )
                * 0.02
            )
            if cfg.temporal_spatial_context_enabled
            else None
        )
        self.ctrl = ControllerHeads(cfg.d_model, cfg.num_actions)
        controller_signal_source = str(cfg.controller_signal_source or "external").lower()
        if controller_signal_source not in {"external", "learned_core", "learned_readout"}:
            raise ValueError(
                "controller_signal_source must be 'external', 'learned_core', or 'learned_readout'"
            )
        self.controller_signal_source = controller_signal_source
        controller_signal_active = bool(
            cfg.controller_signal_enabled or controller_signal_source != "external"
        )
        controller_signal_dim = max(1, int(cfg.controller_signal_dim))
        self.controller_signal_proj = (
            nn.Linear(controller_signal_dim, cfg.d_model, bias=False)
            if controller_signal_active
            else None
        )
        self.controller_signal_head = (
            nn.Linear(cfg.d_model, controller_signal_dim)
            if controller_signal_source != "external"
            else None
        )
        if self.controller_signal_proj is not None:
            nn.init.xavier_uniform_(self.controller_signal_proj.weight)
        if self.controller_signal_head is not None:
            nn.init.xavier_uniform_(self.controller_signal_head.weight)
            nn.init.zeros_(self.controller_signal_head.bias)
        self.residual_gate = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, float(cfg.qtrm_residual_gate_init_bias))
        self.evidence_support_head = nn.Linear(cfg.d_model, 1)
        self.evidence_refute_head = nn.Linear(cfg.d_model, 1)
        self.evidence_missing_head = nn.Linear(cfg.d_model, 1)
        self.evidence_causal_gate_head = nn.Linear(cfg.d_model, 1)
        for head in (
            self.evidence_support_head,
            self.evidence_refute_head,
            self.evidence_missing_head,
            self.evidence_causal_gate_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(
            self.evidence_causal_gate_head.bias,
            float(cfg.evidence_bottleneck_gate_init_bias),
        )
        self.generation_repeat_head = (
            nn.Linear(cfg.d_model, 1) if cfg.generation_verifier_enabled else None
        )
        self.generation_stop_head = (
            nn.Linear(cfg.d_model, 1) if cfg.generation_verifier_enabled else None
        )
        self.generation_quality_head = (
            nn.Linear(cfg.d_model, 1) if cfg.generation_verifier_enabled else None
        )
        for head in (
            self.generation_repeat_head,
            self.generation_stop_head,
            self.generation_quality_head,
        ):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.answer_decision_head = (
            nn.Linear(cfg.d_model, 1) if cfg.answer_decision_head_enabled else None
        )
        answer_decision_feature_dim = max(0, int(cfg.answer_decision_feature_dim))
        self.answer_decision_feature_norm = (
            nn.LayerNorm(answer_decision_feature_dim)
            if cfg.answer_decision_head_enabled and answer_decision_feature_dim > 0
            else None
        )
        self.answer_decision_feature_proj = (
            nn.Linear(answer_decision_feature_dim, cfg.d_model)
            if cfg.answer_decision_head_enabled and answer_decision_feature_dim > 0
            else None
        )
        answer_decision_feature_hidden_dim = max(
            1,
            int(getattr(cfg, "answer_decision_feature_hidden_dim", 32)),
        )
        self.answer_decision_feature_head = (
            nn.Sequential(
                nn.Linear(answer_decision_feature_dim, answer_decision_feature_hidden_dim),
                nn.GELU(),
                nn.Linear(answer_decision_feature_hidden_dim, answer_decision_feature_hidden_dim),
                nn.GELU(),
                nn.Linear(answer_decision_feature_hidden_dim, 1),
            )
            if cfg.answer_decision_head_enabled and answer_decision_feature_dim > 0
            else None
        )
        if self.answer_decision_feature_proj is not None:
            nn.init.xavier_uniform_(self.answer_decision_feature_proj.weight)
            nn.init.zeros_(self.answer_decision_feature_proj.bias)
        if self.answer_decision_feature_head is not None:
            for module in self.answer_decision_feature_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.answer_decision_head is not None:
            nn.init.xavier_uniform_(self.answer_decision_head.weight)
            nn.init.zeros_(self.answer_decision_head.bias)
        self.jepa = JepaWorldModelHead(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            num_actions=cfg.num_actions,
            predictor_layers=cfg.jepa_predictor_layers,
            predictor_dim=cfg.jepa_predictor_dim,
            max_seq_len=cfg.max_seq_len,
            horizon=cfg.jepa_horizon,
            dropout=cfg.dropout,
        )
        self.jepa_sigreg = SIGReg(knots=cfg.jepa_sigreg_knots, num_proj=cfg.jepa_sigreg_num_proj)
        self.core_world_model = (
            JepaWorldModelHead(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                num_actions=cfg.num_actions,
                predictor_layers=cfg.core_world_model_predictor_layers,
                predictor_dim=cfg.core_world_model_predictor_dim,
                max_seq_len=max(
                    1,
                    int(cfg.outer_steps),
                    int(cfg.core_step_conditioning_max_steps),
                ),
                horizon=cfg.core_world_model_horizon,
                dropout=cfg.dropout,
            )
            if cfg.core_world_model_enabled
            else None
        )
        self.core_world_model_sigreg = (
            SIGReg(knots=cfg.jepa_sigreg_knots, num_proj=cfg.jepa_sigreg_num_proj)
            if cfg.core_world_model_enabled
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        text_states: Optional[torch.Tensor] = None,
        workspace_text_states: Optional[torch.Tensor] = None,
        workspace_attention_mask: Optional[torch.Tensor] = None,
        donor_logits: Optional[torch.Tensor] = None,
        disable_workspace: bool = False,
        disable_core: bool = False,
        disable_coda: bool = False,
        disable_qtrm_residual: bool = False,
        disable_donor_context: bool = False,
        disable_workspace_memory_context: bool = False,
        disable_workspace_memory_gate: bool = False,
        disable_core_context: bool = False,
        disable_core_to_text: bool = False,
        disable_evidence_bottleneck: bool = False,
        disable_evidence_span_reader: bool = False,
        disable_answer_residual_governor: bool = False,
        disable_answer_decision_head: bool = False,
        disable_answer_decision_features: bool = False,
        evidence_span_reader_context: str = "workspace",
        workspace_only_context: bool = False,
        core_world_model_actions: Optional[torch.Tensor] = None,
        temporal_spatial_context: Optional[torch.Tensor] = None,
        controller_signal: Optional[torch.Tensor] = None,
        controller_signal_mask: Optional[torch.Tensor] = None,
        answer_decision_features: Optional[torch.Tensor] = None,
        disable_controller_signal: bool = False,
        disable_temporal_spatial_context: bool = False,
        disable_transition_state: bool = False,
        enable_core_halt: Optional[bool] = None,
        return_core_depth_logits: bool = False,
        return_core_depth_text_logits: bool = False,
        return_features_only: bool = False,
    ) -> dict:
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        disable_core = bool(disable_core or not self.cfg.core_enabled)

        text_seq = self.text_embed(input_ids)
        input_text_seq = text_seq
        input_text_mask = attention_mask
        jepa_latents = self.jepa_encoder(text_seq, attention_mask=attention_mask)
        jepa_latents = self.jepa_encoder_norm(jepa_latents)
        jepa_outputs = self.jepa(jepa_latents, attention_mask=attention_mask)

        seq = text_seq
        seq, attention_mask, temporal_spatial_context_token_count = (
            self._prepend_temporal_spatial_context(
                seq,
                attention_mask,
                temporal_spatial_context=temporal_spatial_context,
                disabled=disable_temporal_spatial_context,
            )
        )

        if text_states is not None and not disable_donor_context:
            seq, attention_mask = self.projector(seq, text_states, text_mask=attention_mask)

        if visual_features is not None and not disable_donor_context:
            seq, attention_mask = self.projector(seq, visual_features, text_mask=attention_mask)

        workspace_memory_token_count = 0
        workspace_memory_present = text_seq.new_zeros((b,))
        workspace_reader_states = text_seq.new_empty((b, 0, self.cfg.d_model))
        workspace_reader_mask = attention_mask.new_zeros((b, 0))
        has_workspace_memory = (
            workspace_text_states is not None
            and not disable_workspace
            and not disable_workspace_memory_context
        )
        workspace_input_seq = seq
        workspace_input_mask = attention_mask
        if has_workspace_memory:
            workspace_input_seq, workspace_input_mask = self.projector(
                workspace_input_seq,
                workspace_text_states,
                text_mask=workspace_input_mask,
                feature_mask=workspace_attention_mask,
            )
            workspace_reader_token_count = min(
                int(workspace_text_states.shape[1]),
                int(self.projector.max_visual_tokens),
            )
            workspace_reader_states = workspace_input_seq[:, :workspace_reader_token_count]
            if workspace_input_mask is not None:
                workspace_reader_mask = workspace_input_mask[:, :workspace_reader_token_count]
            workspace_memory_token_count = self._feature_token_count(
                workspace_text_states,
                workspace_attention_mask,
            )
            workspace_memory_present = self._feature_present_mask(
                workspace_text_states,
                workspace_attention_mask,
            )

        if disable_workspace:
            seq = self.prelude(seq, attention_mask=attention_mask)
            text_context_seq = seq
            text_context_mask = attention_mask
        elif workspace_only_context:
            workspace_seq = self.prelude(workspace_input_seq, attention_mask=workspace_input_mask)
            workspace_mask_input = workspace_input_mask
            text_context_seq = self.prelude(input_text_seq, attention_mask=input_text_mask)
            text_context_mask = input_text_mask
            seq = workspace_seq
            attention_mask = workspace_mask_input
        elif has_workspace_memory:
            workspace_seq = self.prelude(workspace_input_seq, attention_mask=workspace_input_mask)
            workspace_mask_input = workspace_input_mask
            text_context_seq = self.prelude(seq, attention_mask=attention_mask)
            text_context_mask = attention_mask
            seq = workspace_seq
            attention_mask = workspace_mask_input
        else:
            seq = self.prelude(seq, attention_mask=attention_mask)
            text_context_seq = seq
            text_context_mask = attention_mask
        if disable_workspace:
            workspace = seq.new_zeros((b, self.cfg.workspace_tokens, self.cfg.d_model))
            workspace_mask = torch.ones(
                workspace.shape[:2],
                device=workspace.device,
                dtype=attention_mask.dtype,
            )
            z_l = workspace
            z_h = workspace
            trajectory = []
            core_halt_info = self._empty_core_halt_info(workspace)
            core_depth_states = self._empty_core_depth_states(workspace)
            transition_state_info = self._compute_transition_state_outputs(
                core_depth_states,
                disabled=True,
            )
            transition_state_code_info = self._compute_transition_state_code_outputs(
                core_depth_states,
                disabled=True,
            )
            transition_state_finality_logits = self._compute_transition_state_finality_logits(
                core_depth_states,
                disabled=True,
            )
            primitive_transition_info = self._compute_primitive_transition_outputs(
                core_depth_states,
                prompt_context_seq=text_context_seq,
                prompt_context_mask=text_context_mask,
                disabled=True,
            )
            core_depth_last_logits = self._empty_core_depth_last_logits(workspace)
            core_depth_text_logits = self._empty_core_depth_text_logits(workspace, s)
            workspace_update_gate_mean = workspace.new_empty((b, 0))
            core_context_gate_mean = workspace.new_empty((b, 0))
        else:
            workspace, workspace_info = self.workspace(
                seq,
                context_mask=attention_mask,
                return_info=True,
                disable_memory_gate=disable_workspace_memory_gate,
            )
            workspace_update_gate_mean = workspace_info["update_gate_mean"]
            workspace_mask = torch.ones(
                workspace.shape[:2],
                device=workspace.device,
                dtype=attention_mask.dtype,
            )
            if disable_core:
                z_l = workspace
                z_h = workspace
                trajectory = []
                core_halt_info = self._empty_core_halt_info(workspace)
                core_context_gate_mean = workspace.new_empty((b, 0))
            else:
                z_l, z_h, trajectory, core_halt_info = self.core(
                    workspace,
                    attention_mask=workspace_mask,
                    context_states=seq,
                    context_mask=attention_mask,
                    disable_context=disable_core_context,
                    enable_halt=enable_core_halt,
                )
                core_context_gate_mean = core_halt_info["context_gate_mean"]
            core_depth_states = self._core_depth_states(trajectory, workspace)
            transition_state_info = self._compute_transition_state_outputs(
                core_depth_states,
                disabled=bool(disable_transition_state or disable_core),
            )
            transition_state_code_info = self._compute_transition_state_code_outputs(
                core_depth_states,
                disabled=bool(disable_transition_state or disable_core),
            )
            transition_state_finality_logits = self._compute_transition_state_finality_logits(
                core_depth_states,
                disabled=bool(disable_transition_state or disable_core),
            )
            primitive_transition_info = self._compute_primitive_transition_outputs(
                core_depth_states,
                prompt_context_seq=text_context_seq,
                prompt_context_mask=text_context_mask,
                disabled=bool(disable_core),
            )
            core_depth_last_logits = (
                self._core_depth_last_logits(
                    trajectory,
                    text_context_seq=text_context_seq,
                    text_context_mask=text_context_mask,
                    workspace_mask=workspace_mask,
                    transition_state_features=transition_state_info["features"],
                    transition_state_code_embeddings=transition_state_code_info[
                        "embeddings"
                    ],
                )
                if return_core_depth_logits
                else self._empty_core_depth_last_logits(workspace)
            )
            core_depth_text_logits = (
                self._core_depth_text_logits(
                    trajectory,
                    text_context_seq=text_context_seq,
                    text_context_mask=text_context_mask,
                    workspace_mask=workspace_mask,
                    input_seq_len=s,
                    transition_state_features=transition_state_info["features"],
                    transition_state_code_embeddings=transition_state_code_info[
                        "embeddings"
                    ],
                )
                if return_core_depth_text_logits
                else self._empty_core_depth_text_logits(workspace, s)
            )
        transition_state_text_logits = self._compute_transition_state_text_logits(
            transition_state_info["features"]
        )
        core_to_text_gate_mean = text_context_seq.new_empty((b, 0))
        core_output_blend_gate_mean = text_context_seq.new_empty((b, 0))
        if (
            self.core_output_blend_gate is not None
            and not disable_workspace
            and not disable_core
        ):
            core_output_blend_gate = torch.sigmoid(self.core_output_blend_gate(workspace))
            gate_min = min(max(float(self.cfg.core_output_blend_min), 0.0), 1.0)
            if gate_min != 0.0:
                core_output_blend_gate = (
                    gate_min + (1.0 - gate_min) * core_output_blend_gate
                )
            z_h = workspace + core_output_blend_gate * (z_h - workspace)
            z_l = workspace + core_output_blend_gate * (z_l - workspace)
            core_output_blend_gate_mean = core_output_blend_gate.squeeze(-1).mean(dim=1)
        if (
            self.core_to_text_cross is not None
            and self.core_to_text_norm_text is not None
            and self.core_to_text_norm_core is not None
            and self.core_to_text_gate is not None
            and not disable_workspace
            and not disable_core_to_text
        ):
            core_delta = self.core_to_text_cross(
                self.core_to_text_norm_text(text_context_seq),
                self.core_to_text_norm_core(z_h),
                workspace_mask,
            )
            core_to_text_gate = torch.sigmoid(self.core_to_text_gate(text_context_seq))
            gate_min = min(max(float(self.cfg.core_to_text_gate_min), 0.0), 1.0)
            if gate_min != 0.0:
                core_to_text_gate = gate_min + (1.0 - gate_min) * core_to_text_gate
            text_context_seq = text_context_seq + core_to_text_gate * core_delta
            core_to_text_gate_mean = core_to_text_gate.mean(dim=(1, 2))
        if not disable_workspace:
            seq = torch.cat([z_h, text_context_seq], dim=1)
            attention_mask = torch.cat([workspace_mask, text_context_mask], dim=1)
        core_world_model_outputs = self._core_world_model_outputs(
            core_depth_states,
            actions=core_world_model_actions,
        )
        pooled = z_h[:, -1, :]
        evidence_info = self._compute_evidence_bottleneck(
            pooled,
            workspace_memory_present=workspace_memory_present,
        )
        if evidence_span_reader_context not in {"workspace", "input"}:
            raise ValueError("evidence_span_reader_context must be 'workspace' or 'input'")
        if evidence_span_reader_context == "input":
            reader_states = text_context_seq[:, -s:]
            reader_mask = (
                text_context_mask[:, -s:]
                if text_context_mask is not None and text_context_mask.numel() != 0
                else attention_mask.new_ones((b, s))
            )
        else:
            reader_states = workspace_reader_states
            reader_mask = workspace_reader_mask
        evidence_span_info = self._compute_evidence_span_reader(
            text_context_seq,
            text_context_mask,
            reader_states
            if not disable_evidence_span_reader
            else reader_states[:, :0],
            reader_mask
            if not disable_evidence_span_reader
            else reader_mask[:, :0],
        )
        if not disable_coda:
            seq = self.coda(seq, attention_mask=attention_mask)
        seq = self.norm(seq)
        generation_pooled = self._last_valid_hidden(seq, attention_mask)
        controller_pooled = generation_pooled * float(self.cfg.controller_signal_base_scale)
        controller_signal_logits = generation_pooled.new_empty((b, 0))
        controller_signal_pred = generation_pooled.new_empty((b, 0))
        controller_signal_injected = False
        signal = None
        if self.controller_signal_proj is not None and not disable_controller_signal:
            if self.controller_signal_source != "external":
                if self.controller_signal_head is None:
                    raise RuntimeError("learned signal source requires controller_signal_head")
                if self.controller_signal_source == "learned_core":
                    signal_source = pooled
                    if disable_core or disable_workspace:
                        signal_source = torch.zeros_like(signal_source)
                elif self.controller_signal_source == "learned_readout":
                    signal_source = generation_pooled
                else:
                    signal_source = generation_pooled
                controller_signal_logits = self.controller_signal_head(signal_source)
                signal = torch.sigmoid(controller_signal_logits)
                controller_signal_pred = signal
            elif controller_signal is not None:
                signal = controller_signal.to(
                    device=controller_pooled.device,
                    dtype=controller_pooled.dtype,
                )
                if signal.ndim != 2 or signal.shape[0] != b:
                    raise ValueError("controller_signal must have shape [batch, signal_dim]")
                controller_signal_pred = signal
        if signal is not None and self.controller_signal_proj is not None:
            expected_dim = self.controller_signal_proj.in_features
            if signal.shape[1] != expected_dim:
                raise ValueError(
                    f"controller_signal dim must be {expected_dim}, got {signal.shape[1]}"
                )
            if controller_signal_mask is not None:
                mask = controller_signal_mask.to(
                    device=signal.device,
                    dtype=signal.dtype,
                )
                if mask.ndim == 1:
                    mask = mask.view(1, -1)
                if mask.shape[-1] != signal.shape[-1]:
                    raise ValueError("controller_signal_mask dim must match controller_signal")
                signal = signal * mask
            controller_pooled = controller_pooled + self.controller_signal_proj(signal)
            controller_signal_injected = True
        generation_info = self._compute_generation_verifier(generation_pooled)
        answer_decision_info = self._compute_answer_decision(
            generation_pooled,
            answer_decision_features=answer_decision_features,
            disabled=disable_answer_decision_head,
            features_disabled=disable_answer_decision_features,
        )
        ctrl = self.ctrl(controller_pooled)
        if return_features_only:
            return {
                **generation_info,
                **answer_decision_info,
                "z_l": z_l,
                "z_h": z_h,
                "pooled": pooled,
                "controller_pooled": controller_pooled,
                "controller_signal_logits": controller_signal_logits,
                "controller_signal_pred": controller_signal_pred,
                "controller_signal_used": controller_pooled.new_tensor(
                    1.0 if controller_signal_injected else 0.0
                ),
                "generation_verifier_pooled": generation_pooled,
                "trajectory_len": torch.tensor(len(trajectory), device=seq.device),
                "core_q_halt_logits": core_halt_info["q_halt_logits"],
                "core_q_continue_logits": core_halt_info["q_continue_logits"],
                "core_halted": core_halt_info["halted"],
                "core_steps": core_halt_info["steps"],
                "core_depth_states": core_depth_states,
                "core_depth_last_logits": core_depth_last_logits,
                "core_depth_text_logits": core_depth_text_logits,
                "transition_state_logits": transition_state_info["logits"],
                "transition_state_features": transition_state_info["features"],
                "transition_state_text_logits": transition_state_text_logits,
                "transition_state_code_logits": transition_state_code_info["logits"],
                "transition_state_code_embeddings": transition_state_code_info[
                    "embeddings"
                ],
                "transition_state_finality_logits": transition_state_finality_logits,
                "primitive_transition_operation_logits": primitive_transition_info[
                    "operation_logits"
                ],
                "core_context_gate_mean": core_context_gate_mean,
                "core_output_blend_gate_mean": core_output_blend_gate_mean,
                "core_to_text_gate_mean": core_to_text_gate_mean,
                "temporal_spatial_context_token_count": temporal_spatial_context_token_count,
                "workspace_update_gate_mean": workspace_update_gate_mean,
                "workspace_memory_token_count": workspace_memory_token_count,
                "workspace_memory_present": workspace_memory_present,
                **ctrl,
            }
        qtrm_logits = self.lm_head(seq) * float(self.cfg.qtrm_logits_scale)
        answer_bottleneck_logits = self._empty_answer_bottleneck_logits(qtrm_logits, s)
        answer_bottleneck_hidden = seq.new_empty((b, 0, self.cfg.d_model))
        core_loop_readout_logits = self._empty_core_loop_readout_logits(qtrm_logits)
        core_loop_readout_hidden = self._empty_core_loop_readout_hidden(seq)
        answer_state_loop_logits = self._empty_answer_state_loop_logits(qtrm_logits)
        answer_state_loop_hidden = self._empty_answer_state_loop_hidden(seq)
        answer_state_loop_depth_hidden = self._empty_answer_state_loop_depth_hidden(seq, s)
        answer_residual_governor_logits = qtrm_logits.new_empty((b, 0))
        answer_residual_governor_gate = qtrm_logits.new_empty((b, 0))
        qtrm_residual_logits = qtrm_logits
        if self.answer_bottleneck_cross is not None:
            text_offset = qtrm_logits.shape[1] - s
            core_required_but_disabled = bool(
                self.cfg.answer_bottleneck_requires_core
                and (disable_workspace or disable_core)
            )
            if core_required_but_disabled:
                answer_bottleneck_logits = qtrm_logits.new_zeros(
                    (b, s, qtrm_logits.shape[-1])
                )
                qtrm_residual_logits = torch.zeros_like(qtrm_logits)
            else:
                answer_bottleneck_logits, answer_bottleneck_hidden = self._compute_answer_bottleneck_outputs(
                    seq,
                    z_h=z_h,
                    workspace_mask=workspace_mask,
                    input_seq_len=s,
                )
                answer_text_residual = answer_bottleneck_logits
                if self.cfg.answer_bottleneck_requires_workspace_memory:
                    answer_text_residual = (
                        answer_text_residual
                        * workspace_memory_present.to(
                            device=answer_text_residual.device,
                            dtype=answer_text_residual.dtype,
                        )[:, None, None]
                    )
                residual_prefix = qtrm_logits.new_zeros(
                    (b, text_offset, qtrm_logits.shape[-1])
                )
                qtrm_residual_logits = torch.cat(
                    [residual_prefix, answer_text_residual],
                    dim=1,
                )
        if (
            self.answer_residual_governor is not None
            and answer_bottleneck_hidden.numel() != 0
        ):
            text_offset = qtrm_residual_logits.shape[1] - s
            if disable_answer_residual_governor:
                answer_residual_governor_logits = qtrm_logits.new_zeros((b, s))
                answer_residual_governor_gate = qtrm_logits.new_ones((b, s))
            else:
                answer_residual_governor_logits = self.answer_residual_governor(
                    answer_bottleneck_hidden
                ).squeeze(-1)
                answer_residual_governor_gate = torch.sigmoid(
                    answer_residual_governor_logits
                )
                gate_min = min(
                    max(float(self.cfg.answer_residual_governor_min), 0.0),
                    1.0,
                )
                if gate_min != 0.0:
                    answer_residual_governor_gate = (
                        gate_min
                        + (1.0 - gate_min) * answer_residual_governor_gate
                    )
                gated_answer_residual = (
                    qtrm_residual_logits[:, text_offset:, :]
                    * answer_residual_governor_gate[:, :, None]
                )
                qtrm_residual_logits = torch.cat(
                    [qtrm_residual_logits[:, :text_offset, :], gated_answer_residual],
                    dim=1,
                )
        if self.core_loop_readout_cross is not None:
            text_offset = qtrm_logits.shape[1] - s
            core_required_but_disabled = bool(
                self.cfg.core_loop_readout_requires_core
                and (disable_workspace or disable_core or len(trajectory) == 0)
            )
            if core_required_but_disabled:
                core_loop_readout_logits = qtrm_logits.new_zeros(
                    (b, s, qtrm_logits.shape[-1])
                )
                core_loop_readout_hidden = seq.new_zeros((b, s, self.cfg.d_model))
                qtrm_residual_logits = torch.zeros_like(qtrm_logits)
            else:
                (
                    core_loop_readout_logits,
                    core_loop_readout_hidden,
                ) = self._compute_core_loop_readout_outputs(
                    text_context_seq,
                    z_h=z_h,
                    workspace_mask=workspace_mask,
                    input_seq_len=s,
                )
                residual_prefix = qtrm_logits.new_zeros(
                    (b, text_offset, qtrm_logits.shape[-1])
                )
                qtrm_residual_logits = torch.cat(
                    [residual_prefix, core_loop_readout_logits],
                    dim=1,
                )
        if self.answer_state_loop_cross is not None:
            text_offset = qtrm_logits.shape[1] - s
            core_required_but_disabled = bool(
                self.cfg.answer_state_loop_requires_core
                and (disable_workspace or disable_core or len(trajectory) == 0)
            )
            if core_required_but_disabled:
                answer_state_loop_logits = qtrm_logits.new_zeros(
                    (b, s, qtrm_logits.shape[-1])
                )
                answer_state_loop_hidden = seq.new_zeros((b, s, self.cfg.d_model))
                answer_state_loop_depth_hidden = seq.new_zeros(
                    (b, 0, s, self.cfg.d_model)
                )
                qtrm_residual_logits = torch.zeros_like(qtrm_logits)
            else:
                (
                    answer_state_loop_logits,
                    answer_state_loop_hidden,
                    answer_state_loop_depth_hidden,
                ) = self._compute_answer_state_loop_outputs(
                    text_context_seq,
                    trajectory=trajectory,
                    workspace_mask=workspace_mask,
                    input_seq_len=s,
                    transition_state_features=transition_state_info["features"],
                    transition_state_code_embeddings=transition_state_code_info[
                        "embeddings"
                    ],
                )
                residual_prefix = qtrm_logits.new_zeros(
                    (b, text_offset, qtrm_logits.shape[-1])
                )
                qtrm_residual_logits = torch.cat(
                    [residual_prefix, answer_state_loop_logits],
                    dim=1,
                )
        if self.cfg.qtrm_residual_clamp is not None:
            clamp = abs(float(self.cfg.qtrm_residual_clamp))
            qtrm_residual_logits = qtrm_residual_logits.clamp(min=-clamp, max=clamp)
        residual_gate = qtrm_logits.new_ones((b,))
        if self.cfg.qtrm_residual_gate_enabled:
            residual_gate = self._compute_residual_gate(z_h)
            qtrm_residual_logits = qtrm_residual_logits * residual_gate[:, None, None]
        if (
            self.cfg.evidence_bottleneck_enabled
            and self.cfg.evidence_bottleneck_applies_to_residual
            and not disable_evidence_bottleneck
        ):
            evidence_gate = evidence_info["evidence_bottleneck_gate"]
            qtrm_residual_logits = qtrm_residual_logits * evidence_gate[:, None, None]
        if disable_qtrm_residual:
            qtrm_residual_logits = torch.zeros_like(qtrm_residual_logits)
        donor_qtrm_conflict_gate = qtrm_logits.new_empty((b, 0))
        logits = qtrm_residual_logits
        if donor_logits is not None and self.cfg.donor_logits_scale != 0.0:
            if donor_logits.shape[:2] != (b, s):
                raise ValueError(
                    "donor_logits must have shape [batch, input_seq_len, vocab_size]"
                )
            if donor_logits.shape[-1] != self.cfg.vocab_size:
                raise ValueError("donor_logits vocab size must match model vocab_size")
            text_offset = logits.shape[1] - s
            text_residual_logits = qtrm_residual_logits[:, text_offset:, :]
            donor_text_logits = donor_logits.to(device=logits.device, dtype=logits.dtype)
            donor_qtrm_conflict_gate = self._compute_donor_qtrm_conflict_gate(
                text_residual_logits,
                donor_text_logits,
            )
            text_residual_logits = (
                text_residual_logits * donor_qtrm_conflict_gate[:, :, None]
            )
            fused_text_logits = (
                text_residual_logits
                + donor_text_logits * float(self.cfg.donor_logits_scale)
            )
            logits = torch.cat(
                [qtrm_residual_logits[:, :text_offset, :], fused_text_logits],
                dim=1,
            )

        return {
            "logits": logits,
            "qtrm_logits": qtrm_logits,
            "qtrm_residual_logits": qtrm_residual_logits,
            "answer_bottleneck_logits": answer_bottleneck_logits,
            "core_loop_readout_logits": core_loop_readout_logits,
            "core_loop_readout_hidden": core_loop_readout_hidden,
            "answer_state_loop_logits": answer_state_loop_logits,
            "answer_state_loop_hidden": answer_state_loop_hidden,
            "answer_state_loop_depth_hidden": answer_state_loop_depth_hidden,
            "answer_residual_governor_logits": answer_residual_governor_logits,
            "answer_residual_governor_gate": answer_residual_governor_gate,
            "qtrm_residual_gate": residual_gate,
            "donor_qtrm_conflict_gate": donor_qtrm_conflict_gate,
            **evidence_info,
            **evidence_span_info,
            **generation_info,
            **answer_decision_info,
            "z_l": z_l,
            "z_h": z_h,
            "pooled": pooled,
            "controller_pooled": controller_pooled,
            "controller_signal_logits": controller_signal_logits,
            "controller_signal_pred": controller_signal_pred,
            "controller_signal_used": controller_pooled.new_tensor(
                1.0 if controller_signal_injected else 0.0
            ),
            "generation_verifier_pooled": generation_pooled,
            "jepa_pred": jepa_outputs["pred"],
            "jepa_target": jepa_outputs["target"],
            "jepa_latents": jepa_outputs["latents"],
            "jepa_latent_mask": jepa_outputs["latent_mask"],
            "jepa_mask": jepa_outputs["mask"],
            "trajectory_len": torch.tensor(len(trajectory), device=seq.device),
            "core_q_halt_logits": core_halt_info["q_halt_logits"],
            "core_q_continue_logits": core_halt_info["q_continue_logits"],
            "core_halted": core_halt_info["halted"],
            "core_steps": core_halt_info["steps"],
            "core_depth_states": core_depth_states,
            "core_depth_last_logits": core_depth_last_logits,
            "core_depth_text_logits": core_depth_text_logits,
            "transition_state_logits": transition_state_info["logits"],
            "transition_state_features": transition_state_info["features"],
            "transition_state_text_logits": transition_state_text_logits,
            "transition_state_code_logits": transition_state_code_info["logits"],
            "transition_state_code_embeddings": transition_state_code_info["embeddings"],
            "transition_state_finality_logits": transition_state_finality_logits,
            "primitive_transition_operation_logits": primitive_transition_info[
                "operation_logits"
            ],
            "core_world_model_pred": core_world_model_outputs["pred"],
            "core_world_model_target": core_world_model_outputs["target"],
            "core_world_model_latents": core_world_model_outputs["latents"],
            "core_world_model_latent_mask": core_world_model_outputs["latent_mask"],
            "core_world_model_mask": core_world_model_outputs["mask"],
            "core_context_gate_mean": core_context_gate_mean,
            "core_output_blend_gate_mean": core_output_blend_gate_mean,
            "core_to_text_gate_mean": core_to_text_gate_mean,
            "temporal_spatial_context_token_count": temporal_spatial_context_token_count,
            "workspace_update_gate_mean": workspace_update_gate_mean,
            "workspace_memory_token_count": workspace_memory_token_count,
            "workspace_memory_present": workspace_memory_present,
            **ctrl,
        }

    def _prepend_temporal_spatial_context(
        self,
        seq: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        temporal_spatial_context: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = seq.new_tensor(0, dtype=torch.long)
        if (
            disabled
            or temporal_spatial_context is None
            or self.temporal_spatial_context_proj is None
            or self.temporal_spatial_context_norm is None
            or self.temporal_spatial_context_pos is None
        ):
            return seq, attention_mask, token_count
        context = temporal_spatial_context.to(device=seq.device, dtype=seq.dtype)
        if context.ndim == 2:
            context = context[:, None, :]
        if context.ndim != 3:
            raise ValueError("temporal_spatial_context must have shape [batch, dim] or [batch, tokens, dim]")
        if context.shape[0] != seq.shape[0]:
            raise ValueError("temporal_spatial_context batch size must match input_ids")
        expected_dim = int(self.temporal_spatial_context_proj.in_features)
        if context.shape[-1] != expected_dim:
            raise ValueError(
                f"temporal_spatial_context feature dim must be {expected_dim}, got {context.shape[-1]}"
            )
        max_tokens = int(self.cfg.temporal_spatial_context_max_tokens)
        context = context[:, :max_tokens, :]
        if context.shape[1] == 0:
            return seq, attention_mask, token_count
        context_tokens = self.temporal_spatial_context_proj(context)
        context_tokens = context_tokens + self.temporal_spatial_context_pos[
            : context_tokens.shape[1]
        ].unsqueeze(0).to(dtype=context_tokens.dtype, device=context_tokens.device)
        context_tokens = self.temporal_spatial_context_norm(context_tokens)
        context_mask = attention_mask.new_ones(
            (attention_mask.shape[0], context_tokens.shape[1])
        )
        return (
            torch.cat([context_tokens, seq], dim=1),
            torch.cat([context_mask, attention_mask], dim=1),
            seq.new_tensor(int(context_tokens.shape[1]), dtype=torch.long),
        )

    def _compute_residual_gate(self, z_h: torch.Tensor) -> torch.Tensor:
        gate_input = z_h[:, -1, :]
        if self.cfg.qtrm_residual_gate_normalize:
            gate_input = gate_input * torch.rsqrt(
                gate_input.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6)
            )
        gate = torch.sigmoid(self.residual_gate(gate_input).squeeze(-1))
        gate_min = min(max(float(self.cfg.qtrm_residual_gate_min), 0.0), 1.0)
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        return gate

    def _compute_donor_qtrm_conflict_gate(
        self,
        qtrm_text_logits: torch.Tensor,
        donor_text_logits: torch.Tensor,
    ) -> torch.Tensor:
        gate = qtrm_text_logits.new_ones(qtrm_text_logits.shape[:2])
        if not self.cfg.donor_qtrm_conflict_gate_enabled:
            return gate
        conflict_scale = min(
            max(float(self.cfg.donor_qtrm_conflict_qtrm_scale), 0.0),
            1.0,
        )
        donor_top = donor_text_logits.argmax(dim=-1)
        qtrm_top = qtrm_text_logits.argmax(dim=-1)
        return torch.where(
            donor_top != qtrm_top,
            gate.new_full(gate.shape, conflict_scale),
            gate,
        )

    def _compute_evidence_bottleneck(
        self,
        pooled: torch.Tensor,
        *,
        workspace_memory_present: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        support_logit = self.evidence_support_head(pooled).squeeze(-1)
        refute_logit = self.evidence_refute_head(pooled).squeeze(-1)
        missing_logit = self.evidence_missing_head(pooled).squeeze(-1)
        causal_gate_logit = self.evidence_causal_gate_head(pooled).squeeze(-1)
        gate_logit = causal_gate_logit + support_logit - refute_logit - missing_logit
        gate = torch.sigmoid(gate_logit)
        gate_min = min(max(float(self.cfg.evidence_bottleneck_gate_min), 0.0), 1.0)
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        if self.cfg.evidence_bottleneck_suppress_without_workspace:
            gate = gate * workspace_memory_present.to(device=gate.device, dtype=gate.dtype)
        return {
            "evidence_support_logits": support_logit,
            "evidence_refute_logits": refute_logit,
            "evidence_missing_logits": missing_logit,
            "evidence_causal_gate_logits": causal_gate_logit,
            "evidence_bottleneck_gate_logits": gate_logit,
            "evidence_bottleneck_gate": gate,
        }

    def _compute_generation_verifier(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            self.generation_repeat_head is None
            or self.generation_stop_head is None
            or self.generation_quality_head is None
        ):
            empty = pooled.new_empty((pooled.shape[0], 0))
            return {
                "generation_repeat_logits": empty,
                "generation_stop_logits": empty,
                "generation_quality_logits": empty,
            }
        return {
            "generation_repeat_logits": self.generation_repeat_head(pooled).squeeze(-1),
            "generation_stop_logits": self.generation_stop_head(pooled).squeeze(-1),
            "generation_quality_logits": self.generation_quality_head(pooled).squeeze(-1),
        }

    def _compute_answer_decision(
        self,
        pooled: torch.Tensor,
        *,
        answer_decision_features: Optional[torch.Tensor] = None,
        disabled: bool = False,
        features_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        empty = pooled.new_empty((pooled.shape[0], 0))
        if self.answer_decision_head is None or disabled:
            return {
                "answer_decision_logits": empty,
                "answer_decision_hidden_logits": empty,
                "answer_decision_feature_logits": empty,
            }
        decision_input = pooled
        feature_logits = pooled.new_zeros((pooled.shape[0],))
        feature_logits_available = False
        if self.answer_decision_feature_proj is not None and not features_disabled:
            feature_dim = int(self.answer_decision_feature_proj.in_features)
            if answer_decision_features is None:
                features = pooled.new_zeros((pooled.shape[0], feature_dim))
            else:
                features = answer_decision_features.to(
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
                if features.ndim == 1:
                    features = features.view(1, -1)
                if features.shape != (pooled.shape[0], feature_dim):
                    raise ValueError(
                        "answer_decision_features must have shape "
                        f"[batch, {feature_dim}]"
                    )
            projected_features = features
            if self.answer_decision_feature_norm is not None:
                projected_features = self.answer_decision_feature_norm(projected_features)
            decision_input = decision_input + self.answer_decision_feature_proj(projected_features)
            if self.answer_decision_feature_head is not None:
                feature_logits = self.answer_decision_feature_head(features).squeeze(-1)
                feature_logits_available = True
        hidden_logits = self.answer_decision_head(decision_input).squeeze(-1)
        decision_logits = (
            feature_logits.to(dtype=hidden_logits.dtype)
            if feature_logits_available
            else hidden_logits
        )
        return {
            "answer_decision_logits": decision_logits,
            "answer_decision_hidden_logits": hidden_logits,
            "answer_decision_feature_logits": feature_logits.to(dtype=hidden_logits.dtype),
        }

    def _compute_evidence_span_reader(
        self,
        question_states: torch.Tensor,
        question_mask: torch.Tensor,
        workspace_states: torch.Tensor,
        workspace_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = question_states.shape[0]
        if (
            self.evidence_span_query_norm is None
            or self.evidence_span_workspace_norm is None
            or self.evidence_span_query_proj is None
            or self.evidence_span_start_key is None
            or self.evidence_span_end_key is None
            or self.evidence_span_no_answer_head is None
            or workspace_states.numel() == 0
        ):
            empty = question_states.new_empty((b, 0))
            return {
                "evidence_span_start_logits": empty,
                "evidence_span_end_logits": empty,
                "evidence_span_no_answer_logits": question_states.new_zeros((b,)),
            }

        query_hidden = self._last_valid_hidden(question_states, question_mask)
        query = self.evidence_span_query_proj(self.evidence_span_query_norm(query_hidden))
        workspace = self.evidence_span_workspace_norm(workspace_states)
        start_keys = self.evidence_span_start_key(workspace)
        end_keys = self.evidence_span_end_key(workspace)
        scale = query.shape[-1] ** -0.5
        start_logits = torch.einsum("bd,btd->bt", query, start_keys) * scale
        end_logits = torch.einsum("bd,btd->bt", query, end_keys) * scale
        if workspace_mask.numel() != 0:
            valid = workspace_mask.to(device=start_logits.device, dtype=torch.bool)
            start_logits = start_logits.masked_fill(valid.logical_not(), -1.0e4)
            end_logits = end_logits.masked_fill(valid.logical_not(), -1.0e4)
        read_scores = torch.maximum(start_logits, end_logits)
        read_weights = torch.softmax(read_scores.float(), dim=-1).to(dtype=workspace.dtype)
        workspace_read = torch.einsum("bt,btd->bd", read_weights, workspace)
        no_answer_logits = self.evidence_span_no_answer_head(
            query_hidden + workspace_read
        ).squeeze(-1)
        return {
            "evidence_span_start_logits": start_logits,
            "evidence_span_end_logits": end_logits,
            "evidence_span_no_answer_logits": no_answer_logits,
        }

    def _empty_answer_bottleneck_logits(
        self,
        logits: torch.Tensor,
        input_seq_len: int,
    ) -> torch.Tensor:
        return logits.new_empty((logits.shape[0], int(input_seq_len), logits.shape[-1]))

    def _empty_core_loop_readout_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.new_empty((logits.shape[0], 0, logits.shape[-1]))

    def _empty_core_loop_readout_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_empty((hidden.shape[0], 0, hidden.shape[-1]))

    def _empty_answer_state_loop_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.new_empty((logits.shape[0], 0, logits.shape[-1]))

    def _empty_answer_state_loop_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_empty((hidden.shape[0], 0, hidden.shape[-1]))

    def _empty_answer_state_loop_depth_hidden(
        self,
        hidden: torch.Tensor,
        input_seq_len: int,
    ) -> torch.Tensor:
        return hidden.new_empty((hidden.shape[0], 0, int(input_seq_len), hidden.shape[-1]))

    def _compute_answer_bottleneck_logits(
        self,
        hidden: torch.Tensor,
        *,
        z_h: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
    ) -> torch.Tensor:
        logits, _ = self._compute_answer_bottleneck_outputs(
            hidden,
            z_h=z_h,
            workspace_mask=workspace_mask,
            input_seq_len=input_seq_len,
        )
        return logits

    def _compute_answer_bottleneck_outputs(
        self,
        hidden: torch.Tensor,
        *,
        z_h: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.answer_bottleneck_query_norm is None
            or self.answer_bottleneck_workspace_norm is None
            or self.answer_bottleneck_cross is None
            or self.answer_bottleneck_output_norm is None
        ):
            empty_logits = self._empty_answer_bottleneck_logits(
                hidden.new_empty(
                    (hidden.shape[0], hidden.shape[1], self.cfg.vocab_size)
                ),
                input_seq_len,
            )
            return empty_logits, hidden.new_empty((hidden.shape[0], 0, hidden.shape[-1]))
        text_offset = hidden.shape[1] - int(input_seq_len)
        text_hidden = hidden[:, text_offset:, :]
        answer_hidden = self.answer_bottleneck_cross(
            self.answer_bottleneck_query_norm(text_hidden),
            self.answer_bottleneck_workspace_norm(z_h),
            workspace_mask,
        )
        answer_hidden = self.answer_bottleneck_output_norm(answer_hidden)
        logits = self.lm_head(answer_hidden) * float(self.cfg.qtrm_logits_scale)
        return logits, answer_hidden

    def _compute_core_loop_readout_outputs(
        self,
        text_context_seq: torch.Tensor,
        *,
        z_h: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.core_loop_readout_query_norm is None
            or self.core_loop_readout_state_norm is None
            or self.core_loop_readout_cross is None
            or self.core_loop_readout_output_norm is None
        ):
            empty_logits = self._empty_answer_bottleneck_logits(
                text_context_seq.new_empty(
                    (text_context_seq.shape[0], text_context_seq.shape[1], self.cfg.vocab_size)
                ),
                input_seq_len,
            )
            return empty_logits, self._empty_core_loop_readout_hidden(text_context_seq)
        text_hidden = text_context_seq[:, -int(input_seq_len) :, :]
        loop_hidden = self.core_loop_readout_cross(
            self.core_loop_readout_query_norm(text_hidden),
            self.core_loop_readout_state_norm(z_h),
            workspace_mask,
        )
        loop_hidden = self.core_loop_readout_output_norm(loop_hidden)
        logits = self.lm_head(loop_hidden) * float(self.cfg.qtrm_logits_scale)
        return logits, loop_hidden

    def _compute_answer_state_loop_outputs(
        self,
        text_context_seq: torch.Tensor,
        *,
        trajectory: list[torch.Tensor],
        workspace_mask: torch.Tensor,
        input_seq_len: int,
        transition_state_features: Optional[torch.Tensor] = None,
        transition_state_code_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.answer_state_loop_query_norm is None
            or self.answer_state_loop_state_norm is None
            or self.answer_state_loop_cross is None
            or self.answer_state_loop_gate is None
            or self.answer_state_loop_output_norm is None
            or not trajectory
        ):
            empty_logits = self._empty_answer_state_loop_logits(
                text_context_seq.new_empty(
                    (
                        text_context_seq.shape[0],
                        text_context_seq.shape[1],
                        self.cfg.vocab_size,
                    )
                )
            )
            return (
                empty_logits,
                self._empty_answer_state_loop_hidden(text_context_seq),
                self._empty_answer_state_loop_depth_hidden(text_context_seq, input_seq_len),
            )
        y = text_context_seq[:, -int(input_seq_len) :, :]
        states = []
        gate_min = min(max(float(self.cfg.answer_state_loop_gate_min), 0.0), 1.0)
        transition_active = (
            transition_state_features is not None
            and transition_state_features.numel() != 0
            and self.transition_state_to_answer is not None
            and self.transition_state_answer_gate is not None
        )
        transition_code_active = (
            transition_state_code_embeddings is not None
            and transition_state_code_embeddings.numel() != 0
        )
        transition_gate_min = min(
            max(float(self.cfg.transition_state_answer_gate_min), 0.0),
            1.0,
        )
        for step_index, state in enumerate(trajectory):
            state_for_cross = state
            state_mask = workspace_mask
            if (
                transition_code_active
                and step_index < int(transition_state_code_embeddings.shape[1])
            ):
                code_delta = transition_state_code_embeddings[:, step_index, :].unsqueeze(1)
                y = self.answer_state_loop_output_norm(y + code_delta)
                code_mask = workspace_mask.new_ones((workspace_mask.shape[0], 1))
                if bool(self.cfg.transition_state_code_only_answer_loop):
                    state_for_cross = code_delta
                    state_mask = code_mask
                else:
                    state_for_cross = torch.cat([code_delta, state], dim=1)
                    state_mask = torch.cat([code_mask, workspace_mask], dim=1)
            if transition_active and step_index < int(transition_state_features.shape[1]):
                transition_delta = self.transition_state_to_answer(
                    transition_state_features[:, step_index, :]
                ).unsqueeze(1)
                transition_gate = torch.sigmoid(self.transition_state_answer_gate(y))
                if transition_gate_min != 0.0:
                    transition_gate = (
                        transition_gate_min
                        + (1.0 - transition_gate_min) * transition_gate
                    )
                y = self.answer_state_loop_output_norm(
                    y + transition_gate * transition_delta
                )
            delta = self.answer_state_loop_cross(
                self.answer_state_loop_query_norm(y),
                self.answer_state_loop_state_norm(state_for_cross),
                state_mask,
            )
            gate = torch.sigmoid(self.answer_state_loop_gate(y))
            if gate_min != 0.0:
                gate = gate_min + (1.0 - gate_min) * gate
            y = self.answer_state_loop_output_norm(y + gate * delta)
            states.append(y)
        logits = self.lm_head(y) * float(self.cfg.qtrm_logits_scale)
        depth_hidden = torch.stack(states, dim=1)
        return logits, y, depth_hidden

    @staticmethod
    def _last_valid_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask must match hidden batch/sequence shape")
        indices = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, indices.to(device=hidden.device)]

    def _feature_token_count(
        self,
        feature_states: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> int:
        token_limit = min(int(feature_states.shape[1]), int(self.projector.max_visual_tokens))
        if feature_mask is None:
            return token_limit
        if feature_mask.numel() == 0:
            return 0
        clipped = feature_mask[:, :token_limit]
        return int(clipped.to(dtype=torch.long).sum(dim=1).max().detach().cpu().item())

    def _feature_present_mask(
        self,
        feature_states: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = feature_states.shape[0]
        token_limit = min(int(feature_states.shape[1]), int(self.projector.max_visual_tokens))
        if feature_mask is None:
            return feature_states.new_ones((b,))
        if feature_mask.numel() == 0 or token_limit <= 0:
            return feature_states.new_zeros((b,))
        clipped = feature_mask[:, :token_limit]
        return clipped.to(device=feature_states.device, dtype=torch.bool).any(dim=1).to(feature_states.dtype)

    @staticmethod
    def _empty_core_halt_info(workspace: torch.Tensor) -> dict[str, torch.Tensor]:
        b = workspace.shape[0]
        return {
            "q_halt_logits": workspace.new_empty((b, 0), dtype=torch.float32),
            "q_continue_logits": workspace.new_empty((b, 0), dtype=torch.float32),
            "halted": torch.zeros(b, device=workspace.device, dtype=torch.bool),
            "steps": torch.zeros(b, device=workspace.device, dtype=torch.long),
        }

    @staticmethod
    def _empty_core_depth_states(workspace: torch.Tensor) -> torch.Tensor:
        b = workspace.shape[0]
        d = workspace.shape[-1]
        return workspace.new_empty((b, 0, d))

    def _empty_core_depth_last_logits(self, workspace: torch.Tensor) -> torch.Tensor:
        b = workspace.shape[0]
        return workspace.new_empty((b, 0, self.cfg.vocab_size))

    def _empty_core_depth_text_logits(
        self,
        workspace: torch.Tensor,
        input_seq_len: int,
    ) -> torch.Tensor:
        b = workspace.shape[0]
        return workspace.new_empty((b, 0, int(input_seq_len), self.cfg.vocab_size))

    def _compute_transition_state_outputs(
        self,
        core_depth_states: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if self.transition_state_predictor is None:
            empty_logits = core_depth_states.new_empty((b, 0, 0))
            return {"logits": empty_logits, "features": empty_logits}
        state_dim = int(self.transition_state_predictor.state_dim)
        if disabled or steps == 0:
            logits = core_depth_states.new_zeros((b, steps, state_dim))
            return {"logits": logits, "features": logits}
        outputs = self.transition_state_predictor(core_depth_states)
        return {
            "logits": outputs["transition_state_logits"],
            "features": outputs["transition_state_features"],
        }

    def _compute_transition_state_text_logits(
        self,
        transition_state_features: torch.Tensor,
    ) -> torch.Tensor:
        b = transition_state_features.shape[0]
        steps = transition_state_features.shape[1]
        if self.transition_state_to_answer is None or transition_state_features.numel() == 0:
            return transition_state_features.new_empty((b, steps, self.cfg.vocab_size))
        hidden = self.transition_state_to_answer(transition_state_features)
        return self.lm_head(hidden) * float(self.cfg.qtrm_logits_scale)

    def _compute_transition_state_code_outputs(
        self,
        core_depth_states: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.transition_state_code_norm is None
            or self.transition_state_code_head is None
            or self.transition_state_code_embed is None
        ):
            return {
                "logits": core_depth_states.new_empty((b, 0, 0)),
                "embeddings": core_depth_states.new_empty((b, 0, self.cfg.d_model)),
            }
        codebook_size = int(self.transition_state_code_head.out_features)
        if disabled or steps == 0:
            return {
                "logits": core_depth_states.new_zeros((b, steps, codebook_size)),
                "embeddings": core_depth_states.new_zeros((b, steps, self.cfg.d_model)),
            }
        logits = self.transition_state_code_head(
            self.transition_state_code_norm(core_depth_states)
        )
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=core_depth_states.dtype)
        embeddings = weights @ self.transition_state_code_embed.weight.to(
            dtype=core_depth_states.dtype,
            device=core_depth_states.device,
        )
        return {"logits": logits, "embeddings": embeddings}

    def _compute_transition_state_finality_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.transition_state_finality_norm is None
            or self.transition_state_finality_head is None
        ):
            return core_depth_states.new_empty((b, 0))
        if disabled or steps == 0:
            return core_depth_states.new_zeros((b, steps))
        logits = self.transition_state_finality_head(
            self.transition_state_finality_norm(core_depth_states)
        )
        return logits.squeeze(-1)

    def _compute_primitive_transition_outputs(
        self,
        core_depth_states: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.primitive_transition_norm is None
            or self.primitive_transition_operation_head is None
        ):
            empty = core_depth_states.new_empty((b, 0, 0))
            return {"operation_logits": empty}
        num_operations = int(
            self.primitive_transition_operation_head[-1].out_features
        )
        if disabled or steps == 0:
            logits = core_depth_states.new_zeros((b, steps, num_operations))
            return {"operation_logits": logits}
        features = self.primitive_transition_norm(core_depth_states)
        if self.primitive_transition_prompt_norm is not None:
            if (
                self.primitive_transition_prompt_cross is not None
                and self.primitive_transition_prompt_query_norm is not None
                and self.primitive_transition_prompt_context_norm is not None
                and prompt_context_seq is not None
            ):
                prompt_context = self.primitive_transition_prompt_cross(
                    self.primitive_transition_prompt_query_norm(core_depth_states),
                    self.primitive_transition_prompt_context_norm(prompt_context_seq),
                    prompt_context_mask,
                )
                prompt_context = self.primitive_transition_prompt_norm(prompt_context)
            elif prompt_context_seq is None:
                prompt_context = core_depth_states.new_zeros((b, self.cfg.d_model))
            else:
                if prompt_context_mask is None:
                    prompt_context = prompt_context_seq.mean(dim=1)
                else:
                    prompt_mask = prompt_context_mask.to(
                        device=prompt_context_seq.device,
                        dtype=prompt_context_seq.dtype,
                    ).unsqueeze(-1)
                    denom = prompt_mask.sum(dim=1).clamp_min(1.0)
                    prompt_context = (prompt_context_seq * prompt_mask).sum(dim=1) / denom
                prompt_context = self.primitive_transition_prompt_norm(prompt_context)
                prompt_context = prompt_context.unsqueeze(1).expand(-1, steps, -1)
            features = torch.cat([features, prompt_context], dim=-1)
        logits = self.primitive_transition_operation_head(features)
        return {"operation_logits": logits}

    def _core_world_model_outputs(
        self,
        core_depth_states: torch.Tensor,
        *,
        actions: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        b, steps, _ = core_depth_states.shape
        latent_mask = torch.ones((b, steps), device=core_depth_states.device, dtype=torch.bool)
        if self.core_world_model is None or steps <= int(self.cfg.core_world_model_horizon):
            return {
                "pred": core_depth_states.new_empty((b, 0, self.cfg.d_model)),
                "target": core_depth_states.new_empty((b, 0, self.cfg.d_model)),
                "latents": core_depth_states,
                "latent_mask": latent_mask,
                "mask": torch.ones((b, 0), device=core_depth_states.device, dtype=torch.bool),
            }
        return self.core_world_model(
            core_depth_states,
            attention_mask=latent_mask,
            actions=actions,
        )

    @staticmethod
    def _core_depth_states(trajectory: list[torch.Tensor], workspace: torch.Tensor) -> torch.Tensor:
        if not trajectory:
            return QTRMMultimodalModel._empty_core_depth_states(workspace)
        return torch.stack([state[:, 0, :] for state in trajectory], dim=1)

    def _core_depth_last_logits(
        self,
        trajectory: list[torch.Tensor],
        *,
        text_context_seq: torch.Tensor,
        text_context_mask: torch.Tensor,
        workspace_mask: torch.Tensor,
        transition_state_features: Optional[torch.Tensor] = None,
        transition_state_code_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not trajectory:
            return self._empty_core_depth_last_logits(text_context_seq)
        if self.answer_state_loop_cross is not None:
            depth_logits = []
            for state_prefix in range(1, len(trajectory) + 1):
                prefix_logits, _, _ = self._compute_answer_state_loop_outputs(
                    text_context_seq,
                    trajectory=trajectory[:state_prefix],
                    workspace_mask=workspace_mask,
                    input_seq_len=1,
                    transition_state_features=(
                        transition_state_features[:, :state_prefix, :]
                        if transition_state_features is not None
                        and transition_state_features.numel() != 0
                        else transition_state_features
                    ),
                    transition_state_code_embeddings=(
                        transition_state_code_embeddings[:, :state_prefix, :]
                        if transition_state_code_embeddings is not None
                        and transition_state_code_embeddings.numel() != 0
                        else transition_state_code_embeddings
                    ),
                )
                depth_logits.append(prefix_logits[:, -1, :])
            return torch.stack(depth_logits, dim=1)
        depth_logits = []
        for state in trajectory:
            if self.core_loop_readout_cross is not None:
                logits, _ = self._compute_core_loop_readout_outputs(
                    text_context_seq,
                    z_h=state,
                    workspace_mask=workspace_mask,
                    input_seq_len=1,
                )
                last_logits = logits[:, -1, :]
            else:
                seq = torch.cat([state, text_context_seq], dim=1)
                attention_mask = torch.cat([workspace_mask, text_context_mask], dim=1)
                hidden = self.coda(seq, attention_mask=attention_mask)
                hidden = self.norm(hidden)
                last_logits = self.lm_head(hidden[:, -1, :]) * float(self.cfg.qtrm_logits_scale)
            depth_logits.append(last_logits)
        return torch.stack(depth_logits, dim=1)

    def _core_depth_text_logits(
        self,
        trajectory: list[torch.Tensor],
        *,
        text_context_seq: torch.Tensor,
        text_context_mask: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
        transition_state_features: Optional[torch.Tensor] = None,
        transition_state_code_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not trajectory:
            return self._empty_core_depth_text_logits(text_context_seq, input_seq_len)
        if self.answer_state_loop_cross is not None:
            _, _, depth_hidden = self._compute_answer_state_loop_outputs(
                text_context_seq,
                trajectory=trajectory,
                workspace_mask=workspace_mask,
                input_seq_len=input_seq_len,
                transition_state_features=transition_state_features,
                transition_state_code_embeddings=transition_state_code_embeddings,
            )
            return self.lm_head(depth_hidden) * float(self.cfg.qtrm_logits_scale)
        depth_logits = []
        input_seq_len = int(input_seq_len)
        for state in trajectory:
            if self.core_loop_readout_cross is not None:
                logits, _ = self._compute_core_loop_readout_outputs(
                    text_context_seq,
                    z_h=state,
                    workspace_mask=workspace_mask,
                    input_seq_len=input_seq_len,
                )
                depth_logits.append(logits)
            else:
                seq = torch.cat([state, text_context_seq], dim=1)
                attention_mask = torch.cat([workspace_mask, text_context_mask], dim=1)
                hidden = self.coda(seq, attention_mask=attention_mask)
                hidden = self.norm(hidden)
                text_hidden = hidden[:, -input_seq_len:, :]
                depth_logits.append(self.lm_head(text_hidden) * float(self.cfg.qtrm_logits_scale))
        return torch.stack(depth_logits, dim=1)
