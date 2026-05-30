from __future__ import annotations
import math
from typing import Optional
import torch
from torch import nn

from .config import QTRMConfig
from .blocks import QTRMBlockStack
from .workspace import LatentWorkspace
from .multimodal_projector import MultimodalProjector
from .core import QTRMCoreCarry, QTRMRecursiveCore
from .heads import ControllerHeads
from .norm import RMSNorm
from .attention import CrossAttention
from .world_model import JepaWorldModelHead, SIGReg
from .agentic.transition_controller import TransitionStatePredictor


def _top_logit_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] < 2:
        return logits.squeeze(-1)
    top2 = logits.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def compute_donor_qtrm_conflict_gate(
    qtrm_text_logits: torch.Tensor,
    donor_text_logits: torch.Tensor,
    *,
    enabled: bool,
    mode: str,
    conflict_scale: float,
    boost_scale: float,
    margin_threshold: float,
) -> torch.Tensor:
    gate = qtrm_text_logits.new_ones(qtrm_text_logits.shape[:2])
    if not enabled:
        return gate

    safe_conflict_scale = min(max(float(conflict_scale), 0.0), 1.0)
    donor_top = donor_text_logits.argmax(dim=-1)
    qtrm_top = qtrm_text_logits.argmax(dim=-1)
    conflict = donor_top != qtrm_top
    mode = str(mode or "downscale")

    if mode in {"downscale", "legacy"}:
        return torch.where(
            conflict,
            gate.new_full(gate.shape, safe_conflict_scale),
            gate,
        )

    if mode in {"adaptive_margin", "margin"}:
        qtrm_margin = _top_logit_margin(qtrm_text_logits)
        donor_margin = _top_logit_margin(donor_text_logits)
        qtrm_wins = qtrm_margin >= donor_margin + float(margin_threshold)
        conflict_gate = torch.where(
            qtrm_wins,
            gate.new_full(gate.shape, max(float(boost_scale), 0.0)),
            gate.new_full(gate.shape, safe_conflict_scale),
        )
        return torch.where(conflict, conflict_gate, gate)

    raise ValueError(f"unknown donor_qtrm_conflict_gate_mode: {mode}")


class QTRMMultimodalModel(nn.Module):
    """Standalone multimodal QTRM model."""

    def __init__(self, cfg: QTRMConfig):
        super().__init__()
        self.cfg = cfg
        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)
        self.text_position_embed = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model)
            if cfg.text_position_embed_enabled
            else None
        )
        if self.text_position_embed is not None:
            nn.init.normal_(self.text_position_embed.weight, mean=0.0, std=0.02)
        self.token_numeric_value_embed = (
            nn.Embedding(int(cfg.token_numeric_value_vocab_size), cfg.d_model)
            if bool(cfg.token_numeric_value_embedding_enabled)
            else None
        )
        self.token_numeric_value_gate = (
            nn.Parameter(torch.tensor(float(cfg.token_numeric_value_gate_init_bias)))
            if bool(cfg.token_numeric_value_embedding_enabled)
            else None
        )
        if self.token_numeric_value_embed is not None:
            nn.init.normal_(self.token_numeric_value_embed.weight, mean=0.0, std=0.02)
        self.token_numeric_source_slot_embed = (
            nn.Embedding(int(cfg.token_numeric_source_slot_vocab_size), cfg.d_model)
            if bool(cfg.token_numeric_source_slot_embedding_enabled)
            else None
        )
        self.token_numeric_source_slot_pos = (
            nn.Embedding(int(cfg.token_numeric_source_slot_max_slots), cfg.d_model)
            if bool(cfg.token_numeric_source_slot_embedding_enabled)
            else None
        )
        self.token_numeric_source_slot_gate = (
            nn.Parameter(
                torch.tensor(float(cfg.token_numeric_source_slot_gate_init_bias))
            )
            if bool(cfg.token_numeric_source_slot_embedding_enabled)
            else None
        )
        self.token_numeric_source_slot_parity_head = (
            nn.Linear(cfg.d_model, 2)
            if bool(cfg.token_numeric_source_slot_embedding_enabled)
            else None
        )
        source_slot_predicate_feedback_enabled = (
            bool(cfg.token_numeric_source_slot_embedding_enabled)
            and bool(cfg.token_numeric_source_slot_predicate_feedback_enabled)
        )
        self.token_numeric_source_slot_predicate_head = (
            nn.Linear(cfg.d_model, 2)
            if source_slot_predicate_feedback_enabled
            else None
        )
        self.token_numeric_source_slot_predicate_embed = (
            nn.Embedding(2, cfg.d_model)
            if source_slot_predicate_feedback_enabled
            else None
        )
        self.token_numeric_source_slot_predicate_gate = (
            nn.Parameter(
                torch.tensor(
                    float(cfg.token_numeric_source_slot_predicate_gate_init_bias)
                )
            )
            if source_slot_predicate_feedback_enabled
            else None
        )
        if self.token_numeric_source_slot_embed is not None:
            nn.init.normal_(
                self.token_numeric_source_slot_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.token_numeric_source_slot_pos is not None:
            nn.init.normal_(
                self.token_numeric_source_slot_pos.weight,
                mean=0.0,
                std=0.02,
            )
        if self.token_numeric_source_slot_predicate_head is not None:
            nn.init.xavier_uniform_(self.token_numeric_source_slot_predicate_head.weight)
            nn.init.zeros_(self.token_numeric_source_slot_predicate_head.bias)
        if self.token_numeric_source_slot_predicate_embed is not None:
            nn.init.normal_(
                self.token_numeric_source_slot_predicate_embed.weight,
                mean=0.0,
                std=0.02,
            )
        self.prelude = QTRMBlockStack(cfg, cfg.n_prelude_layers, causal=True, attn_every=cfg.attn_every)
        self.jepa_encoder = QTRMBlockStack(cfg, cfg.jepa_encoder_layers, causal=True, attn_every=cfg.attn_every)
        self.jepa_encoder_norm = RMSNorm(cfg.d_model)
        self.core = QTRMRecursiveCore(cfg)
        self.core_depth_readout_query = (
            nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
            if cfg.core_depth_readout_enabled
            else None
        )
        self.core_depth_readout_query_norm = (
            RMSNorm(cfg.d_model) if cfg.core_depth_readout_enabled else None
        )
        self.core_depth_readout_state_norm = (
            RMSNorm(cfg.d_model) if cfg.core_depth_readout_enabled else None
        )
        self.core_depth_readout_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.core_depth_readout_enabled
            else None
        )
        self.core_depth_readout_output_norm = (
            RMSNorm(cfg.d_model) if cfg.core_depth_readout_enabled else None
        )
        order_bottleneck_enabled = bool(cfg.core_transition_order_bottleneck_enabled)
        order_bottleneck_classes = max(
            1,
            int(cfg.core_transition_order_bottleneck_num_classes),
        )
        order_bottleneck_hidden_dim = int(
            cfg.core_transition_order_bottleneck_hidden_dim or cfg.d_model
        )
        self.core_transition_order_bottleneck_query = (
            nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            if order_bottleneck_enabled
            else None
        )
        self.core_transition_order_bottleneck_query_norm = (
            RMSNorm(cfg.d_model) if order_bottleneck_enabled else None
        )
        self.core_transition_order_bottleneck_context_norm = (
            RMSNorm(cfg.d_model) if order_bottleneck_enabled else None
        )
        self.core_transition_order_bottleneck_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if order_bottleneck_enabled
            else None
        )
        self.core_transition_order_bottleneck_head = (
            nn.Sequential(
                nn.Linear(cfg.d_model, order_bottleneck_hidden_dim),
                nn.GELU(),
                nn.Linear(order_bottleneck_hidden_dim, order_bottleneck_classes),
            )
            if order_bottleneck_enabled
            else None
        )
        self.core_transition_order_bottleneck_embed = (
            nn.Embedding(order_bottleneck_classes, cfg.d_model)
            if order_bottleneck_enabled
            else None
        )
        self.core_transition_order_bottleneck_gate = (
            nn.Linear(cfg.d_model, 1) if order_bottleneck_enabled else None
        )
        self.core_transition_order_bottleneck_output_norm = (
            RMSNorm(cfg.d_model) if order_bottleneck_enabled else None
        )
        if self.core_transition_order_bottleneck_query is not None:
            nn.init.normal_(
                self.core_transition_order_bottleneck_query,
                mean=0.0,
                std=0.02,
            )
        if self.core_transition_order_bottleneck_head is not None:
            for module in self.core_transition_order_bottleneck_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_transition_order_bottleneck_embed is not None:
            nn.init.normal_(
                self.core_transition_order_bottleneck_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_transition_order_bottleneck_gate is not None:
            nn.init.zeros_(self.core_transition_order_bottleneck_gate.weight)
            nn.init.constant_(
                self.core_transition_order_bottleneck_gate.bias,
                float(cfg.core_transition_order_bottleneck_gate_init_bias),
            )
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
        self.answer_state_loop_recurrent_norm = (
            RMSNorm(cfg.d_model)
            if (
                cfg.answer_state_loop_enabled
                and cfg.answer_state_loop_recurrent_block_enabled
            )
            else None
        )
        self.answer_state_loop_recurrent_stack = (
            QTRMBlockStack(
                cfg,
                max(1, int(cfg.answer_state_loop_recurrent_layers)),
                causal=True,
                attn_every=1,
            )
            if (
                cfg.answer_state_loop_enabled
                and cfg.answer_state_loop_recurrent_block_enabled
            )
            else None
        )

        # RI-4: Placeholder for hybrid block as the actual recurrent engine
        # (instead of side stack + residual). Set externally or in RI-4 setup.
        self.answer_state_loop_hybrid_recurrent_block = None
        self._ri4_hybrid_recurrent_slot_state = None  # carried across answer loop steps for RI-4 hybrid recurrent
        self.answer_state_loop_recurrent_gate = (
            nn.Linear(cfg.d_model, 1)
            if (
                cfg.answer_state_loop_enabled
                and cfg.answer_state_loop_recurrent_block_enabled
            )
            else None
        )
        self.answer_state_loop_halt_head = (
            nn.Linear(cfg.d_model, 1)
            if cfg.answer_state_loop_enabled and cfg.answer_state_loop_halt_enabled
            else None
        )
        mythos_update_enabled = (
            cfg.answer_state_loop_enabled
            and cfg.answer_state_loop_recurrent_block_enabled
            and bool(cfg.answer_state_loop_mythos_update_enabled)
        )
        self.answer_state_loop_mythos_log_A = (
            nn.Parameter(torch.zeros(cfg.d_model)) if mythos_update_enabled else None
        )
        self.answer_state_loop_mythos_log_dt = (
            nn.Parameter(torch.full((1,), float(cfg.answer_state_loop_mythos_log_dt_init)))
            if mythos_update_enabled
            else None
        )
        self.answer_state_loop_mythos_input_B = (
            nn.Parameter(
                torch.full(
                    (cfg.d_model,),
                    float(cfg.answer_state_loop_mythos_input_injection_init),
                )
            )
            if mythos_update_enabled
            else None
        )
        mythos_loop_count = max(
            1,
            int(cfg.outer_steps),
            int(cfg.core_step_conditioning_max_steps),
        )
        self.answer_state_loop_mythos_loop_index = (
            nn.Embedding(mythos_loop_count, cfg.d_model)
            if mythos_update_enabled and bool(cfg.answer_state_loop_mythos_loop_index_enabled)
            else None
        )
        mythos_lora_rank = max(0, int(cfg.answer_state_loop_mythos_lora_rank))
        self.answer_state_loop_mythos_lora_down = (
            nn.Linear(cfg.d_model, mythos_lora_rank, bias=False)
            if mythos_update_enabled and mythos_lora_rank > 0
            else None
        )
        self.answer_state_loop_mythos_lora_up = (
            nn.Linear(mythos_lora_rank, cfg.d_model, bias=False)
            if mythos_update_enabled and mythos_lora_rank > 0
            else None
        )
        self.answer_state_loop_mythos_lora_scale = (
            nn.Embedding(mythos_loop_count, mythos_lora_rank)
            if mythos_update_enabled and mythos_lora_rank > 0
            else None
        )
        answer_lm_adapter_enabled = (
            cfg.answer_state_loop_enabled
            and cfg.answer_state_loop_lm_adapter_enabled
            and int(cfg.answer_state_loop_lm_adapter_rank) > 0
        )
        answer_lm_adapter_rank = max(1, int(cfg.answer_state_loop_lm_adapter_rank))
        self.answer_state_loop_lm_adapter_down = (
            nn.Linear(cfg.d_model, answer_lm_adapter_rank, bias=False)
            if answer_lm_adapter_enabled
            else None
        )
        self.answer_state_loop_lm_adapter_up = (
            nn.Linear(answer_lm_adapter_rank, cfg.vocab_size, bias=False)
            if answer_lm_adapter_enabled
            else None
        )
        hidden_bridge_enabled = (
            cfg.answer_state_loop_enabled
            and bool(cfg.answer_state_loop_hidden_bridge_enabled)
        )
        hidden_bridge_dim = int(
            cfg.answer_state_loop_hidden_bridge_hidden_dim or cfg.d_model
        )
        self.answer_state_loop_hidden_bridge_norm = (
            RMSNorm(cfg.d_model) if hidden_bridge_enabled else None
        )
        self.answer_state_loop_hidden_bridge_down = (
            nn.Linear(cfg.d_model, hidden_bridge_dim)
            if hidden_bridge_enabled
            else None
        )
        self.answer_state_loop_hidden_bridge_up = (
            nn.Linear(hidden_bridge_dim, cfg.d_model)
            if hidden_bridge_enabled
            else None
        )
        free_transformer_latent_enabled = (
            cfg.answer_state_loop_enabled
            and bool(cfg.answer_state_loop_free_transformer_latent_enabled)
        )
        free_transformer_latent_dim = int(
            cfg.answer_state_loop_free_transformer_latent_dim or cfg.d_model
        )
        self.answer_state_loop_free_transformer_prior_norm = (
            RMSNorm(cfg.d_model) if free_transformer_latent_enabled else None
        )
        self.answer_state_loop_free_transformer_posterior_norm = (
            RMSNorm(cfg.d_model) if free_transformer_latent_enabled else None
        )
        self.answer_state_loop_free_transformer_prior_mu = (
            nn.Linear(cfg.d_model, free_transformer_latent_dim)
            if free_transformer_latent_enabled
            else None
        )
        self.answer_state_loop_free_transformer_prior_logvar = (
            nn.Linear(cfg.d_model, free_transformer_latent_dim)
            if free_transformer_latent_enabled
            else None
        )
        self.answer_state_loop_free_transformer_posterior_mu = (
            nn.Linear(cfg.d_model, free_transformer_latent_dim)
            if free_transformer_latent_enabled
            else None
        )
        self.answer_state_loop_free_transformer_posterior_logvar = (
            nn.Linear(cfg.d_model, free_transformer_latent_dim)
            if free_transformer_latent_enabled
            else None
        )
        self.answer_state_loop_free_transformer_latent_up = (
            nn.Linear(free_transformer_latent_dim, cfg.d_model)
            if free_transformer_latent_enabled
            else None
        )
        self.answer_state_loop_free_transformer_gate = (
            nn.Linear(cfg.d_model, 1) if free_transformer_latent_enabled else None
        )
        next_token_decoder_enabled = (
            cfg.answer_state_loop_enabled
            and bool(cfg.answer_state_loop_next_token_decoder_enabled)
            and int(cfg.answer_state_loop_next_token_decoder_layers) > 0
        )
        self.answer_state_loop_next_token_decoder_norm = (
            RMSNorm(cfg.d_model) if next_token_decoder_enabled else None
        )
        self.answer_state_loop_next_token_decoder_stack = (
            QTRMBlockStack(
                cfg,
                max(1, int(cfg.answer_state_loop_next_token_decoder_layers)),
                causal=True,
                attn_every=1,
            )
            if next_token_decoder_enabled
            else None
        )
        self.answer_state_loop_next_token_decoder_gate = (
            nn.Linear(cfg.d_model, 1) if next_token_decoder_enabled else None
        )
        prev_token_decoder_enabled = (
            next_token_decoder_enabled
            and bool(cfg.answer_state_loop_next_token_decoder_prev_token_enabled)
        )
        self.answer_state_loop_next_token_decoder_prev_token_norm = (
            RMSNorm(cfg.d_model) if prev_token_decoder_enabled else None
        )
        self.answer_state_loop_next_token_decoder_prev_token_fuse = (
            nn.Linear(cfg.d_model * 2, cfg.d_model)
            if prev_token_decoder_enabled
            else None
        )
        self.answer_state_loop_next_token_decoder_prev_token_gate = (
            nn.Linear(cfg.d_model, 1) if prev_token_decoder_enabled else None
        )
        future_token_decoder_enabled = (
            cfg.answer_state_loop_enabled
            and bool(cfg.answer_state_loop_future_token_decoder_enabled)
            and int(cfg.answer_state_loop_future_token_max_tokens) > 0
        )
        self.answer_state_loop_future_token_positions = (
            nn.Parameter(
                torch.empty(
                    int(cfg.answer_state_loop_future_token_max_tokens),
                    cfg.d_model,
                )
            )
            if future_token_decoder_enabled
            else None
        )
        talker_enabled = (
            cfg.answer_state_loop_enabled
            and bool(cfg.answer_state_loop_talker_enabled)
            and int(cfg.answer_state_loop_talker_layers) > 0
        )
        self.answer_state_loop_talker_norm = (
            RMSNorm(cfg.d_model) if talker_enabled else None
        )
        self.answer_state_loop_talker_stack = (
            QTRMBlockStack(
                cfg,
                max(1, int(cfg.answer_state_loop_talker_layers)),
                causal=True,
                attn_every=1,
            )
            if talker_enabled
            else None
        )
        self.answer_state_loop_talker_gate = (
            nn.Linear(cfg.d_model, 1) if talker_enabled else None
        )
        selective_context_enabled = (
            cfg.answer_state_loop_enabled
            and cfg.answer_state_loop_selective_context_enabled
        )
        self.answer_state_loop_selective_query = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if selective_context_enabled
            else None
        )
        self.answer_state_loop_selective_key = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if selective_context_enabled
            else None
        )
        if self.answer_state_loop_gate is not None:
            nn.init.zeros_(self.answer_state_loop_gate.weight)
            nn.init.constant_(
                self.answer_state_loop_gate.bias,
                float(cfg.answer_state_loop_gate_init_bias),
            )
        if self.answer_state_loop_recurrent_gate is not None:
            nn.init.zeros_(self.answer_state_loop_recurrent_gate.weight)
            nn.init.constant_(
                self.answer_state_loop_recurrent_gate.bias,
                float(cfg.answer_state_loop_recurrent_gate_init_bias),
            )
        if self.answer_state_loop_halt_head is not None:
            nn.init.zeros_(self.answer_state_loop_halt_head.weight)
            nn.init.constant_(
                self.answer_state_loop_halt_head.bias,
                float(cfg.answer_state_loop_halt_init_bias),
            )
        if self.answer_state_loop_mythos_loop_index is not None:
            nn.init.normal_(
                self.answer_state_loop_mythos_loop_index.weight,
                mean=0.0,
                std=0.02,
            )
        if self.answer_state_loop_mythos_lora_down is not None:
            nn.init.xavier_uniform_(self.answer_state_loop_mythos_lora_down.weight)
        if self.answer_state_loop_mythos_lora_up is not None:
            nn.init.zeros_(self.answer_state_loop_mythos_lora_up.weight)
        if self.answer_state_loop_mythos_lora_scale is not None:
            nn.init.ones_(self.answer_state_loop_mythos_lora_scale.weight)
        if self.answer_state_loop_lm_adapter_down is not None:
            nn.init.xavier_uniform_(self.answer_state_loop_lm_adapter_down.weight)
        if self.answer_state_loop_lm_adapter_up is not None:
            nn.init.zeros_(self.answer_state_loop_lm_adapter_up.weight)
        if self.answer_state_loop_hidden_bridge_down is not None:
            nn.init.xavier_uniform_(self.answer_state_loop_hidden_bridge_down.weight)
            nn.init.zeros_(self.answer_state_loop_hidden_bridge_down.bias)
        if self.answer_state_loop_hidden_bridge_up is not None:
            nn.init.zeros_(self.answer_state_loop_hidden_bridge_up.weight)
            nn.init.zeros_(self.answer_state_loop_hidden_bridge_up.bias)
        if self.answer_state_loop_next_token_decoder_gate is not None:
            nn.init.zeros_(self.answer_state_loop_next_token_decoder_gate.weight)
            nn.init.constant_(
                self.answer_state_loop_next_token_decoder_gate.bias,
                float(cfg.answer_state_loop_next_token_decoder_gate_init_bias),
            )
        if self.answer_state_loop_next_token_decoder_prev_token_fuse is not None:
            nn.init.zeros_(
                self.answer_state_loop_next_token_decoder_prev_token_fuse.weight
            )
            nn.init.zeros_(
                self.answer_state_loop_next_token_decoder_prev_token_fuse.bias
            )
            with torch.no_grad():
                eye = torch.eye(
                    cfg.d_model,
                    device=self.answer_state_loop_next_token_decoder_prev_token_fuse.weight.device,
                    dtype=self.answer_state_loop_next_token_decoder_prev_token_fuse.weight.dtype,
                )
                self.answer_state_loop_next_token_decoder_prev_token_fuse.weight[
                    :, : cfg.d_model
                ].copy_(eye)
        if self.answer_state_loop_next_token_decoder_prev_token_gate is not None:
            nn.init.zeros_(
                self.answer_state_loop_next_token_decoder_prev_token_gate.weight
            )
            nn.init.constant_(
                self.answer_state_loop_next_token_decoder_prev_token_gate.bias,
                float(
                    cfg.answer_state_loop_next_token_decoder_prev_token_gate_init_bias
                ),
            )
        if self.answer_state_loop_free_transformer_gate is not None:
            nn.init.zeros_(self.answer_state_loop_free_transformer_gate.weight)
            nn.init.constant_(
                self.answer_state_loop_free_transformer_gate.bias,
                float(cfg.answer_state_loop_free_transformer_gate_init_bias),
            )
        if self.answer_state_loop_future_token_positions is not None:
            nn.init.normal_(
                self.answer_state_loop_future_token_positions,
                mean=0.0,
                std=0.02,
            )
        if self.answer_state_loop_talker_gate is not None:
            nn.init.zeros_(self.answer_state_loop_talker_gate.weight)
            nn.init.constant_(
                self.answer_state_loop_talker_gate.bias,
                float(cfg.answer_state_loop_talker_gate_init_bias),
            )
        if self.answer_state_loop_selective_query is not None:
            nn.init.xavier_uniform_(self.answer_state_loop_selective_query.weight)
        if self.answer_state_loop_selective_key is not None:
            nn.init.xavier_uniform_(self.answer_state_loop_selective_key.weight)
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
        transition_state_joint_size = max(1, int(cfg.transition_state_joint_size))
        self.transition_state_joint_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.transition_state_joint_enabled else None
        )
        self.transition_state_joint_head = (
            nn.Linear(cfg.d_model, transition_state_joint_size)
            if cfg.transition_state_joint_enabled
            else None
        )
        transition_state_joint_prompt_context_enabled = (
            cfg.transition_state_joint_enabled
            and cfg.transition_state_joint_prompt_context_enabled
        )
        self.transition_state_joint_prompt_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if transition_state_joint_prompt_context_enabled
            else None
        )
        self.transition_state_joint_prompt_context_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if transition_state_joint_prompt_context_enabled
            else None
        )
        transition_state_joint_prompt_token_attention_enabled = (
            transition_state_joint_prompt_context_enabled
            and cfg.transition_state_joint_prompt_token_attention_enabled
        )
        self.transition_state_joint_prompt_query_norm = (
            nn.LayerNorm(cfg.d_model)
            if transition_state_joint_prompt_token_attention_enabled
            else None
        )
        self.transition_state_joint_prompt_token_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if transition_state_joint_prompt_token_attention_enabled
            else None
        )
        self.transition_state_joint_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if transition_state_joint_prompt_token_attention_enabled
            else None
        )
        self.transition_state_joint_prompt_cross_norm = (
            nn.LayerNorm(cfg.d_model)
            if transition_state_joint_prompt_token_attention_enabled
            else None
        )
        self.transition_state_joint_prompt_cross_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if transition_state_joint_prompt_token_attention_enabled
            else None
        )
        self.transition_state_joint_operation_residual = (
            nn.Linear(
                max(1, int(cfg.primitive_transition_num_operations)),
                transition_state_joint_size,
                bias=False,
            )
            if cfg.transition_state_joint_enabled
            and cfg.transition_state_joint_operation_residual_enabled
            and cfg.primitive_transition_enabled
            and int(cfg.primitive_transition_num_operations) > 0
            else None
        )
        self.transition_state_joint_code_residual = (
            nn.Linear(
                max(1, int(cfg.transition_state_codebook_size)),
                transition_state_joint_size,
                bias=False,
            )
            if cfg.transition_state_joint_enabled
            and cfg.transition_state_joint_code_residual_enabled
            and cfg.transition_state_code_enabled
            and int(cfg.transition_state_codebook_size) > 0
            else None
        )
        transition_phase_num_classes = max(1, int(cfg.transition_phase_num_classes))
        transition_phase_hidden_dim = int(
            cfg.transition_phase_hidden_dim or cfg.d_model
        )
        self.transition_phase_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.transition_phase_enabled else None
        )
        self.transition_phase_prompt_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            else None
        )
        self.transition_phase_prompt_query_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_prompt_token_attention_enabled
            else None
        )
        self.transition_phase_prompt_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_prompt_token_attention_enabled
            else None
        )
        self.transition_phase_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_prompt_token_attention_enabled
            else None
        )
        self.transition_phase_global_query = (
            nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_global_prompt_query_enabled
            else None
        )
        self.transition_phase_global_query_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_global_prompt_query_enabled
            else None
        )
        self.transition_phase_global_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_global_prompt_query_enabled
            else None
        )
        self.transition_phase_global_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_global_prompt_query_enabled
            else None
        )
        self.transition_phase_global_cross_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_phase_enabled
            and cfg.transition_phase_prompt_context_enabled
            and cfg.transition_phase_global_prompt_query_enabled
            else None
        )
        transition_phase_context_multiplier = 0
        if cfg.transition_phase_prompt_context_enabled:
            transition_phase_context_multiplier = (
                2 if cfg.transition_phase_prompt_token_attention_enabled else 1
            )
            if cfg.transition_phase_global_prompt_query_enabled:
                transition_phase_context_multiplier += 1
        transition_phase_input_dim = cfg.d_model * (
            1 + transition_phase_context_multiplier
        )
        self.transition_phase_head = (
            nn.Sequential(
                nn.Linear(transition_phase_input_dim, transition_phase_hidden_dim),
                nn.GELU(),
                nn.Linear(transition_phase_hidden_dim, transition_phase_num_classes),
            )
            if cfg.transition_phase_enabled
            else None
        )
        self.transition_state_joint_phase_residual = (
            nn.Sequential(
                nn.Linear(
                    cfg.d_model + transition_phase_num_classes,
                    transition_phase_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(transition_phase_hidden_dim, transition_state_joint_size),
            )
            if cfg.transition_state_joint_enabled
            and cfg.transition_state_joint_phase_residual_enabled
            and cfg.transition_phase_enabled
            else None
        )
        transition_state_joint_answer_enabled = (
            cfg.answer_state_loop_enabled
            and cfg.transition_state_joint_enabled
            and cfg.transition_state_joint_answer_bridge_enabled
        )
        self.transition_state_joint_answer_proj = (
            nn.Linear(transition_state_joint_size, cfg.d_model, bias=False)
            if transition_state_joint_answer_enabled
            else None
        )
        self.transition_state_joint_answer_gate = (
            nn.Linear(cfg.d_model, 1)
            if transition_state_joint_answer_enabled
            else None
        )
        transition_state_final_answer_enabled = (
            cfg.answer_state_loop_enabled
            and cfg.transition_state_joint_enabled
            and cfg.transition_state_final_answer_binder_enabled
        )
        self.transition_state_final_answer_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if transition_state_final_answer_enabled
            else None
        )
        self.transition_state_final_answer_gate = (
            nn.Linear(cfg.d_model, 1)
            if transition_state_final_answer_enabled
            else None
        )
        if self.transition_state_joint_head is not None:
            nn.init.xavier_uniform_(self.transition_state_joint_head.weight)
            nn.init.zeros_(self.transition_state_joint_head.bias)
        if self.transition_state_joint_prompt_context_proj is not None:
            nn.init.zeros_(self.transition_state_joint_prompt_context_proj.weight)
        if self.transition_state_joint_prompt_cross_proj is not None:
            nn.init.zeros_(self.transition_state_joint_prompt_cross_proj.weight)
        if self.transition_state_joint_operation_residual is not None:
            nn.init.zeros_(self.transition_state_joint_operation_residual.weight)
        if self.transition_state_joint_code_residual is not None:
            nn.init.zeros_(self.transition_state_joint_code_residual.weight)
        if self.transition_phase_head is not None:
            for module in self.transition_phase_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.transition_phase_global_query is not None:
            nn.init.normal_(self.transition_phase_global_query, mean=0.0, std=0.02)
        if self.transition_state_joint_phase_residual is not None:
            for module in self.transition_state_joint_phase_residual:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            final_linear = self.transition_state_joint_phase_residual[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
        if self.transition_state_joint_answer_proj is not None:
            nn.init.xavier_uniform_(self.transition_state_joint_answer_proj.weight)
        if self.transition_state_joint_answer_gate is not None:
            nn.init.zeros_(self.transition_state_joint_answer_gate.weight)
            nn.init.constant_(
                self.transition_state_joint_answer_gate.bias,
                float(cfg.transition_state_joint_answer_gate_init_bias),
            )
        if self.transition_state_final_answer_proj is not None:
            nn.init.xavier_uniform_(self.transition_state_final_answer_proj.weight)
        if self.transition_state_final_answer_gate is not None:
            nn.init.zeros_(self.transition_state_final_answer_gate.weight)
            nn.init.constant_(
                self.transition_state_final_answer_gate.bias,
                float(cfg.transition_state_final_answer_gate_init_bias),
            )
        transition_state_sequence_max_tokens = max(
            1,
            int(cfg.transition_state_sequence_max_tokens),
        )
        self.transition_state_sequence_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.transition_state_sequence_enabled else None
        )
        self.transition_state_sequence_pos_embed = (
            nn.Embedding(transition_state_sequence_max_tokens, cfg.d_model)
            if cfg.transition_state_sequence_enabled
            else None
        )
        self.transition_state_sequence_head = (
            nn.Linear(cfg.d_model, cfg.vocab_size)
            if cfg.transition_state_sequence_enabled
            else None
        )
        if self.transition_state_sequence_pos_embed is not None:
            nn.init.normal_(
                self.transition_state_sequence_pos_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.transition_state_sequence_head is not None:
            nn.init.xavier_uniform_(self.transition_state_sequence_head.weight)
            nn.init.zeros_(self.transition_state_sequence_head.bias)
        transition_value_state_max_tokens = max(
            1,
            int(cfg.transition_value_state_max_tokens),
        )
        transition_value_state_vocab_size = max(
            1,
            int(cfg.transition_value_state_vocab_size),
        )
        self.transition_value_state_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.transition_value_state_enabled else None
        )
        self.transition_value_state_pos_embed = (
            nn.Embedding(transition_value_state_max_tokens, cfg.d_model)
            if cfg.transition_value_state_enabled
            else None
        )
        self.transition_value_state_head = (
            nn.Linear(cfg.d_model, transition_value_state_vocab_size)
            if cfg.transition_value_state_enabled
            else None
        )
        if self.transition_value_state_pos_embed is not None:
            nn.init.normal_(
                self.transition_value_state_pos_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.transition_value_state_head is not None:
            nn.init.xavier_uniform_(self.transition_value_state_head.weight)
            nn.init.zeros_(self.transition_value_state_head.bias)
        factorized_value_state_max_tokens = max(
            1,
            int(cfg.factorized_value_state_max_tokens),
        )
        factorized_value_state_vocab_size = max(
            1,
            int(cfg.factorized_value_state_vocab_size),
        )
        factorized_value_state_kind_size = max(
            0,
            int(cfg.factorized_value_state_kind_size),
        )
        role_value_state_num_roles = max(0, int(cfg.role_value_state_num_roles))
        role_value_state_vocab_size = max(
            1,
            int(cfg.role_value_state_vocab_size or factorized_value_state_vocab_size),
        )
        factorized_value_state_hidden_dim = int(
            cfg.factorized_value_state_hidden_dim or cfg.d_model
        )
        self.factorized_value_state_init = (
            nn.Parameter(
                torch.randn(1, factorized_value_state_max_tokens, cfg.d_model) * 0.02
            )
            if cfg.factorized_value_state_enabled
            else None
        )
        self.factorized_value_state_step_embed = (
            nn.Embedding(max(1, int(cfg.outer_steps)), cfg.d_model)
            if cfg.factorized_value_state_enabled
            else None
        )
        self.factorized_value_state_action_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.factorized_value_state_enabled
            else None
        )
        self.factorized_value_state_prompt_query_norm = (
            RMSNorm(cfg.d_model) if cfg.factorized_value_state_enabled else None
        )
        self.factorized_value_state_prompt_context_norm = (
            RMSNorm(cfg.d_model) if cfg.factorized_value_state_enabled else None
        )
        self.factorized_value_state_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.factorized_value_state_enabled
            else None
        )
        self.factorized_value_state_update = (
            nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, factorized_value_state_hidden_dim),
                nn.GELU(),
                nn.Linear(factorized_value_state_hidden_dim, cfg.d_model),
            )
            if cfg.factorized_value_state_enabled
            else None
        )
        self.factorized_value_state_output_norm = (
            nn.LayerNorm(cfg.d_model) if cfg.factorized_value_state_enabled else None
        )
        self.factorized_value_state_head = (
            nn.Linear(cfg.d_model, factorized_value_state_vocab_size)
            if cfg.factorized_value_state_enabled
            else None
        )
        self.factorized_value_state_kind_head = (
            nn.Linear(cfg.d_model, factorized_value_state_kind_size)
            if cfg.factorized_value_state_enabled
            and factorized_value_state_kind_size > 0
            else None
        )
        self.role_value_state_role_embed = (
            nn.Embedding(role_value_state_num_roles, cfg.d_model)
            if cfg.factorized_value_state_enabled
            and cfg.role_value_state_enabled
            and role_value_state_num_roles > 0
            else None
        )
        self.role_value_state_query_norm = (
            RMSNorm(cfg.d_model)
            if self.role_value_state_role_embed is not None
            else None
        )
        self.role_value_state_slot_norm = (
            RMSNorm(cfg.d_model)
            if self.role_value_state_role_embed is not None
            else None
        )
        self.role_value_state_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if self.role_value_state_role_embed is not None
            else None
        )
        self.role_value_state_head = (
            nn.Linear(cfg.d_model, role_value_state_vocab_size)
            if self.role_value_state_role_embed is not None
            else None
        )
        core_role_value_state_num_roles = max(
            0,
            int(cfg.core_role_value_state_num_roles),
        )
        core_role_value_state_vocab_size = max(
            1,
            int(
                cfg.core_role_value_state_vocab_size
                or cfg.role_value_state_vocab_size
                or factorized_value_state_vocab_size
            ),
        )
        self.core_role_value_state_embed = (
            nn.Embedding(core_role_value_state_num_roles, cfg.d_model)
            if cfg.core_role_value_state_enabled
            and core_role_value_state_num_roles > 0
            else None
        )
        self.core_role_value_state_norm = (
            nn.LayerNorm(cfg.d_model)
            if self.core_role_value_state_embed is not None
            else None
        )
        self.core_role_value_state_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if self.core_role_value_state_embed is not None
            else None
        )
        core_role_answer_bridge_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_state_answer_bridge_enabled)
        )
        self.core_role_value_state_answer_value_embed = (
            nn.Embedding(core_role_value_state_vocab_size, cfg.d_model)
            if core_role_answer_bridge_enabled
            else None
        )
        self.core_role_value_state_answer_norm = (
            nn.LayerNorm(cfg.d_model) if core_role_answer_bridge_enabled else None
        )
        self.core_role_value_state_answer_gate = (
            nn.Linear(cfg.d_model, 1) if core_role_answer_bridge_enabled else None
        )
        core_role_answer_prompt_context_enabled = (
            core_role_answer_bridge_enabled
            and bool(cfg.core_role_value_state_answer_prompt_context_enabled)
        )
        self.core_role_value_state_answer_prompt_query_norm = (
            RMSNorm(cfg.d_model) if core_role_answer_prompt_context_enabled else None
        )
        self.core_role_value_state_answer_prompt_context_norm = (
            RMSNorm(cfg.d_model) if core_role_answer_prompt_context_enabled else None
        )
        self.core_role_value_state_answer_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if core_role_answer_prompt_context_enabled
            else None
        )
        self.core_role_value_state_answer_prompt_gate = (
            nn.Linear(cfg.d_model, 1) if core_role_answer_prompt_context_enabled else None
        )
        self.core_role_value_state_answer_prompt_output_norm = (
            RMSNorm(cfg.d_model) if core_role_answer_prompt_context_enabled else None
        )
        core_role_answer_final_binder_enabled = (
            core_role_answer_bridge_enabled
            and bool(cfg.core_role_value_state_answer_final_binder_enabled)
        )
        self.core_role_value_state_answer_final_proj = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if core_role_answer_final_binder_enabled
            else None
        )
        self.core_role_value_state_answer_final_gate = (
            nn.Linear(cfg.d_model, 1)
            if core_role_answer_final_binder_enabled
            else None
        )
        self.core_role_value_state_answer_final_norm = (
            RMSNorm(cfg.d_model) if core_role_answer_final_binder_enabled else None
        )
        core_role_vocab_renderer_enabled = (
            core_role_answer_bridge_enabled
            and bool(cfg.core_role_value_state_vocab_renderer_enabled)
        )
        self.core_role_value_state_vocab_renderer_query_norm = (
            RMSNorm(cfg.d_model) if core_role_vocab_renderer_enabled else None
        )
        self.core_role_value_state_vocab_renderer_state_norm = (
            RMSNorm(cfg.d_model) if core_role_vocab_renderer_enabled else None
        )
        self.core_role_value_state_vocab_renderer_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if core_role_vocab_renderer_enabled
            else None
        )
        self.core_role_value_state_vocab_renderer_gate = (
            nn.Linear(cfg.d_model, 1) if core_role_vocab_renderer_enabled else None
        )
        self.core_role_value_state_vocab_renderer_output_norm = (
            RMSNorm(cfg.d_model) if core_role_vocab_renderer_enabled else None
        )
        vocab_renderer_rank = max(
            1,
            int(cfg.core_role_value_state_vocab_renderer_rank),
        )
        self.core_role_value_state_vocab_renderer_down = (
            nn.Linear(cfg.d_model, vocab_renderer_rank, bias=False)
            if core_role_vocab_renderer_enabled
            else None
        )
        self.core_role_value_state_vocab_renderer_up = (
            nn.Linear(vocab_renderer_rank, cfg.vocab_size, bias=False)
            if core_role_vocab_renderer_enabled
            else None
        )
        core_role_prompt_extract_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_state_prompt_extract_enabled)
        )
        self.core_role_value_state_prompt_query_norm = (
            RMSNorm(cfg.d_model) if core_role_prompt_extract_enabled else None
        )
        self.core_role_value_state_prompt_context_norm = (
            RMSNorm(cfg.d_model) if core_role_prompt_extract_enabled else None
        )
        self.core_role_value_state_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if core_role_prompt_extract_enabled
            else None
        )
        self.core_role_value_state_prompt_gate = (
            nn.Linear(cfg.d_model, 1) if core_role_prompt_extract_enabled else None
        )
        self.core_role_value_state_prompt_output_norm = (
            RMSNorm(cfg.d_model) if core_role_prompt_extract_enabled else None
        )
        core_source_position_binder_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_source_position_binder_enabled)
        )
        core_source_position_binder_hidden_dim = int(
            cfg.core_source_position_binder_hidden_dim or cfg.d_model
        )
        core_source_position_binder_heads = int(
            cfg.core_source_position_binder_heads or cfg.n_heads
        )
        if (
            core_source_position_binder_heads <= 0
            or core_source_position_binder_hidden_dim
            % core_source_position_binder_heads
            != 0
        ):
            core_source_position_binder_heads = 1
        core_source_position_binder_max_positions = max(
            1,
            int(cfg.core_source_position_binder_max_positions or cfg.max_seq_len),
        )
        self.core_source_position_binder_input_proj = (
            nn.Linear(cfg.d_model, core_source_position_binder_hidden_dim)
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_position_embed = (
            nn.Embedding(
                core_source_position_binder_max_positions,
                core_source_position_binder_hidden_dim,
            )
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_slot_queries = (
            nn.Parameter(
                torch.randn(
                    core_role_value_state_num_roles,
                    core_source_position_binder_hidden_dim,
                )
                * 0.02
            )
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_logit_gate = (
            nn.Parameter(
                torch.full(
                    (1, 1, core_role_value_state_num_roles, 1),
                    float(cfg.core_source_position_binder_gate_init_bias),
                )
            )
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=core_source_position_binder_hidden_dim,
                    nhead=core_source_position_binder_heads,
                    dim_feedforward=core_source_position_binder_hidden_dim * 4,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=max(1, int(cfg.core_source_position_binder_layers)),
            )
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_head = (
            nn.Sequential(
                nn.LayerNorm(core_source_position_binder_hidden_dim),
                nn.Linear(
                    core_source_position_binder_hidden_dim,
                    core_source_position_binder_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    core_source_position_binder_hidden_dim,
                    core_role_value_state_vocab_size,
                ),
            )
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_value_embed = (
            nn.Embedding(core_role_value_state_vocab_size, cfg.d_model)
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_state_gate = (
            nn.Parameter(
                torch.full(
                    (1, core_role_value_state_num_roles, 1),
                    float(cfg.core_source_position_binder_state_gate_init_bias),
                )
            )
            if core_source_position_binder_enabled
            else None
        )
        self.core_source_position_binder_state_norm = (
            nn.LayerNorm(cfg.d_model) if core_source_position_binder_enabled else None
        )
        core_source_position_binder_query_state_enabled = (
            core_source_position_binder_enabled
            and bool(cfg.core_source_position_binder_query_state_enabled)
        )
        self.core_source_position_binder_query_state_proj = (
            nn.Linear(core_source_position_binder_hidden_dim, cfg.d_model)
            if core_source_position_binder_query_state_enabled
            else None
        )
        self.core_source_position_binder_query_state_gate = (
            nn.Parameter(
                torch.full(
                    (1, core_role_value_state_num_roles, 1),
                    float(cfg.core_source_position_binder_query_state_gate_init_bias),
                )
            )
            if core_source_position_binder_query_state_enabled
            else None
        )
        self.core_source_position_binder_query_state_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_source_position_binder_query_state_enabled
            else None
        )
        core_source_value_binder_enabled = (
            core_source_position_binder_enabled
            and bool(cfg.core_source_value_binder_enabled)
        )
        self.core_source_value_binder_head = (
            nn.Sequential(
                nn.LayerNorm(core_source_position_binder_hidden_dim),
                nn.Linear(
                    core_source_position_binder_hidden_dim,
                    core_source_position_binder_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    core_source_position_binder_hidden_dim,
                    core_role_value_state_vocab_size,
                ),
            )
            if core_source_value_binder_enabled
            else None
        )
        self.core_source_value_binder_logit_gate = (
            nn.Parameter(
                torch.full(
                    (1, 1, core_role_value_state_num_roles, 1),
                    float(cfg.core_source_value_binder_gate_init_bias),
                )
            )
            if core_source_value_binder_enabled
            else None
        )
        self.core_source_value_binder_value_embed = (
            nn.Embedding(core_role_value_state_vocab_size, cfg.d_model)
            if core_source_value_binder_enabled
            else None
        )
        self.core_source_value_binder_state_gate = (
            nn.Parameter(
                torch.full(
                    (1, core_role_value_state_num_roles, 1),
                    float(cfg.core_source_value_binder_state_gate_init_bias),
                )
            )
            if core_source_value_binder_enabled
            else None
        )
        self.core_source_value_binder_state_norm = (
            nn.LayerNorm(cfg.d_model) if core_source_value_binder_enabled else None
        )
        if self.core_source_position_binder_position_embed is not None:
            nn.init.normal_(
                self.core_source_position_binder_position_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_source_position_binder_head is not None:
            for module in self.core_source_position_binder_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_source_position_binder_value_embed is not None:
            nn.init.normal_(
                self.core_source_position_binder_value_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_source_position_binder_query_state_proj is not None:
            nn.init.xavier_uniform_(
                self.core_source_position_binder_query_state_proj.weight
            )
            nn.init.zeros_(self.core_source_position_binder_query_state_proj.bias)
        if self.core_source_value_binder_head is not None:
            for module in self.core_source_value_binder_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_source_value_binder_value_embed is not None:
            nn.init.normal_(
                self.core_source_value_binder_value_embed.weight,
                mean=0.0,
                std=0.02,
            )
        core_role_prompt_self_condition_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_state_prompt_self_condition_enabled)
        )
        self.core_role_value_state_prompt_self_condition_value_embed = (
            nn.Embedding(core_role_value_state_vocab_size, cfg.d_model)
            if core_role_prompt_self_condition_enabled
            else None
        )
        self.core_role_value_state_prompt_self_condition_gate = (
            nn.Linear(cfg.d_model, 1)
            if core_role_prompt_self_condition_enabled
            else None
        )
        self.core_role_value_state_prompt_self_condition_norm = (
            RMSNorm(cfg.d_model) if core_role_prompt_self_condition_enabled else None
        )
        self.core_role_value_state_prompt_self_condition_output_norm = (
            RMSNorm(cfg.d_model) if core_role_prompt_self_condition_enabled else None
        )
        core_role_prompt_parity_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_state_prompt_parity_enabled)
        )
        self.core_role_value_state_prompt_parity_norm = (
            RMSNorm(cfg.d_model) if core_role_prompt_parity_enabled else None
        )
        self.core_role_value_state_prompt_parity_head = (
            nn.Linear(cfg.d_model, 2) if core_role_prompt_parity_enabled else None
        )
        self.core_role_value_state_prompt_parity_embed = (
            nn.Embedding(2, cfg.d_model) if core_role_prompt_parity_enabled else None
        )
        self.core_role_value_state_prompt_parity_gate = (
            nn.Linear(cfg.d_model, 1) if core_role_prompt_parity_enabled else None
        )
        core_role_value_template_num_templates = max(
            0,
            int(cfg.core_role_value_template_num_templates),
        )
        core_role_value_template_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_template_codec_enabled)
            and core_role_value_template_num_templates > 0
        )
        core_role_value_template_hidden_dim = int(
            cfg.core_role_value_template_hidden_dim or cfg.d_model
        )
        self.core_role_value_template_context_norm = (
            nn.LayerNorm(cfg.d_model) if core_role_value_template_enabled else None
        )
        self.core_role_value_template_head = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_role_value_template_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    core_role_value_template_hidden_dim,
                    core_role_value_template_num_templates,
                ),
            )
            if core_role_value_template_enabled
            else None
        )
        core_role_value_template_factorized_enabled = (
            core_role_value_template_enabled
            and bool(cfg.core_role_value_template_factorized_enabled)
        )
        core_role_value_template_length_classes = max(
            1,
            int(cfg.core_role_value_template_length_classes),
        )
        core_role_value_template_parity_classes = max(
            1,
            int(cfg.core_role_value_template_parity_classes),
        )
        core_role_value_template_offset_classes = max(
            1,
            int(cfg.core_role_value_template_offset_classes),
        )
        self.core_role_value_template_length_head = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_role_value_template_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    core_role_value_template_hidden_dim,
                    core_role_value_template_length_classes,
                ),
            )
            if core_role_value_template_factorized_enabled
            else None
        )
        self.core_role_value_template_parity_head = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_role_value_template_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    core_role_value_template_hidden_dim,
                    core_role_value_template_parity_classes,
                ),
            )
            if core_role_value_template_factorized_enabled
            else None
        )
        self.core_role_value_template_offset_head = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_role_value_template_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    core_role_value_template_hidden_dim,
                    core_role_value_template_offset_classes,
                ),
            )
            if core_role_value_template_factorized_enabled
            else None
        )
        core_role_value_template_max_steps = max(
            1,
            int(cfg.core_role_value_template_max_steps),
        )
        self.core_role_value_template_table = (
            nn.Parameter(
                torch.empty(
                    core_role_value_template_num_templates,
                    core_role_value_template_max_steps,
                    core_role_value_state_num_roles,
                    core_role_value_state_vocab_size,
                )
            )
            if core_role_value_template_enabled
            else None
        )
        core_role_value_transition_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_transition_enabled)
        )
        core_role_value_transition_hidden_dim = int(
            cfg.core_role_value_transition_hidden_dim or cfg.d_model
        )
        self.core_role_value_transition_input_norm = (
            nn.LayerNorm(cfg.d_model) if core_role_value_transition_enabled else None
        )
        self.core_role_value_transition_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_role_value_transition_hidden_dim),
                nn.GELU(),
                nn.Linear(core_role_value_transition_hidden_dim, cfg.d_model),
            )
            if core_role_value_transition_enabled
            else None
        )
        self.core_role_value_transition_output_norm = (
            nn.LayerNorm(cfg.d_model) if core_role_value_transition_enabled else None
        )
        self.core_role_value_transition_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if core_role_value_transition_enabled
            else None
        )
        core_role_value_delta_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_role_value_delta_enabled)
        )
        core_role_value_delta_hidden_dim = int(
            cfg.core_role_value_delta_hidden_dim or cfg.d_model
        )
        self.core_role_value_delta_step_embed = (
            nn.Embedding(max(1, int(cfg.outer_steps)), cfg.d_model)
            if core_role_value_delta_enabled
            else None
        )
        self.core_role_value_delta_input_norm = (
            nn.LayerNorm(cfg.d_model) if core_role_value_delta_enabled else None
        )
        self.core_role_value_delta_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_role_value_delta_hidden_dim),
                nn.GELU(),
                nn.Linear(core_role_value_delta_hidden_dim, cfg.d_model),
            )
            if core_role_value_delta_enabled
            else None
        )
        self.core_role_value_delta_gate = (
            nn.Linear(cfg.d_model, 1) if core_role_value_delta_enabled else None
        )
        self.core_role_value_delta_output_norm = (
            nn.LayerNorm(cfg.d_model) if core_role_value_delta_enabled else None
        )
        core_value_delta_codebook_size = max(
            0,
            int(cfg.core_value_delta_codebook_size),
        )
        core_value_delta_code_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_value_delta_code_enabled)
            and core_value_delta_codebook_size > 0
        )
        self.core_value_delta_code_input_norm = (
            nn.LayerNorm(cfg.d_model) if core_value_delta_code_enabled else None
        )
        self.core_value_delta_code_head = (
            nn.Linear(cfg.d_model, core_value_delta_codebook_size)
            if core_value_delta_code_enabled
            else None
        )
        self.core_value_delta_code_embed = (
            nn.Embedding(core_value_delta_codebook_size, cfg.d_model)
            if core_value_delta_code_enabled
            else None
        )
        self.core_value_delta_code_gate = (
            nn.Linear(cfg.d_model, 1) if core_value_delta_code_enabled else None
        )
        typed_algorithmic_enabled = bool(cfg.typed_algorithmic_value_state_enabled)
        typed_algorithmic_slots = max(
            1,
            int(cfg.typed_algorithmic_value_state_max_list_slots),
        )
        typed_algorithmic_offset_vocab = max(
            1,
            int(cfg.typed_algorithmic_value_state_offset_vocab_size),
        )
        typed_algorithmic_scalar_vocab = max(
            1,
            int(cfg.typed_algorithmic_value_state_scalar_vocab_size),
        )
        typed_algorithmic_kind_size = max(
            0,
            int(cfg.typed_algorithmic_value_state_kind_size),
        )
        self.typed_algorithmic_value_state_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_enabled else None
        )
        self.typed_algorithmic_kind_head = (
            nn.Linear(cfg.d_model, typed_algorithmic_kind_size)
            if typed_algorithmic_enabled and typed_algorithmic_kind_size > 0
            else None
        )
        self.typed_algorithmic_raw_list_offset_head = (
            nn.Linear(
                cfg.d_model,
                typed_algorithmic_slots * typed_algorithmic_offset_vocab,
            )
            if typed_algorithmic_enabled
            else None
        )
        self.typed_algorithmic_doubled_list_offset_head = (
            nn.Linear(
                cfg.d_model,
                typed_algorithmic_slots * typed_algorithmic_offset_vocab,
            )
            if typed_algorithmic_enabled
            else None
        )
        self.typed_algorithmic_scalar_coeff_head = (
            nn.Linear(cfg.d_model, typed_algorithmic_scalar_vocab)
            if typed_algorithmic_enabled
            else None
        )
        self.typed_algorithmic_scalar_residual_head = (
            nn.Linear(cfg.d_model, typed_algorithmic_scalar_vocab)
            if typed_algorithmic_enabled
            else None
        )
        self.typed_algorithmic_scalar_residual_delta_head = (
            nn.Linear(cfg.d_model, typed_algorithmic_scalar_vocab)
            if typed_algorithmic_enabled
            and bool(cfg.typed_algorithmic_value_state_residual_delta_enabled)
            else None
        )
        self.typed_algorithmic_scalar_offset_head = (
            nn.Linear(cfg.d_model, typed_algorithmic_scalar_vocab)
            if typed_algorithmic_enabled
            and bool(cfg.typed_algorithmic_value_state_scalar_offset_enabled)
            else None
        )
        self.typed_algorithmic_final_residual_head = (
            nn.Linear(cfg.d_model, typed_algorithmic_scalar_vocab)
            if typed_algorithmic_enabled
            else None
        )
        typed_algorithmic_scalar_regression_enabled = (
            typed_algorithmic_enabled
            and bool(cfg.typed_algorithmic_value_state_scalar_regression_enabled)
        )
        self.typed_algorithmic_scalar_coeff_value_head = (
            nn.Linear(cfg.d_model, 1)
            if typed_algorithmic_scalar_regression_enabled
            else None
        )
        self.typed_algorithmic_scalar_offset_value_head = (
            nn.Linear(cfg.d_model, 1)
            if typed_algorithmic_scalar_regression_enabled
            and bool(cfg.typed_algorithmic_value_state_scalar_offset_enabled)
            else None
        )
        self.typed_algorithmic_scalar_residual_value_head = (
            nn.Linear(cfg.d_model, 1)
            if typed_algorithmic_scalar_regression_enabled
            else None
        )
        self.typed_algorithmic_final_residual_value_head = (
            nn.Linear(cfg.d_model, 1)
            if typed_algorithmic_scalar_regression_enabled
            else None
        )
        typed_algorithmic_answer_bridge_enabled = (
            typed_algorithmic_enabled
            and bool(cfg.typed_algorithmic_value_state_answer_bridge_enabled)
        )
        self.typed_algorithmic_value_state_answer_bridge_proj = (
            nn.Linear(2 * typed_algorithmic_scalar_vocab, cfg.d_model, bias=False)
            if typed_algorithmic_answer_bridge_enabled
            else None
        )
        self.typed_algorithmic_value_state_answer_bridge_norm = (
            RMSNorm(cfg.d_model) if typed_algorithmic_answer_bridge_enabled else None
        )
        self.typed_algorithmic_value_state_answer_bridge_gate = (
            nn.Linear(cfg.d_model, 1) if typed_algorithmic_answer_bridge_enabled else None
        )
        typed_algorithmic_prompt_context_enabled = (
            typed_algorithmic_enabled
            and bool(cfg.typed_algorithmic_value_state_prompt_context_enabled)
        )
        self.typed_algorithmic_prompt_query_norm = (
            RMSNorm(cfg.d_model) if typed_algorithmic_prompt_context_enabled else None
        )
        self.typed_algorithmic_prompt_context_norm = (
            RMSNorm(cfg.d_model) if typed_algorithmic_prompt_context_enabled else None
        )
        self.typed_algorithmic_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if typed_algorithmic_prompt_context_enabled
            else None
        )
        self.typed_algorithmic_prompt_gate = (
            nn.Linear(cfg.d_model, 1)
            if typed_algorithmic_prompt_context_enabled
            else None
        )
        self.typed_algorithmic_prompt_output_norm = (
            RMSNorm(cfg.d_model) if typed_algorithmic_prompt_context_enabled else None
        )
        typed_algorithmic_recurrent_enabled = (
            typed_algorithmic_enabled
            and bool(cfg.typed_algorithmic_value_state_recurrent_enabled)
        )
        typed_algorithmic_recurrent_hidden_dim = int(
            cfg.typed_algorithmic_value_state_recurrent_hidden_dim or cfg.d_model
        )
        self.typed_algorithmic_recurrent_step_embed = (
            nn.Embedding(max(1, int(cfg.outer_steps)), cfg.d_model)
            if typed_algorithmic_recurrent_enabled
            else None
        )
        self.typed_algorithmic_recurrent_joint_proj = (
            nn.Linear(max(1, int(cfg.transition_state_joint_size)), cfg.d_model)
            if typed_algorithmic_recurrent_enabled
            else None
        )
        self.typed_algorithmic_recurrent_primitive_proj = (
            nn.Linear(max(1, int(cfg.primitive_transition_num_operations)), cfg.d_model)
            if typed_algorithmic_recurrent_enabled
            and bool(cfg.typed_algorithmic_value_state_primitive_conditioning_enabled)
            and int(cfg.primitive_transition_num_operations) > 0
            else None
        )
        self.typed_algorithmic_recurrent_input_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_recurrent_enabled else None
        )
        self.typed_algorithmic_recurrent_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, typed_algorithmic_recurrent_hidden_dim),
                nn.GELU(),
                nn.Linear(typed_algorithmic_recurrent_hidden_dim, cfg.d_model),
            )
            if typed_algorithmic_recurrent_enabled
            else None
        )
        self.typed_algorithmic_recurrent_gate = (
            nn.Linear(cfg.d_model, 1) if typed_algorithmic_recurrent_enabled else None
        )
        self.typed_algorithmic_recurrent_output_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_recurrent_enabled else None
        )
        typed_algorithmic_subregisters_enabled = (
            typed_algorithmic_recurrent_enabled
            and bool(cfg.typed_algorithmic_value_state_subregisters_enabled)
        )
        self.typed_algorithmic_list_subregister_input_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_list_subregister_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, typed_algorithmic_recurrent_hidden_dim),
                nn.GELU(),
                nn.Linear(typed_algorithmic_recurrent_hidden_dim, cfg.d_model),
            )
            if typed_algorithmic_subregisters_enabled
            else None
        )
        self.typed_algorithmic_list_subregister_gate = (
            nn.Linear(cfg.d_model, 1) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_list_subregister_output_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_scalar_subregister_input_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_scalar_subregister_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, typed_algorithmic_recurrent_hidden_dim),
                nn.GELU(),
                nn.Linear(typed_algorithmic_recurrent_hidden_dim, cfg.d_model),
            )
            if typed_algorithmic_subregisters_enabled
            else None
        )
        self.typed_algorithmic_scalar_subregister_gate = (
            nn.Linear(cfg.d_model, 1) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_scalar_subregister_output_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_final_subregister_input_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_final_subregister_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, typed_algorithmic_recurrent_hidden_dim),
                nn.GELU(),
                nn.Linear(typed_algorithmic_recurrent_hidden_dim, cfg.d_model),
            )
            if typed_algorithmic_subregisters_enabled
            else None
        )
        self.typed_algorithmic_final_subregister_gate = (
            nn.Linear(cfg.d_model, 1) if typed_algorithmic_subregisters_enabled else None
        )
        self.typed_algorithmic_final_subregister_output_norm = (
            nn.LayerNorm(cfg.d_model) if typed_algorithmic_subregisters_enabled else None
        )
        typed_algorithmic_residual_feedback_enabled = (
            typed_algorithmic_recurrent_enabled
            and bool(cfg.typed_algorithmic_value_state_residual_feedback_enabled)
        )
        self.typed_algorithmic_scalar_residual_feedback_proj = (
            nn.Linear(typed_algorithmic_scalar_vocab, cfg.d_model)
            if typed_algorithmic_residual_feedback_enabled
            else None
        )
        self.typed_algorithmic_final_residual_feedback_proj = (
            nn.Linear(typed_algorithmic_scalar_vocab, cfg.d_model)
            if typed_algorithmic_residual_feedback_enabled
            else None
        )
        core_typed_register_num_operations = max(
            0,
            int(cfg.core_typed_register_num_operations),
        )
        core_typed_register_enabled = (
            self.core_role_value_state_embed is not None
            and bool(cfg.core_typed_register_executor_enabled)
            and core_typed_register_num_operations > 0
        )
        core_typed_register_hidden_dim = int(
            cfg.core_typed_register_hidden_dim or cfg.d_model
        )
        self.core_typed_register_context_norm = (
            nn.LayerNorm(cfg.d_model) if core_typed_register_enabled else None
        )
        self.core_typed_register_role_norm = (
            nn.LayerNorm(cfg.d_model) if core_typed_register_enabled else None
        )
        self.core_typed_register_operation_head = (
            nn.Linear(cfg.d_model, core_typed_register_num_operations)
            if core_typed_register_enabled
            else None
        )
        self.core_typed_register_operation_embed = (
            nn.Embedding(core_typed_register_num_operations, cfg.d_model)
            if core_typed_register_enabled
            else None
        )
        self.core_typed_register_input_norm = (
            nn.LayerNorm(cfg.d_model) if core_typed_register_enabled else None
        )
        self.core_typed_register_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_typed_register_hidden_dim),
                nn.GELU(),
                nn.Linear(core_typed_register_hidden_dim, cfg.d_model),
            )
            if core_typed_register_enabled
            else None
        )
        self.core_typed_register_gate = (
            nn.Linear(cfg.d_model, 1) if core_typed_register_enabled else None
        )
        self.core_typed_register_output_norm = (
            nn.LayerNorm(cfg.d_model) if core_typed_register_enabled else None
        )
        self.core_typed_register_value_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if core_typed_register_enabled
            else None
        )
        self.core_typed_register_transition_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if core_typed_register_enabled
            else None
        )
        core_typed_register_value_feedback_enabled = (
            core_typed_register_enabled
            and bool(cfg.core_typed_register_value_feedback_enabled)
        )
        self.core_typed_register_value_feedback_embed = (
            nn.Embedding(core_role_value_state_vocab_size, cfg.d_model)
            if core_typed_register_value_feedback_enabled
            else None
        )
        self.core_typed_register_value_feedback_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_typed_register_value_feedback_enabled
            else None
        )
        self.core_typed_register_value_feedback_gate = (
            nn.Linear(cfg.d_model, 1)
            if core_typed_register_value_feedback_enabled
            else None
        )
        self.core_typed_register_value_feedback_output_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_typed_register_value_feedback_enabled
            else None
        )
        if self.factorized_value_state_step_embed is not None:
            nn.init.normal_(
                self.factorized_value_state_step_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.factorized_value_state_action_proj is not None:
            nn.init.xavier_uniform_(self.factorized_value_state_action_proj.weight)
        if self.factorized_value_state_update is not None:
            for module in self.factorized_value_state_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.factorized_value_state_head is not None:
            nn.init.xavier_uniform_(self.factorized_value_state_head.weight)
            nn.init.zeros_(self.factorized_value_state_head.bias)
        if self.factorized_value_state_kind_head is not None:
            nn.init.xavier_uniform_(self.factorized_value_state_kind_head.weight)
            nn.init.zeros_(self.factorized_value_state_kind_head.bias)
        if self.role_value_state_role_embed is not None:
            nn.init.normal_(
                self.role_value_state_role_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.role_value_state_head is not None:
            nn.init.xavier_uniform_(self.role_value_state_head.weight)
            nn.init.zeros_(self.role_value_state_head.bias)
        if self.core_role_value_state_embed is not None:
            nn.init.normal_(
                self.core_role_value_state_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_role_value_state_head is not None:
            nn.init.xavier_uniform_(self.core_role_value_state_head.weight)
            nn.init.zeros_(self.core_role_value_state_head.bias)
        if self.core_role_value_state_answer_value_embed is not None:
            nn.init.normal_(
                self.core_role_value_state_answer_value_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_role_value_state_answer_gate is not None:
            nn.init.zeros_(self.core_role_value_state_answer_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_answer_gate.bias,
                float(cfg.core_role_value_state_answer_bridge_gate_init_bias),
            )
        if self.core_role_value_state_answer_prompt_gate is not None:
            nn.init.zeros_(self.core_role_value_state_answer_prompt_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_answer_prompt_gate.bias,
                float(cfg.core_role_value_state_answer_prompt_gate_init_bias),
            )
        if self.core_role_value_state_answer_final_proj is not None:
            nn.init.xavier_uniform_(self.core_role_value_state_answer_final_proj.weight)
        if self.core_role_value_state_answer_final_gate is not None:
            nn.init.zeros_(self.core_role_value_state_answer_final_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_answer_final_gate.bias,
                float(cfg.core_role_value_state_answer_final_gate_init_bias),
            )
        if self.core_role_value_state_vocab_renderer_gate is not None:
            nn.init.zeros_(self.core_role_value_state_vocab_renderer_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_vocab_renderer_gate.bias,
                float(cfg.core_role_value_state_vocab_renderer_gate_init_bias),
            )
        if self.core_role_value_state_vocab_renderer_down is not None:
            nn.init.xavier_uniform_(self.core_role_value_state_vocab_renderer_down.weight)
        if self.core_role_value_state_vocab_renderer_up is not None:
            nn.init.zeros_(self.core_role_value_state_vocab_renderer_up.weight)
        if self.core_role_value_state_prompt_gate is not None:
            nn.init.zeros_(self.core_role_value_state_prompt_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_prompt_gate.bias,
                float(cfg.core_role_value_state_prompt_extract_gate_init_bias),
            )
        if self.core_role_value_state_prompt_self_condition_value_embed is not None:
            nn.init.normal_(
                self.core_role_value_state_prompt_self_condition_value_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_role_value_state_prompt_self_condition_gate is not None:
            nn.init.zeros_(self.core_role_value_state_prompt_self_condition_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_prompt_self_condition_gate.bias,
                float(cfg.core_role_value_state_prompt_self_condition_gate_init_bias),
            )
        if self.core_role_value_state_prompt_parity_head is not None:
            nn.init.xavier_uniform_(self.core_role_value_state_prompt_parity_head.weight)
            nn.init.zeros_(self.core_role_value_state_prompt_parity_head.bias)
        if self.core_role_value_state_prompt_parity_embed is not None:
            nn.init.normal_(
                self.core_role_value_state_prompt_parity_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_role_value_state_prompt_parity_gate is not None:
            nn.init.zeros_(self.core_role_value_state_prompt_parity_gate.weight)
            nn.init.constant_(
                self.core_role_value_state_prompt_parity_gate.bias,
                float(cfg.core_role_value_state_prompt_parity_gate_init_bias),
            )
        if self.core_role_value_template_head is not None:
            for module in self.core_role_value_template_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        for factor_head in (
            self.core_role_value_template_length_head,
            self.core_role_value_template_parity_head,
            self.core_role_value_template_offset_head,
        ):
            if factor_head is not None:
                for module in factor_head:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
        if self.core_role_value_template_table is not None:
            nn.init.normal_(self.core_role_value_template_table, mean=0.0, std=0.02)
        if self.core_role_value_transition_update is not None:
            for module in self.core_role_value_transition_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_role_value_transition_head is not None:
            nn.init.xavier_uniform_(self.core_role_value_transition_head.weight)
            nn.init.zeros_(self.core_role_value_transition_head.bias)
        if self.core_role_value_delta_step_embed is not None:
            nn.init.normal_(
                self.core_role_value_delta_step_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_role_value_delta_update is not None:
            for module in self.core_role_value_delta_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_role_value_delta_gate is not None:
            nn.init.zeros_(self.core_role_value_delta_gate.weight)
            nn.init.constant_(
                self.core_role_value_delta_gate.bias,
                float(cfg.core_role_value_delta_gate_init_bias),
            )
        if self.core_value_delta_code_head is not None:
            nn.init.xavier_uniform_(self.core_value_delta_code_head.weight)
            nn.init.zeros_(self.core_value_delta_code_head.bias)
        if self.core_value_delta_code_embed is not None:
            nn.init.normal_(
                self.core_value_delta_code_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_value_delta_code_gate is not None:
            nn.init.zeros_(self.core_value_delta_code_gate.weight)
            nn.init.constant_(
                self.core_value_delta_code_gate.bias,
                float(cfg.core_value_delta_code_gate_init_bias),
            )
        for head in (
            self.typed_algorithmic_kind_head,
            self.typed_algorithmic_raw_list_offset_head,
            self.typed_algorithmic_doubled_list_offset_head,
            self.typed_algorithmic_scalar_coeff_head,
            self.typed_algorithmic_scalar_residual_head,
            self.typed_algorithmic_scalar_residual_delta_head,
            self.typed_algorithmic_scalar_offset_head,
            self.typed_algorithmic_final_residual_head,
            self.typed_algorithmic_scalar_coeff_value_head,
            self.typed_algorithmic_scalar_offset_value_head,
            self.typed_algorithmic_scalar_residual_value_head,
            self.typed_algorithmic_final_residual_value_head,
        ):
            if head is not None:
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
        if self.typed_algorithmic_value_state_answer_bridge_proj is not None:
            nn.init.xavier_uniform_(
                self.typed_algorithmic_value_state_answer_bridge_proj.weight
            )
        if self.typed_algorithmic_value_state_answer_bridge_gate is not None:
            nn.init.zeros_(self.typed_algorithmic_value_state_answer_bridge_gate.weight)
            nn.init.constant_(
                self.typed_algorithmic_value_state_answer_bridge_gate.bias,
                float(cfg.typed_algorithmic_value_state_answer_bridge_gate_init_bias),
            )
        if self.typed_algorithmic_prompt_gate is not None:
            nn.init.zeros_(self.typed_algorithmic_prompt_gate.weight)
            nn.init.constant_(
                self.typed_algorithmic_prompt_gate.bias,
                float(cfg.typed_algorithmic_value_state_prompt_gate_init_bias),
            )
        if self.typed_algorithmic_recurrent_step_embed is not None:
            nn.init.normal_(
                self.typed_algorithmic_recurrent_step_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.typed_algorithmic_recurrent_joint_proj is not None:
            nn.init.xavier_uniform_(self.typed_algorithmic_recurrent_joint_proj.weight)
            nn.init.zeros_(self.typed_algorithmic_recurrent_joint_proj.bias)
        if self.typed_algorithmic_recurrent_primitive_proj is not None:
            nn.init.xavier_uniform_(
                self.typed_algorithmic_recurrent_primitive_proj.weight
            )
            nn.init.zeros_(self.typed_algorithmic_recurrent_primitive_proj.bias)
        if self.typed_algorithmic_recurrent_update is not None:
            for module in self.typed_algorithmic_recurrent_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.typed_algorithmic_recurrent_gate is not None:
            nn.init.zeros_(self.typed_algorithmic_recurrent_gate.weight)
            nn.init.constant_(
                self.typed_algorithmic_recurrent_gate.bias,
                float(cfg.typed_algorithmic_value_state_recurrent_gate_init_bias),
            )
        for update in (
            self.typed_algorithmic_list_subregister_update,
            self.typed_algorithmic_scalar_subregister_update,
            self.typed_algorithmic_final_subregister_update,
        ):
            if update is not None:
                for module in update:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
        for gate in (
            self.typed_algorithmic_list_subregister_gate,
            self.typed_algorithmic_scalar_subregister_gate,
            self.typed_algorithmic_final_subregister_gate,
        ):
            if gate is not None:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(
                    gate.bias,
                    float(cfg.typed_algorithmic_value_state_recurrent_gate_init_bias),
                )
        for proj in (
            self.typed_algorithmic_scalar_residual_feedback_proj,
            self.typed_algorithmic_final_residual_feedback_proj,
        ):
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        if self.core_typed_register_operation_head is not None:
            nn.init.xavier_uniform_(self.core_typed_register_operation_head.weight)
            nn.init.zeros_(self.core_typed_register_operation_head.bias)
        if self.core_typed_register_operation_embed is not None:
            nn.init.normal_(
                self.core_typed_register_operation_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_typed_register_update is not None:
            for module in self.core_typed_register_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_typed_register_gate is not None:
            nn.init.zeros_(self.core_typed_register_gate.weight)
            nn.init.constant_(
                self.core_typed_register_gate.bias,
                float(cfg.core_typed_register_gate_init_bias),
            )
        if self.core_typed_register_value_head is not None:
            nn.init.xavier_uniform_(self.core_typed_register_value_head.weight)
            nn.init.zeros_(self.core_typed_register_value_head.bias)
        if self.core_typed_register_transition_head is not None:
            nn.init.xavier_uniform_(self.core_typed_register_transition_head.weight)
            nn.init.zeros_(self.core_typed_register_transition_head.bias)
        if self.core_typed_register_value_feedback_embed is not None:
            nn.init.normal_(
                self.core_typed_register_value_feedback_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_typed_register_value_feedback_gate is not None:
            nn.init.zeros_(self.core_typed_register_value_feedback_gate.weight)
            nn.init.constant_(
                self.core_typed_register_value_feedback_gate.bias,
                float(cfg.core_typed_register_value_feedback_gate_init_bias),
            )
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
        core_primitive_role_value_enabled = (
            bool(cfg.core_primitive_role_value_executor_enabled)
            and self.core_role_value_state_embed is not None
            and primitive_transition_num_operations > 0
            and core_role_value_state_vocab_size > 0
        )
        core_primitive_role_value_mlp_enabled = (
            core_primitive_role_value_enabled
            and bool(cfg.core_primitive_role_value_mlp_enabled)
        )
        core_primitive_role_value_table_enabled = (
            core_primitive_role_value_enabled
            and not core_primitive_role_value_mlp_enabled
        )
        self.core_primitive_role_value_source_mix = (
            nn.Parameter(
                torch.zeros(
                    primitive_transition_num_operations,
                    core_role_value_state_num_roles,
                    core_role_value_state_num_roles,
                )
            )
            if core_primitive_role_value_table_enabled
            else None
        )
        self.core_primitive_role_value_value_transition = (
            nn.Parameter(
                torch.zeros(
                    primitive_transition_num_operations,
                    core_role_value_state_num_roles,
                    core_role_value_state_vocab_size,
                    core_role_value_state_vocab_size,
                )
            )
            if core_primitive_role_value_table_enabled
            else None
        )
        self.core_primitive_role_value_bias = (
            nn.Parameter(
                torch.zeros(
                    primitive_transition_num_operations,
                    core_role_value_state_num_roles,
                    core_role_value_state_vocab_size,
                )
            )
            if core_primitive_role_value_table_enabled
            else None
        )
        if self.core_primitive_role_value_source_mix is not None:
            with torch.no_grad():
                roles = int(core_role_value_state_num_roles)
                vocab = int(core_role_value_state_vocab_size)
                diag_roles = torch.arange(roles)
                self.core_primitive_role_value_source_mix[
                    :, diag_roles, diag_roles
                ] = 2.0
                diag_values = torch.arange(vocab)
                self.core_primitive_role_value_value_transition[
                    :, :, diag_values, diag_values
                ] = 0.25
        core_primitive_role_value_hidden_dim = int(
            cfg.core_primitive_role_value_hidden_dim or cfg.d_model
        )
        self.core_primitive_role_value_value_embed = (
            nn.Embedding(core_role_value_state_vocab_size, cfg.d_model)
            if core_primitive_role_value_mlp_enabled
            else None
        )
        self.core_primitive_role_value_operation_embed = (
            nn.Embedding(primitive_transition_num_operations, cfg.d_model)
            if core_primitive_role_value_mlp_enabled
            else None
        )
        core_primitive_role_value_role_mixer_enabled = (
            core_primitive_role_value_mlp_enabled
            and bool(cfg.core_primitive_role_value_role_mixer_enabled)
        )
        role_mixer_heads = int(
            cfg.core_primitive_role_value_role_mixer_heads or cfg.n_heads
        )
        if role_mixer_heads <= 0 or int(cfg.d_model) % int(role_mixer_heads) != 0:
            role_mixer_heads = 1
        self.core_primitive_role_value_role_mixer_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_primitive_role_value_role_mixer_enabled
            else None
        )
        self.core_primitive_role_value_role_mixer = (
            nn.MultiheadAttention(
                cfg.d_model,
                num_heads=role_mixer_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            if core_primitive_role_value_role_mixer_enabled
            else None
        )
        core_primitive_role_value_prompt_context_enabled = (
            core_primitive_role_value_mlp_enabled
            and bool(cfg.core_primitive_role_value_prompt_context_enabled)
        )
        self.core_primitive_role_value_prompt_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_primitive_role_value_prompt_context_enabled
            else None
        )
        self.core_primitive_role_value_prompt_context_adapter = (
            nn.Linear(cfg.d_model, cfg.d_model)
            if core_primitive_role_value_prompt_context_enabled
            else None
        )
        core_primitive_role_value_prompt_token_attention_enabled = (
            core_primitive_role_value_prompt_context_enabled
            and bool(cfg.core_primitive_role_value_prompt_token_attention_enabled)
        )
        self.core_primitive_role_value_prompt_query_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_primitive_role_value_prompt_token_attention_enabled
            else None
        )
        self.core_primitive_role_value_prompt_token_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_primitive_role_value_prompt_token_attention_enabled
            else None
        )
        self.core_primitive_role_value_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if core_primitive_role_value_prompt_token_attention_enabled
            else None
        )
        self.core_primitive_role_value_prompt_token_output_norm = (
            nn.LayerNorm(cfg.d_model)
            if core_primitive_role_value_prompt_token_attention_enabled
            else None
        )
        self.core_primitive_role_value_input_norm = (
            nn.LayerNorm(cfg.d_model) if core_primitive_role_value_mlp_enabled else None
        )
        self.core_primitive_role_value_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, core_primitive_role_value_hidden_dim),
                nn.GELU(),
                nn.Linear(core_primitive_role_value_hidden_dim, cfg.d_model),
            )
            if core_primitive_role_value_mlp_enabled
            else None
        )
        self.core_primitive_role_value_update_gate = (
            nn.Linear(cfg.d_model, 1)
            if core_primitive_role_value_mlp_enabled
            and bool(cfg.core_primitive_role_value_update_gate_enabled)
            else None
        )
        self.core_primitive_role_value_source_value_gate = (
            nn.Parameter(
                torch.full(
                    (1, core_role_value_state_num_roles, 1),
                    float(cfg.core_primitive_role_value_source_value_gate_init_bias),
                )
            )
            if core_primitive_role_value_mlp_enabled
            and bool(cfg.core_primitive_role_value_source_value_conditioning_enabled)
            else None
        )
        self.core_primitive_role_value_output_norm = (
            nn.LayerNorm(cfg.d_model) if core_primitive_role_value_mlp_enabled else None
        )
        self.core_primitive_role_value_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if core_primitive_role_value_mlp_enabled
            else None
        )
        core_primitive_operation_heads_enabled = (
            core_primitive_role_value_mlp_enabled
            and bool(cfg.core_primitive_role_value_operation_specific_heads_enabled)
        )
        self.core_primitive_role_value_operation_heads = (
            nn.ModuleList(
                [
                    nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
                    for _ in range(max(1, int(primitive_transition_num_operations)))
                ]
            )
            if core_primitive_operation_heads_enabled
            else None
        )
        core_primitive_field_heads_enabled = (
            core_primitive_role_value_mlp_enabled
            and bool(cfg.core_primitive_role_value_field_specific_heads_enabled)
        )
        self.core_primitive_role_value_list_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if core_primitive_field_heads_enabled
            else None
        )
        self.core_primitive_role_value_scalar_head = (
            nn.Linear(cfg.d_model, core_role_value_state_vocab_size)
            if core_primitive_field_heads_enabled
            else None
        )
        self.core_primitive_typed_selector = (
            nn.Linear(5, 1)
            if core_primitive_role_value_enabled
            and core_typed_register_enabled
            and bool(cfg.core_primitive_typed_selector_enabled)
            else None
        )
        if self.core_primitive_role_value_value_embed is not None:
            nn.init.normal_(
                self.core_primitive_role_value_value_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_primitive_role_value_operation_embed is not None:
            nn.init.normal_(
                self.core_primitive_role_value_operation_embed.weight,
                mean=0.0,
                std=0.02,
            )
        if self.core_primitive_role_value_update is not None:
            for module in self.core_primitive_role_value_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.core_primitive_role_value_prompt_context_adapter is not None:
            nn.init.zeros_(self.core_primitive_role_value_prompt_context_adapter.weight)
            nn.init.zeros_(self.core_primitive_role_value_prompt_context_adapter.bias)
        if self.core_primitive_role_value_update_gate is not None:
            nn.init.zeros_(self.core_primitive_role_value_update_gate.weight)
            nn.init.constant_(
                self.core_primitive_role_value_update_gate.bias,
                float(cfg.core_primitive_role_value_update_gate_init_bias),
            )
        if self.core_primitive_role_value_head is not None:
            nn.init.xavier_uniform_(self.core_primitive_role_value_head.weight)
            nn.init.zeros_(self.core_primitive_role_value_head.bias)
        if self.core_primitive_role_value_operation_heads is not None:
            for head in self.core_primitive_role_value_operation_heads:
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
        if self.core_primitive_role_value_list_head is not None:
            nn.init.xavier_uniform_(self.core_primitive_role_value_list_head.weight)
            nn.init.zeros_(self.core_primitive_role_value_list_head.bias)
        if self.core_primitive_role_value_scalar_head is not None:
            nn.init.xavier_uniform_(self.core_primitive_role_value_scalar_head.weight)
            nn.init.zeros_(self.core_primitive_role_value_scalar_head.bias)
        if self.core_primitive_typed_selector is not None:
            nn.init.zeros_(self.core_primitive_typed_selector.weight)
            nn.init.constant_(
                self.core_primitive_typed_selector.bias,
                float(cfg.core_primitive_typed_selector_init_bias),
            )
        transition_source_router_hidden_dim = int(
            cfg.transition_source_router_hidden_dim or cfg.d_model
        )
        self.transition_source_router_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_source_router_enabled
            else None
        )
        self.transition_source_router_prompt_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_source_router_enabled
            and cfg.transition_source_router_prompt_context_enabled
            else None
        )
        self.transition_source_router_prompt_query_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_source_router_enabled
            and cfg.transition_source_router_prompt_context_enabled
            and cfg.transition_source_router_prompt_token_attention_enabled
            else None
        )
        self.transition_source_router_prompt_context_norm = (
            nn.LayerNorm(cfg.d_model)
            if cfg.transition_source_router_enabled
            and cfg.transition_source_router_prompt_context_enabled
            and cfg.transition_source_router_prompt_token_attention_enabled
            else None
        )
        self.transition_source_router_prompt_cross = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.transition_source_router_enabled
            and cfg.transition_source_router_prompt_context_enabled
            and cfg.transition_source_router_prompt_token_attention_enabled
            else None
        )
        transition_source_router_context_multiplier = 0
        if cfg.transition_source_router_prompt_context_enabled:
            transition_source_router_context_multiplier = (
                2 if cfg.transition_source_router_prompt_token_attention_enabled else 1
            )
        transition_source_router_input_dim = cfg.d_model * (
            1 + transition_source_router_context_multiplier
        )
        self.transition_source_router_head = (
            nn.Sequential(
                nn.Linear(
                    transition_source_router_input_dim,
                    transition_source_router_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(transition_source_router_hidden_dim, 2),
            )
            if cfg.transition_source_router_enabled
            else None
        )
        if self.transition_source_router_head is not None:
            for module in self.transition_source_router_head:
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
        if controller_signal_source not in {
            "external",
            "learned_core",
            "learned_core_trajectory",
            "learned_readout",
        }:
            raise ValueError(
                "controller_signal_source must be 'external', 'learned_core', "
                "'learned_core_trajectory', or 'learned_readout'"
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
        controller_signal_hidden_dim = int(getattr(cfg, "controller_signal_hidden_dim", 0))
        if controller_signal_source == "external":
            self.controller_signal_head = None
        elif controller_signal_source == "learned_core_trajectory":
            controller_signal_head_input_dim = cfg.d_model * max(1, int(cfg.outer_steps))
            if controller_signal_hidden_dim > 0:
                self.controller_signal_head = nn.Sequential(
                    nn.Linear(controller_signal_head_input_dim, controller_signal_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(controller_signal_hidden_dim, controller_signal_dim),
                )
            else:
                self.controller_signal_head = nn.Linear(
                    controller_signal_head_input_dim,
                    controller_signal_dim,
                )
        elif controller_signal_hidden_dim > 0:
            self.controller_signal_head = nn.Sequential(
                nn.Linear(cfg.d_model, controller_signal_hidden_dim),
                nn.SiLU(),
                nn.Linear(controller_signal_hidden_dim, controller_signal_dim),
            )
        else:
            self.controller_signal_head = nn.Linear(cfg.d_model, controller_signal_dim)
        if self.controller_signal_proj is not None:
            nn.init.xavier_uniform_(self.controller_signal_proj.weight)
        if self.controller_signal_head is not None:
            for module in self.controller_signal_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
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
        token_numeric_value_ids: Optional[torch.Tensor] = None,
        token_numeric_source_slot_ids: Optional[torch.Tensor] = None,
        token_numeric_source_slot_token_ids: Optional[torch.Tensor] = None,
        token_numeric_source_slot_token_span_ids: Optional[torch.Tensor] = None,
        token_numeric_source_slot_token_span_mask: Optional[torch.Tensor] = None,
        token_numeric_source_slot_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        text_states: Optional[torch.Tensor] = None,
        workspace_text_states: Optional[torch.Tensor] = None,
        workspace_attention_mask: Optional[torch.Tensor] = None,
        donor_logits: Optional[torch.Tensor] = None,
        disable_workspace: bool = False,
        disable_core: bool = False,
        disable_coda: bool = False,
        disable_qtrm_residual: bool = False,
        disable_qtrm_residual_gate: bool = False,
        disable_donor_context: bool = False,
        disable_workspace_memory_context: bool = False,
        disable_workspace_memory_gate: bool = False,
        disable_core_context: bool = False,
        disable_core_state_carry: bool = False,
        zero_core_trajectory: bool = False,
        disable_core_role_value_delta: bool = False,
        disable_core_value_delta_code: bool = False,
        disable_typed_algorithmic_value_state: bool = False,
        disable_typed_algorithmic_value_state_recurrent: bool = False,
        disable_core_typed_register_executor: bool = False,
        disable_core_primitive_role_value_executor: bool = False,
        disable_core_primitive_prompt_context: bool = False,
        disable_token_numeric_source_slots: bool = False,
        disable_core_role_value_prompt_extract: bool = False,
        disable_core_source_position_binder: bool = False,
        disable_core_source_position_binder_query_state: bool = False,
        disable_core_source_value_binder: bool = False,
        disable_core_role_value_answer_bridge: bool = False,
        disable_core_role_value_answer_final_binder: bool = False,
        disable_core_role_value_vocab_renderer: bool = False,
        disable_answer_state_loop_recurrent: bool = False,
        disable_typed_algorithmic_value_state_answer_bridge: bool = False,
        disable_answer_state_loop_selective_context: bool = False,
        force_answer_state_loop_dense_context: bool = False,
        disable_answer_state_loop_finality_selector: bool = False,
        disable_answer_state_loop_finality_gate: bool = False,
        disable_answer_state_loop_halt_gate: bool = False,
        disable_answer_state_loop_hidden_bridge: bool = False,
        disable_answer_state_loop_next_token_decoder: bool = False,
        disable_answer_state_loop_free_transformer_latent: bool = False,
        disable_answer_state_loop_talker: bool = False,
        disable_transition_state_joint_answer_bridge: bool = False,
        disable_transition_state_final_answer_binder: bool = False,
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
        core_carry: Optional[QTRMCoreCarry] = None,
        # RI-4: Optional external memory residual (from hybrid final state after real prompt + MSA slots)
        # Injected as residual on the thinking hidden *before* the final lm_head.
        # This is the proper architectural path (replaces previous monkey-patch attributes).
        ri4_memory_residual: Optional[torch.Tensor] = None,
        ri4_memory_residual_scale: float = 0.3,
        return_core_carry: bool = False,
        core_transition_feedback_operation_targets: Optional[torch.Tensor] = None,
        core_transition_feedback_finality_targets: Optional[torch.Tensor] = None,
        core_transition_feedback_teacher_forcing: bool = False,
        return_core_depth_logits: bool = False,
        return_core_depth_text_logits: bool = False,
        return_features_only: bool = False,
        logit_token_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        disable_core = bool(disable_core or not self.cfg.core_enabled)

        text_seq = self.text_embed(input_ids)
        if token_numeric_value_ids is not None and self.token_numeric_value_embed is not None:
            if tuple(token_numeric_value_ids.shape) != tuple(input_ids.shape):
                raise ValueError("token_numeric_value_ids must match input_ids shape")
            numeric_ids = token_numeric_value_ids.to(
                device=input_ids.device,
                dtype=torch.long,
            ).clamp(
                min=0,
                max=int(self.token_numeric_value_embed.num_embeddings) - 1,
            )
            numeric_delta = self.token_numeric_value_embed(numeric_ids)
            if self.token_numeric_value_gate is not None:
                numeric_gate = torch.sigmoid(
                    self.token_numeric_value_gate.to(
                        device=text_seq.device,
                        dtype=text_seq.dtype,
                    )
                )
                gate_min = min(
                    max(float(self.cfg.token_numeric_value_gate_min), 0.0),
                    1.0,
                )
                if gate_min > 0.0:
                    numeric_gate = gate_min + (1.0 - gate_min) * numeric_gate
                numeric_delta = numeric_gate * numeric_delta
            text_seq = text_seq + numeric_delta
        if self.text_position_embed is not None:
            pos_ids = torch.arange(
                s,
                dtype=torch.long,
                device=input_ids.device,
            ).clamp(max=self.text_position_embed.num_embeddings - 1)
            text_seq = text_seq + self.text_position_embed(pos_ids).unsqueeze(0)
        source_slot_token_count = 0
        source_slot_parity_logits = text_seq.new_empty((b, 0, 2))
        source_slot_predicate_logits = text_seq.new_empty((b, 0, 2))
        source_position_state_delta = text_seq.new_empty((b, 0, self.cfg.d_model))
        source_position_prompt_logits = text_seq.new_empty((b, 0, 0, 0))
        source_position_context_token_ids = input_ids
        source_position_context_mask = attention_mask
        if (
            token_numeric_source_slot_ids is not None
            and self.token_numeric_source_slot_embed is not None
            and not bool(disable_token_numeric_source_slots)
        ):
            if token_numeric_source_slot_ids.ndim != 2:
                raise ValueError(
                    "token_numeric_source_slot_ids must have shape [batch, slots]"
                )
            if int(token_numeric_source_slot_ids.shape[0]) != b:
                raise ValueError(
                    "token_numeric_source_slot_ids batch must match input_ids"
                )
            slot_limit = min(
                int(token_numeric_source_slot_ids.shape[1]),
                int(self.cfg.token_numeric_source_slot_max_slots),
            )
            if slot_limit > 0:
                slot_ids = token_numeric_source_slot_ids[:, :slot_limit].to(
                    device=input_ids.device,
                    dtype=torch.long,
                ).clamp(
                    min=0,
                    max=int(self.token_numeric_source_slot_embed.num_embeddings) - 1,
                )
                source_slot_seq = self.token_numeric_source_slot_embed(slot_ids)
                if self.token_numeric_source_slot_pos is not None:
                    slot_positions = torch.arange(
                        slot_limit,
                        dtype=torch.long,
                        device=input_ids.device,
                    ).clamp(
                        max=int(self.token_numeric_source_slot_pos.num_embeddings) - 1
                    )
                    source_slot_seq = (
                        source_slot_seq
                        + self.token_numeric_source_slot_pos(slot_positions).unsqueeze(0)
                    )
                if self.token_numeric_source_slot_gate is not None:
                    source_slot_gate = torch.sigmoid(
                        self.token_numeric_source_slot_gate.to(
                            device=text_seq.device,
                            dtype=text_seq.dtype,
                        )
                    )
                    gate_min = min(
                        max(
                            float(self.cfg.token_numeric_source_slot_gate_min),
                            0.0,
                        ),
                        1.0,
                    )
                    if gate_min > 0.0:
                        source_slot_gate = (
                            gate_min + (1.0 - gate_min) * source_slot_gate
                        )
                    source_slot_seq = source_slot_gate * source_slot_seq
                if (
                    self.token_numeric_source_slot_predicate_head is not None
                    and self.token_numeric_source_slot_predicate_embed is not None
                    and self.token_numeric_source_slot_predicate_gate is not None
                ):
                    source_slot_predicate_logits = (
                        self.token_numeric_source_slot_predicate_head(source_slot_seq)
                    )
                    predicate_probs = torch.softmax(
                        source_slot_predicate_logits.float(),
                        dim=-1,
                    ).to(dtype=source_slot_seq.dtype)
                    predicate_delta = predicate_probs @ (
                        self.token_numeric_source_slot_predicate_embed.weight.to(
                            device=source_slot_seq.device,
                            dtype=source_slot_seq.dtype,
                        )
                    )
                    predicate_gate = torch.sigmoid(
                        self.token_numeric_source_slot_predicate_gate.to(
                            device=source_slot_seq.device,
                            dtype=source_slot_seq.dtype,
                        )
                    )
                    predicate_gate_min = min(
                        max(
                            float(
                                self.cfg.token_numeric_source_slot_predicate_gate_min
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                    if predicate_gate_min > 0.0:
                        predicate_gate = (
                            predicate_gate_min
                            + (1.0 - predicate_gate_min) * predicate_gate
                        )
                    source_slot_seq = source_slot_seq + predicate_gate * predicate_delta
                if token_numeric_source_slot_mask is None:
                    source_slot_mask = (
                        slot_ids != 0
                    ).to(device=input_ids.device, dtype=attention_mask.dtype)
                else:
                    if token_numeric_source_slot_mask.ndim != 2:
                        raise ValueError(
                            "token_numeric_source_slot_mask must have shape [batch, slots]"
                        )
                    if int(token_numeric_source_slot_mask.shape[0]) != b:
                        raise ValueError(
                            "token_numeric_source_slot_mask batch must match input_ids"
                        )
                    source_slot_mask = token_numeric_source_slot_mask[
                        :, :slot_limit
                    ].to(device=input_ids.device, dtype=attention_mask.dtype)
                text_seq = torch.cat([source_slot_seq, text_seq], dim=1)
                attention_mask = torch.cat([source_slot_mask, attention_mask], dim=1)
                source_slot_token_count = slot_limit
        input_text_seq = text_seq
        input_text_mask = attention_mask
        selector_info = {
            "selected_logits": text_seq.new_empty((b, 0, 0, 0)),
            "gate": text_seq.new_empty((b, 0, 0)),
            "gate_mean": text_seq.new_empty((b, 0)),
        }
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
        if (
            int(source_slot_token_count) > 0
            and self.token_numeric_source_slot_parity_head is not None
        ):
            source_slot_parity_logits = self.token_numeric_source_slot_parity_head(
                text_context_seq[:, : int(source_slot_token_count), :]
            )
        core_transition_order_bottleneck_info = (
            self._empty_core_transition_order_bottleneck_info(seq)
        )
        if disable_workspace:
            workspace = seq.new_zeros((b, self.cfg.workspace_tokens, self.cfg.d_model))
            workspace_mask = torch.ones(
                workspace.shape[:2],
                device=workspace.device,
                dtype=attention_mask.dtype,
            )
            core_state_mask = workspace_mask
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
            transition_state_joint_logits = self._compute_transition_state_joint_logits(
                core_depth_states,
                disabled=True,
            )
            transition_state_joint_answer_embeddings = (
                self._compute_transition_state_joint_answer_embeddings(
                    transition_state_joint_logits,
                    disabled=True,
                )
            )
            transition_state_final_answer_embedding = (
                self._compute_transition_state_final_answer_embedding(
                    core_depth_states,
                    transition_state_joint_logits,
                    disabled=True,
                )
            )
            transition_state_sequence_logits = (
                self._compute_transition_state_sequence_logits(
                    core_depth_states,
                    disabled=True,
                )
            )
            transition_value_state_logits = (
                self._compute_transition_value_state_logits(
                    core_depth_states,
                    disabled=True,
                )
            )
            factorized_value_state_info = (
                self._compute_factorized_value_state_outputs(
                    core_depth_states,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=True,
                )
            )
            factorized_value_state_logits = factorized_value_state_info["slot_logits"]
            factorized_value_state_kind_logits = factorized_value_state_info[
                "kind_logits"
            ]
            role_value_state_logits = factorized_value_state_info["role_logits"]
            core_role_value_state_logits = self._empty_core_role_value_state_logits(
                core_depth_states
            )
            core_role_value_state_prompt_logits = (
                self._empty_core_role_value_state_logits(
                    core_depth_states.new_empty((b, 0, core_depth_states.shape[-1]))
                )
            )
            core_source_value_prompt_logits = self._empty_core_role_value_state_logits(
                core_depth_states.new_empty((b, 0, core_depth_states.shape[-1]))
            )
            core_role_value_state_prompt_parity_logits = workspace.new_empty((b, 0))
            core_role_value_transition_logits = self._empty_core_role_value_transition_logits(
                core_depth_states
            )
            core_role_value_delta_gate_mean = workspace.new_empty((b, 0))
            core_value_delta_code_logits = self._empty_core_value_delta_code_logits(
                core_depth_states
            )
            core_value_delta_code_gate_mean = workspace.new_empty((b, 0))
            core_typed_register_info = self._empty_core_typed_register_outputs(
                core_depth_states
            )
            core_role_value_template_logits = workspace.new_empty((b, 0))
            typed_algorithmic_value_state_info = (
                self._compute_typed_algorithmic_value_state_outputs(
                    core_depth_states,
                    disabled=True,
                    recurrent_disabled=True,
                )
            )
            typed_algorithmic_value_state_answer_bridge_info = (
                self._compute_typed_algorithmic_value_state_answer_bridge(
                    typed_algorithmic_value_state_info,
                    reference=core_depth_states,
                    disabled=True,
                )
            )
            if self.factorized_value_state_head is not None:
                transition_value_state_logits = factorized_value_state_logits
            primitive_transition_info = self._compute_primitive_transition_outputs(
                core_depth_states,
                prompt_context_seq=(
                    None if bool(disable_core_primitive_prompt_context) else text_context_seq
                ),
                prompt_context_mask=(
                    None if bool(disable_core_primitive_prompt_context) else text_context_mask
                ),
                disabled=True,
            )
            (
                core_primitive_role_value_state_logits,
                core_primitive_role_value_update_gate,
            ) = (
                self._compute_core_primitive_role_value_state_logits(
                    primitive_transition_info,
                    prompt_logits=core_role_value_state_prompt_logits,
                    source_value_logits=core_source_value_prompt_logits,
                    fallback_logits=core_role_value_state_logits,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    reference=core_depth_states,
                    disabled=True,
                    disable_prompt_context=bool(
                        disable_core_primitive_prompt_context
                    ),
                )
            )
            transition_state_joint_logits = (
                self._apply_transition_state_joint_operation_residual(
                    transition_state_joint_logits,
                    primitive_transition_info,
                    disabled=True,
                )
            )
            transition_state_joint_logits = (
                self._apply_transition_state_joint_code_residual(
                    transition_state_joint_logits,
                    transition_state_code_info,
                    disabled=True,
                )
            )
            transition_phase_logits = self._compute_transition_phase_logits(
                core_depth_states,
                prompt_context_seq=text_context_seq,
                prompt_context_mask=text_context_mask,
                disabled=True,
            )
            transition_state_joint_logits = (
                self._apply_transition_state_joint_phase_residual(
                    transition_state_joint_logits,
                    transition_phase_logits,
                    core_depth_states,
                    disabled=True,
                )
            )
            transition_source_router_logits = (
                self._compute_transition_source_router_logits(
                    core_depth_states,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=True,
                )
            )
            core_role_value_state_answer_bridge_info = (
                self._compute_core_role_value_state_answer_bridge(
                    core_role_value_state_logits,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=bool(disable_core_role_value_answer_bridge),
                )
            )
            core_role_value_state_answer_final_embedding = (
                self._compute_core_role_value_state_answer_final_embedding(
                    core_role_value_state_answer_bridge_info["tokens"],
                    disabled=bool(
                        disable_core_role_value_answer_bridge
                        or disable_core_role_value_answer_final_binder
                    ),
                )
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
            core_state_mask = workspace_mask
            if disable_core:
                z_l = workspace
                z_h = workspace
                trajectory = []
                core_halt_info = self._empty_core_halt_info(workspace)
                core_context_gate_mean = workspace.new_empty((b, 0))
                core_role_value_state_logits = self._empty_core_role_value_state_logits(
                    self._empty_core_depth_states(workspace)
                )
                core_role_value_state_prompt_logits = (
                    self._empty_core_role_value_state_logits(
                        self._empty_core_depth_states(workspace)
                    )
                )
                core_source_value_prompt_logits = (
                    self._empty_core_role_value_state_logits(
                        self._empty_core_depth_states(workspace)
                    )
                )
                source_position_state_delta = workspace.new_empty(
                    (b, 0, self.cfg.d_model)
                )
                core_role_value_state_prompt_parity_logits = workspace.new_empty(
                    (b, 0)
                )
                core_role_value_transition_logits = (
                    self._empty_core_role_value_transition_logits(
                        self._empty_core_depth_states(workspace)
                    )
                )
                core_role_value_delta_gate_mean = workspace.new_empty((b, 0))
                core_value_delta_code_logits = self._empty_core_value_delta_code_logits(
                    self._empty_core_depth_states(workspace)
                )
                core_value_delta_code_gate_mean = workspace.new_empty((b, 0))
                core_typed_register_info = self._empty_core_typed_register_outputs(
                    self._empty_core_depth_states(workspace)
                )
                core_role_value_template_logits = workspace.new_empty((b, 0))
                typed_algorithmic_value_state_info = (
                    self._compute_typed_algorithmic_value_state_outputs(
                        self._empty_core_depth_states(workspace),
                        disabled=True,
                        recurrent_disabled=True,
                    )
                )
                typed_algorithmic_value_state_answer_bridge_info = (
                    self._compute_typed_algorithmic_value_state_answer_bridge(
                        typed_algorithmic_value_state_info,
                        reference=self._empty_core_depth_states(workspace),
                        disabled=True,
                    )
                )
            else:
                core_workspace = workspace
                core_workspace_mask = workspace_mask
                core_role_token_start = None
                core_role_token_count = 0
                core_transition_order_bottleneck_info = (
                    self._compute_core_transition_order_bottleneck(
                        text_context_seq,
                        text_context_mask,
                        reference=workspace,
                    )
                )
                order_token = core_transition_order_bottleneck_info["token"]
                if int(order_token.shape[1]) > 0:
                    core_workspace = torch.cat([order_token, core_workspace], dim=1)
                    order_mask = workspace_mask.new_ones(
                        (b, int(order_token.shape[1]))
                    )
                    core_workspace_mask = torch.cat(
                        [order_mask, core_workspace_mask],
                        dim=1,
                    )
                core_role_value_state_prompt_logits = (
                    self._empty_core_role_value_state_logits(
                        self._empty_core_depth_states(workspace)
                    )
                )
                core_source_value_prompt_logits = (
                    self._empty_core_role_value_state_logits(
                        self._empty_core_depth_states(workspace)
                    )
                )
                source_position_state_delta = workspace.new_empty(
                    (b, 0, self.cfg.d_model)
                )
                core_role_value_state_prompt_parity_logits = workspace.new_empty(
                    (b, 0)
                )
                if self.core_role_value_state_embed is not None:
                    core_role_tokens = self.core_role_value_state_embed.weight.to(
                        device=workspace.device,
                        dtype=workspace.dtype,
                    ).unsqueeze(0).expand(b, -1, -1)
                    if (
                        self.core_role_value_state_prompt_query_norm is not None
                        and self.core_role_value_state_prompt_context_norm is not None
                        and self.core_role_value_state_prompt_cross is not None
                        and self.core_role_value_state_prompt_gate is not None
                        and self.core_role_value_state_prompt_output_norm is not None
                        and not bool(disable_core_role_value_prompt_extract)
                    ):
                        prompt_context = self.core_role_value_state_prompt_context_norm(
                            seq
                        ).to(dtype=core_role_tokens.dtype)
                        prompt_delta = self.core_role_value_state_prompt_cross(
                            self.core_role_value_state_prompt_query_norm(
                                core_role_tokens
                            ),
                            prompt_context,
                            attention_mask,
                        )
                        prompt_gate = torch.sigmoid(
                            self.core_role_value_state_prompt_gate(core_role_tokens)
                        )
                        core_role_tokens = (
                            self.core_role_value_state_prompt_output_norm(
                                core_role_tokens + prompt_gate * prompt_delta
                            )
                        )
                    if (
                        self.core_role_value_state_prompt_parity_norm is not None
                        and self.core_role_value_state_prompt_parity_head is not None
                        and self.core_role_value_state_prompt_parity_embed is not None
                        and self.core_role_value_state_prompt_parity_gate is not None
                    ):
                        prompt_mask = attention_mask.to(dtype=seq.dtype).unsqueeze(-1)
                        prompt_denom = prompt_mask.sum(dim=1).clamp_min(1.0)
                        prompt_summary = (seq * prompt_mask).sum(dim=1) / prompt_denom
                        parity_hidden = self.core_role_value_state_prompt_parity_norm(
                            prompt_summary
                        )
                        core_role_value_state_prompt_parity_logits = (
                            self.core_role_value_state_prompt_parity_head(
                                parity_hidden
                            )
                        )
                        parity_probs = torch.softmax(
                            core_role_value_state_prompt_parity_logits.float(),
                            dim=-1,
                        ).to(dtype=core_role_tokens.dtype)
                        parity_embed = self.core_role_value_state_prompt_parity_embed.weight.to(
                            device=core_role_tokens.device,
                            dtype=core_role_tokens.dtype,
                        )
                        parity_state = parity_probs @ parity_embed
                        parity_gate = torch.sigmoid(
                            self.core_role_value_state_prompt_parity_gate(
                                parity_hidden
                            )
                        ).to(dtype=core_role_tokens.dtype)
                        core_role_tokens = core_role_tokens + (
                            parity_gate[:, None, :] * parity_state[:, None, :]
                        )
                    if self.core_typed_register_value_head is not None:
                        core_role_value_state_prompt_logits = (
                            self.core_typed_register_value_head(core_role_tokens)
                            .unsqueeze(1)
                        )
                        if (
                            self.core_role_value_state_prompt_self_condition_value_embed
                            is not None
                            and self.core_role_value_state_prompt_self_condition_gate
                            is not None
                            and self.core_role_value_state_prompt_self_condition_norm
                            is not None
                            and self.core_role_value_state_prompt_self_condition_output_norm
                            is not None
                        ):
                            prompt_value_probs = torch.softmax(
                                core_role_value_state_prompt_logits.squeeze(1).float(),
                                dim=-1,
                            ).to(dtype=core_role_tokens.dtype)
                            prompt_value_embed = (
                                self.core_role_value_state_prompt_self_condition_value_embed.weight.to(
                                    device=core_role_tokens.device,
                                    dtype=core_role_tokens.dtype,
                                )
                            )
                            prompt_value_state = prompt_value_probs @ prompt_value_embed
                            prompt_value_hidden = (
                                self.core_role_value_state_prompt_self_condition_norm(
                                    core_role_tokens
                                )
                            )
                            prompt_value_gate = torch.sigmoid(
                                self.core_role_value_state_prompt_self_condition_gate(
                                    prompt_value_hidden
                                )
                            ).to(dtype=core_role_tokens.dtype)
                            gate_min = min(
                                max(
                                    float(
                                        self.cfg.core_role_value_state_prompt_self_condition_gate_min
                                    ),
                                    0.0,
                                ),
                                1.0,
                            )
                            if gate_min > 0.0:
                                prompt_value_gate = gate_min + (
                                    1.0 - gate_min
                                ) * prompt_value_gate
                            core_role_tokens = (
                                self.core_role_value_state_prompt_self_condition_output_norm(
                                    core_role_tokens
                                    + prompt_value_gate * prompt_value_state
                                )
                            )
                            core_role_value_state_prompt_logits = (
                                self.core_typed_register_value_head(core_role_tokens)
                                .unsqueeze(1)
                            )
                    else:
                        core_role_value_state_prompt_logits = (
                            self._empty_core_role_value_state_logits(
                                self._empty_core_depth_states(workspace)
                            )
                        )
                    source_position_context_seq = text_context_seq
                    source_position_context_mask = text_context_mask
                    source_position_context_token_ids = input_ids
                    if int(source_slot_token_count) > 0:
                        source_slot_copy_token_ids = token_numeric_source_slot_ids[
                            :, : int(source_slot_token_count)
                        ].to(device=input_ids.device, dtype=input_ids.dtype)
                        if token_numeric_source_slot_token_ids is not None:
                            if token_numeric_source_slot_token_ids.ndim != 2:
                                raise ValueError(
                                    "token_numeric_source_slot_token_ids must have shape [batch, slots]"
                                )
                            if int(token_numeric_source_slot_token_ids.shape[0]) != b:
                                raise ValueError(
                                    "token_numeric_source_slot_token_ids batch must match input_ids"
                                )
                            source_slot_copy_token_ids = (
                                token_numeric_source_slot_token_ids[
                                    :, : int(source_slot_token_count)
                                ].to(device=input_ids.device, dtype=input_ids.dtype)
                            )
                        source_position_context_token_ids = torch.cat(
                            [
                                source_slot_copy_token_ids,
                                input_ids,
                            ],
                            dim=1,
                        )
                    if bool(self.cfg.core_source_position_binder_source_slots_only):
                        if int(source_slot_token_count) > 0:
                            source_position_source_seq = (
                                input_text_seq
                                if bool(
                                    self.cfg.core_source_position_binder_raw_source_slots_enabled
                                )
                                else text_context_seq
                            )
                            source_position_source_mask = (
                                input_text_mask
                                if bool(
                                    self.cfg.core_source_position_binder_raw_source_slots_enabled
                                )
                                else text_context_mask
                            )
                            source_position_context_seq = source_position_source_seq[
                                :, : int(source_slot_token_count), :
                            ]
                            source_position_context_token_ids = source_slot_copy_token_ids
                            source_position_context_mask = (
                                source_position_source_mask[
                                    :, : int(source_slot_token_count)
                                ]
                                if source_position_source_mask is not None
                                else None
                            )
                        else:
                            source_position_context_seq = None
                            source_position_context_mask = None
                    (
                        source_position_prompt_logits,
                        source_position_query_states,
                    ) = self._compute_core_source_position_binder_context(
                        source_position_context_seq,
                        source_position_context_mask,
                        reference=workspace,
                        disabled=bool(disable_core_source_position_binder),
                    )
                    if int(source_position_prompt_logits.shape[1]) > 0:
                        gate = torch.sigmoid(
                            self.core_source_position_binder_logit_gate.to(
                                device=source_position_prompt_logits.device,
                                dtype=source_position_prompt_logits.dtype,
                            )
                        )
                        gate_min = min(
                            max(
                                float(
                                    self.cfg.core_source_position_binder_gate_min
                                ),
                                0.0,
                            ),
                            1.0,
                        )
                        if gate_min > 0.0:
                            gate = gate_min + (1.0 - gate_min) * gate
                        if (
                            core_role_value_state_prompt_logits.ndim == 4
                            and tuple(core_role_value_state_prompt_logits.shape)
                            == tuple(source_position_prompt_logits.shape)
                        ):
                            core_role_value_state_prompt_logits = (
                                core_role_value_state_prompt_logits
                                + gate
                                * (
                                    source_position_prompt_logits
                                    - core_role_value_state_prompt_logits
                                )
                            )
                        else:
                            core_role_value_state_prompt_logits = (
                                gate * source_position_prompt_logits
                            )
                        source_position_state_delta = (
                            self._compute_core_source_position_binder_state_delta(
                                source_position_prompt_logits,
                                reference=core_role_tokens,
                            )
                        )
                        if tuple(source_position_state_delta.shape) == tuple(
                            core_role_tokens.shape
                        ):
                            core_role_tokens = (
                                self.core_source_position_binder_state_norm(
                                    core_role_tokens + source_position_state_delta
                                )
                                if self.core_source_position_binder_state_norm
                                is not None
                                else core_role_tokens + source_position_state_delta
                            )
                        core_source_value_prompt_logits = (
                            self._compute_core_source_value_binder_logits(
                                source_position_query_states,
                                reference=workspace,
                                disabled=bool(disable_core_source_value_binder),
                            )
                        )
                        source_value_state_delta = (
                            self._compute_core_source_value_binder_state_delta(
                                core_source_value_prompt_logits,
                                reference=core_role_tokens,
                            )
                        )
                        if tuple(source_value_state_delta.shape) == tuple(
                            core_role_tokens.shape
                        ):
                            core_role_tokens = (
                                self.core_source_value_binder_state_norm(
                                    core_role_tokens + source_value_state_delta
                                )
                                if self.core_source_value_binder_state_norm
                                is not None
                                else core_role_tokens + source_value_state_delta
                            )
                        source_position_query_state_delta = (
                            self._compute_core_source_position_binder_query_state_delta(
                                source_position_query_states,
                                reference=core_role_tokens,
                                disabled=bool(
                                    disable_core_source_position_binder_query_state
                                ),
                            )
                        )
                        if tuple(source_position_query_state_delta.shape) == tuple(
                            core_role_tokens.shape
                        ):
                            core_role_tokens = (
                                self.core_source_position_binder_query_state_norm(
                                    core_role_tokens
                                    + source_position_query_state_delta
                                )
                                if self.core_source_position_binder_query_state_norm
                                is not None
                                else core_role_tokens
                                + source_position_query_state_delta
                            )
                    core_role_token_start = int(core_workspace.shape[1])
                    core_role_token_count = int(core_role_tokens.shape[1])
                    core_workspace = torch.cat([core_workspace, core_role_tokens], dim=1)
                    core_role_mask = workspace_mask.new_ones(
                        (b, int(core_role_tokens.shape[1]))
                    )
                    core_workspace_mask = torch.cat(
                        [core_workspace_mask, core_role_mask],
                        dim=1,
                    )
                z_l, z_h, trajectory, core_halt_info = self.core(
                    core_workspace,
                    attention_mask=core_workspace_mask,
                    context_states=seq,
                    context_mask=attention_mask,
                    disable_context=disable_core_context,
                    state_carry_start=core_role_token_start,
                    state_carry_count=core_role_token_count,
                    disable_state_carry=disable_core_state_carry,
                    enable_halt=enable_core_halt,
                    carry=core_carry,
                    return_carry=return_core_carry,
                    transition_feedback_operation_targets=(
                        core_transition_feedback_operation_targets
                    ),
                    transition_feedback_finality_targets=(
                        core_transition_feedback_finality_targets
                    ),
                    transition_feedback_teacher_forcing=(
                        core_transition_feedback_teacher_forcing
                    ),
                    transition_order_conditioning=(
                        core_transition_order_bottleneck_info["token"]
                    ),
                )
                if bool(zero_core_trajectory):
                    z_l = torch.zeros_like(z_l)
                    z_h = torch.zeros_like(z_h)
                    trajectory = [torch.zeros_like(state) for state in trajectory]
                core_state_mask = core_workspace_mask
                core_context_gate_mean = core_halt_info["context_gate_mean"]
                core_role_value_state_logits = self._compute_core_role_value_state_logits(
                    trajectory,
                    role_token_start=core_role_token_start,
                    reference=workspace,
                )
                core_role_value_transition_logits = (
                    self._compute_core_role_value_transition_logits(
                        trajectory,
                        role_token_start=core_role_token_start,
                        reference=workspace,
                    )
                )
                (
                    core_role_value_delta_logits,
                    core_role_value_delta_gate_mean,
                ) = self._compute_core_role_value_delta_logits(
                    trajectory,
                    role_token_start=core_role_token_start,
                    reference=workspace,
                    disabled=bool(disable_core_role_value_delta),
                )
                if int(core_role_value_delta_logits.shape[1]) > 0:
                    core_role_value_state_logits = core_role_value_delta_logits
                (
                    core_value_delta_code_logits,
                    core_value_delta_role_logits,
                    core_value_delta_code_gate_mean,
                ) = self._compute_core_value_delta_code_outputs(
                    trajectory,
                    role_token_start=core_role_token_start,
                    reference=workspace,
                    disabled=bool(disable_core_value_delta_code),
                )
                if int(core_value_delta_role_logits.shape[1]) > 0:
                    core_role_value_state_logits = core_value_delta_role_logits
                core_typed_register_info = self._compute_core_typed_register_outputs(
                    trajectory,
                    role_token_start=core_role_token_start,
                    reference=workspace,
                    disabled=bool(disable_core_typed_register_executor),
                )
                core_role_value_template_info = (
                    self._compute_core_role_value_template_outputs(
                        trajectory,
                        role_token_start=core_role_token_start,
                        reference=workspace,
                        disabled=bool(disable_core_typed_register_executor),
                    )
                )
                core_role_value_template_logits = core_role_value_template_info[
                    "template_logits"
                ]
                if int(core_role_value_template_info["value_logits"].shape[1]) > 0:
                    core_typed_register_info = {
                        **core_typed_register_info,
                        "value_logits": core_role_value_template_info["value_logits"],
                    }
                if (
                    bool(
                        self.cfg.core_typed_register_prompt_first_transition_readout_enabled
                    )
                    and int(core_role_value_state_prompt_logits.shape[1]) > 0
                    and int(core_typed_register_info["transition_logits"].shape[1]) > 0
                ):
                    prompt_first_value_logits = core_role_value_state_prompt_logits[
                        :, :1, :, :
                    ]
                    core_typed_register_info = {
                        **core_typed_register_info,
                        "value_logits": torch.cat(
                            [
                                prompt_first_value_logits,
                                core_typed_register_info["transition_logits"],
                            ],
                            dim=1,
                        ),
                    }
                if int(core_typed_register_info["value_logits"].shape[1]) > 0:
                    core_role_value_state_logits = core_typed_register_info[
                        "value_logits"
                    ]
                if core_role_token_start is not None:
                    z_l = z_l[:, :core_role_token_start, :]
                    z_h = z_h[:, :core_role_token_start, :]
                    trajectory = [
                        state[:, :core_role_token_start, :]
                        for state in trajectory
                    ]
                    core_state_mask = core_state_mask[:, :core_role_token_start]
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
            transition_state_joint_logits = self._compute_transition_state_joint_logits(
                core_depth_states,
                prompt_context_seq=text_context_seq,
                prompt_context_mask=text_context_mask,
                disabled=bool(disable_transition_state or disable_core),
            )
            transition_state_joint_answer_embeddings = (
                self._compute_transition_state_joint_answer_embeddings(
                    transition_state_joint_logits,
                    disabled=bool(
                        disable_transition_state
                        or disable_core
                        or disable_transition_state_joint_answer_bridge
                    ),
                )
            )
            transition_state_final_answer_embedding = (
                self._compute_transition_state_final_answer_embedding(
                    core_depth_states,
                    transition_state_joint_logits,
                    disabled=bool(
                        disable_transition_state
                        or disable_core
                        or disable_transition_state_final_answer_binder
                    ),
                )
            )
            transition_state_sequence_logits = (
                self._compute_transition_state_sequence_logits(
                    core_depth_states,
                    disabled=bool(disable_transition_state or disable_core),
                )
            )
            transition_value_state_logits = (
                self._compute_transition_value_state_logits(
                    core_depth_states,
                    disabled=bool(disable_transition_state or disable_core),
                )
            )
            factorized_value_state_info = (
                self._compute_factorized_value_state_outputs(
                    core_depth_states,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=bool(disable_transition_state or disable_core),
                )
            )
            factorized_value_state_logits = factorized_value_state_info["slot_logits"]
            factorized_value_state_kind_logits = factorized_value_state_info[
                "kind_logits"
            ]
            role_value_state_logits = factorized_value_state_info["role_logits"]
            if disable_core:
                core_role_value_state_logits = self._empty_core_role_value_state_logits(
                    core_depth_states
                )
                core_role_value_transition_logits = (
                    self._empty_core_role_value_transition_logits(core_depth_states)
                )
                core_role_value_delta_gate_mean = workspace.new_empty((b, 0))
                core_value_delta_code_logits = self._empty_core_value_delta_code_logits(
                    core_depth_states
                )
                core_value_delta_code_gate_mean = workspace.new_empty((b, 0))
                core_typed_register_info = self._empty_core_typed_register_outputs(
                    core_depth_states
                )
            if self.factorized_value_state_head is not None:
                transition_value_state_logits = factorized_value_state_logits
            primitive_transition_info = self._compute_primitive_transition_outputs(
                core_depth_states,
                prompt_context_seq=(
                    None if bool(disable_core_primitive_prompt_context) else text_context_seq
                ),
                prompt_context_mask=(
                    None if bool(disable_core_primitive_prompt_context) else text_context_mask
                ),
                disabled=bool(disable_core),
            )
            typed_algorithmic_value_state_info = (
                self._compute_typed_algorithmic_value_state_outputs(
                    core_depth_states,
                    transition_state_joint_logits=transition_state_joint_logits,
                    primitive_operation_logits=primitive_transition_info.get(
                        "operation_logits"
                    ),
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=bool(
                        disable_transition_state
                        or disable_core
                        or disable_typed_algorithmic_value_state
                    ),
                    recurrent_disabled=bool(
                        disable_typed_algorithmic_value_state_recurrent
                    ),
                )
            )
            typed_algorithmic_value_state_answer_bridge_info = (
                self._compute_typed_algorithmic_value_state_answer_bridge(
                    typed_algorithmic_value_state_info,
                    reference=core_depth_states,
                    disabled=bool(
                        disable_transition_state
                        or disable_core
                        or disable_typed_algorithmic_value_state
                        or disable_typed_algorithmic_value_state_answer_bridge
                    ),
                )
            )
            (
                core_primitive_role_value_state_logits,
                core_primitive_role_value_update_gate,
            ) = (
                self._compute_core_primitive_role_value_state_logits(
                    primitive_transition_info,
                    prompt_logits=core_role_value_state_prompt_logits,
                    source_value_logits=core_source_value_prompt_logits,
                    fallback_logits=core_role_value_state_logits,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    reference=core_depth_states,
                    disabled=bool(
                        disable_core or disable_core_primitive_role_value_executor
                    ),
                    disable_prompt_context=bool(
                        disable_core_primitive_prompt_context
                    ),
                )
            )
            selector_info = self._compute_core_primitive_typed_selector_outputs(
                primitive_logits=core_primitive_role_value_state_logits,
                typed_logits=core_typed_register_info["value_logits"],
                reference=core_depth_states,
                disabled=bool(disable_core),
            )
            if int(selector_info["selected_logits"].shape[1]) > 0:
                core_role_value_state_logits = selector_info["selected_logits"]
            elif int(core_primitive_role_value_state_logits.shape[1]) > 0:
                core_role_value_state_logits = core_primitive_role_value_state_logits
            transition_state_joint_logits = (
                self._apply_transition_state_joint_operation_residual(
                    transition_state_joint_logits,
                    primitive_transition_info,
                    disabled=bool(disable_transition_state or disable_core),
                )
            )
            transition_state_joint_logits = (
                self._apply_transition_state_joint_code_residual(
                    transition_state_joint_logits,
                    transition_state_code_info,
                    disabled=bool(disable_transition_state or disable_core),
                )
            )
            transition_phase_logits = self._compute_transition_phase_logits(
                core_depth_states,
                prompt_context_seq=text_context_seq,
                prompt_context_mask=text_context_mask,
                disabled=bool(disable_core),
            )
            transition_state_joint_logits = (
                self._apply_transition_state_joint_phase_residual(
                    transition_state_joint_logits,
                    transition_phase_logits,
                    core_depth_states,
                    disabled=bool(disable_transition_state or disable_core),
                )
            )
            transition_source_router_logits = (
                self._compute_transition_source_router_logits(
                    core_depth_states,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=bool(disable_core),
                )
            )
            core_role_value_state_answer_bridge_info = (
                self._compute_core_role_value_state_answer_bridge(
                    core_role_value_state_logits,
                    prompt_context_seq=text_context_seq,
                    prompt_context_mask=text_context_mask,
                    disabled=bool(disable_core_role_value_answer_bridge),
                )
            )
            core_role_value_state_answer_final_embedding = (
                self._compute_core_role_value_state_answer_final_embedding(
                    core_role_value_state_answer_bridge_info["tokens"],
                    disabled=bool(
                        disable_core_role_value_answer_bridge
                        or disable_core_role_value_answer_final_binder
                    ),
                )
            )
            core_depth_last_logits = (
                self._core_depth_last_logits(
                    trajectory,
                    text_context_seq=text_context_seq,
                    text_context_mask=text_context_mask,
                    workspace_mask=core_state_mask,
                    transition_state_features=transition_state_info["features"],
                    transition_state_code_embeddings=transition_state_code_info[
                        "embeddings"
                    ],
                    transition_state_joint_answer_embeddings=(
                        transition_state_joint_answer_embeddings
                    ),
                    typed_algorithmic_answer_tokens=(
                        typed_algorithmic_value_state_answer_bridge_info["tokens"]
                    ),
                    core_role_value_answer_tokens=core_role_value_state_answer_bridge_info[
                        "tokens"
                    ],
                    disable_answer_state_loop_selective_context=bool(
                        disable_answer_state_loop_selective_context
                    ),
                    force_answer_state_loop_dense_context=bool(
                        force_answer_state_loop_dense_context
                    ),
                    disable_answer_state_loop_hidden_bridge=bool(
                        disable_answer_state_loop_hidden_bridge
                    ),
                    disable_answer_state_loop_talker=bool(
                        disable_answer_state_loop_talker
                    ),
                )
                if return_core_depth_logits
                else self._empty_core_depth_last_logits(workspace)
            )
            core_depth_text_logits = (
                self._core_depth_text_logits(
                    trajectory,
                    text_context_seq=text_context_seq,
                    text_context_mask=text_context_mask,
                    workspace_mask=core_state_mask,
                    input_seq_len=s,
                    transition_state_features=transition_state_info["features"],
                    transition_state_code_embeddings=transition_state_code_info[
                        "embeddings"
                    ],
                    transition_state_joint_answer_embeddings=(
                        transition_state_joint_answer_embeddings
                    ),
                    typed_algorithmic_answer_tokens=(
                        typed_algorithmic_value_state_answer_bridge_info["tokens"]
                    ),
                    core_role_value_answer_tokens=core_role_value_state_answer_bridge_info[
                        "tokens"
                    ],
                    disable_answer_state_loop_selective_context=bool(
                        disable_answer_state_loop_selective_context
                    ),
                    force_answer_state_loop_dense_context=bool(
                        force_answer_state_loop_dense_context
                    ),
                    disable_answer_state_loop_hidden_bridge=bool(
                        disable_answer_state_loop_hidden_bridge
                    ),
                    disable_answer_state_loop_talker=bool(
                        disable_answer_state_loop_talker
                    ),
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
                core_state_mask,
            )
            core_to_text_gate = torch.sigmoid(self.core_to_text_gate(text_context_seq))
            gate_min = min(max(float(self.cfg.core_to_text_gate_min), 0.0), 1.0)
            if gate_min != 0.0:
                core_to_text_gate = gate_min + (1.0 - gate_min) * core_to_text_gate
            text_context_seq = text_context_seq + core_to_text_gate * core_delta
            core_to_text_gate_mean = core_to_text_gate.mean(dim=(1, 2))
        if not disable_workspace:
            seq = torch.cat([z_h, text_context_seq], dim=1)
            attention_mask = torch.cat([core_state_mask, text_context_mask], dim=1)
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
                elif self.controller_signal_source == "learned_core_trajectory":
                    signal_source = self._controller_signal_trajectory_features(
                        core_depth_states,
                        target_steps=max(1, int(self.cfg.outer_steps)),
                        d_model=int(self.cfg.d_model),
                    )
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
                "core_carry": core_halt_info.get("carry", None),
                "core_state_carry_gate_mean": core_halt_info[
                    "state_carry_gate_mean"
                ],
                "core_transition_feedback_operation_logits": core_halt_info[
                    "transition_feedback_operation_logits"
                ],
                "core_transition_feedback_finality_logits": core_halt_info[
                    "transition_feedback_finality_logits"
                ],
                "core_transition_feedback_gate_mean": core_halt_info[
                    "transition_feedback_gate_mean"
                ],
                "core_transition_order_bottleneck_logits": (
                    core_transition_order_bottleneck_info["logits"]
                ),
                "core_transition_order_bottleneck_gate_mean": (
                    core_transition_order_bottleneck_info["gate_mean"]
                ),
                "core_transition_order_conditioning_gate_mean": core_halt_info[
                    "transition_order_conditioning_gate_mean"
                ],
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
                "transition_state_joint_logits": transition_state_joint_logits,
                "transition_state_joint_answer_embeddings": (
                    transition_state_joint_answer_embeddings
                ),
                "transition_state_final_answer_embedding": (
                    transition_state_final_answer_embedding
                ),
                "transition_state_sequence_logits": transition_state_sequence_logits,
                "transition_value_state_logits": transition_value_state_logits,
                "factorized_value_state_logits": factorized_value_state_logits,
                "factorized_value_state_kind_logits": factorized_value_state_kind_logits,
                "role_value_state_logits": role_value_state_logits,
                "typed_algorithmic_kind_logits": typed_algorithmic_value_state_info[
                    "kind_logits"
                ],
                "typed_algorithmic_raw_list_offset_logits": (
                    typed_algorithmic_value_state_info["raw_list_offset_logits"]
                ),
                "typed_algorithmic_doubled_list_offset_logits": (
                    typed_algorithmic_value_state_info["doubled_list_offset_logits"]
                ),
                "typed_algorithmic_scalar_coeff_logits": (
                    typed_algorithmic_value_state_info["scalar_coeff_logits"]
                ),
                "typed_algorithmic_scalar_coeff_value": (
                    typed_algorithmic_value_state_info["scalar_coeff_value"]
                ),
                "typed_algorithmic_scalar_offset_logits": (
                    typed_algorithmic_value_state_info["scalar_offset_logits"]
                ),
                "typed_algorithmic_scalar_offset_value": (
                    typed_algorithmic_value_state_info["scalar_offset_value"]
                ),
                "typed_algorithmic_scalar_residual_logits": (
                    typed_algorithmic_value_state_info["scalar_residual_logits"]
                ),
                "typed_algorithmic_scalar_residual_value": (
                    typed_algorithmic_value_state_info["scalar_residual_value"]
                ),
                "typed_algorithmic_scalar_residual_delta_logits": (
                    typed_algorithmic_value_state_info[
                        "scalar_residual_delta_logits"
                    ]
                ),
                "typed_algorithmic_final_residual_logits": (
                    typed_algorithmic_value_state_info["final_residual_logits"]
                ),
                "typed_algorithmic_final_residual_value": (
                    typed_algorithmic_value_state_info["final_residual_value"]
                ),
                "typed_algorithmic_value_state_answer_bridge_tokens": (
                    typed_algorithmic_value_state_answer_bridge_info["tokens"]
                ),
                "typed_algorithmic_value_state_answer_bridge_gate_mean": (
                    typed_algorithmic_value_state_answer_bridge_info["gate_mean"]
                ),
                "core_role_value_state_logits": core_role_value_state_logits,
                "core_role_value_state_prompt_logits": (
                    core_role_value_state_prompt_logits
                ),
                "core_source_value_prompt_logits": core_source_value_prompt_logits,
                "core_role_value_state_prompt_parity_logits": (
                    core_role_value_state_prompt_parity_logits
                ),
                "core_primitive_role_value_state_logits": (
                    core_primitive_role_value_state_logits
                ),
                "core_primitive_role_value_update_gate": (
                    core_primitive_role_value_update_gate
                ),
                "core_role_value_template_logits": core_role_value_template_logits,
                "core_role_value_state_answer_bridge_gate_mean": (
                    core_role_value_state_answer_bridge_info["gate_mean"]
                ),
                "core_role_value_state_answer_final_embedding": (
                    core_role_value_state_answer_final_embedding
                ),
                "core_role_value_delta_gate_mean": core_role_value_delta_gate_mean,
                "core_value_delta_code_logits": core_value_delta_code_logits,
                "core_value_delta_code_gate_mean": core_value_delta_code_gate_mean,
                "core_typed_register_operation_logits": core_typed_register_info[
                    "operation_logits"
                ],
                "core_typed_register_value_logits": core_typed_register_info[
                    "value_logits"
                ],
                "core_typed_register_transition_logits": core_typed_register_info[
                    "transition_logits"
                ],
                "core_typed_register_gate_mean": core_typed_register_info[
                    "gate_mean"
                ],
                "core_role_value_transition_logits": core_role_value_transition_logits,
                "primitive_transition_operation_logits": primitive_transition_info[
                    "operation_logits"
                ],
                "transition_phase_logits": transition_phase_logits,
                "transition_source_router_logits": transition_source_router_logits,
                "core_context_gate_mean": core_context_gate_mean,
                "core_output_blend_gate_mean": core_output_blend_gate_mean,
                "core_to_text_gate_mean": core_to_text_gate_mean,
                "answer_state_loop_recurrent_gate_mean": seq.new_empty((b, 0)),
                "answer_state_loop_halt_logits": seq.new_empty((b, 0)),
                "temporal_spatial_context_token_count": temporal_spatial_context_token_count,
                "token_numeric_source_slot_token_count": source_slot_token_count,
                "token_numeric_source_slot_parity_logits": source_slot_parity_logits,
                "token_numeric_source_slot_predicate_logits": (
                    source_slot_predicate_logits
                ),
                "workspace_update_gate_mean": workspace_update_gate_mean,
                "workspace_memory_token_count": workspace_memory_token_count,
                "workspace_memory_present": workspace_memory_present,
                **ctrl,
            }
        logit_token_indices_tensor = None
        logit_prev_token_ids = input_ids
        if logit_token_indices is not None:
            logit_token_indices_tensor = torch.as_tensor(
                logit_token_indices,
                device=input_ids.device,
                dtype=torch.long,
            ).reshape(-1)
            if int(logit_token_indices_tensor.numel()) == 0:
                raise ValueError("logit_token_indices must not be empty")
            if (
                int(logit_token_indices_tensor.min().item()) < 0
                or int(logit_token_indices_tensor.max().item()) >= int(s)
            ):
                raise ValueError("logit_token_indices must index input token positions")
            logit_seq = seq[:, -int(s) :, :].index_select(
                1,
                logit_token_indices_tensor,
            )
            logit_prev_token_ids = input_ids.index_select(
                1,
                logit_token_indices_tensor,
            )
            logit_input_len = int(logit_token_indices_tensor.numel())
            selected_logit_positions_only = True
        else:
            logit_seq = seq
            logit_input_len = int(s)
            selected_logit_positions_only = False

        # Apply RI-4 memory residual uniformly via the helper (clean, no duplication).
        logit_seq = self._apply_ri4_memory_residual(logit_seq)

    def _apply_ri4_memory_residual(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply RI-4 memory residual to a hidden tensor (for uniform participation
        across answer paths). Uses the same sources as the main injection.
        """
        # Prefer passed kwarg if the method has access (via self or closure), else PoC attribute.
        # For simplicity in the current structure, read from the same fallback sources.
        ri4_res = getattr(self, '_ri4_memory_residual', None)
        ri4_scale = float(getattr(self, '_ri4_memory_residual_scale', 0.3))
        if ri4_res is None:
            # Fallback to the kwarg if somehow passed differently, but in practice the attribute is the PoC source.
            pass
        if ri4_res is not None:
            try:
                res = ri4_res
                if res.dim() == 1:
                    res = res.unsqueeze(0).unsqueeze(0)
                elif res.dim() == 2:
                    res = res.unsqueeze(1)
                hidden = hidden + res.to(device=hidden.device, dtype=hidden.dtype) * ri4_scale
            except Exception:
                pass
        return hidden

        qtrm_logits = self.lm_head(logit_seq) * float(self.cfg.qtrm_logits_scale)
        answer_bottleneck_logits = self._empty_answer_bottleneck_logits(
            qtrm_logits,
            logit_input_len,
        )
        answer_bottleneck_hidden = seq.new_empty((b, 0, self.cfg.d_model))
        core_loop_readout_logits = self._empty_core_loop_readout_logits(qtrm_logits)
        core_loop_readout_hidden = self._empty_core_loop_readout_hidden(seq)
        answer_state_loop_logits = self._empty_answer_state_loop_logits(qtrm_logits)
        answer_state_loop_future_token_logits = (
            self._empty_answer_state_loop_future_token_logits(qtrm_logits)
        )
        core_role_value_vocab_renderer_logits = (
            self._empty_core_role_value_state_vocab_renderer_logits(
                qtrm_logits,
                logit_input_len,
            )
        )
        answer_state_loop_hidden = self._empty_answer_state_loop_hidden(seq)
        answer_state_loop_depth_hidden = self._empty_answer_state_loop_depth_hidden(
            seq,
            logit_input_len,
        )
        answer_state_loop_recurrent_gate_mean = seq.new_empty((b, 0))
        answer_state_loop_halt_logits = seq.new_empty((b, 0))
        answer_state_loop_free_transformer_latent_kl = seq.new_zeros(())
        answer_state_loop_free_transformer_gate_mean = seq.new_zeros(())
        answer_residual_governor_logits = qtrm_logits.new_empty((b, 0))
        answer_residual_governor_gate = qtrm_logits.new_empty((b, 0))
        qtrm_residual_logits = qtrm_logits
        if self.answer_bottleneck_cross is not None:
            text_offset = 0 if selected_logit_positions_only else qtrm_logits.shape[1] - s
            core_required_but_disabled = bool(
                self.cfg.answer_bottleneck_requires_core
                and (disable_workspace or disable_core)
            )
            if core_required_but_disabled:
                answer_bottleneck_logits = qtrm_logits.new_zeros(
                    (b, logit_input_len, qtrm_logits.shape[-1])
                )
                qtrm_residual_logits = torch.zeros_like(qtrm_logits)
            else:
                answer_bottleneck_logits, answer_bottleneck_hidden = self._compute_answer_bottleneck_outputs(
                    seq,
                    z_h=z_h,
                    workspace_mask=core_state_mask,
                    input_seq_len=s,
                    query_token_indices=logit_token_indices_tensor,
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
                if selected_logit_positions_only:
                    qtrm_residual_logits = answer_text_residual
                else:
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
            text_offset = (
                0
                if selected_logit_positions_only
                else qtrm_residual_logits.shape[1] - s
            )
            if disable_answer_residual_governor:
                answer_residual_governor_logits = qtrm_logits.new_zeros(
                    (b, logit_input_len)
                )
                answer_residual_governor_gate = qtrm_logits.new_ones(
                    (b, logit_input_len)
                )
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
                if selected_logit_positions_only:
                    qtrm_residual_logits = gated_answer_residual
                else:
                    qtrm_residual_logits = torch.cat(
                        [
                            qtrm_residual_logits[:, :text_offset, :],
                            gated_answer_residual,
                        ],
                        dim=1,
                    )
        if self.core_loop_readout_cross is not None:
            text_offset = 0 if selected_logit_positions_only else qtrm_logits.shape[1] - s
            core_required_but_disabled = bool(
                self.cfg.core_loop_readout_requires_core
                and (disable_workspace or disable_core or len(trajectory) == 0)
            )
            if core_required_but_disabled:
                core_loop_readout_logits = qtrm_logits.new_zeros(
                    (b, logit_input_len, qtrm_logits.shape[-1])
                )
                core_loop_readout_hidden = seq.new_zeros(
                    (b, logit_input_len, self.cfg.d_model)
                )
                qtrm_residual_logits = torch.zeros_like(qtrm_logits)
            else:
                (
                    core_loop_readout_logits,
                    core_loop_readout_hidden,
                ) = self._compute_core_loop_readout_outputs(
                    text_context_seq,
                    z_h=z_h,
                    workspace_mask=core_state_mask,
                    input_seq_len=s,
                    query_token_indices=logit_token_indices_tensor,
                )
                if selected_logit_positions_only:
                    qtrm_residual_logits = core_loop_readout_logits
                else:
                    residual_prefix = qtrm_logits.new_zeros(
                        (b, text_offset, qtrm_logits.shape[-1])
                    )
                    qtrm_residual_logits = torch.cat(
                        [residual_prefix, core_loop_readout_logits],
                        dim=1,
                    )
        if self.answer_state_loop_cross is not None:
            text_offset = 0 if selected_logit_positions_only else qtrm_logits.shape[1] - s
            core_required_but_disabled = bool(
                self.cfg.answer_state_loop_requires_core
                and (disable_workspace or disable_core or len(trajectory) == 0)
            )
            if core_required_but_disabled:
                answer_state_loop_logits = qtrm_logits.new_zeros(
                    (b, logit_input_len, qtrm_logits.shape[-1])
                )
                answer_state_loop_future_token_logits = (
                    self._empty_answer_state_loop_future_token_logits(qtrm_logits)
                )
                answer_state_loop_hidden = seq.new_zeros(
                    (b, logit_input_len, self.cfg.d_model)
                )
                answer_state_loop_depth_hidden = seq.new_zeros(
                    (b, 0, logit_input_len, self.cfg.d_model)
                )
                answer_state_loop_halt_logits = seq.new_empty((b, 0))
                answer_state_loop_free_transformer_latent_kl = seq.new_zeros(())
                answer_state_loop_free_transformer_gate_mean = seq.new_zeros(())
                qtrm_residual_logits = torch.zeros_like(qtrm_logits)
            else:
                (
                    answer_state_loop_logits,
                    answer_state_loop_hidden,
                    answer_state_loop_depth_hidden,
                    answer_state_loop_recurrent_gate_mean,
                    answer_state_loop_halt_logits,
                    answer_state_loop_free_transformer_latent_kl,
                    answer_state_loop_free_transformer_gate_mean,
                ) = self._compute_answer_state_loop_outputs(
                    text_context_seq,
                    trajectory=trajectory,
                    text_context_mask=text_context_mask,
                    workspace_mask=core_state_mask,
                    input_seq_len=s,
                    transition_state_features=transition_state_info["features"],
                    transition_state_code_embeddings=transition_state_code_info[
                        "embeddings"
                    ],
                    transition_state_joint_answer_embeddings=(
                        transition_state_joint_answer_embeddings
                    ),
                    transition_state_final_answer_embedding=(
                        transition_state_final_answer_embedding
                    ),
                    transition_state_joint_logits=transition_state_joint_logits,
                    typed_algorithmic_answer_tokens=(
                        typed_algorithmic_value_state_answer_bridge_info["tokens"]
                    ),
                    core_role_value_answer_tokens=core_role_value_state_answer_bridge_info[
                        "tokens"
                    ],
                    core_role_value_final_answer_embedding=(
                        core_role_value_state_answer_final_embedding
                    ),
                    prev_token_ids=logit_prev_token_ids,
                    query_token_indices=logit_token_indices_tensor,
                    disable_recurrent_block=bool(disable_answer_state_loop_recurrent),
                    disable_selective_context=bool(
                        disable_answer_state_loop_selective_context
                    ),
                    force_dense_context=bool(force_answer_state_loop_dense_context),
                    disable_finality_gate=bool(
                        disable_answer_state_loop_finality_gate
                        or disable_transition_state
                    ),
                    disable_halt_gate=bool(disable_answer_state_loop_halt_gate),
                    disable_hidden_bridge=bool(
                        disable_answer_state_loop_hidden_bridge
                    ),
                    disable_next_token_decoder=bool(
                        disable_answer_state_loop_next_token_decoder
                    ),
                    disable_free_transformer_latent=bool(
                        disable_answer_state_loop_free_transformer_latent
                    ),
                    disable_talker=bool(disable_answer_state_loop_talker),
                )
                if not bool(disable_answer_state_loop_finality_selector):
                    (
                        answer_state_loop_logits,
                        answer_state_loop_hidden,
                    ) = self._select_answer_state_loop_by_finality(
                        answer_state_loop_logits,
                        answer_state_loop_hidden,
                        answer_state_loop_depth_hidden,
                        transition_state_joint_logits,
                        disabled=bool(disable_transition_state),
                        disable_hidden_bridge=bool(
                            disable_answer_state_loop_hidden_bridge
                        ),
                        disable_next_token_decoder=bool(
                            disable_answer_state_loop_next_token_decoder
                        ),
                        disable_free_transformer_latent=bool(
                            disable_answer_state_loop_free_transformer_latent
                        ),
                        prev_token_ids=logit_prev_token_ids,
                    )
                answer_state_loop_future_token_logits = (
                    self._answer_state_loop_future_token_logits(
                        answer_state_loop_hidden,
                        disable_hidden_bridge=bool(
                            disable_answer_state_loop_hidden_bridge
                        ),
                        disable_next_token_decoder=bool(
                            disable_answer_state_loop_next_token_decoder
                        ),
                    )
                )
                if selected_logit_positions_only:
                    qtrm_residual_logits = answer_state_loop_logits
                else:
                    residual_prefix = qtrm_logits.new_zeros(
                        (b, text_offset, qtrm_logits.shape[-1])
                    )
                    qtrm_residual_logits = torch.cat(
                        [residual_prefix, answer_state_loop_logits],
                        dim=1,
                    )
        if self.core_role_value_state_vocab_renderer_cross is not None:
            text_offset = (
                0
                if selected_logit_positions_only
                else qtrm_residual_logits.shape[1] - s
            )
            if bool(
                self.cfg.core_role_value_state_vocab_renderer_replace_residual_enabled
            ):
                if selected_logit_positions_only:
                    qtrm_residual_logits = torch.zeros_like(qtrm_residual_logits)
                else:
                    qtrm_residual_logits = torch.cat(
                        [
                            qtrm_residual_logits[:, :text_offset, :],
                            torch.zeros_like(qtrm_residual_logits[:, text_offset:, :]),
                        ],
                        dim=1,
                    )
            role_value_vocab_renderer_extra_tokens = None
            if bool(
                self.cfg.core_role_value_state_vocab_renderer_transition_context_enabled
            ):
                extra_tokens = []
                if (
                    transition_state_joint_answer_embeddings is not None
                    and transition_state_joint_answer_embeddings.ndim == 3
                    and transition_state_joint_answer_embeddings.numel() != 0
                ):
                    extra_tokens.append(transition_state_joint_answer_embeddings)
                if (
                    transition_state_final_answer_embedding is not None
                    and transition_state_final_answer_embedding.ndim == 3
                    and transition_state_final_answer_embedding.numel() != 0
                ):
                    extra_tokens.append(transition_state_final_answer_embedding)
                primitive_operation_logits = primitive_transition_info.get(
                    "operation_logits"
                )
                if (
                    primitive_operation_logits is not None
                    and primitive_operation_logits.ndim == 3
                    and primitive_operation_logits.numel() != 0
                    and self.core_primitive_role_value_operation_embed is not None
                ):
                    op_embed = (
                        self.core_primitive_role_value_operation_embed.weight.to(
                            device=primitive_operation_logits.device,
                            dtype=primitive_operation_logits.dtype,
                        )
                    )
                    op_probs = torch.softmax(
                        primitive_operation_logits.float(),
                        dim=-1,
                    ).to(dtype=primitive_operation_logits.dtype)
                    extra_tokens.append(op_probs @ op_embed)
                if extra_tokens:
                    role_value_vocab_renderer_extra_tokens = torch.cat(
                        extra_tokens,
                        dim=1,
                    )
            if bool(
                self.cfg.core_role_value_state_vocab_renderer_source_state_tokens_enabled
            ):
                if (
                    source_position_state_delta.ndim == 3
                    and source_position_state_delta.numel() != 0
                    and int(source_position_state_delta.shape[0]) == b
                    and int(source_position_state_delta.shape[-1]) == self.cfg.d_model
                ):
                    source_tokens = source_position_state_delta.to(
                        device=text_context_seq.device,
                        dtype=text_context_seq.dtype,
                    )
                    if role_value_vocab_renderer_extra_tokens is None:
                        role_value_vocab_renderer_extra_tokens = source_tokens
                    else:
                        role_value_vocab_renderer_extra_tokens = torch.cat(
                            [
                                role_value_vocab_renderer_extra_tokens,
                                source_tokens,
                            ],
                            dim=1,
                        )
            core_role_value_vocab_renderer_logits = (
                self._compute_core_role_value_state_vocab_renderer_logits(
                    text_context_seq,
                    core_role_value_state_answer_bridge_info["tokens"],
                    extra_state_tokens=role_value_vocab_renderer_extra_tokens,
                    source_copy_position_logits=(
                        self._mask_source_copy_position_logits_to_answer_roles(
                            self._select_source_copy_position_logits_for_renderer(
                                source_position_prompt_logits=source_position_prompt_logits,
                                core_role_value_state_logits=core_role_value_state_logits,
                                core_primitive_role_value_state_logits=(
                                    core_primitive_role_value_state_logits
                                ),
                                bridge_tokens=core_role_value_state_answer_bridge_info[
                                    "tokens"
                                ],
                            )
                        )
                    ),
                    source_copy_token_ids=source_position_context_token_ids,
                    source_copy_token_mask=source_position_context_mask,
                    source_copy_token_span_ids=token_numeric_source_slot_token_span_ids,
                    source_copy_token_span_mask=token_numeric_source_slot_token_span_mask,
                    source_copy_input_ids=input_ids,
                    input_seq_len=s,
                    query_token_indices=logit_token_indices_tensor,
                    disabled=bool(
                        disable_core_role_value_answer_bridge
                        or disable_core_role_value_vocab_renderer
                    ),
                )
            )
            if selected_logit_positions_only:
                qtrm_residual_logits = (
                    qtrm_residual_logits + core_role_value_vocab_renderer_logits
                )
            else:
                qtrm_residual_logits = torch.cat(
                    [
                        qtrm_residual_logits[:, :text_offset, :],
                        qtrm_residual_logits[:, text_offset:, :]
                        + core_role_value_vocab_renderer_logits,
                    ],
                    dim=1,
                )
        if self.cfg.qtrm_residual_clamp is not None:
            clamp = abs(float(self.cfg.qtrm_residual_clamp))
            qtrm_residual_logits = qtrm_residual_logits.clamp(min=-clamp, max=clamp)
        residual_gate = qtrm_logits.new_ones((b,))
        if self.cfg.qtrm_residual_gate_enabled and not disable_qtrm_residual_gate:
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
            text_offset = 0 if selected_logit_positions_only else logits.shape[1] - s
            text_residual_logits = qtrm_residual_logits[:, text_offset:, :]
            donor_text_logits = donor_logits.to(device=logits.device, dtype=logits.dtype)
            if selected_logit_positions_only:
                donor_text_logits = donor_text_logits.index_select(
                    1,
                    logit_token_indices_tensor,
                )
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
            if selected_logit_positions_only:
                logits = fused_text_logits
            else:
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
            "answer_state_loop_future_token_logits": (
                answer_state_loop_future_token_logits
            ),
            "core_role_value_vocab_renderer_logits": (
                core_role_value_vocab_renderer_logits
            ),
            "answer_state_loop_hidden": answer_state_loop_hidden,
            "answer_state_loop_depth_hidden": answer_state_loop_depth_hidden,
            "answer_state_loop_recurrent_gate_mean": answer_state_loop_recurrent_gate_mean,
            "answer_state_loop_halt_logits": answer_state_loop_halt_logits,
            "answer_state_loop_free_transformer_latent_kl": (
                answer_state_loop_free_transformer_latent_kl
            ),
            "answer_state_loop_free_transformer_gate_mean": (
                answer_state_loop_free_transformer_gate_mean
            ),
            "token_numeric_source_slot_token_count": source_slot_token_count,
            "token_numeric_source_slot_parity_logits": source_slot_parity_logits,
            "token_numeric_source_slot_predicate_logits": (
                source_slot_predicate_logits
            ),
            "answer_residual_governor_logits": answer_residual_governor_logits,
            "answer_residual_governor_gate": answer_residual_governor_gate,
            "qtrm_residual_gate": residual_gate,
            "donor_qtrm_conflict_gate": donor_qtrm_conflict_gate,
            "logit_token_indices": (
                logit_token_indices_tensor
                if logit_token_indices_tensor is not None
                else input_ids.new_empty((0,), dtype=torch.long)
            ),
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
            "core_carry": core_halt_info.get("carry", None),
            "core_state_carry_gate_mean": core_halt_info["state_carry_gate_mean"],
            "core_transition_feedback_operation_logits": core_halt_info[
                "transition_feedback_operation_logits"
            ],
            "core_transition_feedback_finality_logits": core_halt_info[
                "transition_feedback_finality_logits"
            ],
            "core_transition_feedback_gate_mean": core_halt_info[
                "transition_feedback_gate_mean"
            ],
            "core_transition_order_bottleneck_logits": (
                core_transition_order_bottleneck_info["logits"]
            ),
            "core_transition_order_bottleneck_gate_mean": (
                core_transition_order_bottleneck_info["gate_mean"]
            ),
            "core_transition_order_conditioning_gate_mean": core_halt_info[
                "transition_order_conditioning_gate_mean"
            ],
            "core_depth_states": core_depth_states,
            "core_depth_last_logits": core_depth_last_logits,
            "core_depth_text_logits": core_depth_text_logits,
            "transition_state_logits": transition_state_info["logits"],
            "transition_state_features": transition_state_info["features"],
            "transition_state_text_logits": transition_state_text_logits,
            "transition_state_code_logits": transition_state_code_info["logits"],
            "transition_state_code_embeddings": transition_state_code_info["embeddings"],
            "transition_state_finality_logits": transition_state_finality_logits,
            "transition_state_joint_logits": transition_state_joint_logits,
            "transition_state_joint_answer_embeddings": (
                transition_state_joint_answer_embeddings
            ),
            "transition_state_final_answer_embedding": (
                transition_state_final_answer_embedding
            ),
            "transition_state_sequence_logits": transition_state_sequence_logits,
            "transition_value_state_logits": transition_value_state_logits,
            "factorized_value_state_logits": factorized_value_state_logits,
            "factorized_value_state_kind_logits": factorized_value_state_kind_logits,
            "role_value_state_logits": role_value_state_logits,
            "typed_algorithmic_kind_logits": typed_algorithmic_value_state_info[
                "kind_logits"
            ],
            "typed_algorithmic_raw_list_offset_logits": (
                typed_algorithmic_value_state_info["raw_list_offset_logits"]
            ),
            "typed_algorithmic_doubled_list_offset_logits": (
                typed_algorithmic_value_state_info["doubled_list_offset_logits"]
            ),
            "typed_algorithmic_scalar_coeff_logits": (
                typed_algorithmic_value_state_info["scalar_coeff_logits"]
            ),
            "typed_algorithmic_scalar_coeff_value": (
                typed_algorithmic_value_state_info["scalar_coeff_value"]
            ),
            "typed_algorithmic_scalar_offset_logits": (
                typed_algorithmic_value_state_info["scalar_offset_logits"]
            ),
            "typed_algorithmic_scalar_offset_value": (
                typed_algorithmic_value_state_info["scalar_offset_value"]
            ),
            "typed_algorithmic_scalar_residual_logits": (
                typed_algorithmic_value_state_info["scalar_residual_logits"]
            ),
            "typed_algorithmic_scalar_residual_value": (
                typed_algorithmic_value_state_info["scalar_residual_value"]
            ),
            "typed_algorithmic_scalar_residual_delta_logits": (
                typed_algorithmic_value_state_info["scalar_residual_delta_logits"]
            ),
            "typed_algorithmic_final_residual_logits": (
                typed_algorithmic_value_state_info["final_residual_logits"]
            ),
            "typed_algorithmic_final_residual_value": (
                typed_algorithmic_value_state_info["final_residual_value"]
            ),
            "typed_algorithmic_value_state_answer_bridge_tokens": (
                typed_algorithmic_value_state_answer_bridge_info["tokens"]
            ),
            "typed_algorithmic_value_state_answer_bridge_gate_mean": (
                typed_algorithmic_value_state_answer_bridge_info["gate_mean"]
            ),
            "core_role_value_state_logits": core_role_value_state_logits,
            "core_role_value_state_prompt_logits": core_role_value_state_prompt_logits,
            "core_source_position_prompt_logits": source_position_prompt_logits,
            "core_source_value_prompt_logits": core_source_value_prompt_logits,
            "core_role_value_state_prompt_parity_logits": (
                core_role_value_state_prompt_parity_logits
            ),
            "core_primitive_role_value_state_logits": (
                core_primitive_role_value_state_logits
            ),
            "core_primitive_role_value_update_gate": (
                core_primitive_role_value_update_gate
            ),
            "core_primitive_typed_selector_gate": selector_info["gate"],
            "core_primitive_typed_selector_gate_mean": selector_info["gate_mean"],
            "core_role_value_template_logits": core_role_value_template_logits,
            "core_role_value_state_answer_bridge_gate_mean": (
                core_role_value_state_answer_bridge_info["gate_mean"]
            ),
            "core_role_value_state_answer_final_embedding": (
                core_role_value_state_answer_final_embedding
            ),
            "core_role_value_delta_gate_mean": core_role_value_delta_gate_mean,
            "core_value_delta_code_logits": core_value_delta_code_logits,
            "core_value_delta_code_gate_mean": core_value_delta_code_gate_mean,
            "core_typed_register_operation_logits": core_typed_register_info[
                "operation_logits"
            ],
            "core_typed_register_value_logits": core_typed_register_info[
                "value_logits"
            ],
            "core_typed_register_transition_logits": core_typed_register_info[
                "transition_logits"
            ],
            "core_typed_register_gate_mean": core_typed_register_info["gate_mean"],
            "core_role_value_transition_logits": core_role_value_transition_logits,
            "primitive_transition_operation_logits": primitive_transition_info[
                "operation_logits"
            ],
            "transition_phase_logits": transition_phase_logits,
            "transition_source_router_logits": transition_source_router_logits,
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
        return compute_donor_qtrm_conflict_gate(
            qtrm_text_logits,
            donor_text_logits,
            enabled=bool(self.cfg.donor_qtrm_conflict_gate_enabled),
            mode=str(getattr(self.cfg, "donor_qtrm_conflict_gate_mode", "downscale")),
            conflict_scale=float(self.cfg.donor_qtrm_conflict_qtrm_scale),
            boost_scale=float(
                getattr(self.cfg, "donor_qtrm_conflict_qtrm_boost_scale", 1.0)
            ),
            margin_threshold=float(
                getattr(self.cfg, "donor_qtrm_conflict_margin_threshold", 0.0)
            ),
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

    def _empty_answer_state_loop_future_token_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
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
        query_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, _ = self._compute_answer_bottleneck_outputs(
            hidden,
            z_h=z_h,
            workspace_mask=workspace_mask,
            input_seq_len=input_seq_len,
            query_token_indices=query_token_indices,
        )
        return logits

    def _compute_answer_bottleneck_outputs(
        self,
        hidden: torch.Tensor,
        *,
        z_h: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
        query_token_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_seq_len = (
            int(query_token_indices.numel())
            if query_token_indices is not None
            else int(input_seq_len)
        )
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
                output_seq_len,
            )
            return empty_logits, hidden.new_empty((hidden.shape[0], 0, hidden.shape[-1]))
        text_offset = hidden.shape[1] - int(input_seq_len)
        text_hidden = hidden[:, text_offset:, :]
        if query_token_indices is not None:
            text_hidden = text_hidden.index_select(1, query_token_indices)
        answer_hidden = self.answer_bottleneck_cross(
            self.answer_bottleneck_query_norm(text_hidden),
            self.answer_bottleneck_workspace_norm(z_h),
            workspace_mask,
        )
        answer_hidden = self.answer_bottleneck_output_norm(answer_hidden)
        answer_hidden = self._apply_ri4_memory_residual(answer_hidden)
        logits = self.lm_head(answer_hidden) * float(self.cfg.qtrm_logits_scale)
        return logits, answer_hidden

    def _compute_core_loop_readout_outputs(
        self,
        text_context_seq: torch.Tensor,
        *,
        z_h: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
        query_token_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_seq_len = (
            int(query_token_indices.numel())
            if query_token_indices is not None
            else int(input_seq_len)
        )
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
                output_seq_len,
            )
            return empty_logits, self._empty_core_loop_readout_hidden(text_context_seq)
        text_hidden = text_context_seq[:, -int(input_seq_len) :, :]
        if query_token_indices is not None:
            text_hidden = text_hidden.index_select(1, query_token_indices)
        loop_hidden = self.core_loop_readout_cross(
            self.core_loop_readout_query_norm(text_hidden),
            self.core_loop_readout_state_norm(z_h),
            workspace_mask,
        )
        loop_hidden = self.core_loop_readout_output_norm(loop_hidden)
        logits = self.lm_head(loop_hidden) * float(self.cfg.qtrm_logits_scale)
        return logits, loop_hidden

    def _answer_state_loop_bridge_hidden(
        self,
        hidden: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or self.answer_state_loop_hidden_bridge_norm is None
            or self.answer_state_loop_hidden_bridge_down is None
            or self.answer_state_loop_hidden_bridge_up is None
        ):
            return hidden
        bridge = self.answer_state_loop_hidden_bridge_up(
            torch.nn.functional.gelu(
                self.answer_state_loop_hidden_bridge_down(
                    self.answer_state_loop_hidden_bridge_norm(hidden)
                )
            )
        )
        return hidden + bridge * float(self.cfg.answer_state_loop_hidden_bridge_scale)

    def _answer_state_loop_lm_logits(
        self,
        hidden: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        disable_hidden_bridge: bool = False,
        disable_next_token_decoder: bool = False,
        disable_free_transformer_latent: bool = False,
        free_transformer_posterior_context: Optional[torch.Tensor] = None,
        return_free_transformer_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, free_transformer_kl, free_transformer_gate_mean = (
            self._answer_state_loop_free_transformer_latent_hidden(
                hidden,
                posterior_context=free_transformer_posterior_context,
                disabled=bool(disable_free_transformer_latent),
            )
        )
        hidden = self._answer_state_loop_bridge_hidden(
            hidden,
            disabled=bool(disable_hidden_bridge),
        )
        hidden = self._answer_state_loop_next_token_decoder_hidden(
            hidden,
            prev_token_ids=prev_token_ids,
            disabled=bool(disable_next_token_decoder),
        )
        hidden = self._apply_ri4_memory_residual(hidden)
        logits = self.lm_head(hidden) * float(self.cfg.qtrm_logits_scale)
        if (
            self.answer_state_loop_lm_adapter_down is not None
            and self.answer_state_loop_lm_adapter_up is not None
        ):
            adapter_logits = self.answer_state_loop_lm_adapter_up(
                self.answer_state_loop_lm_adapter_down(hidden)
            )
            logits = logits + adapter_logits * float(
                self.cfg.answer_state_loop_lm_adapter_scale
            )
        if return_free_transformer_info:
            return logits, free_transformer_kl, free_transformer_gate_mean
        return logits

    def _answer_state_loop_free_transformer_latent_hidden(
        self,
        hidden: torch.Tensor,
        *,
        posterior_context: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty_scalar = hidden.new_zeros(())
        if (
            disabled
            or self.answer_state_loop_free_transformer_prior_norm is None
            or self.answer_state_loop_free_transformer_posterior_norm is None
            or self.answer_state_loop_free_transformer_prior_mu is None
            or self.answer_state_loop_free_transformer_prior_logvar is None
            or self.answer_state_loop_free_transformer_posterior_mu is None
            or self.answer_state_loop_free_transformer_posterior_logvar is None
            or self.answer_state_loop_free_transformer_latent_up is None
            or self.answer_state_loop_free_transformer_gate is None
            or hidden.numel() == 0
        ):
            return hidden, empty_scalar, empty_scalar
        prior_source = hidden.mean(dim=1)
        prior_source = self.answer_state_loop_free_transformer_prior_norm(
            prior_source
        )
        prior_mu = self.answer_state_loop_free_transformer_prior_mu(prior_source)
        prior_logvar = self.answer_state_loop_free_transformer_prior_logvar(
            prior_source
        ).clamp(-12.0, 8.0)
        posterior_active = (
            self.training
            and bool(self.cfg.answer_state_loop_free_transformer_posterior_train_enabled)
            and posterior_context is not None
            and posterior_context.numel() != 0
        )
        if posterior_active:
            posterior_source = posterior_context.mean(dim=1)
            posterior_source = self.answer_state_loop_free_transformer_posterior_norm(
                posterior_source
            )
            posterior_mu = self.answer_state_loop_free_transformer_posterior_mu(
                posterior_source
            )
            posterior_logvar = self.answer_state_loop_free_transformer_posterior_logvar(
                posterior_source
            ).clamp(-12.0, 8.0)
            std = torch.exp(0.5 * posterior_logvar)
            z = posterior_mu + torch.randn_like(std) * std
            prior_var = torch.exp(prior_logvar)
            posterior_var = torch.exp(posterior_logvar)
            kl = 0.5 * (
                prior_logvar
                - posterior_logvar
                + (posterior_var + (posterior_mu - prior_mu).pow(2))
                / prior_var.clamp_min(1e-6)
                - 1.0
            )
            kl = kl.mean()
        else:
            z = prior_mu
            kl = empty_scalar
        latent_delta = self.answer_state_loop_free_transformer_latent_up(z)
        latent_delta = latent_delta.unsqueeze(1).expand_as(hidden)
        gate = torch.sigmoid(self.answer_state_loop_free_transformer_gate(hidden))
        gate_min = min(
            max(float(self.cfg.answer_state_loop_free_transformer_gate_min), 0.0),
            1.0,
        )
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        if self.answer_state_loop_output_norm is not None:
            hidden = self.answer_state_loop_output_norm(hidden + gate * latent_delta)
        else:
            hidden = hidden + gate * latent_delta
        return hidden, kl, gate.mean()

    def _answer_state_loop_next_token_decoder_hidden(
        self,
        hidden: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or self.answer_state_loop_next_token_decoder_norm is None
            or self.answer_state_loop_next_token_decoder_stack is None
            or self.answer_state_loop_next_token_decoder_gate is None
            or hidden.numel() == 0
        ):
            return hidden
        original_shape = hidden.shape
        if hidden.ndim < 3:
            return hidden
        if hidden.ndim > 3:
            hidden = hidden.reshape(-1, original_shape[-2], original_shape[-1])
            if prev_token_ids is not None:
                prev_token_ids = prev_token_ids.reshape(-1, original_shape[-2])
        hidden = self._answer_state_loop_next_token_decoder_prev_token_hidden(
            hidden,
            prev_token_ids=prev_token_ids,
            disabled=disabled,
        )
        proposal = self.answer_state_loop_next_token_decoder_stack(
            self.answer_state_loop_next_token_decoder_norm(hidden)
        )
        gate = torch.sigmoid(self.answer_state_loop_next_token_decoder_gate(hidden))
        gate_min = min(
            max(float(self.cfg.answer_state_loop_next_token_decoder_gate_min), 0.0),
            1.0,
        )
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        if self.answer_state_loop_output_norm is not None:
            decoded = self.answer_state_loop_output_norm(
                hidden + gate * (proposal - hidden)
            )
        else:
            decoded = hidden + gate * (proposal - hidden)
        if len(original_shape) > 3:
            decoded = decoded.reshape(original_shape)
        return decoded

    def _answer_state_loop_next_token_decoder_prev_token_hidden(
        self,
        hidden: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or prev_token_ids is None
            or self.answer_state_loop_next_token_decoder_prev_token_norm is None
            or self.answer_state_loop_next_token_decoder_prev_token_fuse is None
            or self.answer_state_loop_next_token_decoder_prev_token_gate is None
            or hidden.numel() == 0
        ):
            return hidden
        if hidden.ndim != 3 or prev_token_ids.ndim != 2:
            return hidden
        if tuple(prev_token_ids.shape) != tuple(hidden.shape[:2]):
            raise ValueError(
                "prev_token_ids must have shape [batch, seq] matching hidden"
            )
        prev_token_ids = prev_token_ids.to(
            device=hidden.device,
            dtype=torch.long,
        ).clamp(min=0, max=int(self.text_embed.num_embeddings) - 1)
        prev_hidden = self.text_embed(prev_token_ids).to(dtype=hidden.dtype)
        prev_hidden = self.answer_state_loop_next_token_decoder_prev_token_norm(
            prev_hidden
        )
        proposal = self.answer_state_loop_next_token_decoder_prev_token_fuse(
            torch.cat([hidden, prev_hidden], dim=-1)
        )
        gate = torch.sigmoid(
            self.answer_state_loop_next_token_decoder_prev_token_gate(hidden)
        )
        gate_min = min(
            max(
                float(
                    self.cfg.answer_state_loop_next_token_decoder_prev_token_gate_min
                ),
                0.0,
            ),
            1.0,
        )
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        return hidden + gate * (proposal - hidden)

    def _answer_state_loop_talker_hidden(
        self,
        hidden: torch.Tensor,
        *,
        depth_hidden: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or
            self.answer_state_loop_talker_norm is None
            or self.answer_state_loop_talker_stack is None
            or self.answer_state_loop_talker_gate is None
            or hidden.numel() == 0
        ):
            return hidden
        talker_input = hidden
        if depth_hidden is not None and depth_hidden.numel() != 0:
            trajectory_context = depth_hidden.mean(dim=1)
            if trajectory_context.shape == hidden.shape:
                talker_input = talker_input + trajectory_context
        proposal = self.answer_state_loop_talker_stack(
            self.answer_state_loop_talker_norm(talker_input)
        )
        gate = torch.sigmoid(self.answer_state_loop_talker_gate(hidden))
        gate_min = min(max(float(self.cfg.answer_state_loop_talker_gate_min), 0.0), 1.0)
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        if self.answer_state_loop_output_norm is not None:
            return self.answer_state_loop_output_norm(
                hidden + gate * (proposal - hidden)
            )
        return hidden + gate * (proposal - hidden)

    def _answer_state_loop_future_token_logits(
        self,
        hidden: torch.Tensor,
        *,
        disable_hidden_bridge: bool = False,
        disable_next_token_decoder: bool = False,
    ) -> torch.Tensor:
        if (
            self.answer_state_loop_future_token_positions is None
            or hidden.numel() == 0
        ):
            return self._empty_answer_state_loop_future_token_logits(
                hidden.new_empty((hidden.shape[0], 0, self.cfg.vocab_size))
            )
        position_delta = self.answer_state_loop_future_token_positions.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        position_delta = position_delta * float(
            self.cfg.answer_state_loop_future_token_position_scale
        )
        future_hidden = hidden[:, -1:, :] + position_delta.unsqueeze(0)
        if self.answer_state_loop_output_norm is not None:
            future_hidden = self.answer_state_loop_output_norm(future_hidden)
        return self._answer_state_loop_lm_logits(
            future_hidden,
            disable_hidden_bridge=bool(disable_hidden_bridge),
            disable_next_token_decoder=bool(disable_next_token_decoder),
        )

    def _compute_answer_state_loop_outputs(
        self,
        text_context_seq: torch.Tensor,
        *,
        trajectory: list[torch.Tensor],
        text_context_mask: torch.Tensor,
        workspace_mask: torch.Tensor,
        input_seq_len: int,
        transition_state_features: Optional[torch.Tensor] = None,
        transition_state_code_embeddings: Optional[torch.Tensor] = None,
        transition_state_joint_answer_embeddings: Optional[torch.Tensor] = None,
        transition_state_final_answer_embedding: Optional[torch.Tensor] = None,
        transition_state_joint_logits: Optional[torch.Tensor] = None,
        typed_algorithmic_answer_tokens: Optional[torch.Tensor] = None,
        core_role_value_answer_tokens: Optional[torch.Tensor] = None,
        core_role_value_final_answer_embedding: Optional[torch.Tensor] = None,
        prev_token_ids: Optional[torch.Tensor] = None,
        query_token_indices: Optional[torch.Tensor] = None,
        disable_recurrent_block: bool = False,
        disable_selective_context: bool = False,
        force_dense_context: bool = False,
        disable_finality_gate: bool = False,
        disable_halt_gate: bool = False,
        disable_hidden_bridge: bool = False,
        disable_next_token_decoder: bool = False,
        disable_free_transformer_latent: bool = False,
        disable_talker: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        output_seq_len = (
            int(query_token_indices.numel())
            if query_token_indices is not None
            else int(input_seq_len)
        )
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
                self._empty_answer_state_loop_depth_hidden(
                    text_context_seq,
                    output_seq_len,
                ),
                text_context_seq.new_empty((text_context_seq.shape[0], 0)),
                text_context_seq.new_empty((text_context_seq.shape[0], 0)),
                text_context_seq.new_zeros(()),
                text_context_seq.new_zeros(()),
            )
        y = text_context_seq[:, -int(input_seq_len) :, :]
        if query_token_indices is not None:
            y = y.index_select(1, query_token_indices)
        text_query = y
        core_state_only = bool(
            getattr(self.cfg, "answer_state_loop_core_state_only_enabled", False)
        )
        if core_state_only:
            y = torch.zeros_like(text_query)
        answer_input = y
        states = []
        recurrent_gate_means = []
        halt_logits_per_step = []
        gate_min = min(max(float(self.cfg.answer_state_loop_gate_min), 0.0), 1.0)
        recurrent_active = (
            not disable_recurrent_block
            and (
                (
                    self.answer_state_loop_recurrent_norm is not None
                    and self.answer_state_loop_recurrent_stack is not None
                    and self.answer_state_loop_recurrent_gate is not None
                )
                or (self.answer_state_loop_hybrid_recurrent_block is not None)
            )
        )
        recurrent_gate_min = min(
            max(float(self.cfg.answer_state_loop_recurrent_gate_min), 0.0),
            1.0,
        )
        mythos_update_active = (
            recurrent_active
            and self.answer_state_loop_mythos_log_A is not None
            and self.answer_state_loop_mythos_log_dt is not None
            and self.answer_state_loop_mythos_input_B is not None
        )
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
        transition_joint_answer_active = (
            transition_state_joint_answer_embeddings is not None
            and transition_state_joint_answer_embeddings.numel() != 0
            and self.transition_state_joint_answer_gate is not None
        )
        transition_final_answer_active = (
            transition_state_final_answer_embedding is not None
            and transition_state_final_answer_embedding.numel() != 0
            and self.transition_state_final_answer_gate is not None
        )
        role_value_final_answer_active = (
            core_role_value_final_answer_embedding is not None
            and core_role_value_final_answer_embedding.numel() != 0
            and self.core_role_value_state_answer_final_gate is not None
        )
        finality_gate_scores = self._answer_state_loop_finality_scores(
            transition_state_joint_logits
        )
        finality_gate_active = (
            not disable_finality_gate
            and bool(self.cfg.answer_state_loop_finality_gate_enabled)
            and finality_gate_scores is not None
            and finality_gate_scores.numel() != 0
        )
        halt_gate_active = (
            not disable_halt_gate
            and bool(self.cfg.answer_state_loop_halt_gate_enabled)
            and self.answer_state_loop_halt_head is not None
        )
        mythos_act_active = (
            mythos_update_active
            and not disable_halt_gate
            and bool(self.cfg.answer_state_loop_mythos_act_enabled)
            and self.answer_state_loop_halt_head is not None
        )
        terminal_seen = y.new_zeros((y.shape[0], 1, 1))
        terminal_state = y.new_zeros(y.shape)
        act_cumulative = y.new_zeros((y.shape[0], 1, 1))
        act_output = y.new_zeros(y.shape)
        role_value_answer_active = (
            core_role_value_answer_tokens is not None
            and core_role_value_answer_tokens.numel() != 0
        )
        typed_algorithmic_answer_active = (
            typed_algorithmic_answer_tokens is not None
            and typed_algorithmic_answer_tokens.numel() != 0
        )
        transition_gate_min = min(
            max(float(self.cfg.transition_state_answer_gate_min), 0.0),
            1.0,
        )
        transition_joint_answer_gate_min = min(
            max(float(self.cfg.transition_state_joint_answer_gate_min), 0.0),
            1.0,
        )
        transition_final_answer_gate_min = min(
            max(float(self.cfg.transition_state_final_answer_gate_min), 0.0),
            1.0,
        )
        role_value_final_answer_gate_min = min(
            max(float(self.cfg.core_role_value_state_answer_final_gate_min), 0.0),
            1.0,
        )
        for step_index, state in enumerate(trajectory):
            query_y = text_query if core_state_only else y
            state_for_cross = state
            state_mask = workspace_mask

            # RI-4 A-Mode integration safety (Most-Deficient closure):
            # The answer_state_loop state preparation was written assuming richer
            # workspace/trajectory shapes. In tiny diagnostics and early hybrid
            # recurrent engine integration, state_for_cross can end up with seq dim=1
            # (or very small), causing "index out of bounds for dimension 1 with size 1"
            # in later cats, _select, gather, or cross-attn.
            # This guard ensures we always have enough seq dimension to reach the
            # recurrent proposal (hybrid block) without changing behavior on normal
            # rich-trajectory paths.
            min_state_seq = max(2, int(getattr(self.cfg, 'workspace_tokens', 2) or 2))
            cur_seq = int(state_for_cross.shape[1]) if state_for_cross.dim() == 3 else 0
            if cur_seq > 0 and cur_seq < min_state_seq:
                pad_len = min_state_seq - cur_seq
                pad = state_for_cross.new_zeros(
                    (state_for_cross.shape[0], pad_len, state_for_cross.shape[2])
                )
                state_for_cross = torch.cat([state_for_cross, pad], dim=1)
                pad_mask = state_mask.new_ones((state_mask.shape[0], pad_len))
                state_mask = torch.cat([state_mask, pad_mask], dim=1)

            # When the hybrid recurrent engine is attached (the RI-4 goal), or when
            # force_dense is requested, we prefer to bypass complex selection that
            # can still have edge-case fragility in the current integration state.
            effective_force_dense = bool(force_dense_context) or bool(self.answer_state_loop_hybrid_recurrent_block is not None)
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
                transition_gate = torch.sigmoid(
                    self.transition_state_answer_gate(query_y)
                )
                if transition_gate_min != 0.0:
                    transition_gate = (
                        transition_gate_min
                        + (1.0 - transition_gate_min) * transition_gate
                    )
                y = self.answer_state_loop_output_norm(
                    y + transition_gate * transition_delta
                )
            if (
                transition_joint_answer_active
                and step_index < int(transition_state_joint_answer_embeddings.shape[1])
            ):
                joint_delta = transition_state_joint_answer_embeddings[
                    :, step_index, :
                ].unsqueeze(1)
                joint_gate = torch.sigmoid(
                    self.transition_state_joint_answer_gate(query_y)
                )
                if transition_joint_answer_gate_min != 0.0:
                    joint_gate = (
                        transition_joint_answer_gate_min
                        + (1.0 - transition_joint_answer_gate_min) * joint_gate
                    )
                y = self.answer_state_loop_output_norm(y + joint_gate * joint_delta)
            if (
                role_value_answer_active
                and step_index < int(core_role_value_answer_tokens.shape[1])
            ):
                value_tokens = core_role_value_answer_tokens[:, step_index, :, :]
                value_mask = workspace_mask.new_ones(
                    (workspace_mask.shape[0], int(value_tokens.shape[1]))
                )
                state_for_cross = torch.cat([value_tokens, state_for_cross], dim=1)
                state_mask = torch.cat([value_mask, state_mask], dim=1)
            if (
                typed_algorithmic_answer_active
                and step_index < int(typed_algorithmic_answer_tokens.shape[1])
            ):
                value_tokens = typed_algorithmic_answer_tokens[:, step_index, :, :]
                value_mask = workspace_mask.new_ones(
                    (workspace_mask.shape[0], int(value_tokens.shape[1]))
                )
                state_for_cross = torch.cat([value_tokens, state_for_cross], dim=1)
                state_mask = torch.cat([value_mask, state_mask], dim=1)
            state_for_cross, state_mask = self._select_answer_state_loop_context(
                query_y,
                state_for_cross,
                state_mask,
                text_context_seq,
                text_context_mask,
                disabled=bool(disable_selective_context or core_state_only),
                force_dense=effective_force_dense,
            )
            delta = self.answer_state_loop_cross(
                self.answer_state_loop_query_norm(query_y),
                self.answer_state_loop_state_norm(state_for_cross),
                state_mask,
            )
            gate = torch.sigmoid(self.answer_state_loop_gate(query_y))
            if gate_min != 0.0:
                gate = gate_min + (1.0 - gate_min) * gate
            y = self.answer_state_loop_output_norm(y + gate * delta)
            y = self._apply_ri4_memory_residual(y)
            if recurrent_active:
                recurrent_input = y
                if (
                    mythos_update_active
                    and self.answer_state_loop_mythos_loop_index is not None
                ):
                    loop_idx = min(
                        int(step_index),
                        self.answer_state_loop_mythos_loop_index.num_embeddings - 1,
                    )
                    loop_delta = self.answer_state_loop_mythos_loop_index(
                        torch.tensor(loop_idx, device=y.device, dtype=torch.long)
                    ).view(1, 1, -1)
                    loop_dim = int(
                        self.cfg.answer_state_loop_mythos_loop_dim
                        or max(2, self.cfg.d_model // 8)
                    )
                    loop_dim = max(0, min(loop_dim, int(y.shape[-1])))
                    if loop_dim > 0:
                        recurrent_input = recurrent_input.clone()
                        recurrent_input[..., :loop_dim] = (
                            recurrent_input[..., :loop_dim]
                            + loop_delta[..., :loop_dim].to(dtype=y.dtype)
                        )
                if self.answer_state_loop_hybrid_recurrent_block is not None:
                    # RI-4 (A-mode): hybrid block IS the answer_state_loop recurrent engine.
                    # One-Body native: SparseSlotRouter + persistent slots participate directly
                    # in every recurrent proposal step inside the trajectory loop.
                    # Slot state is carried on the model instance across steps for this case.
                    # Falls back to classic recurrent_stack when the attr is None (zero behavior change).
                    hybrid_in = self.answer_state_loop_recurrent_norm(recurrent_input).unsqueeze(1)
                    hybrid_out, new_slot = self.answer_state_loop_hybrid_recurrent_block(
                        hybrid_in,
                        slot_state=getattr(self, '_ri4_hybrid_recurrent_slot_state', None),
                    )
                    recurrent_proposal = hybrid_out.squeeze(1)
                    self._ri4_hybrid_recurrent_slot_state = new_slot
                else:
                    recurrent_proposal = self.answer_state_loop_recurrent_stack(
                        self.answer_state_loop_recurrent_norm(recurrent_input)
                    )
                if (
                    mythos_update_active
                    and self.answer_state_loop_mythos_lora_down is not None
                    and self.answer_state_loop_mythos_lora_up is not None
                    and self.answer_state_loop_mythos_lora_scale is not None
                ):
                    loop_idx = min(
                        int(step_index),
                        self.answer_state_loop_mythos_lora_scale.num_embeddings - 1,
                    )
                    scale = self.answer_state_loop_mythos_lora_scale(
                        torch.tensor(loop_idx, device=y.device, dtype=torch.long)
                    ).to(dtype=y.dtype)
                    lora_hidden = (
                        self.answer_state_loop_mythos_lora_down(recurrent_proposal)
                        * scale.view(1, 1, -1)
                    )
                    recurrent_proposal = (
                        recurrent_proposal
                        + self.answer_state_loop_mythos_lora_up(lora_hidden)
                    )
                recurrent_gate = torch.sigmoid(
                    self.answer_state_loop_recurrent_gate(y)
                )
                if recurrent_gate_min != 0.0:
                    recurrent_gate = (
                        recurrent_gate_min
                        + (1.0 - recurrent_gate_min) * recurrent_gate
                    )
                recurrent_delta = recurrent_gate * (recurrent_proposal - y)
                if mythos_update_active:
                    A = torch.exp(
                        -torch.exp(
                            (
                                self.answer_state_loop_mythos_log_dt
                                + self.answer_state_loop_mythos_log_A
                            ).clamp(-20, 20)
                        )
                    ).to(dtype=y.dtype)
                    B = self.answer_state_loop_mythos_input_B.to(dtype=y.dtype)
                    y = self.answer_state_loop_output_norm(
                        A.view(1, 1, -1) * y
                        + B.view(1, 1, -1) * answer_input
                        + recurrent_delta
                    )
                else:
                    y = self.answer_state_loop_output_norm(y + recurrent_delta)
                y = self._apply_ri4_memory_residual(y)
                recurrent_gate_means.append(recurrent_gate.squeeze(-1).mean(dim=1))
            halt_logit = None
            if self.answer_state_loop_halt_head is not None:
                halt_logit = self.answer_state_loop_halt_head(y[:, -1, :]).squeeze(-1)
                halt_logits_per_step.append(halt_logit)
            if mythos_act_active and halt_logit is not None:
                temperature = max(
                    float(self.cfg.answer_state_loop_halt_gate_temperature),
                    1e-6,
                )
                halt_prob = torch.sigmoid(halt_logit.to(dtype=y.dtype) / temperature)
                halt_prob = halt_prob[:, None, None]
                threshold = min(
                    max(float(self.cfg.answer_state_loop_mythos_act_threshold), 1e-6),
                    1.0,
                )
                still_running = (act_cumulative < threshold).to(dtype=y.dtype)
                remainder = (1.0 - act_cumulative).clamp(min=0.0, max=1.0)
                act_weight = torch.where(
                    act_cumulative + halt_prob >= threshold,
                    remainder,
                    halt_prob,
                )
                act_weight = act_weight * still_running
                act_output = act_output + act_weight * y
                act_cumulative = (act_cumulative + act_weight).clamp(0.0, 1.0)
            elif halt_gate_active and halt_logit is not None:
                mode = str(self.cfg.answer_state_loop_halt_gate_mode or "soft").lower()
                if mode == "hard_first":
                    halt_prob = (halt_logit > 0.0).to(dtype=y.dtype)
                else:
                    temperature = max(
                        float(self.cfg.answer_state_loop_halt_gate_temperature),
                        1e-6,
                    )
                    halt_prob = torch.sigmoid(halt_logit.to(dtype=y.dtype) / temperature)
                halt_prob = halt_prob[:, None, None]
                newly_terminal = (1.0 - terminal_seen) * halt_prob
                terminal_state = newly_terminal * y + (1.0 - newly_terminal) * terminal_state
                terminal_seen = (terminal_seen + newly_terminal).clamp(0.0, 1.0)
                y = terminal_seen * terminal_state + (1.0 - terminal_seen) * y
            if finality_gate_active and step_index < int(finality_gate_scores.shape[1]):
                finality_score = finality_gate_scores[:, step_index].to(dtype=y.dtype)
                mode = str(self.cfg.answer_state_loop_finality_gate_mode or "soft").lower()
                if mode == "hard_first":
                    final_prob = (finality_score > 0.0).to(dtype=y.dtype)
                else:
                    temperature = max(
                        float(self.cfg.answer_state_loop_finality_gate_temperature),
                        1e-6,
                    )
                    final_prob = torch.sigmoid(finality_score / temperature)
                final_prob = final_prob[:, None, None]
                newly_terminal = (1.0 - terminal_seen) * final_prob
                terminal_state = newly_terminal * y + (1.0 - newly_terminal) * terminal_state
                terminal_seen = (terminal_seen + newly_terminal).clamp(0.0, 1.0)
                y = terminal_seen * terminal_state + (1.0 - terminal_seen) * y
            states.append(y)
        if mythos_act_active:
            y = act_output + (1.0 - act_cumulative).clamp(0.0, 1.0) * y
            if states:
                states[-1] = y
        if transition_final_answer_active:
            final_delta = transition_state_final_answer_embedding[:, :1, :]
            final_gate = torch.sigmoid(self.transition_state_final_answer_gate(y))
            if transition_final_answer_gate_min != 0.0:
                final_gate = (
                    transition_final_answer_gate_min
                    + (1.0 - transition_final_answer_gate_min) * final_gate
                )
            y = self.answer_state_loop_output_norm(y + final_gate * final_delta)
            if states:
                states[-1] = y
        if role_value_final_answer_active:
            final_delta = core_role_value_final_answer_embedding[:, :1, :]
            final_gate = torch.sigmoid(
                self.core_role_value_state_answer_final_gate(y)
            )
            if role_value_final_answer_gate_min != 0.0:
                final_gate = (
                    role_value_final_answer_gate_min
                    + (1.0 - role_value_final_answer_gate_min) * final_gate
                )
            y = self.answer_state_loop_output_norm(y + final_gate * final_delta)
            if states:
                states[-1] = y
        depth_hidden = torch.stack(states, dim=1)
        y = self._answer_state_loop_talker_hidden(
            y,
            depth_hidden=depth_hidden,
            disabled=bool(disable_talker),
        )
        if states:
            depth_hidden = depth_hidden.clone()
            depth_hidden[:, -1, :, :] = y
        logits, free_transformer_latent_kl, free_transformer_gate_mean = (
            self._answer_state_loop_lm_logits(
                y,
                prev_token_ids=prev_token_ids,
                disable_hidden_bridge=bool(disable_hidden_bridge),
                disable_next_token_decoder=bool(disable_next_token_decoder),
                disable_free_transformer_latent=bool(disable_free_transformer_latent),
                free_transformer_posterior_context=(
                    text_context_seq[:, -int(input_seq_len) :, :]
                ),
                return_free_transformer_info=True,
            )
        )
        recurrent_gate_mean = (
            torch.stack(recurrent_gate_means, dim=1)
            if recurrent_gate_means
            else text_context_seq.new_empty((text_context_seq.shape[0], 0))
        )
        halt_logits = (
            torch.stack(halt_logits_per_step, dim=1)
            if halt_logits_per_step
            else text_context_seq.new_empty((text_context_seq.shape[0], 0))
        )
        return (
            logits,
            y,
            depth_hidden,
            recurrent_gate_mean,
            halt_logits,
            free_transformer_latent_kl,
            free_transformer_gate_mean,
        )

    def _answer_state_loop_finality_scores(
        self,
        transition_state_joint_logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if transition_state_joint_logits is None or transition_state_joint_logits.numel() == 0:
            return None
        joint_size = int(transition_state_joint_logits.shape[-1])
        if joint_size < 2 or joint_size % 2 != 0:
            return None
        code_count = joint_size // 2
        pair_logits = transition_state_joint_logits.reshape(
            transition_state_joint_logits.shape[0],
            transition_state_joint_logits.shape[1],
            code_count,
            2,
        )
        nonfinal_score = torch.logsumexp(pair_logits[..., 0], dim=-1)
        final_score = torch.logsumexp(pair_logits[..., 1], dim=-1)
        return final_score - nonfinal_score

    def _select_answer_state_loop_by_finality(
        self,
        answer_logits: torch.Tensor,
        answer_hidden: torch.Tensor,
        depth_hidden: torch.Tensor,
        transition_state_joint_logits: torch.Tensor,
        *,
        disabled: bool = False,
        disable_hidden_bridge: bool = False,
        disable_next_token_decoder: bool = False,
        disable_free_transformer_latent: bool = False,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            disabled
            or not bool(self.cfg.answer_state_loop_finality_selector_enabled)
            or depth_hidden.numel() == 0
            or transition_state_joint_logits.numel() == 0
        ):
            return answer_logits, answer_hidden
        finality_score = self._answer_state_loop_finality_scores(
            transition_state_joint_logits
        )
        if finality_score is None:
            return answer_logits, answer_hidden
        mode = str(self.cfg.answer_state_loop_finality_selector_mode or "soft").lower()
        if mode == "hard_max":
            indices = finality_score.float().argmax(dim=1)
            weights = torch.nn.functional.one_hot(
                indices,
                num_classes=int(finality_score.shape[1]),
            ).to(device=depth_hidden.device, dtype=depth_hidden.dtype)
        elif mode == "hard_first":
            final_mask = finality_score > 0.0
            fallback = finality_score.float().argmax(dim=1)
            positions = torch.arange(
                int(finality_score.shape[1]),
                device=finality_score.device,
            ).expand_as(finality_score)
            first_indices = positions.masked_fill(~final_mask, positions.shape[1]).min(dim=1).values
            indices = torch.where(
                first_indices < positions.shape[1],
                first_indices,
                fallback,
            )
            weights = torch.nn.functional.one_hot(
                indices,
                num_classes=int(finality_score.shape[1]),
            ).to(device=depth_hidden.device, dtype=depth_hidden.dtype)
        else:
            temperature = max(
                float(self.cfg.answer_state_loop_finality_selector_temperature),
                1e-6,
            )
            weights = torch.softmax(finality_score.float() / temperature, dim=1).to(
                dtype=depth_hidden.dtype
            )
        selected_hidden = torch.einsum("bs,bstd->btd", weights, depth_hidden)
        selected_logits = self._answer_state_loop_lm_logits(
            selected_hidden,
            prev_token_ids=prev_token_ids,
            disable_hidden_bridge=bool(disable_hidden_bridge),
            disable_next_token_decoder=bool(disable_next_token_decoder),
            disable_free_transformer_latent=bool(disable_free_transformer_latent),
        )
        return selected_logits, selected_hidden

    def _select_answer_state_loop_context(
        self,
        y: torch.Tensor,
        state_for_cross: torch.Tensor,
        state_mask: torch.Tensor,
        text_context_seq: torch.Tensor,
        text_context_mask: torch.Tensor,
        *,
        disabled: bool = False,
        force_dense: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            disabled
            or self.answer_state_loop_selective_query is None
            or self.answer_state_loop_selective_key is None
        ):
            return state_for_cross, state_mask
        top_k = int(self.cfg.answer_state_loop_selective_context_top_k)
        if top_k <= 0:
            return state_for_cross, state_mask
        candidates = torch.cat([state_for_cross, text_context_seq], dim=1)
        candidate_mask = torch.cat([state_mask, text_context_mask], dim=1)
        if bool(force_dense):
            return candidates, candidate_mask
        if candidates.shape[1] <= 1 or top_k >= int(candidates.shape[1]):
            return candidates, candidate_mask

        query = self.answer_state_loop_selective_query(y.mean(dim=1))
        keys = self.answer_state_loop_selective_key(candidates)
        scores = torch.einsum("bd,bnd->bn", query, keys) / (query.shape[-1] ** 0.5)
        valid = candidate_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        top_indices = torch.topk(scores.float(), k=top_k, dim=1).indices
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
        selected = torch.gather(candidates, dim=1, index=gather_index)
        selected_mask = torch.gather(candidate_mask, dim=1, index=top_indices)
        return selected, selected_mask

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

    def _empty_core_transition_order_bottleneck_info(
        self,
        reference: torch.Tensor,
        *,
        steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        b = reference.shape[0]
        classes = max(
            1,
            int(self.cfg.core_transition_order_bottleneck_num_classes),
        )
        return {
            "token": reference.new_empty((b, 0, self.cfg.d_model)),
            "logits": reference.new_empty((b, int(steps), classes), dtype=torch.float32),
            "gate_mean": reference.new_empty((b, 0)),
        }

    def _compute_core_transition_order_bottleneck(
        self,
        prompt_context_seq: Optional[torch.Tensor],
        prompt_context_mask: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        if (
            self.core_transition_order_bottleneck_query is None
            or self.core_transition_order_bottleneck_query_norm is None
            or self.core_transition_order_bottleneck_context_norm is None
            or self.core_transition_order_bottleneck_cross is None
            or self.core_transition_order_bottleneck_head is None
            or self.core_transition_order_bottleneck_embed is None
            or self.core_transition_order_bottleneck_gate is None
            or self.core_transition_order_bottleneck_output_norm is None
        ):
            return self._empty_core_transition_order_bottleneck_info(reference)

        b = reference.shape[0]
        classes = int(self.core_transition_order_bottleneck_embed.num_embeddings)
        if disabled or prompt_context_seq is None:
            return {
                "token": reference.new_empty((b, 0, self.cfg.d_model)),
                "logits": reference.new_zeros((b, 1, classes), dtype=torch.float32),
                "gate_mean": reference.new_empty((b, 0)),
            }

        query = self.core_transition_order_bottleneck_query.to(
            device=reference.device,
            dtype=reference.dtype,
        ).expand(b, -1, -1)
        context = self.core_transition_order_bottleneck_cross(
            self.core_transition_order_bottleneck_query_norm(query),
            self.core_transition_order_bottleneck_context_norm(prompt_context_seq),
            prompt_context_mask,
        )
        logits = self.core_transition_order_bottleneck_head(context).float()
        probs = torch.softmax(logits.to(dtype=reference.dtype), dim=-1)
        order_embed = torch.matmul(
            probs,
            self.core_transition_order_bottleneck_embed.weight.to(
                device=reference.device,
                dtype=reference.dtype,
            ),
        )
        gate = torch.sigmoid(self.core_transition_order_bottleneck_gate(context))
        token = self.core_transition_order_bottleneck_output_norm(
            context + gate * order_embed
        )
        return {
            "token": token,
            "logits": logits,
            "gate_mean": gate.squeeze(-1).to(dtype=reference.dtype),
        }

    def _empty_core_halt_info(self, workspace: torch.Tensor) -> dict[str, torch.Tensor]:
        b = workspace.shape[0]
        return {
            "q_halt_logits": workspace.new_empty((b, 0), dtype=torch.float32),
            "q_continue_logits": workspace.new_empty((b, 0), dtype=torch.float32),
            "halted": torch.zeros(b, device=workspace.device, dtype=torch.bool),
            "steps": torch.zeros(b, device=workspace.device, dtype=torch.long),
            "context_gate_mean": workspace.new_empty((b, 0)),
            "state_carry_gate_mean": workspace.new_empty((b, 0)),
            "transition_feedback_operation_logits": workspace.new_empty(
                (b, 0, max(1, int(self.cfg.core_transition_feedback_num_operations))),
                dtype=torch.float32,
            ),
            "transition_feedback_finality_logits": workspace.new_empty(
                (b, 0),
                dtype=torch.float32,
            ),
            "transition_feedback_gate_mean": workspace.new_empty((b, 0)),
            "transition_order_conditioning_gate_mean": workspace.new_empty((b, 0)),
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

    def _compute_transition_state_joint_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.transition_state_joint_norm is None
            or self.transition_state_joint_head is None
        ):
            return core_depth_states.new_empty((b, 0, 0))
        joint_size = int(self.transition_state_joint_head.out_features)
        if disabled or steps == 0:
            return core_depth_states.new_zeros((b, steps, joint_size))
        features = core_depth_states
        if (
            self.transition_state_joint_prompt_context_norm is not None
            and self.transition_state_joint_prompt_context_proj is not None
            and prompt_context_seq is not None
            and prompt_context_seq.numel() != 0
        ):
            if prompt_context_mask is None:
                prompt_context = prompt_context_seq.mean(dim=1)
            else:
                prompt_mask = prompt_context_mask.to(
                    device=prompt_context_seq.device,
                    dtype=prompt_context_seq.dtype,
                ).unsqueeze(-1)
                denom = prompt_mask.sum(dim=1).clamp_min(1.0)
                prompt_context = (prompt_context_seq * prompt_mask).sum(dim=1) / denom
            prompt_delta = self.transition_state_joint_prompt_context_proj(
                self.transition_state_joint_prompt_context_norm(prompt_context)
            ).to(dtype=core_depth_states.dtype)
            features = features + (
                float(self.cfg.transition_state_joint_prompt_context_scale)
                * prompt_delta.unsqueeze(1)
            )
            if (
                self.transition_state_joint_prompt_query_norm is not None
                and self.transition_state_joint_prompt_token_context_norm is not None
                and self.transition_state_joint_prompt_cross is not None
                and self.transition_state_joint_prompt_cross_norm is not None
                and self.transition_state_joint_prompt_cross_proj is not None
            ):
                prompt_token_context = self.transition_state_joint_prompt_cross(
                    self.transition_state_joint_prompt_query_norm(core_depth_states),
                    self.transition_state_joint_prompt_token_context_norm(
                        prompt_context_seq
                    ),
                    prompt_context_mask,
                )
                prompt_token_delta = self.transition_state_joint_prompt_cross_proj(
                    self.transition_state_joint_prompt_cross_norm(prompt_token_context)
                ).to(dtype=core_depth_states.dtype)
                features = features + (
                    float(self.cfg.transition_state_joint_prompt_context_scale)
                    * prompt_token_delta
                )
        return self.transition_state_joint_head(
            self.transition_state_joint_norm(features)
        )

    def _apply_transition_state_joint_operation_residual(
        self,
        transition_state_joint_logits: torch.Tensor,
        primitive_transition_info: dict[str, torch.Tensor],
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or self.transition_state_joint_operation_residual is None
            or transition_state_joint_logits.numel() == 0
        ):
            return transition_state_joint_logits
        operation_logits = primitive_transition_info.get("operation_logits")
        if (
            operation_logits is None
            or operation_logits.numel() == 0
            or operation_logits.shape[:2] != transition_state_joint_logits.shape[:2]
        ):
            return transition_state_joint_logits
        operation_probs = torch.softmax(operation_logits.float(), dim=-1).to(
            dtype=transition_state_joint_logits.dtype
        )
        residual = self.transition_state_joint_operation_residual(operation_probs)
        return transition_state_joint_logits + (
            float(self.cfg.transition_state_joint_operation_residual_scale) * residual
        )

    def _apply_transition_state_joint_code_residual(
        self,
        transition_state_joint_logits: torch.Tensor,
        transition_state_code_info: dict[str, torch.Tensor],
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or self.transition_state_joint_code_residual is None
            or transition_state_joint_logits.numel() == 0
        ):
            return transition_state_joint_logits
        code_logits = transition_state_code_info.get("logits")
        if (
            code_logits is None
            or code_logits.numel() == 0
            or code_logits.shape[:2] != transition_state_joint_logits.shape[:2]
        ):
            return transition_state_joint_logits
        code_probs = torch.softmax(code_logits.float(), dim=-1).to(
            dtype=transition_state_joint_logits.dtype
        )
        residual = self.transition_state_joint_code_residual(code_probs)
        return transition_state_joint_logits + (
            float(self.cfg.transition_state_joint_code_residual_scale) * residual
        )

    def _apply_transition_state_joint_phase_residual(
        self,
        transition_state_joint_logits: torch.Tensor,
        transition_phase_logits: torch.Tensor,
        core_depth_states: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or self.transition_state_joint_phase_residual is None
            or transition_state_joint_logits.numel() == 0
        ):
            return transition_state_joint_logits
        if (
            transition_phase_logits is None
            or transition_phase_logits.numel() == 0
            or transition_phase_logits.shape[:2] != transition_state_joint_logits.shape[:2]
            or core_depth_states.shape[:2] != transition_state_joint_logits.shape[:2]
        ):
            return transition_state_joint_logits
        phase_probs = torch.softmax(transition_phase_logits.float(), dim=-1).to(
            dtype=transition_state_joint_logits.dtype
        )
        if bool(self.cfg.transition_state_joint_phase_residual_centered):
            ref = int(self.cfg.transition_state_joint_phase_reference_class)
            if 0 <= ref < int(phase_probs.shape[-1]):
                reference = torch.zeros_like(phase_probs)
                reference[..., ref] = 1.0
                phase_probs = phase_probs - reference
        residual_input = torch.cat(
            [core_depth_states.to(dtype=transition_state_joint_logits.dtype), phase_probs],
            dim=-1,
        )
        residual = self.transition_state_joint_phase_residual(residual_input)
        if bool(self.cfg.transition_state_joint_phase_residual_gated_by_nonreference):
            ref = int(self.cfg.transition_state_joint_phase_reference_class)
            if 0 <= ref < int(phase_probs.shape[-1]):
                gate = 1.0 - phase_probs[..., ref : ref + 1]
                if bool(self.cfg.transition_state_joint_phase_residual_detach_gate):
                    gate = gate.detach()
                residual = residual * gate
        return transition_state_joint_logits + (
            float(self.cfg.transition_state_joint_phase_residual_scale) * residual
        )

    def _compute_transition_state_joint_answer_embeddings(
        self,
        transition_state_joint_logits: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = transition_state_joint_logits.shape[0]
        steps = transition_state_joint_logits.shape[1]
        if self.transition_state_joint_answer_proj is None:
            return transition_state_joint_logits.new_empty((b, 0, self.cfg.d_model))
        if disabled or steps == 0:
            return transition_state_joint_logits.new_zeros((b, steps, self.cfg.d_model))
        weights = torch.softmax(transition_state_joint_logits.float(), dim=-1).to(
            dtype=transition_state_joint_logits.dtype
        )
        return self.transition_state_joint_answer_proj(weights)

    def _compute_transition_state_final_answer_embedding(
        self,
        core_depth_states: torch.Tensor,
        transition_state_joint_logits: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if self.transition_state_final_answer_proj is None:
            return core_depth_states.new_empty((b, 0, self.cfg.d_model))
        if disabled or steps == 0:
            return core_depth_states.new_zeros((b, 1, self.cfg.d_model))
        if transition_state_joint_logits.shape[:2] != core_depth_states.shape[:2]:
            return core_depth_states.new_zeros((b, 1, self.cfg.d_model))
        joint_size = int(transition_state_joint_logits.shape[-1])
        if joint_size < 2 or joint_size % 2 != 0:
            return core_depth_states.new_zeros((b, 1, self.cfg.d_model))
        code_count = joint_size // 2
        pair_logits = transition_state_joint_logits.reshape(
            b,
            steps,
            code_count,
            2,
        )
        nonfinal_score = torch.logsumexp(pair_logits[..., 0], dim=-1)
        final_score = torch.logsumexp(pair_logits[..., 1], dim=-1)
        finality_score = final_score - nonfinal_score
        temperature = max(
            float(self.cfg.transition_state_final_answer_temperature),
            1e-6,
        )
        depth_weights = torch.softmax(finality_score.float() / temperature, dim=1).to(
            dtype=core_depth_states.dtype
        )
        selected_state = torch.einsum("bs,bsd->bd", depth_weights, core_depth_states)
        return self.transition_state_final_answer_proj(selected_state).unsqueeze(1)

    def _compute_transition_state_sequence_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.transition_state_sequence_norm is None
            or self.transition_state_sequence_pos_embed is None
            or self.transition_state_sequence_head is None
        ):
            return core_depth_states.new_empty((b, 0, 0, self.cfg.vocab_size))
        max_tokens = int(self.transition_state_sequence_pos_embed.num_embeddings)
        if disabled or steps == 0:
            return core_depth_states.new_zeros(
                (b, steps, max_tokens, self.cfg.vocab_size)
            )
        states = self.transition_state_sequence_norm(core_depth_states)
        pos_ids = torch.arange(max_tokens, device=core_depth_states.device)
        pos = self.transition_state_sequence_pos_embed(pos_ids).to(
            dtype=core_depth_states.dtype
        )
        features = states.unsqueeze(2) + pos.view(1, 1, max_tokens, -1)
        logits = self.transition_state_sequence_head(features)
        return logits * float(self.cfg.qtrm_logits_scale)

    def _compute_transition_value_state_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.transition_value_state_norm is None
            or self.transition_value_state_pos_embed is None
            or self.transition_value_state_head is None
        ):
            vocab = max(1, int(self.cfg.transition_value_state_vocab_size))
            return core_depth_states.new_empty((b, 0, 0, vocab))
        max_tokens = int(self.transition_value_state_pos_embed.num_embeddings)
        vocab = int(self.transition_value_state_head.out_features)
        if disabled or steps == 0:
            return core_depth_states.new_zeros((b, steps, max_tokens, vocab))
        states = self.transition_value_state_norm(core_depth_states)
        pos_ids = torch.arange(max_tokens, device=core_depth_states.device)
        pos = self.transition_value_state_pos_embed(pos_ids).to(
            dtype=core_depth_states.dtype
        )
        features = states.unsqueeze(2) + pos.view(1, 1, max_tokens, -1)
        return self.transition_value_state_head(features)

    def _compute_factorized_value_state_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        return self._compute_factorized_value_state_outputs(
            core_depth_states,
            prompt_context_seq=prompt_context_seq,
            prompt_context_mask=prompt_context_mask,
            disabled=disabled,
        )["slot_logits"]

    def _compute_factorized_value_state_outputs(
        self,
        core_depth_states: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        kind_size = max(0, int(self.cfg.factorized_value_state_kind_size))
        role_count = max(0, int(self.cfg.role_value_state_num_roles))
        if (
            self.factorized_value_state_init is None
            or self.factorized_value_state_step_embed is None
            or self.factorized_value_state_action_proj is None
            or self.factorized_value_state_update is None
            or self.factorized_value_state_output_norm is None
            or self.factorized_value_state_head is None
        ):
            vocab = max(1, int(self.cfg.factorized_value_state_vocab_size))
            return {
                "slot_logits": core_depth_states.new_empty((b, 0, 0, vocab)),
                "kind_logits": core_depth_states.new_empty((b, 0, kind_size)),
                "role_logits": core_depth_states.new_empty((b, 0, role_count, vocab)),
            }
        max_tokens = int(self.factorized_value_state_init.shape[1])
        vocab = int(self.factorized_value_state_head.out_features)
        role_vocab = (
            int(self.role_value_state_head.out_features)
            if self.role_value_state_head is not None
            else max(1, int(self.cfg.role_value_state_vocab_size or vocab))
        )
        if disabled or steps == 0:
            return {
                "slot_logits": core_depth_states.new_zeros(
                    (b, steps, max_tokens, vocab)
                ),
                "kind_logits": core_depth_states.new_zeros((b, steps, kind_size)),
                "role_logits": core_depth_states.new_zeros(
                    (b, steps, role_count, role_vocab)
                ),
            }
        slots = self.factorized_value_state_init.to(
            device=core_depth_states.device,
            dtype=core_depth_states.dtype,
        ).expand(b, -1, -1)
        prompt_active = (
            prompt_context_seq is not None
            and prompt_context_seq.numel() != 0
            and self.factorized_value_state_prompt_query_norm is not None
            and self.factorized_value_state_prompt_context_norm is not None
            and self.factorized_value_state_prompt_cross is not None
        )
        outputs = []
        kind_outputs = []
        role_outputs = []
        for index in range(steps):
            step_idx = min(index, self.factorized_value_state_step_embed.num_embeddings - 1)
            step_id = torch.tensor(step_idx, device=core_depth_states.device)
            step = self.factorized_value_state_step_embed(step_id).view(1, 1, -1)
            step = step.to(dtype=core_depth_states.dtype)
            action = self.factorized_value_state_action_proj(
                core_depth_states[:, index, :]
            ).unsqueeze(1)
            slots = slots + step + action
            if prompt_active:
                prompt = self.factorized_value_state_prompt_context_norm(
                    prompt_context_seq
                )
                prompt = prompt.to(dtype=slots.dtype)
                slots = slots + self.factorized_value_state_prompt_cross(
                    self.factorized_value_state_prompt_query_norm(slots),
                    prompt,
                    prompt_context_mask,
                )
            slots = slots + self.factorized_value_state_update(slots)
            slots = self.factorized_value_state_output_norm(slots)
            outputs.append(self.factorized_value_state_head(slots))
            if self.factorized_value_state_kind_head is not None:
                kind_outputs.append(self.factorized_value_state_kind_head(slots[:, 0, :]))
            if (
                self.role_value_state_role_embed is not None
                and self.role_value_state_query_norm is not None
                and self.role_value_state_slot_norm is not None
                and self.role_value_state_cross is not None
                and self.role_value_state_head is not None
            ):
                role_queries = self.role_value_state_role_embed.weight.to(
                    device=slots.device,
                    dtype=slots.dtype,
                ).unsqueeze(0).expand(b, -1, -1)
                role_features = role_queries + self.role_value_state_cross(
                    self.role_value_state_query_norm(role_queries),
                    self.role_value_state_slot_norm(slots),
                    None,
                )
                role_outputs.append(self.role_value_state_head(role_features))
        slot_logits = torch.stack(outputs, dim=1)
        if self.factorized_value_state_kind_head is None:
            kind_logits = core_depth_states.new_empty((b, steps, 0))
        else:
            kind_logits = torch.stack(kind_outputs, dim=1)
        if self.role_value_state_head is None:
            role_logits = core_depth_states.new_empty((b, steps, 0, role_vocab))
        else:
            role_logits = torch.stack(role_outputs, dim=1)
        return {
            "slot_logits": slot_logits,
            "kind_logits": kind_logits,
            "role_logits": role_logits,
        }

    def _empty_typed_algorithmic_value_state_outputs(
        self,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = int(reference.shape[0])
        steps = int(reference.shape[1]) if reference.ndim >= 3 else 0
        slots = max(1, int(self.cfg.typed_algorithmic_value_state_max_list_slots))
        offset_vocab = max(
            1,
            int(self.cfg.typed_algorithmic_value_state_offset_vocab_size),
        )
        scalar_vocab = max(
            1,
            int(self.cfg.typed_algorithmic_value_state_scalar_vocab_size),
        )
        kind_size = max(0, int(self.cfg.typed_algorithmic_value_state_kind_size))
        return {
            "kind_logits": reference.new_empty((b, steps, kind_size)),
            "raw_list_offset_logits": reference.new_empty(
                (b, steps, slots, offset_vocab)
            ),
            "doubled_list_offset_logits": reference.new_empty(
                (b, steps, slots, offset_vocab)
            ),
            "scalar_coeff_logits": reference.new_empty((b, steps, scalar_vocab)),
            "scalar_offset_logits": reference.new_empty((b, steps, scalar_vocab)),
            "scalar_residual_logits": reference.new_empty((b, steps, scalar_vocab)),
            "scalar_residual_delta_logits": reference.new_empty(
                (b, steps, scalar_vocab)
            ),
            "final_residual_logits": reference.new_empty((b, steps, scalar_vocab)),
            "scalar_coeff_value": reference.new_empty((b, steps)),
            "scalar_offset_value": reference.new_empty((b, steps)),
            "scalar_residual_value": reference.new_empty((b, steps)),
            "final_residual_value": reference.new_empty((b, steps)),
        }

    def _empty_typed_algorithmic_value_state_answer_bridge(
        self,
        reference: torch.Tensor,
        *,
        keep_steps: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = int(reference.shape[0])
        steps = int(reference.shape[1]) if reference.ndim >= 3 else 0
        token_shape = (b, steps, 1, self.cfg.d_model) if keep_steps else (b, 0, 0, self.cfg.d_model)
        gate_shape = (b, steps) if keep_steps else (b, 0)
        factory = reference.new_zeros if keep_steps else reference.new_empty
        return {
            "tokens": factory(token_shape),
            "gate_mean": factory(gate_shape),
        }

    def _compute_typed_algorithmic_value_state_answer_bridge(
        self,
        typed_algorithmic_value_state_info: dict[str, torch.Tensor],
        *,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        if (
            self.typed_algorithmic_value_state_answer_bridge_proj is None
            or self.typed_algorithmic_value_state_answer_bridge_norm is None
            or self.typed_algorithmic_value_state_answer_bridge_gate is None
        ):
            return self._empty_typed_algorithmic_value_state_answer_bridge(reference)
        b = int(reference.shape[0])
        steps = int(reference.shape[1]) if reference.ndim >= 3 else 0
        if disabled or steps == 0:
            return self._empty_typed_algorithmic_value_state_answer_bridge(
                reference,
                keep_steps=True,
            )
        scalar_logits = typed_algorithmic_value_state_info.get("scalar_residual_logits")
        final_logits = typed_algorithmic_value_state_info.get("final_residual_logits")
        if (
            scalar_logits is None
            or final_logits is None
            or scalar_logits.ndim != 3
            or final_logits.ndim != 3
            or tuple(scalar_logits.shape[:2]) != (b, steps)
            or tuple(final_logits.shape[:2]) != (b, steps)
            or int(scalar_logits.shape[-1]) != int(final_logits.shape[-1])
        ):
            return self._empty_typed_algorithmic_value_state_answer_bridge(reference)
        scalar_probs = torch.softmax(scalar_logits.float(), dim=-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        final_probs = torch.softmax(final_logits.float(), dim=-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        bridge_input = torch.cat([scalar_probs, final_probs], dim=-1)
        tokens = self.typed_algorithmic_value_state_answer_bridge_proj(bridge_input)
        gate = torch.sigmoid(self.typed_algorithmic_value_state_answer_bridge_gate(tokens))
        gate_min = min(
            max(float(self.cfg.typed_algorithmic_value_state_answer_bridge_gate_min), 0.0),
            1.0,
        )
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        tokens = self.typed_algorithmic_value_state_answer_bridge_norm(gate * tokens)
        return {
            "tokens": tokens.unsqueeze(2),
            "gate_mean": gate.squeeze(-1),
        }

    def _compute_typed_algorithmic_value_state_outputs(
        self,
        core_depth_states: torch.Tensor,
        *,
        transition_state_joint_logits: Optional[torch.Tensor] = None,
        primitive_operation_logits: Optional[torch.Tensor] = None,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
        recurrent_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        if (
            self.typed_algorithmic_value_state_norm is None
            or self.typed_algorithmic_kind_head is None
            or self.typed_algorithmic_raw_list_offset_head is None
            or self.typed_algorithmic_doubled_list_offset_head is None
            or self.typed_algorithmic_scalar_coeff_head is None
            or self.typed_algorithmic_scalar_residual_head is None
            or self.typed_algorithmic_final_residual_head is None
        ):
            return self._empty_typed_algorithmic_value_state_outputs(
                core_depth_states.new_empty(
                    (core_depth_states.shape[0], 0, core_depth_states.shape[-1])
                )
            )
        b = int(core_depth_states.shape[0])
        steps = int(core_depth_states.shape[1])
        slots = max(1, int(self.cfg.typed_algorithmic_value_state_max_list_slots))
        offset_vocab = int(
            self.typed_algorithmic_raw_list_offset_head.out_features // slots
        )
        scalar_vocab = int(self.typed_algorithmic_scalar_coeff_head.out_features)
        if disabled or steps == 0:
            return {
                "kind_logits": core_depth_states.new_zeros(
                    (b, steps, int(self.typed_algorithmic_kind_head.out_features))
                ),
                "raw_list_offset_logits": core_depth_states.new_zeros(
                    (b, steps, slots, offset_vocab)
                ),
                "doubled_list_offset_logits": core_depth_states.new_zeros(
                    (b, steps, slots, offset_vocab)
                ),
                "scalar_coeff_logits": core_depth_states.new_zeros(
                    (b, steps, scalar_vocab)
                ),
                "scalar_offset_logits": core_depth_states.new_zeros(
                    (b, steps, scalar_vocab)
                ),
                "scalar_residual_logits": core_depth_states.new_zeros(
                    (b, steps, scalar_vocab)
                ),
                "scalar_residual_delta_logits": core_depth_states.new_zeros(
                    (b, steps, scalar_vocab)
                ),
                "final_residual_logits": core_depth_states.new_zeros(
                    (b, steps, scalar_vocab)
                ),
                "scalar_coeff_value": core_depth_states.new_zeros((b, steps)),
                "scalar_offset_value": core_depth_states.new_zeros((b, steps)),
                "scalar_residual_value": core_depth_states.new_zeros((b, steps)),
                "final_residual_value": core_depth_states.new_zeros((b, steps)),
            }
        hidden = self.typed_algorithmic_value_state_norm(core_depth_states)
        if (
            self.typed_algorithmic_prompt_query_norm is not None
            and self.typed_algorithmic_prompt_context_norm is not None
            and self.typed_algorithmic_prompt_cross is not None
            and self.typed_algorithmic_prompt_gate is not None
            and self.typed_algorithmic_prompt_output_norm is not None
            and prompt_context_seq is not None
            and prompt_context_seq.numel() != 0
        ):
            prompt_context = self.typed_algorithmic_prompt_context_norm(
                prompt_context_seq
            ).to(dtype=hidden.dtype)
            prompt_delta = self.typed_algorithmic_prompt_cross(
                self.typed_algorithmic_prompt_query_norm(hidden),
                prompt_context,
                prompt_context_mask,
            )
            prompt_gate = torch.sigmoid(self.typed_algorithmic_prompt_gate(hidden))
            gate_min = min(
                max(float(self.cfg.typed_algorithmic_value_state_prompt_gate_min), 0.0),
                1.0,
            )
            if gate_min != 0.0:
                prompt_gate = gate_min + (1.0 - gate_min) * prompt_gate
            hidden = self.typed_algorithmic_prompt_output_norm(
                hidden + prompt_gate * prompt_delta
            )
        recurrent_active = (
            not recurrent_disabled
            and self.typed_algorithmic_recurrent_step_embed is not None
            and self.typed_algorithmic_recurrent_joint_proj is not None
            and self.typed_algorithmic_recurrent_input_norm is not None
            and self.typed_algorithmic_recurrent_update is not None
            and self.typed_algorithmic_recurrent_gate is not None
            and self.typed_algorithmic_recurrent_output_norm is not None
        )
        list_hidden = hidden
        scalar_hidden = hidden
        final_hidden = hidden
        if recurrent_active:
            gate_min = min(
                max(float(self.cfg.typed_algorithmic_value_state_recurrent_gate_min), 0.0),
                1.0,
            )
            joint_vocab = int(self.typed_algorithmic_recurrent_joint_proj.in_features)
            subregister_active = (
                self.typed_algorithmic_list_subregister_input_norm is not None
                and self.typed_algorithmic_list_subregister_update is not None
                and self.typed_algorithmic_list_subregister_gate is not None
                and self.typed_algorithmic_list_subregister_output_norm is not None
                and self.typed_algorithmic_scalar_subregister_input_norm is not None
                and self.typed_algorithmic_scalar_subregister_update is not None
                and self.typed_algorithmic_scalar_subregister_gate is not None
                and self.typed_algorithmic_scalar_subregister_output_norm is not None
                and self.typed_algorithmic_final_subregister_input_norm is not None
                and self.typed_algorithmic_final_subregister_update is not None
                and self.typed_algorithmic_final_subregister_gate is not None
                and self.typed_algorithmic_final_subregister_output_norm is not None
            )
            if subregister_active:
                list_carried = core_depth_states.new_zeros(
                    (b, core_depth_states.shape[-1])
                )
                scalar_carried = core_depth_states.new_zeros(
                    (b, core_depth_states.shape[-1])
                )
                final_carried = core_depth_states.new_zeros(
                    (b, core_depth_states.shape[-1])
                )
                list_states = []
                scalar_states = []
                final_states = []
            else:
                carried = core_depth_states.new_zeros((b, core_depth_states.shape[-1]))
                recurrent_states = []
            residual_feedback_active = (
                self.typed_algorithmic_scalar_residual_feedback_proj is not None
                and self.typed_algorithmic_final_residual_feedback_proj is not None
            )
            residual_feedback = core_depth_states.new_zeros(
                (b, core_depth_states.shape[-1])
            )
            for index in range(steps):
                step_index = min(
                    index,
                    int(self.typed_algorithmic_recurrent_step_embed.num_embeddings) - 1,
                )
                step_ids = torch.full(
                    (b,),
                    int(step_index),
                    device=core_depth_states.device,
                    dtype=torch.long,
                )
                step = self.typed_algorithmic_recurrent_step_embed(step_ids).to(
                    dtype=core_depth_states.dtype
                )
                joint = core_depth_states.new_zeros((b, joint_vocab))
                if (
                    transition_state_joint_logits is not None
                    and transition_state_joint_logits.ndim == 3
                    and int(transition_state_joint_logits.shape[1]) > index
                    and int(transition_state_joint_logits.shape[2]) == joint_vocab
                ):
                    joint = torch.softmax(
                        transition_state_joint_logits[:, index, :].float(),
                        dim=-1,
                    ).to(dtype=core_depth_states.dtype)
                joint_state = self.typed_algorithmic_recurrent_joint_proj(joint)
                primitive_state = core_depth_states.new_zeros(
                    (b, core_depth_states.shape[-1])
                )
                if (
                    self.typed_algorithmic_recurrent_primitive_proj is not None
                    and primitive_operation_logits is not None
                    and primitive_operation_logits.ndim == 3
                    and int(primitive_operation_logits.shape[1]) > index
                    and int(primitive_operation_logits.shape[2])
                    == int(self.typed_algorithmic_recurrent_primitive_proj.in_features)
                ):
                    primitive_probs = torch.softmax(
                        primitive_operation_logits[:, index, :].float(),
                        dim=-1,
                    ).to(dtype=core_depth_states.dtype)
                    primitive_state = self.typed_algorithmic_recurrent_primitive_proj(
                        primitive_probs
                    )
                base_input = (
                    hidden[:, index, :]
                    + step
                    + joint_state
                    + primitive_state
                    + residual_feedback
                )
                if subregister_active:
                    list_input = list_carried + base_input
                    list_delta = self.typed_algorithmic_list_subregister_update(
                        self.typed_algorithmic_list_subregister_input_norm(list_input)
                    )
                    list_gate = torch.sigmoid(
                        self.typed_algorithmic_list_subregister_gate(list_input)
                    )
                    scalar_input = scalar_carried + base_input
                    scalar_delta = self.typed_algorithmic_scalar_subregister_update(
                        self.typed_algorithmic_scalar_subregister_input_norm(
                            scalar_input
                        )
                    )
                    scalar_gate = torch.sigmoid(
                        self.typed_algorithmic_scalar_subregister_gate(scalar_input)
                    )
                    final_input = final_carried + base_input
                    final_delta = self.typed_algorithmic_final_subregister_update(
                        self.typed_algorithmic_final_subregister_input_norm(final_input)
                    )
                    final_gate = torch.sigmoid(
                        self.typed_algorithmic_final_subregister_gate(final_input)
                    )
                    if gate_min != 0.0:
                        list_gate = gate_min + (1.0 - gate_min) * list_gate
                        scalar_gate = gate_min + (1.0 - gate_min) * scalar_gate
                        final_gate = gate_min + (1.0 - gate_min) * final_gate
                    list_carried = self.typed_algorithmic_list_subregister_output_norm(
                        list_carried + list_gate * list_delta
                    )
                    scalar_carried = (
                        self.typed_algorithmic_scalar_subregister_output_norm(
                            scalar_carried + scalar_gate * scalar_delta
                        )
                    )
                    final_carried = self.typed_algorithmic_final_subregister_output_norm(
                        final_carried + final_gate * final_delta
                    )
                    list_states.append(list_carried)
                    scalar_states.append(scalar_carried)
                    final_states.append(final_carried)
                    if residual_feedback_active:
                        scalar_residual_probs = torch.softmax(
                            self.typed_algorithmic_scalar_residual_head(
                                scalar_carried
                            ).float(),
                            dim=-1,
                        ).to(dtype=core_depth_states.dtype)
                        final_residual_probs = torch.softmax(
                            self.typed_algorithmic_final_residual_head(
                                final_carried
                            ).float(),
                            dim=-1,
                        ).to(dtype=core_depth_states.dtype)
                        residual_feedback = (
                            self.typed_algorithmic_scalar_residual_feedback_proj(
                                scalar_residual_probs
                            )
                            + self.typed_algorithmic_final_residual_feedback_proj(
                                final_residual_probs
                            )
                        )
                else:
                    recurrent_input = carried + base_input
                    delta = self.typed_algorithmic_recurrent_update(
                        self.typed_algorithmic_recurrent_input_norm(recurrent_input)
                    )
                    gate = torch.sigmoid(
                        self.typed_algorithmic_recurrent_gate(recurrent_input)
                    )
                    if gate_min != 0.0:
                        gate = gate_min + (1.0 - gate_min) * gate
                    carried = self.typed_algorithmic_recurrent_output_norm(
                        carried + gate * delta
                    )
                    recurrent_states.append(carried)
                    if residual_feedback_active:
                        scalar_residual_probs = torch.softmax(
                            self.typed_algorithmic_scalar_residual_head(carried).float(),
                            dim=-1,
                        ).to(dtype=core_depth_states.dtype)
                        final_residual_probs = torch.softmax(
                            self.typed_algorithmic_final_residual_head(carried).float(),
                            dim=-1,
                        ).to(dtype=core_depth_states.dtype)
                        residual_feedback = (
                            self.typed_algorithmic_scalar_residual_feedback_proj(
                                scalar_residual_probs
                            )
                            + self.typed_algorithmic_final_residual_feedback_proj(
                                final_residual_probs
                            )
                        )
            if subregister_active:
                list_hidden = torch.stack(list_states, dim=1)
                scalar_hidden = torch.stack(scalar_states, dim=1)
                final_hidden = torch.stack(final_states, dim=1)
                hidden = (list_hidden + scalar_hidden + final_hidden) / 3.0
            else:
                hidden = torch.stack(recurrent_states, dim=1)
                list_hidden = hidden
                scalar_hidden = hidden
                final_hidden = hidden
        return {
            "kind_logits": self.typed_algorithmic_kind_head(hidden),
            "raw_list_offset_logits": self.typed_algorithmic_raw_list_offset_head(
                list_hidden
            ).view(b, steps, slots, offset_vocab),
            "doubled_list_offset_logits": (
                self.typed_algorithmic_doubled_list_offset_head(list_hidden).view(
                    b,
                    steps,
                    slots,
                    offset_vocab,
                )
            ),
            "scalar_coeff_logits": self.typed_algorithmic_scalar_coeff_head(
                scalar_hidden
            ),
            "scalar_coeff_value": (
                self.typed_algorithmic_scalar_coeff_value_head(scalar_hidden).squeeze(-1)
                if self.typed_algorithmic_scalar_coeff_value_head is not None
                else core_depth_states.new_zeros((b, steps))
            ),
            "scalar_offset_logits": (
                self.typed_algorithmic_scalar_offset_head(scalar_hidden)
                if self.typed_algorithmic_scalar_offset_head is not None
                else core_depth_states.new_zeros((b, steps, scalar_vocab))
            ),
            "scalar_offset_value": (
                self.typed_algorithmic_scalar_offset_value_head(scalar_hidden).squeeze(-1)
                if self.typed_algorithmic_scalar_offset_value_head is not None
                else core_depth_states.new_zeros((b, steps))
            ),
            "scalar_residual_logits": self.typed_algorithmic_scalar_residual_head(
                scalar_hidden
            ),
            "scalar_residual_value": (
                self.typed_algorithmic_scalar_residual_value_head(scalar_hidden).squeeze(-1)
                if self.typed_algorithmic_scalar_residual_value_head is not None
                else core_depth_states.new_zeros((b, steps))
            ),
            "scalar_residual_delta_logits": (
                self.typed_algorithmic_scalar_residual_delta_head(scalar_hidden)
                if self.typed_algorithmic_scalar_residual_delta_head is not None
                else core_depth_states.new_zeros((b, steps, scalar_vocab))
            ),
            "final_residual_logits": self.typed_algorithmic_final_residual_head(
                final_hidden
            ),
            "final_residual_value": (
                self.typed_algorithmic_final_residual_value_head(final_hidden).squeeze(-1)
                if self.typed_algorithmic_final_residual_value_head is not None
                else core_depth_states.new_zeros((b, steps))
            ),
        }

    def _empty_core_role_value_state_logits(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        b = int(reference.shape[0])
        steps = int(reference.shape[1]) if reference.ndim >= 3 else 0
        role_count = max(0, int(self.cfg.core_role_value_state_num_roles))
        vocab = (
            int(self.core_role_value_state_head.out_features)
            if self.core_role_value_state_head is not None
            else max(1, int(self.cfg.core_role_value_state_vocab_size or 1))
        )
        return reference.new_empty((b, steps, role_count, vocab))

    def _empty_core_role_value_transition_logits(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        b = int(reference.shape[0])
        role_count = max(0, int(self.cfg.core_role_value_state_num_roles))
        vocab = (
            int(self.core_role_value_transition_head.out_features)
            if self.core_role_value_transition_head is not None
            else max(1, int(self.cfg.core_role_value_state_vocab_size or 1))
        )
        return reference.new_empty((b, 0, role_count, vocab))

    def _empty_core_role_value_state_answer_bridge(
        self,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = int(reference.shape[0])
        return {
            "tokens": reference.new_empty((b, 0, 0, self.cfg.d_model)),
            "gate_mean": reference.new_empty((b, 0)),
        }

    def _compute_core_role_value_state_answer_bridge(
        self,
        role_value_logits: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        if (
            disabled
            or self.core_role_value_state_embed is None
            or self.core_role_value_state_answer_value_embed is None
            or self.core_role_value_state_answer_norm is None
            or self.core_role_value_state_answer_gate is None
            or role_value_logits.ndim != 4
            or int(role_value_logits.shape[1]) == 0
            or int(role_value_logits.shape[2]) == 0
        ):
            return self._empty_core_role_value_state_answer_bridge(role_value_logits)
        if int(role_value_logits.shape[-1]) != int(
            self.core_role_value_state_answer_value_embed.num_embeddings
        ):
            return self._empty_core_role_value_state_answer_bridge(role_value_logits)
        role_count = int(role_value_logits.shape[2])
        value_weight = self.core_role_value_state_answer_value_embed.weight.to(
            device=role_value_logits.device,
            dtype=role_value_logits.dtype,
        )
        value_probs = torch.softmax(role_value_logits.float(), dim=-1).to(
            dtype=role_value_logits.dtype
        )
        value_tokens = value_probs @ value_weight
        role_tokens = self.core_role_value_state_embed.weight[:role_count].to(
            device=role_value_logits.device,
            dtype=role_value_logits.dtype,
        )
        bridge_tokens = self.core_role_value_state_answer_norm(
            value_tokens + role_tokens.view(1, 1, role_count, -1)
        )
        if (
            self.core_role_value_state_answer_prompt_cross is not None
            and self.core_role_value_state_answer_prompt_query_norm is not None
            and self.core_role_value_state_answer_prompt_context_norm is not None
            and self.core_role_value_state_answer_prompt_gate is not None
            and self.core_role_value_state_answer_prompt_output_norm is not None
            and prompt_context_seq is not None
            and prompt_context_mask is not None
            and prompt_context_seq.numel() != 0
        ):
            b, steps, roles, dim = bridge_tokens.shape
            flat_tokens = bridge_tokens.reshape(b, steps * roles, dim)
            prompt_delta = self.core_role_value_state_answer_prompt_cross(
                self.core_role_value_state_answer_prompt_query_norm(flat_tokens),
                self.core_role_value_state_answer_prompt_context_norm(prompt_context_seq),
                prompt_context_mask,
            )
            prompt_gate = torch.sigmoid(
                self.core_role_value_state_answer_prompt_gate(flat_tokens)
            )
            prompt_gate_min = min(
                max(float(self.cfg.core_role_value_state_answer_prompt_gate_min), 0.0),
                1.0,
            )
            if prompt_gate_min != 0.0:
                prompt_gate = prompt_gate_min + (1.0 - prompt_gate_min) * prompt_gate
            flat_tokens = self.core_role_value_state_answer_prompt_output_norm(
                flat_tokens + prompt_gate * prompt_delta
            )
            bridge_tokens = flat_tokens.reshape(b, steps, roles, dim)
        gate = torch.sigmoid(
            self.core_role_value_state_answer_gate(bridge_tokens.mean(dim=2))
        ).squeeze(-1)
        gate_min = min(
            max(float(self.cfg.core_role_value_state_answer_bridge_gate_min), 0.0),
            1.0,
        )
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        bridge_tokens = bridge_tokens * gate[:, :, None, None].to(
            dtype=bridge_tokens.dtype
        )
        return {"tokens": bridge_tokens, "gate_mean": gate}

    def _empty_core_role_value_state_answer_final_embedding(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return reference.new_empty((int(reference.shape[0]), 0, self.cfg.d_model))

    def _compute_core_role_value_state_answer_final_embedding(
        self,
        bridge_tokens: torch.Tensor,
        *,
        disabled: bool = False,
    ) -> torch.Tensor:
        if (
            disabled
            or self.core_role_value_state_answer_final_proj is None
            or self.core_role_value_state_answer_final_norm is None
            or bridge_tokens.ndim != 4
            or int(bridge_tokens.shape[1]) == 0
            or int(bridge_tokens.shape[2]) == 0
        ):
            return self._empty_core_role_value_state_answer_final_embedding(
                bridge_tokens
            )
        final_tokens = bridge_tokens[:, -1, :, :]
        final_state = self.core_role_value_state_answer_final_norm(
            final_tokens.mean(dim=1)
        )
        return self.core_role_value_state_answer_final_proj(final_state).unsqueeze(1)

    def _empty_core_role_value_state_vocab_renderer_logits(
        self,
        reference: torch.Tensor,
        input_seq_len: int,
    ) -> torch.Tensor:
        return reference.new_zeros(
            (
                int(reference.shape[0]),
                int(input_seq_len),
                int(self.cfg.vocab_size),
            )
        )

    def _select_source_copy_position_logits_for_renderer(
        self,
        *,
        source_position_prompt_logits: torch.Tensor,
        core_role_value_state_logits: torch.Tensor,
        core_primitive_role_value_state_logits: Optional[torch.Tensor] = None,
        bridge_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source_position_prompt_logits.ndim != 4
            or int(source_position_prompt_logits.shape[1]) == 0
            or source_position_prompt_logits.numel() == 0
        ):
            return source_position_prompt_logits
        if (
            bool(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_from_primitive_enabled
            )
            and core_primitive_role_value_state_logits is not None
            and core_primitive_role_value_state_logits.ndim == 4
            and int(core_primitive_role_value_state_logits.shape[1]) > 0
            and int(core_primitive_role_value_state_logits.shape[2])
            == int(bridge_tokens.shape[2])
            and int(core_primitive_role_value_state_logits.shape[0])
            == int(source_position_prompt_logits.shape[0])
            and int(core_primitive_role_value_state_logits.shape[-1]) > 0
        ):
            return core_primitive_role_value_state_logits[:, -1:, :, :]
        if (
            core_role_value_state_logits.ndim == 4
            and int(core_role_value_state_logits.shape[1]) > 0
            and int(core_role_value_state_logits.shape[2])
            == int(bridge_tokens.shape[2])
            and int(core_role_value_state_logits.shape[0])
            == int(source_position_prompt_logits.shape[0])
            and int(core_role_value_state_logits.shape[-1]) > 0
        ):
            return core_role_value_state_logits[:, -1:, :, :]
        return source_position_prompt_logits

    def _mask_source_copy_position_logits_to_answer_roles(
        self,
        source_copy_position_logits: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source_copy_position_logits.ndim != 4
            or int(source_copy_position_logits.shape[2]) == 0
            or int(source_copy_position_logits.shape[-1]) == 0
        ):
            return source_copy_position_logits
        role_count = int(source_copy_position_logits.shape[2])
        answer_roles = max(1, (int(self.cfg.core_role_value_state_num_roles) - 2) // 2)
        answer_start = 0
        answer_end = min(role_count, answer_roles)
        if answer_start <= 0 and answer_end >= role_count:
            return source_copy_position_logits
        masked = source_copy_position_logits.clone()
        masked[:, :, :, :] = -1.0e4
        masked[:, :, :, 0] = 1.0e4
        masked[:, :, answer_start:answer_end, :] = source_copy_position_logits[
            :, :, answer_start:answer_end, :
        ]
        return masked

    def _compute_source_copy_cursor_role_bias(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        source_copy_token_ids: Optional[torch.Tensor],
        source_copy_token_mask: Optional[torch.Tensor],
        query_token_indices: Optional[torch.Tensor],
        output_seq_len: int,
        role_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = (
            int(input_ids.shape[0])
            if input_ids is not None
            else int(source_copy_token_ids.shape[0])
            if source_copy_token_ids is not None and source_copy_token_ids.ndim >= 1
            else 0
        )
        bias = torch.zeros(
            (
                batch_size,
                int(output_seq_len),
                int(role_count),
            ),
            device=device,
            dtype=dtype,
        )
        if (
            not bool(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_cursor_enabled
            )
            or float(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_cursor_bias
            )
            == 0.0
            or input_ids is None
            or source_copy_token_ids is None
            or input_ids.ndim != 2
            or source_copy_token_ids.ndim != 2
            or int(input_ids.shape[0]) != int(source_copy_token_ids.shape[0])
            or int(output_seq_len) <= 0
            or int(role_count) <= 0
        ):
            return bias
        if query_token_indices is None:
            query_positions = torch.arange(
                int(output_seq_len),
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            query_positions = query_token_indices.to(
                device=input_ids.device,
                dtype=torch.long,
            ).reshape(-1)
        if int(query_positions.numel()) != int(output_seq_len):
            return bias
        answer_roles = min(
            int(role_count),
            max(1, (int(self.cfg.core_role_value_state_num_roles) - 2) // 2),
        )
        if answer_roles <= 0:
            return bias
        if source_copy_token_mask is not None and source_copy_token_mask.ndim == 2:
            valid_mask = source_copy_token_mask.to(
                device=source_copy_token_ids.device,
                dtype=torch.bool,
            )
        else:
            valid_mask = source_copy_token_ids != 0
        cursor_bias = float(
            self.cfg.core_role_value_state_vocab_renderer_source_copy_cursor_bias
        )
        for batch_index in range(int(input_ids.shape[0])):
            ids = source_copy_token_ids[batch_index][valid_mask[batch_index]]
            if int(ids.numel()) == 0:
                continue
            valid_count = int(ids.numel())
            unique_ids = torch.unique(ids.to(device=input_ids.device, dtype=torch.long))
            for out_index, raw_query_position in enumerate(query_positions.tolist()):
                query_position = int(raw_query_position)
                if query_position < 0 or query_position >= int(input_ids.shape[1]):
                    continue
                last_token = input_ids[batch_index, query_position]
                if bool((unique_ids == last_token).any().item()):
                    continue
                prefix = input_ids[batch_index, : query_position + 1]
                seen = 0
                for token_id in unique_ids.tolist():
                    seen += int((prefix == int(token_id)).sum().item())
                cursor = max(0, seen - valid_count)
                role_index = min(answer_roles - 1, cursor)
                bias[batch_index, out_index, role_index] = cursor_bias
        return bias

    def _compute_source_copy_span_next_token_ids(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        source_copy_token_span_ids: Optional[torch.Tensor],
        source_copy_token_span_mask: Optional[torch.Tensor],
        query_token_indices: Optional[torch.Tensor],
        output_seq_len: int,
        position_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = (
            int(input_ids.shape[0])
            if input_ids is not None and input_ids.ndim >= 1
            else int(source_copy_token_span_ids.shape[0])
            if source_copy_token_span_ids is not None
            and source_copy_token_span_ids.ndim >= 1
            else 0
        )
        next_ids = torch.zeros(
            (batch_size, int(output_seq_len), int(position_count)),
            device=device,
            dtype=torch.long,
        )
        next_valid = torch.zeros_like(next_ids, dtype=torch.bool)
        if (
            not bool(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_span_enabled
            )
            or input_ids is None
            or source_copy_token_span_ids is None
            or input_ids.ndim != 2
            or source_copy_token_span_ids.ndim != 3
            or int(input_ids.shape[0]) != int(source_copy_token_span_ids.shape[0])
            or int(output_seq_len) <= 0
            or int(position_count) <= 1
        ):
            return next_ids, next_valid
        if query_token_indices is None:
            query_positions = torch.arange(
                int(output_seq_len),
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            query_positions = query_token_indices.to(
                device=input_ids.device,
                dtype=torch.long,
            ).reshape(-1)
        if int(query_positions.numel()) != int(output_seq_len):
            return next_ids, next_valid
        span_ids = source_copy_token_span_ids.to(device=input_ids.device, dtype=torch.long)
        if source_copy_token_span_mask is not None and source_copy_token_span_mask.ndim == 3:
            span_mask = source_copy_token_span_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )
        else:
            span_mask = span_ids != 0
        max_slots = min(int(span_ids.shape[1]), int(position_count) - 1)
        max_pieces = min(
            int(span_ids.shape[2]),
            max(1, int(self.cfg.core_role_value_state_vocab_renderer_source_copy_span_max_pieces)),
        )
        for batch_index in range(int(input_ids.shape[0])):
            for out_index, raw_query_position in enumerate(query_positions.tolist()):
                query_position = int(raw_query_position)
                if query_position < 0 or query_position >= int(input_ids.shape[1]):
                    continue
                prefix = input_ids[batch_index, : query_position + 1].to(
                    device=input_ids.device,
                    dtype=torch.long,
                )
                for slot_index in range(max_slots):
                    valid_piece_mask = span_mask[batch_index, slot_index, :max_pieces]
                    pieces = span_ids[batch_index, slot_index, :max_pieces][
                        valid_piece_mask
                    ]
                    piece_count = int(pieces.numel())
                    if piece_count <= 0:
                        continue
                    class_index = slot_index + 1
                    full_len = min(piece_count, int(prefix.numel()))
                    if full_len == piece_count and torch.equal(
                        prefix[-piece_count:], pieces
                    ):
                        continue
                    next_piece = int(pieces[0].item())
                    max_partial = min(piece_count - 1, int(prefix.numel()))
                    for partial_len in range(max_partial, 0, -1):
                        if torch.equal(prefix[-partial_len:], pieces[:partial_len]):
                            next_piece = int(pieces[partial_len].item())
                            break
                    next_ids[batch_index, out_index, class_index] = next_piece
                    next_valid[batch_index, out_index, class_index] = True
        return next_ids.to(device=device), next_valid.to(device=device)

    def _compute_source_copy_answer_role_cursor_bias(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        source_copy_token_span_ids: Optional[torch.Tensor],
        source_copy_token_span_mask: Optional[torch.Tensor],
        query_token_indices: Optional[torch.Tensor],
        output_seq_len: int,
        role_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = (
            int(input_ids.shape[0])
            if input_ids is not None and input_ids.ndim >= 1
            else int(source_copy_token_span_ids.shape[0])
            if source_copy_token_span_ids is not None
            and source_copy_token_span_ids.ndim >= 1
            else 0
        )
        bias = torch.zeros(
            (batch_size, int(output_seq_len), int(role_count)),
            device=device,
            dtype=dtype,
        )
        if (
            not bool(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled
            )
            or float(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias
            )
            == 0.0
            or input_ids is None
            or source_copy_token_span_ids is None
            or input_ids.ndim != 2
            or source_copy_token_span_ids.ndim != 3
            or int(input_ids.shape[0]) != int(source_copy_token_span_ids.shape[0])
            or int(output_seq_len) <= 0
            or int(role_count) <= 0
        ):
            return bias
        if query_token_indices is None:
            query_positions = torch.arange(
                int(output_seq_len),
                device=input_ids.device,
                dtype=torch.long,
            )
        else:
            query_positions = query_token_indices.to(
                device=input_ids.device,
                dtype=torch.long,
            ).reshape(-1)
        if int(query_positions.numel()) != int(output_seq_len):
            return bias
        span_ids = source_copy_token_span_ids.to(device=input_ids.device, dtype=torch.long)
        if source_copy_token_span_mask is not None and source_copy_token_span_mask.ndim == 3:
            span_mask = source_copy_token_span_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )
        else:
            span_mask = span_ids != 0
        max_pieces = min(
            int(span_ids.shape[2]),
            max(1, int(self.cfg.core_role_value_state_vocab_renderer_source_copy_span_max_pieces)),
        )
        answer_roles = min(
            int(role_count),
            max(1, (int(self.cfg.core_role_value_state_num_roles) - 2) // 2),
        )
        if answer_roles <= 0:
            return bias
        separators = {
            int(token_id)
            for token_id in self.cfg.core_role_value_state_vocab_renderer_source_copy_answer_role_separator_token_ids
        }
        cursor_bias = float(
            self.cfg.core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_bias
        )
        for batch_index in range(int(input_ids.shape[0])):
            source_spans: list[torch.Tensor] = []
            for slot_index in range(int(span_ids.shape[1])):
                piece_mask = span_mask[batch_index, slot_index, :max_pieces]
                pieces = span_ids[batch_index, slot_index, :max_pieces][piece_mask]
                if int(pieces.numel()) > 0:
                    source_spans.append(pieces)
            if not source_spans:
                continue
            source_slot_count = len(source_spans)
            for out_index, raw_query_position in enumerate(query_positions.tolist()):
                query_position = int(raw_query_position)
                if query_position < 0 or query_position >= int(input_ids.shape[1]):
                    continue
                prefix = input_ids[batch_index, : query_position + 1].to(
                    device=input_ids.device,
                    dtype=torch.long,
                )
                if int(prefix.numel()) == 0:
                    continue
                ends_with_full_span = False
                full_occurrences = 0
                for pieces in source_spans:
                    piece_count = int(pieces.numel())
                    if piece_count <= 0:
                        continue
                    if int(prefix.numel()) >= piece_count and torch.equal(
                        prefix[-piece_count:], pieces
                    ):
                        ends_with_full_span = True
                    limit = int(prefix.numel()) - piece_count + 1
                    for start in range(max(0, limit)):
                        if torch.equal(prefix[start : start + piece_count], pieces):
                            full_occurrences += 1
                if ends_with_full_span:
                    continue
                completed_answer_spans = max(0, full_occurrences - source_slot_count)
                last_token = int(prefix[-1].item())
                if completed_answer_spans > 0 and last_token not in separators:
                    continue
                role_index = min(answer_roles - 1, completed_answer_spans)
                bias[batch_index, out_index, role_index] = cursor_bias
        return bias

    def _compute_core_role_value_state_vocab_renderer_logits(
        self,
        text_context_seq: torch.Tensor,
        bridge_tokens: torch.Tensor,
        *,
        extra_state_tokens: Optional[torch.Tensor] = None,
        source_copy_position_logits: Optional[torch.Tensor] = None,
        source_copy_token_ids: Optional[torch.Tensor] = None,
        source_copy_token_mask: Optional[torch.Tensor] = None,
        source_copy_token_span_ids: Optional[torch.Tensor] = None,
        source_copy_token_span_mask: Optional[torch.Tensor] = None,
        source_copy_input_ids: Optional[torch.Tensor] = None,
        input_seq_len: int,
        query_token_indices: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        input_seq_len = int(input_seq_len)
        output_seq_len = (
            int(query_token_indices.numel())
            if query_token_indices is not None
            else input_seq_len
        )
        if (
            disabled
            or self.core_role_value_state_vocab_renderer_query_norm is None
            or self.core_role_value_state_vocab_renderer_state_norm is None
            or self.core_role_value_state_vocab_renderer_cross is None
            or self.core_role_value_state_vocab_renderer_gate is None
            or self.core_role_value_state_vocab_renderer_output_norm is None
            or self.core_role_value_state_vocab_renderer_down is None
            or self.core_role_value_state_vocab_renderer_up is None
            or bridge_tokens.ndim != 4
            or int(bridge_tokens.shape[1]) == 0
            or int(bridge_tokens.shape[2]) == 0
        ):
            return self._empty_core_role_value_state_vocab_renderer_logits(
                text_context_seq,
                output_seq_len,
            )
        query = text_context_seq[:, -input_seq_len:, :]
        if query_token_indices is not None:
            query = query.index_select(1, query_token_indices)
        state_tokens = bridge_tokens.reshape(
            int(bridge_tokens.shape[0]),
            int(bridge_tokens.shape[1]) * int(bridge_tokens.shape[2]),
            int(bridge_tokens.shape[3]),
        )
        if (
            extra_state_tokens is not None
            and extra_state_tokens.ndim == 3
            and int(extra_state_tokens.shape[0]) == int(state_tokens.shape[0])
            and int(extra_state_tokens.shape[-1]) == int(state_tokens.shape[-1])
            and extra_state_tokens.numel() != 0
        ):
            state_tokens = torch.cat(
                [
                    state_tokens,
                    extra_state_tokens.to(
                        device=state_tokens.device,
                        dtype=state_tokens.dtype,
                    ),
                ],
                dim=1,
            )
        state_mask = text_context_seq.new_ones(
            (int(text_context_seq.shape[0]), int(state_tokens.shape[1])),
            dtype=torch.long,
        )
        delta = self.core_role_value_state_vocab_renderer_cross(
            self.core_role_value_state_vocab_renderer_query_norm(query),
            self.core_role_value_state_vocab_renderer_state_norm(state_tokens),
            state_mask,
        )
        gate = torch.sigmoid(self.core_role_value_state_vocab_renderer_gate(query))
        gate_min = min(
            max(float(self.cfg.core_role_value_state_vocab_renderer_gate_min), 0.0),
            1.0,
        )
        if gate_min != 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        hidden = self.core_role_value_state_vocab_renderer_output_norm(
            query + gate * delta
        )
        if bool(self.cfg.core_role_value_state_vocab_renderer_use_lm_head):
            logits = self.lm_head(hidden) * float(self.cfg.qtrm_logits_scale)
        else:
            logits = self.core_role_value_state_vocab_renderer_up(
                self.core_role_value_state_vocab_renderer_down(hidden)
            )
        if (
            bool(self.cfg.core_role_value_state_vocab_renderer_source_copy_enabled)
            and source_copy_position_logits is not None
            and source_copy_token_ids is not None
            and source_copy_position_logits.ndim == 4
            and source_copy_token_ids.ndim == 2
            and int(source_copy_position_logits.shape[0]) == int(logits.shape[0])
            and int(source_copy_token_ids.shape[0]) == int(logits.shape[0])
            and int(source_copy_position_logits.shape[1]) > 0
            and int(source_copy_position_logits.shape[2]) == int(bridge_tokens.shape[2])
            and source_copy_position_logits.numel() != 0
        ):
            position_count = int(source_copy_position_logits.shape[3])
            copy_ids = source_copy_token_ids.to(device=logits.device, dtype=torch.long)
            aligned_ids = copy_ids.new_zeros((int(copy_ids.shape[0]), position_count))
            aligned_valid = torch.zeros_like(aligned_ids, dtype=torch.bool)
            # Source-position targets reserve class 0 for NULL. Class 1 must
            # therefore copy the first source token, class 2 the second, etc.
            copy_width = min(int(copy_ids.shape[1]), max(0, position_count - 1))
            if copy_width > 0:
                aligned_ids[:, 1 : 1 + copy_width] = copy_ids[:, :copy_width]
                aligned_valid[:, 1 : 1 + copy_width] = True
            copy_ids = aligned_ids
            copy_id_valid = aligned_valid
            role_tokens = bridge_tokens[:, -1, :, :].to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            role_keys = self.core_role_value_state_vocab_renderer_state_norm(role_tokens)
            role_scores = torch.matmul(
                hidden.float(),
                role_keys.float().transpose(-1, -2),
            ) / math.sqrt(max(1, int(hidden.shape[-1])))
            role_scores = role_scores + self._compute_source_copy_cursor_role_bias(
                input_ids=source_copy_input_ids,
                source_copy_token_ids=source_copy_token_ids,
                source_copy_token_mask=source_copy_token_mask,
                query_token_indices=query_token_indices,
                output_seq_len=int(role_scores.shape[1]),
                role_count=int(role_scores.shape[2]),
                device=role_scores.device,
                dtype=role_scores.dtype,
            )
            if bool(
                self.cfg.core_role_value_state_vocab_renderer_source_copy_answer_role_cursor_enabled
            ):
                role_scores = role_scores + self._compute_source_copy_answer_role_cursor_bias(
                    input_ids=source_copy_input_ids,
                    source_copy_token_span_ids=source_copy_token_span_ids,
                    source_copy_token_span_mask=source_copy_token_span_mask,
                    query_token_indices=query_token_indices,
                    output_seq_len=int(role_scores.shape[1]),
                    role_count=int(role_scores.shape[2]),
                    device=role_scores.device,
                    dtype=role_scores.dtype,
                )
            position_probs = torch.softmax(
                source_copy_position_logits[:, 0, :, :].float(),
                dim=-1,
            ).to(device=hidden.device, dtype=role_scores.dtype)
            copy_scores = torch.einsum(
                "bor,brs->bos",
                role_scores,
                position_probs,
            ).to(dtype=logits.dtype)
            span_copy_active = (
                bool(
                    self.cfg.core_role_value_state_vocab_renderer_source_copy_span_enabled
                )
                and source_copy_token_span_ids is not None
                and source_copy_token_span_ids.ndim == 3
                and int(source_copy_token_span_ids.shape[0]) == int(logits.shape[0])
            )
            span_next_ids, span_next_valid = (
                self._compute_source_copy_span_next_token_ids(
                    input_ids=source_copy_input_ids,
                    source_copy_token_span_ids=source_copy_token_span_ids,
                    source_copy_token_span_mask=source_copy_token_span_mask,
                    query_token_indices=query_token_indices,
                    output_seq_len=int(logits.shape[1]),
                    position_count=int(position_count),
                    device=logits.device,
                )
            )
            if span_copy_active:
                safe_ids = span_next_ids.masked_fill(~span_next_valid, 0)
                valid = (
                    span_next_valid
                    & (safe_ids >= 0)
                    & (safe_ids < int(logits.shape[-1]))
                )
            else:
                valid = (
                    copy_id_valid
                    & (copy_ids >= 0)
                    & (copy_ids < int(logits.shape[-1]))
                )
                safe_ids = copy_ids.masked_fill(~valid, 0)
            copy_logits = torch.zeros_like(logits)
            if safe_ids.ndim == 2:
                expanded_ids = safe_ids[:, None, :].expand(
                    -1,
                    int(logits.shape[1]),
                    -1,
                )
                scatter_scores = copy_scores.masked_fill(~valid[:, None, :], 0.0)
            else:
                expanded_ids = safe_ids
                scatter_scores = copy_scores.masked_fill(~valid, 0.0)
            copy_logits.scatter_add_(
                -1,
                expanded_ids,
                scatter_scores,
            )
            logits = logits + copy_logits
        candidate_token_ids = self.cfg.core_role_value_state_vocab_renderer_candidate_token_ids
        if candidate_token_ids:
            candidate_ids = torch.as_tensor(
                [int(token_id) for token_id in candidate_token_ids],
                device=logits.device,
                dtype=torch.long,
            )
            candidate_ids = candidate_ids[
                (candidate_ids >= 0) & (candidate_ids < int(logits.shape[-1]))
            ]
            if int(candidate_ids.numel()) == 0:
                logits = torch.zeros_like(logits)
            else:
                masked_logits = torch.zeros_like(logits)
                masked_logits.index_copy_(
                    -1,
                    candidate_ids,
                    logits.index_select(-1, candidate_ids),
                )
                logits = masked_logits
        return logits * float(self.cfg.core_role_value_state_vocab_renderer_scale)

    def _empty_core_value_delta_code_logits(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        b = int(reference.shape[0])
        steps = int(reference.shape[1]) if reference.ndim >= 3 else 0
        role_count = max(0, int(self.cfg.core_role_value_state_num_roles))
        codebook = (
            int(self.core_value_delta_code_head.out_features)
            if self.core_value_delta_code_head is not None
            else max(1, int(self.cfg.core_value_delta_codebook_size or 1))
        )
        return reference.new_empty((b, steps, role_count, codebook))

    def _empty_core_typed_register_outputs(
        self,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = int(reference.shape[0])
        role_count = max(0, int(self.cfg.core_role_value_state_num_roles))
        operation_count = (
            int(self.core_typed_register_operation_head.out_features)
            if self.core_typed_register_operation_head is not None
            else max(1, int(self.cfg.core_typed_register_num_operations or 1))
        )
        vocab = (
            int(self.core_typed_register_value_head.out_features)
            if self.core_typed_register_value_head is not None
            else max(1, int(self.cfg.core_role_value_state_vocab_size or 1))
        )
        return {
            "operation_logits": reference.new_empty((b, 0, operation_count)),
            "value_logits": reference.new_empty((b, 0, role_count, vocab)),
            "transition_logits": reference.new_empty((b, 0, role_count, vocab)),
            "gate_mean": reference.new_empty((b, 0)),
        }

    def _compute_core_role_value_template_outputs(
        self,
        trajectory: list[torch.Tensor],
        *,
        role_token_start: Optional[int],
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = int(reference.shape[0])
        role_count = max(0, int(self.cfg.core_role_value_state_num_roles))
        vocab = max(1, int(self.cfg.core_role_value_state_vocab_size or 1))
        empty = {
            "template_logits": reference.new_empty((b, 0)),
            "value_logits": reference.new_empty((b, 0, role_count, vocab)),
        }
        if (
            disabled
            or self.core_role_value_template_context_norm is None
            or self.core_role_value_template_head is None
            or self.core_role_value_template_table is None
            or self.core_role_value_state_embed is None
            or role_token_start is None
            or not trajectory
        ):
            return empty
        role_count = int(self.core_role_value_state_embed.num_embeddings)
        role_start = int(role_token_start)
        role_end = role_start + role_count
        last_state = trajectory[-1]
        if int(last_state.shape[1]) < role_end:
            return empty
        if role_start > 0:
            context_state = last_state[:, :role_start, :].mean(dim=1)
        else:
            context_state = last_state[:, 0, :]
        role_summary = last_state[:, role_start:role_end, :].mean(dim=1)
        template_input = self.core_role_value_template_context_norm(
            context_state + role_summary
        )
        if (
            self.core_role_value_template_length_head is not None
            and self.core_role_value_template_parity_head is not None
            and self.core_role_value_template_offset_head is not None
        ):
            length_logits = self.core_role_value_template_length_head(template_input)
            parity_logits = self.core_role_value_template_parity_head(template_input)
            offset_logits = self.core_role_value_template_offset_head(template_input)
            length_probs = torch.softmax(length_logits.float(), dim=-1)
            parity_probs = torch.softmax(parity_logits.float(), dim=-1)
            offset_probs = torch.softmax(offset_logits.float(), dim=-1)
            template_probs_float = (
                length_probs[:, :, None, None]
                * parity_probs[:, None, :, None]
                * offset_probs[:, None, None, :]
            ).reshape(b, -1)
            template_count = int(self.core_role_value_template_table.shape[0])
            if int(template_probs_float.shape[1]) < template_count:
                pad = template_probs_float.new_zeros(
                    b,
                    template_count - int(template_probs_float.shape[1]),
                )
                template_probs_float = torch.cat([template_probs_float, pad], dim=-1)
            elif int(template_probs_float.shape[1]) > template_count:
                template_probs_float = template_probs_float[:, :template_count]
                denom = template_probs_float.sum(dim=-1, keepdim=True).clamp_min(
                    1.0e-12
                )
                template_probs_float = template_probs_float / denom
            template_logits = torch.log(template_probs_float.clamp_min(1.0e-30)).to(
                dtype=reference.dtype
            )
            template_probs = template_probs_float.to(dtype=reference.dtype)
        else:
            template_logits = self.core_role_value_template_head(template_input)
            template_probs = torch.softmax(template_logits.float(), dim=-1).to(
                dtype=reference.dtype
            )
        steps = len(trajectory)
        table = self.core_role_value_template_table.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if steps <= int(table.shape[1]):
            step_table = table[:, :steps, :, :]
        else:
            pad = table[:, -1:, :, :].expand(
                -1,
                steps - int(table.shape[1]),
                -1,
                -1,
            )
            step_table = torch.cat([table, pad], dim=1)
        value_logits = torch.einsum("bt,tsrv->bsrv", template_probs, step_table)
        return {
            "template_logits": template_logits,
            "value_logits": value_logits,
        }

    def _compute_core_source_position_binder_context(
        self,
        prompt_context_seq: Optional[torch.Tensor],
        prompt_context_mask: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        empty = self._empty_core_role_value_state_logits(
            reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
        )
        if (
            disabled
            or prompt_context_seq is None
            or prompt_context_seq.ndim != 3
            or self.core_source_position_binder_input_proj is None
            or self.core_source_position_binder_position_embed is None
            or self.core_source_position_binder_slot_queries is None
            or self.core_source_position_binder_encoder is None
            or self.core_source_position_binder_head is None
        ):
            return empty, None
        hidden = self.core_source_position_binder_input_proj(
            prompt_context_seq.to(device=reference.device)
        )
        positions = torch.arange(
            hidden.shape[1],
            dtype=torch.long,
            device=hidden.device,
        ).clamp(max=self.core_source_position_binder_position_embed.num_embeddings - 1)
        hidden = hidden + self.core_source_position_binder_position_embed(
            positions
        ).unsqueeze(0)
        batch = int(hidden.shape[0])
        queries = self.core_source_position_binder_slot_queries.to(
            device=hidden.device,
            dtype=hidden.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        sequence = torch.cat([queries, hidden], dim=1)
        padding_mask = None
        if prompt_context_mask is not None:
            prompt_mask = prompt_context_mask.to(device=hidden.device).bool()
            query_mask = torch.ones(
                batch,
                int(queries.shape[1]),
                dtype=torch.bool,
                device=hidden.device,
            )
            padding_mask = ~torch.cat([query_mask, prompt_mask], dim=1)
        encoded = self.core_source_position_binder_encoder(
            sequence,
            src_key_padding_mask=padding_mask,
        )
        query_states = encoded[:, : int(queries.shape[1]), :]
        logits = self.core_source_position_binder_head(
            query_states
        )
        return logits.unsqueeze(1), query_states

    def _compute_core_source_position_binder_logits(
        self,
        prompt_context_seq: Optional[torch.Tensor],
        prompt_context_mask: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> torch.Tensor:
        logits, _query_states = self._compute_core_source_position_binder_context(
            prompt_context_seq,
            prompt_context_mask,
            reference=reference,
            disabled=disabled,
        )
        return logits

    def _compute_core_source_position_binder_state_delta(
        self,
        source_position_logits: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        empty = reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
        if (
            source_position_logits.ndim != 4
            or int(source_position_logits.shape[1]) == 0
            or self.core_source_position_binder_value_embed is None
            or self.core_source_position_binder_state_gate is None
        ):
            return empty
        logits = source_position_logits[:, 0, :, :]
        if int(logits.shape[1]) != int(reference.shape[1]):
            return empty
        vocab = min(
            int(logits.shape[-1]),
            int(self.core_source_position_binder_value_embed.num_embeddings),
        )
        if vocab <= 0:
            return empty
        probs = torch.softmax(logits[..., :vocab].float(), dim=-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if bool(self.cfg.core_source_position_binder_state_straight_through):
            hard_index = probs.argmax(dim=-1)
            hard = torch.nn.functional.one_hot(hard_index, num_classes=vocab).to(
                device=reference.device,
                dtype=reference.dtype,
            )
            probs = hard + probs - probs.detach()
        value_embed = self.core_source_position_binder_value_embed.weight[
            :vocab
        ].to(device=reference.device, dtype=reference.dtype)
        delta = probs @ value_embed
        gate = torch.sigmoid(
            self.core_source_position_binder_state_gate.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        gate_min = min(
            max(float(self.cfg.core_source_position_binder_state_gate_min), 0.0),
            1.0,
        )
        if gate_min > 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        return gate * delta

    def _compute_core_source_position_binder_query_state_delta(
        self,
        query_states: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> torch.Tensor:
        empty = reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
        if (
            disabled
            or query_states is None
            or query_states.ndim != 3
            or int(query_states.shape[1]) != int(reference.shape[1])
            or self.core_source_position_binder_query_state_proj is None
            or self.core_source_position_binder_query_state_gate is None
        ):
            return empty
        delta = self.core_source_position_binder_query_state_proj(
            query_states.to(device=reference.device, dtype=reference.dtype)
        )
        gate = torch.sigmoid(
            self.core_source_position_binder_query_state_gate.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        gate_min = min(
            max(float(self.cfg.core_source_position_binder_query_state_gate_min), 0.0),
            1.0,
        )
        if gate_min > 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        return gate * delta

    def _compute_core_source_value_binder_logits(
        self,
        query_states: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> torch.Tensor:
        empty = self._empty_core_role_value_state_logits(
            reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
        )
        if (
            disabled
            or query_states is None
            or query_states.ndim != 3
            or self.core_source_value_binder_head is None
        ):
            return empty
        logits = self.core_source_value_binder_head(
            query_states.to(device=reference.device)
        ).unsqueeze(1)
        if self.core_source_value_binder_logit_gate is not None:
            gate = torch.sigmoid(
                self.core_source_value_binder_logit_gate.to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
            )
            gate_min = min(
                max(float(self.cfg.core_source_value_binder_gate_min), 0.0),
                1.0,
            )
            if gate_min > 0.0:
                gate = gate_min + (1.0 - gate_min) * gate
            logits = gate * logits
        return logits

    def _compute_core_source_value_binder_state_delta(
        self,
        source_value_logits: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        empty = reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
        if (
            source_value_logits.ndim != 4
            or int(source_value_logits.shape[1]) == 0
            or int(source_value_logits.shape[2]) != int(reference.shape[1])
            or self.core_source_value_binder_value_embed is None
            or self.core_source_value_binder_state_gate is None
        ):
            return empty
        logits = source_value_logits[:, 0, :, :]
        vocab = min(
            int(logits.shape[-1]),
            int(self.core_source_value_binder_value_embed.num_embeddings),
        )
        if vocab <= 0:
            return empty
        probs = torch.softmax(logits[..., :vocab].float(), dim=-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if bool(self.cfg.core_source_value_binder_state_straight_through):
            hard_index = probs.argmax(dim=-1)
            hard = torch.nn.functional.one_hot(hard_index, num_classes=vocab).to(
                device=reference.device,
                dtype=reference.dtype,
            )
            probs = hard + probs - probs.detach()
        value_embed = self.core_source_value_binder_value_embed.weight[:vocab].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        delta = probs @ value_embed
        gate = torch.sigmoid(
            self.core_source_value_binder_state_gate.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        gate_min = min(
            max(float(self.cfg.core_source_value_binder_state_gate_min), 0.0),
            1.0,
        )
        if gate_min > 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        return gate * delta

    def _compute_core_role_value_state_logits(
        self,
        trajectory: list[torch.Tensor],
        *,
        role_token_start: Optional[int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.core_role_value_state_embed is None
            or self.core_role_value_state_norm is None
            or self.core_role_value_state_head is None
            or role_token_start is None
            or not trajectory
        ):
            return self._empty_core_role_value_state_logits(
                reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
            )
        role_count = int(self.core_role_value_state_embed.num_embeddings)
        role_end = int(role_token_start) + role_count
        role_states = []
        for state in trajectory:
            if int(state.shape[1]) < role_end:
                return self._empty_core_role_value_state_logits(
                    reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
                )
            role_states.append(state[:, int(role_token_start):role_end, :])
        stacked = torch.stack(role_states, dim=1)
        return self.core_role_value_state_head(
            self.core_role_value_state_norm(stacked)
        )

    def _compute_core_role_value_delta_logits(
        self,
        trajectory: list[torch.Tensor],
        *,
        role_token_start: Optional[int],
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        empty_gate = reference.new_empty((reference.shape[0], 0))
        if (
            disabled
            or self.core_role_value_state_embed is None
            or self.core_role_value_state_norm is None
            or self.core_role_value_state_head is None
            or self.core_role_value_delta_step_embed is None
            or self.core_role_value_delta_input_norm is None
            or self.core_role_value_delta_update is None
            or self.core_role_value_delta_gate is None
            or self.core_role_value_delta_output_norm is None
            or role_token_start is None
            or not trajectory
        ):
            return self._empty_core_role_value_state_logits(
                reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
            ), empty_gate
        role_count = int(self.core_role_value_state_embed.num_embeddings)
        role_start = int(role_token_start)
        role_end = role_start + role_count
        role_states = []
        for state in trajectory:
            if int(state.shape[1]) < role_end:
                return self._empty_core_role_value_state_logits(
                    reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
                ), empty_gate
            role_states.append(state[:, role_start:role_end, :])
        previous = role_states[0]
        outputs = []
        gate_means = []
        gate_min = float(self.cfg.core_role_value_delta_gate_min)
        for index, role_state in enumerate(role_states):
            step_index = min(
                int(index),
                int(self.core_role_value_delta_step_embed.num_embeddings) - 1,
            )
            step_id = torch.full(
                (role_state.shape[0],),
                step_index,
                device=role_state.device,
                dtype=torch.long,
            )
            step = self.core_role_value_delta_step_embed(step_id).view(
                role_state.shape[0],
                1,
                -1,
            )
            delta_input = previous + role_state + step.to(dtype=role_state.dtype)
            delta = self.core_role_value_delta_update(
                self.core_role_value_delta_input_norm(delta_input)
            )
            gate = torch.sigmoid(self.core_role_value_delta_gate(delta_input))
            if gate_min > 0.0:
                gate = gate_min + (1.0 - gate_min) * gate
            previous = role_state + gate * (previous + delta)
            corrected = self.core_role_value_delta_output_norm(previous)
            outputs.append(
                self.core_role_value_state_head(
                    self.core_role_value_state_norm(corrected)
                )
            )
            gate_means.append(gate.detach().float().mean(dim=(1, 2)))
        return torch.stack(outputs, dim=1), torch.stack(gate_means, dim=1)

    def _compute_core_value_delta_code_outputs(
        self,
        trajectory: list[torch.Tensor],
        *,
        role_token_start: Optional[int],
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty_gate = reference.new_empty((reference.shape[0], 0))
        empty_depth = reference.new_empty((reference.shape[0], 0, reference.shape[-1]))
        if (
            disabled
            or self.core_role_value_state_embed is None
            or self.core_role_value_state_norm is None
            or self.core_role_value_state_head is None
            or self.core_value_delta_code_input_norm is None
            or self.core_value_delta_code_head is None
            or self.core_value_delta_code_embed is None
            or self.core_value_delta_code_gate is None
            or role_token_start is None
            or not trajectory
        ):
            return (
                self._empty_core_value_delta_code_logits(empty_depth),
                self._empty_core_role_value_state_logits(empty_depth),
                empty_gate,
            )
        role_count = int(self.core_role_value_state_embed.num_embeddings)
        role_start = int(role_token_start)
        role_end = role_start + role_count
        role_states = []
        for state in trajectory:
            if int(state.shape[1]) < role_end:
                return (
                    self._empty_core_value_delta_code_logits(empty_depth),
                    self._empty_core_role_value_state_logits(empty_depth),
                    empty_gate,
                )
            role_states.append(state[:, role_start:role_end, :])

        previous = role_states[0]
        code_outputs = []
        role_outputs = []
        gate_means = []
        gate_min = float(self.cfg.core_value_delta_code_gate_min)
        codebook = int(self.core_value_delta_code_head.out_features)
        code_embed = self.core_value_delta_code_embed.weight.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        for role_state in role_states:
            code_input = role_state + previous
            code_logits = self.core_value_delta_code_head(
                self.core_value_delta_code_input_norm(code_input)
            )
            probs = torch.softmax(code_logits.float(), dim=-1).to(dtype=role_state.dtype)
            hard_index = torch.argmax(probs, dim=-1)
            hard = torch.nn.functional.one_hot(hard_index, num_classes=codebook).to(
                dtype=role_state.dtype,
                device=role_state.device,
            )
            straight_through = hard + probs - probs.detach()
            code_delta = torch.matmul(straight_through, code_embed)
            gate = torch.sigmoid(self.core_value_delta_code_gate(code_input))
            if gate_min > 0.0:
                gate = gate_min + (1.0 - gate_min) * gate
            corrected = role_state + gate * code_delta
            previous = corrected
            code_outputs.append(code_logits)
            role_outputs.append(
                self.core_role_value_state_head(
                    self.core_role_value_state_norm(corrected)
                )
            )
            gate_means.append(gate.detach().float().mean(dim=(1, 2)))
        return (
            torch.stack(code_outputs, dim=1),
            torch.stack(role_outputs, dim=1),
            torch.stack(gate_means, dim=1),
        )

    def _compute_core_typed_register_outputs(
        self,
        trajectory: list[torch.Tensor],
        *,
        role_token_start: Optional[int],
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        empty = self._empty_core_typed_register_outputs(reference)
        if (
            disabled
            or self.core_role_value_state_embed is None
            or self.core_typed_register_context_norm is None
            or self.core_typed_register_role_norm is None
            or self.core_typed_register_operation_head is None
            or self.core_typed_register_operation_embed is None
            or self.core_typed_register_input_norm is None
            or self.core_typed_register_update is None
            or self.core_typed_register_gate is None
            or self.core_typed_register_output_norm is None
            or self.core_typed_register_value_head is None
            or self.core_typed_register_transition_head is None
            or role_token_start is None
            or not trajectory
        ):
            return empty
        role_count = int(self.core_role_value_state_embed.num_embeddings)
        role_start = int(role_token_start)
        role_end = role_start + role_count
        role_states = []
        context_states = []
        for state in trajectory:
            if int(state.shape[1]) < role_end:
                return empty
            role_states.append(state[:, role_start:role_end, :])
            if role_start > 0:
                context_states.append(state[:, :role_start, :].mean(dim=1))
            else:
                context_states.append(state[:, 0, :])

        registers = role_states[0]
        operation_outputs = []
        value_outputs = []
        transition_outputs = []
        gate_means = []
        gate_min = float(self.cfg.core_typed_register_gate_min)
        operation_embed = self.core_typed_register_operation_embed.weight.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        for role_state, context_state in zip(role_states, context_states):
            role_summary = self.core_typed_register_role_norm(role_state).mean(dim=1)
            operation_input = self.core_typed_register_context_norm(
                context_state + role_summary
            )
            operation_logits = self.core_typed_register_operation_head(operation_input)
            operation_probs = torch.softmax(operation_logits.float(), dim=-1).to(
                dtype=role_state.dtype
            )
            operation_state = torch.matmul(operation_probs, operation_embed).unsqueeze(1)
            register_input = registers + role_state + operation_state
            delta = self.core_typed_register_update(
                self.core_typed_register_input_norm(register_input)
            )
            gate = torch.sigmoid(self.core_typed_register_gate(register_input))
            if gate_min > 0.0:
                gate = gate_min + (1.0 - gate_min) * gate
            registers = self.core_typed_register_output_norm(registers + gate * delta)
            value_logits = self.core_typed_register_value_head(registers)
            if (
                self.core_typed_register_value_feedback_embed is not None
                and self.core_typed_register_value_feedback_norm is not None
                and self.core_typed_register_value_feedback_gate is not None
                and self.core_typed_register_value_feedback_output_norm is not None
            ):
                value_probs = torch.softmax(value_logits.float(), dim=-1).to(
                    dtype=registers.dtype
                )
                value_embed = (
                    self.core_typed_register_value_feedback_embed.weight.to(
                        device=registers.device,
                        dtype=registers.dtype,
                    )
                )
                value_state = value_probs @ value_embed
                value_gate = torch.sigmoid(
                    self.core_typed_register_value_feedback_gate(
                        self.core_typed_register_value_feedback_norm(registers)
                    )
                ).to(dtype=registers.dtype)
                value_gate_min = min(
                    max(float(self.cfg.core_typed_register_value_feedback_gate_min), 0.0),
                    1.0,
                )
                if value_gate_min > 0.0:
                    value_gate = value_gate_min + (1.0 - value_gate_min) * value_gate
                registers = self.core_typed_register_value_feedback_output_norm(
                    registers + value_gate * value_state
                )
                value_logits = self.core_typed_register_value_head(registers)
            operation_outputs.append(operation_logits)
            value_outputs.append(value_logits)
            transition_outputs.append(
                self.core_typed_register_transition_head(registers)
            )
            gate_means.append(gate.detach().float().mean(dim=(1, 2)))
        transition_logits = (
            torch.stack(transition_outputs[:-1], dim=1)
            if len(transition_outputs) > 1
            else empty["transition_logits"]
        )
        value_logits = torch.stack(value_outputs, dim=1)
        if (
            bool(self.cfg.core_typed_register_transition_readout_enabled)
            and int(transition_logits.shape[1]) > 0
        ):
            value_logits = torch.cat(
                [value_logits[:, :1, :, :], transition_logits],
                dim=1,
            )
        return {
            "operation_logits": torch.stack(operation_outputs, dim=1),
            "value_logits": value_logits,
            "transition_logits": transition_logits,
            "gate_mean": torch.stack(gate_means, dim=1),
        }

    def _compute_core_role_value_transition_logits(
        self,
        trajectory: list[torch.Tensor],
        *,
        role_token_start: Optional[int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.core_role_value_state_embed is None
            or self.core_role_value_transition_input_norm is None
            or self.core_role_value_transition_update is None
            or self.core_role_value_transition_output_norm is None
            or self.core_role_value_transition_head is None
            or role_token_start is None
            or len(trajectory) < 2
        ):
            return self._empty_core_role_value_transition_logits(reference)
        role_count = int(self.core_role_value_state_embed.num_embeddings)
        role_start = int(role_token_start)
        role_end = role_start + role_count
        role_states = []
        context_states = []
        for state in trajectory[:-1]:
            if int(state.shape[1]) < role_end:
                return self._empty_core_role_value_transition_logits(reference)
            role_states.append(state[:, role_start:role_end, :])
            if role_start > 0:
                context_states.append(state[:, :role_start, :].mean(dim=1))
            else:
                context_states.append(state[:, 0, :])
        previous_roles = torch.stack(role_states, dim=1)
        previous_context = torch.stack(context_states, dim=1).unsqueeze(2)
        source = self.core_role_value_transition_input_norm(
            previous_roles + previous_context
        )
        updated = source + self.core_role_value_transition_update(source)
        return self.core_role_value_transition_head(
            self.core_role_value_transition_output_norm(updated)
        )

    def _compute_core_primitive_role_value_update_logits(
        self,
        hidden: torch.Tensor,
        *,
        operation_probs: Optional[torch.Tensor] = None,
        role_count: int,
    ) -> torch.Tensor:
        if (
            self.core_primitive_role_value_operation_heads is not None
            and operation_probs is not None
            and operation_probs.ndim == 2
            and int(operation_probs.shape[1]) > 0
        ):
            op_count = min(
                int(operation_probs.shape[1]),
                len(self.core_primitive_role_value_operation_heads),
            )
            per_op_logits = torch.stack(
                [
                    self.core_primitive_role_value_operation_heads[index](hidden)
                    for index in range(op_count)
                ],
                dim=1,
            )
            op = operation_probs[:, :op_count].to(
                device=per_op_logits.device,
                dtype=per_op_logits.dtype,
            )
            return torch.einsum("bo,borv->brv", op, per_op_logits)
        if (
            self.core_primitive_role_value_list_head is None
            or self.core_primitive_role_value_scalar_head is None
            or self.core_primitive_role_value_head is None
        ):
            return self.core_primitive_role_value_head(hidden)
        max_list_fields = max(1, (int(role_count) - 2) // 2)
        scalar_start = min(int(role_count), 2 * max_list_fields)
        scalar_end = min(int(role_count), scalar_start + 2)
        chunks = []
        if scalar_start > 0:
            chunks.append(
                self.core_primitive_role_value_list_head(
                    hidden[:, :scalar_start, :]
                )
            )
        if scalar_end > scalar_start:
            chunks.append(
                self.core_primitive_role_value_scalar_head(
                    hidden[:, scalar_start:scalar_end, :]
                )
            )
        if scalar_end < int(role_count):
            chunks.append(self.core_primitive_role_value_head(hidden[:, scalar_end:, :]))
        if not chunks:
            return self.core_primitive_role_value_head(hidden)
        return torch.cat(chunks, dim=1)

    def _compute_core_primitive_role_value_state_logits(
        self,
        primitive_transition_info: dict[str, torch.Tensor],
        *,
        prompt_logits: torch.Tensor,
        source_value_logits: Optional[torch.Tensor] = None,
        fallback_logits: torch.Tensor,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        reference: torch.Tensor,
        disabled: bool = False,
        disable_prompt_context: bool = False,
    ) -> torch.Tensor:
        operation_logits = primitive_transition_info.get("operation_logits")
        prompt_context = primitive_transition_info.get("prompt_context")
        if bool(disable_prompt_context):
            prompt_context = None
            prompt_context_seq = None
            prompt_context_mask = None
        role_count = max(0, int(self.cfg.core_role_value_state_num_roles))
        vocab = max(1, int(self.cfg.core_role_value_state_vocab_size or 1))
        b = int(reference.shape[0])
        empty = reference.new_empty((b, 0, role_count, vocab))
        empty_gate = reference.new_empty((b, 0, role_count))
        if (
            disabled
            or operation_logits is None
            or operation_logits.ndim != 3
            or int(operation_logits.shape[1]) == 0
        ):
            return empty, empty_gate

        steps = int(operation_logits.shape[1])
        if prompt_logits.ndim == 4 and int(prompt_logits.shape[1]) > 0:
            current_logits = prompt_logits[:, 0, :, :]
        elif fallback_logits.ndim == 4 and int(fallback_logits.shape[1]) > 0:
            current_logits = fallback_logits[:, 0, :, :]
        else:
            current_logits = reference.new_zeros((b, role_count, vocab))
        if tuple(current_logits.shape[1:]) != (role_count, vocab):
            return empty, empty_gate

        if (
            self.core_primitive_role_value_value_embed is not None
            and self.core_primitive_role_value_operation_embed is not None
            and self.core_primitive_role_value_input_norm is not None
            and self.core_primitive_role_value_update is not None
            and self.core_primitive_role_value_output_norm is not None
            and self.core_primitive_role_value_head is not None
            and self.core_role_value_state_embed is not None
        ):
            value_embed = self.core_primitive_role_value_value_embed.weight.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            operation_embed = (
                self.core_primitive_role_value_operation_embed.weight.to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
            )
            role_embed = self.core_role_value_state_embed.weight.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            source_value_state = None
            if (
                self.core_primitive_role_value_source_value_gate is not None
                and source_value_logits is not None
                and source_value_logits.ndim == 4
                and int(source_value_logits.shape[1]) > 0
                and tuple(source_value_logits.shape[2:]) == (role_count, vocab)
            ):
                source_value_probs = torch.softmax(
                    source_value_logits[:, 0, :, :].float(),
                    dim=-1,
                ).to(device=reference.device, dtype=reference.dtype)
                source_value_state = torch.matmul(source_value_probs, value_embed)
                source_value_gate = torch.sigmoid(
                    self.core_primitive_role_value_source_value_gate.to(
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                )
                gate_min = min(
                    max(
                        float(
                            self.cfg.core_primitive_role_value_source_value_gate_min
                        ),
                        0.0,
                    ),
                    1.0,
                )
                if gate_min > 0.0:
                    source_value_gate = gate_min + (
                        1.0 - gate_min
                    ) * source_value_gate
                source_value_state = source_value_gate * source_value_state
            outputs = []
            gate_outputs = []
            for step in range(steps):
                state_probs = torch.softmax(current_logits.float(), dim=-1).to(
                    dtype=reference.dtype
                )
                op_probs = torch.softmax(
                    operation_logits[:, step, :].float(),
                    dim=-1,
                ).to(dtype=reference.dtype)
                value_state = torch.matmul(state_probs, value_embed)
                operation_state = torch.matmul(op_probs, operation_embed).unsqueeze(1)
                hidden = value_state + operation_state + role_embed.unsqueeze(0)
                if source_value_state is not None:
                    hidden = hidden + source_value_state
                if (
                    prompt_context_seq is not None
                    and prompt_context_seq.ndim == 3
                    and self.core_primitive_role_value_prompt_query_norm is not None
                    and self.core_primitive_role_value_prompt_token_context_norm is not None
                    and self.core_primitive_role_value_prompt_cross is not None
                    and self.core_primitive_role_value_prompt_token_output_norm is not None
                ):
                    prompt_tokens = prompt_context_seq.to(
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                    prompt_token_delta = self.core_primitive_role_value_prompt_cross(
                        self.core_primitive_role_value_prompt_query_norm(hidden),
                        self.core_primitive_role_value_prompt_token_context_norm(
                            prompt_tokens
                        ),
                        prompt_context_mask,
                    )
                    hidden = self.core_primitive_role_value_prompt_token_output_norm(
                        hidden + prompt_token_delta.to(dtype=reference.dtype)
                    )
                if (
                    prompt_context is not None
                    and prompt_context.ndim == 3
                    and int(prompt_context.shape[1]) > step
                    and self.core_primitive_role_value_prompt_context_norm is not None
                    and self.core_primitive_role_value_prompt_context_adapter is not None
                ):
                    prompt_state = prompt_context[:, step, :].to(dtype=reference.dtype)
                    prompt_state = self.core_primitive_role_value_prompt_context_norm(
                        prompt_state
                    )
                    prompt_state = self.core_primitive_role_value_prompt_context_adapter(
                        prompt_state
                    )
                    hidden = hidden + prompt_state.to(dtype=reference.dtype).unsqueeze(1)
                if (
                    self.core_primitive_role_value_role_mixer is not None
                    and self.core_primitive_role_value_role_mixer_norm is not None
                ):
                    mixer_input = self.core_primitive_role_value_role_mixer_norm(
                        hidden
                    )
                    mixed, _ = self.core_primitive_role_value_role_mixer(
                        mixer_input,
                        mixer_input,
                        mixer_input,
                        need_weights=False,
                    )
                    hidden = hidden + mixed
                hidden = hidden + self.core_primitive_role_value_update(
                    self.core_primitive_role_value_input_norm(hidden)
                )
                hidden = self.core_primitive_role_value_output_norm(hidden)
                update_logits = self._compute_core_primitive_role_value_update_logits(
                    hidden,
                    operation_probs=op_probs,
                    role_count=role_count,
                )
                residual_delta_enabled = bool(
                    self.cfg.core_primitive_role_value_residual_delta_enabled
                )
                next_logits = (
                    current_logits + update_logits
                    if residual_delta_enabled
                    else update_logits
                )
                if self.core_primitive_role_value_update_gate is not None:
                    update_gate = torch.sigmoid(
                        self.core_primitive_role_value_update_gate(hidden)
                    ).to(dtype=reference.dtype)
                    gate_min = min(
                        max(
                            float(
                                self.cfg.core_primitive_role_value_update_gate_min
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                    if gate_min > 0.0:
                        update_gate = gate_min + (1.0 - gate_min) * update_gate
                    next_logits = (
                        current_logits + update_gate * update_logits
                        if residual_delta_enabled
                        else current_logits
                        + update_gate * (next_logits - current_logits)
                    )
                    gate_outputs.append(update_gate.squeeze(-1))
                current_logits = next_logits
                outputs.append(current_logits)
            update_gate_tensor = (
                torch.stack(gate_outputs, dim=1) if gate_outputs else empty_gate
            )
            return torch.stack(outputs, dim=1), update_gate_tensor

        if (
            self.core_primitive_role_value_source_mix is None
            or self.core_primitive_role_value_value_transition is None
            or self.core_primitive_role_value_bias is None
        ):
            return empty, empty_gate

        source_mix = torch.softmax(
            self.core_primitive_role_value_source_mix.float(),
            dim=-1,
        ).to(device=reference.device, dtype=reference.dtype)
        value_transition = self.core_primitive_role_value_value_transition.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        bias = self.core_primitive_role_value_bias.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        outputs = []
        for step in range(steps):
            state_probs = torch.softmax(current_logits.float(), dim=-1).to(
                dtype=reference.dtype
            )
            op_probs = torch.softmax(operation_logits[:, step, :].float(), dim=-1).to(
                dtype=reference.dtype
            )
            mixed = torch.einsum("bsv,ots->botv", state_probs, source_mix)
            per_operation_logits = (
                torch.einsum("botu,otuv->botv", mixed, value_transition) + bias
            )
            current_logits = torch.einsum(
                "bo,botv->btv",
                op_probs,
                per_operation_logits,
            )
            outputs.append(current_logits)
        return torch.stack(outputs, dim=1), empty_gate

    def _compute_core_primitive_typed_selector_outputs(
        self,
        *,
        primitive_logits: torch.Tensor,
        typed_logits: torch.Tensor,
        reference: torch.Tensor,
        disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = int(reference.shape[0])
        empty_logits = reference.new_empty((b, 0, 0, 0))
        empty_gate_full = reference.new_empty((b, 0, 0))
        empty_gate = reference.new_empty((b, 0))
        if (
            disabled
            or self.core_primitive_typed_selector is None
            or primitive_logits.ndim != 4
            or typed_logits.ndim != 4
        ):
            return {
                "selected_logits": empty_logits,
                "gate": empty_gate_full,
                "gate_mean": empty_gate,
            }
        steps = min(int(primitive_logits.shape[1]), int(typed_logits.shape[1]))
        roles = min(int(primitive_logits.shape[2]), int(typed_logits.shape[2]))
        vocab = min(int(primitive_logits.shape[3]), int(typed_logits.shape[3]))
        if steps <= 0 or roles <= 0 or vocab <= 1:
            return {
                "selected_logits": empty_logits,
                "gate": empty_gate_full,
                "gate_mean": empty_gate,
            }
        primitive = primitive_logits[:, :steps, :roles, :vocab]
        typed = typed_logits[:, :steps, :roles, :vocab]
        primitive_top2 = primitive.float().topk(k=2, dim=-1).values
        typed_top2 = typed.float().topk(k=2, dim=-1).values
        primitive_margin = primitive_top2[..., 0] - primitive_top2[..., 1]
        typed_margin = typed_top2[..., 0] - typed_top2[..., 1]
        features = torch.stack(
            [
                primitive_margin,
                typed_margin,
                primitive_margin - typed_margin,
                primitive_top2[..., 0],
                typed_top2[..., 0],
            ],
            dim=-1,
        ).to(device=reference.device, dtype=reference.dtype)
        gate = torch.sigmoid(self.core_primitive_typed_selector(features)).to(
            dtype=reference.dtype
        )
        selected = typed + gate * (primitive - typed)
        return {
            "selected_logits": selected,
            "gate": gate.squeeze(-1),
            "gate_mean": gate.detach().float().mean(dim=(2, 3)),
        }

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
            prompt_context = core_depth_states.new_zeros((b, steps, self.cfg.d_model))
            return {"operation_logits": logits, "prompt_context": prompt_context}
        features = self.primitive_transition_norm(core_depth_states)
        prompt_context = None
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
                prompt_context = core_depth_states.new_zeros(
                    (b, steps, self.cfg.d_model)
                )
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
        if prompt_context is None:
            prompt_context = core_depth_states.new_zeros((b, steps, self.cfg.d_model))
        return {"operation_logits": logits, "prompt_context": prompt_context}

    def _compute_transition_phase_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if self.transition_phase_norm is None or self.transition_phase_head is None:
            return core_depth_states.new_empty((b, 0, 0))
        num_classes = int(self.transition_phase_head[-1].out_features)
        if disabled or steps == 0:
            return core_depth_states.new_zeros((b, steps, num_classes))
        features = self.transition_phase_norm(core_depth_states)
        if self.transition_phase_prompt_norm is not None:
            if prompt_context_seq is None:
                prompt_mean = core_depth_states.new_zeros((b, self.cfg.d_model))
            elif prompt_context_mask is None:
                prompt_mean = prompt_context_seq.mean(dim=1)
            else:
                prompt_mask = prompt_context_mask.to(
                    device=prompt_context_seq.device,
                    dtype=prompt_context_seq.dtype,
                ).unsqueeze(-1)
                denom = prompt_mask.sum(dim=1).clamp_min(1.0)
                prompt_mean = (prompt_context_seq * prompt_mask).sum(dim=1) / denom
            prompt_mean = self.transition_phase_prompt_norm(prompt_mean)
            prompt_mean = prompt_mean.unsqueeze(1).expand(-1, steps, -1)
            if (
                self.transition_phase_prompt_cross is not None
                and self.transition_phase_prompt_query_norm is not None
                and self.transition_phase_prompt_context_norm is not None
                and prompt_context_seq is not None
            ):
                prompt_token_context = self.transition_phase_prompt_cross(
                    self.transition_phase_prompt_query_norm(core_depth_states),
                    self.transition_phase_prompt_context_norm(prompt_context_seq),
                    prompt_context_mask,
                )
                prompt_token_context = self.transition_phase_prompt_norm(
                    prompt_token_context
                )
                features = torch.cat([features, prompt_mean, prompt_token_context], dim=-1)
            else:
                features = torch.cat([features, prompt_mean], dim=-1)
            if (
                self.transition_phase_global_query is not None
                and self.transition_phase_global_query_norm is not None
                and self.transition_phase_global_context_norm is not None
                and self.transition_phase_global_cross is not None
                and self.transition_phase_global_cross_norm is not None
                and prompt_context_seq is not None
            ):
                global_query = self.transition_phase_global_query.expand(b, -1, -1)
                global_context = self.transition_phase_global_cross(
                    self.transition_phase_global_query_norm(global_query),
                    self.transition_phase_global_context_norm(prompt_context_seq),
                    prompt_context_mask,
                )
                global_context = self.transition_phase_global_cross_norm(global_context)
                global_context = global_context.expand(-1, steps, -1)
            elif self.transition_phase_global_query is not None:
                global_context = core_depth_states.new_zeros((b, steps, self.cfg.d_model))
            else:
                global_context = None
            if global_context is not None:
                features = torch.cat([features, global_context], dim=-1)
        return self.transition_phase_head(features)

    def _compute_transition_source_router_logits(
        self,
        core_depth_states: torch.Tensor,
        *,
        prompt_context_seq: Optional[torch.Tensor] = None,
        prompt_context_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        steps = core_depth_states.shape[1]
        if (
            self.transition_source_router_norm is None
            or self.transition_source_router_head is None
        ):
            return core_depth_states.new_empty((b, 0, 0))
        if disabled or steps == 0:
            return core_depth_states.new_zeros((b, steps, 2))
        features = self.transition_source_router_norm(core_depth_states)
        if self.transition_source_router_prompt_norm is not None:
            if prompt_context_seq is None:
                prompt_mean = core_depth_states.new_zeros((b, self.cfg.d_model))
            elif prompt_context_mask is None:
                prompt_mean = prompt_context_seq.mean(dim=1)
            else:
                prompt_mask = prompt_context_mask.to(
                    device=prompt_context_seq.device,
                    dtype=prompt_context_seq.dtype,
                ).unsqueeze(-1)
                denom = prompt_mask.sum(dim=1).clamp_min(1.0)
                prompt_mean = (prompt_context_seq * prompt_mask).sum(dim=1) / denom
            prompt_mean = self.transition_source_router_prompt_norm(prompt_mean)
            prompt_mean = prompt_mean.unsqueeze(1).expand(-1, steps, -1)
            if (
                self.transition_source_router_prompt_cross is not None
                and self.transition_source_router_prompt_query_norm is not None
                and self.transition_source_router_prompt_context_norm is not None
                and prompt_context_seq is not None
            ):
                prompt_token_context = self.transition_source_router_prompt_cross(
                    self.transition_source_router_prompt_query_norm(core_depth_states),
                    self.transition_source_router_prompt_context_norm(prompt_context_seq),
                    prompt_context_mask,
                )
                prompt_token_context = self.transition_source_router_prompt_norm(
                    prompt_token_context
                )
                features = torch.cat([features, prompt_mean, prompt_token_context], dim=-1)
            else:
                features = torch.cat([features, prompt_mean], dim=-1)
        return self.transition_source_router_head(features)

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

    def _core_depth_states(self, trajectory: list[torch.Tensor], workspace: torch.Tensor) -> torch.Tensor:
        if not trajectory:
            return QTRMMultimodalModel._empty_core_depth_states(workspace)
        if (
            self.core_depth_readout_query is not None
            and self.core_depth_readout_query_norm is not None
            and self.core_depth_readout_state_norm is not None
            and self.core_depth_readout_cross is not None
            and self.core_depth_readout_output_norm is not None
        ):
            readouts = []
            for state in trajectory:
                query = self.core_depth_readout_query.to(
                    device=state.device,
                    dtype=state.dtype,
                ).expand(state.shape[0], -1, -1)
                readout = self.core_depth_readout_cross(
                    self.core_depth_readout_query_norm(query),
                    self.core_depth_readout_state_norm(state),
                )
                readout = self.core_depth_readout_output_norm(query + readout)
                readouts.append(readout.squeeze(1))
            return torch.stack(readouts, dim=1)
        return torch.stack([state[:, 0, :] for state in trajectory], dim=1)

    @staticmethod
    def _controller_signal_trajectory_features(
        core_depth_states: torch.Tensor,
        *,
        target_steps: int,
        d_model: int,
    ) -> torch.Tensor:
        b = core_depth_states.shape[0]
        target_steps = max(1, int(target_steps))
        d_model = int(d_model)
        if core_depth_states.ndim != 3 or core_depth_states.shape[-1] != d_model:
            return core_depth_states.new_zeros((b, target_steps * d_model))
        steps = int(core_depth_states.shape[1])
        if steps >= target_steps:
            selected = core_depth_states[:, :target_steps, :]
        else:
            pad = core_depth_states.new_zeros((b, target_steps - steps, d_model))
            selected = torch.cat([core_depth_states, pad], dim=1)
        return selected.reshape(b, target_steps * d_model)

    def _core_depth_last_logits(
        self,
        trajectory: list[torch.Tensor],
        *,
        text_context_seq: torch.Tensor,
        text_context_mask: torch.Tensor,
        workspace_mask: torch.Tensor,
        transition_state_features: Optional[torch.Tensor] = None,
        transition_state_code_embeddings: Optional[torch.Tensor] = None,
        transition_state_joint_answer_embeddings: Optional[torch.Tensor] = None,
        typed_algorithmic_answer_tokens: Optional[torch.Tensor] = None,
        core_role_value_answer_tokens: Optional[torch.Tensor] = None,
        disable_answer_state_loop_selective_context: bool = False,
        force_answer_state_loop_dense_context: bool = False,
        disable_answer_state_loop_hidden_bridge: bool = False,
        disable_answer_state_loop_talker: bool = False,
    ) -> torch.Tensor:
        if not trajectory:
            return self._empty_core_depth_last_logits(text_context_seq)
        if self.answer_state_loop_cross is not None:
            depth_logits = []
            for state_prefix in range(1, len(trajectory) + 1):
                prefix_logits, *_ = self._compute_answer_state_loop_outputs(
                    text_context_seq,
                    trajectory=trajectory[:state_prefix],
                    text_context_mask=text_context_mask,
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
                    transition_state_joint_answer_embeddings=(
                        transition_state_joint_answer_embeddings[:, :state_prefix, :]
                        if transition_state_joint_answer_embeddings is not None
                        and transition_state_joint_answer_embeddings.numel() != 0
                        else transition_state_joint_answer_embeddings
                    ),
                    typed_algorithmic_answer_tokens=(
                        typed_algorithmic_answer_tokens[:, :state_prefix, :, :]
                        if typed_algorithmic_answer_tokens is not None
                        and typed_algorithmic_answer_tokens.numel() != 0
                        else typed_algorithmic_answer_tokens
                    ),
                    core_role_value_answer_tokens=(
                        core_role_value_answer_tokens[:, :state_prefix, :, :]
                        if core_role_value_answer_tokens is not None
                        and core_role_value_answer_tokens.numel() != 0
                        else core_role_value_answer_tokens
                    ),
                    disable_selective_context=bool(
                        disable_answer_state_loop_selective_context
                    ),
                    force_dense_context=bool(
                        force_answer_state_loop_dense_context
                    ),
                    disable_hidden_bridge=bool(
                        disable_answer_state_loop_hidden_bridge
                    ),
                    disable_talker=bool(disable_answer_state_loop_talker),
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
        transition_state_joint_answer_embeddings: Optional[torch.Tensor] = None,
        typed_algorithmic_answer_tokens: Optional[torch.Tensor] = None,
        core_role_value_answer_tokens: Optional[torch.Tensor] = None,
        disable_answer_state_loop_selective_context: bool = False,
        force_answer_state_loop_dense_context: bool = False,
        disable_answer_state_loop_hidden_bridge: bool = False,
        disable_answer_state_loop_talker: bool = False,
    ) -> torch.Tensor:
        if not trajectory:
            return self._empty_core_depth_text_logits(text_context_seq, input_seq_len)
        if self.answer_state_loop_cross is not None:
            _, _, depth_hidden, *_ = self._compute_answer_state_loop_outputs(
                text_context_seq,
                trajectory=trajectory,
                text_context_mask=text_context_mask,
                workspace_mask=workspace_mask,
                input_seq_len=input_seq_len,
                transition_state_features=transition_state_features,
                transition_state_code_embeddings=transition_state_code_embeddings,
                transition_state_joint_answer_embeddings=(
                    transition_state_joint_answer_embeddings
                ),
                typed_algorithmic_answer_tokens=typed_algorithmic_answer_tokens,
                core_role_value_answer_tokens=core_role_value_answer_tokens,
                disable_selective_context=bool(
                    disable_answer_state_loop_selective_context
                ),
                force_dense_context=bool(force_answer_state_loop_dense_context),
                disable_hidden_bridge=bool(disable_answer_state_loop_hidden_bridge),
                disable_talker=bool(disable_answer_state_loop_talker),
            )
            return self._answer_state_loop_lm_logits(
                depth_hidden,
                disable_hidden_bridge=bool(disable_answer_state_loop_hidden_bridge),
            )
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
