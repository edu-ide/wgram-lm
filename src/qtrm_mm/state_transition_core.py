"""
Strict TRM (Tiny Recursive Model) Dual-State Core.

Implements the authentic TRM philosophy:
- Dual Latent States: z_H and z_L.
- Shared Recurrence: THE SAME module is used for both L-level and H-level updates.
- Minimalist Design: Information flows through a single shared reasoning engine.

Architecture:
    z_L = Shared_Core(z_L, z_H, op_t)
    z_H = Shared_Core(z_H, z_L, None)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .config import QTRMConfig
from .norm import RMSNorm
from .mixers import FLADeltaMixer

# Operation constants
N_OPERATIONS = 4  # ADD, MUL, SUB, FINAL
OP_ADD = 0
OP_MUL = 1
OP_SUB = 2
OP_FINAL = 3


@dataclass
class StateTransitionOutput:
    """Output of the state transition core."""
    state_trajectory: torch.Tensor  # (B, T+1, d_state)
    state_digit_logits: torch.Tensor  # (B, T+1, 10)
    answer_logits: torch.Tensor  # (B, 10)
    operation_logits: Optional[torch.Tensor]  # (B, T, n_ops)
    state_norms: torch.Tensor  # (B, T+1)
    transition_norms: torch.Tensor  # (B, T)
    state_cosines: torch.Tensor  # (B, T)
    stochastic_mu_norms: Optional[torch.Tensor] = None  # (T,)
    stochastic_std_means: Optional[torch.Tensor] = None  # (T,)
    stochastic_noise_norms: Optional[torch.Tensor] = None  # (T,)
    stochastic_posterior_kls: Optional[torch.Tensor] = None  # (T,)
    working_register_norms: Optional[torch.Tensor] = None  # (T,)
    working_register_gate_means: Optional[torch.Tensor] = None  # (T,)
    working_register_role_cosines: Optional[torch.Tensor] = None  # (T,)
    working_register_trajectory: Optional[torch.Tensor] = None  # (B, T+1, R, D)
    typed_value_register_norms: Optional[torch.Tensor] = None  # (T,)
    typed_value_register_gate_means: Optional[torch.Tensor] = None  # (T,)
    typed_value_register_trajectory: Optional[torch.Tensor] = None  # (B, T+1, V, D)
    typed_digit_register_norms: Optional[torch.Tensor] = None  # (T,)
    typed_digit_register_gate_means: Optional[torch.Tensor] = None  # (T,)
    typed_digit_register_trajectory: Optional[torch.Tensor] = None  # (B, T+1, V*(digits+1), D)
    semantic_token_feedback_gate_means: Optional[torch.Tensor] = None  # (T,)
    semantic_token_feedback_entropies: Optional[torch.Tensor] = None  # (T,)


class SharedReasoningCore(nn.Module):
    """
    The shared core module used for both z_L and z_H updates.
    
    In TRM, weight sharing is the key to efficient and robust reasoning.
    """

    def __init__(
        self,
        d_state: int,
        n_operations: int,
        hidden_dim: Optional[int] = None,
        transition_scale_init: float = 1.0,
        layerscale: bool = False,
        layerscale_init: float = 1e-5,
        gate_type: str = "tanh",
        gate_bias_init: float = 0.5,
    ):
        super().__init__()
        self.d_state = d_state
        self.n_operations = n_operations
        self.hidden_dim = hidden_dim or d_state * 4
        self.transition_scale = nn.Parameter(torch.tensor(float(transition_scale_init)))
        self.layerscale = layerscale
        self.layerscale_init = layerscale_init
        self.gate_type = gate_type
        self.gate_bias_init = gate_bias_init

        # Shared projection for the inputs [state_main, state_side, op_vec]
        # In TRM, op_vec might be zero when updating z_H
        self.input_proj = nn.Linear(d_state * 3, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, d_state)

        # Gating mechanism
        self.gate = nn.Linear(d_state, 1)

        # LayerScale parameter
        if self.layerscale:
            self.layerscale_gamma = nn.Parameter(torch.full((d_state,), float(layerscale_init)))

        # Normalization
        self.norm = RMSNorm(d_state)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.constant_(self.gate.bias, self.gate_bias_init)

    def forward(
        self,
        z_main: torch.Tensor,
        z_side: torch.Tensor,
        op_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generic transition step.
        If updating z_L: z_main=z_L, z_side=z_H, op_vec=op_t
        If updating z_H: z_main=z_H, z_side=z_L, op_vec=0
        """
        combined = torch.cat([z_main, z_side, op_vec], dim=-1)
        
        # Reasoning step
        hidden = torch.nn.functional.gelu(self.input_proj(combined))
        delta = self.output_proj(hidden) * self.transition_scale.to(hidden.dtype)
        
        # Gating
        if self.gate_type == "sigmoid":
            g = torch.sigmoid(self.gate(z_main))
        else:
            g = torch.tanh(self.gate(z_main))
        
        # Residual update with normalization
        if self.layerscale:
            update = (delta.to(z_main.dtype) * g) * self.layerscale_gamma.to(z_main.dtype)
        else:
            update = delta.to(z_main.dtype) * g
        z_next = self.norm(z_main + update)
        
        return z_next


class MiniGatedDeltaReasoningCore(nn.Module):
    """Identity-biased delta-memory update for the shared recurrent core.

    This is a drop-in replacement for SharedReasoningCore. It keeps the TRM
    dual-state call contract, but replaces the MLP residual transition with a
    bounded gated delta update inspired by delta-rule recurrent memories.
    """

    def __init__(
        self,
        d_state: int,
        n_operations: int,
        hidden_dim: Optional[int] = None,
        transition_scale_init: float = 1.0,
        layerscale: bool = False,
        layerscale_init: float = 1e-5,
        gate_type: str = "tanh",
        gate_bias_init: float = 0.5,
    ):
        super().__init__()
        self.d_state = d_state
        self.n_operations = n_operations
        self.hidden_dim = hidden_dim or d_state * 2
        self.transition_scale = nn.Parameter(torch.tensor(float(transition_scale_init)))
        self.layerscale = layerscale
        self.layerscale_init = layerscale_init

        self.input_proj = nn.Linear(d_state * 3, self.hidden_dim)
        self.q_proj = nn.Linear(d_state, d_state)
        self.k_proj = nn.Linear(self.hidden_dim, d_state)
        self.v_proj = nn.Linear(self.hidden_dim, d_state)
        self.beta_proj = nn.Linear(self.hidden_dim, d_state)
        self.decay_proj = nn.Linear(self.hidden_dim, d_state)
        self.gate_proj = nn.Linear(self.hidden_dim, d_state)

        if self.layerscale:
            self.layerscale_gamma = nn.Parameter(torch.full((d_state,), float(layerscale_init)))

        self.norm = RMSNorm(d_state)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, -1.0)
        nn.init.xavier_uniform_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, 2.0)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(
        self,
        z_main: torch.Tensor,
        z_side: torch.Tensor,
        op_vec: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([z_main, z_side, op_vec], dim=-1)
        hidden = torch.nn.functional.gelu(self.input_proj(combined))

        query = torch.tanh(self.q_proj(z_main))
        key = torch.sigmoid(self.k_proj(hidden))
        value = torch.tanh(self.v_proj(hidden))
        beta = torch.sigmoid(self.beta_proj(hidden))
        decay = torch.sigmoid(self.decay_proj(hidden))
        gate = torch.sigmoid(self.gate_proj(hidden))

        prediction = z_main * query
        delta = beta * key * (value - prediction)
        candidate = decay * z_main + (1.0 - decay) * (z_main + delta)
        update = gate * (candidate - z_main)

        if self.layerscale:
            scaled_update = update.to(z_main.dtype) * self.layerscale_gamma.to(z_main.dtype)
        else:
            scaled_update = update.to(z_main.dtype)

        z_next = self.norm(z_main + scaled_update * self.transition_scale.to(z_main.dtype))
        return z_next


class StateReadoutHead(nn.Module):
    """Readout head for digit prediction."""
    def __init__(self, d_state: int, n_digits: int = 10):
        super().__init__()
        self.norm = RMSNorm(d_state)
        self.head = nn.Linear(d_state, n_digits)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(z))


class StateTransitionCore(nn.Module):
    """
    TRM-style Dual-State Core with Shared Recurrence.
    
    This implements the "TRM-llm" philosophy:
    - Dual states (z_H, z_L)
    - SHARED reasoning module (SharedReasoningCore)
    - Direct state supervision
    """

    def __init__(
        self,
        cfg: QTRMConfig,
        d_state: Optional[int] = None,
        n_operations: Optional[int] = None,
        n_steps: Optional[int] = None,
        transition_scale_init: float = 1.0,
        step_embedding_std: Optional[float] = None,
        update_schedule: str = "nested",
        core_update: str = "mlp",
        stochastic_high_level_guidance: bool = False,
        stochastic_high_level_scale: float = 0.05,
        stochastic_high_level_min_std: float = 1e-4,
        stochastic_high_level_max_std: float = 0.2,
        stochastic_high_level_eval: bool = False,
        stochastic_posterior_guidance: bool = False,
        stochastic_transition_mode: str = "delta",
        lattice_feedback_mode: str = "none",
        lattice_feedback_scale: float = 0.0,
        lattice_feedback_threshold: float = 0.5,
        workspace_cross_attention: bool = False,
        workspace_cross_attention_heads: int = 4,
        workspace_cross_attention_scale: float = 1.0,
        operation_arg_conditioning: bool = False,
        continuous_time: bool = False,
        layerscale: bool = False,
        layerscale_init: float = 1e-5,
        gate_type: str = "tanh",
        gate_bias_init: float = 0.5,
        working_register_enabled: bool = False,
        working_register_slots: int = 4,
        working_register_update_scale: float = 0.25,
        working_register_feedback_scale: float = 1.0,
        working_register_gate_init_bias: float = -2.0,
        working_register_summary_mode: str = "mean",
        working_register_role_conditioning: bool = False,
        working_register_role_anchor_scale: float = 0.0,
        working_register_update_mode: str = "all",
        working_register_source_attention: bool = False,
        working_register_source_attention_scale: float = 0.0,
        typed_value_registers: bool = False,
        source_numeric_feature_dim: int = 0,
        typed_value_update_scale: float = 0.25,
        typed_value_update_mode: str = "residual",
        typed_digit_registers: bool = False,
        typed_digit_register_digits: int = 6,
        typed_digit_update_scale: float = 0.25,
        semantic_token_feedback: bool = False,
        semantic_token_feedback_scale: float = 0.0,
        semantic_token_feedback_temperature: float = 1.0,
        semantic_token_feedback_gate_init_bias: float = -2.0,
        semantic_token_feedback_score_mode: str = "cosine",
        semantic_token_feedback_teacher_forcing: float = 0.0,
    ):
        super().__init__()
        self.d_state = d_state or cfg.d_model
        self.n_operations = n_operations or int(cfg.num_actions)
        self.n_steps = n_steps or int(cfg.outer_steps)
        self.step_embedding_std = step_embedding_std
        self.update_schedule = str(update_schedule)
        self.core_update = str(core_update)
        self.stochastic_transition_mode = str(stochastic_transition_mode)
        self.stochastic_high_level_guidance = bool(stochastic_high_level_guidance) or self.stochastic_transition_mode == "true_gram"
        self.stochastic_high_level_scale = float(stochastic_high_level_scale)
        self.stochastic_high_level_min_std = float(stochastic_high_level_min_std)
        self.stochastic_high_level_max_std = float(stochastic_high_level_max_std)
        self.stochastic_high_level_eval = bool(stochastic_high_level_eval)
        self.stochastic_posterior_guidance = bool(stochastic_posterior_guidance)
        self.lattice_feedback_mode = str(lattice_feedback_mode)
        self.lattice_feedback_scale = float(lattice_feedback_scale)
        self.lattice_feedback_threshold = float(lattice_feedback_threshold)
        self.workspace_cross_attention = bool(workspace_cross_attention)
        self.workspace_cross_attention_heads = int(workspace_cross_attention_heads)
        self.workspace_cross_attention_scale = float(workspace_cross_attention_scale)
        self.operation_arg_conditioning = bool(operation_arg_conditioning)
        self.continuous_time = bool(continuous_time)

        self.layerscale = bool(layerscale)
        self.layerscale_init = float(layerscale_init)
        self.gate_type = str(gate_type)
        self.gate_bias_init = float(gate_bias_init)
        self.working_register_enabled = bool(working_register_enabled)
        self.working_register_slots = int(working_register_slots)
        self.working_register_update_scale = float(working_register_update_scale)
        self.working_register_feedback_scale = float(working_register_feedback_scale)
        self.working_register_gate_init_bias = float(working_register_gate_init_bias)
        self.working_register_summary_mode = str(working_register_summary_mode)
        self.working_register_role_conditioning = bool(working_register_role_conditioning)
        self.working_register_role_anchor_scale = float(working_register_role_anchor_scale)
        self.working_register_update_mode = str(working_register_update_mode)
        self.working_register_source_attention = bool(working_register_source_attention)
        self.working_register_source_attention_scale = float(working_register_source_attention_scale)
        self.typed_value_registers = bool(typed_value_registers)
        self.source_numeric_feature_dim = int(source_numeric_feature_dim)
        self.typed_value_update_scale = float(typed_value_update_scale)
        self.typed_value_update_mode = str(typed_value_update_mode)
        self.typed_digit_registers = bool(typed_digit_registers)
        self.typed_digit_register_digits = int(typed_digit_register_digits)
        self.typed_digit_update_scale = float(typed_digit_update_scale)
        self.semantic_token_feedback = bool(semantic_token_feedback)
        self.semantic_token_feedback_scale = float(semantic_token_feedback_scale)
        self.semantic_token_feedback_temperature = float(semantic_token_feedback_temperature)
        self.semantic_token_feedback_gate_init_bias = float(semantic_token_feedback_gate_init_bias)
        self.semantic_token_feedback_score_mode = str(semantic_token_feedback_score_mode)
        self.semantic_token_feedback_teacher_forcing = float(semantic_token_feedback_teacher_forcing)

        if self.stochastic_transition_mode not in {"delta", "true_gram"}:
            raise ValueError(
                "stochastic_transition_mode must be one of {'delta', 'true_gram'}, "
                f"got: {self.stochastic_transition_mode}"
            )
        if self.update_schedule not in {"nested", "two_stream"}:
            raise ValueError(f"update_schedule must be one of {{'nested', 'two_stream'}}, got: {self.update_schedule}")
        if self.core_update not in {"mlp", "mini_gated_delta"}:
            raise ValueError(f"core_update must be one of {{'mlp', 'mini_gated_delta'}}, got: {self.core_update}")
        if self.lattice_feedback_mode not in {"none", "soft", "threshold"}:
            raise ValueError(
                "lattice_feedback_mode must be one of {'none', 'soft', 'threshold'}, "
                f"got: {self.lattice_feedback_mode}"
            )
        if self.lattice_feedback_scale < 0:
            raise ValueError("lattice_feedback_scale must be >= 0")
        if not 0.0 <= self.lattice_feedback_threshold <= 1.0:
            raise ValueError("lattice_feedback_threshold must be in [0, 1]")
        if self.workspace_cross_attention_heads <= 0:
            raise ValueError("workspace_cross_attention_heads must be positive")
        if self.d_state % self.workspace_cross_attention_heads != 0:
            raise ValueError("d_state must be divisible by workspace_cross_attention_heads")
        if self.workspace_cross_attention_scale < 0:
            raise ValueError("workspace_cross_attention_scale must be >= 0")
        if self.stochastic_high_level_scale < 0:
            raise ValueError("stochastic_high_level_scale must be >= 0")
        if self.stochastic_high_level_min_std < 0:
            raise ValueError("stochastic_high_level_min_std must be >= 0")
        if self.stochastic_high_level_max_std < self.stochastic_high_level_min_std:
            raise ValueError("stochastic_high_level_max_std must be >= stochastic_high_level_min_std")
        if self.working_register_slots <= 0:
            raise ValueError("working_register_slots must be positive")
        if self.working_register_update_scale < 0:
            raise ValueError("working_register_update_scale must be >= 0")
        if self.working_register_feedback_scale < 0:
            raise ValueError("working_register_feedback_scale must be >= 0")
        if self.working_register_summary_mode not in {"mean", "query_attention", "query_dot"}:
            raise ValueError(
                "working_register_summary_mode must be one of {'mean', 'query_attention', 'query_dot'}, "
                f"got: {self.working_register_summary_mode}"
            )
        if self.working_register_role_anchor_scale < 0:
            raise ValueError("working_register_role_anchor_scale must be >= 0")
        if self.working_register_source_attention_scale < 0:
            raise ValueError("working_register_source_attention_scale must be >= 0")
        if self.source_numeric_feature_dim < 0:
            raise ValueError("source_numeric_feature_dim must be >= 0")
        if self.typed_value_registers and self.source_numeric_feature_dim <= 0:
            raise ValueError("typed_value_registers requires source_numeric_feature_dim > 0")
        if self.typed_digit_registers and self.source_numeric_feature_dim <= 0:
            raise ValueError("typed_digit_registers requires source_numeric_feature_dim > 0")
        if self.typed_value_update_scale < 0:
            raise ValueError("typed_value_update_scale must be >= 0")
        if self.typed_value_update_mode not in {"residual", "gated_delta"}:
            raise ValueError(
                "typed_value_update_mode must be one of {'residual', 'gated_delta'}, "
                f"got: {self.typed_value_update_mode}"
            )
        if self.typed_digit_register_digits <= 0:
            raise ValueError("typed_digit_register_digits must be positive")
        if self.typed_digit_update_scale < 0:
            raise ValueError("typed_digit_update_scale must be >= 0")
        if self.working_register_update_mode not in {"all", "cyclic"}:
            raise ValueError(
                "working_register_update_mode must be one of {'all', 'cyclic'}, "
                f"got: {self.working_register_update_mode}"
            )
        if self.semantic_token_feedback_scale < 0:
            raise ValueError("semantic_token_feedback_scale must be >= 0")
        if self.semantic_token_feedback_temperature <= 0:
            raise ValueError("semantic_token_feedback_temperature must be > 0")
        if self.semantic_token_feedback_score_mode not in {"cosine", "dot"}:
            raise ValueError(
                "semantic_token_feedback_score_mode must be one of {'cosine', 'dot'}, "
                f"got: {self.semantic_token_feedback_score_mode}"
            )
        if not 0.0 <= self.semantic_token_feedback_teacher_forcing <= 1.0:
            raise ValueError("semantic_token_feedback_teacher_forcing must be in [0, 1]")
        
        # Shared core for ALL updates
        core_cls = MiniGatedDeltaReasoningCore if self.core_update == "mini_gated_delta" else SharedReasoningCore
        self.shared_core = core_cls(
            d_state=self.d_state,
            n_operations=self.n_operations,
            transition_scale_init=transition_scale_init,
            layerscale=self.layerscale,
            layerscale_init=self.layerscale_init,
            gate_type=self.gate_type,
            gate_bias_init=self.gate_bias_init,
        )
        
        # Operation embeddings
        self.op_embed = nn.Embedding(self.n_operations, self.d_state)
        nn.init.normal_(self.op_embed.weight, std=0.02)
        self.operation_arg_embed = nn.Embedding(10, self.d_state)
        self.initial_label_embed = nn.Embedding(10, self.d_state)
        nn.init.normal_(self.operation_arg_embed.weight, std=0.02)
        nn.init.normal_(self.initial_label_embed.weight, std=0.02)
        
        # State initializers
        self.z_h_init = nn.Linear(self.d_state, self.d_state)
        self.z_l_start = nn.Parameter(torch.zeros(self.d_state))
        
        # Step embeddings
        self.step_embed = nn.Embedding(self.n_steps + 1, self.d_state)
        if step_embedding_std is not None:
            nn.init.normal_(self.step_embed.weight, std=float(step_embedding_std))

        if self.continuous_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, self.d_state)
            )
            nn.init.normal_(self.time_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.time_mlp[0].bias)
            nn.init.normal_(self.time_mlp[2].weight, std=0.02)
            nn.init.zeros_(self.time_mlp[2].bias)
        
        # Readout
        self.state_readout = StateReadoutHead(self.d_state)
        self.op_head = nn.Linear(self.d_state, self.n_operations)

        # LDT-style lattice feedback: the alive-candidate set is projected
        # back into the recurrent state between internal solve steps.
        self.lattice_feedback_proj = nn.Linear(10, self.d_state, bias=False)
        self.lattice_feedback_norm = RMSNorm(self.d_state)
        nn.init.zeros_(self.lattice_feedback_proj.weight)

        # DCAOR: optional per-step cross-attention over the preserved Qwen
        # token workspace. This removes the single-vector operand queue
        # bottleneck while keeping the normal recurrent answer path.
        self.workspace_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_state,
            num_heads=self.workspace_cross_attention_heads,
            batch_first=True,
        )
        self.workspace_cross_attn_norm = RMSNorm(self.d_state)

        # Bounded typed working register. This is intentionally generic: slots
        # are typed persistent memory cells, not a digit-specific executor. The
        # recurrent thought state must write/read through this bounded workpad.
        self.working_register_type_embed = nn.Parameter(
            torch.empty(self.working_register_slots, self.d_state)
        )
        self.working_register_init = nn.Linear(self.d_state, self.d_state)
        register_update_factors = 5 if self.working_register_role_conditioning else 4
        self.working_register_update_input_dim = self.d_state * register_update_factors
        self.working_register_update_norm = RMSNorm(self.working_register_update_input_dim)
        self.working_register_update = nn.Linear(self.working_register_update_input_dim, self.d_state)
        self.working_register_gate = nn.Linear(self.working_register_update_input_dim, 1)
        self.working_register_readout = nn.Linear(self.d_state, self.d_state, bias=False)
        self.working_register_norm = RMSNorm(self.d_state)
        self.working_register_summary_norm = RMSNorm(self.d_state)
        self.working_register_query_norm = RMSNorm(self.d_state * 2)
        self.working_register_query_score = nn.Linear(self.d_state * 2, 1)
        self.working_register_source_attn = nn.MultiheadAttention(
            embed_dim=self.d_state,
            num_heads=self.workspace_cross_attention_heads,
            batch_first=True,
        )
        self.working_register_source_norm = RMSNorm(self.d_state)
        self.typed_value_source_proj = (
            nn.Linear(self.source_numeric_feature_dim, self.d_state)
            if self.source_numeric_feature_dim > 0
            else None
        )
        self.typed_value_source_attn = nn.MultiheadAttention(
            embed_dim=self.d_state,
            num_heads=self.workspace_cross_attention_heads,
            batch_first=True,
        )
        self.typed_value_update_norm = RMSNorm(self.d_state * 5)
        self.typed_value_update = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_gate = nn.Linear(self.d_state * 5, 1)
        self.typed_value_delta_query = nn.Linear(self.d_state, self.d_state)
        self.typed_value_delta_key = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_delta_value = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_delta_beta = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_erase_gate = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_retain_gate = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_write_gate = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_value_execute_gate = nn.Linear(self.d_state * 5, 1)
        self.typed_value_norm = RMSNorm(self.d_state)
        self.typed_digit_source_proj = (
            nn.Linear(self.source_numeric_feature_dim, self.d_state)
            if self.source_numeric_feature_dim > 0
            else None
        )
        self.typed_digit_place_embed = nn.Parameter(
            torch.empty(self.typed_digit_register_digits + 1, self.d_state)
        )
        self.typed_digit_source_attn = nn.MultiheadAttention(
            embed_dim=self.d_state,
            num_heads=self.workspace_cross_attention_heads,
            batch_first=True,
        )
        self.typed_digit_update_norm = RMSNorm(self.d_state * 5)
        self.typed_digit_delta_query = nn.Linear(self.d_state, self.d_state)
        self.typed_digit_delta_key = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_digit_delta_value = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_digit_delta_beta = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_digit_erase_gate = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_digit_retain_gate = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_digit_write_gate = nn.Linear(self.d_state * 5, self.d_state)
        self.typed_digit_execute_gate = nn.Linear(self.d_state * 5, 1)
        self.typed_digit_norm = RMSNorm(self.d_state)
        self.semantic_token_feedback_norm = RMSNorm(self.d_state * 2)
        self.semantic_token_feedback_gate = nn.Linear(self.d_state * 2, 1)
        self.semantic_token_feedback_state_norm = RMSNorm(self.d_state)
        nn.init.normal_(self.working_register_type_embed, std=0.02)
        nn.init.xavier_uniform_(self.working_register_init.weight)
        nn.init.zeros_(self.working_register_init.bias)
        nn.init.xavier_uniform_(self.working_register_update.weight)
        nn.init.zeros_(self.working_register_update.bias)
        nn.init.zeros_(self.working_register_gate.weight)
        nn.init.constant_(self.working_register_gate.bias, self.working_register_gate_init_bias)
        nn.init.xavier_uniform_(self.working_register_readout.weight)
        nn.init.zeros_(self.working_register_query_score.weight)
        nn.init.zeros_(self.working_register_query_score.bias)
        if self.typed_value_source_proj is not None:
            nn.init.xavier_uniform_(self.typed_value_source_proj.weight)
            nn.init.zeros_(self.typed_value_source_proj.bias)
        nn.init.xavier_uniform_(self.typed_value_update.weight)
        nn.init.zeros_(self.typed_value_update.bias)
        nn.init.zeros_(self.typed_value_gate.weight)
        nn.init.constant_(self.typed_value_gate.bias, self.working_register_gate_init_bias)
        nn.init.xavier_uniform_(self.typed_value_delta_query.weight)
        nn.init.zeros_(self.typed_value_delta_query.bias)
        nn.init.xavier_uniform_(self.typed_value_delta_key.weight)
        nn.init.zeros_(self.typed_value_delta_key.bias)
        nn.init.xavier_uniform_(self.typed_value_delta_value.weight)
        nn.init.zeros_(self.typed_value_delta_value.bias)
        nn.init.xavier_uniform_(self.typed_value_delta_beta.weight)
        nn.init.constant_(self.typed_value_delta_beta.bias, -1.0)
        nn.init.xavier_uniform_(self.typed_value_erase_gate.weight)
        nn.init.constant_(self.typed_value_erase_gate.bias, -1.0)
        nn.init.xavier_uniform_(self.typed_value_retain_gate.weight)
        nn.init.constant_(self.typed_value_retain_gate.bias, 2.0)
        nn.init.xavier_uniform_(self.typed_value_write_gate.weight)
        nn.init.constant_(self.typed_value_write_gate.bias, -1.0)
        nn.init.zeros_(self.typed_value_execute_gate.weight)
        nn.init.constant_(self.typed_value_execute_gate.bias, -1.0)
        if self.typed_digit_source_proj is not None:
            nn.init.xavier_uniform_(self.typed_digit_source_proj.weight)
            nn.init.zeros_(self.typed_digit_source_proj.bias)
        nn.init.normal_(self.typed_digit_place_embed, std=0.02)
        nn.init.xavier_uniform_(self.typed_digit_delta_query.weight)
        nn.init.zeros_(self.typed_digit_delta_query.bias)
        nn.init.xavier_uniform_(self.typed_digit_delta_key.weight)
        nn.init.zeros_(self.typed_digit_delta_key.bias)
        nn.init.xavier_uniform_(self.typed_digit_delta_value.weight)
        nn.init.zeros_(self.typed_digit_delta_value.bias)
        nn.init.xavier_uniform_(self.typed_digit_delta_beta.weight)
        nn.init.constant_(self.typed_digit_delta_beta.bias, -1.0)
        nn.init.xavier_uniform_(self.typed_digit_erase_gate.weight)
        nn.init.constant_(self.typed_digit_erase_gate.bias, -1.0)
        nn.init.xavier_uniform_(self.typed_digit_retain_gate.weight)
        nn.init.constant_(self.typed_digit_retain_gate.bias, 2.0)
        nn.init.xavier_uniform_(self.typed_digit_write_gate.weight)
        nn.init.constant_(self.typed_digit_write_gate.bias, -1.0)
        nn.init.zeros_(self.typed_digit_execute_gate.weight)
        nn.init.constant_(self.typed_digit_execute_gate.bias, -1.0)
        nn.init.zeros_(self.semantic_token_feedback_gate.weight)
        nn.init.constant_(self.semantic_token_feedback_gate.bias, self.semantic_token_feedback_gate_init_bias)
        with torch.no_grad():
            self.working_register_update.weight.mul_(0.05)
            self.working_register_readout.weight.mul_(0.05)
            self.typed_value_update.weight.mul_(0.05)
        
        # Injection gate
        self.injection_gate = nn.Parameter(torch.tensor(0.2))

        # GRAM-style stochastic guidance on the slower high-level latent state.
        # Kept opt-in and near-identity so legacy checkpoints remain comparable.
        self.stochastic_guidance_norm = RMSNorm(self.d_state * 2)
        self.stochastic_guidance_hidden = nn.Linear(self.d_state * 2, self.d_state * 2)
        self.stochastic_guidance_out = nn.Linear(self.d_state * 2, self.d_state * 2)
        nn.init.xavier_uniform_(self.stochastic_guidance_hidden.weight)
        nn.init.zeros_(self.stochastic_guidance_hidden.bias)
        nn.init.zeros_(self.stochastic_guidance_out.weight)
        nn.init.zeros_(self.stochastic_guidance_out.bias)
        with torch.no_grad():
            self.stochastic_guidance_out.bias[self.d_state :].fill_(-5.0)
        self.posterior_label_embed = nn.Embedding(10, self.d_state)
        self.posterior_guidance_norm = RMSNorm(self.d_state * 3)
        self.posterior_guidance_hidden = nn.Linear(self.d_state * 3, self.d_state * 2)
        self.posterior_guidance_out = nn.Linear(self.d_state * 2, self.d_state * 2)
        nn.init.normal_(self.posterior_label_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.posterior_guidance_hidden.weight)
        nn.init.zeros_(self.posterior_guidance_hidden.bias)
        nn.init.zeros_(self.posterior_guidance_out.weight)
        nn.init.zeros_(self.posterior_guidance_out.bias)
        with torch.no_grad():
            self.posterior_guidance_out.bias[self.d_state :].fill_(-5.0)
        self.true_gram_prior_norm = RMSNorm(self.d_state * 3)
        self.true_gram_prior_hidden = nn.Linear(self.d_state * 3, self.d_state * 2)
        self.true_gram_prior_out = nn.Linear(self.d_state * 2, self.d_state * 2)
        self.true_gram_posterior_norm = RMSNorm(self.d_state * 4)
        self.true_gram_posterior_hidden = nn.Linear(self.d_state * 4, self.d_state * 2)
        self.true_gram_posterior_out = nn.Linear(self.d_state * 2, self.d_state * 2)
        nn.init.xavier_uniform_(self.true_gram_prior_hidden.weight)
        nn.init.zeros_(self.true_gram_prior_hidden.bias)
        nn.init.zeros_(self.true_gram_prior_out.weight)
        nn.init.zeros_(self.true_gram_prior_out.bias)
        nn.init.xavier_uniform_(self.true_gram_posterior_hidden.weight)
        nn.init.zeros_(self.true_gram_posterior_hidden.bias)
        nn.init.zeros_(self.true_gram_posterior_out.weight)
        nn.init.zeros_(self.true_gram_posterior_out.bias)

    def _clamp_stochastic_std(self, raw_std: torch.Tensor) -> torch.Tensor:
        std = torch.nn.functional.softplus(raw_std)
        return (std + self.stochastic_high_level_min_std).clamp(max=self.stochastic_high_level_max_std)

    def _stochastic_prior_distribution(self, z_h: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        guidance_input = torch.cat([z_h, ctx], dim=-1)
        hidden = torch.nn.functional.gelu(
            self.stochastic_guidance_hidden(self.stochastic_guidance_norm(guidance_input))
        )
        mu, raw_std = self.stochastic_guidance_out(hidden).chunk(2, dim=-1)
        return mu, self._clamp_stochastic_std(raw_std)

    def _true_gram_prior_distribution(
        self,
        prev_z_h: torch.Tensor,
        op_vec: torch.Tensor,
        ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_input = torch.cat([prev_z_h, op_vec.to(prev_z_h.dtype), ctx], dim=-1)
        hidden = torch.nn.functional.gelu(
            self.true_gram_prior_hidden(self.true_gram_prior_norm(prior_input))
        )
        mu, raw_std = self.true_gram_prior_out(hidden).chunk(2, dim=-1)
        return mu, self._clamp_stochastic_std(raw_std)

    def _stochastic_posterior_distribution(
        self,
        z_h: torch.Tensor,
        ctx: torch.Tensor,
        posterior_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_vec = self.posterior_label_embed(posterior_labels.to(torch.long).clamp(min=0, max=9))
        posterior_input = torch.cat([z_h, ctx, label_vec.to(z_h.dtype)], dim=-1)
        hidden = torch.nn.functional.gelu(
            self.posterior_guidance_hidden(self.posterior_guidance_norm(posterior_input))
        )
        mu, raw_std = self.posterior_guidance_out(hidden).chunk(2, dim=-1)
        return mu, self._clamp_stochastic_std(raw_std)

    def _true_gram_posterior_distribution(
        self,
        prev_z_h: torch.Tensor,
        op_vec: torch.Tensor,
        ctx: torch.Tensor,
        posterior_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_vec = self.posterior_label_embed(posterior_labels.to(torch.long).clamp(min=0, max=9))
        posterior_input = torch.cat([prev_z_h, op_vec.to(prev_z_h.dtype), ctx, label_vec.to(prev_z_h.dtype)], dim=-1)
        hidden = torch.nn.functional.gelu(
            self.true_gram_posterior_hidden(self.true_gram_posterior_norm(posterior_input))
        )
        mu, raw_std = self.true_gram_posterior_out(hidden).chunk(2, dim=-1)
        return mu, self._clamp_stochastic_std(raw_std)

    @staticmethod
    def _diagonal_gaussian_kl(
        q_mu: torch.Tensor,
        q_std: torch.Tensor,
        p_mu: torch.Tensor,
        p_std: torch.Tensor,
    ) -> torch.Tensor:
        q_var = q_std.float().pow(2)
        p_var = p_std.float().pow(2).clamp_min(1e-8)
        mean_delta = (q_mu.float() - p_mu.float()).pow(2)
        kl = 0.5 * ((q_var + mean_delta) / p_var - 1.0 + 2.0 * (p_std.float().log() - q_std.float().log()))
        return kl.sum(dim=-1).mean()

    def _apply_stochastic_high_level_guidance(
        self,
        z_h: torch.Tensor,
        ctx: torch.Tensor,
        posterior_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.stochastic_high_level_guidance or self.stochastic_high_level_scale == 0:
            zero = z_h.new_zeros(())
            return z_h, zero, zero, zero, zero

        prior_mu, prior_std = self._stochastic_prior_distribution(z_h, ctx)
        mu, std = prior_mu, prior_std
        posterior_kl = z_h.new_zeros(())
        if self.stochastic_posterior_guidance and posterior_labels is not None:
            posterior_mu, posterior_std = self._stochastic_posterior_distribution(z_h, ctx, posterior_labels)
            posterior_kl = self._diagonal_gaussian_kl(posterior_mu, posterior_std, prior_mu, prior_std).to(z_h.dtype)
            mu, std = posterior_mu, posterior_std

        if self.training or self.stochastic_high_level_eval:
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)

        stochastic_delta = (mu + std * eps).to(z_h.dtype) * self.stochastic_high_level_scale
        z_next = z_h + stochastic_delta
        return z_next, mu.norm(dim=-1).mean(), std.mean(), stochastic_delta.norm(dim=-1).mean(), posterior_kl

    def _posterior_labels_for_step(
        self,
        posterior_labels: Optional[torch.Tensor],
        step: int,
    ) -> Optional[torch.Tensor]:
        if posterior_labels is None:
            return None
        if posterior_labels.ndim == 2:
            return posterior_labels[:, min(step, posterior_labels.size(1) - 1)]
        return posterior_labels

    def _apply_true_gram_transition(
        self,
        prev_z_h: torch.Tensor,
        op_vec: torch.Tensor,
        ctx: torch.Tensor,
        *,
        posterior_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prior_mu, prior_std = self._true_gram_prior_distribution(prev_z_h, op_vec, ctx)
        mu, std = prior_mu, prior_std
        posterior_kl = prev_z_h.new_zeros(())
        if self.stochastic_posterior_guidance and posterior_labels is not None:
            posterior_mu, posterior_std = self._true_gram_posterior_distribution(
                prev_z_h,
                op_vec,
                ctx,
                posterior_labels,
            )
            posterior_kl = self._diagonal_gaussian_kl(posterior_mu, posterior_std, prior_mu, prior_std).to(prev_z_h.dtype)
            mu, std = posterior_mu, posterior_std

        if self.training or self.stochastic_high_level_eval:
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)

        stochastic_noise = (std * eps).to(prev_z_h.dtype)
        z_next = (mu.to(prev_z_h.dtype) + stochastic_noise).to(prev_z_h.dtype)
        return z_next, mu.norm(dim=-1).mean(), std.mean(), stochastic_noise.norm(dim=-1).mean(), posterior_kl

    def _lattice_feedback_delta(self, candidate_alive: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.lattice_feedback_mode == "none" or self.lattice_feedback_scale == 0:
            return None
        feedback = self.lattice_feedback_proj(candidate_alive.to(self.lattice_feedback_proj.weight.dtype))
        feedback = self.lattice_feedback_norm(feedback).to(dtype)
        return feedback * float(self.lattice_feedback_scale)

    def _update_candidate_lattice(self, candidate_alive: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
        if self.lattice_feedback_mode == "none" or self.lattice_feedback_scale == 0:
            return candidate_alive
        candidate_probs = torch.sigmoid(self.state_readout(z_h).float()).detach()
        if self.lattice_feedback_mode == "soft":
            return torch.minimum(candidate_alive.float(), candidate_probs).to(candidate_alive.dtype)

        previous = candidate_alive
        next_alive = (candidate_alive.float() > 0).logical_and(
            candidate_probs >= float(self.lattice_feedback_threshold)
        )
        next_alive = next_alive.to(candidate_alive.dtype)
        empty = next_alive.sum(dim=-1, keepdim=True).eq(0)
        return torch.where(empty, previous, next_alive)

    def _apply_semantic_token_feedback(
        self,
        z_h: torch.Tensor,
        semantic_feedback_basis: Optional[torch.Tensor],
        semantic_feedback_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feed the current thought's Qwen-token belief back into the thought.

        The basis is supplied by the Qwen wrapper from the same LM-head label
        token directions used for evaluated answers. This keeps the recurrent
        state inside the speaker's language instead of adding a detached probe.
        """
        if (
            not self.semantic_token_feedback
            or self.semantic_token_feedback_scale == 0
            or semantic_feedback_basis is None
        ):
            zero = z_h.new_zeros(())
            return z_h, zero, zero
        if semantic_feedback_basis.ndim != 2 or semantic_feedback_basis.size(-1) != z_h.size(-1):
            raise ValueError(
                "semantic_feedback_basis must have shape (n_labels, d_state); "
                f"got {tuple(semantic_feedback_basis.shape)} for d_state={z_h.size(-1)}"
            )

        basis = semantic_feedback_basis.to(device=z_h.device, dtype=z_h.dtype)
        if self.semantic_token_feedback_score_mode == "cosine":
            z_scores = F.normalize(z_h.float(), dim=-1)
            basis_scores = F.normalize(basis.float(), dim=-1)
            logits = z_scores @ basis_scores.t()
        else:
            # Match the LM-head speaker more closely: answers are dot products
            # between the normalized thought hidden state and token directions.
            z_scores = self.semantic_token_feedback_state_norm(z_h).float()
            basis_scores = basis.float()
            logits = (z_scores @ basis_scores.t()) / math.sqrt(float(z_h.size(-1)))
        logits = logits / float(self.semantic_token_feedback_temperature)
        probs = torch.softmax(logits, dim=-1).to(z_h.dtype)
        if (
            self.training
            and self.semantic_token_feedback_teacher_forcing > 0
            and semantic_feedback_labels is not None
        ):
            labels = semantic_feedback_labels.to(device=z_h.device, dtype=torch.long).clamp(
                min=0,
                max=probs.size(-1) - 1,
            )
            teacher_probs = F.one_hot(labels, num_classes=probs.size(-1)).to(probs.dtype)
            mix = float(self.semantic_token_feedback_teacher_forcing)
            probs = probs * (1.0 - mix) + teacher_probs * mix
        feedback = probs @ basis

        gate_input = torch.cat([z_h, feedback], dim=-1)
        gate = torch.sigmoid(
            self.semantic_token_feedback_gate(
                self.semantic_token_feedback_norm(gate_input).to(self.semantic_token_feedback_gate.weight.dtype)
            )
        ).to(z_h.dtype)
        z_next = z_h + float(self.semantic_token_feedback_scale) * gate * (feedback - z_h)
        z_next = self.semantic_token_feedback_state_norm(z_next)

        entropy = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(float(probs.size(-1)))
        return z_next, gate.mean(), entropy.mean().to(z_h.dtype)

    @staticmethod
    def _workspace_summary(
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if workspace_attention_mask is None:
            return workspace.mean(dim=1)
        mask = workspace_attention_mask.to(device=workspace.device, dtype=workspace.dtype).unsqueeze(-1)
        return (workspace * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _workspace_step_context(
        self,
        z_h: torch.Tensor,
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.workspace_cross_attention or self.workspace_cross_attention_scale == 0:
            return None
        if workspace.size(1) <= 1:
            return None
        attn_dtype = self.workspace_cross_attn.in_proj_weight.dtype
        query = z_h.unsqueeze(1).to(attn_dtype)
        memory = workspace.to(attn_dtype)
        key_padding_mask = None
        if workspace_attention_mask is not None:
            key_padding_mask = workspace_attention_mask.to(device=workspace.device, dtype=torch.bool).logical_not()
        context, _ = self.workspace_cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        context = context.squeeze(1).to(z_h.dtype)
        return self.workspace_cross_attn_norm(context) * float(self.workspace_cross_attention_scale)

    @staticmethod
    def _workspace_key_padding_mask(
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if workspace_attention_mask is None or workspace_attention_mask.ndim != 2:
            return None
        if (
            workspace_attention_mask.size(0) != workspace.size(0)
            or workspace_attention_mask.size(1) != workspace.size(1)
        ):
            return None
        mask = workspace_attention_mask.to(device=workspace.device, dtype=torch.bool).logical_not()
        all_masked = mask.all(dim=1)
        if bool(all_masked.any()):
            mask = mask.clone()
            mask[all_masked] = False
        return mask

    def _source_condition_working_register(
        self,
        register_state: torch.Tensor,
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            not self.working_register_source_attention
            or float(self.working_register_source_attention_scale) <= 0.0
        ):
            return register_state
        attn_dtype = self.working_register_source_attn.in_proj_weight.dtype
        source_context, _ = self.working_register_source_attn(
            register_state.to(attn_dtype),
            workspace.to(attn_dtype),
            workspace.to(attn_dtype),
            key_padding_mask=self._workspace_key_padding_mask(workspace, workspace_attention_mask),
            need_weights=False,
        )
        source_context = self.working_register_source_norm(source_context.to(register_state.dtype))
        return self.working_register_norm(
            register_state + source_context * float(self.working_register_source_attention_scale)
        )

    def _init_working_register(
        self,
        ctx: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
        workspace_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = self.working_register_init(ctx).unsqueeze(1)
        typed = self.working_register_type_embed.unsqueeze(0).to(dtype=base.dtype, device=base.device)
        register_state = self.working_register_norm(base + typed)
        if workspace is not None:
            register_state = self._source_condition_working_register(
                register_state,
                workspace,
                workspace_attention_mask,
            )
        return register_state

    def _init_typed_value_register(
        self,
        source_numeric_features: Optional[torch.Tensor],
        source_numeric_feature_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.typed_value_registers:
            return None, None, None
        if source_numeric_features is None:
            raise ValueError("typed_value_registers requires source_numeric_features at forward time")
        if self.typed_value_source_proj is None:
            raise ValueError("typed_value_registers requires source_numeric_feature_dim > 0")
        if source_numeric_features.ndim != 3:
            raise ValueError("source_numeric_features must have shape [batch, slots, features]")
        if int(source_numeric_features.size(0)) != int(batch_size):
            raise ValueError("source_numeric_features batch must match workspace batch")
        if int(source_numeric_features.size(-1)) != self.source_numeric_feature_dim:
            raise ValueError("source_numeric_features last dimension must match source_numeric_feature_dim")
        if source_numeric_feature_mask is None:
            source_mask = torch.ones(
                source_numeric_features.shape[:2],
                device=device,
                dtype=dtype,
            )
        else:
            if source_numeric_feature_mask.ndim != 2:
                raise ValueError("source_numeric_feature_mask must have shape [batch, slots]")
            if tuple(source_numeric_feature_mask.shape) != tuple(source_numeric_features.shape[:2]):
                raise ValueError("source_numeric_feature_mask must match source_numeric_features")
            source_mask = source_numeric_feature_mask.to(device=device, dtype=dtype)
        projected = self.typed_value_source_proj(
            source_numeric_features.to(
                device=device,
                dtype=self.typed_value_source_proj.weight.dtype,
            )
        ).to(dtype)
        source_value = self.typed_value_norm(projected) * source_mask.unsqueeze(-1)
        value_register = source_value.clone()
        return value_register, source_mask, source_value

    def _working_register_role_embed(self, register_state: torch.Tensor) -> torch.Tensor:
        return self.working_register_type_embed.unsqueeze(0).to(
            dtype=register_state.dtype,
            device=register_state.device,
        ).expand(register_state.size(0), -1, -1)

    def _working_register_role_cosine(self, register_state: torch.Tensor) -> torch.Tensor:
        role_embed = self._working_register_role_embed(register_state)
        register_norm = torch.nn.functional.normalize(register_state.float(), dim=-1)
        role_norm = torch.nn.functional.normalize(role_embed.float(), dim=-1)
        return (register_norm * role_norm).sum(dim=-1).mean()

    def _working_register_summary(
        self,
        register_state: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.working_register_summary_mode == "query_dot" and query is not None:
            register_norm = torch.nn.functional.normalize(register_state.float(), dim=-1)
            query_norm = torch.nn.functional.normalize(query.float(), dim=-1).unsqueeze(1)
            scores = (register_norm * query_norm).sum(dim=-1)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1).to(register_state.dtype)
            summary = (register_state * weights).sum(dim=1)
        elif self.working_register_summary_mode == "query_attention" and query is not None:
            b, r, _ = register_state.shape
            query_slots = query.unsqueeze(1).expand(b, r, -1).to(register_state.dtype)
            score_input = torch.cat([register_state, query_slots], dim=-1)
            score_input = self.working_register_query_norm(score_input)
            scores = self.working_register_query_score(score_input).squeeze(-1)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1).to(register_state.dtype)
            summary = (register_state * weights).sum(dim=1)
        else:
            summary = register_state.mean(dim=1)
        return self.working_register_summary_norm(summary)

    @staticmethod
    def _combine_registers(
        working_register: Optional[torch.Tensor],
        typed_value_register: Optional[torch.Tensor],
        typed_digit_register: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        registers = [
            register
            for register in (working_register, typed_value_register, typed_digit_register)
            if register is not None
        ]
        if not registers:
            return None
        dtype = registers[0].dtype
        return torch.cat([register.to(dtype) for register in registers], dim=1)

    @staticmethod
    def _value_key_padding_mask(
        source_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if source_mask is None:
            return None
        mask = source_mask.to(dtype=torch.bool).logical_not()
        all_masked = mask.all(dim=1)
        if bool(all_masked.any()):
            mask = mask.clone()
            mask[all_masked] = False
        return mask

    def _update_typed_value_register(
        self,
        value_register: torch.Tensor,
        source_value_register: torch.Tensor,
        source_mask: torch.Tensor,
        z_h: torch.Tensor,
        z_l: torch.Tensor,
        op_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, r, _ = value_register.shape
        attn_dtype = self.typed_value_source_attn.in_proj_weight.dtype
        source_context, _ = self.typed_value_source_attn(
            value_register.to(attn_dtype),
            source_value_register.to(attn_dtype),
            source_value_register.to(attn_dtype),
            key_padding_mask=self._value_key_padding_mask(source_mask),
            need_weights=False,
        )
        source_context = self.typed_value_norm(source_context.to(value_register.dtype))
        z_h_expanded = z_h.unsqueeze(1).expand(b, r, -1)
        z_l_expanded = z_l.unsqueeze(1).expand(b, r, -1)
        op_expanded = op_vec.unsqueeze(1).expand(b, r, -1).to(value_register.dtype)
        update_input = torch.cat(
            [value_register, source_context, z_h_expanded, z_l_expanded, op_expanded],
            dim=-1,
        )
        update_input = self.typed_value_update_norm(update_input)
        if self.typed_value_update_mode == "gated_delta":
            query = torch.tanh(self.typed_value_delta_query(value_register))
            key = torch.sigmoid(self.typed_value_delta_key(update_input))
            target_value = torch.tanh(self.typed_value_delta_value(update_input))
            beta = torch.sigmoid(self.typed_value_delta_beta(update_input))
            erase = torch.sigmoid(self.typed_value_erase_gate(update_input)).to(value_register.dtype)
            retain = torch.sigmoid(self.typed_value_retain_gate(update_input)).to(value_register.dtype)
            write = torch.sigmoid(self.typed_value_write_gate(update_input)).to(value_register.dtype)
            execute = torch.sigmoid(self.typed_value_execute_gate(update_input)).to(value_register.dtype)

            prediction = value_register * query.to(value_register.dtype)
            erased_state = value_register - execute * erase * key.to(value_register.dtype) * prediction
            candidate = retain * erased_state + (1.0 - retain) * source_context
            delta = beta.to(value_register.dtype) * key.to(value_register.dtype) * target_value.to(
                value_register.dtype
            )
            value_register = self.typed_value_norm(
                candidate + execute * write * delta * float(self.typed_value_update_scale)
            )
            gate = execute * 0.5 * (erase.mean(dim=-1, keepdim=True) + write.mean(dim=-1, keepdim=True))
        else:
            raw_delta = torch.tanh(self.typed_value_update(update_input))
            gate = torch.sigmoid(self.typed_value_gate(update_input)).to(value_register.dtype)
            value_register = self.typed_value_norm(
                value_register + gate * raw_delta * float(self.typed_value_update_scale)
            )
        value_mask = source_mask.to(device=value_register.device, dtype=value_register.dtype).unsqueeze(-1)
        value_register = value_register * value_mask
        valid_count = value_mask.sum().clamp(min=1.0)
        gate_mean = (gate * value_mask).sum() / valid_count
        norm_mean = (value_register.norm(dim=-1) * source_mask.to(value_register.dtype)).sum() / valid_count.squeeze()
        return value_register, gate_mean, norm_mean

    def _init_typed_digit_register(
        self,
        source_numeric_features: Optional[torch.Tensor],
        source_numeric_feature_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.typed_digit_registers:
            return None, None, None
        if source_numeric_features is None:
            raise ValueError("typed_digit_registers requires source_numeric_features at forward time")
        if self.typed_digit_source_proj is None:
            raise ValueError("typed_digit_registers requires source_numeric_feature_dim > 0")
        if source_numeric_features.ndim != 3:
            raise ValueError("source_numeric_features must have shape [batch, slots, features]")
        if int(source_numeric_features.size(0)) != int(batch_size):
            raise ValueError("source_numeric_features batch must match workspace batch")
        if int(source_numeric_features.size(-1)) != self.source_numeric_feature_dim:
            raise ValueError("source_numeric_features last dimension must match source_numeric_feature_dim")
        if source_numeric_feature_mask is None:
            source_mask = torch.ones(source_numeric_features.shape[:2], device=device, dtype=dtype)
        else:
            if source_numeric_feature_mask.ndim != 2:
                raise ValueError("source_numeric_feature_mask must have shape [batch, slots]")
            if tuple(source_numeric_feature_mask.shape) != tuple(source_numeric_features.shape[:2]):
                raise ValueError("source_numeric_feature_mask must match source_numeric_features")
            source_mask = source_numeric_feature_mask.to(device=device, dtype=dtype)

        projected = self.typed_digit_source_proj(
            source_numeric_features.to(
                device=device,
                dtype=self.typed_digit_source_proj.weight.dtype,
            )
        ).to(dtype)
        projected = self.typed_digit_norm(projected)
        place_embed = self.typed_digit_place_embed.to(device=device, dtype=dtype)
        source_digit = projected.unsqueeze(2) + place_embed.unsqueeze(0).unsqueeze(0)
        b, slots, places, d_state = source_digit.shape
        source_digit = self.typed_digit_norm(source_digit.reshape(b, slots * places, d_state))
        digit_mask = source_mask.unsqueeze(-1).expand(b, slots, places).reshape(b, slots * places)
        source_digit = source_digit * digit_mask.unsqueeze(-1)
        return source_digit.clone(), digit_mask, source_digit

    def _update_typed_digit_register(
        self,
        digit_register: torch.Tensor,
        source_digit_register: torch.Tensor,
        source_mask: torch.Tensor,
        z_h: torch.Tensor,
        z_l: torch.Tensor,
        op_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, r, _ = digit_register.shape
        attn_dtype = self.typed_digit_source_attn.in_proj_weight.dtype
        source_context, _ = self.typed_digit_source_attn(
            digit_register.to(attn_dtype),
            source_digit_register.to(attn_dtype),
            source_digit_register.to(attn_dtype),
            key_padding_mask=self._value_key_padding_mask(source_mask),
            need_weights=False,
        )
        source_context = self.typed_digit_norm(source_context.to(digit_register.dtype))
        z_h_expanded = z_h.unsqueeze(1).expand(b, r, -1)
        z_l_expanded = z_l.unsqueeze(1).expand(b, r, -1)
        op_expanded = op_vec.unsqueeze(1).expand(b, r, -1).to(digit_register.dtype)
        update_input = torch.cat(
            [digit_register, source_context, z_h_expanded, z_l_expanded, op_expanded],
            dim=-1,
        )
        update_input = self.typed_digit_update_norm(update_input)
        query = torch.tanh(self.typed_digit_delta_query(digit_register))
        key = torch.sigmoid(self.typed_digit_delta_key(update_input))
        target_value = torch.tanh(self.typed_digit_delta_value(update_input))
        beta = torch.sigmoid(self.typed_digit_delta_beta(update_input))
        erase = torch.sigmoid(self.typed_digit_erase_gate(update_input)).to(digit_register.dtype)
        retain = torch.sigmoid(self.typed_digit_retain_gate(update_input)).to(digit_register.dtype)
        write = torch.sigmoid(self.typed_digit_write_gate(update_input)).to(digit_register.dtype)
        execute = torch.sigmoid(self.typed_digit_execute_gate(update_input)).to(digit_register.dtype)

        prediction = digit_register * query.to(digit_register.dtype)
        erased_state = digit_register - execute * erase * key.to(digit_register.dtype) * prediction
        candidate = retain * erased_state + (1.0 - retain) * source_context
        delta = beta.to(digit_register.dtype) * key.to(digit_register.dtype) * target_value.to(
            digit_register.dtype
        )
        digit_register = self.typed_digit_norm(
            candidate + execute * write * delta * float(self.typed_digit_update_scale)
        )
        gate = execute * 0.5 * (erase.mean(dim=-1, keepdim=True) + write.mean(dim=-1, keepdim=True))
        digit_mask = source_mask.to(device=digit_register.device, dtype=digit_register.dtype).unsqueeze(-1)
        digit_register = digit_register * digit_mask
        valid_count = digit_mask.sum().clamp(min=1.0)
        gate_mean = (gate * digit_mask).sum() / valid_count
        norm_mean = (digit_register.norm(dim=-1) * source_mask.to(digit_register.dtype)).sum() / valid_count.squeeze()
        return digit_register, gate_mean, norm_mean

    def _update_working_register(
        self,
        register_state: torch.Tensor,
        z_h: torch.Tensor,
        z_l: torch.Tensor,
        op_vec: torch.Tensor,
        step_index: int,
        workspace: Optional[torch.Tensor] = None,
        workspace_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, r, _ = register_state.shape
        z_h_expanded = z_h.unsqueeze(1).expand(b, r, -1)
        z_l_expanded = z_l.unsqueeze(1).expand(b, r, -1)
        op_expanded = op_vec.unsqueeze(1).expand(b, r, -1).to(register_state.dtype)
        update_parts = [register_state, z_h_expanded, z_l_expanded, op_expanded]
        if self.working_register_role_conditioning:
            update_parts.append(self._working_register_role_embed(register_state))
        update_input = torch.cat(update_parts, dim=-1)
        update_input = self.working_register_update_norm(update_input)
        raw_delta = torch.tanh(self.working_register_update(update_input))
        gate = torch.sigmoid(self.working_register_gate(update_input)).to(register_state.dtype)
        if self.working_register_update_mode == "cyclic":
            slot_index = int(step_index) % self.working_register_slots
            mask = torch.zeros(b, r, 1, device=register_state.device, dtype=register_state.dtype)
            mask[:, slot_index, :] = 1.0
            gate = gate * mask
        register_state = self.working_register_norm(
            register_state + gate * raw_delta * float(self.working_register_update_scale)
        )
        if self.working_register_role_anchor_scale > 0:
            role_delta = self._working_register_role_embed(register_state)
            register_state = self.working_register_norm(
                register_state + role_delta * float(self.working_register_role_anchor_scale)
            )
        if workspace is not None:
            register_state = self._source_condition_working_register(
                register_state,
                workspace,
                workspace_attention_mask,
            )
        return register_state, gate.mean(), register_state.norm(dim=-1).mean(), self._working_register_role_cosine(register_state)

    def forward(
        self,
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor] = None,
        operation_ids: Optional[torch.Tensor] = None,
        operation_arg_ids: Optional[torch.Tensor] = None,
        initial_labels: Optional[torch.Tensor] = None,
        operation_soft: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        initial_state: Optional[torch.Tensor] = None,
        posterior_labels: Optional[torch.Tensor] = None,
        semantic_feedback_basis: Optional[torch.Tensor] = None,
        source_numeric_features: Optional[torch.Tensor] = None,
        source_numeric_feature_mask: Optional[torch.Tensor] = None,
    ) -> StateTransitionOutput:
        b, w, d = workspace.shape
        n_steps = n_steps or self.n_steps
        
        # Initial z_H
        ctx = self._workspace_summary(workspace, workspace_attention_mask)
        z_h = self.z_h_init(ctx)
        if self.operation_arg_conditioning and initial_labels is not None:
            z_h = z_h + self.initial_label_embed(initial_labels.to(torch.long).clamp(min=0, max=9)).to(z_h.dtype)
        if initial_state is not None:
            gamma = torch.sigmoid(self.injection_gate)
            z_h = gamma * z_h + (1 - gamma) * initial_state
        
        # Initial z_L
        z_l = self.z_l_start.unsqueeze(0).expand(b, -1)
        
        # Add step 0 embedding to z_H
        if self.continuous_time:
            time_val = z_h.new_tensor([[0.0, 1.0 / n_steps]]).expand(b, -1)
            time_emb = self.time_mlp(time_val)
            z_h = z_h + time_emb
        else:
            z_h = z_h + self.step_embed.weight[0]
        
        state_trajectory = [z_h]
        state_norms = [z_h.norm(dim=-1).mean()]
        transition_norms = []
        state_cosines = []
        stochastic_mu_norms = []
        stochastic_std_means = []
        stochastic_noise_norms = []
        stochastic_posterior_kls = []
        working_register_gate_means = []
        working_register_norms = []
        working_register_role_cosines = []
        typed_value_register_gate_means = []
        typed_value_register_norms = []
        typed_digit_register_gate_means = []
        typed_digit_register_norms = []
        semantic_token_feedback_gate_means = []
        semantic_token_feedback_entropies = []
        candidate_alive = workspace.new_ones(b, 10)
        working_register = (
            self._init_working_register(ctx, workspace, workspace_attention_mask)
            if self.working_register_enabled
            else None
        )
        typed_value_register, typed_value_mask, typed_value_source = self._init_typed_value_register(
            source_numeric_features,
            source_numeric_feature_mask,
            batch_size=b,
            device=workspace.device,
            dtype=workspace.dtype,
        )
        typed_digit_register, typed_digit_mask, typed_digit_source = self._init_typed_digit_register(
            source_numeric_features,
            source_numeric_feature_mask,
            batch_size=b,
            device=workspace.device,
            dtype=workspace.dtype,
        )
        combined_register = self._combine_registers(
            working_register,
            typed_value_register,
            typed_digit_register,
        )
        working_register_trajectory = [combined_register] if combined_register is not None else []
        typed_value_register_trajectory = [typed_value_register] if typed_value_register is not None else []
        typed_digit_register_trajectory = [typed_digit_register] if typed_digit_register is not None else []
        
        zero_op = workspace.new_zeros(b, d)
        
        for t in range(n_steps):
            prev_z_h = z_h
            lattice_delta = self._lattice_feedback_delta(candidate_alive, z_h.dtype)
            if lattice_delta is not None:
                z_h = z_h + lattice_delta
            
            # Get operation
            if operation_ids is not None:
                op_vec = self.op_embed(operation_ids[:, t].to(torch.long))
            elif operation_soft is not None:
                op_vec = operation_soft[:, t] @ self.op_embed.weight.to(dtype=z_h.dtype)
            else:
                op_vec = zero_op
            if self.operation_arg_conditioning and operation_arg_ids is not None:
                op_vec = op_vec + self.operation_arg_embed(
                    operation_arg_ids[:, t].to(torch.long).clamp(min=0, max=9)
                ).to(op_vec.dtype)

            combined_register = self._combine_registers(
                working_register,
                typed_value_register,
                typed_digit_register,
            )
            if combined_register is not None:
                register_summary = self._working_register_summary(combined_register, query=z_h)
                op_vec = op_vec + self.working_register_readout(register_summary).to(op_vec.dtype) * float(
                    self.working_register_feedback_scale
                )

            workspace_context = self._workspace_step_context(z_h, workspace, workspace_attention_mask)
            if workspace_context is not None:
                op_vec = op_vec + workspace_context

            step_posterior_labels = self._posterior_labels_for_step(posterior_labels, t)
            if self.stochastic_transition_mode == "true_gram":
                z_h, stochastic_mu_norm, stochastic_std_mean, stochastic_noise_norm, stochastic_posterior_kl = (
                    self._apply_true_gram_transition(
                        z_h,
                        op_vec,
                        ctx,
                        posterior_labels=step_posterior_labels,
                    )
                )
            else:
                stochastic_mu_norm = z_h.new_zeros(())
                stochastic_std_mean = z_h.new_zeros(())
                stochastic_noise_norm = z_h.new_zeros(())
                stochastic_posterior_kl = z_h.new_zeros(())
            
            if self.update_schedule == "two_stream":
                # Flat two-stream recurrence: solution and scratch states both
                # read the previous pair, instead of nesting z_H behind the
                # freshly-updated z_L. This tests compute placement without
                # changing checkpoint tensor shapes.
                prev_z_l = z_l
                z_l_next = self.shared_core(z_l, z_h, op_vec)
                z_h_next = self.shared_core(z_h, prev_z_l, op_vec)
                z_l, z_h = z_l_next, z_h_next
            else:
                # 1. Update z_L using Shared Core
                z_l = self.shared_core(z_l, z_h, op_vec)

                # 2. Update z_H using Shared Core (op_vec is zero for z_H update)
                z_h = self.shared_core(z_h, z_l, zero_op)

            if self.stochastic_transition_mode == "delta":
                z_h, stochastic_mu_norm, stochastic_std_mean, stochastic_noise_norm, stochastic_posterior_kl = (
                    self._apply_stochastic_high_level_guidance(z_h, ctx, posterior_labels=posterior_labels)
                )
            stochastic_mu_norms.append(stochastic_mu_norm)
            stochastic_std_means.append(stochastic_std_mean)
            stochastic_noise_norms.append(stochastic_noise_norm)
            stochastic_posterior_kls.append(stochastic_posterior_kl)

            if working_register is not None:
                working_register, register_gate_mean, register_norm, register_role_cosine = self._update_working_register(
                    working_register,
                    z_h,
                    z_l,
                    op_vec,
                    t,
                    workspace,
                    workspace_attention_mask,
                )
                working_register_gate_means.append(register_gate_mean)
                working_register_norms.append(register_norm)
                working_register_role_cosines.append(register_role_cosine)
            if typed_value_register is not None:
                assert typed_value_source is not None
                assert typed_value_mask is not None
                typed_value_register, typed_value_gate_mean, typed_value_norm = self._update_typed_value_register(
                    typed_value_register,
                    typed_value_source,
                    typed_value_mask,
                    z_h,
                    z_l,
                    op_vec,
                )
                typed_value_register_gate_means.append(typed_value_gate_mean)
                typed_value_register_norms.append(typed_value_norm)
                typed_value_register_trajectory.append(typed_value_register)
            if typed_digit_register is not None:
                assert typed_digit_source is not None
                assert typed_digit_mask is not None
                typed_digit_register, typed_digit_gate_mean, typed_digit_norm = self._update_typed_digit_register(
                    typed_digit_register,
                    typed_digit_source,
                    typed_digit_mask,
                    z_h,
                    z_l,
                    op_vec,
                )
                typed_digit_register_gate_means.append(typed_digit_gate_mean)
                typed_digit_register_norms.append(typed_digit_norm)
                typed_digit_register_trajectory.append(typed_digit_register)
            combined_register = self._combine_registers(
                working_register,
                typed_value_register,
                typed_digit_register,
            )
            if combined_register is not None:
                register_summary = self._working_register_summary(combined_register, query=z_h)
                z_h = z_h + self.working_register_readout(register_summary).to(z_h.dtype) * float(
                    self.working_register_feedback_scale
                )
                working_register_trajectory.append(combined_register)
            
            # Add step embedding
            if self.continuous_time:
                time_val = z_h.new_tensor([[(t + 1.0) / n_steps, 1.0 / n_steps]]).expand(b, -1)
                time_emb = self.time_mlp(time_val)
                z_h = z_h + time_emb
            else:
                z_h = z_h + self.step_embed.weight[min(t + 1, self.n_steps)]
            
            # Injection
            gamma = torch.sigmoid(self.injection_gate)
            z_h = gamma * z_h + (1 - gamma) * self.z_h_init(ctx)
            z_h, semantic_gate_mean, semantic_entropy = self._apply_semantic_token_feedback(
                z_h,
                semantic_feedback_basis,
                semantic_feedback_labels=step_posterior_labels,
            )
            if self.semantic_token_feedback and self.semantic_token_feedback_scale > 0:
                semantic_token_feedback_gate_means.append(semantic_gate_mean)
                semantic_token_feedback_entropies.append(semantic_entropy)
            candidate_alive = self._update_candidate_lattice(candidate_alive, z_h)
            
            # Telemetry
            state_trajectory.append(z_h)
            state_norms.append(z_h.norm(dim=-1).mean())
            
            delta = z_h - prev_z_h
            transition_norms.append(delta.norm(dim=-1).mean())
            
            cos = F.cosine_similarity(prev_z_h, z_h, dim=-1).mean()
            state_cosines.append(cos)
            
        trajectory = torch.stack(state_trajectory, dim=1)
        state_digit_logits = self.state_readout(trajectory)
        answer_logits = state_digit_logits[:, -1, :]
        operation_logits = self.op_head(trajectory[:, 1:, :])
        stochastic_mu_tensor = torch.stack(stochastic_mu_norms).to(workspace.device) if self.stochastic_high_level_guidance else None
        stochastic_std_tensor = torch.stack(stochastic_std_means).to(workspace.device) if self.stochastic_high_level_guidance else None
        stochastic_noise_tensor = torch.stack(stochastic_noise_norms).to(workspace.device) if self.stochastic_high_level_guidance else None
        stochastic_kl_tensor = (
            torch.stack(stochastic_posterior_kls).to(workspace.device)
            if self.stochastic_high_level_guidance and self.stochastic_posterior_guidance
            else None
        )
        working_register_norm_tensor = (
            torch.stack(working_register_norms).to(workspace.device)
            if self.working_register_enabled and working_register_norms
            else None
        )
        working_register_gate_tensor = (
            torch.stack(working_register_gate_means).to(workspace.device)
            if self.working_register_enabled and working_register_gate_means
            else None
        )
        working_register_role_cosine_tensor = (
            torch.stack(working_register_role_cosines).to(workspace.device)
            if self.working_register_enabled and working_register_role_cosines
            else None
        )
        working_register_trajectory_tensor = (
            torch.stack(working_register_trajectory, dim=1).to(workspace.device)
            if working_register_trajectory
            else None
        )
        typed_value_register_norm_tensor = (
            torch.stack(typed_value_register_norms).to(workspace.device)
            if self.typed_value_registers and typed_value_register_norms
            else None
        )
        typed_value_register_gate_tensor = (
            torch.stack(typed_value_register_gate_means).to(workspace.device)
            if self.typed_value_registers and typed_value_register_gate_means
            else None
        )
        typed_value_register_trajectory_tensor = (
            torch.stack(typed_value_register_trajectory, dim=1).to(workspace.device)
            if self.typed_value_registers and typed_value_register_trajectory
            else None
        )
        typed_digit_register_norm_tensor = (
            torch.stack(typed_digit_register_norms).to(workspace.device)
            if self.typed_digit_registers and typed_digit_register_norms
            else None
        )
        typed_digit_register_gate_tensor = (
            torch.stack(typed_digit_register_gate_means).to(workspace.device)
            if self.typed_digit_registers and typed_digit_register_gate_means
            else None
        )
        typed_digit_register_trajectory_tensor = (
            torch.stack(typed_digit_register_trajectory, dim=1).to(workspace.device)
            if self.typed_digit_registers and typed_digit_register_trajectory
            else None
        )
        semantic_token_feedback_gate_tensor = (
            torch.stack(semantic_token_feedback_gate_means).to(workspace.device)
            if self.semantic_token_feedback and semantic_token_feedback_gate_means
            else None
        )
        semantic_token_feedback_entropy_tensor = (
            torch.stack(semantic_token_feedback_entropies).to(workspace.device)
            if self.semantic_token_feedback and semantic_token_feedback_entropies
            else None
        )
        
        return StateTransitionOutput(
            state_trajectory=trajectory,
            state_digit_logits=state_digit_logits,
            answer_logits=answer_logits,
            operation_logits=operation_logits,
            state_norms=torch.tensor(state_norms, device=workspace.device),
            transition_norms=torch.tensor(transition_norms, device=workspace.device),
            state_cosines=torch.tensor(state_cosines, device=workspace.device),
            stochastic_mu_norms=stochastic_mu_tensor,
            stochastic_std_means=stochastic_std_tensor,
            stochastic_noise_norms=stochastic_noise_tensor,
            stochastic_posterior_kls=stochastic_kl_tensor,
            working_register_norms=working_register_norm_tensor,
            working_register_gate_means=working_register_gate_tensor,
            working_register_role_cosines=working_register_role_cosine_tensor,
            working_register_trajectory=working_register_trajectory_tensor,
            typed_value_register_norms=typed_value_register_norm_tensor,
            typed_value_register_gate_means=typed_value_register_gate_tensor,
            typed_value_register_trajectory=typed_value_register_trajectory_tensor,
            typed_digit_register_norms=typed_digit_register_norm_tensor,
            typed_digit_register_gate_means=typed_digit_register_gate_tensor,
            typed_digit_register_trajectory=typed_digit_register_trajectory_tensor,
            semantic_token_feedback_gate_means=semantic_token_feedback_gate_tensor,
            semantic_token_feedback_entropies=semantic_token_feedback_entropy_tensor,
        )


class HybridStateTransitionCore(nn.Module):
    """
    Qwen3.5-style Hybrid State Transition Core (3:1 GatedDeltaNet & Full-Attention layout).
    """

    def __init__(
        self,
        cfg: QTRMConfig,
        d_state: Optional[int] = None,
        n_operations: Optional[int] = None,
        n_steps: Optional[int] = None,
        transition_scale_init: float = 1.0,
        step_embedding_std: Optional[float] = None,
        update_schedule: str = "nested",
        continuous_time: bool = False,
    ):
        super().__init__()
        self.d_state = d_state or cfg.d_model
        self.n_operations = n_operations or int(cfg.num_actions)
        self.n_steps = n_steps or int(cfg.outer_steps)
        self.step_embedding_std = step_embedding_std
        self.update_schedule = str(update_schedule)
        self.continuous_time = bool(continuous_time)

        # 3 GatedDeltaMixer sublayers (FLADeltaMixer with double-fallback support)
        self.delta_mixers = nn.ModuleList([
            FLADeltaMixer(d_model=self.d_state, n_heads=4, backend="fla_gated_delta", strict=False)
            for _ in range(3)
        ])

        # 1 Full-Attention synchronization layer
        self.sync_attn = nn.MultiheadAttention(embed_dim=self.d_state, num_heads=4, batch_first=True)
        self.norm_attn = RMSNorm(self.d_state)

        # Operation embeddings
        self.op_embed = nn.Embedding(self.n_operations, self.d_state)
        nn.init.normal_(self.op_embed.weight, std=0.02)

        # State initializers
        self.z_h_init = nn.Linear(self.d_state, self.d_state)
        self.z_l_start = nn.Parameter(torch.zeros(self.d_state))

        # Step embeddings
        self.step_embed = nn.Embedding(self.n_steps + 1, self.d_state)
        if step_embedding_std is not None:
            nn.init.normal_(self.step_embed.weight, std=float(step_embedding_std))

        if self.continuous_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, self.d_state)
            )
            nn.init.normal_(self.time_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.time_mlp[0].bias)
            nn.init.normal_(self.time_mlp[2].weight, std=0.02)
            nn.init.zeros_(self.time_mlp[2].bias)

        # Readout
        self.state_readout = StateReadoutHead(self.d_state)
        self.op_head = nn.Linear(self.d_state, self.n_operations)

        # Injection gate
        self.injection_gate = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor] = None,
        operation_ids: Optional[torch.Tensor] = None,
        operation_arg_ids: Optional[torch.Tensor] = None,
        initial_labels: Optional[torch.Tensor] = None,
        operation_soft: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        initial_state: Optional[torch.Tensor] = None,
        posterior_labels: Optional[torch.Tensor] = None,
        semantic_feedback_basis: Optional[torch.Tensor] = None,
    ) -> StateTransitionOutput:
        b, w, d = workspace.shape
        n_steps = n_steps or self.n_steps

        # Initial z_H
        ctx = StateTransitionCore._workspace_summary(workspace, workspace_attention_mask)
        z_h = self.z_h_init(ctx)
        if initial_state is not None:
            gamma = torch.sigmoid(self.injection_gate)
            z_h = gamma * z_h + (1 - gamma) * initial_state

        # Initial z_L (maintained as simple zero/start embedding if needed, or matched style)
        z_l = self.z_l_start.unsqueeze(0).expand(b, -1)

        # Add step 0 embedding to z_H
        if self.continuous_time:
            time_val = z_h.new_tensor([[0.0, 1.0 / n_steps]]).expand(b, -1)
            time_emb = self.time_mlp(time_val)
            z_h = z_h + time_emb
        else:
            z_h = z_h + self.step_embed.weight[0]

        state_trajectory = [z_h]
        state_norms = [z_h.norm(dim=-1).mean()]
        transition_norms = []
        state_cosines = []

        # We construct a sequential trajectory history to feed into sequence mixers
        state_history = [z_h.unsqueeze(1)]

        for t in range(n_steps):
            prev_z_h = z_h

            # Get operation
            if operation_ids is not None:
                op_vec = self.op_embed(operation_ids[:, t].to(torch.long))
            elif operation_soft is not None:
                op_vec = operation_soft[:, t] @ self.op_embed.weight.to(dtype=z_h.dtype)
            else:
                op_vec = workspace.new_zeros(b, d)

            # Sequence input formed by concatenating trajectory states
            seq_in = torch.cat(state_history, dim=1)  # (B, t+1, d_state)

            # Apply Qwen3.5 3:1 Hybrid block logic:
            if (t + 1) % 4 == 0:
                # 1 Full-Attention Layer (Periodic synchronization point)
                attn_out, _ = self.sync_attn(seq_in, seq_in, seq_in)
                z_next = self.norm_attn(seq_in[:, -1, :] + attn_out[:, -1, :])
            else:
                # 3 GatedDeltaNet Recurrent Mixer layers
                mixer_idx = t % 3
                mixed_seq = self.delta_mixers[mixer_idx](seq_in)
                z_next = seq_in[:, -1, :] + mixed_seq[:, -1, :] + op_vec

            # Add step embedding
            if self.continuous_time:
                time_val = z_next.new_tensor([[(t + 1.0) / n_steps, 1.0 / n_steps]]).expand(b, -1)
                time_emb = self.time_mlp(time_val)
                z_next = z_next + time_emb
            else:
                z_next = z_next + self.step_embed.weight[min(t + 1, self.n_steps)]

            # Injection
            gamma = torch.sigmoid(self.injection_gate)
            z_next = gamma * z_next + (1 - gamma) * self.z_h_init(ctx)

            z_h = z_next
            state_history.append(z_h.unsqueeze(1))

            # Telemetry tracking
            state_trajectory.append(z_h)
            state_norms.append(z_h.norm(dim=-1).mean())

            delta = z_h - prev_z_h
            transition_norms.append(delta.norm(dim=-1).mean())

            cos = F.cosine_similarity(prev_z_h, z_h, dim=-1).mean()
            state_cosines.append(cos)

        trajectory = torch.stack(state_trajectory, dim=1)
        state_digit_logits = self.state_readout(trajectory)
        answer_logits = state_digit_logits[:, -1, :]
        operation_logits = self.op_head(trajectory[:, 1:, :])

        return StateTransitionOutput(
            state_trajectory=trajectory,
            state_digit_logits=state_digit_logits,
            answer_logits=answer_logits,
            operation_logits=operation_logits,
            state_norms=torch.tensor(state_norms, device=workspace.device),
            transition_norms=torch.tensor(transition_norms, device=workspace.device),
            state_cosines=torch.tensor(state_cosines, device=workspace.device),
        )
