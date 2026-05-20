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
    ):
        super().__init__()
        self.d_state = d_state
        self.n_operations = n_operations
        self.hidden_dim = hidden_dim or d_state * 4
        
        # Shared projection for the inputs [state_main, state_side, op_vec]
        # In TRM, op_vec might be zero when updating z_H
        self.input_proj = nn.Linear(d_state * 3, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, d_state)
        
        # Gating mechanism
        self.gate = nn.Linear(d_state, 1)
        
        # Normalization
        self.norm = RMSNorm(d_state)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.constant_(self.gate.bias, 0.5)

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
        delta = self.output_proj(hidden)
        
        # Gating
        g = torch.tanh(self.gate(z_main))
        
        # Residual update with normalization
        z_next = self.norm(z_main + delta * g)
        
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
    ):
        super().__init__()
        self.d_state = d_state or cfg.d_model
        self.n_operations = n_operations or int(cfg.num_actions)
        self.n_steps = n_steps or int(cfg.outer_steps)
        
        # Shared core for ALL updates
        self.shared_core = SharedReasoningCore(
            d_state=self.d_state,
            n_operations=self.n_operations,
        )
        
        # Operation embeddings
        self.op_embed = nn.Embedding(self.n_operations, self.d_state)
        nn.init.normal_(self.op_embed.weight, std=0.02)
        
        # State initializers
        self.z_h_init = nn.Linear(self.d_state, self.d_state)
        self.z_l_start = nn.Parameter(torch.zeros(self.d_state))
        
        # Step embeddings
        self.step_embed = nn.Embedding(self.n_steps + 1, self.d_state)
        
        # Readout
        self.state_readout = StateReadoutHead(self.d_state)
        self.op_head = nn.Linear(self.d_state, self.n_operations)
        
        # Injection gate
        self.injection_gate = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        workspace: torch.Tensor,
        operation_ids: Optional[torch.Tensor] = None,
        operation_soft: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> StateTransitionOutput:
        b, w, d = workspace.shape
        n_steps = n_steps or self.n_steps
        
        # Initial z_H
        ctx = workspace.mean(dim=1)
        z_h = self.z_h_init(ctx)
        if initial_state is not None:
            gamma = torch.sigmoid(self.injection_gate)
            z_h = gamma * z_h + (1 - gamma) * initial_state
        
        # Initial z_L
        z_l = self.z_l_start.unsqueeze(0).expand(b, -1)
        
        # Add step 0 embedding to z_H
        z_h = z_h + self.step_embed.weight[0]
        
        state_trajectory = [z_h]
        state_norms = [z_h.norm(dim=-1).mean()]
        transition_norms = []
        state_cosines = []
        
        zero_op = workspace.new_zeros(b, d)
        
        for t in range(n_steps):
            prev_z_h = z_h
            
            # Get operation
            if operation_ids is not None:
                op_vec = self.op_embed(operation_ids[:, t].to(torch.long))
            elif operation_soft is not None:
                op_vec = operation_soft[:, t] @ self.op_embed.weight.to(dtype=z_h.dtype)
            else:
                op_vec = zero_op
            
            # 1. Update z_L using Shared Core
            z_l = self.shared_core(z_l, z_h, op_vec)
            
            # 2. Update z_H using Shared Core (op_vec is zero for z_H update)
            z_h = self.shared_core(z_h, z_l, zero_op)
            
            # Add step embedding
            z_h = z_h + self.step_embed.weight[min(t + 1, self.n_steps)]
            
            # Injection
            gamma = torch.sigmoid(self.injection_gate)
            z_h = gamma * z_h + (1 - gamma) * self.z_h_init(ctx)
            
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
        
        return StateTransitionOutput(
            state_trajectory=trajectory,
            state_digit_logits=state_digit_logits,
            answer_logits=answer_logits,
            operation_logits=operation_logits,
            state_norms=torch.tensor(state_norms, device=workspace.device),
            transition_norms=torch.tensor(transition_norms, device=workspace.device),
            state_cosines=torch.tensor(state_cosines, device=workspace.device),
        )
