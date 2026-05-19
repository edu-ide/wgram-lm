"""
State-Transition-First Core.

Operation-conditioned recurrent state machine where intermediate states
are the PRIMARY supervised target, not a residual adapter.

Architecture:
    prompt -> compressor -> z_0 (initial state)
    z_t -> transition(z_t, op_t) -> z_{t+1}
    z_final -> answer_head -> answer logits

Key difference from current QTRM:
    - State prediction is the PRIMARY loss (not answer CE)
    - Answer flows ONLY through the state path (no donor bypass)
    - Operations are explicit embeddings conditioning each transition
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

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
    # Full trajectory of states: [z_0, z_1, ..., z_T]
    state_trajectory: list[torch.Tensor]
    # Digit logits at each depth step (for state supervision)
    state_digit_logits: torch.Tensor  # (B, T+1, 10)
    # Final answer logits (from last state only)
    answer_logits: torch.Tensor  # (B, 10)
    # Operation logits at each step (for operation supervision)
    operation_logits: Optional[torch.Tensor]  # (B, T, n_ops) or None
    # Telemetry
    state_norms: torch.Tensor  # (B, T+1)
    transition_norms: torch.Tensor  # (B, T)
    state_cosines: torch.Tensor  # (B, T) - cosine between consecutive states


class OperationConditionedTransition(nn.Module):
    """
    Operation-conditioned state transition function.
    
    z_{t+1} = z_t + Transition(z_t, op_t)
    
    Where op_t is an operation embedding that conditions the transition.
    This makes the transition explicit: each operation transforms the state
    toward the next intermediate answer.
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
        
        # Operation embeddings - each operation has a learned vector
        self.op_embed = nn.Embedding(n_operations, d_state)
        
        # Transition MLP: takes [current_state, operation] -> delta
        self.transition = nn.Sequential(
            nn.LayerNorm(d_state * 2),
            nn.Linear(d_state * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, d_state),
            # Gate to control transition magnitude
            nn.Sigmoid(),
        )
        
        # Separate gate for the magnitude
        self.transition_gate = nn.Linear(d_state, 1, bias=False)
        nn.init.zeros_(self.transition_gate.weight)
        
        # Output normalization
        self.output_norm = RMSNorm(d_state)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.op_embed.weight, mean=0.0, std=0.02)
        # Initialize transition with near-zero output (residual-friendly)
        for module in self.transition:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        z_t: torch.Tensor,  # (B, d_state)
        op_ids: torch.Tensor,  # (B,) or (B, n_ops) for one-hot/softmax
        op_soft: Optional[torch.Tensor] = None,  # (B, n_ops) softmax weights
    ) -> torch.Tensor:
        """
        Apply one transition step.
        
        Args:
            z_t: Current state (B, d_state)
            op_ids: Operation IDs (B,) for hard supervision
            op_soft: Soft operation weights (B, n_ops) for soft mixing
        
        Returns:
            z_{t+1}: Next state (B, d_state)
        """
        # Get operation vector
        if op_soft is not None and op_soft.dim() == 2:
            # Soft operation mixing
            op_vec = (op_soft @ self.op_embed.weight.to(dtype=z_t.dtype))
        else:
            # Hard operation lookup
            op_vec = self.op_embed(op_ids.to(torch.long))
        
        # Concatenate current state with operation
        combined = torch.cat([z_t, op_vec], dim=-1)
        
        # Compute gated transition
        raw_delta = self.transition(combined)
        # The last sigmoid layer already applied; raw_delta is in [0, 1]
        # Scale to reasonable range
        delta = raw_delta * math.sqrt(self.d_state)
        
        # Apply gate to control how much to update
        gate = torch.tanh(self.transition_gate(z_t)).squeeze(-1)
        delta = delta * gate.unsqueeze(-1)
        
        # Residual update
        z_next = self.output_norm(z_t + delta)
        
        return z_next


class StateReadoutHead(nn.Module):
    """
    Read digit logits from a state.
    
    Maps the continuous state to digit 0-9 logits.
    This is the supervised target for intermediate states.
    """

    def __init__(self, d_state: int, n_digits: int = 10):
        super().__init__()
        self.d_state = d_state
        self.n_digits = n_digits
        
        self.norm = RMSNorm(d_state)
        self.head = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.GELU(),
            nn.Linear(d_state * 2, n_digits),
        )
        
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, d_state) -> (B, n_digits) digit logits."""
        return self.head(self.norm(z))


class StateTransitionCore(nn.Module):
    """
    State-transition-first recurrent core.
    
    The primary difference from the current QTRM core:
    1. State prediction is the PRIMARY learning objective
    2. Each transition is explicitly conditioned on an operation
    3. The answer flows ONLY through the final state
    4. No donor bypass - the state path is mandatory
    """

    def __init__(
        self,
        cfg: QTRMConfig,
        d_state: Optional[int] = None,
        n_operations: Optional[int] = None,
        n_steps: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_state = d_state or cfg.d_model
        self.n_operations = n_operations or max(1, int(cfg.num_actions))
        self.n_steps = n_steps or max(1, int(cfg.outer_steps))
        
        # Initial state projection: workspace/context -> z_0
        self.state_init = nn.Sequential(
            nn.LayerNorm(self.d_state),
            nn.Linear(self.d_state, self.d_state * 2),
            nn.GELU(),
            nn.Linear(self.d_state * 2, self.d_state),
            RMSNorm(self.d_state),
        )
        
        # Operation-conditioned transition function
        self.transition = OperationConditionedTransition(
            d_state=self.d_state,
            n_operations=self.n_operations,
        )
        
        # Step conditioning (like Parcae-style loop index)
        self.step_embed = nn.Embedding(self.n_steps + 1, self.d_state)
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        
        # State readout head for digit supervision at each step
        self.state_readout = StateReadoutHead(self.d_state)
        
        # Answer head - reads ONLY from the final state
        self.answer_head = StateReadoutHead(self.d_state)
        
        # Operation prediction head (optional, for supervised operation learning)
        self.op_head = nn.Sequential(
            RMSNorm(self.d_state),
            nn.Linear(self.d_state, self.n_operations),
        )
        nn.init.zeros_(self.op_head[1].weight)
        nn.init.zeros_(self.op_head[1].bias)
        
        # Stable input injection (Parcae-style decay)
        self.input_injection_gate = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()

    def _init_weights(self):
        for module in self.state_init:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        workspace: torch.Tensor,  # (B, W, d_state) - compressed prompt/context
        operation_ids: Optional[torch.Tensor] = None,  # (B, T) - hard op targets
        operation_soft: Optional[torch.Tensor] = None,  # (B, T, n_ops) - soft ops
        n_steps: Optional[int] = None,
        initial_state: Optional[torch.Tensor] = None,  # (B, d_state) for carry
    ) -> StateTransitionOutput:
        """
        Run the state transition loop.
        
        Args:
            workspace: Compressed prompt/context representation
            operation_ids: Hard operation IDs for each step
            operation_soft: Soft operation weights for each step
            n_steps: Number of transition steps (defaults to self.n_steps)
            initial_state: Optional carry state from previous call
        
        Returns:
            StateTransitionOutput with trajectory, logits, and telemetry
        """
        b, w, d = workspace.shape
        n_steps = n_steps or self.n_steps
        
        # Initial state: mean-pool workspace + step embedding
        z = workspace.mean(dim=1)  # (B, d_state)
        z = self.state_init(z)
        
        if initial_state is not None:
            # Blend with carry state (Parcae-style stable injection)
            gamma = torch.sigmoid(self.input_injection_gate)
            z = gamma * z + (1 - gamma) * initial_state
        
        # Add step 0 embedding
        step0 = self.step_embed.weight[0:1]  # (1, d_state)
        z = z + step0
        
        # Store trajectory
        state_trajectory = [z]
        state_norms = [z.pow(2).mean(dim=-1).sqrt()]
        
        # Transition loop
        transition_norms = []
        state_cosines = []
        
        for t in range(n_steps):
            prev_z = z
            
            # Get operation for this step
            if operation_ids is not None and t < operation_ids.shape[1]:
                op_t = operation_ids[:, t]
            elif operation_soft is not None and t < operation_soft.shape[1]:
                op_t_soft = operation_soft[:, t, :]
                op_t = None  # Will use soft weights
            else:
                # Default: use operation 0 (identity-like)
                op_t = workspace.new_zeros(b, dtype=torch.long)
                op_t_soft = None
            
            # Apply transition
            z = self.transition(
                z,
                op_ids=op_t,
                op_soft=op_t_soft if op_t is None else None,
            )
            
            # Add step conditioning
            step_idx = min(t + 1, self.n_steps)
            step_emb = self.step_embed.weight[step_idx:step_idx + 1]
            z = z + step_emb
            
            # Stable input injection (keep connection to original workspace)
            gamma = torch.sigmoid(self.input_injection_gate)
            workspace_mean = workspace.mean(dim=1)
            z = gamma * z + (1 - gamma) * self.state_init(workspace_mean)
            
            # Store
            state_trajectory.append(z)
            state_norms.append(z.pow(2).mean(dim=-1).sqrt())
            
            # Transition norm (for telemetry)
            delta = z - prev_z
            transition_norms.append(delta.pow(2).mean(dim=-1).sqrt())
            
            # Cosine similarity between consecutive states
            cos = torch.sum(prev_z * z, dim=-1) / (
                prev_z.pow(2).sum(dim=-1).sqrt() * z.pow(2).sum(dim=-1).sqrt() + 1e-8
            )
            state_cosines.append(cos)
        
        # Stack trajectory: (B, T+1, d_state)
        trajectory = torch.stack(state_trajectory, dim=1)
        
        # State digit logits at each step: (B, T+1, 10)
        state_digit_logits = self.state_readout(trajectory)
        
        # Answer logits from FINAL state only: (B, 10)
        answer_logits = self.answer_head(state_trajectory[-1])
        
        # Operation logits at each step (optional)
        operation_logits = None
        if len(state_trajectory) > 1:
            # Use states 1..T for operation prediction
            op_states = torch.stack(state_trajectory[1:], dim=1)
            operation_logits = self.op_head(op_states)
        
        return StateTransitionOutput(
            state_trajectory=trajectory,
            state_digit_logits=state_digit_logits,
            answer_logits=answer_logits,
            operation_logits=operation_logits,
            state_norms=torch.stack(state_norms, dim=1),
            transition_norms=torch.stack(transition_norms, dim=1),
            state_cosines=torch.stack(state_cosines, dim=1),
        )
