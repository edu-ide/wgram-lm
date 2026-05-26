"""
Qwen Backbone + State-Transition-First Core Integration.

Integrates the StateTransitionCore into the existing Qwen backbone
infrastructure. The Qwen model serves as a frozen (or partially trainable)
prompt compressor that feeds into the state transition core.

Architecture:
    input_ids -> Qwen backbone -> hidden states -> compressor -> workspace
    workspace -> StateTransitionCore -> state trajectory
    state trajectory -> answer head -> answer logits

The answer flows ONLY through the state path. The Qwen logits are used
only as a baseline for causality gating.
"""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import nn

from .config import QTRMConfig
from .core import QTRMRecursiveCore
from .norm import RMSNorm
from .state_transition_core import (
    N_OPERATIONS,
    StateReadoutHead,
    StateTransitionCore,
    HybridStateTransitionCore,
    StateTransitionOutput,
)
from .qwen_backbone_qtrm import (
    QwenBackboneQTRMReport,
    _config_int,
    _find_qwen_text_model,
    _find_ouro_text_model,
    _normalise_layer_indices,
)


class QwenBackboneStateTransition(nn.Module):
    """
    Qwen backbone with state-transition-first core.
    
    The Qwen model provides the prompt compressor and baseline logits.
    The state transition core is the PRIMARY reasoning mechanism where
    intermediate states are the supervised target.
    
    Key properties:
    - Qwen is frozen (or partially trainable) as prompt compressor
    - State transition core is fully trainable
    - Answer flows ONLY through the final state (no donor bypass)
    - Qwen logits are used only for causality gating
    """
    
    def __init__(
        self,
        qwen_model: nn.Module,
        *,
        model_id: str = "",
        core_config: Optional[QTRMConfig] = None,
        max_seq_len: int = 512,
        freeze_qwen: bool = True,
        d_state: Optional[int] = None,
        n_operations: int = N_OPERATIONS,
        n_steps: int = 4,
        compressor_hidden_dim: Optional[int] = None,
        mandatory_core: bool = True,
        core_impl: str = "state_transition",
        workspace_pooling: str = "mean",
        recurrent_readout_pooling: str = "final",
        recurrent_readout_temperature: float = 1.0,
        transition_scale_init: float = 1.0,
        step_embedding_std: Optional[float] = None,
        state_update_schedule: str = "nested",
        latent_feedback_passes: int = 1,
        core_update: str = "mlp",
        correction_feedback: bool = False,
        correction_feedback_scale: float = 1.0,
        correction_feedback_gate_init_bias: float = -1.0,
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
        semantic_token_feedback: bool = False,
        semantic_token_feedback_scale: float = 0.0,
        semantic_token_feedback_temperature: float = 1.0,
        semantic_token_feedback_gate_init_bias: float = -2.0,
        semantic_token_feedback_score_mode: str = "cosine",
        semantic_token_feedback_teacher_forcing: float = 0.0,
        trajectory_reward_mode: str = "final",
        answer_path: str = "state_head",
        source_numeric_feature_dim: int = 0,
        typed_value_registers: bool = False,
        typed_value_update_scale: float = 0.25,
        typed_value_update_mode: str = "residual",
        typed_digit_registers: bool = False,
        typed_digit_register_digits: int = 6,
        typed_digit_update_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        self.model_id = str(model_id)
        self.mandatory_core = bool(mandatory_core)
        self.core_impl = str(core_impl)
        self.workspace_pooling = str(workspace_pooling)
        self.recurrent_readout_pooling = str(recurrent_readout_pooling)
        self.recurrent_readout_temperature = float(recurrent_readout_temperature)
        self.state_update_schedule = str(state_update_schedule)
        self.latent_feedback_passes = int(latent_feedback_passes)
        self.core_update = str(core_update)
        self.correction_feedback = bool(correction_feedback)
        self.correction_feedback_scale = float(correction_feedback_scale)
        self.stochastic_high_level_guidance = bool(stochastic_high_level_guidance)
        self.stochastic_high_level_scale = float(stochastic_high_level_scale)
        self.stochastic_high_level_eval = bool(stochastic_high_level_eval)
        self.stochastic_posterior_guidance = bool(stochastic_posterior_guidance)
        self.stochastic_transition_mode = str(stochastic_transition_mode)
        self.lattice_feedback_mode = str(lattice_feedback_mode)
        self.lattice_feedback_scale = float(lattice_feedback_scale)
        self.lattice_feedback_threshold = float(lattice_feedback_threshold)
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
        self.semantic_token_feedback = bool(semantic_token_feedback)
        self.semantic_token_feedback_scale = float(semantic_token_feedback_scale)
        self.semantic_token_feedback_temperature = float(semantic_token_feedback_temperature)
        self.semantic_token_feedback_gate_init_bias = float(semantic_token_feedback_gate_init_bias)
        self.semantic_token_feedback_score_mode = str(semantic_token_feedback_score_mode)
        self.semantic_token_feedback_teacher_forcing = float(semantic_token_feedback_teacher_forcing)
        self.trajectory_reward_mode = str(trajectory_reward_mode)
        self.answer_path = str(answer_path)
        self.source_numeric_feature_dim = int(source_numeric_feature_dim)
        self.typed_value_registers = bool(typed_value_registers)
        self.typed_value_update_scale = float(typed_value_update_scale)
        self.typed_value_update_mode = str(typed_value_update_mode)
        self.typed_digit_registers = bool(typed_digit_registers)
        self.typed_digit_register_digits = int(typed_digit_register_digits)
        self.typed_digit_update_scale = float(typed_digit_update_scale)

        if self.core_impl not in {"state_transition", "hybrid_state_transition"}:
            raise ValueError(
                f"only core_impl in {{'state_transition', 'hybrid_state_transition'}} supported, got: {self.core_impl}"
            )
        if self.answer_path not in {"state_head", "lm_head"}:
            raise ValueError(f"answer_path must be one of {{'state_head', 'lm_head'}}, got: {self.answer_path}")
        if self.workspace_pooling not in {"mean", "last", "attention", "sequence", "none"}:
            raise ValueError(
                "workspace_pooling must be one of {'mean', 'last', 'attention', 'sequence', 'none'}, "
                f"got: {self.workspace_pooling}"
            )
        if self.recurrent_readout_pooling not in {"final", "mean", "attention", "sharp_attention", "hybrid_gate"}:
            raise ValueError(
                "recurrent_readout_pooling must be one of "
                "{'final', 'mean', 'attention', 'sharp_attention', 'hybrid_gate'}, "
                f"got: {self.recurrent_readout_pooling}"
            )
        if self.trajectory_reward_mode not in {"final", "rich"}:
            raise ValueError("trajectory_reward_mode must be one of {'final', 'rich'}")
        if self.recurrent_readout_temperature <= 0:
            raise ValueError("recurrent_readout_temperature must be > 0")
        if self.latent_feedback_passes < 1:
            raise ValueError("latent_feedback_passes must be >= 1")
        if self.core_update not in {"mlp", "mini_gated_delta"}:
            raise ValueError(f"core_update must be one of {{'mlp', 'mini_gated_delta'}}, got: {self.core_update}")
        if self.source_numeric_feature_dim < 0:
            raise ValueError("source_numeric_feature_dim must be >= 0")
        if self.typed_value_registers and self.source_numeric_feature_dim <= 0:
            raise ValueError("typed_value_registers requires source_numeric_feature_dim > 0")
        if self.typed_digit_registers and self.source_numeric_feature_dim <= 0:
            raise ValueError("typed_digit_registers requires source_numeric_feature_dim > 0")
        if self.typed_value_registers and self.core_impl != "state_transition":
            raise ValueError("typed_value_registers are currently supported only by core_impl='state_transition'")
        if self.typed_digit_registers and self.core_impl != "state_transition":
            raise ValueError("typed_digit_registers are currently supported only by core_impl='state_transition'")
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
        
        # Get Qwen dimensions
        hidden_size = _config_int(self.qwen.config, "hidden_size", 2048)
        self.hidden_size = hidden_size
        self.d_state = d_state or hidden_size
        self.n_operations = n_operations
        self.n_steps = n_steps
        
        # Freeze Qwen if requested
        if bool(freeze_qwen):
            for parameter in self.qwen.parameters():
                parameter.requires_grad_(False)
            self.qwen.eval()
        
        # Compressor: Qwen hidden states -> workspace for state transition
        # This compresses the sequence of hidden states into a fixed-size
        # workspace representation that the state transition core operates on
        compressor_dim = compressor_hidden_dim or self.d_state
        self.compressor = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, compressor_dim),
            nn.GELU(),
            nn.Linear(compressor_dim, self.d_state),
            RMSNorm(self.d_state),
        )
        self.workspace_attention = nn.Linear(hidden_size, 1)
        self.recurrent_readout_attention = nn.Linear(self.d_state, 1)
        self.recurrent_readout_gate = nn.Linear(self.d_state * 2, 1)
        self.thought_to_lm = nn.Identity() if self.d_state == hidden_size else nn.Linear(self.d_state, hidden_size)
        self.semantic_lm_to_thought = nn.Identity() if self.d_state == hidden_size else nn.Linear(hidden_size, self.d_state)
        self.correction_error_head = nn.Linear(self.d_state, 10)
        self.correction_error_embed = nn.Embedding(10, self.d_state)
        self.correction_confidence_proj = nn.Linear(2, self.d_state)
        self.correction_feedback_gate = nn.Linear(self.d_state * 2, 1)
        self.correction_feedback_norm = RMSNorm(self.d_state)
        self.trajectory_reward_head = nn.Linear(self.d_state, 1)
        self.trajectory_reward_rich_head = nn.Sequential(
            RMSNorm(self.d_state * 4 + 4),
            nn.Linear(self.d_state * 4 + 4, self.d_state),
            nn.GELU(),
            nn.Linear(self.d_state, 1),
        )
        self.source_numeric_proj = (
            nn.Sequential(
                nn.Linear(self.source_numeric_feature_dim, self.d_state, bias=False),
                RMSNorm(self.d_state),
            )
            if self.source_numeric_feature_dim > 0
            else None
        )
        
        # Initialize compressor with near-identity mapping
        for module in self.compressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.workspace_attention.weight)
        nn.init.zeros_(self.workspace_attention.bias)
        nn.init.zeros_(self.recurrent_readout_attention.weight)
        nn.init.zeros_(self.recurrent_readout_attention.bias)
        nn.init.zeros_(self.recurrent_readout_gate.weight)
        nn.init.zeros_(self.recurrent_readout_gate.bias)
        if isinstance(self.thought_to_lm, nn.Linear):
            nn.init.xavier_uniform_(self.thought_to_lm.weight)
            nn.init.zeros_(self.thought_to_lm.bias)
        if isinstance(self.semantic_lm_to_thought, nn.Linear):
            nn.init.xavier_uniform_(self.semantic_lm_to_thought.weight)
            nn.init.zeros_(self.semantic_lm_to_thought.bias)
        nn.init.xavier_uniform_(self.correction_error_head.weight)
        nn.init.zeros_(self.correction_error_head.bias)
        nn.init.normal_(self.correction_error_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.correction_confidence_proj.weight)
        nn.init.zeros_(self.correction_confidence_proj.bias)
        nn.init.zeros_(self.correction_feedback_gate.weight)
        nn.init.constant_(self.correction_feedback_gate.bias, float(correction_feedback_gate_init_bias))
        nn.init.zeros_(self.trajectory_reward_head.weight)
        nn.init.zeros_(self.trajectory_reward_head.bias)
        for module in self.trajectory_reward_rich_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        rich_last = self.trajectory_reward_rich_head[-1]
        if isinstance(rich_last, nn.Linear):
            nn.init.zeros_(rich_last.weight)
            nn.init.zeros_(rich_last.bias)
        if self.source_numeric_proj is not None:
            source_numeric_linear = self.source_numeric_proj[0]
            if isinstance(source_numeric_linear, nn.Linear):
                nn.init.xavier_uniform_(source_numeric_linear.weight)
        
        # State transition core
        qtrm_cfg = core_config or QTRMConfig(
            d_model=self.d_state,
            num_actions=n_operations,
            outer_steps=n_steps,
        )
        if self.core_impl == "hybrid_state_transition":
            self.core = HybridStateTransitionCore(
                cfg=qtrm_cfg,
                d_state=self.d_state,
                n_operations=n_operations,
                n_steps=n_steps,
                transition_scale_init=transition_scale_init,
                step_embedding_std=step_embedding_std,
                update_schedule=self.state_update_schedule,
                continuous_time=self.continuous_time,
            )
        else:
            self.core = StateTransitionCore(
                cfg=qtrm_cfg,
                d_state=self.d_state,
                n_operations=n_operations,
                n_steps=n_steps,
                transition_scale_init=transition_scale_init,
                step_embedding_std=step_embedding_std,
                update_schedule=self.state_update_schedule,
                core_update=self.core_update,
                stochastic_high_level_guidance=self.stochastic_high_level_guidance,
                stochastic_high_level_scale=stochastic_high_level_scale,
                stochastic_high_level_min_std=stochastic_high_level_min_std,
                stochastic_high_level_max_std=stochastic_high_level_max_std,
                stochastic_high_level_eval=stochastic_high_level_eval,
                stochastic_posterior_guidance=stochastic_posterior_guidance,
                stochastic_transition_mode=self.stochastic_transition_mode,
                lattice_feedback_mode=self.lattice_feedback_mode,
                lattice_feedback_scale=self.lattice_feedback_scale,
                lattice_feedback_threshold=self.lattice_feedback_threshold,
                workspace_cross_attention=self.workspace_pooling in {"sequence", "none"},
                operation_arg_conditioning=self.operation_arg_conditioning,
                continuous_time=self.continuous_time,
                layerscale=self.layerscale,
                layerscale_init=self.layerscale_init,
                gate_type=self.gate_type,
                gate_bias_init=self.gate_bias_init,
                working_register_enabled=self.working_register_enabled,
                working_register_slots=self.working_register_slots,
                working_register_update_scale=self.working_register_update_scale,
                working_register_feedback_scale=self.working_register_feedback_scale,
                working_register_gate_init_bias=self.working_register_gate_init_bias,
                working_register_summary_mode=self.working_register_summary_mode,
                working_register_role_conditioning=self.working_register_role_conditioning,
                working_register_role_anchor_scale=self.working_register_role_anchor_scale,
                working_register_update_mode=self.working_register_update_mode,
                working_register_source_attention=self.working_register_source_attention,
                working_register_source_attention_scale=self.working_register_source_attention_scale,
                typed_value_registers=self.typed_value_registers,
                source_numeric_feature_dim=self.source_numeric_feature_dim,
                typed_value_update_scale=self.typed_value_update_scale,
                typed_value_update_mode=self.typed_value_update_mode,
                typed_digit_registers=self.typed_digit_registers,
                typed_digit_register_digits=self.typed_digit_register_digits,
                typed_digit_update_scale=self.typed_digit_update_scale,
                semantic_token_feedback=self.semantic_token_feedback,
                semantic_token_feedback_scale=self.semantic_token_feedback_scale,
                semantic_token_feedback_temperature=self.semantic_token_feedback_temperature,
                semantic_token_feedback_gate_init_bias=self.semantic_token_feedback_gate_init_bias,
                semantic_token_feedback_score_mode=self.semantic_token_feedback_score_mode,
                semantic_token_feedback_teacher_forcing=self.semantic_token_feedback_teacher_forcing,
            )
        
        # Core output normalization
        self.core_out_norm = RMSNorm(self.d_state)
        
        # Answer head: reads from the final state only
        self.answer_head = StateReadoutHead(self.d_state)
        
        # State readout for intermediate supervision. Share it with the core so
        # LDT-style candidate feedback and supervised state logits use the same
        # candidate-lattice projection.
        self.state_readout = self.core.state_readout
        
        # Store for label token IDs
        self.label_token_ids = None

    def _attention_recurrent_readout(self, trajectory: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Read transition states with learned attention and return state plus weights."""
        states = trajectory[:, 1:, :]
        attn_input = states.to(self.recurrent_readout_attention.weight.dtype)
        scores = self.recurrent_readout_attention(attn_input).squeeze(-1)
        scores = scores / float(temperature)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1), weights.squeeze(-1)

    def _pool_recurrent_readout(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Read answer state from the recurrent trajectory."""
        if self.recurrent_readout_pooling == "final":
            return trajectory[:, -1, :]
        states = trajectory[:, 1:, :]
        if self.recurrent_readout_pooling == "mean":
            return states.mean(dim=1)
        if self.recurrent_readout_pooling == "attention":
            readout, _ = self._attention_recurrent_readout(trajectory, temperature=1.0)
            return readout
        if self.recurrent_readout_pooling == "sharp_attention":
            readout, _ = self._attention_recurrent_readout(
                trajectory,
                temperature=self.recurrent_readout_temperature,
            )
            return readout
        attention_state, _ = self._attention_recurrent_readout(
            trajectory,
            temperature=self.recurrent_readout_temperature,
        )
        final_state = trajectory[:, -1, :]
        gate_input = torch.cat([final_state, attention_state], dim=-1).to(self.recurrent_readout_gate.weight.dtype)
        gate = torch.sigmoid(self.recurrent_readout_gate(gate_input)).to(final_state.dtype)
        return gate * attention_state + (1.0 - gate) * final_state

    def _recurrent_readout_telemetry(self, trajectory: torch.Tensor) -> dict[str, torch.Tensor]:
        """Small observability hook for adaptive readout experiments."""
        result: dict[str, torch.Tensor] = {}
        if self.recurrent_readout_pooling in {"attention", "sharp_attention", "hybrid_gate"}:
            temperature = 1.0 if self.recurrent_readout_pooling == "attention" else self.recurrent_readout_temperature
            _, weights = self._attention_recurrent_readout(trajectory, temperature=temperature)
            clamped = weights.clamp_min(1e-8)
            result["qtrm_readout_attention_entropy"] = -(clamped * clamped.log()).sum(dim=-1)
        if self.recurrent_readout_pooling == "hybrid_gate":
            attention_state, _ = self._attention_recurrent_readout(
                trajectory,
                temperature=self.recurrent_readout_temperature,
            )
            final_state = trajectory[:, -1, :]
            gate_input = torch.cat([final_state, attention_state], dim=-1).to(self.recurrent_readout_gate.weight.dtype)
            result["qtrm_readout_gate"] = torch.sigmoid(self.recurrent_readout_gate(gate_input)).squeeze(-1)
        return result

    def _answer_correction_feedback(
        self,
        readout_state: torch.Tensor,
        answer_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict an answer-delta token and turn it into a second-pass correction state."""
        normalized = self.core_out_norm(readout_state)
        error_logits = self.correction_error_head(normalized)
        error_probs = torch.softmax(error_logits.float(), dim=-1).to(readout_state.dtype)
        error_vector = error_probs @ self.correction_error_embed.weight.to(readout_state.dtype)

        answer_probs = torch.softmax(answer_logits.float(), dim=-1)
        top2 = torch.topk(answer_probs, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)
        entropy = -(answer_probs * answer_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(answer_probs.size(-1)), device=answer_probs.device))
        confidence_features = torch.cat([entropy, 1.0 - margin], dim=-1).to(readout_state.dtype)
        confidence_vector = self.correction_confidence_proj(confidence_features).to(readout_state.dtype)

        correction_vector = self.correction_feedback_norm(error_vector + confidence_vector)
        gate_input = torch.cat([readout_state, correction_vector], dim=-1).to(self.correction_feedback_gate.weight.dtype)
        gate = torch.sigmoid(self.correction_feedback_gate(gate_input)).to(readout_state.dtype)
        correction_state = readout_state + float(self.correction_feedback_scale) * gate * correction_vector
        return correction_state, error_logits, gate.squeeze(-1)
    
    def set_label_token_ids(self, token_ids: list[int]) -> None:
        """Set label token IDs for digit choices."""
        self.label_token_ids = torch.tensor(token_ids, dtype=torch.long)

    def _lm_hidden_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Project a recurrent thought state into Qwen's LM-head hidden language."""
        lm_head = getattr(self.qwen, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("answer_path='lm_head' requires qwen.lm_head")
        return self.thought_to_lm(self.core_out_norm(state)).to(lm_head.weight.dtype)

    def _lm_head_logits_from_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a thought state through Qwen's LM head and gather label logits."""
        lm_head = getattr(self.qwen, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("answer_path='lm_head' requires qwen.lm_head")
        thought = self._lm_hidden_from_state(state)
        vocab_logits = lm_head(thought)
        if self.label_token_ids is None:
            raise RuntimeError("answer_path='lm_head' requires label_token_ids for 10-way synthetic scoring")
        label_ids = self.label_token_ids.to(device=vocab_logits.device, dtype=torch.long)
        return vocab_logits.index_select(dim=-1, index=label_ids), vocab_logits

    def _semantic_feedback_basis(self) -> Optional[torch.Tensor]:
        """Return LM-head label-token directions projected into thought space."""
        if not self.semantic_token_feedback or self.semantic_token_feedback_scale == 0:
            return None
        lm_head = getattr(self.qwen, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("semantic token feedback requires qwen.lm_head")
        if self.label_token_ids is None:
            raise RuntimeError("semantic token feedback requires label_token_ids")
        label_ids = self.label_token_ids.to(device=lm_head.weight.device, dtype=torch.long)
        label_basis = lm_head.weight.index_select(0, label_ids)
        return self.semantic_lm_to_thought(label_basis)

    def _answer_logits_from_state(self, state: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.answer_path == "lm_head":
            return self._lm_head_logits_from_state(state)
        return self.answer_head(self.core_out_norm(state)), None

    def compute_trajectory_reward_logits(
        self,
        *,
        state_trajectory: torch.Tensor,
        readout_state: torch.Tensor,
        answer_logits: torch.Tensor,
        detach_state: bool = False,
    ) -> torch.Tensor:
        """Score a sampled thought trajectory for GRAM/LPRM selection.

        The default mode preserves the legacy final-state scorer. The rich mode
        gives the selector the same kind of evidence a human judge would use:
        final thought, average path, start-to-end movement, sharp readout state,
        and answer confidence statistics.
        """
        trajectory = state_trajectory.detach() if detach_state else state_trajectory
        readout = readout_state.detach() if detach_state else readout_state
        logits = answer_logits.detach() if detach_state else answer_logits
        normalized_final = self.core_out_norm(readout)
        if self.trajectory_reward_mode == "final":
            return self.trajectory_reward_head(normalized_final).squeeze(-1)

        step_states = trajectory[:, 1:, :] if trajectory.size(1) > 1 else trajectory
        mean_state = self.core_out_norm(step_states.mean(dim=1))
        delta_state = self.core_out_norm(readout - trajectory[:, 0, :])
        attention_state, _ = self._attention_recurrent_readout(
            trajectory,
            temperature=max(float(self.recurrent_readout_temperature), 1e-6),
        )
        attention_state = self.core_out_norm(attention_state)

        probs = torch.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            margin = top2[:, :1]
        else:
            margin = top2[:, :1] - top2[:, 1:2]
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(probs.size(-1)), device=probs.device))
        max_prob = top2[:, :1]
        logit_scale = logits.float().std(dim=-1, keepdim=True)
        scalar_features = torch.cat([margin, entropy, max_prob, logit_scale], dim=-1).to(normalized_final.dtype)

        reward_features = torch.cat(
            [
                normalized_final,
                mean_state.to(normalized_final.dtype),
                delta_state.to(normalized_final.dtype),
                attention_state.to(normalized_final.dtype),
                scalar_features,
            ],
            dim=-1,
        )
        return self.trajectory_reward_rich_head(reward_features).squeeze(-1)
    
    def _compress_workspace(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compress Qwen hidden states into workspace representation.
        
        Args:
            hidden_states: Qwen hidden states (B, T, H)
            attention_mask: Attention mask (B, T)
        
        Returns:
            workspace: Compressed workspace (B, W, d_state)
        """
        if self.workspace_pooling in {"sequence", "none"}:
            workspace = self.compressor(hidden_states)
            if attention_mask is not None:
                workspace = workspace * attention_mask.unsqueeze(-1).to(workspace.dtype)
            return workspace

        if self.workspace_pooling == "last":
            if attention_mask is not None:
                lengths = attention_mask.to(torch.long).sum(dim=1).clamp(min=1)
                row = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[row, lengths.to(hidden_states.device) - 1]
            else:
                pooled = hidden_states[:, -1, :]
        elif self.workspace_pooling == "attention":
            attn_input = hidden_states.to(self.workspace_attention.weight.dtype)
            scores = self.workspace_attention(attn_input).squeeze(-1)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.to(torch.bool).logical_not(), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(dim=1)
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)
        
        # Compress to state space
        workspace = self.compressor(pooled.unsqueeze(1))  # (B, 1, d_state)
        return workspace

    def _workspace_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.workspace_pooling not in {"sequence", "none"}:
            return None
        return attention_mask

    def _append_source_numeric_workspace(
        self,
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor],
        source_numeric_features: Optional[torch.Tensor],
        source_numeric_feature_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if source_numeric_features is None:
            return workspace, workspace_attention_mask
        if self.source_numeric_proj is None:
            raise ValueError(
                "source_numeric_features require source_numeric_feature_dim > 0 "
                "when building QwenBackboneStateTransition"
            )
        if source_numeric_features.ndim != 3:
            raise ValueError("source_numeric_features must have shape [batch, slots, features]")
        if int(source_numeric_features.size(0)) != int(workspace.size(0)):
            raise ValueError("source_numeric_features batch must match workspace batch")
        if int(source_numeric_features.size(-1)) != self.source_numeric_feature_dim:
            raise ValueError(
                "source_numeric_features last dimension must match source_numeric_feature_dim"
            )
        source_slots = self.source_numeric_proj(
            source_numeric_features.to(
                device=workspace.device,
                dtype=self.source_numeric_proj[0].weight.dtype,
            )
        ).to(workspace.dtype)
        if source_numeric_feature_mask is None:
            source_mask = source_slots.new_ones(source_slots.size(0), source_slots.size(1))
        else:
            if source_numeric_feature_mask.ndim != 2:
                raise ValueError("source_numeric_feature_mask must have shape [batch, slots]")
            if tuple(source_numeric_feature_mask.shape) != tuple(source_slots.shape[:2]):
                raise ValueError("source_numeric_feature_mask must match source numeric slots")
            source_mask = source_numeric_feature_mask.to(device=workspace.device, dtype=workspace.dtype)
        source_slots = source_slots * source_mask.unsqueeze(-1)
        next_workspace = torch.cat([workspace, source_slots], dim=1)
        if workspace_attention_mask is None:
            if self.workspace_pooling in {"sequence", "none"}:
                base_mask = workspace.new_ones(workspace.size(0), workspace.size(1))
            else:
                base_mask = workspace.new_ones(workspace.size(0), workspace.size(1))
        else:
            base_mask = workspace_attention_mask.to(device=workspace.device, dtype=source_mask.dtype)
        return next_workspace, torch.cat([base_mask, source_mask], dim=1)

    @staticmethod
    def _append_workspace_state(
        workspace: torch.Tensor,
        workspace_attention_mask: Optional[torch.Tensor],
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        next_workspace = torch.cat([workspace, state.unsqueeze(1)], dim=1)
        if workspace_attention_mask is None:
            return next_workspace, None
        extra = torch.ones(
            workspace_attention_mask.size(0),
            1,
            device=workspace_attention_mask.device,
            dtype=workspace_attention_mask.dtype,
        )
        return next_workspace, torch.cat([workspace_attention_mask, extra], dim=1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        force_core_off: bool = False,
        operation_ids: Optional[torch.Tensor] = None,
        operation_arg_ids: Optional[torch.Tensor] = None,
        initial_labels: Optional[torch.Tensor] = None,
        operation_soft: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        posterior_labels: Optional[torch.Tensor] = None,
        source_numeric_features: Optional[torch.Tensor] = None,
        source_numeric_feature_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (B, T)
            attention_mask: Attention mask (B, T)
            force_core_off: If True, return Qwen logits directly (baseline)
            operation_ids: Hard operation IDs for each step (B, T_steps)
            operation_soft: Soft operation weights (B, T_steps, n_ops)
            n_steps: Number of transition steps
            return_dict: Return as dict or tuple
        
        Returns:
            Dict with logits, state trajectory, and telemetry
        """
        # Qwen backbone forward (frozen or partially trainable)
        with torch.set_grad_enabled(not all(
            not p.requires_grad for p in self.qwen.parameters()
        )):
            qwen_outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                **kwargs,
            )
        
        # Get last hidden state
        if hasattr(qwen_outputs, 'hidden_states') and qwen_outputs.hidden_states:
            hidden_states = qwen_outputs.hidden_states[-1]
        elif hasattr(qwen_outputs, 'last_hidden_state'):
            hidden_states = qwen_outputs.last_hidden_state
        else:
            hidden_states = qwen_outputs[0] if isinstance(qwen_outputs, tuple) else qwen_outputs
        
        # Baseline logits (for causality gating)
        baseline_logits = None
        if hasattr(qwen_outputs, 'logits') and qwen_outputs.logits is not None:
            baseline_logits = qwen_outputs.logits
        elif hasattr(self.qwen, 'lm_head'):
            # Some Qwen models share embeddings and lm_head
            baseline_logits = self.qwen.lm_head(hidden_states)
        
        # Force core off returns baseline
        if bool(force_core_off):
            if not return_dict:
                return (baseline_logits,)
            return {
                "logits": baseline_logits,
                "qtrm_core_step_states": None,
                "state_digit_logits": None,
                "answer_logits": None,
                "operation_logits": None,
                "baseline_logits": baseline_logits,
            }

        # Compress to workspace
        workspace = self._compress_workspace(hidden_states, attention_mask)
        workspace_attention_mask = self._workspace_attention_mask(attention_mask)
        workspace, workspace_attention_mask = self._append_source_numeric_workspace(
            workspace,
            workspace_attention_mask,
            source_numeric_features,
            source_numeric_feature_mask,
        )
        semantic_feedback_basis = self._semantic_feedback_basis()

        # State transition core. Optional correction feedback predicts an
        # answer-delta token after the first latent pass and conditions a
        # second core pass through the normal answer path.
        core_workspace = workspace
        core_workspace_attention_mask = workspace_attention_mask
        core_output = None
        readout_state = None
        first_answer_logits = None
        correction_error_logits = None
        correction_gate = None
        for feedback_pass in range(self.latent_feedback_passes):
            core_output = self.core(
                workspace=core_workspace,
                workspace_attention_mask=core_workspace_attention_mask,
                operation_ids=operation_ids,
                operation_arg_ids=operation_arg_ids,
                initial_labels=initial_labels,
                operation_soft=operation_soft,
                n_steps=n_steps,
                initial_state=readout_state if feedback_pass > 0 else None,
                posterior_labels=posterior_labels,
                semantic_feedback_basis=semantic_feedback_basis,
                source_numeric_features=source_numeric_features,
                source_numeric_feature_mask=source_numeric_feature_mask,
            )
            readout_state = self._pool_recurrent_readout(core_output.state_trajectory)
            if feedback_pass + 1 < self.latent_feedback_passes:
                core_workspace, core_workspace_attention_mask = self._append_workspace_state(
                    workspace,
                    workspace_attention_mask,
                    readout_state,
        )

        if self.correction_feedback:
            first_answer_logits, _ = self._answer_logits_from_state(readout_state)
            correction_state, correction_error_logits, correction_gate = self._answer_correction_feedback(
                readout_state=readout_state,
                answer_logits=first_answer_logits,
            )
            correction_workspace, correction_workspace_attention_mask = self._append_workspace_state(
                workspace,
                workspace_attention_mask,
                correction_state,
            )
            core_output = self.core(
                workspace=correction_workspace,
                workspace_attention_mask=correction_workspace_attention_mask,
                operation_ids=operation_ids,
                operation_arg_ids=operation_arg_ids,
                initial_labels=initial_labels,
                operation_soft=operation_soft,
                n_steps=n_steps,
                initial_state=correction_state,
                posterior_labels=posterior_labels,
                semantic_feedback_basis=semantic_feedback_basis,
                source_numeric_features=source_numeric_features,
                source_numeric_feature_mask=source_numeric_feature_mask,
            )
            readout_state = self._pool_recurrent_readout(core_output.state_trajectory)

        assert core_output is not None
        assert readout_state is not None

        # Normalize final state for answer head
        readout_telemetry = self._recurrent_readout_telemetry(core_output.state_trajectory)
        normalized_final = self.core_out_norm(readout_state)

        # Answer logits from the configured thought-state path.
        answer_logits, lm_answer_logits = self._answer_logits_from_state(readout_state)
        trajectory_reward_logits = self.compute_trajectory_reward_logits(
            state_trajectory=core_output.state_trajectory,
            readout_state=readout_state,
            answer_logits=answer_logits,
        )
        
        # State digit logits for intermediate supervision
        state_digit_logits = self.state_readout(core_output.state_trajectory)
        
        # Operation logits
        operation_logits = core_output.operation_logits
        
        if not return_dict:
            return (answer_logits, state_digit_logits, operation_logits)
        
        return {
            "logits": answer_logits,
            "qtrm_core_step_states": core_output.state_trajectory,
            "qtrm_workspace": workspace,
            "qtrm_workspace_attention_mask": workspace_attention_mask,
            "state_digit_logits": state_digit_logits,
            "answer_logits": answer_logits,
            "qtrm_answer_path": self.answer_path,
            "qtrm_lm_answer_logits": lm_answer_logits,
            "qtrm_trajectory_reward_logits": trajectory_reward_logits,
            "qtrm_first_answer_logits": first_answer_logits,
            "qtrm_correction_error_logits": correction_error_logits,
            "qtrm_correction_gate": correction_gate,
            "qtrm_readout_state": readout_state,
            "operation_logits": operation_logits,
            "state_norms": core_output.state_norms,
            "transition_norms": core_output.transition_norms,
            "state_cosines": core_output.state_cosines,
            "qtrm_stochastic_mu_norms": core_output.stochastic_mu_norms,
            "qtrm_stochastic_std_means": core_output.stochastic_std_means,
            "qtrm_stochastic_noise_norms": core_output.stochastic_noise_norms,
            "qtrm_stochastic_posterior_kls": core_output.stochastic_posterior_kls,
            "qtrm_working_register_norms": core_output.working_register_norms,
            "qtrm_working_register_gate_means": core_output.working_register_gate_means,
            "qtrm_working_register_role_cosines": core_output.working_register_role_cosines,
            "qtrm_working_register_trajectory": core_output.working_register_trajectory,
            "qtrm_typed_value_register_norms": core_output.typed_value_register_norms,
            "qtrm_typed_value_register_gate_means": core_output.typed_value_register_gate_means,
            "qtrm_typed_value_register_trajectory": core_output.typed_value_register_trajectory,
            "qtrm_typed_digit_register_norms": core_output.typed_digit_register_norms,
            "qtrm_typed_digit_register_gate_means": core_output.typed_digit_register_gate_means,
            "qtrm_typed_digit_register_trajectory": core_output.typed_digit_register_trajectory,
            "qtrm_semantic_token_feedback_gate_means": core_output.semantic_token_feedback_gate_means,
            "qtrm_semantic_token_feedback_entropies": core_output.semantic_token_feedback_entropies,
            "baseline_logits": baseline_logits,
            "qtrm_latent_feedback_passes": self.latent_feedback_passes,
            **readout_telemetry,
        }
    
    def freeze_qwen_parameters(self) -> None:
        """Freeze all Qwen parameters."""
        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)
        self.qwen.eval()
    
    def set_qwen_partial_trainable(
        self,
        *,
        layer_indices: Optional[Sequence[int]] = None,
        train_embeddings: bool = False,
        train_lm_head: bool = False,
        train_final_norm: bool = False,
    ) -> dict[str, object]:
        """Freeze Qwen then unfreeze only specified parts."""
        self.freeze_qwen_parameters()
        text_model = _find_qwen_text_model(self.qwen)
        layers = getattr(text_model, "layers", None)
        
        if layers is None:
            return {"unfrozen_layers": []}
        
        selected = _normalise_layer_indices(
            layer_indices,
            num_layers=len(layers),
            default=(),
        ) if layer_indices else ()
        
        for index in selected:
            for parameter in layers[int(index)].parameters():
                parameter.requires_grad_(True)
        
        if bool(train_embeddings):
            embed = getattr(text_model, "embed_tokens", None)
            if embed is not None:
                for parameter in embed.parameters():
                    parameter.requires_grad_(True)
        
        if bool(train_lm_head):
            lm_head = getattr(self.qwen, "lm_head", None)
            if lm_head is not None:
                for parameter in lm_head.parameters():
                    parameter.requires_grad_(True)
        
        if bool(train_final_norm):
            norm = getattr(text_model, "norm", None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad_(True)
        
        return {
            "unfrozen_layers": list(selected),
            "train_embeddings": train_embeddings,
            "train_lm_head": train_lm_head,
            "train_final_norm": train_final_norm,
        }
    
    def get_report(self) -> QwenBackboneQTRMReport:
        """Get model report."""
        qwen_params = sum(p.numel() for p in self.qwen.parameters())
        qwen_trainable = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        core_params = sum(p.numel() for p in self.parameters() if not str(p).startswith('qwen.'))
        core_trainable = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad and not str(p).startswith('qwen.')
        )
        
        return QwenBackboneQTRMReport(
            model_id=self.model_id,
            vocab_size=_config_int(self.qwen.config, "vocab_size", 152064),
            hidden_size=self.hidden_size,
            qwen_parameters=qwen_params,
            qwen_trainable_parameters=qwen_trainable,
            qtrm_parameters=core_params,
            qtrm_trainable_parameters=core_trainable,
            runtime_donor=False,
            integrated_qwen_backbone=True,
            standalone_graph=False,
            mandatory_core=self.mandatory_core,
            core_impl=self.core_impl,
            core_update=self.core_update,
        )


def build_qwen_state_transition_model(
    model_id: str,
    *,
    d_state: Optional[int] = None,
    n_operations: int = N_OPERATIONS,
    n_steps: int = 4,
    freeze_qwen: bool = True,
    max_seq_len: int = 512,
    workspace_pooling: str = "mean",
    recurrent_readout_pooling: str = "final",
    recurrent_readout_temperature: float = 1.0,
    transition_scale_init: float = 1.0,
    step_embedding_std: Optional[float] = None,
    state_update_schedule: str = "nested",
    latent_feedback_passes: int = 1,
    core_update: str = "mlp",
    correction_feedback: bool = False,
    correction_feedback_scale: float = 1.0,
    correction_feedback_gate_init_bias: float = -1.0,
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
    operation_arg_conditioning: bool = False,
    continuous_time: bool = False,
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
    semantic_token_feedback: bool = False,
    semantic_token_feedback_scale: float = 0.0,
    semantic_token_feedback_temperature: float = 1.0,
    semantic_token_feedback_gate_init_bias: float = -2.0,
    semantic_token_feedback_score_mode: str = "cosine",
    semantic_token_feedback_teacher_forcing: float = 0.0,
    trajectory_reward_mode: str = "final",
    answer_path: str = "state_head",
    source_numeric_feature_dim: int = 0,
    typed_value_registers: bool = False,
    typed_value_update_scale: float = 0.25,
    typed_value_update_mode: str = "residual",
    typed_digit_registers: bool = False,
    typed_digit_register_digits: int = 6,
    typed_digit_update_scale: float = 0.25,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> tuple[QwenBackboneStateTransition, Any]:
    """
    Build a Qwen backbone with state-transition-first core.
    
    Args:
        model_id: HuggingFace model ID or local path
        d_state: State dimension (defaults to Qwen hidden size)
        n_operations: Number of operations
        n_steps: Number of transition steps
        freeze_qwen: Whether to freeze Qwen backbone
        max_seq_len: Maximum sequence length
        workspace_pooling: How to compress Qwen hidden states: mean, last,
            attention, sequence, or none. sequence/none preserve token
            workspaces and enable recurrent cross-attention routing.
        recurrent_readout_pooling: How to read answer state from recurrent trajectory
        recurrent_readout_temperature: Temperature for sharp/hybrid trajectory attention
        transition_scale_init: Initial LayerScale multiplier for recurrent deltas
        step_embedding_std: Optional normal init std for recurrent step embeddings
        state_update_schedule: Recurrent update schedule: nested or two_stream
        latent_feedback_passes: Number of latent core passes with recurrent readout feedback
        core_update: Shared recurrent update: mlp or mini_gated_delta
        correction_feedback: Enable answer-delta conditioned correction pass
        lattice_feedback_mode: Candidate lattice feedback mode: none, soft, or threshold
        lattice_feedback_scale: Strength of recurrent candidate-lattice feedback
        lattice_feedback_threshold: Threshold for hard candidate elimination feedback
        operation_arg_conditioning: Add oracle digit-argument embeddings to
            operation vectors as an upper-bound diagnostic for operand routing
        working_register_enabled: Add a bounded typed working-memory bank to
            the normal recurrent thought path.
        semantic_token_feedback: Feed the thought state's own Qwen label-token
            belief back into the recurrent state at every solve step.
        trajectory_reward_mode: final uses the legacy final-state reward head;
            rich gives the selector trajectory and confidence features.
        answer_path: state_head uses the legacy 10-way state head; lm_head uses
            Qwen's LM head over the recurrent semantic thought state.
        source_numeric_feature_dim: Optional feature width for source-number
            slots appended to the token workspace.
        typed_value_registers: Add source numeric/list value slots to the
            recurrent working-register trajectory and update them each step.
        typed_value_update_mode: residual keeps the legacy single-gate residual
            value update; gated_delta uses separate retain/write/execute gates
            for typed value slots.
        typed_digit_registers: Add explicit digit-column plus carry-pocket
            slots to the same recurrent working-register trajectory.
        dtype: Model dtype
        device: Model device
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is required for Qwen backbone") from e
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model
    load_dtype = dtype or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Build state transition model
    state_model = QwenBackboneStateTransition(
        model,
        model_id=model_id,
        d_state=d_state,
        n_operations=n_operations,
        n_steps=n_steps,
        freeze_qwen=freeze_qwen,
        max_seq_len=max_seq_len,
        workspace_pooling=workspace_pooling,
        recurrent_readout_pooling=recurrent_readout_pooling,
        recurrent_readout_temperature=recurrent_readout_temperature,
        transition_scale_init=transition_scale_init,
        step_embedding_std=step_embedding_std,
        state_update_schedule=state_update_schedule,
        latent_feedback_passes=latent_feedback_passes,
        core_update=core_update,
        correction_feedback=correction_feedback,
        correction_feedback_scale=correction_feedback_scale,
        correction_feedback_gate_init_bias=correction_feedback_gate_init_bias,
        stochastic_high_level_guidance=stochastic_high_level_guidance,
        stochastic_high_level_scale=stochastic_high_level_scale,
        stochastic_high_level_min_std=stochastic_high_level_min_std,
        stochastic_high_level_max_std=stochastic_high_level_max_std,
        stochastic_high_level_eval=stochastic_high_level_eval,
        stochastic_posterior_guidance=stochastic_posterior_guidance,
        stochastic_transition_mode=stochastic_transition_mode,
        lattice_feedback_mode=lattice_feedback_mode,
        lattice_feedback_scale=lattice_feedback_scale,
        lattice_feedback_threshold=lattice_feedback_threshold,
        operation_arg_conditioning=operation_arg_conditioning,
        continuous_time=continuous_time,
        working_register_enabled=working_register_enabled,
        working_register_slots=working_register_slots,
        working_register_update_scale=working_register_update_scale,
        working_register_feedback_scale=working_register_feedback_scale,
        working_register_gate_init_bias=working_register_gate_init_bias,
        working_register_summary_mode=working_register_summary_mode,
        working_register_role_conditioning=working_register_role_conditioning,
        working_register_role_anchor_scale=working_register_role_anchor_scale,
        working_register_update_mode=working_register_update_mode,
        working_register_source_attention=working_register_source_attention,
        working_register_source_attention_scale=working_register_source_attention_scale,
        semantic_token_feedback=semantic_token_feedback,
        semantic_token_feedback_scale=semantic_token_feedback_scale,
        semantic_token_feedback_temperature=semantic_token_feedback_temperature,
        semantic_token_feedback_gate_init_bias=semantic_token_feedback_gate_init_bias,
        semantic_token_feedback_score_mode=semantic_token_feedback_score_mode,
        semantic_token_feedback_teacher_forcing=semantic_token_feedback_teacher_forcing,
        trajectory_reward_mode=trajectory_reward_mode,
        answer_path=answer_path,
        source_numeric_feature_dim=source_numeric_feature_dim,
        typed_value_registers=typed_value_registers,
        typed_value_update_scale=typed_value_update_scale,
        typed_value_update_mode=typed_value_update_mode,
        typed_digit_registers=typed_digit_registers,
        typed_digit_register_digits=typed_digit_register_digits,
        typed_digit_update_scale=typed_digit_update_scale,
        **kwargs,
    )
    
    # Move to device
    if device is not None:
        state_model = state_model.to(device)
    
    # Setup label token IDs
    digit_chars = [str(i) for i in range(10)]
    label_token_ids = [
        tokenizer.encode(d, add_special_tokens=False)[0]
        for d in digit_chars
    ]
    state_model.set_label_token_ids(label_token_ids)
    
    return state_model, tokenizer
