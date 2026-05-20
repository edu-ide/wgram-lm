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
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        self.model_id = str(model_id)
        self.mandatory_core = bool(mandatory_core)
        self.core_impl = str(core_impl)
        
        if self.core_impl != "state_transition":
            raise ValueError(f"only core_impl='state_transition' supported, got: {self.core_impl}")
        
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
        
        # Initialize compressor with near-identity mapping
        for module in self.compressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # State transition core
        qtrm_cfg = core_config or QTRMConfig(
            d_model=self.d_state,
            num_actions=n_operations,
            outer_steps=n_steps,
        )
        self.core = StateTransitionCore(
            cfg=qtrm_cfg,
            d_state=self.d_state,
            n_operations=n_operations,
            n_steps=n_steps,
        )
        
        # Core output normalization
        self.core_out_norm = RMSNorm(self.d_state)
        
        # Answer head: reads from the final state only
        self.answer_head = StateReadoutHead(self.d_state)
        
        # State readout for intermediate supervision
        self.state_readout = StateReadoutHead(self.d_state)
        
        # Store for label token IDs
        self.label_token_ids = None
    
    def set_label_token_ids(self, token_ids: list[int]) -> None:
        """Set label token IDs for digit choices."""
        self.label_token_ids = torch.tensor(token_ids, dtype=torch.long)
    
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
        # Option 1: Mean-pool over attended tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            # Weighted mean pooling
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Compress to state space
        workspace = self.compressor(pooled.unsqueeze(1))  # (B, 1, d_state)
        return workspace
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        force_core_off: bool = False,
        operation_ids: Optional[torch.Tensor] = None,
        operation_soft: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
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
        
        # State transition core
        core_output = self.core(
            workspace=workspace,
            operation_ids=operation_ids,
            operation_soft=operation_soft,
            n_steps=n_steps,
        )
        
        # Normalize final state for answer head
        final_state = core_output.state_trajectory[:, -1, :]
        normalized_final = self.core_out_norm(final_state)
        
        # Answer logits from normalized final state
        answer_logits = self.answer_head(normalized_final)
        
        # State digit logits for intermediate supervision
        state_digit_logits = self.state_readout(core_output.state_trajectory)
        
        # Operation logits
        operation_logits = core_output.operation_logits
        
        if not return_dict:
            return (answer_logits, state_digit_logits, operation_logits)
        
        return {
            "logits": answer_logits,
            "qtrm_core_step_states": core_output.state_trajectory,
            "state_digit_logits": state_digit_logits,
            "answer_logits": answer_logits,
            "operation_logits": operation_logits,
            "state_norms": core_output.state_norms,
            "transition_norms": core_output.transition_norms,
            "state_cosines": core_output.state_cosines,
            "baseline_logits": baseline_logits,
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
        )


def build_qwen_state_transition_model(
    model_id: str,
    *,
    d_state: Optional[int] = None,
    n_operations: int = N_OPERATIONS,
    n_steps: int = 4,
    freeze_qwen: bool = True,
    max_seq_len: int = 512,
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
