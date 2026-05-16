from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

from .config import QTRMConfig
from .blocks import QTRMBlockStack
from .stability import StableInject
from .norm import RMSNorm
from .attention import CrossAttention


@dataclass
class QTRMCoreCarry:
    z_l: torch.Tensor
    z_h: torch.Tensor
    halted: torch.Tensor
    steps: torch.Tensor

    def detached(self) -> "QTRMCoreCarry":
        return QTRMCoreCarry(
            z_l=self.z_l.detach(),
            z_h=self.z_h.detach(),
            halted=self.halted.detach(),
            steps=self.steps.detach(),
        )


class QTRMRecursiveCore(nn.Module):
    """TRM-style z_L/z_H recurrent latent workspace core."""

    def __init__(self, cfg: QTRMConfig):
        super().__init__()
        self.cfg = cfg
        core_causal = bool(getattr(cfg, "core_causal", False))
        self.fast_stack = QTRMBlockStack(
            cfg,
            cfg.n_core_layers,
            causal=core_causal,
            attn_every=cfg.attn_every,
        )
        self.slow_stack = QTRMBlockStack(
            cfg,
            cfg.n_core_layers,
            causal=core_causal,
            attn_every=cfg.attn_every,
        )
        self.inject_l = StableInject(cfg.d_model) if cfg.use_stable_inject else None
        self.inject_h = StableInject(cfg.d_model) if cfg.use_stable_inject else None
        self.step_conditioning = (
            nn.Embedding(max(1, int(cfg.core_step_conditioning_max_steps)), cfg.d_model)
            if cfg.core_step_conditioning_enabled
            else None
        )
        if self.step_conditioning is not None:
            nn.init.normal_(self.step_conditioning.weight, mean=0.0, std=0.02)
        self.transition_order_conditioning_norm = (
            RMSNorm(cfg.d_model)
            if cfg.core_transition_order_step_conditioning_enabled
            else None
        )
        self.transition_order_conditioning_gate = (
            nn.Linear(cfg.d_model, 1)
            if cfg.core_transition_order_step_conditioning_enabled
            else None
        )
        if self.transition_order_conditioning_gate is not None:
            nn.init.zeros_(self.transition_order_conditioning_gate.weight)
            nn.init.constant_(
                self.transition_order_conditioning_gate.bias,
                float(cfg.core_transition_order_step_conditioning_gate_init_bias),
            )
        self.z_l_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.z_h_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.norm_l = RMSNorm(cfg.d_model)
        self.norm_h = RMSNorm(cfg.d_model)
        self.context_cross_l = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.core_context_enabled
            else None
        )
        self.context_cross_h = (
            CrossAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            if cfg.core_context_enabled
            else None
        )
        self.context_gate_l = nn.Linear(cfg.d_model, 1) if cfg.core_context_enabled else None
        self.context_gate_h = nn.Linear(cfg.d_model, 1) if cfg.core_context_enabled else None
        if self.context_gate_l is not None and self.context_gate_h is not None:
            for gate in (self.context_gate_l, self.context_gate_h):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, float(cfg.core_context_gate_init_bias))
        feedback_ops = max(1, int(cfg.core_transition_feedback_num_operations))
        self.transition_feedback_norm = (
            RMSNorm(cfg.d_model) if cfg.core_transition_feedback_enabled else None
        )
        self.transition_feedback_operation_head = (
            nn.Linear(cfg.d_model, feedback_ops)
            if cfg.core_transition_feedback_enabled
            else None
        )
        self.transition_feedback_finality_head = (
            nn.Linear(cfg.d_model, 1) if cfg.core_transition_feedback_enabled else None
        )
        self.transition_feedback_operation_embed = (
            nn.Embedding(feedback_ops, cfg.d_model)
            if cfg.core_transition_feedback_enabled
            else None
        )
        self.transition_feedback_finality_proj = (
            nn.Linear(1, cfg.d_model) if cfg.core_transition_feedback_enabled else None
        )
        self.transition_feedback_gate = (
            nn.Linear(cfg.d_model, 1) if cfg.core_transition_feedback_enabled else None
        )
        self.transition_feedback_update_norm_l = (
            RMSNorm(cfg.d_model) if cfg.core_transition_feedback_enabled else None
        )
        self.transition_feedback_update_norm_h = (
            RMSNorm(cfg.d_model) if cfg.core_transition_feedback_enabled else None
        )
        if self.transition_feedback_operation_head is not None:
            nn.init.xavier_uniform_(self.transition_feedback_operation_head.weight)
            nn.init.zeros_(self.transition_feedback_operation_head.bias)
        if self.transition_feedback_finality_head is not None:
            nn.init.xavier_uniform_(self.transition_feedback_finality_head.weight)
            nn.init.zeros_(self.transition_feedback_finality_head.bias)
        if self.transition_feedback_operation_embed is not None:
            nn.init.normal_(self.transition_feedback_operation_embed.weight, mean=0.0, std=0.02)
        if self.transition_feedback_finality_proj is not None:
            nn.init.xavier_uniform_(self.transition_feedback_finality_proj.weight)
            nn.init.zeros_(self.transition_feedback_finality_proj.bias)
        if self.transition_feedback_gate is not None:
            nn.init.zeros_(self.transition_feedback_gate.weight)
            nn.init.constant_(
                self.transition_feedback_gate.bias,
                float(cfg.core_transition_feedback_gate_init_bias),
            )
        carry_hidden_dim = int(cfg.core_state_carry_hidden_dim or cfg.d_model)
        self.state_carry_norm = RMSNorm(cfg.d_model) if cfg.core_state_carry_enabled else None
        self.state_carry_update = (
            nn.Sequential(
                nn.Linear(cfg.d_model, carry_hidden_dim),
                nn.GELU(),
                nn.Linear(carry_hidden_dim, cfg.d_model),
            )
            if cfg.core_state_carry_enabled
            else None
        )
        self.state_carry_gate = (
            nn.Linear(cfg.d_model, 1) if cfg.core_state_carry_enabled else None
        )
        if self.state_carry_update is not None:
            for module in self.state_carry_update:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.state_carry_gate is not None:
            nn.init.zeros_(self.state_carry_gate.weight)
            nn.init.constant_(
                self.state_carry_gate.bias,
                float(cfg.core_state_carry_gate_init_bias),
            )
        self.halt_head = nn.Linear(cfg.d_model, 2) if cfg.core_halt_enabled else None
        if self.halt_head is not None:
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, float(cfg.core_halt_init_bias))

    def forward(
        self,
        workspace: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_states: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        disable_context: bool = False,
        state_carry_start: Optional[int] = None,
        state_carry_count: int = 0,
        disable_state_carry: bool = False,
        enable_halt: Optional[bool] = None,
        carry: Optional[QTRMCoreCarry] = None,
        return_carry: bool = False,
        transition_feedback_operation_targets: Optional[torch.Tensor] = None,
        transition_feedback_finality_targets: Optional[torch.Tensor] = None,
        transition_feedback_teacher_forcing: bool = False,
        transition_order_conditioning: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        b, w, d = workspace.shape
        fresh_z_l = workspace + self.z_l_init
        fresh_z_h = workspace + self.z_h_init
        carry_steps_start = torch.zeros(
            b,
            device=workspace.device,
            dtype=torch.long,
        )
        if carry is None:
            z_l = fresh_z_l
            z_h = fresh_z_h
        else:
            if tuple(carry.z_l.shape) != tuple(workspace.shape) or tuple(carry.z_h.shape) != tuple(workspace.shape):
                raise ValueError("core carry state shape must match workspace shape")
            reset_mask = carry.halted.to(device=workspace.device, dtype=torch.bool)
            if tuple(reset_mask.shape) != (b,):
                raise ValueError("core carry halted shape must match batch size")
            carry_steps_start = carry.steps.to(device=workspace.device, dtype=torch.long)
            if tuple(carry_steps_start.shape) != (b,):
                raise ValueError("core carry steps shape must match batch size")
            reset = reset_mask.view(b, 1, 1)
            z_l = torch.where(reset, fresh_z_l, carry.z_l.to(device=workspace.device, dtype=workspace.dtype))
            z_h = torch.where(reset, fresh_z_h, carry.z_h.to(device=workspace.device, dtype=workspace.dtype))
            carry_steps_start = torch.where(
                reset_mask,
                torch.zeros_like(carry_steps_start),
                carry_steps_start,
            )
        trajectory = []
        q_halt_steps = []
        q_continue_steps = []
        context_gate_means = []
        state_carry_gate_means = []
        feedback_operation_steps = []
        feedback_finality_steps = []
        feedback_gate_means = []
        halted = torch.zeros(b, device=workspace.device, dtype=torch.bool)
        steps_per_sample = carry_steps_start.clone()
        enable_halt = bool(self.cfg.core_halt_enabled if enable_halt is None else enable_halt)
        freeze_halted_state = bool(
            enable_halt and self.cfg.core_halt_freeze_halted_state_enabled
        )
        trm_no_grad_inner_cycles = bool(
            self.cfg.core_trm_no_grad_inner_cycles_enabled
            and int(self.cfg.h_cycles) > 1
        )
        exploration_prob = min(max(float(self.cfg.core_halt_exploration_prob), 0.0), 1.0)
        exploration_min_steps = max(1, int(self.cfg.core_halt_exploration_min_steps))
        exploration_active = bool(
            self.training
            and enable_halt
            and exploration_prob > 0.0
            and int(self.cfg.outer_steps) > 1
        )
        exploration_mask = (
            torch.rand(b, device=workspace.device) < exploration_prob
            if exploration_active
            else torch.zeros(b, device=workspace.device, dtype=torch.bool)
        )
        order_conditioning = None
        order_conditioning_gate_mean = workspace.new_empty((b, 0))
        if (
            transition_order_conditioning is not None
            and self.transition_order_conditioning_norm is not None
            and self.transition_order_conditioning_gate is not None
        ):
            order_conditioning = transition_order_conditioning.to(
                device=workspace.device,
                dtype=workspace.dtype,
            )
            if order_conditioning.ndim == 2:
                order_conditioning = order_conditioning.unsqueeze(1)
            if order_conditioning.ndim != 3 or tuple(order_conditioning.shape) != (
                b,
                1,
                d,
            ):
                raise ValueError(
                    "transition_order_conditioning must have shape [batch, d_model] "
                    "or [batch, 1, d_model]"
                )
            order_conditioning = self.transition_order_conditioning_norm(
                order_conditioning
            )
            order_gate = torch.sigmoid(
                self.transition_order_conditioning_gate(order_conditioning)
            )
            order_conditioning_gate_mean = order_gate.squeeze(-1)
            order_conditioning = (
                order_conditioning
                * order_gate
                * float(self.cfg.core_transition_order_step_conditioning_scale)
            )
        context_active = (
            context_states is not None
            and not disable_context
            and self.context_cross_l is not None
            and self.context_cross_h is not None
            and self.context_gate_l is not None
            and self.context_gate_h is not None
        )
        state_carry_active = (
            self.state_carry_norm is not None
            and self.state_carry_update is not None
            and self.state_carry_gate is not None
            and state_carry_start is not None
            and int(state_carry_count) > 0
            and not disable_state_carry
        )
        loop_id = 0
        for outer in range(self.cfg.outer_steps):
            active_at_start = ~halted if freeze_halted_state else torch.ones_like(halted)
            z_l_before_outer = z_l
            z_h_before_outer = z_h
            if self.step_conditioning is not None:
                step_idx = min(int(outer), self.step_conditioning.num_embeddings - 1)
                step_id = torch.tensor(step_idx, device=workspace.device, dtype=torch.long)
                step = self.step_conditioning(step_id).view(1, 1, d)
                step = step * float(self.cfg.core_step_conditioning_scale)
                z_l = self.norm_l(z_l + step)
                z_h = self.norm_h(z_h + step)
            if order_conditioning is not None:
                z_l = self.norm_l(z_l + order_conditioning)
                z_h = self.norm_h(z_h + order_conditioning)
            for h in range(self.cfg.h_cycles):
                grad_enabled = torch.is_grad_enabled() and not (
                    trm_no_grad_inner_cycles and h < (self.cfg.h_cycles - 1)
                )
                with torch.set_grad_enabled(grad_enabled):
                    for l in range(self.cfg.l_cycles):
                        source = z_h + workspace
                        if context_active:
                            context_delta = self.context_cross_l(
                                self.norm_l(z_l),
                                context_states,
                                context_mask,
                            )
                            context_gate = torch.sigmoid(self.context_gate_l(z_l))
                            source = source + context_gate * context_delta
                            context_gate_means.append(context_gate.squeeze(-1).mean(dim=1))
                        if self.inject_l is not None:
                            source = self.inject_l(z_l, source, loop_id=loop_id)
                        z_l = self.norm_l(z_l + source)
                        z_l = self.fast_stack(z_l, attention_mask=attention_mask)
                        loop_id += 1
                    source_h = z_l
                    if context_active:
                        context_delta_h = self.context_cross_h(
                            self.norm_h(z_h),
                            context_states,
                            context_mask,
                        )
                        context_gate_h = torch.sigmoid(self.context_gate_h(z_h))
                        source_h = source_h + context_gate_h * context_delta_h
                        context_gate_means.append(context_gate_h.squeeze(-1).mean(dim=1))
                    if self.inject_h is not None:
                        source_h = self.inject_h(z_h, source_h, loop_id=loop_id)
                    z_h = self.norm_h(z_h + source_h)
                    z_h = self.slow_stack(z_h, attention_mask=attention_mask)
                    loop_id += 1
            if freeze_halted_state and bool(halted.any().detach().cpu().item()):
                active_mask = active_at_start.view(b, 1, 1)
                z_l = torch.where(active_mask, z_l, z_l_before_outer)
                z_h = torch.where(active_mask, z_h, z_h_before_outer)
            if state_carry_active:
                start = max(0, int(state_carry_start))
                end = min(int(z_h.shape[1]), start + int(state_carry_count))
                if start < end:
                    carry_tokens = z_h[:, start:end, :]
                    carry_delta = self.state_carry_update(
                        self.state_carry_norm(carry_tokens)
                    )
                    carry_gate = torch.sigmoid(self.state_carry_gate(carry_tokens))
                    gate_min = float(self.cfg.core_state_carry_gate_min)
                    if gate_min != 0.0:
                        carry_gate = carry_gate * (1.0 - gate_min) + gate_min
                    carried = self.norm_h(carry_tokens + carry_gate * carry_delta)
                    z_h = z_h.clone()
                    z_l = z_l.clone()
                    z_h[:, start:end, :] = carried
                    z_l[:, start:end, :] = carried
                    state_carry_gate_means.append(
                        carry_gate.squeeze(-1).mean(dim=1)
                    )
            trajectory.append(z_h)
            steps_per_sample = steps_per_sample + (
                active_at_start.to(torch.long)
                if freeze_halted_state
                else torch.ones_like(steps_per_sample)
            )
            if self.halt_head is not None:
                q_logits = self.halt_head(z_h[:, 0, :]).to(torch.float32)
                q_halt = q_logits[..., 0]
                q_continue = q_logits[..., 1]
                q_halt_steps.append(q_halt)
                q_continue_steps.append(q_continue)
                if enable_halt and (outer + 1) >= max(1, int(self.cfg.core_halt_min_steps)):
                    if self.cfg.core_halt_use_continue:
                        should_halt = q_halt > q_continue
                    else:
                        should_halt = q_halt > 0
                    if exploration_active and (outer + 1) < exploration_min_steps:
                        should_halt = should_halt & ~exploration_mask
                    halted = halted | should_halt if freeze_halted_state else should_halt
                    if bool(halted.all().detach().cpu().item()):
                        break
            if self.transition_feedback_norm is not None:
                feedback_state = self.transition_feedback_norm(z_h[:, 0, :])
                operation_logits = self.transition_feedback_operation_head(
                    feedback_state
                )
                finality_logits = self.transition_feedback_finality_head(
                    feedback_state
                ).squeeze(-1)
                feedback_operation_steps.append(operation_logits)
                feedback_finality_steps.append(finality_logits)
                operation_weights = torch.softmax(
                    operation_logits.float(),
                    dim=-1,
                ).to(dtype=z_h.dtype)
                if (
                    bool(transition_feedback_teacher_forcing)
                    and transition_feedback_operation_targets is not None
                    and outer < int(transition_feedback_operation_targets.shape[1])
                ):
                    op_targets = transition_feedback_operation_targets[
                        :, outer
                    ].to(device=z_h.device, dtype=torch.long)
                    op_valid = op_targets >= 0
                    if bool(op_valid.any().detach().cpu().item()):
                        op_clamped = op_targets.clamp(
                            min=0,
                            max=int(operation_logits.shape[-1]) - 1,
                        )
                        teacher_weights = torch.zeros_like(operation_weights)
                        teacher_weights.scatter_(
                            1,
                            op_clamped.view(-1, 1),
                            1.0,
                        )
                        operation_weights = torch.where(
                            op_valid.view(-1, 1),
                            teacher_weights,
                            operation_weights,
                        )
                operation_feedback = operation_weights @ (
                    self.transition_feedback_operation_embed.weight.to(dtype=z_h.dtype)
                )
                finality_signal = torch.sigmoid(finality_logits.float()).to(
                    dtype=z_h.dtype
                ).unsqueeze(-1)
                if (
                    bool(transition_feedback_teacher_forcing)
                    and transition_feedback_finality_targets is not None
                    and outer < int(transition_feedback_finality_targets.shape[1])
                ):
                    finality_targets = transition_feedback_finality_targets[
                        :, outer
                    ].to(device=z_h.device, dtype=z_h.dtype)
                    finality_valid = finality_targets >= 0
                    if bool(finality_valid.any().detach().cpu().item()):
                        finality_signal = torch.where(
                            finality_valid.view(-1, 1),
                            finality_targets.clamp(0, 1).view(-1, 1),
                            finality_signal,
                        )
                finality_feedback = self.transition_feedback_finality_proj(
                    finality_signal
                )
                feedback = operation_feedback + finality_feedback
                feedback_gate = torch.sigmoid(
                    self.transition_feedback_gate(feedback_state)
                ).to(dtype=z_h.dtype)
                feedback_gate_means.append(feedback_gate.squeeze(-1))
                feedback = (
                    float(self.cfg.core_transition_feedback_scale)
                    * feedback_gate
                    * feedback
                ).unsqueeze(1)
                z_l = z_l.clone()
                z_h = z_h.clone()
                z_l[:, :1, :] = self.transition_feedback_update_norm_l(
                    z_l[:, :1, :] + feedback
                )
                z_h[:, :1, :] = self.transition_feedback_update_norm_h(
                    z_h[:, :1, :] + feedback
                )
            if self.cfg.truncated_recurrence and self.training:
                z_l = z_l.detach()
                z_h = z_h.detach()
        steps = len(trajectory)
        if q_halt_steps:
            q_halt_logits = torch.stack(q_halt_steps, dim=1)
            q_continue_logits = torch.stack(q_continue_steps, dim=1)
        else:
            q_halt_logits = workspace.new_empty((b, 0), dtype=torch.float32)
            q_continue_logits = workspace.new_empty((b, 0), dtype=torch.float32)
        if context_gate_means:
            context_gate_mean = torch.stack(context_gate_means, dim=1)
        else:
            context_gate_mean = workspace.new_empty((b, 0))
        if state_carry_gate_means:
            state_carry_gate_mean = torch.stack(state_carry_gate_means, dim=1)
        else:
            state_carry_gate_mean = workspace.new_empty((b, 0))
        if feedback_operation_steps:
            feedback_operation_logits = torch.stack(feedback_operation_steps, dim=1)
            feedback_finality_logits = torch.stack(feedback_finality_steps, dim=1)
            feedback_gate_mean = torch.stack(feedback_gate_means, dim=1)
        else:
            feedback_ops = max(1, int(self.cfg.core_transition_feedback_num_operations))
            feedback_operation_logits = workspace.new_empty(
                (b, 0, feedback_ops),
                dtype=torch.float32,
            )
            feedback_finality_logits = workspace.new_empty((b, 0), dtype=torch.float32)
            feedback_gate_mean = workspace.new_empty((b, 0))
        halt_info = {
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "halted": halted if enable_halt else torch.zeros_like(halted),
            "steps": steps_per_sample,
            "context_gate_mean": context_gate_mean,
            "state_carry_gate_mean": state_carry_gate_mean,
            "transition_feedback_operation_logits": feedback_operation_logits,
            "transition_feedback_finality_logits": feedback_finality_logits,
            "transition_feedback_gate_mean": feedback_gate_mean,
            "transition_order_conditioning_gate_mean": order_conditioning_gate_mean,
        }
        if return_carry:
            halt_info["carry"] = QTRMCoreCarry(
                z_l=z_l,
                z_h=z_h,
                halted=halt_info["halted"],
                steps=steps_per_sample,
            ).detached()
        return z_l, z_h, trajectory, halt_info
