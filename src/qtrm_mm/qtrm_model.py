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
from .world_model import JepaWorldModelHead, SIGReg


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
        )
        self.projector = MultimodalProjector(
            cfg.d_model, cfg.visual_dim, cfg.max_visual_tokens, cfg.n_heads,
        )
        self.ctrl = ControllerHeads(cfg.d_model, cfg.num_actions)
        self.residual_gate = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, float(cfg.qtrm_residual_gate_init_bias))
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        text_states: Optional[torch.Tensor] = None,
        donor_logits: Optional[torch.Tensor] = None,
        disable_workspace: bool = False,
        disable_core: bool = False,
        disable_coda: bool = False,
        disable_qtrm_residual: bool = False,
        disable_donor_context: bool = False,
        workspace_only_context: bool = False,
        enable_core_halt: Optional[bool] = None,
        return_core_depth_logits: bool = False,
    ) -> dict:
        b, s = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        text_seq = self.text_embed(input_ids)
        input_text_seq = text_seq
        input_text_mask = attention_mask
        jepa_latents = self.jepa_encoder(text_seq, attention_mask=attention_mask)
        jepa_latents = self.jepa_encoder_norm(jepa_latents)
        jepa_outputs = self.jepa(jepa_latents, attention_mask=attention_mask)

        seq = text_seq

        if text_states is not None and not disable_donor_context:
            seq, attention_mask = self.projector(seq, text_states, text_mask=attention_mask)

        if visual_features is not None and not disable_donor_context:
            seq, attention_mask = self.projector(seq, visual_features, text_mask=attention_mask)

        if workspace_only_context and not disable_workspace:
            workspace_seq = self.prelude(seq, attention_mask=attention_mask)
            workspace_mask_input = attention_mask
            text_context_seq = self.prelude(input_text_seq, attention_mask=input_text_mask)
            text_context_mask = input_text_mask
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
            core_depth_last_logits = self._empty_core_depth_last_logits(workspace)
        else:
            workspace = self.workspace(seq, context_mask=attention_mask)
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
            else:
                z_l, z_h, trajectory, core_halt_info = self.core(
                    workspace,
                    attention_mask=workspace_mask,
                    enable_halt=enable_core_halt,
                )
            core_depth_states = self._core_depth_states(trajectory, workspace)
            core_depth_last_logits = (
                self._core_depth_last_logits(
                    trajectory,
                    text_context_seq=text_context_seq,
                    text_context_mask=text_context_mask,
                    workspace_mask=workspace_mask,
                )
                if return_core_depth_logits
                else self._empty_core_depth_last_logits(workspace)
            )

            seq = torch.cat([z_h, text_context_seq], dim=1)
            attention_mask = torch.cat([workspace_mask, text_context_mask], dim=1)
        if not disable_coda:
            seq = self.coda(seq, attention_mask=attention_mask)
        seq = self.norm(seq)
        qtrm_logits = self.lm_head(seq) * float(self.cfg.qtrm_logits_scale)
        qtrm_residual_logits = qtrm_logits
        if self.cfg.qtrm_residual_clamp is not None:
            clamp = abs(float(self.cfg.qtrm_residual_clamp))
            qtrm_residual_logits = qtrm_residual_logits.clamp(min=-clamp, max=clamp)
        residual_gate = qtrm_logits.new_ones((b,))
        if self.cfg.qtrm_residual_gate_enabled:
            residual_gate = self._compute_residual_gate(z_h)
            qtrm_residual_logits = qtrm_residual_logits * residual_gate[:, None, None]
        if disable_qtrm_residual:
            qtrm_residual_logits = torch.zeros_like(qtrm_residual_logits)
        logits = qtrm_logits
        if donor_logits is not None and self.cfg.donor_logits_scale != 0.0:
            if donor_logits.shape[:2] != (b, s):
                raise ValueError(
                    "donor_logits must have shape [batch, input_seq_len, vocab_size]"
                )
            if donor_logits.shape[-1] != self.cfg.vocab_size:
                raise ValueError("donor_logits vocab size must match model vocab_size")
            text_offset = logits.shape[1] - s
            logits = qtrm_residual_logits.clone()
            logits[:, text_offset:, :] = (
                logits[:, text_offset:, :]
                + donor_logits.to(device=logits.device, dtype=logits.dtype)
                * float(self.cfg.donor_logits_scale)
            )

        pooled = z_h[:, -1, :]
        ctrl = self.ctrl(pooled)

        return {
            "logits": logits,
            "qtrm_logits": qtrm_logits,
            "qtrm_residual_logits": qtrm_residual_logits,
            "qtrm_residual_gate": residual_gate,
            "z_l": z_l,
            "z_h": z_h,
            "pooled": pooled,
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
            **ctrl,
        }

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
    ) -> torch.Tensor:
        if not trajectory:
            return self._empty_core_depth_last_logits(text_context_seq)
        depth_logits = []
        for state in trajectory:
            seq = torch.cat([state, text_context_seq], dim=1)
            attention_mask = torch.cat([workspace_mask, text_context_mask], dim=1)
            hidden = self.coda(seq, attention_mask=attention_mask)
            hidden = self.norm(hidden)
            last_logits = self.lm_head(hidden[:, -1, :]) * float(self.cfg.qtrm_logits_scale)
            depth_logits.append(last_logits)
        return torch.stack(depth_logits, dim=1)
