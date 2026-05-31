from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .config import WGRAMV2Config


@dataclass
class IMTAOutput:
    chunk_hidden: torch.Tensor
    selector_weights: torch.Tensor
    diversity_loss: torch.Tensor
    route_entropy_loss: torch.Tensor
    route_balance_loss: torch.Tensor
    metrics: dict[str, float | int]


class InternalMultiTrajectoryAdapter(nn.Module):
    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        self.config = config
        d_model = int(config.d_model)
        k = int(config.imta_trajectories)
        self.offsets = nn.Parameter(torch.zeros(k, d_model))
        self.adapter_gate_logit = nn.Parameter(torch.full((k,), float(config.imta_adapter_gate_init)))
        self.post_adapter_gate_logit = nn.Parameter(torch.full((k,), float(config.imta_post_adapter_gate_init)))
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(k)
            ]
        )
        self.post_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(k)
            ]
        )
        self.selector = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.route_queries = nn.Parameter(torch.zeros(k, d_model))
        self.route_bias = nn.Parameter(torch.zeros(k))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if int(self.config.imta_trajectories) > 1:
            nn.init.normal_(self.offsets, mean=0.0, std=0.02)
            with torch.no_grad():
                self.offsets[0].zero_()
        for idx, adapter in enumerate(self.adapters):
            last = adapter[-1]
            if isinstance(last, nn.Linear):
                if idx == 0:
                    nn.init.zeros_(last.weight)
                else:
                    nn.init.normal_(last.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last.bias)
        for idx, adapter in enumerate(self.post_adapters):
            last = adapter[-1]
            if isinstance(last, nn.Linear):
                if idx == 0:
                    nn.init.zeros_(last.weight)
                else:
                    nn.init.normal_(last.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last.bias)
        head = self.selector[-1]
        if isinstance(head, nn.Linear):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if int(self.config.imta_trajectories) > 1 and float(self.config.imta_selector_route_query_std) > 0.0:
            nn.init.normal_(self.route_queries, mean=0.0, std=float(self.config.imta_selector_route_query_std))
            with torch.no_grad():
                self.route_queries[0].zero_()
                self.route_bias.copy_(torch.linspace(-0.02, 0.02, steps=self.route_bias.numel()))

    def forward(
        self,
        chunk_states: torch.Tensor,
        chunk_valid: torch.Tensor,
        *,
        core_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        speaker_norm: nn.Module,
    ) -> IMTAOutput:
        batch, chunk_len, d_model = chunk_states.shape
        k = int(self.config.imta_trajectories)
        if k <= 1:
            hidden = core_fn(chunk_states, chunk_valid)
            return IMTAOutput(
                chunk_hidden=hidden,
                selector_weights=hidden.new_ones((batch, 1)),
                diversity_loss=hidden.new_zeros(()),
                route_entropy_loss=hidden.new_zeros(()),
                route_balance_loss=hidden.new_zeros(()),
                metrics={
                    "imta_trajectory_count": 1,
                    "imta_adapter_delta_norm": 0.0,
                    "imta_raw_selector_entropy": 0.0,
                    "imta_selector_entropy": 0.0,
                    "imta_raw_selector_effective_routes": 1.0,
                    "imta_selector_effective_routes": 1.0,
                    "imta_diversity_loss": 0.0,
                    "imta_route_min_probability": 0.0,
                    "imta_route_entropy_floor": 0.0,
                    "imta_route_entropy_loss": 0.0,
                    "imta_route_balance_loss": 0.0,
                    "imta_raw_route_mass_min": 1.0,
                    "imta_raw_route_mass_max": 1.0,
                    "imta_route_mass_min": 1.0,
                    "imta_route_mass_max": 1.0,
                    "imta_route_entropy_below_floor_fraction": 0.0,
                },
            )

        offsets = self.offsets[:k].to(device=chunk_states.device, dtype=chunk_states.dtype)
        trajectories = chunk_states.unsqueeze(1).expand(batch, k, chunk_len, d_model)
        trajectories = trajectories + offsets.view(1, k, 1, d_model)
        adapter_deltas = torch.stack([self.adapters[idx](chunk_states) for idx in range(k)], dim=1)
        gates = torch.sigmoid(self.adapter_gate_logit[:k]).to(device=chunk_states.device, dtype=chunk_states.dtype)
        anchor_mask = torch.ones((1, k, 1, 1), dtype=chunk_states.dtype, device=chunk_states.device)
        anchor_mask[:, 0] = 0.0
        adapter_delta = gates.view(1, k, 1, 1) * adapter_deltas * anchor_mask
        trajectories = trajectories + adapter_delta
        if self.training and float(self.config.imta_noise_std) > 0.0:
            noise = torch.randn_like(trajectories) * float(self.config.imta_noise_std)
            noise[:, 0] = 0.0
            trajectories = trajectories + noise

        flat_states = trajectories.reshape(batch * k, chunk_len, d_model)
        flat_valid = chunk_valid.unsqueeze(1).expand(batch, k, chunk_len).reshape(batch * k, chunk_len)
        flat_hidden = core_fn(flat_states, flat_valid)
        hidden = flat_hidden.reshape(batch, k, chunk_len, d_model)
        post_adapter_deltas = torch.stack([self.post_adapters[idx](hidden[:, idx]) for idx in range(k)], dim=1)
        post_gates = torch.sigmoid(self.post_adapter_gate_logit[:k]).to(device=hidden.device, dtype=hidden.dtype)
        post_adapter_delta = post_gates.view(1, k, 1, 1) * post_adapter_deltas * anchor_mask
        hidden = hidden + post_adapter_delta

        valid = chunk_valid.to(hidden.dtype).view(batch, 1, chunk_len, 1)
        pooled = hidden.mul(valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        speaker_probe = speaker_norm(pooled)
        selector_logits = self.selector(torch.cat([pooled, speaker_probe], dim=-1)).squeeze(-1)
        query_logits = torch.einsum(
            "bkd,kd->bk",
            speaker_probe.float(),
            self.route_queries[:k].to(device=speaker_probe.device).float(),
        ) / max(float(d_model) ** 0.5, 1.0)
        selector_logits = selector_logits + query_logits.to(selector_logits.dtype)
        selector_logits = selector_logits + self.route_bias[:k].to(device=selector_logits.device, dtype=selector_logits.dtype)
        raw_selector_weights = torch.softmax(selector_logits / float(self.config.imta_selector_temperature), dim=-1)
        min_route_probability = min(
            max(float(self.config.imta_route_min_probability), 0.0),
            (1.0 / float(k)) - 1.0e-6,
        )
        if min_route_probability > 0.0:
            selector_weights = raw_selector_weights * (1.0 - min_route_probability * float(k))
            selector_weights = selector_weights + min_route_probability
        else:
            selector_weights = raw_selector_weights
        selected_pre_norm = (selector_weights.view(batch, k, 1, 1) * hidden).sum(dim=1)
        selected = self.output_norm(selected_pre_norm) * chunk_valid.to(hidden.dtype).unsqueeze(-1)

        normed = F.normalize(pooled.float(), dim=-1)
        cosine = torch.matmul(normed, normed.transpose(1, 2))
        off_diag_mask = ~torch.eye(k, dtype=torch.bool, device=cosine.device).view(1, k, k)
        off_diag = cosine.masked_select(off_diag_mask.expand(batch, -1, -1))
        diversity_loss = F.relu(off_diag).mean().to(chunk_states.dtype) if int(off_diag.numel()) else selected.new_zeros(())
        raw_entropy = -(raw_selector_weights * raw_selector_weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        entropy = -(selector_weights * selector_weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        max_entropy = math.log(float(k))
        entropy_floor = min(max(float(self.config.imta_route_entropy_floor), 0.0), max_entropy)
        route_entropy_loss = F.relu(raw_entropy.new_tensor(entropy_floor) - raw_entropy).mean().to(chunk_states.dtype)
        raw_route_mass = raw_selector_weights.mean(dim=0)
        route_mass = selector_weights.mean(dim=0)
        uniform_mass = route_mass.new_full((k,), 1.0 / float(k))
        route_balance_loss = (
            raw_route_mass.clamp_min(1.0e-8) * (raw_route_mass.clamp_min(1.0e-8) / uniform_mass).log()
        ).sum().to(chunk_states.dtype)
        below_floor = raw_entropy < raw_entropy.new_tensor(entropy_floor)
        return IMTAOutput(
            chunk_hidden=selected,
            selector_weights=selector_weights,
            diversity_loss=diversity_loss,
            route_entropy_loss=route_entropy_loss,
            route_balance_loss=route_balance_loss,
            metrics={
                "imta_trajectory_count": int(k),
                "imta_adapter_delta_norm": float(adapter_delta.detach().float().norm(dim=-1).mean().cpu().item()),
                "imta_post_adapter_delta_norm": float(post_adapter_delta.detach().float().norm(dim=-1).mean().cpu().item()),
                "imta_selected_pre_norm_norm": float(
                    selected_pre_norm.detach().float().norm(dim=-1)[chunk_valid.bool()].mean().cpu().item()
                )
                if bool(chunk_valid.any())
                else 0.0,
                "imta_selected_post_norm_norm": float(
                    selected.detach().float().norm(dim=-1)[chunk_valid.bool()].mean().cpu().item()
                )
                if bool(chunk_valid.any())
                else 0.0,
                "imta_raw_selector_entropy": float(raw_entropy.detach().float().mean().cpu().item()),
                "imta_selector_entropy": float(entropy.detach().float().mean().cpu().item()),
                "imta_raw_selector_effective_routes": float(raw_entropy.detach().float().exp().mean().cpu().item()),
                "imta_selector_effective_routes": float(entropy.detach().float().exp().mean().cpu().item()),
                "imta_diversity_loss": float(diversity_loss.detach().float().cpu().item()),
                "imta_route_min_probability": float(min_route_probability),
                "imta_route_entropy_floor": float(entropy_floor),
                "imta_route_entropy_loss": float(route_entropy_loss.detach().float().cpu().item()),
                "imta_route_balance_loss": float(route_balance_loss.detach().float().cpu().item()),
                "imta_raw_route_mass_min": float(raw_route_mass.detach().float().min().cpu().item()),
                "imta_raw_route_mass_max": float(raw_route_mass.detach().float().max().cpu().item()),
                "imta_route_mass_min": float(route_mass.detach().float().min().cpu().item()),
                "imta_route_mass_max": float(route_mass.detach().float().max().cpu().item()),
                "imta_route_entropy_below_floor_fraction": float(below_floor.detach().float().mean().cpu().item()),
            },
        )
