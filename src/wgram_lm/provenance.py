"""
Provenance reasoning components extracted for native QTRM architecture (I→G→A A-stage).

Originally developed in the Stage102 experimental track (scripts/605, 607, 608).
This module makes the full causal path (graph reasoner + data world model + gated register)
available as first-class, importable components inside src/wgram_lm/.

Promotion status is tracked in component_registry.py under the I→G→A protocol.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProvenanceGraphReasoner(nn.Module):
    """Tiny graph-to-register module for source authority and claim support.

    Extracted from Stage102B (605). Used as the graph frontend for the final
    free-form answer path (Stage102Z).
    """

    def __init__(self, d_model: int, max_sources: int = 16, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = int(hidden_dim or d_model)
        self.source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.trust_proj = nn.Linear(1, int(d_model))
        self.support_proj = nn.Linear(1, int(d_model))
        self.norm = nn.LayerNorm(int(d_model))
        self.message = nn.Sequential(
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, int(d_model)),
            nn.Tanh(),
        )
        self.authority_gate = nn.Sequential(
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: list[dict[str, Any]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not features:
            raise ValueError("features must not be empty")
        source_index = torch.tensor(
            [int(item["source_index"]) for item in features],
            dtype=torch.long,
            device=device,
        )
        source_index = source_index.clamp(0, self.source_embedding.num_embeddings - 1)
        source_verified = torch.tensor(
            [float(item["source_verified"]) for item in features],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        claim_supported = torch.tensor(
            [float(item["claim_supported"]) for item in features],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        state = self.source_embedding(source_index)
        state = state + self.trust_proj(source_verified) + self.support_proj(claim_supported)
        state = self.norm(state)
        authority = self.authority_gate(state)
        register = authority * self.message(state)
        metrics = {
            "rows": int(len(features)),
            "mean_authority": float(authority.detach().float().mean().cpu().item()),
            "min_authority": float(authority.detach().float().min().cpu().item()),
            "max_authority": float(authority.detach().float().max().cpu().item()),
            "mean_source_verified": float(source_verified.detach().float().mean().cpu().item()),
            "mean_claim_supported": float(claim_supported.detach().float().mean().cpu().item()),
        }
        return register, metrics


class ProvenanceDataWorldModel(nn.Module):
    """Small energy model for label-free provenance consistency.

    Extracted from Stage102D (607). Provides the world-model residual signal.
    """

    def __init__(self, d_model: int = 32, max_sources: int = 16, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = int(hidden_dim or max(32, d_model * 2))
        self.source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.verified_source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.context_source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.context_verified_source_embedding = nn.Embedding(int(max_sources), int(d_model))
        self.scalar_proj = nn.Linear(4, int(d_model))
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, int(d_model)),
            nn.LayerNorm(int(d_model)),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(int(d_model), width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        examples: list[dict[str, Any]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("examples must not be empty")
        source_index = torch.tensor(
            [int(item["source_index"]) for item in examples],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.source_embedding.num_embeddings - 1)
        verified_source_index = torch.tensor(
            [int(item["verified_source_index"]) for item in examples],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.verified_source_embedding.num_embeddings - 1)
        context_source_index = torch.tensor(
            [int(item.get("context_source_index", item["source_index"])) for item in examples],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.context_source_embedding.num_embeddings - 1)
        context_verified_source_index = torch.tensor(
            [
                int(item.get("context_verified_source_index", item["verified_source_index"]))
                for item in examples
            ],
            dtype=torch.long,
            device=device,
        ).clamp(0, self.context_verified_source_embedding.num_embeddings - 1)
        scalars = torch.tensor(
            [
                [
                    float(item["observed_source_verified"]),
                    float(item["claim_supported"]),
                    float(item.get("context_source_verified", item["observed_source_verified"])),
                    float(item.get("context_claim_supported", item["claim_supported"])),
                ]
                for item in examples
            ],
            dtype=torch.float32,
            device=device,
        )
        src = self.source_embedding(source_index)
        vsrc = self.verified_source_embedding(verified_source_index)
        csrc = self.context_source_embedding(context_source_index)
        cvsrc = self.context_verified_source_embedding(context_verified_source_index)
        scalar = self.scalar_proj(scalars)
        x = src + vsrc + csrc + cvsrc + scalar
        h = self.encoder(x)
        energy = self.energy_head(h).squeeze(-1)
        return energy, h


class WorldModelGatedAnswerRegister(nn.Module):
    """Fuse a graph register with data-world residuals.

    Extracted from Stage102E (608). This is the final gated fusion that produces
    the signal intended to be injected into the normal LM head path.
    """

    def __init__(
        self,
        *,
        d_model: int,
        graph_reasoner: nn.Module,
        world_model: nn.Module,
        world_d_model: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        width = int(hidden_dim or max(int(d_model), int(world_d_model) * 2))
        self.graph_reasoner = graph_reasoner
        self.world_model = world_model
        self.world_to_delta = nn.Sequential(
            nn.Linear(int(world_d_model) + 1, width),
            nn.SiLU(),
            nn.Linear(width, int(d_model)),
        )
        self.world_gate = nn.Sequential(
            nn.Linear(int(world_d_model) + 1, width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        graph_features: dict[str, Any],
        world_example: dict[str, Any],
        *,
        device: torch.device,
        world_off: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        graph_register, graph_metrics = self.graph_reasoner([graph_features], device=device)
        if bool(world_off):
            return graph_register, {
                "world_energy": 0.0,
                "world_gate": 0.0,
                **{f"graph_{key}": value for key, value in graph_metrics.items()},
            }
        energy, latent = self.world_model([world_example], device=device)
        signal = torch.cat([latent.float(), energy.float().unsqueeze(1)], dim=1)
        gate = self.world_gate(signal).to(dtype=graph_register.dtype)
        delta = self.world_to_delta(signal).to(dtype=graph_register.dtype)
        register = graph_register + gate * delta
        return register, {
            "world_energy": float(energy.detach().float().mean().cpu().item()),
            "world_gate": float(gate.detach().float().mean().cpu().item()),
            **{f"graph_{key}": value for key, value in graph_metrics.items()},
        }


__all__ = [
    "ProvenanceGraphReasoner",
    "ProvenanceDataWorldModel",
    "WorldModelGatedAnswerRegister",
    "build_provenance_register_from_config",
]


def build_provenance_register_from_config(cfg: "QTRMConfig") -> Optional["WorldModelGatedAnswerRegister"]:
    """Factory to create the full native provenance register from QTRMConfig.

    This is the recommended way for config-driven architecture-ization.
    Returns None if the feature is disabled.
    """
    if not getattr(cfg, "core_provenance_register_enabled", False):
        return None

    prov_d = int(getattr(cfg, "core_provenance_register_dim", 64))
    graph = ProvenanceGraphReasoner(d_model=prov_d, max_sources=8)
    world = ProvenanceDataWorldModel(d_model=32, max_sources=8)
    reg = WorldModelGatedAnswerRegister(
        d_model=prov_d,
        graph_reasoner=graph,
        world_model=world,
        world_d_model=32,
    )
    return reg