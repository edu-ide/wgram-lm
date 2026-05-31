from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import WGRAMV2Config


@dataclass
class ChunkEncoding:
    chunk_states: torch.Tensor
    chunk_valid: torch.Tensor
    dechunk_indices: torch.Tensor
    dechunk_has_completed_chunk: torch.Tensor
    metrics: dict[str, float | int | str]


class CausalByteChunkEncoder(nn.Module):
    """BLT-style causal byte chunk summarizer.

    Each selected boundary represents only bytes available up to that boundary.
    Future bytes never enter an earlier chunk state.
    """

    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        self.config = config
        d_model = int(config.d_model)
        self.boundary_scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.length_embed = nn.Embedding(int(config.patch_size) + 1, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.boundary_scorer[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, float(self.config.boundary_initial_logit))
        nn.init.normal_(self.length_embed.weight, mean=0.0, std=0.02)

    def _select_boundaries(
        self,
        byte_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        boundary_probs: torch.Tensor,
    ) -> list[list[int]]:
        batch, seq_len, _ = byte_embeddings.shape
        patch_size = int(self.config.patch_size)
        valid_mask = attention_mask.bool()
        if bool(self.config.force_fixed_boundaries):
            rows: list[list[int]] = []
            for row_idx in range(batch):
                valid_len = int(valid_mask[row_idx].sum().detach().cpu().item())
                positions = list(range(patch_size - 1, valid_len, patch_size))
                if valid_len > 0 and (not positions or positions[-1] != valid_len - 1):
                    positions.append(valid_len - 1)
                rows.append(positions)
            return rows

        rows = []
        threshold = float(self.config.dynamic_boundary_threshold)
        for row_idx in range(batch):
            valid_len = int(valid_mask[row_idx].sum().detach().cpu().item())
            positions: list[int] = []
            last = -1
            for pos in range(valid_len):
                hard_cap = pos - last >= patch_size
                learned_boundary = bool(boundary_probs[row_idx, pos].detach().cpu().item() >= threshold)
                if hard_cap or (learned_boundary and pos > last):
                    positions.append(pos)
                    last = pos
            if valid_len > 0 and (not positions or positions[-1] != valid_len - 1):
                positions.append(valid_len - 1)
            rows.append(positions)
        return rows

    def forward(
        self,
        byte_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ChunkEncoding:
        del input_ids
        batch, seq_len, d_model = byte_embeddings.shape
        valid_mask = attention_mask.bool()
        boundary_logits = self.boundary_scorer(byte_embeddings).squeeze(-1)
        boundary_logits = boundary_logits.masked_fill(~valid_mask, float(self.config.boundary_initial_logit))
        boundary_probs = torch.sigmoid(boundary_logits)
        selected_positions = self._select_boundaries(byte_embeddings, attention_mask, boundary_probs)
        max_chunks = max(1, max((len(row) for row in selected_positions), default=1))
        chunk_states = byte_embeddings.new_zeros((batch, max_chunks, d_model))
        chunk_valid = torch.zeros((batch, max_chunks), dtype=torch.bool, device=byte_embeddings.device)
        dechunk_indices = torch.zeros((batch, seq_len), dtype=torch.long, device=byte_embeddings.device)
        dechunk_has_completed_chunk = torch.zeros((batch, seq_len), dtype=torch.bool, device=byte_embeddings.device)
        summary_count = 0
        token_count = 0
        nonboundary_tokens = 0

        for row_idx, row_positions in enumerate(selected_positions):
            valid_len = int(attention_mask[row_idx].bool().sum().detach().cpu().item())
            for chunk_idx, boundary_pos in enumerate(row_positions):
                start = int(row_positions[chunk_idx - 1]) + 1 if chunk_idx > 0 else 0
                end = min(valid_len, int(boundary_pos) + 1)
                if end <= start:
                    start = int(boundary_pos)
                    end = min(valid_len, int(boundary_pos) + 1)
                states = byte_embeddings[row_idx, start:end]
                if int(states.shape[0]) <= 0:
                    continue
                recency = torch.linspace(1.0, 2.0, steps=int(states.shape[0]), dtype=states.dtype, device=states.device)
                local_boundary_probs = boundary_probs[row_idx, start:end].to(dtype=states.dtype)
                soft_weights = recency * (0.5 + local_boundary_probs)
                causal_mean = (states * soft_weights.unsqueeze(-1)).sum(dim=0) / soft_weights.sum().clamp_min(1.0e-6)
                boundary_embedding = byte_embeddings[row_idx, int(boundary_pos)]
                boundary_strength = boundary_probs[row_idx, int(boundary_pos)].to(dtype=states.dtype)
                boundary_embedding = boundary_embedding * (0.5 + boundary_strength)
                summary = self.summary_proj(torch.cat([boundary_embedding, causal_mean], dim=-1))
                length_id = torch.tensor(min(int(states.shape[0]), int(self.config.patch_size)), dtype=torch.long, device=states.device)
                chunk_states[row_idx, chunk_idx] = summary + self.length_embed(length_id)
                chunk_valid[row_idx, chunk_idx] = True
                summary_count += 1
                token_count += int(states.shape[0])
                nonboundary_tokens += max(0, int(states.shape[0]) - 1)
            cursor = -1
            for pos in range(valid_len):
                while cursor + 1 < len(row_positions) and row_positions[cursor + 1] <= pos:
                    cursor += 1
                if cursor >= 0:
                    dechunk_indices[row_idx, pos] = cursor + 1
                    dechunk_has_completed_chunk[row_idx, pos] = True

        metrics: dict[str, float | int | str] = {
            "boundary_state_source": "causal_chunk_summary",
            "dechunk_context_mode": "completed_chunk_or_bos",
            "dechunk_bos_fraction": float((valid_mask & ~dechunk_has_completed_chunk).detach().float().sum().cpu().item())
            / float(max(1, int(valid_mask.detach().sum().cpu().item()))),
            "boundary_selection_mode": "fixed_cap" if bool(self.config.force_fixed_boundaries) else "dynamic_soft_causal",
            "dynamic_boundary_threshold": float(self.config.dynamic_boundary_threshold),
            "dynamic_boundary_mean_probability": float(
                boundary_probs.masked_select(valid_mask).detach().float().mean().cpu().item()
            )
            if bool(valid_mask.any())
            else 0.0,
            "dynamic_boundary_hard_rate": float(
                (boundary_probs.masked_select(valid_mask) >= float(self.config.dynamic_boundary_threshold))
                .detach()
                .float()
                .mean()
                .cpu()
                .item()
            )
            if bool(valid_mask.any())
            else 0.0,
            "chunk_count": int(summary_count),
            "causal_chunk_summary_mean_len": float(token_count) / float(max(1, summary_count)),
            "causal_chunk_summary_nonboundary_tokens": int(nonboundary_tokens),
        }
        return ChunkEncoding(
            chunk_states=chunk_states,
            chunk_valid=chunk_valid,
            dechunk_indices=dechunk_indices,
            dechunk_has_completed_chunk=dechunk_has_completed_chunk,
            metrics=metrics,
        )
