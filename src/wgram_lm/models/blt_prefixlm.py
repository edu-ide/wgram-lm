from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .blt_components import BLTDLocalDecoder, NextImplicitByteProjector


IGNORE_LABEL_ID = -100


class BLTDByteLatentPrefixLM(nn.Module):
    """Minimal BLT-D-4 style model around the existing recurrent global core."""

    def __init__(
        self,
        *,
        global_core: nn.Module,
        vocab_size: int,
        d_model: int,
        patch_size: int = 4,
        mask_token_id: int | None = None,
        local_layers: int = 2,
        local_heads: int = 4,
        dropout: float = 0.0,
        clean_boundary_current_latent: bool = True,
        decoder_latent_mode: str = "add",
        patch_boundary_mode: str = "fixed",
        dynamic_min_patch_size: int = 2,
        dynamic_soft_patch_size: int = 0,
        hbf_boundary_threshold: float = 0.35,
        nitp_enabled: bool = False,
        nitp_hidden_dim: int = 0,
        answer_readback_mode: str = "none",
        answer_readback_gate_init: float = -4.0,
        answer_readback_temperature: float = 1.0,
        hnet_one_body_byte_gate_init: float = -2.0,
        hnet_one_body_latent_gate_init: float = 2.0,
        imta_trajectories: int = 1,
        imta_noise_std: float = 0.0,
        imta_selector_temperature: float = 1.0,
        imta_adapter_gate_init: float = -1.0,
        own_latent_prediction_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.global_core = global_core
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.mask_token_id = int(mask_token_id if mask_token_id is not None else vocab_size - 1)
        self.clean_boundary_current_latent = bool(clean_boundary_current_latent)
        self.decoder_latent_mode = str(decoder_latent_mode)
        self.patch_boundary_mode = str(patch_boundary_mode)
        self.dynamic_min_patch_size = int(dynamic_min_patch_size)
        self.dynamic_soft_patch_size = int(dynamic_soft_patch_size)
        self.hbf_boundary_threshold = float(hbf_boundary_threshold)
        self.nitp_enabled = bool(nitp_enabled)
        self.answer_readback_mode = str(answer_readback_mode)
        self.answer_readback_temperature = float(answer_readback_temperature)
        self.hnet_one_body_byte_gate_init = float(hnet_one_body_byte_gate_init)
        self.hnet_one_body_latent_gate_init = float(hnet_one_body_latent_gate_init)
        self.imta_trajectories = max(1, int(imta_trajectories))
        self.imta_noise_std = max(0.0, float(imta_noise_std))
        self.imta_selector_temperature = max(1.0e-6, float(imta_selector_temperature))
        self.imta_adapter_gate_init = float(imta_adapter_gate_init)
        self.own_latent_prediction_enabled = bool(own_latent_prediction_enabled)
        self.last_pack_metrics: dict[str, float | int] = {}
        self.last_boundary_metrics: dict[str, float | int] = {}
        self.last_readback_metrics: dict[str, float | int | str] = {}
        self._last_own_latent_prediction_loss: torch.Tensor | None = None
        self._last_imta_diversity_loss: torch.Tensor | None = None
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.decoder_latent_mode not in {"add", "cross", "add_cross", "hier_add", "hier_add_cross", "one_body"}:
            raise ValueError(
                "decoder_latent_mode must be one of: add, cross, add_cross, "
                "hier_add, hier_add_cross, one_body"
            )
        if self.answer_readback_mode not in {
            "none",
            "self_embedding",
            "anchor_embedding",
            "selected_anchor_embedding",
        }:
            raise ValueError(
                "answer_readback_mode must be one of: none, self_embedding, "
                "anchor_embedding, selected_anchor_embedding"
            )
        if self.patch_boundary_mode not in {
            "fixed",
            "utf8_entropy",
            "byteflow_proxy",
            "hbf_byteflow",
            "blt_ngram_entropy",
            "learned_primary",
            "learned_boundary",
            "hnet_dechunk",
            "hnetpp_flow_dechunk",
        }:
            raise ValueError(
                "patch_boundary_mode must be one of: fixed, utf8_entropy, byteflow_proxy, hbf_byteflow, blt_ngram_entropy, learned_primary, learned_boundary, hnet_dechunk, hnetpp_flow_dechunk"
            )
        if self.dynamic_min_patch_size <= 0:
            raise ValueError("dynamic_min_patch_size must be positive")
        if self.mask_token_id < 0 or self.mask_token_id >= self.vocab_size:
            raise ValueError("mask_token_id must be inside vocab")
        use_cross_attention = self.decoder_latent_mode in {"cross", "add_cross", "hier_add_cross"}
        self.byte_embed = nn.Embedding(int(vocab_size), int(d_model))
        self.byte_pos_embed = nn.Embedding(int(patch_size), int(d_model))
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(int(d_model) * int(patch_size)),
            nn.Linear(int(d_model) * int(patch_size), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.semantic_boundary_scorer = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), 1),
        )
        self.semantic_chunk_proj = nn.Sequential(
            nn.LayerNorm(int(d_model) * 2),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.hierarchical_chunk_proj = nn.Sequential(
            nn.LayerNorm(int(d_model) * 2),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.hierarchical_chunk_gate = nn.Sequential(
            nn.LayerNorm(int(d_model) * 2),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), 1),
        )
        self.hnet_byte_speaker = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(vocab_size)),
        )
        self.hnet_causal_speaker = BLTDLocalDecoder(
            int(d_model),
            int(vocab_size),
            patch_size=int(patch_size),
            n_heads=int(local_heads),
            n_layers=int(local_layers),
            dropout=float(dropout),
            causal=True,
            cross_attention=False,
        )
        self.hnet_causal_speaker.head.weight = self.hnet_byte_speaker[-1].weight
        self.hnet_latent_bridge = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.hnet_byte_residual_gate_logit = nn.Parameter(
            torch.tensor(float(hnet_one_body_byte_gate_init), dtype=torch.float32)
        )
        self.hnet_latent_residual_gate_logit = nn.Parameter(
            torch.tensor(float(hnet_one_body_latent_gate_init), dtype=torch.float32)
        )
        self.imta_trajectory_offsets = nn.Parameter(torch.zeros(self.imta_trajectories, int(d_model)))
        self.imta_trajectory_adapter_gate_logit = nn.Parameter(
            torch.full((self.imta_trajectories,), float(imta_adapter_gate_init), dtype=torch.float32)
        )
        self.imta_trajectory_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(d_model)),
                    nn.Linear(int(d_model), int(d_model)),
                    nn.GELU(),
                    nn.Linear(int(d_model), int(d_model)),
                )
                for _ in range(self.imta_trajectories)
            ]
        )
        self.imta_trajectory_selector = nn.Sequential(
            nn.LayerNorm(int(d_model) * 2),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), 1),
        )
        self.own_latent_predictor = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.answer_anchor_head = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(vocab_size)),
        )
        self.answer_workspace_selector = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), 1),
        )
        self.register_buffer(
            "ngram_unigram_surprisal",
            torch.zeros(int(vocab_size), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ngram_bigram_surprisal",
            torch.zeros((int(vocab_size), int(vocab_size)), dtype=torch.float32),
            persistent=False,
        )
        self.patch_len_embed = nn.Embedding(int(patch_size) + 1, int(d_model))
        self.bos_latent = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.answer_readback_gate_logit = nn.Parameter(
            torch.tensor(float(answer_readback_gate_init), dtype=torch.float32)
        )
        self.clean_decoder = BLTDLocalDecoder(
            int(d_model),
            int(vocab_size),
            patch_size=int(patch_size),
            n_heads=int(local_heads),
            n_layers=int(local_layers),
            dropout=float(dropout),
            causal=True,
            cross_attention=use_cross_attention,
        )
        self.diffusion_decoder = BLTDLocalDecoder(
            int(d_model),
            int(vocab_size),
            patch_size=int(patch_size),
            n_heads=int(local_heads),
            n_layers=int(local_layers),
            dropout=float(dropout),
            causal=False,
            cross_attention=use_cross_attention,
        )
        self.nitp_projector = (
            NextImplicitByteProjector(int(d_model), hidden_dim=int(nitp_hidden_dim))
            if self.nitp_enabled
            else None
        )
        self.reset_parameters()

    def set_ngram_entropy_tables(self, unigram: torch.Tensor, bigram: torch.Tensor) -> None:
        if tuple(unigram.shape) != (self.vocab_size,):
            raise ValueError(f"unigram table must have shape ({self.vocab_size},), got {tuple(unigram.shape)}")
        if tuple(bigram.shape) != (self.vocab_size, self.vocab_size):
            raise ValueError(
                f"bigram table must have shape ({self.vocab_size}, {self.vocab_size}), got {tuple(bigram.shape)}"
            )
        self.ngram_unigram_surprisal.copy_(unigram.detach().to(device=self.ngram_unigram_surprisal.device, dtype=torch.float32))
        self.ngram_bigram_surprisal.copy_(bigram.detach().to(device=self.ngram_bigram_surprisal.device, dtype=torch.float32))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.byte_pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.patch_len_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bos_latent, mean=0.0, std=0.02)
        final_bridge = self.hnet_latent_bridge[-1]
        if isinstance(final_bridge, nn.Linear):
            nn.init.zeros_(final_bridge.weight)
            nn.init.zeros_(final_bridge.bias)
        if int(self.imta_trajectories) > 1:
            nn.init.normal_(self.imta_trajectory_offsets, mean=0.0, std=0.02)
            with torch.no_grad():
                self.imta_trajectory_offsets[0].zero_()
        else:
            nn.init.zeros_(self.imta_trajectory_offsets)
        for adapter_idx, adapter in enumerate(self.imta_trajectory_adapters):
            final_adapter = adapter[-1]
            if isinstance(final_adapter, nn.Linear):
                if int(adapter_idx) == 0:
                    nn.init.zeros_(final_adapter.weight)
                    nn.init.zeros_(final_adapter.bias)
                else:
                    nn.init.normal_(final_adapter.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(final_adapter.bias)
        selector_head = self.imta_trajectory_selector[-1]
        if isinstance(selector_head, nn.Linear):
            nn.init.zeros_(selector_head.weight)
            nn.init.zeros_(selector_head.bias)
        final_own_latent = self.own_latent_predictor[-1]
        if isinstance(final_own_latent, nn.Linear):
            nn.init.zeros_(final_own_latent.weight)
            nn.init.zeros_(final_own_latent.bias)

    def _grouped_patch_embeddings(
        self,
        grouped_ids: torch.Tensor,
        grouped_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_ids = grouped_ids.clamp(min=0, max=self.vocab_size - 1)
        byte_embeddings = self.byte_embed(safe_ids)
        valid = grouped_mask.to(byte_embeddings.dtype).unsqueeze(-1)
        grouped_byte_embeddings = byte_embeddings * valid
        patch_lengths = grouped_mask.to(torch.long).sum(dim=-1).clamp(min=0, max=self.patch_size)
        if self.patch_boundary_mode in {"learned_primary", "learned_boundary"}:
            boundary_logits = self.semantic_boundary_scorer(grouped_byte_embeddings).squeeze(-1)
            boundary_logits = boundary_logits.masked_fill(~grouped_mask.bool(), -20.0)
            gates = torch.sigmoid(boundary_logits) * grouped_mask.to(boundary_logits.dtype)
            positions = torch.arange(self.patch_size, device=grouped_ids.device, dtype=gates.dtype).view(1, 1, -1)
            recency = 1.0 + positions / float(max(1, self.patch_size - 1))
            weights = (0.25 + gates) * recency * grouped_mask.to(gates.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            semantic_mean = (grouped_byte_embeddings * weights.unsqueeze(-1)).sum(dim=2)
            hard_mean = grouped_byte_embeddings.sum(dim=2) / patch_lengths.clamp(min=1).to(grouped_byte_embeddings.dtype).unsqueeze(-1)
            patch_embeddings = self.semantic_chunk_proj(torch.cat([semantic_mean, hard_mean], dim=-1))
            patch_embeddings = patch_embeddings + self.patch_len_embed(patch_lengths)
            valid_gates = gates[grouped_mask.bool()]
            if int(valid_gates.numel()) > 0:
                clipped = valid_gates.clamp(1e-6, 1.0 - 1e-6)
                entropy = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())
                self.last_pack_metrics = {
                    "learned_chunk_gate_mean": float(valid_gates.mean().detach().cpu().item()),
                    "learned_chunk_gate_entropy": float(entropy.mean().detach().cpu().item()),
                    "learned_chunk_gate_std": float(valid_gates.std(unbiased=False).detach().cpu().item()),
                }
            else:
                self.last_pack_metrics = {
                    "learned_chunk_gate_mean": 0.0,
                    "learned_chunk_gate_entropy": 0.0,
                    "learned_chunk_gate_std": 0.0,
                }
            self.last_pack_metrics.update(self.last_boundary_metrics)
            return safe_ids, grouped_byte_embeddings, patch_lengths, patch_embeddings
        patch_embeddings = self.patch_proj(
            grouped_byte_embeddings.reshape(
                grouped_ids.shape[0],
                grouped_ids.shape[1],
                self.patch_size * self.d_model,
            )
        ) + self.patch_len_embed(patch_lengths)
        self.last_pack_metrics = dict(self.last_boundary_metrics)
        return safe_ids, grouped_byte_embeddings, patch_lengths, patch_embeddings

    def _pad_to_patch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
        batch, seq_len = input_ids.shape
        patch_size = self.patch_size
        latent_len = int(math.ceil(seq_len / float(patch_size)))
        padded_len = latent_len * patch_size
        pad_len = padded_len - seq_len
        if pad_len <= 0:
            return input_ids, attention_mask, labels, seq_len, latent_len
        input_ids = F.pad(input_ids, (0, pad_len), value=0)
        attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        if labels is not None:
            labels = F.pad(labels, (0, pad_len), value=IGNORE_LABEL_ID)
        return input_ids, attention_mask, labels, seq_len, latent_len

    @staticmethod
    def _token_byte_value(token_id: int) -> int:
        return int(token_id) - 2 if 2 <= int(token_id) <= 257 else -1

    @staticmethod
    def _is_utf8_continuation(byte_value: int) -> bool:
        return 0x80 <= int(byte_value) <= 0xBF

    @staticmethod
    def _byte_kind(byte_value: int) -> str:
        b = int(byte_value)
        if b < 0:
            return "special"
        if b >= 0x80:
            return "utf8"
        if 48 <= b <= 57:
            return "digit"
        if 65 <= b <= 90 or 97 <= b <= 122:
            return "alpha"
        if b in {9, 10, 13, 32}:
            return "space"
        return "punct"

    def _should_close_dynamic_patch(self, patch_tokens: list[int], next_token: int | None) -> bool:
        if not patch_tokens:
            return False
        current_len = len(patch_tokens)
        if next_token is None:
            return True
        if current_len >= self.patch_size:
            return True
        current_byte = self._token_byte_value(patch_tokens[-1])
        next_byte = self._token_byte_value(int(next_token))
        # Keep UTF-8 codepoints intact.  Korean and other non-ASCII scripts are
        # exactly where byte-level tokenizers should avoid arbitrary cuts.
        if self._is_utf8_continuation(next_byte):
            return False
        if current_len < self.dynamic_min_patch_size:
            return False
        if current_byte < 0 or next_byte < 0:
            return True
        if current_byte >= 0x80:
            return True
        current_kind = self._byte_kind(current_byte)
        next_kind = self._byte_kind(next_byte)
        if next_byte >= 0x80:
            return True
        if current_kind in {"space", "punct"}:
            return True
        if current_kind != next_kind:
            return True
        if self.dynamic_soft_patch_size > 0 and current_len >= self.dynamic_soft_patch_size:
            return True
        return False

    def _pack_fixed_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        padded_ids, padded_mask, padded_labels, _, latent_len = self._pad_to_patch(
            input_ids,
            attention_mask,
            labels,
        )
        assert padded_labels is not None
        batch = int(input_ids.shape[0])
        grouped_ids = padded_ids.reshape(batch, latent_len, self.patch_size)
        grouped_mask = padded_mask.reshape(batch, latent_len, self.patch_size)
        grouped_labels = padded_labels.reshape(batch, latent_len, self.patch_size)
        patch_lengths = grouped_mask.to(torch.long).sum(dim=-1)
        valid_patches = int((patch_lengths > 0).sum().detach().cpu().item())
        valid_bytes = int(grouped_mask.sum().detach().cpu().item())
        return grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches

    def _pack_dynamic_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        batch = int(input_ids.shape[0])
        rows: list[list[tuple[list[int], list[int]]]] = []
        for row_idx in range(batch):
            valid_len = int(mask_cpu[row_idx].sum().item())
            row_ids = [int(v) for v in ids_cpu[row_idx, :valid_len].tolist()]
            row_labels = [int(v) for v in labels_cpu[row_idx, :valid_len].tolist()]
            patches: list[tuple[list[int], list[int]]] = []
            current_ids: list[int] = []
            current_labels: list[int] = []
            for pos, token_id in enumerate(row_ids):
                current_ids.append(int(token_id))
                current_labels.append(int(row_labels[pos]))
                next_token = row_ids[pos + 1] if pos + 1 < len(row_ids) else None
                if self._should_close_dynamic_patch(current_ids, next_token):
                    patches.append((current_ids, current_labels))
                    current_ids = []
                    current_labels = []
            if current_ids:
                patches.append((current_ids, current_labels))
            rows.append(patches or [([], [])])
        latent_len = max(len(row) for row in rows)
        grouped_ids = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grouped_mask = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        grouped_labels = torch.full(
            (batch, latent_len, self.patch_size),
            int(IGNORE_LABEL_ID),
            dtype=labels.dtype,
            device=input_ids.device,
        )
        valid_bytes = 0
        valid_patches = 0
        for row_idx, patches in enumerate(rows):
            for patch_idx, (patch_ids, patch_labels) in enumerate(patches):
                if not patch_ids:
                    continue
                patch_len = min(len(patch_ids), self.patch_size)
                grouped_ids[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_ids[:patch_len],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                grouped_labels[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_labels[:patch_len],
                    dtype=labels.dtype,
                    device=input_ids.device,
                )
                grouped_mask[row_idx, patch_idx, :patch_len] = 1
                valid_bytes += int(patch_len)
                valid_patches += 1
        return grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches

    def _pack_byteflow_proxy_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        with torch.no_grad():
            safe_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
            embeddings = self.byte_embed(safe_ids).detach().float()
            embeddings = F.normalize(embeddings, dim=-1)
            if embeddings.shape[1] > 1:
                change_scores = 1.0 - (embeddings[:, :-1] * embeddings[:, 1:]).sum(dim=-1)
                change_scores = change_scores.detach().cpu()
            else:
                change_scores = torch.zeros(
                    (embeddings.shape[0], 0),
                    dtype=torch.float32,
                )
        batch = int(input_ids.shape[0])
        rows: list[list[tuple[list[int], list[int]]]] = []
        for row_idx in range(batch):
            valid_len = int(mask_cpu[row_idx].sum().item())
            row_ids = [int(v) for v in ids_cpu[row_idx, :valid_len].tolist()]
            row_labels = [int(v) for v in labels_cpu[row_idx, :valid_len].tolist()]
            if not row_ids:
                rows.append([([], [])])
                continue
            candidate_positions: list[tuple[int, float]] = []
            for pos in range(1, len(row_ids)):
                prev_byte = self._token_byte_value(row_ids[pos - 1])
                this_byte = self._token_byte_value(row_ids[pos])
                if self._is_utf8_continuation(this_byte):
                    continue
                if pos < self.dynamic_min_patch_size:
                    continue
                score = float(change_scores[row_idx, pos - 1].item()) if pos - 1 < change_scores.shape[1] else 0.0
                if prev_byte < 0 or this_byte < 0:
                    score += 1.0
                if prev_byte >= 0x80 or this_byte >= 0x80:
                    score += 0.25
                if self._byte_kind(prev_byte) != self._byte_kind(this_byte):
                    score += 0.25
                candidate_positions.append((pos, score))
            desired_patches = max(1, int(math.ceil(len(row_ids) / float(max(1, self.dynamic_min_patch_size)))))
            boundary_budget = max(0, desired_patches - 1)
            boundaries = [0, len(row_ids)]
            for candidate_pos, _ in sorted(
                candidate_positions,
                key=lambda item: float(item[1]),
                reverse=True,
            ):
                if len(boundaries) - 2 >= boundary_budget:
                    break
                pos = int(candidate_pos)
                left = max(boundary for boundary in boundaries if boundary < pos)
                right = min(boundary for boundary in boundaries if boundary > pos)
                if pos - left < self.dynamic_min_patch_size:
                    continue
                if right - pos < self.dynamic_min_patch_size:
                    continue
                boundaries.append(pos)
                boundaries.sort()
            chosen = set(boundaries[1:-1])
            patches: list[tuple[list[int], list[int]]] = []
            current_ids: list[int] = []
            current_labels: list[int] = []
            for pos, token_id in enumerate(row_ids):
                if current_ids and (len(current_ids) >= self.patch_size or pos in chosen):
                    patches.append((current_ids, current_labels))
                    current_ids = []
                    current_labels = []
                current_ids.append(int(token_id))
                current_labels.append(int(row_labels[pos]))
            if current_ids:
                patches.append((current_ids, current_labels))
            rows.append(patches or [([], [])])
        latent_len = max(len(row) for row in rows)
        grouped_ids = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grouped_mask = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        grouped_labels = torch.full(
            (batch, latent_len, self.patch_size),
            int(IGNORE_LABEL_ID),
            dtype=labels.dtype,
            device=input_ids.device,
        )
        valid_bytes = 0
        valid_patches = 0
        for row_idx, patches in enumerate(rows):
            for patch_idx, (patch_ids, patch_labels) in enumerate(patches):
                if not patch_ids:
                    continue
                patch_len = min(len(patch_ids), self.patch_size)
                grouped_ids[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_ids[:patch_len],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                grouped_labels[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_labels[:patch_len],
                    dtype=labels.dtype,
                    device=input_ids.device,
                )
                grouped_mask[row_idx, patch_idx, :patch_len] = 1
                valid_bytes += int(patch_len)
                valid_patches += 1
        return grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches

    def _ngram_entropy_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            safe_ids = input_ids.clamp(min=0, max=self.vocab_size - 1).to(torch.long)
            prev_ids = F.pad(safe_ids[:, :-1], (1, 0), value=0)
            bigram_scores = self.ngram_bigram_surprisal[prev_ids, safe_ids]
            unigram_scores = self.ngram_unigram_surprisal[safe_ids]
            if safe_ids.shape[1] > 0:
                bigram_scores[:, 0] = unigram_scores[:, 0]
            return (bigram_scores * attention_mask.to(bigram_scores.dtype)).detach().cpu()

    def _pack_blt_ngram_entropy_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        entropy_scores = self._ngram_entropy_scores(input_ids, attention_mask)
        batch = int(input_ids.shape[0])
        rows: list[list[tuple[list[int], list[int]]]] = []
        selected_boundaries = 0
        score_sum = 0.0
        for row_idx in range(batch):
            valid_len = int(mask_cpu[row_idx].sum().item())
            row_ids = [int(v) for v in ids_cpu[row_idx, :valid_len].tolist()]
            row_labels = [int(v) for v in labels_cpu[row_idx, :valid_len].tolist()]
            if not row_ids:
                rows.append([([], [])])
                continue
            candidate_positions: list[tuple[int, float]] = []
            for pos in range(1, len(row_ids)):
                this_byte = self._token_byte_value(row_ids[pos])
                if self._is_utf8_continuation(this_byte):
                    continue
                score = float(entropy_scores[row_idx, pos].item()) if pos < entropy_scores.shape[1] else 0.0
                candidate_positions.append((pos, score))
            target_patch_size = max(1, int(self.dynamic_min_patch_size))
            desired_patches = max(1, int(math.ceil(len(row_ids) / float(target_patch_size))))
            boundary_budget = max(0, desired_patches - 1)
            boundaries = [0, len(row_ids)]
            for candidate_pos, score in sorted(candidate_positions, key=lambda item: float(item[1]), reverse=True):
                if len(boundaries) - 2 >= boundary_budget:
                    break
                pos = int(candidate_pos)
                left = max(boundary for boundary in boundaries if boundary < pos)
                right = min(boundary for boundary in boundaries if boundary > pos)
                if pos - left < target_patch_size:
                    continue
                if right - pos < target_patch_size:
                    continue
                boundaries.append(pos)
                boundaries.sort()
                selected_boundaries += 1
                score_sum += float(score)
            chosen = set(boundaries[1:-1])
            patches: list[tuple[list[int], list[int]]] = []
            current_ids: list[int] = []
            current_labels: list[int] = []
            for pos, token_id in enumerate(row_ids):
                if current_ids and (len(current_ids) >= self.patch_size or pos in chosen):
                    patches.append((current_ids, current_labels))
                    current_ids = []
                    current_labels = []
                current_ids.append(int(token_id))
                current_labels.append(int(row_labels[pos]))
            if current_ids:
                patches.append((current_ids, current_labels))
            rows.append(patches or [([], [])])
        latent_len = max(len(row) for row in rows)
        grouped_ids = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grouped_mask = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        grouped_labels = torch.full(
            (batch, latent_len, self.patch_size),
            int(IGNORE_LABEL_ID),
            dtype=labels.dtype,
            device=input_ids.device,
        )
        valid_bytes = 0
        valid_patches = 0
        for row_idx, patches in enumerate(rows):
            for patch_idx, (patch_ids, patch_labels) in enumerate(patches):
                if not patch_ids:
                    continue
                patch_len = min(len(patch_ids), self.patch_size)
                grouped_ids[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_ids[:patch_len],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                grouped_labels[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_labels[:patch_len],
                    dtype=labels.dtype,
                    device=labels.device,
                )
                grouped_mask[row_idx, patch_idx, :patch_len] = 1
                valid_bytes += int(patch_len)
                valid_patches += 1
        self.last_boundary_metrics = {
            "ngram_entropy_selected_boundaries": int(selected_boundaries),
            "ngram_entropy_boundary_score_mean": float(score_sum / float(max(1, selected_boundaries))),
        }
        return grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches

    def _hbf_boundary_scores(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            safe_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
            embeddings = self.byte_embed(safe_ids).detach().float()
            if embeddings.shape[1] <= 1:
                return torch.zeros((embeddings.shape[0], 0), dtype=torch.float32)
            normalized = F.normalize(embeddings, dim=-1)
            cosine_change = 1.0 - (normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1)
            l2_change = (embeddings[:, 1:] - embeddings[:, :-1]).float().pow(2).mean(dim=-1).sqrt()
            row_scale = l2_change.mean(dim=1, keepdim=True).clamp_min(1e-6)
            coding_proxy = (l2_change / row_scale).clamp(max=4.0) / 4.0
            return (0.7 * cosine_change + 0.3 * coding_proxy).detach().cpu()

    def _should_close_hbf_patch(
        self,
        patch_tokens: list[int],
        next_token: int | None,
        boundary_score: float,
    ) -> bool:
        if not patch_tokens:
            return False
        current_len = len(patch_tokens)
        if next_token is None:
            return True
        if current_len >= self.patch_size:
            return True
        current_byte = self._token_byte_value(patch_tokens[-1])
        next_byte = self._token_byte_value(int(next_token))
        # H-Net-style hierarchy starts by respecting the byte structure of a
        # character.  Korean UTF-8 bytes should become one local unit before the
        # global recurrent core sees the compressed note.
        if self._is_utf8_continuation(next_byte):
            return False
        if current_len < self.dynamic_min_patch_size:
            return False
        if current_byte < 0 or next_byte < 0:
            return True
        current_kind = self._byte_kind(current_byte)
        next_kind = self._byte_kind(next_byte)
        target_len = int(self.dynamic_soft_patch_size) if int(self.dynamic_soft_patch_size) > 0 else int(self.dynamic_min_patch_size)
        if current_len < target_len:
            return False
        if current_byte >= 0x80 or next_byte >= 0x80:
            return True
        if current_kind in {"space", "punct"}:
            return True
        if next_kind in {"space", "punct"}:
            return True
        if current_kind != next_kind:
            return True
        if float(boundary_score) >= float(self.hbf_boundary_threshold):
            return True
        return False

    def _pack_hbf_byteflow_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        boundary_scores = self._hbf_boundary_scores(input_ids)
        batch = int(input_ids.shape[0])
        rows: list[list[tuple[list[int], list[int]]]] = []
        for row_idx in range(batch):
            valid_len = int(mask_cpu[row_idx].sum().item())
            row_ids = [int(v) for v in ids_cpu[row_idx, :valid_len].tolist()]
            row_labels = [int(v) for v in labels_cpu[row_idx, :valid_len].tolist()]
            patches: list[tuple[list[int], list[int]]] = []
            current_ids: list[int] = []
            current_labels: list[int] = []
            for pos, token_id in enumerate(row_ids):
                current_ids.append(int(token_id))
                current_labels.append(int(row_labels[pos]))
                next_token = row_ids[pos + 1] if pos + 1 < len(row_ids) else None
                score = float(boundary_scores[row_idx, pos].item()) if pos < boundary_scores.shape[1] else 0.0
                if self._should_close_hbf_patch(current_ids, next_token, score):
                    patches.append((current_ids, current_labels))
                    current_ids = []
                    current_labels = []
            if current_ids:
                patches.append((current_ids, current_labels))
            rows.append(patches or [([], [])])
        latent_len = max(len(row) for row in rows)
        grouped_ids = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grouped_mask = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        grouped_labels = torch.full(
            (batch, latent_len, self.patch_size),
            int(IGNORE_LABEL_ID),
            dtype=labels.dtype,
            device=input_ids.device,
        )
        valid_bytes = 0
        valid_patches = 0
        for row_idx, patches in enumerate(rows):
            for patch_idx, (patch_ids, patch_labels) in enumerate(patches):
                if not patch_ids:
                    continue
                patch_len = min(len(patch_ids), self.patch_size)
                grouped_ids[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_ids[:patch_len],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                grouped_labels[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_labels[:patch_len],
                    dtype=labels.dtype,
                    device=input_ids.device,
                )
                grouped_mask[row_idx, patch_idx, :patch_len] = 1
                valid_bytes += int(patch_len)
                valid_patches += 1
        return grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches

    def _learned_boundary_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            safe_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
            byte_embeddings = self.byte_embed(safe_ids)
            logits = self.semantic_boundary_scorer(byte_embeddings).squeeze(-1)
            probs = torch.sigmoid(logits)
            probs = probs * attention_mask.to(probs.dtype)
        return probs.detach().cpu()

    def _should_close_learned_boundary_patch(
        self,
        patch_tokens: list[int],
        next_token: int | None,
        boundary_prob: float,
    ) -> bool:
        if not patch_tokens:
            return False
        current_len = len(patch_tokens)
        if next_token is None:
            return True
        if current_len >= self.patch_size:
            return True
        next_byte = self._token_byte_value(int(next_token))
        if self._is_utf8_continuation(next_byte):
            return False
        if current_len < self.dynamic_min_patch_size:
            return False
        return float(boundary_prob) >= float(self.hbf_boundary_threshold)

    def _pack_learned_boundary_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        boundary_probs = self._learned_boundary_probabilities(input_ids, attention_mask)
        batch = int(input_ids.shape[0])
        rows: list[list[tuple[list[int], list[int]]]] = []
        learned_boundary_count = 0
        for row_idx in range(batch):
            valid_len = int(mask_cpu[row_idx].sum().item())
            row_ids = [int(v) for v in ids_cpu[row_idx, :valid_len].tolist()]
            row_labels = [int(v) for v in labels_cpu[row_idx, :valid_len].tolist()]
            patches: list[tuple[list[int], list[int]]] = []
            current_ids: list[int] = []
            current_labels: list[int] = []
            for pos, token_id in enumerate(row_ids):
                current_ids.append(int(token_id))
                current_labels.append(int(row_labels[pos]))
                next_token = row_ids[pos + 1] if pos + 1 < len(row_ids) else None
                prob = float(boundary_probs[row_idx, pos].item()) if pos < boundary_probs.shape[1] else 0.0
                if self._should_close_learned_boundary_patch(current_ids, next_token, prob):
                    if next_token is not None and len(current_ids) < self.patch_size:
                        learned_boundary_count += 1
                    patches.append((current_ids, current_labels))
                    current_ids = []
                    current_labels = []
            if current_ids:
                patches.append((current_ids, current_labels))
            rows.append(patches or [([], [])])
        latent_len = max(len(row) for row in rows)
        grouped_ids = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grouped_mask = torch.zeros(
            (batch, latent_len, self.patch_size),
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        grouped_labels = torch.full(
            (batch, latent_len, self.patch_size),
            int(IGNORE_LABEL_ID),
            dtype=labels.dtype,
            device=labels.device,
        )
        valid_bytes = 0
        valid_patches = 0
        for row_idx, patches in enumerate(rows):
            for patch_idx, (patch_ids, patch_labels) in enumerate(patches):
                if not patch_ids:
                    continue
                patch_len = min(len(patch_ids), self.patch_size)
                grouped_ids[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_ids[:patch_len],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                grouped_labels[row_idx, patch_idx, :patch_len] = torch.tensor(
                    patch_labels[:patch_len],
                    dtype=labels.dtype,
                    device=labels.device,
                )
                grouped_mask[row_idx, patch_idx, :patch_len] = 1
                valid_bytes += int(patch_len)
                valid_patches += 1
        valid_probs = boundary_probs[mask_cpu.bool()]
        if int(valid_probs.numel()) > 0:
            self.last_boundary_metrics = {
                "learned_boundary_prob_mean": float(valid_probs.mean().item()),
                "learned_boundary_prob_std": float(valid_probs.std(unbiased=False).item()),
                "learned_boundary_valid_boundaries": int(learned_boundary_count),
            }
        else:
            self.last_boundary_metrics = {
                "learned_boundary_prob_mean": 0.0,
                "learned_boundary_prob_std": 0.0,
                "learned_boundary_valid_boundaries": 0,
            }
        return grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches

    def pack_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        self.last_boundary_metrics = {}
        if self.patch_boundary_mode == "fixed":
            return self._pack_fixed_patches(input_ids, attention_mask, labels)
        if self.patch_boundary_mode == "learned_primary":
            return self._pack_fixed_patches(input_ids, attention_mask, labels)
        if self.patch_boundary_mode == "learned_boundary":
            return self._pack_learned_boundary_patches(input_ids, attention_mask, labels)
        if self.patch_boundary_mode == "byteflow_proxy":
            return self._pack_byteflow_proxy_patches(input_ids, attention_mask, labels)
        if self.patch_boundary_mode == "blt_ngram_entropy":
            return self._pack_blt_ngram_entropy_patches(input_ids, attention_mask, labels)
        if self.patch_boundary_mode == "hbf_byteflow":
            return self._pack_hbf_byteflow_patches(input_ids, attention_mask, labels)
        return self._pack_dynamic_patches(input_ids, attention_mask, labels)

    def _global_hidden(
        self,
        patch_embeddings: torch.Tensor,
        *,
        think_steps: int,
    ) -> torch.Tensor:
        core = self.global_core
        x = patch_embeddings
        if getattr(core, "position_embedding_mode", "") in {"learned", "randomized"}:
            seq_len = int(x.shape[1])
            position_ids = core._position_ids(seq_len, x.device)
            x = x + core.pos_embed(position_ids)
        return core._forward_embedded_impl(
            x,
            think_steps=int(think_steps),
            return_hidden=True,
        )

    def encode_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        safe_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
        padded_ids, padded_mask, _, original_len, latent_len = self._pad_to_patch(
            safe_ids,
            attention_mask,
            None,
        )
        byte_embeddings = self.byte_embed(padded_ids)
        valid = padded_mask.to(byte_embeddings.dtype).unsqueeze(-1)
        byte_embeddings = byte_embeddings * valid
        batch = int(byte_embeddings.shape[0])
        grouped_ids = padded_ids.reshape(batch, latent_len, self.patch_size)
        grouped_mask = padded_mask.reshape(batch, latent_len, self.patch_size)
        _, _, _, patch_embeddings = self._grouped_patch_embeddings(grouped_ids, grouped_mask)
        hidden = self._global_hidden(patch_embeddings, think_steps=int(think_steps))
        return hidden, padded_ids, padded_mask, original_len

    def shifted_patch_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, latent_len, _ = hidden.shape
        bos = self.bos_latent.expand(batch, 1, self.d_model).to(dtype=hidden.dtype)
        if latent_len <= 1:
            return bos
        return torch.cat([bos, hidden[:, :-1]], dim=1)

    def clean_patch_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        shifted = self.shifted_patch_condition(hidden)
        if not bool(self.clean_boundary_current_latent):
            return shifted
        # For positions at the end of a fixed byte patch, the full patch is
        # already visible in the prefix.  BLT-style decoding should let the
        # local decoder use that current patch latent to predict the first byte
        # of the next patch.  Earlier positions still use the previous latent to
        # avoid looking through future bytes inside the same patch.
        end_mask = torch.zeros(
            (1, 1, self.patch_size, 1),
            dtype=torch.bool,
            device=hidden.device,
        )
        end_mask[:, :, self.patch_size - 1, :] = True
        return torch.where(end_mask, hidden.unsqueeze(2), shifted.unsqueeze(2))

    def hierarchical_patch_condition(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch, latent_len, _ = hidden.shape
        bos = self.bos_latent.expand(batch, 1, self.d_model).to(dtype=hidden.dtype)
        prev_one = self.shifted_patch_condition(hidden)
        if latent_len <= 1:
            prev_two = bos
        else:
            prev_two = torch.cat([bos, bos, hidden[:, :-2]], dim=1)[:, :latent_len]
        pair = torch.cat([prev_two, prev_one], dim=-1)
        gate = torch.sigmoid(self.hierarchical_chunk_gate(pair))
        upper_memory = torch.tanh(self.hierarchical_chunk_proj(pair))
        condition = gate.unsqueeze(2) * upper_memory.unsqueeze(2)
        condition = condition.expand(batch, latent_len, self.patch_size, self.d_model)
        gate_values = gate.detach().float()
        memory_norm = upper_memory.detach().float().norm(dim=-1).mean()
        metrics = {
            "hier_chunk_gate_mean": float(gate_values.mean().cpu().item()),
            "hier_chunk_gate_std": float(gate_values.std(unbiased=False).cpu().item()),
            "hier_chunk_memory_norm": float(memory_norm.cpu().item()),
        }
        return condition.to(dtype=hidden.dtype), metrics

    def prefix_latent_context(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, latent_len, _ = hidden.shape
        bos = self.bos_latent.expand(batch, 1, self.d_model).to(dtype=hidden.dtype)
        context_all = torch.cat([bos, hidden], dim=1)
        patch_idx = torch.arange(latent_len, device=hidden.device).view(latent_len, 1)
        source_idx = torch.arange(latent_len, device=hidden.device).view(1, latent_len)
        valid_hidden = source_idx < patch_idx
        valid = torch.cat(
            [
                torch.ones((latent_len, 1), dtype=torch.bool, device=hidden.device),
                valid_hidden,
            ],
            dim=1,
        )
        context = context_all.unsqueeze(1).expand(batch, latent_len, latent_len + 1, self.d_model)
        key_padding_mask = (~valid).unsqueeze(0).expand(batch, latent_len, latent_len + 1)
        return (
            context.reshape(batch * latent_len, latent_len + 1, self.d_model),
            key_padding_mask.reshape(batch * latent_len, latent_len + 1),
        )

    def _hnet_boundary_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
        safe_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
        valid_mask = attention_mask.bool()
        byte_embeddings = self.byte_embed(safe_ids)
        byte_embeddings = byte_embeddings * valid_mask.to(byte_embeddings.dtype).unsqueeze(-1)
        boundary_logits = self.semantic_boundary_scorer(byte_embeddings).squeeze(-1)
        boundary_probs = torch.sigmoid(boundary_logits) * valid_mask.to(boundary_logits.dtype)

        batch, seq_len = input_ids.shape
        hard_boundary = torch.zeros((batch, seq_len), dtype=torch.bool, device=input_ids.device)
        selected_positions: list[list[int]] = []
        use_hnetpp_flow = self.patch_boundary_mode == "hnetpp_flow_dechunk"
        hnetpp_flow_scores = self._hbf_boundary_scores(input_ids) if bool(use_hnetpp_flow) else None
        hnetpp_flow_score_sum = 0.0
        hnetpp_flow_score_count = 0
        hnetpp_flow_selected = 0
        for row_idx in range(batch):
            valid_len = int(valid_mask[row_idx].sum().detach().cpu().item())
            row_positions: list[int] = []
            last_boundary = -10**9
            for pos in range(valid_len):
                token_id = int(safe_ids[row_idx, pos].detach().cpu().item())
                byte_value = self._token_byte_value(token_id)
                semantic_score = float(boundary_probs[row_idx, pos].detach().cpu().item())
                flow_score = 0.0
                if (
                    bool(use_hnetpp_flow)
                    and hnetpp_flow_scores is not None
                    and pos > 0
                    and pos - 1 < hnetpp_flow_scores.shape[1]
                ):
                    flow_score = float(hnetpp_flow_scores[row_idx, pos - 1].item())
                    hnetpp_flow_score_sum += float(flow_score)
                    hnetpp_flow_score_count += 1
                combined_score = (0.65 * semantic_score) + (0.35 * flow_score)
                is_boundary = False
                if pos == 0:
                    is_boundary = True
                elif self._is_utf8_continuation(byte_value):
                    is_boundary = False
                elif pos - last_boundary >= self.patch_size:
                    is_boundary = True
                elif (
                    pos - last_boundary >= self.dynamic_min_patch_size
                    and (
                        semantic_score >= float(self.hbf_boundary_threshold)
                        or (
                            bool(use_hnetpp_flow)
                            and (
                                flow_score >= float(self.hbf_boundary_threshold)
                                or combined_score >= float(self.hbf_boundary_threshold)
                            )
                        )
                    )
                ):
                    is_boundary = True
                if bool(is_boundary):
                    hard_boundary[row_idx, pos] = True
                    row_positions.append(pos)
                    last_boundary = pos
                    if (
                        bool(use_hnetpp_flow)
                        and pos > 0
                        and flow_score >= float(self.hbf_boundary_threshold)
                        and semantic_score < float(self.hbf_boundary_threshold)
                    ):
                        hnetpp_flow_selected += 1
            if valid_len > 0 and not row_positions:
                hard_boundary[row_idx, 0] = True
                row_positions.append(0)
            selected_positions.append(row_positions)

        max_selected = max((len(row) for row in selected_positions), default=1)
        max_selected = max(1, int(max_selected))
        selected_embeddings = torch.zeros(
            (batch, max_selected, self.d_model),
            dtype=byte_embeddings.dtype,
            device=input_ids.device,
        )
        selected_valid = torch.zeros(
            (batch, max_selected),
            dtype=torch.bool,
            device=input_ids.device,
        )
        selected_probs = torch.zeros(
            (batch, max_selected),
            dtype=boundary_probs.dtype,
            device=input_ids.device,
        )
        dechunk_indices = torch.zeros((batch, seq_len), dtype=torch.long, device=input_ids.device)
        chunk_summary_count = 0
        chunk_summary_token_count = 0
        chunk_summary_nonboundary_token_count = 0
        chunk_summary_norm_total = 0.0
        for row_idx, row_positions in enumerate(selected_positions):
            if not row_positions:
                continue
            valid_len = int(valid_mask[row_idx].sum().detach().cpu().item())
            for selected_idx, boundary_pos in enumerate(row_positions):
                chunk_start = int(row_positions[selected_idx - 1]) + 1 if selected_idx > 0 else 0
                chunk_end = min(int(valid_len), int(boundary_pos) + 1)
                if chunk_end <= chunk_start:
                    chunk_start = int(boundary_pos)
                    chunk_end = min(int(valid_len), int(boundary_pos) + 1)
                chunk_states = byte_embeddings[row_idx, chunk_start:chunk_end]
                chunk_len = int(chunk_states.shape[0])
                if chunk_len <= 0:
                    continue
                recency = torch.linspace(
                    1.0,
                    2.0,
                    steps=chunk_len,
                    dtype=chunk_states.dtype,
                    device=input_ids.device,
                )
                causal_mean = (chunk_states * recency.unsqueeze(-1)).sum(dim=0) / recency.sum().clamp_min(1.0e-6)
                boundary_embedding = byte_embeddings[row_idx, int(boundary_pos)]
                summary = self.semantic_chunk_proj(torch.cat([boundary_embedding, causal_mean], dim=-1))
                len_id = torch.tensor(
                    min(chunk_len, int(self.patch_size)),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                summary = summary + self.patch_len_embed(len_id)
                selected_embeddings[row_idx, selected_idx] = summary
                chunk_summary_count += 1
                chunk_summary_token_count += int(chunk_len)
                chunk_summary_nonboundary_token_count += max(0, int(chunk_len) - 1)
                chunk_summary_norm_total += float(summary.detach().float().norm().cpu().item())
            selected_valid[row_idx, : len(row_positions)] = True
            index_tensor = torch.tensor(row_positions, dtype=torch.long, device=input_ids.device)
            selected_probs[row_idx, : len(row_positions)] = boundary_probs[row_idx, index_tensor]
            cursor = 0
            for pos in range(valid_len):
                while cursor + 1 < len(row_positions) and row_positions[cursor + 1] <= pos:
                    cursor += 1
                dechunk_indices[row_idx, pos] = cursor

        trajectory_count = int(self.imta_trajectories)
        imta_adapter_delta_norm = selected_embeddings.new_zeros(())
        imta_adapter_gate_mean = selected_embeddings.new_zeros(())
        imta_diversity_loss = selected_embeddings.new_zeros(())
        imta_diversity_mean_cosine = selected_embeddings.new_zeros(())
        if trajectory_count > 1:
            offsets = self.imta_trajectory_offsets[:trajectory_count].to(
                device=input_ids.device,
                dtype=selected_embeddings.dtype,
            )
            trajectory_embeddings = selected_embeddings.unsqueeze(1).expand(
                batch,
                trajectory_count,
                max_selected,
                self.d_model,
            )
            trajectory_embeddings = trajectory_embeddings + offsets.view(1, trajectory_count, 1, self.d_model)
            adapter_deltas = torch.stack(
                [
                    self.imta_trajectory_adapters[idx](selected_embeddings)
                    for idx in range(trajectory_count)
                ],
                dim=1,
            )
            adapter_gates = torch.sigmoid(self.imta_trajectory_adapter_gate_logit[:trajectory_count]).to(
                device=input_ids.device,
                dtype=selected_embeddings.dtype,
            )
            anchor_mask = torch.ones(
                (1, trajectory_count, 1, 1),
                dtype=selected_embeddings.dtype,
                device=input_ids.device,
            )
            anchor_mask[:, 0] = 0.0
            adapter_delta = adapter_gates.view(1, trajectory_count, 1, 1) * adapter_deltas * anchor_mask
            trajectory_embeddings = trajectory_embeddings + adapter_delta
            imta_adapter_delta_norm = adapter_delta.detach().float().norm(dim=-1).mean().to(selected_embeddings.dtype)
            imta_adapter_gate_mean = adapter_gates[1:].detach().float().mean().to(selected_embeddings.dtype)
            if self.training and float(self.imta_noise_std) > 0.0:
                noise = torch.randn_like(trajectory_embeddings) * float(self.imta_noise_std)
                noise[:, 0] = 0.0
                trajectory_embeddings = trajectory_embeddings + noise
            selected_hidden_flat = self._global_hidden(
                trajectory_embeddings.reshape(batch * trajectory_count, max_selected, self.d_model),
                think_steps=int(think_steps),
            )
            selected_hidden = selected_hidden_flat.reshape(batch, trajectory_count, max_selected, self.d_model)
        else:
            selected_hidden = self._global_hidden(selected_embeddings, think_steps=int(think_steps)).unsqueeze(1)

        own_latent_loss = selected_hidden.new_zeros(())
        own_latent_targets = 0
        own_latent_cosine = selected_hidden.new_zeros(())
        own_latent_pred_norm = selected_hidden.new_zeros(())
        own_latent_target_norm = selected_hidden.new_zeros(())
        if bool(self.own_latent_prediction_enabled) and max_selected > 1:
            pair_mask = selected_valid[:, :-1] & selected_valid[:, 1:]
            pair_mask = pair_mask.unsqueeze(1).expand(batch, trajectory_count, max_selected - 1)
            if bool(pair_mask.any()):
                source_latent = selected_hidden[:, :, :-1]
                target_latent = selected_hidden[:, :, 1:].detach()
                predicted_latent = source_latent + self.own_latent_predictor(source_latent)
                selected_pred = predicted_latent[pair_mask]
                selected_target = target_latent[pair_mask]
                pred_normed = F.normalize(selected_pred.float(), dim=-1)
                target_normed = F.normalize(selected_target.float(), dim=-1)
                cosine = (pred_normed * target_normed).sum(dim=-1).clamp(min=-1.0, max=1.0)
                cosine_loss = 1.0 - cosine
                smooth_l1 = F.smooth_l1_loss(
                    selected_pred.float(),
                    selected_target.float(),
                    reduction="none",
                ).mean(dim=-1)
                own_latent_loss = (cosine_loss + 0.1 * smooth_l1).mean().to(selected_hidden.dtype)
                own_latent_targets = int(selected_target.shape[0])
                own_latent_cosine = cosine.mean().to(selected_hidden.dtype)
                own_latent_pred_norm = selected_pred.detach().float().norm(dim=-1).mean().to(selected_hidden.dtype)
                own_latent_target_norm = selected_target.detach().float().norm(dim=-1).mean().to(selected_hidden.dtype)
        self._last_own_latent_prediction_loss = own_latent_loss
        ema_hidden = torch.zeros_like(selected_hidden)
        running = torch.zeros(
            (batch, trajectory_count, self.d_model),
            dtype=selected_hidden.dtype,
            device=input_ids.device,
        )
        selected_probs = selected_probs.clamp(min=1e-4, max=1.0 - 1e-4).to(selected_hidden.dtype)
        for selected_idx in range(max_selected):
            p = selected_probs[:, selected_idx].view(batch, 1, 1)
            running = p * selected_hidden[:, :, selected_idx] + (1.0 - p) * running
            ema_hidden[:, :, selected_idx] = running
        dechunked_candidates = torch.gather(
            ema_hidden,
            dim=2,
            index=dechunk_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(batch, trajectory_count, seq_len, self.d_model),
        )
        bridged_latent_candidates = dechunked_candidates + self.hnet_latent_bridge(dechunked_candidates)
        if self.decoder_latent_mode == "one_body":
            byte_gate = torch.sigmoid(self.hnet_byte_residual_gate_logit).to(bridged_latent_candidates.dtype)
            latent_gate = torch.sigmoid(self.hnet_latent_residual_gate_logit).to(bridged_latent_candidates.dtype)
            token_hidden_candidates = (
                byte_gate * byte_embeddings.unsqueeze(1)
                + latent_gate * bridged_latent_candidates
            )
            metric_byte_gate = byte_gate
            metric_latent_gate = latent_gate
        else:
            metric_byte_gate = bridged_latent_candidates.new_tensor(1.0)
            metric_latent_gate = boundary_probs.unsqueeze(1).unsqueeze(-1).to(bridged_latent_candidates.dtype)
            token_hidden_candidates = byte_embeddings.unsqueeze(1) + metric_latent_gate * bridged_latent_candidates
        token_hidden_candidates = token_hidden_candidates * valid_mask.to(token_hidden_candidates.dtype).view(
            batch,
            1,
            seq_len,
            1,
        )
        if trajectory_count > 1:
            valid_float = valid_mask.to(token_hidden_candidates.dtype).view(batch, 1, seq_len, 1)
            pooled = token_hidden_candidates.mul(valid_float).sum(dim=2) / valid_float.sum(dim=2).clamp_min(1.0)
            speaker_probe = self.hnet_causal_speaker.norm(pooled)
            selector_features = torch.cat([pooled, speaker_probe], dim=-1)
            selector_logits = self.imta_trajectory_selector(selector_features).squeeze(-1)
            selector_weights = torch.softmax(
                selector_logits / float(self.imta_selector_temperature),
                dim=-1,
            )
            selector_view = selector_weights.view(batch, trajectory_count, 1, 1)
            token_hidden = (selector_view * token_hidden_candidates).sum(dim=1)
            bridged_latent = (selector_view * bridged_latent_candidates).sum(dim=1)
            dechunked = (selector_view * dechunked_candidates).sum(dim=1)
            metric_latent_for_norm = (
                metric_latent_gate
                if metric_latent_gate.ndim <= 0
                else (selector_view * metric_latent_gate).sum(dim=1)
            )
            selector_entropy = -(selector_weights * selector_weights.clamp_min(1.0e-8).log()).sum(dim=-1)
            selector_confidence = selector_weights.max(dim=-1).values
            top2 = torch.topk(selector_weights, k=min(2, trajectory_count), dim=-1).values
            selector_margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
            pooled_std = pooled.detach().float().std(dim=1, unbiased=False).norm(dim=-1)
            normed_pooled = F.normalize(pooled.float(), dim=-1)
            trajectory_cosine = torch.matmul(normed_pooled, normed_pooled.transpose(1, 2))
            off_diag_mask = ~torch.eye(
                trajectory_count,
                dtype=torch.bool,
                device=trajectory_cosine.device,
            ).view(1, trajectory_count, trajectory_count)
            off_diag = trajectory_cosine.masked_select(off_diag_mask.expand(batch, -1, -1))
            if int(off_diag.numel()) > 0:
                imta_diversity_mean_cosine = off_diag.mean().to(token_hidden.dtype)
                imta_diversity_loss = F.relu(off_diag).mean().to(token_hidden.dtype)
        else:
            token_hidden = token_hidden_candidates[:, 0]
            bridged_latent = bridged_latent_candidates[:, 0]
            dechunked = dechunked_candidates[:, 0]
            metric_latent_for_norm = metric_latent_gate
            selector_entropy = token_hidden.new_zeros((batch,), dtype=token_hidden.dtype)
            selector_confidence = token_hidden.new_ones((batch,), dtype=token_hidden.dtype)
            selector_margin = token_hidden.new_ones((batch,), dtype=token_hidden.dtype)
            pooled_std = token_hidden.new_zeros((batch,), dtype=token_hidden.dtype)
        self._last_imta_diversity_loss = imta_diversity_loss
        token_hidden = token_hidden * valid_mask.to(token_hidden.dtype).unsqueeze(-1)

        valid_probs = boundary_probs[valid_mask]
        selected_count = int(hard_boundary.sum().detach().cpu().item())
        valid_count = int(valid_mask.sum().detach().cpu().item())
        if bool(valid_mask.any()):
            boundary_rate = boundary_probs[valid_mask].mean()
        else:
            boundary_rate = boundary_probs.sum() * 0.0
        metrics: dict[str, float | int] = {
            "latent_len": int(max_selected),
            "byte_len": int(valid_count),
            "compression_ratio": float(valid_count) / float(max(1, selected_count)),
            "hnet_selected_len": int(max_selected),
            "hnet_mean_selected_len": float(np.mean([len(row) for row in selected_positions])) if selected_positions else 0.0,
            "hnet_dechunked_tokens": int(valid_count),
            "hnet_causal_chunk_summary_count": int(chunk_summary_count),
            "hnet_causal_chunk_summary_mean_len": float(chunk_summary_token_count)
            / float(max(1, chunk_summary_count)),
            "hnet_causal_chunk_summary_nonboundary_tokens": int(chunk_summary_nonboundary_token_count),
            "hnet_causal_chunk_summary_mean_norm": float(chunk_summary_norm_total)
            / float(max(1, chunk_summary_count)),
            "learned_boundary_valid_boundaries": int(selected_count),
            "boundary_prob_rate": float(boundary_rate.detach().float().cpu().item()),
            "hnet_one_body_answer_path": int(self.decoder_latent_mode == "one_body"),
            "hnet_byte_residual_gate": float(metric_byte_gate.detach().float().mean().cpu().item()),
            "hnet_latent_residual_gate": float(metric_latent_gate.detach().float().mean().cpu().item()),
            "imta_trajectory_count": int(trajectory_count),
            "imta_active": int(trajectory_count > 1),
            "imta_noise_std": float(self.imta_noise_std),
            "imta_selector_temperature": float(self.imta_selector_temperature),
            "imta_adapter_gate_mean": float(imta_adapter_gate_mean.detach().float().cpu().item()),
            "imta_adapter_delta_norm": float(imta_adapter_delta_norm.detach().float().cpu().item()),
            "imta_selector_entropy": float(selector_entropy.detach().float().mean().cpu().item()),
            "imta_selector_confidence": float(selector_confidence.detach().float().mean().cpu().item()),
            "imta_selector_margin": float(selector_margin.detach().float().mean().cpu().item()),
            "imta_trajectory_state_std": float(pooled_std.detach().float().mean().cpu().item()),
            "imta_diversity_loss": float(imta_diversity_loss.detach().float().cpu().item()),
            "imta_diversity_mean_cosine": float(imta_diversity_mean_cosine.detach().float().cpu().item()),
            "own_latent_prediction_enabled": int(bool(self.own_latent_prediction_enabled)),
            "own_latent_prediction_loss": float(own_latent_loss.detach().float().cpu().item()),
            "own_latent_prediction_targets": int(own_latent_targets),
            "own_latent_prediction_cosine": float(own_latent_cosine.detach().float().cpu().item()),
            "own_latent_prediction_pred_norm": float(own_latent_pred_norm.detach().float().cpu().item()),
            "own_latent_prediction_target_norm": float(own_latent_target_norm.detach().float().cpu().item()),
        }
        if bool(valid_mask.any()):
            byte_component = (metric_byte_gate * byte_embeddings)[valid_mask]
            latent_gate_for_norm = metric_latent_for_norm
            if latent_gate_for_norm.ndim == 4 and latent_gate_for_norm.shape[1] == 1:
                latent_gate_for_norm = latent_gate_for_norm[:, 0]
            latent_component = (latent_gate_for_norm * bridged_latent)[valid_mask]
            byte_norm = byte_component.detach().float().norm(dim=-1).mean()
            latent_norm = latent_component.detach().float().norm(dim=-1).mean()
            metrics["hnet_latent_to_byte_norm_ratio"] = float(
                (latent_norm / byte_norm.clamp_min(1.0e-6)).cpu().item()
            )
        else:
            metrics["hnet_latent_to_byte_norm_ratio"] = 0.0
        if int(valid_probs.numel()) > 0:
            metrics.update(
                {
                    "learned_boundary_prob_mean": float(valid_probs.detach().float().mean().cpu().item()),
                    "learned_boundary_prob_std": float(valid_probs.detach().float().std(unbiased=False).cpu().item()),
                }
            )
        else:
            metrics.update(
                {
                    "learned_boundary_prob_mean": 0.0,
                    "learned_boundary_prob_std": 0.0,
                }
            )
        if bool(use_hnetpp_flow):
            metrics.update(
                {
                    "hnetpp_flow_boundary_score_mean": float(hnetpp_flow_score_sum) / float(max(1, hnetpp_flow_score_count)),
                    "hnetpp_flow_boundary_score_count": int(hnetpp_flow_score_count),
                    "hnetpp_flow_selected_boundaries": int(hnetpp_flow_selected),
                }
            )
        self.last_pack_metrics = dict(metrics)
        return token_hidden, safe_ids, valid_mask, boundary_probs, boundary_rate, metrics

    def _hnet_forward_losses(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
        diffusion_weight: float = 0.0,
        diffusion_mask_prob: float = 0.0,
        boundary_prior_weight: float = 0.0,
        boundary_target_ratio: float = 0.0,
        qwen_boundary_targets: torch.Tensor | None = None,
        qwen_boundary_prior_weight: float = 0.0,
        cot_anchor_loss_weight: float = 0.0,
        cot_anchor_max_targets: int = 512,
        workspace_selector_critic_weight: float = 0.0,
        workspace_selector_critic_temperature: float = 0.25,
        workspace_selector_final_ce_critic_weight: float = 0.0,
        workspace_selector_final_ce_critic_temperature: float = 0.25,
        workspace_selector_final_ce_critic_max_candidates: int = 16,
        workspace_selector_final_ce_critic_max_targets: int = 512,
        imta_diversity_weight: float = 0.0,
        own_latent_prediction_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        token_hidden, safe_ids, valid_mask, boundary_probs, boundary_rate, hnet_metrics = self._hnet_boundary_states(
            input_ids,
            attention_mask,
            think_steps=int(think_steps),
        )
        speaker_hidden = self.hnet_causal_speaker.forward_hidden(token_hidden)
        speaker_hidden = speaker_hidden * valid_mask.to(speaker_hidden.dtype).unsqueeze(-1)
        pre_readback_hidden = speaker_hidden
        speaker_hidden = self.apply_answer_readback(
            speaker_hidden,
            self.answer_embedding_weight(),
            attention_mask=attention_mask,
        )
        clean_logits = self.hnet_causal_speaker.head(speaker_hidden)
        clean_loss = F.cross_entropy(
            clean_logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=IGNORE_LABEL_ID,
        )
        cot_anchor_loss, cot_anchor_metrics = self.cot_anchor_regularization_loss(
            pre_readback_hidden,
            labels,
            max_targets=int(cot_anchor_max_targets),
        )
        workspace_selector_loss, workspace_selector_metrics = self.workspace_selector_critic_loss(
            pre_readback_hidden,
            labels,
            temperature=float(workspace_selector_critic_temperature),
        )
        final_ce_selector_loss = clean_loss.new_zeros(())
        final_ce_selector_metrics = self.zero_workspace_selector_final_ce_metrics()
        if float(workspace_selector_final_ce_critic_weight) > 0.0:
            final_ce_selector_loss, final_ce_selector_metrics = self.workspace_selector_final_ce_critic_loss(
                pre_readback_hidden,
                labels,
                temperature=float(workspace_selector_final_ce_critic_temperature),
                max_candidates=int(workspace_selector_final_ce_critic_max_candidates),
                max_targets=int(workspace_selector_final_ce_critic_max_targets),
            )
        boundary_prior_loss = clean_loss.new_zeros(())
        if float(boundary_prior_weight) > 0.0:
            target = clean_loss.new_tensor(float(boundary_target_ratio))
            boundary_prior_loss = (boundary_rate.float() - target).pow(2)
        qwen_boundary_prior_loss = clean_loss.new_zeros(())
        qwen_boundary_target_rate = 0.0
        qwen_boundary_accuracy = 0.0
        qwen_boundary_targets_seen = 0
        if float(qwen_boundary_prior_weight) > 0.0 and qwen_boundary_targets is not None:
            aligned_targets = qwen_boundary_targets[:, : input_ids.shape[1]].to(
                device=boundary_probs.device,
                dtype=boundary_probs.dtype,
            )
            valid = valid_mask[:, : aligned_targets.shape[1]].bool()
            if bool(valid.any()):
                probs = boundary_probs[:, : aligned_targets.shape[1]].clamp(min=1e-4, max=1.0 - 1e-4)
                targets = aligned_targets.clamp(min=0.0, max=1.0)
                valid_probs = probs[valid]
                valid_targets = targets[valid]
                positive = valid_targets >= 0.5
                negative = ~positive
                losses = []
                if bool(positive.any()):
                    losses.append(-valid_probs[positive].log().mean())
                if bool(negative.any()):
                    losses.append(-(1.0 - valid_probs[negative]).log().mean())
                if losses:
                    qwen_boundary_prior_loss = torch.stack(losses).mean()
                qwen_boundary_target_rate = float(valid_targets.float().mean().detach().cpu().item())
                qwen_boundary_accuracy = float(
                    valid_probs.ge(0.5).eq(valid_targets.ge(0.5)).float().mean().detach().cpu().item()
                )
                qwen_boundary_targets_seen = int(valid_targets.numel())
        own_latent_prediction_loss = clean_loss.new_zeros(())
        if self._last_own_latent_prediction_loss is not None:
            own_latent_prediction_loss = self._last_own_latent_prediction_loss.to(
                device=clean_loss.device,
                dtype=clean_loss.dtype,
            )
        imta_diversity_loss = clean_loss.new_zeros(())
        if self._last_imta_diversity_loss is not None:
            imta_diversity_loss = self._last_imta_diversity_loss.to(
                device=clean_loss.device,
                dtype=clean_loss.dtype,
            )
        diffusion_loss = clean_loss.new_zeros(())
        diffusion_targets = 0
        if float(diffusion_weight) > 0.0:
            sampled = torch.rand_like(safe_ids.float()) < float(diffusion_mask_prob)
            sampled = sampled & valid_mask.bool()
            if not bool(sampled.any()):
                sampled = valid_mask.bool()
            direct_byte_hidden = self.byte_embed(safe_ids.clamp(min=0, max=self.vocab_size - 1))
            direct_byte_hidden = direct_byte_hidden * valid_mask.to(direct_byte_hidden.dtype).unsqueeze(-1)
            if self.decoder_latent_mode == "one_body":
                direct_byte_hidden = direct_byte_hidden * torch.sigmoid(self.hnet_byte_residual_gate_logit).to(
                    direct_byte_hidden.dtype
                )
            latent_hidden = token_hidden - direct_byte_hidden
            mask_hidden = self.byte_embed(
                torch.full_like(safe_ids, int(self.mask_token_id)).clamp(min=0, max=self.vocab_size - 1)
            )
            if self.decoder_latent_mode == "one_body":
                mask_hidden = mask_hidden * torch.sigmoid(self.hnet_byte_residual_gate_logit).to(mask_hidden.dtype)
            masked_hidden = torch.where(
                sampled.unsqueeze(-1),
                mask_hidden + latent_hidden,
                token_hidden,
            )
            diffusion_hidden = self.hnet_causal_speaker.forward_hidden(masked_hidden)
            diffusion_logits = self.hnet_causal_speaker.head(diffusion_hidden)
            diffusion_targets = int(sampled.sum().detach().cpu().item())
            diffusion_loss = F.cross_entropy(
                diffusion_logits.reshape(-1, self.vocab_size)[sampled.reshape(-1)],
                safe_ids.reshape(-1)[sampled.reshape(-1)],
            )
        loss = (
            clean_loss
            + float(diffusion_weight) * diffusion_loss
            + float(boundary_prior_weight) * boundary_prior_loss
            + float(qwen_boundary_prior_weight) * qwen_boundary_prior_loss
            + float(cot_anchor_loss_weight) * cot_anchor_loss
            + float(workspace_selector_critic_weight) * workspace_selector_loss
            + float(workspace_selector_final_ce_critic_weight) * final_ce_selector_loss
            + float(imta_diversity_weight) * imta_diversity_loss
            + float(own_latent_prediction_weight) * own_latent_prediction_loss
        )
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "clean_loss": float(clean_loss.detach().cpu().item()),
            "diffusion_loss": float(diffusion_loss.detach().cpu().item()),
            "diffusion_targets": int(diffusion_targets),
            "nitp_loss": 0.0,
            "nitp_targets": 0,
            "nitp_cosine_similarity": 0.0,
            "nitp_predicted_norm": 0.0,
            "nitp_target_norm": 0.0,
            "boundary_prior_loss": float(boundary_prior_loss.detach().cpu().item()),
            "qwen_boundary_prior_loss": float(qwen_boundary_prior_loss.detach().cpu().item()),
            "qwen_boundary_target_rate": float(qwen_boundary_target_rate),
            "qwen_boundary_accuracy": float(qwen_boundary_accuracy),
            "qwen_boundary_targets": int(qwen_boundary_targets_seen),
            "imta_diversity_weight": float(imta_diversity_weight),
            "own_latent_prediction_weight": float(own_latent_prediction_weight),
        }
        metrics.update(hnet_metrics)
        metrics.update(cot_anchor_metrics)
        metrics.update(workspace_selector_metrics)
        metrics.update(final_ce_selector_metrics)
        metrics.update(
            {
                "answer_readback_gate_mean": float(self.last_readback_metrics.get("answer_readback_gate_mean", 0.0)),
                "answer_readback_expected_norm": float(
                    self.last_readback_metrics.get("answer_readback_expected_norm", 0.0)
                ),
                "cot_anchor_readback_entropy": float(self.last_readback_metrics.get("cot_anchor_entropy", 0.0)),
                "cot_anchor_readback_confidence": float(
                    self.last_readback_metrics.get("cot_anchor_confidence", 0.0)
                ),
                "answer_workspace_selection_entropy": float(
                    self.last_readback_metrics.get("answer_workspace_selection_entropy", 0.0)
                ),
                "answer_workspace_selection_confidence": float(
                    self.last_readback_metrics.get("answer_workspace_selection_confidence", 0.0)
                ),
            }
        )
        return loss, metrics

    def forward_losses(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
        diffusion_weight: float,
        diffusion_mask_prob: float,
        nitp_loss_weight: float = 0.0,
        nitp_max_targets: int = 0,
        boundary_prior_weight: float = 0.0,
        boundary_target_ratio: float = 0.0,
        qwen_boundary_targets: torch.Tensor | None = None,
        qwen_boundary_prior_weight: float = 0.0,
        cot_anchor_loss_weight: float = 0.0,
        cot_anchor_max_targets: int = 512,
        workspace_selector_critic_weight: float = 0.0,
        workspace_selector_critic_temperature: float = 0.25,
        workspace_selector_final_ce_critic_weight: float = 0.0,
        workspace_selector_final_ce_critic_temperature: float = 0.25,
        workspace_selector_final_ce_critic_max_candidates: int = 16,
        workspace_selector_final_ce_critic_max_targets: int = 512,
        imta_diversity_weight: float = 0.0,
        own_latent_prediction_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        if self.patch_boundary_mode in {"hnet_dechunk", "hnetpp_flow_dechunk"}:
            return self._hnet_forward_losses(
                input_ids,
                labels,
                attention_mask,
                think_steps=int(think_steps),
                diffusion_weight=float(diffusion_weight),
                diffusion_mask_prob=float(diffusion_mask_prob),
                boundary_prior_weight=float(boundary_prior_weight),
                boundary_target_ratio=float(boundary_target_ratio),
                qwen_boundary_targets=qwen_boundary_targets,
                qwen_boundary_prior_weight=float(qwen_boundary_prior_weight),
                cot_anchor_loss_weight=float(cot_anchor_loss_weight),
                cot_anchor_max_targets=int(cot_anchor_max_targets),
                workspace_selector_critic_weight=float(workspace_selector_critic_weight),
                workspace_selector_critic_temperature=float(workspace_selector_critic_temperature),
                workspace_selector_final_ce_critic_weight=float(workspace_selector_final_ce_critic_weight),
                workspace_selector_final_ce_critic_temperature=float(workspace_selector_final_ce_critic_temperature),
                workspace_selector_final_ce_critic_max_candidates=int(
                workspace_selector_final_ce_critic_max_candidates
                ),
                workspace_selector_final_ce_critic_max_targets=int(workspace_selector_final_ce_critic_max_targets),
                imta_diversity_weight=float(imta_diversity_weight),
                own_latent_prediction_weight=float(own_latent_prediction_weight),
            )
        grouped_ids, grouped_mask, grouped_labels, latent_len, valid_bytes, valid_patches = self.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )
        safe_ids, grouped_byte_embeddings, _, patch_embeddings = self._grouped_patch_embeddings(
            grouped_ids,
            grouped_mask,
        )
        hidden = self._global_hidden(patch_embeddings, think_steps=int(think_steps))
        use_latent_add = self.decoder_latent_mode in {
            "add",
            "add_cross",
            "hier_add",
            "hier_add_cross",
            "one_body",
        }
        use_byte_decoder_input = self.decoder_latent_mode != "one_body"
        use_hierarchical = self.decoder_latent_mode in {"hier_add", "hier_add_cross"}
        hierarchical_metrics: dict[str, float] = {}
        if bool(use_latent_add):
            clean_cond = self.clean_patch_condition(hidden)
            diffusion_cond = self.shifted_patch_condition(hidden).unsqueeze(2)
        else:
            clean_cond = torch.zeros(
                (*hidden.shape[:2], self.patch_size, self.d_model),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            diffusion_cond = torch.zeros(
                (*hidden.shape[:2], 1, self.d_model),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        if bool(use_hierarchical):
            hierarchical_cond, hierarchical_metrics = self.hierarchical_patch_condition(hidden)
            clean_cond = clean_cond + hierarchical_cond
        decoder_context = None
        decoder_context_mask = None
        if self.decoder_latent_mode in {"cross", "add_cross", "hier_add_cross"}:
            decoder_context, decoder_context_mask = self.prefix_latent_context(hidden)
        pos_ids = torch.arange(self.patch_size, device=input_ids.device)
        pos = self.byte_pos_embed(pos_ids).view(1, 1, self.patch_size, self.d_model)

        if bool(use_byte_decoder_input):
            clean_byte_input = grouped_byte_embeddings
        else:
            clean_byte_input = torch.zeros_like(grouped_byte_embeddings)
        clean_input = clean_byte_input + clean_cond + pos
        clean_hidden = self.clean_decoder.forward_hidden(
            clean_input.reshape(-1, self.patch_size, self.d_model),
            context=decoder_context,
            context_key_padding_mask=decoder_context_mask,
        ).reshape(input_ids.shape[0], latent_len, self.patch_size, self.d_model)
        pre_readback_hidden = clean_hidden
        clean_hidden = self.apply_answer_readback(
            clean_hidden,
            self.answer_embedding_weight(),
            attention_mask=grouped_mask,
        )
        clean_logits = self.clean_decoder.head(clean_hidden).reshape(
            input_ids.shape[0],
            latent_len,
            self.patch_size,
            self.vocab_size,
        )
        clean_labels = grouped_labels
        clean_loss = F.cross_entropy(
            clean_logits.reshape(-1, self.vocab_size),
            clean_labels.reshape(-1),
            ignore_index=IGNORE_LABEL_ID,
        )
        cot_anchor_loss, cot_anchor_metrics = self.cot_anchor_regularization_loss(
            pre_readback_hidden,
            clean_labels,
            max_targets=int(cot_anchor_max_targets),
        )
        workspace_selector_loss, workspace_selector_metrics = self.workspace_selector_critic_loss(
            pre_readback_hidden,
            clean_labels,
            temperature=float(workspace_selector_critic_temperature),
        )
        final_ce_selector_loss = clean_loss.new_zeros(())
        final_ce_selector_metrics = self.zero_workspace_selector_final_ce_metrics()
        if float(workspace_selector_final_ce_critic_weight) > 0.0:
            final_ce_selector_loss, final_ce_selector_metrics = self.workspace_selector_final_ce_critic_loss(
                pre_readback_hidden,
                clean_labels,
                temperature=float(workspace_selector_final_ce_critic_temperature),
                max_candidates=int(workspace_selector_final_ce_critic_max_candidates),
                max_targets=int(workspace_selector_final_ce_critic_max_targets),
            )
        nitp_loss = clean_loss.new_zeros(())
        nitp_targets = 0
        nitp_cosine = 0.0
        nitp_predicted_norm = 0.0
        nitp_target_norm = 0.0
        if self.nitp_projector is not None and float(nitp_loss_weight) > 0.0:
            target_mask = clean_labels.reshape(-1) != IGNORE_LABEL_ID
            if bool(target_mask.any()):
                target_hidden = clean_hidden.reshape(-1, self.d_model)[target_mask]
                target_ids = clean_labels.reshape(-1)[target_mask]
                if int(nitp_max_targets) > 0 and int(target_ids.numel()) > int(nitp_max_targets):
                    indices = torch.linspace(
                        0,
                        int(target_ids.numel()) - 1,
                        steps=int(nitp_max_targets),
                        device=target_ids.device,
                    ).long()
                    target_hidden = target_hidden[indices]
                    target_ids = target_ids[indices]
                predicted = self.nitp_projector(target_hidden)
                with torch.no_grad():
                    target_embedding = self.byte_embed(target_ids).detach()
                predicted_norm = F.normalize(predicted.float(), dim=-1)
                target_norm = F.normalize(target_embedding.float(), dim=-1)
                cosine = (predicted_norm * target_norm).sum(dim=-1)
                nitp_loss = (1.0 - cosine).mean()
                nitp_targets = int(target_ids.numel())
                nitp_cosine = float(cosine.mean().detach().cpu().item())
                nitp_predicted_norm = float(predicted.float().norm(dim=-1).mean().detach().cpu().item())
                nitp_target_norm = float(target_embedding.float().norm(dim=-1).mean().detach().cpu().item())

        diffusion_loss = clean_loss.new_zeros(())
        diffusion_targets = 0
        if float(diffusion_weight) > 0.0:
            valid_mask = grouped_mask.bool()
            sampled = torch.rand_like(grouped_ids.float()) < float(diffusion_mask_prob)
            sampled = sampled & valid_mask
            if not bool(sampled.any()):
                sampled = valid_mask
            corrupted_ids = torch.where(
                sampled,
                torch.full_like(grouped_ids, int(self.mask_token_id)),
                safe_ids,
            ).clamp(min=0, max=self.vocab_size - 1)
            corrupted_embeddings = self.byte_embed(corrupted_ids)
            valid = grouped_mask.to(corrupted_embeddings.dtype).unsqueeze(-1)
            corrupted_grouped = corrupted_embeddings * valid
            if not bool(use_byte_decoder_input):
                corrupted_grouped = torch.zeros_like(corrupted_grouped)
            diffusion_input = corrupted_grouped + diffusion_cond + pos
            diffusion_logits = self.diffusion_decoder(
                diffusion_input.reshape(-1, self.patch_size, self.d_model),
                context=decoder_context,
                context_key_padding_mask=decoder_context_mask,
            ).reshape(input_ids.shape[0], latent_len, self.patch_size, self.vocab_size)
            target_ids = safe_ids.reshape(-1)
            mask_flat = sampled.reshape(-1)
            diffusion_targets = int(mask_flat.sum().detach().cpu().item())
            diffusion_loss = F.cross_entropy(
                diffusion_logits.reshape(-1, self.vocab_size)[mask_flat],
                target_ids[mask_flat],
            )

        loss = (
            clean_loss
            + float(diffusion_weight) * diffusion_loss
            + float(nitp_loss_weight) * nitp_loss
            + float(cot_anchor_loss_weight) * cot_anchor_loss
            + float(workspace_selector_critic_weight) * workspace_selector_loss
            + float(workspace_selector_final_ce_critic_weight) * final_ce_selector_loss
        )
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "clean_loss": float(clean_loss.detach().cpu().item()),
            "diffusion_loss": float(diffusion_loss.detach().cpu().item()),
            "diffusion_targets": int(diffusion_targets),
            "nitp_loss": float(nitp_loss.detach().cpu().item()),
            "nitp_targets": int(nitp_targets),
            "nitp_cosine_similarity": float(nitp_cosine),
            "nitp_predicted_norm": float(nitp_predicted_norm),
            "nitp_target_norm": float(nitp_target_norm),
            "boundary_prior_loss": 0.0,
            "qwen_boundary_prior_loss": 0.0,
            "qwen_boundary_target_rate": 0.0,
            "qwen_boundary_accuracy": 0.0,
            "qwen_boundary_targets": 0,
            "imta_diversity_weight": float(imta_diversity_weight),
            "imta_diversity_loss": 0.0,
            "imta_diversity_mean_cosine": 0.0,
            "imta_adapter_gate_mean": 0.0,
            "imta_adapter_delta_norm": 0.0,
            "own_latent_prediction_enabled": int(bool(self.own_latent_prediction_enabled)),
            "own_latent_prediction_weight": float(own_latent_prediction_weight),
            "own_latent_prediction_loss": 0.0,
            "own_latent_prediction_targets": 0,
            "own_latent_prediction_cosine": 0.0,
            "own_latent_prediction_pred_norm": 0.0,
            "own_latent_prediction_target_norm": 0.0,
            "latent_len": int(latent_len),
            "byte_len": int(valid_bytes),
            "compression_ratio": float(valid_bytes) / float(max(1, valid_patches)),
        }
        metrics.update(self.last_pack_metrics)
        metrics.update(hierarchical_metrics)
        metrics.update(cot_anchor_metrics)
        metrics.update(workspace_selector_metrics)
        metrics.update(final_ce_selector_metrics)
        metrics.update(
            {
                "answer_readback_gate_mean": float(self.last_readback_metrics.get("answer_readback_gate_mean", 0.0)),
                "answer_readback_expected_norm": float(
                    self.last_readback_metrics.get("answer_readback_expected_norm", 0.0)
                ),
                "cot_anchor_readback_entropy": float(self.last_readback_metrics.get("cot_anchor_entropy", 0.0)),
                "cot_anchor_readback_confidence": float(
                    self.last_readback_metrics.get("cot_anchor_confidence", 0.0)
                ),
                "answer_workspace_selection_entropy": float(
                    self.last_readback_metrics.get("answer_workspace_selection_entropy", 0.0)
                ),
                "answer_workspace_selection_confidence": float(
                    self.last_readback_metrics.get("answer_workspace_selection_confidence", 0.0)
                ),
            }
        )
        return loss, metrics

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
        external_register: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward_logits_and_decoder_hidden(
            input_ids,
            attention_mask,
            think_steps=int(think_steps),
            external_register=external_register,
        )
        return logits

    def answer_embedding_weight(self) -> torch.Tensor:
        if self.patch_boundary_mode in {"hnet_dechunk", "hnetpp_flow_dechunk"}:
            return self.hnet_causal_speaker.head.weight
        return self.clean_decoder.head.weight

    def cot_anchor_regularization_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        max_targets: int = 512,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        flat_hidden = hidden.reshape(-1, self.d_model)
        flat_labels = labels.reshape(-1)
        mask = flat_labels != IGNORE_LABEL_ID
        if not bool(mask.any()):
            zero = flat_hidden.sum() * 0.0
            return zero, {
                "cot_anchor_loss": 0.0,
                "cot_anchor_targets": 0,
                "cot_anchor_entropy": 0.0,
                "cot_anchor_accuracy": 0.0,
            }
        selected_hidden = flat_hidden[mask]
        selected_labels = flat_labels[mask].clamp(min=0, max=self.vocab_size - 1)
        if int(max_targets) > 0 and int(selected_labels.numel()) > int(max_targets):
            indices = torch.linspace(
                0,
                int(selected_labels.numel()) - 1,
                steps=int(max_targets),
                device=selected_labels.device,
            ).long()
            selected_hidden = selected_hidden[indices]
            selected_labels = selected_labels[indices]
        logits = self.answer_anchor_head(selected_hidden)
        loss = F.cross_entropy(logits.float(), selected_labels)
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        accuracy = logits.argmax(dim=-1).eq(selected_labels).float().mean()
        return loss.to(hidden.dtype), {
            "cot_anchor_loss": float(loss.detach().cpu().item()),
            "cot_anchor_targets": int(selected_labels.numel()),
            "cot_anchor_entropy": float(entropy.detach().cpu().item()),
            "cot_anchor_accuracy": float(accuracy.detach().cpu().item()),
        }

    def workspace_selector_critic_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        batch = int(hidden.shape[0])
        flat_hidden = hidden.reshape(batch, -1, self.d_model)
        flat_labels = labels.reshape(batch, -1)
        length = min(int(flat_hidden.shape[1]), int(flat_labels.shape[1]))
        flat_hidden = flat_hidden[:, :length]
        flat_labels = flat_labels[:, :length]
        valid = flat_labels != IGNORE_LABEL_ID
        active = valid.any(dim=-1)
        if not bool(active.any()):
            zero = flat_hidden.sum() * 0.0
            return zero, {
                "answer_workspace_selector_loss": 0.0,
                "answer_workspace_selector_targets": 0,
                "answer_workspace_selector_target_entropy": 0.0,
                "answer_workspace_selector_target_confidence": 0.0,
                "answer_workspace_selector_selection_entropy": 0.0,
                "answer_workspace_selector_selection_confidence": 0.0,
                "answer_workspace_selector_target_argmax_match": 0.0,
                "answer_workspace_selector_target_best_index": 0,
            }

        selector_logits = self.answer_workspace_selector(flat_hidden).squeeze(-1).float()
        anchor_logits = self.answer_anchor_head(flat_hidden).float()
        safe_labels = flat_labels.clamp(min=0, max=self.vocab_size - 1)
        anchor_ce = F.cross_entropy(
            anchor_logits.reshape(-1, self.vocab_size),
            safe_labels.reshape(-1),
            reduction="none",
        ).reshape(batch, length)

        active_valid = valid[active]
        active_selector_logits = selector_logits[active].masked_fill(~active_valid, -1.0e9)
        score_temperature = float(max(1e-6, temperature))
        target_scores = (-anchor_ce.detach()[active] / score_temperature).masked_fill(~active_valid, -1.0e9)
        target_distribution = F.softmax(target_scores, dim=-1)
        selector_log_probs = F.log_softmax(active_selector_logits, dim=-1)
        loss = -(target_distribution * selector_log_probs).sum(dim=-1).mean()

        selector_distribution = F.softmax(active_selector_logits, dim=-1)
        target_entropy = -(
            target_distribution * target_distribution.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        selector_entropy = -(
            selector_distribution * selector_distribution.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        target_best = target_distribution.argmax(dim=-1)
        selector_best = selector_distribution.argmax(dim=-1)
        target_argmax_match = selector_best.eq(target_best).float().mean()

        return loss.to(hidden.dtype), {
            "answer_workspace_selector_loss": float(loss.detach().cpu().item()),
            "answer_workspace_selector_targets": int(active_valid.sum().detach().cpu().item()),
            "answer_workspace_selector_target_entropy": float(target_entropy.detach().cpu().item()),
            "answer_workspace_selector_target_confidence": float(
                target_distribution.max(dim=-1).values.mean().detach().cpu().item()
            ),
            "answer_workspace_selector_selection_entropy": float(selector_entropy.detach().cpu().item()),
            "answer_workspace_selector_selection_confidence": float(
                selector_distribution.max(dim=-1).values.mean().detach().cpu().item()
            ),
            "answer_workspace_selector_target_argmax_match": float(
                target_argmax_match.detach().cpu().item()
            ),
            "answer_workspace_selector_target_best_index": int(target_best[0].detach().cpu().item()),
        }

    def zero_workspace_selector_final_ce_metrics(self) -> dict[str, float | int]:
        return {
            "answer_workspace_final_ce_selector_loss": 0.0,
            "answer_workspace_final_ce_selector_targets": 0,
            "answer_workspace_final_ce_selector_candidate_count": 0,
            "answer_workspace_final_ce_selector_target_entropy": 0.0,
            "answer_workspace_final_ce_selector_target_confidence": 0.0,
            "answer_workspace_final_ce_selector_selection_entropy": 0.0,
            "answer_workspace_final_ce_selector_selection_confidence": 0.0,
            "answer_workspace_final_ce_selector_target_argmax_match": 0.0,
            "answer_workspace_final_ce_selector_target_best_index": 0,
            "answer_workspace_final_ce_selector_best_ce": 0.0,
            "answer_workspace_final_ce_selector_mean_ce": 0.0,
            "answer_workspace_final_ce_selector_worst_ce": 0.0,
            "answer_workspace_final_ce_selector_improvement_over_mean_ce": 0.0,
        }

    def speaker_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.patch_boundary_mode in {"hnet_dechunk", "hnetpp_flow_dechunk"}:
            return self.hnet_causal_speaker.head(hidden)
        return self.clean_decoder.head(hidden)

    def workspace_selector_final_ce_critic_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float = 0.25,
        max_candidates: int = 16,
        max_targets: int = 512,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        batch = int(hidden.shape[0])
        flat_hidden = hidden.reshape(batch, -1, self.d_model)
        flat_labels = labels.reshape(batch, -1)
        length = min(int(flat_hidden.shape[1]), int(flat_labels.shape[1]))
        flat_hidden = flat_hidden[:, :length]
        flat_labels = flat_labels[:, :length]
        valid = flat_labels != IGNORE_LABEL_ID
        if not bool(valid.any()):
            zero = flat_hidden.sum() * 0.0
            return zero, self.zero_workspace_selector_final_ce_metrics()

        score_temperature = float(max(1e-6, temperature))
        readback_temperature = float(max(1e-6, self.answer_readback_temperature))
        gate = torch.sigmoid(self.answer_readback_gate_logit).to(dtype=flat_hidden.dtype, device=flat_hidden.device)
        speaker = self.answer_embedding_weight().to(dtype=flat_hidden.dtype, device=flat_hidden.device)

        row_losses: list[torch.Tensor] = []
        target_entropies: list[torch.Tensor] = []
        target_confidences: list[torch.Tensor] = []
        selection_entropies: list[torch.Tensor] = []
        selection_confidences: list[torch.Tensor] = []
        target_argmax_matches: list[torch.Tensor] = []
        best_indices: list[int] = []
        best_ces: list[torch.Tensor] = []
        mean_ces: list[torch.Tensor] = []
        worst_ces: list[torch.Tensor] = []
        candidate_total = 0
        target_total = 0

        for batch_idx in range(batch):
            valid_indices = torch.nonzero(valid[batch_idx], as_tuple=False).squeeze(-1)
            if int(valid_indices.numel()) == 0:
                continue
            candidate_indices = valid_indices
            if int(max_candidates) > 0 and int(candidate_indices.numel()) > int(max_candidates):
                take = torch.linspace(
                    0,
                    int(candidate_indices.numel()) - 1,
                    steps=int(max_candidates),
                    device=candidate_indices.device,
                ).long()
                candidate_indices = candidate_indices[take]
            target_indices = valid_indices
            if int(max_targets) > 0 and int(target_indices.numel()) > int(max_targets):
                take = torch.linspace(
                    0,
                    int(target_indices.numel()) - 1,
                    steps=int(max_targets),
                    device=target_indices.device,
                ).long()
                target_indices = target_indices[take]

            candidate_hidden = flat_hidden[batch_idx, candidate_indices]
            target_hidden = flat_hidden[batch_idx, target_indices]
            target_labels = flat_labels[batch_idx, target_indices].clamp(min=0, max=self.vocab_size - 1)
            anchor_logits = self.answer_anchor_head(candidate_hidden).float()
            anchor_probs = F.softmax(anchor_logits / readback_temperature, dim=-1).to(flat_hidden.dtype)
            expected = anchor_probs @ speaker

            candidate_count = int(candidate_indices.numel())
            target_count = int(target_indices.numel())
            refined = target_hidden.unsqueeze(0) + gate * expected.unsqueeze(1)
            logits = self.speaker_logits_from_hidden(refined.reshape(candidate_count * target_count, self.d_model))
            candidate_ce = F.cross_entropy(
                logits.float(),
                target_labels.repeat(candidate_count),
                reduction="none",
            ).reshape(candidate_count, target_count).mean(dim=-1)

            target_distribution = F.softmax(-candidate_ce.detach() / score_temperature, dim=-1)
            selector_logits = self.answer_workspace_selector(candidate_hidden).squeeze(-1).float()
            selector_log_probs = F.log_softmax(selector_logits, dim=-1)
            selector_distribution = F.softmax(selector_logits, dim=-1)
            row_losses.append(-(target_distribution * selector_log_probs).sum())

            target_entropy = -(
                target_distribution * target_distribution.clamp_min(1e-9).log()
            ).sum()
            selector_entropy = -(
                selector_distribution * selector_distribution.clamp_min(1e-9).log()
            ).sum()
            target_best = target_distribution.argmax(dim=-1)
            selector_best = selector_distribution.argmax(dim=-1)
            target_entropies.append(target_entropy)
            target_confidences.append(target_distribution.max(dim=-1).values)
            selection_entropies.append(selector_entropy)
            selection_confidences.append(selector_distribution.max(dim=-1).values)
            target_argmax_matches.append(selector_best.eq(target_best).float())
            best_indices.append(int(candidate_indices[target_best].detach().cpu().item()))
            best_ces.append(candidate_ce.min())
            mean_ces.append(candidate_ce.mean())
            worst_ces.append(candidate_ce.max())
            candidate_total += candidate_count
            target_total += target_count

        if not row_losses:
            zero = flat_hidden.sum() * 0.0
            return zero, self.zero_workspace_selector_final_ce_metrics()

        loss = torch.stack(row_losses).mean()
        best_ce = torch.stack(best_ces).mean()
        mean_ce = torch.stack(mean_ces).mean()
        worst_ce = torch.stack(worst_ces).mean()
        return loss.to(hidden.dtype), {
            "answer_workspace_final_ce_selector_loss": float(loss.detach().cpu().item()),
            "answer_workspace_final_ce_selector_targets": int(target_total),
            "answer_workspace_final_ce_selector_candidate_count": int(candidate_total),
            "answer_workspace_final_ce_selector_target_entropy": float(
                torch.stack(target_entropies).mean().detach().cpu().item()
            ),
            "answer_workspace_final_ce_selector_target_confidence": float(
                torch.stack(target_confidences).mean().detach().cpu().item()
            ),
            "answer_workspace_final_ce_selector_selection_entropy": float(
                torch.stack(selection_entropies).mean().detach().cpu().item()
            ),
            "answer_workspace_final_ce_selector_selection_confidence": float(
                torch.stack(selection_confidences).mean().detach().cpu().item()
            ),
            "answer_workspace_final_ce_selector_target_argmax_match": float(
                torch.stack(target_argmax_matches).mean().detach().cpu().item()
            ),
            "answer_workspace_final_ce_selector_target_best_index": int(best_indices[0]),
            "answer_workspace_final_ce_selector_best_ce": float(best_ce.detach().cpu().item()),
            "answer_workspace_final_ce_selector_mean_ce": float(mean_ce.detach().cpu().item()),
            "answer_workspace_final_ce_selector_worst_ce": float(worst_ce.detach().cpu().item()),
            "answer_workspace_final_ce_selector_improvement_over_mean_ce": float(
                (mean_ce - best_ce).detach().cpu().item()
            ),
        }

    def apply_answer_readback(
        self,
        hidden: torch.Tensor,
        speaker_weight: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_readback_metrics = {
            "answer_readback_mode": str(self.answer_readback_mode),
            "answer_readback_gate_mean": 0.0,
            "answer_readback_expected_norm": 0.0,
            "cot_anchor_entropy": 0.0,
            "cot_anchor_confidence": 0.0,
            "answer_workspace_selection_entropy": 0.0,
            "answer_workspace_selection_confidence": 0.0,
        }
        if self.answer_readback_mode == "none":
            return hidden
        if self.answer_readback_mode not in {"self_embedding", "anchor_embedding", "selected_anchor_embedding"}:
            raise ValueError(f"unsupported answer_readback_mode: {self.answer_readback_mode}")
        temp = float(max(1e-6, self.answer_readback_temperature))
        if self.answer_readback_mode in {"anchor_embedding", "selected_anchor_embedding"}:
            preliminary_logits = self.answer_anchor_head(hidden)
        else:
            preliminary_logits = F.linear(hidden, speaker_weight.to(dtype=hidden.dtype, device=hidden.device))
        probs = F.softmax(preliminary_logits.float() / temp, dim=-1).to(hidden.dtype)
        speaker = speaker_weight.to(dtype=hidden.dtype, device=hidden.device)
        expected = probs @ speaker
        gate = torch.sigmoid(self.answer_readback_gate_logit).to(dtype=hidden.dtype, device=hidden.device)
        probs_float = probs.detach().float()
        entropy = -(probs_float * probs_float.clamp_min(1e-9).log()).sum(dim=-1).mean()
        confidence = probs_float.max(dim=-1).values.mean()
        selection_entropy = hidden.new_tensor(0.0)
        selection_confidence = hidden.new_tensor(0.0)
        if self.answer_readback_mode == "selected_anchor_embedding":
            batch = int(hidden.shape[0])
            flat_hidden = hidden.reshape(batch, -1, self.d_model)
            flat_expected = expected.reshape(batch, -1, self.d_model)
            selector_logits = self.answer_workspace_selector(flat_hidden).squeeze(-1).float()
            if attention_mask is not None:
                valid = attention_mask.reshape(batch, -1).to(device=selector_logits.device).bool()
                valid = valid | ~valid.any(dim=-1, keepdim=True)
                selector_logits = selector_logits.masked_fill(~valid, -1.0e9)
            selection = F.softmax(selector_logits, dim=-1).to(hidden.dtype)
            workspace = (selection.unsqueeze(-1) * flat_expected).sum(dim=1)
            broadcast_shape = (batch,) + (1,) * (hidden.ndim - 2) + (self.d_model,)
            expected = workspace.reshape(broadcast_shape).expand_as(hidden)
            selection_float = selection.detach().float()
            selection_entropy = -(
                selection_float * selection_float.clamp_min(1e-9).log()
            ).sum(dim=-1).mean()
            selection_confidence = selection_float.max(dim=-1).values.mean()
        refined = hidden + gate * expected
        self.last_readback_metrics = {
            "answer_readback_mode": str(self.answer_readback_mode),
            "answer_readback_gate_mean": float(gate.detach().float().cpu().item()),
            "answer_readback_expected_norm": float(expected.detach().float().norm(dim=-1).mean().cpu().item()),
            "cot_anchor_entropy": float(entropy.detach().cpu().item()),
            "cot_anchor_confidence": float(confidence.detach().cpu().item()),
            "answer_workspace_selection_entropy": float(selection_entropy.detach().cpu().item()),
            "answer_workspace_selection_confidence": float(selection_confidence.detach().cpu().item()),
        }
        return refined

    def forward_logits_and_decoder_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        think_steps: int,
        external_register: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.patch_boundary_mode in {"hnet_dechunk", "hnetpp_flow_dechunk"}:
            token_hidden, _, _, _, _, _ = self._hnet_boundary_states(
                input_ids,
                attention_mask,
                think_steps=int(think_steps),
            )
            if external_register is not None:
                register = external_register.to(device=token_hidden.device, dtype=token_hidden.dtype)
                if register.ndim == 2:
                    register = register.unsqueeze(1)
                if register.ndim != 3 or register.shape[0] != token_hidden.shape[0] or register.shape[-1] != self.d_model:
                    raise ValueError(
                        "external_register must have shape [batch, d_model] or "
                        "[batch, slots, d_model]"
                    )
                token_hidden = token_hidden + register.mean(dim=1, keepdim=True)
            speaker_hidden = self.hnet_causal_speaker.forward_hidden(token_hidden)
            speaker_hidden = self.apply_answer_readback(
                speaker_hidden,
                self.answer_embedding_weight(),
                attention_mask=attention_mask,
            )
            return self.hnet_causal_speaker.head(speaker_hidden), speaker_hidden
        labels = torch.full_like(input_ids, int(IGNORE_LABEL_ID))
        grouped_ids, grouped_mask, _, latent_len, _, _ = self.pack_patches(
            input_ids,
            attention_mask,
            labels,
        )
        _, grouped_byte_embeddings, _, patch_embeddings = self._grouped_patch_embeddings(
            grouped_ids,
            grouped_mask,
        )
        hidden = self._global_hidden(patch_embeddings, think_steps=int(think_steps))
        if external_register is not None:
            register = external_register.to(device=hidden.device, dtype=hidden.dtype)
            if register.ndim == 2:
                register = register.unsqueeze(1)
            if register.ndim != 3 or register.shape[0] != hidden.shape[0] or register.shape[-1] != self.d_model:
                raise ValueError(
                    "external_register must have shape [batch, d_model] or "
                    "[batch, slots, d_model]"
                )
            hidden = hidden + register.mean(dim=1, keepdim=True)
        use_latent_add = self.decoder_latent_mode in {
            "add",
            "add_cross",
            "hier_add",
            "hier_add_cross",
            "one_body",
        }
        use_byte_decoder_input = self.decoder_latent_mode != "one_body"
        use_hierarchical = self.decoder_latent_mode in {"hier_add", "hier_add_cross"}
        if bool(use_latent_add):
            clean_cond = self.clean_patch_condition(hidden)
        else:
            clean_cond = torch.zeros(
                (*hidden.shape[:2], self.patch_size, self.d_model),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        if bool(use_hierarchical):
            hierarchical_cond, _ = self.hierarchical_patch_condition(hidden)
            clean_cond = clean_cond + hierarchical_cond
        decoder_context = None
        decoder_context_mask = None
        if self.decoder_latent_mode in {"cross", "add_cross", "hier_add_cross"}:
            decoder_context, decoder_context_mask = self.prefix_latent_context(hidden)
        pos_ids = torch.arange(self.patch_size, device=input_ids.device)
        pos = self.byte_pos_embed(pos_ids).view(1, 1, self.patch_size, self.d_model)
        if bool(use_byte_decoder_input):
            clean_byte_input = grouped_byte_embeddings
        else:
            clean_byte_input = torch.zeros_like(grouped_byte_embeddings)
        clean_input = clean_byte_input + clean_cond + pos
        clean_hidden = self.clean_decoder.forward_hidden(
            clean_input.reshape(-1, self.patch_size, self.d_model),
            context=decoder_context,
            context_key_padding_mask=decoder_context_mask,
        ).reshape(input_ids.shape[0], latent_len, self.patch_size, self.d_model)
        clean_hidden = self.apply_answer_readback(
            clean_hidden,
            self.answer_embedding_weight(),
            attention_mask=grouped_mask,
        )
        logits = self.clean_decoder.head(clean_hidden).reshape(
            input_ids.shape[0],
            latent_len * self.patch_size,
            self.vocab_size,
        )
        token_hidden = clean_hidden.reshape(input_ids.shape[0], latent_len * self.patch_size, self.d_model)
        if self.patch_boundary_mode == "fixed":
            return logits[:, : input_ids.shape[1]], token_hidden[:, : input_ids.shape[1]]
        return logits, token_hidden
