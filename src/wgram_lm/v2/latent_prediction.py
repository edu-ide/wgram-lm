from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import WGRAMV2Config


@dataclass
class LatentPredictionOutput:
    loss: torch.Tensor
    targets: int
    cosine: torch.Tensor
    target_source: str


class OwnLatentPredictor(nn.Module):
    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        d_model = int(config.d_model)
        self.enabled = bool(config.own_latent_prediction_enabled)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        chunk_hidden: torch.Tensor,
        chunk_valid: torch.Tensor,
        *,
        target_hidden: torch.Tensor | None = None,
    ) -> LatentPredictionOutput:
        zero = chunk_hidden.new_zeros(())
        if not self.enabled or int(chunk_hidden.shape[1]) <= 1:
            return LatentPredictionOutput(loss=zero, targets=0, cosine=zero, target_source="disabled")
        pair_mask = chunk_valid[:, :-1] & chunk_valid[:, 1:]
        if not bool(pair_mask.any()):
            return LatentPredictionOutput(loss=zero, targets=0, cosine=zero, target_source="empty")
        if target_hidden is None:
            target_hidden = chunk_hidden
        source = chunk_hidden[:, :-1]
        target = target_hidden[:, 1:].detach()
        pred = self.net(source)
        selected_pred = pred[pair_mask]
        selected_target = target[pair_mask]
        pred_normed = F.normalize(selected_pred.float(), dim=-1)
        target_normed = F.normalize(selected_target.float(), dim=-1)
        cosine = (pred_normed * target_normed).sum(dim=-1).clamp(-1.0, 1.0)
        smooth = F.smooth_l1_loss(selected_pred.float(), selected_target.float(), reduction="none").mean(dim=-1)
        loss = (1.0 - cosine + 0.1 * smooth).mean().to(chunk_hidden.dtype)
        return LatentPredictionOutput(
            loss=loss,
            targets=int(selected_target.shape[0]),
            cosine=cosine.mean().to(chunk_hidden.dtype),
            target_source="next_causal_chunk_state",
        )
