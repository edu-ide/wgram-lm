from __future__ import annotations
import torch
from torch import nn
from .norm import RMSNorm


class StableInject(nn.Module):
    """Parcae-style stable source injection.

    This is a conservative implementation: normalize source, project with spectral
    normalization, gate injection magnitude, and add loop embedding.
    """

    def __init__(self, d_model: int, max_loops: int = 128):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.utils.parametrizations.spectral_norm(nn.Linear(d_model, d_model, bias=False))
        self.gate = nn.Linear(d_model, d_model, bias=True)
        self.loop_emb = nn.Embedding(max_loops, d_model)
        nn.init.zeros_(self.gate.bias)

    def forward(self, target: torch.Tensor, source: torch.Tensor, loop_id: int = 0) -> torch.Tensor:
        loop_id = min(loop_id, self.loop_emb.num_embeddings - 1)
        s = self.norm(source)
        inj = self.proj(s)
        g = torch.sigmoid(self.gate(target))
        le = self.loop_emb(torch.tensor(loop_id, device=target.device)).view(1, 1, -1)
        return inj * g + le
