from __future__ import annotations
import torch
from torch import nn


class ControllerHeads(nn.Module):
    def __init__(self, d_model: int, num_actions: int):
        super().__init__()
        self.halt = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, 1)
        self.energy = nn.Linear(d_model, 1)
        self.action = nn.Linear(d_model, num_actions)
        self.retrieve_text = nn.Linear(d_model, 1)
        self.retrieve_image = nn.Linear(d_model, 1)
        self.verify = nn.Linear(d_model, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "halt_logits": self.halt(pooled).squeeze(-1),
            "value_logits": self.value(pooled).squeeze(-1),
            "energy": self.energy(pooled).squeeze(-1),
            "action_logits": self.action(pooled),
            "retrieve_text_logits": self.retrieve_text(pooled).squeeze(-1),
            "retrieve_image_logits": self.retrieve_image(pooled).squeeze(-1),
            "verify_logits": self.verify(pooled).squeeze(-1),
        }
