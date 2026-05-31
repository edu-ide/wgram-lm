"""
Explicit Multi-Trajectory + Answer-attractor Scorer

Major unapplied track from IMTA SSOT.

This provides the "A" (Answer-attractor scorer) that was largely missing or emergent.

Multiple internal trajectories compete, scorer evaluates per-trajectory attractor strength,
and the strongest influences the final z_h / answer path.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn

class MultiTrajectoryScorer(nn.Module):
    """
    Explicit scorer for multiple reasoning trajectories inside the recurrent core.
    """

    def __init__(self, d_model: int, num_trajectories: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_trajectories = num_trajectories

        # Simple scorer: projects trajectory state to a strength score
        self.strength_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, trajectory_states: torch.Tensor) -> torch.Tensor:
        """
        trajectory_states: [num_trajectories, B, d] or [B, num_trajectories, d]
        Returns scores [B, num_trajectories]
        """
        if trajectory_states.dim() == 3 and trajectory_states.shape[0] == self.num_trajectories:
            # [K, B, d] -> [B, K, d]
            trajectory_states = trajectory_states.permute(1, 0, 2)

        scores = self.strength_proj(trajectory_states).squeeze(-1)  # [B, K]
        return torch.sigmoid(scores)

    def aggregate(self, trajectory_states: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Weighted aggregation of trajectories based on scores."""
        # scores: [B, K]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, K, 1]
        aggregated = (trajectory_states * weights).sum(dim=1)  # [B, d]
        return aggregated