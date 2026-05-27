"""
Gold State Loader for 642/637 checkpoints.

Provides proper loading and projection of historical gold states
into the current architecture dimensions for deep integration.
"""

import torch
from pathlib import Path
from typing import List, Dict, Optional

def load_642_gold_states(
    ckpt_path: str = "local_eval/642_adaptive_fine_tuned_200step/adaptive_phase2_checkpoint.pt",
    device: str = "cpu",
    target_dim: int = 256,
    max_vectors: int = 8
) -> List[torch.Tensor]:
    """
    Load multiple useful latent-like tensors from the 642 gold checkpoint
    and project them to target_dim.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    candidates = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.numel() > 300:
            flat = v.flatten()
            candidates.append((k, flat))

    # Prefer latent/bos related
    candidates.sort(key=lambda x: ("latent" in x[0].lower() or "bos" in x[0].lower()), reverse=True)

    selected = []
    for name, vec in candidates[:max_vectors]:
        if vec.shape[0] != target_dim:
            proj = torch.nn.Linear(vec.shape[0], target_dim, bias=False).to(device)
            vec = proj(vec)
        selected.append(vec)

    print(f"Loaded {len(selected)} gold vectors from 642, projected to dim={target_dim}")
    return selected


def get_composite_gold_signal(
    gold_vectors: List[torch.Tensor],
    batch: int,
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """Create a composite gold signal for injection."""
    if weights is None:
        weights = [0.1 / (i + 1) for i in range(len(gold_vectors))]

    signal = torch.zeros_like(gold_vectors[0]).unsqueeze(0).repeat(batch, 1)
    for vec, w in zip(gold_vectors, weights):
        signal += vec.unsqueeze(0).repeat(batch, 1) * w
    return signal
