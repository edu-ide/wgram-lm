"""Token-Superposition Training helpers.

These helpers implement the loss-side primitive from Token-Superposition
Training (TST): one prediction is trained against a bag of future tokens using
equal-weight multi-hot cross entropy. They do not yet change the QTRM model
forward path; full TST input superposition needs an embedding-level forward
hook in the native LM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_bags(token_ids: torch.Tensor, *, bag_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-overlapping current bags and next-token target bags.

    For a token stream segmented into bags `[0:s]`, `[s:2s]`, ... this returns
    all current bags except the last and all next bags except the first:

    - input_bags[:, i] = tokens[:, i*s:(i+1)*s]
    - target_bags[:, i] = tokens[:, (i+1)*s:(i+2)*s]

    This preserves causal ordering at bag granularity.
    """
    if int(bag_size) <= 0:
        raise ValueError("bag_size must be positive")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, seq]")
    usable = (int(token_ids.shape[1]) // int(bag_size)) * int(bag_size)
    if usable < int(bag_size) * 2:
        raise ValueError("token_ids must contain at least two full bags")
    bags = token_ids[:, :usable].reshape(token_ids.shape[0], usable // int(bag_size), int(bag_size))
    return bags[:, :-1, :], bags[:, 1:, :]


def superpose_embeddings(
    embedding_weight: torch.Tensor,
    input_bags: torch.Tensor,
) -> torch.Tensor:
    """Average token embeddings inside each input bag."""
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [vocab, dim]")
    if input_bags.ndim != 3:
        raise ValueError("input_bags must have shape [batch, bags, bag_size]")
    embedded = F.embedding(input_bags, embedding_weight)
    return embedded.mean(dim=2)


def multi_hot_cross_entropy(
    logits: torch.Tensor,
    target_bags: torch.Tensor,
    *,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Equal-weight multi-hot cross entropy over target token bags.

    `logits` has shape `[batch, bags, vocab]`; `target_bags` has shape
    `[batch, bags, bag_size]`. The loss is equivalent to averaging standard CE
    over every target token in each bag, which is the simple MCE form used by
    TST for next-bag prediction.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, bags, vocab]")
    if target_bags.ndim != 3:
        raise ValueError("target_bags must have shape [batch, bags, bag_size]")
    if tuple(logits.shape[:2]) != tuple(target_bags.shape[:2]):
        raise ValueError("logits and target_bags batch/bag dimensions must match")
    expanded_logits = logits.unsqueeze(2).expand(
        target_bags.shape[0],
        target_bags.shape[1],
        target_bags.shape[2],
        logits.shape[-1],
    )
    ce_kwargs = {}
    if ignore_index is not None:
        ce_kwargs["ignore_index"] = int(ignore_index)
    return F.cross_entropy(
        expanded_logits.reshape(-1, logits.shape[-1]),
        target_bags.reshape(-1),
        **ce_kwargs,
    )
