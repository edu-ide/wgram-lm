from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def token_text(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def topk_token_report(
    logits: torch.Tensor,
    tokenizer: Any = None,
    k: int = 10,
) -> list[dict[str, float | int | str]]:
    """Return a small human-readable top-k report for one logits vector."""
    if logits.ndim != 1:
        logits = logits.reshape(-1, logits.shape[-1])[-1]
    k = min(int(k), logits.shape[-1])
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)
    out = []
    for prob, token_id in zip(top_probs, top_ids):
        tid = int(token_id.detach().cpu().item())
        out.append(
            {
                "token_id": tid,
                "token": token_text(tokenizer, tid),
                "prob": _as_float(prob),
                "logit": _as_float(logits[tid]),
            }
        )
    return out


def residual_logit_telemetry(
    donor_logits: torch.Tensor,
    fused_logits: torch.Tensor,
    *,
    tokenizer: Any = None,
    donor_logits_scale: float = 1.0,
) -> dict[str, float | int | bool | str]:
    """Compare donor-only logits with fused donor-plus-residual logits.

    The QTRM residual path fuses donor logits inside the model. This helper
    treats the scaled donor distribution as the reference policy and reports how
    much the final fused distribution moved.
    """
    if donor_logits.ndim != 1:
        donor_logits = donor_logits.reshape(-1, donor_logits.shape[-1])[-1]
    if fused_logits.ndim != 1:
        fused_logits = fused_logits.reshape(-1, fused_logits.shape[-1])[-1]
    if donor_logits.shape != fused_logits.shape:
        raise ValueError(
            "donor_logits and fused_logits must have the same final vocab shape"
        )

    donor_scaled = donor_logits.float() * float(donor_logits_scale)
    fused = fused_logits.float()
    residual = fused - donor_scaled

    donor_log_probs = torch.log_softmax(donor_scaled, dim=-1)
    fused_log_probs = torch.log_softmax(fused, dim=-1)
    donor_probs = torch.softmax(donor_scaled, dim=-1)
    fused_probs = torch.softmax(fused, dim=-1)

    donor_top = int(donor_scaled.argmax(dim=-1).detach().cpu().item())
    fused_top = int(fused.argmax(dim=-1).detach().cpu().item())
    residual_top = int(residual.argmax(dim=-1).detach().cpu().item())
    residual_bottom = int(residual.argmin(dim=-1).detach().cpu().item())
    kl_fused_to_donor = (fused_probs * (fused_log_probs - donor_log_probs)).sum()
    kl_donor_to_fused = (donor_probs * (donor_log_probs - fused_log_probs)).sum()

    return {
        "donor_logits_scale": float(donor_logits_scale),
        "donor_top_id": donor_top,
        "donor_top_token": token_text(tokenizer, donor_top),
        "donor_top_prob": _as_float(donor_probs[donor_top]),
        "fused_top_id": fused_top,
        "fused_top_token": token_text(tokenizer, fused_top),
        "fused_top_prob": _as_float(fused_probs[fused_top]),
        "argmax_changed": bool(donor_top != fused_top),
        "kl_fused_to_donor": _as_float(kl_fused_to_donor),
        "kl_donor_to_fused": _as_float(kl_donor_to_fused),
        "residual_l2_norm": _as_float(torch.linalg.vector_norm(residual, ord=2)),
        "residual_linf_norm": _as_float(residual.abs().max()),
        "residual_mean_abs": _as_float(residual.abs().mean()),
        "residual_top_id": residual_top,
        "residual_top_token": token_text(tokenizer, residual_top),
        "residual_top_value": _as_float(residual[residual_top]),
        "residual_bottom_id": residual_bottom,
        "residual_bottom_token": token_text(tokenizer, residual_bottom),
        "residual_bottom_value": _as_float(residual[residual_bottom]),
    }


def next_token_diagnostics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, float | int]:
    """Compute teacher-forced next-token probes.

    QTRM prepends donor/workspace tokens to the logits stream. `offset` points to
    the first logit position aligned with `input_ids[:, 0]`.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if input_ids.shape[1] < 2:
        raise ValueError("input_ids must contain at least two tokens")

    aligned = logits[:, offset : offset + input_ids.shape[1] - 1, :].float()
    targets = input_ids[:, 1:].to(device=aligned.device)
    if aligned.shape[:2] != targets.shape:
        raise ValueError(
            "logits/input_ids alignment failed: "
            f"aligned={tuple(aligned.shape)}, targets={tuple(targets.shape)}, offset={offset}"
        )

    if attention_mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool, device=aligned.device)
    else:
        mask = attention_mask[:, 1:].to(device=aligned.device, dtype=torch.bool)

    if not mask.any():
        return {
            "valid_tokens": 0,
            "loss": 0.0,
            "ppl": 1.0,
            "target_rank_mean": 0.0,
            "target_rank_median": 0.0,
            "target_top1_acc": 0.0,
            "target_top5_acc": 0.0,
            "entropy_mean": 0.0,
            "max_prob_mean": 0.0,
        }

    flat_logits = aligned[mask]
    flat_targets = targets[mask]
    loss = F.cross_entropy(flat_logits, flat_targets)
    probs = torch.softmax(flat_logits, dim=-1)
    log_probs = torch.log_softmax(flat_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    target_logits = flat_logits.gather(1, flat_targets[:, None])
    ranks = (flat_logits > target_logits).sum(dim=-1) + 1
    top5_k = min(5, flat_logits.shape[-1])
    top5 = torch.topk(flat_logits, k=top5_k, dim=-1).indices
    top1 = flat_logits.argmax(dim=-1)

    return {
        "valid_tokens": int(mask.sum().detach().cpu().item()),
        "loss": _as_float(loss),
        "ppl": _as_float(torch.exp(loss.clamp(max=20.0))),
        "target_rank_mean": _as_float(ranks.float().mean()),
        "target_rank_median": _as_float(ranks.float().median()),
        "target_top1_acc": _as_float((top1 == flat_targets).float().mean()),
        "target_top5_acc": _as_float((top5 == flat_targets[:, None]).any(dim=-1).float().mean()),
        "entropy_mean": _as_float(entropy.mean()),
        "max_prob_mean": _as_float(max_prob.mean()),
    }


def _ngrams(ids: Sequence[int], n: int) -> Iterable[tuple[int, ...]]:
    for idx in range(0, max(0, len(ids) - n + 1)):
        yield tuple(ids[idx : idx + n])


def repetition_stats(
    token_ids: Sequence[int],
    *,
    prompt_len: int = 0,
    ngram_sizes: Sequence[int] = (1, 2, 3, 4),
) -> dict[str, float | int | None]:
    """Report repeated-token and repeated-ngram rates for generated completion."""
    completion = [int(x) for x in token_ids[int(prompt_len) :]]
    stats: dict[str, float | int | None] = {
        "prompt_tokens": int(prompt_len),
        "completion_tokens": len(completion),
        "max_token_run": 0,
        "most_common_token_id": None,
        "most_common_token_count": 0,
    }
    if not completion:
        for n in ngram_sizes:
            stats[f"repeated_{n}gram_rate"] = 0.0
        return stats

    max_run = 1
    current_run = 1
    for prev, cur in zip(completion, completion[1:]):
        if cur == prev:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    counts = Counter(completion)
    token_id, token_count = counts.most_common(1)[0]
    stats["max_token_run"] = max_run
    stats["most_common_token_id"] = token_id
    stats["most_common_token_count"] = token_count

    for n in ngram_sizes:
        grams = list(_ngrams(completion, int(n)))
        if not grams:
            stats[f"repeated_{n}gram_rate"] = 0.0
            continue
        gram_counts = Counter(grams)
        repeated = sum(count - 1 for count in gram_counts.values() if count > 1)
        stats[f"repeated_{n}gram_rate"] = repeated / len(grams)
    return stats
