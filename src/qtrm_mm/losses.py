from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def next_token_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Standard causal full-sequence CE. In donor/workspace modes, use prefix-only
    # sampling if leakage is a concern. Smoke training uses ordinary causal stack.
    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:].clone()
    if attention_mask is not None:
        targets = targets.masked_fill(attention_mask[:, 1:].to(torch.bool).logical_not(), -100)
    return F.cross_entropy(
        logits[:, offset:-1].reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def jepa_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float().detach(), dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(torch.bool)
    if not mask.any():
        return values.sum() * 0.0
    return values.masked_select(mask).mean()


def jepa_world_model_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    latents: Optional[torch.Tensor] = None,
    latent_mask: Optional[torch.Tensor] = None,
    sigreg=None,
    sigreg_weight: float = 0.09,
) -> torch.Tensor:
    """LeWM-style next-embedding loss plus SIGReg anti-collapse regularizer.

    Unlike I-JEPA/older JEPA-WM variants, LeWM does not stop-gradient the target
    branch during training. The encoder and predictor are optimized end-to-end.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if pred.numel() == 0:
        return pred.sum() * 0.0

    per_transition = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    pred_loss = _masked_mean(per_transition, mask)
    if sigreg is None or latents is None or sigreg_weight == 0:
        return pred_loss
    return pred_loss + sigreg_weight * sigreg(latents, latent_mask)


def controller_aux_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    # Weak regularizers to prevent immediate collapse in smoke training.
    loss = 0.0
    halt = outputs["halt_logits"]
    action = outputs["action_logits"]
    loss = loss + 0.01 * halt.float().pow(2).mean()
    probs = action.float().softmax(dim=-1)
    entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
    loss = loss - 0.001 * entropy
    return loss


def _expand_halt_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    if target.ndim == 1:
        target = target[:, None].expand_as(logits)
    if target.shape != logits.shape:
        raise ValueError("halt target must have shape [batch] or match halt logits")
    return target


def core_halt_loss(
    outputs: dict[str, torch.Tensor],
    target_halt: Optional[torch.Tensor] = None,
    target_continue: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_halt = outputs.get("core_q_halt_logits")
    if q_halt is None:
        anchor = outputs["logits"]
        return anchor.sum() * 0.0
    if q_halt.numel() == 0 or target_halt is None:
        return q_halt.sum() * 0.0

    halt_target = _expand_halt_target(target_halt, q_halt)
    loss = F.binary_cross_entropy_with_logits(q_halt.float(), halt_target.float())

    q_continue = outputs.get("core_q_continue_logits")
    if target_continue is not None and q_continue is not None and q_continue.numel() > 0:
        continue_target = _expand_halt_target(target_continue, q_continue)
        loss = loss + F.binary_cross_entropy_with_logits(
            q_continue.float(),
            continue_target.float(),
        )
    return loss


def qtrm_smoke_loss(
    model,
    input_ids: torch.Tensor,
    *,
    jepa_weight: float = 0.1,
    aux_weight: float = 1.0,
    core_halt_weight: float = 0.0,
    **kwargs,
):
    labels = kwargs.get("labels")
    core_halt_targets = kwargs.get("core_halt_targets")
    core_continue_targets = kwargs.get("core_continue_targets")
    private_keys = {"labels", "core_halt_targets", "core_continue_targets"}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in private_keys}
    outputs = model(input_ids=input_ids, **model_kwargs)
    offset = outputs["logits"].shape[1] - input_ids.shape[1]
    lm = next_token_lm_loss(
        outputs["logits"],
        input_ids,
        offset=offset,
        attention_mask=model_kwargs.get("attention_mask"),
        labels=labels,
    )
    jepa = jepa_world_model_loss(
        outputs["jepa_pred"],
        outputs["jepa_target"],
        outputs.get("jepa_mask"),
        latents=outputs.get("jepa_latents"),
        latent_mask=outputs.get("jepa_latent_mask"),
        sigreg=getattr(model, "jepa_sigreg", None),
        sigreg_weight=getattr(getattr(model, "cfg", None), "jepa_sigreg_weight", 0.09),
    )
    aux = controller_aux_loss(outputs)
    core_halt = core_halt_loss(
        outputs,
        target_halt=core_halt_targets,
        target_continue=core_continue_targets,
    )
    loss = (
        lm
        + float(jepa_weight) * jepa
        + float(aux_weight) * aux
        + float(core_halt_weight) * core_halt
    )
    return (
        loss,
        {
            "loss": loss.detach(),
            "lm": lm.detach(),
            "jepa": jepa.detach(),
            "aux": aux.detach(),
            "core_halt": core_halt.detach(),
        },
        outputs,
    )
