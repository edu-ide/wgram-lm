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
    token_logits = logits[:, offset:-1]
    if not torch.any(targets != -100):
        return token_logits.sum() * 0.0
    return F.cross_entropy(
        token_logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def simpo_margin_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    *,
    beta: float = 2.0,
    margin: float = 0.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SimPO-style sequence-level preference loss from average log-probs."""
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("chosen and rejected log-probs must have the same shape")
    advantage = (chosen_logps.float() - rejected_logps.float()) - float(margin)
    per_sample = F.softplus(-float(beta) * advantage)
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).view_as(per_sample)
    weights = weights.clamp_min(0.0)
    if not torch.any(weights > 0):
        return per_sample.sum() * 0.0
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def sequence_average_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:]
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    token_logits = logits[:, offset:-1]
    if token_logits.shape[:2] != targets.shape:
        raise ValueError("logits/text target shape mismatch while computing sequence log-prob")
    safe_targets = targets.masked_fill(valid.logical_not(), 0)
    log_probs = token_logits.float().log_softmax(dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    return _masked_mean_per_sample(token_logps, valid.to(device=token_logps.device))


def canonical_causal_ablation_loss(
    chosen_logps: torch.Tensor,
    ablation_logps: list[torch.Tensor],
    *,
    beta: float = 2.0,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Prefer the full canonical answer path over explicit internal ablations.

    Rejected/ablated log-probs are stop-gradient targets. The goal is to make
    the full SSOT path better, not to teach the shared model to sabotage its
    own ablated paths.
    """
    if not ablation_logps:
        zero = chosen_logps.sum() * 0.0
        return zero, {
            "canonical_causal_margin": zero,
            "canonical_causal_full_logp": chosen_logps.mean().detach(),
            "canonical_causal_ablation_logp": zero,
        }
    rejected = torch.stack([logp.to(device=chosen_logps.device) for logp in ablation_logps], dim=0)
    rejected_mean = rejected.mean(dim=0)
    loss = simpo_margin_loss(
        chosen_logps,
        rejected_mean.detach(),
        beta=beta,
        margin=margin,
    )
    return loss, {
        "canonical_causal_margin": (chosen_logps - rejected_mean).mean().detach(),
        "canonical_causal_full_logp": chosen_logps.mean().detach(),
        "canonical_causal_ablation_logp": rejected_mean.mean().detach(),
    }


def _canonical_causal_forward_kwargs(model_kwargs: dict, mode: str) -> dict:
    ablation_kwargs = dict(model_kwargs)
    if mode == "core_off":
        ablation_kwargs["disable_core"] = True
    elif mode == "workspace_off":
        ablation_kwargs["disable_workspace"] = True
    elif mode == "workspace_memory_off":
        ablation_kwargs["disable_workspace_memory_context"] = True
    elif mode == "core_context_off":
        ablation_kwargs["disable_core_context"] = True
    elif mode == "core_to_text_off":
        ablation_kwargs["disable_core_to_text"] = True
    elif mode == "evidence_bottleneck_off":
        ablation_kwargs["disable_evidence_bottleneck"] = True
    elif mode == "span_reader_off":
        ablation_kwargs["disable_evidence_span_reader"] = True
    elif mode == "transition_state_off":
        ablation_kwargs["disable_transition_state"] = True
    else:
        raise ValueError(f"unknown canonical causal ablation mode: {mode}")
    return ablation_kwargs


def donor_logit_distillation_loss(
    student_logits: torch.Tensor,
    donor_logits: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized KD loss from donor logits to the fused/student policy.

    `beta=0` matches ordinary teacher-to-student forward KL. `beta=1` matches
    the reverse-KL direction used by MiniLLM/GKD-style variants.
    """
    if donor_logits is None:
        return student_logits.sum() * 0.0
    target_source = labels if labels is not None else input_ids
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    if not valid.any():
        return student_logits.sum() * 0.0

    student = student_logits[:, offset:-1]
    donor = donor_logits[:, : student.shape[1]].to(device=student.device, dtype=student.dtype)
    if student.shape != donor.shape:
        raise ValueError("student logits and donor logits must align for distillation")

    temperature = max(float(temperature), 1e-6)
    beta_value = min(max(float(beta), 0.0), 1.0)
    student_log_probs = F.log_softmax(student.float() / temperature, dim=-1)
    donor_log_probs = F.log_softmax(donor.float() / temperature, dim=-1)

    if beta_value == 0.0:
        per_vocab = F.kl_div(student_log_probs, donor_log_probs, reduction="none", log_target=True)
    elif beta_value == 1.0:
        per_vocab = F.kl_div(donor_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_tensor = torch.tensor(beta_value, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_tensor),
                    donor_log_probs + torch.log(beta_tensor),
                ]
            ),
            dim=0,
        )
        kl_donor = F.kl_div(mixture_log_probs, donor_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = beta_tensor * kl_donor + (1.0 - beta_tensor) * kl_student

    per_token = per_vocab.sum(dim=-1)
    return per_token.masked_select(valid.to(device=per_token.device)).mean()


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


def action_policy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action_logits = outputs.get("action_logits")
    if action_logits is None or target is None:
        anchor = outputs["logits"]
        zero = anchor.sum() * 0.0
        return zero, {"action_acc": zero}

    logits = action_logits.float()
    target = target.to(device=logits.device, dtype=torch.long).view(-1)
    if logits.ndim != 2:
        raise ValueError("action_logits must have shape [batch, num_actions]")
    if target.shape[0] != logits.shape[0]:
        raise ValueError("action target batch size must match action logits")

    valid = (target >= 0) & (target < logits.shape[-1])
    if not torch.any(valid):
        zero = logits.sum() * 0.0
        return zero, {"action_acc": zero}

    per_sample = F.cross_entropy(logits, target.clamp_min(0), reduction="none")
    if sample_weight is not None:
        weights = sample_weight.to(device=logits.device, dtype=per_sample.dtype).view(-1)
        weights = weights.clamp_min(0.0) * valid.to(dtype=per_sample.dtype)
        if not torch.any(weights > 0):
            loss = per_sample.sum() * 0.0
        else:
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample.masked_select(valid).mean()

    pred = logits.argmax(dim=-1)
    acc_values = (pred == target).to(dtype=logits.dtype).masked_select(valid)
    action_acc = acc_values.mean() if acc_values.numel() else logits.sum() * 0.0
    return loss, {"action_acc": action_acc.detach()}


def controller_signal_prediction_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    signal_logits = outputs.get("controller_signal_logits")
    if signal_logits is None or signal_logits.numel() == 0 or target is None:
        anchor = outputs["logits"]
        zero = anchor.sum() * 0.0
        return zero, {"controller_signal_acc": zero}

    logits = signal_logits.float()
    target = target.to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        raise ValueError("controller signal target must match signal logits shape")

    per_dim = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per_sample = per_dim.mean(dim=-1)
    if sample_weight is not None:
        weights = sample_weight.to(device=logits.device, dtype=per_sample.dtype).view(-1)
        weights = weights.clamp_min(0.0)
        if not torch.any(weights > 0):
            loss = per_sample.sum() * 0.0
        else:
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample.mean()

    pred = (torch.sigmoid(logits) >= 0.5).to(dtype=target.dtype)
    signal_acc = (pred == target).to(dtype=logits.dtype).mean()
    return loss, {"controller_signal_acc": signal_acc.detach()}


def answer_decision_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decision_logits = outputs.get("answer_decision_logits")
    if decision_logits is None or decision_logits.numel() == 0 or target is None:
        anchor = outputs["logits"]
        zero = anchor.sum() * 0.0
        return zero, {"answer_decision_acc": zero, "answer_decision_block_prob": zero}

    logits = decision_logits.float()
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    target = target.to(device=logits.device, dtype=logits.dtype).view_as(logits)

    per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if sample_weight is not None:
        weights = sample_weight.to(device=logits.device, dtype=per_sample.dtype).view_as(per_sample)
        weights = weights.clamp_min(0.0)
        if not torch.any(weights > 0):
            loss = per_sample.sum() * 0.0
        else:
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample.mean()

    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).to(dtype=target.dtype)
    acc = (pred == target).to(dtype=logits.dtype).mean()
    return loss, {
        "answer_decision_acc": acc.detach(),
        "answer_decision_block_prob": prob.mean().detach(),
    }


def answer_residual_governor_loss(
    outputs: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    *,
    labels: Optional[torch.Tensor] = None,
    donor_logits: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gate_logits = outputs.get("answer_residual_governor_logits")
    anchor = outputs["logits"]
    zero = anchor.sum() * 0.0
    if gate_logits is None or gate_logits.numel() == 0 or donor_logits is None:
        return zero, {
            "answer_residual_governor_acc": zero,
            "answer_residual_governor_open_rate": zero,
            "answer_residual_governor_target_open_rate": zero,
        }

    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:]
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    token_gate_logits = gate_logits[:, : targets.shape[1]].float()
    if token_gate_logits.shape != targets.shape:
        raise ValueError("answer residual governor logits must align to input tokens")
    donor_slice = donor_logits[:, : targets.shape[1]].to(
        device=token_gate_logits.device,
        dtype=token_gate_logits.dtype,
    )
    if donor_slice.shape[:2] != targets.shape:
        raise ValueError("donor logits must align to answer residual governor labels")

    safe_targets = targets.masked_fill(valid.logical_not(), 0)
    donor_wrong = donor_slice.argmax(dim=-1) != safe_targets
    target = donor_wrong.to(dtype=token_gate_logits.dtype)
    if not torch.any(valid):
        return zero, {
            "answer_residual_governor_acc": zero,
            "answer_residual_governor_open_rate": zero,
            "answer_residual_governor_target_open_rate": zero,
        }

    per_token = F.binary_cross_entropy_with_logits(
        token_gate_logits,
        target,
        reduction="none",
    )
    loss = per_token.masked_select(valid.to(device=per_token.device)).mean()
    prob = torch.sigmoid(token_gate_logits)
    pred = prob >= 0.5
    valid_on_device = valid.to(device=pred.device)
    acc = (pred == donor_wrong.to(device=pred.device)).to(dtype=prob.dtype)
    return loss, {
        "answer_residual_governor_acc": acc.masked_select(valid_on_device).mean().detach(),
        "answer_residual_governor_open_rate": prob.masked_select(valid_on_device).mean().detach(),
        "answer_residual_governor_target_open_rate": target.masked_select(
            valid.to(device=target.device)
        ).mean().detach(),
    }


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


def _valid_next_token_mask(
    target_source: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = target_source[:, 1:] != -100
    if attention_mask is not None:
        valid = valid & attention_mask[:, 1:].to(torch.bool)
    return valid


def repetition_unlikelihood_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize high probability on adjacent wrong-repeat candidates.

    This is deliberately narrow: it targets the failure mode seen in probes
    ("Freeze Freeze ..." / "world of the world") without applying a broad
    no-repeat rule that would punish legitimate gold repeated words.
    """
    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:]
    prev_candidates = target_source[:, :-1]
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    candidate_mask = valid & (prev_candidates >= 0) & (targets != prev_candidates)

    token_logits = logits[:, offset:-1]
    if token_logits.shape[:2] != targets.shape:
        raise ValueError("logits/text target shape mismatch while computing repetition loss")
    if not candidate_mask.any():
        return token_logits.sum() * 0.0

    safe_candidates = prev_candidates.masked_fill(candidate_mask.logical_not(), 0)
    probs = token_logits.float().softmax(dim=-1)
    repeated_probs = probs.gather(dim=-1, index=safe_candidates.unsqueeze(-1)).squeeze(-1)
    per_token = -torch.log1p(-repeated_probs.clamp(max=1.0 - 1e-6))
    return per_token.masked_select(candidate_mask.to(device=per_token.device)).mean()


def greedy_token_margin_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    donor_logits: Optional[torch.Tensor] = None,
    margin: float = 0.0,
    only_donor_errors: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Force gold next tokens to beat greedy competitors in final fused logits.

    CE can improve average log-prob without changing argmax when donor logits
    dominate. This loss directly targets the greedy-decode failure mode by
    requiring the supervised token to exceed the strongest non-target token by
    a margin on answer-label positions.
    """
    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:]
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    token_logits = logits[:, offset:-1]
    if token_logits.shape[:2] != targets.shape:
        raise ValueError("logits/text target shape mismatch while computing greedy margin")
    if token_logits.shape[-1] <= 1:
        zero = token_logits.sum() * 0.0
        return zero, {
            "greedy_token_win_rate": zero,
            "greedy_token_active_rate": zero,
            "greedy_token_donor_error_rate": zero,
        }
    if not torch.any(valid):
        zero = token_logits.sum() * 0.0
        return zero, {
            "greedy_token_win_rate": zero,
            "greedy_token_active_rate": zero,
            "greedy_token_donor_error_rate": zero,
        }

    safe_targets = targets.masked_fill(valid.logical_not(), 0)
    target_logits = token_logits.gather(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    target_mask = torch.zeros_like(token_logits, dtype=torch.bool)
    target_mask.scatter_(-1, safe_targets.unsqueeze(-1), True)
    competitor_logits = token_logits.float().masked_fill(target_mask, -torch.inf)
    top_competitor = competitor_logits.max(dim=-1).values

    active = valid
    donor_error_rate = token_logits.new_tensor(0.0)
    if donor_logits is not None:
        donor_slice = donor_logits[:, : token_logits.shape[1]].to(
            device=token_logits.device,
            dtype=token_logits.dtype,
        )
        if donor_slice.shape != token_logits.shape:
            raise ValueError("donor logits and fused logits must align for greedy margin")
        donor_wrong = donor_slice.argmax(dim=-1) != safe_targets
        donor_valid_wrong = donor_wrong & valid
        donor_error_rate = (
            donor_valid_wrong.float().sum()
            / valid.float().sum().clamp_min(1.0)
        ).detach()
        if only_donor_errors:
            active = donor_valid_wrong

    active_rate = (
        active.float().sum() / valid.float().sum().clamp_min(1.0)
    ).detach()
    if not torch.any(active):
        zero = token_logits.sum() * 0.0
        return zero, {
            "greedy_token_win_rate": zero,
            "greedy_token_active_rate": active_rate,
            "greedy_token_donor_error_rate": donor_error_rate,
        }

    required_margin = float(margin)
    violations = F.relu(required_margin + top_competitor - target_logits.float())
    loss = violations.masked_select(active.to(device=violations.device)).mean()
    wins = (target_logits.float() >= top_competitor + required_margin).to(
        dtype=token_logits.dtype
    )
    win_rate = wins.masked_select(active.to(device=wins.device)).mean().detach()
    return loss, {
        "greedy_token_win_rate": win_rate,
        "greedy_token_active_rate": active_rate,
        "greedy_token_donor_error_rate": donor_error_rate,
    }


def donor_correct_margin_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    donor_logits: Optional[torch.Tensor] = None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve donor-correct next tokens against QTRM residual overwrites.

    The donor-error margin teaches QTRM where to intervene. This companion loss
    teaches the opposite: when the frozen donor already puts the gold token on
    top, the fused model must not let the residual move a wrong token above it.
    """
    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:]
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    token_logits = logits[:, offset:-1]
    if token_logits.shape[:2] != targets.shape:
        raise ValueError("logits/text target shape mismatch while computing donor-correct margin")

    zero = token_logits.sum() * 0.0
    if donor_logits is None or token_logits.shape[-1] <= 1 or not torch.any(valid):
        return zero, {
            "donor_correct_margin_win_rate": zero,
            "donor_correct_margin_active_rate": zero,
        }

    donor_slice = donor_logits[:, : token_logits.shape[1]].to(
        device=token_logits.device,
        dtype=token_logits.dtype,
    )
    if donor_slice.shape != token_logits.shape:
        raise ValueError("donor logits and fused logits must align for donor-correct margin")

    safe_targets = targets.masked_fill(valid.logical_not(), 0)
    donor_correct = donor_slice.argmax(dim=-1) == safe_targets
    active = valid & donor_correct
    active_rate = (
        active.float().sum() / valid.float().sum().clamp_min(1.0)
    ).detach()
    if not torch.any(active):
        return zero, {
            "donor_correct_margin_win_rate": zero,
            "donor_correct_margin_active_rate": active_rate,
        }

    target_logits = token_logits.gather(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    target_mask = torch.zeros_like(token_logits, dtype=torch.bool)
    target_mask.scatter_(-1, safe_targets.unsqueeze(-1), True)
    competitor_logits = token_logits.float().masked_fill(target_mask, -torch.inf)
    top_competitor = competitor_logits.max(dim=-1).values
    required_margin = float(margin)
    violations = F.relu(required_margin + top_competitor - target_logits.float())
    loss = violations.masked_select(active.to(device=violations.device)).mean()
    wins = (target_logits.float() >= top_competitor + required_margin).to(
        dtype=token_logits.dtype
    )
    win_rate = wins.masked_select(active.to(device=wins.device)).mean().detach()
    return loss, {
        "donor_correct_margin_win_rate": win_rate,
        "donor_correct_margin_active_rate": active_rate,
    }


def generation_verifier_loss(
    outputs: dict[str, torch.Tensor],
    *,
    repeat_target: Optional[torch.Tensor] = None,
    stop_target: Optional[torch.Tensor] = None,
    quality_target: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    repeat_logits = outputs.get("generation_repeat_logits")
    stop_logits = outputs.get("generation_stop_logits")
    quality_logits = outputs.get("generation_quality_logits")
    if (
        repeat_logits is None
        or stop_logits is None
        or quality_logits is None
        or repeat_logits.numel() == 0
        or stop_logits.numel() == 0
        or quality_logits.numel() == 0
    ):
        anchor = outputs["logits"]
        zero = anchor.sum() * 0.0
        return zero, {
            "repeat_prob": zero,
            "stop_prob": zero,
            "quality_prob": zero,
        }

    losses = [
        _weighted_bce_with_logits(repeat_logits, repeat_target, sample_weight=sample_weight, default=0.0),
        _weighted_bce_with_logits(stop_logits, stop_target, sample_weight=sample_weight, default=0.0),
        _weighted_bce_with_logits(quality_logits, quality_target, sample_weight=sample_weight, default=1.0),
    ]
    loss = sum(losses) / float(len(losses))
    return loss, {
        "repeat_prob": repeat_logits.float().sigmoid().mean().detach(),
        "stop_prob": stop_logits.float().sigmoid().mean().detach(),
        "quality_prob": quality_logits.float().sigmoid().mean().detach(),
    }


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    target: Optional[torch.Tensor],
    *,
    sample_weight: Optional[torch.Tensor],
    default: float,
) -> torch.Tensor:
    target_tensor = _target_like(logits, target, default=default)
    per_sample = F.binary_cross_entropy_with_logits(
        logits.float(),
        target_tensor.float(),
        reduction="none",
    )
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).view_as(per_sample)
    weights = weights.clamp_min(0.0)
    if not torch.any(weights > 0):
        return per_sample.sum() * 0.0
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def _masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(mask.logical_not(), 0.0)
    denom = mask.sum(dim=-1).clamp_min(1).to(values.dtype)
    return masked.sum(dim=-1) / denom


def infer_core_halt_targets(
    outputs: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    *,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    offset: int = 0,
    verifier_passed: Optional[torch.Tensor] = None,
    donor_logits: Optional[torch.Tensor] = None,
    donor_logits_scale: float = 1.0,
    donor_kl_threshold: Optional[float] = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_source = labels if labels is not None else input_ids
    targets = target_source[:, 1:]
    valid = _valid_next_token_mask(target_source, attention_mask=attention_mask)
    logits = outputs["logits"][:, offset:-1]
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits/text target shape mismatch while inferring core halt targets")

    safe_targets = targets.masked_fill(valid.logical_not(), 0)
    predicted = logits.argmax(dim=-1)
    token_correct = (predicted == safe_targets) | valid.logical_not()
    has_valid = valid.any(dim=-1)
    exact_next_token_pass = token_correct.all(dim=-1) & has_valid
    halt_pass = exact_next_token_pass
    diagnostics = {
        "valid_sample_rate": has_valid.float().mean().detach(),
        "exact_next_token_pass_rate": exact_next_token_pass.float().mean().detach(),
    }

    if verifier_passed is not None:
        verifier_mask = verifier_passed.to(device=halt_pass.device, dtype=torch.bool)
        diagnostics["verifier_pass_rate"] = verifier_mask.float().mean().detach()
        halt_pass = halt_pass & verifier_mask

    if donor_logits is not None and donor_kl_threshold is not None:
        donor_slice = donor_logits[:, : logits.shape[1]].to(device=logits.device, dtype=logits.dtype)
        donor_slice = donor_slice * float(donor_logits_scale)
        fused_log_probs = logits.float().log_softmax(dim=-1)
        donor_log_probs = donor_slice.float().log_softmax(dim=-1)
        fused_probs = fused_log_probs.exp()
        per_token_kl = (fused_probs * (fused_log_probs - donor_log_probs)).sum(dim=-1)
        kl_per_sample = _masked_mean_per_sample(per_token_kl, valid)
        donor_kl_pass = kl_per_sample <= float(donor_kl_threshold)
        diagnostics["donor_kl_pass_rate"] = donor_kl_pass.float().mean().detach()
        diagnostics["donor_kl_mean"] = kl_per_sample.float().mean().detach()
        halt_pass = halt_pass & donor_kl_pass

    halt_targets = halt_pass.to(dtype=torch.float32)
    diagnostics["halt_target_pos_rate"] = halt_targets.mean().detach()
    diagnostics["halt_target_neg_rate"] = (1.0 - halt_targets).mean().detach()
    if return_diagnostics:
        return halt_targets, diagnostics
    return halt_targets


def infer_core_halt_targets_from_teacher_depth(
    outputs: dict[str, torch.Tensor],
    *,
    similarity_threshold: float = 0.995,
    logit_kl_threshold: float = 0.05,
    min_step: int = 1,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    depth_logits = outputs.get("core_depth_last_logits")
    if depth_logits is not None and depth_logits.shape[1] > 0:
        if depth_logits.ndim != 3:
            raise ValueError("core_depth_last_logits must have shape [batch, steps, vocab_size]")
        logits = depth_logits.detach().float()
        b, steps, _ = logits.shape
        step_log_probs = logits.log_softmax(dim=-1)
        final_log_probs = step_log_probs[:, -1:, :]
        step_probs = step_log_probs.exp()
        kl_to_final = (step_probs * (step_log_probs - final_log_probs)).sum(dim=-1)
        centered_logits = logits - logits.mean(dim=-1, keepdim=True)
        logit_similarity = (
            F.normalize(centered_logits, dim=-1)
            * F.normalize(centered_logits[:, -1:, :], dim=-1)
        ).sum(dim=-1)
        top1_match = logits.argmax(dim=-1) == logits[:, -1:, :].argmax(dim=-1)
        stable = (
            top1_match
            & (logit_similarity >= float(similarity_threshold))
            & (kl_to_final <= float(logit_kl_threshold))
        )
        stable[:, -1] = True

        halt, cont, first_stable = _teacher_depth_targets_from_stable(
            stable,
            anchor=depth_logits,
            min_step=min_step,
        )
        diagnostics = {
            "teacher_depth_halt_pos_rate": halt.float().mean().detach(),
            "teacher_depth_halt_neg_rate": cont.float().mean().detach(),
            "teacher_depth_earliest_step_mean": (first_stable.float() + 1.0).mean().detach(),
            "teacher_depth_logit_kl_mean": kl_to_final.float().mean().detach(),
            "teacher_depth_logit_similarity_mean": logit_similarity.float().mean().detach(),
            "teacher_depth_top1_match_rate": top1_match.float().mean().detach(),
            "teacher_depth_step1_stable_rate": stable[:, 0].float().mean().detach(),
        }
        if return_diagnostics:
            return halt, cont, diagnostics
        return halt, cont

    depth_states = outputs.get("core_depth_states")
    if depth_states is None:
        raise ValueError("core_depth_states are required for teacher-depth halt targets")
    if depth_states.ndim != 3:
        raise ValueError("core_depth_states must have shape [batch, steps, d_model]")

    b, steps, _ = depth_states.shape
    if steps == 0:
        q_halt = outputs.get("core_q_halt_logits")
        if q_halt is None:
            raise ValueError("empty core_depth_states require core_q_halt_logits for shape")
        halt = q_halt.new_empty(q_halt.shape)
        cont = q_halt.new_empty(q_halt.shape)
        diagnostics = {
            "teacher_depth_halt_pos_rate": halt.sum() * 0.0,
            "teacher_depth_halt_neg_rate": halt.sum() * 0.0,
            "teacher_depth_earliest_step_mean": halt.sum() * 0.0,
        }
        if return_diagnostics:
            return halt, cont, diagnostics
        return halt, cont

    states = depth_states.detach().float()
    final = F.normalize(states[:, -1:, :], dim=-1)
    normalized = F.normalize(states, dim=-1)
    similarity = (normalized * final).sum(dim=-1)
    stable = similarity >= float(similarity_threshold)
    stable[:, -1] = True
    min_step = max(1, int(min_step))
    if min_step > 1:
        stable[:, : min(min_step - 1, steps)] = False

    halt, cont, first_stable = _teacher_depth_targets_from_stable(
        stable,
        anchor=depth_states,
        min_step=1,
    )

    diagnostics = {
        "teacher_depth_halt_pos_rate": halt.float().mean().detach(),
        "teacher_depth_halt_neg_rate": cont.float().mean().detach(),
        "teacher_depth_earliest_step_mean": (first_stable.float() + 1.0).mean().detach(),
        "teacher_depth_similarity_mean": similarity.float().mean().detach(),
        "teacher_depth_step1_stable_rate": stable[:, 0].float().mean().detach(),
    }
    if return_diagnostics:
        return halt, cont, diagnostics
    return halt, cont


def _teacher_depth_targets_from_stable(
    stable: torch.Tensor,
    *,
    anchor: torch.Tensor,
    min_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, steps = stable.shape
    min_step = max(1, int(min_step))
    if min_step > 1:
        stable = stable.clone()
        stable[:, : min(min_step - 1, steps)] = False
    step_index = torch.arange(steps, device=anchor.device).view(1, steps).expand(b, steps)
    first_stable = torch.where(
        stable.to(device=anchor.device),
        step_index,
        step_index.new_full((b, steps), steps),
    ).min(dim=1).values
    halt = (step_index >= first_stable[:, None]).to(device=anchor.device, dtype=anchor.dtype)
    cont = 1.0 - halt
    return halt, cont, first_stable


def qtrm_smoke_loss(
    model,
    input_ids: torch.Tensor,
    *,
    jepa_weight: float = 0.1,
    aux_weight: float = 1.0,
    core_halt_weight: float = 0.0,
    core_halt_auto_targets: bool = False,
    core_halt_target_mode: str = "exact",
    core_halt_donor_kl_threshold: Optional[float] = None,
    core_halt_teacher_depth_threshold: float = 0.995,
    core_halt_teacher_depth_logit_kl_threshold: float = 0.05,
    core_halt_teacher_depth_min_step: int = 1,
    student_lm_weight: float = 0.0,
    donor_kl_weight: float = 0.0,
    donor_kl_beta: float = 0.0,
    donor_kl_temperature: float = 1.0,
    repeat_unlikelihood_weight: float = 0.0,
    greedy_token_margin_weight: float = 0.0,
    greedy_token_margin: float = 0.0,
    greedy_token_margin_only_donor_errors: bool = False,
    donor_correct_margin_weight: float = 0.0,
    donor_correct_margin: float = 0.0,
    preference_weight: float = 0.0,
    preference_beta: float = 2.0,
    preference_margin: float = 0.0,
    workspace_contrastive_weight: float = 0.0,
    workspace_contrastive_beta: float = 2.0,
    workspace_contrastive_margin: float = 0.0,
    logical_evidence_weight: float = 0.0,
    causal_evidence_gate_weight: float = 0.0,
    core_world_model_weight: float = 0.0,
    generation_verifier_weight: float = 0.0,
    evidence_span_reader_weight: float = 0.0,
    evidence_span_no_answer_span_suppression_weight: float = 0.0,
    answer_decision_weight: float = 0.0,
    answer_residual_governor_weight: float = 0.0,
    canonical_causal_weight: float = 0.0,
    canonical_causal_beta: float = 2.0,
    canonical_causal_margin: float = 0.0,
    canonical_causal_ablation_modes: Optional[list[str]] = None,
    action_policy_weight: float = 0.0,
    controller_signal_weight: float = 0.0,
    **kwargs,
):
    lm_weight = float(kwargs.pop("lm_weight", 1.0))
    labels = kwargs.get("labels")
    core_halt_targets = kwargs.get("core_halt_targets")
    core_continue_targets = kwargs.get("core_continue_targets")
    verifier_passed = kwargs.get("verifier_passed")
    preference_rejected_input_ids = kwargs.get("preference_rejected_input_ids")
    preference_rejected_labels = kwargs.get("preference_rejected_labels")
    preference_rejected_attention_mask = kwargs.get("preference_rejected_attention_mask")
    preference_sample_weight = kwargs.get("preference_sample_weight")
    workspace_counterfactual_text_states = kwargs.get("workspace_counterfactual_text_states")
    workspace_counterfactual_attention_mask = kwargs.get("workspace_counterfactual_attention_mask")
    logical_support_target = kwargs.get("logical_support_target")
    logical_refute_target = kwargs.get("logical_refute_target")
    logical_missing_target = kwargs.get("logical_missing_target")
    causal_evidence_target = kwargs.get("causal_evidence_target")
    generation_verifier_repeat_target = kwargs.get("generation_verifier_repeat_target")
    generation_verifier_stop_target = kwargs.get("generation_verifier_stop_target")
    generation_verifier_quality_target = kwargs.get("generation_verifier_quality_target")
    generation_verifier_sample_weight = kwargs.get("generation_verifier_sample_weight")
    evidence_span_start_target = kwargs.get("evidence_span_start_target")
    evidence_span_end_target = kwargs.get("evidence_span_end_target")
    evidence_span_no_answer_target = kwargs.get("evidence_span_no_answer_target")
    evidence_span_sample_weight = kwargs.get("evidence_span_sample_weight")
    answer_decision_target = kwargs.get("answer_decision_target")
    answer_decision_sample_weight = kwargs.get("answer_decision_sample_weight")
    action_targets = kwargs.get("action_targets")
    action_sample_weight = kwargs.get("action_sample_weight")
    controller_signal_target = kwargs.get("controller_signal_target")
    controller_signal_sample_weight = kwargs.get("controller_signal_sample_weight")
    core_halt_target_diagnostics = {}
    private_keys = {
        "labels",
        "core_halt_targets",
        "core_continue_targets",
        "verifier_passed",
        "preference_rejected_input_ids",
        "preference_rejected_labels",
        "preference_rejected_attention_mask",
        "preference_rejected_text_states",
        "preference_rejected_donor_logits",
        "preference_sample_weight",
        "workspace_counterfactual_text_states",
        "workspace_counterfactual_attention_mask",
        "logical_support_target",
        "logical_refute_target",
        "logical_missing_target",
        "causal_evidence_target",
        "generation_verifier_repeat_target",
        "generation_verifier_stop_target",
        "generation_verifier_quality_target",
        "generation_verifier_sample_weight",
        "evidence_span_start_target",
        "evidence_span_end_target",
        "evidence_span_no_answer_target",
        "evidence_span_sample_weight",
        "answer_decision_target",
        "answer_decision_sample_weight",
        "action_targets",
        "action_sample_weight",
        "controller_signal_target",
        "controller_signal_sample_weight",
    }
    model_kwargs = {k: v for k, v in kwargs.items() if k not in private_keys}
    if core_halt_auto_targets and core_halt_target_mode == "teacher_depth":
        model_kwargs.setdefault("enable_core_halt", False)
        model_kwargs.setdefault("return_core_depth_logits", True)
    outputs = model(input_ids=input_ids, **model_kwargs)
    offset = outputs["logits"].shape[1] - input_ids.shape[1]
    lm = next_token_lm_loss(
        outputs["logits"],
        input_ids,
        offset=offset,
        attention_mask=model_kwargs.get("attention_mask"),
        labels=labels,
    )
    student_lm = next_token_lm_loss(
        outputs.get("qtrm_logits", outputs["logits"]),
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
    core_world_model = jepa_world_model_loss(
        outputs.get("core_world_model_pred", outputs["logits"].new_empty((input_ids.shape[0], 0, 1))),
        outputs.get("core_world_model_target", outputs["logits"].new_empty((input_ids.shape[0], 0, 1))),
        outputs.get("core_world_model_mask"),
        latents=outputs.get("core_world_model_latents"),
        latent_mask=outputs.get("core_world_model_latent_mask"),
        sigreg=getattr(model, "core_world_model_sigreg", None),
        sigreg_weight=getattr(
            getattr(model, "cfg", None),
            "core_world_model_sigreg_weight",
            getattr(getattr(model, "cfg", None), "jepa_sigreg_weight", 0.09),
        ),
    )
    aux = controller_aux_loss(outputs)
    donor_kl = donor_logit_distillation_loss(
        outputs.get("qtrm_logits", outputs["logits"]),
        model_kwargs.get("donor_logits"),
        input_ids,
        offset=offset,
        attention_mask=model_kwargs.get("attention_mask"),
        labels=labels,
        beta=donor_kl_beta,
        temperature=donor_kl_temperature,
    )
    repeat_ul = repetition_unlikelihood_loss(
        outputs["logits"],
        input_ids,
        offset=offset,
        attention_mask=model_kwargs.get("attention_mask"),
        labels=labels,
    )
    greedy_margin, greedy_margin_metrics = greedy_token_margin_loss(
        outputs["logits"],
        input_ids,
        offset=offset,
        attention_mask=model_kwargs.get("attention_mask"),
        labels=labels,
        donor_logits=model_kwargs.get("donor_logits"),
        margin=greedy_token_margin,
        only_donor_errors=greedy_token_margin_only_donor_errors,
    )
    donor_correct_margin_value, donor_correct_margin_metrics = (
        donor_correct_margin_loss(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=model_kwargs.get("attention_mask"),
            labels=labels,
            donor_logits=model_kwargs.get("donor_logits"),
            margin=donor_correct_margin,
        )
    )
    generation_verifier, generation_verifier_probs = generation_verifier_loss(
        outputs,
        repeat_target=generation_verifier_repeat_target,
        stop_target=generation_verifier_stop_target,
        quality_target=generation_verifier_quality_target,
        sample_weight=generation_verifier_sample_weight,
    )
    evidence_span_reader, evidence_span_metrics = evidence_span_reader_loss(
        outputs,
        start_target=evidence_span_start_target,
        end_target=evidence_span_end_target,
        no_answer_target=evidence_span_no_answer_target,
        sample_weight=evidence_span_sample_weight,
        no_answer_span_suppression_weight=(
            evidence_span_no_answer_span_suppression_weight
        ),
    )
    answer_decision, answer_decision_metrics = answer_decision_loss(
        outputs,
        target=answer_decision_target,
        sample_weight=answer_decision_sample_weight,
    )
    answer_residual_governor, answer_residual_governor_metrics = (
        answer_residual_governor_loss(
            outputs,
            input_ids,
            labels=labels,
            donor_logits=model_kwargs.get("donor_logits"),
            attention_mask=model_kwargs.get("attention_mask"),
        )
    )
    action_policy, action_policy_metrics = action_policy_loss(
        outputs,
        target=action_targets,
        sample_weight=action_sample_weight,
    )
    controller_signal, controller_signal_metrics = controller_signal_prediction_loss(
        outputs,
        target=controller_signal_target,
        sample_weight=controller_signal_sample_weight,
    )
    preference = outputs["logits"].sum() * 0.0
    preference_chosen_logps = preference
    preference_rejected_logps = preference
    if preference_rejected_input_ids is not None and float(preference_weight) != 0.0:
        rejected_model_kwargs = dict(model_kwargs)
        if preference_rejected_attention_mask is not None:
            rejected_model_kwargs["attention_mask"] = preference_rejected_attention_mask
        else:
            rejected_model_kwargs.pop("attention_mask", None)
        if kwargs.get("preference_rejected_text_states") is not None:
            rejected_model_kwargs["text_states"] = kwargs["preference_rejected_text_states"]
        else:
            rejected_model_kwargs.pop("text_states", None)
        if kwargs.get("preference_rejected_donor_logits") is not None:
            rejected_model_kwargs["donor_logits"] = kwargs["preference_rejected_donor_logits"]
        else:
            rejected_model_kwargs.pop("donor_logits", None)
        rejected_outputs = model(input_ids=preference_rejected_input_ids, **rejected_model_kwargs)
        rejected_offset = rejected_outputs["logits"].shape[1] - preference_rejected_input_ids.shape[1]
        preference_chosen_logps = sequence_average_logprob(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=model_kwargs.get("attention_mask"),
            labels=labels,
        )
        preference_rejected_logps = sequence_average_logprob(
            rejected_outputs["logits"],
            preference_rejected_input_ids,
            offset=rejected_offset,
            attention_mask=preference_rejected_attention_mask,
            labels=preference_rejected_labels,
        )
        preference = simpo_margin_loss(
            preference_chosen_logps,
            preference_rejected_logps,
            beta=preference_beta,
            margin=preference_margin,
            sample_weight=preference_sample_weight,
        )
    workspace_contrastive = outputs["logits"].sum() * 0.0
    workspace_true_logps = workspace_contrastive
    workspace_counterfactual_logps = workspace_contrastive
    counterfactual_outputs = None
    if (
        workspace_counterfactual_text_states is not None
        and (
            float(workspace_contrastive_weight) != 0.0
            or float(logical_evidence_weight) != 0.0
            or float(causal_evidence_gate_weight) != 0.0
        )
    ):
        counterfactual_model_kwargs = dict(model_kwargs)
        counterfactual_model_kwargs["workspace_text_states"] = workspace_counterfactual_text_states
        if workspace_counterfactual_attention_mask is not None:
            counterfactual_model_kwargs["workspace_attention_mask"] = (
                workspace_counterfactual_attention_mask
            )
        else:
            counterfactual_model_kwargs.pop("workspace_attention_mask", None)
        counterfactual_outputs = model(input_ids=input_ids, **counterfactual_model_kwargs)
    if counterfactual_outputs is not None and float(workspace_contrastive_weight) != 0.0:
        counterfactual_offset = counterfactual_outputs["logits"].shape[1] - input_ids.shape[1]
        workspace_true_logps = sequence_average_logprob(
            outputs["logits"],
            input_ids,
            offset=offset,
            attention_mask=model_kwargs.get("attention_mask"),
            labels=labels,
        )
        workspace_counterfactual_logps = sequence_average_logprob(
            counterfactual_outputs["logits"],
            input_ids,
            offset=counterfactual_offset,
            attention_mask=model_kwargs.get("attention_mask"),
            labels=labels,
        )
        workspace_contrastive = simpo_margin_loss(
            workspace_true_logps,
            workspace_counterfactual_logps,
            beta=workspace_contrastive_beta,
            margin=workspace_contrastive_margin,
            sample_weight=preference_sample_weight,
        )
    logical_evidence = outputs["logits"].sum() * 0.0
    causal_evidence_gate = outputs["logits"].sum() * 0.0
    logical_support_prob = logical_evidence
    counterfactual_support_prob = logical_evidence
    evidence_gate_mean = causal_evidence_gate
    counterfactual_gate_mean = causal_evidence_gate
    if float(logical_evidence_weight) != 0.0:
        logical_evidence, logical_support_prob, counterfactual_support_prob = (
            logical_evidence_verifier_loss(
                outputs,
                support_target=logical_support_target,
                refute_target=logical_refute_target,
                missing_target=logical_missing_target,
                counterfactual_outputs=counterfactual_outputs,
            )
        )
    if float(causal_evidence_gate_weight) != 0.0:
        causal_evidence_gate, evidence_gate_mean, counterfactual_gate_mean = (
            causal_evidence_gate_loss(
                outputs,
                target=causal_evidence_target,
                counterfactual_outputs=counterfactual_outputs,
            )
        )
    canonical_causal = outputs["logits"].sum() * 0.0
    canonical_causal_metrics = {
        "canonical_causal_margin": canonical_causal,
        "canonical_causal_full_logp": canonical_causal,
        "canonical_causal_ablation_logp": canonical_causal,
    }
    if float(canonical_causal_weight) != 0.0:
        modes = list(
            canonical_causal_ablation_modes
            or ["core_off", "workspace_off", "evidence_bottleneck_off"]
        )
        full_answer_logits = outputs.get(
            "qtrm_residual_logits",
            outputs.get("qtrm_logits", outputs["logits"]),
        )
        full_logps = sequence_average_logprob(
            full_answer_logits,
            input_ids,
            offset=offset,
            attention_mask=model_kwargs.get("attention_mask"),
            labels=labels,
        )
        ablation_logps = []
        for mode in modes:
            ablation_kwargs = _canonical_causal_forward_kwargs(model_kwargs, mode)
            ablation_outputs = model(input_ids=input_ids, **ablation_kwargs)
            ablation_answer_logits = ablation_outputs.get(
                "qtrm_residual_logits",
                ablation_outputs.get("qtrm_logits", ablation_outputs["logits"]),
            )
            ablation_offset = ablation_answer_logits.shape[1] - input_ids.shape[1]
            ablation_logps.append(
                sequence_average_logprob(
                    ablation_answer_logits,
                    input_ids,
                    offset=ablation_offset,
                    attention_mask=ablation_kwargs.get("attention_mask"),
                    labels=labels,
                )
            )
        canonical_causal, canonical_causal_metrics = canonical_causal_ablation_loss(
            full_logps,
            ablation_logps,
            beta=canonical_causal_beta,
            margin=canonical_causal_margin,
        )
    if core_halt_targets is None and core_halt_auto_targets:
        if core_halt_target_mode == "exact":
            core_halt_targets, core_halt_target_diagnostics = infer_core_halt_targets(
                outputs,
                input_ids,
                labels=labels,
                attention_mask=model_kwargs.get("attention_mask"),
                offset=offset,
                verifier_passed=verifier_passed,
                donor_logits=model_kwargs.get("donor_logits"),
                donor_logits_scale=getattr(getattr(model, "cfg", None), "donor_logits_scale", 1.0),
                donor_kl_threshold=core_halt_donor_kl_threshold,
                return_diagnostics=True,
            )
            if core_continue_targets is None:
                core_continue_targets = 1.0 - core_halt_targets
        elif core_halt_target_mode == "teacher_depth":
            core_halt_targets, core_continue_targets, core_halt_target_diagnostics = (
                infer_core_halt_targets_from_teacher_depth(
                    outputs,
                    similarity_threshold=core_halt_teacher_depth_threshold,
                    logit_kl_threshold=core_halt_teacher_depth_logit_kl_threshold,
                    min_step=core_halt_teacher_depth_min_step,
                    return_diagnostics=True,
                )
            )
        else:
            raise ValueError(f"unknown core_halt_target_mode: {core_halt_target_mode}")
    core_halt = core_halt_loss(
        outputs,
        target_halt=core_halt_targets,
        target_continue=core_continue_targets,
    )
    loss = (
        lm_weight * lm
        + float(student_lm_weight) * student_lm
        + float(jepa_weight) * jepa
        + float(aux_weight) * aux
        + float(core_halt_weight) * core_halt
        + float(donor_kl_weight) * donor_kl
        + float(repeat_unlikelihood_weight) * repeat_ul
        + float(greedy_token_margin_weight) * greedy_margin
        + float(donor_correct_margin_weight) * donor_correct_margin_value
        + float(preference_weight) * preference
        + float(workspace_contrastive_weight) * workspace_contrastive
        + float(logical_evidence_weight) * logical_evidence
        + float(causal_evidence_gate_weight) * causal_evidence_gate
        + float(core_world_model_weight) * core_world_model
        + float(generation_verifier_weight) * generation_verifier
        + float(evidence_span_reader_weight) * evidence_span_reader
        + float(answer_decision_weight) * answer_decision
        + float(answer_residual_governor_weight) * answer_residual_governor
        + float(canonical_causal_weight) * canonical_causal
        + float(action_policy_weight) * action_policy
        + float(controller_signal_weight) * controller_signal
    )
    metrics = {
        "loss": loss.detach(),
        "lm": lm.detach(),
        "student_lm": student_lm.detach(),
        "jepa": jepa.detach(),
        "core_world_model": core_world_model.detach(),
        "aux": aux.detach(),
        "core_halt": core_halt.detach(),
        "donor_kl": donor_kl.detach(),
        "repeat_ul": repeat_ul.detach(),
        "greedy_token_margin": greedy_margin.detach(),
        "greedy_token_win_rate": greedy_margin_metrics["greedy_token_win_rate"],
        "greedy_token_active_rate": greedy_margin_metrics["greedy_token_active_rate"],
        "greedy_token_donor_error_rate": greedy_margin_metrics[
            "greedy_token_donor_error_rate"
        ],
        "donor_correct_margin": donor_correct_margin_value.detach(),
        "donor_correct_margin_win_rate": donor_correct_margin_metrics[
            "donor_correct_margin_win_rate"
        ],
        "donor_correct_margin_active_rate": donor_correct_margin_metrics[
            "donor_correct_margin_active_rate"
        ],
        "generation_verifier": generation_verifier.detach(),
        "generation_repeat_prob": generation_verifier_probs["repeat_prob"],
        "generation_stop_prob": generation_verifier_probs["stop_prob"],
        "generation_quality_prob": generation_verifier_probs["quality_prob"],
        "evidence_span_reader": evidence_span_reader.detach(),
        "evidence_span_start_acc": evidence_span_metrics["start_acc"],
        "evidence_span_end_acc": evidence_span_metrics["end_acc"],
        "evidence_span_no_answer_prob": evidence_span_metrics["no_answer_prob"],
        "evidence_span_no_answer_span_score": evidence_span_metrics[
            "no_answer_span_score"
        ],
        "answer_decision": answer_decision.detach(),
        "answer_decision_acc": answer_decision_metrics["answer_decision_acc"],
        "answer_decision_block_prob": answer_decision_metrics["answer_decision_block_prob"],
        "answer_residual_governor": answer_residual_governor.detach(),
        "answer_residual_governor_acc": answer_residual_governor_metrics[
            "answer_residual_governor_acc"
        ],
        "answer_residual_governor_open_rate": answer_residual_governor_metrics[
            "answer_residual_governor_open_rate"
        ],
        "answer_residual_governor_target_open_rate": (
            answer_residual_governor_metrics[
                "answer_residual_governor_target_open_rate"
            ]
        ),
        "action_policy": action_policy.detach(),
        "action_acc": action_policy_metrics["action_acc"],
        "controller_signal": controller_signal.detach(),
        "controller_signal_acc": controller_signal_metrics["controller_signal_acc"],
        "preference": preference.detach(),
        "preference_chosen_logp": preference_chosen_logps.mean().detach(),
        "preference_rejected_logp": preference_rejected_logps.mean().detach(),
        "preference_margin_logp": (preference_chosen_logps - preference_rejected_logps).mean().detach(),
        "workspace_contrastive": workspace_contrastive.detach(),
        "workspace_true_logp": workspace_true_logps.mean().detach(),
        "workspace_counterfactual_logp": workspace_counterfactual_logps.mean().detach(),
        "workspace_margin_logp": (
            workspace_true_logps - workspace_counterfactual_logps
        ).mean().detach(),
        "logical_evidence": logical_evidence.detach(),
        "logical_support_prob": logical_support_prob.mean().detach(),
        "counterfactual_support_prob": counterfactual_support_prob.mean().detach(),
        "causal_evidence_gate": causal_evidence_gate.detach(),
        "evidence_gate_mean": evidence_gate_mean.mean().detach(),
        "counterfactual_gate_mean": counterfactual_gate_mean.mean().detach(),
        "canonical_causal": canonical_causal.detach(),
        "canonical_causal_margin": canonical_causal_metrics[
            "canonical_causal_margin"
        ],
        "canonical_causal_full_logp": canonical_causal_metrics[
            "canonical_causal_full_logp"
        ],
        "canonical_causal_ablation_logp": canonical_causal_metrics[
            "canonical_causal_ablation_logp"
        ],
    }
    metrics.update(core_halt_target_diagnostics)
    return loss, metrics, outputs


def _target_like(
    logits: torch.Tensor,
    target: Optional[torch.Tensor],
    *,
    default: float,
) -> torch.Tensor:
    if target is None:
        return logits.new_full(logits.shape, float(default))
    return target.to(device=logits.device, dtype=logits.dtype).view_as(logits)


def _bce_head_loss(
    logits: torch.Tensor,
    target: Optional[torch.Tensor],
    *,
    default: float,
) -> torch.Tensor:
    target_tensor = _target_like(logits, target, default=default)
    return F.binary_cross_entropy_with_logits(logits.float(), target_tensor.float())


def logical_evidence_verifier_loss(
    outputs: dict[str, torch.Tensor],
    *,
    support_target: Optional[torch.Tensor] = None,
    refute_target: Optional[torch.Tensor] = None,
    missing_target: Optional[torch.Tensor] = None,
    counterfactual_outputs: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_logits = outputs.get("evidence_support_logits")
    if support_logits is None:
        anchor = outputs["logits"]
        zero = anchor.sum() * 0.0
        return zero, zero, zero

    support_loss = _bce_head_loss(support_logits, support_target, default=1.0)
    refute_loss = _bce_head_loss(outputs["evidence_refute_logits"], refute_target, default=0.0)
    missing_loss = _bce_head_loss(outputs["evidence_missing_logits"], missing_target, default=0.0)
    loss = support_loss + 0.5 * (refute_loss + missing_loss)
    support_prob = support_logits.float().sigmoid()
    counterfactual_support_prob = support_prob.new_zeros(support_prob.shape)

    if counterfactual_outputs is not None and counterfactual_outputs.get("evidence_support_logits") is not None:
        cf_support_logits = counterfactual_outputs["evidence_support_logits"]
        cf_refute_logits = counterfactual_outputs["evidence_refute_logits"]
        cf_missing_logits = counterfactual_outputs["evidence_missing_logits"]
        loss = loss + _bce_head_loss(cf_support_logits, None, default=0.0)
        loss = loss + 0.5 * _bce_head_loss(cf_refute_logits, None, default=1.0)
        loss = loss + 0.5 * _bce_head_loss(cf_missing_logits, None, default=1.0)
        counterfactual_support_prob = cf_support_logits.float().sigmoid()
    return loss, support_prob, counterfactual_support_prob


def causal_evidence_gate_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target: Optional[torch.Tensor] = None,
    counterfactual_outputs: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_logits = outputs.get("evidence_bottleneck_gate_logits")
    gate = outputs.get("evidence_bottleneck_gate")
    if gate_logits is None:
        anchor = outputs["logits"]
        zero = anchor.sum() * 0.0
        return zero, zero, zero

    loss = _bce_head_loss(gate_logits, target, default=1.0)
    gate_mean = gate.float() if gate is not None else gate_logits.float().sigmoid()
    counterfactual_gate = gate_mean.new_zeros(gate_mean.shape)
    if counterfactual_outputs is not None and counterfactual_outputs.get("evidence_bottleneck_gate_logits") is not None:
        cf_gate_logits = counterfactual_outputs["evidence_bottleneck_gate_logits"]
        cf_gate = counterfactual_outputs.get("evidence_bottleneck_gate")
        loss = loss + _bce_head_loss(cf_gate_logits, None, default=0.0)
        counterfactual_gate = cf_gate.float() if cf_gate is not None else cf_gate_logits.float().sigmoid()
    return loss, gate_mean, counterfactual_gate


def evidence_span_reader_loss(
    outputs: dict[str, torch.Tensor],
    *,
    start_target: Optional[torch.Tensor] = None,
    end_target: Optional[torch.Tensor] = None,
    no_answer_target: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    no_answer_span_suppression_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    start_logits = outputs.get("evidence_span_start_logits")
    end_logits = outputs.get("evidence_span_end_logits")
    no_answer_logits = outputs.get("evidence_span_no_answer_logits")
    anchor = outputs["logits"]
    zero = anchor.sum() * 0.0
    if start_logits is None or end_logits is None or no_answer_logits is None:
        return zero, {
            "start_acc": zero.detach(),
            "end_acc": zero.detach(),
            "no_answer_prob": zero.detach(),
            "no_answer_span_score": zero.detach(),
        }
    if start_logits.numel() == 0 or end_logits.numel() == 0:
        no_answer_prob = no_answer_logits.float().sigmoid().mean() if no_answer_logits.numel() else zero
        return zero, {
            "start_acc": zero.detach(),
            "end_acc": zero.detach(),
            "no_answer_prob": no_answer_prob.detach(),
            "no_answer_span_score": zero.detach(),
        }

    b = start_logits.shape[0]
    if start_target is None:
        start = torch.full((b,), -100, dtype=torch.long, device=start_logits.device)
    else:
        start = start_target.to(device=start_logits.device, dtype=torch.long).view(-1)
    if end_target is None:
        end = torch.full((b,), -100, dtype=torch.long, device=end_logits.device)
    else:
        end = end_target.to(device=end_logits.device, dtype=torch.long).view(-1)
    if sample_weight is None:
        weights = torch.ones((b,), dtype=start_logits.dtype, device=start_logits.device)
    else:
        weights = sample_weight.to(device=start_logits.device, dtype=start_logits.dtype).view(-1).clamp_min(0.0)

    valid_span = (start >= 0) & (end >= 0) & (weights > 0)
    span_loss = zero
    start_acc = zero
    end_acc = zero
    if valid_span.any():
        start_ce = F.cross_entropy(start_logits.float(), start, reduction="none", ignore_index=-100)
        end_ce = F.cross_entropy(end_logits.float(), end, reduction="none", ignore_index=-100)
        span_weights = weights.masked_fill(valid_span.logical_not(), 0.0)
        denom = span_weights.sum().clamp_min(1e-8)
        span_loss = ((start_ce + end_ce) * span_weights).sum() / denom
        start_acc = (
            (start_logits.argmax(dim=-1) == start).to(start_logits.dtype) * span_weights
        ).sum() / denom
        end_acc = (
            (end_logits.argmax(dim=-1) == end).to(end_logits.dtype) * span_weights
        ).sum() / denom

    if no_answer_target is None:
        no_answer = torch.zeros((b,), dtype=no_answer_logits.dtype, device=no_answer_logits.device)
    else:
        no_answer = no_answer_target.to(device=no_answer_logits.device, dtype=no_answer_logits.dtype).view(-1)
    no_answer_loss = F.binary_cross_entropy_with_logits(
        no_answer_logits.float(),
        no_answer.float(),
        reduction="none",
    )
    if weights.numel() == no_answer_loss.numel() and torch.any(weights > 0):
        no_answer_loss = (no_answer_loss * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        no_answer_loss = no_answer_loss.mean()
    no_answer_prob = no_answer_logits.float().sigmoid().mean()
    no_answer_span_score = zero
    span_suppression_loss = zero
    no_answer_mask = (no_answer >= 0.5) & (weights > 0)
    if no_answer_mask.any():
        max_start = start_logits.float().max(dim=-1).values
        max_end = end_logits.float().max(dim=-1).values
        best_span_score = torch.maximum(max_start, max_end)
        no_answer_weights = weights.masked_fill(no_answer_mask.logical_not(), 0.0)
        denom = no_answer_weights.sum().clamp_min(1e-8)
        no_answer_span_score = (
            best_span_score * no_answer_weights
        ).sum() / denom
        if float(no_answer_span_suppression_weight) != 0.0:
            span_suppression = F.softplus(best_span_score)
            span_suppression_loss = (
                span_suppression * no_answer_weights
            ).sum() / denom

    return (
        span_loss
        + no_answer_loss
        + float(no_answer_span_suppression_weight) * span_suppression_loss
    ), {
        "start_acc": start_acc.detach(),
        "end_acc": end_acc.detach(),
        "no_answer_prob": no_answer_prob.detach(),
        "no_answer_span_score": no_answer_span_score.detach(),
    }
