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


def _valid_next_token_mask(
    target_source: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    valid = target_source[:, 1:] != -100
    if attention_mask is not None:
        valid = valid & attention_mask[:, 1:].to(torch.bool)
    return valid


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
    **kwargs,
):
    labels = kwargs.get("labels")
    core_halt_targets = kwargs.get("core_halt_targets")
    core_continue_targets = kwargs.get("core_continue_targets")
    verifier_passed = kwargs.get("verifier_passed")
    core_halt_target_diagnostics = {}
    private_keys = {
        "labels",
        "core_halt_targets",
        "core_continue_targets",
        "verifier_passed",
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
        lm
        + float(jepa_weight) * jepa
        + float(aux_weight) * aux
        + float(core_halt_weight) * core_halt
    )
    metrics = {
        "loss": loss.detach(),
        "lm": lm.detach(),
        "jepa": jepa.detach(),
        "aux": aux.detach(),
        "core_halt": core_halt.detach(),
    }
    metrics.update(core_halt_target_diagnostics)
    return loss, metrics, outputs
