# Reverse I→G→A Plan: One-Body Port of Stochastic Recurrent Breadth

**Linked Reconstruction**: 2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md
**Inductive Bias Map Entry**: stochastic recurrent breadth
**Current Date**: 2026-05-30
**Owner**: research-driven-architecture-debugging process

---

## Goal

Restore the training-time stochastic recurrent breadth inductive bias (the missing piece from the 5.53~5.56 era) into the current canonical `QTRMRecursiveCore` in a clean, One-Body-compliant, fully ablatable form — or explicitly decide not to, with evidence.

---

## Stage I: Improvement (Narrow Contract + Minimal Causal Test)

### Narrow Contract Definition (Must Be Closed Before Any Generalization Claim)

**What we are porting**:
- The core effect: during training recurrence, occasionally (or at configurable points) sample a small perturbation to the high-level state z_h from a learned distribution, instead of pure deterministic update.
- Two modes (both historically useful):
  - "delta" mode: small additive stochastic delta (cheaper, more stable).
  - "replace" / true_gram mode: replace z_h with sample from prior (or posterior) — stronger exploration pressure.

**Minimal Required Surface (for the narrow gate)**:
- Config flags (in `QTRMConfig`):
  - `core_stochastic_breadth_enabled: bool = False`
  - `core_stochastic_mode: str = "delta"`  # or "true_gram"
  - `core_stochastic_scale: float = 0.05`
  - `core_stochastic_high_level_min_std`, `max_std`
  - `core_stochastic_apply_prob: float = 0.3` (or every_n_steps)
  - `core_stochastic_posterior_guidance: bool = False` (for later)
- Ablation: `core_stochastic_breadth_ablation_zero: bool` — when True, the mechanism must become a pure no-op (exact same computation graph and numbers as disabled).
- Implementation location: inside `QTRMRecursiveCore` (new small heads or methods, called from forward at safe points — after slow-tier / memory fusion, before or after attractor pressure, before workspace broadcast).
- One-Body hard requirement: the noised (or original) z_h must still flow only through the normal path to the LM head. No side answer channels.

**Success Criteria for I-Stage (Narrow Gate)**:
- Implement the minimal version.
- Run a controlled proxy (small curriculum, 642-gold style or compatible synthetic) with stochastic_breadth_on vs ablation_zero.
- Show measurable difference in at least one of:
  - Recurrent state trajectory diversity (cosine distance between parallel unrolls, or norm of injected noise).
  - Attractor behavior or long-horizon coherence under the new dynamics.
  - Downstream answer margin on hard families (even if small).
- The ablation_zero version must be numerically identical to a clean "breadth disabled" run.
- Full documentation + ablation table attached to the Inductive Bias Map entry.

**Hard Rejects for I-Stage**:
- "We added a small noise head" without the ablation_zero actually forcing identity behavior.
- Using the old `state_transition_core` machinery directly (violates One-Body + new core contract).
- Claiming success from post-hoc scoring only.

---

### Proposed Minimal Implementation Sketch (One-Body Compliant)

Inside `QTRMRecursiveCore.__init__` (when `core_stochastic_breadth_enabled`):

```python
self.stochastic_breadth_prior = nn.Sequential(
    RMSNorm(cfg.d_model * 2),
    nn.Linear(cfg.d_model * 2, cfg.d_model * 2),
    nn.GELU(),
    nn.Linear(cfg.d_model * 2, cfg.d_model * 2),
)
# similar for posterior if enabled
```

In the recurrent step (after memory/ALRMC/slow-tier fusion, at a controlled point):

```python
if self.cfg.core_stochastic_breadth_enabled and not self.cfg.core_stochastic_breadth_ablation_zero:
    if random or step condition:
        ctx = pooled or memory_signal or z_h.mean(1)
        prior_input = torch.cat([z_h_pooled, ctx], dim=-1)
        mu, raw_std = self.stochastic_breadth_prior(...).chunk(2, -1)
        std = softplus(raw_std).clamp(min=..., max=...)
        if self.training:
            eps = torch.randn_like(std)
            noise = (mu + std * eps) * self.cfg.core_stochastic_scale
        else:
            noise = mu * self.cfg.core_stochastic_scale
        z_h = z_h + noise.unsqueeze(1)   # or replace in true_gram mode
```

Add the corresponding loss term (KL) when posterior guidance is active, exposed cleanly.

All of this must be easily zeroed by the ablation flag (early return of zeros or identity).

---

## Stage G + A (High Level)

- After narrow gate passes: multi-seed + composition with workspaces + attractor + provenance + gold rehearsal.
- Full flag surface in config + carry if needed.
- Update registry (new or updated entry with `active_in_primary_onebody_path=true`).
- Update SSOTs with the exact new ablation contract.
- Mark the old legacy entry more clearly as "historical reference implementation".

---

## Immediate Next Work (after this plan is reviewed)

1. Add the config flags to `src/wgram_lm/config.py` (with clear comments referencing this reconstruction).
2. Implement the minimal heads + wiring + ablation_zero inside `core.py`.
3. Write the small diagnostic runner or extend an existing Phase0/1 script to run the narrow gate.
4. Produce the first ablation table and attach it to the Inductive Bias Map + this decision record.

This plan follows the strict I→G→A contract in both directions (forward for the new port + the Reverse obligation created by the pivot).

**Do not proceed to large-scale training or "mega" claims on the current core until the I-stage narrow gate for this bias is executed or this Reverse track is explicitly closed.**