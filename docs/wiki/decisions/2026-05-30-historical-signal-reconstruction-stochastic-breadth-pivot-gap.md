# Historical Signal Reconstruction Gate: Stochastic Recurrent Breadth (GRAM/PTRM Inductive Bias)

**Date**: 2026-05-30  
**Branch/Context**: feat/architecture-integration-2026-05 (post new-thought-structure pivot + Phase 0-3 mega restoration)  
**Trigger**: Updated research-driven-architecture-debugging skill (Pivot Safety + Reverse I→G→A section) + explicit user request to follow the skill after diagnosing the gap.  
**Status**: Gate executed. Reverse I→G→A initiated (Improvement stage pending).

---

## 1. Signal Identification

**Signal Name**: Adaptive Rehearsal 5.53~5.56 (gold recipe on 642/637 base) + related early GRAM/PTRM recurrent performance.

**Peak Evidence**:
- state_ablation_median ~5.53-5.56 on compatible small checkpoints.
- Strong causal drop when rehearsal + scheduled binding decay + attractor protection were ablated.
- Multiple "accepted" internal notes and early Stage56/58 PTRM results.

**Minimal Set of Inductive Biases Implicated** (from historical analysis):
1. Scheduled external/scheduled binding decay (0.40 → 0.04).
2. Hard-family focus + preference mix in rehearsal.
3. ALRMC-style importance protection on gold Phase-1 base.
4. **Stochastic recurrent breadth during training** (GRAM/PTRM-style): training-time generation of multiple noisy high-level trajectories via learned prior/posterior sampling + noise injection into z_h.

This reconstruction focuses on bias #4, which was the one most completely lost after the pivot.

---

## 2. What the Bias Actually Was (Precise Mechanism)

**Location in Legacy Code** (the version that produced the signal):
- `src/qtrm_mm/state_transition_core.py`
  - `stochastic_transition_mode = "true_gram"` or `"delta"`
  - `_apply_true_gram_transition(...)` (primary path for strong signals): 
    - Computes learned prior (and optional posterior) Gaussian over high-level state.
    - During training: `z_h = mu + std * eps` (direct replacement, not additive delta).
    - Posterior guidance using answer labels.
    - KL divergence term.
  - `_apply_stochastic_high_level_guidance(...)`: smaller additive noise version.
- Activated inside the dual-state recurrence loop (after or instead of the shared_core update on z_h).
- Exposed via `QwenBackboneWithStateTransition` and trainers 510-523 series.

**Effect on Training Dynamics**:
- Forces the recurrent high-level latent (the "thought" state) to maintain trajectory diversity *throughout training*, not just at inference.
- Prevents premature collapse of the latent manifold.
- Creates natural exploration + selection pressure inside the recurrence itself.

**Distinction from Current Approximations**:
- Current `core_multi_trajectory_enabled` + `MultiTrajectoryScorer` (added in Mega C) is **post-hoc** scoring/aggregation over past memory_buffer states. It does not inject stochasticity into the recurrence during the forward pass of training. It is a weaker, different bias.

---

## 3. Current Status After the Pivot (2026-05-26 new thought structure + subsequent work)

**Primary Architecture Today**: `QTRMRecursiveCore` in `src/qtrm_mm/core.py` (fast_stack + slow_stack BlockStack, gated workspaces, monotonic attractor, provenance register, adaptive rehearsal scaffolding, learned slow tier, gold structural bias, equation binding, etc.).

**Stochastic Breadth Status**: Completely absent from the recurrent forward path.
- No Gaussian prior/posterior networks on z_h.
- No training-time noise injection into the high-level state.
- No `stochastic_transition_mode`, `stochastic_high_level_guidance`, or equivalent.
- The entire `state_transition_core.py` machinery is isolated in a legacy parallel path (never called by the current core).

**component_registry.py Status** (pre-audit):
- `"state_transition_core"` listed as `PROMOTED` with note "Reusable GRAM/PTRM-style state transition core family."
- This was technically true for the *library code*, but misleading for the *active architecture*. The bias was not active in the One-Body path being trained.

**SSOT Status**:
- `internal-multitrajectory-answer-attractor-ssot.md` explicitly lists "GRAM/PTRM stochastic breadth off (K=1 vs K>1)" as a **mandatory** ablation for any IMTA promotion claim.
- The ablation cannot be executed on the current primary code. This is a direct violation of the SSOT promotion gate.

**Restoration/Mega Work Performed So Far**:
- Extensive gold state injection, structural bias into ALRMC/memory/slow-tier/attractor, adaptive rehearsal 5.56 proxies, Stage102Z provenance, workspaces, etc.
- All valuable, but none restored the training-time stochastic trajectory generation bias.

**Conclusion of Reconstruction**:
The 5.53~5.56 signal was a *composite*. Significant parts of the recipe have been ported or proxied. The stochastic recurrent breadth component was silently dropped during the pivot (intentionally isolated for checkpoint compatibility, never followed by a Reverse I→G→A decision to port or discard).

---

## 4. Reverse I→G→A Decision Record

**Decision**: Initiate Reverse I→G→A for this bias (do not treat as automatically discarded).

**Rationale**:
- The bias had clear historical causal contribution in combination with other accepted elements.
- The current post-hoc multi-trajectory approximation is not equivalent (different timing and strength of pressure on the latent during training).
- The SSOT still treats stochastic breadth as a required axis. We cannot honestly claim progress on the SSOT while the required ablation is impossible to run.

**Next Steps (Reverse I→G→A Stages)**:

**Improvement (I) Stage — Immediate**:
- Define narrow contract for a One-Body-compliant version:
  - Small learned prior (and optional posterior) head(s) that can be optionally applied to z_h at chosen recurrence steps (controlled by `core_stochastic_breadth_enabled` + `core_stochastic_mode` + `core_stochastic_scale` + `core_stochastic_apply_every_n`).
  - Support for both "delta" (additive) and "replace" (true_gram style) modes.
  - Clean `core_stochastic_breadth_ablation_zero` that completely disables sampling + noise (returns identity behavior).
  - KL term exposed as optional auxiliary loss when posterior guidance is used.
  - Must preserve One-Body: final answer still comes only through normal LM head from the (possibly noised) recurrent state. No side organs.
- Implement minimal version inside `QTRMRecursiveCore` (new small modules or methods, wired in forward after slow-tier / before workspace broadcast or at configurable points).
- Run narrow gate: small curriculum + direct comparison of stochastic_breadth_on vs ablation_zero on a 5.56-compatible proxy setup. Measure effect on state coherence, attractor behavior, and downstream answer margin.

**Generalization (G) Stage**:
- Multi-seed survival.
- Composition with the already-promoted mechanisms (workspaces + attractor + provenance + gold rehearsal).
- Language non-regression.
- Broader family stress.

**Architecture-ization (A) Stage**:
- Full wiring with config flags + carry support if needed.
- Update `component_registry.py` (change state_transition_core note + add new entry for the native One-Body port if it graduates).
- Update SSOTs with the new ablation contract.
- Add to Inductive Bias Map as "ported".

**Hard Constraints**:
- Must survive `core_stochastic_breadth_ablation_zero`.
- Must not break existing gold structural integration or adaptive rehearsal paths.
- Must remain fully ablatable and documented.

**Risk / Deferral Option**:
If the I-stage narrow gate shows weak or negative effect in the new core (different recurrence dynamics than the old SharedReasoningCore), we may document "deliberately not ported — different inductive regime" with evidence. That decision must be explicit and recorded.

---

## 5. Immediate Action Items (from this gate)

1. (Done) Created `docs/wiki/architecture/inductive-bias-map.md` with this bias as the first entry.
2. (In progress) Strengthen `component_registry.py` entry for `state_transition_core`.
3. Produce concrete Improvement-stage implementation plan + diff skeleton for the port (next session).
4. Update 2026-05-28 ablation wiki with link to this reconstruction.
5. When user issues high-momentum "mega" directives in future, explicitly reference this gate and insert the required Reverse step.

---

**Linked Skill Sections**:
- research-driven-architecture-debugging → "Pivot Safety, Historical Inductive Bias Preservation, and Reverse I→G→A"
- research-driven-architecture-debugging → "Improvement → Generalization → Architecture-ization (I→G→A) Promotion Loop" (the forward direction)

**This document is the required output of the Historical Signal Reconstruction Gate for the stochastic breadth case.**

No further major "mega" architectural claims on the current core should be made until the Reverse I→G→A for this bias is either completed or explicitly closed with recorded rationale.