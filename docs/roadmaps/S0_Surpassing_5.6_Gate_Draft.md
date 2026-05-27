# S0: Surpassing 5.6 Gate — Refined Draft (2026-05-30)

**Status**: Refined based on available evidence. Ready for user review and finalization.

## 1. Primary Metric (Tier 1 - Current Backbone)

**Name**: State Robustness under Ablation (Proxy for historical `state_ablation_median`)

**Definition** (from `DOWNSTREAM_EVAL_PLAN.md`):
- After running the curriculum, measure how much ablating or noising the recurrent state (especially high-level z_h) degrades future trajectory quality / next-step prediction accuracy on synthetic reasoning sequences.
- Compute relative degradation (with vs without ablation).
- Higher robustness (lower degradation under ablation) = better state quality.

This is the best currently executable proxy on the existing small-scale backbone.

**Future Tier 2 (Real Goal)**:
- Once a proper hard-family downstream harness exists on the current (or larger) backbone, switch the primary metric to the original `state_ablation_median` on held-out hard-family cases.

## 2. Historical Baseline

From the 5.53~5.56 gold runs (Inductive Bias Map + evidence package):
- The strongest signal was achieved using the **full 5.56 Adaptive Rehearsal gold recipe** on 642 gold.
- Reported performance level: **state_ablation_median in the ~5.53 ~ 5.56 range** on the historical hard-family evaluation.
- Critical causal ingredients that produced the signal:
  - 642 gold structural bias (attractor-baked starting state)
  - Scheduled external binding decay (0.40 → 0.04)
  - Attractor protection during rehearsal (strength 0.7)
  - Stochastic recurrent breadth (K>1 noisy trajectories) applied throughout the curriculum

**Known fact**: Removing stochastic breadth or during-rehearsal attractor protection caused **large drops** in the final state_ablation_median.

**Current limitation**: Exact numerical best value on the original hard-family eval is not precisely recorded in the current evidence package (many downstream evals were marked as future work). We will treat the ~5.53-5.56 range as the reference band until a concrete number is recovered or re-measured.

## 3. Required Ablations (Causal Contribution Test)

To claim any improvement is due to the new backbone + 5.56 dynamics (not artifacts), the following ablations **must** show material drops on the primary metric:

1. `stochastic_breadth_ablation_zero` (core_stochastic_breadth_ablation_zero = True)
2. Gold structural injection disabled (`gold_state_injection_alpha = 0`)
3. Attractor protection during rehearsal disabled (`attractor_protection_during_rehearsal = 0`)
4. Recurrence depth / core significantly reduced (where the architecture allows clean ablation)

**Quantitative bar** (to be finalized in S0):
- Each ablation must cause at least **0.02~0.03 drop** in the primary robustness metric (to be calibrated during S1).

## 4. Curriculum Reproduction Requirements (for fair comparison)

To count as "reproducing the 5.56 recipe":

- Use the same `RehearsalConfig` parameters as the historical gold recipe (scheduled_binding_decay_start=0.40, end=0.04, attractor_protection_during_rehearsal=0.7, etc.)
- Use real 642 gold injection when possible (via `--gold_path`)
- Enable full stochastic breadth (`core_stochastic_breadth_enabled=True`) throughout the entire curriculum
- Match curriculum length (target: 150–200+ steps for long-horizon comparison)
- Use the same rehearsal importance mechanism and gold injection timing

## 5. Success Criteria (Proposed — Tier 1 Proxy)

To pass S0 and claim "meaningful progress toward surpassing 5.6":

**Condition A (Improvement)**:
- New backbone + faithfully reproduced 5.56 curriculum achieves **clearly better state robustness** than:
  - The same curriculum with stochastic breadth disabled, and
  - Historical small-scale 5.56 runs on the old backbone (using the proxy metric)

**Condition B (Causality)**:
- All required ablations in section 3 must produce material drops (≥0.02–0.03 on the proxy metric).

**Stretch Goal for later claiming "surpassed 5.6"**:
- On Tier 2 (real hard-family state_ablation_median), exceed the upper end of the historical ~5.53-5.56 band by a statistically meaningful margin while preserving all ablations.

## 6. Minimum Validation Scale (Tier 1)

- Curriculum length: ≥ 150 steps
- At least 3–5 independent seeds
- Full ablation matrix executed on the best checkpoints
- Consistent results across seeds

## Open Items (to be closed during S0 finalization)

1. Exact historical best proxy or downstream number we will treat as the hard baseline.
2. Precise numerical threshold for "material drop" in ablations (will be calibrated in early S1 runs).
3. Decision on when to move from Tier 1 (proxy) to Tier 2 (real hard-family eval).

---

**Next micro-step for S0**:
Gather the most precise available numbers from the 5.56 evidence package and past trainer logs to replace the "~5.53-5.56 range" with the best concrete reference we can establish.

Once that is done, S0 can be declared closed and we move to S1.