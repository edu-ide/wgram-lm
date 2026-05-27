# G-Stage Synthesis Report — 5.56 Curriculum + Stochastic Breadth (2026-05-30)

**Status**: Post-180-step + partial matrix execution phase

## Executive Summary (Efficient View)

After extensive execution of real 642 gold runs (50-step, 180-step, and matrix variants), the following has been **actually demonstrated**:

### Strong, Reproducible Evidence
1. **Stochastic Breadth is Causally Distinct and Contributes Positively**
   - When ablated (stoch_zero), diversity drops from ~6.0–6.5 to exactly 0.0 under identical real-gold + curriculum conditions.
   - This holds across multiple independent runs.
   - Real gold path attempts consistently produce higher diversity than pure synthetic runs when breadth is enabled.

2. **The 5.56 Curriculum Dynamics are Stable at Scale**
   - Scheduled decay (0.40 → ~0.04) behaves cleanly over 180 steps.
   - Attractor protection remains effective.
   - Removing stochastic breadth does not destabilize the other components (decay and protection curves remain nearly identical).

### What Has Not Been Demonstrated (and may not be demonstrable in current setup)
- Meaningful improvement in **state robustness** after ablation (the original spirit of the 5.5x signal).
- Current probe (even after v1 metric + v2 structured inputs) returns near-zero signal on all variants. This is a limitation of the small synthetic training regime, not lack of execution.

## What "G-Stage" Has Actually Achieved Here

**G-stage success criteria (training dynamics + causal evidence side)**: **Partially but meaningfully achieved.**

We have:
- Long-horizon stability data (180 steps).
- Clean ablation contrast on real gold data.
- Evidence of positive interaction between real gold structural bias and stochastic breadth.

**G-stage success criteria (recovering original downstream 5.5x robustness)**: **Not achieved, and current setup is poorly suited for it.**

## Efficient Recommendation (No More Probe Grinding)

Continuing to iterate the current `probe_state_ablation_robustness.py` on this small synthetic setup is low-leverage.

**High-efficiency path forward**:

1. **Freeze the probe work** at current state (v2) with clear documentation of its limitations.
2. **Synthesize and claim what we actually proved** in the training dynamics + causal ablation domain (the two points above).
3. **Decide on one of two real branches**:
   - **Branch A (Efficient & Honest)**: Declare G-stage complete for the *training dynamics* portion. Document limitations on downstream robustness. Move to A-stage decision or re-scope.
   - **Branch B (High Cost)**: Accept that meaningful state robustness measurement requires leaving the current small synthetic regime (larger model, real structured data, or actual hard-family eval harness).

## Artifacts Available for Decision

- Causal contrast document: `02_real_gold_runs/Gstage_Causal_Contrast_180vs100.md`
- Full matrix (partial): `local_556_real642_full_gstage_matrix_fixed_20260527_1256`
- All previous runs and analyses consolidated in this package.

**Bottom line**: We have real, non-trivial evidence on the dynamics side. We do not have (and are unlikely to get with current tools) the original-style downstream robustness signal.

Stop polishing the probe. Decide which branch (A or B) to take.

---

## Branch Decision (2026-05-30)

**User Decision**: Branch B

**Meaning**: We will not stop at the current small synthetic regime. We accept the higher cost to attempt measuring real state robustness / recovering signal closer to the original 5.5x.

This means shifting resources toward:
- Larger model capacity
- More structured / meaningful data
- Potentially porting the best curriculum logic into a regime where proper downstream evaluation (hard-family, state ablation on real tasks) is feasible.

The previous G-stage work (training dynamics + causal evidence on small scale) will be treated as solid foundation, not the final claim.

**Immediate implication**: Probe iteration on current small synthetic setup is deprioritized. Focus moves to defining the minimal viable "Branch B" setup.
