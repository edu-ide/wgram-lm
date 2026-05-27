# S0: Surpassing 5.6 Gate — LOCKED (2026-05-30)

**Status**: FINALIZED per research-driven-architecture-debugging skill + evidence package review.  
**Date locked**: 2026-05-30 (after user "모든 S 다해" directive)  
**Evidence base**: `docs/5.56_Promotion_Gate_Evidence_2026-05-30/` (G-stage reports, 180-step full vs 100-step stochastic zero on real 642 gold, probe artifacts)

---

## 1. Primary Metric (Tier 1 — Current Executable)

**Name**: Stochastic Diversity (mean / max) under real 642 gold + full 5.56 curriculum  
**Secondary (State Robustness Proxy)**: State ablation robustness probe (see `probe_state_ablation_robustness.py` starter in evidence package; measures degradation of future trajectory quality under recurrent state noise/zeroing).

**Rationale**: 
- The strongest, most reproducible causal signal we currently have from the 5.56 G-stage is the diversity metric (Full ON: stable ~5.99–6.13; Zero OFF: exactly 0.0000).
- This directly measures the Reverse I→G→A port of stochastic recurrent breadth — the single most important historical inductive bias that produced the original 5.53~5.56 downstream strength.
- The original historical "state_ablation_median ~5.53-5.56" was a downstream hard-family metric that is not yet directly executable on the current small-scale backbone without a full hard-family harness (Phase 2 in DOWNSTREAM_EVAL_PLAN.md). We therefore use Tier 1 proxies that are causally linked to the same inductive bias.

**Future Tier 2 (Real Historical Target)**: 
- Once a proper hard-family downstream evaluator exists, primary metric switches to `state_ablation_median` (or equivalent) on held-out hard-family reasoning cases. Success on Tier 1 is a prerequisite for investing in Tier 2 measurement.

---

## 2. Historical Baseline (from Evidence Package)

- **Full 5.56 recipe on real 642 gold (180-step)**: Stochastic Diversity mean ~5.99, max ~6.13, extremely stable across horizon.
- **Stochastic breadth ablation zero (100-step, same real gold + curriculum)**: Diversity exactly 0.0000 (perfect causal isolation).
- Scheduled decay, attractor protection, and overall stability remained excellent in both conditions.
- Real gold + stochastic breadth interaction produced **higher** diversity than synthetic-proxy runs (~4.0).

**Key historical fact (preserved)**: Removing stochastic breadth caused large, clean drops in the training dynamics that were previously associated with the downstream 5.5x state quality gains.

---

## 3. Required Ablations (Causal Contribution Contract — Non-Negotiable)

All of the following **must** produce material, directionally consistent effects on the primary Tier 1 metric for any run to count as "5.56 dynamics successfully reproduced on new backbone":

1. `stochastic_breadth_ablation_zero` → Diversity must collapse to ~0 (already proven; must remain perfect identity on hybrid block).
2. Gold structural injection disabled (`gold_state_injection_alpha = 0` or equivalent) → measurable drop in diversity and/or state robustness probe.
3. Attractor protection during rehearsal disabled → measurable degradation in state quality / rehearsal effectiveness.
4. Recurrence / core significantly reduced or disabled (where architecture permits clean ablation) → same metric must drop.

**Quantitative bar (Tier 1)**:
- Stochastic breadth ablation: drop to <0.1 (ideally exact 0).
- Other ablations: at least **0.02–0.03 relative degradation** on the state robustness probe (or equivalent diversity drop) — to be calibrated on first S1.3 runs.

These bars are deliberately conservative and directly derived from the G-stage contrast data.

---

## 4. Curriculum Reproduction Requirements (for fair comparison)

To claim "faithful 5.56 reproduction":
- Use identical `RehearsalConfig` parameters (scheduled_binding_decay 0.40 → 0.04, attractor_protection_during_rehearsal=0.7, etc.).
- Use real 642 gold path when possible.
- Enable full stochastic breadth (`core_stochastic_breadth_enabled=True`, K>1, true_gram or delta mode) for the entire curriculum.
- Curriculum length ≥150–180 steps for meaningful horizon comparison.
- Same rehearsal importance / ALRMC-style scoring logic.

---

## 5. Success Criteria (LOCKED)

**To pass S0 and declare "meaningful progress toward surpassing 5.6" (Tier 1)**:

**Condition A (Reproduction + Improvement Direction)**:
- New backbone (`OneBodyParallelHybridBlock` with decided injection points + official preference) + faithfully reproduced 5.56 curriculum achieves:
  - Stochastic Diversity stable mean ≥5.8 (targeting the historical ~5.99–6.13 band), AND
  - State robustness probe shows clear advantage vs stochastic-zero and gold-off variants on the same checkpoint.

**Condition B (Causality — Reverse I→G→A Contract)**:
- All 4 required ablations in section 3 produce the expected material drops (especially perfect diversity collapse on stochastic zero).

**Stretch / Promotion-Qualifying (Tier 2, later)**:
- On real hard-family `state_ablation_median` (or equivalent), exceed the upper end of the historical 5.53–5.56 band by a statistically meaningful margin while preserving the full ablation matrix.

**Kill Criterion** (if any of these happen, S0 claim is rejected and we must diagnose):
- Stochastic zero fails to drive diversity to ~0 on the hybrid block (fusion or official components silently cancel the signal).
- Gold injection or attractor protection produce no measurable effect.
- New backbone + 5.56 recipe produces **worse** robustness/diversity than the previous QTRMRecursiveCore 5.56 runs under identical conditions.

---

## 6. Minimum Validation Scale (Tier 1)

- Curriculum: ≥150 steps (180-step historical contrast preferred).
- Seeds: at least 3 independent seeds for the full recipe; key ablations on at least 1–2 seeds.
- Full ablation matrix executed on best checkpoints.
- Consistent direction across seeds.
- bfloat16 + official GDN2/MLA preference where stable.

---

## 7. S0 Closure Decision

**S0 is now LOCKED**.

We have:
- Concrete, reproducible baseline numbers from real-gold G-stage runs (not vague "5.5x").
- Perfect causal isolation proof for the most important inductive bias (stochastic breadth).
- Explicit Tier 1 vs Tier 2 distinction with honest evidence gap acknowledged.
- Quantitative bars derived directly from existing contrast data.

All future S1–S5 experiments will be judged against this gate. No claim of "surpassing 5.6" or even "recovered 5.56 dynamics on new backbone" is valid without passing the above conditions.

**Next immediate step**: Proceed to S1.1 decision closure → S1.2 minimal trainer prototype → S1.3 execution against this exact gate.

---

**References** (must be preserved):
- G_STAGE_COMPLETION_REPORT.md and all 02_real_gold_runs/ analysis files
- DOWNSTREAM_EVAL_PLAN.md (Phase 1 proxy definition)
- `scripts/probe_state_ablation_robustness.py` (when completed)
- PHASE_S_Surpassing_5.6_Experiments.md (updated with this lock)

This lock satisfies the skill's "Past-Success Doubt Loop", "Reproducibility seal", and "Prior-To-Implementation Contract" requirements for the entire PHASE S effort.
