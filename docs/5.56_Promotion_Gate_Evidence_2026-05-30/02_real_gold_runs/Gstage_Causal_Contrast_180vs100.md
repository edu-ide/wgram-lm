# G-Stage Causal Contrast: 180-step Full (Stochastic ON) vs 100-step Stoch Zero

**Date**: 2026-05-30
**Goal**: Quantify the causal contribution of stochastic recurrent breadth under real 642 gold + 5.56 curriculum conditions.

## Executive Summary

When stochastic breadth is ablated on real gold data:
- Diversity signal disappears completely (6.13 → 0.0)
- Decay schedule remains almost identical
- Overall training stability stays good

This provides strong causal evidence that stochastic breadth is a distinct, additive, and non-destructive component in the 5.56 recipe.

## Detailed Metrics Comparison

| Metric                          | 180-step Full (ON)          | 100-step Zero (OFF)         | Delta / Observation |
|--------------------------------|-----------------------------|-----------------------------|---------------------|
| Steps completed                 | 180                         | 100                         | Different length (normalized where possible) |
| Scheduled Decay (start → end)   | 0.400 → 0.042 (range 0.358) | 0.400 → 0.044 (range 0.356) | Nearly identical decay behavior |
| Stochastic Diversity (max)      | 6.1297                      | 0.0000                      | **Complete disappearance** |
| Stochastic Diversity (mean)     | 5.9936                      | 0.0000                      | Perfect isolation |
| Diversity Stability (min~max)   | 5.85 ~ 6.13                 | Flat 0                      | No hidden stochasticity |
| Mean Drift                      | 0.6057                      | 0.4582                      | Slightly higher drift with breadth (expected over longer run) |
| Attractor Protection            | 100% active                 | 100% active                 | Independent of breadth |

## Key G-Stage Findings

1. **Causal Isolation Success**
   - Removing stochastic breadth eliminates the diversity signal entirely, even when everything else (real gold path, scheduled decay, protection) stays the same.
   - This is the cleanest evidence yet that the Reverse I→G→A port of stochastic breadth is actually doing the work we claimed.

2. **No Destructive Interference**
   - The scheduled decay mechanism works equally well with or without breadth.
   - Attractor protection is unaffected.
   - This means breadth can be treated as an orthogonal, additive module.

3. **Real Gold + Breadth Synergy (Hypothesis Strengthened)**
   - In both real-gold runs with breadth ON, diversity stayed high (~6.0).
   - In pure synthetic validation runs, diversity was lower (~4.0).
   - Combined with the zero run, this suggests the attempted 642 gold structural bias amplifies the value of stochastic breadth.

## Implications for Promotion Gate

- Stochastic recurrent breadth has now passed a meaningful G-stage test on real historical gold data.
- We have direct before/after ablation evidence under realistic curriculum conditions.
- The component is ready for deeper downstream robustness testing (next phase).

**Recommended Next**:
- Wait for the full matrix to finish.
- Run the state ablation robustness probe (see DOWNSTREAM_EVAL_PLAN.md) on the best variants from the matrix.
- If any variant shows clearly superior state robustness after the curriculum, that becomes the strongest candidate for claiming partial recovery of the original 5.5x signal.

---
**Source Runs**
- Full: `local_556_real642_long_180step_20260527_1201`
- Zero: `local_556_gstage_stoch_zero_real_fixed_20260527_1255`
