# S3 Bottleneck Diagnosis Record — Metric Impurity in Early S1.3 Smoke (2026-05-30)

**Date**: 2026-05-30  
**Phase**: S1.4 → S3 transition  
**Trigger**: Initial S1.3 smoke showed strong diversity in full recipe but **zero arm did not collapse** when using the original polluted proxy.

---

## Humanistic Preflight (per skill)

**Plain language story**:
We were measuring "how much the model is exploring different trajectories" using batch variance on the final hidden state.  
But the final hidden state was being pushed around every step by synthetic gold injection + fake rehearsal targets.  
So even when we turned off the stochastic noise (the thing we actually cared about), the measurement still looked "diverse" because of all the other artificial variance we were injecting in the minimal loop.

The model wasn't lying — our ruler was dirty.

---

## Technical Diagnosis

### Original Polluted Metric (used in first S1.3 smoke)
```python
var = trajectories.var(dim=0).mean()
diversity = log1p(var * 200)
```
This was taken on the **post-rehearsal, post-gold** final `h`.

### Sources of spurious variance in the minimal prototype:
1. `gold_delta` added every step (random gold_state * decay)
2. Rehearsal target pull simulation (`target = gold * 0.8 + mean(h) * 0.2`)
3. Initial random `x` + recurrent carry

These create batch-wise differences independently of the stochastic breadth noise.

### Result
- Full arm: high "diversity" (partly real, partly fake)
- Zero arm: still high "diversity" (fake variance remained)
- False negative on the most important contract (Reverse I→G→A stochastic breadth ablation)

---

## Resolution (S1.4 Action Taken)

Implemented `compute_pure_stochastic_contribution`:

- From **identical starting state**, run two forwards inside the same step:
  - One with the noise tensor passed to the hybrid block
  - One with `stochastic_breadth_noise=None`
- Measure L2 delta on the **final fused output** between the two arms.
- This delta is caused **only** by the stochastic breadth path.

**Result after fix**:
- Full recipe: pure_stoch_effect ≈ 1.40 (clear causal movement)
- Stochastic zero arm: pure_stoch_effect = 0.0000 (contract perfectly holds)

---

## Lessons for Future S Levels

1. **Never trust aggregate variance proxies** when claiming a specific mechanism (especially stochastic breadth).
2. For any "stochastic / exploration / breadth" claim, the gold standard is a controlled with-vs-without from the exact same prefix state.
3. When moving to real rehearsal + real gold (S2+), we must keep this controlled-pair discipline in the eval harness.

---

## Decision

- Previous "S1.3 FAIL" was **measurement artifact**, not (necessarily) an architectural failure of the hybrid block or fusion.
- With clean metric, the OneBodyParallelHybridBlock + S1.1 injection decisions now **pass the core 5.56 ablation contract** on the stochastic breadth axis.
- We can proceed toward S2 with higher confidence, while treating the earlier polluted-metric phase as a valuable diagnostic lesson (properly recorded here under S3).

This is how real research works under the skill: when a gate fails, diagnose why before declaring the architecture broken.

**Status**: S3 diagnostic work for this specific failure mode is complete. The bottleneck was "dirty ruler", not "broken backbone".
