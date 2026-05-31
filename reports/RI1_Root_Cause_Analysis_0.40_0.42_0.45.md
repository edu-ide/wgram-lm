# Root Cause Analysis: Why 0.42 and 0.45 Failed While 0.40 Succeeded

## Executive Summary

Pushing internalization weight beyond 0.40 (to 0.42 and especially 0.45) in a short 50-step continuation caused late-run attractor collapse instead of improved RI-1 inductive bias. The primary culprit was **unintended double high pressure** created by an overly aggressive auto depth_consistency_weight boost combined with high internalization.

## Detailed Evidence

### 1. Trajectory Patterns
- **0.40**: Classic "pay early cost, strong late recovery" (EqR/Solve-the-Loop expected pattern). Peak int_mse ~0.214 at step 550 → best 0.156/6.40 at step 570 → final 0.164/6.10.
- **0.42**: Initial high cost (0.261 at start), partial recovery to ~0.204-0.205, then sharp reversal after step 620, ending 0.269/3.71.
- **0.45**: High cost, almost no recovery (very flat), then gradual worsening, ending 0.250/4.00.

The failure mode for both 0.42 and 0.45 was **late-run regression**, not just "failed to recover."

### 2. The Critical Code Change (Introduced Before 0.42/0.45)
In `scripts/train_hybrid_ri4_real_continuation_minimal.py`:

```python
if internal_fast_recurrent and depth_consistency_weight == 0:
    base = 0.06
    int_w = attractor_internalization_weight
    if int_w >= 0.40:
        base = 0.15   # ← This was too aggressive
    ...
```

- For the successful **0.40 run**: depth_consistency_weight remained at safe **0.06**.
- For **0.42 and 0.45 runs**: it was auto-boosted to **0.10 or 0.15**.

This meant we were simultaneously applying:
- Very high equilibrium internalization pressure, **and**
- Very high shortcut/depth consistency pressure.

In a short continuation (only 50 steps), this combination over-constrained the learning dynamics.

### 3. Why This Broke RI-1 Goals
The goal of RI-1 is to instill "deeper is better" inductive bias through variable depth training + strong but balanced attractor pressure.

When two strong pressures (internalization + depth consistency) are stacked without sufficient training time or proper ramping:
- The model cannot properly shape a flexible, depth-scalable attractor basin.
- Instead, it collapses into rigid or poor local equilibria.
- Late in training, when variable depth sampling hits higher effective depths, the over-constrained system fails → regression.

This matches exactly what we observed: early/partial recovery followed by sharp late collapse once higher depths were sampled more.

### 4. Hyperparameter Sensitivity
- 0.40 appears to be near a sweet spot under the current short-horizon + current noise/RI schedule.
- Jumping +0.02 (to 0.42) or +0.05 (to 0.45) crossed a threshold where the combined pressures became destructive.
- The papers (EqR, Solve the Loop) emphasize careful balancing of noise, relaxation, and pressure. We had the mechanisms but not the balanced scheduling for aggressive jumps in short continuations.

### 5. Other Contributing Factors (Secondary)
- Short horizon (50 steps) gave little room for the model to adapt to the new higher pressure regime.
- Noise schedule and RI scale were not re-tuned when internalization was increased.
- The aggressive auto depth_consistency boost was an untested change introduced during the push for "D" (more pressure).

## Conclusion on "Paper vs Implementation" Question

This was **not** primarily a case of "not reading the papers" or "wrong implementation of the core mechanisms."

The core EqR + Solve-the-Loop ideas (H/L cycles, explicit NI, RI, internalization, Anderson, variable depth) were correctly ported and active.

The failure was in **experimental judgment and hyperparameter control**:
- We over-estimated how much additional pressure the current substrate + short continuation could absorb.
- We introduced an unvalidated aggressive interaction (high int + high depth_cons) without sufficient safeguards or ramping.

The papers describe powerful ideas, but they are highly sensitive to exact balances, especially when stacking multiple strong regularizers in limited training time.

## Immediate Code Fix Applied
The auto depth_consistency boost was made much more conservative (max 0.10 even at very high int, and milder boosts below 0.45). This change is now in the codebase.
