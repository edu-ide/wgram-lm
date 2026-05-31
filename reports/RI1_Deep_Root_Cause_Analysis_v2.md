# Deep Root Cause Analysis: Why Only 0.40 Succeeded (Final Version with 5 Runs)

## Executive Summary

Across five carefully controlled experiments (all 50-step continuations from the same 590 checkpoint), only the original **0.40 internalization** configuration produced strong late recovery and an excellent final attractor basin (0.164 / 6.10).

All four attempts to increase pressure beyond this point — whether by raising internalization weight (0.41, 0.42, 0.45) or by adding depth_consistency on top of 0.40 — resulted in late-run attractor collapse. The worst result came from **0.40 + depth_consistency 0.08**.

The core issue is **over-constraint** in a short training horizon.

## Detailed Evidence from All Five Runs

| Run                        | Peak Quality (approx.) | Final Quality     | Behavior after step 614 |
|----------------------------|------------------------|-------------------|-------------------------|
| 0.40 (Reference)           | 0.156 / 6.40 (step 570)| 0.164 / 6.10      | Stable / excellent     |
| 0.41                       | 0.173 / 5.79           | 0.223 / 4.49      | Reversal → collapse    |
| 0.45                       | ~0.235 / 4.25          | 0.250 / 4.00      | Flat → gradual collapse|
| 0.42                       | 0.205 / 4.90           | 0.270 / 3.71      | Reversal → sharp collapse |
| 0.40 + dcons 0.08          | 0.205 / 4.88           | 0.275 / 3.63      | Early plateau → sharpest collapse |

Key pattern: The higher the combined (internalization + depth_consistency) pressure, the earlier the regression begins and the worse the final basin becomes.

## Primary Root Cause: Double High Pressure in Short Horizon

### The Dangerous Combination
When both of the following are applied together in only 50 steps:

1. High **attractor internalization** (forcing equilibrium)
2. Elevated **depth_consistency** (forcing longer-depth rollouts to produce better final states)

The model is over-constrained. It cannot simultaneously:
- Satisfy a very tight equilibrium condition, and
- Maintain the flexibility needed for depth scaling.

In a short continuation, there is insufficient time for the model to adapt its representations to satisfy both constraints at once. The result is a brittle basin that works for a while but collapses when variable depth sampling increases later in training.

### Why 0.40 Worked
- Internalization = 0.40 (high but workable)
- Depth consistency remained at the safe default of **0.06**

This combination allowed strong basin shaping from internalization without over-constraining the system via depth consistency. The model could still develop the inductive bias that "more depth is better."

### Why Adding Depth Consistency on Top of 0.40 Failed (The 5th Run)
The experiment "0.40 + dcons 0.08" was designed to test whether a small, controlled boost in depth consistency on the proven 0.40 base could help. Instead, it produced the **worst final result** of all (0.275 / 3.63).

This is strong evidence that, in the current short-horizon + current noise/RI schedule setup, **any meaningful increase in depth_consistency pressure on top of 0.40 is net harmful**.

### Secondary Factors That Amplified the Problem
- Sudden pressure jumps (instead of gradual ramping)
- No re-tuning of noise schedule or RI scale when increasing other pressures
- Very short 50-step horizon gave almost no room for the model to adapt to the new pressure regime

## Important Distinction: Paper Ideas vs. Experimental Execution

The core ideas from EqR and "Solve the Loop" (hierarchical noisy fixed-point, explicit internalization, depth consistency as a training signal, etc.) are likely still valid.

The failure here was **not** in the implementation of those ideas, but in:
- The **aggressive stacking** of multiple strong pressures without sufficient safeguards or training time.
- Treating "more pressure" as a simple linear lever rather than a highly sensitive, context-dependent one.

## Conclusion

Under the current experimental conditions (short 50-step continuations from a strong 0.40 base, current noise/RI schedule, current substrate):

- **0.40 internalization + low depth consistency (≈0.06)** is the current practical sweet spot.
- Increasing internalization beyond 0.40, or adding depth consistency on top of it, reliably produces late-run attractor collapse rather than improved RI-1 scaling.

Any attempt to push further must address the over-constraint problem (longer horizon, pressure ramping, reduced secondary pressures, or re-tuning of noise/RI).
