# RI-1 Attractor Pressure Experiment: Final 4-Run Comparison

**Reference Checkpoint**: `hybrid_ri4_cont_step590.pt` (produced by the successful 0.40 training)
**Test Setup**: All three higher-pressure runs (0.41 / 0.42 / 0.45) are 50-step continuations from the same 590 checkpoint.

## Final Results (step 640)

| Run              | Final int_mse | Final densing_sig | Verdict          | Key Characteristic |
|------------------|---------------|-------------------|------------------|--------------------|
| **0.40** (Reference) | **0.16406**   | **6.10**          | **Success**      | Strong late recovery |
| **0.41** (Safe)      | 0.22266       | 4.49              | **Failure**      | Late regression |
| **0.42**             | 0.26953       | 3.71              | **Failure**      | Severe late regression |
| **0.45**             | 0.25000       | 4.00              | **Failure**      | Late regression |

## Detailed Trajectory Comparison (Selected Steps)

### 0.40 Reference (the run that created the 590 checkpoint)
- Early phase: Clear penalty (peaked ~0.214 at step 550, densing ~4.68)
- Recovery phase: Strong and sustained from ~step 558
- Peak quality: step 570 (0.15625 / 6.40)
- Final (step 590): 0.16406 / 6.10 → **Excellent basin quality**

### 0.41 Run (internalization 0.41 + fixed conservative depth_cons)
- Start (592): 0.21289 / 4.70
- Strong recovery: 600 (0.18848 / 5.31) → 608 (0.17578 / 5.69) → 612 (0.17383 / 5.75)
- Peak in this run: ~step 612–614 (0.17285 / 5.79)
- Reversal begins: step 616+
- Final (640): **0.22266 / 4.49** → Clear late regression

### 0.42 Run (internalization 0.42 + aggressive depth_cons 0.15)
- Start (592): 0.26172 / 3.82 (very high initial cost)
- Partial recovery: 600 (0.22363 / 4.47) → 608–620 (~0.204–0.205 / 4.88–4.90)
- Reversal: step 620+
- Sharp collapse after 624
- Final (640): **0.26953 / 3.71** → Worst final basin

### 0.45 Run (internalization 0.45 + aggressive depth_cons)
- Start (592): 0.24707 / 4.05
- Almost no meaningful recovery (very flat 0.235–0.237 range until ~624)
- Slow then accelerating worsening after 624
- Final (640): **0.25000 / 4.00**

## When the Runs Diverge

- **Up to ~step 608–612**: 0.41 shows the best recovery among the three tests.
- **After step 614–616**: All three higher-pressure runs begin to lose momentum.
- **After step 620**: Clear divergence — 0.41 starts mild regression, 0.42 and 0.45 accelerate into severe regression.
- The higher the combined pressure (internalization + depth_consistency), the earlier and more violently the regression appears.

## Summary of Findings

1. **0.40 remains the clear winner** by a large margin in attractor quality (int_mse + densing).
2. Increasing internalization beyond 0.40 in this short-horizon setup consistently leads to late-run attractor collapse.
3. The "more pressure = better RI-1" approach failed in this experimental regime.
4. Even the "safe" 0.41 version (with fixed conservative depth_consistency) eventually entered regression, although later and milder than 0.42/0.45.
