# RI-1 Attractor Pressure Experiments: Final 5-Run Comparison Report

**Reference Checkpoint**: `hybrid_ri4_cont_step590.pt`
**Test Design**: All higher-pressure variants (0.41, 0.42, 0.45, 0.40+dcons0.08) are 50-step continuations from the same 590 checkpoint using the EqR-style attractor substrate (H=3/L=8, Anderson in L, etc.).

## Final Results (Step 640)

| Run                              | Final int_mse | Final densing_sig | Outcome                  | Ranking |
|----------------------------------|---------------|-------------------|--------------------------|---------|
| **0.40** (Reference)             | **0.16406**   | **6.10**          | Strong late recovery     | 1 (Best) |
| **0.41** (Safe, fixed code)      | 0.22266       | 4.49              | Late regression          | 2       |
| **0.45**                         | 0.25000       | 4.00              | Severe late regression   | 3       |
| **0.42**                         | 0.26953       | 3.71              | Severe late regression   | 4       |
| **0.40 + dcons 0.08**            | **0.27539**   | **3.63**          | Most severe regression   | 5 (Worst) |

## Key Trajectory Observations

### 0.40 Reference (the run that produced the 590 checkpoint)
- Early penalty phase: int_mse rose to ~0.214 (step 550), densing fell to ~4.68
- Strong recovery phase: Began around step 558–562
- Peak quality: step 570 (int_mse 0.15625 / densing 6.40)
- Final (step 590): 0.16406 / 6.10 → Excellent, stable high-quality basin

### 0.41 Run
- Early recovery: Strong from step 596 (0.20020 / 5.00) → 612 (0.17383 / 5.75)
- Peak in run: ~step 612–614 (0.17285 / 5.79)
- Reversal: Started after step 614–616
- Final: 0.22266 / 4.49 → Clear late collapse

### 0.42 Run
- Very high initial cost: step 592 (0.26172 / 3.82)
- Partial recovery: Peaked around step 608–620 (~0.204–0.205 / 4.88–4.90)
- Reversal: Began after step 620, accelerated sharply after 624
- Final: 0.26953 / 3.71 → Worst basin quality

### 0.45 Run
- High initial cost with almost no meaningful recovery
- Very flat until ~step 624 (int_mse ~0.235–0.237)
- Gradual then accelerating regression after 624
- Final: 0.25000 / 4.00

### 0.40 + dcons 0.08 (This run)
- Highest initial cost among all tests: step 592 (0.27930 / 3.58)
- Slow, limited recovery: Reached best around step 610–612 (~0.205 / 4.88)
- Reversal: Began after step 612–614, accelerated sharply after 620
- Final: 0.27539 / 3.63 → Worst final numbers overall

## When and How the Runs Diverge

- **Up to ~step 608–612**: 0.41 showed the strongest recovery among the pressure tests.
- **After step 614**: All higher-pressure variants began losing ground.
- **After step 620**: Clear divergence into regression. The higher the combined internalization + depth_consistency pressure, the earlier and more violent the collapse.
- Adding even a "mild" depth_consistency boost (0.08) on top of proven 0.40 internalization caused the worst outcome, suggesting that any increase in depth_consistency pressure is currently harmful in this short-horizon regime.

## Overall Verdict

- **0.40 remains the clear best configuration** tested so far.
- Any attempt to increase pressure beyond 0.40 (whether by raising internalization or by adding depth_consistency) led to late-run attractor collapse in 50-step continuations.
- The "more pressure = better RI-1" hypothesis has been falsified under the current experimental conditions (short horizon, current noise/RI schedule, current substrate).
