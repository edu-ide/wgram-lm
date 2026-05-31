# RI-1 Attractor Pressure Experiments: Final 6-Run Comparison Report

**Reference Checkpoint**: `hybrid_ri4_cont_step590.pt` (produced by the successful 0.40 training)
**Test Design**: All variants are 50-step continuations from the same 590 checkpoint.

## Final Results (Step 640)

| Run                                      | Final int_mse | Final densing_sig | Outcome                  | Ranking |
|------------------------------------------|---------------|-------------------|--------------------------|---------|
| **0.40** (Original reference)            | **0.16406**   | **6.10**          | Strong late recovery     | 1 (Best) |
| **0.40 + explicit dcons 0.06** (Clean Baseline) | 0.18262       | 5.48              | Good recovery            | 2       |
| **0.41** (Safe)                          | 0.22266       | 4.49              | Late regression          | 3       |
| **0.45**                                 | 0.25000       | 4.00              | Severe late regression   | 4       |
| **0.42**                                 | 0.26953       | 3.71              | Severe late regression   | 5       |
| **0.40 + dcons 0.08**                    | 0.27539       | 3.63              | Most severe regression   | 6 (Worst) |

## Key Observations

- **0.40 remains the clear winner** by a significant margin.
- The clean baseline (0.40 + explicit dcons 0.06) produced solid recovery (0.18262 / 5.48) without any late regression — validating the safe recipe.
- Any attempt to meaningfully increase pressure beyond this point (higher internalization or higher depth_consistency) led to late-run attractor collapse.
- Adding even a small depth_consistency boost on top of 0.40 (the 0.08 experiment) produced the worst final result of all.

## When Regression Begins

- 0.41: Regression started after ~step 614–616
- 0.45: Very slow improvement until ~624, then regression
- 0.42: Reversal after ~620, sharp after 624
- 0.40 + dcons 0.08: Plateau then sharp regression after ~614–616

Higher combined pressure generally led to earlier and more severe regression.

## Conclusion

Under the current 50-step continuation setup with the current noise/RI schedule:

- **0.40 internalization + low depth consistency (0.06)** is the current optimal and most robust configuration tested.
- Increasing pressure (internalization or depth consistency) beyond this point is counterproductive in short horizons.
