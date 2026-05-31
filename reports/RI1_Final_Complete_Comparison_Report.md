# RI-1 Attractor Pressure Experiments: Final Complete Comparison Report

**Reference Checkpoint**: `hybrid_ri4_cont_step590.pt`
**Test Design**: All variants from 590 are 50-step or 150-step continuations. The long-horizon run is from step 540 for 200 steps.

## Final Results

| Run                                      | Final int_mse | Final densing_sig | Outcome                  | Ranking |
|------------------------------------------|---------------|-------------------|--------------------------|---------|
| **0.40** (Original, long training)       | **0.16406**   | **6.10**          | Strong recovery          | 1 (Best) |
| **0.40 + explicit dcons 0.06** (50 steps from 590) | 0.18262       | 5.48              | Good recovery            | 2       |
| **0.41** (50 steps from 590)             | 0.22266       | 4.49              | Late regression          | 3       |
| **0.40 + dcons=0** (100 steps from 590)  | 0.24316       | 4.11              | Severe late regression   | 4       |
| **0.45** (50 steps from 590)             | 0.25000       | 4.00              | Severe late regression   | 5       |
| **0.42** (50 steps from 590)             | 0.26953       | 3.71              | Severe late regression   | 6       |
| **0.40 + dcons 0.08** (50 steps from 590)| 0.27539       | 3.63              | Most severe regression   | 7 (Worst) |
| **0.40 + dcons 0.06** (150 steps from 540) | 0.20605     | 4.85              | Late regression          | - (Worse than 50-step version) |

## Key Insights

- The original 0.40 (long training) remains the best by far.
- The safe 0.40 + low dcons (0.06) in 50-step continuation from 590 is the best among all tested variants from 590.
- The long-horizon 0.40 (200 steps from 540) ended at 0.20605 / 4.85 — worse than the shorter 50-step version from 590.
- Any deviation from the safe 0.40 + low dcons balance (higher internalization, higher dcons, dcons=0, or longer continuation from 590/540) leads to late-run attractor collapse.

## When Regression Begins

- 0.41: After ~step 614–616
- 0.45: After ~step 624
- 0.42: After ~step 620 (sharp after 624)
- 0.40 + dcons 0.08: After ~step 614–616
- Long-horizon 0.40 from 540: Multiple waves, major regression after ~step 698

Higher or prolonged pressure consistently leads to earlier and more severe regression.

## Conclusion

Under the current conditions (short-to-medium continuations, current noise/RI schedule):

- **0.40 internalization + low depth consistency (~0.06)** is the current most robust and highest-performing configuration.
- Pushing beyond this balance in any way (higher pressure or longer continuation without other changes) leads to late-run attractor collapse.
