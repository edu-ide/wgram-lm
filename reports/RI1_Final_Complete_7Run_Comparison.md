# RI-1 Attractor Pressure Experiments: Final Complete 7-Run Comparison

**Reference Checkpoint**: `hybrid_ri4_cont_step590.pt`
**Test Design**: All variants from 590 are 50-step or 150-step continuations.

## Final Results

| Run                                      | Final int_mse | Final densing_sig | Outcome                  | Ranking |
|------------------------------------------|---------------|-------------------|--------------------------|---------|
| **0.40** (Original, long training)       | **0.16406**   | **6.10**          | Strong recovery          | 1 (Best) |
| **0.40 + explicit dcons 0.06** (50 steps)| 0.18262       | 5.48              | Good recovery            | 2       |
| **0.41** (50 steps)                      | 0.22266       | 4.49              | Late regression          | 3       |
| **0.40 + dcons=0** (100 steps)           | 0.24316       | 4.11              | Severe late regression   | 4       |
| **0.45** (50 steps)                      | 0.25000       | 4.00              | Severe late regression   | 5       |
| **0.42** (50 steps)                      | 0.26953       | 3.71              | Severe late regression   | 6       |
| **0.40 + dcons 0.08** (50 steps)         | 0.27539       | 3.63              | Most severe regression   | 7 (Worst) |
| **0.40 + dcons 0.06** (150 steps)        | 0.20605       | 4.85              | Late regression          | - (Worse than 50-step version) |

## Key Insights

- The original 0.40 (long training) remains the best by far.
- The safe 0.40 + low dcons (0.06) in 50-step continuation is the best among all tested variants from 590.
- Removing depth_consistency entirely (dcons=0) caused severe early damage and ended worse than 0.41.
- Adding any depth_consistency on top of 0.40 (0.08) produced the absolute worst result.
- Even the safe 0.40 recipe in 150-step continuation from 590 regressed to a worse final (0.206/4.85) than the 50-step version (0.182/5.48).

## Conclusion

Under the current conditions:

- **0.40 internalization + low depth consistency (~0.06)** is the current most robust and highest-performing configuration.
- Deviating from this balance (increasing internalization, increasing depth_consistency, or removing depth_consistency) consistently leads to late-run attractor collapse in short-to-medium continuations from the 590 state.
