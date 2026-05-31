# Final Complete Root Cause Analysis (All Experiments)

## All Experiments Summary

- Original 0.40 (long training): Best final 0.16406 / 6.10
- Clean 0.40 + dcons 0.06 (50 steps from 590): 0.18262 / 5.48 (good recovery)
- 0.41 (50 steps from 590): 0.22266 / 4.49 (regression)
- 0.40 + dcons=0 (100 steps from 590): 0.24316 / 4.11 (severe regression)
- 0.45 (50 steps from 590): 0.25000 / 4.00 (regression)
- 0.42 (50 steps from 590): 0.26953 / 3.71 (severe regression)
- 0.40 + dcons 0.08 (50 steps from 590): 0.27539 / 3.63 (worst)
- 0.40 + dcons 0.06 (200 steps from 540): 0.20605 / 4.85 (regression, worse than 50-step version from 590)

## Core Finding

**0.40 internalization + low depth consistency (~0.06) is the current optimal and most robust balance** for this substrate in moderate-length continuations.

Any deviation from this balance in the tested setups led to late-run attractor collapse:

- Increasing internalization (0.41+): Regression
- Adding depth_consistency on top of 0.40 (0.08): Worst result
- Removing depth_consistency entirely (0): Severe early damage + regression
- Even the safe 0.40 recipe in very long continuation from an earlier checkpoint (200 steps from 540) regressed to a worse final (0.20605 / 4.85) than the shorter 50-step version from 590 (0.18262 / 5.48).

## Why This Pattern Occurs

The safe 0.40 + low dcons 0.06 combination allows the model to shape a strong, flexible attractor basin during the initial training phase. Once at a strong checkpoint (e.g., 590), the basin is already high-quality.

When we deviate (higher pressure, removing secondary pressure, or over-applying the safe recipe in long continuation without other changes), the model is forced into a different optimization path that, in limited training time, leads to brittle or over-constrained basins that fail in the later stages when variable depth sampling increases.

The long-horizon 0.40 experiment (200 steps from 540) showing worse final than the 50-step version from 590 suggests that even the safe recipe has limits when over-applied in continuation from a already-strong checkpoint without other modifications.

## Practical Takeaway

To push beyond the current best:

- Use significantly longer training from earlier checkpoints with the safe recipe (not just continuation from 590).
- Focus on other levers (noise/RI schedule optimization, proposal quality, architecture, etc.) rather than increasing pressure.
- Introduce gradual ramping for any changes.
- The combination of 0.40 internalization + low depth consistency remains the current practical ceiling for stable, high-quality attractor basins in this setup.
