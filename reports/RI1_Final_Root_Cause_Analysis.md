# Final Root Cause Analysis: Why the Safe 0.40 Recipe Succeeded and Pressure Increases Failed

## Summary of All Experiments

We conducted a systematic series of 50-step continuations from the same strong 590 checkpoint:

- Clean 0.40 baseline (explicit dcons 0.06): Final 0.18262 / 5.48 → Good recovery, no regression
- 0.41: Final 0.22266 / 4.49 → Regression
- 0.42: Final 0.26953 / 3.71 → Severe regression
- 0.45: Final 0.25000 / 4.00 → Regression
- 0.40 + dcons 0.08: Final 0.27539 / 3.63 → Most severe regression

The original 0.40 run (which produced the 590 checkpoint) achieved the best final basin of all (0.16406 / 6.10) thanks to much longer training.

## Core Finding

**0.40 internalization + low depth consistency (~0.06) is currently the optimal and most robust point** under short-horizon conditions.

Any meaningful increase in pressure — whether by raising internalization weight or by increasing depth_consistency on top of 0.40 — led to late-run attractor collapse in this experimental regime.

## Why This Happened

1. **Double Constraint Problem**
   When both high internalization (forcing tight equilibrium) and elevated depth_consistency (forcing longer rollouts to produce better final states) are applied together in only 50 steps, the model becomes over-constrained. It cannot simultaneously satisfy both demands while maintaining the flexibility needed for depth scaling.

2. **Short Horizon Limitation**
   The 50-step continuation gives very little time for the model to adapt its internal representations when pressure is suddenly increased. The original 0.40 success benefited from a much longer training process that allowed gradual basin shaping.

3. **No Compensation in Other Levers**
   When we increased internalization or depth_consistency, we did not sufficiently retune noise schedule, RI scale, or provide longer adaptation time. The system was simply pushed beyond its current capacity.

4. **Particularly Damaging: Adding Depth Consistency on Top of 0.40**
   The experiment "0.40 + dcons 0.08" produced the worst final result. This suggests that once internalization is already at the 0.40 level, further increasing depth_consistency pressure adds harmful constraint without meaningful benefit in short continuations.

## Practical Takeaway

Under the current substrate + 50-step continuation + current noise/RI schedule:

- **Do not increase internalization beyond 0.40.**
- **Do not increase depth_consistency beyond the safe low range (~0.06).**
- The combination of 0.40 internalization + low depth consistency is the current sweet spot for stable, high-quality attractor basins.

Any future attempt to push RI-1 scaling further must either:
- Use significantly longer training horizons, or
- Improve other parts of the system (noise scheduling, proposal injection, architecture, etc.) before adding more pressure.
