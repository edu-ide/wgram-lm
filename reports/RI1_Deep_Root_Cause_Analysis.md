# Deep Root Cause Analysis: Why Only 0.40 Succeeded

## Core Question
Why did the 0.40 configuration produce strong late recovery and an excellent final attractor basin (0.164 / 6.10), while 0.41, 0.42, and 0.45 all exhibited late-run regression despite starting from the same checkpoint?

## Primary Root Cause: Over-Constraining via Combined High Pressures in Short Horizon

### 1. The Critical Interaction Effect
The dominant failure mode was the **simultaneous application of two strong pressures** during a very short continuation (only 50 steps):

- High **attractor internalization** (equilibrium pressure)
- High **depth_consistency** pressure (shortcut / monotonic depth pressure)

In the 0.40 run:
- Internalization weight = 0.40
- Depth consistency weight = **0.06** (mild, default safe value)

In the failed runs:
- 0.41: Internalization = 0.41 + depth_cons ≈ 0.08 (mild boost)
- 0.42: Internalization = 0.42 + depth_cons = **0.15** (aggressive)
- 0.45: Internalization = 0.45 + depth_cons = **0.15** (aggressive)

When both pressures are high at the same time in a short horizon, the model is forced to satisfy two demanding constraints simultaneously:
- Make the attractor state very close to equilibrium (high internalization)
- Make longer-depth rollouts produce strictly better final states than short ones (high depth consistency)

This double constraint appears to have prevented the formation of flexible, depth-scalable attractor basins. Instead, the system collapsed into more rigid or brittle equilibria that failed when variable depth sampling increased later in training.

### 2. Why the Regression Appears Late
- Early in the continuation (steps 592–610), the model can still make progress under the high pressure because it is mostly refining existing representations.
- Once variable depth sampling begins to frequently use higher effective depths (around step 610–620+), the over-constrained basin reveals its weakness.
- The model cannot properly utilize the additional thinking steps → performance on the attractor metrics degrades.

This explains why 0.41 lasted longer before regressing than 0.42/0.45 (lower combined pressure), but still eventually failed.

### 3. Hyperparameter Sensitivity in Short Continuations
The 50-step horizon is extremely short for such aggressive pressure changes. The original 0.40 success was likely the result of a fortunate sweet spot where internalization was high enough to shape the basin but not so high (when combined with depth consistency) that it over-constrained the system.

Jumping even +0.01–0.02 beyond this point, especially while also raising depth consistency, crossed a critical threshold.

### 4. Secondary Contributing Factors
- Noise schedule and RI scale were not re-tuned when internalization was increased.
- No gradual ramp was used for the pressure increase (sudden jump from the 590 state).
- The auto depth_consistency boost logic (introduced earlier) amplified the problem for 0.42 and 0.45.

## Key Lesson

For this particular substrate and short-horizon continuation setup, **0.40 appears to be near the practical upper limit** for internalization weight when depth consistency is also active.

Pushing higher without:
- Much longer training horizons,
- Careful pressure ramping, and/or
- Reduction in the secondary pressure (depth consistency)

leads to late-run attractor collapse rather than improved RI-1 scaling.
