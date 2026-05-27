# Probe Improvement v1 (2026-05-30) — Trajectory Divergence Metric

**Change made**:
- Replaced the old "average state norm difference" with a proper **parallel rollout divergence** metric.
- After the ablation point, we now continue two branches (clean vs ablated) and measure the average L2 distance between their z_h states over the remaining steps.

**Result of the improvement**:
- Still getting 0.0000 on both the 180-step and 100-step checkpoints (and on the matrix variants).

**Root cause**:
The input is pure i.i.d. Gaussian noise every step. In this regime, the model has very little "state" to maintain, so zeroing z_h and continuing with the same random input produces almost no lasting divergence.

**Conclusion**:
Metric improvement alone (v1) is not sufficient. The next necessary improvement is **input quality**.

**Recommended v2 direction** (next in sequence):
- Replace random Gaussian workspace with structured, slowly evolving or "reasoning-like" sequences (e.g., low-frequency patterns, repeated motifs, or simple state machines).
- This will force the model to actually use its recurrent state, making ablation effects visible.

This v1 change is still valuable as infrastructure (the divergence measurement code is correct and reusable).
