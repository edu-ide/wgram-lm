# Probe Improvement v2 (2026-05-30) — Structured Temporally Correlated Inputs

**Change made**:
- Added `generate_structured_workspace_sequence()` that creates AR(1)-like slowly drifting patterns + low-frequency modulation instead of i.i.d. Gaussian noise every step.

**Result**:
- Still 0.0000 divergence on both the 180-step Full and 100-step Zero checkpoints (and matrix variants).

**Analysis**:
Even with temporally correlated inputs, hard-zeroing z_h at the midpoint does not create measurable lasting effect in the future trajectory under the current small model + synthetic curriculum training regime.

Possible reasons:
- The recurrence in this small d_model=64 model is not strong enough to "hold" information against the incoming input stream.
- Zeroing the entire state is too destructive; the model resets and follows the new input immediately.
- The "structured" input we created is still not complex/persistent enough to require long-term state maintenance.

**Conclusion**:
v2 (better inputs) was a necessary step but not sufficient by itself.

**Recommended v3 direction** (next micro-step in sequence):
- Combine structured inputs with **milder ablation** (e.g., noise instead of hard zero, or masking only part of the state).
- Or make the input even more persistent (very slow random walk or actual low-dimensional hidden process that the model must track).
- Measure not raw Euclidean distance, but normalized divergence or prediction consistency of some derived quantity.

We are systematically stress-testing what is required to make a useful state robustness probe on this architecture.
