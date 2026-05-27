## Gold Handling Hardening (2026-05-30)

During the first real 642 gold attempts, load_gold_proxy sometimes fell back to a 1D synthetic vector even when --gold_path was provided.

This caused shape mismatch in `AdaptiveRehearsal.inject_gold_state`.

**Fix applied** (in train_556_full_curriculum_minimal.py):
- When gold_state ends up as 1D d_model vector (synthetic fallback case), we defensively set gold_state = None.
- This preserves scheduled decay, stochastic breadth, and attractor protection.
- Clear log message is printed so the run is still valid for curriculum dynamics study.

This is a temporary safety measure until better shape adaptation for real 642 gold states is implemented.
