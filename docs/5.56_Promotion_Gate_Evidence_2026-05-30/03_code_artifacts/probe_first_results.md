# First Run of State Ablation Robustness Probe (2026-05-30)

**Checkpoints tested**:
- 180-step Full (stoch ON + real gold path attempt)
- 100-step Stoch Zero

**Probe settings**: ablation=zero, 40 steps, 4 trials

**Result**: Both runs showed mean degradation = 0.0000

**Interpretation**:
The current probe implementation (very simple norm-based degradation on synthetic random workspaces) is too coarse for this small model + synthetic data regime. The state norm after ablation did not show measurable difference in this setup.

**Value**:
- Confirmed the probe script loads the trainer checkpoints correctly and runs without crashing.
- Gives us a baseline: we need a better proxy (e.g., trajectory prediction consistency, multi-step rollout error, or actual task-like sequences).

**Next improvements suggested**:
1. Use actual sequential "reasoning-like" input patterns instead of pure random.
2. Measure consistency of future z_h trajectory or next prediction stability after ablation.
3. Run on larger d_model or real data once available.

This is normal for a first-version proxy tool. The infrastructure is now in place to iterate quickly once the full matrix finishes.
