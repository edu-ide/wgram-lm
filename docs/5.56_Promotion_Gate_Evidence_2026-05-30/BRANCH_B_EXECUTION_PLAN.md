# Branch B Execution Plan — Moving Beyond Small Synthetic Regime (2026-05-30)

**Decision**: User selected Branch B (pay the cost to chase real robustness signal).

## Goal of Branch B
Obtain measurable evidence (or clear failure) on whether the 5.56 curriculum + stochastic breadth combination can produce the kind of state robustness that was historically associated with the 5.5x numbers — outside of the current tiny synthetic training setup.

## Core Problem with Current Setup
- d_model ~64, purely synthetic random (or mildly structured) data
- State ablation probes return near-zero signal even after improvements
- This regime is too weak to test the original claim

## Branch B Strategic Options (Ranked by Feasibility)

### Option 1: Scale Up Within Current Architecture (Recommended starting point)
- Increase d_model significantly (e.g., 256~512)
- Increase recurrence depth / outer steps
- Use more structured, longer-horizon synthetic data that actually requires state maintenance
- Retrain or continue curriculum on larger model
- Re-run improved robustness probe

**Pros**: Keeps everything in the same codebase and training loop. Faster iteration.
**Cons**: Still synthetic. May still not be enough for real 5.5x-level signal.

### Option 2: Port Curriculum Logic to a Stronger Backbone
- Take the best ideas from the 5.56 curriculum (scheduled decay, gold injection, attractor protection, stochastic breadth) and implement them on top of a larger, more capable core (e.g., bigger QTRM variant, or even a standard transformer with added recurrence).
- Evaluate on actual hard-family style tasks or held-out reasoning datasets.

**Pros**: Much higher chance of seeing real robustness effects.
**Cons**: Higher engineering cost. Requires new training/eval infrastructure.

### Option 3: Hybrid (Most Pragmatic)
- First do Option 1 (scale the current trainer) as a quick test.
- If signal still weak, move to Option 2 with lessons learned.

## Recommended Immediate Next Actions (Efficient Order)

1. **Define Minimum Viable Scale for Option 1**
   - Decide target d_model (suggest starting at 256 or 384).
   - Decide on data improvements (how to make "reasoning-like" sequences at larger scale).
   - Estimate training cost.

2. **Upgrade the Probe in Parallel**
   - The current probe is too weak. While scaling the model, design v3+ probe that works with larger capacity (better metrics, structured rollouts, consistency of derived outputs, etc.).

3. **Quick Feasibility Check**
   - Run a small pilot: increase d_model modestly (e.g., 128) + better structured data for 50-100 steps.
   - See if the probe starts showing non-trivial divergence. If yes → full Branch B investment. If still dead → seriously consider Option 2 or accept fundamental limits.

4. **Documentation & Decision Gate**
   - After the pilot, write a short "Branch B Pilot Results" note.
   - Then decide whether to go all-in on Option 1 or jump to Option 2.

## Success Criteria for Branch B
- At least one curriculum variant shows clearly higher state robustness (via improved probe or real downstream) compared to strong ablations (especially stochastic zero).
- Or, clear negative result: even at larger scale, the 5.56 recipe does not produce the expected robustness advantage.

## Risk
Branch B will consume significantly more compute and engineering time than everything done so far. It should only be pursued if the user is willing to accept possible negative results.

---
**Status**: Branch B chosen. Awaiting user confirmation on starting scale (Option 1 pilot) and target timeline.
