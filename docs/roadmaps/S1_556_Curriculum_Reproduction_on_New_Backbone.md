# S1: 5.56 Curriculum Faithful Reproduction on the New Backbone

**Date**: 2026-05-30
**Phase**: PHASE S (Surpassing 5.6 Experiments)
**Status**: Execution Started (S0 still being refined in parallel)

## Goal

Reproduce the **full 5.56 Adaptive Rehearsal gold recipe** as faithfully as possible on the new experimental backbone (`OneBodyParallelHybridBlock` with Gating v2 + preference for official GDN2 / official MLA), while preserving all critical contracts:

- Stochastic recurrent breadth (with clean ablation_zero)
- Gold structural injection from 642
- Scheduled external binding decay (0.40 → 0.04)
- Attractor protection during rehearsal
- Overall One-Body discipline

## Why This Matters for Surpassing 5.6

We cannot claim to have surpassed (or even meaningfully improved upon) 5.6 unless we first prove that the **same training dynamics** that produced the historical signal can be made to work on the new architecture.

If the 5.56 curriculum "doesn't work well" on the new backbone (e.g., stochastic breadth loses its effect, gold injection becomes ineffective, attractor protection breaks), then any later claim of surpassing 5.6 will be on shaky ground.

## Current Gap Analysis

### Old 5.56 Trainer Setup
- Built around `QTRMRecursiveCore`
- Uses `AdaptiveRehearsal` module for curriculum logic
- Stochastic breadth injected inside `QTRMRecursiveCore`
- Gold injection and rehearsal happen at the core level

### New Experimental Backbone
- `OneBodyParallelHybridBlock` (recurrence via Gating v2 or official GDN2, attention via GQA or official MLA)
- Vector gated fusion between branches
- Currently lives mostly in isolated test scripts and the hybrid block itself
- Not yet integrated into the main 5.56 training loop

## S1 Key Challenges (to be solved in order)

1. **Curriculum Dynamics Porting**
   - Where does scheduled binding decay live?
   - Where does gold structural injection happen?
   - Where does attractor protection during rehearsal apply?
   - How does stochastic breadth interact with the new parallel hybrid + fusion structure?

2. **Ablation Contract Preservation**
   - `stochastic_breadth_ablation_zero` must still give clean identity behavior on the fused output.
   - Gold injection off must still be a meaningful ablation.
   - Attractor protection off must still be testable.

3. **Training Loop Integration**
   - Can we run the 5.56 trainer (or a close variant) using `OneBodyParallelHybridBlock` as the core recurrence/attention mechanism?
   - Or do we need a new minimal trainer for the hybrid architecture?

4. **Official Component Compatibility**
   - When using official GDN2 or official MLA inside the hybrid block, do the 5.56 dynamics still function correctly (especially stochastic breadth and rehearsal)?

## Proposed S1 Execution Order (순서대로)

### S1.1: Curriculum Dynamics Mapping
- Map every critical 5.56 dynamic (decay, gold injection, protection, stochastic) to the new hybrid block architecture.
- Identify exact insertion points in `OneBodyParallelHybridBlock` forward and the rehearsal module.
- Document any required changes.

### S1.2: Minimal Trainer Prototype
- Either extend the existing 5.56 trainer or create a minimal new trainer that can run 5.56-style curriculum using the hybrid block.
- Goal: be able to run at least 50–100 steps of the 5.56 recipe on the new architecture.

### S1.3: Contract Validation Runs
- Run small controlled experiments with:
  - Full 5.56 recipe
  - Stochastic breadth zeroed
  - Gold injection disabled
  - Attractor protection disabled
- Verify that the primary proxy metric (state robustness under ablation) still shows the expected causal drops.

### S1.4: Official Component Stress Test (optional but recommended)
- Repeat key runs using `delta_backend=official_gated_delta2` and `attention_type=mla` (when stable).
- Confirm contracts still hold.

## Success Criteria for S1

- We can run a recognizable version of the 5.56 curriculum on the new hybrid backbone.
- All critical 5.56 ablations produce measurable, directionally consistent effects on state robustness.
- The system does not silently bypass or break the historical inductive biases.

## Current Blockers (as of 2026-05-30)

- `OneBodyParallelHybridBlock` is not yet wired into any real training loop that supports the full rehearsal + gold injection machinery.
- Official MLA still has some runtime/kernel fragility in small-scale experiments.
- Exact insertion points for rehearsal dynamics inside the new parallel hybrid + fusion structure are not yet designed in detail.

## Recommended Next Micro-Steps (Immediate)

1. Create a detailed "5.56 Dynamics on Hybrid Block" design document (mapping every mechanism).
2. Decide on the minimal training path for S1 (extend old trainer vs new minimal trainer for hybrid).
3. Begin implementation of the highest-priority integration points.

---

**User Note**: S0 is still being refined in parallel (per previous choice). S1 work can proceed on the assumption that the S0 gate will be defined around state robustness / state_ablation_median style metrics. Any major S0 changes will be back-propagated into S1 planning.