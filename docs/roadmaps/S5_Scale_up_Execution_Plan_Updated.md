# S5 Scale-up & Family Validation — Execution Plan Update (2026-05-30, after S1.4 progress)

**Context**: After S1.4 clean metric success, the hybrid backbone has demonstrated that it can carry the core 5.56 stochastic breadth inductive bias with proper causal ownership.

**S5 Goal** (unchanged from original):
Once S1 contract is solid and S2 shows hybrid advantage (or parity with lower complexity), scale the approach to:
- Longer curriculum (300+ steps)
- Multiple independent seeds (minimum 5)
- More realistic gold sources (real 642 checkpoints + larger synthetic families)
- Official GDN2 + MLA full stack
- Hard-family proxy or real downstream eval when available

## Recommended S5 Ladder (post S2)

1. **S5.1** — 150~200 step multi-seed on current hybrid prototype (using the clean metric + full rehearsal wiring when ready)
2. **S5.2** — Switch to official_gated_delta2 + official MLA as default inside hybrid
3. **S5.3** — Increase model size / depth modestly while keeping the same OneBodyParallelHybridBlock structure
4. **S5.4** — Real hard-family / state_ablation_median style eval (Tier 2 of S0 gate)
5. **S5.5** — Family-balanced + OOD stress (if we have the data)

## Success Bar for S5 (to claim "surpassed 5.6 direction")

- Hybrid + 5.56 recipe beats or matches historical best on the S0 Tier 1 clean metrics at 2x+ the horizon
- All key ablations still causal
- At least one seed sweep shows the advantage is not a single lucky run
- When Tier 2 downstream becomes available: clear exceed of historical 5.53~5.56 band with causal story intact

This document will be expanded the moment S2 produces its first comparison table.
