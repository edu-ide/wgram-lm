# S2 Controlled Comparison — Execution Plan (Active Phase)

**Date**: 2026-06-01  
**Current Phase**: S2 (after successful S1.5 100-step validation)

## S2 Objective

Determine whether the new OneBodyParallelHybridBlock (with S1.1 decisions + S1.5 faithful rehearsal) delivers a measurable improvement over the historical 5.56 gold recipe on the previous backbone, under matched curriculum conditions.

## Comparison Protocol (Strict)

**Curriculum Conditions (must be matched)**:
- 100 steps (current best hybrid data point; can scale to 150 later)
- Scheduled binding decay: 0.40 → 0.04
- Gold structural injection alpha: 0.25 (modulated by decay)
- Attractor protection during rehearsal: 0.7
- Stochastic breadth: K>1, delta mode, scale 0.06

**Metrics (S0_LOCKED aligned)**:
1. Pure Stochastic Effect Size (primary)
2. State Robustness under Ablation
3. Full ablation matrix deltas (stoch zero, gold off, protection off)

**Data Sources**:
- Candidate (Hybrid): Runs from `scripts/train_556_on_parallel_hybrid_minimal.py` with S1.5 logic
- Baseline (Historical): Data from `docs/5.56_Promotion_Gate_Evidence_2026-05-30/` at closest matched length, or fresh re-runs of the old trainer

**Success Criteria for "Hybrid Wins S2"**:
- Pure stochastic effect on hybrid ≥ baseline + 0.15
- Stochastic zero arm on hybrid shows cleaner contract (closer to 0.00)
- Robustness on hybrid ≥ baseline

## Current Data Status (2026-06-01)

**Hybrid (Candidate) — Ready (Multi-seed)**:
- 3 seeds at 100 steps (S1.5 rehearsal):
  - Seed 1: 1.3984
  - Seed 2: 1.4531
  - Seed 3: 1.4609
- Average pure effect ~1.437
- Zero arm consistently 0.0000 across all runs
- Robustness 1.000
- `s2_collect_hybrid_data.py` helper script created for easy multi-seed + ablation collection.

**Historical (Baseline) — Needs Work**:
- Evidence package has excellent 50/180 step real-gold runs, but not exact 100-step clean-probe numbers yet.
- Next step: either extract comparable numbers or re-run old trainer with the clean probe.

## Immediate Next Steps (in order)

1. Run 2-3 additional hybrid seeds at 100 steps (different random seeds).
2. Analyze evidence package runs to extract or approximate 100-step equivalent metrics for the old backbone.
3. Execute the comparison script with real numbers from both sides.
4. Produce final S2 report + decision (promote hybrid / needs S4 fix / no advantage).

This plan will be updated after each major data collection step.
