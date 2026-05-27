# Branch B - Option 2: Direct Execution Plan (Large Scale Jump)

**Date**: 2026-05-30  
**Decision**: User selected Option 2 (Direct) instead of modest pilot.

## Goal
Make a serious attempt to recover (or clearly falsify) the original 5.5x-level state robustness signal by moving to a regime where the 5.56 curriculum + stochastic breadth can actually express its potential.

This means leaving the "small synthetic toy regime" (d_model ~64, pure random data) and committing to real scale and real evaluation.

## Strategic Options for Option 2

### Path A: Scale the Current Architecture Aggressively
- Target: d_model 512 ~ 1024+, deeper recurrence, longer outer steps.
- Data: High-quality, long-horizon structured synthetic data (or early real reasoning traces).
- Keep the same training loop (curriculum + stochastic breadth).
- Evaluation: Build a proper state ablation + hard-family style evaluator on this scale.

**Risk**: Still synthetic. May scale but not capture the real inductive bias.

### Path B: Port Curriculum Logic to a Stronger Existing Backbone
- Take the proven 5.56 ideas (scheduled decay, gold structural injection, attractor protection during rehearsal, stochastic breadth) and implement them on top of a significantly stronger core (larger QTRM variant, or even a standard large transformer with added recurrence/memory).
- Train/evaluate on datasets that actually test long-horizon reasoning and state robustness.
- This is closer to how the original 5.5x signal was produced.

**Risk**: Higher engineering cost, but much higher chance of seeing real effects.

### Path C: Hybrid (Most Practical)
- Phase 1: Aggressive scale-up of current architecture (Path A) as a bridge.
- Phase 2: Parallel or sequential port to stronger backbone (Path B) using lessons from Phase 1.
- This reduces risk of going all-in on one path too early.

## Recommended Phased Approach for Option 2

**Phase 0 (Immediate - 1~2 weeks)**
- Define target scale and backbone choice (A, B, or C).
- Design the minimal viable evaluation harness for "state robustness after curriculum".
- Design the data pipeline for high-quality, long-horizon training data.
- Estimate compute budget and timeline.

**Phase 1 (Bridge)**
- Scale current QTRMRecursiveCore to a point where the probe starts showing non-trivial signal (target: clear difference between full curriculum vs strong ablations).
- Use this to validate that the 5.56 ideas still work at larger scale.
- Iterate on the robustness probe until it becomes a reliable signal.

**Phase 2 (Main Bet)**
- Port the matured curriculum + stochastic breadth logic to the stronger backbone chosen in Phase 0.
- Train with real(ish) long-horizon data.
- Run proper downstream evaluation (state ablation on hard cases, hard-family performance, etc.).

**Phase 3 (Decision)**
- Compare against strong baselines and ablations.
- Decide: Partial recovery of 5.5x signal? Clear negative? Need even larger scale?

## Critical Success Factors

1. **Evaluation is King**
   - Without a good way to measure "state robustness after curriculum", scaling is meaningless.
   - Must build or adapt a real evaluation suite early.

2. **Data Quality > Model Size (initially)**
   - Moving from pure random to structured, persistent, reasoning-like sequences will likely give more signal than just making the model bigger with bad data.

3. **Treat stochastic breadth as a first-class citizen**
   - In Branch B, stochastic breadth should be properly integrated and ablated as a core variable, not an afterthought.

## Immediate Next Decisions Needed from User

To start Option 2 efficiently, please answer:

1. **Target Backbone Strategy**
   - A: Aggressively scale the current QTRMRecursiveCore family.
   - B: Port to a significantly stronger existing architecture (specify preference if any).
   - C: Hybrid (start with A, parallel work on B).

2. **Target Scale Range (rough)**
   - d_model target? (e.g. 512, 1024, etc.)
   - Rough parameter count goal?

3. **Evaluation Priority**
   - What should the first real downstream metric be? (state ablation on structured tasks, hard-family accuracy, something else?)

4. **Timeline & Resource Expectation**
   - How aggressive do you want this? (fast dirty experiment vs serious multi-month effort)

Once we have answers to the above, I can produce a concrete Phase 0 task list with owners, dependencies, and estimated effort.

---
**Current Status**: Branch B + Option 2 locked in. Awaiting user answers to the 4 questions above to begin detailed planning and execution.
