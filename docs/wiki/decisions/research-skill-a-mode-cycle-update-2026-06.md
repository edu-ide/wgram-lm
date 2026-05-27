# Research-Driven-Architecture-Debugging Skill Update — A-Mode Cycle (2026-06)

This document records that the canonical skill file
`/home/tripleyoung/.agents/skills/research-driven-architecture-debugging/SKILL.md`
was updated to include the explicit 4-step cycle requested by the user during RI-4 One-Body work.

## The Added Cycle (exact user language)

> 미흡 한번에 수정 → 실험 → 결과 확인 → 개선

Full section title in the skill:
**Holistic Largest-Gap Closure + Immediate Experiment Cycle (A-Mode Research Loop)**

## When the cycle is preferred

- User explicitly says "A 방식으로", "한번에 수정", "미흡한거 한번에"
- The dominant remaining insufficiency is a first-class architectural invariant (e.g. side-car vs core recurrent engine)
- Incremental "one most-deficient piece" steps would produce many intermediate states that cannot be meaningfully evaluated until the full structural move is complete.

## Non-negotiable guardrails (still enforced inside the cycle)

- One-Body Covenant
- Full ablation matrix on the new mechanism
- Immediate experiment after the holistic structural change (no long theory phase)
- Dedicated safe branch + frequent git checkpoints
- Wiki + component_registry update with evidence

## Relation to existing I→G→A

This cycle is a **complement**, not a replacement, for the existing Improvement→Generalization→Architecture-ization loop. It is the sanctioned path when the user judges that the current largest gap requires a coherent holistic move rather than another granular increment.

## Evidence of application in this repo

- Branch: feat/ri4-hybrid-answer-state-loop-recurrent-engine
- Structural change: OneBodyParallelHybridBlock attached as real answer_state_loop recurrent engine (removal of side-car residual path for the 4 hybrid RI-4 modes)
- Immediate experiment: scripts/smoke_ri4_a_mode_hybrid_recurrent_engine.py (4 ablation variants, instrumentation proving call path)
- First result committed: docs/wiki/decisions/ri4_a_mode_hybrid_recurrent_first_smoke_2026-06.txt

This update makes the user's requested operating mode part of the official research skill guidance for future work.

## Second Update (same session)

Added the mandatory prioritization rule:

**Most-Deficient + Highest-Value-First Prioritization Principle (최종 목표 필요 조건 중 가장 미흡하면서 가장 가치 있는 것부터)**

This rule must be applied before deciding whether to use granular incremental work or the new A-Mode holistic cycle. It requires maintaining a living assessment of insufficiency vs. strategic value across all necessary conditions of the final goal (RI-1~RI-7 in this project), and always targeting the current #1 (most insufficient + highest value) item.

