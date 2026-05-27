# Model Architecture Versioning (RI-4 / Reasoning Substrate)

**Date**: 2026-05-27 (initial creation, based on full historical audit)
**Purpose**: Prevent confusion between different eras of the recurrent reasoning engine. Every major change to the core recurrence mechanism + inductive biases gets a version.

## Versioning Scheme

- **Major (vX.0)**: Fundamental change to the recurrent engine / One-Body contract.
- **Minor (vX.Y)**: Significant addition of inductive bias or curriculum (e.g., full 5.56 recipe port).
- **Patch**: Small fixes, guards, or measurement improvements.

## Current Version History

### v0.x — Legacy / Pre-Pivot Core (before ~2026-05-26)
- Core: StateTransitionCore (true_gram / delta modes with learned prior/posterior)
- Key scripts: 5xx series (510-530): StateTransition, verifiers, selectors, pool selection, typed register.
- Inductive biases: Strong training-time stochastic breadth via StateTransitionCore, early verifier-style selection.
- Notes: High signals in component-level experiments. Many 5xx scripts focused on trajectory selection and state transition.

### v0.5 — 5.56 Full Adaptive Rehearsal Curriculum Era (May 2026)
- Core: QTRMRecursiveCore + AdaptiveRehearsal (full curriculum)
- Key scripts: 
  - train_556_full_curriculum_minimal.py
  - run_556_ablation_matrix.py
  - train_556_rehearsal_smoke_real.py
  - analyze_556_curriculum_metrics.py
- Inductive biases (the "composite recipe"):
  - Scheduled binding decay (0.40 → 0.04)
  - Attractor protection **during rehearsal steps**
  - Stochastic recurrent breadth (integrated)
  - Real 642 gold structural bias
  - ALRMC-style importance protection on gold
- Measurement focus: state_ablation_median, drift, stochastic_diversity, gold_dist (not the later narrow cosine proxy).
- Notes: This is the source of the strongest historical "5.56 signals". Separate from the 5xx numbering — it is a synthesis/reproduction effort, not a direct continuation of 530.

### v1.0 — Hybrid RI-4 Recurrent Engine (post 2026-05-26 pivot, current mainline)
- Core: OneBodyParallelHybridBlock (as the actual recurrent engine inside answer_state_loop) + SparseSlotRouter (RI-4)
- Key scripts:
  - train_hybrid_ri4_real_continuation_minimal.py (and its accuracy push variants)
  - train_556_on_parallel_hybrid_minimal.py (bridge)
  - smoke_ri4_a_mode_hybrid_recurrent_engine.py
- Inductive biases (ongoing Reverse I→G→A):
  - Stochastic breadth ported into the hybrid block (delta + true_gram modes, gold-conditioned posterior)
  - Partial 5.56 curriculum (scheduled decay + boosted protection)
  - RI-4 4-way ablation contract (slots, persistence, router, hybrid participation)
- Measurement: narrow heldout cosine proxy on 72.jsonl + heldout_answer_pressure_loss (temporary for accuracy visibility).
- Notes: 
  - Created after the "new-thought-structure pivot".
  - Goal: Make the hybrid block the real recurrent engine while preserving 5.56 rehearsal inductive biases.
  - Current accuracy work (the 50% plateau runs) is on this v1.0 substrate.
  - Many recent radical experiments (coarse recurrence, decoupled bank, LEM, etc.) are explorations on v1.0.

### v1.1 — v1.0 + Full v0.5 Curriculum Rehearsal Engine (late June 2026)
- Change: `AdaptiveRehearsal.full_curriculum_rehearsal_step` becomes the **primary** rehearsal path for all gold_structured runs (instead of thin manual router update or direct step_rehearsal with singleton buffer).
- Key wiring changes (in train_hybrid_ri4_real_continuation_minimal.py):
  - Persistent rolling `rehearsal_memory_buffer` (size 8) fed from post-hybrid states every step.
  - `full_curriculum_rehearsal_step(gold_state=..., memory_buffer=..., stochastic_breadth_fn=...)` is the one call site for gold_structured.
  - Scheduled binding decay is now actually exercised *inside* rehearsal for gold injection scaling.
  - Attractor protection during rehearsal and importance-based selection from real buffer history are now live.
- Block side (already present): `rehearsal_gold_target` + 1.6x scale on gold-conditioned posterior already wired in OneBodyParallelHybridBlock.
- Effect on accuracy cycle: This closes the "imported but not driving" gap. Previous architecture_restored / gold_protection runs were still on the thin path.
- Version tag: All future runs on this trainer with gold_structured should be labeled "v1.1 + full 5.56 curriculum rehearsal primary".

### v1.2 — v1.1 + Architectural Trajectory Guardrail inside Hybrid Recurrence (current, June 2026)
- Change (direct response to "이전 버전 아키텍처 참고하면서 개선+run 반복" + git history audit of 5xx StateTransitionCore + verifier/selector): K-candidate sampling + progress-aware verifier-style scoring + selection is now executed **inside** `OneBodyParallelHybridBlock._stochastic_breadth` at every micro recurrent step (when --v0x_trajectory_selection K>1 and gold_target present).
- Why: Outer-only rehearsal selection (even the refined v1.1 progress+softmax) was insufficient for 400-step horizons because error compounding happens *inside* the 8-step answer_state_loop calls to the block. v0.x succeeded because the guardrail (true_gram per-step sampling + K-cand selection) was architectural, not post-hoc.
- Implementation: In blocks.py, when K>1 the prior/posterior draws K different next-states; a lightweight proxy (dist-to-gold + delta-progress, exactly matching historical 5xx/outer logic) acts as the verifier; only the best trajectory's update is committed. Gradients and state evolution now strongly prefer good local trajectories.
- Trainer wiring: The existing --v0x_trajectory_selection flag now controls both outer (optional) and the new internal architectural K (primary lever). After build_hybrid_stack, the value is pushed into each block's `_internal_k_trajectory`.
- Expected effect: Long-horizon stability on strict 192-style heldout (pure_recursive_reasoning_heldout_72.jsonl forced choice) should rise above the 2/8~1/8 plateau. This is the minimal faithful port of the v0.x inductive bias that the accuracy cycle required.
- Version tag: All runs using K>1 on this trainer = "v1.2 + arch guardrail". Update model_architecture_versioning.md and run headers on every iteration.

## Key Differences That Matter for Accuracy Transfer

- v0.5 (5.56): Full `AdaptiveRehearsal.full_curriculum_rehearsal_step` with explicit protection *during* rehearsal injection + strong training-time stochastic + real rolling buffer selection.
- v1.0 (Hybrid): Hybrid block as engine + thinner rehearsal update via router. Partial restorations (gold posterior in block, 1.6x scale, manual protection boost) were not sufficient.
- **v1.1 (current)**: v1.0 engine + full v0.5 curriculum rehearsal as the *primary path* for gold_structured. This is the first time the composite recipe (decay-scaled gold injection inside rehearsal + protection + buffer importance) is actually driving the v1.0 loop.

The persistent ~50% plateau on narrow proxy was the direct symptom of the rehearsal engine remaining on the thin path even when "restoration" labels were used. v1.1 is the architecture correction the accuracy cycle required.

## Recommendation

- Every new major architecture change (new recurrence engine, new memory topology, new curriculum integration) should increment the version.
- Document the version in:
  - The main trainer script header
  - docs/wiki/architecture/
  - Run logs / metrics.json
- When starting a new accuracy cycle, explicitly state "Running on v1.0 + [specific restoration]".

## Next Steps (updated 2026-06)

1. (done) v1.1 header + full `full_curriculum_rehearsal_step` as primary path in the trainer.
2. All gold_structured resume runs from step350+ must be explicitly labeled "v1.1".
3. After 100-150 steps on v1.1, run strict 192-style forced_choice eval on pure_recursive_reasoning_heldout_72.jsonl (no more reliance on narrow proxy alone).
4. If v1.1 still plateaus, the gap has moved from "rehearsal engine not wired" to "something deeper in the hybrid recurrence or loss signal".

When reviewing any run, the first question must be: "Which architecture version + which piece of the 5.56 recipe was actually driving the loop?"

## Historical Design Trade-off: Explicit Protection Mechanisms vs. Emergent Reasoning

### The Core Question
Why did the project move away from the strong explicit protection mechanisms that existed in v0.x (StateTransitionCore, verifier-style selection, K-candidate trajectory management, strong stochastic control), and what were the long-term consequences?

### v0.x Protection Mechanisms
In the pre-pivot era (v0.x), the architecture contained relatively strong built-in safeguards:
- Explicit trajectory diversity through StateTransitionCore (true_gram / delta modes with learned prior/posterior).
- Verifier / selector / pool selection mechanisms that actively filtered or chose among candidate reasoning paths.
- Strong training-time stochastic breadth that prevented the model from collapsing into narrow, low-quality trajectories too early.

These mechanisms functioned as **architectural guardrails** — they made it harder for the model to completely abandon good reasoning strategies during training, even without perfect external supervision.

### The v1.0 Pivot Decision
Around the 2026-05-26 "new-thought-structure pivot," the project deliberately moved away from many of these explicit protections when adopting the new OneBodyParallelHybridBlock + SparseSlotRouter substrate (v1.0).

**Main motivations at the time:**
- Desire to move beyond heavily hand-crafted, explicit control mechanisms.
- Belief that a cleaner, more integrated hybrid substrate could enable more *emergent* and general reasoning (following the success pattern seen in large-scale models like Codex/GPT-3).
- Architectural constraints of the new hybrid block made it difficult to directly port the old explicit verifier/selector stack.
- Philosophical shift: "Let the model discover good reasoning strategies on its own rather than forcing them through architectural constraints."

In short, the project chose to **remove or significantly weaken explicit protection mechanisms** in pursuit of a more scalable, less engineered form of reasoning.

### Observed Consequences
This choice had clear long-term side effects, which became especially visible during long-horizon training:

- When training was short (e.g., ~150 steps), reasoning heldout could still reach decent levels (~50% in some v1.1 runs).
- When training was extended significantly (e.g., 400 steps) under the same recipe, reasoning heldout often stagnated early (e.g., stuck at ~12% from step 120~360) even while training loss continued to drop.
- The model became more prone to "giving up on hard reasoning" and collapsing into easier, lower-quality strategies over long training — a classic symptom of insufficient trajectory protection and selection pressure.

This pattern ("loss goes down, but real reasoning does not improve or even degrades with longer training") was much harder to observe in short experiments and only became obvious once longer runs were attempted.

### Contrast with Successful Frontier Approaches
Interestingly, the most successful general models (especially post-2023 reasoning systems) did **not** simply remove protection mechanisms. Instead, they re-introduced strong verification and selection pressure, just in different forms:

- Inference-time search + verification (e.g., o1-style reasoning traces).
- Process supervision and outcome verification.
- Heavy use of test-time compute for trajectory selection.
- RL-based methods that effectively reward "good thinking processes."

In other words, frontier labs largely moved in the opposite direction from the v1.0 pivot: they accepted that pure emergence without strong selection/verification mechanisms was insufficient for reliable, high-quality reasoning, and began rebuilding protection layers (albeit at inference time and through training signals rather than purely architectural means).

Codex itself succeeded with relatively light explicit protection not because protection is generally unnecessary, but because code has unusually strong statistical regularities and local recoverability — a property that does not transfer well to mathematical or recursive reasoning.

### Key Lesson
The removal of v0.x-style protection mechanisms was not a neutral or universally correct decision. It was a deliberate bet on emergence and architectural minimalism. While this bet produced some gains in flexibility and substrate cleanliness, it also re-exposed a fundamental difficulty:

**In domains where errors compound severely (such as mathematical and recursive reasoning), the absence of strong trajectory protection and selection pressure makes long-horizon training unstable and often counterproductive for actual reasoning capability.**

v1.1 partially recovered some of the lost rehearsal dynamics, but it did not fully restore the level of explicit selection pressure that existed in v0.x. This remaining gap continues to manifest when the system is pushed to longer training horizons.

### Open Question
Is it possible to achieve robust, scalable reasoning in the current hybrid substrate *without* re-introducing strong protection/selection mechanisms (in some form)? Or will any attempt to significantly scale reasoning in this direction eventually require rebuilding equivalent guardrails — whether architectural, training-signal-based, or inference-time?

This remains one of the central open questions for the current architecture.
