# Missing Inductive Biases Restoration Roadmap (2026-06)

**Date**: 2026-06  
**Status**: Living document — created following user request "누락된것들 wiki 에 정리하고 마일스톤 잡고 단계별로 진행하자"

**Purpose**: Provide a single source of truth for all historically important inductive biases that were partially or completely lost / weakened during architectural pivots (especially the new-thought-structure pivot and the shift to OneBodyParallelHybridBlock + answer_state_loop as the primary recurrent engine). Define clear milestones and a phased, auditable plan to restore them (or deliberately discard with evidence) while following the project's Pivot Safety, Reverse I→G→A, and research-driven-architecture-debugging protocols.

**Core Problem Context**:
After extensive parallel falsification (including deepest substrate attacks), the RI-4 hybrid substrate consistently reproduces `persistent_carry_rate = 1.0` with no meaningful ablation signal on memory selectivity. Historical 5.53~5.56 signals were composite. One major piece — training-time stochastic recurrent breadth — was identified as "the one most completely lost." Other biases exist in weakened or incomplete form. This roadmap turns that diagnosis into an executable, milestone-driven restoration program.

---

## 1. Authoritative Inventory of Missing / Weak Biases (as of 2026-06)

Prioritized by (1) historical causal contribution to 5.53~5.56 signals, (2) current SSOT violation risk, (3) feasibility of restoration inside current One-Body hybrid substrate.

### Tier 1: Critical — Most Completely Lost (Direct SSOT Violation Risk)

**1. Training-time Stochastic Recurrent Breadth (GRAM/PTRM-style)**
- **Historical Impact**: Core of 5.53~5.56 Adaptive Rehearsal success. Created K-trajectory diversity *during training* via learned prior/posterior on z_h (true_gram replace mode). Prevented latent collapse and enabled memory to learn selectivity.
- **Current Status in Primary Path (RI-4 hybrid engine)**: 
  - Legacy `state_transition_core` remains `active_in_primary_onebody_path=False`.
  - Partial I-stage port existed in `QTRMRecursiveCore`.
  - **2026-06 Restoration**: Self-contained learned prior + delta/true_gram generation added directly inside `OneBodyParallelHybridBlock` (the active engine delegated from answer_state_loop). External noise path preserved for compatibility. Full ablation_zero contract.
- **SSOT Impact**: Directly violates `internal-multitrajectory-answer-attractor-ssot.md` mandatory ablation ("GRAM/PTRM stochastic breadth off (K=1 vs K>1)").
- **Severity**: ★★★★★ (Highest)
- **Reverse I→G→A Status**: I-stage executed (hybrid block implementation + trainer wiring + safety process). Needs G-stage (multi-seed + composition) + A-stage (executable gate promotion + SSOT update).
- **Key Evidence**: inductive-bias-map.md, 2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md, recent 2026-06 RIGA record.

### Tier 2: Weak / Incomplete (Scaffolding Exists, Effect Not Strong/Causal)

**2. Elastic / Variable Recurrence Depth (Real Policy Learning)**
- Historical gains from training with variable/random depth + inference variable unroll.
- Current: Basic flags + scaffolding present (`core_elastic_depth_*`, effective_outer_steps). "Real learning of depth policy (not just random) is still weak."
- Severity: ★★★☆☆
- Reverse I→G→A: Scaffolding done; needs dedicated narrow gate for learned policy.

**3. Learned Slow-Tier Hierarchical Memory Policy**
- 4-way decision head (load/evict/compress/ignore) with gold structural bias.
- Current: Module + gold bias injection exists. "Policy is not yet strongly causal in most ablations. Still early."
- Severity: ★★★☆☆
- Reverse I→G→A: I-stage scaffolding; needs narrow gate on the decision head itself.

**4. Full-Strength 570-style Depth-wise Monotonic Answer Attractor**
- True comparison against actual recent K states in memory buffer + counterfactual variants.
- Current: Partial port (depth-wise pressure + buffer usage) in core. "Full true monotonic over actual recent K states + counterfactual" still being refined.
- Severity: ★★☆☆☆ (I-stage largely complete)

### Tier 3: Composite / Partially Ported (Requires Integration Work)

**5. Full 5.56 Adaptive Rehearsal Curriculum Strength on Current Hybrid Substrate**
- Composite: 642 gold + scheduled binding decay + attractor protection during rehearsal + stochastic breadth.
- Current: Gold/decay/protection instrumented and executable. Stochastic was the critical missing piece (now in active restoration). Full long-horizon (150-600 step) matrix + downstream hard-family / state_ablation_median evidence still pending.
- Severity: ★★★★☆ (composite effect)
- Note: This is the "highest-priority remaining historical signal" per prior deep dives.

**Healthy / Good Examples (for contrast)**
- Gated Thought Workspaces + ALRMC-aligned Importance Broadcast: Fully wired, one of the strongest causal I→G→A successes.
- One-Body contract and many provenance/attractor pieces: Strong.

**Higher-Level Substrate Hypothesis (from ri4_substrate_doubt_synthesis_2026-06.md)**
Even after restoring the above, the fundamental pattern "tight micro-step recurrent latent thinking + memory participation during thinking" may itself be a structural local minimum. This is tracked separately but informs prioritization (restoration work must be paired with substrate diagnostics).

---

## 2. Overall Strategy

- Every restoration follows **Reverse I→G→A** (Improvement → Generalization → Architecture-ization) with explicit records.
- Every major change follows the **Pivot Safety process** (mandatory impact review, executable ablations, component_registry updates, loud warnings).
- No promotion claim is valid until the relevant SSOT-mandated ablations are **executable** on the primary path and produce data.
- **Triple-Track Evaluation Principle (mandatory from 2026-06 user feedback progression: "A,B 를 적절히 다 해야될거 같음 너는 너무 A 만해" → "narrow memory heldout도" → "추론능력이 얼마나 상승하는지" → "학습 효율도 체크하고 리서치 drive 스킬 업데이트해")**: Every restoration / M-milestone artifact must produce paired evidence on the *identical* checkpoint:
  - A (Mechanism / Causal Track): v2 real-heldout driver (measure_continuation_hybrid_192 or successor) delivering the full ablation matrix (persistent_carry_rate, carry events, state_ablation drop, 4-way RI-4 contract).
  - B (Narrow Capability Sanity Track) — two legs: narrow reasoning (pure_recursive_reasoning_heldout_72) + narrow memory (memory_reasoning_heldout_expanded_72, distractors/multi-hop/selective recall).
  - C (Training Efficiency / Learning Dynamics Track): loss/step curves, convergence speed, final loss at matched step count, variance/stability under fixed data budget. Direct with-vs-without contrast on the identical trainer+data. "Codex style" full convergence + dynamics required before capability or "restoration ineffective" claims. Explicit C-debt note required if trainer only logs norm (no loss/tensorboard), as occurred in the 2026-06 GRAM-posterior_long 200-step run.
  - Interpretation rule: A flat 1.0 / no-ablation negative may be read as substrate doubt / M4 candidate only after A + B + C (or explicit C debt) are on record for the same artifact. Any leg missing = incomplete evidence.
  - Hygiene: hybrid continuation trainers must (1) emit loadable artifacts for reasoning+memory drivers, (2) emit loss/step + TensorBoard at reasonable frequency. The continuation_minimal trainer used for the critical M1 GRAM restoration (only sparse `norm=` prints) is the documented negative example that triggered C-track elevation.
- Measurement always uses the v2 real-heldout harness + 4-way RI-4 ablation contract where applicable.
- Parallel waves allowed only when they do not dilute focus on the current milestone.

---

## 3. Milestones (Measurable, Time-Bound Where Possible)

**M0: Documentation & Safety Foundation** (Completed 2026-06)
- This roadmap created and linked from index, inductive-bias-map, substrate doubt synthesis, and main RI-4 log.
- Pivot Safety process (`docs/wiki/process/pivot-safety-and-inductive-bias-preservation.md`), executable SSOT gates (`scripts/gates/check_ssot_stochastic_breadth.py`), and anti-drift culture docs live.
- component_registry hardened with `active_in_primary_onebody_path` flag + warning / assertion helpers.
- Reverse I→G→A template + first record (`2026-06-reverse-iga-stochastic-breadth-hybrid-engine.md`) created and followed for the stochastic breadth restoration.
- First smoke of the restored bias executed successfully (60 steps, real 642 gold, internal learned prior active).

**M1: Stochastic Recurrent Breadth — Full Executable + First Causal Evidence** (Current Focus, 2026-06 progress)
- Success Criteria:
  - Learned prior generation inside OneBodyParallelHybridBlock is the default when `--enable_stochastic_breadth` (true_gram mode supported).
  - `scripts/gates/check_ssot_stochastic_breadth.py --strict` passes cleanly.
  - Real 642 gold continuation with breadth armed produces clear stochastic effects.
  - 192 real-heldout v2 measurement (A: carry/ablation matrix) **and** paired B-track narrow sanity (192_eval forced_choice / families on the same artifact or driver-equivalent) both exist and are recorded.
- Latest Progress (2026-06):
  - I-stage implementation complete (self-contained prior in hybrid block + trainer wiring fix).
  - First smoke (60 steps, d=128, real 642 gold, `--enable_stochastic_breadth`): completed successfully. Checkpoints saved in `checkpoints/hybrid_ri4_learned_prior_test/`.
  - Pivot Safety warning still correctly fires for legacy state_transition_core (desired behavior).
  - Next immediate: run v2 192 measurement harness on these checkpoints + compare against previous flat 1.0 baselines.
- Required Deliverables: Full matrix + G-stage data.
- Timeline target: Immediate next wave.

**M2: Elastic Depth + Slow-Tier Policy — Policy Learning Strengthened**
- Narrow gates for learned depth policy and slow-tier 4-way decisions.
- Ablations show causal contribution (not just scaffolding).
- Integration with restored stochastic breadth.

**M3: Full 5.56 Composite Reproducibility on Hybrid Substrate**
- Long-horizon (200–600 step) matrix on real 642 gold with all pieces (including restored stochastic breadth) armed.
- State_ablation_median / hard-family answer quality movement vs historical anchors.
- Decision: "Composite promotes on current substrate" or "Document why it does not and pivot higher."

**M4: Global Decision Gate** (Triggered 2026-06)

**Trigger Evidence**:
- M1 direct contrast (200-step continuation with restored internal learned stochastic breadth vs matched run without it) both returned persistent_carry_rate = 1.0 in v2 192 measurement.
- This is the cleanest test to date on whether restoring the historically most-lost inductive bias inside the current RI-4 hybrid engine improves selectivity.
- Result: No measurable improvement from the restoration in the current regime.

**Decision Required**:
- Whether the current substrate family (tight micro-step recurrent latent thinking + memory participation during thinking, even after restoring Tier 1–3 historical biases) is exhausted.
- Go / No-Go on "continue local restoration + tuning inside this substrate" vs "higher-level architectural jump" (non-recurrent thinking phase, radically decoupled memory, different temporal granularity of recurrence, etc.).

**Immediate M4 Actions**:
1. Complete all Tier 1–3 bias status summary (restored / insufficient / to be discarded).
2. Run parallel substrate doubt diagnostics (smallest non-recurrent generative thinking experiments).
3. Produce explicit recommendation document with evidence.
4. User decision on whether to commit to a structural jump.

---

## 5.1 Starter Actions for M2 (Elastic Depth + Slow-Tier Policy)

**Concrete first sub-tasks (to be executed after M1 first measurement wave):**
- Add `core_elastic_depth_learn_policy` flag + small policy head in core / hybrid block.
- Add `core_slow_tier_learn_policy` strengthening (better supervision on the 4-way head during rehearsal).
- Create narrow gate experiment launcher: `scripts/launch_m2_policy_learning.py` (elastic + slow-tier on/off).
- Update roadmap + inductive-bias-map with M2 I-stage contract.

## 5.2 Starter Actions for M3 (Full 5.56 Composite)

**First sub-tasks:**
- Extend the existing 5.56 continuation trainer to support 200–400 step runs with the restored stochastic breadth + all other pieces.
- Define canonical M3 matrix (full composite vs each piece ablated) using the v2 192 harness.
- Add long-run resume + checkpoint hygiene improvements if needed.
- Target: First 200+ step real 642 M3 matrix launch.

## 5.3 M4 Decision Criteria (Draft)

Go criteria for current hybrid + restored biases family:
- At least 2 of Tier 1–2 biases show clear carry_rate movement (< 0.95) + ablation drop (> 5–10% relative) on real heldout.
- Full 5.56 composite on hybrid produces state_ablation_median within 0.3 of historical 5.53–5.56 anchors on compatible width.
- No new "most-deficient" blocker appears after 2 full waves of M1–M3.

If criteria not met after M3: Trigger explicit "higher substrate jump" decision (decoupled bank evolution, non-recurrent thinking, etc.) with full documentation.

---

## 4. Phased Execution Playbook (Step-by-Step per Bias)

For each bias:
1. **Historical Signal Reconstruction** (if not already done) — exact mechanism, code paths, evidence.
2. **Current Gap Audit** — code locations, config, ablation status, component_registry flag.
3. **Reverse I→G→A Record** (mandatory before code change).
4. **Narrow I-Stage Contract** — minimal One-Body + ablation_zero implementation.
5. **Executable Gate** — update or create `scripts/gates/` check.
6. **Experiment Wave** — continuation + v2 measurement + ablation matrix.
7. **G/A Decision** — promote, iterate, or discard with evidence.
8. **Wiki Update** — inductive-bias-map + this roadmap + main logs.

Safety gates at every step: Pivot Safety review, loud warnings, registry updates.

---

## 5. Immediate Next Actions (Post 2026-06 Stochastic Breadth I-Stage)

1. Complete M1 measurement wave on the current "learned prior restored" continuation runs (carry_rate, stochastic_diversity, ablation contrast).
2. Update inductive-bias-map entry for Stochastic Breadth to reflect 2026-06 hybrid engine restoration.
3. Open M2 narrow gate planning (Elastic Depth policy + Slow-Tier).
4. Parallel: Continue substrate doubt diagnostics (non-recurrent thinking) as higher-level falsification.

---

## 6. Links & Cross-References

- Inductive Bias Map: `docs/wiki/architecture/inductive-bias-map.md`
- Historical Reconstruction (stochastic focus): `docs/wiki/decisions/2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md`
- IMTA SSOT (mandatory ablations): `docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md`
- Substrate Doubt Synthesis: `docs/wiki/decisions/ri4_substrate_doubt_synthesis_2026-06.md`
- Pivot Safety Process: `docs/wiki/process/pivot-safety-and-inductive-bias-preservation.md`
- 2026-06 Stochastic Breadth RIGA Record: `docs/wiki/decisions/2026-06-reverse-iga-stochastic-breadth-hybrid-engine.md`
- Executable Gate: `scripts/gates/check_ssot_stochastic_breadth.py`

**Last Updated**: 2026-06 (created + initial population after stochastic breadth hybrid engine restoration)

**Owner**: Follows research-driven-architecture-debugging skill + explicit user "단계별로 진행" directive.

---

This document is the single place to track progress on "solving the missing pieces." All future work on these biases must reference and update it.
---

## 7. Progress Log (Live Updates)

**2026-06 (current session)**:
- M0: Fully completed and documented (safety processes + first RIGA record).
- M1:
  - I-stage: Learned prior self-generation implemented inside OneBodyParallelHybridBlock.
  - First smoke (60 steps, real 642 gold, `--enable_stochastic_breadth`): Success. Checkpoints produced in `hybrid_ri4_learned_prior_test/`.
  - M1 measurement (v2 192 harness on step60.pt): Launched with corrected flags (previous attempt failed on unsupported `--out_dir`).
  - Pivot Safety warnings firing correctly throughout all runs.
- M2–M4: Concrete starter sub-tasks and decision criteria already added to this document (sections 5.1–5.3).
- All work committed (including roadmap creation) and recorded verbatim in main RI-4 living log.

**Current State**: M1 measurement running in background. While it completes, M2–M4 planning is already live in this roadmap. Sequential milestone progression is active.


**M1 Measurement Result (2026-06)**:
- Checkpoint from 60-step continuation with internal learned prior active.
- v2 192 measurement (8 real cases, 6 steps): persistent_carry_rate = 1.0 (same as all prior RI-4 baselines).
- Engine was exercised with proper RI-4 carry (88 slot carry events observed).
- Conclusion so far: Restoring the bias in a short 60-step continuation was not sufficient to move the selectivity signal.
- Next required for M1: 
  1. Longer continuation (200+ steps) with the restored bias.
  2. Explicit on/off contrast (breadth during continuation + ablation during measurement).
  3. true_gram mode vs delta comparison.

This data point is now part of the official M1 record.


**M2 Starter Code Initiated (2026-06)**:
- First code change for M2: `core_elastic_depth_learn_policy` flag exposed in OneBodyParallelHybridBlock.
- Commit: 745160d
- This runs in parallel with M1 measurement analysis. Full M2 narrow gate will be built after M1 results are processed.

All milestones are now being advanced sequentially with concrete artifacts and honest data recording.


---

## M1 Current Status & Next Sub-Step (as of latest measurement)

**Latest Data Point (2026-06)**:
- Checkpoint: 60-step continuation with internal learned prior active (true stochastic breadth restoration in the active RI-4 engine).
- v2 192 measurement (8 real cases, 6 thinking steps): persistent_carry_rate = 1.0
- Engine was fully exercised with proper RI-4 persistent carry (88 slot carry events).
- Honest result: Short restoration + short continuation did not produce selectivity movement.

**M1 Next Concrete Sub-Step (defined now)**:
1. Launch longer continuation (minimum 200 steps, target 300+) using the restored internal learned prior (in true_gram mode where possible) on real 642 gold.
2. Produce multiple checkpoints.
3. Run full v2 192 measurement + explicit stochastic_breadth ablation contrast (on vs zero) on the longer checkpoints.
4. Compare against historical "no internal prior" baselines.

This sub-step must be completed before declaring M1 "first causal evidence achieved" or "needs higher intervention".

**Status**: Sub-step 1 (longer run) is being prepared for immediate launch.


**M1 Next Sub-Step Execution (2026-06)**:
- 200-step continuation with the restored internal learned prior officially launched (real 642 gold, --enable_stochastic_breadth).
- This directly follows the "longer run + later contrast" sub-step defined after the 60-step 1.0 measurement.
- Expected to produce better checkpoints for the critical on/off measurement.

M1 remains the current active milestone. M2–M4 planning artifacts exist and can receive light parallel work if M1 experiments are running in background.


**M1 Sub-Step Execution Result (2026-06)**:
- 200-step continuation with restored internal learned prior: **Completed successfully**.
- 5 checkpoints produced (40~200 steps).
- Stable training dynamics, real 642 gold used.
- This directly fulfills the "longer run" part of the M1 sub-step defined after the 60-step 1.0 measurement.

**Immediate Next Action for M1**:
Launch v2 192 measurement + stochastic_breadth ablation contrast (full vs breadth_ablate) on the step200.pt (and ideally step120/160 for comparison).

This contrast is the key to determining whether the restored bias is producing measurable selectivity improvement.


**M1 Progress Update (2026-06)**:
- 200-step continuation with restored bias: Success (5 checkpoints).
- M1 critical contrast measurement (v2 192 on step200.pt): Launched in background.
- Honest data from 60-step run (1.0) recorded.

**M2 Light Parallel Starter**:
- Basic learnable depth policy head added to OneBodyParallelHybridBlock (linear projection from pooled state).
- Commit: f341ea3

M1 remains the active focus. M2 receives minimal parallel scaffolding as allowed by the roadmap.


**M1 Critical Data Point (2026-06)**:
- 200-step continuation with restored internal learned prior → v2 192 measurement: **persistent_carry_rate = 1.0**
- Engine was properly exercised with RI-4 carry (120 slot carries observed).
- This is the longest checkpoint we have tested with the restored bias active.

**Honest M1 Assessment**:
Even after implementing the historically missing training-time stochastic breadth inside the active RI-4 engine and running a 200-step continuation, the current proxy measurement still shows flat 1.0 carry rate.

**Defined Next Sub-Step for M1** (to be executed next):
1. Produce a matched "no stochastic breadth" 200-step continuation (same config, just without --enable_stochastic_breadth).
2. Run identical v2 192 measurement on both "with restored breadth" and "without" checkpoints.
3. Direct contrast will give the clearest signal on whether the restored bias is moving the needle.

If this contrast also shows no difference, M1 will need escalation (longer horizons, different mode, stronger integration, or honest conclusion that this bias alone is insufficient in the current substrate).

This keeps M1 execution rigorous and sequential.


**M1 Next Sub-Step Execution (2026-06)**:
- Direct contrast continuation launched: 200 steps *without* stochastic breadth (matched to the previous "with restored bias" run).
- This fulfills the "produce matched no-breadth continuation for direct comparison" action defined after the 1.0 measurement on the restored-bias 200-step checkpoint.

M1 contrast data (with vs without the restored bias on equivalent long runs) will be the key input for deciding whether M1 has produced causal evidence or requires escalation.


**M1 Contrast Measurement Launched (2026-06)**:
- Matched "no stochastic breadth" 200-step continuation completed successfully.
- v2 192 measurement on its step200.pt (direct contrast to the previous "with restored bias" measurement) is now running in background.

This is the key data generation step for M1. Once both measurements are available, we can perform the direct With vs Without restored bias comparison and make the call on M1 progress.

M1 remains the current active focus.


**M1 Direct Contrast Result (2026-06) - Key Finding**:
- 200-step "with restored internal learned prior" vs "without stochastic breadth": 
  **Both returned persistent_carry_rate = 1.0** in identical v2 192 measurement.

This is the cleanest apples-to-apples test we have on the impact of restoring the historically most completely lost inductive bias (GRAM/PTRM training-time stochastic recurrent breadth) inside the active RI-4 hybrid engine.

**Current Assessment for M1**:
Restoring this bias and giving it 200 steps of real gold continuation did not produce measurable improvement in the core selectivity metric (persistent_carry_rate) under the current v2 192 proxy.

This is important negative data. It suggests that this particular restoration, at least at the current scale/integration/horizon, is not sufficient by itself to solve the persistent 1.0 problem.

**Recommended Next for M1 (or escalation to M4)**:
- Option A: Escalate M1 (much longer runs 500+, stronger true_gram integration, different injection points, etc.)
- Option B: Treat this as evidence that the bias restoration alone is insufficient on this substrate and move toward the higher-level substrate doubt hypothesis (M4 direction).

This data point should be treated as a major milestone input.


**M1 GRAM-like Upgrade Test Run (2026-06)**:
- 200-step continuation launched with the upgraded stochastic breadth in OneBodyParallelHybridBlock (learned prior + new posterior guidance scaffolding active via trainer patch).
- Output: checkpoints/hybrid_ri4_gram_posterior_long (checkpoints up to step200)
- Status: Completed cleanly, norms stable.
- This is the first meaningful long-horizon execution of the GRAM-like training-time stochastic trajectory modeling upgrade inside the active RI-4 hybrid engine.

This run produces a new comparison point for M1. Measurement on its step200.pt is the immediate next action.


**M1 GRAM-like Upgrade (1번) Test Result (2026-06)**:
- 200-step continuation with the upgraded stochastic breadth (learned prior + new posterior guidance) + trainer forcing posterior guidance during training.
- v2 192 measurement on step200: **persistent_carry_rate = 1.0**
- Same result as the previous "with restored bias" 200-step point.

This is the first meaningful long-horizon data on the 1번 GRAM-like upgrade (posterior guidance active).

Current reading: Making the stochastic breadth inside the hybrid block more GRAM-like (adding posterior scaffolding) did not move the selectivity needle in the current 200-step + current substrate regime.

This data point strengthens the case for M4-level structural considerations rather than further local improvements to the stochastic breadth mechanism within the existing hybrid engine.

