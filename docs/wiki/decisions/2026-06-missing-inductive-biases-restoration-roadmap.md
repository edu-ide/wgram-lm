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
  - 192 real-heldout v2 measurement shows carry_rate movement and/or ablation drop.
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

**M4: Global Decision Gate**
- All Tier 1–3 biases either (a) restored with executable ablations + positive causal data, (b) deliberately discarded with recorded evidence, or (c) explicitly deferred with hard review date.
- Combined with substrate doubt diagnostics (NRG-TP style non-recurrent thinking tests).
- Go / No-Go on "this substrate family + restored historical biases is sufficient" vs. higher-level architectural jump.

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