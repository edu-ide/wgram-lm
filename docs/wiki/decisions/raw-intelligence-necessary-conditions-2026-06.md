# Raw Intelligence / Actual Reasoning Necessary Conditions — SSOT (2026-06)

**Status**: Canonical SSOT replacing the 2026-05-02 "Raw Intelligence Gates" definition for all new work.  
**Date**: 2026-06  
**Context**: Updated for the OneBodyParallelHybridBlock architecture (recurrence-primary + attention sync), MSA as first-class internal memory mechanism, deep 5.56 Adaptive Rehearsal integration, Gating v2, and the explicit goal of a ~1B model surpassing Qwen3.6-27B-class models via architecture (not scale).

This document supersedes:
- `raw-intelligence-gates.md` (2026-05-02 version — historical, pre-hybrid, pre-MSA elevation)
- Older pure-recursive depth gates that assumed the old QTRMRecursiveCore substrate

The old definition (pure recursive depth + memory on/off + composition, no-retrieval) remains valuable historical context and its eval infrastructure (`src/qtrm_mm/eval/raw_intelligence_gate.py`, pure_recursive_reasoning_* scripts and datasets) should be **extended**, not discarded.

---

## 1. Operational Definition of Raw Intelligence (Actual Reasoning) in 2026-06 Context

"Raw intelligence" / "actual reasoning" means: **the model performs non-trivial latent-space iterative computation that is causally necessary for high-quality answers on held-out problems, without relying on retrieval shortcuts, visible CoT crutches, or donor policy credit.**

From [QTRM Terminology](../concepts/qtrm-terminology.md) (operational version):

> Intermediate latent computation changes the final answer, the change improves correctness, disabling the claimed computation removes the gain, and the behavior transfers to held-out cases.

In the current architecture this specifically requires:

- The **OneBodyParallelHybridBlock** (recurrence-primary path with GatedDeltaNet-2 / V2 + attention every 4 + vector per-dimension fusion) is the substrate for the iterative computation.
- **MSA-style sparse selective memory access** (Raven-inspired slots or layer-wise chunk routing over past latent states / thought traces / gold states) is causally active inside the loop.
- **5.56 Adaptive Rehearsal dynamics** (stochastic breadth, real gold structural injection, attractor protection, scheduled decay) act as the trainable write/rehearsal policy that improves the quality and stability of that latent computation.
- All gains must survive clean ablations (core/recurrence off, sparse router off, 5.56 components zeroed, donor scale reduced) on no-retrieval, held-out reasoning tasks.

Raw intelligence claims are **not** allowed from:
- Better formatting or fluency while core-off matches full.
- Gains that disappear when stochastic breadth / gold injection / attractor protection are ablated.
- Donor credit (donor-only or core-off still solves the hard cases).

---

## 2. Updated Necessary Conditions for Raw Intelligence (2026-06)

These are the minimal conditions that must be verifiably met (via PoC experiments with clean metrics and ablations) before stronger "raw intelligence" or "1B >> 27B reasoning" claims are made on the current architecture.

They are deliberately mapped to the 7 S2 PoC conditions for 1B vs 27B (see [S2 PoC Verification Plan](../../roadmaps/S2_PoC_Verification_Plan_for_1B_vs_27B.md)).

### RI-1: Causal Test-Time Compute Scaling via Hybrid Recurrence Depth (maps to S2 #2)
Increasing the number of recurrence steps / latent iterations in the OneBodyParallelHybridBlock (with attention sync every 4) must produce large, predictable, monotonic gains on held-out raw reasoning tasks.

**Required evidence**:
- Depth sweep (1 / 4 / 8 / 12+ steps) on no-retrieval heldout sets.
- Full hybrid depth beats "recurrence-off" (attention-only mode) and donor-only baselines by a clear margin.
- Gains are not explained by simple repetition or donor leakage.

**Primary proxies** (while full raw reasoning evals are expensive):
- Clean "pure stochastic effect" (or equivalent depth-sensitive metric) as function of horizon.
- State robustness probe under increasing depth.

### RI-2: Long-Horizon Latent State Stability via Attractor Dynamics + Sparse Memory (maps to S2 #3)
The recurrent latent state (z_h / thought traces inside the hybrid recurrence) must remain high-quality and usable over dozens to hundreds of steps without rapid degradation or collapse into repetition / drift.

**Required evidence**:
- Robustness probe (state perturbation or carry ablation) shows graceful degradation at long horizons (80–200+ steps) on real-gold or hard reasoning curricula.
- Attractor protection (0.7 during rehearsal) and any MSA-style persistence (untouched slots or low routing weight) are causally responsible for the stability.

### RI-3: Causal Contribution of 5.56 Inductive Biases to Raw Reasoning (maps to S2 #4)
Stochastic recurrent breadth, real gold (bos_latent) structural injection, attractor protection during rehearsal, and scheduled binding decay must not be decorative — they must show large, clean causal drops on raw intelligence metrics when ablated.

**Required evidence**:
- Full ablation matrix (full 5.56 recipe vs stoch_zero vs gold_off vs protection_off vs decay_disabled) on the hybrid, using both the clean pure stochastic effect proxy **and** direct raw reasoning heldout accuracy / diversity / robustness.
- Stochastic breadth remains the dominant driver (as seen in S2), but gold injection and protection must still show measurable positive contribution to long-horizon raw reasoning stability and correctness.

**Substrate update (2026-06 attractor-centric + brain-mimetic triple memory)**:
In the new OneBodyParallelHybridBlock + attractor-centric substrate with BrainMimeticTripleMemory, "stochastic recurrent breadth" is realized as **structured, data-aware K-trajectory mental simulation** (inside ActiveWorkingMemory) modulated by StabilizingAttractorMemory, ProvenanceEpisodicMemory, and the Predictive Data Intuition surprise signal. 
This modern form must demonstrate equivalent or stronger causal contribution than historical GRAM/PTRM-style breadth. The ablation (`brain_mimetic_stochastic` off or `data_intuition_ablation_zero=True`) must produce clear drops on the same gates. Historical prior/posterior noise injection on z_h is no longer the canonical form; the structured mental simulation + surprise modulation inside the triple memory is the required realization.

### RI-4: Sparse Selective Memory Access is Causally Active Inside the Latent Reasoning Loop (MSA/Raven-style) (maps to S2 #1 + #3)
The model must use structured sparse routing (top-k / learned router over past latent states, thought chunks, or memory slots) rather than dense updates or simple FIFO rehearsal. Untouched or low-weight memory must exhibit high persistence (anti-interference).

**Current Status (2026-06)**: First concrete implementation work started — this was the *most insufficient* condition at the time of the audit.
- `src/qtrm_mm/memory/sparse_slot_router.py` created (full `SparseSlotRouter` with `set_ablation` for clean experiments).
- Read path integrated into `TorchGatedDeltaNet2MixerV2` (mixers.py).
- **Le-TTT & LeJEPA Substrate (2026-06 Upgrade)**: Exclusively adopted the **Le-TTT (Lean Joint-Embedding Test-Time Training)** and **LeJEPA SIGReg** paradigm as the canonical realization. Long-term memory is represented as learnable fast weights within `DecoupledLatentMemoryBank` updated in-place via online gradient descent on a JEPA-style prediction error. Isotropic representation collapse is mathematically prevented via Sketched Isotropic Gaussian Regularization (SIGReg).
- Detailed status + next micro-steps: see [RI PoC Execution Plan](../../roadmaps/RI_Raw_Intelligence_PoC_Execution_Plan_2026-06.md#P2.1).

**Required evidence** (once fully wired + trained):
- Router-on vs router-off (dense) or top-k=0 ablations produce clear drops on long-horizon recall, compositional reasoning, and state stability.
- Non-selected slots show near-perfect persistence (with `carry_rate < 0.90` showing active selectivity).
- 5.56 rehearsal and router co-evolve (router chooses *what* to update, 5.56 policy controls *how* with gold + attractor + stochastic + JEPA prediction error surprise).
- **SIGReg Loss Active**: The SIGReg covariance/variance regularization is actively optimized during continuation to maintain isotropic Gaussian distribution of slots and states.

Raven-style slots with Le-TTT memory bank is the official canonical design.

### RI-5: Efficient One-Body Hybrid Synergy for Raw Reasoning (maps to S2 #5)
The parallel hybrid design (recurrence-primary + attention-secondary with vector gating) must produce positive synergy for raw intelligence, not destructive interference.

**Required evidence**:
- Full hybrid > pure-recurrence version (attention heads = 0 or disabled) on depth scaling and long-horizon stability.
- Vector per-dimension gating (Gating v2) + stochastic injection points are causally helpful (ablate the gate or the recurrence bias and measure drop).
- Attention sync every 4 improves global coherence without diluting the iterative latent computation.

### RI-6: Low Training Waste — Core Mechanisms Are Actually Used for Reasoning (maps to S2 #6)
During training with the 5.56 curriculum on the hybrid, the model must be forced to use the recurrence depth, sparse memory routing, and rehearsal policy for actual capability gains rather than shortcuts.

**Required evidence**:
- Training curves + intermediate checkpoints show that ablating the core mechanisms (depth, router, 5.56 components) hurts held-out raw reasoning even mid-training.
- No "cheating" via donor scale or visible context leakage on the no-retrieval raw reasoning splits.

### RI-7: Better Data Efficiency via Stronger Priors for Reasoning (maps to S2 #7)
The combination of OneBodyParallelHybrid + MSA sparse memory structure + 5.56 inductive biases must extract more raw reasoning capability from the same (or less) high-quality data than weaker substrates.

**Required evidence**:
- Matched data budget comparisons (historical core vs current hybrid) on raw reasoning heldouts show superior sample efficiency.
- Real 642 gold + curriculum produces larger causal gains on the new architecture than on the old one (already directionally shown in S2 ~5.5× gap).

---

## 3. Measurement & Evaluation Discipline

All raw intelligence claims must use:
- **No-retrieval / no-evidence** heldout splits (extend the existing `pure_recursive_reasoning_heldout_72.jsonl` and related sets with harder compositional, multi-hop latent, and long-horizon families).
- **Clean ablations** with identity behavior when disabled (especially stochastic breadth ablation_zero must still be perfect).
- Primary metrics: answer correctness on heldout + diversity / trajectory quality under depth + state robustness under long horizons.
- Secondary / cheaper proxies: pure stochastic effect (already implemented and validated in S2), depth-output diversity, robustness probe.

**Causal Sensitivity / "인과성 직관" Diagnostic (mandatory when claiming Predictive Data Intuition progress)**:
When work on Predictive Data Intuition (or data-grounded world model components) is active, the following must be measured and reported alongside A/B/C tracks:
- Intervention consistency probe: On synthetic causal families, perturbing a causally upstream latent factor must produce coherent downstream changes in the model's internal trajectories and final answer distribution (while non-causal correlations do not).
- What-if / counterfactual consistency in latent space: Applying "what-if" perturbations to K mental simulation trajectories must yield predictable, non-hallucinated shifts consistent with the Predictive Data Intuition model's own predictions.
- Surprise-causal correlation: High-surprise events (from the data intuition predictor) must show statistically stronger binding or influence in ProvenanceEpisodicMemory and attractor stabilization than low-surprise events.
These diagnostics do not create a new top-level RI-8 at this time, but are required evidence that the combination of predictive world model + structured stochastic mental simulation is producing genuine causal sensitivity rather than mere statistical pattern matching.
- Task-family labeling (parallelizable vs sequential/stochastic counting) as recommended in the old gates document.

The existing `raw_intelligence_gate.py` and scripts were extended on
2026-05-29 with the first executable hybrid modes:

- `hybrid_recurrence_depth_1_no_evidence`
- `hybrid_recurrence_depth_4_no_evidence`
- `hybrid_recurrence_depth_8_no_evidence`
- `hybrid_recurrence_depth_12_no_evidence`
- `hybrid_recurrence_off_no_evidence`
- `hybrid_stochastic_breadth_off_no_evidence`
- `hybrid_556_full_no_evidence`
- `hybrid_556_stoch_zero_no_evidence`
- `hybrid_556_gold_off_no_evidence`
- `hybrid_556_protection_off_no_evidence`
- `hybrid_556_decay_disabled_no_evidence`
- `ri4_sparse_persistent_memory`
- `hybrid_recurrence_depth_scaling`
- `hybrid_556_causal_matrix`

**Quick status snapshot**: See [RI Status Snapshot (May 2026)](ri-status-snapshot-2026-05.md) for a concise overview of what was recently advanced (especially RI-4) and remaining priorities.

Still required for full promotion:

- Full `556_full / stoch_zero / gold_off / protection_off / decay_disabled`
  matrix on trained checkpoints. The gate and eval modes now exist; the trained
  run evidence still must be produced.
- Router-on/off, persistent-memory-off, chunk-shuffle, and distractor-robustness
  runs on harder heldout datasets.
- Hybrid vs pure-recurrence vs attention-only trained comparisons.

---

## 4. Relationship to the 7 S2 Necessary Conditions for 1B >> 27B

| Raw Intelligence Condition | Primary S2 Overlap | Notes |
|----------------------------|--------------------|-------|
| RI-1 Test-Time Scaling      | S2 #2             | Core shared axis |
| RI-2 Long-Horizon Stability | S2 #3             | Attractor + MSA sparse persistence |
| RI-3 5.56 Causal Contribution | S2 #4           | Highest immediate priority (ablation matrix on real gold) |
| RI-4 Sparse Memory Access   | S2 #1 + #3        | The new structural piece (Raven/MSA slots) |
| RI-5 Hybrid Synergy         | S2 #5             | Direct |
| RI-6 Low Training Waste     | S2 #6             | Shared |
| RI-7 Data Efficiency        | S2 #7             | Shared |

Raw intelligence is the **reasoning-specific sharpening** of the general 1B>>27B necessary conditions, with heavy emphasis on no-retrieval causal latent computation.

---

## 5. Current Status & Immediate Highest-Value Work (2026-06)

**Completed / Strong Direction**:
- OneBodyParallelHybridBlock + Gating v2 + faithful 5.56 rehearsal on hybrid (S1/S2).
- Real 642 gold direct baseline infrastructure.
- Clean "pure stochastic effect" metric + state robustness probe.
- ~5.5× gap evidence at 120 steps real-gold (hybrid advantage on the stochastic / memory-quality proxy).
- 2026-05-29 closure: strict stochastic breadth gate now recognizes the active
  OneBodyParallelHybridBlock replacement; hybrid recurrence depth modes and
  RI-4 sparse persistent memory gate are wired into the raw intelligence eval
  harness; RI-4 A-mode smoke covers direct recurrent, pure delegation, and
  192-style real-tensor forced paths; the hybrid depth gate rejects
  non-monotonic ladders instead of accepting any isolated gain; the RI-3 full
  5.56 causal matrix gate is wired for stochastic-zero, gold-off,
  protection-off, and decay-disabled ablations.

**Highest immediate priorities for Raw Intelligence**:
1. Run the full 5.56 ablation matrix on the hybrid using both the cheap clean proxy **and** direct raw reasoning heldouts (this simultaneously advances S2 PoC #3 and RI-3).
2. Run the new hybrid recurrence depth ladder on no-retrieval heldouts and require monotonic depth scaling against recurrence-off.
3. Run trained MSA/sparse-slot causality: router-on/off, persistent-memory-off, chunk-shuffle, and distractor robustness for RI-2 and RI-4.
4. 150–200+ step horizon scaling experiments on real gold with the hybrid + full 5.56 recipe (RI-1 + RI-2).
5. Add attractor/fixed-point residual and halt telemetry in the next loop, following the EqR/LT2 paper evidence.

These are the concrete next steps that complete the necessary conditions for raw intelligence under the current architecture.

**Detailed executable plan**: [RI Raw Intelligence PoC Execution Plan (2026-06)](../../roadmaps/RI_Raw_Intelligence_PoC_Execution_Plan_2026-06.md) — priority-ordered experiments, exact file targets, success signals, and reuse of existing S2 real-gold + clean probe infrastructure.

---

## 6. Navigation

- S2 PoC Verification Plan: [../../roadmaps/S2_PoC_Verification_Plan_for_1B_vs_27B.md](../../roadmaps/S2_PoC_Verification_Plan_for_1B_vs_27B.md)
- PHASE S Strategy: [../../roadmaps/PHASE_S_Surpassing_5.6_Experiments.md](../../roadmaps/PHASE_S_Surpassing_5.6_Experiments.md)
- MSA Memory Architecture SSOT: [../concepts/memory-architecture-msa.md](../concepts/memory-architecture-msa.md) (includes positioning options)
- Historical Raw Gates (pre-2026-06): [raw-intelligence-gates.md](./raw-intelligence-gates.md)
- Actual Reasoning Roadmap (historical framing): [actual-reasoning-architecture-roadmap.md](./actual-reasoning-architecture-roadmap.md)
- Terminology (operational definition of actual reasoning): [../concepts/qtrm-terminology.md](../concepts/qtrm-terminology.md)
- Implementation: `src/qtrm_mm/blocks.py` (OneBodyParallelHybridBlock), `src/qtrm_mm/eval/raw_intelligence_gate.py`, `scripts/191_build_raw_intelligence_gate.py`, `scripts/192_eval_raw_intelligence.py`, rehearsal logic in S2 training scripts.

---

**This is the living SSOT for raw intelligence necessary conditions.** All future raw-intelligence experiments, claims, and architecture decisions on the hybrid + MSA + 5.56 substrate must reference and be consistent with it. Updates require corresponding updates to the S2 PoC plan and major decision records.

Old gates and infrastructure are not deleted — they are the foundation we extend.
