# Memory Architecture Pillar (MSA First-Class) — SSOT

**Status**: Canonical single source of truth for the memory architecture track as of 2026-06.
**Last Major Update**: 2026-06 (post user clarification "memory sparse attention MSA 임" + "wiki 를 정리해").
**Supersedes**: Pre-2026-06 workspace-memory-architecture.md framing for new architecture work.

---

## 1. Strategic Context: Why Memory Architecture is First-Class for 1B >> 27B

The project's explicit long-term goal is **architecture innovation that allows a ~1B-scale model to meaningfully surpass Qwen3.6-27B-class models** (not via brute scale, but via superior per-parameter efficiency, test-time compute scaling, and inductive biases).

In the necessary conditions analysis (see [S2 PoC Verification Plan for 1B vs 27B](../roadmaps/S2_PoC_Verification_Plan_for_1B_vs_27B.md)), **memory architecture** was identified as a missing first-class pillar:

- Parameter count alone is a losing game for 1B vs 27B.
- The decisive advantage must come from **how the model reads, writes, rehearses, and selectively accesses its own latent computation history** over long horizons without degradation.
- MSA (Memory Sparse Attention) is the clearest current external reference for exactly this capability: stable performance from 16K → 100M tokens via sparse top-k routing over compressed latent memory blocks, document-wise position handling, and Memory Parallel inference.

**MSA is not a "future nice-to-have". It is the reference architecture for the memory substrate that the One-Body Parallel Hybrid + 5.56 inductive biases must ultimately realize or surpass inside the canonical causal path.**

---

## 2. MSA Reference (Source Paper)

- **Paper**: Memory Sparse Attention (arXiv:2603.23516)
- **Core Idea**: Separate memory *capacity* from active reasoning context. Route sparsely over document-level latent/KV summaries instead of stuffing all tokens into the transformer's context window.
- **Key Mechanisms**:
  - Sparse top-k memory selection
  - Compressed per-document latent/KV states
  - Document-wise RoPE / position handling
  - Memory Interleave (retrieve → expand context → generate cycles for multi-hop)
- **Claimed Scaling**: <9% degradation from 16K to 100M tokens; 100M-token inference on 2×A800.

**QTRM Mapping (updated)**:
- External MemoryOS (Harrier + FAISS + reranker) remains the 100M+ token pool (runtime scale axis).
- The **model-internal** memory architecture must implement MSA-like *selective, sparse, importance-weighted access and rehearsal* over its own recurrent latent states (z_h / thought traces).
- This must happen **inside the One-Body causal path** (no separate memory organ taped on).

See full source notes: [Memory Sparse Attention](../sources/memory-sparse-attention.md).

---

## 3. Current Realization: One-Body Parallel Hybrid + 5.56 Adaptive Rehearsal as "Latent MSA"

The 2026-05~06 architecture (OneBodyParallelHybridBlock, Gating v2, official FLA GDN2/MLA) + faithful 5.56 curriculum port is the **active implementation vehicle** for MSA principles inside the recurrence:

### 3.1 Substrate: OneBodyParallelHybridBlock (형태 1, canonical QTRMBlockStack with attn_every=4)
- Recurrence-primary (GatedDeltaNet-2 or TorchGatedDeltaNet2MixerV2) for iterative latent thinking.
- Attention-secondary (MLA or GQA) every 4 blocks for synchronization / global mixing.
- Vector-valued per-dimension gating fusion (v0.2) with temperature, recurrence bias, and stochastic injection points.
- All inside a single causal body: prompt tokens → recurrent thought core → same LM head.

This satisfies the One-Body Architecture SSOT while providing a modern, high-capacity substrate for long-horizon latent computation.

### 3.2 Memory Policy: Adaptive Rehearsal 5.56 Gold Recipe (the "MSA inside")
The 5.56 signals are re-interpreted as a **trainable, ablatable latent memory read/write/rehearsal policy**:

| 5.56 Mechanism              | MSA / Latent Memory Analogy                          | Role in 1B >> 27B Thesis                  |
|-----------------------------|-----------------------------------------------------|-------------------------------------------|
| Stochastic recurrent breadth (ablation_zero) | Sparse sampling / exploration over past thought trajectories | Prevents dense collapse; enables test-time scaling via breadth |
| Real 642 gold (bos_latent) structural injection | High-value "document" / memory block writes        | Strong priors + data efficiency           |
| Attractor protection (0.7 during rehearsal) | Importance scoring + selective preservation (anti-dilution) | Long-horizon state stability              |
| Scheduled binding decay (0.40 → 0.04)     | Temporal decay / forgetting policy                  | Capacity management without catastrophic interference |
| Full curriculum rehearsal   | Memory Interleave / repeated selective access       | Causal contribution of memory mechanisms  |

**Pure Stochastic Effect** (the clean primary metric in S0~S2) is the operationalization of "MSA-style sparse memory signal is causally active": when stochastic breadth is ablated to zero, the effect must collapse cleanly to ~0.

**Real-gold direct baseline infrastructure** (`run_s2_historical_baseline.py` + robust 642 bos_latent loading) provides the matched apples-to-apples measurement of memory policy quality on both old core and new hybrid.

### 3.3 Early Evidence (S2 Controlled Comparison, 2026-06)
- **120-step, exact same real 642 gold, clean pure stochastic effect**:
  - Historical QTRMRecursiveCore: **0.2714**
  - OneBodyParallelHybridBlock + 5.56 faithful: **1.496** (multi-seed avg, 3 seeds)
  - **~5.5× gap**
- Hybrid state robustness: **1.000** across measurements.
- Stochastic breadth is the dominant driver (gold injection off or protection off does not collapse the signal; zero arm is exactly 0.0).
- Hybrid shows continued scaling / stability advantage at 150 steps vs historical plateau/decline.

These numbers are direct, real-gold, matched-horizon, clean-probe measurements (not reconstructions). Full details and ablation matrices: [S2 Interim Verdict Report](../roadmaps/S2_Interim_Verdict_Report.md) and the 5.56 Promotion Gate Evidence package.

This is the strongest current signal that the new memory substrate + policy carries the high-signal 5.56 inductive biases **better** than the historical backbone — a necessary (though not yet sufficient) condition for the 1B >> 27B thesis.

---

## 4. Explicit Sparse Memory Track (ALRMC + MSA-style Top-k)

Parallel / future work (documented in May 28 decisions and core experiments):

- MSA-style sparse top-k (k=4) over growing pooled z_h buffer inside the recurrent core (already prototyped in older core.py forward).
- ALRMC (Adaptive Latent Rehearsal Memory Core) v0: importance scoring + rehearsal on the sparse buffer.
- Goal: explicit O(k) selective retrieval that remains stable as the number of "memory items" (past latent states) grows to dozens or hundreds — exactly the anti-dilution property MSA demonstrates at document scale.

This track is compatible with the One-Body Covenant and the current hybrid direction. The 5.56 rehearsal dynamics are the **policy**; explicit sparse routing is the **access mechanism** that can be layered on top.

See decision record: `2026-05-28-ablation-study-plan-literature-extensions.md` (Option 2 MSA-style sparse memory signal + ALRMC v0).

---

## 5. Necessary Conditions Mapping (Memory-Centric View)

**Raw Intelligence / Actual Reasoning** is the reasoning-specific sharpening of the 1B>>27B goal. See the dedicated canonical SSOT: [Raw Intelligence / Actual Reasoning Necessary Conditions (2026-06)](../decisions/raw-intelligence-necessary-conditions-2026-06.md). It defines 7 updated conditions (RI-1 through RI-7) that heavily depend on MSA-style sparse memory access + 5.56 rehearsal policy inside the One-Body Hybrid.

From the 1B>>27B PoC plan, the memory architecture directly attacks:

1. **Per-parameter efficiency** — High-quality latent states via selective rehearsal (not dense memorization).
2. **Test-time compute scaling** — Stochastic breadth + depth inside recurrence scales reasoning without prompt bloat.
3. **Long-horizon state stability** — Attractor protection + sparse access = no rapid degradation (current hybrid robustness 1.0).
4. **Causal contribution of 5.56 inductive biases** — The memory policy itself must be ablatable and show large clean drops (S2 priority).
5. **Low training waste** — Curriculum + gold injection forces the memory mechanisms to be used for actual capability, not shortcuts.
6. **Hybrid synergy** — Recurrence (iterative memory) + attention (global sync) must amplify rather than dilute the memory signal.

PoC verification priority (highest value work):
- Full ablation matrix on hybrid with real gold (PoC 3 — causal 5.56 memory policy).
- Longer horizon (150-200 step) + systematic robustness at multiple scales (PoC 1+2).
- Hybrid vs pure-recurrence controlled run (PoC 4+5).

---

## 6. One-Body Covenant & Ablation Discipline (Non-Negotiable)

All memory architecture work must obey:

- **Canonical causal path**: prompt → OneBodyParallelHybridBlock stack (recurrence-primary) → same hidden state → LM head. No side-car memory organs whose outputs bypass the core for final answers.
- **Ablation contract**: every memory mechanism (stochastic breadth, gold injection, attractor protection, future explicit sparse router, ALRMC importance) must have a clean zero/off mode that produces measurable, reproducible drops on the primary metric (pure stochastic effect + answer-level causality where applicable).
- **KISS / Reverse I→G→A / Prior-To-Implementation Contract**: New memory features are added only after historical high-signal behavior (5.56) is faithfully reproduced and measured on the new substrate.

See:
- [One-Body Architecture SSOT](../architecture/one-body-architecture-ssot.md)
- [Universal LLM Causal Path Contract](../architecture/universal-llm-causal-path-contract.md)
- [KISS YAGNI DRY SSOT Contract](../architecture/kiss-yagni-dry-ssot-contract.md)

---

## 7. Navigation & Living References

**Primary Documents (read in this order for current state)**:
1. This page (synthesis)
2. [S2 PoC Verification Plan for 1B vs 27B](../roadmaps/S2_PoC_Verification_Plan_for_1B_vs_27B.md) — necessary conditions + PoC matrix
3. [PHASE S: Surpassing 5.6 Experiments](../roadmaps/PHASE_S_Surpassing_5.6_Experiments.md) — S0~S5 strategy
4. [S2 Interim Verdict Report](../roadmaps/S2_Interim_Verdict_Report.md) — current 5.5× real-gold evidence
5. [S0 Gate LOCKED](../roadmaps/S0_Surpassing_5.6_Gate_LOCKED.md) + S1/S2 detailed plans

**Implementation**:
- `src/wgram_lm/blocks.py` — OneBodyParallelHybridBlock + Gating v2 + stochastic injection points
- `scripts/train_556_on_parallel_hybrid_minimal.py` — faithful 5.56 rehearsal on hybrid + clean probes
- `scripts/run_s2_historical_baseline.py` — production-grade real 642 gold direct baseline system

**Historical / Parallel Tracks**:
- [Workspace Memory Architecture (legacy framing)](workspace-memory-architecture.md) — pre-OneBody, pre-MSA-elevation view
- [Gated Latent Memory](gated-latent-memory.md)
- [Neural Long-Term Memory](neural-long-term-memory.md)
- Decision: `2026-05-28-ablation-study-plan-literature-extensions.md` (MSA top-k + ALRMC prototype)
- Source: [Memory Sparse Attention](../sources/memory-sparse-attention.md) + Qwen3.5 Full-MSA Fork decision

**Evidence Package**:
- `docs/5.56_Promotion_Gate_Evidence_2026-05-30/` (real 642 gold runs, stochastic diversity, 50/100/180 step analyses)

---

## Current Active Snapshot (2026-06) — Highest Value Work

**S2 Status (Controlled Comparison with real 642 gold, clean pure stochastic effect metric)**:
- Hybrid (OneBodyParallelHybridBlock + faithful 5.56): **1.496 avg** (3 seeds) at 120 steps
- Historical baseline (direct same real gold): **0.2714** at 120 steps
- **Gap: ~5.5×** in favor of new architecture
- Hybrid robustness: 1.000 sustained
- Stochastic breadth confirmed dominant causal driver

**Immediate Highest-Value Next Steps** (per S2 PoC plan):
1. Full ablation matrix on hybrid at 120 steps with real gold (causal contribution of 5.56 memory policy — PoC 3 priority)
2. 150-200 step horizon extension + multi-horizon robustness probes
3. Hybrid vs pure-recurrence isolation run

All governed by the 7 necessary conditions in the [S2 PoC Verification Plan](../roadmaps/S2_PoC_Verification_Plan_for_1B_vs_27B.md). PHASE S (S0~S5) is the execution framework. MSA + ALRMC explicit sparse routing is the parallel architectural extension track.

These steps are also the fastest path to validating several of the updated **Raw Intelligence Necessary Conditions** (especially RI-2 Long-Horizon Stability, RI-3 5.56 Causal Contribution, RI-5 Hybrid Synergy). See the dedicated [RI PoC Execution Plan](../roadmaps/RI_Raw_Intelligence_PoC_Execution_Plan_2026-06.md) + the 2026-06 RI SSOT for the reasoning-specific harness and experiment details.

---

## 8. Open Questions & Next Synthesis Work

- How to port explicit MSA-style top-k sparse routing + document-wise compression into the current hybrid recurrence without violating One-Body or adding destructive interference?
- What is the minimal "memory router head" (or integrated mechanism) that gives causal long-horizon gains on top of the existing 5.56 rehearsal policy?
- At what scale (parameter count + horizon) does the current hybrid + 5.56 memory policy start showing the <9% degradation resistance that MSA claims externally?

These are the highest-leverage questions once the current S2/S3 ablation matrix on real gold is complete.

---

**This page is the living memory architecture SSOT. All future memory-related architecture decisions, experiments, and claims must reference and be consistent with it.** Updates require corresponding updates to the linked PoC plan, PHASE S roadmaps, and major decision records.