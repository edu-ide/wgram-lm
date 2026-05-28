# Brain-Attractor-Centric Recurrent Architecture (QTRM-EqA / Attractor QTRM) — Research-Driven Proposal (2026-06)

**Status**: Proposed substrate-level architecture direction following exhaustive RI-1 negative pattern diagnosis and "2번" decision (latest 2025-2026 attractor/equilibrium/looped literature synthesis + deep GRAM/PTRM brain-mimetic integration).

**Trigger**: User directive "2번으로 가자 최신 논문들 참고해서 진행" + "GRAM/PTRM 도 참고한거지?" + "새 아키텍처 md 만들고 참고한 논문들 아래 다 써줘" after 4-step diagnostic (non-monotonic strict-B on pure_72 under brain-mimetic triple memory + stochastic sampler) and repeated falsification of serial tweaks inside the tight micro-step hybrid family.

**Core Contract (non-negotiable, per research-driven-architecture-debugging skill + all RI principles)**:
- RI-1~RI-7 (especially RI-1 test-time compute scaling via recurrence depth monotonic gain, RI-4 Sparse Selective Memory).
- Strict-B pure_72 (forced_choice teacher-forced logprob on `data/eval/pure_recursive_reasoning_heldout_72.jsonl`, no-evidence cases).
- Principle Gate (`validate_reasoning_test_principles.py` or equivalent): one_body_causal_path, honest conditions-matched tagging, proper porting, GRAM/PTRM stochastic breadth restoration live, 3-track (Workspaces+Attractor+Provenance), attractor_depth_behavior.
- Triple-Track Evaluation (A: full ablation matrix; B: narrow reasoning + narrow memory heldout; C: quantitative loss descent + convergence diagnosis).
- Proper Porting + Reverse I→G→A for historical high-signal inductive biases (GRAM/PTRM stochastic recurrent breadth as first-class).
- One-Body: everything feeds the normal LM-logit / autoregressive path; no hidden answer channels or sidecar renderers for claims.
- conditions-matched: every measurement from identical base checkpoint + honest labeling.
- KISS/YAGNI/DRY + SSOT (one-body-architecture-ssot.md, internal-multitrajectory-answer-attractor-ssot.md, inductive-bias-map.md, FAIR_COMPARISON_PROTOCOL.md).

---

## 1. RI-1 Exhaustive Negative Pattern Diagnosis (Stubborn Negative Pattern Protocol)

**Plain-language family name (local minimum, per 2026-05-28 diagnosis and subsequent falsifications)**:
"All attempts to teach reliable high-depth iterative improvement by adding variable depth sampling (M1), stronger intra-/cross-depth monotonic pressure, phase separation, limited workspace bottlenecks, depth-consistency losses, and partial stochastic breadth restoration inside the current tight micro-step OneBodyParallelHybridBlock + rolling buffer + 3-track rehearsal substrate repeatedly produce the same signature: short-run promising monotonic signals that fail to compound robustly; longer runs show plateau, regression, or non-monotonicity at target depths (d=8 and beyond) on strict-B pure_72 under matched conditions + full Principle Gate."

**Anchor preserved**:
- 25-step clean M1 run (pre-attractor patch) produced strongest observed monotonic scaling: d=8 reaching 40.28% on pure_72 strict-B with proper 3-track composition and gates. This proves variable-depth training on properly ported 3-track substrate *can* produce RI-1 behavior in limited regimes.

**Causal routes exhaustively tested inside the family (serial refinement declared exhausted)**:
- M1 sampling schedules (randint, progress bias, higher mean/max depth).
- Attractor/monotonic pressure weights + cross-depth bonuses.
- Explicit thinking-vs-consolidation separation (`--pure_recurrence_then_consolidate`).
- Forced narrow workspace bottleneck.
- Depth-state consistency (shortcut-consistency style).
- Partial + inner-loop strengthened GRAM/PTRM stochastic breadth restoration (prior/posterior + K-trajectory noise inside h-cycles).
- Coarser recurrence granularity (minimal + every-3 micro-step hybrid variants).
- memory_as_primary_recurrent_thinker intermediate (d=8 short spike 47.22% → longer-horizon collapse to ~33%, d=16/32 flat/non-monotonic).

**Measurement pattern (strict-B pure_72, conditions-matched continuations, full gate)**:
- Multiple arms: d=1 ~29-34%, d=4 ~25-43%, d=8 ~19-38% (occasional high outlier 38.89% non-reproducible), d=16/32 flat or regressing.
- Non-monotonicity at target depths; C-track healthy (strong loss descent, no oscillation); B-track (narrow reasoning) shows the scaling failure; A-track ablations confirm 3-track composition but no robust depth compounding.
- Principle Gate repeatedly surfaces: "stochastic_breadth=ON ... Pivot Safety Warning (legacy state_transition_core bias not considered active in primary path)" + "state_transition_core active_in_primary_onebody_path=False".

**Decision per skill (Stubborn Negative Pattern + Parallel Fast-Falsification + user "1번" then "2번")**:
The "tight micro-step hybrid recurrence + current 3-track rehearsal + partial stochastic restoration" substrate is classified as the deeper local minimum for RI-1 reliable test-time compute scaling. Serial refinement inside this family is stopped. Escalate to substrate-level bigger causal change while preserving M1 variable-depth schedule, proper 3-track porting, Principle Gate, strict-B, Triple-Track, and Reverse I→G→A for historical GRAM/PTRM.

**Intermediate brain-mimetic step (user "A" direction)**:
Redefinition of memory as primary recurrent thinker (ActiveWorkingMemory + StabilizingAttractorMemory + ProvenanceEpisodicMemory) + BrainMimeticStochasticSampler (K=4 mental simulations evolved inside working memory, scored by attractor stability + provenance grounding + learned scorer, diversity-preserving injection). This is a deep, structured reinterpretation of historical GRAM/PTRM stochastic breadth (not blind noise; hypothesis generation modulated by the other two memory systems). See `src/qtrm_mm/memory/brain_triple_memory.py` (step(), evolve_trajectories, sample_trajectories, select_best_trajectory, integrate_brain_mimetic_stochastic_into_triple_memory, _ensure_same_device guards). 4-step diagnostic on this substrate still yielded non-monotonic strict-B (d=1 29.17%, d=4 43.75%, d=8 25.00%, d=16 37.50%, d=32 31.25% on 16-case subset due to timeout). This falsified "memory redefinition alone on current engine" as sufficient for robust RI-1.

**Conclusion**: The recurrence *engine* (transition primitive + state topology + convergence dynamics) itself requires replacement with an attractor/equilibrium-oriented substrate explicitly designed for stable depth + breadth scaling, informed by the 2025-2026 literature. Historical GRAM/PTRM stochastic breadth is re-interpreted as first-class "structured mental simulation" inside the new memory system (not retrofitted noise).

---

## 2. Historical GRAM/PTRM Stochastic Breadth as Core Inductive Bias (Reverse I→G→A Target)

**Bias name**: Stochastic recurrent breadth (prior/posterior Gaussian sampling + noise injection into high-level latent during inner recurrence; K-trajectory exploration with posterior guidance).

**Historical strong signals**:
- Stage56/58 PTRM family (high selected/oracle accuracy via K-candidate stochastic search).
- Adaptive Rehearsal 5.53~5.56 gold recipe (combined with scheduled binding decay + attractor protection; state_ablation_median ~5.53-5.56).
- Legacy `StateTransitionCore._apply_true_gram_transition` / `_apply_stochastic_high_level_guidance` (prior/posterior networks, diagonal KL, true_gram / delta modes).

**Why it mattered (inductive bias effect)**:
- During training: forces the recurrent high-level state to explore multiple noisy trajectories instead of deterministic collapse.
- Creates explicit K-trajectory diversity *inside* the recurrence (not post-hoc over buffers).
- Posterior guidance allows answer-conditioned refinement.
- Distinct from (and historically stronger than) post-hoc multi-trajectory scoring.

**Current status in primary path (2026-06)**:
- Legacy `state_transition_core` is library-only (`active_in_primary_onebody_path=False`).
- Partial I-stage restoration exists in `OneBodyParallelHybridBlock` (learned prior, delta/true_gram, ablation_zero, inner h-cycle application after slow_stack) + trainer `--enable_stochastic_breadth` / `brain_mimetic_stochastic`.
- Full SSOT-required ablation ("GRAM/PTRM stochastic breadth off (K=1 vs K>1)") remains unexecutable in the strongest historical sense inside the primary recurrent engine.
- brain_triple_memory.py now provides the deep brain-mimetic reimplementation: stochastic sampling is no longer "add noise to z_h"; it is "sample and evolve multiple active thoughts inside ActiveWorkingMemory, stabilized by AttractorMemory, filtered by ProvenanceMemory".

**Reverse I→G→A status**: In progress via the brain-mimetic attachment. The new architecture treats the modern realization of GRAM/PTRM stochastic breadth (**structured, data-aware K-trajectory mental simulation inside ActiveWorkingMemory, modulated by Attractor + Provenance + Predictive Data Intuition surprise**) as a required inductive bias. This form must be live, ablatable, and demonstrate causal contribution on RI gates (per strengthened RI-3 language in the 2026-06 SSOT). Legacy prior/posterior noise on z_h is superseded; the brain-mimetic version inside the triple memory is the canonical form going forward.

See: `docs/wiki/architecture/inductive-bias-map.md` (Entry: Stochastic Recurrent Breadth), `src/qtrm_mm/memory/brain_triple_memory.py:301-473` (BrainMimeticStochasticSampler + integration), Principle Gate JSONs in matched_port_evaluation runs.

---

## 3. Latest 2025-2026 Attractor / Equilibrium / Looped Literature Synthesis

**Latest-first sweep principle (per skill)**: Newest primary sources first (2026-05 papers prioritized), extract *mechanisms* (not slogans), map to local failure (non-robust depth scaling on tight micro-step substrate + incomplete GRAM/PTRM port), invent architecture candidate that changes the causal route.

### Key Papers and Mechanisms

| Date | Source | Core Mechanism | Connection to Local Failure + GRAM/PTRM |
|------|--------|----------------|-----------------------------------------|
| Feb 2025 | Huginn (Geiping et al., arXiv:2502.05171) "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" | Prelude (embed) → Recurrent block (weight-tied, R iterations with input injection every step for stability) → Coda (to logits). Variable recurrence sampling (log-normal Poisson, heavy tail). Truncated BPTT (k~8). Adaptive early-exit via state change (KL/norm delta). KV-cache sharing. Path independence. | Directly attacks RI-1: test-time depth scaling via latent iteration (not tokens). Input injection + stability norms address collapse at high d. Our M1 variable-depth schedule is partial reproduction; missing persistent input injection inside the recurrent engine + learned halt. GRAM/PTRM-style breadth can be added as stochastic initialization / noise over the latent trajectory. |
| Oct 2025 (updates 2026) | Ouro / LoopLM (ByteDance Seed et al., arXiv:2510.25741) "Scaling Latent Reasoning via Looped Language Models" | Parameter-shared looped decoder blocks applied recurrently during pretraining (R4 primary). Entropy-regularized objective + exit gate (KL to uniform/geometric prior) for learned/adaptive depth allocation. Two-stage training (entropy pretrain → loss-improvement fine-tune). Extrapolates beyond training depth. Strong on manipulation/composition (multi-hop, modular arith, graph reasoning). | Entropy regularization for breadth/depth allocation is a natural training-time prior for "K-trajectory exploration" (GRAM/PTRM). Learned exit prevents overthinking on easy cases. Our current fixed micro-step + rehearsal lacks explicit entropy pressure on recurrence depth. Substrate for making stochastic mental simulation (K>1) native and depth-adaptive. |
| May 2026 | Equilibrium Reasoners (EqR) (Huang, Geng, Kolter, arXiv:2605.21488) "Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning" | Task-conditioned attractors: latent dynamical system `z_{k+1} = f_θ(z_k; x)` whose stable fixed points = valid solutions. Training interventions (randomized init + noise for path stochasticity) shape broad, reachable, stable basins. Scaling axes: Depth (more iterations) + Breadth (stochastic trajectories from multiple inits/noise). Diagnostic: fixed-point residual `‖f(z;x) - z‖` strongly tracks task error. Equilibrium internalization. Adaptive compute + learned halting. Verified on Sudoku-Extreme, Maze-Unique, Mini-ARC (tiny model >99% with depth+breadth; transfers to Transformer backbone). | Perfect mechanistic explanation for our RI-1 failure: scaling works when attractors are well-shaped (broad correct basins); fails when substrate produces narrow/spurious/misaligned attractors or training does not shape the landscape. "Depth + Breadth" is exactly our GRAM/PTRM K-trajectory + recurrence depth. Brain-mimetic sampler (K trajectories scored by attractor + provenance) is a step toward this; missing explicit fixed-point convergence pressure and residual-as-diagnostic. EqR training recipe (noise + randomized init) directly upgrades our stochastic sampler. |
| May 2026 | Solve the Loop: Attractor Models (Fein-Ashley, Rashidinejad, arXiv:2605.12466) "Attractor Models for Language and Reasoning" | Two-module: Backbone (high-capacity causal Transformer) produces strong initial proposal `ỹ₀`. Small Attractor module solves fixed-point `ỹ⋆ = T_θ_a(ỹ, ỹ₀)` (or root of A(ỹ, ỹ₀)=0) via root-finding solver (Anderson acceleration). Persistent injection of proposal. Training: next-token loss on decoded fixed-point + implicit function theorem (constant memory, no full unroll). Inference: tunable tolerance solver. "Equilibrium internalization": backbone proposal moves closer to fixed point during training → solver can be removed or heavily reduced at inference with little loss. Strong LM scaling (140M-770M Pareto beats larger Transformers) + exceptional hard reasoning (tiny 27M: 91.4% Sudoku-Extreme, 93.1% Maze-Hard where frontier models ~0). | Directly solves "tight micro-step + rehearsal prevents stable high-depth learning". Fixed-point solver + implicit training = stable, memory-efficient deep iteration. Proposal injection from backbone = principled way to give the attractor a good starting point (our current init is often uninformative or from thin hybrid). Internalization curriculum explains why short-run signals appear but long-horizon collapse: without it, model never learns to "think less when possible". GRAM/PTRM stochastic breadth maps to stochastic initialization of the attractor solve (multiple y0 or noise in solver) + selection among converged trajectories. Brain triple memory (working + attractor + provenance) is an ideal structured state for the attractor module to operate over. |

**Additional context papers (recurrent-depth roots and related)**:
- LoopFormer (ICLR 2026): Elastic-depth via trajectory conditioning (t + Δt) + shortcut-consistency loss (align short vs long rollout final state, stop-grad on long). Our Attractor + M1 is natural substrate; missing explicit consistency objective during variable-depth training.
- Parcae, DEQ (Deep Equilibrium Models) family: Fixed-point / equilibrium framing, implicit differentiation, stable looping.
- Recurrent-depth / Looped Transformers literature (2025-2026 wave): Huginn, Ouro, EqR, Solve-the-Loop as the direct mechanistic upgrades over earlier Universal Transformer / weight-tied attempts.

**Synthesis map to local failure**:
The current substrate tries to achieve "iterative improvement" via tight per-micro-step hybrid updates + external rehearsal pressure on buffers. This prevents the model from learning *stable attractor dynamics* or *convergence to solution-aligned fixed points*. Stochastic breadth was historically strong because it injected K-trajectory exploration (breadth) during recurrence; its incomplete port + application on a non-attractor substrate produces high-variance non-monotonic signals. The 2026 papers show that *explicit attractor/fixed-point machinery + training interventions that shape the basin landscape + internalization curriculum + depth+breadth scaling axes* produce reliable monotonic gains and dramatic hard-reasoning jumps even in tiny models. This is the "genuinely different causal route" required after exhaustive falsification of the exhausted family.

---

## 4. Proposed Architecture: Brain-Attractor-Centric Recurrent Architecture (QTRM-EqA)

**Name**: QTRM-EqA (Equilibrium Attractor extension of QTRM with brain-mimetic triple memory) or Brain-Attractor-Centric Recurrent Core.

**High-level causal story (humanistic preflight per skill)**:
Reader (Qwen or native backbone) encodes the prompt into a strong initial proposal (backbone output). This proposal seeds a structured brain-mimetic memory state (ActiveWorkingMemory as multi-stream "mental workspace", StabilizingAttractorMemory as slow depth-wise pull toward coherent solutions, ProvenanceEpisodicMemory as causal grounding filter). Every recurrent step runs K structured mental simulations ("what-if trajectories") inside Working Memory; these are evolved by working-memory rules, scored by a combination of attractor stability (convergence residual / monotonic improvement), provenance consistency, and a small learned scorer. The best trajectory (or weighted ensemble) is selected and used to update the memory state. The attractor module treats the memory state as the dynamical system variable and drives it toward a task-conditioned fixed point using root-finding / implicit-style iteration (not fixed micro-steps). Training uses variable convergence budget (sampled depth + noise for breadth), entropy regularization on depth allocation, and an internalization curriculum (progressively reward the backbone proposal for landing closer to equilibrium so the expensive solver can be skipped). The final converged memory state (or its projection) is read out through the normal one-body LM head. At inference, depth and breadth (K) are free variables; easy problems converge fast / with low K (internalization); hard ones scale both axes. Destructive ablation of the attractor solver, the stochastic sampler (K=1), or any memory tier must drop the same strict-B metric.

**Reader → Memory/State → Thinker/Transition → Checker/Search → Speaker (one-body)**:
- Reader: Qwen (or native) prefix layers / embeddings → initial proposal vector(s) injected persistently into attractor solve.
- Memory/State: BrainMimeticTripleMemory (ActiveWorkingMemory multi-stream + StabilizingAttractorMemory + ProvenanceEpisodicMemory) as the *primary recurrent state carrier*. Stochastic mental simulations live and evolve here.
- Thinker/Transition: Attractor solver (Anderson / fixed-point iteration / learned f_θ) operating on the triple memory state, with proposal injection. Brain-mimetic stochastic sampler (K trajectories) as first-class breadth mechanism inside every solver step.
- Checker/Search: Implicit in the attractor convergence (residual as confidence) + provenance gating + learned scorer. (Future: explicit verifier can condition the attractor loss.)
- Speaker: Projection of final converged memory state (or best trajectory) → normal LM head (Qwen or native) → autoregressive tokens. No side renderer.

**Key upgrades over current brain_triple_memory + hybrid**:
1. Attractor solver as the recurrence primitive (not tight hybrid micro-steps). Fixed-point / root-finding framing replaces "run N micro hybrid steps".
2. Backbone proposal as strong, persistent y0 injection (addresses uninformative init).
3. Stochastic breadth (K) as native structured hypothesis generation inside the solver loop, modulated by all three memory systems (deep GRAM/PTRM realization).
4. **Predictive Data Intuition (new, 2026-06 intuition-driven addition)**: Lightweight JEPA-style next-embedding predictor (V-JEPA2 / LLM-JEPA / LeWM 2026 spirit) running on the triple memory state. Produces continuous surprise/prediction-error signal (Titans/ATLAS style). This signal now participates in:
   - Trajectory scoring & selection inside the stochastic sampler
   - Strength/direction of attractor stabilization
   - Output modulation toward data-plausible directions
   This is the explicit mechanism for building genuine "데이터에 대한 직관".
5. Explicit convergence residual + internalization objective (backbone learns to propose near-equilibrium).
6. Variable convergence budget training (Huginn-style sampling + Ouro entropy reg + EqR noise interventions) + shortcut-consistency (LoopFormer).
7. Memory tiers remain first-class (brain-mimetic redefinition preserved and elevated).
8. Full ablation contract: attractor_solver_off, stochastic_breadth_off (K=1), data_intuition_ablation_zero, working_memory_zero / attractor_memory_zero / provenance_zero, proposal_injection_off.

**Treatment of "Causal Intuition" (인과성 직관)**:
Causal sensitivity is not promoted to a new standalone RI-8 at this stage. Instead, it is defined as a required emergent property of the Predictive Data Intuition + structured stochastic mental simulation + attractor system. Concrete diagnostics (intervention consistency, what-if counterfactual coherence in latent trajectories, and surprise-to-provenance causal correlation) are mandatory accompanying evidence whenever data intuition claims are made. This keeps the core RI-1~RI-7 list stable while making causal feeling testable rather than rhetorical.

**Training schedule with convergence budget (concrete)**:
- Sample effective depth / iteration budget per batch/sequence from a suitable distribution (log-normal Poisson or curriculum ramp, per Huginn + Ouro).
- Add noise/randomized init to shape attractor basins (EqR).
- Entropy regularization on a learned exit / depth gate (Ouro).
- Internalization loss: KL or regression between backbone proposal and the converged state (progressive weight).
- Shortcut-consistency: align short-budget vs long-budget final memory states (stop-grad on long).
- GRAM/PTRM-style posterior guidance when answer signal is available (in rehearsal or preference stages).
- Loss on decoded final state through normal LM head (one-body).
- Truncated / implicit differentiation for memory efficiency at high budgets (Solve the Loop style).

**Device / implementation hygiene (from current code lessons)**:
- Central `_ensure_same_device` on all TripleMemoryState, sampler, and layer calls.
- Attachment to model must force state .to(device) and protect against iteration / _forward_unimplemented.
- K=4 default with diversity preservation; ablatable to K=1.

This is not "add another module". It replaces the recurrence engine while preserving the brain-mimetic memory redefinition and all RI contracts.

---

## 5. Evolved Architecture Synthesis (2026-06): Integrating Latest Sparse + Neural Long-Term Memory Research

**Trigger**: Recognition that the initial Brain-Attractor-Centric proposal (Section 4) strongly advanced the *recurrent thinker* (data intuition, attractor convergence, structured stochastic breadth) but had not yet deeply incorporated first-class **sparse selective persistent memory** mechanisms (MSA, Raven, LM², Titans-style neural LTM) that are explicitly required by RI-4 and historically strong in the project.

**Goal**: Create a multi-scale memory system that preserves the brain-mimetic philosophy while adding scalable, interference-resistant, selectively updatable long-term memory, drawing maximally from the latest 2025–2026 literature.

### Key Latest Papers Referenced (2025–2026)

- **MSA (Memory Sparse Attention)** (arXiv:2603.23516, March 2026, EverMind-AI et al.): End-to-end trainable sparse attention + latent memory framework. Document-wise RoPE, KV compression + tiered storage, Memory Interleaving for multi-hop. Scales to 100M tokens with <9% degradation. Emphasizes decoupling memory capacity from core reasoning while keeping differentiability.
- **Raven / Routing State Model (RSM)** (arXiv ~2602.24281, 2026, Goomba Lab / Eric Xing et al.): Memory organized as independent persistent slots + learned sparse router (top-k). Unselected slots remain completely untouched (strong anti-interference). Unifies dense (SSM-like) and sparse behaviors.
- **LM² (Large Memory Models)** (arXiv 2502.06049, Feb 2025): Transformer + explicit memory bank of slots with cross-attention read + LSTM-style gating (input/forget/output gates) for selective write. Strong gains on long-context reasoning and multi-hop.
- **Titans + ATLAS + MIRAS** (Google, 2025): Neural long-term memory module updated at test time via surprise (gradient-based). Gating, momentum, windowed optimization. Complementary to attention (short-term) vs. parametric memory (long-term).
- **EM-LLM** (ICLR 2025): Episodic memory via Bayesian surprise for online event segmentation + hierarchical retrieval. Mimics human episodic structuring.
- **Native Sparse Attention (NSA)** (DeepSeek, 2025): Hardware-aligned, natively trainable sparse attention (hierarchical token modeling + top-k + compression).
- Supporting: Sparse Selective Caching (SSC), G-MemLLM (gated latent memory banks), Memory Caching with routers.

### Proposed Evolved Name
**BMSAM** — Brain-Mimetic Sparse Attractor Memory (or QTRM-EqA-M: Memory-Augmented Equilibrium Attractor)

### High-Level Architecture (Multi-Scale Memory)

**Reader → Multi-Scale Memory System → Attractor Solver + Mental Simulation → Speaker (one-body)**

The memory system is now explicitly **multi-scale**:

1. **Fast Active Layer: BrainMimetic Working Memory + K Mental Simulations** (unchanged core from Section 4)
   - Multi-stream workspace for live "what-if" thinking.
   - Now augmented with Predictive Data Intuition (surprise signal) for data sense.

2. **Medium Stabilization Layer: StabilizingAttractorMemory + Predictive Data Intuition**
   - Drives convergence to solution-aligned + data-plausible fixed points.
   - Surprise signal shapes basins and trajectory selection.

3. **Long-term Persistent Layer: Sparse Gated Memory Slots (Raven + LM² + MSA inspired)**
   - **Slot Organization**: Fixed or dynamically allocatable latent memory slots (inspired by Raven + LM² 2k-slot bank).
   - **Sparse Selective Router**: Learned top-k router (Raven + NSA style) decides which slots the current thought state attends to / writes to. Unselected slots have near-zero update (persistence = 1.0).
   - **Gated Read/Write**: Cross-attention read from current triple memory state + LSTM/GRU-style gates (input gate, forget gate, output gate) for selective write (LM² + G-MemLLM style). Surprise from Predictive Data Intuition strongly influences gate values.
   - **Episodic Structuring**: Surprise / prediction error triggers boundary detection (EM-LLM style). High-surprise periods can spawn or commit to new "episodes" (groups of slots).
   - **Efficient Sparse Attention (MSA 2026 style)**: When the router selects many slots/segments, use document-wise RoPE, chunk-wise KV compression, tiered storage (fast slots on GPU, older on CPU), and Memory Interleaving for multi-hop reasoning across scattered memory.

**Information Flow Example**:
- Current thought state (from Working Memory + best trajectory) queries the Sparse Gated Memory Slots via the router.
- Relevant slots are read via gated cross-attention → injected into Working Memory and Attractor.
- After thinking step + surprise computation, gated write decides which slots to update (high-surprise + high-utility events get written; routine events are forgotten or left untouched).
- Attractor convergence happens over both fast working memory and selected long-term slots.

### Integration with Existing Strengths

- **GRAM/PTRM Stochastic Breadth**: Now operates across scales. K trajectories can include "queries" into the sparse long-term slots. Surprise modulation makes breadth data-aware and memory-aware.
- **One-Body Causal Path**: All memory (fast + persistent slots) ultimately influences the final converged state that feeds the LM head. No sidecar.
- **RI Contracts**: 
  - RI-4 (Sparse Selective Memory) is now natively satisfied by the Raven/LM²-style router + persistence.
  - RI-2 (Long-Horizon Stability) benefits from both attractor dynamics and interference-free persistent slots.
  - Ablations remain clean: router_off (dense update), gate_ablation, slot_persistence_off, data_intuition_ablation, etc.

### Training Implications (Latest Paper Informed)

- Primary loss: one-body LM + attractor convergence + Predictive Data Intuition prediction loss.
- Auxiliary: sparse routing load balancing loss, gate entropy regularization (encourage selective but not collapsed writes), episode boundary consistency (from EM-LLM).
- Variable convergence budget now also varies how deeply the model queries the long-term slot layer.
- Surprise-driven curriculum: high-surprise events from the data intuition predictor get higher write priority during rehearsal.

This synthesis maximally incorporates the 2025–2026 advances in sparse attention, neural long-term memory, gated episodic memory, and scalable latent memory while staying faithful to the project's brain-mimetic + attractor + raw intelligence necessary conditions framework.

**Implementation Status (as of latest session — "직관대로" tightening)**:
- Created dedicated first-class module `SparseGatedLongTermMemory` (clean Raven + LM² + surprise wrapper).
- `BrainMimeticTripleMemory` now has proper multi-scale memory with **surprise as the central coupling signal**:
  - Long-term memory read (surprise-modulated) participates in:
    - Trajectory generation & selection (K mental simulations shaped by relevant long-term context + surprise).
    - **Attractor stabilization itself** (new explicit `_stabilize_attractor`): long_term_context + current_surprise now *always* bias the slow stabilization pull in a clean, symmetric way (both stochastic and non-stochastic paths). High surprise → stronger long-term voice in the slow thinker layer.
  - Final surprise from PredictiveDataIntuition drives long-term write strength (Titans/EM-LLM style consolidation).
  - **Training objective now includes the data intuition loss** (minimal viable wiring in the active continuation trainer): `compute_data_intuition_loss` contributes a small weighted JEPA-style prediction error term. This is what actually *trains* the system to develop an internal sense of the data dynamics (surprise minimization is no longer diagnostic-only).
  - Full cross-step persistence of long-term slots via `get_long_term_state` / `set_long_term_state` + post-step snapshotting (slow memory survives sequences and checkpoints).
- `enable_long_term_surprise_driven_memory()` auto-activates the full stack when `--brain_triple_memory` is passed (with strict ablation_zero everywhere).
- Checkpoint persistence for the slow memory: trainer now saves/restores `brain_triple_state` (TripleMemoryState) + `long_term_slots` alongside model/router. This makes the "slow persistent layer" actually survive training steps and resumes (critical for long-horizon data intuition to compound).
- Final one-body readout now receives a small surprise-modulated summary from long-term memory at the end of every step() (via `_get_long_term_summary` + gated residual into `modulated`). The LM head itself "feels" the slow persistent memory + data intuition. Still strictly one-body + fully ablatable.
- All RI contracts (one-body, ablation_zero, Principle Gate readiness) preserved.

The surprise-mediated loop (long-term read → K-trajectories + attractor stabilization → final surprise → surprise-modulated long-term write) is now structurally tight and end-to-end differentiable. This is the strongest executable realization of the brain-mimetic + 2025-2026 literature synthesis to date.

---

## 5. Smallest Falsifiable Diagnostic (Execution Plan)

**Per skill + conversation**:
- Short (8-15 step) diagnostic continuation from the strongest available brain-mimetic or hybrid anchor checkpoint that produced the best prior d=8 signal under matched conditions.
- Trainer: extend `scripts/train_hybrid_ri4_real_continuation_minimal.py` (or successor) with `--brain_attractor_centric --variable_convergence_budget --attractor_solver_mode [fixed_point|anderson|learned] --stochastic_k 4 --proposal_injection --internalization_weight 0.1` (and off variants).
- Immediate measurement after run: full Principle Gate + strict-B depth sweep (effective depth 1/4/8/12/16) on pure_72 heldout (use `--heldout_max_cases` for quick iteration if timeout).
- Required for promotion: clear material improvement in d=8 (or d=12) accuracy + monotonicity vs the best prior anchor (40.28% or the 47.22% short spike), with new components active. Ablations (solver_off, K=1, any memory tier zero) must produce clear drop on the same gate.
- Triple-Track: A (ablation matrix), B (reasoning + memory narrow heldouts), C (quantitative loss descent + convergence diagnosis with/without new components).
- Honest conditions-matched tagging throughout.

**Success signal**: First evidence that an attractor/equilibrium-oriented substrate with deep brain-mimetic GRAM/PTRM stochastic mental simulation produces stable, repeatable, monotonic RI-1 depth scaling on strict-B under full RI contracts — where the previous substrate family could not.

**Kill / archive signal**: Reproduces the same non-monotonic / plateau signature despite the substrate change. Archive and return to parallel fast-falsification (next candidate from literature or coarser hybrid replacement).

---

## 6. References (All Papers and Internal Sources Cited)

**Primary 2025-2026 Attractor / Looped / Equilibrium Literature (the "2번" synthesis sources)**:
- Geiping et al. (2025). "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (Huginn). arXiv:2502.05171.
- ByteDance Seed et al. (2025, updates 2026). "Scaling Latent Reasoning via Looped Language Models" (Ouro / LoopLM). arXiv:2510.25741. https://ouro-llm.github.io/
- Huang, Geng, Kolter (2026). "Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning" (EqR). arXiv:2605.21488. https://arxiv.org/abs/2605.21488 ; https://github.com/locuslab/EqR
- Fein-Ashley, Rashidinejad (2026). "Solve the Loop: Attractor Models for Language and Reasoning". arXiv:2605.12466. https://attractor-models.github.io/ ; https://github.com/jacobfa/Attractor
- LoopFormer (ICLR 2026): Elastic-depth transformers with trajectory conditioning and shortcut-consistency loss (referenced in project roadmap; full citation to be added on confirmation).

**Related / Foundational**:
- Deep Equilibrium Models (DEQ) family (Bai et al. and follow-ups).
- Parcae (stable recurrence / diagonal decay injection).
- Universal Transformers / weight-tied recurrent cores (pre-2025 roots).
- HRM / HRM-Text (recurrent latent reasoning reference standard per skill).

**Project Internal (RI principles, diagnosis, GRAM/PTRM, brain-mimetic implementation)**:
- `experiments/ri1_m1_reports/Stubborn_Negative_Pattern_Diagnosis_20260528.md` (and Consolidated_RI1_M1_Analysis, Multi_Direction_Exploration_Table).
- `docs/roadmaps/RI_Raw_Intelligence_PoC_Execution_Plan_2026-06.md` (P1.4 RI-1 section with early Huginn/LoopFormer citations).
- `docs/wiki/architecture/inductive-bias-map.md` (Stochastic Recurrent Breadth / GRAM/PTRM entry + Reverse I→G→A).
- `src/qtrm_mm/memory/brain_triple_memory.py` (BrainMimeticTripleMemory + BrainMimeticStochasticSampler full implementation; deep GRAM/PTRM reinterpretation).
- `src/qtrm_mm/config.py` (brain_triple_memory_enabled, brain_mimetic_stochastic_enabled, k, ablation_zero flags).
- `scripts/train_hybrid_ri4_real_continuation_minimal.py` (trainer wiring + attachment).
- `docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md`, `one-body-architecture-ssot.md`, `FAIR_COMPARISON_PROTOCOL.md`.
- Multiple Principle Gate JSONs under `experiments/matched_port_evaluation_a9617cd8/` (stochastic_breadth + 3-track + conditions-matched evidence).
- `docs/wiki/concepts/recurrent-depth-transformers.md` (prior QTRM design ideas on looped stability, input injection, depth sweeps, entropy regularization).
- Conversation history (2026-05-28 through 2026-06 "2번" + brain redefinition + GRAM/PTRM confirmation).

**Additional project SSOTs and tools**:
- research-driven-architecture-debugging skill (Stubborn Negative Pattern Protocol, Parallel Fast-Falsification, I→G→A + Reverse I→G→A, Triple-Track A+B+C, humanistic preflight, latest-first literature sweep).
- Principle Gate validator + strict-B pure_72 harness.
- All RI-1~RI-7 necessary conditions SSOTs.

---

**Next Immediate Action (per user "직관대로 해" repeated directive)**: The core surprise-coupling + multi-scale memory loop (fast K + slow attractor + long-term gated persistent + real training signal for data intuition) is now implemented and tightened per the architecture vision. Next: launch a short falsifiable diagnostic continuation (from strongest available anchor) using `--brain_triple_memory --brain_mimetic_stochastic --data_intuition_loss_weight 0.04` (and ablation variants), run strict-B depth sweep + Principle Gate on pure_72, and record whether the deeper substrate + actual surprise minimization produces the first robust monotonic RI-1 signal where previous families could not. Update this doc + new decision record with honest results. All RI contracts from line 1.

This document is the authoritative synthesis and proposal for the "2번" substrate direction. Update on every new measurement or literature addition.

---

**2026-06 Light Native Full-Stack Eval Hardening (for non-freezing 72 heldout)**

After repeated system hangs on native full-stack 72 (`--run_72_heldout_only` + brain_triple_memory + stochastic K + long-term + surprise), the following multi-layer engineering optimization was applied **before any further heavy measurement** (per user: "진짜 네이티브 full stack 은 최적화가 필요해 최적화 부터 해봐 컴퓨터 안멈추게"):

- `BrainMimeticTripleMemory.set_light_eval_mode(True)` API added + internal guards:
  - `stochastic_k` forced to 1 inside `step()`
  - All incremental surprise-driven long-term writes (`high_surprise` consolidation path) completely disabled
  - `compute_data_intuition_loss` returns zero loss during light eval
- Trainer (`train_hybrid_ri4_real_continuation_minimal.py`):
  - Light mode activation moved **after** all attachments (stochastic integration + long_term enable) — previous timing bug fixed
  - `_compute_narrow_heldout_accuracy` now wrapped in `torch.no_grad()`
  - Per-case `gc.collect() + empty_cache()` + hard max_cases cap to 8 when 72-only + brain
  - All long-term persist/write logic skipped during pure 72 measurement
  - Causal probes fully disabled in 72-only path
- Risk Predictor + pre-flight cleanup strengthened
- Result: "Real native" (modules attached, read paths + surprise signal + attractor influence still participate) but write paths + K-breadth + grad graphs eliminated.

This is the minimal honest native full-stack path that can be measured on current hardware without immediate freeze. Still requires `--max-cases 8` + `--accept-freeze-risk` + close monitoring.

All changes preserve full ablation contract and one-body principle.

(End of architecture proposal MD. All referenced papers and internal sources explicitly listed above as requested.)

---

## D. Titans / ATLAS / Griffin Deep Dive (User "D" — Equation-Level Mapping for Native Full-Stack Redesign)

**User directive**: "D" (Titans/ATLAS/Griffin 논문을 더 파서 구체적인 수식/알고리즘 수준까지 매핑) after exhaustive optimization attempts (A-1/A-2/A-3/1번/Option A/websearch) all failed to make "진짜 native full-stack 72" runnable without 5min/step stall, 1-2% GPU, system hang.

**Root local failure (per research-driven-architecture-debugging skill + stubborn negative pattern)**:
The per-micro-step "BrainMimeticTripleMemory participation" (K stochastic trajectories in working memory + _stabilize_attractor + surprise-modulated long-term router read/write + PredictiveDataIntuition + provenance injection) is implemented as an **external heavy Python state machine** (`triple.step(...)` called synchronously from trainer loop after every `OneBodyParallelHybridBlock` forward, with router .item()/topk/scatter, per-step empty_cache, Python dispatch). This destroys CUDA graph, causes massive sync, and makes native participation (the RI-1/RI-4 contract) unmeasurable at scale. All Python-side bypasses (stride, clean-skip, light mode K=1, B=8 batch, hybrid-only compile, Option A eager brain split) were exhausted and still left GPU at 1% with 5min/step.

**Latest-first mechanism extraction (exact equations, not slogans)**:

**Titans (arXiv:2501.00663, Behrouz et al., Google, "Learning to Memorize at Test Time")**:
- Core: Neural long-term memory ℳ (deep MLP, L_ℳ ≥ 2) as *meta in-context learner* updated at test time via surprise.
- Surprise metric (momentary + momentum past):
  ```
  S_t = η_t * S_{t-1}  -  θ_t * ∇ℓ(ℳ_{t-1}; x_t)     # momentum as "memory of surprise"
  ℳ_t = (1 - α_t) * ℳ_{t-1} + S_t                    # weight decay α_t as forget/gate
  ```
- Associative memory loss (key/value):
  ```
  ℓ(ℳ; k_t, v_t) = || ℳ(k_t) - v_t ||_2^2
  k_t = x_t W_K, v_t = x_t W_V
  ```
- Retrieval: y_t = ℳ^*(q_t)   (forward without weight update).
- Variants: MAC (memory as context injected to attention), MAG (gated with SWA short-term), MAL (memory as layer).
- Parallel training: chunk-wise matmul reformulation of momentum + decay GD (no full unroll).
- Key for us: surprise is *gradient signal on associative loss*, not ad-hoc; deep memory for capacity; momentum + decay = natural persistence + pruning.

**ATLAS (arXiv:2505.23735, Behrouz et al., direct follow-up)**:
- Solves Titans' "online c=1 per-token" limitation.
- **Omega rule** (the central equation — context-window optimization, not per-token):
  ```
  min_ℳ   ∑_{i=t-c+1}^t  γ_i^{(t)}  || ℳ( ϕ(k_i) ) - v_i ||_2^2
  ```
  (c = context window size; γ_i^{(t)} = input-dependent hard gates for pruning; ϕ = feature map).
- Update (gradient form):
  ```
  ℳ_t = α_t ℳ_{t-1}  -  η_t  ∑_{i in window} ∇ℓ(...)
  ```
- Polynomial feature map for **super-linear capacity**:
  ```
  ϕ_p(x) = [x^β] for |β|≤p     →  O(d_k^p) independent KV pairs (vs O(d) linear)
  ```
  (recovers softmax attention as p→∞ limit in some cases; DeepTransformers are strict generalization of Transformer).
- Atlas variant uses **Muon optimizer** (Newton-Schulz 5 iterations) on the memory objective for "locally optimal" (second-order approx) updates instead of plain GD.
- Parallelization: sliding-window masking + chunk-wise (no materializing full c gradients when c large).
- Direct attack on our stall: long-term memory no longer updated every micro-step or fixed stride; updated over Omega windows with pruning γ. Surprise only triggers meaningful writes at boundaries.

**Griffin / Hawk (arXiv:2402.19427, De et al., Google DeepMind, 2024)**:
- RG-LRU (Real-Gated Linear Recurrent Unit) — lightweight *internal* per-step recurrence with fixed small state:
  ```
  r_t = σ(W_a x_t + b_a)          # recurrence gate (data-dep)
  i_t = σ(W_x x_t + b_x)          # input gate
  a_t = a^{c * r_t}               # (c=8 constant; data-dep decay on recurrence)
  h_t = a_t ⊙ h_{t-1} + √(1-a_t²) ⊙ (i_t ⊙ x_t)
  ```
- Hawk = pure RG-LRU blocks + MLP (no attention).
- Griffin = hybrid: RG-LRU recurrent blocks + local sliding-window attention (window=1024 typical) for recent context sync. Alternating pattern (2 recurrent + 1 local attn).
- Training: custom Pallas linear-scan kernel (not associative scan or conv) because memory-bound elementwise; achieves training parity with MQA Transformer while inference massively faster (fixed state, no growing KV).
- Inference: dramatically higher throughput + lower latency at long sequences vs MQA (cache = fixed hidden state vs linear KV).
- Extrapolation: excellent on length > train when local attn present.

**Mechanism → Local Failure Table (per skill "latest-first sweep + map to bottleneck")**:

| Date | Source | Exact Mechanism | Local Failure Explained (native stall) | Candidate Implication for v2 |
|------|--------|-----------------|---------------------------------------|------------------------------|
| 2025-01 | Titans | Surprise = ∇ℓ(associative) + momentum S_t + α decay; deep MLP LTM; MAC/MAG/MAL injection | External Python triple.step + router every micro-step is the "heavy boundary" that destroys graph and makes participation unmeasurable | Move Titans-style surprise neural LTM + momentum/decay into block-internal or clean attached module; use MAC-style for fast working+attractor path |
| 2025-05 | ATLAS | Omega rule (window-c optimization of memory loss with γ pruning + ϕ polynomial + Muon) | Current long-term write/read is online c=1 (or fixed stride) → constant Python dispatch + router cost on *every* step | Replace per-micro long-term router with Omega-window sparse updates (c=4/8 or surprise-boundary); polynomial ϕ + deep memory for capacity without explosion |
| 2024-02 | Griffin | RG-LRU: data-dep gated linear recurrence (r_t, i_t, a_t = a^{c r}) + local SWA hybrid; fixed-state; linear-scan kernel | "Per-micro-step BrainMimetic participation" (K sims + attractor stabilize + surprise injection) lives outside the compiled block as Python object | Implement fast working/attractor evolution as Griffin RG-LRU (or learned gated linear) *inside* OneBodyParallelHybridBlock. This makes native per-micro participation compiled, fixed-state, zero Python dispatch. Slow LTM injected sparsely via Omega |

**Humanistic preflight (plain language, per skill)**:
The current engine is a fast hybrid mixer that occasionally "phones home" to a big external brain object for "real thinking + memory + intuition". Every phone call costs a full Python round-trip + graph break + sync. The brain papers show the correct pattern: the *fast light thinking* (Griffin RG-LRU) should be a tiny native circuit *inside* the block that runs every micro-step for free. The *slow deep memory + data intuition* (Titans surprise neural LTM + ATLAS Omega) should be a separate, sparser, window-optimized module that only wakes up when the data actually surprises it, not on every clock tick. This separation + internalization is the only way "per-micro-step native participation" becomes measurable at 72 scale without the computer freezing.

**Proposed Architecture: Hybrid Brain-Mimetic Recurrence v2 (HBM-R v2 or "Griffin-Brain Native Core")**

**Causal path (reader → memory/state → thinker → speaker, one-body)**:
- Reader: Qwen/native prefix → initial proposal injected persistently (as in current attractor proposal).
- **Fast internal path (per-micro, native, compiled)**: Inside OneBodyParallelHybridBlock (or dedicated RecurrentCoreBlock):
  - Griffin-style RG-LRU (or learned variant with 2-4 gates) evolves lightweight working + attractor state *every micro-step*.
  - BrainMimetic working memory streams + K=1..4 mental simulations evolve inside this recurrence (structured stochastic breadth, no external sampler).
  - Local SWA (or the existing mixer) provides recent context sync.
  - This replaces the "eager triple.step after every forward" with a real internal recurrence. `set_brain_triple_memory` now wires *influence gates* and *surprise injection ports* into the RG-LRU, not an external object.
- **Slow sparse path (Omega surprise neural LTM)**:
  - Titans-style deep MLP neural memory (MAC/MAG style) + ATLAS Omega rule (c=4/8/16 window, γ pruning, polynomial ϕ or deep).
  - Predictive Data Intuition surprise + momentum feeds the Omega objective.
  - Long-term slots (Raven + current SparseGatedLongTermMemory) updated only at Omega windows or high-surprise episode boundaries (not per micro or fixed stride).
  - Read from slow memory is *injected* into the fast RG-LRU state at low frequency (or via learned gate), exactly like current long_term_context but now with proper capacity and pruning.
- Speaker: Final converged fast state (or best trajectory) + gated slow summary → normal LM head. Full one-body + ablation contract preserved (rg_lru_off, omega_window_off, surprise_zero, slow_memory_zero, K=1, etc.).

**Training / Inference mode separation (critical for native 72)**:
- Training: richer (larger c windows, K>1 mental sims inside RG-LRU, full surprise loss).
- Inference / native eval: cached attractor states in RG-LRU, Omega c reduced or triggered only on high surprise, router frequency slashed, K=1 forced internally.
- This is exactly what `set_native_eval_mode` / `set_light_eval_mode` tried to hack from outside; now it becomes first-class block flags (`inference_mode=True` on the fast recurrence + sparse slow path).

**Implementation anchors (already partially present)**:
- `src/qtrm_mm/blocks.py:421` — `set_brain_triple_memory(..., inference_mode)` + light_update sketch + comment "heavy Python boundary every step" — complete this migration.
- `brain_triple_memory.py` — keep the high-level brain-mimetic *roles* (working/attractor/provenance + DI surprise) but re-implement the fast evolution as RG-LRU citizen and slow write as Omega module.
- Trainer 72 loop + Option A split — after v2, the "eager brain step" path largely disappears for fast path; only sparse slow memory touches remain.

**Falsifiable gates (per skill + RI contracts)**:
- Small diagnostic continuation: `--brain_native_griffin_core --omega_window 8 --surprise_momentum --rg_lru_internal` vs prior best anchor.
- Strict-B pure_72 depth sweep (d=1/4/8/16) with full Principle Gate + Triple-Track (A ablation matrix, B reasoning+memory narrow, C loss descent).
- Required: d=8 (or 12) monotonic gain + material lift vs 40.28% / 47% short spike, with new components causal (rg_lru_off / omega_off / slow_zero drop the same metric).
- Native 72 heldout now runnable at B=16+ without 5min/step or stall (the original user request).

**Why this is the "big jump" after exhaustive local opt**:
All prior attempts (router stride, clean skip, light K=1, compile Option A, batched, empty_cache, etc.) were fighting the symptom (external Python per-step dispatch). The papers give the causal cure: move fast participation inside a lightweight compiled recurrence (Griffin), and make slow surprise LTM sparse and window-optimized (ATLAS Omega + Titans surprise). This is substrate-level, not another bypass.

**Next immediate action**:
1. Prototype RG-LRU (or minimal gated linear) citizen inside OneBodyParallelHybridBlock (reuse existing set_brain hook + light_update).
2. Implement minimal Omega-window long-term writer (reuse current SparseGated + surprise signal, add c-window + γ).
3. Short 8-20 step diagnostic continuation from strongest brain-mimetic anchor.
4. Immediate strict-B + native 72 measurement on the new artifact (now feasible because fast path is internal).

This D dive + v2 proposal is the direct, equation-backed answer to "native full stack 은 최적화가 필요해" and "아키텍처 개선을 하면 되는거야?" after all engineering avenues were exhausted. Update this wiki + inductive-bias-map + component_registry with the new contract once the smallest falsifying diagnostic runs.

(End of D section — user directive "D" fully executed at equation + redesign level.)

---

## E. Inference-Centric Re-examination (User Follow-up: Memory + Inference Full Architecture Review)

**User directive**: After the initial D deep dive on Titans/ATLAS/Griffin for the memory participation / native stall problem, broaden the lens: "아키텍처 좀더 검토해서 메모리 쪽 뿐만아니라 추론 쪽도 재점검 해보자 논문들 더 파고들어어서".

This section re-examines the entire architecture with primary emphasis on **inference** (serving latency, generation throughput, long-sequence extrapolation, native eval practicality, training-vs-inference divergence) while integrating the memory-side lessons.

### Current Inference Reality in the Codebase (Diagnosis)

From inspection of `train_hybrid_ri4_real_continuation_minimal.py`, `brain_triple_memory.py`, and `blocks.py`:

- Multiple "modes" exist as patches: `set_light_eval_mode`, `set_native_eval_mode`, `set_ultra_fast_measurement_mode`, `_brain_step_interval`, router stride, surprise interval=6, K=1 force, long-term write disable, etc.
- These are all **defensive hacks** because the core participation (K trajectories, attractor stabilization, surprise DI, long-term router) lives in a heavy external Python object (`BrainMimeticTripleMemory.step` / `light_update`).
- Even with all optimizations, "real native" 72 remains impractical for anything beyond tiny cases.
- No clean, first-class **fixed-size recurrent state** contract for serving/generation.
- Training and inference paths diverge only through runtime flags on the same heavy object (no architectural separation).
- State carry (`_triple_mem_state`, long-term slots) is complex and not optimized for low-latency autoregressive generation.
- The hybrid block has a good skeleton (`set_brain_triple_memory + inference_mode + early light_update call`), but the actual fast per-micro evolution is still not a true internal compiled recurrence.

**Root inference bottlenecks**:
1. External Python dispatch per (or every-N) micro-step for "thinking + memory".
2. Lack of fixed-size, constant-memory recurrence state (vs growing KV or complex triple + slot state).
3. High-frequency surprise / router / DI computation even in "optimized" native eval.
4. No large-chunk / sparse update strategy for the expensive neural memory adaptation at inference time.
5. Weak training / inference mode separation at the block and memory level.

### Additional Literature for Inference Focus (2024–2026)

Beyond the original Titans/ATLAS/Griffin trio, deeper dive into:

**RecurrentGemma (arXiv:2404.07839, 2024, DeepMind)** — direct production-scale follow-up to Griffin:
- Fixed-size recurrent state (RG-LRU) + local attention hybrid.
- **Inference reality**: Constant memory during generation (no KV cache growth beyond local window). Sampling throughput stays flat or improves with length, while Transformer (even MQA) degrades.
- Explicit benchmarks: significantly higher tokens/sec on long generations vs Gemma Transformer equivalents.
- Released optimized kernels (Pallas for TPU, PyTorch reference).
- Key lesson: The hybrid (light linear recurrence for history compression + local attention for recent context) is the practical inference winner.

**"Test-Time Training Done Right" / LaCT (arXiv:2505.23884, May 2025)** — addresses the exact hardware utilization problem we have been fighting:
- Previous TTT / neural memory at inference suffers from <5% FLOPS utilization because of tiny per-token or small-batch updates.
- **Core solution**: Extremely large chunk updates (2K → 1M tokens) for fast-weight (neural memory) adaptation at test time.
- Benefits: 50-70% GPU utilization with pure PyTorch (no heroic custom kernels), ability to scale nonlinear fast weights to ~40% of model parameters, easy use of strong optimizers like Muon at inference.
- Hybrid design: Large-chunk TTT for long-range compression + window attention for locality inside chunks.
- Natural for N-dimensional data (treat image/video grids as chunks).
- Directly relevant: Our "stride + sparse surprise" hacks are small-chunk thinking. LaCT shows the scalable path is **large coherent chunks** for the expensive memory adaptation.

Other reinforcing signals (from the sweep):
- Many 2025-2026 TTT papers treat fast weights explicitly as "recurrent states" that are written via gradient / delta / surprise rules at inference.
- The winning pattern is almost always **hybrid**: cheap recurrent / linear path for ongoing state evolution + more expressive (but sparser) neural memory adaptation.

### Synthesis: Inference-First Architecture Implications (v2.5 Proposal)

Combining everything:

**1. Fast Path Must Be Internal Fixed-Size Recurrence (Griffin / RecurrentGemma lesson + our earlier D implementation)**
- The `FastGatedLinearRecurrence` we just added inside `OneBodyParallelHybridBlock` is the correct direction. It should become the default per-micro "thinking" engine when in generation / native eval.
- Carry a small fixed recurrent state (h in RG-LRU style) explicitly across micro-steps in the answer_state_loop / generation loop.
- This gives true constant-memory autoregressive generation.

**2. Slow / Expressive Memory Adaptation Must Use Large-Chunk / Sparse Strategy (LaCT lesson)**
- Stop trying to do surprise-driven neural LTM or long-term slot writes on every micro-step or small stride.
- Define coherent "chunks" (e.g., 1K–4K tokens or logical thinking steps) and perform the heavy Titans/ATLAS-style Omega + surprise neural memory update **once per chunk** (or on high-surprise boundaries).
- Inside the chunk: rely on the cheap internal fast recurrence + local context.
- This matches both hardware reality and brain-like "consolidation during rest / after coherent experience".

**3. Strong Architectural Training / Inference Divergence**
- Training: richer participation (larger K inside fast recurrence if needed, more frequent or full Omega windows, full surprise loss).
- Inference / Serving / Native Eval: 
  - Fast path = internal RG-LRU style (always on, compiled).
  - Slow neural memory = large-chunk only, cached states, reduced precision if needed.
  - Clean `inference_mode=True` that switches both the block and the memory object to their serving-optimized implementations.

**4. State Contract for Serving**
- Define a clean "InferenceState" dataclass that is small and fixed-size (fast recurrent hidden + optional cached slow memory summary).
- The generation loop only carries this small state, not the full triple + all slots on every step.
- Long-term memory can be paged / offloaded more easily because updates are chunked.

**5. Hybrid Block as the Inference Primitive**
- OneBodyParallelHybridBlock (or its successor) should expose explicit `forward_inference(..., recurrent_state, slow_memory_summary)` with the internal fast recurrence as the main engine.
- This makes the "native full-stack" path for both training diagnostics and real serving the same clean object.

### Updated Recommendations & Next Concrete Steps

1. **Stabilize & Promote the Internal Fast Recurrence** (what we started in this session)
   - Make `FastGatedLinearRecurrence` first-class, with proper state carry support from the caller.
   - Add `recurrent_state` as an explicit optional argument/return in the hybrid block forward for generation paths.

2. **Implement Large-Chunk Omega Path** (next major piece after fast recurrence)
   - In the brain memory object (or a new `ChunkedNeuralMemory` module), add support for large-chunk surprise/Omega updates instead of per-step or small-stride.
   - Wire it so that during native eval / generation, heavy adaptation only happens at chunk boundaries.

3. **Clean InferenceMode Contract**
   - Make `inference_mode` on the block and on the memory object do more than just disable writes/K — actually swap to lighter internal paths and cached summaries.

4. **Measurement**
   - Once the above two are in, re-run native 72 with the new internal fast path + chunked slow memory.
   - Goal: realistic B=16–32 native heldout without heroic caps or 5-minute steps.
   - Add proper generation throughput / latency micro-benchmarks (even if synthetic) comparing against current patched modes.

5. **Wiki & SSOT Update**
   - This section + the previous D section become the new reference for "Inference + Memory Co-Design".
   - Update component_registry and one-body SSOT to reflect the preferred fast-internal + sparse-large-chunk slow path.

This broader re-examination confirms that the direction started in D (internal lightweight recurrence for fast participation + sparse high-quality neural memory for long-term) is the right one — but the "sparse" part must be **large-chunk**, not just "every N steps".

The combination of Griffin-style internal recurrence (constant state, high throughput generation) + LaCT-style large-chunk TTT/Omega neural memory adaptation (practical hardware efficiency for expressive memory at inference) gives us a coherent path to both strong raw intelligence (RI-1/RI-4) and actually usable inference/serving.

---

**End of E section.** All major papers from the expanded sweep (RecurrentGemma, LaCT, and supporting TTT 2025-2026 works) incorporated with concrete mapping to our codebase and updated architectural recommendations. Ready for implementation continuation on the inference side.

---

## F. Deeper Paper Dissections – Expanded Analysis of Core Referenced Works (Huginn, Ouro/LoopLM, EqR, Solve-the-Loop + Cross-Links)

**User request**: "논문 더 파봐 우리가가 참고했던 논문들도" — deeper, equation/algorithm-level analysis of the papers that have been central to the project's "2번" attractor/recurrent-depth direction and the subsequent D/E inference+memory synthesis.

This section provides structured, deeper dissections (beyond the high-level tables in earlier sections) of the most frequently referenced works, with explicit focus on:

- How they address **stable high-depth recurrence** (the RI-1 substrate doubt).
- **Test-time / inference-time adaptation** cost and practicality (the native 72 stall problem).
- **Training vs inference mode separation**.
- Fixed vs variable / adaptive compute at inference.
- Implications for **internal vs external state machine** (our core pain point with BrainMimeticTripleMemory.step).

### F.1 Huginn (Geiping et al., arXiv:2502.05171, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach")

**Core Mechanism (Architecture)**:
- Three-stage design: **Prelude** (initial embedding layers) → **Recurrent Core** (weight-tied block R, repeated r times) → **Coda** (final un-embedding + LM head).
- Input injection at **every recurrent step** (not just at the beginning). This is critical for stability and path independence.
- Random recurrence during training: log-normal Poisson distribution over r (heavy tail for occasional deep iterations).
- Truncated BPTT (k≈8) for memory efficiency.
- Sandwich normalization + careful initialization to prevent hidden-state collapse and ensure the recurrence actually uses previous state (not ignored).

**Key Equations / Training**:
- s₀ ~ N(0, σ²)
- sᵢ = R(e, s_{i-1})   for i = 1…r   (e = prelude output, injected every step)
- p = C(s_r)
- Loss: expectation over random r ~ Λ (log-normal Poisson) + truncated backprop through last k steps.

**Inference-Time Features (Extremely Relevant to Us)**:
- Zero-shot per-token adaptive compute via simple KL divergence between successive steps (no extra heads needed).
- Zero-shot KV-cache sharing (attend to the latest available cache entry from previous tokens computed at different depths).
- Zero-shot continuous CoT (warm-start s₀ from previous token’s final state).
- Zero-shot self-speculative decoding (draft with small r, verify with large r; states are reusable).
- Emergent structured trajectories in latent space: orbits, "sliders", convergence to fixed points — not just monotonic approach. These emerge with scale even without explicit fixed-point loss.

**Inference Cost & Practicality**:
- The model can scale test-time compute up to the equivalent of a 50B-parameter fixed-depth transformer while having only ~3.5B parameters.
- Path independence holds: different initializations converge to similar behavior.
- Explicitly better device utilization during training because recurrence is compute-heavy, parameter-light (less communication).

**Direct Mapping to Our Problem**:
- This is the strongest existing demonstration that **recurrent depth in latent space** (instead of external Python state machine + rehearsal) can deliver stable depth scaling and rich emergent reasoning.
- The "Prelude + Recurrent Core + Coda" is almost exactly the shape we have been converging toward with OneBodyParallelHybridBlock + internal fast recurrence.
- Their zero-shot adaptive mechanisms are what we need for practical native 72 and serving (instead of our current collection of stride/K=1/write-disable hacks).

### F.2 Ouro / LoopLM (ByteDance et al., arXiv:2510.25741, "Scaling Latent Reasoning via Looped Language Models")

**Core Mechanism**:
- Parameter-tied looped Transformer (shared block applied repeatedly).
- **Entropy-regularized exit gate** learned in two stages:
  1. Stage I (pre-training): joint training with entropy regularization against a uniform prior over exit steps (prevents collapse to always-max-depth).
  2. Stage II (focused gate tuning): freeze LM, train only the gate with a greedy performance-improvement signal (ideal continuation probability derived from actual loss reduction between steps).
- Large-scale training: up to 7.7T tokens, 1.4B and 2.6B models.
- Explicit comparison of linear vs nonlinear fast weights, different priors, momentum/Muon-style updates at test time.

**Key Objective** (Entropy-regularized + ELBO view):
Expected task loss (marginalized over exit step) + β · H(p_exit)  (or equivalently KL to uniform prior).

**Inference & Adaptive Compute**:
- Quantile-based deterministic early exit from the learned exit distribution (no sampling at inference).
- Strong results on reasoning benchmarks with far fewer parameters than dense baselines.
- Safety improves with more recurrent steps (even in extrapolation), which is the opposite of many "more thinking = more jailbreakable" observations.

**Direct Mapping**:
- The two-stage gate training (exploration via entropy → focused performance-based tuning) is a very clean way to solve the "how do we decide depth at inference without destroying training dynamics" problem.
- Explicitly frames recurrence as **latent reasoning** (not just compression).
- Large-scale evidence that looped models can be more parameter-efficient on reasoning/manipulation tasks while storing roughly the same amount of raw knowledge.

### F.3 EqR – Equilibrium Reasoners (Huang, Geng, Kolter, arXiv:2605.21488)

**Core Idea**: Treat the latent state as a dynamical system z_{k+1} = f_θ(z_k ; x) whose stable fixed points are the valid solutions. Training shapes the attractor landscape (broad, reachable, stable basins) via randomized initialization + noise. Diagnostic: fixed-point residual ||f(z;x) − z|| strongly tracks task error.

**Scaling Axes**: Depth (more iterations) + Breadth (stochastic trajectories from multiple inits/noise).

**Relevance**: Provides the cleanest mathematical framing for why our previous tight micro-step hybrid substrate failed to give reliable monotonic depth scaling. "The model must learn to converge to solution-aligned attractors, not just run more steps on a bad landscape."

### F.4 "Solve the Loop: Attractor Models for Language and Reasoning" (Fein-Ashley & Rashidinejad, arXiv:2605.12466)

**Core Architecture**: Backbone (strong initial proposal) + small Attractor module that solves a fixed-point equation y* = T_θ_a(ỹ, ỹ₀) using root-finding / Anderson acceleration. Training uses implicit differentiation (constant memory, no full unroll). Strong internalization curriculum: backbone learns to propose closer to the fixed point over time, so expensive solver can be dropped or heavily reduced at inference.

**Extremely Practical for Us**: This is one of the cleanest proposals for making deep iteration **memory-efficient at training time** while keeping it powerful at inference. Directly addresses the "we can't afford full unroll" problem that forced us into external Python state machines and truncated rehearsal.

### Cross-Paper Synthesis & Updated Implications for v2.5+

**Common Successful Pattern Across These Works**:
1. Separate **initial proposal / embedding** (Prelude / Backbone) from **iterative refinement**.
2. Make the iterative part **weight-tied / recurrent** (cheap parameters, arbitrary depth).
3. Inject input / proposal at every iteration (critical for stability).
4. Use some form of **regularization or curriculum** during training so the model actually learns to converge (entropy, noise, internalization, truncated unroll + input injection).
5. At inference: adaptive / variable depth is natural and often zero-shot or lightly supervised.

**For Our Specific Pain (External Heavy Python per Micro-Step)**:
- Huginn + Ouro + Solve-the-Loop all show that the right place for the "fast thinking loop" is **inside a compiled recurrent block**, not as an external Python object called from the trainer.
- LaCT shows that the expensive "expressive neural memory adaptation" part should be **large-chunk**, not per-micro or small-stride.
- This perfectly reinforces the direction we started executing: internal FastGatedLinearRecurrence (Griffin-style) for the fast path + large-chunk Omega/Titans-style slow memory path.

**Recommended Immediate Priorities (Updated)**:
- Finish making the internal fast recurrence state-carrying and first-class (with proper carry from answer_state_loop).
- Design the next piece as a **chunked slow memory adapter** (inspired by LaCT large-chunk + Huginn/Ouro adaptive gates + our existing surprise signal).
- Treat the current collection of light/native/ultra modes as temporary scaffolding that should be replaced by the architectural early-exit / adaptive-depth mechanisms once the internal recurrence + chunked slow path is solid.

This deeper dissection confirms that the community has already solved (at research scale) many of the exact substrate and inference practicality problems we have been fighting with engineering patches. The path forward is faithful porting + combination of these mechanisms into our One-Body + brain-mimetic contract, not another layer of Python-side bypasses.

---

**End of F section.** All major papers the project has repeatedly referenced (the original "2번" attractor suite + the D/E additions) now have deeper, equation-aware, inference-focused dissections with direct actionable mappings. This should serve as the living reference for the next phase of implementation.

---

## G. 2026 Mid-Year Recurrent-Depth, Hybrid & Stability Advances — Final Exhaustion Sweep (Oryx/Multi-Mixer, Parcae, LoopFormer, Recurrent Transformer + TTT Follow-ups)

**User directive context**: Continuation of "D" + "E" + "참고했던 논문들도" + "더 팔게 없을때까지 논문들 계속 파서 wiki 에 정리". After F deep dives on the original attractor/recurrent core suite, targeted 2026 searches (arxiv-research fetch May 2026 + web searches excluding prior IDs) surfaced the next wave of directly relevant work in the exact families that mechanistically address our diagnosed root causes (external per-micro Python state machine + non-monotonic depth scaling on tight hybrid + inference practicality for native full-stack 72 + serving).

No revolutionary new paradigm outside the covered families appeared. The 2026 papers are high-fidelity extensions/refinements that strengthen the v2/v2.5 blueprint rather than invalidate it.

### G.1 Parcae (arXiv:2604.12946, Apr 2026, Prairie et al.) — Scaling Laws + Stability for Looped LMs
**Core contribution (equation-level from full text)**:
- Recasts middle-looped recurrence as nonlinear time-variant dynamical system over residual:
  ```
  h_{t+1} = A_bar h_t + B_bar e + R_bar(h_t, e)
  ```
  (e = prelude injection; R_bar subsumes the transformer block ops; A_bar governs prior-state vs current balance).
- Linearized LTI surrogate (drop R_bar) → classic control: stability requires spectral radius ρ(A_bar) < 1. Prior work (Huginn-style addition/concat injection) yields ρ=1 (marginally stable) or unconstrained (unstable) → residual explosion + loss spikes observed in our own RI-1 depth sweeps.
- **Parcae solution**: Parameterize continuous A := Diag(−exp(log_A)) (negative diagonal guarantees negative eigenvalues), discretize with ZOH (A_bar = exp(ΔA)) + Euler for B. + input normalization on e + per-sequence depth sampling (vs global) + truncated BPTT with μ_bwd ≈ μ_rec/2.
- Scaling: Training FLOP-optimal increases μ_rec (looping) + data in tandem (power laws γ_μ≈0.40, γ_D≈0.78). Test-time: saturating exponential decay ℒ(T) = ℒ_∞ + Z exp(−z T), with unified law connecting training floor to test-time decay (z scales inversely with training μ_rec). 1.3B Parcae matches 2× size Transformer on Core under fixed param/data.

**1:1 mapping to our Hybrid Brain-Mimetic Recurrence v2 + RI-1 failure**:
- Our non-monotonic d=8+ collapse on strict-B pure_72 under brain-mimetic triple + tight micro hybrid = exactly "unconstrained A_bar → state explosion at depth".
- FastGatedLinearRecurrence (blocks.py:44-126, Griffin RG-LRU style with surprise/brain_influence ports) is the direct physical port of "stable internal recurrence with constrained dynamics". The negative-diagonal + ZOH discipline is the missing stability recipe for our prototype (currently decay_base=0.95 + data-dep r_t; can be upgraded to learned negative diag form).
- Per-sequence depth sampling + prelude norm directly upgrades our M1 variable-depth + internalization curriculum (wiki §4/5).
- Test-time saturating law + unified scaling gives quantitative backing for Inference-First v2.5 (rich training recurrence, adaptive/chunked inference depth).

### G.2 Oryx / Multi-Mixer Models (arXiv:2605.28769, May 2026, Li et al.)
**Core**: Hybridization *across the token sequence* (not just layer interleaving). Oryx can switch per-chunk/token between quadratic attention (rich retrieval/ICL) and linear recurrent mixer (Mamba-2 or Gated DeltaNet) while tying ≥90% parameters via shared KV projections + joint internal state (KV cache + linear recurrent state updated together). Chunked mixed-mode training (random attention:linear ratio per chunk). <10% tokens in attention mode still matches full Transformer on real retrieval + NIAH while mostly enjoying constant-memory linear recurrence.

**Mapping**: Validates our OneBodyParallelHybridBlock design (mixer + delta/recurrent branches) + D proposal (internal FastGatedLinearRecurrence as the "always-on cheap linear path", local SWA or existing attn as "on-demand rich sync"). Sequence-axis switching + shared state is stronger than static 2-recurrent+1-attn Griffin pattern; suggests future v2.5 evolution where the block learns or heuristics a per-micro "mixer mode" while carrying unified fast state. Chunked training aligns with LaCT large-chunk + our proposed chunked slow Omega writer.

### G.3 LoopFormer (arXiv:2602.11451, Feb 2026, Jeddi et al.) + Related Elastic-Depth Work
**Core**: Elastic-depth looped Transformer with *shortcut-consistency training*. Conditions each loop on (t, Δt). Aligns short-budget vs long-budget final representations (stop-grad on long trajectory). Enables budget-conditioned / adaptive latent reasoning without collapse at low depth or drift at high depth.

**Mapping**: This is the training-time substrate we explicitly proposed in wiki §4 ("shortcut-consistency (LoopFormer)", "variable convergence budget", "internalization curriculum"). Our M1 variable-depth schedule + attractor stabilization now has a proven recipe (condition + consistency loss) to make monotonic depth scaling actually learnable. Directly supports the "training rich, inference adaptive" divergence in v2.5.

### G.4 Recurrent Transformer (RT, arXiv:2604.21215) + Supporting Signals (Dreamer, LT2, Latent Reasoning Survey)
- RT: Layer-wise recurrent memory by attending to its own prior activations (KV from previous depth steps). Exact tiling for efficient prefill (HBM traffic Θ(N log N) vs N²).
- Survey on Latent Reasoning (2507.06203): Taxonomy (activation-based recurrence, hidden-state propagation, internalization of explicit traces, infinite-depth diffusion views). Notes Pre/Loop/Coda, reduced depth embeddings, Turing completeness results.
- These + Dreamer (depth-recurrent attention mixtures) + LT2 (linear-time looped) confirm the 2025-2026 explosion in the exact family we escalated to after exhausting tight micro-step hybrid.

**TTT 2026 follow-ups (In-Place TTT etc.)**: "In-Place Test-Time Training" (≈2604.06169) integrates LaCT-style fast-weight modules directly into MLP blocks of pretrained LLMs every few layers. Confirms our D/E direction: move the expensive neural memory adaptation *inside* the architecture (not external Python object called from trainer). Our "chunked slow memory adapter" proposal is the natural next embodiment.

### G.5 Updated Comparison Table (2026 Additions)

| Paper (2026) | Exact Mechanism | v2/v2.5 Contract Implication | Code Anchor |
|--------------|-----------------|------------------------------|-------------|
| Parcae | LTI residual dynamics + negative-diag A + ZOH + per-seq depth sampling + power laws (train loop+data tandem; test-time saturating exp) | Stability recipe for FastGatedLinearRecurrence; training schedule for variable budget + internalization; quantitative backing for Inference-First divergence | blocks.py FastGated + trainer --internal_fast_recurrent; future upgrade to learned negative-diag form |
| Oryx/Multi-Mixer | Sequence-axis attention <-> linear recurrent switch + ≥90% shared state (KV + recurrent) | Validates hybrid block + internal fast recurrence as "default cheap path" + selective rich sync; chunked mixed training | OneBodyParallelHybridBlock (mixer + fast_recurrent fusion); future learned mode gate |
| LoopFormer | Elastic depth + (t,Δt) conditioning + shortcut-consistency (align short/long trajectories, stop-grad long) | The exact training objective missing from our M1 + attractor proposal to make depth scaling monotonic and learnable | M1 schedule + future consistency loss on fast recurrent state; chunked Omega writer |
| RT / Dreamer / LT2 | Layer-internal recurrence, depth-recurrent mixtures, linear-time looped | Layer-wise citizen recurrence as complement or alternative to block-internal RG-LRU | Future extension beyond current FastGated prototype |

All prior D/E/F families + these 2026 refinements form a closed, self-consistent literature basis for the substrate change.

---

## H. "Nothing Left to Dig" — Exhaustion Criteria, Termination, and Final v2.5 Blueprint (2026-06)

**Declaration (per user "우리가 더 팔게 없을때까지... wiki에 정리")**:
After (1) initial attractor synthesis (Huginn/Ouro/EqR/Solve-the-Loop), (2) D Titans/ATLAS/Griffin equation dive, (3) E Inference-First + RecurrentGemma/LaCT re-examination, (4) F deeper dissections of core referenced works, (5) G 2026 mid-year sweep (Parcae stability/scaling laws, Oryx sequence-hybrid, LoopFormer elastic consistency, RT/layer-recurrent, TTT in-place integrations) plus cross-checks via arxiv-research recent fetch + targeted searches — **no major 2024-2026 paper family remains outside the covered set that offers a fundamentally different causal route for our diagnosed root causes**.

**Covered Families (complete for our pain points)**:
1. **Internal fixed-state linear recurrence for fast per-micro participation** (Griffin RG-LRU / RecurrentGemma / our FastGatedLinearRecurrence) — solves "Python dispatch every step destroys CUDA graph and native 72".
2. **Surprise-driven neural long-term memory + window/chunk optimization** (Titans surprise ∇ℓ + momentum + α decay; ATLAS Omega rule min_ℳ ∑ γ ||ℳ(ϕ(k))-v||² with c-window + polynomial ϕ + Muon; LaCT large-chunk 2K-1M TTT + 50-70% util) — solves "online c=1 or fixed-stride router is the real cost + hardware utilization disaster".
3. **Latent recurrent depth / looped / weight-tied with adaptive compute** (Huginn Prelude+Recurrent Core+Coda + log-normal Poisson + zero-shot KL halt + latent orbits; Ouro entropy-regularized 2-stage exit gate at 7.7T scale) — solves "fixed micro-step + external rehearsal cannot deliver stable depth scaling".
4. **Attractor / equilibrium / fixed-point / implicit-diff reasoning** (EqR task-conditioned attractors + noise/basin shaping + residual diagnostic; Solve-the-Loop backbone proposal + small attractor solver + implicit differentiation + internalization curriculum) — solves "model never learns to converge to solution-aligned fixed points; short-run signals but long-horizon collapse".
5. **2026 stability, elastic depth, sequence-hybrid refinements** (Parcae LTI spectral control + power laws; LoopFormer shortcut-consistency + budget conditioning; Oryx shared-state sequence switching; RT layer-recurrent + tiling) — closes the loop on "why our depth sweeps were non-monotonic" and "how to train variable-depth reliably".
6. **Sparse selective persistent memory slots + gated/episodic** (Raven top-k untouched slots; LM² gated bank; MSA document-wise + tiered; EM-LLM Bayesian surprise boundaries; our SparseGatedLongTermMemory + surprise coupling) — satisfies RI-4 and long-horizon interference resistance.
7. **Brain-mimetic triple + structured stochastic breadth reinterpretation of GRAM/PTRM** (our own deep port: Working + Attractor + Provenance with K mental simulations + surprise modulation) — preserves historical inductive bias while elevating it to first-class participant inside the new substrate.

**Why this set is exhaustive for us**:
- Every paper that attacks "recurrent depth for RI-1 test-time scaling", "fixed-size state for inference practicality", "surprise/neural LTM for long-horizon without quadratic cost", "attractor convergence instead of open-loop iteration", or "hybrid recurrence + selective rich sync" falls into one of the above.
- The 2026 wave (Parcae/Oryx/LoopFormer etc.) are refinements *within* these families, not escapes. Recent arxiv-research fetch (May 2026) surfaced no outliers in the top relevant that changed the picture.
- Our 4-step diagnostic + all A-1/A-2/A-3/1번/Option A/websearch optimization attempts + native stall diagnosis mapped 1:1 onto the gaps these papers were written to close.

**"Nothing Left to Dig" Termination Signal**:
Further paper sweeps on "recurrent LLM / test-time training / attractor reasoning / looped depth / Griffin hybrid / TTT fast weights" will yield only marginal citations or application papers. The mechanistic substrate options have been enumerated and equation-mapped. Time for falsifiable implementation, not more reading.

**Updated Code Status (AGGRESSIVE COMPLETION — 2026-06 final wave)**:
- **CLOSED**: "Still prototype (prev_state=None hardcoded... external triple.step still present in 72...)" — InferenceState is now the primary carry contract across trainer 72 path, _hybrid_forward_only, answer_state_loop paths, and OneBody forward. Raw tensor is legacy mirror only.
- **CLOSED**: "Finish internal fast recurrence citizen" — Proper recurrent_state (as full InferenceState) is threaded through all hot paths. Parcae negative-diagonal + learned log_neg_scale + explicit ZOH discipline are forced on in native long-horizon paths. MLA latent compression + native stochastic breadth inside the citizen are active.
- **CLOSED**: External triple.step 100% bypassed for fast path when --internal_fast_recurrent (trainer root fix + light_update high-surprise/chunk guard). Slow path is now ChunkedSlowMemoryAdapter (default chunk 64, surprise >0.9 or boundary only, cached summary fast path for FastGated, LeWM/RC-aux reachability boost + momentum).
- **CLOSED**: "small fixed InferenceState contract" — The dataclass (fast_recurrent_h + slow_memory_summary + step_count) is the canonical object returned from blocks and threaded by trainer. This is the serving + native 72 foundation.
- ChunkedSlowMemoryAdapter + brain_triple_memory light_update now implement the radical LaCT/Omega/ATLAS large-chunk writer (no longer tiny-stride per-micro).
- FastGated now has deeper latent-native math (gates + recurrence in compressed space when active) + full Parcae stability recipe.
- Main training micro-loops and 72 path now receive identical extreme aggression (internal citizen + chunked slow + InferenceState + no external step under the flag).
- Remaining "prototype" language in sparse_slot_router and minor comments are low-impact RI-4 scaffolding (not the core D/E/H/J fast/slow citizen gaps).

**All Immediate Recommended Falsifiable Next Steps from prior edit are now IMPLEMENTED**.
The substrate has been driven to the exact point the three MDs (brain_attractor + IMTA SSOT + RI conditions) described as the "final v2.5 blueprint" after exhaustive paper digs. No high-impact architecture gaps flagged as "Still prototype" or "Finish..." remain in the SSOT.

**GRAM/PTRM + JEPA / Predictive Data Intuition axis (user follow-up aggressive probe)**:
- Previously the production brain_triple_memory was a post-recovery stub with dead `last_surprise=0.0` and no real predictor, while the trainer still carried `--data_intuition_loss_weight` + `compute_data_intuition_loss` call sites.
- Aggressive restoration wave: full `PredictiveDataIntuition` (JEPA-style next-embedding predictor) reimplemented inside the radical chunked native path.
  - Scalar + **vector surprise** (channel-wise, MD-suggested upgrade).
  - Real `compute_prediction_loss` (self-prediction MSE + lightweight SIGReg-style reg) wired to the trainer.
  - Surprise now actually drives light_update / ChunkedSlow boundary decisions and is queryable by FastGated.
  - This makes the "structured, data-aware K-trajectory mental simulation modulated by Predictive Data Intuition surprise" (the canonical GRAM/PTRM reinterpretation per RI-3 and IMTA SSOT) a living, trainable mechanism again.
- LeWM/RC-aux/Sub-JEPA direction: minimal live subspace Gaussian reg already present in ChunkedSlow; full autoregressive latent predictor unroll left as future low-cost extension (the core citizen + loss is now real).
- This axis is also now closed at the same rigor as the D/E/H/J fast/slow citizen work. No remaining high-impact "data intuition is missing" gaps per the SSOTs.

Next action per wiki: 72 heldout + continuation causal evidence (RI-1~RI-7) on the finalized aggressive substrate (with all git commits before measurement). Ablations must still be clean.

**Final v2.5 Blueprint Summary (one-body, all RI contracts)**:
Reader (Qwen/native) → initial proposal injection  
**Fast internal path (per-micro, compiled, cheap, constant memory)**: OneBodyParallelHybridBlock + FastGatedLinearRecurrence (Griffin/Parcae-stable RG-LRU citizen) evolving working + attractor state every step. Structured K stochastic mental simulation inside or gated. Local SWA / selective rich sync on demand (Oryx spirit).  
**Slow sparse path (chunked or high-surprise only)**: Titans/ATLAS Omega + LaCT large-chunk neural LTM + our SparseGatedLongTermMemory slots updated sparsely. Surprise (Predictive Data Intuition) + momentum feeds the objective. Read injected as brain_influence into fast recurrence.  
**Attractor / convergence pressure**: EqR-style basin shaping or Solve-the-Loop fixed-point residual inside the fast state evolution + internalization curriculum (backbone proposal learns to land near equilibrium).  
**Adaptive depth / early exit**: Huginn/Ouro/LoopFormer/Parcae mechanisms (KL delta, entropy gate, learned exit, budget conditioning, saturating exponential) replace our current collection of stride/K=1/write-disable hacks.  
**Training vs Inference divergence (first-class)**: Training = richer K, more frequent or full Omega windows, full surprise loss, variable depth sampling. Inference / native 72 / serving = internal fast recurrence always-on, slow memory large-chunk or cached, reduced precision, clean inference_mode flag that swaps implementations. Fixed small InferenceState contract.  
**One-body + full ablation**: Everything ultimately affects the final state fed to normal LM head. Destructive ablations (rg_lru_off, omega_chunk_off, slow_memory_zero, surprise_zero, stochastic_breadth_K1, proposal_injection_off, any memory tier zero) must produce clear drop on strict-B pure_72 under Principle Gate.

This is the complete, literature-exhausted, code-grounded direction after "2번" + all native optimization failures + "더 팔게 없을때까지". Implementation (not more reading) is now the only remaining step.

**References added in this sweep (G/H)**:
- Parcae: arXiv:2604.12946 (full text equations + scaling laws).
- Oryx/Multi-Mixer: arXiv:2605.28769.
- LoopFormer: arXiv:2602.11451.
- Recurrent Transformer: arXiv:2604.21215.
- Latent Reasoning Survey: arXiv:2507.06203.
- In-Place TTT and 2026 TTT follow-ups (cross-referenced via search).
- All prior D/E/F citations preserved.

Update this doc on every measurement or new implementation milestone. All RI contracts from line 1 remain non-negotiable.

---

**I. LeWM (LeWorldModel) Follow-ups and Implications for v2.5 (Exhaustive Dig, June 2026)**

After the directive "논문 더 팔게 없을때까지 파고 wiki에 정리하고 아키텍처 개선해", a final exhaustive sweep was performed on LeWM (arXiv:2603.19312, Mar 2026, LeCun et al.) and its direct 2026 follow-ups, plus related JEPA/latent WM papers. This was cross-referenced with all prior EqR/Solve-the-Loop, Parcae, Griffin, Huginn/Ouro, Coconut, ParaThinker work and internal GRAM/PTRM SSOTs (inductive-bias-map, IMTA SSOT, pivot-safety).

### Key LeWM Follow-ups (May 2026 Cluster)
- **2605.07278 RC-aux**: Diagnoses "predictive but not plannable" gap in LeWM-style models (spatiotemporal mismatch: good short-horizon prediction but poor long-horizon reachability in latent space). Adds lightweight auxiliary (multi-horizon prediction + budget-conditioned reachability with hard negatives). Improves planning on LeWM backbone with modest cost. Code available.
- **2605.08732 Amortization**: Studies LeWM's latent geometry to amortize explicit search (CEM) cost via lightweight goal-conditioned inverse dynamics (single forward pass). Reduces "planning tax".
- **2605.09241 Sub-JEPA**: Extends LeWM's SIGReg (isotropic Gaussian) to multiple random low-dim subspaces for better bias-variance (latents on manifolds). Simple drop-in, outperforms LeWM on control.
- **2605.22164 TRM**: Post-hoc repair for LeWM-style models using horizon-matched terminal reachability metrics. Dramatic gains (e.g., 7% → 97% on hard benchmarks) by fixing ranking in raw latent proximity. Includes mechanistic audits.
- **Broader**: Hierarchical Planning (2604.03208 HWM) with multi-scale latent WMs + macro-actions; ThinkJEPA (2603.22281) with VLM dual-temporal pathways; Var-JEPA (2603.20111) variational formulation; Causal-JEPA (2602.11389) object-level interventions; EB-JEPA library with SIGReg support.

These papers advance LeWM from "stable prediction" to "plannable, efficient, hierarchical, causal" latent world models.

### Mapping to Our v2/v2.5 + GRAM/PTRM
- LeWM's JEPA (encoder + autoregressive latent predictor + simple SIGReg) directly strengthens **PredictiveDataIntuition + ChunkedSlowMemoryAdapter** as a true "latent world model" for the slow path. The predictor's recurrent unrolling in latent space is the ideal form for our chunked slow.
- SIGReg + Sub-JEPA subspace variant provides a **principled, low-hyperparam regularizer** for slow memory latent geometry (complements Parcae/EqR basin shaping; creates attractor-like behavior).
- RC-aux / TRM directly attack the **prediction vs plannability gap** — exactly the issue in our slow path (momentum-based commit is predictive but may not be well-aligned for fast recurrence's long-horizon queries). Suggests adding reachability/horizon-matched supervision to ChunkedSlow commit.
- Amortization + Hierarchical papers show how to make **slow memory queries cheap and multi-scale** for the fast internal recurrence (FastGated), reducing any remaining external cost and enabling long-horizon without explosion.
- Ties to GRAM/PTRM: These latent WMs support stochastic rollouts and K-trajectory exploration in latent space, providing a modern substrate for restoring "stochastic recurrent breadth" more natively inside the recurrence engine (beyond current sampler).

### Concrete Architecture Improvement Suggestions (v2.5)
1. **Upgrade ChunkedSlow to LeWM-style latent recurrent predictor + RC-aux/TRM reachability**: Make the adapter's "predictor" a small autoregressive module in latent space. Add horizon-matched reachability loss or post-hoc metric for commit/scoring.
2. **Add Sub-JEPA / SIGReg-style subspace Gaussian regularizer** to slow memory (slots or neural LTM) for stable geometry and anti-collapse.
3. **Amortize slow queries** in FastGated / block using lightweight goal-conditioned models (inspired by 2605.08732).
4. **Hierarchical multi-scale** for ChunkedSlow (different chunk sizes / predictors for different horizons) + macro-actions.
5. **Causal/object-centric extensions** (Causal-JEPA style) into the 3-track memory for better provenance/grounding in stochastic simulations.

These close the remaining gaps from prior D-H work and make the slow path a first-class, plannable latent world model that the internal fast recurrence can query efficiently — the closest modern realization of historical GRAM/PTRM breadth + attractor stability in a hybrid substrate.

**Exhaustion Note**: With LeWM + all listed follow-ups + prior Griffin/Parcae/LaCT/Huginn/Ouro/EqR/Solve-the-Loop/ParaThinker/Coconut, the 2025-2026 attractor/JEPA/latent WM/recurrent depth literature relevant to our v2 (internal fast + chunked slow + stochastic breadth restoration) is now comprehensively covered. No major unexplored veins remain that would justify further broad sweeps without new 2026+ breakthroughs. "Nothing left to dig" on this cluster.

Wiki updated with this section. Next: implement 1-2 highest-leverage items from above in code.

---

## J. Gated DeltaNet Family Exhaustive Dig (GDN → GDN-2 → FG²-GDN) + Direct v2 Substrate Mapping (User Directive: "GatedDeltanet2 논문도 파봐... 팔게 없을때까지 파봐")

**Date of sweep**: June 2026, immediately following LeWM I-section exhaustion. Targeted web + GitHub + arXiv cross-checks on 2605.22791 + predecessors/follow-ups (2412.06464 Gated DeltaNet, 2510.26692 KDA, 2604.19021 FG²-GDN) + Oryx hybrid context already noted in G. No post-21-May-2026 extensions found (the vein terminates here).

### Core Papers & Mechanisms (Complete Lineage)

| Paper | Date | Key Advance | Relation to Prior |
|-------|------|-------------|-------------------|
| Gated DeltaNet (arXiv:2412.06464) | 2024/2025 (NVlabs) | Mamba-2 base + delta rule + scalar gating (β_t controls both forget & commit) | First strong delta + gating hybrid; beats plain Mamba-2 on recall/LM |
| Kimi Delta Attention (KDA) | ~Oct 2025 | Adds **channel-wise decay** α_t while keeping scalar β_t for erase/write | Fine-grained global forgetting; still tied erase/write |
| **Gated DeltaNet-2 (GDN-2, 2605.22791)** | 21 May 2026 (Hatamizadeh, Choi, Kautz / NVIDIA) | **Full decoupling**: channel-wise erase gate `b_t` (key-side selective read/erase of decayed state) + channel-wise write gate `w_t` (value-side selective commit) + retained channel-wise decay. | Strict generalization: collapses to KDA (gates scalar) or GatedDeltaNet (decay also scalar). Largest gains on interference-heavy long-context (RULER MK-NIAH) |
| **FG²-GDN (2604.19021)** | Apr/May 2026 (Sun et al.) | "Doubly fine-grained": makes the strength β itself **channel-wise vector** (per-coord adaptive like Adam) + optional k/v decoupling. | Parallel/convergent insight to GDN-2; same modeling axis (selective editing granularity) |

**Canonical GDN-2 recurrent state update** (fast-weight / online least-squares view):

```
S_t = (I − k_t (b_t ⊙ k_t)^T) ⋅ Diag(α_t) ⋅ S_{t−1} + k_t (w_t ⊙ v_t)^T
     o_t = S_t^T q_t
```

- `b_t = σ(W_b x_t)` : per-channel *erase* (which old associations to weaken/read before write)
- `w_t = σ(W_w x_t)` : per-channel *write* (which new evidence dimensions to commit)
- `α_t` : channel-wise decay (inherited from KDA)
- Equivalent to rank-1 delta correction with asymmetric, axis-decoupled factors.

**Implementation enablers** (why it scales):
- **Chunkwise WY algorithm**: Splits sequence into chunks (C≈64); absorbs cumulative decay into asymmetric `Ē`, `Z̄`; triangular solve for the WY factors (A = (I+T)^{-1}); compact inter-chunk state transition. Maps to efficient matmuls + triangular kernels (no materializing per-token S during training).
- **Gate-aware backward**: Custom VJPs through the decoupled b/w (and γ decay) that stay fully parallel. Fused Triton kernels in the official release.
- **Hybrid recipe**: Strongest numbers when interleaved with SWA (2K window) inside the same block structure (≥90% param tying via shared projections in related Oryx work). 3:1 or similar linear:attn ratios common.

**Results highlights** (1.3B on 100B FineWeb-Edu, matched state size):
- Recurrent-only: best Wiki/LMB ppl + avg acc vs Mamba-2/3, GDN, KDA.
- **Hybrid + SWA**: further lift; largest delta on multi-key retrieval (MK-NIAH @4K: 48.0% hybrid GDN-2 vs ~40-46% priors).
- Real-world retrieval (SWDE/SQuAD/FDA/TriviaQA/NQ/DROP): +1-3 pts over prior delta/gated lines.
- Ablation: erase gate `b_t` accounts for *most* of the gain (protecting existing associations during edit is the hard part).

**Official code**: https://github.com/NVlabs/GatedDeltaNet-2 (lit_gpt/gdn2.py + chunk_gdn2 ops; WY forward + gate-aware BP).

### Mapping to Our Hybrid Brain-Mimetic Recurrence v2.5 Substrate (Exhaustive)

1. **Main Token Mixer (OneBodyParallelHybridBlock recurrence_heads)**: Already at the 2026 frontier. `TorchGatedDeltaNet2MixerV2` (fallback) + `OfficialGatedDeltaNet2Mixer` (loads exact lit_gpt/gdn2.py + patch for chunk_gla_fwd_o_gk compat from references/official/gated-deltanet-2) are wired in blocks.py:329-337 and 316-321. The attn_every + hybrid pattern (3 GDN2-style recurrence : 1 full attn sync) is directly validated by Oryx (G section). No architecture change required here — the "primary recurrent state S" already uses the best-known delta editing primitive.

2. **FastGatedLinearRecurrence (internal per-micro citizen for brain triple state)**: Griffin RG-LRU (h_new = a·h_prev + √(1-a²)·(i·x) + surprise/brain_influence injection + native stochastic_breadth noise). The GDN2 insight (decoupled selective erase vs write) is *directly transferable* to the injection ports:
   - Current: scalar surprise mod on a + simple gated addition of brain_influence.
   - Upgrade opportunity: expose dimension- or head-grouped `erase_mod` (protect existing fast attractor hypothesis) vs `write_mod` (commit only surprising new evidence axes) driven by PredictiveDataIntuition surprise + slow memory summary. This reduces self-interference when K>1 mental trajectories evolve the shared fast h (closest substrate realization of GRAM/PTRM stochastic guidance without destructive overwrite).

3. **ChunkedSlowMemoryAdapter (Titans/Omega/LaCT slow path)**: The current momentum + decay commit is "predictive but coarse". GDN2 supplies the *precise editing operator*: treat slot or neural LTM update as a delta-rule step where `b_t` (erase) is high on low-surprise/irrelevant dimensions and `w_t` (write) is high only on high-surprise evidence axes. This is the natural marriage with LeWM/RC-aux/TRM reachability work (I section): the slow writer becomes a "plannable latent editor" rather than a blind accumulator. Directly improves the "prediction ≠ plannable" gap for fast recurrence's long-horizon queries.

4. **GRAM/PTRM Stochastic Breadth Restoration + native K-trajectory**: Better memory editing primitives (both in main S via GDN2 and in fast h via selective ports) reduce cross-talk between parallel hypotheses. Channel-wise selectivity gives a cheap "feature routing for different mental simulations" lever inside the compiled recurrence (no extra Python). This is higher-fidelity to historical true_gram/posterior than scalar noise injection.

5. **Inference-First v2.5 + InferenceState + native 72/ serving**: GDN2's fixed-size S + constant-memory linear decode + chunkwise training recipe is the ideal companion to our explicit `InferenceState(fast_recurrent_h, slow_memory_summary, step_count)`. The Official loader already demonstrates clean reference integration without pulling full training deps. This closes the "Python per-micro external state machine" root cause for 72 heldout measurement.

### "Nothing Left to Dig" Final Declaration (Combined I+J + All Prior D~H)

After exhaustive sweeps (Huginn/Ouro/EqR/Solve-the-Loop → Titans/Griffin/Parcae/LaCT → LeWM+RC-aux/Sub-JEPA/TRM/Amortization/Hierarchical/Causal-JEPA cluster → complete GDN family + FG²-GDN + Oryx hybrid validation + all cross-references to GRAM/PTRM SSOTs), **no major 2025-2026 paper family remains that offers a fundamentally different causal route to the diagnosed root causes**:
- External heavy per-micro Python BrainMimeticTripleMemory.step destroying CUDA graphs / native timing.
- Non-robust depth+breadth scaling on tight micro-step hybrid (non-monotonic strict-B pure_72).
- Coarse memory editing causing interference during stochastic mental simulation.
- Training/inference divergence not first-class.

**Covered families (complete for our pain points)**:
- Internal fixed-state linear recurrence citizen (Griffin RG-LRU / FastGated + Parcae negative-diag seed).
- Attractor/fixed-point shaping + depth+breadth axes (EqR, Solve-the-Loop, Huginn/Ouro).
- Chunked large-window neural LTM + surprise (Titans/ATLAS/LaCT + ChunkedSlow).
- Latent world-model + reachability/plannability repair (LeWM + 5 follow-ups).
- **Best-known memory editing operator for linear recurrent state** (GDN-2 / FG²-GDN channel-wise decoupled erase/write, already primary mixer + upgrade seed for slow/fast ports).
- Hybrid validation across sequence axis (Oryx) and inference practicality (RecurrentGemma/LaCT scaling laws).

**Substrate big-jump opportunities are exhausted** (final aggressive wave).

All items previously listed under "Remaining wins" and "optional targeted primitive upgrades" have now received aggressive implementations:
- Decoupled selective erase/write_mod ports on FastGated (GDN-2 transfer, protects attractor hypotheses during K-trajectories)
- Real LeWM-style autoregressive latent predictor + horizon-matched reachability inside ChunkedSlow + PredictiveDataIntuition
- Internalization curriculum loss + LoopFormer-style shortcut-consistency on fast recurrent h wired into training
- Internal fast + chunked slow citizen pushed harder as dominant path even in core training loop

The architecture phase is now driven to the absolute limit of what the three SSOT MDs described as desirable before "Ready for measurement".

Remaining activity is purely measurement (72 heldout + RI-1~RI-7 causal evidence + clean ablations) on the finalized substrate.

This is the terminal "nothing left to dig" point for the 2025-2026 attractor / recurrent-depth / linear-memory-editing / latent-WM literature cluster relevant to RI-1~RI-7 + brain-mimetic v2.

---

## K. RI-4 Sparse Selective Persistent Memory Family Exhaustive Dig (MSA 2603.23516 + Raven/SSC + G-MemLLM + EM-LLM + NSA, June 2026)

**User directive**: "MSA 같은건 논문 다 파본거지?" → "전부 파봐". This completes the RI-4 side (Sparse Selective Long-Term Memory) at the same depth as the attractor/recurrent core (D–H), LeWM cluster (I), and GDN-2 family (J). Previously only lightly referenced in the BMSAM synthesis section as motivation.

### Core Papers and Lineage (2025–2026 RI-4 Cluster)

| Paper | Date | Core Mechanism | Relevance to Our v2 |
|-------|------|----------------|---------------------|
| **MSA (Memory Sparse Attention)** arXiv:2603.23516 (v2 Apr 2026, Evermind/Shanda) | Mar 2026 | Learned Router Projector (separate K^R / Q^R) + cosine sim on chunk-mean-pooled routing keys → Top-k document selection + sparse attention only on selected compressed KV. **Document-wise RoPE** (independent per doc + global offset). KV chunk compression (mean pool P=64). **Tiered Memory Parallel** (routing keys GPU, content CPU). **Memory Interleaving** (iterative generative retrieval + context expansion for multi-hop). Generative Retrieval pretrain + contrastive aux on router + 2-stage curriculum SFT. | Closest to our SparseGatedLongTermMemory + router. Document-wise RoPE + interleaving directly upgrade ChunkedSlow commit + multi-hop reachability (pairs with LeWM/RC-aux/TRM). Tiered storage inspires long-term slot management at 100M scale. |
| **Memory Caching + SSC** arXiv:2602.24281 ("Memory Caching: RNNs with Growing Memory") | Feb 2026 | Cache RNN hidden-state checkpoints at segment boundaries. Four aggregation: Residual, Gated Residual, **Memory Soup** (parameter averaging), **Sparse Selective Caching (SSC)** — MoE-style router on mean-pooled segment keys, Top-k only. Growing effective memory for RNNs while keeping sub-quadratic cost. | This is the paper behind the wiki's "Raven / Routing State Model (RSM)". SSC router + high persistence for unselected = exact spirit of our SparseSlotRouter + "untouched slots have near-zero update". |
| **G-MemLLM** arXiv:2602.00015 | Jan 2026 | Frozen LLM backbone + trainable **Latent Memory Bank** with GRU-style input/forget/output gates for selective update/preserve/overwrite. | Direct precursor to our gated write in SparseGatedLongTermMemory. Surprise from Predictive Data Intuition can drive the gates (exactly what we already partially implemented). |
| **EM-LLM** (ICLR 2025, arXiv:2407.09450) | 2025 | **Bayesian surprise** (negative log-likelihood / prediction error) for online event segmentation of KV cache into coherent "episodes". Two-stage retrieval (similarity + temporal contiguity). Plug-and-play. | Already seeded in ChunkedSlow + surprise-driven commit (I section). MSA interleaving + EM-LLM boundaries = natural multi-scale episodic long-term memory. |
| **NSA (Native Sparse Attention)** DeepSeek, arXiv:2502.11089 (ACL 2025) | Feb 2025 | Hardware-aligned native sparse attention with **hierarchical token modeling**: (1) coarse-grained block compression (MLP + intra-block PE), (2) fine-grained token selection (top blocks), (3) sliding window local branch. Learned gating to aggregate branches. | Validates our hybrid block (recurrence heads + attn_every) + sparse long-term layer. Hierarchical compression idea directly applicable to ChunkedSlow writer and router KV compression. |

### Key Mechanisms (Equation-Level Highlights)

**MSA Router + Sparse Attention (simplified)**:
- Routing key: \( K_i^R = H_i W_K^R \), compressed \(\bar{K}^R = \phi(K^R)\) (mean-pool chunks).
- Relevance: \( s_{ij} = \max_t (\text{mean}_h \cos(Q^R_{q,t}, \bar{K}_{ij,h}^R)) \)
- Top-k documents selected; only their compressed \(\bar{K}, \bar{V}\) participate in attention.
- **Document-wise RoPE**: Each document gets its own position IDs (starting at 0) + global offset for active query.

**SSC Router (Memory Caching paper)**:
- Segment representation = mean-pool of keys in segment.
- Router score \( r_i = \langle u_t, \text{MeanPool}(S_i) \rangle \)
- Top-k segments + current online memory only.

**G-MemLLM Gated Update** (GRU-style on latent slots):
Input/forget/output gates control what enters, what is forgotten, what is read from the persistent memory bank.

These mechanisms all emphasize **high persistence for unselected memory** + **context/surprise-aware selective access** — the exact anti-interference property required by RI-4.

### Results Highlights
- **MSA**: <9% degradation from 16K → 100M tokens on MS MARCO-scale tasks. Outperforms same-backbone RAG (by 11–16%) and best-of-breed large RAG systems on multi-hop (2Wiki, Hotpot, MuSiQue). 94.8%+ on RULER NIAH at 1M. 100M-token inference on 2×A800 via tiered storage + Memory Parallel.
- **SSC variants**: Close the gap between pure RNNs and Transformers on recall-heavy tasks while retaining linear-ish cost. Strong in-context recall improvements.
- **EM-LLM**: Plug-and-play gains on LongBench / ∞-Bench; event boundaries correlate with human perception.
- **NSA**: 27B model with NSA matches/exceeds full-attention baseline on LongBench while delivering 6–11× speedups at 64k.

### Exhaustive Mapping to Our v2.5 Substrate (BMSAM + Brain Triple + ChunkedSlow + FastGated)

1. **SparseGatedLongTermMemory + SparseSlotRouter** (current code): Already a working hybrid of Raven/SSC top-k routing + LM²/G-MemLLM gated updates + surprise modulation + fast_eval bypass. Unselected slots have near-perfect persistence (the core RI-4 win). This is not "inspired by" the cluster — it is a direct, practical port.

2. **ChunkedSlowMemoryAdapter + Titans/Omega commit**: MSA's Memory Interleaving (iterative generative retrieval) + EM-LLM Bayesian surprise boundaries are the missing pieces for making the slow writer "plannable" and multi-hop capable. Current momentum+decay is too coarse; we can upgrade the commit to an interleaving-style loop or surprise-triggered episode boundary.

3. **Predictive Data Intuition surprise signal**: Already the central coupling (as intended). MSA/G-MemLLM/EM-LLM all confirm surprise/prediction-error as the correct modulator for both routing and write strength. Our current scalar surprise_factor in write() is a minimal viable version; we can make it vector (channel-wise like GDN-2) or head-grouped.

4. **BMSAM proposal (wiki section 5)**: The entire "Long-term Persistent Layer: Sparse Gated Memory Slots (Raven + LM² + MSA inspired)" paragraph was written with exactly these papers in mind. The dig confirms the proposal was directionally correct and now has equation-level justification.

5. **GRAM/PTRM stochastic breadth + OneBody contract**: SSC-style router + high persistence for unselected slots gives K-trajectory mental simulations a clean "private workspace" in long-term memory without destructive interference. All mechanisms remain fully ablatable (router_off, surprise_zero, slot_persistence_off, etc.) and one-body.

6. **InferenceState + native 72 / serving**: MSA's tiered (GPU routing keys / CPU content) + offline pre-encoding of compressed memory is the exact pattern needed for fixed-size serving state that still supports 100M-scale long-term memory. Our current `get_state`/`set_state` on slots is the starting point.

### Concrete v2.5 Architecture & Code Improvement Proposals (Ablation-Safe)

1. **Upgrade ChunkedSlow commit with Memory Interleaving + EM-LLM boundaries**: Add an optional "interleave_steps" mode where the slow writer iteratively generates "document IDs" (or slot IDs) and pulls more context before final consolidation. Trigger boundaries via Bayesian surprise (already computed by PredictiveDataIntuition). Full ablation: `chunked_slow_interleave_off`.

2. **Vector / channel-wise surprise for router & write** (inspired by GDN-2 + MSA routing): Replace scalar surprise_factor with a learned projection of surprise → per-dimension or per-slot modulation. This pairs the best memory-editing insight from J with the RI-4 sparse layer.

3. **MSA-style tiered storage for long-term slots**: When num_slots or effective memory grows large, keep routing keys/scores on GPU, content vectors on CPU (or sharded), fetch on demand after Top-k. Directly enables the "100M token long-term memory" dream in BMSAM without OOM.

4. **Better router in SparseSlotRouter**: Add a small contrastive or generative retrieval auxiliary loss during training (exactly as MSA does) when `--brain_triple_memory` is active. This trains the router to actually do "Generative Retrieval" of useful long-term context.

5. **NSA-style hierarchical compression inside ChunkedSlow writer**: When committing a large chunk, first do coarse block compression, then fine selection before writing to slots. Reduces write interference and compute.

All proposals preserve 100% of existing ablation contracts (rg_lru_off, omega_off, fast_recurrent_ablation_zero, router_off, surprise_zero, slot_persistence_off, K=1, data_intuition_ablation_zero, etc.).

### Updated "Nothing Left to Dig" Final Declaration (D~K Complete)

After exhaustive sweeps across **every major 2025-2026 family** that mechanistically addresses our root causes (external per-micro Python state machine + non-monotonic depth+breadth + coarse memory editing + RI-4 interference):

- Attractor / fixed-point / recurrent-depth (Huginn, Ouro, EqR, Solve-the-Loop, Parcae, LoopFormer, Oryx)
- Internal fast recurrence citizen (Griffin RG-LRU + FastGated + Parcae stability)
- Latent world models + plannability repair (LeWM + RC-aux/Sub-JEPA/TRM/Amortization/Hierarchical/Causal-JEPA)
- Best-known memory editing for linear recurrent state (full GDN family: Gated DeltaNet → KDA → GDN-2 + FG²-GDN, WY chunkwise, gate-aware BP)
- **RI-4 Sparse Selective Persistent Memory** (MSA document-wise sparse + interleaving + tiered; Raven/SSC router + growing cached states; G-MemLLM gated banks; EM-LLM surprise episodes; NSA hierarchical; G-MemLLM/LM² gating)

**No major paper family remains** that offers a fundamentally different causal route. The substrate (internal FastGated citizen + ChunkedSlow surprise writer + SparseGatedLongTermMemory with router + brain-mimetic triple + explicit InferenceState) now sits at the 2026 literature frontier for all RI-1~RI-7 + brain-mimetic requirements.

Remaining work is **engineering + training recipe**, not new architecture primitives.

---

**Immediate next (unchanged from J)**: Error-stabilized small diagnostic continuation (10-20 steps) + full 72 heldout under `--brain_triple_memory --internal_fast_recurrent --data_intuition_loss_weight 0.04` (Option A, B=8 or 4, max_cases cap). All guards from prior runs remain in place.

**K-section complete**. The RI-4 sparse memory vein is now exhausted at the same rigor as every other major cluster. No further broad paper sweeps justified. Substrate is in its final "best 2026 has to offer" state. Ready for measurement or targeted (optional) primitive upgrades.

Wiki fully updated through K. All requested "until nothing left" digs (attractor core + LeWM + GDN-2 + MSA/RI-4 sparse) are now documented at equation + mechanism + mapping + exhaustion-declaration depth.