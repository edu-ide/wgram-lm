# Long-term Backbone Improvement Roadmap (One-Body Strict)
**Project**: QTRM / 5.56 Curriculum Integration  
**Date**: 2026-05-30  
**Context**: Branch B Option 2 (Direct large-scale path) + Strict One-Body principle  
**Philosophy**: Sustainable, phased, evidence-driven improvement. No big-bang rewrites. Every phase must produce measurable progress on state robustness + training dynamics.

## Core Constraints (Non-Negotiable)
- **One-Body**: Everything must live inside the main recurrent core. No side organs, no separate memory banks, no external curriculum heads. Final output always through the normal LM head.
- **Ablation Cleanliness**: Every new mechanism (especially stochastic breadth) must support clean, independent ablation.
- **Qwen3.6-inspired 3:1 + GDN2:Atten lineage**: We start from the current architecture and evolve it, rather than throwing it away.
- **Long-term Goal**: Build a backbone strong enough that the full 5.56 curriculum (scheduled decay + gold structural injection + attractor protection + stochastic breadth) can demonstrate clear state robustness advantages at scale.

## Strategic Direction
We will follow a **hybrid evolution path**:
- Short-term: Strengthen and modernize the existing 3:1 GDN2+Attention structure using 2025-2026 research.
- Medium-term: Evolve toward modern parallel hybrid-head designs (Hymba-style) while staying One-Body.
- Long-term: Prepare a scalable backbone that can host the full 5.56 curriculum at frontier-relevant sizes, either by aggressive scaling of our lineage or principled porting.

This corresponds to a disciplined **Path C (Hybrid)** execution within Branch B Option 2.

## Phase 0: Foundation & Diagnosis (1-2 weeks, Low cost)

**Goals**:
- Deeply understand current backbone limitations vs latest research.
- Establish strong evaluation baselines for state robustness and long-horizon dynamics.
- Formalize One-Body constraints as testable rules.

**Key Actions**:
1. Comprehensive audit of current 3:1 GDN2 + Attention implementation (gating mechanisms, state passing, positional handling, fusion points).
2. Literature deep-dive focused on:
   - Hymba (parallel hybrid heads + meta-tokens)
   - Griffin (gated linear recurrence + local attention)
   - ReGLA, Gated DeltaNet, RWKV-7 improvements in gating and delta rules
   - TransXSSM / Unified RoPE for hybrid models
   - Priming and Liger techniques (for efficient future conversion)
   - Titans-style neural long-term memory ideas (adapted to One-Body)
3. Build or strengthen a "State Robustness Eval Suite" that works at current scale and can scale up (beyond the current simple probe).
4. Define quantitative success criteria for each subsequent phase (e.g., diversity stability, degradation under ablation, long-context needle retrieval, etc.).

**Deliverables**:
- Internal "Backbone Audit & Gap Analysis" document
- Updated evaluation harness
- Phase success criteria document

**Risk Mitigation**: Do not start any architecture changes until this phase is complete.

## Phase 1: Strengthen Current Lineage (2-4 months)

**Goal**: Modernize the existing 3:1 GDN2+Atten backbone using latest gating and hybrid insights, while strictly maintaining One-Body. Achieve clear improvements in state tracking and training dynamics stability.

**Focus Areas** (in recommended order):

1. **Gating & Recurrence Upgrade (Highest impact, lowest risk)**
   - Evolve GDN2 toward Gated DeltaNet / ReGLA / RWKV-7 style vector gating + improved delta rule.
   - Add better forgetting / in-context learning rate mechanisms.
   - Keep the 3:1 structure initially.

2. **Positional & State Coherence**
   - Introduce Unified RoPE or equivalent across attention and GDN2 recurrence.
   - Improve state initialization (meta-token style persistent vectors or learned initial state).

3. **Light Hybridization inside One-Body**
   - Experiment with limited parallel computation inside layers (e.g., small number of attention heads running alongside GDN2 heads, then gated fusion).
   - This is a baby step toward Hymba-style designs without breaking the overall 3:1 philosophy yet.

4. **Stochastic Breadth Native Integration**
   - Make stochastic breadth a first-class, deeply integrated mechanism (not a bolt-on).
   - Ensure it interacts cleanly with the improved gating and positional system.

**Evaluation Focus**:
- State ablation robustness (using improved probe)
- Long-horizon coherence and forgetting behavior
- Training stability when applying 5.56 curriculum (especially with real gold data)
- Downstream needle-in-haystack / long-context recall at current scale

**Success Criteria**:
- Clear improvement in state robustness metrics vs current backbone on the same data.
- Stable training of full 5.56 curriculum (including stochastic breadth) at larger batch/length.
- No regression in core language modeling capability.

**Risk Mitigation**: All changes must be ablatable. Roll back immediately if training becomes unstable.

## Phase 2: Architectural Evolution toward Modern Hybrids (4-8 months)

**Goal**: Evolve the backbone toward 2025-2026 best practices (parallel hybrid heads, better memory mechanisms) while preserving One-Body.

**Key Directions**:
- Move from strict layer-wise 3:1 to **intra-layer parallel hybrid heads** (Hymba-inspired), but keep everything inside the single recurrent body.
- Introduce more sophisticated long-term memory mechanisms inspired by Titans (surprise-driven, meta-learning style updates) adapted to One-Body.
- Explore optimal mixing ratios and fusion strategies through systematic ablations.
- Deeply integrate the full 5.56 curriculum (especially gold structural injection and attractor protection) into the new hybrid design.

**Evaluation Focus**:
- Scale up experiments (progressively larger d_model).
- Head-to-head comparison against strong baselines on long-context and reasoning tasks.
- Rigorous ablation studies of each 5.56 component inside the new architecture.

**Success Criteria**:
- The evolved backbone + 5.56 curriculum shows clear, scalable advantages in state robustness over strong ablated versions.
- Training remains stable and efficient.
- Architecture remains clean One-Body.

## Phase 3: Large-Scale Validation & Path B Preparation (6-12+ months)

**Goal**: Validate the matured backbone at scales where the original 5.5x signal could plausibly appear, and prepare for potential full Path B port if needed.

**Activities**:
- Train frontier-relevant size models with the evolved architecture + full 5.56 curriculum.
- Build production-grade long-horizon reasoning evaluation suite.
- Systematic comparison against pure Transformer baselines and other strong hybrids at similar scale.
- If results are promising but still limited by architecture, begin principled design of a "One-Body native large backbone" (either heavily evolved current lineage or clean-slate design informed by all previous phases).
- Document lessons for potential porting to external strong backbones while preserving One-Body philosophy.

**Success Criteria**:
- Reproduce or exceed historical 5.5x-level robustness signals at meaningful scale, or produce a clear, well-documented negative result with lessons.
- Have a production-viable backbone that can serve as the foundation for future work.

## Cross-Cutting Principles (All Phases)

- **One-Body First**: Every design decision must pass the One-Body test.
- **Ablation-Driven Development**: Never add a mechanism without a clear, cheap ablation.
- **Evidence over Hype**: Every phase must produce publishable-quality ablations and scaling curves.
- **Curriculum Co-Design**: The backbone and 5.56 curriculum (especially stochastic breadth) must be developed together, not sequentially.
- **Sustainability**: Prefer incremental, reversible changes. Avoid hero commits that break training for months.

## Risk Management

- **Biggest risk**: Over-investing in architecture improvements without parallel progress on data quality and evaluation.
- **Mitigation**: Every phase must include data and eval improvements in parallel with architecture work.
- **Compute risk**: Large-scale experiments are expensive. Use heavy checkpointing, early stopping criteria, and strong proxy tasks before committing big runs.

## Success Metrics (Overall)

- Clear, reproducible improvement in state robustness / long-horizon reasoning attributable to the 5.56 curriculum on the improved backbone.
- Architecture that remains clean, One-Body, and ablatable even at large scale.
- A credible path (or well-documented impossibility) toward recovering the original 5.5x historical signal.

---

**Status**: This roadmap is the long-term, sustainable plan for backbone improvement under strict One-Body while pursuing Branch B Option 2.

**Next Immediate Action**: User confirmation on starting Phase 0 scope and resource allocation.
