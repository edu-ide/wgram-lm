# Phase 0: Backbone Audit & Literature Synthesis (One-Body Strict)

**Date**: 2026-05-30  
**Status**: In Progress

## 1. Current Backbone Characterization (Initial Audit)

From code inspection (`src/qtrm_mm/mixers.py`, `core.py`, `qwen_backbone_qtrm.py`):

- The backbone is a **Qwen3.6-inspired decoder** with a custom hybrid recurrence layer.
- Core recurrence primitive: **Gated DeltaNet-2** (via `OfficialGatedDeltaNet2Mixer` + FLA backends) + Attention.
- The "3:1" structure appears to be a layer-wise or block-wise ratio of Gated Delta (GDN2) blocks to Attention blocks.
- Strong emphasis on One-Body: everything flows through the main recurrent state; no obvious side memory banks in the core path.
- They already use modern gated recurrence (Gated DeltaNet-2), which aligns well with 2025-2026 research (Gated DeltaNet frequently appears as one of the stronger options in hybrid ablations).

**Strengths identified**:
- Already using one of the better gated recurrence primitives (GDN2).
- Clean One-Body design philosophy.
- Good adapter structure for different backends (FLA, official references, torch fallback).

**Gaps vs 2025-2026 SOTA** (preliminary):
- Still using relatively traditional **layer-wise alternation** (3:1 style) rather than modern **intra-layer parallel hybrid heads** (Hymba style is currently stronger in many ablations).
- Gating, while advanced, can still benefit from latest vector gating + delta rule refinements (RWKV-7, ReGLA).
- Limited evidence of meta-tokens / learned persistent state initializers (Hymba strength).
- Positional handling between recurrence and attention may not be fully unified.

## 2. High-Impact Paper Recommendations (One-Body Compatible)

Prioritized for your 3:1 GDN2 + Attention lineage + strict One-Body:

### Tier 1 (Highest immediate impact, relatively contained changes)

1. **Hymba-style Parallel Hybrid Heads** (arXiv 2411.13676 + follow-ups)
   - Instead of sequential 3 GDN2 then 1 Attention, run GDN2 heads and Attention heads **in parallel inside the same layer**, then fuse.
   - This is currently one of the strongest patterns in 2025-2026.
   - Can be done while keeping One-Body (everything still happens inside the layer's hidden state flow).
   - Recommendation: Start experimenting with 3:1 or 4:1 parallel head ratio (recurrence heads : attention heads) inside layers.

2. **Gated DeltaNet + RWKV-7 Vector Gating Improvements** (ReGLA, RWKV-7, Gated DeltaNet papers)
   - Your GDN2 is already close. The latest refinements focus on:
     - Vector-valued (per-channel) gating instead of scalar.
     - Better delta rule formulations.
     - Improved normalization and stability techniques (ReGLA).
   - This is a high-ROI change because it directly upgrades the recurrence primitive you already use.

3. **Meta-tokens / Learnable Persistent State** (Hymba + Titans ideas)
   - Learnable vectors prepended or used as initial state.
   - Acts as a form of persistent memory / cache initializer.
   - Very One-Body friendly (just part of the initial hidden state or prefix).

### Tier 2 (Important but more architectural)

4. **Unified Positional Encoding** (TransXSSM / Unified RoPE approaches)
   - Apply consistent rotary or relative position information to both attention and the recurrence state update.
   - Reduces misalignment issues common in hybrids.

5. **Better Fusion Strategies**
   - Learnable gating or normalization-based fusion between parallel recurrence and attention outputs (Hymba ablations are useful here).

## 3. Recommended Phase 1 Focus (First 3-4 months)

Based on the audit and literature:

**Priority Order**:
1. Upgrade the Gated DeltaNet-2 gating mechanism with latest vector gating + delta refinements (biggest quality win with contained change).
2. Introduce parallel hybrid heads (move away from strict sequential 3:1 toward intra-layer parallel).
3. Add meta-token / learnable persistent state initializer.
4. Unify positional encoding across recurrence and attention.

All changes must be:
- Fully ablatable (especially stochastic breadth).
- Keep the overall One-Body flow.
- Tested with the 5.56 curriculum (including real gold data where possible).

## 4. Evaluation Upgrades Needed in Parallel

- Improve the current state ablation robustness probe (the v2 structured input version is a good start, but needs further work on input quality and metric).
- Add long-context needle-in-haystack style evals that scale with model size.
- Build systematic ablations for each 5.56 component inside the new hybrid design.

## 5. Risk & Sequencing Notes

- Do not attempt full parallel hybrid head redesign until gating upgrade (item 1) shows clear gains.
- Keep a strong "current best" branch at all times.
- Every major change must be validated with at least one 5.56 curriculum training run (even if small scale initially).

---

**Next Action (Phase 0 continuation)**: Deep dive into specific implementation of GDN2 in your references + detailed proposal for "Gating v2" based on latest Gated DeltaNet / RWKV-7 techniques.

This document will be iteratively updated as Phase 0 progresses.

## 3. Detailed Findings on Current Recurrence Primitive (GDN2)

From deeper inspection of `mixers.py`:

The project already has a sophisticated gated recurrence system:
- `OfficialGatedDeltaNet2Mixer`: Adapter for the official GatedDeltaNet-2 (from NVlabs gated-deltanet-2 repo via lit_gpt).
- `FLADeltaMixer`: Adapter for flash-linear-attention implementations of GatedDelta / KDA.
- `TorchGatedDeltaMixer`: Pure PyTorch reference implementation (for debugging).

This means the current "GDN2" is already one of the stronger modern gated linear recurrence primitives (Gated DeltaNet family is frequently ranked very high in 2025-2026 hybrid ablations, often beating or matching Mamba-2 in expressiveness when combined with attention).

The "3:1 gdn2:atten" structure the user mentioned likely refers to how these GDN2 mixers are alternated with standard Attention layers inside the Qwen-based stacks (in `qwen_backbone_qtrm.py` or custom layer definitions), using a 3 recurrence : 1 attention ratio per block or per stage.

**Current Strengths**:
- Already using a top-tier gated recurrence (GatedDeltaNet-2).
- Good engineering with multiple backends (official, FLA, torch fallback).
- Clean separation via mixer adapters.

**Current Limitations (vs latest papers)**:
- The gating logic inside the official GDN2 is from ~2024. Papers like ReGLA (2025), RWKV-7 (2025), and latest Gated DeltaNet refinements have proposed specific improvements in:
  - Vector-valued (per-channel) gating instead of simpler forms.
  - Better delta rule formulations and in-context learning rate mechanisms.
  - Improved normalization and stability techniques to prevent vanishing/exploding issues in long recurrence.
- The hybrid arrangement is still relatively traditional layer-wise alternation. Latest strong results (Hymba 2024/2025 and follow-ups) favor **parallel hybrid heads** within the same layer (recurrence heads + attention heads running in parallel, then fused).

## 4. Recommended First Improvement: Gating v2 (Highest Leverage, Lowest Risk)

**Proposal for Phase 1, Step 1**:

Upgrade the recurrence gating mechanism from current GDN2 to a "Gated DeltaNet-2 + ReGLA/RWKV-7 refinements" version.

Specific technical directions (drawn from latest papers):

- Adopt **vector gating** (independent gate per head or per channel) as emphasized in RWKV-7 and latest Gated DeltaNet work.
- Incorporate refined delta rule with explicit "in-context learning rate" or forget gate modulation (RWKV-7 style).
- Add the normalization and variance control techniques from ReGLA for better training stability at depth.
- Keep the overall interface identical so the rest of the One-Body system (z_l / z_h dual recurrence, 5.56 curriculum integration, stochastic breadth) does not need changes.

**Why this first?**
- Highest expected quality gain per engineering hour.
- Directly builds on what they already have (they are already on GatedDeltaNet-2).
- Fully compatible with strict One-Body.
- Can be A/B tested cleanly against current GDN2.
- Provides a stronger recurrence primitive before attempting the bigger architectural shift to parallel hybrid heads.

**Risk**: Low. Can be implemented as a drop-in improved mixer.

**Next micro-step (if user approves)**: I can produce a detailed technical spec + pseudocode for the "Gated DeltaNet-2 v2" mixer, including exact equations from the key papers, adapted to their current interface.

This keeps everything strictly sequential and low-risk while moving toward a backbone that can better support the 5.56 curriculum at scale under One-Body constraints.
