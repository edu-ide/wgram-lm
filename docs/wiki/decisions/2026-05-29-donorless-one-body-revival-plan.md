# 2026-05-29 Donorless Born-One-Body Track Revival Plan (Option 2)

**Date**: 2026-05-29  
**Context**: After successfully unlocking real ablation margin in the hybrid (Qwen donor + RI-4 dynamic slots) track, the user explicitly requested parallel progress on the pure donorless / HRM-Text-style born-one-body direction.

## Strategic Rationale

The hybrid track (even with dynamic memory) still has the fundamental limitation that a strong frozen Qwen backbone supplies both:
- The initial token representations (reader)
- A large fraction of the final output distribution (base policy)

True claims about "our recurrent core + memory system performing reasoning" are always partially confounded. The cleanest falsifiable path remains a **fully native, donor-free, one-body student** trained from (near) scratch or from a small native BPE/BLT reader.

This is the spiritual continuation of:
- The `ri7_clean_reference/` minimal donorless one-body scaffold
- `scripts/260_train_donorless_recurrent_depth_probe.py` line of experiments
- The HRM-Text one-body contract (reader → recurrent core → same LM head, no side bridges)

## Current Assets (as of 2026-05-29)

Positive:
- `OneBodyParallelHybridBlock` + internal Griffin-style fast recurrence + Parcae stability + RI-4 SparseSlotRouter is now the strongest recurrent engine we have ever had inside a true one-body block.
- The dynamic write patch + surprise modulation are now proven to deliver measurable capability.
- `src/qtrm_mm/models/blt_prefixlm.py` and related BLT one-body components already exist with `--decoder-latent-mode one_body` support.
- `references/ri7_clean_reference/` contains a clean, minimal contract + trainer for donorless RI-1~RI-7 gates.

Gaps / Risks:
- Most recent large-scale training recipes and data pipelines are heavily entangled with the Qwen donor adapter path.
- Native tokenizer + embedding (BPE or semantic BLT) + small random-init LM head has historically shown slow language acquisition and repetition collapse before any interesting depth scaling appears.
- No recent "donorless + current OneBodyHybrid + active memory" joint run exists at meaningful scale.

## Proposed Minimal Revival Path (Low-Risk, High-Signal)

Phase 0 (1–2 days)
- [ ] Create a clean "donorless_hybrid_ri4" training entrypoint that re-uses the current best `OneBodyParallelHybridBlock` stack + SparseSlotRouter + surprise write, but with:
  - Pure BPE (or current best stable reader) tokenizer
  - Small randomly initialized embedding + LM head
  - No `QwenDonorAdapter` at all
  - Same 5.56-style rehearsal + gold curriculum where possible
- [ ] Port the minimal data loading + heldout gate harness from `ri7_clean_reference/` or `260_*` into this new script so we can run the exact same RI gates without donor contamination.

Phase 1 (Validation)
- Tiny-scale (d128~d256) overfit + depth scaling probe on synthetic logical/arithmetic data (same as the historical 260 success).
- Confirm that with the new dynamic write + surprise logic, we still see clean depth gain and memory ablation drop in the **complete absence** of any external backbone.

Phase 2 (Language)
- Small native pretraining run (BPE + OneBodyHybrid + active memory) on the same general text mix used in recent successful hybrid runs, but without donor hidden injection or donor logit fusion.
- Primary success criteria: non-degenerate free generation + measurable held-out CE improvement from added recurrence depth and memory.

Phase 3 (Scale Decision)
- Only after Phase 1+2 show clean signals do we consider larger donorless runs. Otherwise we treat the hybrid track (now with proven memory) as the pragmatic research vehicle and keep the donorless track as the "pure reference" for architectural claims.

## Immediate Next Concrete Steps

1. Inventory the exact minimal set of components needed to run a donorless version of the current best hybrid recipe (tokenization, data, model construction without donor, training loop).
2. Produce a single launch script + config example (e.g. `scripts/630_train_donorless_hybrid_ri4_minimal.py` or equivalent) that a new researcher can run end-to-end with one command.
3. Run the synthetic logical depth probe first (fastest falsifier).

## Relationship to the Hybrid Track

These two tracks are **not in competition** for the next 4–6 weeks. The hybrid track (now with working memory) is the fastest way to generate publishable "memory matters" numbers at 2B-donor scale. The donorless track is the only way to make uncontaminated architectural claims about the recurrent core itself.

Both should be kept alive and cross-pollinated (lessons from surprise write, internal fast recurrence, etc. should flow both ways).

## Decision Record

User explicitly chose to document the RI-4 write-lock breakthrough in the wiki **and** to advance Option 2 in parallel. This document is the authoritative planning artifact for that second axis as of 2026-05-29.