# Reverse I→G→A Decision Record: Stochastic Recurrent Breadth in Active RI-4 Hybrid Engine

**Component / Bias**: Training-time stochastic recurrent breadth (GRAM/PTRM-style prior/posterior sampling) inside the active recurrent engine (OneBodyParallelHybridBlock + answer_state_loop delegation)

**Triggering Event**: Discovery during substrate doubt analysis + "all missing pieces" audit that the historically most-critical inductive bias (the one most completely lost after the new-thought-structure pivot) had never been restored in the actual RI-4 recurrent path being trained and measured.

**Date**: 2026-06

**Author**: Grok (following explicit user directive "git commit 하면서 누락된거 전부 아키텍처에 추가하면서 개선되는지 실험 진행해")

---

## 1. What Was Isolated or At Risk

The only mechanism capable of delivering **training-time generative K-trajectory diversity** inside recurrence (learned prior/posterior on high-level state, true_gram replace mode, clean ablation_zero) lived exclusively in the legacy `state_transition_core.py`.

After the pivot to OneBodyParallelHybridBlock as the real recurrent proposal engine inside answer_state_loop:
- The hybrid block only had skeleton flags + ability to *consume* external noise.
- No learned prior generation happened inside the active engine.
- This made the mandatory ablation declared in `internal-multitrajectory-answer-attractor-ssot.md` ("GRAM/PTRM stochastic breadth off") unexecutable on the primary training path.
- component_registry correctly marked `state_transition_core` as `active_in_primary_onebody_path=False`.

This was the single largest identified cause of the persistent "no selectivity learning" pattern (persistent_carry_rate = 1.0 with zero ablation signal).

---

## 2. Decision Taken

**Chosen Option**: Clean minimal port of self-contained stochastic breadth generation directly into `OneBodyParallelHybridBlock`.

**Rationale**:
- The active recurrent engine for RI-4 (and future experiments) *is* the hybrid block when delegated from answer_state_loop.
- Adding the bias generation capability inside it is the only way to make the historical inductive bias (and the SSOT that depends on it) executable again on the primary path.
- Kept fully One-Body: noise is injected into the recurrent hidden state before fusion; final answer still comes only through the normal LM head.
- Full ablation contract preserved (`core_stochastic_breadth_ablation_zero` must produce perfect identity behavior).

**Implementation Summary**:
- Added learned `stochastic_breadth_prior` network inside the hybrid block (same structure as the partial port in QTRMRecursiveCore).
- Self-generation logic in `forward` (delta mode default + true_gram replace mode supported).
- Respects all existing config flags + ablation_zero.
- Works with the (B,1,D) shapes produced by answer_state_loop delegation.
- External noise path kept for backward compatibility.

---

## 3. Conditions / Constraints

- Must never break the verified RI-4 4-way ablation contract (slots, persistence, router, hybrid participation).
- Must survive `core_stochastic_breadth_ablation_zero=True` with zero behavioral change.
- One-Body preserved at all times.
- No change to the SparseSlotRouter or LEM paths unless they explicitly opt in.

---

## 4. Evidence & Cross-References

- Main gap diagnosis: `docs/wiki/architecture/inductive-bias-map.md` (Stochastic Recurrent Breadth entry)
- Historical reconstruction: `docs/wiki/decisions/2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md`
- SSOT that was being violated: `docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md`
- New process followed: `docs/wiki/process/pivot-safety-and-inductive-bias-preservation.md`
- Code change: `src/wgram_lm/blocks.py` (prior network + self-generation logic in forward)
- Registry update (to be done in same commit): `src/wgram_lm/architecture/component_registry.py`

---

## 5. Post-Decision Verification Plan (Experiments)

After this commit:
- Run real 642 gold continuation with `--enable_stochastic_breadth --core_stochastic_mode true_gram` (or delta).
- Full v2 192 real-heldout measurement + 4-way RI-4 ablation matrix (including stochastic_breadth zero).
- Track: persistent_carry_rate movement, stochastic_diversity during training, ablation drop on carry when breadth is zeroed.
- Compare against previous baselines where the bias was not present in the engine.

If selectivity (carry_rate < 1.0 with clear ablation signal) appears where it previously did not, this constitutes direct causal evidence that restoring the missing training-time trajectory diversity bias was high-leverage.

---

## 6. Notes for Future Pivots

This change was deliberately done as a **Reverse I→G→A** following the new anti-drift process we built after discovering the previous silent loss. Future major engine changes must treat this bias (and the executable gate `scripts/gates/check_ssot_stochastic_breadth.py --strict`) as a first-class constraint.

**This record was created as part of the same change set being committed.**

---

**Status**: Change implemented + this record created. Ready for git commit + experiment wave.