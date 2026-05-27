# Reverse I→G→A Decision Record Template

**Component / Bias**: [e.g. state_transition_core / GRAM/PTRM training-time stochastic recurrent breadth]

**Triggering Event**: [e.g. "new-thought-structure pivot", "replacement of QTRMRecursiveCore with OneBodyParallelHybridBlock as the recurrent engine inside answer_state_loop", etc.]

**Date**: YYYY-MM-DD

**Author / Reviewer**:

---

## 1. What Was Isolated or At Risk

Describe the historical mechanism and the SSOT(s) that depend on it.

Example:
> The only implementation of training-time prior/posterior sampling on the high-level recurrent state (`_apply_true_gram_transition` + stochastic_high_level_guidance in `state_transition_core.py`) was isolated during the pivot for checkpoint compatibility reasons. `internal-multitrajectory-answer-attractor-ssot.md` explicitly lists "GRAM/PTRM stochastic breadth off (K=1 vs K>1)" as a mandatory promotion gate ablation.

---

## 2. Why It Happened

- Technical reason (compatibility, shape contracts, performance, etc.)
- Process reason (time pressure, Most-Deficient focus, "we'll come back to it later")

---

## 3. Decision Options Considered

| Option | Description | Pros | Cons | Recommended? |
|--------|-------------|------|------|--------------|
| Clean Port | Re-implement the bias inside the new primary path with full ablation_zero contract | Preserves historical signal + SSOT compliance | Engineering cost + risk of changing recurrence dynamics | |
| Equivalent Replacement | Design a new mechanism that delivers similar training-time trajectory diversity | Opportunity to improve | May not be equivalent in strength | |
| Deliberate Discard | Explicitly decide that the new substrate does not need / should not have this bias | Honest, reduces complexity | Loses the inductive bias that contributed to past positive signals | |
| Temporary Deferral | Document that we will revisit within N weeks / after specific milestone | Pragmatic | High risk of becoming permanent drift | |

---

## 4. Decision Taken

**Chosen Option**: [Port / Replace / Discard / Defer]

**Justification** (1-2 paragraphs):

**Conditions / Constraints**:

**Owner** (who is responsible for executing or monitoring):

**Deadline / Review Date**:

---

## 5. Evidence & Cross-References

- SSOT(s) impacted:
- Historical signal reconstruction:
- Inductive bias map entry:
- Related decision logs / pivot review:
- Code locations (before and after):

---

## 6. Post-Decision Verification

- [ ] The chosen option has been implemented (or explicitly not implemented in the case of Discard)
- [ ] The relevant `component_registry.py` entry has been updated (`active_in_primary_onebody_path`)
- [ ] `scripts/gates/check_ssot_stochastic_breadth.py` (or equivalent) passes or fails as expected
- [ ] The main architecture decision log for the triggering pivot references this record

**Verification Date**:
**Verifier**:

---

## 7. Notes for Future Pivots

Any lessons or process improvements discovered while doing this Reverse I→G→A.

---

**This template must be filled and linked from the Pivot Impact Review whenever a PROMOTED component or SSOT-declared mandatory bias is at risk during a major architectural change.**