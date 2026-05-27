# Pivot Safety & Inductive Bias Preservation Process

**Purpose**: Prevent high-value historical inductive biases (documented in SSOTs) from silently disappearing during major architectural pivots or refactors.

**Status**: Active process (2026-06). Applies to any change that moves the "primary one-body forward path".

---

## The Failure Mode We Are Defending Against

Example (real incident):
- `internal-multitrajectory-answer-attractor-ssot.md` (2026-05-25) explicitly required "GRAM/PTRM stochastic breadth off (K>1 vs K=1)" as a **mandatory promotion gate ablation**.
- During the "new-thought-structure pivot" (~2026-05-26), the only component that could deliver real training-time stochastic breadth (`state_transition_core.py`) was isolated.
- No Reverse I→G→A decision was recorded.
- Months later, the primary training path (QTRMRecursiveCore → OneBodyParallelHybridBlock + answer_state_loop) had **no executable way** to satisfy the SSOT's own gate.
- The SSOT document still existed. The code reality had drifted.

This process exists so that "we wrote it in the wiki" is never again accepted as sufficient protection.

---

## Level 1: Mandatory Pivot Impact Review (Required Before Any Major Change)

Any of the following triggers a **mandatory Pivot Impact Review**:

- New core recurrence engine (e.g. replacing BlockStack, introducing OneBodyParallelHybridBlock as the actual recurrent proposal engine inside answer_state_loop)
- Structural change to the main forward path that affects how recurrent state (z_h / thought state) is updated
- Isolation, deprecation, or heavy refactoring of any component listed in `component_registry.py` with `status=PROMOTED`
- Changes that affect the ability to run ablations declared mandatory in any architecture SSOT

### Required Output (must be recorded before the pivot branch can be considered closed)

1. **List of affected SSOTs and inductive biases**
   - Which architecture SSOTs mention mechanisms that touch the changed area?
   - Example: `internal-multitrajectory-answer-attractor-ssot.md` → stochastic recurrent breadth (GRAM/PTRM)

2. **Executable status after the pivot**
   - For each critical mechanism: "Will the mandatory ablation still be runnable on the new primary path?"
   - If "no", record the decision: **Port cleanly** / **Replace with equivalent** / **Deliberate discard** (with justification)

3. **Reverse I→G→A Decision Record**
   - If any historical bias is at risk, create (or update) a Reverse I→G→A record (see template below).
   - This record must be linked from the pivot decision log.

4. **component_registry update**
   - Any component whose `active_in_primary_onebody_path` changes must have its registry entry updated in the same PR/branch.

**Enforcement**: This checklist must be copied into the main decision log entry for the pivot (or a dedicated `pivot-impact-review-*.md` file) and cross-referenced.

---

## Level 2: Executable SSOT Gates (Strongly Recommended)

Critical promotion requirements from SSOTs should not remain as wiki text only.

Example gate target:
- `internal-multitrajectory-answer-attractor-ssot.md` → "GRAM/PTRM stochastic breadth off" must be executable.

See:
- `scripts/gates/check_ssot_stochastic_breadth.py` (created 2026-06)

This script (or equivalent) should be runnable as part of any promotion or major experiment triage. If the mechanism is missing from the primary path, the gate fails loudly.

---

## Level 3: Cultural & Code-Level Defenses

1. **Loud runtime / import warnings** in `core.py`, `config.py`, and `component_registry.py` whenever a declared critical bias is known to be missing from the active path.
2. **Never treat "we documented it" as protection.** The existence of an SSOT is evidence of intent, not evidence of implementation.
3. **Reverse I→G→A is a first-class deliverable**, not an afterthought. It carries the same weight as shape contracts or ablation matrix closure.
4. Reviewers of major refactors are required to ask: "Which SSOTs will become non-executable because of this change?"

---

## Quick Reference: Current Known At-Risk Biases (2026-06)

- GRAM/PTRM training-time stochastic recurrent breadth (true_gram / prior-posterior sampling on z_h during recurrence)
  - SSOT: `internal-multitrajectory-answer-attractor-ssot.md`
  - Historical signal owner: 5.53~5.56 Adaptive Rehearsal results
  - Current status: Only partial I-stage port in `QTRMRecursiveCore`. Not active in the RI-4 OneBodyParallelHybridBlock + answer_state_loop primary path.
  - Registry entry: `state_transition_core`

Update this list whenever a new Pivot Impact Review is completed.

---

## Related Documents

- `docs/wiki/architecture/inductive-bias-map.md`
- `docs/wiki/decisions/2026-05-30-historical-signal-reconstruction-stochastic-breadth-pivot-gap.md`
- `docs/wiki/architecture/internal-multitrajectory-answer-attractor-ssot.md`
- `src/qtrm_mm/architecture/component_registry.py`
- Research-driven-architecture-debugging skill (Pivot Safety + Reverse I→G→A section)

**Last updated**: 2026-06 (after the discovery that the IMTA SSOT's core ablation was unexecutable on the active RI-4 substrate)