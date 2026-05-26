# 2026-05-26 Stage117 vs Stage118 Generated Algebra Traps

## Summary
Stage118 (fixed-parrot algebra diagnostic, 60 steps) was evaluated per the handoff.

**Result on primary GD gate (official_gdsuite_choice_probe_2pertask, 44 rows, 20 valid):**

- Stage118 last (step 60): **accuracy 1.0**, mean_margin +1.208, min_margin +0.022, **accepted: True**
- All algebra variants now pass (major improvement over Stage117's 0.85 accuracy and negative min margin on algebra).

Stage117 anchor (from handoff):
- accuracy 0.85, mean_margin +1.167, min_margin -0.455, accepted false

**Side-by-Side Comparison (Stage117 last vs Stage118 last)**

**GD Gate (primary target - official_gdsuite_choice_probe):**
- Stage118 last: **accuracy 1.0**, mean_margin +1.208, min_margin +0.022, accepted **True**
- Stage117: accuracy 0.85, mean_margin +1.167, min_margin -0.455, accepted False
- **Clear win** on the exact bottleneck (algebra under misleading repetition).

**Language Heldout (8 cases):**
- Stage118: loss **7.62**, token_accuracy **0.259**
- Stage117: loss 11.05, token_accuracy 0.105
- **Improved** (better than the previous anchor).

**Direct Generation (12 cases):**
- Stage118: exact **0/12**, prefix_token_accuracy **0.133**
- Stage117: exact 1/12, prefix_token_accuracy 0.317
- **Material regression** on generation quality and first-token behavior.

## Decision

Stage118 delivered a **strong targeted improvement** on the persistent algebra-under-misleading-demonstration problem.

However, it caused a clear regression on the direct generation gate (exact match and prefix accuracy), which is important for one-body claims.

**Per handoff promotion rule** ("GD improves **and** language/generation do not materially regress"):

→ **Stage118 is not promoted** as the new main local anchor at this time.

It is recorded as a high-value diagnostic that further "more preference pressure / more fixed-parrot exposure" has diminishing or negative returns on the overall one-body path.

**Current local anchor remains Stage117** (generated algebra traps, 100 steps).

## Recommended Next Move

Escalate to a structural change in how the recurrent state handles the final equation and calculation, rather than continuing scalar preference tuning.

See "Stage119 Direction" section below.

## Decision
Stage118 shows clear progress on the specific "misleading repeated answer + small algebra calculation" bottleneck that has been the persistent wall since Stage113-114.

It is a successful **diagnostic** that "more targeted exposure to the exact failure mode" (fixed parrot numbers close to heldout failures) can move the one-body preference without immediately destroying everything else.

However, because this is still a narrow "fixed-parrot" curriculum on top of the same basic preference objective, it is treated as a strong local repair rather than a fundamental route change.

**Current local anchor remains Stage117** until a full language + generation preservation comparison is completed and documented.

If Stage118 passes language/generation non-regression at similar or better quality than Stage117, it can become the new promoted local checkpoint for this line of work.

## Next High-Probability Direction (if no further scalar gains)
If further "fixed parrot" or stronger CE variants stop moving the remaining hard algebra cases (or regress language), the project should move to a structural change in the answer path rather than more replay pressure:

Candidate: **Stage119 one-body equation-state readback**
- Same BPE reader
- Recurrent state must explicitly bind/reconstruct the final equation fields
- Same hidden state must prefer the solved answer
- Same LM head speaks it

This matches the plain-language diagnosis in the handoff:
"The model does not need more exposure to the exact wrong answer token. It needs a stronger normal answer path for binding the final equation and performing the small calculation before the LM head speaks."

## Commands Run (for reproducibility)
(See handoff document for the exact evaluation loop + preservation gates that were executed.)

## References
- Handoff: docs/wiki/handoffs/2026-05-26-stage118-local-gd-preference-handoff.md
- Previous: docs/wiki/decisions/2026-05-26-stage114-stage116-hard-algebra-followup.md
- Trainer: scripts/625_train_bpe_gd_preference.py + 626 algebra trap builder
