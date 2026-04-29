# Recurrent-Depth Transformers

Recurrent-depth transformers reuse one or more blocks multiple times along the
depth axis. They trade extra inference/training FLOPs for fewer unique
parameters.

Useful QTRM design ideas:

1. Prelude / recurrent block / coda split.
   Let the prelude encode tokens once, run a bounded recurrent workspace, then
   decode through a coda. QTRM already has a workspace core, but its boundaries
   should be documented and ablated.

2. Stable input injection.
   The recurrent state should keep access to the original encoded input. Parcae
   uses a diagonal stable update with decay in `(0, 1)`, while current QTRM uses
   spectral-normalized projection plus a gate. The Parcae-style decay is easier
   to reason about and easier to test.

3. Recurrence-depth evaluation.
   A looped model should be evaluated at multiple depths, not only the training
   depth. Depth sweeps can show whether extra loops help, saturate, or collapse.

4. Gradient budget control.
   Parcae separates no-grad recurrence steps from backpropagated recurrence
   steps. QTRM currently detaches only after outer steps. A no-grad/with-grad
   split may reduce memory while preserving long recurrent computation.

5. Loop-index conditioning.
   Shared recurrent weights may need a depth signal so early and late iterations
   can specialize. Current QTRM's `StableInject` has a learned loop embedding,
   which is aligned with this idea.

6. Telemetry before claims.
   Track recurrent state norm, recurrent residual norm, contraction/spectral
   factor, depth-sweep validation loss, and logit entropy. Without these, a
   repeated-token collapse can hide inside the recurrent block.

Do not assume:

- More loops always improve reasoning.
- Latent recurrence is equivalent to chain-of-thought.
- A speculative community reconstruction is a source of truth for a closed
  frontier model.

Current QTRM action:

- Add tests/metrics for stable injection contraction and recurrence depth sweep.
- Compare current `StableInject` with a Parcae-style diagonal injection behind a
  config flag.
- Keep TRM carry/ACT and Parcae looped-LM recurrence as separate design axes.

## Classification Wording

QTRM can be called looped or recurrent-depth in a limited architectural sense:
the recursive core repeatedly updates latent workspace states `z_l` and `z_h`.
It should not be called a standalone loop LM yet, because Qwen donor logits are
the base language policy and QTRM currently contributes a residual.

Preferred wording:

```text
QTRM is a Qwen-backed looped latent-workspace residual adapter.
```

Avoid:

```text
QTRM is a standalone loop LM.
QTRM has proven latent reasoning.
```
