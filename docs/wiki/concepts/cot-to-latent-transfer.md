# CoT To Latent Transfer

Status: architecture concept, 2026-04-29.

## Position

QTRM should not be trained to always print long chain-of-thought at inference.
The more defensible design is:

```text
explicit CoT / verifier trace = teacher supervision
latent workspace loop        = internal repeated reasoning
short answer / action        = final output
```

This aligns QTRM with Coconut/CODI/looped-transformer references while avoiding
the weak claim that hidden states are automatically faithful reasoning.

## Training Path

1. **Trace SFT bootstrap**
   Train on short structured traces for retrieval, conflict, abstention, and
   critical synthesis. These traces teach the target reasoning policy.

2. **Latent distillation**
   Use explicit traces as teacher runs and train QTRM latent states to carry the
   same decision information. CODI suggests hidden-state alignment around a
   designated latent token; QTRM's designated latent target is the workspace
   `z_H` state.

3. **Residual correctness**
   The fused output must improve target answer likelihood or answer accuracy
   over donor-only. Otherwise the latent loop is not useful.

4. **Halting supervision**
   Train a halt head from correctness and confidence signals:

   ```text
   target_halt = answer_is_correct and verifier_passed and residual_is_stable
   target_continue = not target_halt and steps < max_steps
   ```

5. **Inference**
   Run QTRM latent loops until either the halt head stops or `halt_max_steps` is
   reached, then emit only a short answer, `NEEDS_SEARCH`, or a verifier-facing
   structured result.

## Current Implementation

Implemented in code:

- `QTRMConfig.core_halt_enabled`
- `QTRMConfig.core_halt_min_steps`
- `QTRMConfig.core_halt_use_continue`
- `QTRMRecursiveCore.halt_head`
- `TrainConfig.loss_core_halt_weight`
- `TrainConfig.core_halt_auto_targets`
- `TrainConfig.core_halt_donor_kl_threshold`
- `core_halt_loss`
- `infer_core_halt_targets`
- model outputs:
  - `core_q_halt_logits`
  - `core_q_continue_logits`
  - `core_halted`
  - `core_steps`
- `enable_core_halt` forward flag

Current behavior:

- If `core_halt_enabled=False`, existing behavior is unchanged.
- If `core_halt_enabled=True`, the core records halt/continue telemetry.
- If `enable_core_halt=True`, the current batch can stop early when all samples
  satisfy the halt decision.
- If training data supplies `core_halt_targets`, `qtrm_smoke_loss` can train
  `core_q_halt_logits` through `loss_core_halt_weight`.
- If `core_halt_auto_targets=True`, `qtrm_smoke_loss` can infer halt targets
  from exact token correctness, optional verifier pass/fail, and optional
  fused-vs-donor KL stability.

Important limitation:

The current implementation is not full TRM ACT. It does not yet have persistent
carry, per-sequence halt/reset, or halt exploration. Automatic target
construction exists, but it is a conservative proxy, not a proof that the latent
loop has reasoned faithfully.

## Automatic Halt Target Rule

The current automatic halt target is:

```text
target_halt =
  exact_next_token_correct
  and optional verifier_passed
  and optional KL(fused_logits || donor_logits) <= threshold
```

This is intentionally strict. It only tells the halt head "this state appears
safe enough to stop"; it does not teach which intermediate latent step was
semantically necessary. That stronger signal requires step-wise teacher runs or
TRM-style exploration.

## Why CoT Is Not Removed

CoT remains useful for:

- generating teacher traces;
- verifying whether the model followed the intended reasoning policy;
- constructing preference pairs and self-improvement examples;
- auditing failure modes.

CoT should not be treated as the production output format for QTRM unless the
task explicitly asks for explanation. Long visible CoT would make QTRM behave
like a text-pattern LM instead of a compact latent reasoning adapter.

## Required Evals

Before claiming latent reasoning:

- `residual > donor_only` on held-out evidence-sensitive tasks;
- `residual > workspace_off`;
- `residual > core_off`;
- halting saves steps without lowering accuracy;
- latent distillation beats trace-SFT-only at equal token budget;
- adversarial/distractor tests do not show shortcut dependence.

If these fail, the correct conclusion is that QTRM has useful engineering
hooks, not proven latent reasoning.
