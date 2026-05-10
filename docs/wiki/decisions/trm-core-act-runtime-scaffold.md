# TRM Core ACT Runtime Scaffold

Status: implementation scaffold, 2026-05-07.

## Decision

Add two TRM-faithful runtime mechanics to `QTRMRecursiveCore` behind explicit
config flags:

```text
core_halt_init_bias
core_trm_no_grad_inner_cycles_enabled
core_halt_freeze_halted_state_enabled
core_halt_exploration_prob
core_halt_exploration_min_steps
core_halt_loss_mode=q_value
core_halt_q_value_gamma
```

Also add an explicit detached carry API:

```text
QTRMCoreCarry(z_l, z_h, halted, steps)
QTRMRecursiveCore(..., carry=..., return_carry=True)
QTRMMultimodalModel(..., core_carry=..., return_core_carry=True)
scripts/192_eval_raw_intelligence.py mode:
  qtrm_core_halt_carry_steps_N_no_evidence
```

These are not new reasoning heads. They make the existing TRM/QTRM recursive
core behave closer to Samsung TRM's update schedule and ACT wrapper.

## Implemented

`core_halt_init_bias=-5.0` zero-initializes the halt head weights and sets both
halt/continue biases to a conservative negative value, matching the official
TRM practice of preventing random early halt behavior at initialization.

`core_trm_no_grad_inner_cycles_enabled=true` changes each outer recurrent step
to:

```text
H_cycles - 1 inner H/L cycles: no grad
final H/L cycle: grad
```

This follows the official TRM pattern where most inner recurrence is compute
and only the last inner cycle receives direct backpropagation.

`core_halt_freeze_halted_state_enabled=true` makes halted batch rows keep their
own `z_L/z_H` state while other rows continue until the batch reaches
`outer_steps` or all rows halt. The output `core_steps` becomes per-sequence
when this mode is enabled.

`QTRMCoreCarry` stores detached `z_L/z_H`, per-row `halted`, and per-row
`steps`. On the next call, rows whose previous carry has `halted=true` reset to
the fresh workspace state, while unfinished rows continue from their previous
latent state. This is the first reset-on-halt across calls scaffold.

`core_halt_exploration_prob > 0` delays early halt during training for sampled
rows until at least `core_halt_exploration_min_steps`. This follows the
official TRM idea that the halt head should not immediately collapse to the
first confident stop during training. Exploration is training-only; eval halt
behavior remains deterministic.

`core_halt_loss_mode=q_value` makes the halt head train as a two-action value
head instead of only as two independent BCE classifiers. `q_halt` regresses
toward the value of stopping at the current depth, and `q_continue` regresses
toward discounted continuation value via `core_halt_q_value_gamma`. This better
matches the runtime rule `halt if q_halt > q_continue`.

`qtrm_core_halt_carry_steps_N_no_evidence` is the first task-level eval harness
for the carry API. In causal forced-choice and generation scoring, it feeds the
previous forward's `core_carry` into the next token/prefix forward and requests
the next detached carry. This keeps the path no-retrieval and no-hidden-
evidence while measuring whether recurrent state continuation helps.

## Boundaries

Still missing before claiming full TRM faithfulness:

- accepted held-out depth/ACT gate proving the carry runtime improves raw
  reasoning over no-carry/core-off/donor baselines.

This scaffold is therefore the first runtime correction, not the full TRM
training recipe.

## Tests

Added tests in `tests/test_core_halting.py` and `tests/test_losses.py`:

```text
test_core_halt_head_uses_trm_conservative_initialization
test_core_trm_no_grad_inner_cycles_only_backpropagates_last_h_cycle
test_core_respects_outer_torch_no_grad_context
test_core_trm_act_freezes_halted_samples_until_batch_finishes
test_core_returns_detached_explicit_carry_for_continuation
test_core_resets_halted_carry_rows_on_next_call
test_model_forward_can_return_and_reuse_core_carry
test_core_carry_is_public_package_api
test_core_halt_exploration_delays_early_halt_during_training_only
test_core_halt_loss_can_train_q_value_halt_continue_targets
test_qtrm_smoke_loss_can_use_q_value_core_halt_mode
test_mode_runtime_maps_core_halt_carry_without_hidden_evidence
test_score_case_record_marks_core_carry_runtime
test_causal_choice_can_reuse_core_carry_across_prefix_steps
```

Promotion gate:

```text
core_trm_no_grad_inner_cycles_enabled=true
core_halt_freeze_halted_state_enabled=true
core_halt_exploration_prob>0 during halt-head training
core_halt_loss_mode=q_value for halt-head training
qtrm_core_halt_carry_steps_N_no_evidence included in ACT eval
full > core_off
depth/ACT gate improves held-out raw reasoning
generation remains a separate renderer claim
```
