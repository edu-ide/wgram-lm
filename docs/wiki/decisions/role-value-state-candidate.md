# Role-Value State Candidate

Date: 2026-05-05

## Question

Can QTRM learn exact algorithmic value content if each value is bound to a
stable role, instead of being predicted from generic slots?

## Previous Failure

The generic structured slot path was rejected:

```text
kind accuracy:          1.0000
content slot accuracy:  0.0500
trace exact:            0/32
action-code exact:      32/32
```

Root cause: one generic vocabulary had to represent unrelated semantics
including raw list offsets, doubled list offsets, scalar coefficients,
residuals, final residuals, and padding.

## Prior-Backed Candidate

Use role-conditioned variable slots:

```text
accepted recursive action trajectory
  -> factorized recurrent value slots
  -> stable role embeddings
  -> role cross-attention over value slots
  -> shared value classifier per role
```

Stable roles for the first mixed-composition gate:

```text
0..3  raw list offsets
4..7  doubled list offsets
8     scalar coefficient
9     scalar/final residual
```

This borrows the role/filler pressure from TPR-style representations and the
slot-binding pressure from Slot Attention, while keeping the QTRM path neural
and ablatable.

## Experiment

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_role_value_state_s120.yaml
```

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_role_value_state_s480_from_mixed_s720/last.pt
```

Training:

```text
steps: 480
trainable policy: role_value_state_only
loss: role-value CE only
init: accepted mixed-composition action checkpoint
```

Held-out role-value result:

```text
rows:                 32
trace exact:          0/32
step exact:           0/256
value accuracy:       48/624 = 0.0769
```

Action-code preservation:

```text
trace exact:          32/32
halted exact:         32/32
step accuracy:        1.0000
finality accuracy:    1.0000
```

Observed prediction pattern:

```text
role values collapse to repeated classes such as 1, 5, and 13
```

## Decision

Reject as a final value-bearing state architecture.

The role-conditioned path is a real improvement over the generic-slot baseline
on value accuracy, but it still fails the exact-state gate. It preserves the
accepted action controller, which is useful, but the values are still a readout
probe over the action trajectory rather than a causal recurrent variable.

## Acceptance Gate Used

```text
held-out eval:
  content role accuracy > 0.05 generic-slot baseline
  trace exact > 0/32

preservation:
  action-code exact remains 32/32

causality:
  role-value path off/shuffle/dropout should drop role/value metrics
```

## Kill Criterion

If content role accuracy stays near the generic-slot baseline or trace exact
remains 0/32 after the short falsification run, reject readout-only role slots.
The next candidate must move role-bound values into the recurrent core update,
not add another probe head.

The kill criterion fired on 2026-05-05.

## Next Candidate

Move role-bound values into the mandatory recurrent state:

```text
prompt tokens
  -> workspace/core
  -> action-state update
  -> role-value recurrent update
  -> next core step reads role-value state
  -> role-value logits
```

The falsifiable difference from this rejected candidate is that the next core
step must receive the previous role-value state, not merely expose a side
readout after the action trajectory has already been computed.
