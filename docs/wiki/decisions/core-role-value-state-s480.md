# Core Role-Value State S480

Date: 2026-05-05

## Question

Does moving role-bound value tokens into the mandatory recurrent core make
QTRM learn exact internal value state, instead of merely reading values out
after the action trajectory is already computed?

## Prior

This candidate follows the role/filler and slot-binding line summarized in:

- `docs/wiki/sources/role-filler-variable-slots.md`

The key design change from the rejected readout-only role-value head is that
role tokens are appended to the workspace before the recursive core runs. The
core therefore updates action/workspace tokens and role-value tokens in the
same recurrent loop.

## Experiment

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_state_s120.yaml
```

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_s480_from_mixed_s720/last.pt
```

Training:

```text
steps: 480
init: accepted mixed-composition action checkpoint
trainable policy: core_role_value_state_only
loss: core role-value CE only
```

Held-out core role-value result:

```text
rows:                 32
trace exact:          0/32
step exact:           0/256
value accuracy:       100/624 = 0.1603
```

Action-code preservation:

```text
trace exact:          32/32
halted exact:         32/32
step accuracy:        1.0000
finality accuracy:    1.0000
```

## Decision

Partial scaffold accepted, final value-state claim rejected.

The core-role path improves value accuracy over the rejected readout-only
role-value path:

```text
readout-only role values: 48/624  = 0.0769
core role values:        100/624 = 0.1603
```

However, exact state reconstruction remains 0/32. This is not enough to claim
that QTRM has learned a reliable latent value transition law.

## Root Cause Hypothesis

The current run keeps the existing recurrent core frozen. The new role tokens
and classifier can exploit some latent trajectory structure, but the core is
not trained to make those role-value tokens causally useful for subsequent
recursive steps.

In other words, the signal moved into the recurrent loop, but the recurrent
transition itself did not receive enough value-state pressure.

## Next Candidate

Train the recurrent core and role-value tokens jointly while preserving the
accepted action controller:

```text
loss =
  action-code CE / finality CE preservation
  + core role-value CE
  + transition-off / shuffle / dropout ablation gate
```

Acceptance gate:

```text
action-code exact: remains near 32/32
value accuracy:    beats 0.1603
trace exact:       rises above 0/32
ablation:          role/core transition off lowers value metrics
```

Kill criterion:

```text
If joint core training still improves in-sample role accuracy but held-out
trace exact remains 0/32, stop adding heads and generate explicit
step-to-step role-transition supervision.
```
