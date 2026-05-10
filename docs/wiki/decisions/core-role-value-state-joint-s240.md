# Core Role-Value State Joint S240

Date: 2026-05-05

## Question

If role-bound value tokens already live inside the recurrent core, does opening
the core itself during training improve held-out value-state recovery while
preserving the accepted action controller?

## Experiment

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_state_joint_s120.yaml
```

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_s240_from_core_role_s480/last.pt
```

Training:

```text
steps: 240
init: core-role-value-state S480 scaffold
trainable policy: core_and_role_value_state
loss:
  transition-state joint CE: 1.0
  core role-value CE:        1.0
```

## Result

Held-out core role-value:

```text
rows:                 32
trace exact:          0/32
step exact:           0/256
value accuracy:       144/624 = 0.2308
```

Action-code preservation:

```text
trace exact:          32/32
halted exact:         32/32
step accuracy:        1.0000
finality accuracy:    1.0000
```

Progression so far:

```text
generic algorithmic slots: 0.0500
role readout-only:        0.0769
core role tokens:         0.1603
joint core role tokens:   0.2308
```

## Decision

Accept the direction, reject final exact-state claim.

Opening the recurrent core improves held-out value accuracy without regressing
the accepted action-code policy. That is a real causal-architecture signal:
the value path should be part of the mandatory recurrent core, not a detached
readout.

However, exact value-state recovery is still 0/32. This is not yet a reliable
algorithmic value transition model.

## Next Bottleneck

The model is still trained mainly on final role values for each observed depth.
It is not forced to learn the explicit step-to-step value transition law.

The next candidate should add transition-level supervision:

```text
previous role-value state
  + current action/state code
  -> next role-value state
```

Acceptance gate:

```text
value accuracy:  > 0.2308
trace exact:     > 0/32
action-code:     remains near 32/32
ablation:        previous-role-state off/shuffle drops value metrics
```

Kill criterion:

```text
If explicit step-to-step supervision still produces trace exact 0/32, stop
trying to recover exact integer/list values through classification heads and
switch to a neural-symbolic scratchpad/runtime-value representation.
```
