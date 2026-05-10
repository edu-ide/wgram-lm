# Core Role-Value State Len579 S240

Date: 2026-05-05

## Question

Was the exact value-state failure caused only by the architecture, or also by
the gate data omitting longer list lengths from the role-value training split?

## Gate Fix

The previous mixed-only train split used list length 5, while the held-out eval
used lengths 7 and 9. That made roles 3 and 7 plus wider residual classes
largely unseen during value-state training.

The mixed composition builder now accepts train list lengths:

```text
scripts/235_build_mixed_family_composition_gate.py
  --train-list-lengths 5,7,9
```

Generated artifacts:

```text
data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.jsonl
data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5_mixed_only.jsonl
data/eval/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_eval50000_v6to7_len7_9.jsonl
data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5.summary.json
```

The eval still holds out surface variants 6 and 7.

## Experiment

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt
```

Training:

```text
init: core-role-value joint S240
steps: 240
train mixed lengths: 5, 7, 9
eval mixed lengths:  7, 9
loss:
  transition-state joint CE: 1.0
  core role-value CE:        1.0
```

## Result

Held-out core role-value:

```text
rows:                 32
trace exact:          0/32
step exact:           16/256 = 0.0625
value accuracy:       184/624 = 0.2949
```

Action-code preservation:

```text
trace exact:          32/32
halted exact:         32/32
step accuracy:        1.0000
finality accuracy:    1.0000
```

Progression:

```text
generic algorithmic slots:     0.0500
role readout-only:             0.0769
core role tokens:              0.1603
joint core role tokens:        0.2308
len579 joint core role tokens: 0.2949
```

## Decision

Accept as the current best value-state scaffold, reject final exact-state
claim.

The result proves that part of the previous failure was a gate/data confound:
when longer list lengths are included in value-state training, held-out value
accuracy improves and step-exact becomes non-zero for the first time.

However, full trace exact remains 0/32. The model still does not reliably bind
all prompt list elements and arithmetic residuals into role-value state.

## Next Bottleneck

Prompt-to-role binding is now the likely bottleneck.

The role tokens enter the recurrent core, but they are initialized as static
role embeddings. They must discover list element bindings through the general
core context path. That is probably too indirect for exact variable binding.

Next candidate:

```text
prompt token states
  -> role-conditioned extraction attention
  -> role-value tokens
  -> mandatory recurrent core
  -> role-value transition loss
```

Acceptance gate:

```text
value accuracy:  > 0.2949
step exact:      > 16/256
trace exact:     > 0/32
action-code:     remains near 32/32
ablation:        role extraction off/shuffle drops value metrics
```
