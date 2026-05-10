# Core Role-Value Delta-Only S120

Date: 2026-05-05

## Question

Can a small recurrent value-delta adapter improve the core role-value state
probe while freezing the accepted action-code/core/head baseline?

## Root Architecture Claim

```text
canonical prompt/donor stream
-> mandatory recurrent core with role-value tokens
-> compact recurrent delta adapter over role-token trajectory
-> frozen role-value head
```

The claim is narrow: the adapter should improve value-state reasoning without
changing the action-code policy or relying on MemoryOS/retrieval.

## Baseline

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

held-out len7/9 mixed composition, 32 cases:
  value accuracy: 184/624 = 0.2949
  step exact:      16/256 = 0.0625
  trace exact:      0/32

action-code:
  exact:      32/32
  step acc:  256/256
```

## Implementation

Added disabled-by-default model fields:

```text
core_role_value_delta_enabled
core_role_value_delta_hidden_dim
core_role_value_delta_gate_init_bias
core_role_value_delta_gate_min
```

Added output telemetry:

```text
core_role_value_delta_gate_mean
```

Added eval ablation:

```text
scripts/238_eval_qtrm_algorithmic_value_state.py --disable-core-role-value-delta
```

Added trainable policy:

```text
trainable_param_policy: core_role_value_delta_only
```

The policy trains only:

```text
core_role_value_delta_*
```

It freezes:

```text
core.*
transition_state_joint_*
answer_state_loop_*
core_role_value_state_head
core_role_value_state_embed
text_embed
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_delta_only_s120.yaml
```

## Control: Untrained Delta Config

The untrained delta config preserves the baseline exactly with gate bias `-8.0`:

```text
value accuracy: 184/624 = 0.2949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Therefore any later regression is caused by delta training, not by module
presence.

## Experiment A: LR 1e-4

Training:

```text
steps: 120
lr: 1e-4
trainable params: 529,921
role-value CE: 1.0
transition-state joint CE: 1.0

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_delta_only_len579_s120_from_len579_s240/last.pt
```

Held-out value result:

```text
value accuracy: 112/624 = 0.1795
step exact:       0/256
trace exact:      0/32
```

Delta-off ablation on the same checkpoint:

```text
value accuracy: 184/624 = 0.2949
step exact:      16/256 = 0.0625
```

Action-code preservation:

```text
exact:         32/32
step accuracy: 256/256 = 1.0
finality acc:  256/256 = 1.0
halted exact:  32/32
```

Interpretation:

```text
The adapter is isolated from the accepted action-code path, but its active
value correction harms held-out value-state predictions.
```

## Experiment B: LR 1e-5

Training:

```text
steps: 120
lr: 1e-5

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_delta_only_len579_s120_lr1e5_from_len579_s240/last.pt
```

Held-out value result:

```text
value accuracy: 184/624 = 0.2949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Interpretation:

```text
The conservative run ties the baseline. It does not provide a falsifiable
raw-intelligence improvement.
```

## Decision

Reject as canonical.

The module is useful as a safe probe because:

```text
untrained delta preserves baseline;
delta-off ablation recovers baseline;
action-code remains 32/32 when only delta is trained.
```

But it fails promotion because:

```text
best held-out value result: 184/624 tie, not >184/624
best held-out step exact:  16/256 tie, not >16/256
trace exact remains:       0/32
```

## Root Cause

The failure is likely structural:

```text
A free MLP delta can move role-token hidden states, but it does not force the
model to represent exact arithmetic/list value transitions. It can learn
surface corrections on train slices while harming held-out role values.
```

Next candidate should not be another unconstrained hidden-state perturbation.
It should make the missing state variable explicit:

```text
action code -> typed value-delta code -> value-state update -> role-value logits
```

Promotion requires a state ablation where zeroing/shuffling the value-delta code
lowers held-out value accuracy while full beats the 184/624 baseline.

