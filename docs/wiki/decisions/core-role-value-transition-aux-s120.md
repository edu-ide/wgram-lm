# Core Role-Value Transition Auxiliary S120

Date: 2026-05-05

## Question

Can an explicit previous-role-state transition auxiliary loss make the
mandatory recurrent core behave more like a real latent reasoning loop, instead
of only matching role-value slots independently at each depth?

## Failure Ledger

Failure:

```text
Core role-value state reaches non-zero held-out value accuracy, but full exact
state traces remain 0/32 and deeper core steps do not show a clean reasoning
improvement.
```

Evidence:

```text
canonical baseline:
  checkpoint:
    local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

  held-out len7/9 mixed composition:
    value accuracy: 184/624 = 0.2949
    step exact:      16/256 = 0.0625
    trace exact:     0/32
```

Known limitation class:

```text
Side-head state supervision can create a diagnostic scaffold without forcing
the latent state transition itself to become answer-causal or depth-improving.
```

Root architecture hypothesis:

```text
Role-value tokens are inside the recurrent core, but the training signal still
allows the model to learn per-depth slot readouts rather than a robust
next-state transition.
```

Could the big structure be wrong?

```text
Yes. If deeper recursion only adds more supervised slots without improving
later-step correctness, the current role-value head is a probe, not a
general-purpose reasoning mechanism.
```

Information path needed:

```text
previous latent state -> next latent state -> later decision/answer
```

Current information path:

```text
prompt/donor states -> workspace -> recurrent core with role tokens
  -> role-value side logits
```

Local fix attempted:

```text
Add core_role_value_transition_logits:
  previous role tokens + previous workspace summary
  -> transition head
  -> next-step role-value targets
```

Acceptance gate:

```text
value accuracy > 0.2949
step exact     > 16/256
trace exact    > 0/32
no action-code regression
```

Kill criterion:

```text
If held-out value/step exact regress under two weight settings, reject the
transition auxiliary as canonical and treat it as a probe-only component.
```

## Implementation

Added disabled-by-default transition auxiliary path:

```text
model config:
  core_role_value_transition_enabled
  core_role_value_transition_hidden_dim

model output:
  core_role_value_transition_logits

train loss:
  --algorithmic-role-value-transition-ce-weight

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_core_role_value_transition_joint_s120.yaml
```

The transition head is causal with respect to the role trajectory: it uses
previous role states and previous workspace summary to predict the next
role-value targets. It does not read the future role state.

## Verification

Targeted tests:

```text
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_core_halting \
  tests.test_training_checkpoint_init \
  tests.test_pure_recursive_depth_supervised_train_script

result: 141 tests OK
```

Smoke:

```text
steps: 2
transition samples at depth 2: 2
saved:
  local_eval/smoke_core_role_value_transition_joint_s2/last.pt
```

## Experiment A: Full Transition Weight

Training:

```text
init:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt

out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_transition_len579_s120_from_len579_s240/last.pt

steps: 120
lr: 2e-5
role-value CE: 1.0
transition auxiliary CE: 1.0
```

Held-out result:

```text
value accuracy: 96/624 = 0.1538
step exact:      0/256
trace exact:     0/32
```

Decision: reject.

## Experiment B: Lower Transition Weight

Training:

```text
out:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_transition_len579_s120_w01_from_len579_s240/last.pt

steps: 120
lr: 1e-5
role-value CE: 1.0
transition auxiliary CE: 0.1
```

Held-out result:

```text
value accuracy: 96/624 = 0.1538
step exact:      0/256
trace exact:     0/32
```

Decision: reject.

## Control: New Config Without Training

The canonical baseline checkpoint loaded into the transition-enabled config
without transition training preserves the baseline:

```text
value accuracy: 184/624 = 0.2949
step exact:      16/256 = 0.0625
trace exact:     0/32
```

Therefore the regression is caused by transition auxiliary training, not by the
new module being present at inference.

## Core-Step Sweep

Canonical baseline, same held-out set:

```text
core steps 1:
  value accuracy: 64/120 = 0.5333
  step exact:     16/32  = 0.5000

core steps 2:
  value accuracy: 96/240 = 0.4000
  step exact:     16/64  = 0.2500

core steps 4:
  value accuracy: 120/368 = 0.3261
  step exact:      16/128 = 0.1250

core steps 8:
  value accuracy: 184/624 = 0.2949
  step exact:      16/256 = 0.0625
```

Per-depth value accuracy at core steps 8:

```text
depth 1: 64/120 = 0.5333
depth 2: 32/120 = 0.2667
depth 3: 16/64  = 0.2500
depth 4: 8/64   = 0.1250
depth 5: 16/64  = 0.2500
depth 6: 16/64  = 0.2500
depth 7: 16/64  = 0.2500
depth 8: 16/64  = 0.2500
```

Interpretation:

```text
The current role-value path does not yet show monotonic raw recursive
reasoning improvement with more latent steps. It is strongest on the earliest
visible binding-like stage and then weakly repeats partial later-stage
patterns.
```

## Decision

Reject as canonical architecture improvement.

Keep the code path disabled by default because it is useful as a diagnostic
probe, but do not use transition auxiliary training for promotion unless a
future schedule proves no regression and a held-out improvement.

## Next Architecture Direction

Do not add more side heads to hide the failure. The next candidate must make
the transition state itself part of the causal recurrent loop:

```text
prompt tokens
  -> initial latent state
  -> recurrent core step t
  -> explicit latent state z_t
  -> learned transition update z_{t+1}
  -> answer/state readout
```

Acceptance gate:

```text
core_steps 1 < core_steps 2 < core_steps 4 or core_steps 8
on final-answer or exact-state held-out metrics,
with core-off and transition-off both dropping the advantage.
```

Until that gate passes, role-value slots should be described as a structured
probe/scaffold, not as a final general LLM reasoning mechanism.
