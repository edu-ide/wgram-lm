# Transition Value-State S480

Date: 2026-05-05

Status: rejected.

## Claim

After rejecting full-vocab state-sequence readouts, test whether a compact
digit/comma/minus value-state head can recover value-bearing internal state
without destroying the accepted mixed-composition action policy.

Accepted baseline:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt

action-code full exact:      32/32
transition-off exact:         0/32
code-shuffle exact:           0/32
code-dropout exact:           0/32
```

## Candidate

Added a compact transition value-state path:

```text
config fields:
  model.transition_value_state_enabled
  model.transition_value_state_max_tokens
  model.transition_value_state_vocab_size

head:
  transition_value_state_head

target alphabet:
  0..9, comma, minus

trainable policy:
  core_and_value_state
```

The goal was to avoid the 248K full-token projection used by the rejected
state-sequence probes and force the core to expose bounded numeric/list state.

## Training

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_value_state_s120.yaml

init checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt

output checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_s480_from_mixed_s720/last.pt

loss:
  transition_state_joint_ce_weight=2.0
  transition_value_state_ce_weight=1.0
```

## Held-Out Result

Value-state eval:

```text
json:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_s480_from_mixed_s720/eval_mixed_composition_value_state_full.json

rows:              32
trace exact:        0/32
step exact:         0.0000
token accuracy:     0.3713
```

Action-code preservation eval:

```text
json:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_s480_from_mixed_s720/eval_mixed_composition_action_code_full.json

rows:              32
trace exact:       10/32
halted exact:      10/32
step accuracy:      0.8477
finality accuracy:  0.8633
```

## Decision

Reject.

The compact head gives a non-trivial token signal, but it does not recover any
complete value trace. More importantly, joint value-state training regresses the
accepted action-code policy from `32/32` to `10/32`.

This confirms the current bottleneck:

```text
Adding a probe/readout for values is not enough.
Training value recovery through the same core can contaminate the already
accepted latent action controller.
```

## Research Update

The next direction is documented in:

```text
docs/wiki/decisions/state-factorized-qtrm-core-research-plan.md
```

Prior work motivating the redesign:

```text
Neural Algorithmic Reasoning:
  latent processor should execute state transitions.

Dreamer/RSSM:
  recurrent state, latent state, action, and value signals should be separable.

Factored Latent Action World Models:
  factorized state/action transition is a better fit than a monolithic latent.

TRM / latent reasoning:
  recursion is necessary but not sufficient; value preservation needs its own
  causal gate.
```

## Next Architecture Constraint

The next candidate must protect the accepted action policy while adding
value-bearing recurrent state.

Required gate:

```text
1. action-code full exact remains 32/32
2. value-state exact rises above 0/32
3. value-state path is causally used, not only decoded after the fact
4. disabling value-state reduces final mixed-composition success
```

Preferred direction:

```text
State-Factorized Core

action_state: frozen or KL/preserve-constrained latent action policy
value_state: compact differentiable slots updated by the recurrent core
bridge: value_state may condition the next recurrent update and answer path
gate: ablate action_state and value_state separately
```

Kill criterion:

```text
If value supervision improves only token probes while action exact regresses or
final answers do not depend on value_state, reject the candidate.
```

## Value-State-Only Control

Follow-up control from the state-factorized research plan:

```text
checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_only_s480_from_mixed_s720/last.pt

trainable:
  transition_value_state_* only
```

Result:

```text
value-state trace exact: 0/32
value-state token acc:   0.3794

action-code exact:       32/32
halted exact:            32/32
```

Interpretation:

```text
The action controller is not inherently fragile when frozen.
The missing piece is exact value content in the recurrent latent state, not
merely a better readout.
```
