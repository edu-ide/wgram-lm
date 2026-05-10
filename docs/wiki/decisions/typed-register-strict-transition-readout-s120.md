# Typed Register Strict Transition Readout S120

Date: 2026-05-05

Status: rejected.

## Question

Can value reasoning improve if depth > 1 role-value logits are read from the
previous register state's transition prediction instead of an independent
current-state value head?

## Why

`typed-register-transition-consistency-s120` failed because transition CE was
only auxiliary:

```text
full: 104/624
off:  184/624
```

This experiment makes transition prediction the causal value readout path:

```text
depth 1 value logits  = value_head(register_state[1])
depth >1 value logits = transition_head(register_state[depth-1])
```

## Implementation

Added:

```text
core_typed_register_transition_readout_enabled
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_strict_transition_readout_s120.yaml
```

## Acceptance

Promote only if:

```text
full value accuracy > 184/624
step exact          > 16/256
trace exact         > 0/32
action-code exact   = 32/32
typed-register-off  < full
```

## Result

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_strict_transition_readout_len579_s120_from_len579_s240/last.pt
```

Held-out value gate:

```text
full:
  correct_values: 64/624 = 0.1025641026
  step exact:      0/256
  trace exact:     0/32

typed-register-off:
  correct_values: 184/624 = 0.2948717949
  step exact:      16/256
  trace exact:     0/32
```

Action-code preservation:

```text
exact:       32/32
step_acc:     1.0000
finality_acc: 1.0000
halted_exact: 32/32
```

## Decision

Reject.

The strict readout made the recurrent transition causal for depth > 1 value
logits, but it damaged value accuracy more than the auxiliary transition
variant. The ablation restores the baseline exactly, so the typed-register
transition path is the harmful component, not the base checkpoint.

The useful signal remains narrow: action selection is solved, but the role
value update is not. Further local variants of typed-register heads should be
paused unless they change the root information path or the training target.

## Next

Escalate from typed-register readout tuning to a root candidate:

```text
prompt-token stream
-> mandatory recurrent core
-> learned latent operation/state update
-> canonical LM answer path
```

The next probe should avoid a parallel role-vocab answer channel. Exact
intermediate values may still be supervised for diagnostics, but promotion
requires the recurrent latent state to improve the universal LLM causal path
under depth and core-off ablations.
