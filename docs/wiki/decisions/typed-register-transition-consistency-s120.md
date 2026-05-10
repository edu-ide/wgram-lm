# Typed Register Transition Consistency S120

Date: 2026-05-05

Status: rejected experiment.

## Question

Can typed registers improve exact value reasoning if the internal register state
is trained to predict the next depth's role-value state, not only the current
depth readout?

## Why This Follows The Failures

Rejected probes:

```text
typed-register v1 full:       106/624
typed-register v2 process CE: 102/624
prompt-binder full:           104/624
baseline:                     184/624
```

Process-code CE and prompt-token access both preserve action-code behavior, but
neither makes value state useful. The likely missing signal is recurrent value
transition credit:

```text
register_state[t] -> predicted_role_value_state[t+1]
```

## Architecture

Same universal LLM path:

```text
prompt tokens
-> donor hidden states
-> QTRM workspace + mandatory core
-> typed operation selector
-> persistent typed registers
-> current value logits + next-state transition logits
-> LM answer path / eval readout
```

No external solver is used at inference. Solver-derived role-value states are
training targets only.

## Implementation

Added:

```text
core_typed_register_transition_logits
--core-typed-register-transition-ce-weight
core_typed_register_transition_ce_loss
```

Config:

```text
configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_consistency_s120.yaml
```

## Command

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src .venv/bin/python scripts/196_train_pure_recursive_depth_supervised.py \
  --config configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_typed_register_transition_consistency_s120.yaml \
  --data-jsonl data/filtered/pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_len579_train40000_v0to5_mixed_only.jsonl \
  --init-checkpoint local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_core_role_value_state_joint_len579_s240_from_joint_s240/last.pt \
  --steps 120 --depth-steps 1,2,4,8 \
  --out-dir local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_transition_consistency_len579_s120_from_len579_s240 \
  --final-logit-ce-weight 0.0 --depth-final-ce-weight 0.0 --progress-margin-weight 0.0 \
  --core-typed-register-ce-weight 1.0 \
  --core-typed-register-operation-ce-weight 0.5 \
  --core-typed-register-transition-ce-weight 0.5
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

Reject if full remains below baseline or register-off restores performance.

## Result

Checkpoint:

```text
local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_typed_register_transition_consistency_len579_s120_from_len579_s240/last.pt
```

Held-out full:

```text
value accuracy: 104/624 = 0.1666666667
step exact:       0/256 = 0.0
trace exact:      0/32
```

Typed-register-off:

```text
value accuracy: 184/624 = 0.2948717949
step exact:      16/256 = 0.0625
trace exact:      0/32
```

Action-code preservation:

```text
exact:        32/32
step acc:      1.0
finality acc:  1.0
halted exact: 32/32
```

## Decision

Reject as canonical.

Transition CE did not improve the value path because the transition head was
trained as an auxiliary head while the evaluated value readout still came from
the independent current-state value head:

```text
full: 104/624
off:  184/624
```

## Next Candidate

Make the transition prediction causal for the value readout:

```text
depth 1 value logits = current register value head
depth >1 value logits = previous register transition head
```

This is stricter because the recurrent transition itself becomes the role-value
readout path instead of an auxiliary loss.
