# Transition Joint List Transfer Long S120

Date: 2026-05-05

## Claim

Scale the accepted within-family list paraphrase gate to a harder split:
list_transform remains present in training, but eval uses held-out surface
variants, longer list lengths, and a disjoint value range.

This is still not full operation-family zero-shot. It tests whether the dense
terminal_v2 joint-state path survives length/value shift inside a known
primitive family.

## Artifacts

```text
builder:
  scripts/234_build_list_transfer_gate.py

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_s120.yaml

train:
  data/filtered/pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_train20000_v0to5.jsonl

eval:
  data/eval/pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_eval30000_v6to7_len7_9.jsonl

summary:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/list_transfer_long_summary.json

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/last.pt

evals:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/eval_list_transfer_long_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/eval_list_transfer_long_transition_off.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/eval_list_transfer_long_code_shuffle.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_long_s120_from_oodstress/eval_list_transfer_long_code_dropout_to_hold.json
```

## Split

```text
train:
  rows: 240
  families:
    arithmetic_chain: 64
    symbolic_binding: 64
    boolean_logic: 64
    list_transform: 48
  list variants:
    0, 1, 2, 3, 4, 5
  list lengths:
    5

eval:
  rows: 32
  family:
    list_transform
  list variants:
    6, 7
  list lengths:
    7, 9
  value range:
    starts at 30001+
```

## Result

```text
full:
  step_acc:      1.0000
  exact:         32/32
  halted exact:  32/32

transition-state-off:
  step_acc:      0.1250
  exact:         0/32
  halted exact:  0/32

code shuffle, swap code 1 and 2:
  step_acc:      0.8750
  exact:         0/32
  halted exact:  0/32

code dropout to hold code 4:
  step_acc:      0.7500
  exact:         0/32
  halted exact:  0/32
```

## Decision

Accept as a stronger Stage 1 within-family list transfer gate:

```text
The dense terminal_v2 joint-state path generalizes from train length 5 to
held-out lengths 7 and 9 under held-out list paraphrase variants and a disjoint
value range. The accepted behavior depends causally on both the transition-state
path and action-code semantics.
```

Boundary:

```text
This does not prove unseen operation invention, open-ended reasoning, or ASI.
It proves a narrow but useful scaling step: the recurrent transition-state core
is no longer merely memorizing exact list length, exact values, or seen list
surface variants.
```

## Next Gate

The next promotion should test composition rather than more same-family
surface scaling:

```text
1. Add mixed-family multi-step tasks where list state feeds arithmetic,
   symbolic, or boolean state.
2. Add held-out chain/list lengths with donor-only, core-off, state-off, and
   code-shuffle baselines.
3. Move toward neural transition-state content prediction after mixed-family
   composition remains stable.
```
