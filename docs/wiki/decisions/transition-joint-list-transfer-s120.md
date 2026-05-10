# Transition Joint List Transfer S120

Date: 2026-05-05

## Claim

After rejecting full list-family zero-shot transfer, test a fairer Stage 1 gate:
list_transform is present in training, but selected list paraphrase clusters are
held out.

## Artifacts

```text
builder:
  scripts/234_build_list_transfer_gate.py

config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_s120.yaml

train:
  data/filtered/pure_recursive_transition_joint_dense_terminal_v2_list_transfer_train18000_v0to5.jsonl

eval:
  data/eval/pure_recursive_transition_joint_dense_terminal_v2_list_transfer_eval19000_v6to7.jsonl

summary:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/list_transfer_summary.json

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/last.pt

evals:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/eval_list_transfer_v6to7_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/eval_list_transfer_v6to7_transition_off.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/eval_list_transfer_v6to7_code_shuffle.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_list_transfer_s120_from_oodstress/eval_list_transfer_v6to7_code_dropout_to_hold.json
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

eval:
  rows: 16
  family:
    list_transform
  list variants:
    6, 7
```

This is intentionally not full family-zero-shot. It tests list paraphrase
cluster transfer while preserving the list operation family in train.

## Result

```text
full:
  step_acc:      1.0000
  exact:         16/16
  halted exact:  16/16

transition-state-off:
  step_acc:      0.1250
  exact:         0/16
  halted exact:  0/16

code shuffle, swap code 1 and 2:
  step_acc:      0.8750
  exact:         0/16
  halted exact:  0/16

code dropout to hold code 4:
  step_acc:      0.7500
  exact:         0/16
  halted exact:  0/16
```

## Decision

Accept as a Stage 1 list-surface transfer gate:

```text
The dense terminal_v2 joint-state path generalizes from seen list variants to
held-out list paraphrase variants, and the result depends causally on the
transition-state path and action-code semantics.
```

Do not promote to broad family-zero-shot reasoning:

```text
Full list-family holdout remains rejected. This result proves within-family
surface transfer, not invention of an unseen operation family.
```

## Next Gate

The next promotion should increase difficulty without jumping back to the
unfair full family-zero-shot gate:

```text
1. Increase eval rows and held-out variant clusters.
2. Add held-out list value ranges and longer list lengths.
3. Add mixed-family multi-step tasks where list state must compose with another
   primitive family.
4. Move from action-code classification to neural transition-state content
   prediction only after the above remains stable under ablations.
```
