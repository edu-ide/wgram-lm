# Transition Joint Dense S120

Date: 2026-05-05

## Claim

Dense 1..N recurrent transition targets should remove the sparse-depth halt
ambiguity seen in the previous joint-state head.

## Artifacts

```text
dense target builder:
  scripts/232_build_dense_transition_targets.py

role_v1 config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_s120.yaml

role_v1 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_s120_from_oodstress/last.pt

role_v1 eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_s120_from_oodstress/eval_transition_joint_dense_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_s120_from_oodstress/eval_transition_joint_dense_list_holdout_transition_off.json

terminal_v2 config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_s120.yaml

terminal_v2 checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_s120_from_oodstress/last.pt

terminal_v2 eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_s120_from_oodstress/eval_transition_joint_dense_terminal_v2_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_s120_from_oodstress/eval_transition_joint_dense_terminal_v2_list_holdout_transition_off.json
```

## Result

```text
role_v1 dense:
  full code step_acc:      0.8750
  full finality step_acc:  0.8750
  full trace exact:        0/64
  full halted exact:       0/64
  off code step_acc:       0.1250
  off finality step_acc:   0.1250

terminal_v2 dense:
  full code step_acc:      0.7500
  full finality step_acc:  0.8750
  full trace exact:        0/64
  full halted exact:       0/64
  off code step_acc:       0.1250
  off finality step_acc:   0.1250
```

## Diagnosis

Dense labels fixed one local problem:

```text
transition-state path is causal:
  role_v1 full 0.8750 vs off 0.1250
  terminal_v2 full 0.7500 vs off 0.1250
```

They did not fix the Stage 1 reasoning gate:

```text
strict trace exact:  0/64
halted trace exact: 0/64
```

The representative terminal_v2 held-out list pattern is:

```text
target codes:
  [0, 1, 4, 4, 4, 4, 4, 4]

predicted codes:
  [0, 2, 3, 4, 4, 4, 4, 4]

target finality:
  [0, 1, 1, 1, 1, 1, 1, 1]

predicted finality:
  [0, 0, 1, 1, 1, 1, 1, 1]
```

This means the recurrent path is learning a transition sequence, but it maps
the held-out list family to the arithmetic/nonterminal compose path at depth 2.
The current bottleneck is no longer only sparse supervision or code/finality
head disagreement. It is prompt-grounded terminal routing and semantic family
transfer.

## Decision

Reject as Stage 1 promotion.

Keep as a useful causal diagnostic because disabling the transition state
collapses the metric, but do not claim learned latent reasoning yet. The model
has a causal recurrent transition path, not a general reusable latent operation
model.

## Next Hypothesis

The next smallest experiment should test whether the failure is caused by
missing terminality diversity in training data or by an architecture-level
prompt-routing weakness.

Candidate gate:

```text
1. Build mixed terminality augmentation:
   - train examples where similar prompt surfaces require terminal compose;
   - train examples where similar prompt surfaces require nonterminal compose;
   - keep list_transform held out at the operation-family level.

2. Add counterfactual terminality pairs:
   - same surface family, changed required halt depth;
   - evaluate whether depth 2 switches between code 1 and code 2.

3. Promote only if:
   - halted exact > 0/64 on held-out list;
   - full still beats transition-state-off;
   - action-code shuffle/dropout breaks performance.
```

Kill criterion:

```text
If mixed terminality augmentation improves in-family training but held-out
list still maps to arithmetic terminality, stop tuning the codebook and replace
the Stage 1 target with a neural state-transition objective that predicts
intermediate state content directly.
```
