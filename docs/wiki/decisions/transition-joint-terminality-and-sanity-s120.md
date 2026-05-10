# Transition Joint Terminality And Sanity S120

Date: 2026-05-05

## Claim

The dense joint-state failure may be caused by terminality target ambiguity or
missing terminal/nonterminal diversity, not by the recurrent transition path
itself.

## Artifacts

```text
terminality counterfactual builder:
  scripts/233_build_terminality_counterfactual_targets.py

action-terminal dense target builder:
  scripts/232_build_dense_transition_targets.py --finality-mode action_terminal

terminality-aug checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminality_aug_s120_from_oodstress/last.pt

terminality-aug eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminality_aug_s120_from_oodstress/eval_transition_joint_dense_terminality_aug_list_holdout_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminality_aug_s120_from_oodstress/eval_transition_joint_dense_terminality_aug_list_holdout_transition_off.json

all-families sanity checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_allfamilies_s120_from_oodstress/last.pt

all-families sanity eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_allfamilies_s120_from_oodstress/eval_transition_joint_dense_terminal_v2_allfamilies_list17000_full.json
  local_eval/qwen35_2b_pure_recursive_transition_joint_dense_terminal_v2_allfamilies_s120_from_oodstress/eval_transition_joint_dense_terminal_v2_allfamilies_list17000_transition_off.json
```

## Data Target Fix

The previous answer-match finality target produced process ambiguity:

```text
nonterminal action with finality=1 appeared in training rows when an
intermediate boolean state happened to equal the final answer.
```

For terminal_v2, the corrected target is:

```text
nonterminal codes:
  0, 2 -> finality 0

terminal/hold codes:
  1, 3, 4 -> finality 1
```

This is now available through:

```text
scripts/232_build_dense_transition_targets.py --finality-mode action_terminal
```

## Terminality Aug Result

Terminality augmentation added 128 rows while keeping list_transform fully held
out:

```text
base rows:  192
added rows: 128
total rows: 320

added:
  arithmetic_terminal_depth2: 64
  symbolic_nonterminal_depth3: 64
```

Result on held-out list_transform:

```text
full:
  step_acc:      0.7500
  finality_acc:  0.8750
  exact:         0/64
  halted_exact:  0/64

transition-state-off:
  step_acc:      0.1250
  finality_acc:  0.1250
  exact:         0/64
  halted_exact:  0/64
```

Decision:

```text
Reject terminality augmentation as a family-zero-shot fix.
```

The failure stayed identical:

```text
target list codes:
  [0, 1, 4, 4, 4, 4, 4, 4]

predicted codes:
  [0, 2, 3, 4, 4, 4, 4, 4]
```

## All-Families Sanity Result

The sanity control trained on all primitive families, including list_transform,
with a held-out list index range:

```text
train:
  data/filtered/pure_recursive_transition_joint_dense_terminal_v2_allfamilies_train16000.jsonl
  rows: 256
  families: arithmetic_chain, boolean_logic, list_transform, symbolic_binding

eval:
  data/eval/pure_recursive_transition_joint_dense_terminal_v2_list_eval17000.jsonl
  rows: 64
  family: list_transform
```

Result:

```text
full:
  step_acc:      1.0000
  finality_acc:  1.0000
  exact:         64/64
  halted_exact:  64/64

transition-state-off:
  step_acc:      0.1250
  finality_acc:  0.1250
  exact:         0/64
  halted_exact:  0/64
```

Representative full prediction:

```text
predicted codes:
  [0, 1, 4, 4, 4, 4, 4, 4]

target codes:
  [0, 1, 4, 4, 4, 4, 4, 4]

halted_depth:
  2
```

## Decision

Accept the all-families result as a Stage 1 mechanism sanity control:

```text
The recurrent transition-state path can learn the terminal_v2 dense joint trace
for list_transform and is causally necessary, because transition-state-off
drops from 64/64 to 0/64.
```

Reject broad Stage 1 family-transfer promotion:

```text
The same architecture still fails when list_transform is completely absent
from train. The bottleneck is zero-shot semantic family transfer, not the
ability to learn list transition traces once the surface family is present.
```

## Next Gate

The next fair promotion gate should not be full list-family zero-shot. It should
use a curriculum between in-distribution list and impossible family-zero-shot:

```text
1. Train on list_transform surfaces with held-out value ranges.
2. Hold out one list operation surface or paraphrase cluster.
3. Add action-code shuffle/dropout ablations.
4. Promote only if full beats transition-state-off and shuffle/dropout while
   maintaining halted exact on the held-out list slice.
```

If this passes, move from action-code classification to a neural state-content
transition objective. If it fails, the compact joint-state path is too
template-bound and should not be scaled.
