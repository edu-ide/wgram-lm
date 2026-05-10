# Transition Joint S120

Date: 2026-05-05

## Claim

The recurrent core can avoid code/finality disagreement if latent action code
and finality are predicted through one compact joint transition-state head.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_s120.yaml

train:
  data/filtered/pure_recursive_transition_finality_family_holdout_list_train.jsonl

eval:
  data/eval/pure_recursive_transition_finality_family_holdout_list_eval.jsonl

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_s120_from_oodstress/last.pt

full eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_s120_from_oodstress/eval_transition_joint_list_holdout_full.json

transition-state-off eval:
  local_eval/qwen35_2b_pure_recursive_transition_joint_s120_from_oodstress/eval_transition_joint_list_holdout_transition_off.json
```

## Result

```text
full:
  code step_acc:      0.7500
  finality step_acc:  0.7500
  trace exact:        0/64
  halted exact:       0/64

transition-state-off:
  code step_acc:      0.2500
  finality step_acc:  0.2500
  trace exact:        0/64
  halted exact:       0/64
```

## Failure

The joint path is causal because full beats transition-state-off, but it does
not pass the raw recursive reasoning gate.

Representative held-out list-transform pattern:

```text
target joint states by labelled depth:
  depth 1: code 0, nonfinal
  depth 2: code 1, final
  depth 4: code 3, final
  depth 8: code 3, final

predicted joint states:
  [0, 2, 7, 5, 7, 2, 5, 7]
```

Decoded:

```text
depth 1: code 0, nonfinal  correct
depth 2: code 1, nonfinal  code correct, finality late
depth 3: code 3, final     unlabelled near-miss halt
depth 4: code 2, final     code wrong
depth 8: code 3, final     correct
```

## Decision

Reject as Stage 1 promotion:

- strict trace exact remains 0/64;
- halted exact remains 0/64;
- the model still does not align the final/hold transition to the labelled
  recurrent depth.

Keep as useful diagnosis:

- the path is causal;
- the model learns much of the prefix state;
- the new bottleneck is not only head disagreement. It is sparse/depth-skipped
  transition supervision and halt latency.

## Next Hypothesis

Do not add another independent head. The next falsifiable experiment should
make the transition target dense across recurrent steps:

```text
label depths 1..N directly
train joint state for every recurrent step
accept delayed halt only as a diagnostic metric
promote only if strict held-out halted exact rises above 0/64
```

For list-transform cases this means depth 3 should no longer be unlabelled; it
should be a supervised hold/final state. For arithmetic, the dense transition
schedule must be derived from the solver trace rather than guessed from powers
of two.
