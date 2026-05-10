# Transition-State Text S120

Date: 2026-05-05

## Claim

The recursive core can learn semantic intermediate-state tokens from latent
depth states on held-out list-transform tasks, without fixed operation ids.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_text_s120.yaml

train:
  data/filtered/pure_recursive_transition_finality_family_holdout_list_train.jsonl

eval:
  data/eval/pure_recursive_transition_finality_family_holdout_list_eval.jsonl

eval script:
  scripts/231_eval_qtrm_transition_state_text.py

low-lr checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_s120_from_oodstress/last.pt

high-lr checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_lr1e3_s120_from_oodstress/last.pt

depth-contrast checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_text_depthcontrast_lr1e3_s120_from_oodstress/last.pt
```

## Target Fix

The first version accidentally scored the tokenizer's leading space token for
many numeric/list targets. The train and eval paths were corrected to use the
first content token for transition-state text supervision.

## Results

```text
lr=5e-5:
  full step_acc:                 0.0000
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64

lr=1e-3:
  full step_acc:                 0.2500
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64

lr=1e-3 + depth contrast:
  full step_acc:                 0.2500
  transition-state-off step_acc: 0.0000
  trace exact:                   0/64
```

The high-lr run produced a causal signal, but it collapsed to the first
list-transform state token at every depth. It therefore matched depth 1 and
missed depths 2/4/8.

The depth-contrast run did not break the collapse. Held-out examples still
predicted the same content token at every recurrent depth:

```text
predicted: [16, 16, 16, 16, 16, 16, 16, 16]
targets:   [16, 18, --, 18, --, --, --, 18]
```

## Decision

Accepted only as a narrow causal semantic-token probe:

- the transition-state path can affect held-out semantic token prediction;
- the disabled transition-state path scores zero.

Rejected as recursive semantic transition:

- trace exact remains 0/64;
- the model does not learn depth-varying state updates;
- the current text path through a frozen LM head is optimization-sensitive and
  collapses to a shallow state token.
- adding a local anti-collapse depth-contrast loss did not change the held-out
  failure pattern.

## Next Gate

The next experiment should not add another scalar loss to the same frozen
full-vocabulary LM projection. That path has now failed under low LR, high LR,
and depth contrast.

Replacement candidate:

```text
prompt -> recurrent core -> compact semantic state head -> transition decoder
```

The compact head should predict a small task-local semantic state vocabulary or
state embedding before any frozen LM vocabulary projection. Acceptance still
requires held-out depth 2/4/8 improvement, trace exact above zero, and
transition-state-off failure.
