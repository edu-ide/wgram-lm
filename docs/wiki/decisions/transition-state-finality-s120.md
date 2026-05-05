# Transition-State Finality S120

Date: 2026-05-05

## Claim

The recursive core can learn a prompt-grounded terminal/non-terminal signal
from latent depth states on held-out list-transform tasks.

This is a narrower claim than reusable action-code reasoning. It only tests
whether the core state carries causal information about when a recursive trace
has reached a final answer.

## Artifacts

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_finality_s120.yaml

train:
  data/filtered/pure_recursive_transition_finality_family_holdout_list_train.jsonl

eval:
  data/eval/pure_recursive_transition_finality_family_holdout_list_eval.jsonl

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/last.pt

full eval:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/eval_transition_finality_list_holdout_full.json

transition-state-off eval:
  local_eval/qwen35_2b_pure_recursive_transition_finality_s120_from_oodstress/eval_transition_finality_list_holdout_transition_off.json
```

## Result

```text
full finality step accuracy:                 0.7500
transition-state-off finality step accuracy: 0.2500
full finality trace exact:                   0/64
transition-state-off finality trace exact:   0/64
```

The evaluation was corrected so a zero finality logit is treated as a
non-final tie. Without that correction, the disabled transition-state path was
artificially inflated because all-zero logits were counted as final.

## Decision

Accepted as a narrow causal finality signal:

- the full transition-state finality head beats the transition-state-off
  ablation on held-out list-transform depth labels;
- the signal is produced by the core-depth path, not by code labels or answer
  text loss.

Rejected as a Stage 1 reasoning promotion:

- trace exact remains 0/64;
- this does not solve action semantics, answer generation, or open-ended
  recursive reasoning;
- the next bottleneck is semantic latent-state transition, not just a separate
  halt/finality bit.

## Next Gate

Train a neural transition-state semantics probe that predicts the next latent
state or answer-state delta without operation ids. Require both:

- held-out family improvement over transition-state-off;
- non-zero trace exact on list-transform tasks.
