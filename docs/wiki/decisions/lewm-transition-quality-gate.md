# LeWM Transition Quality Gate

Date: 2026-05-03

## Question

The LeWM core run did not improve raw answer accuracy. Did it at least improve
recursive-core transition prediction?

## Gate

Measure LeWorldModel-style next-latent prediction quality over recursive core
states:

```text
z_H[t] -> core_world_model -> predicted z_H[t+1]
target: actual z_H[t+1]
metric: masked MSE over valid transitions
```

This gate intentionally does not claim answer correctness. It checks whether
the world-model head learned the transition target at all, then compares that
with existing raw-answer hits.

## Implementation

```text
scripts/200_eval_core_world_model_transition.py
tests/test_core_world_model_transition_eval.py
```

The script:

- loads no-retrieval raw reasoning cases;
- runs QTRM core steps 1/2/4/8;
- computes `core_world_model_pred` vs `core_world_model_target` MSE;
- merges answer hits from existing raw eval JSONL by `(case id, mode)`;
- writes JSONL records and a summary JSON.

## Results

CE-only S200 checkpoint loaded under the LeWM config, leaving the world-model
head random:

```text
checkpoint:
  runs/qwen35_2b_4090_verified_reasoning_s200/last.pt
records:
  runs/eval/verified_reasoning_ce_only_s200_transition_max5.jsonl
summary:
  runs/eval/verified_reasoning_ce_only_s200_transition_max5_summary.json

mean transition MSE:
  core_steps_2: 1.3253
  core_steps_4: 1.3295
  core_steps_8: 1.3322
answer hit rate for QTRM modes:
  0.0
```

CE + LeWM core S200 checkpoint:

```text
checkpoint:
  runs/qwen35_2b_4090_verified_reasoning_lewm_core_s200/last.pt
records:
  runs/eval/verified_reasoning_lewm_core_s200_transition_max5.jsonl
summary:
  runs/eval/verified_reasoning_lewm_core_s200_transition_max5_summary.json

mean transition MSE:
  core_steps_2: 0.00816
  core_steps_4: 0.00786
  core_steps_8: 0.00778
answer hit rate for QTRM modes:
  0.0
```

## Decision

LeWM is doing something real: it dramatically improves prediction of the
current recursive-core latent trajectory.

But this is not raw-intelligence progress yet:

```text
transition prediction improved: yes
answer accuracy improved: no
transition MSE correlated with answer hits: not shown
canonical promotion: rejected
```

## Root Diagnosis

The current LeWM target is self-referential:

```text
predict the next latent state produced by the same current core
```

If the core trajectory is not already answer-causal, predicting it better can
just learn the model's own latent motion without improving reasoning.

The architecture needs a semantically anchored transition target.

## Next Architecture Candidates

1. Verifiable symbolic transition target:
   train the recurrent core on tasks where each latent transition corresponds
   to a known state update, then require transition error to predict final
   answer correctness.

2. Answer-progress transition target:
   make the world model predict future target-logprob improvement or answer
   state, not only next latent MSE.

3. Planner-style action outcome target:
   attach action-conditioned predictions to explicit `OBSERVE -> COMPUTE ->
   VERIFY -> ANSWER` state changes and reject if action/world-model-off
   ablations do not drop.

Recommended next:
start with candidate 1 because it is the cleanest falsification gate. It tests
whether the recursive core can learn real state transitions before we ask it to
improve open-ended language reasoning.
