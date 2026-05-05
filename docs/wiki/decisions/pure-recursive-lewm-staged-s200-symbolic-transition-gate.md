# Pure Recursive LeWM Staged S200 Symbolic Transition Gate

Date: 2026-05-03

## Question

The LeWM core-world-model loss can learn recursive latent transitions. Does it
also make the recursive core form verifiable symbolic intermediate states?

## Setup

Baseline:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
checkpoint:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
symbolic transition eval:
  runs/eval/symbolic_transition_canonical_s160_max16.jsonl
  runs/eval/symbolic_transition_canonical_s160_max16_summary.json
```

Verified-reasoning LeWM comparison:

```text
config:
  configs/qwen35_2b_4090_verified_reasoning_lewm_core_s200.yaml
checkpoint:
  runs/qwen35_2b_4090_verified_reasoning_lewm_core_s200/last.pt
symbolic transition eval:
  runs/eval/symbolic_transition_lewm_core_s200_max16.jsonl
  runs/eval/symbolic_transition_lewm_core_s200_max16_summary.json
```

Pure-recursive staged LeWM run:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_lewm_staged_s200.yaml
init:
  runs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160/last.pt
checkpoint:
  runs/qwen35_2b_4090_pure_recursive_lewm_staged_s200/last.pt
raw depth gate:
  runs/eval/pure_recursive_lewm_staged_s200_depth_gate_16.jsonl
  docs/wiki/decisions/pure-recursive-lewm-staged-s200-depth-gate-16-summary.json
symbolic transition eval:
  runs/eval/symbolic_transition_pure_recursive_lewm_staged_s200_max16.jsonl
  runs/eval/symbolic_transition_pure_recursive_lewm_staged_s200_max16_summary.json
latent transition eval:
  runs/eval/pure_recursive_lewm_staged_s200_transition_max16.jsonl
  runs/eval/pure_recursive_lewm_staged_s200_transition_max16_summary.json
```

The symbolic transition gate scores each held-out row against
`depth_targets[core_steps]`, not only the final answer.

## Implementation

Added:

```text
scripts/201_eval_symbolic_transition_gate.py
tests/test_symbolic_transition_gate_script.py
configs/qwen35_2b_4090_pure_recursive_lewm_staged_s200.yaml
```

Runner update:

```text
scripts/197_run_pure_recursive_depth_supervised_train.sh
  CORE_WORLD_MODEL_WEIGHT
  --core-world-model-weight
```

## Results

Symbolic transition accuracy, `max_cases=16`, four core depths:

```text
canonical S160:
  18/64 = 0.28125
  core1: 6/16
  core2: 2/16
  core4: 5/16
  core8: 5/16

verified-reasoning LeWM S200:
  19/64 = 0.296875
  core1: 7/16
  core2: 2/16
  core4: 5/16
  core8: 5/16

pure-recursive staged LeWM S200:
  18/64 = 0.28125
  core1: 6/16
  core2: 2/16
  core4: 5/16
  core8: 5/16
```

The pure-recursive staged LeWM raw depth gate is still accepted:

```text
donor: 5/16
core_off: 0/16
core1: 4/16
core2: 5/16
core4: 6/16
core8: 6/16
status: accepted
```

Its latent transition MSE is low:

```text
core2 MSE: 0.00647
core4 MSE: 0.00740
core8 MSE: 0.00783
transition_mse_hit_pearson: 0.06636
```

## Decision

Do not promote LeWM as a semantic transition architecture yet.

Current evidence:

```text
LeWM latent transition prediction: learned
raw depth gate: accepted on 16 held-out cases
symbolic intermediate-state gate: unchanged from canonical
semantic promotion: rejected
```

## Root Diagnosis

The LeWM target is still self-latent:

```text
predict current core latent t+1 from current core latent t
```

This can be learned without forcing the latent to encode the intended symbolic
state update, such as arithmetic partial sums, binding hops, boolean subresults,
or list-transform intermediate results.

The causal path needed for the claim is:

```text
prompt tokens -> mandatory recursive state update -> symbolic state readout -> answer
```

The current path proves:

```text
prompt tokens -> mandatory recursive state update -> predictable next latent
```

Those are not equivalent.

## Next Architecture Candidates

1. Semantic transition head:
   add a small readout from each recurrent state to the staged symbolic target,
   train it with depth-specific CE, and keep it attached to the same state used
   by answer formation.

2. All-depth answer-state CE:
   set nonzero all-depth CE so every intermediate recurrent state is directly
   answer-readable, not only the final depth state for the scheduled run.

3. World-model predicts answer-progress:
   replace or augment next-latent MSE with prediction of future target logprob
   improvement or symbolic target identity.

Recommended next:
candidate 1 as a diagnostic, then candidate 2 if the diagnostic proves the
state contains the information but the main answer readout fails to use it.
