# State-Factorized QTRM Core Research Plan

Date: 2026-05-05

Status: active plan.

## Problem

The accepted mixed-composition checkpoint proves a causal latent action
controller:

```text
action-code full exact: 32/32
transition-off:          0/32
code-shuffle:            0/32
code-dropout:            0/32
```

But value-bearing internal state is not yet proven:

```text
full state sequence exact: 0/32
compact value-state exact: 0/32
```

The compact value-state joint run also regressed the accepted action policy:

```text
action-code exact after joint value training: 10/32
```

Therefore the current bottleneck is not a decoder/head problem. It is a state
factorization problem.

## Prior Work To Use

Neural Algorithmic Reasoning:

```text
Use a latent processor that executes state transitions, not only answer
classification. This supports explicit supervision/ablation of intermediate
algorithmic states.
```

Dreamer/RSSM world-model family:

```text
Separate recurrent deterministic state, latent state, action/policy, and value
signals. This supports action-conditioned latent rollout without forcing every
signal into one undifferentiated vector.
```

Factored Latent Action World Models:

```text
Factor state and latent action so each factor can predict its next value. This
is the closest prior for splitting QTRM action_state from value_state.
```

Latent reasoning / looped-LM / TRM family:

```text
Keep recurrent latent computation as the raw-intelligence target, but do not
assume recursion alone preserves variable values. Value transition must be
measured separately.
```

Recurrent Memory Transformer / ARMT:

```text
Use recurrent memory/state tokens as a reference for future MSA/LM2 memory
integration, but keep MemoryOS/RAG outside this raw recursive reasoning gate.
```

## Root Architecture Claim

```text
QTRM should become a recurrent latent reasoning model whose recursive core
causally updates action_state and value_state, then renders an answer from the
resulting state.
```

Falsification:

```text
If value_state can be decoded only by a probe but disabling it does not hurt
final answers, it is not real reasoning state.

If value_state training damages action_state, the state representation is not
properly factorized.
```

## Ranked Architecture Candidates

### A. Value-State-Only Readout Control

Freeze the accepted core/action path and train only the compact value-state
head.

Purpose:

```text
Distinguish "value information already exists but is unreadable" from
"value information is absent from the current latent trajectory".
```

Accept only as a diagnostic if:

```text
action-code exact remains 32/32
value-state exact rises above 0/32
```

Reject if:

```text
value-state exact remains 0/32
```

### B. State-Factorized Core

Add separate recurrent state factors:

```text
action_state: preserve accepted action/finality policy
value_state: compact slots for intermediate numeric/list values
confidence_state: halt/unknown/metacognition later
```

Transition:

```text
action_t = policy(action_state_t, prompt_context)
value_state_{t+1} = F(value_state_t, action_t, prompt_context)
answer = read(value_state_final, action_state_final)
```

Required gate:

```text
action-code exact: 32/32
value-state exact: >0/32, then scaled upward
value_state_off: final answer drops
action_state_off: action trace drops
```

### C. Neural Algorithmic Processor Core

Replace the current monolithic core trajectory with a processor-style recurrent
cell trained directly on state transitions.

Use only if B fails, because it is a larger architecture change.

## Immediate Experiment

Run A before implementing B:

```text
trainable_param_policy: transition_value_state_only
init checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_composition_balanced_interleave_s720_from_s240/last.pt
```

Decision:

```text
If A fails, current latent trajectory lacks stable value content. Proceed to B.
If A succeeds, keep the accepted core frozen and design a causal bridge from
value_state into final answers without modifying action_state.
```

## Experiment A Result

Value-state-only control:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_transition_joint_dense_terminal_v2_value_state_only_s120.yaml

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_value_state_only_s480_from_mixed_s720/last.pt

trainable:
  transition_value_state_* only
```

Held-out value-state:

```text
rows:          32
trace exact:    0/32
step exact:     0.0000
token acc:      0.3794
```

Action-code preservation:

```text
rows:          32
trace exact:   32/32
halted exact:  32/32
step acc:       1.0000
finality acc:   1.0000
```

Decision:

```text
Action policy can be preserved if the core/action path is frozen.
But the accepted latent trajectory still does not contain enough exact value
content for complete state recovery.
```

Next:

```text
Proceed to B: State-Factorized Core.
Do not spend more local effort on probe/readout heads.
```

## Experiment B Minimal Result

Minimal factorized value slots were implemented and evaluated:

```text
decision:
  docs/wiki/decisions/factorized-value-state-s480.md

checkpoint:
  local_eval/qwen35_2b_pure_recursive_transition_joint_dynamic_halt_v3_mixed_factorized_value_state_s480_from_mixed_s720/last.pt
```

Result:

```text
value-state trace exact: 0/32
value-state token acc:   0.3713
action-code exact:       32/32
halted exact:            32/32
```

Decision:

```text
Reject minimal factorized digit-sequence path.
Proceed to structured neural-algorithmic state targets.
```
