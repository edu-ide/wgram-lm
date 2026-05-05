# LeWM Demoted From Canonical Single-Trace TRM

Date: 2026-05-03

## Decision

LeWorldModel/core-world-model is no longer part of the canonical QTRM answer
architecture.

```text
Canonical:
  prompt tokens
  -> frozen donor hidden-state context
  -> latent workspace
  -> mandatory single-trace recursive core
  -> answer-state loop
  -> LM logits

Canonical settings:
  core_world_model_enabled = false
  loss_core_world_model_weight = 0.0
```

LeWM remains implemented and documented as an experimental probe.

## Why

The LeWM head learned to predict recursive-core latent transitions, but this did
not prove semantic reasoning:

```text
latent transition MSE: improved strongly
symbolic intermediate-state accuracy: unchanged
answer accuracy: not improved by the LeWM objective
```

The root issue is that the current target is self-latent:

```text
z_H[t] -> predict z_H[t+1]
```

That can teach the world-model head to follow the core's own latent motion
without forcing the core to represent arithmetic partial states, binding hops,
boolean subresults, contradiction checks, or answer-progress states.

## Canonical Claim

The near-term raw-intelligence claim is now narrower and easier to falsify:

```text
Can one mandatory TRM trace become better as recursive depth increases?
```

The accepted canonical baseline is:

```text
config:
  configs/qwen35_2b_4090_pure_recursive_answer_state_loop_causal_prefix_s160.yaml
contract:
  no retrieval
  no hidden evidence
  no MemoryOS shortcut
  no LeWM loss
  donor logits disabled
  causal forced-choice depth gate
```

## Promotion Gate For LeWM

LeWM can return to the canonical architecture only if at least one of these
passes on held-out cases:

```text
Semantic transition gate:
  internal depth states predict labelled intermediate symbolic targets
  better than the non-LeWM single-trace TRM baseline.

Answer-causal gate:
  LeWM-on improves final reasoning accuracy,
  and LeWM-off drops while donor/core baselines are controlled.

Planner gate:
  action-conditioned predictions improve action selection or verification,
  and wrong-future predictions are penalized or abstained from.
```

Until then, LeWM is instrumentation, not canonical architecture.
