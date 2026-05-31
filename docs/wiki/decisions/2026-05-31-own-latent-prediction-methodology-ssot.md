# 2026-05-31 Own-Latent Prediction Methodology SSOT

## Decision

Adopt arXiv:2605.27734 as a methodology source for QTRM/BLT latent learning:

```text
Learn the hidden grammar by predicting same-model latent states.
Keep token CE as the speaking contract.
Keep IMTA/GRAM/PTRM as the internal breadth/search contract.
Do not let a world-model branch become a detached answer path.
```

Source note:

- [Own-Latent Prediction Sample Complexity](../sources/own-latent-prediction-sample-complexity.md)
- arXiv:2605.27734: <https://arxiv.org/abs/2605.27734>

## Why This Matters

The recent 82M BLT runs show the exact failure mode this paper warns about:

```text
byte/token loss improves
boundaries mature
but free generation still loops or answers weakly
```

The compatible diagnosis is:

```text
The model learned part of the byte distribution, but not enough of the latent
compositional structure that makes the next answer state stable.
```

Therefore, the next architecture direction should not be "only tune decoding"
or "only make the local speaker bigger." It should add an own-latent prediction
objective over the same internal states that later feed the answer speaker.

## Non-Conflict Matrix

| Mechanism | Role | Not allowed to claim |
|---|---|---|
| Token / byte CE | Teaches the model to speak through the LM head | Sufficient evidence of reasoning by itself |
| Own-latent prediction | Teaches sample-efficient latent grammar/state discovery | Replacement for token CE or answer evaluation |
| LeWM / JEPA-style world model | Probe or auxiliary latent predictor | Canonical answer path before semantic/answer-causal gate |
| Answer attractor | Pulls recurrent states toward answer-facing basins across depth | Proof of reasoning without free-generation/depth lift |
| GRAM/PTRM / IMTA | Creates and selects/aggregates internal trajectory breadth | External candidate reranker or oracle answer table |
| MSA / memory | Sparse long-memory route into the reader | Shortcut around one-body state or evidence gates |
| OPUS | Selects useful data/update windows | Architecture novelty or substitute for causal ablations |

## Canonical BLT Integration Target

For the active BLT PrefixLM line, own-latent prediction should be wired like
this:

```text
input bytes
-> learned BLT boundary states
-> NativeQTRMETDLM recurrent latent states
-> K IMTA/GRAM/PTRM internal trajectories
-> own-latent predictor over selected/dechunked/recurrent states
-> hnet one-body bridge
-> hnet_causal_speaker
-> same LM head answer
```

The predictor may supervise:

- future selected-boundary latent states,
- masked selected-boundary latent states,
- dechunked recurrent latent states,
- recurrent core states at depth `t+1` from depth `t`,
- trajectory-consensus latent states after IMTA selection.

The predictor must not supervise:

- a detached text answer table,
- an external verifier answer,
- a target whose loss can fall while the answer path ignores it,
- a separate speaker head that bypasses `hnet_causal_speaker` or the canonical
  LM head.

## H-JEPA / Stacking Rule

The paper argues that own-latent methods such as data2vec can implicitly learn
hierarchical latent prediction. Local rule:

```text
Do not add a separate stacked H-JEPA tower until one same-body own-latent
predictor has been ablated and found insufficient.
```

This prevents a conflict with the current one-body requirement. The first
implementation should be a same-body latent objective, not a second
architecture path.

## Relation To The Old LeWM Demotion

Old local result:

```text
LeWM learned the recursive latent transition, but symbolic/answer gates did
not improve.
```

Updated read after arXiv:2605.27734:

```text
That rejected a non-semantic or non-answer-causal implementation, not the
general principle of learning from own latents.
```

So LeWM remains off by default in canonical answer-path claims, but
own-latent prediction becomes a required methodology candidate for the next BLT
architecture revision.

## Promotion Gates

A run may claim this methodology helped only if it reports:

- token-CE-only vs own-latent-aux comparison under matched data and optimizer,
- latent-predictor-off ablation,
- same-mouth LM-head free-generation samples,
- EOS/special-token and repetition metrics from decoded generation,
- depth sweep showing `think_steps` can matter,
- K sweep if IMTA/GRAM/PTRM is enabled,
- latent MSE or cosine improvement plus answer-quality improvement,
- evidence that old anchors do not regress.

## Current Status

```text
2026-05-31 smoke/current run:
  same-body own-latent prediction is implemented in BLTDByteLatentPrefixLM
  as own_latent_predictor over recurrent BLT boundary states

active training switch:
  --own-latent-prediction-weight > 0 activates the auxiliary

current long run:
  /mnt/nvme0n1p2/tmp/20260531_82M_HNET_IMTA_K3_OWNLATENT_FREEGEN_RUN
  uses IMTA K=3 plus own_latent_prediction_weight=0.05

still required before promotion:
  token-CE-only/K=1/latent-off ablations and same-LM-head free-generation lift
```
