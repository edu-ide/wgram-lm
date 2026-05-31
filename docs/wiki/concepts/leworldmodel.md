# LeWorldModel

LeWorldModel is the preferred JEPA world-model reference for QTRM's
experimental world-model probes. It is not part of the current canonical
single-trace TRM answer path.

Core contract:

- encode observations into latent embeddings
- predict next embeddings autoregressively from context embeddings and actions
- train end-to-end with MSE on next embeddings
- prevent collapse with SIGReg
- no training-time stop-gradient target branch
- no EMA target encoder

QTRM implication:

- `src/wgram_lm/world_model.py` should keep LeWM-style next-latent prediction.
- TRM-style recursive `z_H` trajectories can also be trained with the same
  LeWM objective: predict the next core state from earlier core states plus an
  action trace.
- If we use older I-JEPA/JEPA-WM stop-gradient ideas, they must be explicitly
  marked as a deviation.
- Canonical QTRM runs keep `core_world_model_enabled=false` and
  `loss_core_world_model_weight=0.0` until a semantic transition or
  answer-causal gate passes.

## Current QTRM Mapping

Token-latent path:

```text
input tokens -> causal JEPA encoder -> latent[t]
latent[t] -> predictor -> latent[t+1]
```

Core-trajectory path:

```text
workspace evidence + prompt
-> TRM recursive core
-> z_H[0:T]

z_H[0:T-1] + action trace
-> LeWM-style predictor
-> predicted z_H[1:T]
```

The first implemented core action trace is simple and inspectable:

```text
OBSERVE or RETRIEVE -> VERIFY -> ANSWER
```

Files:

- `src/wgram_lm/wgram_model.py`
- `src/wgram_lm/world_model.py`
- `src/wgram_lm/losses.py`
- `src/wgram_lm/training/train.py`
- `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`
- `scripts/128_run_lewm_core_world_model_probe.sh`

## Boundary

LeWorldModel is not the whole answer to human-like intuition. It is useful for
next-state prediction, surprise, compact planning priors, and future multimodal
sequences, but the current QTRM result shows that self-latent prediction alone
can be non-semantic.

## 2026-05-31 Own-Latent Prediction Update

arXiv:2605.27734 strengthens the methodology case for learning from a model's
own latents rather than only from visible tokens. It does not reverse the local
LeWM demotion by itself.

Read the two facts together:

```text
2605.27734:
  own-latent prediction can be much more sample-efficient than token-level
  objectives for discovering hierarchical latent structure.

local LeWM probes:
  a next-latent objective can still learn non-semantic state motion unless its
  target is on the answer-causal path and passes semantic gates.
```

Updated rule:

```text
Keep LeWM/world-model modules probe-only for canonical answer-path claims.
Adopt same-body own-latent prediction as a required methodology candidate for
the next BLT/IMTA revision.
Promote it only if latent-predictor-off ablation hurts same-LM-head answers,
not merely because latent MSE improves.
```

See:

- [Own-Latent Prediction Sample Complexity](../sources/own-latent-prediction-sample-complexity.md)
- [2026-05-31 Own-Latent Prediction Methodology SSOT](../decisions/2026-05-31-own-latent-prediction-methodology-ssot.md)

Current decision:

```text
canonical single-trace TRM: LeWM off
experimental world-model probes: LeWM retained
promotion gate: semantic transition and/or answer-causal improvement required
```

It should not be used as the factual truth mechanism. For fake-info and
fact-seeking tasks, QTRM still needs retrieval, source metadata, atomic claim
verification, conflict arbitration, and temporal reasoning. The combined target
is:

```text
latent predictive priors
+ evidence retrieval
+ source/time-aware verification
+ conflict arbitration
+ trace-based self-improvement
```

See [Fact Verification Reasoning](fact-verification-reasoning.md).
