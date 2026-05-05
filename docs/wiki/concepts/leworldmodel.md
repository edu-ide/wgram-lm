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

- `src/qtrm_mm/world_model.py` should keep LeWM-style next-latent prediction.
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

- `src/qtrm_mm/qtrm_model.py`
- `src/qtrm_mm/world_model.py`
- `src/qtrm_mm/losses.py`
- `src/qtrm_mm/training/train.py`
- `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`
- `scripts/128_run_lewm_core_world_model_probe.sh`

## Boundary

LeWorldModel is not the whole answer to human-like intuition. It is useful for
next-state prediction, surprise, compact planning priors, and future multimodal
sequences, but the current QTRM result shows that self-latent prediction alone
can be non-semantic.

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
