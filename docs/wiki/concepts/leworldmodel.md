# LeWorldModel

LeWorldModel is the current preferred JEPA world-model reference for QTRM.

Core contract:

- encode observations into latent embeddings
- predict next embeddings autoregressively from context embeddings and actions
- train end-to-end with MSE on next embeddings
- prevent collapse with SIGReg
- no training-time stop-gradient target branch
- no EMA target encoder

QTRM implication:

- `src/qtrm_mm/world_model.py` should keep LeWM-style next-latent prediction.
- If we use older I-JEPA/JEPA-WM stop-gradient ideas, they must be explicitly
  marked as a deviation.

## Boundary

LeWorldModel is not the whole answer to human-like intuition. It is the current
preferred reference for predictive latent world modeling: useful for next-state
prediction, surprise, compact planning priors, and future multimodal sequences.

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
