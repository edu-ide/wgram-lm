# LeWM Integration Notes

## Reference Baseline

The QTRM JEPA path now uses LeWorldModel as the newest primary reference:

- `references/official/le-wm/jepa.py`
  - `encode()` creates latent embeddings
  - `predict()` predicts next embeddings from context embeddings and action embeddings
  - rollout is autoregressive at inference time
- `references/official/le-wm/train.py`
  - training uses `pred_loss = (pred_emb - tgt_emb).pow(2).mean()`
  - training adds `sigreg_loss`
  - no stop-gradient target branch and no EMA target encoder
- `references/official/le-wm/module.py`
  - `SIGReg`
  - `ARPredictor`
  - AdaLN-zero action conditioning

Older JEPA references remain useful for comparison:

- `references/official/jepa-wms`: action-conditioned JEPA-WM planning study
- `references/official/eb_jepa`: JEPA unroll patterns and older anti-collapse losses
- `references/official/ijepa`: masked latent prediction with stop-grad/EMA target branch

## Current QTRM Adaptation

QTRM adapts LeWM from pixel trajectories to token latents:

```text
input_ids
  -> text embedding
  -> causal JEPA encoder
  -> latent sequence emb[0:T]

emb[0:T-1] + optional action embeddings
  -> LeWM-style causal autoregressive predictor
  -> pred_emb[1:T]

emb[1:T]
  -> next-embedding target

loss = mse(pred_emb, emb[1:T]) + lambda * SIGReg(emb)
```

Implemented files:

- `src/wgram_lm/world_model.py`
  - `JepaWorldModelHead`
  - `ActionConditionedFuturePredictor`
  - `AdaLNZeroBlock`
  - `SIGReg`
- `src/wgram_lm/wgram_model.py`
  - adds a causal JEPA latent encoder
  - adds optional core-trajectory LeWM prediction over recursive `z_H` states
  - exposes `jepa_pred`, `jepa_target`, `jepa_latents`, `jepa_latent_mask`, and `jepa_mask`
  - exposes `core_world_model_pred`, `core_world_model_target`,
    `core_world_model_latents`, `core_world_model_latent_mask`, and
    `core_world_model_mask` when enabled
- `src/wgram_lm/losses.py`
  - `jepa_world_model_loss` now uses MSE plus optional SIGReg
  - `core_world_model_weight` applies the same LeWM loss to recursive core
    trajectories
- `tests/test_jepa_world_model.py`
  - verifies next-latent shapes, padding transition masks, end-to-end target gradients, SIGReg integration, and core trajectory prediction

## What Changed

Before:

```text
pooled z_h -> MLP -> last token embedding
```

That was only a smoke-test auxiliary objective.

Intermediate older-JEPA version:

```text
latent[t] -> predictor -> stopgrad(latent[t+1])
```

That matched I-JEPA/older JEPA-WM style better, but not the newest LeWM contract.

Now:

```text
latent[t] -> predictor -> latent[t+1]
SIGReg(latent[0:T]) prevents collapse
```

This is closer to LeWorldModel: end-to-end next-embedding prediction from
latents, action-conditioned predictor support, and a Gaussian latent
regularizer instead of stop-gradient/EMA.

Core-world-model probe:

```text
MemoryOS evidence / visible prompt
  -> Latent Workspace
  -> TRM recursive core
  -> z_H[0], z_H[1], z_H[2]

z_H[0:T-1] + action trace(RETRIEVE/VERIFY/ANSWER)
  -> LeWM-style predictor
  -> predicted z_H[1:T]

loss = mse(predicted z_H, next z_H) + lambda * SIGReg(z_H trajectory)
```

This directly connects LeWM to the TRM-like reasoning loop instead of only to
token embeddings. The first action trace is deliberately simple:
`OBSERVE` for no evidence, `RETRIEVE` when workspace evidence exists, then
`VERIFY`, then `ANSWER`.

## Still Not Done

This is not a full LeWM reproduction.

Remaining gaps:

1. Add real action traces for `LOOK`, `RETRIEVE`, `VERIFY`, `ANSWER`, then pass
   those actions into the predictor. The current implementation has a fixed
   first probe action trace, not learned or human-verified action labels.
2. Add multimodal/video latents from a real vision encoder, not only text token
   latents.
3. Add autoregressive rollout evaluation over multiple future steps.
4. Add bisection/search support for the SIGReg weight.
5. Keep Qwen/HF `generate()` as the baseline generator; the QTRM JEPA path is an
   auxiliary world-model/controller objective until metrics justify promotion.

Probe entry point:

```bash
HF_HOME=/mnt/nvme1n1p2/hf-cache-qtrm PYTHONPATH=src MAX_CASES=4 \
  bash scripts/128_run_lewm_core_world_model_probe.sh
```
