# QTRM World Model

Current code:

- `src/wgram_lm/world_model.py`
- `src/wgram_lm/losses.py`
- `src/wgram_lm/wgram_model.py`

Reference source:

- `docs/wiki/sources/leworldmodel.md`

Status:

- Partially aligned with LeWM, but demoted to probe-only for the canonical QTRM
  architecture.

Aligned:

- next-latent prediction
- end-to-end target gradients
- SIGReg anti-collapse regularizer
- action-conditioned predictor interface
- AdaLN-zero style conditioning block
- optional LeWM-style prediction over recursive core `z_H` trajectories

Remaining gaps:

- only fixed probe action traces so far; no learned or human-labeled action traces
- text-token latents only; no pixel/video latent trajectory
- no multi-step rollout evaluation
- no SIGReg weight search

Current probe:

- `configs/qwen35_2b_4090_lewm_core_world_model_probe_s050.yaml`
- `scripts/128_run_lewm_core_world_model_probe.sh`

The probe keeps the evidence-bottleneck setup and adds a weak
`loss_core_world_model_weight` so TRM-style latent reasoning states are trained
to predict their next state under an explicit action trace.

Canonical boundary:

- `core_world_model_enabled=false`
- `loss_core_world_model_weight=0.0`
- LeWM must not be on the canonical answer path until a semantic transition or
  answer-causal gate passes.

Reason:

- The LeWM head learned the current recursive latent trajectory, but the
  symbolic intermediate-state gate did not improve. Predicting `z_H[t+1]` from
  `z_H[t]` can model the core's own latent motion without improving reasoning.

2026-05-31 correction:

- arXiv:2605.27734 is now adopted as evidence that own-latent prediction is a
  serious sample-efficiency methodology for discovering hidden hierarchical
  structure.
- This does not make the existing world-model branch canonical. It means the
  next canonical candidate should be a same-body latent-prediction objective
  whose targets are already on the answer-causal path.
- Promotion still requires latent-predictor-off ablation and same-LM-head
  answer improvement, not only lower latent transition loss.

See:

- `docs/wiki/sources/own-latent-prediction-sample-complexity.md`
- `docs/wiki/decisions/2026-05-31-own-latent-prediction-methodology-ssot.md`
