# QTRM World Model

Current code:

- `src/qtrm_mm/world_model.py`
- `src/qtrm_mm/losses.py`
- `src/qtrm_mm/qtrm_model.py`

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
