# QTRM World Model

Current code:

- `src/qtrm_mm/world_model.py`
- `src/qtrm_mm/losses.py`
- `src/qtrm_mm/qtrm_model.py`

Reference source:

- `docs/wiki/sources/leworldmodel.md`

Status:

- Partially aligned with LeWM.

Aligned:

- next-latent prediction
- end-to-end target gradients
- SIGReg anti-collapse regularizer
- action-conditioned predictor interface
- AdaLN-zero style conditioning block

Remaining gaps:

- no real action traces yet
- text-token latents only; no pixel/video latent trajectory
- no multi-step rollout evaluation
- no SIGReg weight search
