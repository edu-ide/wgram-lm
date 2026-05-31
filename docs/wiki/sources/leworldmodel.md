# LeWorldModel Source

Source:

- Repo: `references/official/le-wm`
- Upstream: `https://github.com/lucas-maes/le-wm`
- Commit: `bf04d3e8c375`
- Paper: `references/papers/leworldmodel_2603.19312.pdf`
- Paper URL: `https://arxiv.org/abs/2603.19312`

Key implementation files:

- `jepa.py`
- `train.py`
- `module.py`

QTRM relevance:

- Current preferred JEPA world-model objective.
- Uses next-embedding MSE plus SIGReg.
- Does not use stop-gradient target encoder or EMA during training.
- Provides predictive latent priors, not factual verification. Pair it with
  `docs/wiki/sources/fact-verification-and-fake-info.md` for truthfulness and
  fake-info handling.

2026-05-31 relation to own-latent prediction theory:

- arXiv:2605.27734 strengthens the general case for learning from same-model
  latent states rather than only visible tokens.
- It does not automatically promote the existing LeWM branch into the
  canonical answer path.
- Local use should be same-body and answer-causal: the predicted latent state
  must feed the normal decoder/LM head, and the claim must survive
  latent-predictor-off ablation.

See:

- `docs/wiki/sources/own-latent-prediction-sample-complexity.md`
- `docs/wiki/decisions/2026-05-31-own-latent-prediction-methodology-ssot.md`
