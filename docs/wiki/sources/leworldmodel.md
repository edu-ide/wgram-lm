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
