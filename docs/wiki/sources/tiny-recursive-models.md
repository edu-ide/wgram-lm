# Tiny Recursive Models Source

Source:

- Repo: `references/official/tiny-recursive-models`
- Upstream: `https://github.com/SamsungSAILMontreal/TinyRecursiveModels`
- Commit: `c01103738605`
- Paper: `references/papers/tiny_recursive_models_2510.04871.pdf`
- Paper URL: `https://arxiv.org/abs/2510.04871`

Key implementation files:

- `models/recursive_reasoning/trm.py`
- `models/recursive_reasoning/trm_singlez.py`
- `models/losses.py`
- `config/arch/trm.yaml`

QTRM relevance:

- Reference for recursive z_H/z_L state updates.
- Important details: carry state, reset-on-halt, no-grad inner cycles, detach new
  carry, and ACT halt/continue heads.

Correction recorded 2026-05-19:

- TRM is not a single-state model in the official implementation. It keeps
  `z_H` and `z_L`.
- TRM's simplification relative to HRM is that it removes the separate
  `H_level` module: both `z_L` and `z_H` are updated through the same shared
  recurrent block.
- Therefore QTRM should distinguish `dual state` from `dual module` when
  comparing HRM-style and TRM-style cores.
