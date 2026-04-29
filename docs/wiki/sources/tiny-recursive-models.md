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
