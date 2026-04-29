# Gated DeltaNet Source

Source:

- Repo: `references/official/gated-delta-net`
- Upstream: `https://github.com/NVlabs/GatedDeltaNet`
- Commit: `b53d6d3a1612`
- Paper: `references/papers/gated_delta_networks_2412.06464.pdf`
- Paper URL: `https://arxiv.org/abs/2412.06464`

Key implementation files:

- `lit_gpt/gated_delta_net.py`
- `lit_gpt/gated_delta_rule_ops/chunk.py`
- `lit_gpt/model.py`
- `lit_gpt/config.py`

QTRM relevance:

- Official reference for the delta/recurrent mixer path.
- The repo README recommends FLA for faster kernels and varlen support.
- QTRM should treat local `torch_gated_delta` as a debug fallback only.
