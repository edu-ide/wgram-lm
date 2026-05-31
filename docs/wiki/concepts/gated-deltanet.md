# Gated DeltaNet

Gated DeltaNet is the official reference for QTRM's delta/recurrent token mixer
axis.

Core contract:

- q/k/v projections
- short convolution on q/k/v
- beta gate for delta update
- decay gate for memory management
- chunked gated delta rule
- output norm and swish gate
- hybrid placement with attention layers

QTRM implication:

- `src/wgram_lm/mixers.py::TorchGatedDeltaMixer` is not official Gated DeltaNet.
- Production path should prefer FLA `GatedDeltaNet` or a faithful adapter to the
  NVLabs/FLA interface.
- `official_gated_delta2` must not fall back to `TorchGatedDeltaMixer`.
  Missing official code, missing kernels, or missing pinned ptxas is a launch
  failure, not a reason to change backend silently.
- Any non-official mixer must be labelled debug/smoke only.
