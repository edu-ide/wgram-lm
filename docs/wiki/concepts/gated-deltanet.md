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

- `src/qtrm_mm/mixers.py::TorchGatedDeltaMixer` is not official Gated DeltaNet.
- Production path should prefer FLA `GatedDeltaNet` or a faithful adapter to the
  NVLabs/FLA interface.
- Any fallback must be labelled debug/smoke only.
