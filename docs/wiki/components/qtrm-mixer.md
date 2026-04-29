# QTRM Mixer

Current code:

- `src/qtrm_mm/mixers.py`
- `src/qtrm_mm/blocks.py`

Reference source:

- `docs/wiki/sources/gated-deltanet.md`

Status:

- Official adapter path added; local fallback remains experimental.

Findings:

- `TorchGatedDeltaMixer` is a simple bounded recurrent mixer.
- Its own docstring says it is not official KDA.
- Official Gated DeltaNet uses short convolution, q/k/v projections, beta/decay
  gates, chunked gated delta rule, and output gated norm.
- QTRM now imports official FLA GatedDeltaNet through `from fla.layers import
  GatedDeltaNet` when `delta_backend: fla_gated_delta` is selected.
- `strict_backends: true` raises if the official backend is not importable.
- `strict_backends: false` falls back to `TorchGatedDeltaMixer` and marks
  `is_official_backend=False`.

Gate before long training:

- Install FLA and run the adapter against the real `GatedDeltaNet`, not only the
  fake-symbol unit test.
- Verify real output shape, dtype, causal behavior, cache behavior, and mask
  behavior.
- Keep `torch_gated_delta` for smoke only.
