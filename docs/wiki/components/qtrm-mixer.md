# QTRM Mixer

Current code:

- `src/wgram_lm/mixers.py`
- `src/wgram_lm/blocks.py`

Reference source:

- `docs/wiki/sources/gated-deltanet.md`

Status:

- `official_gated_delta2 is fail-fast`: no Torch fallback, no runtime fallback,
  and no official-looking checkpoint from a non-official mixer.
- `TorchGatedDeltaMixer` remains a debug/smoke mixer only. It is not a fallback
  for an official GDN2 claim.
- Canonical LT2 choice is fixed to the Full+GDN 3:1 schedule:
  `GatedDelta/GDN, GatedDelta/GDN, GatedDelta/GDN, full attention`.
  In code this is `attn_every=4`. Do not keep shopping among `attn_every`
  ratios for the main path; ratio changes are ablations only.
- Runtime evidence must distinguish the requested backend from the backend that
  actually executed. After 2026-05-25, an `official_gated_delta2` run must stop
  if the official module, CUDA kernel, or pinned ptxas path is unavailable.

Findings:

- `TorchGatedDeltaMixer` is a simple bounded recurrent mixer.
- Its own docstring says it is not official KDA.
- Official Gated DeltaNet uses short convolution, q/k/v projections, beta/decay
  gates, chunked gated delta rule, and output gated norm.
- QTRM imports official FLA GatedDeltaNet through `from fla.layers import
  GatedDeltaNet` when the official backend is selected.
- For `official_gated_delta2`, `strict_backends` is effectively mandatory. The
  builder forces strict behavior and raises instead of constructing
  `TorchGatedDeltaMixer`.
- BLT/Data-IO training now logs `actual_delta_runtime`,
  `delta_runtime_fallback_active_count`, and related TensorBoard scalars. Treat
  `actual_delta_runtime=official_runtime` as the evidence for a real official
  GDN2 run; do not infer it from `args.delta_backend=official_gated_delta2`.
- 문과적으로 말하면: 세 번은 빠른 working-memory mixer로 생각을 굴리고, 네
  번째마다 full attention으로 전체 문맥과 다시 맞춘다. 이것이 우리가 채택한
  LT2 방식이다. LT2를 안 하는 것이 아니라 LT2의 Full+GDN 3:1 가지를 표준으로
  고정한다.

Gate before long training:

- Install FLA and run the adapter against the real `GatedDeltaNet`, not only the
  fake-symbol unit test.
- Verify real output shape, dtype, causal behavior, cache behavior, and mask
  behavior.
- Keep `torch_gated_delta` for smoke only.
- For DGX GB10, both env vars must be explicit:

  ```bash
  REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
  TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
  ```

  If either is missing, the launcher must fail with
  `missing required ptxas contract` or `missing required ptxas`.
- If logs contain `ptxas fatal: Value 'sm_121a'`, the run is not official and
  must stop. Do not continue by silently changing ptxas or mixer backend.
- Before continuing from an existing checkpoint, run:

  ```bash
  REQUIRED_TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas \
  TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas \
  bash scripts/559_run_stage95_blt_partial_then_full_dgx.sh preflight
  ```

  The underlying checker is
  `scripts/613_preflight_official_gdn2_contract.py`. It rejects legacy fallback
  checkpoint keys before they can become a confusing resume base. The Stage95
  launcher passes `--official-smoke ${OFFICIAL_GDN2_PREFLIGHT_SMOKE}` and
  defaults to `forward_auto`, so a CUDA machine also checks that the official
  mixer can execute a tiny forward pass before long training starts.
- `scripts/559_run_stage95_blt_partial_then_full_dgx.sh run` calls the same
  preflight automatically before partial training and before full continuation.
  The explicit `preflight` action is still useful for checking a run before
  launching it under `nohup`.
- Plain-language rule: an official GDN2 experiment is like testing a specific
  engine. If the engine is not installed, the right answer is "do not drive",
  not "secretly install a different engine and keep the same experiment name".
